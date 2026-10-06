#include "gpt2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <new>
#include <stdexcept>

#include "kernels.h"
#include "threads.h"

namespace {

constexpr int32_t kMagic = 20261005;
constexpr std::size_t kHeaderBytes = 64;
constexpr std::align_val_t kAlign{64};

// steady_clock, not omp_get_wtime(): MinGW's libgomp timer has 1 ms resolution.
double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Adds the wall time of f() to bucket.
template <class F>
void timed(double& bucket, F&& f) {
  const double t0 = now_s();
  f();
  bucket += now_s() - t0;
}

void transpose_in_place(float* W, int in, int out, std::vector<float>& tmp) {
  tmp.assign(W, W + (std::size_t)in * out);
  for (int i = 0; i < in; ++i)
    for (int o = 0; o < out; ++o) W[(std::size_t)o * in + i] = tmp[(std::size_t)i * out + o];
}

}  // namespace

GPT2::GPT2(const std::string& path, const Quantization& quant) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char header[kHeaderBytes];
  f.read(header, kHeaderBytes);
  int32_t h[6];
  std::memcpy(h, header, sizeof h);
  if (!f || h[0] != kMagic) throw std::runtime_error(path + ": bad header");
  cfg_ = {h[1], h[2], h[3], h[4], h[5]};
  const std::size_t d = cfg_.d_model, L = cfg_.n_layer;
  if (!quant.widths.empty() && quant.widths.size() != 4 * L)
    throw std::invalid_argument("a quantization needs one width per block linear layer");
  if (quant.head_bits != 0 && quant.head_bits != 8) throw std::invalid_argument("the head is FP32 or 8-bit");
  const auto width = [&](std::size_t index) { return quant.widths.empty() ? 0 : quant.widths[index]; };

  const std::size_t per_block = 12 * d * d + 13 * d;
  const std::size_t n_floats = cfg_.vocab * d + cfg_.n_ctx * d + L * per_block + 2 * d;
  f.seekg(0, std::ios::end);
  if ((std::size_t)f.tellg() != kHeaderBytes + 4 * n_floats) throw std::runtime_error(path + ": wrong size");
  f.seekg(kHeaderBytes);

  // FP32 weights of quantized layers are skipped, so the buffer only holds what is kept.
  std::size_t skipped = 0;
  for (std::size_t i = 0; i < 4 * L; ++i)
    if (width(i)) skipped += (std::size_t)module_shape((int)i, (int)d, cfg_.vocab).first *
                             module_shape((int)i, (int)d, cfg_.vocab).second;
  weights_ = static_cast<float*>(::operator new(4 * (n_floats - skipped), kAlign));

  // Read tensors in file order (README.md), transposing the FP32 Conv1D weights of each block.
  float* p = weights_;
  std::vector<float> tmp;
  auto take = [&](std::size_t n) {
    f.read(reinterpret_cast<char*>(p), 4 * n);
    const float* t = p;
    p += n;
    return t;
  };
  auto take_linear = [&](std::size_t index) {  // a block linear layer: weight [in, out], then bias [out]
    const auto [in, out] = module_shape((int)index, (int)d, cfg_.vocab);
    Linear layer;
    layer.in = in, layer.out = out;
    if (const int bits = width(index)) {
      f.seekg(4 * (std::streamoff)in * out, std::ios::cur);
      layer.q = load_qlinear(quant.dir, bits, (int)index, (int)d, cfg_.vocab, quant.family);
    } else {
      f.read(reinterpret_cast<char*>(p), 4 * (std::size_t)in * out);
      transpose_in_place(p, in, out, tmp);
      layer.w = p;
      p += (std::size_t)in * out;
    }
    layer.b = take(out);
    return layer;
  };
  wte_ = take(cfg_.vocab * d);
  wpe_ = take(cfg_.n_ctx * d);
  for (std::size_t l = 0; l < L; ++l) {
    Block b;
    b.ln1_w = take(d), b.ln1_b = take(d);
    b.attn = take_linear(4 * l), b.attn_proj = take_linear(4 * l + 1);
    b.ln2_w = take(d), b.ln2_b = take(d);
    b.fc = take_linear(4 * l + 2), b.mlp_proj = take_linear(4 * l + 3);
    blocks_.push_back(std::move(b));
  }
  lnf_w_ = take(d), lnf_b_ = take(d);
  if (!f) throw std::runtime_error(path + ": short read");

  head_.in = (int)d, head_.out = cfg_.vocab;  // the head is wte, [vocab, d], with no bias
  if (quant.head_bits) head_.q = load_qlinear(quant.dir, 8, 48, (int)d, cfg_.vocab, quant.family);
  else head_.w = wte_;

  const std::size_t T = cfg_.n_ctx;
  h_.resize(T * d), ln_.resize(T * d), att_.resize(T * d), proj_.resize(T * d);
  qkv_.resize(T * 3 * d), fc_.resize(T * 4 * d);
  k_cache_.resize(L * T * d), v_cache_.resize(L * T * d);
}

GPT2::~GPT2() { ::operator delete(weights_, kAlign); }

void GPT2::forward_sequence(const int* tokens, int T, float* logits) {
  reset();
  forward(tokens, T, logits);
}

void GPT2::decode_step(int token, float* logits) { forward(&token, 1, logits); }

double GPT2::parameter_bytes_per_token() const {
  const auto layer_bytes = [](const Linear& l) {
    return (l.q.bits ? (double)qbytes(l.q) : 4.0 * l.in * l.out) + (l.b ? 4.0 * l.out : 0.0);
  };
  const double d = cfg_.d_model;
  double bytes = layer_bytes(head_) + 4 * (2 * d + 2 * d);  // head, final LayerNorm, rows of wte and wpe
  for (const Block& b : blocks_)
    bytes += layer_bytes(b.attn) + layer_bytes(b.attn_proj) + layer_bytes(b.fc) + layer_bytes(b.mlp_proj) + 4 * 4 * d;
  return bytes;
}

void GPT2::apply(const Linear& layer, float* y, const float* x, int T) {
  if (!layer.q.bits) return linear(y, x, layer.w, layer.b, T, layer.in, layer.out);
  // Quantized: every row gets the arithmetic of a single matrix-vector product, so a decode step and the
  // same position inside a sequence stay bit-identical.
  qmatmul(layer.q, x, y, T);
  if (layer.b)
    for (int t = 0; t < T; ++t) add(y + (std::size_t)t * layer.out, layer.b, layer.out);
}

void GPT2::forward(const int* tokens, int T, float* logits) {
  const double t0 = now_s();
  const int d = cfg_.d_model;
  if (T < 1 || pos_ + T > cfg_.n_ctx) throw std::runtime_error("context length exceeded");
  for (int t = 0; t < T; ++t) {
    if (tokens[t] < 0 || tokens[t] >= cfg_.vocab) throw std::runtime_error("token id out of range");
    const float* te = wte_ + (std::size_t)tokens[t] * d;
    const float* pe = wpe_ + (std::size_t)(pos_ + t) * d;
    for (int i = 0; i < d; ++i) h_[(std::size_t)t * d + i] = te[i] + pe[i];
  }
  for (int l = 0; l < cfg_.n_layer; ++l) {
    const Block& b = blocks_[l];
    double& lin = timings_.linear;
    layernorm(ln_.data(), h_.data(), b.ln1_w, b.ln1_b, T, d);
    timed(lin, [&] { apply(b.attn, qkv_.data(), ln_.data(), T); });
    timed(timings_.attention, [&] { attention(l, T); });
    timed(lin, [&] { apply(b.attn_proj, proj_.data(), att_.data(), T); });
    add(h_.data(), proj_.data(), (std::size_t)T * d);
    layernorm(ln_.data(), h_.data(), b.ln2_w, b.ln2_b, T, d);
    timed(lin, [&] { apply(b.fc, fc_.data(), ln_.data(), T); });
    gelu(fc_.data(), (std::size_t)T * 4 * d);
    timed(lin, [&] { apply(b.mlp_proj, proj_.data(), fc_.data(), T); });
    add(h_.data(), proj_.data(), (std::size_t)T * d);
  }
  layernorm(ln_.data(), h_.data(), lnf_w_, lnf_b_, T, d);
  timed(timings_.linear, [&] { apply(head_, logits, ln_.data(), T); });
  pos_ += T;
  timings_.total += now_s() - t0;
}

void GPT2::attention(int layer, int T) {
  const int d = cfg_.d_model, n_head = cfg_.n_head, hs = d / n_head;
  const float scale = 1.0f / std::sqrt((float)hs);
  float* K = k_cache_.data() + (std::size_t)layer * cfg_.n_ctx * d;
  float* V = v_cache_.data() + (std::size_t)layer * cfg_.n_ctx * d;

  // Append this chunk's keys and values. A row of qkv is [q | k | v]; head h uses entries h*hs..h*hs+hs-1.
  for (int t = 0; t < T; ++t) {
    const float* row = qkv_.data() + (std::size_t)t * 3 * d;
    std::copy_n(row + d, d, K + (std::size_t)(pos_ + t) * d);
    std::copy_n(row + 2 * d, d, V + (std::size_t)(pos_ + t) * d);
  }

  // One task per (position, head).
  parallel_for(T * n_head, [&](int begin, int end) {
    thread_local std::vector<float> s;  // attention scores, one buffer per thread
    s.resize(cfg_.n_ctx);
    for (int task = begin; task < end; ++task) {
      const int t = task / n_head, h = task % n_head;
      const int n = pos_ + t + 1;  // causal: position pos_+t sees keys 0..pos_+t
      const float* q = qkv_.data() + (std::size_t)t * 3 * d + h * hs;
      for (int j = 0; j < n; ++j) s[j] = dot(q, K + (std::size_t)j * d + h * hs, hs) * scale;
      softmax(s.data(), n);
      float* out = att_.data() + (std::size_t)t * d + h * hs;
      std::fill_n(out, hs, 0.f);
      for (int j = 0; j < n; ++j) {
        const float* v = V + (std::size_t)j * d + h * hs;
        for (int i = 0; i < hs; ++i) out[i] += s[j] * v[i];
      }
    }
  });
}
