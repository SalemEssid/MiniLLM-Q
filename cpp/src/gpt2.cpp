#include "gpt2.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <new>
#include <stdexcept>

#include "kernels.h"

namespace {

constexpr int32_t kMagic = 20261005;
constexpr std::size_t kHeaderBytes = 64;
constexpr std::align_val_t kAlign{64};

void transpose_in_place(float* W, int in, int out, std::vector<float>& tmp) {
  tmp.assign(W, W + (std::size_t)in * out);
  for (int i = 0; i < in; ++i)
    for (int o = 0; o < out; ++o) W[(std::size_t)o * in + i] = tmp[(std::size_t)i * out + o];
}

}  // namespace

GPT2::GPT2(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char header[kHeaderBytes];
  f.read(header, kHeaderBytes);
  int32_t h[6];
  std::memcpy(h, header, sizeof h);
  if (!f || h[0] != kMagic) throw std::runtime_error(path + ": bad header");
  cfg_ = {h[1], h[2], h[3], h[4], h[5]};
  const std::size_t d = cfg_.d_model, L = cfg_.n_layer;

  const std::size_t per_block = 12 * d * d + 13 * d;
  const std::size_t n_floats = cfg_.vocab * d + cfg_.n_ctx * d + L * per_block + 2 * d;
  f.seekg(0, std::ios::end);
  if ((std::size_t)f.tellg() != kHeaderBytes + 4 * n_floats) throw std::runtime_error(path + ": wrong size");
  f.seekg(kHeaderBytes);
  weights_ = static_cast<float*>(::operator new(4 * n_floats, kAlign));
  f.read(reinterpret_cast<char*>(weights_), 4 * n_floats);
  if (!f) throw std::runtime_error(path + ": short read");

  // Hand out pointers in file order (README.md), transposing the four Conv1D weights of each block.
  float* p = weights_;
  std::vector<float> tmp;
  auto take = [&p](std::size_t n) { const float* t = p; p += n; return t; };
  auto take_linear = [&](std::size_t in, std::size_t out) {
    transpose_in_place(p, (int)in, (int)out, tmp);
    return take(in * out);
  };
  wte_ = take(cfg_.vocab * d);
  wpe_ = take(cfg_.n_ctx * d);
  for (std::size_t l = 0; l < L; ++l) {
    Block b{};
    b.ln1_w = take(d), b.ln1_b = take(d);
    b.attn_w = take_linear(d, 3 * d), b.attn_b = take(3 * d);
    b.attn_proj_w = take_linear(d, d), b.attn_proj_b = take(d);
    b.ln2_w = take(d), b.ln2_b = take(d);
    b.fc_w = take_linear(d, 4 * d), b.fc_b = take(4 * d);
    b.mlp_proj_w = take_linear(4 * d, d), b.mlp_proj_b = take(d);
    blocks_.push_back(b);
  }
  lnf_w_ = take(d), lnf_b_ = take(d);

  const std::size_t T = cfg_.n_ctx;
  h_.resize(T * d), ln_.resize(T * d), att_.resize(T * d), proj_.resize(T * d);
  qkv_.resize(T * 3 * d), fc_.resize(T * 4 * d);
  k_cache_.resize(L * T * d), v_cache_.resize(L * T * d);
  scores_.resize((std::size_t)omp_get_max_threads() * T);
}

GPT2::~GPT2() { ::operator delete(weights_, kAlign); }

void GPT2::forward_sequence(const int* tokens, int T, float* logits) {
  reset();
  forward(tokens, T, logits);
}

void GPT2::decode_step(int token, float* logits) { forward(&token, 1, logits); }

void GPT2::forward(const int* tokens, int T, float* logits) {
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
    layernorm(ln_.data(), h_.data(), b.ln1_w, b.ln1_b, T, d);
    linear(qkv_.data(), ln_.data(), b.attn_w, b.attn_b, T, d, 3 * d);
    attention(l, T);
    linear(proj_.data(), att_.data(), b.attn_proj_w, b.attn_proj_b, T, d, d);
    add(h_.data(), proj_.data(), (std::size_t)T * d);
    layernorm(ln_.data(), h_.data(), b.ln2_w, b.ln2_b, T, d);
    linear(fc_.data(), ln_.data(), b.fc_w, b.fc_b, T, d, 4 * d);
    gelu(fc_.data(), (std::size_t)T * 4 * d);
    linear(proj_.data(), fc_.data(), b.mlp_proj_w, b.mlp_proj_b, T, 4 * d, d);
    add(h_.data(), proj_.data(), (std::size_t)T * d);
  }
  layernorm(ln_.data(), h_.data(), lnf_w_, lnf_b_, T, d);
  linear(logits, ln_.data(), wte_, nullptr, T, d, cfg_.vocab);  // the head is wte, [vocab, d]
  pos_ += T;
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

#pragma omp parallel for collapse(2) schedule(static)
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < n_head; ++h) {
      const int n = pos_ + t + 1;  // causal: position pos_+t sees keys 0..pos_+t
      const float* q = qkv_.data() + (std::size_t)t * 3 * d + h * hs;
      float* s = scores_.data() + (std::size_t)omp_get_thread_num() * cfg_.n_ctx;
      for (int j = 0; j < n; ++j) s[j] = dot(q, K + (std::size_t)j * d + h * hs, hs) * scale;
      softmax(s, n);
      float* out = att_.data() + (std::size_t)t * d + h * hs;
      std::fill_n(out, hs, 0.f);
      for (int j = 0; j < n; ++j) {
        const float* v = V + (std::size_t)j * d + h * hs;
        for (int i = 0; i < hs; ++i) out[i] += s[j] * v[i];
      }
    }
}
