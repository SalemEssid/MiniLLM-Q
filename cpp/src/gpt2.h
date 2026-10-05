#pragma once
// GPT-2 small in FP32: loader, forward pass and KV cache. Each of the 48 block linear layers, and the head,
// can instead be weight-only quantized (quant.h).
#include <string>
#include <vector>

#include "quant.h"

struct Config {
  int n_layer, n_head, d_model, vocab, n_ctx;
};

// Which linear layers are quantized, to how many bits, and with which kernel family.
struct Quantization {
  std::vector<int> widths;  // empty, or 48 entries in allocation order (block 0's c_attn, c_proj, c_fc,
                            // mlp.c_proj, then block 1's, ...); 0 keeps a layer in FP32
  int head_bits = 0;        // 0 (FP32) or 8
  Family family = Family::Unpack;
  std::string dir;          // folder holding gpt2_q{b}.bin
};

class GPT2 {
 public:
  // Reads gpt2_124M.bin (format in README.md) and transposes every FP32 Conv1D weight from [in, out] to
  // [out, in], so each output of a linear layer reads one contiguous row, like the head reads wte. The FP32
  // weights of quantized layers are skipped, not loaded.
  explicit GPT2(const std::string& path, const Quantization& quant = {});
  ~GPT2();
  GPT2(const GPT2&) = delete;
  GPT2& operator=(const GPT2&) = delete;

  // Logits at every position of tokens[0..T), starting from an empty cache. logits: [T, vocab].
  void forward_sequence(const int* tokens, int T, float* logits);
  // Logits for one more token at position pos(), reading and extending the KV cache. logits: [vocab].
  void decode_step(int token, float* logits);

  void reset() { pos_ = 0; }
  int pos() const { return pos_; }
  const Config& config() const { return cfg_; }

  // Bytes of parameters a decode step reads: every linear layer (weights, scales and biases), the
  // LayerNorms, and one row each of wte and wpe. The KV cache comes on top.
  double parameter_bytes_per_token() const;

  // Wall time spent in forward() and in its linear layers and attention, for the Phase 2 breakdown.
  struct Timings {
    double total = 0, linear = 0, attention = 0;
  };
  const Timings& timings() const { return timings_; }
  void reset_timings() { timings_ = {}; }

 private:
  struct Linear {
    int in = 0, out = 0;
    const float* w = nullptr;  // FP32 weights [out, in], when not quantized
    const float* b = nullptr;  // bias, or null
    QLinear q;                 // quantized weights, when q.bits > 0
  };
  struct Block {
    const float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    Linear attn, attn_proj, fc, mlp_proj;
  };

  // y[T, out] = x[T, in] @ W + b, FP32 or quantized.
  void apply(const Linear& layer, float* y, const float* x, int T);
  // Runs tokens[0..T) at positions pos()..pos()+T-1 and appends their keys and values to the cache.
  // Both entry points go through here; they differ only in T and in the starting position.
  void forward(const int* tokens, int T, float* logits);
  // att_ = softmax(q k^T / sqrt(head size)) v for every head, over cached positions 0..pos()+t.
  void attention(int layer, int T);

  Config cfg_{};
  float* weights_ = nullptr;  // every FP32 tensor kept from the file, 64-byte aligned
  const float *wte_, *wpe_, *lnf_w_, *lnf_b_;
  std::vector<Block> blocks_;
  Linear head_;
  int pos_ = 0;
  Timings timings_;

  // Activations for up to n_ctx positions; KV cache as [layer][position][d_model].
  std::vector<float> h_, ln_, qkv_, att_, proj_, fc_, k_cache_, v_cache_;
};
