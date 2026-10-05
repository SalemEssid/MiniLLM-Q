#pragma once
// GPT-2 small in FP32: loader, forward pass and KV cache.
#include <string>
#include <vector>

struct Config {
  int n_layer, n_head, d_model, vocab, n_ctx;
};

class GPT2 {
 public:
  // Reads gpt2_124M.bin (format in README.md) and transposes every Conv1D weight from [in, out] to
  // [out, in], so each output of a linear layer reads one contiguous row, like the head reads wte.
  explicit GPT2(const std::string& path);
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

 private:
  struct Block {
    const float *ln1_w, *ln1_b, *attn_w, *attn_b, *attn_proj_w, *attn_proj_b;
    const float *ln2_w, *ln2_b, *fc_w, *fc_b, *mlp_proj_w, *mlp_proj_b;
  };

  // Runs tokens[0..T) at positions pos()..pos()+T-1 and appends their keys and values to the cache.
  // Both entry points go through here; they differ only in T and in the starting position.
  void forward(const int* tokens, int T, float* logits);
  // att_ = softmax(q k^T / sqrt(head size)) v for every head, over cached positions 0..pos()+t.
  void attention(int layer, int T);

  Config cfg_{};
  float* weights_ = nullptr;  // every tensor of the file, 64-byte aligned
  const float *wte_, *wpe_, *lnf_w_, *lnf_b_;
  std::vector<Block> blocks_;
  int pos_ = 0;

  // Activations for up to n_ctx positions; KV cache as [layer][position][d_model].
  std::vector<float> h_, ln_, qkv_, att_, proj_, fc_, k_cache_, v_cache_, scores_;
};
