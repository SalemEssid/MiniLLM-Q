# MiniLLM-Q

A GPT-2 small inference engine in C++, used to test whether sensitivity-guided mixed-precision quantization
turns saved bits into decode speed on a laptop CPU. The plan, phases and gates are in
[the guide](<document/MiniLLM-Q guide.tex>).

## Setup (Windows)

- Python 3.11: `python -m venv .venv`, then `.venv\Scripts\python -m pip install -r requirements.txt`.
- C++: the MSYS2 UCRT64 toolchain (`pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake
  mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-gdb`) with `C:\msys64\ucrt64\bin` on `PATH`.

## Phase 0: weights, tokens, reference outputs

```
.venv\Scripts\python python\export_weights.py     # data/gpt2_124M.bin, checked bit for bit on read-back
.venv\Scripts\python python\tokenize_wikitext.py  # data/wikitext2_test.u16, data/wikitext2_train.u16
.venv\Scripts\python python\make_reference.py     # data/ref_logits.f32, data/ref_greedy.u16
```

Each script writes a JSON file next to its outputs with shapes, SHA-256 checksums and package versions.
Nothing in Phase 0 is random; the seed is fixed at 0 anyway. The binaries are not tracked: run the three
scripts to recreate them (about 5 minutes plus the GPT-2 and WikiText-2 downloads).

## File formats

All files are little-endian and have no compression.

### `data/gpt2_124M.bin` (497,759,296 bytes)

| Bytes | Content |
|---|---|
| 0–23 | 6 × int32: magic 20261005, n_layer 12, n_head 12, d_model 768, vocab 50257, n_ctx 1024 |
| 24–63 | zeros |
| 64– | 148 float32 tensors, back to back, in the order below |

Order: `wte` [50257, 768], `wpe` [1024, 768]; then for each of the 12 blocks `ln_1.weight` [768],
`ln_1.bias` [768], `attn.c_attn.weight` [768, 2304], `attn.c_attn.bias` [2304], `attn.c_proj.weight`
[768, 768], `attn.c_proj.bias` [768], `ln_2.weight` [768], `ln_2.bias` [768], `mlp.c_fc.weight` [768, 3072],
`mlp.c_fc.bias` [3072], `mlp.c_proj.weight` [3072, 768], `mlp.c_proj.bias` [768]; then `ln_f.weight` [768],
`ln_f.bias` [768]. The byte offset of each tensor is in `data/gpt2_124M.json`.

- Linear weights keep the Conv1D layout `[in, out]`, row-major: `y = x @ W + b`.
- `c_attn` outputs `[q | k | v]`, 768 values each; head `h` uses entries `64h .. 64h+63` of each.
- The LM head is `wte` itself (`logits = ln_f(h) @ wte^T`) and is not stored twice.
- Every tensor is a multiple of 64 bytes long, so each one starts 64-byte aligned.
- LayerNorm uses ε = 1e-5; GELU is the tanh approximation (`gelu_new`).

### `data/wikitext2_test.u16`, `data/wikitext2_train.u16`

Raw uint16 token ids with no header: WikiText-2 (`wikitext-2-raw-v1`) lines joined with `"\n\n"`, then
tokenized with the GPT-2 tokenizer. The test split has 287,644 tokens and the train split 2,428,601.
Final perplexities use test; calibration slices come from train.

### Reference outputs (described in `data/reference.json`)

- **Prompt:** test tokens [64, 128). This is the first 64-token window whose 100-step greedy run has every
  top-1 minus top-2 logit margin ≥ 0.01. The window at offset 0 has a near-tie of 0.0011, where a correct
  engine could pick the other token.
- **`ref_logits.f32`:** [64, 50257] float32. Row `p` holds the logits after `prompt[0..p]`: row 0 after the
  first token, row 63 after the whole prompt.
- **`ref_greedy.u16`:** the 100 greedy tokens after the prompt (argmax at every step, no stop at EOS). The
  smallest margin is 0.021.
- **Perplexity:** test tokens [0, 32768) as 32 non-overlapping windows of 1024. Each window scores its 1023
  next-token predictions, and ppl = exp(mean NLL) = **31.8220**. The mean NLL of each window is in
  `reference.json`, to locate a mismatch.
