# MiniLLM-Q

A GPT-2 small (124M) inference engine in C++. The question: **when does a mixed-precision allocation turn
into faster decoding on a laptop CPU?** It compares two quantized kernel families (unpack-dequantize and
bit-plane) and uses llama.cpp as an outside baseline. The full plan and pass gates are in
[the guide](<document/MiniLLM-Q guide.tex>).

Machine: Intel i5-1135G7 (4 cores, AVX-512), 8 GB RAM, Windows 11.

## Status

| Phase | What | Result |
|---|---|---|
| 0 | Weights, tokens, reference outputs | Done |
| 1 | FP32 C++ engine | Done: matches PyTorch, perplexity 31.8222 vs 31.8220 |
| 2 | Bandwidth and FP32 decode speed | Done: 54.4 tokens/s, 80% of the bandwidth ceiling |
| 3 | Quantized kernels, two families | Done: matches Python within 7e-6 |
| 4 | Sensitivity and interaction scans | Scans done; written discussion still to do |
| 5 | Closed-form allocation | Allocations scored; derivations and BAQ comparison still to do |
| 6–7 | Search, experiments | Not started |

## Setup (Windows)

- Python 3.11: `python -m venv .venv`, then `.venv\Scripts\python -m pip install -r requirements.txt`.
- C++: MSYS2 UCRT64 (`pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake
  mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-gdb`), with `C:\msys64\ucrt64\bin` on `PATH`.
- Build: `cmake -S cpp -B cpp/build -G Ninja`, then `cmake --build cpp/build`.

## Phase 0: weights, tokens, reference outputs

```
.venv\Scripts\python python\export_weights.py     # data/gpt2_124M.bin
.venv\Scripts\python python\tokenize_wikitext.py  # data/wikitext2_test.u16, data/wikitext2_train.u16
.venv\Scripts\python python\make_reference.py     # data/ref_logits.f32, data/ref_greedy.u16
```

- Each script also writes a JSON file with shapes, SHA-256 checksums and package versions.
- The binaries are not in git. These three scripts recreate them in about 5 minutes, plus downloads.

## Phase 1: FP32 C++ engine

```
cpp\build\phase1_check.exe       # gates (a)-(d); writes results/phase1.json
```

- `cpp/src/gpt2.{h,cpp}`: loader, forward pass, KV cache. `decode_step` and `forward_sequence` share one
  code path, so they give bit-identical results.
- `cpp/src/kernels.{h,cpp}`: `linear`, `layernorm`, `gelu`, `softmax`, `add`.
- The loader transposes each weight to `[out, in]`, so every output is one contiguous dot product.

**Result:** all four gates pass. Logits: cosine 1.0, max error 2.7e-4. The 100 greedy tokens are
identical to PyTorch. The KV cache matches the full forward pass exactly. Perplexity is 31.8222 against
31.8220 in Python.

## Phase 2: bandwidth and FP32 decode speed

```
cpp\build\bandwidth.exe [MB]          # read bandwidth at 1, 2, 4, 8 threads
cpp\build\phase2_bench.exe            # decode speed + roofline; writes results/phase2.json
cpp\build\phase1_check.exe --quick    # fast correctness check after any kernel change
```

**Result** (on mains power): bandwidth 34.8 GB/s, so decode speed can't exceed 68.4 tokens/s. With 8
threads it reaches 54.4 ± 2.9 tokens/s (80% of that ceiling). All later speed runs use 8 threads.

What made it fast:

- An AVX2 dot product with four accumulators (`kernels.h`).
- A spinning thread pool (`threads.h`), because a libgomp (OpenMP) parallel region costs 40–140 µs on
  Windows.
- Opting out of Windows power throttling (`tools/bench_env.h`).
- Timing with `steady_clock`: `omp_get_wtime()` only has 1 ms resolution with MinGW.

## Phase 3: quantized kernels, two families

```
.venv\Scripts\python python\export_quantized.py      # data/gpt2_q{2,3,4,6,8}.bin
.venv\Scripts\python python\make_quant_reference.py  # reference outputs for the gates
cpp\build\phase3_check.exe [--skip-ppl] [--ppl-widths 8,4]   # gates (a), (b); results/phase3.json
cpp\build\phase3_bench.exe [--decode]                         # gate (c); results/phase3_bench.json
```

- **Format:** asymmetric, groups of 128 inputs, FP16 scale and minimum per group. Python quantizes; C++
  only repacks the codes.
- **Family U (unpack-dequantize):** codes are stored in byte-aligned fields (3 bits = 2 + 1, 6 = 4 + 2, as
  in llama.cpp). The kernel unpacks them, converts to float and multiplies (AVX-512).
- **Family P (bit-plane):** each bit of the codes is stored separately. A 16-entry table of partial sums
  for every 4 inputs replaces the multiplications. One kernel serves every width.
- **An allocation** is 48 widths (0 = FP32), the head at FP32 or 8 bits, and a family.
- `qmatmul` handles 64 rows at a time with exactly the arithmetic of a decode step. A 32-window
  perplexity takes about 3 minutes.

**Result:**

- **(a)** All 42 kernel cases pass. Max error 7e-7 (limit 1e-5).
- **(b)** Perplexity matches Python for both families: 31.8307 vs 31.8305 at 8 bits, and 36.3900 vs
  36.3898 at 4 bits.
- **(c)** Speed is measured for every width and family, but on battery. Rerun `phase3_bench --decode` on
  mains power for the record.

## Phase 4: sensitivity and interaction scans

```
.venv\Scripts\python python\sensitivity_scan.py [--only scan|pairs|allocations|near]   # results/phase4_scan.json
.venv\Scripts\python python\analyze_scan.py                                            # results/phase4_analysis.json + 5 figures
```

Each evaluation measures perplexity on the first 16,384 tokens of the WikiText-2 **train** split (the test
split is never used here). It takes about 40 s. The scan runs:

- **FP32 baseline** (1 evaluation): perplexity 32.24.
- **Single modules** (240): each of the 48 modules at 2, 3, 4, 6 and 8 bits, with the rest in FP32.
- **Pairs** (60): two modules at 3 bits, stratified by distance and type.
- **Far allocations** (20): 40 random size-neutral moves away from uniform 3 or 4 bits.
- **Near allocations** (20): uniform 3 and 4 bits, plus 9 per budget made with only 2–6 moves.

Running it:

- It saves after every evaluation. If it stops, run it again and it continues where it left off.
- It keeps Windows from sleeping while it runs, but **the lid must stay open**. Use mains power.
- The full run takes about 4 hours.

**Results** (`results/phase4_analysis.json`, figures in `results/phase4_*.png`):

- **Bits vs. damage:** loss falls about as 2^-2b. The fitted exponent has median 1.94 (the model assumes
  2) and ranges from 1.07 to 3.53 across modules.
- **Where the damage is:** at 3 bits, the top 5 of 48 modules cause 32% of it. They are mostly the
  attention input projections (`attn.c_attn`) of blocks 2–8.
- **2-bit outliers:** some modules break at 2 bits (`h.0.mlp.c_proj`: perplexity 381). Ranking modules
  with and without 2-bit gives different orders (Kendall τ 0.64).
- **Pairs:** two modules' damage adds up (median ratio 0.99–1.04 per group).
- **Whole allocations (near uniform):** the sum of single-module damage ranks allocations well (Spearman
  0.96) but underestimates the damage. At 4 bits it is close: uniform 4-bit measures perplexity 37.4
  against a predicted 36.5. At 3 bits it fails: uniform 3-bit measures 166 against a predicted 56, so errors
  stop adding up below 4 bits.
- **Whole allocations (far from uniform):** with 6–16 modules at 2 bits, the model breaks (perplexity
  96–4,700), and the real damage is about 1.8× the sum of the single-module damage.

## Phase 5: closed-form allocation

```
.venv\Scripts\python python\allocate.py [--no-eval]   # results/phase5_allocations.json
```

For each width set (A = {2, 4, 8}, B = {2, 3, 4, 6, 8}) and budget (an average of 3, 3.5 or 4 bits), it
builds three allocations:

- **rd** (main): the rate–distortion closed form, using c_l from Phase 4.
- **quadratic** (comparison): MAPLE's quadratic moved to bits, using s_l and b*_l.
- **greedy** (reference): the measured Phase 4 table alone.

Each closed-form solution is clamped to [2, 8], rounded to the width set, then adjusted greedily to hit the
budget exactly. Allocations are scored on the same calibration slice as Phase 4; the test split is not
touched. The closed forms are in one marked block at the top of `allocate.py`. They answer the guide's
derivation tasks, so skip that block if you want to derive them first.

**Results** (calibration perplexity; FP32 is 32.24; best per row in bold):

| Budget | Set | Uniform | rd | quadratic | greedy |
|---|---|---|---|---|---|
| 3 | A | – | 502 | **365** | **365** |
| 3 | B | 166 | **142** | 319 | 177 |
| 3.5 | A | – | **75.6** | 88.0 | 90.4 |
| 3.5 | B | – | **45.2** | 65.3 | 45.5 |
| 4 | A | 37.4 | 37.4 | 37.4 | 37.4 |
| 4 | B | 37.4 | 36.6 | 36.8 | **36.3** |

- **rd beats quadratic** at every budget except set A at 3 bits.
- **Set B beats set A** at every budget below 4 bits (at 3.5 bits: 45 vs 76).
- **At 4 bits**, set A can only be uniform. In set B every method beats uniform, but only by 2–3%.
- **At 3 bits**, the sum of single-module damage is far too optimistic again (B/3/rd: predicted 59,
  measured 142).
- **Caveat:** the same data is used to fit the inputs and to score the results. Phase 7 uses the test
  split and LAMBADA.

## File formats

All files are little-endian, uncompressed.

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
`ln_f.bias` [768]. Each tensor's byte offset is in `data/gpt2_124M.json`.

- Linear weights keep the Conv1D layout `[in, out]`, row-major: `y = x @ W + b`.
- `c_attn` outputs `[q | k | v]`, 768 values each; head `h` uses entries `64h .. 64h+63` of each.
- The LM head is `wte` itself (`logits = ln_f(h) @ wte^T`); it is not stored twice.
- Every tensor is a multiple of 64 bytes long, so each starts 64-byte aligned.
- LayerNorm uses ε = 1e-5; GELU is the tanh approximation (`gelu_new`).

### `data/wikitext2_test.u16`, `data/wikitext2_train.u16`

Raw uint16 token ids, no header. WikiText-2 (`wikitext-2-raw-v1`) lines are joined with `"\n\n"` and
tokenized with the GPT-2 tokenizer. Test: 287,644 tokens. Train: 2,428,601. Final perplexities use test;
calibration uses train.

### Reference outputs (described in `data/reference.json`)

- **Prompt:** test tokens [64, 128). It is the first 64-token window whose 100 greedy steps all have a
  top-1 vs top-2 logit margin of at least 0.01 (offset 0 has a near-tie of 0.0011).
- **`ref_prompt.u16`:** the 64 prompt tokens.
- **`ref_logits.f32`:** [64, 50257] float32. Row `p` holds the logits after `prompt[0..p]`.
- **`ref_greedy.u16`:** the 100 greedy tokens after the prompt (argmax, no stop at EOS). Smallest margin:
  0.021.
- **Perplexity:** test tokens [0, 32768) as 32 non-overlapping windows of 1024; each window scores its 1023
  next-token predictions. ppl = exp(mean NLL) = **31.8220**. Each window's mean NLL is in
  `reference.json`, to locate a mismatch.
