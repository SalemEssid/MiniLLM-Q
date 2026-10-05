# MiniLLM-Q

A GPT-2 small inference engine in C++, used to measure when a mixed-precision allocation turns into decode
speed on a laptop CPU, with two kernel families (unpack-dequantize and bit-plane) and llama.cpp as an
external baseline. The plan, phases and gates are in [the guide](<document/MiniLLM-Q guide.tex>).

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

## Phase 1: FP32 C++ engine

```
cmake -S cpp -B cpp/build -G Ninja
cmake --build cpp/build
cpp\build\phase1_check.exe       # gates (a)-(d) against the Phase 0 references; writes results/phase1.json
```

- `cpp/src/gpt2.{h,cpp}`: loader, forward pass, KV cache. `forward_sequence(tokens)` and `decode_step(token)`
  share one code path, so a decode step is bit-identical to the same position in a sequence.
- `cpp/src/kernels.{h,cpp}`: `linear`, `layernorm`, `gelu`, `softmax`, `add`.
- Layout choice: the file keeps Conv1D `[in, out]`, and the loader transposes each block weight to
  `[out, in]` so every output of a linear layer is one contiguous dot product, as for the head on `wte` and
  the column-wise layout of the Phase 3 quantized kernels.

## Phase 2: bandwidth, roofline, fast FP32 decode

```
cpp\build\bandwidth.exe [MB]          # read bandwidth at 1, 2, 4, 8 threads (default 256 MB)
cpp\build\phase2_bench.exe            # decode protocol + roofline; writes results/phase2.json
cpp\build\phase1_check.exe --quick    # gates (a)-(c) in seconds, after every kernel change
```

`phase2_bench` follows the guide's protocol: 64-token prompt, 256 decode steps, 1 warm-up and 5 timed runs
per thread count. It records the CPU, the power source, the free RAM and every run's speed. Run it on mains
power for the gate.

**Gate result** (mains power, `results/phase2.json`): bandwidth 34.8 GB/s, ceiling 68.4 tokens/s.
8 threads reach 54.4 ± 2.9 tokens/s, 80% of the ceiling, and 2 or more threads all exceed 70%. The gate asked
for 50–70%. The sweep fixes **8 threads** for all later speed runs.

What it took to reach the bandwidth limit:

- **AVX2 dot product with four accumulators** (`kernels.h`). The compiler's `omp simd` reduction kept one
  accumulator, so every FMA waited for the previous one. Alone, the matrix-vector product now streams at
  the full measured bandwidth (16.9 GB/s on 1 thread, 31.4 GB/s on 4).
- **A thread pool instead of OpenMP** (`threads.h`). With MinGW's libgomp on Windows, entering a parallel
  region costs 40–140 µs, and a decode step enters about 100 of them. Spinning workers start in about 1 µs.
- **Power-throttling opt-out in the benchmarks** (`tools/bench_env.h`). Windows 11 runs background
  processes at efficient clocks, especially on battery. Opting out and raising the priority removed most of
  the run-to-run spread.
- Timing uses `std::chrono::steady_clock`, because `omp_get_wtime()` has 1 ms resolution with MinGW.

Decode still uses the same `dot` as `forward_sequence`, so gate (c) stays bit-identical.

## Phase 3: quantized kernels, two families

```
.venv\Scripts\python python\export_quantized.py      # data/gpt2_q{2,3,4,6,8}.bin (+ gpt2_qbits.json)
.venv\Scripts\python python\make_quant_reference.py  # data/qref/ (gate a), data/quant_reference.json (gate b)
cpp\build\phase3_check.exe [--skip-ppl] [--ppl-widths 8,4]   # gates (a), (b); results/phase3.json
cpp\build\phase3_bench.exe [--decode]                         # gate (c); results/phase3_bench.json
```

- **Format** (`python/quant.py`, `cpp/src/quant.h`): asymmetric, groups of 128 inputs, FP16 scale and
  minimum per group and output column. Python quantizes; C++ only repacks the exported codes.
- **Family U** (unpack-dequantize): codes are split into byte-aligned fields (3 = 2 + 1 and 6 = 4 + 2 bits,
  as in llama.cpp's Q3_K and Q6_K). One 16-byte load, a shift and a mask give one field of 16 codes; then
  convert to float and FMA, with AVX-512.
- **Family P** (bit-plane): each bit of the codes is a plane. Every 4 inputs get a 16-entry FP32 table of
  partial sums, built once per product, and `_mm512_permutexvar_ps` looks up 16 output columns at once.
  There are no multiplications inside a group, and the same kernel serves every width.
- `qmatvec_reference` decodes every code from the packed layout and sums in double. It checks the packing
  against Python, and the AVX-512 kernels against it.
- An allocation is a `Quantization`: 48 widths (0 = FP32), the head at FP32 or 8 bits, and a family. The
  FP32 weights of quantized layers are never loaded.

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
- **`ref_prompt.u16`:** the 64 prompt tokens.
- **`ref_logits.f32`:** [64, 50257] float32. Row `p` holds the logits after `prompt[0..p]`: row 0 after the
  first token, row 63 after the whole prompt.
- **`ref_greedy.u16`:** the 100 greedy tokens after the prompt (argmax at every step, no stop at EOS). The
  smallest margin is 0.021.
- **Perplexity:** test tokens [0, 32768) as 32 non-overlapping windows of 1024. Each window scores its 1023
  next-token predictions, and ppl = exp(mean NLL) = **31.8220**. The mean NLL of each window is in
  `reference.json`, to locate a mismatch.
