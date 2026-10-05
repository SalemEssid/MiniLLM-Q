"""Phase 0, step 4: reference outputs that the C++ engine must reproduce (Phase 1 gate).

Prompt: the first 64-token window of data/wikitext2_test.u16 (offsets 0, 64, 128, ...) whose greedy
continuation has every top-1 minus top-2 logit margin >= 0.01. At a near-tie, a correct C++ engine (logit
error up to 1e-3 under gate (a)) could pick the other token and fail gate (b) by chance.
  ref_prompt.u16  [64] the prompt tokens
  ref_logits.f32  [64, 50257] float32: row p holds the logits after prompt[0..p]
                  (row 0 = after the first token, row 63 = after the whole prompt)
  ref_greedy.u16  [100] greedy continuation of the prompt (argmax every step, no stop at EOS)
  reference.json  prompt, greedy tokens with their top-1 minus top-2 margins, perplexity reference, versions
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, MODEL_ID, VOCAB, load_model, load_tokenizer, sha256, versions, write_json

PROMPT_LEN, N_GREEDY = 64, 100
MIN_MARGIN = 0.01
PPL_TOKENS, WINDOW = 32 * 1024, 1024  # non-overlapping windows: stride = window
CHUNK = 256                           # head applied 256 positions at a time to keep memory low
SEED = 0


def greedy_decode(m, prompt):
    """Logits at every prompt position, then N_GREEDY argmax steps with the KV cache."""
    out = m(prompt[None], use_cache=True)
    prompt_logits = out.logits[0]
    tokens, margins = [], []
    for _ in range(N_GREEDY):
        top = torch.topk(out.logits[0, -1], 2)
        tokens.append(int(top.indices[0]))
        margins.append(float(top.values[0] - top.values[1]))
        out = m(top.indices[:1][None], past_key_values=out.past_key_values, use_cache=True)
    return prompt_logits, tokens, margins


@torch.inference_mode()
def main():
    torch.manual_seed(SEED)  # nothing below is random; the seed is fixed anyway and recorded
    m, tok = load_model(), load_tokenizer()
    test = np.fromfile(DATA / "wikitext2_test.u16", dtype="<u2").astype(np.int64)

    rejected = []
    for start in range(0, len(test) - PROMPT_LEN, PROMPT_LEN):
        prompt = torch.from_numpy(test[start:start + PROMPT_LEN])
        prompt_logits, greedy, margins = greedy_decode(m, prompt)
        if min(margins) >= MIN_MARGIN:
            break
        rejected.append({"offset": start, "min_margin": min(margins), "step": int(np.argmin(margins))})
    prompt.numpy().astype("<u2").tofile(DATA / "ref_prompt.u16")
    prompt_logits.numpy().astype("<f4").tofile(DATA / "ref_logits.f32")
    np.array(greedy, dtype="<u2").tofile(DATA / "ref_greedy.u16")

    # Cross-check: one uncached pass over prompt + continuation picks the same tokens.
    full = torch.cat([prompt, torch.tensor(greedy)])
    assert m(full[None]).logits[0, PROMPT_LEN - 1:-1].argmax(-1).tolist() == greedy, \
        "cached and uncached greedy decoding disagree"

    # Perplexity: each window predicts its tokens 1..1023 from the tokens before them in the same window.
    windows = torch.from_numpy(test[:PPL_TOKENS]).view(-1, WINDOW)
    nll, t0 = [], time.perf_counter()
    for w in windows:
        h = m.transformer(w[None]).last_hidden_state[0, :-1]
        total = sum(F.cross_entropy(m.lm_head(h[i:i + CHUNK]), w[1 + i:1 + i + CHUNK], reduction="sum").item()
                    for i in range(0, WINDOW - 1, CHUNK))
        nll.append(total / (WINDOW - 1))
    seconds = time.perf_counter() - t0
    ppl = math.exp(sum(nll) / len(nll))

    write_json(DATA / "reference.json", {
        "model": MODEL_ID,
        "revision": getattr(m.config, "_commit_hash", None),
        "attn_implementation": m.config._attn_implementation,
        "seed": SEED,
        "torch_threads": torch.get_num_threads(),
        "prompt": {"file": "ref_prompt.u16", "sha256": sha256(DATA / "ref_prompt.u16"),
                   "source": f"wikitext2_test.u16[{start}:{start + PROMPT_LEN}]", "tokens": prompt.tolist(),
                   "text": tok.decode(prompt),
                   "rule": f"first 64-token window at offsets 0, 64, ... whose greedy margins are all >= {MIN_MARGIN}",
                   "rejected": rejected},
        "logits": {"file": "ref_logits.f32", "shape": [PROMPT_LEN, VOCAB], "dtype": "float32, little-endian",
                   "rows": "row p = logits after prompt[0..p]", "sha256": sha256(DATA / "ref_logits.f32")},
        "greedy": {"file": "ref_greedy.u16", "dtype": "uint16, little-endian",
                   "sha256": sha256(DATA / "ref_greedy.u16"), "tokens": greedy,
                   "text": tok.decode(greedy), "min_margin": min(margins), "margins": margins},
        "perplexity": {"tokens": f"wikitext2_test.u16[0:{PPL_TOKENS}]", "window": WINDOW, "stride": WINDOW,
                       "predictions_per_window": WINDOW - 1, "ppl": ppl, "window_mean_nll": nll,
                       "seconds": round(seconds, 1)},
        "versions": versions(),
    })
    print(f"prompt at offset {start} ({len(rejected)} earlier windows rejected): {tok.decode(prompt)!r}")
    print(f"greedy: {tok.decode(greedy)!r}")
    print(f"smallest greedy margin: {min(margins):.4f} (step {int(np.argmin(margins))})")
    print(f"perplexity on {PPL_TOKENS:,} test tokens (window = stride = {WINDOW}): {ppl:.4f} in {seconds:.0f} s")


if __name__ == "__main__":
    main()
