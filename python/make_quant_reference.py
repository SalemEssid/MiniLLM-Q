"""Phase 3 references for the C++ quantized kernels.

Gate (a), data/qref/: y = x @ fake_quant(W, b) in float64 for test modules of every shape, at every width.
  x768.f32, x3072.f32               inputs, float32, N(0, 1) with a fixed seed
  y_m{index}_q{b}.f64               outputs, float64; index into MODULES, 48 = the head (8 bits only)
Gate (b), data/quant_reference.json: perplexity with all 48 block modules fake-quantized to b bits (head in
FP32), on the same 32 windows of 1024 test tokens as the Phase 1 reference.
"""
import math
import time

import numpy as np
import torch

from common import DATA, load_model, versions, window_nll, write_json
from quant import MODULES, WIDTHS, conv1d, fake_quant

TEST_MODULES = [2, 3, 44, 45]  # h.0.mlp.c_fc (768->3072), h.0.mlp.c_proj (3072->768), h.11.attn.c_attn, h.11.attn.c_proj
HEAD = 48
PPL_WIDTHS = [8, 4]
PPL_TOKENS, WINDOW = 32 * 1024, 1024
SEED = 0


@torch.inference_mode()
def kernel_references(model):
    out = DATA / "qref"
    out.mkdir(exist_ok=True)
    gen = torch.Generator().manual_seed(SEED)
    xs = {n: torch.randn(n, generator=gen) for n in (768, 3072)}
    for n, x in xs.items():
        x.numpy().astype("<f4").tofile(out / f"x{n}.f32")
    cases = [(i, conv1d(model, MODULES[i]).weight, WIDTHS) for i in TEST_MODULES]
    cases.append((HEAD, model.transformer.wte.weight.T.contiguous(), [8]))
    for index, W, widths in cases:
        for b in widths:
            y = xs[W.shape[0]].double() @ fake_quant(W.float(), b).double()
            y.numpy().astype("<f8").tofile(out / f"y_m{index}_q{b}.f64")
    print(f"kernel references: {len(TEST_MODULES)} modules x {len(WIDTHS)} widths + the head at 8 bits")


def uniform_perplexity(bits, tokens):
    model = load_model()
    with torch.inference_mode():
        for name in MODULES:
            w = conv1d(model, name).weight
            w.copy_(fake_quant(w.float(), bits))
    t0 = time.perf_counter()
    nll = window_nll(model, tokens, WINDOW)
    ppl = math.exp(sum(nll) / len(nll))
    print(f"uniform {bits}-bit blocks, FP32 head: perplexity {ppl:.4f} ({time.perf_counter() - t0:.0f} s)")
    return ppl, nll


def main():
    kernel_references(load_model())
    test = torch.from_numpy(np.fromfile(DATA / "wikitext2_test.u16", dtype="<u2").astype(np.int64))
    ref = {"tokens": f"wikitext2_test.u16[0:{PPL_TOKENS}]", "window": WINDOW, "stride": WINDOW, "head": "FP32"}
    for b in PPL_WIDTHS:
        ppl, nll = uniform_perplexity(b, test[:PPL_TOKENS])
        ref[f"ppl_q{b}"] = ppl
        ref[f"window_mean_nll_q{b}"] = nll
    ref["versions"] = versions()
    write_json(DATA / "quant_reference.json", ref)


if __name__ == "__main__":
    main()
