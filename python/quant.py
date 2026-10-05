"""The quantization format of the guide (Section 6.1), shared by the export, the references and later the scans.

Asymmetric round-to-nearest with groups of G = 128 input channels: W ~= s * q + m, with an FP16 scale s and
minimum m per group and output column, and codes q in 0..2^b - 1.
"""
import torch

G = 128
WIDTHS = [2, 3, 4, 6, 8]
KINDS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
# The 48 block modules in allocation order: block 0's four modules, then block 1's, ...
MODULES = [f"transformer.h.{i}.{k}" for i in range(12) for k in KINDS]


def quantize(W, bits, G=G):
    """W: [in, out] (Conv1D layout). Returns codes [in/G, G, out] uint8 and FP16-exact scales, mins [in/G, 1, out]."""
    qmax = 2**bits - 1
    Wg = W.reshape(-1, G, W.shape[1])  # [in/G, G, out]: groups along the input dim
    m = Wg.amin(dim=1, keepdim=True).half().float()  # FP16, exactly as stored
    s = ((Wg.amax(dim=1, keepdim=True) - m) / qmax).clamp_min(1e-6).half().float()
    q = torch.clamp(torch.round((Wg - m) / s), 0, qmax)
    return q.to(torch.uint8), s, m


def fake_quant(W, bits, G=G):
    q, s, m = quantize(W, bits, G)
    return (q.float() * s + m).reshape_as(W)


def conv1d(model, name):
    """The Conv1D module called name; its weight is [in, out]."""
    return model.get_submodule(name)
