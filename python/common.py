"""Paths, the gpt2_124M.bin layout and small helpers shared by the Python scripts.

The C++ loader mirrors TENSORS: same names, same order, same shapes.
"""
import hashlib
import json
import os
import platform
import struct
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL_ID = "openai-community/gpt2"

MAGIC = 20261005
N_LAYER, N_HEAD, D, VOCAB, N_CTX = 12, 12, 768, 50257, 1024
# Six int32 then zeros. Every tensor is a multiple of 64 bytes long, so all of them start 64-byte aligned.
HEADER_BYTES = 64


def _block(i):
    p = f"transformer.h.{i}."
    return [
        (p + "ln_1.weight", (D,)), (p + "ln_1.bias", (D,)),
        (p + "attn.c_attn.weight", (D, 3 * D)), (p + "attn.c_attn.bias", (3 * D,)),
        (p + "attn.c_proj.weight", (D, D)), (p + "attn.c_proj.bias", (D,)),
        (p + "ln_2.weight", (D,)), (p + "ln_2.bias", (D,)),
        (p + "mlp.c_fc.weight", (D, 4 * D)), (p + "mlp.c_fc.bias", (4 * D,)),
        (p + "mlp.c_proj.weight", (4 * D, D)), (p + "mlp.c_proj.bias", (D,)),
    ]


# (name, shape) in file order. Linear weights keep the Conv1D layout [in, out].
# lm_head.weight is not written: it is the same tensor as wte.
TENSORS = (
    [("transformer.wte.weight", (VOCAB, D)), ("transformer.wpe.weight", (N_CTX, D))]
    + [t for i in range(N_LAYER) for t in _block(i)]
    + [("transformer.ln_f.weight", (D,)), ("transformer.ln_f.bias", (D,))]
)
N_PARAMS = 124_439_808


def header_bytes():
    return struct.pack("<6i", MAGIC, N_LAYER, N_HEAD, D, VOCAB, N_CTX).ljust(HEADER_BYTES, b"\0")


def layout():
    """(name, shape, byte offset) for every tensor in gpt2_124M.bin."""
    out, off = [], HEADER_BYTES
    for name, shape in TENSORS:
        out.append((name, shape, off))
        off += 4 * int(np.prod(shape))
    return out


def read_weights(path):
    """Memory-map gpt2_124M.bin and return {name: float32 array}, checking the header and the size."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    assert bytes(raw[:HEADER_BYTES]) == header_bytes(), "header mismatch"
    out = {}
    for name, shape, off in layout():
        out[name] = np.frombuffer(raw, dtype="<f4", count=int(np.prod(shape)), offset=off).reshape(shape)
    assert raw.size == HEADER_BYTES + 4 * N_PARAMS, f"file is {raw.size} bytes"
    return out


def load_model():
    from transformers import GPT2LMHeadModel

    # Eager attention: an explicit softmax(QK^T/sqrt(d))V, the same math the C++ engine implements.
    return GPT2LMHeadModel.from_pretrained(MODEL_ID, attn_implementation="eager").eval()


def load_tokenizer():
    from transformers import GPT2TokenizerFast

    return GPT2TokenizerFast.from_pretrained(MODEL_ID)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    import datasets
    import tokenizers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
    }


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
