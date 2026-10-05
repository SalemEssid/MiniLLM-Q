"""Phase 3: quantize every block module once per width, and the head at 8 bits, into data/gpt2_q{b}.bin.

Quantizing a module depends only on its weights and its width, so an allocation is just a list of 48 widths
and the C++ loader takes each module from the file of its width. File layout, little-endian:
  header  64 bytes: int32 magic 20261006, bits, G, n_tensors, then zeros
  then for every tensor, in MODULES order (and the head last, in the 8-bit file only):
    codes   uint8   [in, out]    one code per weight, unpacked, Conv1D layout
    scales  float16 [in/G, out]
    mins    float16 [in/G, out]
The head is wte^T, [768, 50257]. C++ repacks the codes into the layout of its kernel family at load time.
"""
import struct

import numpy as np
import torch

from common import DATA, load_model, sha256, versions, write_json
from quant import G, MODULES, WIDTHS, conv1d, quantize

MAGIC = 20261006
HEADER_BYTES = 64
COLUMN_CHUNK = 4096  # quantize this many output columns at a time to keep memory low (the head has 50,257)


def write_tensor(f, W, bits):
    """Quantize W [in, out] column chunk by column chunk; columns are independent."""
    codes, scales, mins = [], [], []
    for c in range(0, W.shape[1], COLUMN_CHUNK):
        q, s, m = quantize(W[:, c:c + COLUMN_CHUNK].contiguous(), bits)
        codes.append(q.reshape(W.shape[0], -1).numpy())
        scales.append(s[:, 0].half().numpy())
        mins.append(m[:, 0].half().numpy())
    for part, dtype in ((codes, "<u1"), (scales, "<f2"), (mins, "<f2")):
        f.write(np.concatenate(part, axis=1).astype(dtype).tobytes())


@torch.inference_mode()
def main():
    model = load_model()
    tensors = [(name, conv1d(model, name).weight) for name in MODULES]
    head = ("head (wte^T)", model.transformer.wte.weight.T)
    manifest = {"magic": MAGIC, "G": G, "header_bytes": HEADER_BYTES,
                "tensor_layout": "codes uint8 [in, out], scales float16 [in/G, out], mins float16 [in/G, out]",
                "files": {}}
    for bits in WIDTHS:
        todo = tensors + ([head] if bits == 8 else [])
        path = DATA / f"gpt2_q{bits}.bin"
        entries = []
        with open(path, "wb") as f:
            f.write(struct.pack("<4i", MAGIC, bits, G, len(todo)).ljust(HEADER_BYTES, b"\0"))
            for name, W in todo:
                entries.append({"name": name, "shape": list(W.shape), "offset": f.tell()})
                write_tensor(f, W.float(), bits)
        manifest["files"][str(bits)] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
                                        "tensors": entries}
        print(f"wrote {path.relative_to(DATA.parent)}: {len(todo)} tensors, {path.stat().st_size / 1e6:.1f} MB")
    manifest["versions"] = versions()
    write_json(DATA / "gpt2_qbits.json", manifest)


if __name__ == "__main__":
    main()
