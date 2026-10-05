"""Phase 0, step 2: export GPT-2 small to data/gpt2_124M.bin and check that it loads back bit for bit."""
import numpy as np
import torch

from common import (DATA, HEADER_BYTES, MODEL_ID, N_PARAMS, TENSORS, header_bytes, layout, load_model,
                    read_weights, sha256, versions, write_json)


def main():
    m = load_model()
    sd = m.state_dict()
    names = [name for name, _ in TENSORS]

    # What is left out must be the tied head or a non-weight buffer (older transformers keep attn.bias masks).
    extra = set(sd) - set(names)
    assert all(k == "lm_head.weight" or k.endswith(("attn.bias", "attn.masked_bias")) for k in extra), extra
    assert torch.equal(sd["lm_head.weight"], sd["transformer.wte.weight"]), "head is not tied to wte"
    for name, shape in TENSORS:
        assert tuple(sd[name].shape) == shape, (name, tuple(sd[name].shape), shape)
    assert sum(sd[n].numel() for n in names) == N_PARAMS

    DATA.mkdir(exist_ok=True)
    path = DATA / "gpt2_124M.bin"
    with open(path, "wb") as f:
        f.write(header_bytes())
        for name in names:
            f.write(sd[name].float().numpy().astype("<f4").tobytes())

    # Gate: every tensor read back from the file equals the state_dict bit for bit.
    back = read_weights(path)
    for name in names:
        assert np.array_equal(back[name].view("<u4"), sd[name].numpy().view("<u4")), name
    del back

    write_json(DATA / "gpt2_124M.json", {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "model": MODEL_ID,
        "revision": getattr(m.config, "_commit_hash", None),
        "header": {"bytes": HEADER_BYTES, "int32_fields": ["magic", "n_layer", "n_head", "d_model", "vocab", "n_ctx"],
                   "padding": "zeros up to 64 bytes"},
        "dtype": "float32, little-endian",
        "linear_layout": "Conv1D [in, out]; logits = ln_f(h) @ wte^T (head tied to wte, not stored twice)",
        "n_params": N_PARAMS,
        "tensors": [{"name": n, "shape": list(s), "offset": o} for n, s, o in layout()],
        "versions": versions(),
    })
    print(f"wrote {path.relative_to(DATA.parent)}: {len(names)} tensors, {N_PARAMS:,} params, "
          f"{path.stat().st_size:,} bytes; read-back matches bit for bit")


if __name__ == "__main__":
    main()
