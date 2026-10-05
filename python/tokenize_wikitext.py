"""Phase 0, step 3: tokenize WikiText-2 (raw) with the GPT-2 tokenizer and save the ids as uint16.

test  -> final perplexity numbers (touched once per final allocation)
train -> calibration slices for the sensitivity scan and the search (Phase 4 onwards)
"""
import numpy as np

from common import DATA, MODEL_ID, load_tokenizer, sha256, versions, write_json

DATASET, CONFIG = "Salesforce/wikitext", "wikitext-2-raw-v1"
JOIN = "\n\n"  # lines joined as in the Hugging Face perplexity guide


def main():
    from datasets import load_dataset  # after common, which silences the hub's symlink warning

    tok = load_tokenizer()
    ds = load_dataset(DATASET, CONFIG)
    DATA.mkdir(exist_ok=True)
    info = {"dataset": DATASET, "config": CONFIG, "join": JOIN, "tokenizer": MODEL_ID,
            "dtype": "uint16, little-endian", "splits": {}}
    for split in ["test", "train"]:
        text = JOIN.join(ds[split]["text"])
        ids = np.array(tok(text, verbose=False)["input_ids"], dtype=np.int64)
        assert ids.min() >= 0 and ids.max() < 65536
        path = DATA / f"wikitext2_{split}.u16"
        ids.astype("<u2").tofile(path)
        info["splits"][split] = {"file": path.name, "rows": len(ds[split]), "tokens": int(ids.size),
                                 "sha256": sha256(path)}
        print(f"{split}: {len(ds[split]):,} rows -> {ids.size:,} tokens -> {path.relative_to(DATA.parent)}")
    info["versions"] = versions()
    write_json(DATA / "wikitext2.json", info)


if __name__ == "__main__":
    main()
