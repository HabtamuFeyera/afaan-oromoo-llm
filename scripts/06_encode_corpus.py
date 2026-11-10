"""Phase 2, step 3: encode the corpus once into token-ID files for Phase 3.

Writes data/tokens/{train,val}.bin (uint16 token IDs, readable with
numpy.memmap) and data/tokens/meta.json.

Lines are encoded in parallel and joined with the newline token. This gives
exactly the same IDs as encoding the whole file at once, because the
pre-tokenizer always splits a lone "\\n" before a word into its own piece
(checked on a sample below).

Usage: python scripts/06_encode_corpus.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.tokenizer import EOT, get_tokenizer  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "tokens"


def encode_file(tok, path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    newline_id = tok.encode("\n")
    assert len(newline_id) == 1, newline_id

    # EOT lines are encoded directly; everything else goes through the batch encoder.
    encoded = tok.encode_batch([l for l in lines if l != EOT])
    it = iter(encoded)
    ids: list[int] = []
    for i, line in enumerate(lines):
        if i:
            ids.extend(newline_id)
        ids.extend([tok.eot_token] if line == EOT else next(it))

    sample = "\n".join(lines[:3000] + [EOT] + lines[-500:])
    sample_ids = tok.encode(sample)
    expected = ids[: len(tok.encode("\n".join(lines[:3000])))]
    assert sample_ids[: len(expected)] == expected, "line-wise encoding differs from whole-text encoding"

    arr = np.array(ids, dtype=np.uint16)
    assert arr.max() < tok.n_vocab
    return arr


def main():
    tok = get_tokenizer()
    assert tok.n_vocab < 2**16
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": "tokenizer/oromo-bpe.json", "vocab_size": tok.n_vocab,
            "eot_token": tok.eot_token, "dtype": "uint16"}
    splits = ["train", "val"] + (["val_telegram"] if (PROC / "val_telegram.txt").exists() else [])
    for split in splits:
        arr = encode_file(tok, PROC / f"{split}.txt")
        arr.tofile(OUT / f"{split}.bin")
        meta[f"{split}_tokens"] = int(arr.size)
        print(f"{split}: {arr.size:,} tokens -> data/tokens/{split}.bin")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
