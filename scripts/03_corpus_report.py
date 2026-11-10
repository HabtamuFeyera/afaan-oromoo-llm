"""Phase 1, step 3: size report for the processed corpus.

Counts words, characters and GPT-2 BPE tokens (the tokenizer the book uses),
so Phase 2 can compare an Oromo tokenizer against this baseline.

Usage: python scripts/03_corpus_report.py
"""

import json
import re
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def main():
    enc = tiktoken.get_encoding("gpt2")
    report = {}
    splits = ["train", "val"] + (["val_telegram"] if (PROC / "val_telegram.txt").exists() else [])
    for split in splits:
        text = (PROC / f"{split}.txt").read_text(encoding="utf-8")
        words = WORD_RE.findall(text)
        n_tok = len(enc.encode(text, allowed_special={"<|endoftext|>"}))
        report[split] = {
            "characters": len(text),
            "words": len(words),
            "unique_words": len({w.lower() for w in words}),
            "gpt2_tokens": n_tok,
            "gpt2_chars_per_token": round(len(text) / n_tok, 2),
            "gpt2_tokens_per_word": round(n_tok / len(words), 2),
        }

    sample = "Afaan Oromoo afaan ummata Oromoo kan Gaanfa Afrikaa keessa jiraatuuti."
    ids = enc.encode(sample)
    report["gpt2_example"] = {
        "text": sample,
        "tokens": len(ids),
        "pieces": [enc.decode([i]) for i in ids],
    }

    (PROC / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
