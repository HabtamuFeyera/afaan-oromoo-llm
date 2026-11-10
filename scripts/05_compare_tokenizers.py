"""Phase 2, step 2: compare tokenizers on the held-out validation text.

Compared: GPT-2 (the book's tokenizer), NaolBM/Oromo-BBPE (an existing 12k Oromo
tokenizer), and our oromo-bpe-<size>.json files.

Metrics (validation text, <|endoftext|> lines removed)
- chars/token, tokens/word: compression; more chars per token = shorter sequences
- whole-word %: share of word occurrences that are a single token
- round-trip: decode(encode(text)) == text
- rare-token %: share of the vocabulary seen fewer than 50 times in the training
  text. A small model learns little about those tokens.
- embedding params: vocab x 384 (the planned Phase 3 width)

Writes tokenizer/comparison.json.
Usage: python scripts/05_compare_tokenizers.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import tiktoken
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
TOKDIR = ROOT / "tokenizer"
EOT = "<|endoftext|>"
EMB_DIM = 384
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def load_text(split):
    lines = (PROC / f"{split}.txt").read_text(encoding="utf-8").split("\n")
    return [l for l in lines if l and l != EOT]


class HF:
    def __init__(self, name, path):
        self.name, self.tok = name, Tokenizer.from_file(str(path))
        self.vocab_size = self.tok.get_vocab_size()

    def encode_batch(self, lines):
        return [e.ids for e in self.tok.encode_batch(lines, add_special_tokens=False)]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def encode_word(self, word):
        return self.tok.encode(word, add_special_tokens=False).ids


class TikToken:
    def __init__(self):
        self.name, self.enc = "gpt2 (book)", tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

    def encode_batch(self, lines):
        return self.enc.encode_ordinary_batch(lines)

    def decode(self, ids):
        return self.enc.decode(ids)

    def encode_word(self, word):
        return self.enc.encode_ordinary(word)


def evaluate(tk, val_lines, train_lines, word_counts):
    val_ids = tk.encode_batch(val_lines)
    n_tok = sum(map(len, val_ids))
    n_chars = sum(map(len, val_lines))
    n_words = sum(word_counts.values())

    # Words appear mid-sentence with a leading space, so measure them that way.
    whole = sum(c for w, c in word_counts.items() if len(tk.encode_word(" " + w)) == 1)

    sample = val_lines[:2000]
    roundtrip = all(tk.decode(ids) == line for ids, line in zip(tk.encode_batch(sample), sample))

    freq = Counter()
    for ids in tk.encode_batch(train_lines):
        freq.update(ids)
    rare = sum(1 for i in range(tk.vocab_size) if freq[i] < 50)

    return {
        "vocab_size": tk.vocab_size,
        "val_tokens": n_tok,
        "chars_per_token": round(n_chars / n_tok, 2),
        "tokens_per_word": round(n_tok / n_words, 2),
        "whole_word_pct": round(100 * whole / n_words, 1),
        "roundtrip_ok": roundtrip,
        "rare_token_pct": round(100 * rare / tk.vocab_size, 1),
        "embedding_params_M": round(tk.vocab_size * EMB_DIM / 1e6, 1),
    }


def main():
    val_lines = load_text("val")
    train_lines = load_text("train")
    word_counts = Counter(w for l in val_lines for w in WORD_RE.findall(l))

    naol = hf_hub_download("NaolBM/Oromo-BBPE", "tokenizer.json", local_dir=TOKDIR / "naolbm-oromo-bbpe")
    tokenizers = [TikToken(), HF("NaolBM/Oromo-BBPE", naol)]
    for p in sorted(TOKDIR.glob("oromo-bpe-*.json"), key=lambda p: int(p.stem.split("-")[-1])):
        tokenizers.append(HF(p.stem, p))

    results = {}
    header = f"{'tokenizer':<20}{'vocab':>7}{'chars/tok':>10}{'tok/word':>9}{'whole%':>8}{'rare%':>7}{'emb M':>7}  roundtrip"
    print(header)
    for tk in tokenizers:
        r = evaluate(tk, val_lines, train_lines, word_counts)
        results[tk.name] = r
        print(f"{tk.name:<20}{r['vocab_size']:>7}{r['chars_per_token']:>10}{r['tokens_per_word']:>9}"
              f"{r['whole_word_pct']:>8}{r['rare_token_pct']:>7}{r['embedding_params_M']:>7}  {r['roundtrip_ok']}")

    (TOKDIR / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
