"""Phase 1, step 2: clean, deduplicate and split the raw Oromo text.

Inputs (from 01_download.py)
- data/raw/afriberta/afaanoromoo/{train,eval}.txt  one shuffled sentence per line
- data/raw/wikipedia/20231101.om/*.parquet          one article per row

- data/interim/telegram_{train,val}.jsonl            optional, from 14_import_telegram.py

Outputs (data/processed/)
- train.txt, val.txt   AfriBERTa sentences one per line, then Wikipedia articles,
                       then Telegram posts (if imported); documents are separated
                       by <|endoftext|> lines
- val_telegram.txt     held-out Telegram posts (only if imported)
- stats.json           counts kept/dropped per rule and per source
- dropped_sample.txt   a few dropped lines per rule, for spot checks

Telegram posts are processed last, so they never change the other sources'
output: train.txt starts with exactly the same text, val.txt is identical, and
any Telegram line that repeats existing text is dropped as a duplicate.

Usage: python scripts/02_build_corpus.py [--no-telegram]
"""

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.contamination import HeadlineFilter  # noqa: E402
from oromo_llm.text import (  # noqa: E402
    ARABIC_RE, BOILERPLATE_RE, CYRILLIC_RE, DOT_LEADER_RE, ENGLISH_STOP, ETHIOPIC_RE, WORD_RE, normalize,
)

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

EOT = "<|endoftext|>"
WIKI_VAL_FRACTION = 0.05
MIN_WORDS = 3

def reject_reason(line: str):
    """Return the rule name that rejects this line, or None to keep it."""
    words = WORD_RE.findall(line)
    if len(words) < MIN_WORDS:
        return "too_short"
    if BOILERPLATE_RE.search(line):
        return "boilerplate"
    if DOT_LEADER_RE.search(line):
        return "dot_leader"
    if CYRILLIC_RE.search(line) or ARABIC_RE.search(line):
        return "other_script"
    letters = sum(ch.isalpha() for ch in line)
    if len(ETHIOPIC_RE.findall(line)) > 0.1 * letters:
        return "ethiopic"
    if letters < 0.5 * len(line):
        return "mostly_non_letters"
    english = sum(w.lower() in ENGLISH_STOP for w in words)
    if english >= 3 and english / len(words) >= 0.25:
        return "english"
    return None


def dedup_key(line: str) -> str:
    """Case- and punctuation-insensitive fingerprint for exact-duplicate removal."""
    core = " ".join(w.lower() for w in WORD_RE.findall(line))
    return hashlib.blake2b(core.encode("utf-8"), digest_size=12).hexdigest()


# --- builders ----------------------------------------------------------------


class Builder:
    def __init__(self):
        self.seen = set()
        self.stats = defaultdict(Counter)
        self.dropped = defaultdict(list)
        # Training lines that contain a MasakhaNEWS dev/test headline are removed,
        # so the topic and headline evaluations stay clean (oromo_llm/contamination.py).
        news_ready = (ROOT / "data" / "news" / "test.jsonl").exists()
        self.headlines = HeadlineFilter() if news_ready else None

    def clean_line(self, raw: str, source: str):
        line = normalize(raw)
        reason = reject_reason(line) if line else "empty"
        if reason is None and self.headlines and "train" in source and self.headlines.contains_headline(line):
            reason = "eval_headline"
        if reason is None:
            key = dedup_key(line)
            if key in self.seen:
                reason = "duplicate"
            else:
                self.seen.add(key)
        if reason:
            self.stats[source][f"dropped_{reason}"] += 1
            if len(self.dropped[reason]) < 15 and line:
                self.dropped[reason].append(f"[{source}] {line[:200]}")
            return None
        self.stats[source]["kept_lines"] += 1
        return line


def afriberta_lines(builder: Builder, split: str):
    path = RAW / "afriberta" / "afaanoromoo" / f"{split}.txt"
    source = f"afriberta_{split}"
    with open(path, encoding="utf-8") as f:
        for raw in f:
            builder.stats[source]["input_lines"] += 1
            line = builder.clean_line(raw, source)
            if line:
                yield line


def wiki_articles(builder: Builder):
    """Yield (split, article_text) for each usable Wikipedia article."""
    path = next((RAW / "wikipedia" / "20231101.om").glob("*.parquet"))
    df = pd.read_parquet(path)
    for art_id, text in zip(df["id"], df["text"]):
        split = "val" if int(hashlib.md5(art_id.encode()).hexdigest(), 16) % 100 < WIKI_VAL_FRACTION * 100 else "train"
        source = f"wikipedia_{split}"
        builder.stats[source]["input_articles"] += 1
        kept = []
        for raw in text.split("\n"):
            if not raw.strip():
                continue
            builder.stats[source]["input_lines"] += 1
            line = builder.clean_line(raw, source)
            if line:
                kept.append(line)
        if kept:
            builder.stats[source]["kept_articles"] += 1
            yield split, "\n".join(kept)


def telegram_posts(builder: Builder, split: str):
    """Yield cleaned Telegram posts (one document each) from 14_import_telegram.py output."""
    path = ROOT / "data" / "interim" / f"telegram_{split}.jsonl"
    source = f"telegram_{split}"
    with open(path, encoding="utf-8") as f:
        for row in f:
            post = json.loads(row)
            builder.stats[source]["input_posts"] += 1
            kept = []
            for raw in post["lines"]:
                builder.stats[source]["input_lines"] += 1
                line = builder.clean_line(raw, source)
                if line:
                    kept.append(line)
            if kept:
                builder.stats[source]["kept_posts"] += 1
                yield "\n".join(kept)


def main():
    use_telegram = "--no-telegram" not in sys.argv and (ROOT / "data" / "interim" / "telegram_train.jsonl").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    b = Builder()

    # Validation text is cleaned first, so any train line that repeats a
    # validation line is the one dropped as a duplicate (no train/val leakage).
    val_parts = list(afriberta_lines(b, "eval"))
    train_parts = []

    wiki = {"train": [], "val": []}
    for split, article in wiki_articles(b):
        wiki[split].append(article)
    # Wikipedia val articles were deduped alongside train ones in id order;
    # that is fine because a repeated line is dropped from only one of them.

    train_parts.extend(afriberta_lines(b, "train"))

    random.Random(0).shuffle(train_parts)  # already shuffled upstream; keep deterministic

    def write(path: Path, sentences, articles):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))
            for art in articles:
                f.write(f"\n{EOT}\n{art}")
            f.write("\n")

    # Telegram last: held-out posts first (so a train post never repeats one), then train posts.
    telegram = {"train": [], "val": []}
    if use_telegram:
        telegram["val"] = list(telegram_posts(b, "val"))
        telegram["train"] = list(telegram_posts(b, "train"))

    write(OUT / "train.txt", train_parts, wiki["train"] + telegram["train"])
    write(OUT / "val.txt", val_parts, wiki["val"])
    outputs = ["train", "val"]
    if use_telegram:
        (OUT / "val_telegram.txt").write_text(f"\n{EOT}\n".join(telegram["val"]) + "\n", encoding="utf-8")
        outputs.append("val_telegram")
    else:
        (OUT / "val_telegram.txt").unlink(missing_ok=True)

    summary = {}
    for name in outputs:
        text = (OUT / f"{name}.txt").read_text(encoding="utf-8")
        summary[name] = {
            "bytes": len(text.encode("utf-8")),
            "characters": len(text),
            "words": len(WORD_RE.findall(text)),
            "lines": text.count("\n"),
        }

    stats = {"sources": {k: dict(v) for k, v in b.stats.items()}, "output": summary}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    with open(OUT / "dropped_sample.txt", "w", encoding="utf-8") as f:
        for reason, lines in sorted(b.dropped.items()):
            f.write(f"=== {reason}\n" + "\n".join(lines) + "\n\n")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
