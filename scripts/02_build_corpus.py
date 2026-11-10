"""Phase 1, step 2: clean, deduplicate and split the raw Oromo text.

Inputs (from 01_download.py)
- data/raw/afriberta/afaanoromoo/{train,eval}.txt  one shuffled sentence per line
- data/raw/wikipedia/20231101.om/*.parquet          one article per row

Outputs (data/processed/)
- train.txt, val.txt   AfriBERTa sentences one per line, then Wikipedia articles;
                       documents are separated by <|endoftext|> lines
- stats.json           counts kept/dropped per rule and per source
- dropped_sample.txt   a few dropped lines per rule, for spot checks

Usage: python scripts/02_build_corpus.py
"""

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

EOT = "<|endoftext|>"
WIKI_VAL_FRACTION = 0.05
MIN_WORDS = 3

# --- character normalization -------------------------------------------------

# Every apostrophe-like mark becomes ASCII ' so the hudhaa is one symbol.
APOSTROPHES = "’‘´`ʼʻ′‛"
DOUBLE_QUOTES = "“”„‟«»″"
HYPHENS = "‐‑‒­"
ZERO_WIDTH = "​‌‍⁠﻿"

CHAR_MAP = str.maketrans(
    {**{c: "'" for c in APOSTROPHES},
     **{c: '"' for c in DOUBLE_QUOTES},
     **{c: "-" for c in HYPHENS},
     **{c: "" for c in ZERO_WIDTH},
     "…": "...",
     " ": " "}
)

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
CITATION_RE = re.compile(r"\[\d+(?:[,–-]\d+)*\]")  # [3], [3,4], [3-5]
READ_MORE_RE = re.compile(r"\[?\s*Read More\s*\]?", re.IGNORECASE)
BBC_SUFFIX_RE = re.compile(r"\s*-\s*BBC News Afaan Oromoo\s*$")
MULTISPACE_RE = re.compile(r"[ \t]+")

# --- line filters ------------------------------------------------------------

# Page residue from news sites. Labels are stripped (the sentence around them
# is real Oromo); lines that are only site furniture are dropped.
LABEL_RE = re.compile(
    r"Qabiyyeetti darbi Kutaawwan(?: Oduu Itoophiyaa Viidiyoo Jajjaboo)*"
    r"|Madda suuraa,?(?: (?:Getty Images|Reuters|AFP|EPA|AP|BBC|Social Media|Facebook|Twitter|[A-Z][\w']*(?: [A-Z][\w']*)?))?,?"
    r"|Ibsa waa'ee suuraa,?"
    r"|Getty Images",
    re.IGNORECASE,
)
BOILERPLATE_RE = re.compile(r"Editorial Guidelines|Be the first to comment", re.IGNORECASE)
ELLIPSIS_RUN_RE = re.compile(r"\.{4,9}")
DOT_LEADER_RE = re.compile(r"\.{10,}|(?:\. ){4,}")  # glossary/TOC formatting
ETHIOPIC_RE = re.compile(r"[ሀ-᎟ⶀ-⷟]")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
ARABIC_RE = re.compile(r"[؀-ۿ]")
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")

# Common English function words that are not also Oromo words.
ENGLISH_STOP = {
    "the", "and", "of", "is", "to", "that", "with", "for", "this", "are",
    "was", "were", "be", "by", "from", "which", "have", "has", "it", "on",
    "at", "as", "or", "not", "but", "they", "their", "will", "would", "can",
    "been", "about", "into", "who", "what", "when", "where", "there",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(CHAR_MAP)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = URL_RE.sub(" ", text)
    text = CITATION_RE.sub("", text)
    text = READ_MORE_RE.sub(" ", text)
    text = BBC_SUFFIX_RE.sub("", text)
    text = LABEL_RE.sub(" ", text)
    text = ELLIPSIS_RUN_RE.sub("...", text)
    return MULTISPACE_RE.sub(" ", text).strip()


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

    def clean_line(self, raw: str, source: str):
        line = normalize(raw)
        reason = reject_reason(line) if line else "empty"
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


def main():
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

    write(OUT / "train.txt", train_parts, wiki["train"])
    write(OUT / "val.txt", val_parts, wiki["val"])

    summary = {}
    for name in ("train", "val"):
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
