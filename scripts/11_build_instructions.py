"""Phase 5, step 1: build an Oromo instruction dataset from existing Oromo data.

No Oromo instruction dataset exists, and translating the book's English one
would bring in translation errors. Instead, instruction tasks are made from
real Oromo text, the way the FLAN datasets did it. Every answer is
human-written Oromo; only the short instruction templates below are written
here, and a native speaker should review them.

Tasks
- topic:     MasakhaNEWS article -> its topic, as an Oromo word
- headline:  MasakhaNEWS article -> its real headline
- describe:  Wikipedia title -> the article's first sentence

Splits (no leakage between them)
- train: news train articles + Wikipedia articles from the pretraining train split
- val:   news dev articles + 60 held-out Wikipedia train-split articles
- test:  news test articles + Wikipedia articles from the pretraining val split
         (never seen in pretraining)

Outputs: data/instructions/{train,val,test}.json (book chapter 7 format:
instruction / input / output, plus a "task" field)
Usage: python scripts/11_build_instructions.py
"""

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.text import WORD_RE, normalize  # noqa: E402

NEWS = ROOT / "data" / "news"
WIKI = ROOT / "data" / "raw" / "wikipedia" / "20231101.om"
OUT = ROOT / "data" / "instructions"

TOPIC_OROMO = {
    "politics": "siyaasa",
    "health": "fayyaa",
    "sports": "ispoortii",
    "entertainment": "bashannanaa",
    "technology": "teeknooloojii",
}
CHOICES = "siyaasa, fayyaa, ispoortii, bashannanaa, teeknooloojii"

TEMPLATES = {
    "topic": [
        f"Oduun armaan gadii waa'ee maalii? Tokko filadhu: {CHOICES}.",
        f"Oduu armaan gadii kana ramaddii isaa barreessi: {CHOICES}.",
    ],
    "headline": [
        "Oduu armaan gadiitiif mata duree gabaabaa barreessi.",
        "Mata duree barruu armaan gadii maal ta'uu danda'a?",
    ],
    "describe": [
        "Waa'ee {title} ibsa gabaabaa kenni.",
        "Waa'ee {title} hima tokkoon ibsi.",
    ],
}

# Articles are truncated so that instruction + article + answer fit the 256-token
# context; the truncation itself happens at tokenization time (oromo_llm/instruct.py).
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
WIKI_VAL_FRACTION = 0.05  # must match scripts/02_build_corpus.py


def wiki_split(art_id):
    return "val" if int(hashlib.md5(art_id.encode()).hexdigest(), 16) % 100 < WIKI_VAL_FRACTION * 100 else "train"


def first_sentence(text):
    """First sentence of the first real paragraph, or None if it is not a usable definition."""
    for para in text.split("\n"):
        para = normalize(para)
        if len(WORD_RE.findall(para)) < 6:
            continue
        sent = SENTENCE_END_RE.split(para)[0].strip()
        n = len(WORD_RE.findall(sent))
        if 6 <= n <= 45 and sent.endswith((".", "!", "?")):
            return sent
        return None
    return None


def news_examples(split, rng):
    rows = [json.loads(l) for l in (NEWS / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    out = []
    for r in rows:
        headline, _, body = r["text"].partition("\n")
        out.append({"task": "topic", "instruction": rng.choice(TEMPLATES["topic"]),
                    "input": body, "output": TOPIC_OROMO[r["label"]]})
        out.append({"task": "headline", "instruction": rng.choice(TEMPLATES["headline"]),
                    "input": body, "output": headline.strip()})
    return out


def wiki_examples(rng):
    df = pd.read_parquet(next(WIKI.glob("*.parquet")))
    by_split = {"train": [], "val": []}
    for art_id, title, text in zip(df["id"], df["title"], df["text"]):
        sent = first_sentence(text)
        if not sent:
            continue
        title = normalize(title)
        by_split[wiki_split(art_id)].append({
            "task": "describe",
            "instruction": rng.choice(TEMPLATES["describe"]).format(title=title),
            "input": "", "output": sent,
        })
    return by_split


def main():
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)
    wiki = wiki_examples(rng)
    rng.shuffle(wiki["train"])
    splits = {
        "train": news_examples("train", rng) + wiki["train"][60:],
        "val": news_examples("dev", rng) + wiki["train"][:60],
        "test": news_examples("test", rng) + wiki["val"],
    }
    for name, rows in splits.items():
        if name == "train":
            rng.shuffle(rows)
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {len(rows)} examples {dict(Counter(r['task'] for r in rows))}")


if __name__ == "__main__":
    main()
