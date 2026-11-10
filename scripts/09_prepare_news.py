"""Phase 4, step 1: prepare MasakhaNEWS Oromo (news topic classification).

Input: data/raw/masakhanews/data/orm/{train,dev,test}.tsv, downloaded here if missing
       (masakhane/masakhanews, AFL-3.0). Five topics: entertainment, health,
       politics, sports, technology.

Each example is the headline plus the article text, cleaned with the same
normalize() as the pretraining corpus (so the hudhaa is always ').

Contamination check: the articles are BBC Afaan Oromoo stories, and so is much of
the AfriBERTa pretraining text. For each article we measure the share of its
13-word sequences that also appear in data/processed/train.txt (the method from
the GPT-3 paper). An article is flagged as "seen" when at least half of its
13-grams appear there.

Outputs: data/news/{train,dev,test}.jsonl and data/news/contamination.json
Usage: python scripts/09_prepare_news.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.text import WORD_RE, normalize  # noqa: E402

RAW = ROOT / "data" / "raw" / "masakhanews"
OUT = ROOT / "data" / "news"
N = 13
SEEN_THRESHOLD = 0.5


def ngram_hashes(words, n=N):
    return {hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)}


def words_of(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def main():
    for f in ("train.tsv", "dev.tsv", "test.tsv", "labels.txt"):
        hf_hub_download("masakhane/masakhanews", f"data/orm/{f}", repo_type="dataset", local_dir=RAW)
    labels = (RAW / "data/orm/labels.txt").read_text(encoding="utf-8").split()
    label_id = {l: i for i, l in enumerate(labels)}

    print("indexing 13-grams of the pretraining text...", flush=True)
    pretrain = set()
    with open(ROOT / "data" / "processed" / "train.txt", encoding="utf-8") as f:
        for line in f:
            pretrain |= ngram_hashes(words_of(line))
    print(f"  {len(pretrain):,} distinct 13-grams", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"labels": labels, "ngram": N, "seen_threshold": SEEN_THRESHOLD}
    for split in ("train", "dev", "test"):
        df = pd.read_csv(RAW / f"data/orm/{split}.tsv", sep="\t")
        rows, overlaps = [], []
        for cat, head, body, url in zip(df.category, df.headline, df.text, df.url):
            text = normalize(f"{head}\n{body}")
            grams = ngram_hashes(words_of(text))
            overlap = len(grams & pretrain) / len(grams) if grams else 0.0
            overlaps.append(overlap)
            rows.append({"text": text, "label": cat, "label_id": label_id[cat],
                         "pretrain_overlap": round(overlap, 3), "url": url})
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        seen = sum(o >= SEEN_THRESHOLD for o in overlaps)
        report[split] = {
            "articles": len(rows),
            "labels": dict(Counter(r["label"] for r in rows)),
            "mean_overlap": round(sum(overlaps) / len(overlaps), 3),
            "seen_articles": seen,
            "seen_pct": round(100 * seen / len(rows), 1),
            "any_overlap_pct": round(100 * sum(o > 0 for o in overlaps) / len(rows), 1),
        }
        print(f"{split}: {len(rows)} articles | mean 13-gram overlap {report[split]['mean_overlap']:.1%} | "
              f"seen (>= {SEEN_THRESHOLD:.0%}) {seen} ({report[split]['seen_pct']}%) | "
              f"any overlap {report[split]['any_overlap_pct']}%", flush=True)

    (OUT / "contamination.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
