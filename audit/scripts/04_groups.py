"""Assign every MasakhaNEWS dev/test article to a contamination group, for the effect study.

WURA was split into train (~90%) and eval (~10%) at random, so MasakhaNEWS articles from the same
sources fell into one or the other by chance. That gives a natural randomized comparison:
- wura_train:     the article is in WURA's training split (headline, >=50% of 13-grams, or same URL)
- wura_eval_only: the article is only in WURA's eval split, and in no other audited training split
- nowhere:        in no audited corpus at all (other sources, mostly Somali and Xhosa; not randomized)
- other:          in AfriBERTa or FineWeb-2 training data but not WURA's (excluded from the study)

Output: audit/results/effect_items.jsonl (lang, split, i, group, headline, text)
Usage: python audit/scripts/04_groups.py
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
RAW = AUDIT / "data" / "raw"
RESULTS = AUDIT / "results"
LANGS = ["amh", "hau", "ibo", "orm", "sna", "som", "swa", "tir", "xho", "yor"]  # languages in WURA
csv.field_size_limit(sys.maxsize)


def load(corpus, lang, split):
    path = RESULTS / f"flags_{corpus}_{lang}_{split}.jsonl"
    if not path.exists():
        return None
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def seen(flag, part):
    if flag is None:
        return False
    return (part in flag["headline_found"] or flag["share_13gram_by_file"].get(part, 0) >= 0.5
            or part in flag.get("url_found", []))


def main():
    counts = Counter()
    with open(RESULTS / "effect_items.jsonl", "w", encoding="utf-8") as out:
        for lang in LANGS:
            for split in ("test", "dev"):
                with open(RAW / "masakhanews" / "data" / lang / f"{split}.tsv", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f, delimiter="\t"))
                wura, fw2 = load("wura", lang, split), load("fineweb2", lang, split)
                afb = load("afriberta", lang, split) or [None] * len(rows)
                for row, w, f, a in zip(rows, wura, fw2, afb):
                    other = seen(f, "train") or seen(a, "train")
                    if seen(w, "train"):
                        group = "wura_train"
                    elif seen(w, "eval") and not other:
                        group = "wura_eval_only"
                    elif not seen(w, "eval") and not other:
                        group = "nowhere"
                    else:
                        group = "other"
                    counts[(lang, group)] += 1
                    out.write(json.dumps({"lang": lang, "split": split, "i": w["i"], "group": group,
                                          "headline": row["headline"], "text": row["text"]},
                                         ensure_ascii=False) + "\n")
    for lang in LANGS:
        print(lang, {g: counts[(lang, g)] for g in ("wura_train", "wura_eval_only", "nowhere", "other")})


if __name__ == "__main__":
    main()
