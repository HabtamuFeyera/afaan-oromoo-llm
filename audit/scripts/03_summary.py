"""Combine the per-corpus audits into one table, counting only each corpus's training split.

An evaluation article counts as contaminated in a corpus when the corpus's training split contains
its headline (as a phrase), at least half of its 13-grams, or a page with the same URL.

Writes audit/results/summary.json and prints a Markdown table.
Usage: python audit/scripts/03_summary.py [--split test]
"""

import argparse
import json
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
RESULTS = AUDIT / "results"
CORPORA = ["afriberta", "wura", "fineweb2"]
LANGS = ["amh", "hau", "ibo", "lin", "lug", "orm", "pcm", "run", "sna", "som", "swa", "tir", "xho", "yor"]
NAMES = {"amh": "Amharic", "hau": "Hausa", "ibo": "Igbo", "lin": "Lingala", "lug": "Luganda", "orm": "Oromo",
         "pcm": "Nigerian Pidgin", "run": "Kirundi", "sna": "Shona", "som": "Somali", "swa": "Swahili",
         "tir": "Tigrinya", "xho": "Xhosa", "yor": "Yoruba"}


def flags(corpus, lang, split):
    path = RESULTS / f"flags_{corpus}_{lang}_{split}.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else None


def in_train(f):
    head = "train" in f["headline_found"]
    grams = f["share_13gram_by_file"].get("train", 0) >= 0.5
    url = "train" in f.get("url_found", [])
    return head, grams, url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    split = ap.parse_args().split

    summary = {}
    for lang in LANGS:
        row = {}
        union = None
        for corpus in CORPORA:
            fl = flags(corpus, lang, split)
            if fl is None:
                continue
            marks = [in_train(f) for f in fl]
            hit = [any(m) for m in marks]
            union = hit if union is None else [a or b for a, b in zip(union, hit)]
            row[corpus] = {"articles": len(fl), "headline": sum(m[0] for m in marks),
                           "grams50": sum(m[1] for m in marks), "url": sum(m[2] for m in marks),
                           "any": sum(hit)}
        if union is not None:
            row["any_corpus"] = sum(union)
            row["articles"] = len(union)
            summary[lang] = row

    (RESULTS / f"summary_{split}.json").write_text(json.dumps(summary, indent=2))

    def cell(r, c):
        if c not in r:
            return "–"
        x = r[c]
        return f"{x['any']} ({100 * x['any'] / x['articles']:.0f}%)"

    print(f"| Language | {split} articles | AfriBERTa | WURA | FineWeb-2 | Any corpus |")
    print("|---|---:|---:|---:|---:|---:|")
    tot = {c: [0, 0] for c in CORPORA + ["any_corpus"]}
    for lang, r in summary.items():
        print(f"| {NAMES[lang]} | {r['articles']} | {cell(r, 'afriberta')} | {cell(r, 'wura')} | "
              f"{cell(r, 'fineweb2')} | **{r['any_corpus']} ({100 * r['any_corpus'] / r['articles']:.0f}%)** |")
        for c in CORPORA:
            if c in r:
                tot[c][0] += r[c]["any"]
                tot[c][1] += r[c]["articles"]
        tot["any_corpus"][0] += r["any_corpus"]
        tot["any_corpus"][1] += r["articles"]
    print("| **All** | " + f"{tot['any_corpus'][1]} | " + " | ".join(
        f"{a}/{b} ({100 * a / b:.0f}%)" if b else "–" for a, b in (tot[c] for c in CORPORA)) +
        f" | **{tot['any_corpus'][0]}/{tot['any_corpus'][1]} ({100 * tot['any_corpus'][0] / tot['any_corpus'][1]:.0f}%)** |")


if __name__ == "__main__":
    main()
