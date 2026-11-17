"""Release files: per-article contamination flags and clean evaluation subsets.

For every MasakhaNEWS language and dev/test split, an article is clean when no audited corpus's
training split contains its headline (as a phrase), at least half of its 13-grams, or a page with
the same URL. The release lists, per article (row index in the official TSV), which corpus had it.

Outputs (audit/release/, small, meant for the public repository):
- contamination_<lang>_<split>.tsv   index, afriberta, wura, fineweb2, clean (1/0 per column; "-" when
                                     a corpus does not cover the language)
- clean_counts.json                  clean / total per language and split
Usage: python audit/scripts/07_release.py
"""

import json
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
RESULTS = AUDIT / "results"
RELEASE = AUDIT / "release"
CORPORA = ["afriberta", "wura", "fineweb2"]
LANGS = ["amh", "hau", "ibo", "lin", "lug", "orm", "pcm", "run", "sna", "som", "swa", "tir", "xho", "yor"]


def flags(corpus, lang, split):
    path = RESULTS / f"flags_{corpus}_{lang}_{split}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else None


def in_train(f):
    return ("train" in f["headline_found"] or f["share_13gram_by_file"].get("train", 0) >= 0.5
            or "train" in f.get("url_found", []))


def main():
    RELEASE.mkdir(parents=True, exist_ok=True)
    counts = {}
    for lang in LANGS:
        for split in ("test", "dev"):
            cols = {c: flags(c, lang, split) for c in CORPORA}
            n = len(next(v for v in cols.values() if v))
            lines, clean = ["index\t" + "\t".join(CORPORA) + "\tclean"], 0
            for k in range(n):
                marks = [("-" if cols[c] is None else str(int(in_train(cols[c][k])))) for c in CORPORA]
                ok = "1" not in marks
                clean += ok
                idx = next(v for v in cols.values() if v)[k]["i"]
                lines.append(f"{idx}\t" + "\t".join(marks) + f"\t{int(ok)}")
            (RELEASE / f"contamination_{lang}_{split}.tsv").write_text("\n".join(lines) + "\n")
            counts[f"{lang}_{split}"] = {"clean": clean, "total": n}
    (RELEASE / "clean_counts.json").write_text(json.dumps(counts, indent=2))
    for lang in LANGS:
        t = counts[f"{lang}_test"]
        print(f"{lang}: clean test articles {t['clean']}/{t['total']}")


if __name__ == "__main__":
    main()
