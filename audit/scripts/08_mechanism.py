"""How do MasakhaNEWS test headlines appear in the AfriBERTa corpus?

For every test headline found in AfriBERTa's training split, find the first training line that
contains it and classify the line:
- alone:       the line is the headline (at most 2 extra words)
- short:       headline plus up to 15 extra words (e.g. a caption or a date)
- inside long: the headline sits inside a longer line, typically a flattened web page with menus and
               "related stories" lists
Writes audit/results/mechanism_afriberta.json (counts and one example line per language).
Usage: python audit/scripts/08_mechanism.py
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
RESULTS = AUDIT / "results"
spec = importlib.util.spec_from_file_location("ov", AUDIT / "scripts" / "02_overlap.py")
ov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ov)


def main():
    out = {}
    for lang, folder in ov.CORPORA["afriberta"].items():
        rows = ov.read_split(lang, "test")
        flags = [json.loads(l) for l in (RESULTS / f"flags_afriberta_{lang}_test.jsonl").read_text().splitlines()]
        heads = {" ".join(ov.words(r["headline"])) for r, f in zip(rows, flags) if "train" in f["headline_found"]}
        kinds, example, seen = Counter(), None, set()
        with open(ov.RAW / "afriberta" / folder / "train.txt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                ws = ov.words(line)
                if len(ws) < ov.MIN_HEADLINE_WORDS:
                    continue
                joined = " " + " ".join(ws) + " "
                for h in heads - seen:
                    if f" {h} " in joined:
                        seen.add(h)
                        extra = len(ws) - len(h.split())
                        kind = "alone" if extra <= 2 else "short" if extra <= 15 else "inside_long"
                        kinds[kind] += 1
                        if kind == "inside_long" and example is None:
                            example = line.strip()[:300]
                if len(seen) == len(heads):
                    break
        out[lang] = {"headlines": len(heads), **{k: kinds[k] for k in ("alone", "short", "inside_long")},
                     "example_long_line": example}
        print(lang, {k: kinds[k] for k in ("alone", "short", "inside_long")}, flush=True)
    total = Counter()
    for v in out.values():
        total.update({k: v[k] for k in ("alone", "short", "inside_long")})
    out["all"] = dict(total)
    print("all", dict(total))
    (RESULTS / "mechanism_afriberta.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
