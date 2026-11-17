"""Does a model find the evaluation articles it was trained on easier? (WURA's random split as control)

For each model and language: mean loss on articles in WURA's train split ("seen") versus articles
only in WURA's eval split ("held out", in no other audited training data). WURA assigned documents to
train/eval at random, so the two groups come from the same sources and differ only in whether the
model saw them.

Reported per model:
- gap: mean(held out) - mean(seen), in nats per token (positive = seen articles are easier)
- relative gap: gap / mean(held out)
- AUC: chance that a seen article has lower loss than a held-out one in the same language (0.5 = none)
- p: stratified permutation test (labels shuffled within language), one-sided, 10,000 permutations
For the AfriqueQwen model the same is also computed on loss(afrique) - loss(qwen-base), which
removes how hard each article is for the same model before African-language training.

Usage: python audit/scripts/06_effect.py
Writes audit/results/effect.json
"""

import json
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parent.parent
RESULTS = AUDIT / "results"
MIN_HELD_OUT = 10  # languages with fewer held-out articles enter only the pooled estimate
SEEN, HELD = "wura_train", "wura_eval_only"


def load(model):
    path = RESULTS / f"memorization_{model}.jsonl"
    if not path.exists():
        return None
    return {(r["lang"], r["split"], r["i"]): r for r in map(json.loads, path.read_text().splitlines())}


def auc(seen, held):
    s, h = np.asarray(seen)[:, None], np.asarray(held)[None, :]
    return float(((s < h).sum() + 0.5 * (s == h).sum()) / (s.size * h.size))


def analyse(values):
    """values: list of (lang, group, value). Returns per-language and pooled statistics."""
    langs = sorted({l for l, g, _ in values})
    per_lang, gaps, weights, aucs = {}, [], [], []
    for lang in langs:
        seen = [v for l, g, v in values if l == lang and g == SEEN]
        held = [v for l, g, v in values if l == lang and g == HELD]
        if not seen or not held:
            continue
        gap = float(np.mean(held) - np.mean(seen))
        per_lang[lang] = {"seen": len(seen), "held_out": len(held), "mean_seen": round(float(np.mean(seen)), 4),
                          "mean_held_out": round(float(np.mean(held)), 4), "gap": round(gap, 4),
                          "relative_gap_pct": round(100 * gap / float(np.mean(held)), 2) if np.mean(held) > 0 else None,
                          "auc": round(auc(seen, held), 3), "reported": len(held) >= MIN_HELD_OUT}
        gaps.append(gap)
        weights.append(len(held))
        aucs.append((auc(seen, held), len(held)))
    pooled_gap = float(np.average(gaps, weights=weights))

    # stratified permutation test of the pooled gap
    rng = np.random.default_rng(0)
    by_lang = {l: (np.array([v for ll, g, v in values if ll == l and g in (SEEN, HELD)]),
                   np.array([g == HELD for ll, g, v in values if ll == l and g in (SEEN, HELD)]))
               for l in per_lang}

    def stat(perm):
        gs, ws = [], []
        for l, (v, held) in by_lang.items():
            h = perm[l] if perm else held
            gs.append(v[h].mean() - v[~h].mean())
            ws.append(h.sum())
        return np.average(gs, weights=ws)

    null = [stat({l: rng.permutation(h) for l, (v, h) in by_lang.items()}) for _ in range(10000)]
    p = (1 + sum(x >= pooled_gap for x in null)) / (1 + len(null))
    mean_held = np.average([per_lang[l]["mean_held_out"] for l in per_lang], weights=weights)
    rel = round(100 * pooled_gap / mean_held, 2) if mean_held > 0 else None  # undefined for loss differences
    return per_lang, {"gap": round(pooled_gap, 4), "relative_gap_pct": rel,
                      "auc": round(float(np.average([a for a, _ in aucs], weights=[w for _, w in aucs])), 3),
                      "p_one_sided": round(float(p), 5), "seen": sum(x["seen"] for x in per_lang.values()),
                      "held_out": sum(x["held_out"] for x in per_lang.values())}


def main():
    out = {}
    models = {m: load(m) for m in ("afriteva-base", "afriteva-large", "afrique", "qwen-base",
                                    "afrique-4b", "qwen-4b-base")}
    for name, rows in models.items():
        if rows:
            out[name] = dict(zip(("per_language", "pooled"),
                                 analyse([(r["lang"], r["group"], r["loss"]) for r in rows.values()])))
    for tuned, base, label in (("afrique", "qwen-base", "afrique_minus_qwen_base"),
                               ("afrique-4b", "qwen-4b-base", "afrique-4b_minus_qwen-4b_base")):
        if models[tuned] and models[base]:
            a, b = models[tuned], models[base]
            out[label] = dict(zip(("per_language", "pooled"), analyse(
                [(r["lang"], r["group"], r["loss"] - b[k]["loss"]) for k, r in a.items() if k in b])))
    (RESULTS / "effect.json").write_text(json.dumps(out, indent=2))

    for name, res in out.items():
        p = res["pooled"]
        rel = f" ({p['relative_gap_pct']:+.1f}%)" if p["relative_gap_pct"] is not None else ""
        print(f"\n{name}: pooled gap {p['gap']:+.4f} nats/token{rel}, "
              f"AUC {p['auc']:.3f}, p = {p['p_one_sided']} (seen {p['seen']}, held out {p['held_out']})")
        for lang, r in res["per_language"].items():
            if r["reported"]:
                print(f"  {lang}: seen {r['mean_seen']:.3f} vs held out {r['mean_held_out']:.3f} "
                      f"(gap {r['gap']:+.3f}), AUC {r['auc']:.2f}, "
                      f"n = {r['seen']}/{r['held_out']}")


if __name__ == "__main__":
    main()
