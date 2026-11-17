"""Draw the paper's figures from the saved results (vector PDFs, one ACL column wide).

- audit/paper/figures/contamination.pdf  % of test articles in each corpus's training split, by language
- audit/paper/figures/memorization.pdf   seen-vs-held-out AUC per model, 95% bootstrap intervals
- audit/paper/figures/by_language.pdf    AfriqueQwen-4B (calibrated) AUC per language vs WURA size
Also writes audit/results/auc_intervals.json.
Usage: python audit/scripts/09_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AUDIT = Path(__file__).resolve().parent.parent
RESULTS = AUDIT / "results"
OUT = AUDIT / "paper" / "figures"
ACCENT, GREY, INK, INK2, GRID = "#2a78d6", "#b9b7ae", "#0b0b0b", "#52514e", "#e1e0d9"
WIDTH = 3.15
NAMES = {"amh": "Amharic", "hau": "Hausa", "ibo": "Igbo", "lin": "Lingala", "lug": "Luganda", "orm": "Oromo",
         "pcm": "Nig. Pidgin", "run": "Kirundi", "sna": "Shona", "som": "Somali", "swa": "Swahili",
         "tir": "Tigrinya", "xho": "Xhosa", "yor": "Yoruba"}
# WURA final size per language in GB (Table 6 of the WURA paper)
WURA_GB = {"amh": 1.2, "hau": 0.9, "ibo": 0.2, "orm": 0.06, "sna": 0.5, "swa": 3.5, "tir": 0.03, "yor": 0.1}

plt.rcParams.update({
    "font.size": 7.5, "font.family": "sans-serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def contamination():
    s = json.loads((RESULTS / "summary_test.json").read_text())
    langs = sorted(s, key=lambda l: -s[l]["any_corpus"] / s[l]["articles"])
    cols = [("afriberta", "AfriBERTa"), ("wura", "WURA"), ("fineweb2", "FineWeb-2"), ("any_corpus", "Any")]
    grid = np.full((len(langs), len(cols)), np.nan)
    for i, l in enumerate(langs):
        for j, (c, _) in enumerate(cols):
            if c == "any_corpus":
                grid[i, j] = 100 * s[l][c] / s[l]["articles"]
            elif c in s[l]:
                grid[i, j] = 100 * s[l][c]["any"] / s[l][c]["articles"]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("#f3f2ee")
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(len(langs)):
        for j in range(len(cols)):
            v = grid[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if (not np.isnan(v) and v > 55) else INK,
                    fontweight="bold" if j == len(cols) - 1 else "normal")
    ax.set_xticks(range(len(cols)), [c[1] for c in cols])
    ax.set_yticks(range(len(langs)), [f"{NAMES[l]} ({s[l]['articles']})" for l in langs])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(2.5, color="white", lw=2)
    fig.savefig(OUT / "contamination.pdf")


def item_values(model, base=None):
    def load(m):
        return {(r["lang"], r["split"], r["i"]): r for r in
                map(json.loads, (RESULTS / f"memorization_{m}.jsonl").read_text().splitlines())}
    a = load(model)
    b = load(base) if base else None
    return [(r["lang"], r["group"], r["loss"] - (b[k]["loss"] if b else 0.0)) for k, r in a.items()]


def pooled_auc(values, rng=None):
    aucs, ws = [], []
    for lang in sorted({l for l, _, _ in values}):
        s = np.array([v for l, g, v in values if l == lang and g == "wura_train"])
        h = np.array([v for l, g, v in values if l == lang and g == "wura_eval_only"])
        if len(s) == 0 or len(h) == 0:
            continue
        if rng is not None:
            s, h = rng.choice(s, len(s)), rng.choice(h, len(h))
        aucs.append(((s[:, None] < h[None, :]).mean() + 0.5 * (s[:, None] == h[None, :]).mean()))
        ws.append(len(h))
    return float(np.average(aucs, weights=ws))


def memorization():
    rows = [("AfriTeVa V2 base", "afriteva-base", None, "wura"), ("AfriTeVa V2 large", "afriteva-large", None, "wura"),
            ("AfriqueQwen 0.8B", "afrique", "qwen-base", "wura"), ("AfriqueQwen 4B", "afrique-4b", "qwen-4b-base", "wura"),
            ("Qwen3.5 0.8B base", "qwen-base", None, "control"), ("Qwen3.5 4B base", "qwen-4b-base", None, "control")]
    rng = np.random.default_rng(0)
    res = {}
    for label, m, base, kind in rows:
        vals = item_values(m, base)
        boot = [pooled_auc(vals, rng) for _ in range(1000)]
        res[label] = {"auc": round(pooled_auc(vals), 3), "ci95": [round(float(np.percentile(boot, 2.5)), 3),
                      round(float(np.percentile(boot, 97.5)), 3)], "calibrated": base is not None, "kind": kind}
    (RESULTS / "auc_intervals.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    labels = list(res)
    for i, lab in enumerate(labels):
        r = res[lab]
        y = len(labels) - 1 - i
        color = ACCENT if lab.startswith("AfriqueQwen 4B") else (GREY if r["kind"] == "control" else INK2)
        ax.plot(r["ci95"], [y, y], color=color, lw=1.4)
        ax.plot(r["auc"], y, "o", color=color, ms=4.5)
        ax.text(r["ci95"][1] + 0.008, y, f"{r['auc']:.2f}", va="center", fontsize=6.5, color=INK,
                fontweight="bold" if lab.startswith("AfriqueQwen 4B") else "normal")
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(range(len(labels)), [l + (" *" if res[l]["calibrated"] else "") for l in reversed(labels)])
    ax.set_xlim(0.38, 0.75)
    ax.set_xlabel("AUC (0.5 = no memorization)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "memorization.pdf")
    return res


def by_language():
    e = json.loads((RESULTS / "effect.json").read_text())["afrique-4b_minus_qwen-4b_base"]["per_language"]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    for l, gb in WURA_GB.items():
        a = e[l]["auc"]
        ax.plot(gb, a, "o", color=ACCENT, ms=4.5)
        ax.annotate(NAMES[l], (gb, a), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=INK)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlim(0.02, 6)
    ticks = [0.03, 0.1, 0.3, 1, 3]
    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
    ax.minorticks_off()
    ax.set_ylim(0.35, 0.9)
    ax.set_xlabel("WURA size of the language (GB, log scale)")
    ax.set_ylabel("AUC, AfriqueQwen 4B")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "by_language.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contamination()
    res = memorization()
    by_language()
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
