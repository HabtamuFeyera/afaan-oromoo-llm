"""Draw the paper's figures as vector PDFs from the saved result files.

- figures/val_loss.pdf      validation loss by step for the three corpus sizes (checkpoints/*/log.jsonl)
- figures/topic_f1.pdf      MasakhaNEWS test macro-F1 of five systems (checkpoints/news-classifier-v2/results.json)
- figures/open_models.pdf   bits per character of open models (checkpoints/oromo_skill.json)

Style: grey data with one blue accent on the paper's own model, thin marks,
direct labels, light gridlines; sized for one column of the ACL template (3.15 in).
Usage: python scripts/19_paper_figures.py
"""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CK = ROOT / "checkpoints"
OUT = ROOT / "paper" / "figures"

ACCENT = "#2a78d6"
GREY = "#b9b7ae"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WIDTH = 3.15

plt.rcParams.update({
    "font.size": 7.5, "font.family": "sans-serif", "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def r2(x):
    """Round half up to 2 decimals, as in the paper text (1.535 -> 1.54, not 1.53)."""
    return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def val_loss():
    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    runs = [("oromo-gpt-23m", "Public text, 6.9M words", "#d6d4cb", 1.4),
            ("oromo-gpt-23m-tg", "+ 4 channels, 8.1M", GREY, 1.4),
            ("oromo-gpt-23m-v2", "+ 19 channels, 12.8M (ours)", ACCENT, 1.8)]
    for run, label, color, lw in runs:
        rows = [json.loads(l) for l in (CK / run / "log.jsonl").read_text().splitlines()]
        steps, loss = [r["step"] for r in rows], [r["val_loss"] for r in rows]
        ax.plot(steps, loss, color=color, lw=lw, label=f"{label}: {loss[-1]:.2f}")
    ax.axhline(8.138, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.text(10400, 8.138 - 0.08, "Unigram baseline 8.14", ha="right", va="top", color=INK2, fontsize=6.5)
    ax.set_xlim(0, 10600)
    ax.set_ylim(4.7, 8.4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss (nats/token)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.savefig(OUT / "val_loss.pdf")


def topic_f1():
    res = json.loads((CK / "news-classifier-v2" / "results.json").read_text())["setups"]
    systems = [
        ("majority", "Majority class"),
        ("random_init_all", "GPT, random init"),
        ("pretrained_last_block", "GPT, pretrained, book recipe"),
        ("tfidf_logreg", "TF-IDF + logistic regression"),
        ("pretrained_all", "GPT, pretrained, all layers"),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.75))
    for i, (key, label) in enumerate(systems):
        t = res[key]["test"]
        f1 = t["macro_f1"]["mean"] if isinstance(t["macro_f1"], dict) else t["macro_f1"]
        sd = t["macro_f1"]["std"] if isinstance(t["macro_f1"], dict) else 0.0
        acc = t["accuracy"]["mean"] if isinstance(t["accuracy"], dict) else t["accuracy"]
        best = key == "pretrained_all"
        y = len(systems) - 1 - i
        ax.barh(y, f1, height=0.62, color=ACCENT if best else GREY)
        if sd:
            ax.errorbar(f1, y, xerr=sd, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
        ax.text(f1 + sd + 0.015, y, f"{f1:.3f} ({100 * acc:.1f}%)", va="center", color=INK,
                fontweight="bold" if best else "normal", fontsize=6.5)
    ax.set_yticks(range(len(systems)), [s[1] for s in reversed(systems)])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Test macro-F1 (accuracy in brackets)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "topic_f1.pdf")


def open_models():
    res = json.loads((CK / "oromo_skill.json").read_text())
    models = [
        ("McGill-NLP/AfriqueQwen3.5-0.8B-50Langs", "AfriqueQwen3.5-0.8B"),
        ("oromo-gpt-23m-v2 (ours)", "Ours, 23M"),
        ("checkpoints/qwen3-0.6b-oromo-lora-v2/merged", "Qwen3-0.6B + our LoRA"),
        ("Qwen/Qwen3.5-0.8B-Base", "Qwen3.5-0.8B-Base"),
        ("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base"),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.85))
    for i, (key, label) in enumerate(models):
        r = res[key]
        y = len(models) - 1 - i
        ours = label.startswith("Ours")
        unseen, val = r["telegram_2026"]["bits_per_char"], r["val"]["bits_per_char"]
        ax.barh(y, unseen, height=0.62, color=ACCENT if ours else GREY)
        ax.plot([val, val], [y - 0.38, y + 0.38], color=INK, lw=1.2)
        ax.text(unseen - 0.05, y, r2(unseen), va="center", ha="right", color="white" if ours else INK,
                fontweight="bold" if ours else "normal", fontsize=6.5)
        ax.text(max(val, unseen) + 0.06, y, f"val {r2(val)}", va="center", color=INK2, fontsize=6)
    ax.set_yticks(range(len(models)), [m[1] for m in reversed(models)])
    ax.set_xlim(0, 3.6)
    ax.set_xlabel("Bits per character (lower is better)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "open_models.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val_loss()
    topic_f1()
    open_models()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
