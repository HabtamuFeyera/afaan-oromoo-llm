"""Metrics for the instruction tasks (Phase 5 and Phase 6 use the same code).

- topic: exact match of the answer's first word with the Oromo topic word
         -> accuracy, macro-F1, share of answers that are a valid topic word
- headline / describe: ROUGE-L F1 (longest common word subsequence)
"""

from oromo_llm.classify import scores
from oromo_llm.text import WORD_RE

TOPICS = ["siyaasa", "fayyaa", "ispoortii", "bashannanaa", "teeknooloojii"]


def words(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def rouge_l(pred, gold):
    a, b = words(pred), words(gold)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            dp[i + 1][j + 1] = dp[i][j] + 1 if x == y else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def first_word(text):
    w = words(text)
    return w[0] if w else ""


def evaluate(responses, entries):
    """Scores per task; tasks with no items in `entries` are left out."""
    out = {}
    topic = [(r, e) for r, e in zip(responses, entries) if e["task"] == "topic"]
    if topic:
        preds = [TOPICS.index(first_word(r)) if first_word(r) in TOPICS else -1 for r, _ in topic]
        gold = [TOPICS.index(e["output"]) for _, e in topic]
        s = scores(preds, gold, len(TOPICS))
        out["topic"] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                        "valid_label_pct": sum(p >= 0 for p in preds) / len(preds)}
    for task in ("headline", "describe"):
        pairs = [(r, e["output"]) for r, e in zip(responses, entries) if e["task"] == task]
        if pairs:
            out[task] = {"rouge_l": sum(rouge_l(r, g) for r, g in pairs) / len(pairs),
                         "avg_words": sum(len(words(r)) for r, _ in pairs) / len(pairs)}
    return out
