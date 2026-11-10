"""Phase 5, step 3: evaluate instruction following on the test split.

Compares the pretrained model (before instruction tuning) with the
instruction-tuned one, plus simple baselines, on each task:
- topic:    exact match of the first word of the answer with the Oromo topic word
            -> accuracy and macro-F1 (comparable with Phase 4's classifier)
- headline: ROUGE-L F1 (longest common word subsequence) with the real headline.
            Baseline: the article's first 10 words ("lead").
- describe: ROUGE-L F1 with the Wikipedia article's first sentence. These articles
            were never seen in pretraining, so this measures how much the model
            makes up; the answers are saved for a native speaker to judge.

All answers are written to checkpoints/oromo-gpt-23m-instruct/test_responses.json
(book section 7.7) so a native speaker can review them.
Usage: python scripts/13_evaluate_instructions.py
"""

import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.classify import scores  # noqa: E402
from oromo_llm.instruct import load_entries, respond  # noqa: E402
from oromo_llm.model import GPTModel  # noqa: E402
from oromo_llm.text import WORD_RE  # noqa: E402
from oromo_llm.train import get_device  # noqa: E402

TOPICS = ["siyaasa", "fayyaa", "ispoortii", "bashannanaa", "teeknooloojii"]
RUN = ROOT / "checkpoints" / "oromo-gpt-23m-instruct"


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
    out = {}
    topic = [(r, e) for r, e in zip(responses, entries) if e["task"] == "topic"]
    preds = [TOPICS.index(first_word(r)) if first_word(r) in TOPICS else -1 for r, _ in topic]
    gold = [TOPICS.index(e["output"]) for _, e in topic]
    s = scores(preds, gold, len(TOPICS))
    out["topic"] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                    "valid_label_pct": sum(p >= 0 for p in preds) / len(preds)}
    for task in ("headline", "describe"):
        pairs = [(r, e["output"]) for r, e in zip(responses, entries) if e["task"] == task]
        out[task] = {"rouge_l": sum(rouge_l(r, g) for r, g in pairs) / len(pairs),
                     "avg_words": sum(len(words(r)) for r, _ in pairs) / len(pairs)}
    return out


def load(path, device):
    ckpt = torch.load(path, map_location=device)
    model = GPTModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def main():
    device = get_device()
    entries = load_entries(ROOT / "data" / "instructions" / "test.json")
    models = {
        "pretrained (no instruction tuning)": ROOT / "checkpoints" / "oromo-gpt-23m" / "best.pt",
        "instruction-tuned": RUN / "best.pt",
    }
    results, all_responses = {}, {}
    for name, path in models.items():
        model = load(path, device)
        torch.manual_seed(0)
        responses = []
        for i, e in enumerate(entries):
            n = 8 if e["task"] == "topic" else 40
            responses.append(respond(model, e, device, max_new_tokens=n))
            if (i + 1) % 200 == 0:
                print(f"  {name}: {i + 1}/{len(entries)}", flush=True)
        results[name] = evaluate(responses, entries)
        all_responses[name] = responses

    # Baselines that need no model.
    lead = [" ".join(e["input"].split()[:10]) if e["task"] == "headline" else "" for e in entries]
    results["lead-10 baseline"] = {"headline": {"rouge_l": sum(
        rouge_l(p, e["output"]) for p, e in zip(lead, entries) if e["task"] == "headline") / sum(
        e["task"] == "headline" for e in entries)}}

    review = [{"task": e["task"], "instruction": e["instruction"], "input": e["input"][:300],
               "gold": e["output"], **{k: v[i] for k, v in all_responses.items()}}
              for i, e in enumerate(entries)]
    (RUN / "test_responses.json").write_text(json.dumps(review, indent=1, ensure_ascii=False), encoding="utf-8")
    (RUN / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    for name, r in results.items():
        print(f"\n{name}")
        for task, m in r.items():
            print(f"  {task:<9} " + " | ".join(f"{k} {v:.3f}" for k, v in m.items()))
    print("\nSample answers (instruction-tuned):")
    shown = {"topic": 0, "headline": 0, "describe": 0}
    for row in review:
        if shown[row["task"]] < 3:
            shown[row["task"]] += 1
            print(f"  [{row['task']}] {row['instruction'][:60]}")
            print(f"     model: {row['instruction-tuned'][:140]!r}")
            print(f"     gold:  {row['gold'][:140]!r}")


if __name__ == "__main__":
    main()
