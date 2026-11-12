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

All answers are written to checkpoints/<base>-instruct/test_responses.json
(book section 7.7) so a native speaker can review them.
Usage: python scripts/13_evaluate_instructions.py [--base oromo-gpt-23m-v2]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.instruct import load_entries, respond  # noqa: E402
from oromo_llm.metrics import evaluate, rouge_l  # noqa: E402
from oromo_llm.model import GPTModel  # noqa: E402
from oromo_llm.train import get_device  # noqa: E402

def load(path, device):
    ckpt = torch.load(path, map_location=device)
    model = GPTModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="oromo-gpt-23m", help="pretraining run under checkpoints/")
    base = ap.parse_args().base
    RUN = ROOT / "checkpoints" / f"{base}-instruct"
    device = get_device()
    entries = load_entries(ROOT / "data" / "instructions" / "test.json")
    models = {
        "pretrained (no instruction tuning)": ROOT / "checkpoints" / base / "best.pt",
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
