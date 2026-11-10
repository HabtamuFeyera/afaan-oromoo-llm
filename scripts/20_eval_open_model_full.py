"""Phase 6, step 4: score the LoRA instruction-tuned open model on the FULL test set.

scripts/18_instruct_open_model.py evaluated a subset (the first 150 news articles
plus all 81 "describe" items). The MasakhaNEWS test file is sorted by topic, so
that subset held only 3 of the 5 topics. This script loads the saved adapter,
generates answers for the remaining news items, merges them with the saved
ones, and scores every test item, next to the 23M model's answers on the same
items and the "first 10 words" headline baseline.

Outputs: checkpoints/<run-name>/evaluation_full.json, test_responses_full.json
Usage: python scripts/20_eval_open_model_full.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.instruct import load_entries  # noqa: E402
from oromo_llm.metrics import evaluate, rouge_l  # noqa: E402
from oromo_llm.train import get_device  # noqa: E402

spec = importlib.util.spec_from_file_location("s18", ROOT / "scripts" / "18_instruct_open_model.py")
s18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s18)

MODEL = "McGill-NLP/AfriqueQwen3.5-0.8B-50Langs"
RUN = ROOT / "checkpoints" / "afrique-0.8b-instruct-lora"
MAX_LEN = 384
KEY = "after LoRA instruction tuning"


def main():
    device = get_device()
    test = load_entries(ROOT / "data" / "instructions" / "test.json")
    saved = json.loads((RUN / "test_responses.json").read_text(encoding="utf-8"))
    _, subset_idx = s18.test_subset(test)
    have = {i: row[KEY] for i, row in zip(subset_idx, saved)}
    todo = [i for i in range(len(test)) if i not in have]
    print(f"{len(have)} answers saved, generating {len(todo)} more", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, RUN / "adapter").to(device)
    new = s18.generate_all(model, tok, [test[i] for i in todo], MAX_LEN, device)
    have.update(zip(todo, new))
    tuned = [have[i] for i in range(len(test))]

    p5 = json.loads((ROOT / "checkpoints" / "oromo-gpt-23m-instruct" / "test_responses.json").read_text(encoding="utf-8"))
    small = [row["instruction-tuned"] for row in p5]
    lead = [" ".join(e["input"].split()[:10]) if e["task"] == "headline" else "" for e in test]
    heads = [(l, e["output"]) for l, e in zip(lead, test) if e["task"] == "headline"]

    results = {
        "23M instruction-tuned (Phase 5)": evaluate(small, test),
        "AfriqueQwen3.5-0.8B + LoRA": evaluate(tuned, test),
        "first 10 words of the article": {"headline": {"rouge_l": sum(rouge_l(a, b) for a, b in heads) / len(heads)}},
    }
    (RUN / "evaluation_full.json").write_text(json.dumps(results, indent=2))
    rows = [{"task": e["task"], "instruction": e["instruction"], "gold": e["output"], "23M": s, "0.8B": t}
            for e, s, t in zip(test, small, tuned)]
    (RUN / "test_responses_full.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, r in results.items():
        print(f"\n{name}")
        for task, m in r.items():
            print(f"  {task:<9} " + " | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                               for k, v in m.items()))


if __name__ == "__main__":
    main()
