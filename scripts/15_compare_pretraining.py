"""Compare pretraining runs on the same held-out text.

For each checkpoint: loss, perplexity and bits per character on
- val.txt          the original validation set (AfriBERTa eval + Wikipedia val)
- val_telegram.txt held-out Telegram posts (2019-2026 news)
plus samples from the same prompts and random seed, for side-by-side reading.

Usage: python scripts/15_compare_pretraining.py [run-name ...]
       (default: oromo-gpt-23m oromo-gpt-23m-tg)
Writes checkpoints/pretraining_comparison.json.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.data import TOKENS_DIR, create_token_dataloader  # noqa: E402
from oromo_llm.generate import complete  # noqa: E402
from oromo_llm.model import GPTModel  # noqa: E402
from oromo_llm.train import calc_loss_loader, get_device  # noqa: E402

# val_telegram_v1: the Telegram validation posts of the first import (4 channels), kept so
# later runs are compared on exactly the same text; val_telegram: the current, larger set.
SPLITS = [s for s in ["val", "val_telegram_v1", "val_telegram"]
          if (ROOT / "data" / "processed" / f"{s}.txt").exists()]
PROMPTS = ["Afaan Oromoo", "Mootummaan naannoo Oromiyaa", "Qonnaan bultoonni", "Barattootni kutaa 8ffaa"]


def chars_per_token(split):
    text = (ROOT / "data" / "processed" / f"{split}.txt").read_text(encoding="utf-8")
    return len(text) / np.fromfile(TOKENS_DIR / f"{split}.bin", dtype=np.uint16).size


def main():
    runs = sys.argv[1:] or ["oromo-gpt-23m", "oromo-gpt-23m-tg"]
    device = get_device()
    cpt = {s: chars_per_token(s) for s in SPLITS}
    results = {}
    for run in runs:
        ckpt = torch.load(ROOT / "checkpoints" / run / "best.pt", map_location=device)
        model = GPTModel(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        r = {"step": ckpt["step"]}
        with torch.no_grad():
            for s in SPLITS:
                loader = create_token_dataloader(s, batch_size=32, max_length=ckpt["config"]["context_length"],
                                                 shuffle=False)
                loss = calc_loss_loader(loader, model, device)
                r[s] = {"loss": round(loss, 4), "perplexity": round(math.exp(loss), 1),
                        "bits_per_char": round(loss / cpt[s] / math.log(2), 3)}
        torch.manual_seed(0)
        r["samples"] = [complete(model, p, max_new_tokens=50, temperature=0.7, top_k=50) for p in PROMPTS]
        results[run] = r

    out = ROOT / "checkpoints" / "pretraining_comparison.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'run':<20}" + "".join(f"{s + ' loss':>18}{'ppl':>7}{'bpc':>7}" for s in SPLITS))
    for run, r in results.items():
        print(f"{run:<20}" + "".join(f"{r[s]['loss']:>18.3f}{r[s]['perplexity']:>7.0f}{r[s]['bits_per_char']:>7.3f}"
                                     for s in SPLITS))
    for i, p in enumerate(PROMPTS):
        print(f"\n--- {p}")
        for run, r in results.items():
            print(f"  [{run}] {r['samples'][i].replace(chr(10), ' / ')[:220]}")


if __name__ == "__main__":
    main()
