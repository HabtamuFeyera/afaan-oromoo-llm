"""Phase 3: evaluate a pretrained checkpoint.

Reports
- validation loss and perplexity over the whole validation set
- bits per character: loss converted to "bits per character of Oromo text", which
  (unlike perplexity) can be compared across tokenizers
- unigram baseline: a model that only knows how often each token appears.
  The gap to it shows what the model learned about context.
- samples at several temperatures, and the real-word rate: the share of
  generated words that occur in the training text (a rough fluency check
  that needs no Oromo speaker)

Usage: python scripts/08_evaluate.py [checkpoints/oromo-gpt-23m/best.pt]
Writes <run-dir>/evaluation.json.
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.data import TOKENS_DIR, create_token_dataloader  # noqa: E402
from oromo_llm.generate import complete  # noqa: E402
from oromo_llm.model import GPTModel  # noqa: E402
from oromo_llm.train import calc_loss_loader, get_device  # noqa: E402

WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
PROMPTS = [
    "Afaan Oromoo",
    "Magaalaan Finfinnee",
    "Barnoonni",
    "Qonnaan bultoonni",
    "Mootummaan naannoo Oromiyaa",
    "Fayyaa hawaasaa",
]


def unigram_loss(train_ids, val_ids, vocab_size):
    counts = np.bincount(train_ids, minlength=vocab_size).astype(np.float64) + 1.0  # add-one smoothing
    logp = np.log(counts / counts.sum())
    return float(-logp[val_ids].mean())


def main():
    ckpt_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "checkpoints" / "oromo-gpt-23m" / "best.pt"
    device = get_device()
    state = torch.load(ckpt_path, map_location=device)
    cfg = state["config"]
    model = GPTModel(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    val_loader = create_token_dataloader("val", batch_size=32, max_length=cfg["context_length"])
    with torch.no_grad():
        val_loss = calc_loss_loader(val_loader, model, device)

    train_ids = np.fromfile(TOKENS_DIR / "train.bin", dtype=np.uint16)
    val_ids = np.fromfile(TOKENS_DIR / "val.bin", dtype=np.uint16)
    uni = unigram_loss(train_ids, val_ids, cfg["vocab_size"])

    val_text = (ROOT / "data" / "processed" / "val.txt").read_text(encoding="utf-8")
    chars_per_token = len(val_text) / len(val_ids)
    to_bpc = lambda loss: loss / chars_per_token / math.log(2)  # noqa: E731

    train_words = Counter(
        w.lower() for w in WORD_RE.findall((ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8"))
    )

    torch.manual_seed(0)
    samples, gen_words = {}, []
    for temp in (0.0, 0.7, 1.0):
        samples[str(temp)] = []
        for p in PROMPTS:
            text = complete(model, p, max_new_tokens=60, temperature=temp, top_k=None if temp == 0 else 50)
            samples[str(temp)].append(text)
            if temp == 0.7:
                gen_words += [w.lower() for w in WORD_RE.findall(text[len(p):])]
    real_rate = sum(w in train_words for w in gen_words) / max(1, len(gen_words))

    result = {
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "step": state["step"],
        "val_loss": round(val_loss, 4),
        "val_perplexity": round(math.exp(val_loss), 1),
        "val_bits_per_char": round(to_bpc(val_loss), 3),
        "unigram_val_loss": round(uni, 4),
        "unigram_perplexity": round(math.exp(uni), 1),
        "unigram_bits_per_char": round(to_bpc(uni), 3),
        "real_word_rate_t0.7": round(real_rate, 3),
        "samples": samples,
    }
    (ckpt_path.parent / "evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"step {state['step']}: val loss {val_loss:.3f} | perplexity {math.exp(val_loss):.1f} | "
          f"{to_bpc(val_loss):.2f} bits/char")
    print(f"unigram baseline: loss {uni:.3f} | perplexity {math.exp(uni):.1f} | {to_bpc(uni):.2f} bits/char")
    print(f"real-word rate of samples (T=0.7): {real_rate:.1%}")
    for temp, texts in samples.items():
        print(f"\n--- temperature {temp}")
        for t in texts:
            print("  " + t.replace("\n", " / "))


if __name__ == "__main__":
    main()
