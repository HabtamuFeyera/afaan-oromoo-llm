"""Phase 6, step 1: how well do open models model Oromo out of the box?

Bits per character (bpc) on the same Oromo text for every model. bpc is the
average number of bits a model needs per character, so it compares models
with different tokenizers fairly (lower is better). Each model sees about the
same amount of text as context (~1,300 characters): 256 tokens for our model,
whose tokenizer packs ~5 characters per token, and 512 for Qwen's (~2.6).

Test sets
- val:            the first 150k characters of the validation set (AfriBERTa + Wikipedia).
                  AfriqueLLM's training data may contain this text.
- telegram:       the first 150k characters of held-out Telegram posts
- telegram_2026:  held-out Telegram posts from February 2026 or later, after the
                  AfriqueLLM release, so no model here can have seen them (small: 28 posts)

Usage: python scripts/16_oromo_skill.py [model ...]
       models: "ours:<run-name>" or a Hugging Face id
Writes checkpoints/oromo_skill.json.
"""

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.model import GPTModel  # noqa: E402
from oromo_llm.tokenizer import EOT, get_tokenizer  # noqa: E402
from oromo_llm.train import get_device  # noqa: E402

DEFAULT_MODELS = ["ours:oromo-gpt-23m-tg", "Qwen/Qwen3.5-0.8B-Base", "McGill-NLP/AfriqueQwen3.5-0.8B-50Langs"]
SAMPLE_CHARS = 150_000
CONTEXT_CHARS = 1300


def test_sets():
    def doc_text(path):
        return path.read_text(encoding="utf-8").replace(f"\n{EOT}\n", "\n\n")[:SAMPLE_CHARS]

    proc = ROOT / "data" / "processed"
    recent = []
    for line in (ROOT / "data" / "interim" / "telegram_val.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        date = datetime.strptime(row["date"].split(" UTC")[0], "%d %B %Y, %H:%M:%S") if row["date"] else None
        if date and date >= datetime(2026, 2, 1):
            recent.append("\n".join(row["lines"]))
    return {"val": doc_text(proc / "val.txt"),
            "telegram": doc_text(proc / "val_telegram.txt"),
            "telegram_2026": "\n\n".join(recent)}


@torch.no_grad()
def total_nll(logits_fn, ids, ctx, device, batch_size=8):
    """Sum of negative log-likelihoods (nats) of every token after the first,
    with a sliding window: each window scores only its last ctx/2 tokens, which
    all have at least ctx/2 tokens of context."""
    stride = ctx // 2
    windows = []  # (start, first scored position within window)
    for start in range(0, max(1, len(ids) - 1), stride):
        end = min(start + ctx, len(ids))
        first = 1 if start == 0 else ctx - stride
        if end - start > first:
            windows.append((start, end, first))
        if end == len(ids):
            break
    nll, count = 0.0, 0
    for i in range(0, len(windows), batch_size):
        chunk = windows[i:i + batch_size]
        width = max(e - s for s, e, _ in chunk)
        x = torch.zeros(len(chunk), width, dtype=torch.long)
        for j, (s, e, _) in enumerate(chunk):
            x[j, :e - s] = torch.tensor(ids[s:e])
        logp = torch.log_softmax(logits_fn(x.to(device)).float(), dim=-1)
        for j, (s, e, first) in enumerate(chunk):
            targets = x[j, first:e - s].to(device)
            nll -= logp[j, first - 1:e - s - 1].gather(1, targets[:, None]).sum().item()
            count += len(targets)
    return nll, count


def load(name, device):
    """Return (logits_fn, encode, context_tokens, label) for a model name."""
    if name.startswith("ours:"):
        run = name.split(":", 1)[1]
        ckpt = torch.load(ROOT / "checkpoints" / run / "best.pt", map_location=device)
        model = GPTModel(ckpt["config"]).to(device).eval()
        model.load_state_dict(ckpt["model_state_dict"])
        tok = get_tokenizer()
        return model, tok.encode, ckpt["config"]["context_length"], f"{run} (ours)"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    probe = "Afaan Oromoo afaan ummata Oromoo kan Gaanfa Afrikaa keessa jiraatuuti. " * 20
    ctx = round(CONTEXT_CHARS / (len(probe) / len(tok(probe)["input_ids"])) / 64) * 64
    return (lambda x: model(input_ids=x).logits), (lambda t: tok(t)["input_ids"]), ctx, name


def main():
    names = sys.argv[1:] or DEFAULT_MODELS
    device = get_device()
    sets = test_sets()
    print({k: f"{len(v):,} chars" for k, v in sets.items()}, flush=True)
    results = {}
    for name in names:
        t0 = time.time()
        logits_fn, encode, ctx, label = load(name, device)
        r = {"context_tokens": ctx}
        for set_name, text in sets.items():
            ids = encode(text)
            nll, n = total_nll(logits_fn, ids, ctx, device)
            r[set_name] = {"bits_per_char": round(nll / math.log(2) / len(text), 3),
                           "chars_per_token": round(len(text) / len(ids), 2)}
        results[label] = r
        print(f"{label:<45} ctx {ctx:>4} | " + " | ".join(
            f"{s} {r[s]['bits_per_char']:.3f}" for s in sets) + f" | {time.time() - t0:.0f}s", flush=True)
        del logits_fn
        if device.type == "mps":
            torch.mps.empty_cache()

    out = ROOT / "checkpoints" / "oromo_skill.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**old, **results}, indent=2))


if __name__ == "__main__":
    main()
