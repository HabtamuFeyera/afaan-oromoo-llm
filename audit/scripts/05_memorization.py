"""Score each effect-study article with a model: lower loss on seen articles means memorization.

Models
- afriteva-base / afriteva-large (castorini/afriteva_v2_*, T5, pretrained on WURA): span-corruption
  loss, the model's own pretraining task. 15% of tokens are blanked in spans of mean length 3, with
  the same masks for every model run (seeded by article), two masks per article.
- afrique / afrique-4b (McGill-NLP/AfriqueQwen3.5-{0.8B,4B}-50Langs, trained on FineWeb-2 + WURA +
  MADLAD-400) and qwen-base / qwen-4b-base (Qwen/Qwen3.5-{0.8B,4B}-Base, the same models before
  African-language training):
  mean next-token loss.
Each article is "headline\\ntext", cut to MAX_TOKENS tokens. All wura_eval_only and nowhere
articles are scored, plus a fixed random sample of SEEN_PER_LANG wura_train articles per language.

Output: audit/results/memorization_<model>.jsonl
Usage: python audit/scripts/05_memorization.py --model afriteva-base
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

AUDIT = Path(__file__).resolve().parent.parent
RESULTS = AUDIT / "results"
MAX_TOKENS = 256
SEEN_PER_LANG = 150
MODELS = {
    "afriteva-base": ("t5", "castorini/afriteva_v2_base"),
    "afriteva-large": ("t5", "castorini/afriteva_v2_large"),
    "afrique": ("lm", "McGill-NLP/AfriqueQwen3.5-0.8B-50Langs"),
    "qwen-base": ("lm", "Qwen/Qwen3.5-0.8B-Base"),
    "afrique-4b": ("lm", "McGill-NLP/AfriqueQwen3.5-4B-50Langs"),
    "qwen-4b-base": ("lm", "Qwen/Qwen3.5-4B-Base"),
}


def study_items():
    # split on "\n" only: article texts can contain U+2028 line separators, which splitlines() breaks on
    rows = [json.loads(l) for l in (RESULTS / "effect_items.jsonl").read_text(encoding="utf-8").split("\n") if l]
    keep = [r for r in rows if r["group"] in ("wura_eval_only", "nowhere")]
    rng = random.Random(0)
    for lang in sorted({r["lang"] for r in rows}):
        seen = [r for r in rows if r["lang"] == lang and r["group"] == "wura_train"]
        keep += rng.sample(seen, min(SEEN_PER_LANG, len(seen)))
    return keep


# --- T5 span corruption (the T5 paper's noise mask, simplified) ---------------------------------

def noise_mask(length, rng, density=0.15, mean_span=3.0):
    num_noise = max(1, round(length * density))
    num_spans = max(1, round(num_noise / mean_span))
    num_keep = length - num_noise

    def segments(total, n):
        cuts = sorted(rng.sample(range(1, total), n - 1)) if n > 1 and total > n else []
        bounds = [0, *cuts, total]
        return [b - a for a, b in zip(bounds, bounds[1:])] or [total]

    noise, keep = segments(num_noise, num_spans), segments(num_keep, num_spans)
    mask = []
    for k, n in zip(keep, noise):
        mask += [False] * k + [True] * n
    return mask[:length] + [False] * (length - len(mask))


def t5_example(ids, mask, sentinel0):
    inp, tgt, s = [], [], 0
    prev = False
    for tok, m in zip(ids, mask):
        if m:
            if not prev:
                inp.append(sentinel0 - s)
                tgt.append(sentinel0 - s)
                s += 1
            tgt.append(tok)
        else:
            inp.append(tok)
        prev = m
    return inp + [1], tgt + [1]  # eos


def score_t5(name, items, device):
    import sentencepiece as spm
    from huggingface_hub import snapshot_download
    from transformers import T5ForConditionalGeneration
    path = Path(snapshot_download(name, allow_patterns=["*.json", "*.model", "pytorch_model.bin"]))
    sp = spm.SentencePieceProcessor(model_file=str(path / "spiece.model"))
    # AfriTeVa V2 was trained with the last SentencePiece ids as sentinels (149999, 149998, ...), not the
    # 100 extra ids Hugging Face's T5 tokenizer appends: with those, the loss is above chance (~12 nats);
    # with these it is 0.2-0.7 on ordinary text.
    sentinel0 = sp.get_piece_size() - 1
    model = T5ForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32).to(device).eval()
    out = []
    for n, it in enumerate(items):
        ids = sp.encode(it["headline"] + "\n" + it["text"])[:MAX_TOKENS]
        losses = []
        for m in range(2):
            rng = random.Random(f"{it['lang']}-{it['split']}-{it['i']}-{m}")
            inp, tgt = t5_example(ids, noise_mask(len(ids), rng), sentinel0)
            with torch.no_grad():
                loss = model(input_ids=torch.tensor([inp], device=device),
                             labels=torch.tensor([tgt], device=device)).loss
            losses.append(loss.item())
        out.append({**{k: it[k] for k in ("lang", "split", "i", "group")}, "loss": float(np.mean(losses)),
                    "tokens": len(ids)})
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(items)}", flush=True)
    return out


def score_lm(name, items, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    out = []
    for n, it in enumerate(items):
        ids = tok(it["headline"] + "\n" + it["text"])["input_ids"][:MAX_TOKENS]
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            loss = model(input_ids=x, labels=x).loss
        out.append({**{k: it[k] for k in ("lang", "split", "i", "group")}, "loss": loss.item(),
                    "tokens": len(ids)})
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(items)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    args = ap.parse_args()
    kind, name = MODELS[args.model]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    items = study_items()
    t0 = time.time()
    rows = (score_t5 if kind == "t5" else score_lm)(name, items, device)
    path = RESULTS / f"memorization_{args.model}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{args.model}: {len(rows)} articles in {(time.time() - t0) / 60:.1f} min -> {path.name}")


if __name__ == "__main__":
    main()
