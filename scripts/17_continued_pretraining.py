"""Phase 6, step 2: continued pretraining of an open model on Oromo with LoRA.

Takes a general multilingual base model (default Qwen/Qwen3-0.6B-Base, which
uses standard attention and trains reasonably fast on a Mac) and continues
next-token training on our Oromo corpus. Only LoRA adapters train
(low-rank matrices added to every attention and MLP projection, the method
in the book's Appendix E); the original weights stay frozen.

Data: random ~2,000-character chunks of data/processed/train.txt, tokenized with
the model's own tokenizer and packed into 512-token sequences, until the token
budget is reached. The validation and test text is never used.

Outputs: checkpoints/<run-name>/adapter/ (LoRA weights), merged/ (base model with
the adapters merged in, loadable by scripts/16_oromo_skill.py), log.json
Usage: python scripts/17_continued_pretraining.py [--tokens 2500000]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.tokenizer import EOT  # noqa: E402
from oromo_llm.train import get_device, lr_at  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def packed_sequences(tok, n_tokens, seq_len, seed):
    text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8").replace(f"\n{EOT}\n", "\n\n")
    rng = random.Random(seed)
    chunk = 2000
    starts = list(range(0, len(text) - chunk, chunk))
    rng.shuffle(starts)
    buf, seqs = [], []
    eos = tok.eos_token_id
    for s in starts:
        piece = text[s:s + chunk]
        piece = piece[piece.find("\n") + 1:] if s else piece  # start at a line boundary
        buf += tok(piece)["input_ids"] + [eos]
        while len(buf) >= seq_len:
            seqs.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(seqs) * seq_len >= n_tokens:
            break
    return torch.tensor(seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--run-name", default="qwen3-0.6b-oromo-lora")
    ap.add_argument("--tokens", type=int, default=2_500_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    out = ROOT / "checkpoints" / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    data = packed_sequences(tok, args.tokens, args.seq_len, args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=TARGETS, task_type="CAUSAL_LM")).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    steps = len(data) // args.batch_size
    warmup = max(1, steps // 50)
    print(f"{args.model} + LoRA r={args.rank}: {trainable:,} trainable params | "
          f"{len(data)} sequences x {args.seq_len} = {len(data) * args.seq_len:,} tokens | {steps} steps", flush=True)

    log, t0, running = [], time.time(), []
    model.train()
    perm = torch.randperm(len(data))
    for step in range(steps):
        lr = lr_at(step, steps, args.lr, warmup, args.lr * 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        x = data[perm[step * args.batch_size:(step + 1) * args.batch_size]].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad()
        running.append(loss.item())
        if (step + 1) % 25 == 0 or step + 1 == steps:
            rec = {"step": step + 1, "lr": lr, "train_loss": sum(running) / len(running),
                   "tokens_per_sec": round((step + 1) * args.batch_size * args.seq_len / (time.time() - t0))}
            log.append(rec)
            running = []
            eta = (steps - step - 1) * (time.time() - t0) / (step + 1) / 60
            print(f"step {step + 1}/{steps} | loss {rec['train_loss']:.3f} | lr {lr:.1e} | "
                  f"{rec['tokens_per_sec']} tok/s | ~{eta:.0f} min left", flush=True)

    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged")
    tok.save_pretrained(out / "merged")
    (out / "log.json").write_text(json.dumps({"args": vars(args), "trainable_params": trainable,
                                              "minutes": round((time.time() - t0) / 60, 1), "log": log}, indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out.relative_to(ROOT)}/merged", flush=True)


if __name__ == "__main__":
    main()
