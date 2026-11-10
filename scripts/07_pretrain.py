"""Phase 3: pretrain the small Oromo GPT on data/tokens/{train,val}.bin.

Usage
    python scripts/07_pretrain.py                      # full run, defaults below
    python scripts/07_pretrain.py --epochs 1 --max-batches 30   # quick smoke test

Outputs go to checkpoints/<run-name>/: config.json, log.jsonl, best.pt
(lowest validation loss) and last.pt.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.data import create_token_dataloader  # noqa: E402
from oromo_llm.model import OROMO_CONFIG_23M, GPTModel, count_parameters  # noqa: E402
from oromo_llm.train import calc_loss_loader, get_device, train_model  # noqa: E402

SAMPLE_PROMPTS = ["Afaan Oromoo", "Magaalaan Finfinnee", "Barnoonni"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="oromo-gpt-23m")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-freq", type=int, default=250)
    ap.add_argument("--eval-iter", type=int, default=40)
    ap.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (smoke tests)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    cfg = dict(OROMO_CONFIG_23M)
    ctx = cfg["context_length"]

    train_loader = create_token_dataloader("train", batch_size=args.batch_size, max_length=ctx)
    val_loader = create_token_dataloader("val", batch_size=args.batch_size, max_length=ctx)
    if args.max_batches:
        n = args.max_batches * args.batch_size
        train_loader = torch.utils.data.DataLoader(
            Subset(train_loader.dataset, range(n)), batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_loader.dataset.max_length = ctx

    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = ROOT / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"model": cfg, "train": vars(args)}, indent=2))

    print(f"device {device} | params {count_parameters(model):,} | "
          f"{len(train_loader)} train batches/epoch x {args.epochs} epochs | "
          f"{len(train_loader) * args.batch_size * ctx:,} tokens/epoch", flush=True)
    with torch.no_grad():
        model.eval()
        init_val = calc_loss_loader(val_loader, model, device, num_batches=10)
        model.train()
    print(f"initial val loss {init_val:.3f} (random guessing = ln {cfg['vocab_size']} = "
          f"{math.log(cfg['vocab_size']):.3f})", flush=True)

    t0 = time.time()
    best = train_model(
        model, train_loader, val_loader, optimizer, device, cfg, run_dir,
        num_epochs=args.epochs, peak_lr=args.lr, eval_freq=args.eval_freq,
        eval_iter=args.eval_iter, sample_prompts=SAMPLE_PROMPTS,
    )
    minutes = (time.time() - t0) / 60

    # Final score: best checkpoint on the whole validation set.
    state = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        full_val = calc_loss_loader(val_loader, model, device)
    summary = {"best_eval_val_loss": best, "full_val_loss": full_val,
               "full_val_perplexity": math.exp(full_val), "best_step": state["step"],
               "train_minutes": round(minutes, 1)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {minutes:.1f} min | best step {state['step']} | full val loss {full_val:.3f} "
          f"(perplexity {math.exp(full_val):.1f})", flush=True)


if __name__ == "__main__":
    main()
