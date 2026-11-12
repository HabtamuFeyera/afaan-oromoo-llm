"""Phase 5, step 2: instruction fine-tune the pretrained Oromo GPT (book chapter 7).

Starts from checkpoints/<base>/best.pt (default oromo-gpt-23m) and trains all layers on
data/instructions/train.json with AdamW and the chapter 5 loss. Padding and
prompt tokens are masked out. Each learning rate runs for a fixed number of
epochs; the epoch and rate with the lowest validation loss are kept.

Outputs: checkpoints/<base>-instruct/{best.pt, log.json}
Usage: python scripts/12_instruction_finetune.py [--epochs 5] [--lrs 5e-5 2e-4] [--base oromo-gpt-23m-v2]
"""

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.instruct import InstructionDataset, collate, load_entries, respond  # noqa: E402
from oromo_llm.model import GPTModel  # noqa: E402
from oromo_llm.train import calc_loss_batch, calc_loss_loader, get_device  # noqa: E402

DATA = ROOT / "data" / "instructions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-5, 2e-4])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--base", default="oromo-gpt-23m", help="pretraining run under checkpoints/")
    args = ap.parse_args()
    PRETRAINED = ROOT / "checkpoints" / args.base / "best.pt"
    OUT = ROOT / "checkpoints" / f"{args.base}-instruct"

    device = get_device()
    ckpt = torch.load(PRETRAINED, map_location="cpu")
    cfg = ckpt["config"]
    ctx = cfg["context_length"]

    train_entries, val_entries = load_entries(DATA / "train.json"), load_entries(DATA / "val.json")
    coll = partial(collate, device=device)
    val_loader = DataLoader(InstructionDataset(val_entries, ctx), batch_size=16, collate_fn=coll)
    preview = [val_entries[i] for i in (0, 1, len(val_entries) - 1)]  # topic, headline, describe

    OUT.mkdir(parents=True, exist_ok=True)
    log, best = [], {"val_loss": math.inf}
    for lr in args.lrs:
        torch.manual_seed(args.seed)
        model = GPTModel(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        train_loader = DataLoader(InstructionDataset(train_entries, ctx), batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, collate_fn=coll,
                                  generator=torch.Generator().manual_seed(args.seed))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        model.eval()
        with torch.no_grad():
            v0 = calc_loss_loader(val_loader, model, device)
        print(f"\n=== lr {lr:g} | val loss before instruction tuning {v0:.3f}", flush=True)
        for epoch in range(1, args.epochs + 1):
            t0, total = time.time(), 0.0
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = calc_loss_batch(x, y, model, device)
                loss.backward()
                optimizer.step()
                total += loss.item()
            model.eval()
            with torch.no_grad():
                val_loss = calc_loss_loader(val_loader, model, device)
            rec = {"lr": lr, "epoch": epoch, "train_loss": round(total / len(train_loader), 4),
                   "val_loss": round(val_loss, 4), "minutes": round((time.time() - t0) / 60, 2)}
            log.append(rec)
            print(f"  epoch {epoch}: train {rec['train_loss']:.3f} | val {val_loss:.3f} | "
                  f"{rec['minutes']:.1f} min", flush=True)
            for e in preview:
                print(f"    [{e['task']}] {respond(model, e, device)[:120]!r}  (gold: {e['output'][:60]!r})",
                      flush=True)
            if val_loss < best["val_loss"]:
                best = {"val_loss": val_loss, "lr": lr, "epoch": epoch}
                torch.save({"model_state_dict": model.state_dict(), "config": cfg, **best},
                           OUT / "best.pt")
        (OUT / "log.json").write_text(json.dumps({"best": best, "log": log}, indent=2))

    print(f"\nbest: lr {best['lr']:g}, epoch {best['epoch']}, val loss {best['val_loss']:.3f}", flush=True)


if __name__ == "__main__":
    main()
