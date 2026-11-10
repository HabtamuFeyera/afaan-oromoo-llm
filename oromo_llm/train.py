"""Pretraining loop from chapter 5 of the book (listings 5.1-5.3).

The loss functions and the evaluate/sample steps are the book's. The training
loop adds the standard improvements from the book's Appendix D, which is not
in the PDF copy here but is in the book's code repository:
- learning-rate warmup followed by cosine decay
- gradient clipping
It also saves checkpoints and writes a log, since a run here takes about an hour.
"""

import json
import math
import time
from pathlib import Path

import torch

from oromo_llm.generate import complete


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def calc_loss_batch(input_batch, target_batch, model, device):
    """Book listing 5.1 (cross entropy over every position)."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Book listing 5.2 (mean loss over the first num_batches batches)."""
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Book listing 5.3 helper."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def lr_at(step, total_steps, peak_lr, warmup_steps, min_lr):
    """Linear warmup to peak_lr, then cosine decay down to min_lr (Appendix D)."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, model, optimizer, cfg, step, epoch, val_loss):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
    }, path)


def train_model(model, train_loader, val_loader, optimizer, device, cfg, run_dir,
                num_epochs, peak_lr, warmup_frac=0.02, min_lr_frac=0.1, grad_clip=1.0,
                eval_freq=200, eval_iter=50, sample_prompts=()):
    """The book's train_model_simple, plus LR schedule, clipping, checkpoints and logging."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    total_steps = num_epochs * len(train_loader)
    warmup_steps = max(1, int(warmup_frac * total_steps))
    min_lr = peak_lr * min_lr_frac
    tokens_per_step = train_loader.batch_size * train_loader.dataset.max_length

    best_val = float("inf")
    step = 0
    t_last, tokens_since = time.time(), 0
    model.train()

    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            lr =lr_at(step, total_steps, peak_lr, warmup_steps, min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            tokens_since += tokens_per_step

            if step % eval_freq == 0 or step == total_steps:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                elapsed = time.time() - t_last
                record = {
                    "step": step, "epoch": epoch + 1, "lr": lr,
                    "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                    "val_ppl": round(math.exp(val_loss), 1),
                    "tokens_per_sec": round(tokens_since / elapsed),
                }
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(run_dir / "best.pt", model, optimizer, cfg, step, epoch + 1, val_loss)
                    record["saved_best"] = True
                if sample_prompts:
                    record["samples"] = [complete(model, p, max_new_tokens=40) for p in sample_prompts]
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Ep {epoch + 1} (step {step:06d}/{total_steps}): lr {lr:.2e} "
                      f"train {train_loss:.3f} val {val_loss:.3f} (ppl {math.exp(val_loss):.0f}) "
                      f"{record['tokens_per_sec']:,} tok/s", flush=True)
                for s in record.get("samples", []):
                    print("   ", s.replace("\n", " / ")[:200], flush=True)
                t_last, tokens_since = time.time(), 0

        save_checkpoint(run_dir / "last.pt", model, optimizer, cfg, step, epoch + 1, best_val)

    return best_val
