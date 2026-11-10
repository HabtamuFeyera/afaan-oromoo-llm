"""Phase 6, step 3: instruction-tune a larger open model on the Phase 5 data (LoRA).

Default model: McGill-NLP/AfriqueQwen3.5-0.8B-50Langs (Qwen3.5-0.8B after
African-language continued pretraining, Oromo included).

Same data, prompt format (Oromo Alpaca: ### Qajeelfama / ### Galtee / ### Deebii)
and prompt masking as Phase 5. Long articles are cut so the whole example fits
--max-len tokens. Only LoRA adapters train.

Evaluation on a fixed test subset (generation with this architecture is slow on
a Mac): the first 150 test articles for both topic and headline, plus all 81
"describe" items. Scored with oromo_llm/metrics.py, the same code as Phase 5,
for three systems:
- the 23M instruction-tuned model from Phase 5 (its saved answers, same items)
- the open model before instruction tuning (same prompt)
- the open model after LoRA instruction tuning

Outputs: checkpoints/<run-name>/{adapter/, evaluation.json, test_responses.json, log.json}
Usage: python scripts/18_instruct_open_model.py [--epochs 1] [--max-len 384]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.instruct import load_entries  # noqa: E402
from oromo_llm.metrics import evaluate  # noqa: E402
from oromo_llm.train import get_device, lr_at  # noqa: E402

DATA = ROOT / "data" / "instructions"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "out_proj"]
N_NEWS_TEST = 150


def encode(tok, entry, max_len, with_answer=True):
    """Prompt (+ answer + EOS) token IDs and the prompt length; the input is cut to fit."""
    head = tok(f"### Qajeelfama:\n{entry['instruction']}", add_special_tokens=False)["input_ids"]
    body = tok(f"\n\n### Galtee:\n{entry['input']}", add_special_tokens=False)["input_ids"] if entry.get("input") else []
    tail = tok("\n\n### Deebii:\n", add_special_tokens=False)["input_ids"]
    answer = (tok(entry["output"], add_special_tokens=False)["input_ids"][:96] + [tok.eos_token_id]) if with_answer else []
    budget = max_len - len(head) - len(tail) - len(answer)
    prompt = head + body[:max(0, budget)] + tail
    return prompt + answer, len(prompt)


def batches(items, batch_size, pad_id, shuffle, seed=0):
    order = torch.randperm(len(items), generator=torch.Generator().manual_seed(seed)).tolist() if shuffle \
        else list(range(len(items)))
    for i in range(0, len(order) - (batch_size - 1 if shuffle else 0), batch_size):
        chunk = [items[j] for j in order[i:i + batch_size]]
        width = max(len(ids) for ids, _ in chunk)
        x = torch.full((len(chunk), width), pad_id)
        y = torch.full((len(chunk), width), -100)
        att = torch.zeros((len(chunk), width), dtype=torch.long)
        for k, (ids, plen) in enumerate(chunk):
            x[k, :len(ids)] = torch.tensor(ids)
            att[k, :len(ids)] = 1
            y[k, plen:len(ids)] = torch.tensor(ids[plen:])  # loss on the answer only
        yield x, y, att


def test_subset(entries):
    news = [e for e in entries if e["task"] in ("topic", "headline")][:2 * N_NEWS_TEST]
    return news + [e for e in entries if e["task"] == "describe"], \
        [i for i, e in enumerate(entries) if e["task"] in ("topic", "headline")][:2 * N_NEWS_TEST] + \
        [i for i, e in enumerate(entries) if e["task"] == "describe"]


@torch.no_grad()
def generate_all(model, tok, entries, max_len, device):
    model.eval()
    out = []
    for i, e in enumerate(entries):
        n_new = 8 if e["task"] == "topic" else 40
        prompt, _ = encode(tok, e, max_len - n_new, with_answer=False)
        x = torch.tensor([prompt], device=device)
        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n_new,
                           do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        out.append(tok.decode(g[0, len(prompt):], skip_special_tokens=True).strip())
        if (i + 1) % 50 == 0:
            print(f"    generated {i + 1}/{len(entries)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="McGill-NLP/AfriqueQwen3.5-0.8B-50Langs")
    ap.add_argument("--run-name", default="afrique-0.8b-instruct-lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--limit-train", type=int, default=None, help="use only N training examples (smoke tests)")
    ap.add_argument("--limit-test", type=int, default=None, help="evaluate only N test items (smoke tests)")
    ap.add_argument("--skip-base-eval", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(123)
    device = get_device()
    out = ROOT / "checkpoints" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)

    test_all = load_entries(DATA / "test.json")
    test, test_idx = test_subset(test_all)
    if args.limit_test:
        keep = list(range(0, len(test), max(1, len(test) // args.limit_test)))[:args.limit_test]
        test, test_idx = [test[i] for i in keep], [test_idx[i] for i in keep]
    results, responses = {}, {}

    # The Phase 5 model's saved answers, scored on the same items.
    p5 = ROOT / "checkpoints" / "oromo-gpt-23m-instruct" / "test_responses.json"
    if p5.exists():
        saved = json.loads(p5.read_text(encoding="utf-8"))
        responses["23M instruction-tuned (Phase 5)"] = [saved[i]["instruction-tuned"] for i in test_idx]
        results["23M instruction-tuned (Phase 5)"] = evaluate(responses["23M instruction-tuned (Phase 5)"], test)

    if not args.skip_base_eval:
        print("evaluating the model before instruction tuning...", flush=True)
        responses["before instruction tuning"] = generate_all(model, tok, test, args.max_len, device)
        results["before instruction tuning"] = evaluate(responses["before instruction tuning"], test)
        print("  ", results["before instruction tuning"], flush=True)

    train = load_entries(DATA / "train.json")[:args.limit_train]
    items = [encode(tok, e, args.max_len) for e in train]
    val_items = [encode(tok, e, args.max_len) for e in load_entries(DATA / "val.json")]
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * (len(items) // args.batch_size)
    print(f"training: {len(items)} examples, {sum(len(i) for i, _ in items):,} tokens/epoch, {steps} steps",
          flush=True)

    log, step, t0 = [], 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        running = []
        for x, y, att in batches(items, args.batch_size, tok.eos_token_id, shuffle=True, seed=epoch):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps, args.lr, max(1, steps // 30), args.lr * 0.1)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss = model(input_ids=x.to(device), attention_mask=att.to(device), labels=y.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            running.append(loss.item())
            step += 1
            if step % 50 == 0 or step == steps:
                eta = (steps - step) * (time.time() - t0) / step / 60
                print(f"  step {step}/{steps} | loss {sum(running) / len(running):.3f} | ~{eta:.0f} min left",
                      flush=True)
                log.append({"step": step, "train_loss": sum(running) / len(running)})
                running = []
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            vl = [model(input_ids=x.to(device), attention_mask=att.to(device), labels=y.to(device)).loss.item()
                  for x, y, att in batches(val_items, 8, tok.eos_token_id, shuffle=False)]
        log.append({"epoch": epoch + 1, "val_loss": sum(vl) / len(vl)})
        print(f"  epoch {epoch + 1}: val loss {sum(vl) / len(vl):.3f}", flush=True)
    model.save_pretrained(out / "adapter")

    print("evaluating after instruction tuning...", flush=True)
    responses["after LoRA instruction tuning"] = generate_all(model, tok, test, args.max_len, device)
    results["after LoRA instruction tuning"] = evaluate(responses["after LoRA instruction tuning"], test)

    review = [{"task": e["task"], "instruction": e["instruction"], "input": e["input"][:300], "gold": e["output"],
               **{k: v[i] for k, v in responses.items()}} for i, e in enumerate(test)]
    (out / "test_responses.json").write_text(json.dumps(review, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "evaluation.json").write_text(json.dumps(results, indent=2))
    (out / "log.json").write_text(json.dumps({"args": vars(args), "minutes": round((time.time() - t0) / 60, 1),
                                              "log": log}, indent=2))
    for name, r in results.items():
        print(f"\n{name}")
        for task, m in r.items():
            print(f"  {task:<9} " + " | ".join(f"{k} {v:.3f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
