"""Instruction fine-tuning from chapter 7 of the book (listings 7.1-7.6).

Follows the book, with these changes:
- Prompt format: the Alpaca layout with Oromo section names:
  "### Qajeelfama:" (instruction), "### Galtee:" (input), "### Deebii:" (response).
  The long English preamble is dropped to save room in the 256-token context.
- Long inputs (news articles) are truncated so that instruction + input + answer
  fit the context. The book's dataset items were short enough not to need this.
- Prompt tokens are masked out of the loss, so the model learns only to write
  the answer. This is the book's exercise 7.2 variant; the main text masks
  only the padding.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from oromo_llm.generate import generate
from oromo_llm.tokenizer import get_tokenizer

IGNORE_INDEX = -100
MAX_OUTPUT_TOKENS = 96


def format_prompt(entry):
    """Book listing 7.2 (format_input), with Oromo section names."""
    text = f"### Qajeelfama:\n{entry['instruction']}"
    if entry.get("input"):
        text += f"\n\n### Galtee:\n{entry['input']}"
    return text + "\n\n### Deebii:\n"


def encode_example(entry, context_length, tokenizer=None):
    """Token IDs of prompt + answer + <|endoftext|>, and the prompt length.

    If too long, the input text is cut (the instruction and the answer are kept).
    The sequence has at most context_length + 1 tokens, because inputs and
    targets are the sequence shifted by one.
    """
    tok = tokenizer or get_tokenizer()
    head = tok.encode(f"### Qajeelfama:\n{entry['instruction']}")
    body = tok.encode(f"\n\n### Galtee:\n{entry['input']}") if entry.get("input") else []
    tail = tok.encode("\n\n### Deebii:\n")
    answer = tok.encode(entry["output"])[:MAX_OUTPUT_TOKENS] + [tok.eot_token] if "output" in entry else []
    budget = context_length + 1 - len(head) - len(tail) - len(answer)
    prompt = head + body[:max(0, budget)] + tail
    return prompt + answer, len(prompt)


class InstructionDataset(Dataset):
    """Book listing 7.4, storing (token IDs, prompt length) per example."""

    def __init__(self, entries, context_length):
        self.items = [encode_example(e, context_length) for e in entries]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_token_id=0, ignore_index=IGNORE_INDEX, mask_prompt=True, device="cpu"):
    """Book listing 7.5 (custom_collate_fn), plus optional prompt masking.

    Pads with <|endoftext|>. Targets are the inputs shifted by one. Padding after
    the first <|endoftext|> is set to -100 so the loss ignores it, and so are
    the prompt tokens when mask_prompt is on.
    """
    max_len = max(len(ids) for ids, _ in batch)
    inputs, targets = [], []
    for ids, prompt_len in batch:
        padded = ids + [pad_token_id] * (max_len - len(ids))
        x = torch.tensor(padded[:-1])
        y = torch.tensor(padded[1:])
        y[len(ids) - 1:] = ignore_index      # padding (the real <|endoftext|> stays a target)
        if mask_prompt:
            y[:prompt_len - 1] = ignore_index  # the prompt itself
        inputs.append(x)
        targets.append(y)
    return torch.stack(inputs).to(device), torch.stack(targets).to(device)


def load_entries(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@torch.no_grad()
def respond(model, entry, device, max_new_tokens=48, temperature=0.0, top_k=None):
    """Generate the model's answer to one instruction (book section 7.7)."""
    tok = get_tokenizer()
    ctx = model.pos_emb.weight.shape[0]
    prompt, _ = encode_example({k: v for k, v in entry.items() if k != "output"},
                               ctx - max_new_tokens)
    model.eval()
    idx = torch.tensor([prompt], device=device)
    out = generate(model, idx, max_new_tokens, ctx, temperature=temperature, top_k=top_k,
                   eos_id=tok.eot_token)
    return tok.decode(out[0, len(prompt):].tolist()).strip()
