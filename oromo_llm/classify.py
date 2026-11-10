"""Classification fine-tuning from chapter 6 of the book (listings 6.4-6.10).

Follows the book: a pretrained GPT gets a new output head with one output per
class, its loss and accuracy are computed from the last token's output, and it
trains with AdamW and cross entropy. Changes for this task:
- The class logits are read at the last *real* token, not the last position.
  The book's spam messages were padded and read at position -1. News articles
  are truncated to the context length, so this rarely matters, but it is
  correct for short texts.
- Macro-F1 is reported next to accuracy, because the topics are unbalanced.
- The best epoch is chosen on the dev set.
"""

import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from oromo_llm.model import GPTModel
from oromo_llm.tokenizer import get_tokenizer


class TextClassificationDataset(Dataset):
    """Book listing 6.4 (SpamDataset), for JSONL files with "text" and "label_id"."""

    def __init__(self, jsonl_path, max_length, tokenizer=None):
        tokenizer = tokenizer or get_tokenizer()
        rows = [json.loads(l) for l in Path(jsonl_path).read_text(encoding="utf-8").splitlines()]
        pad = tokenizer.eot_token
        self.labels = [r["label_id"] for r in rows]
        self.inputs, self.lengths = [], []
        for ids in tokenizer.encode_batch([r["text"] for r in rows]):
            ids = ids[:max_length]
            self.lengths.append(len(ids))
            self.inputs.append(ids + [pad] * (max_length - len(ids)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.tensor(self.inputs[i]), torch.tensor(self.lengths[i]),
                torch.tensor(self.labels[i]))


def make_loader(path, max_length, batch_size, shuffle, seed=0):
    ds = TextClassificationDataset(path, max_length)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, generator=g)


def build_classifier(cfg, num_classes, pretrained_state=None, trainable="last_block"):
    """Book section 6.5: new output head plus the choice of which layers train.

    trainable: "last_block" (the book's recipe: last transformer block, final
               norm and the new head) or "all".
    """
    model = GPTModel(cfg)
    if pretrained_state is not None:
        model.load_state_dict(pretrained_state)
    model.out_head = torch.nn.Linear(cfg["emb_dim"], num_classes)
    if trainable == "last_block":
        for p in model.parameters():
            p.requires_grad = False
        for module in (model.trf_blocks[-1], model.final_norm, model.out_head):
            for p in module.parameters():
                p.requires_grad = True
    return model


def class_logits(model, inputs, lengths):
    logits = model(inputs)
    return logits[torch.arange(inputs.shape[0], device=inputs.device), lengths - 1]


def calc_loss_batch(batch, model, device):
    """Book listing 6.8, reading the last real token."""
    inputs, lengths, labels = (t.to(device) for t in batch)
    return torch.nn.functional.cross_entropy(class_logits(model, inputs, lengths), labels)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, gold = [], []
    for inputs, lengths, labels in loader:
        logits = class_logits(model, inputs.to(device), lengths.to(device))
        preds += logits.argmax(dim=-1).cpu().tolist()
        gold += labels.tolist()
    return preds, gold


def scores(preds, gold, num_classes):
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and g == c for p, g in zip(preds, gold))
        fp = sum(p == c and g != c for p, g in zip(preds, gold))
        fn = sum(p != c and g == c for p, g in zip(preds, gold))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"accuracy": acc, "macro_f1": sum(f1s) / num_classes, "per_class_f1": f1s}


def train_classifier(model, train_loader, dev_loader, device, num_classes, lr, num_epochs,
                     weight_decay=0.1, log=print):
    """Book listing 6.10 (train_classifier_simple), keeping the best epoch on dev macro-F1."""
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
    best = {"macro_f1": -1.0}
    best_state = None
    history = []
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(batch, model, device)
            loss.backward()
            optimizer.step()
            total += loss.item()
        dev = scores(*predict(model, dev_loader, device), num_classes)
        history.append({"epoch": epoch + 1, "train_loss": total / len(train_loader),
                        "dev_accuracy": dev["accuracy"], "dev_macro_f1": dev["macro_f1"]})
        log(f"    epoch {epoch + 1}: train loss {total / len(train_loader):.3f} | "
            f"dev acc {dev['accuracy']:.3f} | dev macro-F1 {dev['macro_f1']:.3f}")
        if dev["macro_f1"] > best["macro_f1"]:
            best = {**dev, "epoch": epoch + 1}
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best, history
