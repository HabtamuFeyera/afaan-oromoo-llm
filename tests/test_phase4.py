"""Checks for the Phase 4 classification code. Run: python -m pytest tests -q"""

import json

import torch

from oromo_llm.classify import (TextClassificationDataset, build_classifier, class_logits, scores)
from oromo_llm.model import OROMO_CONFIG_23M
from oromo_llm.text import normalize
from oromo_llm.tokenizer import get_tokenizer

SMALL = {**OROMO_CONFIG_23M, "emb_dim": 64, "n_heads": 4, "n_layers": 3, "context_length": 32}


def test_normalize_unifies_hudhaa_and_strips_labels():
    assert normalize("baay’ee  deebi‘uun") == "baay'ee deebi'uun"
    assert normalize("Madda suuraa, Getty Images Oromoon ni dubbate.") == "Oromoon ni dubbate."


def test_book_recipe_trains_only_last_block_norm_and_head():
    model = build_classifier(SMALL, 5, trainable="last_block")
    trainable = {n.split(".")[0] + ("." + n.split(".")[1] if n.startswith("trf_blocks") else "")
                 for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"trf_blocks.2", "final_norm", "out_head"}
    assert model.out_head.out_features == 5
    assert all(p.requires_grad for p in build_classifier(SMALL, 5, trainable="all").parameters())


def test_pads_truncates_and_reads_last_real_token(tmp_path):
    path = tmp_path / "d.jsonl"
    rows = [{"text": "Akkam jirta?", "label_id": 1}, {"text": "Oromoon " * 100, "label_id": 3}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    ds = TextClassificationDataset(path, max_length=32)
    (x0, len0, y0), (x1, len1, _) = ds[0], ds[1]
    assert x0.shape == x1.shape == (32,) and y0.item() == 1
    assert len1.item() == 32 and 0 < len0.item() < 32
    assert (x0[len0:] == get_tokenizer().eot_token).all()

    model = build_classifier(SMALL, 5).eval()
    x, lengths = torch.stack([x0, x1]), torch.stack([len0, len1])
    with torch.no_grad():
        picked = class_logits(model, x, lengths)
        full = model(x)
    assert torch.allclose(picked[0], full[0, len0 - 1]) and torch.allclose(picked[1], full[1, -1])


def test_macro_f1():
    s = scores([0, 0, 1, 2], [0, 1, 1, 2], 3)
    assert s["accuracy"] == 0.75
    assert abs(s["macro_f1"] - (2 / 3 + 2 / 3 + 1) / 3) < 1e-9
