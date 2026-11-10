"""Checks for the Phase 3 model. Run: python -m pytest tests -q"""

import copy
import math

import torch

from oromo_llm.generate import generate, text_to_token_ids, token_ids_to_text
from oromo_llm.model import OROMO_CONFIG_23M, GPTModel, count_parameters
from oromo_llm.train import calc_loss_batch

SMALL = {**OROMO_CONFIG_23M, "emb_dim": 64, "n_heads": 4, "n_layers": 2, "context_length": 32}


def test_fast_attention_matches_book_attention():
    torch.manual_seed(0)
    fast = GPTModel({**SMALL, "fast_attention": True}).eval()
    book = GPTModel({**SMALL, "fast_attention": False}).eval()
    book.load_state_dict(fast.state_dict())
    x = torch.randint(0, SMALL["vocab_size"], (2, 20))
    with torch.no_grad():
        assert torch.allclose(fast(x), book(x), atol=1e-5)


def test_full_config_size_and_shapes():
    model = GPTModel(OROMO_CONFIG_23M)
    n = count_parameters(model)
    assert 22e6 < n < 24e6, n
    x = torch.randint(0, 16000, (2, 256))
    assert model(x).shape == (2, 256, 16000)


def test_initial_loss_near_uniform():
    torch.manual_seed(0)
    model = GPTModel(SMALL).eval()
    x = torch.randint(0, SMALL["vocab_size"], (4, 32))
    loss = calc_loss_batch(x[:, :-1], x[:, 1:], model, "cpu").item()
    assert abs(loss - math.log(SMALL["vocab_size"])) < 0.5


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = GPTModel({**SMALL, "drop_rate": 0.0})
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ids = text_to_token_ids("Afaan Oromoo afaan ummata Oromoo kan Gaanfa Afrikaa keessa jiraatuuti.")
    x, y = ids[:, :-1], ids[:, 1:]
    for _ in range(60):
        opt.zero_grad()
        loss = calc_loss_batch(x, y, model, "cpu")
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
    model.eval()
    out = generate(model, ids[:, :3], max_new_tokens=x.shape[1] - 2, context_size=32)
    assert token_ids_to_text(out).startswith("Afaan Oromoo afaan ummata")
