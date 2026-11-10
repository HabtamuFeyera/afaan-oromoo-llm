"""Checks for the Phase 5 instruction-tuning code. Run: python -m pytest tests -q"""

import torch

from oromo_llm.instruct import IGNORE_INDEX, collate, encode_example, format_prompt
from oromo_llm.tokenizer import get_tokenizer

tok = get_tokenizer()
ENTRY = {"instruction": "Oduun armaan gadii waa'ee maalii?", "input": "Atileetonni Oromoo injifatan.",
         "output": "ispoortii"}


def test_pieces_encode_like_the_whole_prompt():
    ids, prompt_len = encode_example(ENTRY, context_length=256)
    assert ids == tok.encode(format_prompt(ENTRY) + ENTRY["output"]) + [tok.eot_token]
    assert tok.decode(ids[:prompt_len]) == format_prompt(ENTRY)


def test_long_input_is_truncated_but_answer_kept():
    long = {**ENTRY, "input": "Oromoon ni dubbate. " * 300}
    ids, prompt_len = encode_example(long, context_length=256)
    assert len(ids) == 257
    assert tok.decode(ids[prompt_len:-1]) == "ispoortii" and ids[-1] == tok.eot_token
    assert tok.decode(ids[:prompt_len]).endswith("### Deebii:\n")


def test_collate_masks_prompt_and_padding():
    a = encode_example(ENTRY, 256)
    b = encode_example({**ENTRY, "input": "Gabaabaa."}, 256)
    x, y = collate([a, b])
    assert x.shape == y.shape
    ids, plen = a
    kept = (y[0] != IGNORE_INDEX).nonzero().flatten().tolist()
    # Only the answer tokens and the closing <|endoftext|> are targets.
    assert [y[0, i].item() for i in kept] == ids[plen:]
    assert kept[0] == plen - 1
    assert (y[1] != IGNORE_INDEX).sum() == len(b[0]) - b[1]


def test_collate_without_prompt_mask_keeps_prompt_targets():
    x, y = collate([encode_example(ENTRY, 256)], mask_prompt=False)
    assert (y != IGNORE_INDEX).all()
    assert torch.equal(x[0, 1:], y[0, :-1])
