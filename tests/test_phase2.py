"""Checks for the Phase 2 tokenizer and data loaders. Run: python -m pytest tests -q"""

import numpy as np
import torch

from oromo_llm.data import GPTDatasetV1, TOKENS_DIR, TokenFileDataset, create_token_dataloader
from oromo_llm.tokenizer import EOT, get_tokenizer

tok = get_tokenizer()


def test_roundtrip_and_hudhaa():
    text = "Baay'ee galatoomaa, deebi'uun isaa ni ta'a. 'Obboleessi koo' jedhe."
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode(" deebi'uun")) == 1


def test_endoftext_is_special():
    ids = tok.encode(f"Akkam? {EOT} Nagaa.", allowed_special={EOT})
    assert ids.count(tok.eot_token) == 1 and tok.eot_token == 0


def test_book_dataset_matches_token_file(tmp_path):
    text = (TOKENS_DIR.parent / "processed" / "val.txt").read_text(encoding="utf-8")[:20000]
    ids = np.array(tok.encode(text), dtype=np.uint16)
    path = tmp_path / "sample.bin"
    ids.tofile(path)
    book = GPTDatasetV1(text, tok, max_length=32, stride=16)
    ours = TokenFileDataset(path, max_length=32, stride=16)
    assert len(book) == len(ours) > 0
    for i in (0, len(book) // 2, len(book) - 1):
        assert torch.equal(book[i][0], ours[i][0]) and torch.equal(book[i][1], ours[i][1])


def test_loader_shapes_and_shift():
    x, y = next(iter(create_token_dataloader("val", batch_size=4, max_length=256, shuffle=False)))
    assert x.shape == y.shape == (4, 256) and x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert int(x.max()) < tok.n_vocab
