"""Data loaders from chapter 2 of the book, adapted for the Oromo tokenizer.

Two datasets give the same (input, target) windows:
- GPTDatasetV1: the book's listing 2.5, for raw text (small experiments).
- TokenFileDataset: reads the pre-encoded data/tokens/*.bin files with numpy.memmap,
  so the 11M-token corpus is never re-tokenized or held as Python lists.

Each item is (x, y), where y is x shifted one token to the right, and
windows start every `stride` tokens.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from oromo_llm.tokenizer import get_tokenizer

TOKENS_DIR = Path(__file__).resolve().parent.parent / "data" / "tokens"


class GPTDatasetV1(Dataset):
    """Book listing 2.5, unchanged except for the tokenizer it is given."""

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    """Book listing 2.6 with the Oromo tokenizer in place of tiktoken's GPT-2."""
    dataset = GPTDatasetV1(txt, get_tokenizer(), max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)


class TokenFileDataset(Dataset):
    """Same windows as GPTDatasetV1, read from a uint16 token file."""

    def __init__(self, path, max_length, stride):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.max_length, self.stride = max_length, stride
        self.n = len(range(0, len(self.tokens) - max_length, stride))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.stride
        chunk = torch.from_numpy(self.tokens[i:i + self.max_length + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def create_token_dataloader(split="train", batch_size=8, max_length=256, stride=None,
                            shuffle=None, drop_last=True, num_workers=0):
    """Loader over data/tokens/<split>.bin. stride defaults to max_length (no overlap),
    as the book recommends; shuffle defaults to True for train only."""
    stride = stride or max_length
    shuffle = (split == "train") if shuffle is None else shuffle
    dataset = TokenFileDataset(TOKENS_DIR / f"{split}.bin", max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)
