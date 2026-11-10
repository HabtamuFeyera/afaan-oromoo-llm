"""Oromo tokenizer with the same interface as the book's tiktoken GPT-2 tokenizer.

Use it wherever the book writes ``tiktoken.get_encoding("gpt2")``:

    from oromo_llm.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    ids = tokenizer.encode("Akkam jirta?")
    text = tokenizer.decode(ids)

Differences from tiktoken
- ``<|endoftext|>`` in the text is always encoded as the special token (id 0).
  ``allowed_special`` is accepted so book code runs unchanged, but it is not needed.
- ``n_vocab`` is 16,000, so the model config needs ``"vocab_size": tokenizer.n_vocab``.
"""

from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "oromo-bpe.json"
EOT = "<|endoftext|>"


class OromoTokenizer:
    def __init__(self, path: Path = TOKENIZER_PATH):
        self._tok = Tokenizer.from_file(str(path))
        self.n_vocab = self._tok.get_vocab_size()
        self.eot_token = self._tok.token_to_id(EOT)

    def encode(self, text: str, allowed_special=None) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def id_to_token(self, i: int) -> str:
        """The readable text of one token (for inspection)."""
        return self.decode([i])


@lru_cache(maxsize=None)
def get_tokenizer() -> OromoTokenizer:
    return OromoTokenizer()
