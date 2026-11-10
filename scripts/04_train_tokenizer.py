"""Phase 2, step 1: train byte-level BPE tokenizers on the Oromo corpus.

Same family as GPT-2's tokenizer (byte-level BPE, so no unknown tokens), with
one Oromo-specific change: the pre-tokenizer keeps the hudhaa (') inside a word,
so "baay'ee" or "deebi'uun" can become single tokens. GPT-2's rule always cuts
at the apostrophe. A leading or trailing ' (a quote mark) is still split off.

Trains one tokenizer per vocabulary size and saves each as
tokenizer/oromo-bpe-<size>.json. Evaluation lives in 05_compare_tokenizers.py.

Usage: python scripts/04_train_tokenizer.py [sizes...]   (default: 8000 16000 24000 32000)
"""

import sys
import time
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "processed" / "train.txt"
OUT = ROOT / "tokenizer"

EOT = "<|endoftext|>"
# Words (letters joined by inner apostrophes), numbers, punctuation runs, whitespace.
# Like GPT-2, a word carries its leading space.
PRETOKENIZE_PATTERN = r" ?\p{L}+(?:'\p{L}+)*| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZE_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def iter_documents(path: Path, chunk_lines: int = 1000):
    """Yield chunks of text; <|endoftext|> separator lines are skipped."""
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == EOT:
                continue
            buf.append(line)
            if len(buf) >= chunk_lines:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)


def train(vocab_size: int) -> Path:
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOT],  # id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(iter_documents(TRAIN), trainer=trainer)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"oromo-bpe-{vocab_size}.json"
    tok.save(str(path))
    return path


def main():
    sizes = [int(s) for s in sys.argv[1:]] or [8000, 16000, 24000, 32000]
    for size in sizes:
        t0 = time.time()
        path = train(size)
        tok = Tokenizer.from_file(str(path))
        print(f"vocab {size:>6}: saved {path.relative_to(ROOT)} "
              f"(actual vocab {tok.get_vocab_size()}, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
