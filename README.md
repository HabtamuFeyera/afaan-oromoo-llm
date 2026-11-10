# Afaan Oromoo LLM: a GPT for Oromo, built from scratch

A GPT-style language model for **Afaan Oromoo**, one of Africa's most widely spoken languages.
Everything from the tokenizer to the training loop is implemented by hand in PyTorch.

The project follows Sebastian Raschka's *Build a Large Language Model (From Scratch)* (Manning, 2025)
and adapts it to a low-resource language. The book trains on English with GPT-2's tokenizer. Here every
stage is rebuilt around Oromo: the data, a tokenizer that understands the hudhaa (`'`), and a model
sized for the amount of Oromo text that actually exists.

## Roadmap

| Phase | What | Book chapters | Status |
|---|---|---|---|
| 1 | Collect, clean and deduplicate an Oromo corpus | 2 | ✅ Done |
| 2 | Train an Oromo tokenizer (byte-level BPE) | 2 | ✅ Done |
| 3 | Implement GPT and pretrain it on Oromo | 3, 4, 5 | 🏃 Training |
| 4 | Fine-tune for classification (sentiment / news topics) | 6 | Planned |
| 5 | Instruction fine-tuning on Oromo instruction–response pairs | 7 | Planned |
| 6 | Adapt a multilingual open model (continued pretraining + LoRA) and compare | App. E | Planned |

## Results so far

### Phase 1: the corpus

| Split | Words | Characters | Tokens (our tokenizer) | Tokens (GPT-2) |
|---|---:|---:|---:|---:|
| train | 6.91M | 53.7M | **11.07M** | 23.1M |
| val | 0.51M | 3.9M | **0.82M** | 1.7M |

**Sources:** the Afaan Oromoo part of the [AfriBERTa corpus](https://huggingface.co/datasets/castorini/afriberta-corpus)
(mostly news) and Oromo Wikipedia.

**Cleaning:**
- Every apostrophe variant becomes one hudhaa character. The raw text was split almost evenly
  between `'` and `’`, so a model would have seen two spellings of words like *baay'ee*.
- News-site leftovers are removed: photo-caption labels and menu text.
- Lines in other scripts or mostly in English are dropped, and so are lines of junk symbols.
- Near-duplicate lines are removed.
- The cleaner keeps a sample of every line it drops, for spot checks. A first pass caught three
  rules that were discarding real Oromo sentences.
- No validation sentence also appears in the training set.

### Phase 2: a tokenizer for Oromo

GPT-2's tokenizer was trained on English and breaks Oromo into fragments:

```
GPT-2 (28 tokens):  A|fa|an| O|rom|oo| af|aan| u|mm|ata| O|rom|oo| kan| Ga|an|fa| Af|rika|a| ke|essa| j|ira|atu|uti|.
Ours  (12 tokens):  Afaan| Oromoo| afaan| ummata| Oromoo| kan| Gaanfa| Afrikaa| keessa| jiraat|uuti|.
```

The tokenizer is byte-level BPE, like GPT-2's, so no input is ever "unknown". It changes one rule
for Oromo: GPT-2 always cuts at an apostrophe, but here the hudhaa stays inside a word, so
`deebi'uun` and `ta'a` become single tokens.

Results on held-out text:

| Tokenizer | Vocab | Chars/token ↑ | Tokens/word ↓ | Whole words ↑ | Rare tokens* ↓ |
|---|---:|---:|---:|---:|---:|
| GPT-2 (the book's) | 50,257 | 2.34 | 3.29 | 7.5% | 91% |
| NaolBM/Oromo-BBPE | 12,000 | 4.59 | 1.68 | 66.7% | 8% |
| ours, 8k | 8,000 | 4.47 | 1.73 | 64.4% | 1.5% |
| **ours, 16k (chosen)** | 16,000 | **4.98** | **1.55** | **73.8%** | 2.8% |
| ours, 24k | 24,000 | 5.24 | 1.47 | 78.5% | 4.5% |
| ours, 32k | 32,000 | 5.41 | 1.42 | 81.5% | 29.5% |

\*Share of the vocabulary seen fewer than 50 times in training. A small model learns almost nothing
about those tokens.

**Why 16k:** it halves sequence length compared with GPT-2, so each 256-token window holds twice as
much Oromo. 24k saves only about 5% more tokens but costs 50% more embedding weights, and 32k leaves
30% of its vocabulary almost untrained.

### Phase 3: pretraining (in progress)

- **Model:** the book's GPT architecture: 23.0M parameters, 6 layers, 6 heads, 384-wide embeddings,
  256-token context, 16k vocabulary.
- **Training:** 4 passes over the training tokens on an Apple M1 Max (MPS), about 20k tokens/s.
- **Optimizer:** AdamW with learning-rate warmup, cosine decay and gradient clipping.
- **Results:** validation loss, perplexity, bits per character, comparison with a unigram baseline,
  and sample text will appear here when the run finishes.

## Project layout

```
oromo_llm/
  tokenizer.py   Oromo tokenizer; drop-in for the book's tiktoken GPT-2 tokenizer
  data.py        the book's GPTDatasetV1 + a faster loader over pre-encoded tokens
  model.py       the book's GPT: multi-head attention, LayerNorm, GELU, transformer blocks
  generate.py    greedy, temperature and top-k generation
  train.py       the book's loss functions and training loop + warmup, cosine decay, clipping
scripts/
  01_download.py            raw data from Hugging Face
  02_build_corpus.py        cleaning, deduplication, train/val split
  03_corpus_report.py       corpus statistics
  04_train_tokenizer.py     BPE tokenizers at 8k-32k
  05_compare_tokenizers.py  tokenizer comparison
  06_encode_corpus.py       corpus -> uint16 token files
  07_pretrain.py            pretraining
  08_evaluate.py            loss, perplexity, bits/char, baseline, samples
tests/                      tokenizer, data loader and model checks
tokenizer/oromo-bpe.json    the chosen 16k tokenizer
```

## Reproduce

```bash
pip install -r requirements.txt

python scripts/01_download.py
python scripts/02_build_corpus.py
python scripts/03_corpus_report.py
python scripts/04_train_tokenizer.py
python scripts/05_compare_tokenizers.py
python scripts/06_encode_corpus.py
python -m pytest tests -q
python scripts/07_pretrain.py
python scripts/08_evaluate.py
```

Steps 1–6 take a few minutes. Pretraining takes about 40 minutes on an M1 Max; it also runs on
CUDA or on CPU (slowly).

## Changes from the book's code

- **Tokenizer:** an Oromo BPE replaces GPT-2's tiktoken encoding. `oromo_llm.tokenizer.get_tokenizer()`
  has the same `encode`/`decode` interface.
- **Attention:** `fast_attention` uses PyTorch's fused `scaled_dot_product_attention`. A test checks
  that it gives the same output as the book's mask-and-softmax code (set it to `False` to run the
  book's version).
- **Training:** learning-rate warmup, cosine decay and gradient clipping, from the book's Appendix D.
  Training also saves checkpoints and writes a log.
- **Generation:** `generate()` works with batches larger than one.

## Acknowledgements and licenses

- The model and training code are adapted from Sebastian Raschka's
  [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache-2.0), the companion code
  for *Build a Large Language Model (From Scratch)*.
- **Data (not stored in this repository; downloaded by the scripts):**
  - AfriBERTa corpus (Apache-2.0), by Ogueji, Zhu and Lin, *Small Data? No Problem!* (2021).
  - Oromo Wikipedia (CC BY-SA 4.0).
- This repository is licensed under Apache-2.0.
