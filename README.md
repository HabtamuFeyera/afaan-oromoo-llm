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
| 3 | Implement GPT and pretrain it on Oromo | 3, 4, 5 | ✅ Done |
| 4 | Fine-tune for classification (news topics) | 6 | ✅ Done |
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

### Phase 3: pretraining

- **Model:** the book's GPT architecture: 23.0M parameters, 6 layers, 6 heads, 384-wide embeddings,
  256-token context, 16k vocabulary.
- **Training:** 4 passes over the 11M training tokens (5,400 steps of 32 × 256 tokens).
  It took 42 minutes on an Apple M1 Max (MPS) at about 18k tokens/s.
- **Optimizer:** AdamW with weight decay 0.1. The learning rate warms up to 6e-4, then follows a
  cosine decay; gradients are clipped at 1.0.

| Step | Pass | Train loss | Val loss | Val perplexity |
|---:|---:|---:|---:|---:|
| 0 | – | – | 9.85 | ~19,000 (random) |
| 250 | 1 | 7.26 | 7.24 | 1,389 |
| 1,000 | 1 | 5.89 | 5.93 | 376 |
| 2,500 | 2 | 5.24 | 5.39 | 220 |
| 4,000 | 3 | 4.99 | 5.20 | 181 |
| 5,400 | 4 | 4.91 | **5.14** | **171** |

Evaluation of the final model on the whole validation set (`scripts/08_evaluate.py`):

| Metric | GPT (23M) | Unigram baseline* |
|---|---:|---:|
| Validation loss | 5.14 | 8.05 |
| Perplexity (per token) | 171 | 3,147 |
| Bits per character | **1.54** | 2.41 |
| Generated words that are real Oromo words (T=0.7) | 93% | – |

\*Knows only how often each token appears, with no context. The gap to the GPT is what the model
learned about context.

**Samples** (temperature 0.7, top-k 50; the prompt is in bold, ` / ` is a line break):

> **Mootummaan naannoo Oromiyaa** keessa jiran hedduu kan himan Obbo Lammaan, ammas hojiin amma hin
> jirretti waan hin arganne. / Obbo Lidatuu Ayyaaloo, Obbo Jawaar Mahaammad fi Obbo Lammaa Magarsaa
> akka ta'an himaniiru. / ''Humni addaa kun kan raawwatamedha,'' jedhan.

> **Barnoonni** akka jedhanitti, namoota mana mareettis miidhaan qaqqabu akka hin uumamne. / Garuu kan
> isaan irratti bobba'an yoo tahu, isaan keessaa tokko.

**Findings**
- **The output is Oromo, not yet meaning.** Words, the hudhaa (*ta'an*, *bobba'an*), quoting style
  (*''…'' jedhan*) and news-report phrasing are right. The model has absorbed the style of its
  mostly-news training data: names of officials, *jedhan* ("they said"). Sentences are locally
  fluent but don't add up to a coherent story. That's expected at 23M parameters and 11M tokens.
- **Greedy decoding loops.** At temperature 0 the model repeats itself
  (*"Afaan Oromoo, Afaan Oromoo, …"*). Temperature and top-k sampling, from chapter 5 of the book,
  fix this; 0.7 works best here.
- **No overfitting, so more data is the bottleneck.** Validation loss fell at every checkpoint, and
  the train/validation gap stayed small (0.23). More Oromo text, or a larger model with more data,
  should keep improving it.

### Phase 4: news-topic classification

**Task:** [MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews) Oromo. Classify a BBC
Afaan Oromoo article into one of 5 topics: politics, health, sports, entertainment, technology.
There are 1,128 training, 162 dev and 325 test articles, and the topics are unbalanced (technology
has only 50 training articles). So **macro-F1**, which weights every topic equally, is reported
alongside accuracy.

**Method:** following chapter 6 of the book, the 16k-word output layer is replaced by a 5-class
head, and the class is read from the model's output at the last token. Each GPT setup tries
3 learning rates, keeps the epoch with the best dev score (up to 10 epochs), and reruns the best
learning rate with 3 seeds. Test results are mean ± standard deviation over those seeds.

| Setup | Trained params | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| Majority class | – | 30.8% | 0.094 |
| TF-IDF + logistic regression | – | 86.8% | 0.749 |
| Our GPT, **random init**, all layers | 16.9M | 73.1% ± 3.1 | 0.558 ± 0.022 |
| Our GPT, **pretrained**, book recipe (last block + norm + head) | 1.8M | 81.0% ± 0.4 | 0.672 ± 0.033 |
| Our GPT, **pretrained**, all layers | 16.9M | **87.2% ± 0.9** | **0.779 ± 0.015** |

F1 per topic (mean over seeds):

| Topic (training articles) | TF-IDF | GPT random init | GPT pretrained, all layers |
|---|---:|---:|---:|
| sports (280) | 0.98 | 0.90 | 0.93 |
| health (338) | 0.88 | 0.80 | 0.91 |
| politics (350) | 0.87 | 0.76 | 0.90 |
| entertainment (110) | 0.60 | 0.23 | 0.58 |
| technology (50) | 0.42 | 0.11 | **0.58** |

**Findings**
- **Pretraining is what makes it work.** The same architecture trained only on the 1,128 labeled
  articles reaches macro-F1 0.56. Starting from the Phase 3 weights raises it to 0.78
  (+0.22 macro-F1, +14 accuracy points). The gain is largest on the rare topics: technology goes
  from 0.11 to 0.58, because the pretrained model already knows Oromo words it sees only a few
  times in the labeled data.
- **The fully fine-tuned GPT slightly beats a strong classic baseline.** Accuracy is level with
  TF-IDF (87.2% vs 86.8%, inside the seed noise). Macro-F1 is higher (0.779 vs 0.749), almost all
  from technology. TF-IDF is still better on sports, where keywords like team names decide the topic.
- **The book's recipe is a good trade-off.** Training only the last block, the final norm and the
  head (1.8M parameters, a tenth of the model) recovers most of the gain over random init.
  Training all layers adds another 0.11 macro-F1.
- **Contamination was checked.** The articles and much of the pretraining text both come from
  BBC Afaan Oromoo, so `scripts/09_prepare_news.py` measures 13-word overlap (the GPT-3 paper's
  method). Only 1 of 325 test articles appears mostly in the pretraining text (on average, 5.5% of
  an article's 13-grams do). Scores without that article are unchanged to three decimals.

## Project layout

```
oromo_llm/
  tokenizer.py   Oromo tokenizer; drop-in for the book's tiktoken GPT-2 tokenizer
  data.py        the book's GPTDatasetV1 + a faster loader over pre-encoded tokens
  model.py       the book's GPT: multi-head attention, LayerNorm, GELU, transformer blocks
  generate.py    greedy, temperature and top-k generation
  train.py       the book's loss functions and training loop + warmup, cosine decay, clipping
  classify.py    the book's classification fine-tuning (chapter 6) + macro-F1
  text.py        Oromo text normalization (hudhaa, quotes, site labels)
scripts/
  01_download.py            raw data from Hugging Face
  02_build_corpus.py        cleaning, deduplication, train/val split
  03_corpus_report.py       corpus statistics
  04_train_tokenizer.py     BPE tokenizers at 8k-32k
  05_compare_tokenizers.py  tokenizer comparison
  06_encode_corpus.py       corpus -> uint16 token files
  07_pretrain.py            pretraining
  08_evaluate.py            loss, perplexity, bits/char, baseline, samples
  09_prepare_news.py        MasakhaNEWS Oromo + contamination check
  10_finetune_news.py       topic classification: baselines vs fine-tuned GPTs
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
python scripts/09_prepare_news.py
python scripts/10_finetune_news.py --epochs 10
```

Steps 1–6 take a few minutes. On an M1 Max, pretraining takes about 42 minutes and the
classification experiments about 30 minutes. Both also run on CUDA, or slowly on CPU.

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
  - MasakhaNEWS (AFL-3.0), by Adelani et al., *MasakhaNEWS: News Topic Classification for African
    Languages* (2023).
- This repository is licensed under Apache-2.0.
