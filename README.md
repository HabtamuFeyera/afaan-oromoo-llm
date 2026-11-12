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
| 5 | Instruction fine-tuning on Oromo instruction–response pairs | 7 | ✅ Done (learns the format, not the content) |
| 6 | Compare with open multilingual models; LoRA continued pretraining and instruction tuning | App. E | ✅ Done |

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

#### Adding Telegram news channels (local only)

Public Afaan Oromoo Telegram channels, exported with Telegram Desktop ("Export chat history"),
are imported by `scripts/14_import_telegram.py`. The posts are copyrighted by their channels, so
they stay in the git-ignored `data/` folder; only the importer is published.

The first import used 4 channels (OBN, TIKVAH Afaan Oromoo, Jawar Mohammed, Oromia Times; 1.16M
words). The second (v2) uses 19 channels, about 6.66M words before line filtering. The largest:

| Channel | Posts with text | Words kept |
|---|---:|---:|
| FMC Afaan Oromoo | 42,598 | 2.64M |
| OBN (Oromia Broadcasting Network) | 6,303 | 0.83M |
| Burqaa Ispoortii (sports) | 115,594 | 0.75M |
| TIKVAH Afaan Oromoo | 4,652 | 0.43M |
| BBC Afaan Oromoo | 23,949 | 0.43M |
| Jawar Mohammed | 4,866 | 0.40M |
| Addis Standard Afaan Oromoo | 4,485 | 0.39M |
| 12 smaller channels | 12,556 | 0.79M |

`scripts/21_analyze_telegram.py` profiles each channel before import: Oromo vs other languages,
signature lines, topics, overlap with the corpus and with evaluation text, and how much of it is new.

**What the importer does:**
- Strips hashtags, @handles, emojis and links.
- Removes "signature" lines that appear in 3+ posts of a channel. For OBN that's 58,612 lines of
  social-media link blocks, sign-offs and reporter bylines.
- **Drops any post that shares a 13-word sequence with evaluation text** (the validation set and
  the MasakhaNEWS dev/test articles), **or that contains a MasakhaNEWS dev/test headline**
  (`oromo_llm/contamination.py`). 338 posts were dropped this way, 307 of them from BBC.
- Folds styled Unicode letters (𝐒𝐇𝐀𝐑𝐄 → SHARE) with NFKC, and keeps one export per channel when a
  channel was exported twice.
- The corpus builder then applies the usual line filters and deduplication. Telegram is processed
  last, so `val.txt` stays byte-identical across all versions.

**Result:**
- First import: training text grew from 6.91M to 8.07M words (11.07M → 12.97M tokens).
- **v2: training text is 12.90M words (21.45M tokens)**, almost twice the original corpus.
- 5% of posts (by hash) are held out as `val_telegram.txt` (323k words in v2). The first import's
  held-out posts are kept as `val_telegram_v1.txt`, so every run is compared on the same text.
- The Phase 3 model, which never saw these posts, scores perplexity **217** on them, against 171
  on its usual validation set. The channels cover 2019–2026, newer events than the rest of the
  corpus.

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

#### Retraining with the Telegram data

The same model and settings, retrained on the corpus with Telegram added (12.97M tokens, 4 passes,
6,328 steps, 46 minutes). `scripts/15_compare_pretraining.py` scores both models on the same
held-out text:

| Model | Training words | Original val: perplexity | Original val: bits/char | Telegram val: perplexity | Telegram val: bits/char |
|---|---:|---:|---:|---:|---:|
| `oromo-gpt-23m` | 6.91M | 171 | 1.536 | 217 | 1.516 |
| `oromo-gpt-23m-tg` | 8.07M | **163** | **1.522** | **96** | **1.286** |

- **The new text is learned well:** bits per character on held-out Telegram posts fall by 15%
  (perplexity 217 → 96). These channels cover 2019–2026 events and administrative and education
  news that the older corpus lacks. Samples now mention places like *Benishaangul Gumuz* and
  *Naqamtee*, and things like exam results.
- **The old domain improves a little:** 1% fewer bits per character on the original validation set.
  The new text doesn't crowd out the old.
- **Caveats:**
  - The retrained run also took 17% more optimizer steps, because each pass over the data is
    longer, so not all of the gain on the original set is due to the new data.
  - The Telegram validation posts come from the same channels as the training posts (split at
    random by post). Templated posts that recur, like daily statistics or exam announcements, make
    that set easier than truly new text.
  - Validation loss was still falling at the end of training, so more data and longer training
    remain the main levers.
- Phases 4 and 5 above use the original `oromo-gpt-23m`.

#### v2: all the data

Same model, tokenizer and settings, trained on the v2 corpus (21.45M tokens, 4 passes, 10,472
steps, 77 minutes at about 20k tokens/s). Before building it, every line containing a MasakhaNEWS
dev/test headline was removed from all training sources (see Phase 4).

| Model | Training words | Original val: ppl | bits/char | Telegram val v1: ppl | bits/char | Telegram val v2: ppl | bits/char |
|---|---:|---:|---:|---:|---:|---:|---:|
| `oromo-gpt-23m` | 6.91M | 171 | 1.536 | 217 | 1.516 | 217 | 1.550 |
| `oromo-gpt-23m-tg` | 8.07M | 163 | 1.522 | 96 | 1.286 | 117 | 1.372 |
| `oromo-gpt-23m-v2` | 12.90M | **150** | **1.496** | **64** | **1.173** | **38** | **1.047** |

- **The original validation set improves again** (perplexity 163 → 150). It is AfriBERTa and
  Wikipedia text that the new data does not resemble, so this is the cleanest sign that more data
  helps the model in general.
- **No validation line appears in the v2 training text**, checked for all three sets.
- The Telegram scores are flattered by the same caveat as above: channels repost the same stories
  in slightly different words, so the held-out posts are close to training posts.
- At 23M parameters, samples are fluent in phrasing but still repeat themselves and lose the
  thread after a sentence or two.

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
- **Headlines needed a separate check.** A headline is usually shorter than 13 words, so the n-gram
  test misses it. Matching whole headlines found that **81 of the 325 test headlines appear in the
  v1 pretraining text**, almost all through AfriBERTa. This doesn't affect topic classification
  (the labels aren't in the text), but it matters for the Phase 5 headline task. On the 244 clean
  headlines the conclusions hold (Phase 6 model ROUGE-L 0.255 vs 0.218 for the first-10-words
  baseline). From v2 on, the corpus builder removes every line that contains a dev/test headline.

### Phase 5: instruction fine-tuning

**Data:** there is no Oromo instruction dataset, and translating the book's English one would
bring in translation errors. So instruction tasks are made from real Oromo text (the approach the
FLAN datasets used). Every answer is human-written Oromo; only the instruction templates are new.

| Task | Instruction (one of two templates) | Answer | Train / val / test |
|---|---|---|---:|
| topic | *Oduun armaan gadii waa'ee maalii? Tokko filadhu: siyaasa, fayyaa, …* | the topic word | 1,128 / 162 / 325 |
| headline | *Oduu armaan gadiitiif mata duree gabaabaa barreessi.* | the real BBC headline | 1,128 / 162 / 325 |
| describe | *Waa'ee {X} ibsa gabaabaa kenni.* | the first sentence of X's Wikipedia article | 1,502 / 60 / 81 |

The test articles never appear in training. The test "describe" articles were also never seen in
pretraining.

**Method:** the chapter 7 recipe, with these changes:
- **Prompt format:** the Alpaca layout with Oromo section names (`### Qajeelfama:`,
  `### Galtee:`, `### Deebii:`).
- **Loss:** prompt tokens are masked out, so the model trains only on writing the answer (the
  book's exercise 7.2).
- **Truncation:** long articles are cut so instruction + article + answer fit 256 tokens.
- **Training:** the Phase 3 model, all layers trained. Of 2 learning rates × 5 epochs, the best
  validation loss was at 5e-5 after 2 epochs (6.45 → 4.05). With 2e-4 the model memorized the
  training set: training loss fell to 2.0 while validation loss rose.

**Results on the test set (greedy decoding)**

| Task | Metric | Before instruction tuning | After | Reference |
|---|---|---:|---:|---|
| topic | answers that are a valid topic word | 0% | **100%** | – |
| topic | accuracy / macro-F1 | 0% / 0.00 | 50.5% / 0.29 | Phase 4 classifier: 87.2% / 0.78; majority: 30.8% |
| headline | ROUGE-L with the real headline | 0.005 | 0.025 | first 10 words of the article: **0.212** |
| describe | ROUGE-L with the Wikipedia sentence | 0.024 | 0.079 | – |

**Findings**
- **Instruction tuning teaches the format.** Before tuning, the model just keeps writing news
  text after the prompt. After tuning it answers in the right shape every time: a topic word for
  topic questions, a short headline, one descriptive sentence.
- **At 23M parameters it doesn't learn to use the input.** The answers barely depend on what the
  model is given:
  - Topic answers are almost always *fayyaa* or *ispoortii*, and never *teeknooloojii*.
  - 240 of 325 generated headlines mention Covid, while only 53 of the real ones do. The model
    falls back on the most common kind of training headline instead of reading the article.
  - "Describe" answers for unseen entities are invented, often describing a district (*"Aanaan …"*)
    even when asked about a country.
  - A headline that just copies the article's first 10 words scores 8× better.
- **Scale is the missing piece.** The same model reaches 87% on the topic task as a classifier
  (Phase 4), so the knowledge is there. Turning it into instruction following needs more capacity,
  which is why the book uses the 355M GPT-2 for this chapter. This is what Phase 6 tests:
  instruction-tuning a larger, multilingual open model on the same data.
- The instruction templates were written for this project and should be reviewed by a native
  Oromo speaker.

### Phase 6: open multilingual models

**Out-of-the-box Oromo.** Bits per character (lower is better) on the same text, each model given
about 1,300 characters of context (`scripts/16_oromo_skill.py`). "Telegram 2026" is 28 held-out posts
published after AfriqueLLM's January 2026 release, so no model here can have seen them.

| Model | Params | Validation | Telegram | Telegram 2026 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B-Base (general) | 0.6B | 3.135 | 2.628 | 2.633 |
| Qwen3.5-0.8B-Base (general) | 752M | 2.881 | 2.430 | 2.418 |
| Qwen3-0.6B + **our LoRA continued pretraining** (2.5M tokens, 55 min) | 0.6B | 1.798 | 1.478 | 1.458 |
| **Our GPT** (Oromo only) | 23M | 1.512 | 1.360 | 1.381 |
| AfriqueQwen3.5-0.8B (African-language continued pretraining) | 752M | **1.452** | **1.078** | **1.024** |

- **Oromo data beats size.** Our 23M model beats a general 752M model by almost 2× (that model
  continues an Oromo prompt in Afrikaans). The same 752M model after African-language pretraining
  is best everywhere.
- **Our own LoRA continued pretraining** of Qwen3-0.6B cut its bits per character by 43%.

**Instruction tuning at 0.8B.** LoRA instruction tuning of AfriqueQwen3.5-0.8B on the Phase 5 data
(one epoch, about an hour; `scripts/18_instruct_open_model.py`, full-test scoring by
`scripts/20_eval_open_model_full.py`):

| Task | Metric | 23M (Phase 5) | AfriqueQwen 0.8B + LoRA | Reference |
|---|---|---:|---:|---|
| Topic | Accuracy / macro-F1 | 50.5% / 0.29 | **91.1% / 0.87** | Phase 4 classifier: 87.2% / 0.78 |
| Headline | ROUGE-L | 0.025 | **0.254** | First 10 words: 0.212 |
| Describe | ROUGE-L | 0.079 | **0.186** | – |

The failure in Phase 5 was one of scale: with the same 3,758 examples, the larger Oromo-adapted model
reads its input. It gets every topic at F1 ≥ 0.73, and its headlines no longer collapse to Covid
stories. Two caveats: it's a single run, and AfriqueLLM's pretraining data may include BBC Oromo
articles like the test set.

**On a Mac:** Qwen3.5's linear-attention layers have fast kernels only for NVIDIA GPUs, so LoRA
training runs at 90–290 tokens/s. That's why our own continued pretraining used Qwen3.

## Paper

[`paper/`](paper) holds the arXiv version: 8 pages in ACL style. Rebuild the PDF and the upload
archive with `sh paper/build.sh` (needs `brew install tectonic`), and see
[`paper/ARXIV_SUBMISSION.md`](paper/ARXIV_SUBMISSION.md) for the submission steps.

## Project layout

```
oromo_llm/
  tokenizer.py   Oromo tokenizer; drop-in for the book's tiktoken GPT-2 tokenizer
  data.py        the book's GPTDatasetV1 + a faster loader over pre-encoded tokens
  model.py       the book's GPT: multi-head attention, LayerNorm, GELU, transformer blocks
  generate.py    greedy, temperature and top-k generation
  train.py       the book's loss functions and training loop + warmup, cosine decay, clipping
  classify.py    the book's classification fine-tuning (chapter 6) + macro-F1
  instruct.py    the book's instruction fine-tuning (chapter 7): Oromo prompt format, masking
  text.py        Oromo text normalization (hudhaa, quotes, site labels)
  contamination.py keeps evaluation headlines out of training text
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
  11_build_instructions.py  Oromo instruction data from news + Wikipedia
  12_instruction_finetune.py instruction tuning (learning-rate and epoch selection)
  13_evaluate_instructions.py before/after evaluation + all answers for human review
  14_import_telegram.py     optional: Telegram channel exports -> corpus (run before 02)
  15_compare_pretraining.py compare pretraining runs on the same held-out text
  16_oromo_skill.py         bits per character of any model on the same Oromo text
  17_continued_pretraining.py LoRA continued pretraining of an open model on Oromo
  18_instruct_open_model.py LoRA instruction tuning of an open model on the Phase 5 data
  19_paper_figures.py       the paper's figures, drawn from the saved results
  20_eval_open_model_full.py full-test evaluation of the tuned open model
  21_analyze_telegram.py    per-channel profile of the Telegram exports
  prompt.py                 type prompts to the trained models
paper/                      the arXiv paper (LaTeX, figures, build script)
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
python scripts/11_build_instructions.py
python scripts/12_instruction_finetune.py
python scripts/13_evaluate_instructions.py
```

To include Telegram channels, put Telegram Desktop exports under `data/raw/telegram/<channel>/`,
then run `scripts/14_import_telegram.py` (after `09_prepare_news.py`, which it uses for the overlap
check) and rerun `02`, `03` and `06`. Without exports, the pipeline skips Telegram.
The Phase 3–5 results above were produced before the Telegram data was added.

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
