# Contamination audit of MasakhaNEWS (paper 2)

*Already Seen: Most of an African News Benchmark Is in Common Pretraining Corpora, and Larger Models
Remember It.* Draft: [`paper/main.pdf`](paper/main.pdf).

## Findings

- **77% of MasakhaNEWS test articles** (3,752 of 4,872, 14 African languages) are in the training split
  of at least one of AfriBERTa, WURA and FineWeb-2: their headline as a phrase, half of their 13-grams,
  or the same page URL. WURA alone contains 81% of the articles it covers (91-93% for seven languages).
- **13-gram checks miss short text**: 61% of the test headlines found in AfriBERTa sit inside flattened
  web pages (menus, "related stories" lists) with little article text around them.
- **Little clean data remains**: 1,120 test articles (23%); 15-37 for nine languages.
- **Memorization grows with model size**: using WURA's random train/eval split as a natural experiment
  (1,436 seen vs 265 held-out articles), AfriTeVa V2 shows none (AUC 0.50), AfriqueQwen 0.53 at 0.8B and
  0.64 at 4B (p = 0.0001), strongest for low-resource languages. Untrained base models show none.

## Release

- [`release/contamination_<lang>_<split>.tsv`](release): for every MasakhaNEWS dev/test article (row index
  in the official TSV), whether it is in AfriBERTa, WURA or FineWeb-2 training data, and whether it is clean.
- [`release/clean_counts.json`](release/clean_counts.json): clean / total per language and split.
- `results/`: per-article flags and scores (indices, URLs and numbers only; no benchmark or corpus text).

## Reproduce

```bash
python audit/scripts/01_download.py                 # MasakhaNEWS + AfriBERTa corpus
python audit/scripts/01_download.py --wura          # ~10 GB
python audit/scripts/01_download.py --fineweb2      # ~5 GB
python audit/scripts/02_overlap.py --corpus afriberta
python audit/scripts/02_overlap.py --corpus wura --jobs 6
python audit/scripts/02_overlap.py --corpus fineweb2 --jobs 7
python audit/scripts/03_summary.py
python audit/scripts/04_groups.py
for m in afriteva-base afriteva-large afrique qwen-base afrique-4b qwen-4b-base; do
  python audit/scripts/05_memorization.py --model $m
done
python audit/scripts/06_effect.py
python audit/scripts/07_release.py
python audit/scripts/08_mechanism.py
sh audit/paper/build.sh                             # figures (09) + PDF; needs Tectonic
```

Data goes to the git-ignored `audit/data/`. Everything runs on one laptop (Apple M1 Max, 64 GB): the
overlap audit takes under 30 minutes, each model 4-29 minutes.

Note for anyone reusing AfriTeVa V2: its sentinel tokens are the last SentencePiece ids (149999, 149998,
...), not the 100 extra ids the Hugging Face T5 tokenizer appends.
