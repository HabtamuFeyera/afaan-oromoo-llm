# Zenodo upload: paper 2 (contamination audit)

Upload `audit/paper/main.pdf` (rebuild first with `sh audit/paper/build.sh`).

| Field | Value |
|---|---|
| Community (top of the form) | **AfricArXiv - the pan-African Open Access portal** |
| Do you already have a DOI? | **No, I need one** |
| Resource type | Publication / Preprint |
| Title | Already Seen: Most of an African News Benchmark Is in Common Pretraining Corpora, and Larger Models Remember It |
| Publication date | the day you upload |
| Creator | Feyera, Habtamu — ORCID 0009-0008-8698-8940 — Affiliation: Independent researcher (add by hand; do not pick another person from the search) |
| Licence | Creative Commons Attribution 4.0 International |
| Keywords | data contamination; benchmark contamination; African languages; MasakhaNEWS; pretraining data; memorization; low-resource languages; language model evaluation; WURA; FineWeb-2 |
| Languages | English |
| Version | v1 |
| Related works | *Is supplemented by* — URL — https://github.com/HabtamuFeyera/afaan-oromoo-llm/tree/main/audit (Software) |
| Software: Repository URL | https://github.com/HabtamuFeyera/afaan-oromoo-llm |
| Software: Programming language | Python |
| Publisher | leave "Zenodo" |
| Leave empty | Copyright, Contributors, Dates, Funding, Alternate identifiers, References, Publishing information, Domain specific fields |

## Description (paste as the abstract)

MasakhaNEWS is a standard benchmark for African languages, built from news websites such as BBC. We audit its test sets in 14 African languages against the training splits of three pretraining corpora that African language models are built from: AfriBERTa, WURA and FineWeb-2. Matching headlines as phrases, article text by 13-grams and pages by URL, we find that 3,752 of 4,872 test articles (77%) appear in at least one corpus. WURA alone contains 81% of the articles it covers, and 91-93% for seven languages. Only 15-37 clean test articles remain for nine languages. The standard 13-gram test misses most leaked headlines: in AfriBERTa, 61% sit inside flattened web pages with menus and link lists.

To measure whether the overlap matters, we use WURA's random train/eval split as a natural experiment: articles from the same sources that WURA happened to hold out serve as controls. AfriTeVa V2, pretrained on WURA and evaluated on MasakhaNEWS by its authors, shows no measurable memorization (AUC 0.50). AfriqueQwen models trained on WURA do: AUC 0.53 at 0.8B and 0.64 at 4B parameters (p = 0.0001), strongest for the languages with the least data, while the untrained base models show none. Benchmark contamination of African-language evaluation is thus nearly universal, and its effect grows as models scale. We release per-article contamination flags, clean subsets and the audit code.

Code and data: https://github.com/HabtamuFeyera/afaan-oromoo-llm/tree/main/audit
