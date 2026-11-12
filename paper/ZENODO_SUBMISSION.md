# Publishing the preprint on Zenodo (AfricArXiv community)

arXiv needs an endorser for first-time authors. Zenodo (run by CERN) does not: sign in with
ORCID, upload the PDF, and the paper gets a DOI as soon as it is published. Submitting to the
**AfricArXiv** community on Zenodo also lists it in Africa's preprint portal.

Before uploading: make the GitHub repository public, because the paper links to it.

## Steps

1. Go to <https://zenodo.org/communities/africarxiv> and click **New upload**
   (sign in with **ORCID**: 0009-0008-8698-8940).
2. **Files:** upload `paper/main.pdf` (rebuild it first with `sh paper/build.sh`).
3. Fill in the form with the values below.
4. **Communities:** AfricArXiv should already be selected (you started from its page).
5. Click **Submit for review**. AfricArXiv's curators check it, then it is published with a DOI.

## Form values

| Field | Value |
|---|---|
| Do you already have a DOI? | No (Zenodo assigns one) |
| Resource type | Publication / Preprint |
| Title | Building an Afaan Oromoo Language Model from Scratch: Data, Tokenization, Pretraining and Adaptation on a Laptop |
| Publication date | the day you upload |
| Creator | Feyera, Habtamu — ORCID 0009-0008-8698-8940 — Role: none needed — Affiliation: Independent researcher |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Language | English |
| Keywords | Afaan Oromoo; Oromo; low-resource languages; African languages; language models; tokenization; pretraining; instruction tuning; LoRA; MasakhaNEWS |
| Related works | *Is supplemented by* — URL — https://github.com/HabtamuFeyera/afaan-oromoo-llm (Resource type: Software) |
| Version | v1 |

### Description (paste as the abstract)

We build a complete language-model pipeline for Afaan Oromoo, a widely spoken but
under-resourced Cushitic language, on a single laptop (Apple M1 Max). Following the
from-scratch recipe of Raschka (2025), we adapt each stage to Oromo: a 12.76M-word cleaned
corpus that adds 19 public Telegram news channels to AfriBERTa and Wikipedia, filtered so that
no evaluation headline or article text appears in training; a byte-level BPE tokenizer that
keeps the hudhaa (') inside words and halves sequence length relative to GPT-2 (4.98 vs. 2.34
characters per token); and a 23M-parameter GPT pretrained in 77 minutes to 1.50 bits per
character. Growing the corpus from 6.9M to 12.8M words improves held-out text from every source.

Fine-tuned for news-topic classification on MasakhaNEWS, the pretrained model reaches 87.8%
accuracy and 0.809 macro-F1, against 0.577 for the same architecture trained from random
weights and 0.749 for TF-IDF with logistic regression. Instruction tuning on 3,758 examples
built from real Oromo text teaches the 23M model the answer format but not reliably to use its
input; the same data with LoRA on an Oromo-adapted 0.8B model reaches 91.1% topic accuracy and
beats a lead baseline on headlines.

Comparing open models in bits per character, our 23M model outperforms a general 752M
multilingual model by almost 2x (1.49 vs. 2.88 on validation text), while the same 752M model
after African-language continued pretraining is best on every test set, most clearly on 658
posts published after its release (1.15 vs. 1.54 for ours). LoRA continued pretraining of
Qwen3-0.6B on 2.5M Oromo tokens cuts its bits per character by 40-49%. The results point to
Oromo-specific data and tokenization, more than model size, as the main levers for small-scale
Oromo language modeling, and to model size as the limit for instruction following.

Code: https://github.com/HabtamuFeyera/afaan-oromoo-llm

## After it is published

- Add the DOI to the README and to the paper's first page (a "Preprint, DOI: …" line), then
  upload the updated PDF as **a new version** of the same Zenodo record (same concept DOI).
- Later peer-reviewed venues (for example the AfricaNLP workshop) accept papers that are already
  on a preprint server.
- arXiv stays possible later if an endorser turns up (`ARXIV_SUBMISSION.md`).
