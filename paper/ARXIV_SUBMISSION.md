# arXiv submission checklist

Build the PDF and the upload archive with `sh paper/build.sh` (needs Tectonic: `brew install tectonic`).

## Status (November 10, 2026)
- [x] arXiv account created and verified (username `HabtamuFeyera97`, default category cs.CL).
- [x] Submission started; license CC BY 4.0, category cs.CL.
- [ ] **Endorsement for cs.CL.** Forward arXiv's endorsement email (it holds your code) to a qualified endorser
  (3+ arXiv cs papers between 3 months and 5 years ago; check with "Which of the authors of this article
  can endorse?" on any of their arXiv abstract pages).
- [ ] After endorsement: upload `paper/arxiv-source.tar.gz` to the saved submission.

## Before submitting
- [ ] Read the whole PDF once; check every number against the README and `checkpoints/*.json`.
- [ ] Check the Oromo examples and the six instruction templates (Section 7) read naturally.
- [ ] Decide on an affiliation and an email in the author block (`main.tex`, line ~16); "Independent researcher" is the placeholder.
- [ ] Confirm the Acknowledgments wording about AI assistance matches how you see your role.

## Submitting (from your own arXiv account)
1. **Account:** register at arxiv.org/user/register if you don't have one.
2. **Endorsement:** first-time submitters to `cs.CL` may need an endorsement. arXiv shows an endorsement code during submission; send it to someone who has published in cs.CL (e.g. an advisor or colleague) and ask them to endorse you.
3. **Start a new submission** and choose:
   - **License:** CC BY 4.0 (lets others reuse the paper with credit).
   - **Primary category:** `cs.CL` (Computation and Language). Optional cross-list: `cs.LG`.
4. **Upload** `paper/arxiv-source.tar.gz`. arXiv compiles it with pdfLaTeX; `main.bbl` is included because arXiv does not run BibTeX.
5. **Metadata:**
   - **Title:** Building an Afaan Oromoo Language Model from Scratch: Data, Tokenization, Pretraining and Adaptation on a Laptop
   - **Authors:** Habtamu Feyera
   - **Comments:** 8 pages, 3 figures. Code: https://github.com/HabtamuFeyera/afaan-oromoo-llm
   - **Abstract:** paste the plain-text version below (arXiv's form does not take LaTeX commands).
6. **Check the preview PDF** arXiv builds before you confirm. The paper is announced after the next daily cutoff.

## Plain-text abstract

We build a complete language-model pipeline for Afaan Oromoo, a widely spoken but under-resourced Cushitic language, on a single laptop (Apple M1 Max). Following the from-scratch recipe of Raschka (2025), we adapt each stage to Oromo: an 8.07M-word cleaned corpus that adds public Telegram news channels to AfriBERTa and Wikipedia; a byte-level BPE tokenizer that keeps the hudhaa (') inside words and halves sequence length relative to GPT-2 (4.98 vs. 2.34 characters per token); and a 23M-parameter GPT pretrained in 46 minutes to 1.52 bits per character. Fine-tuned for news-topic classification on MasakhaNEWS, the pretrained model reaches 87.2% accuracy and 0.779 macro-F1, against 0.558 for the same architecture trained from random weights and 0.749 for TF-IDF with logistic regression. Instruction tuning on 3,758 examples built from real Oromo text teaches the 23M model the answer format (100% valid topic labels) but not to use its input; the same data with LoRA on an Oromo-adapted 0.8B model reaches 91.1% topic accuracy and beats a lead baseline on headlines. Comparing open models in bits per character, our 23M model outperforms a general 752M multilingual model (1.51 vs. 2.88), while the same 752M model after African-language continued pretraining is best (1.45; 1.02 on posts published after its release). LoRA continued pretraining of Qwen3-0.6B on 2.5M Oromo tokens cuts its bits per character by 43%. The results point to Oromo-specific data and tokenization, more than model size, as the main levers for small-scale Oromo language modeling. Code: https://github.com/HabtamuFeyera/afaan-oromoo-llm
