"""Download the benchmark and the pretraining corpora for the contamination audit.

Benchmark: MasakhaNEWS (masakhane/masakhanews, AFL-3.0), 16 languages, train/dev/test TSVs.
Corpus:    AfriBERTa corpus (castorini/afriberta-corpus, Apache-2.0), 10 of those languages.

Corpus:    WURA (--wura) and FineWeb-2 (--fineweb2), the web corpora behind current African LLMs.

Everything goes to audit/data/raw/ (git-ignored).
Usage: python audit/scripts/01_download.py [--wura | --fineweb2]
"""

import zipfile
from pathlib import Path

import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTPS downloads; the Xet CDN dropped large files
from huggingface_hub import hf_hub_download as _hf_hub_download  # noqa: E402


def hf_hub_download(*args, tries=6, **kwargs):
    """hf_hub_download with retries: large downloads over a home connection fail now and then."""
    for attempt in range(1, tries + 1):
        try:
            return _hf_hub_download(*args, **kwargs)
        except Exception as e:  # network errors come in many types
            if attempt == tries:
                raise
            print(f"  retry {attempt} after {type(e).__name__}", flush=True)
            time.sleep(10 * attempt)

AUDIT = Path(__file__).resolve().parent.parent
RAW = AUDIT / "data" / "raw"

MASAKHANEWS = ["amh", "eng", "fra", "hau", "ibo", "lin", "lug", "orm", "pcm", "run", "sna", "som",
               "swa", "tir", "xho", "yor"]
# MasakhaNEWS code -> AfriBERTa corpus folder ("gahuza" mixes Kinyarwanda and Kirundi)
AFRIBERTA = {"orm": "afaanoromoo", "amh": "amharic", "run": "gahuza", "hau": "hausa", "ibo": "igbo",
             "pcm": "pidgin", "som": "somali", "swa": "swahili", "tir": "tigrinya", "yor": "yoruba"}


# WURA (castorini/wura, documents-v1.0; the AfriTeVa V2 corpus, also used by AfriqueLLM).
# It has Kinyarwanda but not Kirundi, and no Nigerian Pidgin.
WURA = ["orm", "amh", "hau", "ibo", "sna", "som", "swa", "tir", "xho", "yor"]
# FineWeb-2 (HuggingFaceFW/fineweb-2): MasakhaNEWS code -> FineWeb-2 language_script folder
FINEWEB2 = {"orm": "gaz_Latn", "amh": "amh_Ethi", "run": "run_Latn", "hau": "hau_Latn", "ibo": "ibo_Latn",
            "pcm": "pcm_Latn", "som": "som_Latn", "swa": "swh_Latn", "tir": "tir_Ethi", "yor": "yor_Latn",
            "lin": "lin_Latn", "lug": "lug_Latn", "sna": "sna_Latn", "xho": "xho_Latn"}


def download_wura():
    for lang in WURA:
        for split in ("train", "eval"):
            hf_hub_download("castorini/wura", f"documents-v1.0/{split}/{lang}.jsonl", repo_type="dataset",
                            local_dir=RAW / "wura")
        print(f"WURA {lang}", flush=True)


def download_fineweb2():
    from huggingface_hub import HfApi
    api = HfApi()
    for lang, folder in FINEWEB2.items():
        for f in api.list_repo_tree("HuggingFaceFW/fineweb-2", path_in_repo=f"data/{folder}", repo_type="dataset",
                                    recursive=True):
            if f.path.endswith(".parquet"):
                hf_hub_download("HuggingFaceFW/fineweb-2", f.path, repo_type="dataset", local_dir=RAW / "fineweb2")
        print(f"FineWeb-2 {folder}", flush=True)


def main():
    import sys
    if "--wura" in sys.argv:
        return download_wura()
    if "--fineweb2" in sys.argv:
        return download_fineweb2()
    for lang in MASAKHANEWS:
        for split in ("train", "dev", "test"):
            hf_hub_download("masakhane/masakhanews", f"data/{lang}/{split}.tsv", repo_type="dataset",
                            local_dir=RAW / "masakhanews")
    print("MasakhaNEWS: 16 languages", flush=True)

    for lang, folder in AFRIBERTA.items():
        for split in ("train", "eval"):
            out = RAW / "afriberta" / folder
            if (out / f"{split}.txt").exists():
                continue
            path = hf_hub_download("castorini/afriberta-corpus", f"{folder}/{split}.zip", repo_type="dataset",
                                   local_dir=RAW / "afriberta_zip")
            with zipfile.ZipFile(path) as z:
                names = [n for n in z.namelist() if n.endswith(".txt")]
                out.mkdir(parents=True, exist_ok=True)
                with open(out / f"{split}.txt", "wb") as f:
                    for n in names:
                        f.write(z.read(n))
        size = sum(p.stat().st_size for p in (RAW / "afriberta" / folder).glob("*.txt")) / 1e6
        print(f"AfriBERTa {folder}: {size:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
