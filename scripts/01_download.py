"""Phase 1, step 1: download raw Afaan Oromoo text.

Sources
- AfriBERTa corpus, Afaan Oromoo subset (castorini/afriberta-corpus, Apache-2.0)
- Oromo Wikipedia, 2023-11-01 snapshot (wikimedia/wikipedia, CC BY-SA 4.0)

Files land in data/raw/. Re-running skips files that already exist.
"""

import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

SOURCES = [
    ("castorini/afriberta-corpus", "afaanoromoo/train.zip", "afriberta"),
    ("castorini/afriberta-corpus", "afaanoromoo/eval.zip", "afriberta"),
    ("wikimedia/wikipedia", "20231101.om/train-00000-of-00001.parquet", "wikipedia"),
]


def main():
    for repo, filename, subdir in SOURCES:
        out_dir = RAW / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=out_dir,
        )
        path = Path(path)
        print(f"{repo}/{filename} -> {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")

        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    print(f"  contains {member}")
                zf.extractall(path.parent)


if __name__ == "__main__":
    main()
