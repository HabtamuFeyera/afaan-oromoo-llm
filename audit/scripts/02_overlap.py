"""Measure how much MasakhaNEWS evaluation text appears in a pretraining corpus.

Two checks per evaluation article:
- headline: the whole headline (>= MIN_HEADLINE_WORDS words) occurs as a contiguous word
  sequence inside one corpus line. Headlines are usually shorter than 13 words, so the
  standard 13-gram test cannot see them.
- article: the share of the article's 13-grams that occur in the corpus (the GPT-3 test);
  an article counts as "seen" when at least half of them do.
Text is compared after NFKC, lower-casing and unifying apostrophes; punctuation is ignored.

The evaluation side is small, so it is indexed in memory and the corpus is streamed once.

Outputs: audit/results/overlap_<corpus>.json (per language) and
         audit/results/flags_<corpus>_<lang>_<split>.jsonl (per article, for clean splits)
Usage: python audit/scripts/02_overlap.py [--corpus afriberta|wura|fineweb2] [--jobs 4] [--langs orm amh ...] [--splits test dev]
"""

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
RAW = AUDIT / "data" / "raw"
RESULTS = AUDIT / "results"
N = 13
MIN_HEADLINE_WORDS = 4
PROBE = 4
APOSTROPHES = str.maketrans({c: "'" for c in "’‘ʼʻ`´′"})
# Letter runs (with inner apostrophes) and digit runs are separate tokens: scraped pages often glue a
# headline to a date ("...daawwatamuuf8 Hagayya 2019"), which would otherwise hide the match.
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+")
csv.field_size_limit(sys.maxsize)

CORPORA = {
    "afriberta": {"orm": "afaanoromoo", "amh": "amharic", "run": "gahuza", "hau": "hausa", "ibo": "igbo",
                  "pcm": "pidgin", "som": "somali", "swa": "swahili", "tir": "tigrinya", "yor": "yoruba"},
    "wura": {l: l for l in ["orm", "amh", "hau", "ibo", "sna", "som", "swa", "tir", "xho", "yor"]},
    "fineweb2": {"orm": "gaz_Latn", "amh": "amh_Ethi", "run": "run_Latn", "hau": "hau_Latn", "ibo": "ibo_Latn",
                 "pcm": "pcm_Latn", "som": "som_Latn", "swa": "swh_Latn", "tir": "tir_Ethi", "yor": "yor_Latn",
                 "lin": "lin_Latn", "lug": "lug_Latn", "sna": "sna_Latn", "xho": "xho_Latn"},
}


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).translate(APOSTROPHES).lower()
    return WORD_RE.findall(text)


def ngrams(ws):
    return {hash(tuple(ws[i:i + N])) for i in range(len(ws) - N + 1)}


def read_split(lang, split):
    path = RAW / "masakhanews" / "data" / lang / f"{split}.tsv"
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def corpus_files(corpus, lang):
    """(split name, path) pairs; the split name is how results are broken down (train vs eval/test)."""
    code = CORPORA[corpus][lang]
    if corpus == "afriberta":
        return [(p.stem, p) for p in sorted((RAW / corpus / code).glob("*.txt"))]
    if corpus == "wura":
        return [(s, p) for s in ("train", "eval")
                if (p := RAW / "wura" / "documents-v1.0" / s / f"{code}.jsonl").exists()]
    # FineWeb-2 ships Parquet; files an older pyarrow cannot read were converted to JSON Lines (text, url)
    folder = RAW / "fineweb2" / "data" / code
    return [(p.parent.name, p) for p in sorted([*folder.glob("*/*.parquet"), *folder.glob("*/*.jsonl")])]


def url_keys(url):
    """Keys that identify a web page: the URL without scheme, www, query or trailing slash, and for
    BBC pages also the numeric article id (old and new BBC URL styles share it)."""
    url = (url or "").strip().lower()
    if not url:
        return set()
    url = re.sub(r"^[a-z]+://", "", url).split("?")[0].split("#")[0].rstrip("/")
    url = url.removeprefix("www.")
    keys = {url}
    m = re.search(r"(\d{8,})$", url)
    if "bbc." in url and m:
        keys.add("bbc-id:" + m.group(1))
    return keys


def corpus_lines(path, urls=None):
    """Every line of text in one corpus file, whatever its format. Page URLs, where the corpus
    keeps them (WURA, FineWeb-2), are added to `urls`."""
    if path.suffix == ".txt":
        with open(path, encoding="utf-8", errors="replace") as f:
            yield from f
    elif path.suffix == ".jsonl":  # WURA documents: headline, content, category, url
        with open(path, encoding="utf-8", errors="replace") as f:
            for row in f:
                doc = json.loads(row)
                if urls is not None:
                    urls.update(url_keys(doc.get("url")))
                yield doc.get("headline") or ""
                yield from (doc.get("content") or doc.get("text") or "").split("\n")
    else:  # FineWeb-2 parquet: text, url, ...
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = [c for c in ("text", "url") if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=cols, batch_size=2000):
            texts = batch.column(cols.index("text")).to_pylist()
            links = batch.column(cols.index("url")).to_pylist() if "url" in cols else [None] * len(texts)
            for doc, link in zip(texts, links):
                if urls is not None:
                    urls.update(url_keys(link))
                yield from (doc or "").split("\n")


def audit(corpus, lang, splits):
    items = []  # (split, index, headline words, article 13-grams)
    for split in splits:
        for i, row in enumerate(read_split(lang, split)):
            items.append({"split": split, "i": i, "head": words(row["headline"]),
                          "grams": ngrams(words(row["text"])), "url": row.get("url", "")})

    probe = {}  # first PROBE words of a headline -> indices of items
    for k, it in enumerate(items):
        if len(it["head"]) >= MIN_HEADLINE_WORDS:
            probe.setdefault(tuple(it["head"][:PROBE]), []).append(k)
    gram_owner = {}
    for k, it in enumerate(items):
        for g in it["grams"]:
            gram_owner.setdefault(g, []).append(k)

    head_found = [set() for _ in items]  # corpus files (train/eval) whose text contains the headline
    grams_found = [{} for _ in items]  # corpus file -> 13-grams of the article found there
    urls_seen = {}  # corpus file -> URL keys of the pages it contains
    lines = 0
    for name, path in corpus_files(corpus, lang):
        for line in corpus_lines(path, urls_seen.setdefault(name, set())):
            lines += 1
            ws = words(line)
            if len(ws) < PROBE:
                continue
            joined = None
            for j in range(len(ws) - PROBE + 1):
                for k in probe.get(tuple(ws[j:j + PROBE]), ()):
                    if name not in head_found[k]:
                        joined = joined or " " + " ".join(ws) + " "
                        if f" {' '.join(items[k]['head'])} " in joined:
                            head_found[k].add(name)
            if len(ws) >= N:
                for g in ngrams(ws):
                    for k in gram_owner.get(g, ()):
                        grams_found[k].setdefault(name, set()).add(g)

    url_found = [sorted(n for n, keys in urls_seen.items() if url_keys(it["url"]) & keys) for it in items]
    has_urls = any(urls_seen.values())
    out = {}
    for split in splits:
        idx = [k for k, it in enumerate(items) if it["split"] == split]
        eligible = [k for k in idx if len(items[k]["head"]) >= MIN_HEADLINE_WORDS]
        def share_of(k, stems=None):
            found = set().union(*(v for s, v in grams_found[k].items() if stems is None or s in stems))
            return len(found) / len(items[k]["grams"]) if items[k]["grams"] else 0.0

        share = [share_of(k) for k in idx]
        stems = sorted({name for name, _ in corpus_files(corpus, lang)})
        share_by = {s: [share_of(k, {s}) for k in idx] for s in stems}
        heads = [k for k in eligible if head_found[k]]
        out[split] = {
            "articles": len(idx),
            "headlines_eligible": len(eligible),
            "headlines_found": len(heads),
            "headlines_found_pct": round(100 * len(heads) / max(len(eligible), 1), 1),
            "headlines_found_in": {s: sum(s in head_found[k] for k in heads) for s in sorted(set().union(*(head_found[k] for k in heads)))},
            "articles_seen_50pct": sum(s >= 0.5 for s in share),
            "articles_any_13gram": sum(s > 0 for s in share),
            "mean_13gram_share_pct": round(100 * sum(share) / max(len(share), 1), 2),
            "articles_seen_50pct_in": {s: sum(x >= 0.5 for x in v) for s, v in share_by.items()},
            "articles_url_found": sum(bool(url_found[k]) for k in idx) if has_urls else None,
            "articles_url_found_in": {s: sum(s in url_found[k] for k in idx) for s in stems} if has_urls else None,
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / f"flags_{corpus}_{lang}_{split}.jsonl", "w", encoding="utf-8") as f:
            for n, (k, s) in enumerate(zip(idx, share)):
                f.write(json.dumps({"i": items[k]["i"], "headline_found": sorted(head_found[k]),
                                    "share_13gram": round(s, 4),
                                    "share_13gram_by_file": {st: round(v[n], 4) for st, v in share_by.items()},
                                    "url_found": url_found[k],
                                    "url": items[k]["url"]}) + "\n")
    out["corpus_lines"] = lines
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="afriberta", choices=sorted(CORPORA))
    ap.add_argument("--langs", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["test", "dev"])
    ap.add_argument("--jobs", type=int, default=4, help="languages processed in parallel")
    args = ap.parse_args()
    langs = args.langs or [l for l in CORPORA[args.corpus] if corpus_files(args.corpus, l)]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    path = RESULTS / f"overlap_{args.corpus}.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(audit, args.corpus, lang, args.splits): lang for lang in langs}
        done = [(futures[f], f.result()) for f in as_completed(futures)]
    for lang, r in sorted(done):
        results[lang] = r
        t = r["test"]
        print(f"{lang}: test headlines found {t['headlines_found']}/{t['headlines_eligible']} "
              f"({t['headlines_found_pct']}%) | articles >=50% seen {t['articles_seen_50pct']}/{t['articles']} "
              f"| any 13-gram {t['articles_any_13gram']} | mean share {t['mean_13gram_share_pct']}% "
              f"| same URL {t['articles_url_found']} | {r['corpus_lines']:,} corpus lines | {time.time() - t0:.0f}s", flush=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
