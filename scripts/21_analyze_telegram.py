"""Analyse Telegram channel exports before importing them into the corpus.

For every export folder under data/raw/telegram it reports, per channel:
- size: messages, posts with text, date range, posts per month
- duplicate exports of the same channel (overlapping message IDs) and which copy to keep
- language: share of posts that are Oromo, English, Ethiopic-script (Amharic/Tigrinya),
  Arabic-script, or too short to tell, from function-word and script counts
- post length, exact repeated posts, forwarded posts, signature boilerplate
- topic profile from keyword families (politics, health, sports, education, ...)
- overlap with the existing corpus (AfriBERTa + Wikipedia) and with evaluation text
  (validation set, MasakhaNEWS dev/test), using 13-word sequences
- estimated new clean Oromo words after the same cleaning the corpus uses

Writes data/interim/telegram_analysis.json and prints a summary table.
Usage: python scripts/21_analyze_telegram.py
"""

import hashlib
import importlib.util
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.text import WORD_RE, normalize  # noqa: E402

spec = importlib.util.spec_from_file_location("imp", ROOT / "scripts" / "14_import_telegram.py")
imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp)
spec2 = importlib.util.spec_from_file_location("bc", ROOT / "scripts" / "02_build_corpus.py")
bc = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(bc)

URL_RE = re.compile(r"(https?://\S+|www\.\S+|t\.me/\S+|@\w+)")
RAW = ROOT / "data" / "raw" / "telegram"
OUT = ROOT / "data" / "interim" / "telegram_analysis.json"
EOT = "<|endoftext|>"

ENGLISH_FUNC = set("""the and of to in is that for on with as was are this by be from at it an have has
not but they their will which you we our""".split())
TOPICS = {
    "politics": "mootummaa paartii filannoo siyaasaa siyaasa ministira pirezidaantii paarlaamaa mana marii bulchiinsa".split(),
    "security/conflict": "lola waraana poolisii ajjeefaman hidhaman nageenya humnoota loltoota".split(),
    "health": "fayyaa dhukkuba yaalaa hospitaala talaallii qorichaa covid koronaa".split(),
    "sports": "kubbaa taphataa taphattoota garee galchii tapha ispoortii liigii shaampiyoonaa atileetii".split(),
    "education": "barnoota barattoota barsiisota mana barumsaa yuunivarsiitii qormaata kutaa".split(),
    "economy": "diinagdee qarshii gatii daldala misooma investimentii baajata".split(),
    "religion": "rabbii waaqayyo masjiida amantaa islaamaa kiristaana mana sagadaa".split(),
}


def dedup_folders():
    """Group export folders by channel, keep the copy with the most messages."""
    folders = sorted({p.parent for p in RAW.rglob("messages*.html")})
    by_key = defaultdict(list)
    for f in folders:
        key = re.sub(r"_\d+$", "", f.name.removeprefix("ChatExport_")).strip().lower()
        by_key[key].append(f)
    chosen, dropped = {}, []
    for key, fs in by_key.items():
        parsed = {f: list(imp.parse_export(f)) for f in fs}
        best = max(fs, key=lambda f: len(parsed[f]))
        ids_best = {m for m, _, _ in parsed[best]}
        for f in fs:
            if f != best:
                ids = {m for m, _, _ in parsed[f]}
                dropped.append({"folder": str(f.relative_to(RAW)), "kept": str(best.relative_to(RAW)),
                                "messages": len(ids), "share_inside_kept": round(len(ids & ids_best) / max(1, len(ids)), 3)})
        chosen[best.name.removeprefix("ChatExport_")] = (best, parsed[best])
    return chosen, dropped


VOCAB = set()  # known Oromo words (filled in main from the corpus)


def classify(text):
    """Language of a post. Oromo = mostly words already known from the Oromo corpus
    (headlines have few function words, so function words alone under-count Oromo)."""
    text = unicodedata.normalize("NFKC", text)
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 15:
        return "too short"
    eth = sum("ሀ" <= c <= "᎟" for c in letters) / len(letters)
    ara = sum("؀" <= c <= "ۿ" for c in letters) / len(letters)
    if eth > 0.3:
        return "ethiopic script"
    if ara > 0.3:
        return "arabic script"
    words = [w.lower() for w in WORD_RE.findall(URL_RE.sub(" ", text))]
    if len(words) < 4:
        return "too short"
    en = sum(w in ENGLISH_FUNC for w in words) / len(words)
    known = sum(w in VOCAB for w in words if len(w) > 2) / max(1, sum(len(w) > 2 for w in words))
    if en >= 0.12:
        return "english"
    if known >= 0.6:
        return "oromo"
    return "other/unclear"


def ngrams(words, n=13):
    return {hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)}


def gram_set(texts):
    s = set()
    for t in texts:
        for line in t.split("\n"):
            s |= ngrams([w.lower() for w in WORD_RE.findall(line)])
    return s


def main():
    chosen, dropped = dedup_folders()

    proc = ROOT / "data" / "processed"
    train_text = (proc / "train.txt").read_text(encoding="utf-8")
    stats = json.loads((proc / "stats.json").read_text())["sources"]
    parts = train_text.split(f"\n{EOT}\n")
    n_wiki = stats["wikipedia_train"]["kept_articles"]
    base_corpus = parts[:1 + n_wiki]  # AfriBERTa + Wikipedia, without the earlier Telegram import
    print("indexing corpus and evaluation 13-grams...", flush=True)
    corpus_grams = gram_set(base_corpus)
    eval_texts = [(proc / "val.txt").read_text(encoding="utf-8")]
    for split in ("dev", "test"):
        eval_texts += [json.loads(l)["text"] for l in (ROOT / "data" / "news" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    eval_grams = gram_set(eval_texts)
    test_grams = gram_set([json.loads(l)["text"] for l in (ROOT / "data" / "news" / "test.jsonl").read_text(encoding="utf-8").splitlines()])
    seen_keys = {bc.dedup_key(l) for p in base_corpus for l in p.split("\n") if l.strip()}
    counts = Counter(w.lower() for p in base_corpus for w in WORD_RE.findall(p))
    VOCAB.update(w for w, c in counts.items() if c >= 3)
    global_seen = set()  # lines already counted in an earlier channel (alphabetical order)

    report = {"dropped_duplicate_exports": dropped, "channels": {}}
    cross_posts = defaultdict(set)
    for name, (folder, posts) in sorted(chosen.items()):
        dates, langs, lengths, texts = [], Counter(), [], []
        apos = Counter()
        for mid, date, raw in posts:
            if date:
                dates.append(datetime.strptime(date.split(" UTC")[0], "%d %B %Y, %H:%M:%S"))
            lang = classify(raw)
            langs[lang] += 1
            apos["'"] += raw.count("'")
            apos["’"] += raw.count("’") + raw.count("‘")
            cleaned = imp.clean_post(raw)
            texts.append(cleaned)
            lengths.append(sum(len(WORD_RE.findall(l)) for l in cleaned))
            key = hashlib.md5(" ".join(w.lower() for l in cleaned for w in WORD_RE.findall(l)).encode()).hexdigest()
            cross_posts[key].add(name)

        # signature lines: appear in 3+ posts of this channel
        per_line = Counter(k for lines in texts for k in {imp.line_key(l) for l in lines})
        sig = {k for k, c in per_line.items() if c >= imp.SIGNATURE_MIN_POSTS and k}
        all_lines = sum(len(t) for t in texts)
        sig_lines = sum(1 for t in texts for l in t if imp.line_key(l) in sig)
        exact_dup = len(texts) - len({tuple(t) for t in texts})

        # yield after the corpus cleaning rules, deduplicated against the existing corpus
        new_words, kept_posts, corpus_ov, eval_hit, test_hit, oromo_posts = 0, 0, [], 0, 0, 0
        unique_words = 0
        fancy = sum(any(0x1D400 <= ord(ch) <= 0x1D7FF for ch in raw) for _, _, raw in posts)
        local_seen = set()
        topic = Counter()
        for (mid, date, raw), lines in zip(posts, texts):
            if classify(raw) != "oromo":
                continue
            oromo_posts += 1
            lines = [l for l in lines if imp.line_key(l) not in sig]
            words = [w.lower() for l in lines for w in WORD_RE.findall(l)]
            g = ngrams(words)
            if g:
                corpus_ov.append(len(g & corpus_grams) / len(g))
                if g & eval_grams:
                    eval_hit += 1
                if len(g & test_grams) >= max(1, len(g) // 2):
                    test_hit += 1
            wl = set(words)
            for t, kws in TOPICS.items():
                if sum(k in wl for k in kws) >= 2:
                    topic[t] += 1
            if g & eval_grams:
                continue
            post_words = 0
            for line in lines:
                line = normalize(unicodedata.normalize("NFKC", line))  # NFKC turns fancy letters into plain ones
                if not line or bc.reject_reason(line):
                    continue
                k = bc.dedup_key(line)
                if k in seen_keys or k in local_seen:
                    continue
                local_seen.add(k)
                n = len(WORD_RE.findall(line))
                post_words += n
                if k not in global_seen:
                    global_seen.add(k)
                    unique_words += n
            if post_words:
                kept_posts += 1
                new_words += post_words

        months = max(1, round((max(dates) - min(dates)).days / 30.4)) if dates else 1
        report["channels"][name] = {
            "folder": str(folder.relative_to(RAW)),
            "posts_with_text": len(posts),
            "first": min(dates).strftime("%Y-%m-%d") if dates else None,
            "last": max(dates).strftime("%Y-%m-%d") if dates else None,
            "posts_per_month": round(len(posts) / months, 1),
            "language_pct": {k: round(100 * v / max(1, len(posts)), 1) for k, v in langs.most_common()},
            "median_words": statistics.median(lengths) if lengths else 0,
            "short_posts_pct": round(100 * sum(l < 10 for l in lengths) / max(1, len(lengths)), 1),
            "exact_duplicate_posts_pct": round(100 * exact_dup / max(1, len(posts)), 1),
            "signature_lines_pct": round(100 * sig_lines / max(1, all_lines), 1),
            "top_signature_lines": [k[:80] for k, _ in Counter({k: per_line[k] for k in sig}).most_common(4)],
            "apostrophe_curly_pct": round(100 * apos["’"] / max(1, sum(apos.values())), 1),
            "topic_pct_of_oromo_posts": {k: round(100 * v / max(1, oromo_posts), 1) for k, v in topic.most_common()},
            "corpus_overlap_mean_pct": round(100 * statistics.mean(corpus_ov), 1) if corpus_ov else 0,
            "posts_touching_eval_text": eval_hit,
            "posts_mostly_in_masakhanews_test": test_hit,
            "new_clean_words": new_words,
            "new_clean_words_not_in_earlier_channels": unique_words,
            "new_clean_posts": kept_posts,
            "fancy_unicode_posts_pct": round(100 * fancy / max(1, len(posts)), 1),
        }
        c = report["channels"][name]
        print(f"{name[:30]:<30} posts {len(posts):>6} | oromo {c["language_pct"].get("oromo", 0):>5}% | fancy {c["fancy_unicode_posts_pct"]:>4}% | "
              f"median {c['median_words']:>4} w | sig {c['signature_lines_pct']:>4}% | corpus-overlap {c['corpus_overlap_mean_pct']:>4}% | "
              f"eval-touch {eval_hit:>4} | NEW {new_words:>9,} (unique {unique_words:>9,}) words", flush=True)

    shared = Counter()
    for key, chans in cross_posts.items():
        if len(chans) > 1:
            for c in chans:
                shared[c] += 1
    report["posts_also_in_another_channel"] = dict(shared)
    report["total_new_clean_words"] = sum(c["new_clean_words_not_in_earlier_channels"] for c in report["channels"].values())
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ntotal new clean words (deduplicated across channels): {report['total_new_clean_words']:,}"
          f"  ->  {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
