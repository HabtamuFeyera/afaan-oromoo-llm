"""Import Telegram channel exports into the corpus pipeline.

Input: data/raw/telegram/**/messages*.html, from Telegram Desktop's
"Export chat history" (HTML or JSON both work; this reads HTML). These files are
copyrighted by the channels: they stay in the git-ignored data/ folder.

Per post
- the text with line breaks kept; hashtags, @handles, emojis and links removed
- cleaned with the same normalize() as the rest of the corpus
- signature lines (a line that appears in 3+ posts of a channel, such as
  "follow us on YouTube" blocks) removed
- dropped entirely if it shares any 13-word sequence with evaluation text:
  the pretraining validation set or MasakhaNEWS dev/test articles, so Phase 3-5
  results stay fair

5% of posts (by hash) go to a separate Telegram validation set.
Line-level filtering and deduplication against the rest of the corpus happen in
scripts/02_build_corpus.py, which picks these files up.

Outputs: data/interim/telegram_{train,val}.jsonl and telegram_report.json
Usage: python scripts/14_import_telegram.py
"""

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oromo_llm.contamination import HeadlineFilter  # noqa: E402
from oromo_llm.text import WORD_RE, normalize  # noqa: E402

RAW = ROOT / "data" / "raw" / "telegram"
OUT = ROOT / "data" / "interim"
VAL_FRACTION = 0.05
SIGNATURE_MIN_POSTS = 3
N = 13

HASHTAG_RE = re.compile(r"#\w+")
HANDLE_RE = re.compile(r"@\w+")


def strip_symbols(text):
    """Remove emojis and other pictographic symbols, keep letters/punctuation."""
    return "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Sk", "Cs", "Co")
                   and ch not in "️‍")


def parse_export(folder):
    """Yield (message_id, date, text) for every post with text in one export folder."""
    for page in sorted(folder.glob("messages*.html"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 1)):
        soup = BeautifulSoup(page.read_text(encoding="utf-8"), "lxml")
        for msg in soup.select("div.message.default"):
            text_div = msg.select_one("div.body > div.text")
            if text_div is None:
                continue
            for br in text_div.find_all("br"):
                br.replace_with("\n")
            date = msg.select_one("div.date")
            yield msg.get("id", ""), date["title"] if date else "", text_div.get_text()


def clean_post(text):
    lines = []
    for raw in text.split("\n"):
        raw = unicodedata.normalize("NFKC", raw)  # fancy letters such as 𝐒𝐇𝐀𝐑𝐄 become plain SHARE
        line = HANDLE_RE.sub(" ", HASHTAG_RE.sub(" ", strip_symbols(raw)))
        line = normalize(line)
        if line:
            lines.append(line)
    return lines


def line_key(line):
    return " ".join(w.lower() for w in WORD_RE.findall(line))


def ngrams(words):
    return {hash(tuple(words[i:i + N])) for i in range(len(words) - N + 1)}


def eval_ngrams():
    """13-grams of all text that is used for evaluation."""
    grams = set()
    texts = [(ROOT / "data" / "processed" / "val.txt").read_text(encoding="utf-8")]
    for split in ("dev", "test"):
        path = ROOT / "data" / "news" / f"{split}.jsonl"
        texts += [json.loads(l)["text"] for l in path.read_text(encoding="utf-8").splitlines()]
    for t in texts:
        for line in t.split("\n"):
            grams |= ngrams([w.lower() for w in WORD_RE.findall(line)])
    return grams


def channel_exports():
    """One export per channel: when a channel was exported more than once (e.g. "BBC" and
    "BBC_2", or an older copy in another folder), keep the export with the most posts."""
    groups = {}
    for folder in sorted({p.parent for p in RAW.rglob("messages*.html")}):
        channel = re.sub(r"_\d+$", "", folder.name.removeprefix("ChatExport_")).strip()
        posts = list(parse_export(folder))
        best = groups.get(channel.lower())
        if best is None or len(posts) > len(best[2]):
            if best is not None:
                print(f"  skipping smaller export {best[1].relative_to(RAW)} of {channel}", flush=True)
            groups[channel.lower()] = (channel, folder, posts)
        else:
            print(f"  skipping smaller export {folder.relative_to(RAW)} of {channel}", flush=True)
    return sorted(groups.values(), key=lambda g: g[0].lower())


def main():
    exports = channel_exports()
    if not exports:
        sys.exit(f"no Telegram exports found under {RAW}")
    held_out = eval_ngrams()
    headlines = HeadlineFilter()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    out = {"train": [], "val": []}

    for channel, folder, raw_posts in exports:
        posts = [(mid, date, clean_post(text)) for mid, date, text in raw_posts]
        posts = [p for p in posts if p[2]]
        per_line_posts = Counter(k for _, _, lines in posts for k in {line_key(l) for l in lines})
        stats = Counter(posts_with_text=len(posts))
        dates = [d for _, d, _ in posts if d]

        for mid, date, lines in posts:
            kept = [l for l in lines if per_line_posts[line_key(l)] < SIGNATURE_MIN_POSTS]
            stats["signature_lines_removed"] += len(lines) - len(kept)
            if not kept:
                stats["posts_empty_after_cleaning"] += 1
                continue
            words = [w.lower() for l in kept for w in WORD_RE.findall(l)]
            if ngrams(words) & held_out:
                stats["posts_overlapping_eval_text"] += 1
                continue
            if any(headlines.contains_headline(l) for l in kept) or headlines.contains_headline(" ".join(kept)):
                stats["posts_with_eval_headline"] += 1
                continue
            uid = f"{channel}/{mid}"
            split = "val" if int(hashlib.md5(uid.encode()).hexdigest(), 16) % 100 < VAL_FRACTION * 100 else "train"
            out[split].append({"channel": channel, "id": mid, "date": date, "lines": kept})
            stats[f"posts_{split}"] += 1
            stats[f"words_{split}"] += len(words)

        report[channel] = {**stats, "first_post": dates[0] if dates else None,
                           "last_post": dates[-1] if dates else None}
        print(f"{channel}: {stats['posts_with_text']} posts with text -> "
              f"{stats['posts_train']} train / {stats['posts_val']} val | "
              f"{stats['words_train'] + stats['words_val']:,} words | "
              f"{stats['posts_overlapping_eval_text'] + stats['posts_with_eval_headline']} dropped for eval overlap | "
              f"{stats['signature_lines_removed']} signature lines removed", flush=True)

    for split, rows in out.items():
        with open(OUT / f"telegram_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "telegram_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
