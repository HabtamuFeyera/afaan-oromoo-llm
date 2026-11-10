"""Oromo text normalization shared by the corpus builder and the fine-tuning data.

normalize() makes the hudhaa one character, unifies quotes, and strips links,
citation marks and news-site labels. The regexes and word lists are also used
by scripts/02_build_corpus.py to filter lines.
"""

import re
import unicodedata

# --- character normalization -------------------------------------------------

# Every apostrophe-like mark becomes ASCII ' so the hudhaa is one symbol.
APOSTROPHES = "’‘´`ʼʻ′‛"
DOUBLE_QUOTES = "“”„‟«»″"
HYPHENS = "‐‑‒­"
ZERO_WIDTH = "​‌‍⁠﻿"

CHAR_MAP = str.maketrans(
    {**{c: "'" for c in APOSTROPHES},
     **{c: '"' for c in DOUBLE_QUOTES},
     **{c: "-" for c in HYPHENS},
     **{c: "" for c in ZERO_WIDTH},
     "…": "...",
     " ": " "}
)

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
CITATION_RE = re.compile(r"\[\d+(?:[,–-]\d+)*\]")  # [3], [3,4], [3-5]
READ_MORE_RE = re.compile(r"\[?\s*Read More\s*\]?", re.IGNORECASE)
BBC_SUFFIX_RE = re.compile(r"\s*-\s*BBC News Afaan Oromoo\s*$")
MULTISPACE_RE = re.compile(r"[ \t]+")

# --- line filters ------------------------------------------------------------

# Page residue from news sites. Labels are stripped (the sentence around them
# is real Oromo); lines that are only site furniture are dropped.
LABEL_RE = re.compile(
    r"Qabiyyeetti darbi Kutaawwan(?: Oduu Itoophiyaa Viidiyoo Jajjaboo)*"
    r"|Madda suuraa,?(?: (?:Getty Images|Reuters|AFP|EPA|AP|BBC|Social Media|Facebook|Twitter|[A-Z][\w']*(?: [A-Z][\w']*)?))?,?"
    r"|Ibsa waa'ee suuraa,?"
    r"|Getty Images",
    re.IGNORECASE,
)
BOILERPLATE_RE = re.compile(r"Editorial Guidelines|Be the first to comment", re.IGNORECASE)
ELLIPSIS_RUN_RE = re.compile(r"\.{4,9}")
DOT_LEADER_RE = re.compile(r"\.{10,}|(?:\. ){4,}")  # glossary/TOC formatting
ETHIOPIC_RE = re.compile(r"[ሀ-᎟ⶀ-⷟]")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
ARABIC_RE = re.compile(r"[؀-ۿ]")
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")

# Common English function words that are not also Oromo words.
ENGLISH_STOP = {
    "the", "and", "of", "is", "to", "that", "with", "for", "this", "are",
    "was", "were", "be", "by", "from", "which", "have", "has", "it", "on",
    "at", "as", "or", "not", "but", "they", "their", "will", "would", "can",
    "been", "about", "into", "who", "what", "when", "where", "there",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(CHAR_MAP)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = URL_RE.sub(" ", text)
    text = CITATION_RE.sub("", text)
    text = READ_MORE_RE.sub(" ", text)
    text = BBC_SUFFIX_RE.sub("", text)
    text = LABEL_RE.sub(" ", text)
    text = ELLIPSIS_RUN_RE.sub("...", text)
    return MULTISPACE_RE.sub(" ", text).strip()
