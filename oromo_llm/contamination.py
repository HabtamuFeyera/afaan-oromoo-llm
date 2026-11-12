"""Keep evaluation text out of training text.

Two checks, used by the corpus builder and the Telegram importer:
- 13-word sequences (the GPT-3 paper's test) catch shared article text.
- Headlines are often shorter than 13 words, so MasakhaNEWS dev/test
  headlines are matched as whole phrases: any training line that contains
  one (case and punctuation ignored) is removed. This matters because the
  instruction-tuning headline task uses these headlines as answers.
"""

import json
from pathlib import Path

from oromo_llm.text import WORD_RE, normalize

ROOT = Path(__file__).resolve().parent.parent
MIN_HEADLINE_WORDS = 4
PROBE = 4  # words per probe phrase used to find candidate matches quickly


def words_key(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(normalize(text))]


class HeadlineFilter:
    """Finds MasakhaNEWS dev/test headlines inside a line of text."""

    def __init__(self, splits=("dev", "test")):
        self.headlines = []
        self.index = {}
        for split in splits:
            path = ROOT / "data" / "news" / f"{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                head = words_key(json.loads(line)["text"].split("\n")[0])
                if len(head) >= MIN_HEADLINE_WORDS:
                    self.headlines.append(head)
                    self.index.setdefault(tuple(head[:PROBE]), []).append(" ".join(head))

    def __len__(self):
        return len(self.headlines)

    def contains_headline(self, text: str) -> bool:
        w = words_key(text)
        joined = None
        for i in range(len(w) - PROBE + 1):
            cands = self.index.get(tuple(w[i:i + PROBE]))
            if cands:
                joined = joined or " " + " ".join(w) + " "
                if any(f" {c} " in joined for c in cands):
                    return True
        return False
