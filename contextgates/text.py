"""Text utilities shared by gates and selectors.

Ported faithfully from the paper harnesses (task1263/task1264) so that
library behavior is byte-compatible with the published evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "to", "was", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "with",
}


@dataclass
class Document:
    """A candidate document: a title and its sentences, in order."""

    title: str
    sentences: list[str] = field(default_factory=list)

    @staticmethod
    def coerce(value) -> "Document":
        if isinstance(value, Document):
            return value
        if isinstance(value, dict):
            return Document(title=value["title"], sentences=list(value["sentences"]))
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return Document(title=value[0], sentences=list(value[1]))
        raise TypeError(f"cannot coerce {type(value)!r} to Document")


def coerce_docs(docs) -> list[Document]:
    return [Document.coerce(d) for d in docs]


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def normalize_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def lexical_score(question: str, text: str, titles: list[str]) -> float:
    """Question-overlap score used by the Hotpot-or selector (paper Gate 1)."""
    q_terms = terms(question)
    page_terms = terms(text)
    title_terms: set[str] = set()
    for title in titles:
        title_terms |= terms(title)
    overlap = len(q_terms & page_terms)
    title_overlap = len(q_terms & title_terms)
    phrase_bonus = sum(3 for title in titles if title.lower() in question.lower())
    return float(overlap + title_overlap * 3 + phrase_bonus)


def sentence_lines(docs: list[Document]) -> list[dict]:
    """Flatten docs to title-prefixed sentence rows (harness format)."""
    rows = []
    for doc in docs:
        for sent_idx, sentence in enumerate(doc.sentences):
            rows.append(
                {
                    "title": doc.title,
                    "sent_idx": sent_idx,
                    "text": f"{doc.title}: {sentence}",
                }
            )
    return rows


def build_pages(docs: list[Document], page_sentence_count: int = 2) -> list[dict]:
    """Chunk the flattened sentence stream into fixed-size pages.

    NOTE: pages may span document boundaries; this mirrors the paper's Gate 1
    construction exactly (its evidence was produced with this behavior).
    """
    rows = sentence_lines(docs)
    pages = []
    for start in range(0, len(rows), page_sentence_count):
        chunk = rows[start : start + page_sentence_count]
        text = " ".join(item["text"] for item in chunk)
        pages.append(
            {
                "page_index": len(pages),
                "text": text,
                "tokens": approx_tokens(text),
                "titles": sorted({item["title"] for item in chunk}),
                "members": [(item["title"], item["sent_idx"]) for item in chunk],
            }
        )
    return pages
