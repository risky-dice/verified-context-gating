"""Framework-agnostic glue: retrieved chunks <-> contextgates Documents.

Both adapters face the same problem: frameworks hand you a flat list of
retrieved chunks with metadata, while gates reason over titled documents
whose sentence order matters. This module does that mapping once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from ..text import Document

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Cheap sentence split. Swap in nltk/spacy via ``sentence_splitter`` if
    your text needs it; gates only require stable, ordered units."""
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


@dataclass
class ChunkView:
    """One retrieved chunk, framework-neutral."""

    title: str
    text: str
    handle: object  # the original framework object, returned untouched


def group_chunks(
    chunks: Iterable[ChunkView],
    sentence_splitter: Optional[Callable[[str], list[str]]] = None,
) -> tuple[list[Document], dict[str, list[ChunkView]]]:
    """Group chunks by title into Documents, preserving retrieval order.

    Returns (documents, title -> chunks) so callers can map a gate's kept
    sentences back onto original framework objects.
    """
    splitter = sentence_splitter or split_sentences
    order: list[str] = []
    by_title: dict[str, list[ChunkView]] = {}
    for chunk in chunks:
        if chunk.title not in by_title:
            by_title[chunk.title] = []
            order.append(chunk.title)
        by_title[chunk.title].append(chunk)
    docs = [
        Document(
            title=title,
            sentences=[s for c in by_title[title] for s in splitter(c.text)],
        )
        for title in order
    ]
    return docs, by_title


def kept_titles(result) -> set[str]:
    """Titles that survived the gate (empty if the query was rejected)."""
    return {d.title for d in result.selected_docs} if result.admitted else set()
