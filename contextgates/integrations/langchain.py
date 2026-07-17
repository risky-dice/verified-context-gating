"""LangChain integration: verified gates as a document compressor.

    from langchain.retrievers import ContextualCompressionRetriever
    from contextgates import entity_title_gate, GateRegistry
    from contextgates.integrations.langchain import GateDocumentCompressor

    registry = GateRegistry()
    registry.register(entity_title_gate(), evidence=my_report)  # verified first!

    retriever = ContextualCompressionRetriever(
        base_compressor=GateDocumentCompressor(registry=registry),
        base_retriever=my_retriever,
    )

Semantics mirror the LlamaIndex adapter: admitted queries get dropped
documents removed and kept documents trimmed to the gate's sentences;
non-admitted queries pass through untouched (full-context fallback).

Requires langchain-core (not a dependency of contextgates).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..gates import GateRegistry
from .common import ChunkView, group_chunks

try:  # pragma: no cover - exercised only with langchain installed
    from langchain_core.callbacks import Callbacks
    from langchain_core.documents import Document as LCDocument
    from langchain_core.documents.compressor import BaseDocumentCompressor
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "GateDocumentCompressor requires langchain-core: pip install langchain-core"
    ) from exc


def _title_of(doc: "LCDocument", title_key: Optional[str]) -> str:
    meta = doc.metadata or {}
    if title_key and meta.get(title_key):
        return str(meta[title_key])
    for key in ("title", "source", "file_name", "document_title"):
        if meta.get(key):
            return str(meta[key])
    return "untitled"


class GateDocumentCompressor(BaseDocumentCompressor):
    """Applies a verified gate registry to retrieved LangChain documents."""

    registry: GateRegistry
    allow_unverified: bool = False
    title_key: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence["LCDocument"],
        query: str,
        callbacks: Optional["Callbacks"] = None,
        **kwargs: Any,
    ) -> Sequence["LCDocument"]:
        if not documents:
            return documents
        chunks = [
            ChunkView(title=_title_of(d, self.title_key), text=d.page_content, handle=d)
            for d in documents
        ]
        docs, _by_title = group_chunks(chunks)
        result = self.registry.route(
            query, docs, allow_unverified=self.allow_unverified
        )
        if not result.admitted:
            return documents  # full-context fallback

        selected = {d.title: d.sentences for d in result.selected_docs}
        kept: list["LCDocument"] = []
        for chunk in chunks:
            sentences = selected.get(chunk.title)
            if not sentences:
                continue
            keep = [s for s in sentences if s in chunk.text]
            if not keep:
                continue
            meta = dict(chunk.handle.metadata or {})
            meta["contextgates_gate"] = result.gate_name
            meta["contextgates_token_reduction_pct"] = result.token_reduction_pct
            kept.append(LCDocument(page_content=" ".join(keep), metadata=meta))
        return kept or documents

    async def acompress_documents(
        self,
        documents: Sequence["LCDocument"],
        query: str,
        callbacks: Optional["Callbacks"] = None,
        **kwargs: Any,
    ) -> Sequence["LCDocument"]:
        return self.compress_documents(documents, query, callbacks, **kwargs)
