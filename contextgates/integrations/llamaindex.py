"""LlamaIndex integration: drop verified gates into a query engine.

    from llama_index.core import VectorStoreIndex
    from contextgates import entity_title_gate, GateRegistry
    from contextgates.integrations.llamaindex import GateNodePostprocessor

    registry = GateRegistry()
    registry.register(entity_title_gate(), evidence=my_report)  # verified first!

    engine = index.as_query_engine(
        node_postprocessors=[GateNodePostprocessor(registry)]
    )

Semantics: if a gate admits the query, nodes belonging to documents the gate
dropped are removed and admitted nodes keep only the gate's selected
sentences. If no gate admits (or none is verified), nodes pass through
untouched — the full-context fallback.

Requires llama-index-core (not a dependency of contextgates).
"""

from __future__ import annotations

from typing import Any, Optional

from ..gates import GateRegistry
from .common import ChunkView, group_chunks

try:  # pragma: no cover - exercised only with llama-index installed
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "GateNodePostprocessor requires llama-index-core: pip install llama-index-core"
    ) from exc


def _title_of(node: "NodeWithScore") -> str:
    meta = node.node.metadata or {}
    for key in ("title", "file_name", "document_title", "source"):
        if meta.get(key):
            return str(meta[key])
    return node.node.ref_doc_id or node.node.node_id


class GateNodePostprocessor(BaseNodePostprocessor):
    """Applies a verified gate registry to retrieved nodes."""

    registry: GateRegistry
    allow_unverified: bool = False
    title_key: Optional[str] = None

    class Config:  # pydantic v1 style used by llama-index
        arbitrary_types_allowed = True

    def __init__(
        self,
        registry: GateRegistry,
        allow_unverified: bool = False,
        title_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            registry=registry,
            allow_unverified=allow_unverified,
            title_key=title_key,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GateNodePostprocessor"

    def _title(self, node: "NodeWithScore") -> str:
        if self.title_key:
            meta = node.node.metadata or {}
            if meta.get(self.title_key):
                return str(meta[self.title_key])
        return _title_of(node)

    def _postprocess_nodes(
        self,
        nodes: list["NodeWithScore"],
        query_bundle: Optional["QueryBundle"] = None,
    ) -> list["NodeWithScore"]:
        if query_bundle is None or not nodes:
            return nodes
        question = query_bundle.query_str
        chunks = [
            ChunkView(title=self._title(n), text=n.node.get_content(), handle=n)
            for n in nodes
        ]
        docs, by_title = group_chunks(chunks)
        result = self.registry.route(
            question, docs, allow_unverified=self.allow_unverified
        )
        if not result.admitted:
            return nodes  # full-context fallback

        selected = {d.title: d.sentences for d in result.selected_docs}
        kept_nodes: list["NodeWithScore"] = []
        for chunk in chunks:
            sentences = selected.get(chunk.title)
            if not sentences:
                continue  # whole document dropped by the gate
            node = chunk.handle
            keep = [s for s in sentences if s in chunk.text]
            if not keep:
                continue
            new_text = " ".join(keep)
            if new_text != chunk.text:
                node = _with_text(node, new_text)
            _mark(node, result)
            kept_nodes.append(node)
        return kept_nodes or nodes


def _with_text(node: "NodeWithScore", text: str) -> "NodeWithScore":
    clone = node.node.model_copy(deep=True) if hasattr(node.node, "model_copy") else node.node.copy()
    try:
        clone.set_content(text)
    except Exception:  # TextNode fallback
        clone.text = text
    return NodeWithScore(node=clone, score=node.score)


def _mark(node: "NodeWithScore", result) -> None:
    meta = node.node.metadata or {}
    meta["contextgates_gate"] = result.gate_name
    meta["contextgates_token_reduction_pct"] = result.token_reduction_pct
    node.node.metadata = meta
