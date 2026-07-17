"""Context gates: training-free admission predicates + selectors.

A gate admits a (question, documents) pair into a verified regime and
returns a selected sub-context; rejected queries fall back to the full
context. The two built-in gates are the paper's Gate 1 (Hotpot-or) and
Gate 2 (2Wiki-entity), ported byte-compatibly from the evidence harnesses.

A gate in this library is DISABLED by default until you attach an evidence
report produced by ``contextgates.verify`` on your own data. You can bypass
that with ``registry.route(..., allow_unverified=True)`` — at your own risk,
which is exactly the practice this project exists to end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .text import Document, approx_tokens, build_pages, coerce_docs, lexical_score

REL_PATTERN = re.compile(
    r"\b(director|directors|performer|performers|composer|composers|producer|"
    r"producers|screenwriter|screenwriters|editor|editors|founder|founders|"
    r"star|stars|actor|actors|author|authors|creator|creators|"
    r"father|mother|grandfather|grandmother|husband|wife|spouse|son|daughter|"
    r"brother|sister|sibling|siblings|uncle|aunt|child|children|cousin|"
    r"grandson|granddaughter|grandchild)\b"
)
YESNO_PATTERN = re.compile(r"^\s*(are|is|do|does|did|were|was|have|has)\b", re.I)


@dataclass
class GateResult:
    admitted: bool
    gate_name: Optional[str]
    context_text: str
    selected_docs: list[Document]
    token_reduction_pct: float
    verified: bool

    def __bool__(self) -> bool:  # truthy iff admitted
        return self.admitted


@dataclass
class Gate:
    name: str
    admit: Callable[[str, list[Document]], bool]
    select: Callable[[str, list[Document]], list[tuple[str, int]]]
    """select returns kept (title, sent_idx) pairs."""
    evidence: Optional[dict] = None  # attach an EvidenceReport dict to enable

    def apply(self, question: str, docs) -> GateResult:
        documents = coerce_docs(docs)
        full_lines = [
            f"{d.title}: {s}" for d in documents for s in d.sentences
        ]
        full_text = "\n".join(full_lines)
        if not self.admit(question, documents):
            return GateResult(False, None, full_text, documents, 0.0, False)
        kept = set(self.select(question, documents))
        selected_docs: list[Document] = []
        kept_lines: list[str] = []
        for d in documents:
            kept_sents = [
                s for i, s in enumerate(d.sentences) if (d.title, i) in kept
            ]
            if kept_sents:
                selected_docs.append(Document(d.title, kept_sents))
                kept_lines.extend(f"{d.title}: {s}" for s in kept_sents)
        full_tokens = sum(approx_tokens(l) for l in full_lines)
        kept_tokens = sum(approx_tokens(l) for l in kept_lines)
        reduction = (
            (full_tokens - kept_tokens) / full_tokens * 100 if full_tokens else 0.0
        )
        return GateResult(
            True,
            self.name,
            "\n".join(kept_lines),
            selected_docs,
            round(reduction, 3),
            self.evidence is not None,
        )


# ---------------------------------------------------------------- built-ins


def _or_admit(question: str, docs: list[Document]) -> bool:
    return " or " in question.lower()


def _or_select(question: str, docs: list[Document], top_k: int = 10):
    pages = build_pages(docs)
    scored = sorted(
        ((lexical_score(question, p["text"], p["titles"]), p["page_index"]) for p in pages),
        key=lambda t: (-t[0], t[1]),
    )
    keep_pages = {idx for _s, idx in scored[:top_k]}
    kept: list[tuple[str, int]] = []
    for p in pages:
        if p["page_index"] in keep_pages:
            kept.extend(p["members"])
    return kept


def matched_titles(question: str, docs: list[Document]) -> set[str]:
    q = question.lower()
    return {d.title for d in docs if d.title.lower() in q}


def _entity_admit(question: str, docs: list[Document]) -> bool:
    if len(matched_titles(question, docs)) < 2:
        return False
    if REL_PATTERN.search(question.lower()):
        return False
    if YESNO_PATTERN.match(question):
        return False
    return True


def _entity_select(question: str, docs: list[Document]):
    titles = matched_titles(question, docs)
    return [
        (d.title, i)
        for d in docs
        if d.title in titles
        for i in range(len(d.sentences))
    ]


def or_comparison_gate(top_k: int = 10) -> Gate:
    """Paper Gate 1 (\"Hotpot-or\"): explicit 'A or B' comparison form +
    lexical top-k page selection. Verified in the paper at ~45% reduction
    with blind-judged semantic parity on its regime."""
    return Gate(
        name="or_comparison",
        admit=_or_admit,
        select=lambda q, d: _or_select(q, d, top_k=top_k),
    )


def entity_title_gate() -> Gate:
    """Paper Gate 2 (\"2Wiki-entity\"): >=2 verbatim question-matched titles,
    no relational-role/kinship word, not yes/no-form; keeps the named
    documents whole. Verified in the paper at ~69% reduction with a
    replicated in-regime semantic accuracy improvement."""
    return Gate(name="entity_title", admit=_entity_admit, select=_entity_select)


# ----------------------------------------------------------------- registry


@dataclass
class GateRegistry:
    gates: list[Gate] = field(default_factory=list)
    admissions: dict = field(default_factory=dict)  # gate_name -> count
    total: int = 0

    def register(self, gate: Gate, evidence: Optional[dict] = None) -> None:
        if evidence is not None:
            gate.evidence = evidence
        self.gates.append(gate)

    def route(self, question: str, docs, allow_unverified: bool = False) -> GateResult:
        """Apply the first admitting gate; fall back to full context.

        Gates without attached evidence are skipped unless
        ``allow_unverified=True``.
        """
        self.total += 1
        documents = coerce_docs(docs)
        for gate in self.gates:
            if gate.evidence is None and not allow_unverified:
                continue
            result = gate.apply(question, documents)
            if result.admitted:
                self.admissions[gate.name] = self.admissions.get(gate.name, 0) + 1
                return result
        full_lines = [f"{d.title}: {s}" for d in documents for s in d.sentences]
        return GateResult(False, None, "\n".join(full_lines), documents, 0.0, False)

    def admission_rate(self) -> float:
        return sum(self.admissions.values()) / self.total if self.total else 0.0
