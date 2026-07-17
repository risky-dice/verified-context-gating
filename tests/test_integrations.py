"""Adapter tests that do not require the frameworks themselves.

The framework-specific glue is thin; the risky logic is the chunk->Document
grouping and the sentence mapping back onto chunks. Those are tested here.
Framework classes are only import-guarded, so we assert the guard too.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextgates import GateRegistry, entity_title_gate  # noqa: E402
from contextgates.integrations.common import (  # noqa: E402
    ChunkView,
    group_chunks,
    kept_titles,
    split_sentences,
)


def test_split_sentences_basic():
    assert split_sentences("A one. B two! C three?") == ["A one.", "B two!", "C three?"]
    assert split_sentences("   ") == []
    assert split_sentences("no terminator") == ["no terminator"]


def test_group_chunks_preserves_retrieval_order_and_merges_titles():
    chunks = [
        ChunkView("Alpha Film", "Alpha Film is a 1990 drama.", handle="c1"),
        ChunkView("Beta Film", "Beta Film is a 2001 comedy.", handle="c2"),
        ChunkView("Alpha Film", "It won a prize.", handle="c3"),
    ]
    docs, by_title = group_chunks(chunks)
    assert [d.title for d in docs] == ["Alpha Film", "Beta Film"]  # first-seen order
    assert docs[0].sentences == ["Alpha Film is a 1990 drama.", "It won a prize."]
    assert [c.handle for c in by_title["Alpha Film"]] == ["c1", "c3"]


def test_gate_over_grouped_chunks_drops_distractor_titles():
    chunks = [
        ChunkView("Alpha Film", "Alpha Film is a 1990 drama.", handle="c1"),
        ChunkView("Beta Film", "Beta Film is a 2001 comedy.", handle="c2"),
        ChunkView("Rail Transport", "Trains carry freight.", handle="c3"),
    ]
    docs, _ = group_chunks(chunks)
    registry = GateRegistry()
    registry.register(entity_title_gate(), evidence={"passed": True})
    result = registry.route("Which film was released first, Alpha Film or Beta Film?", docs)
    assert result.admitted
    assert kept_titles(result) == {"Alpha Film", "Beta Film"}
    assert result.token_reduction_pct > 0


def test_rejected_query_keeps_everything():
    chunks = [
        ChunkView("Alpha Film", "Alpha Film is a 1990 drama.", handle="c1"),
        ChunkView("Rail Transport", "Trains carry freight.", handle="c2"),
    ]
    docs, _ = group_chunks(chunks)
    registry = GateRegistry()
    registry.register(entity_title_gate(), evidence={"passed": True})
    result = registry.route("When was Alpha Film released?", docs)
    assert not result.admitted
    assert kept_titles(result) == set()  # caller must pass all chunks through


def test_adapters_raise_clear_import_error_without_framework():
    for module, needle in (
        ("contextgates.integrations.llamaindex", "llama-index-core"),
        ("contextgates.integrations.langchain", "langchain-core"),
    ):
        try:
            __import__(module)
        except ImportError as e:
            assert needle in str(e)
        else:
            pass  # framework installed: import succeeded, nothing to assert


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print("all green" if not fails else f"{fails} failures")
    sys.exit(1 if fails else 0)
