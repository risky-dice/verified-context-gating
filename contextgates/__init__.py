"""contextgates — verified context gating for RAG.

Cut RAG input cost only where you have proven it is safe:

    from contextgates import entity_title_gate, GateRegistry, run_ladder

    gate = entity_title_gate()
    report = run_ladder(gate, my_dataset, generate=my_gen, judge=my_judge)
    registry = GateRegistry()
    if report["passed"]:
        registry.register(gate, evidence=report)

    result = registry.route(question, docs)   # falls back to full context
    prompt_context = result.context_text

This package ships no API keys and no model weights: `generate` and `judge`
are callables you provide. Gates are training-free and dependency-free.

Paper: "Verified Context Gating: Training-Free Structural Admission for
Context Selection, with Pre-Registered Quality Evidence" (Lee, 2026).
"""

from .gates import (
    Gate,
    GateRegistry,
    GateResult,
    entity_title_gate,
    matched_titles,
    or_comparison_gate,
)
from .text import Document, approx_tokens, build_pages, lexical_score
from .verify import (
    JudgeThresholds,
    RetentionThresholds,
    build_blinded_pack,
    evaluate_retention,
    run_generation_pairs,
    run_ladder,
    score_blinded_verdicts,
    sign_test_two_sided,
)

__version__ = "0.1.0a0"

__all__ = [
    "Document",
    "Gate",
    "GateRegistry",
    "GateResult",
    "JudgeThresholds",
    "RetentionThresholds",
    "approx_tokens",
    "build_blinded_pack",
    "build_pages",
    "entity_title_gate",
    "evaluate_retention",
    "lexical_score",
    "matched_titles",
    "or_comparison_gate",
    "run_generation_pairs",
    "run_ladder",
    "score_blinded_verdicts",
    "sign_test_two_sided",
]
