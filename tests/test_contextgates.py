"""Tests: gates behave as published, and the ladder gates correctly."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextgates import (  # noqa: E402
    GateRegistry,
    build_blinded_pack,
    entity_title_gate,
    evaluate_retention,
    or_comparison_gate,
    run_ladder,
    score_blinded_verdicts,
    sign_test_two_sided,
)

DOCS = [
    {"title": "Alpha Film", "sentences": ["Alpha Film is a 1990 drama.", "It won a prize."]},
    {"title": "Beta Film", "sentences": ["Beta Film is a 2001 comedy.", "It flopped."]},
    {"title": "Distractor One", "sentences": ["Unrelated text about trains.", "More noise."]},
    {"title": "Distractor Two", "sentences": ["Nothing to do with films.", "Filler."]},
]


def test_entity_gate_admits_and_keeps_named_docs():
    gate = entity_title_gate()
    r = gate.apply("Which film was released first, Alpha Film or Beta Film?", DOCS)
    assert r.admitted
    titles = {d.title for d in r.selected_docs}
    assert titles == {"Alpha Film", "Beta Film"}
    assert "trains" not in r.context_text
    assert r.token_reduction_pct > 0


def test_entity_gate_rejects_relational_and_yesno():
    gate = entity_title_gate()
    assert not gate.apply(
        "Who is the director of Alpha Film or Beta Film?", DOCS
    ).admitted  # relational word
    assert not gate.apply(
        "Are Alpha Film and Beta Film from the same country?", DOCS
    ).admitted  # yes/no form
    assert not gate.apply("Which film was released first?", DOCS).admitted  # <2 titles


def test_or_gate_admission():
    gate = or_comparison_gate()
    assert gate.apply("Which is older, Alpha Film or Beta Film?", DOCS).admitted
    assert not gate.apply("When was Alpha Film released?", DOCS).admitted


def test_rejected_query_gets_full_context_via_registry():
    reg = GateRegistry()
    reg.register(entity_title_gate(), evidence={"passed": True})
    r = reg.route("When was Alpha Film released?", DOCS)
    assert not r.admitted
    assert "trains" in r.context_text  # full context fallback
    assert reg.admission_rate() == 0.0


def test_unverified_gate_is_skipped_by_default():
    reg = GateRegistry()
    reg.register(entity_title_gate())  # no evidence attached
    q = "Which film was released first, Alpha Film or Beta Film?"
    assert not reg.route(q, DOCS).admitted
    assert reg.route(q, DOCS, allow_unverified=True).admitted


def test_retention_layer_scores_only_admitted_rows():
    rows = [
        {
            "question": "Which film was released first, Alpha Film or Beta Film?",
            "docs": DOCS,
            "gold_answer": "Alpha Film",
            "support": [("Alpha Film", 0), ("Beta Film", 0)],
        },
        {"question": "When was Alpha Film released?", "docs": DOCS, "gold_answer": "1990"},
    ]
    s = evaluate_retention(entity_title_gate(), rows)
    assert s["n_total"] == 2 and s["n_admitted"] == 1
    assert s["answer_retention_rate_given_full"] == 1.0
    assert s["mean_support_coverage"] == 1.0
    assert s["passed"] is False  # min_n=100 not met — thresholds bite


def test_blinding_hides_arm_and_unblinds_correctly():
    pairs = [
        {
            "case_id": i,
            "question": "q",
            "gold_answer": "Alpha Film",
            "selected_answer": "Alpha Film" if i % 2 == 0 else "Beta Film",
            "full_answer": "Beta Film",
            "token_reduction_pct": 50.0,
        }
        for i in range(40)
    ]
    pack = build_blinded_pack(pairs, salt="test-salt")
    blob = str(pack)
    assert "selected" not in blob and "full_answer" not in blob
    # Judge marks whichever side says "Alpha Film" correct.
    verdicts = {}
    for item, p in zip(pack, pairs):
        text = item["messages"][1]["content"]
        a = text.split("Answer A:\n")[1].split("\n\nAnswer B:")[0]
        b = text.split("Answer B:\n")[1].split("\n\nReturn")[0]
        verdicts[p["case_id"]] = {
            "a_correct": a.strip() == "Alpha Film",
            "b_correct": b.strip() == "Alpha Film",
        }
    s = score_blinded_verdicts(pairs, verdicts, salt="test-salt")
    assert s["selected_semantic_correct_rate"] == 0.5  # half our selected answers are right
    assert s["full_semantic_correct_rate"] == 0.0
    assert s["discordant"]["selected_only"] == 20
    assert s["passed"] is True


def test_ladder_stops_at_l1_without_generate():
    rows = [
        {
            "question": "Which film was released first, Alpha Film or Beta Film?",
            "docs": DOCS,
            "gold_answer": "Alpha Film",
        }
    ] * 120
    report = run_ladder(entity_title_gate(), rows)
    assert report["layers"][0]["passed"] is True
    assert report["passed"] is False  # L1 alone never enables a gate
    assert "L2/L3 were not run" in report["note"]


def test_sign_test_matches_paper_values():
    assert abs(sign_test_two_sided(29, 7) - 3.126e-4) < 1e-6
    assert abs(sign_test_two_sided(5, 2) - 0.4531) < 1e-3


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
