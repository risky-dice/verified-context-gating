"""Research Task #1271: 2Wiki entity regime frozen holdout.

Frozen candidate (fixed by #1270 search-split ranking before this run):

    boundary: ent2_norel_noyn
        >= 2 context titles appear verbatim (lowercase) in the question
        AND no relational-role/kinship word in the question
        AND question does not start with are/is/do/does/did/were/was/have/has
    selector: sent_full
        all sentences belonging to the matched-title documents

Holdout: ALL accepted rows with ODD dataset index — untouched by #1269 and
#1270, which searched only even indices. Single policy, no sweep, no
post-hoc adjustment.

Gates (project standard, unchanged):
    n >= 100
    answer_retention_rate_given_full >= 0.99
    mean_support_coverage >= 0.98
    perfect_support_rate >= 0.94
    mean_token_reduction_pct >= 35.0
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/external/2wikimultihop/dev.json"
OUTPUT = ROOT / "task1271_2wiki_entity_regime_frozen_holdout_results.json"


def load_task1270_module():
    spec = importlib.util.spec_from_file_location(
        "task1270",
        Path(__file__).parent / "task1270_2wiki_entity_aware_selector_search.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load_task1270_module()
T = S.T

GATES = S.GATES


def main() -> None:
    all_rows = json.loads(DATASET.read_text(encoding="utf-8"))
    holdout_rows = [(i, row) for i, row in enumerate(all_rows) if i % 2 == 1]

    accepted = [
        (index, row)
        for index, row in holdout_rows
        if S.boundary_ent2_norel_noyn(row)
    ]
    type_audit: dict[str, int] = {}
    for _index, row in accepted:
        type_audit[row["type"]] = type_audit.get(row["type"], 0) + 1

    rows = [
        {
            "index": index,
            "question": row["question"],
            "gold_answer": row["answer"],
            **S.entity_sentence_context(row, None),
        }
        for index, row in accepted
    ]
    summary = T.summarize(rows)
    success = (
        summary["n"] >= GATES["min_n"]
        and summary["answer_retention_rate_given_full"]
        >= GATES["answer_retention_rate_given_full"]
        and summary["mean_support_coverage"] >= GATES["mean_support_coverage"]
        and summary["perfect_support_rate"] >= GATES["perfect_support_rate"]
        and summary["mean_token_reduction_pct"] >= GATES["mean_token_reduction_pct"]
    )

    result = {
        "experiment": "Research Task #1271 2Wiki Entity Regime Frozen Holdout",
        "hypothesis": (
            "The #1270 candidate (ent2_norel_noyn + sent_full) passes the "
            "project-standard gates on the untouched odd-index holdout."
        ),
        "mode": "frozen_holdout",
        "dataset": str(DATASET),
        "split": "ALL odd dataset indices (never used by #1269/#1270 searches)",
        "boundary": "ent2_norel_noyn (>=2 verbatim question-matched titles, no role/kinship word, not yes/no-form)",
        "selector": "sent_full (all sentences of matched-title documents)",
        "accepted_case_count": len(accepted),
        "admission_type_audit": type_audit,
        "gates": GATES,
        "summary": summary,
        "success": bool(success),
        "claim_decision": (
            "frozen_2wiki_entity_regime_holdout_passes"
            if success
            else "frozen_2wiki_entity_regime_holdout_fails"
        ),
        "limitations": [
            "Non-generative retention-layer metrics; generative canary still pending.",
            "Boundary word lists were tuned on the search split; this holdout is the clean test of that tuning.",
            "Verbatim-lowercase title matching only.",
        ],
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    view = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(view, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
