"""Research Task #1260: frozen Hotpot-or boundary holdout.

Question:
    Does the #1259 structural admission candidate survive on a disjoint HotpotQA
    comparison slice?

Frozen boundary:
    dataset == hotpot and question contains " or "

This is a local non-generative holdout. No external API is called and no model
is loaded.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import task1235_answer_presence_reality_check as answer_presence  # noqa: E402
import task1251_frozen_cheap_retrieval_generation_canary as canary  # noqa: E402


DATASET = ROOT / "data/external/hotpot_dev_distractor_v1.json"
OUTPUT = ROOT / "task1260_frozen_hotpot_or_boundary_holdout_results.json"
START_INDEX = 3400
COMPARISON_CASE_COUNT = 240


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def boundary_accept(question: str) -> bool:
    return " or " in question.lower()


def collect_holdout_rows() -> list[tuple[int, dict]]:
    rows = load_json(DATASET)
    collected = []
    for index, row in enumerate(rows[START_INDEX:], start=START_INDEX):
        if row.get("type") != "comparison":
            continue
        collected.append((index, row))
        if len(collected) >= COMPARISON_CASE_COUNT:
            break
    return collected


def audit_row(index: int, row: dict) -> dict:
    context = canary.cheap_context(row, "hotpot")
    selected_text = context.pop("selected_text")
    full_text = context.pop("full_text")
    answer_in_full = answer_presence.answer_present(row["answer"], full_text)
    answer_in_selected = answer_presence.answer_present(row["answer"], selected_text)
    return {
        "index": index,
        "dataset": "hotpot",
        "question": row["question"],
        "gold_answer": row["answer"],
        "accepted": boundary_accept(row["question"]),
        "answer_in_full": answer_in_full,
        "answer_in_selected": answer_in_selected,
        "answer_retained_given_full": answer_in_selected if answer_in_full else None,
        **context,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "answer_retention_rate_given_full": 0.0,
            "mean_support_coverage": 0.0,
            "perfect_support_rate": 0.0,
            "strict_safe_rate": 0.0,
            "mean_token_reduction_pct": 0.0,
        }
    retained = [row["answer_retained_given_full"] for row in rows if row["answer_retained_given_full"] is not None]
    return {
        "n": len(rows),
        "answer_in_full_rate": round(mean(row["answer_in_full"] for row in rows), 4),
        "answer_in_selected_rate": round(mean(row["answer_in_selected"] for row in rows), 4),
        "answer_retention_rate_given_full": round(mean(retained), 4) if retained else 0.0,
        "mean_support_coverage": round(mean(row["support_coverage"] for row in rows), 4),
        "perfect_support_rate": round(mean(row["perfect_support"] for row in rows), 4),
        "strict_safe_rate": round(
            mean(
                row["answer_retained_given_full"] is True and row["support_coverage"] >= 0.95
                for row in rows
            ),
            4,
        ),
        "mean_token_reduction_pct": round(mean(row["token_reduction_pct"] for row in rows), 3),
        "mean_selected_page_count": round(mean(row["selected_page_count"] for row in rows), 3),
    }


def main() -> None:
    rows = [audit_row(index, row) for index, row in collect_holdout_rows()]
    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]
    accepted_summary = summarize(accepted)
    rejected_summary = summarize(rejected)
    overall_summary = summarize(rows)
    success = (
        accepted_summary["n"] >= 100
        and accepted_summary["answer_retention_rate_given_full"] >= 0.99
        and accepted_summary["mean_support_coverage"] >= 0.98
        and accepted_summary["perfect_support_rate"] >= 0.94
        and accepted_summary["mean_token_reduction_pct"] >= 30.0
    )
    payload = {
        "experiment": "Research Task #1260 Frozen Hotpot-Or Boundary Holdout",
        "hypothesis": (
            "The #1259 hotpot_or_question admission boundary survives on a "
            "disjoint HotpotQA comparison slice."
        ),
        "dataset": str(DATASET),
        "start_index": START_INDEX,
        "comparison_case_count": len(rows),
        "boundary": "dataset == hotpot and question contains ' or '",
        "summary": {
            "overall": overall_summary,
            "accepted_hotpot_or_question": accepted_summary,
            "rejected_hotpot_non_or_question": rejected_summary,
            "accepted_rate": round(len(accepted) / len(rows), 4) if rows else 0.0,
        },
        "success": bool(success),
        "claim_decision": (
            "frozen_hotpot_or_boundary_holdout_passes"
            if success
            else "frozen_hotpot_or_boundary_holdout_fails"
        ),
        "limitations": [
            "Non-generative audit only.",
            "HotpotQA-only holdout; not cross-dataset evidence.",
            "Answer-string retention is not semantic correctness.",
            "Boundary may still be a HotpotQA formatting artifact.",
        ],
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = dict(payload)
    compact.pop("rows")
    compact.pop("limitations")
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
