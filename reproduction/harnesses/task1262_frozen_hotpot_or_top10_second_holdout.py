"""Research Task #1262: frozen Hotpot-or top10 second holdout.

Question:
    Does the #1261 candidate survive when both boundary and selector depth are
    frozen on a new disjoint HotpotQA comparison slice?

Frozen candidate:
    boundary: dataset == hotpot and question contains " or "
    selector: lexical_top10

This is a local non-generative validation. No external API is called and no
model is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import task1261_hotpot_or_compression_depth_sweep as depth_sweep  # noqa: E402


DATASET = ROOT / "data/external/hotpot_dev_distractor_v1.json"
OUTPUT = ROOT / "task1262_frozen_hotpot_or_top10_second_holdout_results.json"
START_INDEX = 4200
COMPARISON_CASE_COUNT = 240
TOP_K = 10


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows() -> list[tuple[int, dict]]:
    rows = load_json(DATASET)
    comparison_rows = []
    for index, row in enumerate(rows[START_INDEX:], start=START_INDEX):
        if row.get("type") != "comparison":
            continue
        comparison_rows.append((index, row))
        if len(comparison_rows) >= COMPARISON_CASE_COUNT:
            break
    return [
        (index, row)
        for index, row in comparison_rows
        if depth_sweep.boundary_accept(row["question"])
    ]


def summarize(rows: list[dict]) -> dict:
    retained = [row["answer_retained_given_full"] for row in rows if row["answer_retained_given_full"] is not None]
    return {
        "n": len(rows),
        "answer_in_full_rate": round(mean(row["answer_in_full"] for row in rows), 4) if rows else 0.0,
        "answer_in_selected_rate": round(mean(row["answer_in_selected"] for row in rows), 4) if rows else 0.0,
        "answer_retention_rate_given_full": round(mean(retained), 4) if retained else 0.0,
        "mean_support_coverage": round(mean(row["support_coverage"] for row in rows), 4) if rows else 0.0,
        "perfect_support_rate": round(mean(row["perfect_support"] for row in rows), 4) if rows else 0.0,
        "strict_safe_rate": round(
            mean(
                row["answer_retained_given_full"] is True and row["support_coverage"] >= 0.95
                for row in rows
            ),
            4,
        )
        if rows
        else 0.0,
        "mean_token_reduction_pct": round(mean(row["token_reduction_pct"] for row in rows), 3) if rows else 0.0,
        "mean_selected_page_count": round(mean(row["selected_page_count"] for row in rows), 3) if rows else 0.0,
    }


def main() -> None:
    source_rows = collect_rows()
    rows = []
    for index, row in source_rows:
        rows.append(
            {
                "index": index,
                "question": row["question"],
                "gold_answer": row["answer"],
                "strategy": f"lexical_top{TOP_K}",
                **depth_sweep.cheap_context(row, TOP_K),
            }
        )
    summary = summarize(rows)
    success = (
        summary["n"] >= 100
        and summary["answer_retention_rate_given_full"] >= 0.99
        and summary["mean_support_coverage"] >= 0.98
        and summary["perfect_support_rate"] >= 0.94
        and summary["mean_token_reduction_pct"] >= 35.0
    )
    payload = {
        "experiment": "Research Task #1262 Frozen Hotpot-Or Top10 Second Holdout",
        "hypothesis": (
            "The #1261 hotpot_or + lexical_top10 candidate survives when frozen "
            "on a new disjoint HotpotQA comparison slice."
        ),
        "dataset": str(DATASET),
        "start_index": START_INDEX,
        "comparison_case_count": COMPARISON_CASE_COUNT,
        "accepted_hotpot_or_count": len(rows),
        "boundary": "dataset == hotpot and question contains ' or '",
        "selector": f"lexical_top{TOP_K}",
        "success_gate": {
            "n": ">= 100",
            "answer_retention_rate_given_full": ">= 0.99",
            "mean_support_coverage": ">= 0.98",
            "perfect_support_rate": ">= 0.94",
            "mean_token_reduction_pct": ">= 35.0",
        },
        "summary": summary,
        "success": bool(success),
        "claim_decision": (
            "frozen_hotpot_or_top10_second_holdout_passes"
            if success
            else "frozen_hotpot_or_top10_second_holdout_fails"
        ),
        "limitations": [
            "Non-generative validation only.",
            "HotpotQA-only evidence.",
            "No semantic judge or generation quality check.",
            "The boundary may still be dataset-format specific.",
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
