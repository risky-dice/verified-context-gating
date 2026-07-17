"""Research Task #1258: non-Which admission negative control.

Question:
    Is the frozen Which gate actually selecting safer cheap-context cases, or
    would non-Which comparison rows look just as safe under the same cheap
    selector?

This is a local non-generative audit. It does not call external APIs and does
not load a model.
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


OUTPUT = ROOT / "task1258_nonwhich_admission_negative_control_results.json"
SOURCE_RESULTS = (
    ("2wiki_1241", "2wiki", ROOT / "task1241_bridge_frozen_which_gate_validation_results.json"),
    ("2wiki_1242", "2wiki", ROOT / "task1242_bridge_which_gate_second_holdout_results.json"),
    ("hotpot_1243", "hotpot", ROOT / "task1243_hotpot_comparison_which_gate_transfer_results.json"),
    ("hotpot_1244", "hotpot", ROOT / "task1244_hotpot_comparison_which_gate_240_replication_results.json"),
)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def source_rows() -> tuple[dict[int, dict], dict[int, dict]]:
    return canary.source_row_maps()


def collect_items() -> list[dict]:
    items = []
    for source, dataset, path in SOURCE_RESULTS:
        payload = load_json(path)
        for row in payload["rows"]:
            items.append(
                {
                    "source": source,
                    "dataset": dataset,
                    "index": row["index"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "admission_group": "which_accepted" if row["use_selected"] else "nonwhich_rejected",
                    "original_use_selected": row["use_selected"],
                    "original_full_match": row["full_match"],
                    "original_mixed_match": row["mixed_match"],
                }
            )
    return items


def source_row_for(item: dict, two_wiki_by_index: dict[int, dict], hotpot_by_index: dict[int, dict]) -> dict:
    return two_wiki_by_index[item["index"]] if item["dataset"] == "2wiki" else hotpot_by_index[item["index"]]


def audit_item(item: dict, source_row: dict) -> dict:
    context = canary.cheap_context(source_row, item["dataset"])
    selected_text = context.pop("selected_text")
    full_text = context.pop("full_text")
    answer_in_full = answer_presence.answer_present(source_row["answer"], full_text)
    answer_in_selected = answer_presence.answer_present(source_row["answer"], selected_text)
    answer_retained_given_full = answer_in_selected if answer_in_full else None
    return {
        **item,
        **context,
        "answer_in_full": answer_in_full,
        "answer_in_selected": answer_in_selected,
        "answer_retained_given_full": answer_retained_given_full,
    }


def summarize(rows: list[dict]) -> dict:
    retained = [row["answer_retained_given_full"] for row in rows if row["answer_retained_given_full"] is not None]
    return {
        "n": len(rows),
        "answer_in_full_rate": round(mean(row["answer_in_full"] for row in rows), 4) if rows else 0.0,
        "answer_in_selected_rate": round(mean(row["answer_in_selected"] for row in rows), 4) if rows else 0.0,
        "answer_retention_rate_given_full": round(mean(retained), 4) if retained else 0.0,
        "mean_support_coverage": round(mean(row["support_coverage"] for row in rows), 4) if rows else 0.0,
        "perfect_support_rate": round(mean(row["perfect_support"] for row in rows), 4) if rows else 0.0,
        "mean_token_reduction_pct": round(mean(row["token_reduction_pct"] for row in rows), 3) if rows else 0.0,
        "mean_selected_page_count": round(mean(row["selected_page_count"] for row in rows), 3) if rows else 0.0,
    }


def grouped_summary(rows: list[dict]) -> dict:
    by_group = defaultdict(list)
    by_dataset_group = defaultdict(list)
    by_source_group = defaultdict(list)
    for row in rows:
        by_group[row["admission_group"]].append(row)
        by_dataset_group[f"{row['dataset']}:{row['admission_group']}"].append(row)
        by_source_group[f"{row['source']}:{row['admission_group']}"].append(row)
    return {
        "by_admission_group": {name: summarize(items) for name, items in sorted(by_group.items())},
        "by_dataset_group": {name: summarize(items) for name, items in sorted(by_dataset_group.items())},
        "by_source_group": {name: summarize(items) for name, items in sorted(by_source_group.items())},
    }


def main() -> None:
    two_wiki_by_index, hotpot_by_index = source_rows()
    rows = []
    for item in collect_items():
        rows.append(audit_item(item, source_row_for(item, two_wiki_by_index, hotpot_by_index)))

    summaries = grouped_summary(rows)
    accepted = summaries["by_admission_group"].get("which_accepted", {})
    rejected = summaries["by_admission_group"].get("nonwhich_rejected", {})
    answer_retention_gap = round(
        accepted.get("answer_retention_rate_given_full", 0.0)
        - rejected.get("answer_retention_rate_given_full", 0.0),
        4,
    )
    support_gap = round(
        accepted.get("mean_support_coverage", 0.0) - rejected.get("mean_support_coverage", 0.0),
        4,
    )
    perfect_support_gap = round(
        accepted.get("perfect_support_rate", 0.0) - rejected.get("perfect_support_rate", 0.0),
        4,
    )
    token_reduction_gap = round(
        accepted.get("mean_token_reduction_pct", 0.0)
        - rejected.get("mean_token_reduction_pct", 0.0),
        3,
    )
    success = (
        answer_retention_gap >= 0.05
        or support_gap >= 0.05
        or perfect_support_gap >= 0.10
    )
    payload = {
        "experiment": "Research Task #1258 Non-Which Admission Negative Control",
        "hypothesis": (
            "The Which gate is meaningful only if accepted Which rows are safer "
            "than rejected non-Which rows under the same cheap context selector."
        ),
        "source_results": [str(path) for _source, _dataset, path in SOURCE_RESULTS],
        "summary": {
            **summaries,
            "comparative_gaps": {
                "answer_retention_gap_accepted_minus_rejected": answer_retention_gap,
                "mean_support_coverage_gap_accepted_minus_rejected": support_gap,
                "perfect_support_gap_accepted_minus_rejected": perfect_support_gap,
                "mean_token_reduction_gap_accepted_minus_rejected": token_reduction_gap,
            },
        },
        "success": bool(success),
        "claim_decision": (
            "which_admission_has_nonwhich_negative_control_support"
            if success
            else "which_admission_not_distinguished_from_nonwhich_control"
        ),
        "limitations": [
            "Non-generative audit only.",
            "Answer-string retention is not semantic correctness.",
            "The rejected group is not a random sample of all non-Which questions.",
            "Small non-Which count in 2Wiki sources limits confidence.",
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
