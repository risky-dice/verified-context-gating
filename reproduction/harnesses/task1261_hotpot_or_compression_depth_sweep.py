"""Research Task #1261: Hotpot-or compression depth sweep.

Question:
    Can the #1260 frozen Hotpot-or boundary recover the token-reduction gate by
    using a more aggressive lexical selector depth?

Frozen boundary:
    dataset == hotpot and question contains " or "

Only selector depth changes:
    lexical_top8, lexical_top10, lexical_top12

This is a local non-generative sweep. No external API is called and no model is
loaded.
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

import task1193_prefill_hook_selector_120_replication as selector  # noqa: E402
import task1235_answer_presence_reality_check as answer_presence  # noqa: E402


DATASET = ROOT / "data/external/hotpot_dev_distractor_v1.json"
OUTPUT = ROOT / "task1261_hotpot_or_compression_depth_sweep_results.json"
START_INDEX = 3400
COMPARISON_CASE_COUNT = 240
TOP_K_VALUES = (8, 10, 12)
MAX_CONTEXT_CHARS = 8000


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def boundary_accept(question: str) -> bool:
    return " or " in question.lower()


def collect_rows() -> list[tuple[int, dict]]:
    rows = load_json(DATASET)
    collected = []
    for index, row in enumerate(rows[START_INDEX:], start=START_INDEX):
        if row.get("type") != "comparison":
            continue
        if boundary_accept(row["question"]):
            collected.append((index, row))
        if len([1 for _index, item in collected if item.get("type") == "comparison"]) >= COMPARISON_CASE_COUNT:
            # This branch is unreachable because collected contains accepted only,
            # but keep the loop bounded by the dataset length below.
            break
    # Use the first accepted rows found within the same 240-comparison holdout
    # window as #1260, not all rows after START_INDEX.
    comparison_rows = []
    for index, row in enumerate(rows[START_INDEX:], start=START_INDEX):
        if row.get("type") != "comparison":
            continue
        comparison_rows.append((index, row))
        if len(comparison_rows) >= COMPARISON_CASE_COUNT:
            break
    return [(index, row) for index, row in comparison_rows if boundary_accept(row["question"])]


def cheap_context(row: dict, top_k: int) -> dict:
    pages = selector.build_pages(row)
    scored = [
        (selector.lexical_score(row["question"], page), page["page_index"])
        for page in pages
    ]
    indices = {
        page_index
        for _score, page_index in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]
    }
    selected_pages = [page for page in pages if page["page_index"] in indices]
    selected_text = "\n".join(page["text"] for page in selected_pages)[:MAX_CONTEXT_CHARS]
    full_text = "\n".join(page["text"] for page in pages)[:MAX_CONTEXT_CHARS]
    full_tokens = sum(page["tokens"] for page in pages)
    selected_tokens = sum(page["tokens"] for page in selected_pages)
    support_total = sum(page["support_count"] for page in pages)
    support_selected = sum(page["support_count"] for page in selected_pages)
    answer_in_full = answer_presence.answer_present(row["answer"], full_text)
    answer_in_selected = answer_presence.answer_present(row["answer"], selected_text)
    return {
        "selected_tokens": selected_tokens,
        "full_tokens": full_tokens,
        "token_reduction_pct": round((full_tokens - selected_tokens) / full_tokens * 100, 3)
        if full_tokens
        else 0.0,
        "selected_page_count": len(selected_pages),
        "full_page_count": len(pages),
        "support_coverage": support_selected / support_total if support_total else 1.0,
        "perfect_support": support_selected == support_total if support_total else True,
        "answer_in_full": answer_in_full,
        "answer_in_selected": answer_in_selected,
        "answer_retained_given_full": answer_in_selected if answer_in_full else None,
    }


def audit_rows(top_k: int, source_rows: list[tuple[int, dict]]) -> list[dict]:
    rows = []
    for index, row in source_rows:
        rows.append(
            {
                "index": index,
                "question": row["question"],
                "gold_answer": row["answer"],
                "strategy": f"lexical_top{top_k}",
                **cheap_context(row, top_k),
            }
        )
    return rows


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
    results = []
    all_rows = []
    for top_k in TOP_K_VALUES:
        rows = audit_rows(top_k, source_rows)
        summary = summarize(rows)
        success = (
            summary["n"] >= 100
            and summary["answer_retention_rate_given_full"] >= 0.99
            and summary["mean_support_coverage"] >= 0.98
            and summary["perfect_support_rate"] >= 0.94
            and summary["mean_token_reduction_pct"] >= 30.0
        )
        results.append(
            {
                "strategy": f"lexical_top{top_k}",
                "summary": summary,
                "success": bool(success),
            }
        )
        all_rows.extend(rows)

    ranked = sorted(
        results,
        key=lambda item: (
            item["success"],
            item["summary"]["answer_retention_rate_given_full"],
            item["summary"]["mean_support_coverage"],
            item["summary"]["perfect_support_rate"],
            item["summary"]["mean_token_reduction_pct"],
        ),
        reverse=True,
    )
    payload = {
        "experiment": "Research Task #1261 Hotpot-Or Compression Depth Sweep",
        "hypothesis": (
            "A more aggressive lexical selector depth may preserve #1260 safety "
            "while recovering the >=30% token-reduction gate."
        ),
        "dataset": str(DATASET),
        "start_index": START_INDEX,
        "comparison_case_count": COMPARISON_CASE_COUNT,
        "accepted_hotpot_or_count": len(source_rows),
        "boundary": "dataset == hotpot and question contains ' or '",
        "top_k_values": list(TOP_K_VALUES),
        "success_gate": {
            "n": ">= 100",
            "answer_retention_rate_given_full": ">= 0.99",
            "mean_support_coverage": ">= 0.98",
            "perfect_support_rate": ">= 0.94",
            "mean_token_reduction_pct": ">= 30.0",
        },
        "best_strategy": ranked[0],
        "strategy_results": ranked,
        "success": bool(any(item["success"] for item in results)),
        "claim_decision": (
            "hotpot_or_depth_sweep_finds_acceleration_candidate"
            if any(item["success"] for item in results)
            else "hotpot_or_depth_sweep_fails_acceleration_gate"
        ),
        "limitations": [
            "Non-generative audit only.",
            "HotpotQA-only holdout.",
            "No semantic judge.",
            "Selector depth was searched on the holdout, so the chosen depth needs another frozen validation.",
        ],
        "rows": all_rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = dict(payload)
    compact.pop("rows")
    compact.pop("limitations")
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
