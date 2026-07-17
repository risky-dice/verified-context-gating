"""Research Task #1269: 2Wiki regime boundary x selector-depth search (exploratory).

Question:
    Does 2Wiki have its own safe admission regime, with its own selector
    depth, analogous to the Hotpot-or + lexical_top10 regime? #1263 showed the
    Hotpot policy does not transfer verbatim (retention 0.9806, token
    reduction 24.9%); the plausible cause is that 2Wiki documents are shorter,
    so top10 pages barely compress. This task searches boundary x depth
    jointly.

Split discipline (fixed before any metric is computed):
    SEARCH split  = rows with even dataset index (this task, exploratory)
    HOLDOUT split = rows with odd dataset index (reserved for the frozen
                    confirmation task; NOT touched here)

Boundaries (question-structure predicates, all cheap):
    cmp_or        type == comparison and " or " in question
    cmp_all       type == comparison
    bridge_or     type == bridge_comparison and " or " in question
    cmp_which     type == comparison and question startswith "Which"
    cmp_or_which  cmp_or and startswith "Which"
    anycmp_or     (comparison or bridge_comparison) and " or "

Selectors:
    page_topK for K in {4, 6, 8, 10}   (2-sentence pages, #1263 scoring)
    sent_topN for N in {8, 12}         (single-sentence granularity; motivated
                                        by #1268 where sentence selection had
                                        the best support coverage at budget)

Gates per cell (identical to the project-standard #1263 gate):
    n >= 100
    answer_retention_rate_given_full >= 0.99
    mean_support_coverage >= 0.98
    perfect_support_rate >= 0.94
    mean_token_reduction_pct >= 35.0

Exploratory only: any passing cell must survive the frozen odd-split holdout
(#1270) before it becomes a claim. Max 300 accepted rows per boundary.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/external/2wikimultihop/dev.json"
OUTPUT = ROOT / "task1269_2wiki_regime_boundary_search_results.json"
MAX_CASES_PER_BOUNDARY = 300


def load_task1263_module():
    spec = importlib.util.spec_from_file_location(
        "task1263",
        Path(__file__).parent / "task1263_2wiki_hotpot_or_top10_transfer.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_task1263_module()

BOUNDARIES = {
    "cmp_or": lambda row: row["type"] == "comparison" and " or " in row["question"].lower(),
    "cmp_all": lambda row: row["type"] == "comparison",
    "bridge_or": lambda row: row["type"] == "bridge_comparison"
    and " or " in row["question"].lower(),
    "cmp_which": lambda row: row["type"] == "comparison"
    and row["question"].strip().lower().startswith("which"),
    "cmp_or_which": lambda row: row["type"] == "comparison"
    and " or " in row["question"].lower()
    and row["question"].strip().lower().startswith("which"),
    "anycmp_or": lambda row: row["type"] in ("comparison", "bridge_comparison")
    and " or " in row["question"].lower(),
}

PAGE_KS = [4, 6, 8, 10]
SENT_NS = [8, 12]

GATES = {
    "min_n": 100,
    "answer_retention_rate_given_full": 0.99,
    "mean_support_coverage": 0.98,
    "perfect_support_rate": 0.94,
    "mean_token_reduction_pct": 35.0,
}


def sentence_context(row: dict, top_n: int) -> dict:
    sentences = T.sentence_rows(row)
    scored = []
    for pos, item in enumerate(sentences):
        page_like = {"text": item["text"], "titles": [item["title"]]}
        scored.append((T.lexical_score(row["question"], page_like), pos))
    chosen = sorted(
        pos for _s, pos in sorted(scored, key=lambda t: (-t[0], t[1]))[:top_n]
    )
    selected_text = "\n".join(sentences[pos]["text"] for pos in chosen)[
        : T.MAX_CONTEXT_CHARS
    ]
    full_text = "\n".join(item["text"] for item in sentences)[: T.MAX_CONTEXT_CHARS]
    selected_tokens = sum(T.approx_tokens(sentences[pos]["text"]) for pos in chosen)
    full_tokens = sum(T.approx_tokens(item["text"]) for item in sentences)
    support_total = sum(1 for item in sentences if item["is_support"])
    support_selected = sum(1 for pos in chosen if sentences[pos]["is_support"])
    answer_in_full = T.answer_present(row["answer"], full_text)
    answer_in_selected = T.answer_present(row["answer"], selected_text)
    return {
        "selected_tokens": selected_tokens,
        "full_tokens": full_tokens,
        "token_reduction_pct": round(
            (full_tokens - selected_tokens) / full_tokens * 100, 3
        )
        if full_tokens
        else 0.0,
        "selected_page_count": len(chosen),
        "full_page_count": len(sentences),
        "support_coverage": support_selected / support_total if support_total else 1.0,
        "perfect_support": support_selected == support_total if support_total else True,
        "answer_in_full": answer_in_full,
        "answer_in_selected": answer_in_selected,
        "answer_retained_given_full": answer_in_selected if answer_in_full else None,
    }


def cell_passes(summary: dict) -> bool:
    return (
        summary["n"] >= GATES["min_n"]
        and summary["answer_retention_rate_given_full"]
        >= GATES["answer_retention_rate_given_full"]
        and summary["mean_support_coverage"] >= GATES["mean_support_coverage"]
        and summary["perfect_support_rate"] >= GATES["perfect_support_rate"]
        and summary["mean_token_reduction_pct"] >= GATES["mean_token_reduction_pct"]
    )


def main() -> None:
    all_rows = json.loads(DATASET.read_text(encoding="utf-8"))
    search_rows = [
        (index, row) for index, row in enumerate(all_rows) if index % 2 == 0
    ]

    cells = []
    for boundary_name, predicate in BOUNDARIES.items():
        accepted = []
        for index, row in search_rows:
            if predicate(row):
                accepted.append((index, row))
            if len(accepted) >= MAX_CASES_PER_BOUNDARY:
                break

        selectors: list[tuple[str, dict]] = []
        for k in PAGE_KS:
            selectors.append(
                (f"page_top{k}", {"kind": "page", "param": k})
            )
        for n in SENT_NS:
            selectors.append((f"sent_top{n}", {"kind": "sent", "param": n}))

        for selector_name, spec in selectors:
            rows = []
            for index, row in accepted:
                metrics = (
                    T.cheap_context(row, spec["param"])
                    if spec["kind"] == "page"
                    else sentence_context(row, spec["param"])
                )
                rows.append({"index": index, **metrics})
            summary = T.summarize(rows)
            cells.append(
                {
                    "boundary": boundary_name,
                    "selector": selector_name,
                    "summary": summary,
                    "passes_gates": cell_passes(summary),
                }
            )

    passing = [cell for cell in cells if cell["passes_gates"]]
    # Rank passing cells: highest token reduction first, then retention.
    passing.sort(
        key=lambda cell: (
            -cell["summary"]["mean_token_reduction_pct"],
            -cell["summary"]["answer_retention_rate_given_full"],
        )
    )
    best = passing[0] if passing else None

    result = {
        "experiment": "Research Task #1269 2Wiki Regime Boundary x Selector-Depth Search",
        "hypothesis": (
            "2Wiki has its own safe admission regime with its own selector depth, "
            "which #1263's verbatim Hotpot policy transfer missed."
        ),
        "mode": "exploratory_search_split_only",
        "dataset": str(DATASET),
        "split": "even dataset indices only; odd indices reserved for #1270 frozen holdout",
        "max_cases_per_boundary": MAX_CASES_PER_BOUNDARY,
        "gates": GATES,
        "cell_count": len(cells),
        "passing_cell_count": len(passing),
        "best_candidate": best,
        "passing_cells": passing,
        "all_cells": cells,
        "success": best is not None,
        "claim_decision": (
            "exploratory_2wiki_regime_candidate_found"
            if best
            else "no_2wiki_regime_candidate_in_search_grid"
        ),
        "limitations": [
            "Exploratory sweep with 36 cells; any hit must survive the #1270 frozen odd-split holdout.",
            "Non-generative retention-layer metrics only.",
            "Boundaries limited to type/lexical predicates; no learned admission.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    view = {k: v for k, v in result.items() if k not in ("all_cells", "passing_cells")}
    print(json.dumps(view, ensure_ascii=False, indent=2))
    print("\n=== cell grid (retention / coverage / perfect / reduction / n) ===")
    for cell in cells:
        s = cell["summary"]
        flag = "PASS" if cell["passes_gates"] else "    "
        print(
            f"{flag} {cell['boundary']:>13} x {cell['selector']:<11} "
            f"ret={s['answer_retention_rate_given_full']:.4f} "
            f"cov={s['mean_support_coverage']:.4f} "
            f"perf={s['perfect_support_rate']:.4f} "
            f"red={s['mean_token_reduction_pct']:6.2f} n={s['n']}"
        )


if __name__ == "__main__":
    main()
