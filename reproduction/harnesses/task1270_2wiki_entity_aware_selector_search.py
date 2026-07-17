"""Research Task #1270: 2Wiki entity-aware selector search (exploratory, v2).

Question:
    #1269 failed because cheap lexical depth-shrinking drops one of 2Wiki's
    two cross-document supporting facts. Diagnostics on the search split
    showed comparison rows keep 100% of supporting facts inside the documents
    named verbatim in the question (~34% of chars), while bridge_comparison
    keeps 0% there. Can an entity-document selector plus a purely
    question-structural admission predicate form a passing 2Wiki regime?

v1 lesson (recorded): page-granularity selection failed spuriously because
#1263 pages chunk sentences ACROSS document boundaries, so mixed-title pages
were dropped (coverage 0.55). v2 selects at sentence granularity. v1 also
leaked kinship questions ("maternal grandfather of X") through the role-word
filter and lost 4 retention points to yes/no-answer substring artifacts;
v2 extends the role list with kinship terms and adds a yes/no-form exclusion.

Deployable admission predicate (no dataset metadata used):
    ENT2:   >= 2 context titles appear verbatim (lowercase) in the question
    NOREL:  question contains no relational-role or kinship word
    NOYN:   question does not start with are/is/do/does/did/were/was/have/has

Boundaries:
    ent2_norel        ENT2 and NOREL
    ent2_norel_noyn   ENT2 and NOREL and NOYN

Selectors (sentence granularity, matched-title docs only):
    sent_full           all sentences of matched docs
    sent_top3_per_doc   top 3 sentences per matched doc by lexical score
    sent_top2_per_doc   top 2 sentences per matched doc
    sent_top1_per_doc   top 1 sentence per matched doc

Split discipline: even dataset indices only (search split). Odd indices
remain reserved for the #1271 frozen holdout. Gates are the project standard
(n>=100, retention>=0.99, coverage>=0.98, perfect>=0.94, reduction>=35).
Max 300 accepted rows per boundary.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/external/2wikimultihop/dev.json"
OUTPUT = ROOT / "task1270_2wiki_entity_aware_selector_search_results.json"
MAX_CASES_PER_BOUNDARY = 300

REL_PATTERN = re.compile(
    r"\b(director|directors|performer|performers|composer|composers|producer|"
    r"producers|screenwriter|screenwriters|editor|editors|founder|founders|"
    r"star|stars|actor|actors|author|authors|creator|creators|"
    r"father|mother|grandfather|grandmother|husband|wife|spouse|son|daughter|"
    r"brother|sister|sibling|siblings|uncle|aunt|child|children|cousin|"
    r"grandson|granddaughter|grandchild)\b"
)
YESNO_PATTERN = re.compile(r"^\s*(are|is|do|does|did|were|was|have|has)\b", re.I)

GATES = {
    "min_n": 100,
    "answer_retention_rate_given_full": 0.99,
    "mean_support_coverage": 0.98,
    "perfect_support_rate": 0.94,
    "mean_token_reduction_pct": 35.0,
}


def load_task1263_module():
    spec = importlib.util.spec_from_file_location(
        "task1263",
        Path(__file__).parent / "task1263_2wiki_hotpot_or_top10_transfer.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_task1263_module()


def matched_titles(row: dict) -> set[str]:
    question = row["question"].lower()
    return {title for title, _sents in row["context"] if title.lower() in question}


def boundary_ent2_norel(row: dict) -> bool:
    return len(matched_titles(row)) >= 2 and not REL_PATTERN.search(
        row["question"].lower()
    )


def boundary_ent2_norel_noyn(row: dict) -> bool:
    return boundary_ent2_norel(row) and not YESNO_PATTERN.match(row["question"])


BOUNDARIES = {
    "ent2_norel": boundary_ent2_norel,
    "ent2_norel_noyn": boundary_ent2_norel_noyn,
}

SELECTOR_CAPS = {
    "sent_full": None,
    "sent_top3_per_doc": 3,
    "sent_top2_per_doc": 2,
    "sent_top1_per_doc": 1,
}


def entity_sentence_context(row: dict, per_doc_cap: int | None) -> dict:
    titles = matched_titles(row)
    sentences = T.sentence_rows(row)
    matched_positions = [
        pos for pos, item in enumerate(sentences) if item["title"] in titles
    ]
    if per_doc_cap is not None:
        kept = []
        for title in sorted(titles):
            doc_positions = [
                pos for pos in matched_positions if sentences[pos]["title"] == title
            ]
            scored = sorted(
                doc_positions,
                key=lambda pos: (
                    -T.lexical_score(
                        row["question"],
                        {
                            "text": sentences[pos]["text"],
                            "titles": [sentences[pos]["title"]],
                        },
                    ),
                    pos,
                ),
            )
            kept.extend(scored[:per_doc_cap])
        matched_positions = sorted(kept)

    selected_text = "\n".join(sentences[pos]["text"] for pos in matched_positions)[
        : T.MAX_CONTEXT_CHARS
    ]
    full_text = "\n".join(item["text"] for item in sentences)[: T.MAX_CONTEXT_CHARS]
    selected_tokens = sum(
        T.approx_tokens(sentences[pos]["text"]) for pos in matched_positions
    )
    full_tokens = sum(T.approx_tokens(item["text"]) for item in sentences)
    support_total = sum(1 for item in sentences if item["is_support"])
    support_selected = sum(
        1 for pos in matched_positions if sentences[pos]["is_support"]
    )
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
        "selected_page_count": len(matched_positions),
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
    search_rows = [(i, row) for i, row in enumerate(all_rows) if i % 2 == 0]

    type_audit: dict[str, dict[str, int]] = {}
    cells = []
    for boundary_name, predicate in BOUNDARIES.items():
        accepted = []
        audit: dict[str, int] = {}
        for index, row in search_rows:
            if predicate(row):
                if len(accepted) < MAX_CASES_PER_BOUNDARY:
                    accepted.append((index, row))
                audit[row["type"]] = audit.get(row["type"], 0) + 1
        type_audit[boundary_name] = audit

        for selector_name, cap in SELECTOR_CAPS.items():
            rows = [
                {"index": index, **entity_sentence_context(row, cap)}
                for index, row in accepted
            ]
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
    passing.sort(
        key=lambda cell: (
            -cell["summary"]["mean_token_reduction_pct"],
            -cell["summary"]["answer_retention_rate_given_full"],
        )
    )
    best = passing[0] if passing else None

    result = {
        "experiment": "Research Task #1270 2Wiki Entity-Aware Selector Search (v2)",
        "hypothesis": (
            "Entity-document sentence selection plus a question-structural "
            "admission predicate forms a passing 2Wiki regime, fixing #1269's "
            "coverage collapse."
        ),
        "mode": "exploratory_search_split_only",
        "dataset": str(DATASET),
        "split": "even dataset indices only; odd indices reserved for #1271 frozen holdout",
        "v1_lessons": [
            "Page-granularity failed spuriously: #1263 pages chunk across document boundaries, dropping mixed-title pages (coverage 0.55).",
            "Kinship questions leaked through the role filter; kinship terms added.",
            "Yes/no answers caused substring retention artifacts; yes/no-form questions excluded in the _noyn boundary.",
        ],
        "admission_type_audit_full_search_split": type_audit,
        "max_cases_per_boundary": MAX_CASES_PER_BOUNDARY,
        "gates": GATES,
        "cell_count": len(cells),
        "passing_cell_count": len(passing),
        "best_candidate": best,
        "passing_cells": passing,
        "all_cells": cells,
        "success": best is not None,
        "claim_decision": (
            "exploratory_2wiki_entity_regime_candidate_found"
            if best
            else "entity_aware_selector_also_fails_2wiki"
        ),
        "limitations": [
            "Exploratory search split; the single best cell must survive the #1271 frozen odd-split holdout.",
            "Non-generative retention-layer metrics.",
            "Title matching is verbatim-lowercase; paraphrased titles are not admitted.",
            "Role/kinship word list was tuned on this search split (v1 -> v2); the holdout is the only clean test.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    view = {k: v for k, v in result.items() if k not in ("all_cells", "passing_cells")}
    print(json.dumps(view, ensure_ascii=False, indent=2))
    print("\n=== cells ===")
    for cell in cells:
        s = cell["summary"]
        flag = "PASS" if cell["passes_gates"] else "    "
        print(
            f"{flag} {cell['boundary']:>16} x {cell['selector']:<17} "
            f"ret={s['answer_retention_rate_given_full']:.4f} "
            f"cov={s['mean_support_coverage']:.4f} "
            f"perf={s['perfect_support_rate']:.4f} "
            f"red={s['mean_token_reduction_pct']:6.2f} n={s['n']}"
        )


if __name__ == "__main__":
    main()
