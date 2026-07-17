"""Research Task #1259: structural admission boundary search.

Question:
    If the literal Which-prefix gate failed, can cheap observable features find
    a better admission boundary for cheap context reduction?

This is a local search over #1258 diagnostic rows. It does not call external
APIs and does not load a model. Any passing boundary is exploratory and must be
frozen before validation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "task1258_nonwhich_admission_negative_control_results.json"
OUTPUT = ROOT / "task1259_structural_admission_boundary_search_results.json"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def capital_phrases(question: str) -> list[str]:
    phrases = re.findall(r"(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\\s+|$)){1,8}", question)
    cleaned = []
    for phrase in phrases:
        phrase = phrase.strip(" ?.,;:'\"()[]{}")
        if len(phrase) >= 3 and phrase.lower() not in STOPWORDS:
            cleaned.append(phrase)
    return cleaned


def feature_row(row: dict) -> dict:
    question = row["question"]
    q_terms = terms(question)
    caps = capital_phrases(question)
    normalized = question.lower().strip()
    has_or = " or " in normalized
    starts_which = normalized.startswith("which ")
    starts_what = normalized.startswith("what ")
    starts_were = normalized.startswith("were ")
    starts_was = normalized.startswith("was ")
    starts_are = normalized.startswith("are ")
    has_comparison_word = bool(
        re.search(r"\\b(older|younger|larger|smaller|higher|lower|first|last|more|less|both|same)\\b", normalized)
    )
    safety_answer = bool(row["answer_retained_given_full"])
    safety_support = row["support_coverage"] >= 0.95
    safety_perfect = bool(row["perfect_support"])
    safe_strict = safety_answer and safety_support
    return {
        **row,
        "starts_which": starts_which,
        "starts_what": starts_what,
        "starts_were": starts_were,
        "starts_was": starts_was,
        "starts_are": starts_are,
        "has_or": has_or,
        "has_comparison_word": has_comparison_word,
        "capital_phrase_count": len(caps),
        "question_term_count": len(q_terms),
        "question_char_len": len(question),
        "selected_page_count_le_10": row["selected_page_count"] <= 10,
        "selected_page_count_ge_12": row["selected_page_count"] >= 12,
        "token_reduction_ge_20": row["token_reduction_pct"] >= 20,
        "token_reduction_ge_30": row["token_reduction_pct"] >= 30,
        "safe_answer_retention": safety_answer,
        "safe_support_95": safety_support,
        "safe_perfect_support": safety_perfect,
        "safe_strict_answer_and_support95": safe_strict,
    }


def boundary_defs() -> dict[str, callable]:
    return {
        "all_rows": lambda row: True,
        "literal_which": lambda row: row["starts_which"],
        "hotpot_all": lambda row: row["dataset"] == "hotpot",
        "hotpot_or_question": lambda row: row["dataset"] == "hotpot" and row["has_or"],
        "hotpot_comparison_word": lambda row: row["dataset"] == "hotpot" and row["has_comparison_word"],
        "hotpot_or_or_comparison": lambda row: row["dataset"] == "hotpot"
        and (row["has_or"] or row["has_comparison_word"]),
        "2wiki_or_question": lambda row: row["dataset"] == "2wiki" and row["has_or"],
        "or_question_all": lambda row: row["has_or"],
        "comparison_word_all": lambda row: row["has_comparison_word"],
        "or_and_capitals_ge2": lambda row: row["has_or"] and row["capital_phrase_count"] >= 2,
        "hotpot_or_capitals_ge2": lambda row: row["dataset"] == "hotpot"
        and row["has_or"]
        and row["capital_phrase_count"] >= 2,
        "token_reduction_ge30": lambda row: row["token_reduction_ge_30"],
        "token_reduction_ge20": lambda row: row["token_reduction_ge_20"],
        "hotpot_token_reduction_ge30": lambda row: row["dataset"] == "hotpot"
        and row["token_reduction_ge_30"],
        "selected_pages_ge12": lambda row: row["selected_page_count_ge_12"],
        "hotpot_selected_pages_ge12": lambda row: row["dataset"] == "hotpot"
        and row["selected_page_count_ge_12"],
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "accepted_rate": 0.0,
            "answer_retention_rate_given_full": 0.0,
            "mean_support_coverage": 0.0,
            "perfect_support_rate": 0.0,
            "strict_safe_rate": 0.0,
            "mean_token_reduction_pct": 0.0,
        }
    return {
        "n": len(rows),
        "answer_retention_rate_given_full": round(
            mean(row["safe_answer_retention"] for row in rows), 4
        ),
        "mean_support_coverage": round(mean(row["support_coverage"] for row in rows), 4),
        "perfect_support_rate": round(mean(row["safe_perfect_support"] for row in rows), 4),
        "strict_safe_rate": round(
            mean(row["safe_strict_answer_and_support95"] for row in rows), 4
        ),
        "mean_token_reduction_pct": round(mean(row["token_reduction_pct"] for row in rows), 3),
        "by_dataset": {
            dataset: {
                "n": len(items),
                "answer_retention_rate_given_full": round(
                    mean(item["safe_answer_retention"] for item in items), 4
                ),
                "mean_support_coverage": round(mean(item["support_coverage"] for item in items), 4),
                "perfect_support_rate": round(mean(item["safe_perfect_support"] for item in items), 4),
                "strict_safe_rate": round(
                    mean(item["safe_strict_answer_and_support95"] for item in items), 4
                ),
                "mean_token_reduction_pct": round(mean(item["token_reduction_pct"] for item in items), 3),
            }
            for dataset, items in group_by(rows, "dataset").items()
        },
    }


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return dict(groups)


def main() -> None:
    source = load_json(SOURCE)
    rows = [feature_row(row) for row in source["rows"]]
    total = len(rows)
    results = []
    for name, predicate in boundary_defs().items():
        accepted = [row for row in rows if predicate(row)]
        summary = summarize(accepted)
        summary["accepted_rate"] = round(len(accepted) / total, 4) if total else 0.0
        success = (
            summary["n"] >= 80
            and summary["answer_retention_rate_given_full"] >= 0.99
            and summary["mean_support_coverage"] >= 0.98
            and summary["perfect_support_rate"] >= 0.94
            and summary["mean_token_reduction_pct"] >= 30.0
        )
        results.append(
            {
                "boundary": name,
                "summary": summary,
                "success": bool(success),
            }
        )

    ranked = sorted(
        results,
        key=lambda item: (
            item["success"],
            item["summary"]["answer_retention_rate_given_full"],
            item["summary"]["mean_support_coverage"],
            item["summary"]["perfect_support_rate"],
            item["summary"]["mean_token_reduction_pct"],
            item["summary"]["n"],
        ),
        reverse=True,
    )
    feature_counts = {
        "dataset": dict(Counter(row["dataset"] for row in rows)),
        "admission_group": dict(Counter(row["admission_group"] for row in rows)),
        "starts_which": dict(Counter(str(row["starts_which"]) for row in rows)),
        "has_or": dict(Counter(str(row["has_or"]) for row in rows)),
        "has_comparison_word": dict(Counter(str(row["has_comparison_word"]) for row in rows)),
    }
    payload = {
        "experiment": "Research Task #1259 Structural Admission Boundary Search",
        "hypothesis": (
            "After the literal Which gate failed, cheap observable structural "
            "features may identify a safer admission boundary."
        ),
        "source": str(SOURCE),
        "case_count": total,
        "feature_counts": feature_counts,
        "success_gate": {
            "n": ">= 80",
            "answer_retention_rate_given_full": ">= 0.99",
            "mean_support_coverage": ">= 0.98",
            "perfect_support_rate": ">= 0.94",
            "mean_token_reduction_pct": ">= 30.0",
        },
        "best_boundary": ranked[0],
        "successful_boundaries": [item for item in ranked if item["success"]],
        "ranked_boundaries": ranked,
        "success": bool(any(item["success"] for item in ranked)),
        "claim_decision": (
            "structural_boundary_search_finds_candidate"
            if any(item["success"] for item in ranked)
            else "structural_boundary_search_finds_no_candidate"
        ),
        "limitations": [
            "Exploratory search on the same diagnostic rows.",
            "Non-generative safety labels only.",
            "Candidate boundaries must be frozen and validated on held-out rows.",
            "Some features include post-selection token metrics and may not be pure pre-admission signals.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = dict(payload)
    compact.pop("ranked_boundaries")
    compact.pop("limitations")
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
