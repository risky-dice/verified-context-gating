"""Research Task #1263: 2Wiki transfer of frozen Hotpot-or top10.

Question:
    Does the #1262 HotpotQA-style explicit candidate comparison candidate
    transfer to 2WikiMultiHopQA comparison rows?

Frozen candidate:
    boundary: question contains " or "
    selector: lexical_top10

This is a local non-generative falsification. No external API is called and no
model is loaded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from statistics import mean
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))


DATASET = ROOT / "data/external/2wikimultihop/dev.json"
OUTPUT = ROOT / "task1263_2wiki_hotpot_or_top10_transfer_results.json"
START_INDEX = 0
COMPARISON_CASE_COUNT = 300
TOP_K = 10
MAX_CONTEXT_CHARS = 8000
PAGE_SENTENCE_COUNT = 2
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


def boundary_accept(question: str) -> bool:
    return " or " in question.lower()


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
        if boundary_accept(row["question"])
    ]


def terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def sentence_rows(row: dict) -> list[dict]:
    support = {(title, sent_idx) for title, sent_idx in row["supporting_facts"]}
    rows = []
    for title, sentences in row["context"]:
        for sent_idx, sentence in enumerate(sentences):
            rows.append(
                {
                    "title": title,
                    "sent_idx": sent_idx,
                    "text": f"{title}: {sentence}",
                    "is_support": (title, sent_idx) in support,
                }
            )
    return rows


def build_pages(row: dict) -> list[dict]:
    rows = sentence_rows(row)
    pages = []
    for start in range(0, len(rows), PAGE_SENTENCE_COUNT):
        chunk = rows[start : start + PAGE_SENTENCE_COUNT]
        text = " ".join(item["text"] for item in chunk)
        pages.append(
            {
                "page_index": len(pages),
                "text": text,
                "tokens": approx_tokens(text),
                "support_count": sum(1 for item in chunk if item["is_support"]),
                "titles": sorted({item["title"] for item in chunk}),
            }
        )
    return pages


def lexical_score(question: str, page: dict) -> float:
    q_terms = terms(question)
    page_terms = terms(page["text"])
    title_terms = set().union(*(terms(title) for title in page["titles"]))
    overlap = len(q_terms & page_terms)
    title_overlap = len(q_terms & title_terms)
    phrase_bonus = sum(3 for title in page["titles"] if title.lower() in question.lower())
    return float(overlap + title_overlap * 3 + phrase_bonus)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_present(answer: str, text: str) -> bool:
    answer_norm = normalize_text(answer)
    text_norm = normalize_text(text)
    return bool(answer_norm) and answer_norm in text_norm


def cheap_context(row: dict, top_k: int) -> dict:
    pages = build_pages(row)
    scored = [
        (lexical_score(row["question"], page), page["page_index"])
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
    answer_in_full = answer_present(row["answer"], full_text)
    answer_in_selected = answer_present(row["answer"], selected_text)
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
                **cheap_context(row, TOP_K),
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
        "experiment": "Research Task #1263 2Wiki Transfer of Frozen Hotpot-Or Top10",
        "hypothesis": (
            "If the #1262 candidate is not HotpotQA-format specific, the same "
            "question contains 'or' + lexical_top10 policy should pass on 2Wiki "
            "comparison rows."
        ),
        "dataset": str(DATASET),
        "start_index": START_INDEX,
        "comparison_case_count": COMPARISON_CASE_COUNT,
        "accepted_or_count": len(rows),
        "boundary": "dataset == 2wiki and question contains ' or '",
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
            "hotpot_or_top10_transfers_to_2wiki_non_generatively"
            if success
            else "hotpot_or_top10_fails_2wiki_transfer"
        ),
        "limitations": [
            "Non-generative validation only.",
            "String answer-presence is not semantic answer quality.",
            "First 300 2Wiki comparison rows only.",
            "No existing compressor baseline was run.",
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
