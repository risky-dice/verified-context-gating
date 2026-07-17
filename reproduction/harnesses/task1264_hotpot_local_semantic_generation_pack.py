"""Research Task #1264: Hotpot-local semantic generation package.

Question:
    Does the #1262 HotpotQA-local selected context preserve generated answer
    quality compared with full context?

Default mode is dry-run only. It writes generation requests for selected and
full contexts without calling an external API. Live mode requires explicit user
approval, OPENAI_API_KEY, and a budget ceiling.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
HOTPOT_DATASET = ROOT / "data/external/hotpot_dev_distractor_v1.json"
SOURCE = ROOT / "task1262_frozen_hotpot_or_top10_second_holdout_results.json"
REQUESTS = ROOT / "task1264_hotpot_local_semantic_generation_requests.jsonl"
OUTPUT = ROOT / "task1264_hotpot_local_semantic_generation_pack_results.json"
TARGET_CASES = 32
TOP_K = 10
PAGE_SENTENCE_COUNT = 2
MAX_CONTEXT_CHARS = 8000
DEFAULT_MODEL = "gpt-4.1-mini"
INPUT_PRICE_PER_1M = 0.40
OUTPUT_PRICE_PER_1M = 1.60
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

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. Return a concise "
    "answer. If the context is insufficient, say: insufficient context."
)


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_match(gold: str, answer: str) -> bool:
    gold_norm = normalize_text(gold)
    answer_norm = normalize_text(answer)
    return bool(gold_norm) and gold_norm in answer_norm


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * INPUT_PRICE_PER_1M + output_tokens * OUTPUT_PRICE_PER_1M) / 1_000_000


def terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


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


def selected_and_full_context(row: dict) -> dict:
    pages = build_pages(row)
    scored = [
        (lexical_score(row["question"], page), page["page_index"])
        for page in pages
    ]
    selected_indices = {
        page_index
        for _score, page_index in sorted(scored, key=lambda item: (-item[0], item[1]))[:TOP_K]
    }
    selected_pages = [page for page in pages if page["page_index"] in selected_indices]
    selected_text = "\n".join(page["text"] for page in selected_pages)[:MAX_CONTEXT_CHARS]
    full_text = "\n".join(page["text"] for page in pages)[:MAX_CONTEXT_CHARS]
    return {
        "selected_context": selected_text,
        "full_context": full_text,
        "selected_context_tokens": sum(page["tokens"] for page in selected_pages),
        "full_context_tokens": sum(page["tokens"] for page in pages),
        "selected_page_count": len(selected_pages),
        "full_page_count": len(pages),
    }


def risk_bucket(row: dict) -> str:
    if row["support_coverage"] < 0.95:
        return "support_below_095"
    if not row["perfect_support"]:
        return "imperfect_support"
    if row["token_reduction_pct"] < 35:
        return "low_token_reduction"
    if row["token_reduction_pct"] >= 55:
        return "high_token_reduction"
    return "ordinary_pass"


def choose_cases(source_rows: list[dict]) -> list[dict]:
    decorated = [{**row, "risk_bucket": risk_bucket(row)} for row in source_rows]
    selected = []

    def take(bucket: str, count: int) -> None:
        already = {row["index"] for row in selected}
        pool = [row for row in decorated if row["risk_bucket"] == bucket and row["index"] not in already]
        pool = sorted(pool, key=lambda row: (row["token_reduction_pct"], row["index"]))
        selected.extend(pool[:count])

    take("support_below_095", 4)
    take("imperfect_support", 8)
    take("low_token_reduction", 6)
    take("high_token_reduction", 6)
    take("ordinary_pass", TARGET_CASES - len(selected))
    if len(selected) < TARGET_CASES:
        already = {row["index"] for row in selected}
        fill = [row for row in decorated if row["index"] not in already]
        selected.extend(sorted(fill, key=lambda row: row["index"])[: TARGET_CASES - len(selected)])
    return selected[:TARGET_CASES]


def build_user_prompt(question: str, context: str) -> str:
    return (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )


def build_request(case: dict, source: str, context: str) -> dict:
    user_prompt = build_user_prompt(case["question"], context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return {
        "case_id": f"hotpot-{case['index']}-{source}",
        "source_context": source,
        "index": case["index"],
        "question": case["question"],
        "gold_answer": case["gold_answer"],
        "risk_bucket": case["risk_bucket"],
        "token_reduction_pct": case["token_reduction_pct"],
        "support_coverage": case["support_coverage"],
        "perfect_support": case["perfect_support"],
        "messages": messages,
        "estimated_input_tokens": approx_tokens(SYSTEM_PROMPT) + approx_tokens(user_prompt),
        "estimated_output_tokens": 48,
    }


def write_jsonl(items: list[dict]) -> None:
    with REQUESTS.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def call_openai_chat(item: dict, model: str) -> dict:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is required for --live")
    body = json.dumps(
        {
            "model": model,
            "messages": item["messages"],
            "temperature": 0,
            "max_tokens": item["estimated_output_tokens"],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - started) * 1000
    usage = data.get("usage") or {}
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    return {
        "answer": content,
        "latency_ms": round(latency_ms, 3),
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
    }


def summarize_live(items: list[dict]) -> dict:
    generated = [item for item in items if "generated_answer" in item]
    by_index: dict[int, dict[str, dict]] = {}
    for item in generated:
        by_index.setdefault(item["index"], {})[item["source_context"]] = item
    pairs = [pair for pair in by_index.values() if "selected" in pair and "full" in pair]
    selected_matches = [pair["selected"]["exact_match"] for pair in pairs]
    full_matches = [pair["full"]["exact_match"] for pair in pairs]
    selected_latencies = [pair["selected"]["latency_ms"] for pair in pairs]
    full_latencies = [pair["full"]["latency_ms"] for pair in pairs]
    total_prompt = sum(item.get("usage", {}).get("prompt_tokens", 0) for item in generated)
    total_completion = sum(item.get("usage", {}).get("completion_tokens", 0) for item in generated)
    return {
        "generated_request_count": len(generated),
        "paired_case_count": len(pairs),
        "selected_exact_match_rate": round(sum(selected_matches) / len(pairs), 4) if pairs else 0.0,
        "full_exact_match_rate": round(sum(full_matches) / len(pairs), 4) if pairs else 0.0,
        "selected_minus_full_exact_match_rate": round(
            sum(selected_matches) / len(pairs) - sum(full_matches) / len(pairs),
            4,
        )
        if pairs
        else 0.0,
        "mean_selected_latency_ms": round(mean(selected_latencies), 3) if selected_latencies else 0.0,
        "mean_full_latency_ms": round(mean(full_latencies), 3) if full_latencies else 0.0,
        "latency_reduction_pct": round((1 - mean(selected_latencies) / mean(full_latencies)) * 100, 3)
        if selected_latencies and full_latencies and mean(full_latencies)
        else 0.0,
        "actual_tokens": {
            "prompt": total_prompt,
            "completion": total_completion,
            "total": total_prompt + total_completion,
        },
        "estimated_cost_usd_from_usage": round(estimate_cost(total_prompt, total_completion), 6),
        "mean_token_reduction_pct": round(
            mean(pair["selected"]["token_reduction_pct"] for pair in pairs),
            3,
        )
        if pairs
        else 0.0,
    }


def run_live(items: list[dict], model: str, budget_usd: float) -> list[dict]:
    estimated_cost = estimate_cost(
        sum(item["estimated_input_tokens"] for item in items),
        sum(item["estimated_output_tokens"] for item in items),
    )
    if estimated_cost > budget_usd:
        raise SystemExit(
            f"Estimated cost ${estimated_cost:.4f} exceeds budget ${budget_usd:.4f}; refusing live run."
        )
    live_items = []
    actual_prompt = 0
    actual_completion = 0
    for number, item in enumerate(items, start=1):
        projected_cost = estimate_cost(
            actual_prompt + item["estimated_input_tokens"],
            actual_completion + item["estimated_output_tokens"],
        )
        if projected_cost > budget_usd:
            break
        result = call_openai_chat(item, model)
        generated = dict(item)
        generated["live_request_number"] = number
        generated["generation_model"] = model
        generated["generated_answer"] = result["answer"]
        generated["exact_match"] = answer_match(item["gold_answer"], result["answer"])
        generated["latency_ms"] = result["latency_ms"]
        generated["usage"] = result["usage"]
        live_items.append(generated)
        actual_prompt += result["usage"]["prompt_tokens"]
        actual_completion += result["usage"]["completion_tokens"]
        if estimate_cost(actual_prompt, actual_completion) > budget_usd:
            break
    return live_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Call OpenAI API. Default is dry-run only.")
    parser.add_argument("--model", default=os.getenv("OPENAI_GENERATION_MODEL", DEFAULT_MODEL))
    parser.add_argument("--budget-usd", type=float, default=0.05)
    args = parser.parse_args()

    hotpot_rows = json.loads(HOTPOT_DATASET.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_rows = source["rows"]
    cases = choose_cases(source_rows)
    requests = []
    enriched_cases = []
    for case in cases:
        dataset_row = hotpot_rows[case["index"]]
        contexts = selected_and_full_context(dataset_row)
        enriched = {**case, **contexts}
        enriched_cases.append(enriched)
        requests.append(build_request(enriched, "selected", contexts["selected_context"]))
        requests.append(build_request(enriched, "full", contexts["full_context"]))

    write_jsonl(requests)

    estimated_input = sum(item["estimated_input_tokens"] for item in requests)
    estimated_output = sum(item["estimated_output_tokens"] for item in requests)
    by_bucket = Counter(case["risk_bucket"] for case in enriched_cases)
    token_reductions = [case["token_reduction_pct"] for case in enriched_cases]
    payload = {
        "experiment": "Research Task #1264 Hotpot-Local Semantic Generation Package",
        "hypothesis": (
            "The #1262 selected context preserves generated answer quality versus "
            "full context on a diagnostic HotpotQA-local sample."
        ),
        "mode": "live" if args.live else "dry_run",
        "generation_model": args.model,
        "source": str(SOURCE),
        "dataset": str(HOTPOT_DATASET),
        "request_file": str(REQUESTS),
        "case_count": len(enriched_cases),
        "request_count": len(requests),
        "sample_policy": {
            "target_cases": TARGET_CASES,
            "by_risk_bucket": dict(by_bucket),
        },
        "estimated_tokens": {
            "input": estimated_input,
            "output": estimated_output,
            "total": estimated_input + estimated_output,
        },
        "estimated_cost_usd": round(estimate_cost(estimated_input, estimated_output), 6),
        "budget_usd": args.budget_usd,
        "sample_summary": {
            "mean_token_reduction_pct": round(mean(token_reductions), 3) if token_reductions else 0.0,
            "min_token_reduction_pct": round(min(token_reductions), 3) if token_reductions else 0.0,
            "max_token_reduction_pct": round(max(token_reductions), 3) if token_reductions else 0.0,
            "mean_selected_context_tokens": round(mean(case["selected_context_tokens"] for case in enriched_cases), 3)
            if enriched_cases
            else 0.0,
            "mean_full_context_tokens": round(mean(case["full_context_tokens"] for case in enriched_cases), 3)
            if enriched_cases
            else 0.0,
        },
        "approval_required_for_live": True,
        "success": False,
        "claim_decision": "semantic_generation_package_prepared_not_run",
        "limitations": [
            "Dry-run only; no semantic evidence yet.",
            "Requires generation outputs before the judge phase can run.",
            "Diagnostic sample, not a full 123-case evaluation.",
            "No direct compressor baseline is included.",
        ],
        "cases": enriched_cases,
        "requests": requests,
    }
    if args.live:
        live_items = run_live(requests, args.model, args.budget_usd)
        payload["requests"] = live_items
        payload["summary"] = summarize_live(live_items)
        summary = payload["summary"]
        payload["success"] = (
            summary["paired_case_count"] >= 24
            and summary["selected_minus_full_exact_match_rate"] >= -0.10
            and summary["mean_token_reduction_pct"] >= 30.0
            and summary["estimated_cost_usd_from_usage"] <= args.budget_usd
        )
        payload["claim_decision"] = (
            "live_generation_canary_supports_hotpot_selected_context"
            if payload["success"]
            else "live_generation_canary_weakens_or_insufficient_for_hotpot_selected_context"
        )
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = dict(payload)
    compact.pop("cases")
    compact.pop("requests")
    compact.pop("limitations")
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
