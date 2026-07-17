"""Research Task #1266: Hotpot live generation blinded semantic judge.

Question:
    Does the #1264 live exact-match result (selected context preserves
    generated-answer quality) survive a blinded semantic judge?

Default mode is dry-run only. It writes blinded A/B judge requests from the
#1264 live generated answers without calling any external API. The
selected/full assignment is deterministically randomized per case and is NOT
present in the request file, so any judge that only reads the request file is
blind to which answer came from selected context.

Verdict import mode (--import-verdicts) consumes a verdicts JSONL produced by
any judge (external API or local blinded judge agents), unblinds the
assignments, and applies the pre-registered gates:

    paired_case_count >= 32
    selected_semantic_correct_rate >= full_semantic_correct_rate - 0.05
    selected_cost_reduction_vs_full >= 30 (from #1264 live usage)

Live OpenAI judge mode requires explicit user approval, OPENAI_API_KEY, and a
budget ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "task1264_hotpot_local_semantic_generation_pack_results.json"
SOURCE_BACKUP = (
    ROOT / "task1264_hotpot_local_semantic_generation_pack_results.live_backup_20260715.json"
)
REQUESTS = ROOT / "task1266_hotpot_live_generation_semantic_judge_requests.jsonl"
OUTPUT = ROOT / "task1266_hotpot_live_generation_semantic_judge_results.json"
DEFAULT_MODEL = "gpt-4.1-mini"
INPUT_PRICE_PER_1M = 0.40
OUTPUT_PRICE_PER_1M = 1.60
GENERATION_INPUT_PRICE_PER_1M = 0.40
GENERATION_OUTPUT_PRICE_PER_1M = 1.60
JUDGE_OUTPUT_TOKENS_PER_CASE = 96

MIN_PAIRED_CASES = 32
SEMANTIC_DELTA_FLOOR = -0.05
COST_REDUCTION_FLOOR_PCT = 30.0

JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair answer grader. You will see a question, the "
    "gold answer, and two candidate answers labeled A and B. Judge each "
    "candidate on whether it conveys the same answer as the gold answer. "
    "Wording differences, extra explanation, and formatting do not matter; "
    "the conveyed answer entity or choice must match. Return only JSON with "
    'keys: a_correct (boolean), b_correct (boolean), better ("A" | "B" | '
    '"tie" | "neither"), rationale (short string).'
)


def sha_flip(index: int) -> int:
    """Deterministic pre-registered blinding coin: 0 -> A=selected, 1 -> A=full."""
    digest = hashlib.sha256(f"task1266-blind:{index}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 2


def approx_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens * INPUT_PRICE_PER_1M / 1_000_000
        + output_tokens * OUTPUT_PRICE_PER_1M / 1_000_000,
        6,
    )


def load_source() -> dict:
    path = SOURCE
    data = json.loads(path.read_text())
    if data.get("mode") != "live" and SOURCE_BACKUP.exists():
        path = SOURCE_BACKUP
        data = json.loads(path.read_text())
    if data.get("mode") != "live":
        raise SystemExit(
            "No live #1264 results found. The judge needs live generated answers."
        )
    data["_loaded_from"] = str(path)
    return data


def pair_cases(source: dict) -> list[dict]:
    by_index: dict[int, dict] = {}
    for request in source["requests"]:
        entry = by_index.setdefault(
            request["index"],
            {
                "index": request["index"],
                "question": request["question"],
                "gold_answer": request["gold_answer"],
                "risk_bucket": request["risk_bucket"],
                "token_reduction_pct": request["token_reduction_pct"],
            },
        )
        entry[request["source_context"]] = {
            "generated_answer": request["generated_answer"],
            "exact_match": request["exact_match"],
            "usage": request["usage"],
        }
    cases = [
        entry
        for entry in by_index.values()
        if "selected" in entry and "full" in entry
    ]
    cases.sort(key=lambda entry: entry["index"])
    return cases


def build_judge_prompt(question: str, gold: str, answer_a: str, answer_b: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n\n"
        "Return only the JSON object."
    )


def build_requests(cases: list[dict]) -> list[dict]:
    requests = []
    for case in cases:
        flip = sha_flip(case["index"])
        answers = (
            (case["selected"]["generated_answer"], case["full"]["generated_answer"])
            if flip == 0
            else (case["full"]["generated_answer"], case["selected"]["generated_answer"])
        )
        prompt = build_judge_prompt(
            case["question"], case["gold_answer"], answers[0], answers[1]
        )
        requests.append(
            {
                "case_index": case["index"],
                "question": case["question"],
                "gold_answer": case["gold_answer"],
                "answer_a": answers[0],
                "answer_b": answers[1],
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "estimated_input_tokens": approx_tokens(JUDGE_SYSTEM_PROMPT + prompt),
                "estimated_output_tokens": JUDGE_OUTPUT_TOKENS_PER_CASE,
            }
        )
    return requests


def write_jsonl(items: list[dict]) -> None:
    with REQUESTS.open("w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def generation_cost_reduction_pct(cases: list[dict]) -> dict:
    selected_cost = 0.0
    full_cost = 0.0
    for case in cases:
        for source_name, bucket in (("selected", case["selected"]), ("full", case["full"])):
            usage = bucket["usage"]
            cost = (
                usage["prompt_tokens"] * GENERATION_INPUT_PRICE_PER_1M / 1_000_000
                + usage["completion_tokens"] * GENERATION_OUTPUT_PRICE_PER_1M / 1_000_000
            )
            if source_name == "selected":
                selected_cost += cost
            else:
                full_cost += cost
    reduction = 100.0 * (1.0 - selected_cost / full_cost) if full_cost else 0.0
    return {
        "selected_estimated_cost_usd": round(selected_cost, 6),
        "full_estimated_cost_usd": round(full_cost, 6),
        "selected_cost_reduction_vs_full_pct": round(reduction, 3),
    }


def unblind_verdict(case_index: int, verdict: dict) -> dict:
    flip = sha_flip(case_index)
    a_is_selected = flip == 0
    selected_correct = verdict["a_correct"] if a_is_selected else verdict["b_correct"]
    full_correct = verdict["b_correct"] if a_is_selected else verdict["a_correct"]
    better = verdict.get("better", "tie")
    if better == "A":
        better_source = "selected" if a_is_selected else "full"
    elif better == "B":
        better_source = "full" if a_is_selected else "selected"
    else:
        better_source = better
    return {
        "selected_correct": bool(selected_correct),
        "full_correct": bool(full_correct),
        "better_source": better_source,
    }


def summarize_verdicts(
    cases: list[dict], verdicts: dict[int, dict], judge_model: str
) -> dict:
    judged = []
    for case in cases:
        verdict = verdicts.get(case["index"])
        if verdict is None:
            continue
        unblinded = unblind_verdict(case["index"], verdict)
        judged.append(
            {
                "index": case["index"],
                "question": case["question"],
                "gold_answer": case["gold_answer"],
                "risk_bucket": case["risk_bucket"],
                "selected_exact_match": case["selected"]["exact_match"],
                "full_exact_match": case["full"]["exact_match"],
                "verdict_raw": verdict,
                **unblinded,
            }
        )
    paired_case_count = len(judged)
    selected_rate = (
        mean(1.0 if row["selected_correct"] else 0.0 for row in judged)
        if judged
        else 0.0
    )
    full_rate = (
        mean(1.0 if row["full_correct"] else 0.0 for row in judged) if judged else 0.0
    )
    better_counts = {"selected": 0, "full": 0, "tie": 0, "neither": 0}
    for row in judged:
        better_counts[row["better_source"]] = better_counts.get(row["better_source"], 0) + 1

    exact_vs_semantic = {
        "selected_exact_but_semantic_wrong": sum(
            1 for row in judged if row["selected_exact_match"] and not row["selected_correct"]
        ),
        "selected_semantic_but_exact_miss": sum(
            1 for row in judged if row["selected_correct"] and not row["selected_exact_match"]
        ),
        "full_exact_but_semantic_wrong": sum(
            1 for row in judged if row["full_exact_match"] and not row["full_correct"]
        ),
        "full_semantic_but_exact_miss": sum(
            1 for row in judged if row["full_correct"] and not row["full_exact_match"]
        ),
    }

    cost = generation_cost_reduction_pct(cases)
    delta = round(selected_rate - full_rate, 4)
    gates = {
        "paired_case_count_gate": paired_case_count >= MIN_PAIRED_CASES,
        "semantic_delta_gate": delta >= SEMANTIC_DELTA_FLOOR,
        "cost_reduction_gate": cost["selected_cost_reduction_vs_full_pct"]
        >= COST_REDUCTION_FLOOR_PCT,
    }
    success = all(gates.values())
    return {
        "judge_model": judge_model,
        "paired_case_count": paired_case_count,
        "selected_semantic_correct_rate": round(selected_rate, 4),
        "full_semantic_correct_rate": round(full_rate, 4),
        "selected_minus_full_semantic_rate": delta,
        "better_source_counts": better_counts,
        "exact_vs_semantic_disagreement": exact_vs_semantic,
        "generation_cost": cost,
        "gates": gates,
        "success": success,
        "claim_decision": (
            "blinded_semantic_judge_supports_hotpot_selected_context"
            if success
            else "blinded_semantic_judge_rejects_or_incomplete"
        ),
        "judged_cases": judged,
    }


def call_openai_chat(item: dict, model: str) -> dict:
    api_key = os.environ["OPENAI_API_KEY"]
    payload = {
        "model": model,
        "messages": item["messages"],
        "temperature": 0,
        "max_tokens": JUDGE_OUTPUT_TOKENS_PER_CASE * 2,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    verdict = json.loads(body["choices"][0]["message"]["content"])
    verdict["_usage"] = body.get("usage", {})
    verdict["_latency_ms"] = round((time.time() - started) * 1000, 2)
    return verdict


def run_live(requests: list[dict], model: str, budget_usd: float) -> dict[int, dict]:
    spent = 0.0
    verdicts: dict[int, dict] = {}
    for item in requests:
        estimated = estimate_cost(
            item["estimated_input_tokens"], item["estimated_output_tokens"]
        )
        if spent + estimated > budget_usd:
            print(f"Budget ceiling reached at case {item['case_index']}; stopping.")
            break
        verdict = call_openai_chat(item, model)
        usage = verdict.get("_usage", {})
        spent += estimate_cost(
            usage.get("prompt_tokens", item["estimated_input_tokens"]),
            usage.get("completion_tokens", item["estimated_output_tokens"]),
        )
        verdicts[item["case_index"]] = verdict
    print(f"Live judge spend estimate: ${spent:.6f}")
    return verdicts


def load_verdicts_file(path: Path) -> dict[int, dict]:
    verdicts: dict[int, dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            verdicts[int(row["case_index"])] = row
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true", help="Call OpenAI judge. Default is dry-run only."
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--budget-usd", type=float, default=0.03)
    parser.add_argument(
        "--import-verdicts",
        type=Path,
        default=None,
        help="JSONL of blinded verdicts to unblind and score instead of calling an API.",
    )
    parser.add_argument(
        "--judge-label",
        default=None,
        help="Judge model label recorded with imported verdicts.",
    )
    args = parser.parse_args()

    source = load_source()
    cases = pair_cases(source)
    requests = build_requests(cases)
    write_jsonl(requests)

    estimated_input = sum(item["estimated_input_tokens"] for item in requests)
    estimated_output = sum(item["estimated_output_tokens"] for item in requests)
    result = {
        "experiment": "Research Task #1266 Hotpot Live Generation Blinded Semantic Judge",
        "hypothesis": (
            "The #1264 live exact-match result survives blinded semantic judging: "
            "selected-context answers are semantically correct at least as often as "
            "full-context answers minus 0.05."
        ),
        "mode": "dry-run",
        "source": source["_loaded_from"],
        "request_file": str(REQUESTS),
        "blinding": {
            "scheme": "sha256('task1266-blind:{index}') % 2; 0 -> A=selected, 1 -> A=full",
            "note": "Assignment is absent from the request file; judges reading only "
            "the request file are blind to selected/full.",
        },
        "case_count": len(cases),
        "estimated_tokens": {
            "input": estimated_input,
            "output": estimated_output,
            "total": estimated_input + estimated_output,
        },
        "estimated_cost_usd": estimate_cost(estimated_input, estimated_output),
        "gates": {
            "min_paired_cases": MIN_PAIRED_CASES,
            "semantic_delta_floor": SEMANTIC_DELTA_FLOOR,
            "cost_reduction_floor_pct": COST_REDUCTION_FLOOR_PCT,
        },
        "approval_required_for_live": True,
        "success": False,
        "claim_decision": "dry_run_only_no_semantic_verdicts",
    }

    verdicts: dict[int, dict] | None = None
    judge_label = args.judge_label or args.model
    if args.import_verdicts is not None:
        verdicts = load_verdicts_file(args.import_verdicts)
        result["mode"] = "imported-verdicts"
    elif args.live:
        approval = os.getenv("TASK1266_LIVE_APPROVED", "")
        if approval != "yes":
            raise SystemExit(
                "Live judge requires explicit user approval. Set TASK1266_LIVE_APPROVED=yes."
            )
        verdicts = run_live(requests, args.model, args.budget_usd)
        result["mode"] = "live"

    if verdicts is not None:
        summary = summarize_verdicts(cases, verdicts, judge_label)
        result["summary"] = summary
        result["success"] = summary["success"]
        result["claim_decision"] = summary["claim_decision"]

    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2)[:1200])
    if "summary" in result:
        summary_view = {
            k: v for k, v in result["summary"].items() if k != "judged_cases"
        }
        print(json.dumps(summary_view, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
