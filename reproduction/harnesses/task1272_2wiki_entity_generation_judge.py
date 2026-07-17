"""Research Task #1272: 2Wiki entity regime generative canary + blinded judge.

Question:
    Does regime 2 (#1271: ent2_norel_noyn + sent_full, retention-verified at
    n=981) survive the generation layer, the way regime 1 needed #1264/#1266
    after #1262? Selected context here is far more aggressive (68.9% token
    reduction vs Hotpot's 45%), so retention safety does NOT automatically
    imply generative parity.

Recipe (identical to #1264/#1266, applied to regime 2):
    1. Stratified 32-case sample from the 981 frozen holdout rows.
    2. Live gpt-4.1-mini generation, temperature 0, selected vs full.
    3. Blinded A/B judge requests: sha256("task1272-blind:{index}") % 2;
       0 -> A=selected, 1 -> A=full. No labels in the request file.
    4. External blinded judging (Claude 3-pass panel), majority vote,
       verdicts imported via --import-verdicts.

Pre-registered gates:
    generation layer: paired_case_count >= 24,
                      selected_minus_full_exact_match >= -0.10,
                      mean_token_reduction_pct >= 30
    judge layer:      paired_case_count >= 30,
                      selected_semantic_correct_rate >=
                          full_semantic_correct_rate - 0.05,
                      selected_cost_reduction_vs_full >= 30%

Live mode requires explicit user approval (TASK1272_LIVE_APPROVED=yes),
OPENAI_API_KEY, and a budget ceiling (default $0.05).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/external/2wikimultihop/dev.json"
HOLDOUT = ROOT / "task1271_2wiki_entity_regime_frozen_holdout_results.json"
GEN_OUTPUT = ROOT / "task1272_2wiki_entity_generation_results.json"
JUDGE_REQUESTS = ROOT / "task1272_2wiki_entity_judge_requests.jsonl"
OUTPUT = ROOT / "task1272_2wiki_entity_semantic_judge_results.json"
TARGET_CASES = 32

GEN_GATES = {"min_paired": 24, "exact_delta_floor": -0.10, "min_token_reduction": 30.0}
JUDGE_GATES = {
    "min_paired": 30,
    "semantic_delta_floor": -0.05,
    "cost_reduction_floor_pct": 30.0,
}

JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair answer grader. You will see a question, the "
    "gold answer, and two candidate answers labeled A and B. Judge each "
    "candidate on whether it conveys the same answer as the gold answer. "
    "Wording differences, extra explanation, and formatting do not matter; "
    "the conveyed answer entity or choice must match. Return only JSON with "
    'keys: a_correct (boolean), b_correct (boolean), better ("A" | "B" | '
    '"tie" | "neither"), rationale (short string).'
)


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T1264 = load_module("task1264", "task1264_hotpot_local_semantic_generation_pack.py")
S1270 = load_module("task1270", "task1270_2wiki_entity_aware_selector_search.py")
T1263 = S1270.T


def sha_flip(index: int) -> int:
    digest = hashlib.sha256(f"task1272-blind:{index}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 2


def risk_bucket(row: dict) -> str:
    if row["answer_retained_given_full"] is False:
        return "retention_fail"
    if row["support_coverage"] < 1.0:
        return "imperfect_support"
    if row["token_reduction_pct"] < 60.0:
        return "low_reduction"
    if row["token_reduction_pct"] >= 80.0:
        return "high_reduction"
    return "ordinary_pass"


def choose_cases(rows: list[dict]) -> list[dict]:
    decorated = [{**row, "risk_bucket": risk_bucket(row)} for row in rows]
    chosen: list[dict] = []

    def take(bucket: str, count: int) -> None:
        already = {row["index"] for row in chosen}
        pool = sorted(
            (
                row
                for row in decorated
                if row["risk_bucket"] == bucket and row["index"] not in already
            ),
            key=lambda row: row["index"],
        )
        chosen.extend(pool[:count])

    take("retention_fail", 4)
    take("imperfect_support", 4)
    take("low_reduction", 6)
    take("high_reduction", 6)
    take("ordinary_pass", TARGET_CASES - len(chosen))
    if len(chosen) < TARGET_CASES:
        already = {row["index"] for row in chosen}
        filler = sorted(
            (row for row in decorated if row["index"] not in already),
            key=lambda row: row["index"],
        )
        chosen.extend(filler[: TARGET_CASES - len(chosen)])
    return chosen[:TARGET_CASES]


def build_contexts(dataset_row: dict) -> dict:
    titles = S1270.matched_titles(dataset_row)
    sentences = T1263.sentence_rows(dataset_row)
    selected_lines = [
        item["text"] for item in sentences if item["title"] in titles
    ]
    full_lines = [item["text"] for item in sentences]
    return {
        "selected_context": "\n".join(selected_lines)[: T1263.MAX_CONTEXT_CHARS],
        "full_context": "\n".join(full_lines)[: T1263.MAX_CONTEXT_CHARS],
    }


def build_request(case: dict, source: str, context: str) -> dict:
    user_prompt = T1264.build_user_prompt(case["question"], context)
    return {
        "case_id": f"2wiki-{case['index']}-{source}",
        "source_context": source,
        "index": case["index"],
        "question": case["question"],
        "gold_answer": case["gold_answer"],
        "risk_bucket": case["risk_bucket"],
        "token_reduction_pct": case["token_reduction_pct"],
        "support_coverage": case["support_coverage"],
        "messages": [
            {"role": "system", "content": T1264.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "estimated_input_tokens": T1264.approx_tokens(T1264.SYSTEM_PROMPT)
        + T1264.approx_tokens(user_prompt),
        "estimated_output_tokens": 48,
    }


def run_live(requests: list[dict], model: str, budget_usd: float) -> tuple[list[dict], list[dict]]:
    generated, failures = [], []
    prompt_tokens = completion_tokens = 0
    for number, item in enumerate(requests, start=1):
        projected = T1264.estimate_cost(
            prompt_tokens + item["estimated_input_tokens"],
            completion_tokens + item["estimated_output_tokens"],
        )
        if projected > budget_usd:
            print(f"Budget ceiling ${budget_usd} reached; stopping at request {number}.")
            break
        try:
            result = T1264.call_openai_chat(item, model)
        except Exception as error:  # noqa: BLE001
            failures.append({"case_id": item["case_id"], "error": str(error)})
            continue
        merged = dict(item)
        merged["live_request_number"] = number
        merged["generation_model"] = model
        merged["generated_answer"] = result["answer"]
        merged["exact_match"] = T1264.answer_match(item["gold_answer"], result["answer"])
        merged["latency_ms"] = result["latency_ms"]
        merged["usage"] = result["usage"]
        generated.append(merged)
        prompt_tokens += result["usage"]["prompt_tokens"]
        completion_tokens += result["usage"]["completion_tokens"]
    print(
        f"Live spend from usage: ${T1264.estimate_cost(prompt_tokens, completion_tokens):.6f} "
        f"({len(generated)} requests, {len(failures)} failures)"
    )
    return generated, failures


def build_judge_requests(records: list[dict]) -> list[dict]:
    by_index: dict[int, dict[str, dict]] = {}
    for record in records:
        by_index.setdefault(record["index"], {})[record["source_context"]] = record
    items = []
    for index in sorted(by_index):
        pair = by_index[index]
        if "selected" not in pair or "full" not in pair:
            continue
        flip = sha_flip(index)
        answers = (
            (pair["selected"]["generated_answer"], pair["full"]["generated_answer"])
            if flip == 0
            else (pair["full"]["generated_answer"], pair["selected"]["generated_answer"])
        )
        prompt = (
            f"Question:\n{pair['selected']['question']}\n\n"
            f"Gold answer:\n{pair['selected']['gold_answer']}\n\n"
            f"Answer A:\n{answers[0]}\n\n"
            f"Answer B:\n{answers[1]}\n\n"
            "Return only the JSON object."
        )
        items.append(
            {
                "case_index": index,
                "question": pair["selected"]["question"],
                "gold_answer": pair["selected"]["gold_answer"],
                "answer_a": answers[0],
                "answer_b": answers[1],
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
        )
    return items


def generation_cost_split(records: list[dict]) -> dict:
    selected_cost = full_cost = 0.0
    for record in records:
        usage = record["usage"]
        cost = T1264.estimate_cost(usage["prompt_tokens"], usage["completion_tokens"])
        if record["source_context"] == "selected":
            selected_cost += cost
        else:
            full_cost += cost
    reduction = 100.0 * (1.0 - selected_cost / full_cost) if full_cost else 0.0
    return {
        "selected_estimated_cost_usd": round(selected_cost, 6),
        "full_estimated_cost_usd": round(full_cost, 6),
        "selected_cost_reduction_vs_full_pct": round(reduction, 3),
    }


def unblind(case_index: int, verdict: dict) -> dict:
    a_is_selected = sha_flip(case_index) == 0
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


def score_verdicts(gen_payload: dict, verdicts: dict[int, dict], judge_label: str) -> dict:
    by_index: dict[int, dict[str, dict]] = {}
    for record in gen_payload["requests"]:
        by_index.setdefault(record["index"], {})[record["source_context"]] = record
    judged = []
    for index in sorted(by_index):
        pair = by_index[index]
        verdict = verdicts.get(index)
        if verdict is None or "selected" not in pair or "full" not in pair:
            continue
        judged.append(
            {
                "index": index,
                "question": pair["selected"]["question"],
                "gold_answer": pair["selected"]["gold_answer"],
                "risk_bucket": pair["selected"]["risk_bucket"],
                "selected_exact_match": pair["selected"]["exact_match"],
                "full_exact_match": pair["full"]["exact_match"],
                "verdict_raw": {k: v for k, v in verdict.items() if not k.startswith("pass_")},
                **unblind(index, verdict),
            }
        )
    n = len(judged)
    selected_rate = mean(1.0 if r["selected_correct"] else 0.0 for r in judged) if judged else 0.0
    full_rate = mean(1.0 if r["full_correct"] else 0.0 for r in judged) if judged else 0.0
    better_counts = {"selected": 0, "full": 0, "tie": 0, "neither": 0}
    for row in judged:
        better_counts[row["better_source"]] = better_counts.get(row["better_source"], 0) + 1
    cost = generation_cost_split(gen_payload["requests"])
    delta = round(selected_rate - full_rate, 4)
    gates = {
        "paired_case_count_gate": n >= JUDGE_GATES["min_paired"],
        "semantic_delta_gate": delta >= JUDGE_GATES["semantic_delta_floor"],
        "cost_reduction_gate": cost["selected_cost_reduction_vs_full_pct"]
        >= JUDGE_GATES["cost_reduction_floor_pct"],
    }
    success = all(gates.values())
    return {
        "judge_model": judge_label,
        "paired_case_count": n,
        "selected_semantic_correct_rate": round(selected_rate, 4),
        "full_semantic_correct_rate": round(full_rate, 4),
        "selected_minus_full_semantic_rate": delta,
        "better_source_counts": better_counts,
        "generation_cost": cost,
        "gates": gates,
        "success": success,
        "claim_decision": (
            "2wiki_entity_regime_blinded_judge_supports_selected_context"
            if success
            else "2wiki_entity_regime_blinded_judge_rejects_or_incomplete"
        ),
        "judged_cases": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=os.getenv("OPENAI_GENERATION_MODEL", T1264.DEFAULT_MODEL))
    parser.add_argument("--budget-usd", type=float, default=0.05)
    parser.add_argument("--import-verdicts", type=Path, default=None)
    parser.add_argument("--judge-label", default="unspecified-judge")
    args = parser.parse_args()

    if args.import_verdicts is not None:
        gen_payload = json.loads(GEN_OUTPUT.read_text(encoding="utf-8"))
        verdicts: dict[int, dict] = {}
        with args.import_verdicts.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    verdicts[int(row["case_index"])] = row
        summary = score_verdicts(gen_payload, verdicts, args.judge_label)
        result = {
            "experiment": "Research Task #1272 2Wiki Entity Regime Generative Canary + Blinded Judge",
            "mode": "imported-verdicts",
            "generation_results": str(GEN_OUTPUT),
            "blinding": "sha256('task1272-blind:{index}') % 2; 0 -> A=selected, 1 -> A=full",
            "gates": JUDGE_GATES,
            "summary": summary,
            "success": summary["success"],
            "claim_decision": summary["claim_decision"],
        }
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "judged_cases"}, ensure_ascii=False, indent=2))
        return

    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    dataset_rows = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = choose_cases(holdout["rows"])
    requests = []
    for case in cases:
        contexts = build_contexts(dataset_rows[case["index"]])
        requests.append(build_request(case, "selected", contexts["selected_context"]))
        requests.append(build_request(case, "full", contexts["full_context"]))

    estimated_input = sum(item["estimated_input_tokens"] for item in requests)
    estimated_output = sum(item["estimated_output_tokens"] for item in requests)
    estimated_cost = round(T1264.estimate_cost(estimated_input, estimated_output), 6)
    from collections import Counter

    bucket_counts = dict(Counter(case["risk_bucket"] for case in cases))
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "request_count": len(requests),
                "by_risk_bucket": bucket_counts,
                "estimated_cost_usd": estimated_cost,
                "budget_usd": args.budget_usd,
            },
            indent=2,
        )
    )

    failures: list[dict] = []
    generated: list[dict] = []
    if args.live:
        if os.getenv("TASK1272_LIVE_APPROVED", "") != "yes":
            raise SystemExit("Live generation requires TASK1272_LIVE_APPROVED=yes.")
        if estimated_cost > args.budget_usd:
            raise SystemExit(
                f"Estimated cost ${estimated_cost} exceeds budget ${args.budget_usd}."
            )
        generated, failures = run_live(requests, args.model, args.budget_usd)

    judge_items = build_judge_requests(generated)
    with JUDGE_REQUESTS.open("w", encoding="utf-8") as handle:
        for item in judge_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = T1264.summarize_live(generated) if generated else {}
    gen_success = bool(
        generated
        and summary["paired_case_count"] >= GEN_GATES["min_paired"]
        and summary["selected_minus_full_exact_match_rate"] >= GEN_GATES["exact_delta_floor"]
        and summary["mean_token_reduction_pct"] >= GEN_GATES["min_token_reduction"]
    )
    payload = {
        "experiment": "Research Task #1272 2Wiki Entity Regime Generative Canary + Blinded Judge",
        "mode": "live" if args.live else "dry_run",
        "generation_model": args.model,
        "source": str(HOLDOUT),
        "sample_policy": {"target_cases": TARGET_CASES, "by_risk_bucket": bucket_counts},
        "judge_request_file": str(JUDGE_REQUESTS),
        "case_count": len(cases),
        "request_count": len(requests),
        "failures": failures,
        "estimated_cost_usd": estimated_cost,
        "budget_usd": args.budget_usd,
        "gen_gates": GEN_GATES,
        "summary": summary,
        "generation_cost": generation_cost_split(generated) if generated else {},
        "success": gen_success,
        "claim_decision": (
            "2wiki_entity_live_generation_canary_supports_selected_context"
            if gen_success
            else ("2wiki_entity_live_generation_canary_fails_or_incomplete" if args.live else "dry_run_only")
        ),
        "requests": generated if generated else requests,
    }
    GEN_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "requests"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
