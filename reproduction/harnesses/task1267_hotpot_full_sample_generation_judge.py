"""Research Task #1267: Hotpot full-sample live generation + blinded judge scale-up.

Question:
    Does the #1264/#1266 diagnostic result (selected context preserves
    generated-answer semantic quality at ~parity while cutting input cost)
    hold on ALL 123 #1262 accepted Hotpot-or cases, not just the 32-case
    diagnostic sample?

Modes:
    (default)            dry-run: build requests, count reusable #1264 live
                         responses, estimate remaining live cost. No API calls.
    --live               generate the missing selected/full answers with
                         gpt-4.1-mini (temperature 0). Requires user approval,
                         OPENAI_API_KEY, and a budget ceiling. Responses for
                         cases already generated live in #1264 with byte-identical
                         prompts are reused, not re-bought.
    --import-verdicts F  score blinded judge verdicts (same schema as #1266),
                         unblind, and apply the pre-registered gates.

Pre-registered gates (fixed before judging):
    paired_case_count >= 120  (allow up to 3 API failures out of 123)
    selected_semantic_correct_rate >= full_semantic_correct_rate - 0.05
    selected_cost_reduction_vs_full >= 30% (from actual usage)

Blinding: per case A/B = sha256("task1267-blind:{index}") % 2;
0 -> A=selected, 1 -> A=full. The judge request file contains no
selected/full labels.
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
SOURCE_1262 = ROOT / "task1262_frozen_hotpot_or_top10_second_holdout_results.json"
REUSE_1264 = (
    ROOT / "task1264_hotpot_local_semantic_generation_pack_results.live_backup_20260715.json"
)
GEN_OUTPUT = ROOT / "task1267_hotpot_full_sample_generation_results.json"
JUDGE_REQUESTS = ROOT / "task1267_hotpot_full_sample_judge_requests.jsonl"
OUTPUT = ROOT / "task1267_hotpot_full_sample_semantic_judge_results.json"

MIN_PAIRED_CASES = 120
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


def load_task1264_module():
    spec = importlib.util.spec_from_file_location(
        "task1264",
        Path(__file__).parent / "task1264_hotpot_local_semantic_generation_pack.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T1264 = load_task1264_module()


def sha_flip(index: int) -> int:
    digest = hashlib.sha256(f"task1267-blind:{index}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 2


def build_all_requests() -> tuple[list[dict], list[dict]]:
    hotpot_rows = json.loads(T1264.HOTPOT_DATASET.read_text(encoding="utf-8"))
    source_rows = json.loads(SOURCE_1262.read_text(encoding="utf-8"))["rows"]
    cases = sorted(
        ({**row, "risk_bucket": T1264.risk_bucket(row)} for row in source_rows),
        key=lambda row: row["index"],
    )
    requests = []
    enriched_cases = []
    for case in cases:
        dataset_row = hotpot_rows[case["index"]]
        contexts = T1264.selected_and_full_context(dataset_row)
        enriched = {**case, **contexts}
        enriched_cases.append(enriched)
        requests.append(T1264.build_request(enriched, "selected", contexts["selected_context"]))
        requests.append(T1264.build_request(enriched, "full", contexts["full_context"]))
    return enriched_cases, requests


def load_reusable() -> dict[tuple[int, str], dict]:
    if not REUSE_1264.exists():
        return {}
    data = json.loads(REUSE_1264.read_text(encoding="utf-8"))
    if data.get("mode") != "live":
        return {}
    reusable = {}
    for record in data["requests"]:
        if "generated_answer" in record:
            reusable[(record["index"], record["source_context"])] = record
    return reusable


def attach_reused(requests: list[dict], reusable: dict[tuple[int, str], dict]) -> tuple[list[dict], list[dict]]:
    done, pending = [], []
    for item in requests:
        cached = reusable.get((item["index"], item["source_context"]))
        if cached is not None and cached["messages"] == item["messages"]:
            merged = dict(item)
            merged["generation_model"] = cached["generation_model"]
            merged["generated_answer"] = cached["generated_answer"]
            merged["exact_match"] = cached["exact_match"]
            merged["latency_ms"] = cached["latency_ms"]
            merged["usage"] = cached["usage"]
            merged["reused_from_task1264"] = True
            done.append(merged)
        else:
            pending.append(item)
    return done, pending


def run_live_pending(pending: list[dict], model: str, budget_usd: float) -> tuple[list[dict], list[dict]]:
    generated = []
    failures = []
    actual_prompt = 0
    actual_completion = 0
    for number, item in enumerate(pending, start=1):
        projected = T1264.estimate_cost(
            actual_prompt + item["estimated_input_tokens"],
            actual_completion + item["estimated_output_tokens"],
        )
        if projected > budget_usd:
            print(f"Budget ceiling ${budget_usd} would be exceeded; stopping at request {number}.")
            break
        try:
            result = T1264.call_openai_chat(item, model)
        except Exception as error:  # noqa: BLE001 - record and continue
            failures.append({"case_id": item["case_id"], "error": str(error)})
            continue
        merged = dict(item)
        merged["live_request_number"] = number
        merged["generation_model"] = model
        merged["generated_answer"] = result["answer"]
        merged["exact_match"] = T1264.answer_match(item["gold_answer"], result["answer"])
        merged["latency_ms"] = result["latency_ms"]
        merged["usage"] = result["usage"]
        merged["reused_from_task1264"] = False
        generated.append(merged)
        actual_prompt += result["usage"]["prompt_tokens"]
        actual_completion += result["usage"]["completion_tokens"]
    spent = T1264.estimate_cost(actual_prompt, actual_completion)
    print(f"New live spend from usage: ${spent:.6f} ({len(generated)} requests, {len(failures)} failures)")
    return generated, failures


def build_judge_requests(records: list[dict]) -> list[dict]:
    by_index: dict[int, dict[str, dict]] = {}
    for record in records:
        by_index.setdefault(record["index"], {})[record["source_context"]] = record
    judge_items = []
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
        judge_items.append(
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
    return judge_items


def generation_cost_split(records: list[dict]) -> dict:
    selected_cost = 0.0
    full_cost = 0.0
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


def score_verdicts(gen_payload: dict, verdicts: dict[int, dict], judge_label: str) -> dict:
    records = gen_payload["requests"]
    by_index: dict[int, dict[str, dict]] = {}
    for record in records:
        by_index.setdefault(record["index"], {})[record["source_context"]] = record
    judged = []
    for index in sorted(by_index):
        pair = by_index[index]
        verdict = verdicts.get(index)
        if verdict is None or "selected" not in pair or "full" not in pair:
            continue
        unblinded = unblind(index, verdict)
        judged.append(
            {
                "index": index,
                "question": pair["selected"]["question"],
                "gold_answer": pair["selected"]["gold_answer"],
                "risk_bucket": pair["selected"]["risk_bucket"],
                "selected_exact_match": pair["selected"]["exact_match"],
                "full_exact_match": pair["full"]["exact_match"],
                "reused_from_task1264": pair["selected"].get("reused_from_task1264", False),
                "verdict_raw": {k: v for k, v in verdict.items() if not k.startswith("pass_")},
                **unblinded,
            }
        )
    n = len(judged)
    selected_rate = mean(1.0 if r["selected_correct"] else 0.0 for r in judged) if judged else 0.0
    full_rate = mean(1.0 if r["full_correct"] else 0.0 for r in judged) if judged else 0.0
    better_counts = {"selected": 0, "full": 0, "tie": 0, "neither": 0}
    for row in judged:
        better_counts[row["better_source"]] = better_counts.get(row["better_source"], 0) + 1
    cost = generation_cost_split(records)
    delta = round(selected_rate - full_rate, 4)
    gates = {
        "paired_case_count_gate": n >= MIN_PAIRED_CASES,
        "semantic_delta_gate": delta >= SEMANTIC_DELTA_FLOOR,
        "cost_reduction_gate": cost["selected_cost_reduction_vs_full_pct"] >= COST_REDUCTION_FLOOR_PCT,
    }
    success = all(gates.values())
    disagreement = {
        "selected_exact_but_semantic_wrong": sum(
            1 for r in judged if r["selected_exact_match"] and not r["selected_correct"]
        ),
        "selected_semantic_but_exact_miss": sum(
            1 for r in judged if r["selected_correct"] and not r["selected_exact_match"]
        ),
        "full_exact_but_semantic_wrong": sum(
            1 for r in judged if r["full_exact_match"] and not r["full_correct"]
        ),
        "full_semantic_but_exact_miss": sum(
            1 for r in judged if r["full_correct"] and not r["full_exact_match"]
        ),
    }
    subset_rates = {}
    for label, rows in (
        ("reused_32_diagnostic", [r for r in judged if r["reused_from_task1264"]]),
        ("new_91_cases", [r for r in judged if not r["reused_from_task1264"]]),
    ):
        if rows:
            subset_rates[label] = {
                "n": len(rows),
                "selected_semantic_correct_rate": round(
                    mean(1.0 if r["selected_correct"] else 0.0 for r in rows), 4
                ),
                "full_semantic_correct_rate": round(
                    mean(1.0 if r["full_correct"] else 0.0 for r in rows), 4
                ),
            }
    return {
        "judge_model": judge_label,
        "paired_case_count": n,
        "selected_semantic_correct_rate": round(selected_rate, 4),
        "full_semantic_correct_rate": round(full_rate, 4),
        "selected_minus_full_semantic_rate": delta,
        "better_source_counts": better_counts,
        "exact_vs_semantic_disagreement": disagreement,
        "subset_rates": subset_rates,
        "generation_cost": cost,
        "gates": gates,
        "success": success,
        "claim_decision": (
            "full_sample_blinded_judge_supports_hotpot_selected_context"
            if success
            else "full_sample_blinded_judge_rejects_or_incomplete"
        ),
        "judged_cases": judged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=os.getenv("OPENAI_GENERATION_MODEL", T1264.DEFAULT_MODEL))
    parser.add_argument("--budget-usd", type=float, default=0.10)
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
            "experiment": "Research Task #1267 Hotpot Full-Sample Generation + Blinded Semantic Judge",
            "mode": "imported-verdicts",
            "generation_results": str(GEN_OUTPUT),
            "blinding": "sha256('task1267-blind:{index}') % 2; 0 -> A=selected, 1 -> A=full",
            "gates": {
                "min_paired_cases": MIN_PAIRED_CASES,
                "semantic_delta_floor": SEMANTIC_DELTA_FLOOR,
                "cost_reduction_floor_pct": COST_REDUCTION_FLOOR_PCT,
            },
            "summary": summary,
            "success": summary["success"],
            "claim_decision": summary["claim_decision"],
        }
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        view = {k: v for k, v in summary.items() if k != "judged_cases"}
        print(json.dumps(view, ensure_ascii=False, indent=2))
        return

    enriched_cases, requests = build_all_requests()
    reusable = load_reusable()
    done, pending = attach_reused(requests, reusable)
    pending_input = sum(item["estimated_input_tokens"] for item in pending)
    pending_output = sum(item["estimated_output_tokens"] for item in pending)
    estimated_pending_cost = round(T1264.estimate_cost(pending_input, pending_output), 6)
    print(
        json.dumps(
            {
                "case_count": len(enriched_cases),
                "request_count": len(requests),
                "reused_from_task1264": len(done),
                "pending_live_requests": len(pending),
                "estimated_pending_cost_usd": estimated_pending_cost,
                "budget_usd": args.budget_usd,
            },
            indent=2,
        )
    )

    failures: list[dict] = []
    if args.live:
        if os.getenv("TASK1267_LIVE_APPROVED", "") != "yes":
            raise SystemExit(
                "Live generation requires explicit user approval. Set TASK1267_LIVE_APPROVED=yes."
            )
        if estimated_pending_cost > args.budget_usd:
            raise SystemExit(
                f"Estimated pending cost ${estimated_pending_cost} exceeds budget ${args.budget_usd}."
            )
        new_records, failures = run_live_pending(pending, args.model, args.budget_usd)
        all_records = done + new_records
    else:
        all_records = done

    generated = [r for r in all_records if "generated_answer" in r]
    judge_items = build_judge_requests(generated)
    with JUDGE_REQUESTS.open("w", encoding="utf-8") as handle:
        for item in judge_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    payload = {
        "experiment": "Research Task #1267 Hotpot Full-Sample Generation + Blinded Semantic Judge",
        "mode": "live" if args.live else "dry_run",
        "generation_model": args.model,
        "source": str(SOURCE_1262),
        "reuse_source": str(REUSE_1264),
        "judge_request_file": str(JUDGE_REQUESTS),
        "case_count": len(enriched_cases),
        "request_count": len(requests),
        "reused_request_count": len(done),
        "new_live_request_count": len(generated) - len(done) if args.live else 0,
        "failures": failures,
        "estimated_pending_cost_usd": estimated_pending_cost,
        "budget_usd": args.budget_usd,
        "paired_judge_case_count": len(judge_items),
        "summary": T1264.summarize_live(generated) if generated else {},
        "generation_cost": generation_cost_split(generated) if generated else {},
        "requests": generated,
    }
    GEN_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    view = {k: v for k, v in payload.items() if k != "requests"}
    print(json.dumps(view, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
