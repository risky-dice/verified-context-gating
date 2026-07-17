"""Research Task #1275: 2Wiki entity regime large-sample generation + blinded judge.

Question:
    Does the #1272 canary result (semantic parity at 61% cost reduction)
    hold at scale on the #1271 frozen holdout? Target: as many of the 981
    holdout pairs as the approved budget allows, in deterministic index
    order (the sample is therefore a pre-registered prefix, not cherry-picked).

Budget note (recorded for honesty): the full 981 pairs would cost ~$0.60 at
gpt-4.1-mini prices; the approved ceiling for this task is $0.35, so the
task generates pairs in ascending index order until the ceiling would be
exceeded (~550-600 pairs expected). The remaining pairs can be added later
under a separate approval; the prefix design makes the extension clean.

Recipe: identical to #1272 (prompts via the same context builders; 64
already-generated #1272 responses are reused where prompt-identical).
Blinding: sha256("task1275-blind:{index}") % 2; 0 -> A=selected, 1 -> A=full.

Pre-registered gates (judge layer):
    paired_case_count >= 500
    selected_semantic_correct_rate >= full_semantic_correct_rate - 0.05
    selected_cost_reduction_vs_full >= 30%

Live mode requires TASK1275_LIVE_APPROVED=yes, OPENAI_API_KEY, and the
budget ceiling (default $0.33 generation spend for NEW requests).
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
REUSE_1272 = ROOT / "task1272_2wiki_entity_generation_results.json"
GEN_OUTPUT = ROOT / "task1275_2wiki_entity_large_sample_generation_results.json"
JUDGE_REQUESTS = ROOT / "task1275_2wiki_entity_large_sample_judge_requests.jsonl"
OUTPUT = ROOT / "task1275_2wiki_entity_large_sample_judge_results.json"

JUDGE_GATES = {
    "min_paired": 500,
    "semantic_delta_floor": -0.05,
    "cost_reduction_floor_pct": 30.0,
}


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T1272 = load_module("task1272", "task1272_2wiki_entity_generation_judge.py")
T1264 = T1272.T1264


def sha_flip(index: int) -> int:
    return int(hashlib.sha256(f"task1275-blind:{index}".encode()).hexdigest(), 16) % 2


def load_reusable() -> dict[tuple[int, str], dict]:
    if not REUSE_1272.exists():
        return {}
    data = json.loads(REUSE_1272.read_text(encoding="utf-8"))
    reusable = {}
    for record in data.get("requests", []):
        if "generated_answer" in record:
            reusable[(record["index"], record["source_context"])] = record
    return reusable


def build_all_requests() -> list[dict]:
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    dataset_rows = json.loads(DATASET.read_text(encoding="utf-8"))
    requests = []
    for case in sorted(holdout["rows"], key=lambda row: row["index"]):
        case = {**case, "risk_bucket": T1272.risk_bucket(case)}
        contexts = T1272.build_contexts(dataset_rows[case["index"]])
        requests.append(T1272.build_request(case, "selected", contexts["selected_context"]))
        requests.append(T1272.build_request(case, "full", contexts["full_context"]))
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=os.getenv("OPENAI_GENERATION_MODEL", T1264.DEFAULT_MODEL))
    parser.add_argument("--budget-usd", type=float, default=0.33)
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
        # Reuse #1272's scorer but with this task's blinding and gates.
        T1272.sha_flip = sha_flip
        T1272.JUDGE_GATES.update(JUDGE_GATES)
        summary = T1272.score_verdicts(gen_payload, verdicts, args.judge_label)
        summary["claim_decision"] = (
            "2wiki_entity_large_sample_blinded_judge_supports_selected_context"
            if summary["success"]
            else "2wiki_entity_large_sample_blinded_judge_rejects_or_incomplete"
        )
        result = {
            "experiment": "Research Task #1275 2Wiki Entity Regime Large-Sample Generation + Blinded Judge",
            "mode": "imported-verdicts",
            "generation_results": str(GEN_OUTPUT),
            "blinding": "sha256('task1275-blind:{index}') % 2; 0 -> A=selected, 1 -> A=full",
            "gates": JUDGE_GATES,
            "summary": summary,
            "success": summary["success"],
            "claim_decision": summary["claim_decision"],
        }
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "judged_cases"}, ensure_ascii=False, indent=2))
        return

    requests = build_all_requests()
    reusable = load_reusable()
    done, pending = [], []
    for item in requests:
        cached = reusable.get((item["index"], item["source_context"]))
        if cached is not None and cached["messages"] == item["messages"]:
            merged = dict(item)
            for key in ("generation_model", "generated_answer", "exact_match", "latency_ms", "usage"):
                merged[key] = cached[key]
            merged["reused_from_task1272"] = True
            done.append(merged)
        else:
            pending.append(item)
    est_pending = T1264.estimate_cost(
        sum(item["estimated_input_tokens"] for item in pending),
        sum(item["estimated_output_tokens"] for item in pending),
    )
    print(
        json.dumps(
            {
                "total_pairs": len(requests) // 2,
                "reused_requests": len(done),
                "pending_requests": len(pending),
                "estimated_full_pending_cost_usd": round(est_pending, 6),
                "budget_usd": args.budget_usd,
                "note": "generation stops at the budget ceiling; prefix by index order",
            },
            indent=2,
        )
    )

    generated = list(done)
    failures: list[dict] = []
    if args.live:
        if os.getenv("TASK1275_LIVE_APPROVED", "") != "yes":
            raise SystemExit("Live generation requires TASK1275_LIVE_APPROVED=yes.")
        new_records, failures = T1272.run_live(pending, args.model, args.budget_usd)
        generated += new_records

    # Keep only complete pairs.
    by_index: dict[int, dict[str, dict]] = {}
    for record in generated:
        if "generated_answer" in record:
            by_index.setdefault(record["index"], {})[record["source_context"]] = record
    complete = [
        record
        for index in sorted(by_index)
        if len(by_index[index]) == 2
        for record in (by_index[index]["selected"], by_index[index]["full"])
    ]

    T1272.sha_flip = sha_flip
    judge_items = T1272.build_judge_requests(complete)
    with JUDGE_REQUESTS.open("w", encoding="utf-8") as handle:
        for item in judge_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    payload = {
        "experiment": "Research Task #1275 2Wiki Entity Regime Large-Sample Generation + Blinded Judge",
        "mode": "live" if args.live else "dry_run",
        "generation_model": args.model,
        "source": str(HOLDOUT),
        "reuse_source": str(REUSE_1272),
        "judge_request_file": str(JUDGE_REQUESTS),
        "total_holdout_pairs": len(requests) // 2,
        "reused_request_count": len(done),
        "generated_request_count": len(generated),
        "complete_pair_count": len(complete) // 2,
        "failures": failures,
        "budget_usd": args.budget_usd,
        "summary": T1264.summarize_live(complete) if complete else {},
        "generation_cost": T1272.generation_cost_split(complete) if complete else {},
        "requests": complete,
    }
    GEN_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "requests"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
