"""Research Task #1277: Pre-registered superiority replication (remaining 386 pairs).

Hypothesis (PRE-REGISTERED before any generation on this data):
    On the 386 frozen-holdout pairs untouched by #1275 (the complement of
    its 595-pair prefix), named-entity-document selection is semantically
    SUPERIOR to full context for live gpt-4.1-mini generation, under BOTH
    judge families.

Provenance: the superiority hypothesis was born post-hoc in #1275 and
survived a judge-family swap in #1276 on the same data. This task is the
data-level confirmation: fresh pairs, hypothesis and gates fixed first.

Pre-registered gates (ALL must pass):
    G0 paired_case_count >= 350
    Per judge family F in {claude-3pass-majority, external gpt-4.1-mini}:
      G1(F) non-inferiority: selected_rate >= full_rate - 0.05
      G2(F) superiority: selected ahead AND discordant sign test p < 0.05
    G3 generation cost reduction >= 30%

Modes:
    (default)          dry-run: build complement requests, estimate cost
    --live             generate (ceiling default $0.26); writes blinded
                       judge requests (sha256 "task1277-blind:{index}")
    --finalize         score: requires --claude-verdicts; runs the external
                       judge live over the pack (ceiling default $0.07,
                       requires the same approval env var) and applies all
                       gates.

Approval env var: TASK1277_LIVE_APPROVED=yes (covers both live phases).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from math import comb
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/external/2wikimultihop/dev.json"
HOLDOUT = ROOT / "task1271_2wiki_entity_regime_frozen_holdout_results.json"
PREFIX_GEN = ROOT / "task1275_2wiki_entity_large_sample_generation_results.json"
GEN_OUTPUT = ROOT / "task1277_2wiki_entity_replication_generation_results.json"
JUDGE_REQUESTS = ROOT / "task1277_2wiki_entity_replication_judge_requests.jsonl"
EXTERNAL_VERDICTS = ROOT / "task1277_external_judge_verdicts.jsonl"
OUTPUT = ROOT / "task1277_2wiki_entity_superiority_replication_results.json"

GATES = {
    "min_paired_cases": 350,
    "non_inferiority_floor": -0.05,
    "sign_test_p_max": 0.05,
    "cost_reduction_floor_pct": 30.0,
}


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T1272 = load_module("task1272", "task1272_2wiki_entity_generation_judge.py")
T1274 = load_module("task1274", "task1274_external_judge_confirmation.py")
T1264 = T1272.T1264


def sha_flip(index: int) -> int:
    return int(hashlib.sha256(f"task1277-blind:{index}".encode()).hexdigest(), 16) % 2


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n)


def complement_cases() -> list[dict]:
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    prefix = json.loads(PREFIX_GEN.read_text(encoding="utf-8"))
    covered = {record["index"] for record in prefix["requests"]}
    return sorted(
        ({**row, "risk_bucket": T1272.risk_bucket(row)} for row in holdout["rows"] if row["index"] not in covered),
        key=lambda row: row["index"],
    )


def family_summary(rows: list[dict]) -> dict:
    n = len(rows)
    selected_rate = mean(1.0 if r["selected_correct"] else 0.0 for r in rows) if rows else 0.0
    full_rate = mean(1.0 if r["full_correct"] else 0.0 for r in rows) if rows else 0.0
    sel_only = sum(1 for r in rows if r["selected_correct"] and not r["full_correct"])
    full_only = sum(1 for r in rows if r["full_correct"] and not r["selected_correct"])
    p_value = sign_test_two_sided(sel_only, full_only)
    delta = round(selected_rate - full_rate, 4)
    return {
        "n": n,
        "selected_semantic_correct_rate": round(selected_rate, 4),
        "full_semantic_correct_rate": round(full_rate, 4),
        "delta": delta,
        "discordant_selected_only": sel_only,
        "discordant_full_only": full_only,
        "sign_test_p_two_sided": float(f"{p_value:.3e}"),
        "G1_non_inferiority": delta >= GATES["non_inferiority_floor"],
        "G2_superiority": sel_only > full_only and p_value < GATES["sign_test_p_max"],
    }


def unblind_rows(verdicts: dict[int, dict]) -> list[dict]:
    rows = []
    for index, verdict in sorted(verdicts.items()):
        a_is_selected = sha_flip(index) == 0
        rows.append(
            {
                "index": index,
                "selected_correct": bool(verdict["a_correct"] if a_is_selected else verdict["b_correct"]),
                "full_correct": bool(verdict["b_correct"] if a_is_selected else verdict["a_correct"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--claude-verdicts", type=Path, default=None)
    parser.add_argument("--model", default=os.getenv("OPENAI_GENERATION_MODEL", T1264.DEFAULT_MODEL))
    parser.add_argument("--budget-usd", type=float, default=0.26)
    args = parser.parse_args()

    if args.finalize:
        if args.claude_verdicts is None:
            raise SystemExit("--finalize requires --claude-verdicts.")
        gen_payload = json.loads(GEN_OUTPUT.read_text(encoding="utf-8"))
        requests = T1274.load_jsonl(JUDGE_REQUESTS)

        claude_raw = {int(r["case_index"]): r for r in T1274.load_jsonl(args.claude_verdicts)}
        claude_rows = unblind_rows(claude_raw)

        if os.getenv("TASK1277_LIVE_APPROVED", "") != "yes":
            raise SystemExit("External judging requires TASK1277_LIVE_APPROVED=yes.")
        spent = 0.0
        external: dict[int, dict] = {}
        failures = []
        with EXTERNAL_VERDICTS.open("w", encoding="utf-8") as handle:
            for item in requests:
                if spent >= args.budget_usd:
                    print("External-judge budget ceiling reached; stopping.")
                    break
                try:
                    verdict = T1274.call_judge(item["messages"], "gpt-4.1-mini")
                except Exception as error:  # noqa: BLE001
                    failures.append({"case_index": item["case_index"], "error": str(error)})
                    continue
                usage = verdict.get("_usage", {})
                spent += T1274.estimate_cost(
                    usage.get("prompt_tokens", 300), usage.get("completion_tokens", 120)
                )
                external[int(item["case_index"])] = verdict
                handle.write(json.dumps({"case_index": item["case_index"], **verdict}, ensure_ascii=False) + "\n")
        external_rows = unblind_rows(external)

        cost = T1272.generation_cost_split(gen_payload["requests"])
        claude_summary = family_summary(claude_rows)
        external_summary = family_summary(external_rows)
        agreements = []
        for index, verdict in external.items():
            claude_row = claude_raw.get(index)
            if claude_row:
                agreements.append(int(bool(claude_row["a_correct"]) == bool(verdict["a_correct"])))
                agreements.append(int(bool(claude_row["b_correct"]) == bool(verdict["b_correct"])))
        g0 = min(claude_summary["n"], external_summary["n"]) >= GATES["min_paired_cases"]
        g3 = cost["selected_cost_reduction_vs_full_pct"] >= GATES["cost_reduction_floor_pct"]
        success = (
            g0
            and g3
            and claude_summary["G1_non_inferiority"]
            and claude_summary["G2_superiority"]
            and external_summary["G1_non_inferiority"]
            and external_summary["G2_superiority"]
        )
        result = {
            "experiment": "Research Task #1277 Pre-Registered Superiority Replication (386 fresh pairs)",
            "mode": "finalized",
            "gates": GATES,
            "external_judge_spend_usd": round(spent, 6),
            "external_failures": failures,
            "generation_cost": cost,
            "claude_panel": claude_summary,
            "external_judge": external_summary,
            "cross_family_label_agreement": round(mean(agreements), 4) if agreements else None,
            "gate_results": {"G0_min_pairs": g0, "G3_cost_reduction": g3},
            "success": success,
            "claim_decision": (
                "preregistered_superiority_replication_passes"
                if success
                else "preregistered_superiority_replication_fails_or_incomplete"
            ),
        }
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    cases = complement_cases()
    dataset_rows = json.loads(DATASET.read_text(encoding="utf-8"))
    requests = []
    for case in cases:
        contexts = T1272.build_contexts(dataset_rows[case["index"]])
        requests.append(T1272.build_request(case, "selected", contexts["selected_context"]))
        requests.append(T1272.build_request(case, "full", contexts["full_context"]))
    est = T1264.estimate_cost(
        sum(item["estimated_input_tokens"] for item in requests),
        sum(item["estimated_output_tokens"] for item in requests),
    )
    print(
        json.dumps(
            {
                "complement_pairs": len(cases),
                "requests": len(requests),
                "estimated_generation_cost_usd": round(est, 6),
                "budget_usd": args.budget_usd,
            },
            indent=2,
        )
    )

    generated: list[dict] = []
    failures: list[dict] = []
    if args.live:
        if os.getenv("TASK1277_LIVE_APPROVED", "") != "yes":
            raise SystemExit("Live generation requires TASK1277_LIVE_APPROVED=yes.")
        generated, failures = T1272.run_live(requests, args.model, args.budget_usd)

    by_index: dict[int, dict[str, dict]] = {}
    for record in generated:
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
        "experiment": "Research Task #1277 Pre-Registered Superiority Replication (386 fresh pairs)",
        "mode": "live" if args.live else "dry_run",
        "generation_model": args.model,
        "complement_pairs": len(cases),
        "complete_pair_count": len(complete) // 2,
        "failures": failures,
        "judge_request_file": str(JUDGE_REQUESTS),
        "summary": T1264.summarize_live(complete) if complete else {},
        "generation_cost": T1272.generation_cost_split(complete) if complete else {},
        "requests": complete,
    }
    GEN_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "requests"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
