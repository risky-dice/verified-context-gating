"""Research Task #1276: External judge confirmation of the #1275 superiority signal.

Question:
    #1275 found a POST-HOC superiority signal (selected 0.9697 vs full
    0.9059, discordant 44:6, p=3.2e-08) under the Claude judge panel. Does
    it survive an external judge (gpt-4.1-mini)? This targets the specific
    residual concern — judge-model family — not new data (the remaining 386
    holdout pairs are the data-level extension, separate task).

Pre-registered gates (fixed before the external judge runs):
    E1 non-inferiority: selected_rate >= full_rate - 0.05  (as always)
    E2 superiority confirmation: under the external judge, discordant-pair
       sign test two-sided p < 0.05 AND selected_rate > full_rate.
    E2 passing upgrades the #1275 observation from post-hoc to a
       two-judge-family superiority result on this sample.

Reported (not gated): per-label agreement with the Claude majority.

Live mode requires TASK1276_LIVE_APPROVED=yes, OPENAI_API_KEY, and a budget
ceiling (default $0.11).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from math import comb
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
REQUESTS = ROOT / "task1275_2wiki_entity_large_sample_judge_requests.jsonl"
CLAUDE_VERDICTS = ROOT / "task1275_claude_blinded_judge_verdicts.jsonl"
VERDICTS_OUT = ROOT / "task1276_external_judge_verdicts_task1275_pack.jsonl"
OUTPUT = ROOT / "task1276_external_judge_1275_superiority_confirmation_results.json"
BLIND_PREFIX = "task1275-blind"


def load_task1274_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "task1274", Path(__file__).parent / "task1274_external_judge_confirmation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T1274 = load_task1274_module()


def sha_flip(index: int) -> int:
    return int(hashlib.sha256(f"{BLIND_PREFIX}:{index}".encode()).hexdigest(), 16) % 2


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL", T1274.MODEL_DEFAULT))
    parser.add_argument("--budget-usd", type=float, default=0.11)
    args = parser.parse_args()

    requests = T1274.load_jsonl(REQUESTS)
    # Calibrated on #1274 actuals: $0.031474 / 187 judged cases. The naive
    # chars/4 + 120-output estimate overshoots judge calls by ~60%.
    estimated_cost = round(len(requests) * 0.031474 / 187, 6)
    print(
        json.dumps(
            {
                "judge_requests": len(requests),
                "estimated_cost_usd": estimated_cost,
                "budget_usd": args.budget_usd,
            },
            indent=2,
        )
    )

    result = {
        "experiment": "Research Task #1276 External Judge Confirmation of #1275 Superiority",
        "mode": "live" if args.live else "dry_run",
        "judge_model": args.model,
        "gates": {
            "E1_non_inferiority_floor": -0.05,
            "E2_sign_test_p_max": 0.05,
        },
        "estimated_cost_usd": estimated_cost,
        "budget_usd": args.budget_usd,
        "success": False,
        "claim_decision": "dry_run_only",
    }

    if args.live:
        if os.getenv("TASK1276_LIVE_APPROVED", "") != "yes":
            raise SystemExit("Live run requires TASK1276_LIVE_APPROVED=yes.")
        if estimated_cost > args.budget_usd:
            raise SystemExit(f"Estimated ${estimated_cost} exceeds budget ${args.budget_usd}.")
        spent = 0.0
        verdicts: dict[int, dict] = {}
        failures = []
        with VERDICTS_OUT.open("w", encoding="utf-8") as handle:
            for item in requests:
                if spent >= args.budget_usd:
                    print("Budget ceiling reached; stopping.")
                    break
                try:
                    verdict = T1274.call_judge(item["messages"], args.model)
                except Exception as error:  # noqa: BLE001
                    failures.append({"case_index": item["case_index"], "error": str(error)})
                    continue
                usage = verdict.get("_usage", {})
                spent += T1274.estimate_cost(
                    usage.get("prompt_tokens", 300), usage.get("completion_tokens", 120)
                )
                verdicts[int(item["case_index"])] = verdict
                handle.write(
                    json.dumps({"case_index": item["case_index"], **verdict}, ensure_ascii=False)
                    + "\n"
                )

        claude = {int(r["case_index"]): r for r in T1274.load_jsonl(CLAUDE_VERDICTS)}
        rows = []
        agreements = []
        for index, verdict in sorted(verdicts.items()):
            a_is_selected = sha_flip(index) == 0
            selected_correct = bool(verdict["a_correct"] if a_is_selected else verdict["b_correct"])
            full_correct = bool(verdict["b_correct"] if a_is_selected else verdict["a_correct"])
            rows.append({"index": index, "selected_correct": selected_correct, "full_correct": full_correct})
            claude_row = claude.get(index)
            if claude_row:
                agreements.append(int(bool(claude_row["a_correct"]) == bool(verdict["a_correct"])))
                agreements.append(int(bool(claude_row["b_correct"]) == bool(verdict["b_correct"])))

        n = len(rows)
        selected_rate = mean(1.0 if r["selected_correct"] else 0.0 for r in rows) if rows else 0.0
        full_rate = mean(1.0 if r["full_correct"] else 0.0 for r in rows) if rows else 0.0
        delta = round(selected_rate - full_rate, 4)
        sel_only = [r["index"] for r in rows if r["selected_correct"] and not r["full_correct"]]
        full_only = [r["index"] for r in rows if r["full_correct"] and not r["selected_correct"]]
        p_value = sign_test_two_sided(len(sel_only), len(full_only))
        e1 = delta >= -0.05
        e2 = len(sel_only) > len(full_only) and p_value < 0.05
        result.update(
            {
                "actual_cost_usd": round(spent, 6),
                "failures": failures,
                "judged_case_count": n,
                "selected_semantic_correct_rate": round(selected_rate, 4),
                "full_semantic_correct_rate": round(full_rate, 4),
                "selected_minus_full_semantic_rate": delta,
                "discordant": {
                    "selected_only_correct": len(sel_only),
                    "full_only_correct": len(full_only),
                    "sign_test_p_two_sided": float(f"{p_value:.3e}"),
                },
                "agreement_with_claude_majority": round(mean(agreements), 4) if agreements else None,
                "gate_results": {"E1_non_inferiority": e1, "E2_superiority": e2},
                "success": e1 and e2,
                "claim_decision": (
                    "external_judge_confirms_1275_superiority"
                    if e1 and e2
                    else ("external_judge_confirms_non_inferiority_only" if e1 else "external_judge_rejects")
                ),
            }
        )

    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
