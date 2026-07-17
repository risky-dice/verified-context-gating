"""Research Task #1273: Latency-controlled interleaved canary (both regimes).

Question:
    Does selected context actually reduce live OpenAI wall-clock latency?
    This is the last unclaimed axis. Prior signals were suggestive only and
    methodologically weak: #1264 -0.4% (all-selected block then all-full
    block), #1267 +26% and #1272 +11% (sequential, uncontrolled, mean-based,
    tail-sensitive).

Design (pre-registered):
    Cases: 24 regime-1 pairs (Hotpot-or + lexical_top10, prompts byte-identical
    to #1264) and 24 regime-2 pairs (2Wiki entity + sent_full, prompts
    byte-identical to #1272), lowest case indices first (deterministic).

    Ordering: per case, ABBA — arm X, arm Y, arm Y, arm X — so each arm is
    measured twice and order effects cancel within the case. Which arm is X
    alternates across cases (even position: selected first; odd: full first)
    to cancel drift across cases. 2 untimed warm-up calls precede everything.

    Per-case latency per arm = mean of its 2 samples. Paired comparison per
    case; per-regime gates:
        paired_case_count >= 20
        sign test (selected faster), two-sided p < 0.05
        mean paired latency reduction >= 10%
        completion-token comparability: mean_selected_completion_tokens /
            mean_full_completion_tokens in [0.80, 1.25]  (guards against
            latency differences that are really output-length differences)

Claim decisions:
    both pass  -> latency_acceleration_supported_both_regimes
    one passes -> latency_acceleration_supported_<regime>_only
    none       -> latency_acceleration_not_supported

Live mode requires TASK1273_LIVE_APPROVED=yes, OPENAI_API_KEY, and a budget
ceiling (default $0.09). Answers are generated at temperature 0 with the
same max_tokens as the source packs; answer content is recorded but quality
is NOT re-judged here (already covered by #1266/#1272).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from math import comb
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
REGIME1_SOURCE = (
    ROOT / "task1264_hotpot_local_semantic_generation_pack_results.live_backup_20260715.json"
)
REGIME2_SOURCE = ROOT / "task1272_2wiki_entity_generation_results.json"
OUTPUT = ROOT / "task1273_latency_controlled_interleaved_canary_results.json"
CASES_PER_REGIME = 24

GATES = {
    "min_paired_cases": 20,
    "sign_test_p_max": 0.05,
    "min_mean_latency_reduction_pct": 10.0,
    "completion_token_ratio_range": [0.80, 1.25],
}


def load_task1264_module():
    spec = importlib.util.spec_from_file_location(
        "task1264",
        Path(__file__).parent / "task1264_hotpot_local_semantic_generation_pack.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T1264 = load_task1264_module()


def load_pairs(path: Path, regime: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_index: dict[int, dict[str, dict]] = {}
    for record in data["requests"]:
        if "generated_answer" not in record:
            continue
        by_index.setdefault(record["index"], {})[record["source_context"]] = record
    pairs = []
    for index in sorted(by_index):
        pair = by_index[index]
        if "selected" not in pair or "full" not in pair:
            continue
        pairs.append(
            {
                "regime": regime,
                "index": index,
                "question": pair["selected"]["question"],
                "selected_messages": pair["selected"]["messages"],
                "full_messages": pair["full"]["messages"],
                "estimated_tokens": {
                    "selected": pair["selected"]["estimated_input_tokens"],
                    "full": pair["full"]["estimated_input_tokens"],
                },
            }
        )
        if len(pairs) >= CASES_PER_REGIME:
            break
    return pairs


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    p_tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * p_tail)


def timed_call(messages: list[dict], model: str) -> dict:
    item = {"messages": messages, "estimated_output_tokens": 48}
    return T1264.call_openai_chat(item, model)


def run_regime(pairs: list[dict], model: str, spend_state: dict, budget: float) -> list[dict]:
    measured = []
    for position, pair in enumerate(pairs):
        first, second = (
            ("selected", "full") if position % 2 == 0 else ("full", "selected")
        )
        order = [first, second, second, first]
        samples: dict[str, list[dict]] = {"selected": [], "full": []}
        aborted = False
        for arm in order:
            projected = spend_state["cost"] + T1264.estimate_cost(
                pair["estimated_tokens"][arm], 48
            )
            if projected > budget:
                print(f"Budget ceiling ${budget} reached at case {pair['index']}; stopping.")
                aborted = True
                break
            result = timed_call(pair[f"{arm}_messages"], model)
            spend_state["cost"] += T1264.estimate_cost(
                result["usage"]["prompt_tokens"], result["usage"]["completion_tokens"]
            )
            samples[arm].append(result)
        if aborted or len(samples["selected"]) < 2 or len(samples["full"]) < 2:
            break
        measured.append(
            {
                "regime": pair["regime"],
                "index": pair["index"],
                "first_arm": first,
                "selected_latencies_ms": [s["latency_ms"] for s in samples["selected"]],
                "full_latencies_ms": [s["latency_ms"] for s in samples["full"]],
                "selected_mean_ms": round(mean(s["latency_ms"] for s in samples["selected"]), 3),
                "full_mean_ms": round(mean(s["latency_ms"] for s in samples["full"]), 3),
                "selected_completion_tokens": [
                    s["usage"]["completion_tokens"] for s in samples["selected"]
                ],
                "full_completion_tokens": [
                    s["usage"]["completion_tokens"] for s in samples["full"]
                ],
                "selected_prompt_tokens": samples["selected"][0]["usage"]["prompt_tokens"],
                "full_prompt_tokens": samples["full"][0]["usage"]["prompt_tokens"],
            }
        )
    return measured


def summarize_regime(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"paired_case_count": 0, "success": False}
    wins = sum(1 for row in rows if row["selected_mean_ms"] < row["full_mean_ms"])
    losses = n - wins
    p_value = sign_test_two_sided(wins, losses)
    reductions = [
        (row["full_mean_ms"] - row["selected_mean_ms"]) / row["full_mean_ms"] * 100
        for row in rows
    ]
    sel_completion = mean(
        token for row in rows for token in row["selected_completion_tokens"]
    )
    full_completion = mean(
        token for row in rows for token in row["full_completion_tokens"]
    )
    ratio = sel_completion / full_completion if full_completion else 0.0
    low, high = GATES["completion_token_ratio_range"]
    gates = {
        "paired_case_count_gate": n >= GATES["min_paired_cases"],
        "sign_test_gate": wins > losses and p_value < GATES["sign_test_p_max"],
        "mean_reduction_gate": mean(reductions) >= GATES["min_mean_latency_reduction_pct"],
        "completion_token_gate": low <= ratio <= high,
    }
    return {
        "paired_case_count": n,
        "selected_faster_count": wins,
        "full_faster_count": losses,
        "sign_test_p_two_sided": round(p_value, 5),
        "mean_latency_reduction_pct": round(mean(reductions), 3),
        "median_latency_reduction_pct": round(median(reductions), 3),
        "mean_selected_ms": round(mean(row["selected_mean_ms"] for row in rows), 3),
        "mean_full_ms": round(mean(row["full_mean_ms"] for row in rows), 3),
        "mean_prompt_tokens": {
            "selected": round(mean(row["selected_prompt_tokens"] for row in rows), 1),
            "full": round(mean(row["full_prompt_tokens"] for row in rows), 1),
        },
        "completion_token_ratio_selected_over_full": round(ratio, 4),
        "gates": gates,
        "success": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=os.getenv("OPENAI_GENERATION_MODEL", T1264.DEFAULT_MODEL))
    parser.add_argument("--budget-usd", type=float, default=0.09)
    args = parser.parse_args()

    regime1 = load_pairs(REGIME1_SOURCE, "hotpot_or_top10")
    regime2 = load_pairs(REGIME2_SOURCE, "2wiki_entity_sent_full")
    estimated = sum(
        2 * (pair["estimated_tokens"]["selected"] + pair["estimated_tokens"]["full"])
        for pair in regime1 + regime2
    )
    estimated_cost = round(T1264.estimate_cost(estimated, (len(regime1) + len(regime2)) * 4 * 48), 6)
    print(
        json.dumps(
            {
                "regime1_pairs": len(regime1),
                "regime2_pairs": len(regime2),
                "timed_requests": (len(regime1) + len(regime2)) * 4,
                "estimated_cost_usd": estimated_cost,
                "budget_usd": args.budget_usd,
            },
            indent=2,
        )
    )

    result = {
        "experiment": "Research Task #1273 Latency-Controlled Interleaved Canary",
        "hypothesis": (
            "Selected context reduces live API wall-clock latency in both regimes "
            "under order-controlled, paired measurement."
        ),
        "mode": "live" if args.live else "dry_run",
        "model": args.model,
        "design": "per-case ABBA interleaving, first arm alternating across cases, 2 warm-up calls",
        "sources": {"regime1": str(REGIME1_SOURCE), "regime2": str(REGIME2_SOURCE)},
        "cases_per_regime": CASES_PER_REGIME,
        "gates": GATES,
        "estimated_cost_usd": estimated_cost,
        "budget_usd": args.budget_usd,
        "success": False,
        "claim_decision": "dry_run_only",
    }

    if args.live:
        if os.getenv("TASK1273_LIVE_APPROVED", "") != "yes":
            raise SystemExit("Live run requires TASK1273_LIVE_APPROVED=yes.")
        if estimated_cost > args.budget_usd:
            raise SystemExit(
                f"Estimated cost ${estimated_cost} exceeds budget ${args.budget_usd}."
            )
        spend_state = {"cost": 0.0}
        for _ in range(2):  # untimed warm-up
            timed_call(
                [
                    {"role": "system", "content": "Reply with OK."},
                    {"role": "user", "content": "warm-up"},
                ],
                args.model,
            )
        rows1 = run_regime(regime1, args.model, spend_state, args.budget_usd)
        rows2 = run_regime(regime2, args.model, spend_state, args.budget_usd)
        summary1 = summarize_regime(rows1)
        summary2 = summarize_regime(rows2)
        passing = [
            name
            for name, summary in (("regime1", summary1), ("regime2", summary2))
            if summary["success"]
        ]
        if len(passing) == 2:
            claim = "latency_acceleration_supported_both_regimes"
        elif passing:
            claim = f"latency_acceleration_supported_{passing[0]}_only"
        else:
            claim = "latency_acceleration_not_supported"
        result.update(
            {
                "actual_cost_usd": round(spend_state["cost"], 6),
                "regime1_summary": summary1,
                "regime2_summary": summary2,
                "regime1_rows": rows1,
                "regime2_rows": rows2,
                "success": bool(passing),
                "claim_decision": claim,
            }
        )

    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    view = {k: v for k, v in result.items() if k not in ("regime1_rows", "regime2_rows")}
    print(json.dumps(view, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
