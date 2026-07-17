"""Research Task #1274: External judge confirmation over all blinded packs.

Question:
    Do the three blinded-judge results (#1266, #1267, #1272) survive when the
    judge is an EXTERNAL model (gpt-4.1-mini) instead of the Claude subagent
    panel? This removes the last recurring methodological caveat: the Claude
    judge shares a model family with the orchestrating tool.

Method:
    For each pack, send the exact blinded judge messages already on disk
    (label-free; A/B assignment via each task's own sha256 scheme) to
    gpt-4.1-mini (temperature 0, JSON mode). Unblind with the original
    scheme, compute semantic rates, and measure per-label agreement with the
    Claude 3-pass majority verdicts.

Packs:
    #1266: task1266_hotpot_live_generation_semantic_judge_requests.jsonl (32)
    #1267: task1267_hotpot_full_sample_judge_requests.jsonl (123)
    #1272: task1272_2wiki_entity_judge_requests.jsonl (32)

Pre-registered gates:
    per pack: selected_semantic_correct_rate >= full_rate - 0.05
              (same non-inferiority floor as the original tasks)
    overall:  all three packs pass
    reported (not gated): external-vs-Claude per-label agreement rates.

Live mode requires TASK1274_LIVE_APPROVED=yes, OPENAI_API_KEY, and a budget
ceiling (default $0.06).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "task1274_external_judge_confirmation_results.json"
MODEL_DEFAULT = "gpt-4.1-mini"
INPUT_PRICE_PER_1M = 0.40
OUTPUT_PRICE_PER_1M = 1.60
SEMANTIC_DELTA_FLOOR = -0.05

PACKS = [
    {
        "name": "task1266_hotpot_canary",
        "requests": ROOT / "task1266_hotpot_live_generation_semantic_judge_requests.jsonl",
        "claude_verdicts": ROOT / "task1266_claude_blinded_judge_verdicts.jsonl",
        "blind_prefix": "task1266-blind",
    },
    {
        "name": "task1267_hotpot_full_sample",
        "requests": ROOT / "task1267_hotpot_full_sample_judge_requests.jsonl",
        "claude_verdicts": ROOT / "task1267_claude_blinded_judge_verdicts.jsonl",
        "blind_prefix": "task1267-blind",
    },
    {
        "name": "task1272_2wiki_entity_canary",
        "requests": ROOT / "task1272_2wiki_entity_judge_requests.jsonl",
        "claude_verdicts": ROOT / "task1272_claude_blinded_judge_verdicts.jsonl",
        "blind_prefix": "task1272-blind",
    },
]


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_PRICE_PER_1M + output_tokens * OUTPUT_PRICE_PER_1M
    ) / 1_000_000


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def sha_flip(prefix: str, index: int) -> int:
    return int(hashlib.sha256(f"{prefix}:{index}".encode()).hexdigest(), 16) % 2


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def call_judge(messages: list[dict], model: str) -> dict:
    api_key = os.environ["OPENAI_API_KEY"]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 192,
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


def unblind(prefix: str, index: int, verdict: dict) -> dict:
    a_is_selected = sha_flip(prefix, index) == 0
    return {
        "selected_correct": bool(
            verdict["a_correct"] if a_is_selected else verdict["b_correct"]
        ),
        "full_correct": bool(
            verdict["b_correct"] if a_is_selected else verdict["a_correct"]
        ),
    }


def score_pack(pack: dict, verdicts: dict[int, dict]) -> dict:
    claude = {int(row["case_index"]): row for row in load_jsonl(pack["claude_verdicts"])}
    rows = []
    label_agreements = []
    for index, verdict in sorted(verdicts.items()):
        unblinded = unblind(pack["blind_prefix"], index, verdict)
        rows.append(unblinded)
        claude_row = claude.get(index)
        if claude_row is not None:
            label_agreements.append(
                int(bool(claude_row["a_correct"]) == bool(verdict["a_correct"]))
            )
            label_agreements.append(
                int(bool(claude_row["b_correct"]) == bool(verdict["b_correct"]))
            )
    n = len(rows)
    selected_rate = mean(1.0 if r["selected_correct"] else 0.0 for r in rows) if rows else 0.0
    full_rate = mean(1.0 if r["full_correct"] else 0.0 for r in rows) if rows else 0.0
    delta = round(selected_rate - full_rate, 4)
    return {
        "judged_case_count": n,
        "selected_semantic_correct_rate": round(selected_rate, 4),
        "full_semantic_correct_rate": round(full_rate, 4),
        "selected_minus_full_semantic_rate": delta,
        "semantic_delta_gate": delta >= SEMANTIC_DELTA_FLOOR,
        "agreement_with_claude_majority": round(mean(label_agreements), 4)
        if label_agreements
        else None,
        "agreement_label_count": len(label_agreements),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL", MODEL_DEFAULT))
    parser.add_argument("--budget-usd", type=float, default=0.06)
    args = parser.parse_args()

    packs_data = []
    total_estimated_input = 0
    for pack in PACKS:
        requests = load_jsonl(pack["requests"])
        estimated = sum(
            approx_tokens(m["content"]) for r in requests for m in r["messages"]
        )
        total_estimated_input += estimated
        packs_data.append((pack, requests))
    total_requests = sum(len(reqs) for _p, reqs in packs_data)
    estimated_cost = round(
        estimate_cost(total_estimated_input, total_requests * 120), 6
    )
    print(
        json.dumps(
            {
                "total_judge_requests": total_requests,
                "estimated_cost_usd": estimated_cost,
                "budget_usd": args.budget_usd,
            },
            indent=2,
        )
    )

    result = {
        "experiment": "Research Task #1274 External Judge Confirmation",
        "hypothesis": (
            "The #1266/#1267/#1272 blinded-judge conclusions survive when the "
            "judge is external (gpt-4.1-mini) instead of the Claude panel."
        ),
        "mode": "live" if args.live else "dry_run",
        "judge_model": args.model,
        "semantic_delta_floor": SEMANTIC_DELTA_FLOOR,
        "estimated_cost_usd": estimated_cost,
        "budget_usd": args.budget_usd,
        "success": False,
        "claim_decision": "dry_run_only",
    }

    if args.live:
        if os.getenv("TASK1274_LIVE_APPROVED", "") != "yes":
            raise SystemExit("Live run requires TASK1274_LIVE_APPROVED=yes.")
        if estimated_cost > args.budget_usd:
            raise SystemExit(
                f"Estimated cost ${estimated_cost} exceeds budget ${args.budget_usd}."
            )
        spent = 0.0
        pack_summaries = {}
        failures = []
        for pack, requests in packs_data:
            verdicts: dict[int, dict] = {}
            raw_records = []
            for item in requests:
                if spent >= args.budget_usd:
                    print(f"Budget ceiling reached inside {pack['name']}; stopping.")
                    break
                try:
                    verdict = call_judge(item["messages"], args.model)
                except Exception as error:  # noqa: BLE001
                    failures.append({"pack": pack["name"], "case_index": item["case_index"], "error": str(error)})
                    continue
                usage = verdict.get("_usage", {})
                spent += estimate_cost(
                    usage.get("prompt_tokens", 300), usage.get("completion_tokens", 120)
                )
                verdicts[int(item["case_index"])] = verdict
                raw_records.append({"case_index": item["case_index"], **verdict})
            out_path = ROOT / f"task1274_external_judge_verdicts_{pack['name']}.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for record in raw_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            pack_summaries[pack["name"]] = score_pack(pack, verdicts)
        all_pass = all(s["semantic_delta_gate"] for s in pack_summaries.values())
        complete = all(
            s["judged_case_count"] == len(reqs)
            for (p, reqs), s in zip(packs_data, pack_summaries.values())
        )
        result.update(
            {
                "actual_cost_usd": round(spent, 6),
                "failures": failures,
                "pack_summaries": pack_summaries,
                "success": all_pass and complete,
                "claim_decision": (
                    "external_judge_confirms_all_blinded_results"
                    if all_pass and complete
                    else "external_judge_diverges_or_incomplete"
                ),
            }
        )

    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
