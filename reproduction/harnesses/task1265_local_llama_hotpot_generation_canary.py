"""Research Task #1265: local Llama Hotpot generation canary.

Question:
    Does #1264 selected context catastrophically degrade local generated
    answers compared with full context?

This is a weak, local-only canary using an already available GGUF model through
llama-server. It does not call external APIs and does not download models.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
REQUESTS = ROOT / "task1264_hotpot_local_semantic_generation_requests.jsonl"
OUTPUT = ROOT / "task1265_local_llama_hotpot_generation_canary_results.json"
MODEL = Path(os.environ.get("PROMETHEUS_LOCAL_GEN_MODEL", ROOT / "data/models/llama-3.2-1b-instruct-q4_k_m.gguf"))
PORT = int(os.environ.get("PROMETHEUS_LLAMA_PORT", "8126"))
CASE_COUNT = int(os.environ.get("PROMETHEUS_CASE_COUNT", "12"))
PREDICT = int(os.environ.get("PROMETHEUS_N_PREDICT", "32"))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_match(gold: str, answer: str) -> bool:
    gold_norm = normalize_text(gold)
    answer_norm = normalize_text(answer)
    return bool(gold_norm) and gold_norm in answer_norm


def prompt_for(request: dict) -> str:
    system = request["messages"][0]["content"]
    user = request["messages"][1]["content"]
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def read_requests() -> list[dict]:
    items = []
    with REQUESTS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    by_index = defaultdict(dict)
    for item in items:
        by_index[item["index"]][item["source_context"]] = item
    selected = []
    for index in sorted(by_index):
        pair = by_index[index]
        if "selected" in pair and "full" in pair:
            selected.extend([pair["selected"], pair["full"]])
        if len(selected) >= CASE_COUNT * 2:
            break
    return selected


def wait_ready(process: subprocess.Popen, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{PORT}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("llama-server did not become ready")


def start_server() -> subprocess.Popen:
    if not MODEL.exists():
        raise SystemExit(f"Local model missing: {MODEL}")
    command = [
        "llama-server",
        "--model",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--no-webui",
        "--parallel",
        "1",
        "--ctx-size",
        "8192",
        "--n-predict",
        str(PREDICT),
        "--temp",
        "0",
        "--seed",
        "1265",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    wait_ready(process)
    return process


def complete(prompt: str) -> tuple[str, float]:
    payload = {
        "prompt": prompt,
        "n_predict": PREDICT,
        "temperature": 0,
        "seed": 1265,
        "stop": ["<|user|>", "<|system|>", "\n\n"],
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - started) * 1000
    return (data.get("content") or "").strip(), latency_ms


def summarize(rows: list[dict]) -> dict:
    by_index = defaultdict(dict)
    for row in rows:
        by_index[row["index"]][row["source_context"]] = row
    paired = [pair for pair in by_index.values() if "selected" in pair and "full" in pair]
    selected_matches = [pair["selected"]["match"] for pair in paired]
    full_matches = [pair["full"]["match"] for pair in paired]
    selected_latencies = [pair["selected"]["latency_ms"] for pair in paired]
    full_latencies = [pair["full"]["latency_ms"] for pair in paired]
    return {
        "case_count": len(paired),
        "request_count": len(rows),
        "selected_match_rate": round(sum(selected_matches) / len(paired), 4) if paired else 0.0,
        "full_match_rate": round(sum(full_matches) / len(paired), 4) if paired else 0.0,
        "selected_minus_full_match_rate": round(
            sum(selected_matches) / len(paired) - sum(full_matches) / len(paired),
            4,
        )
        if paired
        else 0.0,
        "mean_selected_latency_ms": round(sum(selected_latencies) / len(selected_latencies), 3)
        if selected_latencies
        else 0.0,
        "mean_full_latency_ms": round(sum(full_latencies) / len(full_latencies), 3)
        if full_latencies
        else 0.0,
        "latency_reduction_pct": round(
            (1 - (sum(selected_latencies) / len(selected_latencies)) / (sum(full_latencies) / len(full_latencies))) * 100,
            3,
        )
        if selected_latencies and full_latencies and sum(full_latencies)
        else 0.0,
        "mean_token_reduction_pct": round(
            sum(pair["selected"]["token_reduction_pct"] for pair in paired) / len(paired),
            3,
        )
        if paired
        else 0.0,
    }


def main() -> None:
    requests = read_requests()
    process = start_server()
    rows = []
    error = None
    try:
        for number, item in enumerate(requests, start=1):
            try:
                answer, latency_ms = complete(prompt_for(item))
                rows.append(
                    {
                        "number": number,
                        "case_id": item["case_id"],
                        "source_context": item["source_context"],
                        "index": item["index"],
                        "question": item["question"],
                        "gold_answer": item["gold_answer"],
                        "risk_bucket": item["risk_bucket"],
                        "token_reduction_pct": item["token_reduction_pct"],
                        "answer": answer,
                        "match": answer_match(item["gold_answer"], answer),
                        "latency_ms": round(latency_ms, 3),
                    }
                )
            except (urllib.error.URLError, TimeoutError) as exc:
                error = str(exc)
                break
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    summary = summarize(rows)
    evaluator_has_signal = summary["selected_match_rate"] > 0 or summary["full_match_rate"] > 0
    success = (
        summary["case_count"] >= 8
        and evaluator_has_signal
        and summary["selected_minus_full_match_rate"] >= -0.10
        and summary["mean_token_reduction_pct"] >= 30.0
    )
    payload = {
        "experiment": "Research Task #1265 Local Llama Hotpot Generation Canary",
        "hypothesis": (
            "The #1264 selected context does not catastrophically degrade local "
            "generated answers versus full context."
        ),
        "model": str(MODEL),
        "source": str(REQUESTS),
        "case_count_target": CASE_COUNT,
        "summary": summary,
        "evaluator_has_signal": bool(evaluator_has_signal),
        "success": bool(success),
        "claim_decision": (
            "local_llama_canary_supports_hotpot_selected_context"
            if success
            else "local_llama_canary_insufficient_or_weakens_hotpot_selected_context"
        ),
        "error": error,
        "limitations": [
            "Small local Llama 1B exact-match generation only.",
            "Not a semantic judge.",
            "Diagnostic 12-case subset only.",
            "Local latency is not OpenAI API latency.",
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
