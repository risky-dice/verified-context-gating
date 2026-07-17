"""contextgates quickstart: verify a gate on your data, then route through it.

Run:  python examples/quickstart.py            # L1 only, no API calls, free
      OPENAI_API_KEY=... python examples/quickstart.py --live   # full ladder

The library never holds your keys. `generate` and `judge` are callables you
write; the adapter below is the entire integration surface (~20 lines).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextgates import GateRegistry, entity_title_gate, run_ladder

# ---------------------------------------------------------------- your data
# Rows need: question, docs [{title, sentences}], gold_answer,
# and optionally support [(title, sentence_index), ...] for coverage checks.
DATASET = [
    {
        "question": "Which film was released first, Alpha Film or Beta Film?",
        "docs": [
            {"title": "Alpha Film", "sentences": ["Alpha Film is a 1990 drama.", "It won a prize."]},
            {"title": "Beta Film", "sentences": ["Beta Film is a 2001 comedy.", "It flopped."]},
            {"title": "Rail Transport", "sentences": ["Trains carry freight.", "Noise."]},
            {"title": "Weather", "sentences": ["It rained on Tuesday.", "Filler."]},
        ],
        "gold_answer": "Alpha Film",
        "support": [("Alpha Film", 0), ("Beta Film", 0)],
    },
] * 120  # a real run needs >= 100 admitted rows of YOUR traffic


# ------------------------------------------------------- provider adapters
def openai_call(messages, json_mode=False, model="gpt-4.1-mini"):
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 128}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read().decode())
    return body["choices"][0]["message"]["content"]


def generate(messages):
    return openai_call(messages)


def judge(messages):
    return json.loads(openai_call(messages, json_mode=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="run L2+L3 (costs API money)")
    args = parser.parse_args()

    gate = entity_title_gate()

    # The evidence ladder. Without --live this stops at L1 and the gate stays
    # DISABLED on purpose: retention evidence alone never enables a gate.
    report = run_ladder(
        gate,
        DATASET,
        generate=generate if args.live else None,
        judge=judge if args.live else None,
        salt="quickstart-v1",
    )
    print(json.dumps(report, indent=2)[:1500])

    registry = GateRegistry()
    if report["passed"]:
        registry.register(gate, evidence=report)
        print("\nGate ENABLED by evidence.")
    else:
        print("\nGate NOT enabled — every query will use full context (safe default).")

    result = registry.route(DATASET[0]["question"], DATASET[0]["docs"])
    print(f"admitted={result.admitted} gate={result.gate_name} "
          f"reduction={result.token_reduction_pct}%")
    print("context passed to your LLM:\n" + result.context_text)


if __name__ == "__main__":
    main()
