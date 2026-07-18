"""contextgates command-line interface.

    contextgates verify --data mydata.jsonl [--gate entity_title] [--json out.json]
    contextgates route  --data mydata.jsonl [--gate entity_title]

`verify` runs the free L1 retention layer on your data and prints a report
telling you whether a gate is safe to enable — and, when it is not, exactly
which threshold failed. L2/L3 (generation + judging) need your own LLM
callables and are driven from Python (see examples/quickstart.py); the CLI
deliberately stops at the free, local layer so `verify` never spends money or
needs an API key.

Data format: JSONL, one object per line:
    {"question": "...", "docs": [{"title": "...", "sentences": ["...", "..."]}],
     "gold_answer": "...", "support": [["Title", 0], ...]}   # support optional

`docs` may also be [["Title", ["sent", ...]], ...] or use "text" instead of
"sentences" (auto sentence-split). Rows missing gold_answer are skipped with a
warning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, entity_title_gate, or_comparison_gate
from .gates import GateRegistry
from .verify import RetentionThresholds, evaluate_retention

GATES = {
    "entity_title": entity_title_gate,
    "or_comparison": or_comparison_gate,
}


def _split_sentences(text: str) -> list[str]:
    import re

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _normalize_docs(docs):
    out = []
    for d in docs:
        if isinstance(d, dict):
            title = d.get("title", "")
            if "sentences" in d:
                sents = list(d["sentences"])
            else:
                sents = _split_sentences(d.get("text", ""))
            out.append({"title": title, "sentences": sents})
        elif isinstance(d, (list, tuple)) and len(d) == 2:
            title, body = d
            sents = body if isinstance(body, list) else _split_sentences(str(body))
            out.append({"title": title, "sentences": list(sents)})
    return out


def _load(path: Path):
    rows, skipped = [], 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"  ! line {i}: invalid JSON, skipped", file=sys.stderr)
            skipped += 1
            continue
        if "question" not in obj or "docs" not in obj:
            skipped += 1
            continue
        row = {
            "question": obj["question"],
            "docs": _normalize_docs(obj["docs"]),
            "gold_answer": obj.get("gold_answer", obj.get("answer", "")),
        }
        if obj.get("support"):
            row["support"] = [tuple(s) for s in obj["support"]]
        rows.append(row)
    return rows, skipped


BAR = "─" * 58


def _pct(x):
    return "  n/a" if x is None else f"{x*100:5.1f}%"


def cmd_verify(args) -> int:
    path = Path(args.data)
    if not path.exists():
        print(f"error: data file not found: {path}", file=sys.stderr)
        return 2
    rows, skipped = _load(path)
    if not rows:
        print("error: no usable rows (need question + docs per line)", file=sys.stderr)
        return 2

    names = [args.gate] if args.gate else list(GATES)
    th = RetentionThresholds()
    reports = {}
    print(BAR)
    print(f" contextgates verify  ·  {len(rows)} rows  ·  {path.name}")
    if skipped:
        print(f" ({skipped} rows skipped: missing fields or bad JSON)")
    print(BAR)
    for name in names:
        gate = GATES[name]()
        r = evaluate_retention(gate, rows, th)
        reports[name] = r
        verdict = "PASS ✓" if r["passed"] else "not verified"
        print(f"\n  gate: {name}   [{verdict}]")
        print(f"    admitted            {r['n_admitted']}/{r['n_total']}  ({_pct(r['admission_rate'])} of your traffic)")
        print(f"    answer retention    {_pct(r['answer_retention_rate_given_full'])}   (need >= {th.min_answer_retention_given_full*100:.0f}%)")
        if r["mean_support_coverage"] is not None:
            print(f"    evidence coverage   {_pct(r['mean_support_coverage'])}   (need >= {th.min_mean_support_coverage*100:.0f}%)")
        else:
            print(f"    evidence coverage     n/a   (add 'support' spans to check this)")
        print(f"    token reduction     {r['mean_token_reduction_pct']:5.1f}%   (need >= {th.min_token_reduction_pct:.0f}%)")
        if not r["passed"]:
            fails = [k for k, v in r["checks"].items() if not v]
            print(f"    -> blocked by: {', '.join(fails)}")
            if r["n_admitted"] < th.min_n:
                print(f"       (need >= {th.min_n} admitted rows for a reliable read; give more data)")

    print("\n" + BAR)
    any_pass = any(r["passed"] for r in reports.values())
    if any_pass:
        print(" Next: L1 passed is necessary but NOT sufficient. Run L2/L3")
        print(" (live generation + blinded judging) from Python before enabling —")
        print(" see examples/quickstart.py. Proxy metrics can mislead.")
    else:
        print(" No gate verified at L1 on this data. That is the safe answer:")
        print(" every query keeps full context. Cheap structural gating may")
        print(" simply not fit this traffic (e.g. single-hop or bridge questions).")
    print(BAR)

    if args.json:
        Path(args.json).write_text(json.dumps(reports, ensure_ascii=False, indent=2))
        print(f" wrote {args.json}")
    return 0


def cmd_route(args) -> int:
    path = Path(args.data)
    rows, _ = _load(path)
    if not rows:
        print("error: no usable rows", file=sys.stderr)
        return 2
    reg = GateRegistry()
    name = args.gate or "entity_title"
    reg.register(GATES[name](), evidence={"passed": True})  # demo: force-enabled
    admitted = 0
    saved = []
    for row in rows:
        res = reg.route(row["question"], row["docs"], allow_unverified=True)
        if res.admitted:
            admitted += 1
            saved.append(res.token_reduction_pct)
    mean_red = sum(saved) / len(saved) if saved else 0.0
    print(f" gate {name}: admitted {admitted}/{len(rows)} "
          f"({admitted/len(rows)*100:.1f}%), mean token reduction on admitted {mean_red:.1f}%")
    print(" (demo routing with the gate force-enabled; verify on your data first)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="contextgates", description="Verified context gating for RAG.")
    p.add_argument("--version", action="version", version=f"contextgates {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="run the free L1 retention check on your data")
    v.add_argument("--data", required=True, help="JSONL file: question, docs, gold_answer[, support]")
    v.add_argument("--gate", choices=list(GATES), help="check one gate (default: all)")
    v.add_argument("--json", help="also write the full report to this path")
    v.set_defaults(func=cmd_verify)

    r = sub.add_parser("route", help="demo: show what a gate would admit/save")
    r.add_argument("--data", required=True)
    r.add_argument("--gate", choices=list(GATES))
    r.set_defaults(func=cmd_route)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
