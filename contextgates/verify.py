"""The evidence ladder: verify a gate on YOUR data before enabling it.

Layers (thresholds are pre-specified here; change them BEFORE running, and
keep the changed source under version control — that is the whole point):

  L1 retention  : cheap, local, no LLM calls. Requires gold answers
                  (and optionally gold-evidence sentence ids).
  L2 generation : paired selected/full generations via a caller-supplied
                  ``generate(prompt_messages) -> str`` function.
  L3 judging    : blinded A/B semantic judging via a caller-supplied
                  ``judge(messages) -> dict`` function returning
                  {"a_correct": bool, "b_correct": bool}. The pack builder
                  strips every arm label; assignment is a deterministic
                  sha256 coin recorded only in the report.

The library never holds API keys; you pass callables. Adapters for any
provider are ~10 lines (see examples/quickstart.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import comb
from statistics import mean
from typing import Callable, Optional

from .gates import Gate
from .text import Document, approx_tokens, coerce_docs, normalize_text


# ------------------------------------------------------------------- stats


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n)


def answer_in_text(gold: str, text: str) -> bool:
    gold_norm = normalize_text(gold)
    return bool(gold_norm) and gold_norm in normalize_text(text)


# ------------------------------------------------------------- thresholds


@dataclass
class RetentionThresholds:
    min_n: int = 100
    min_answer_retention_given_full: float = 0.99
    min_mean_support_coverage: float = 0.98
    min_token_reduction_pct: float = 35.0


@dataclass
class JudgeThresholds:
    min_paired: int = 30
    semantic_delta_floor: float = -0.05


# -------------------------------------------------------------------- L1


def evaluate_retention(
    gate: Gate,
    dataset: list[dict],
    thresholds: RetentionThresholds = RetentionThresholds(),
) -> dict:
    """L1. dataset rows: {"question", "docs", "gold_answer",
    optional "support": [(title, sent_idx), ...]}.

    Only ADMITTED rows are scored (rejected rows fall back to full context
    and carry no risk); the admission rate is reported alongside.
    """
    admitted_rows = []
    admitted = 0
    for row in dataset:
        docs = coerce_docs(row["docs"])
        result = gate.apply(row["question"], docs)
        if not result.admitted:
            continue
        admitted += 1
        full_text = "\n".join(f"{d.title}: {s}" for d in docs for s in d.sentences)
        answer_in_full = answer_in_text(row["gold_answer"], full_text)
        entry = {
            "token_reduction_pct": result.token_reduction_pct,
            "answer_in_full": answer_in_full,
            "answer_retained_given_full": (
                answer_in_text(row["gold_answer"], result.context_text)
                if answer_in_full
                else None
            ),
        }
        support = row.get("support")
        if support:
            kept = {
                (d.title, i)
                for d in result.selected_docs
                for i, s in enumerate(_kept_indices(d, docs))
            }
            kept = _kept_pairs(result.selected_docs, docs)
            covered = len(set(map(tuple, support)) & kept)
            entry["support_coverage"] = covered / len(support)
        admitted_rows.append(entry)

    retained = [
        r["answer_retained_given_full"]
        for r in admitted_rows
        if r["answer_retained_given_full"] is not None
    ]
    coverages = [r["support_coverage"] for r in admitted_rows if "support_coverage" in r]
    summary = {
        "layer": "L1_retention",
        "gate": gate.name,
        "n_total": len(dataset),
        "n_admitted": admitted,
        "admission_rate": round(admitted / len(dataset), 4) if dataset else 0.0,
        "answer_retention_rate_given_full": round(mean(retained), 4) if retained else None,
        "mean_support_coverage": round(mean(coverages), 4) if coverages else None,
        "mean_token_reduction_pct": round(
            mean(r["token_reduction_pct"] for r in admitted_rows), 3
        )
        if admitted_rows
        else 0.0,
        "thresholds": asdict(thresholds),
    }
    checks = {
        "n": admitted >= thresholds.min_n,
        "retention": (summary["answer_retention_rate_given_full"] or 0)
        >= thresholds.min_answer_retention_given_full,
        "reduction": summary["mean_token_reduction_pct"]
        >= thresholds.min_token_reduction_pct,
    }
    if coverages:
        checks["coverage"] = summary["mean_support_coverage"] >= thresholds.min_mean_support_coverage
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    return summary


def _kept_pairs(selected_docs: list[Document], original_docs: list[Document]) -> set:
    """Map selected sentences back to (title, original_sent_idx)."""
    by_title = {d.title: d.sentences for d in original_docs}
    pairs = set()
    for d in selected_docs:
        original = by_title.get(d.title, [])
        cursor = 0
        for s in d.sentences:
            for i in range(cursor, len(original)):
                if original[i] == s:
                    pairs.add((d.title, i))
                    cursor = i + 1
                    break
    return pairs


def _kept_indices(doc, docs):  # retained for backwards clarity; unused path
    return doc.sentences


# -------------------------------------------------------------------- L2


DEFAULT_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. Return a concise "
    "answer. If the context is insufficient, say: insufficient context."
)


def run_generation_pairs(
    gate: Gate,
    dataset: list[dict],
    generate: Callable[[list[dict]], str],
    max_cases: int = 32,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict]:
    """L2. Generates paired selected/full answers for admitted rows."""
    pairs = []
    for row in dataset:
        if len(pairs) >= max_cases:
            break
        docs = coerce_docs(row["docs"])
        result = gate.apply(row["question"], docs)
        if not result.admitted:
            continue
        full_text = "\n".join(f"{d.title}: {s}" for d in docs for s in d.sentences)
        answers = {}
        for arm, context in (("selected", result.context_text), ("full", full_text)):
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{row['question']}\n\nAnswer:",
                },
            ]
            answers[arm] = generate(messages)
        pairs.append(
            {
                "case_id": row.get("id", len(pairs)),
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "selected_answer": answers["selected"],
                "full_answer": answers["full"],
                "token_reduction_pct": result.token_reduction_pct,
            }
        )
    return pairs


# -------------------------------------------------------------------- L3


JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair answer grader. You will see a question, the "
    "gold answer, and two candidate answers labeled A and B. Judge each "
    "candidate on whether it conveys the same answer as the gold answer. "
    "Wording differences, extra explanation, and formatting do not matter; "
    "the conveyed answer entity or choice must match. Return only JSON with "
    'keys: a_correct (boolean), b_correct (boolean).'
)


def _flip(salt: str, case_id) -> int:
    return int(hashlib.sha256(f"{salt}:{case_id}".encode()).hexdigest(), 16) % 2


def build_blinded_pack(pairs: list[dict], salt: str) -> list[dict]:
    """Label-free judge requests; assignment recoverable only via the salt."""
    pack = []
    for p in pairs:
        a_is_selected = _flip(salt, p["case_id"]) == 0
        answer_a = p["selected_answer"] if a_is_selected else p["full_answer"]
        answer_b = p["full_answer"] if a_is_selected else p["selected_answer"]
        pack.append(
            {
                "case_id": p["case_id"],
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{p['question']}\n\nGold answer:\n"
                            f"{p['gold_answer']}\n\nAnswer A:\n{answer_a}\n\n"
                            f"Answer B:\n{answer_b}\n\nReturn only the JSON object."
                        ),
                    },
                ],
            }
        )
    return pack


def score_blinded_verdicts(
    pairs: list[dict],
    verdicts: dict,
    salt: str,
    thresholds: JudgeThresholds = JudgeThresholds(),
) -> dict:
    """Unblind verdicts {case_id: {"a_correct","b_correct"}} and gate them."""
    rows = []
    for p in pairs:
        v = verdicts.get(p["case_id"])
        if v is None:
            continue
        a_is_selected = _flip(salt, p["case_id"]) == 0
        rows.append(
            {
                "selected_correct": bool(v["a_correct"] if a_is_selected else v["b_correct"]),
                "full_correct": bool(v["b_correct"] if a_is_selected else v["a_correct"]),
            }
        )
    n = len(rows)
    sel = mean(1.0 if r["selected_correct"] else 0.0 for r in rows) if rows else 0.0
    ful = mean(1.0 if r["full_correct"] else 0.0 for r in rows) if rows else 0.0
    wins = sum(1 for r in rows if r["selected_correct"] and not r["full_correct"])
    losses = sum(1 for r in rows if r["full_correct"] and not r["selected_correct"])
    delta = round(sel - ful, 4)
    summary = {
        "layer": "L3_blinded_judging",
        "paired": n,
        "selected_semantic_correct_rate": round(sel, 4),
        "full_semantic_correct_rate": round(ful, 4),
        "delta": delta,
        "discordant": {"selected_only": wins, "full_only": losses},
        "sign_test_p_two_sided": float(f"{sign_test_two_sided(wins, losses):.3e}"),
        "thresholds": asdict(thresholds),
    }
    checks = {
        "paired": n >= thresholds.min_paired,
        "non_inferiority": delta >= thresholds.semantic_delta_floor,
    }
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    return summary


# ------------------------------------------------------------------ ladder


def run_ladder(
    gate: Gate,
    dataset: list[dict],
    generate: Optional[Callable] = None,
    judge: Optional[Callable] = None,
    salt: str = "contextgates-v1",
    retention_thresholds: RetentionThresholds = RetentionThresholds(),
    judge_thresholds: JudgeThresholds = JudgeThresholds(),
    max_generation_cases: int = 32,
) -> dict:
    """Run L1 (always) and, if callables are given, L2+L3. Returns an
    EvidenceReport dict; attach it via GateRegistry.register(gate, report)
    only if report["passed"] is True."""
    report = {
        "gate": gate.name,
        "salt": salt,
        "layers": [],
        "passed": False,
    }
    l1 = evaluate_retention(gate, dataset, retention_thresholds)
    report["layers"].append(l1)
    if not l1["passed"]:
        return report
    if generate is None or judge is None:
        report["passed"] = False
        report["note"] = (
            "L1 passed but L2/L3 were not run (no generate/judge provided). "
            "Do not enable the gate on retention evidence alone; the paper's "
            "central caution is that proxy layers can mislead in BOTH directions."
        )
        return report
    pairs = run_generation_pairs(gate, dataset, generate, max_cases=max_generation_cases)
    pack = build_blinded_pack(pairs, salt)
    verdicts = {}
    for item in pack:
        raw = judge(item["messages"])
        if isinstance(raw, str):
            raw = json.loads(raw)
        verdicts[item["case_id"]] = raw
    l3 = score_blinded_verdicts(pairs, verdicts, salt, judge_thresholds)
    report["layers"].append(l3)
    report["passed"] = l3["passed"]
    return report
