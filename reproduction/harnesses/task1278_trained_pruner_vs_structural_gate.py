"""Research Task #1278: Trained pruner (Provence) vs. structural gates.

Question:
    Does a trained context pruner (Provence, ICLR 2025, DeBERTa-v3 430M,
    plug-and-play) dominate our training-free structural gates on their own
    admitted regimes? This is the missing arm of the #1268 ladder and the
    load-bearing experiment for the preprint's "training-free floor" claim.

Arms:
    Regime 1 (Hotpot, 123 frozen #1262 cases):
        ours = lexical_top10 (page level, byte-identical to #1262)
        Provence at thresholds {0.1, 0.25, 0.5}, per-document pruning
        (title = document title, always_select_title as shipped).
    Regime 2 (2Wiki, 300 even-split ent2_norel_noyn cases, #1270 convention):
        ours = ent_sent_full (all sentences of question-named docs)
        Provence at the same thresholds.

Metrics per arm (retention layer, as #1268):
    mean_token_reduction_pct, answer_retention_rate_given_full,
    mean_support_coverage, perfect_support_rate.

Pre-registered decision rules (fixed before Provence runs):
    D1 trained_pruner_dominates(regime): some threshold achieves
       reduction >= ours AND retention >= ours AND coverage >= ours.
    D2 structural_gate_holds_floor(regime): no threshold dominates, AND
       every threshold with reduction >= ours - 5pp has retention < 0.99
       or coverage < 0.98 (i.e., cannot match our operating point within
       the project-standard gates).
    Otherwise: mixed_pareto(regime) — report the frontier.

Notes:
    - Provence license is CC BY-NC-ND 4.0: research comparison only.
    - Provence max length 512 tokens -> we prune per document, which is its
      intended retrieved-passage usage.
    - Fully local; no external API calls.

Run with the venv python: .venv-provence/bin/python.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
HOTPOT_SOURCE = ROOT / "task1262_frozen_hotpot_or_top10_second_holdout_results.json"
WIKI_DATASET = ROOT / "data/external/2wikimultihop/dev.json"
OUTPUT = ROOT / "task1278_trained_pruner_vs_structural_gate_results.json"
THRESHOLDS = [0.1, 0.25, 0.5]
WIKI_CASES = 300


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T1264 = load_module("task1264", "task1264_hotpot_local_semantic_generation_pack.py")
S1270 = load_module("task1270", "task1270_2wiki_entity_aware_selector_search.py")
T1263 = S1270.T


def hotpot_cases() -> list[dict]:
    hotpot_rows = json.loads(T1264.HOTPOT_DATASET.read_text(encoding="utf-8"))
    source_rows = json.loads(HOTPOT_SOURCE.read_text(encoding="utf-8"))["rows"]
    cases = []
    for row in sorted(source_rows, key=lambda r: r["index"]):
        dataset_row = hotpot_rows[row["index"]]
        docs = [
            {"title": title, "sentences": sentences}
            for title, sentences in dataset_row["context"]
        ]
        support = {(t, i) for t, i in dataset_row["supporting_facts"]}
        cases.append(
            {
                "regime": "hotpot",
                "index": row["index"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "docs": docs,
                "support": support,
            }
        )
    return cases


def wiki_cases() -> list[dict]:
    rows = json.loads(WIKI_DATASET.read_text(encoding="utf-8"))
    cases = []
    for i, row in enumerate(rows):
        if i % 2:
            continue
        if not S1270.boundary_ent2_norel_noyn(row):
            continue
        docs = [{"title": t, "sentences": s} for t, s in row["context"]]
        support = {(t, si) for t, si in row["supporting_facts"]}
        cases.append(
            {
                "regime": "2wiki",
                "index": i,
                "question": row["question"],
                "gold_answer": row["answer"],
                "docs": docs,
                "support": support,
                "row": row,
            }
        )
        if len(cases) >= WIKI_CASES:
            break
    return cases


def full_text_of(case: dict) -> tuple[str, int]:
    lines = [
        f"{doc['title']}: {sent}" for doc in case["docs"] for sent in doc["sentences"]
    ]
    text = "\n".join(lines)
    return text, sum(T1264.approx_tokens(line) for line in lines)


def evaluate_kept(case: dict, kept: set[tuple[str, int]], kept_tokens: int, full_tokens: int) -> dict:
    kept_lines = [
        f"{doc['title']}: {sent}"
        for doc in case["docs"]
        for idx, sent in enumerate(doc["sentences"])
        if (doc["title"], idx) in kept
    ]
    text = "\n".join(kept_lines)
    full_text, _ = full_text_of(case)
    support = case["support"]
    covered = kept & support
    coverage = len(covered) / len(support) if support else 1.0
    answer_in_full = T1263.answer_present(case["gold_answer"], full_text)
    answer_in_kept = T1263.answer_present(case["gold_answer"], text)
    return {
        "token_reduction_pct": round((full_tokens - kept_tokens) / full_tokens * 100, 3)
        if full_tokens
        else 0.0,
        "support_coverage": coverage,
        "perfect_support": coverage >= 1.0,
        "answer_in_full": answer_in_full,
        "answer_retained_given_full": answer_in_kept if answer_in_full else None,
    }


def ours_hotpot(case: dict) -> dict:
    hotpot_rows = json.loads(T1264.HOTPOT_DATASET.read_text(encoding="utf-8"))
    raise RuntimeError("use precomputed ours_hotpot_all instead")


def summarize(rows: list[dict]) -> dict:
    retained = [r["answer_retained_given_full"] for r in rows if r["answer_retained_given_full"] is not None]
    return {
        "n": len(rows),
        "answer_retention_rate_given_full": round(mean(retained), 4) if retained else 0.0,
        "mean_support_coverage": round(mean(r["support_coverage"] for r in rows), 4),
        "perfect_support_rate": round(mean(1.0 if r["perfect_support"] else 0.0 for r in rows), 4),
        "mean_token_reduction_pct": round(mean(r["token_reduction_pct"] for r in rows), 3),
    }


def run_ours(cases: list[dict]) -> dict[str, list[dict]]:
    """Our selectors, expressed as kept-sentence sets for identical scoring."""
    hotpot_rows = json.loads(T1264.HOTPOT_DATASET.read_text(encoding="utf-8"))
    results: dict[str, list[dict]] = {"hotpot": [], "2wiki": []}
    for case in cases:
        _full_text, full_tokens = full_text_of(case)
        if case["regime"] == "hotpot":
            dataset_row = hotpot_rows[case["index"]]
            pages = T1264.build_pages(dataset_row)
            scored = [
                (T1264.lexical_score(case["question"], page), page["page_index"])
                for page in pages
            ]
            top = {
                pi for _s, pi in sorted(scored, key=lambda t: (-t[0], t[1]))[: T1264.TOP_K]
            }
            sentences = T1264.sentence_rows(dataset_row)
            kept = set()
            kept_tokens = 0
            # pages chunk the global sentence list in order; map back to (title, sent_idx)
            for page_index, start in enumerate(range(0, len(sentences), T1264.PAGE_SENTENCE_COUNT)):
                if page_index in top:
                    for item in sentences[start : start + T1264.PAGE_SENTENCE_COUNT]:
                        kept.add((item["title"], item["sent_idx"]))
                        kept_tokens += T1264.approx_tokens(item["text"])
            results["hotpot"].append(evaluate_kept(case, kept, kept_tokens, full_tokens))
        else:
            titles = S1270.matched_titles(case["row"])
            kept = set()
            kept_tokens = 0
            for doc in case["docs"]:
                if doc["title"] in titles:
                    for idx, sent in enumerate(doc["sentences"]):
                        kept.add((doc["title"], idx))
                        kept_tokens += T1264.approx_tokens(f"{doc['title']}: {sent}")
            results["2wiki"].append(evaluate_kept(case, kept, kept_tokens, full_tokens))
    return results


def run_provence(cases: list[dict], thresholds: list[float]) -> dict[tuple[str, float], list[dict]]:
    from transformers import AutoModel

    provence = AutoModel.from_pretrained(
        "naver/provence-reranker-debertav3-v1", trust_remote_code=True
    )
    out: dict[tuple[str, float], list[dict]] = {
        (case_regime, th): [] for case_regime in ("hotpot", "2wiki") for th in thresholds
    }
    for number, case in enumerate(cases, start=1):
        _full_text, full_tokens = full_text_of(case)
        passages = [" ".join(doc["sentences"]) for doc in case["docs"]]
        titles = [doc["title"] for doc in case["docs"]]
        for th in thresholds:
            result = provence.process(
                question=[case["question"]],
                context=[passages],
                title=[titles],
                threshold=th,
                always_select_title=True,
                enable_warnings=False,
            )
            pruned_list = result["pruned_context"][0]
            kept = set()
            kept_tokens = 0
            for doc, pruned in zip(case["docs"], pruned_list):
                for idx, sent in enumerate(doc["sentences"]):
                    if sent.strip() and sent.strip() in pruned:
                        kept.add((doc["title"], idx))
                        kept_tokens += T1264.approx_tokens(f"{doc['title']}: {sent}")
            out[(case["regime"], th)].append(
                evaluate_kept(case, kept, kept_tokens, full_tokens)
            )
        if number % 50 == 0:
            print(f"provence: {number}/{len(cases)} cases done", flush=True)
    return out


def decide(regime: str, ours: dict, provence_points: dict[float, dict]) -> str:
    dominated = any(
        p["mean_token_reduction_pct"] >= ours["mean_token_reduction_pct"]
        and p["answer_retention_rate_given_full"] >= ours["answer_retention_rate_given_full"]
        and p["mean_support_coverage"] >= ours["mean_support_coverage"]
        for p in provence_points.values()
    )
    if dominated:
        return "trained_pruner_dominates"
    floor_holds = all(
        (p["mean_token_reduction_pct"] < ours["mean_token_reduction_pct"] - 5.0)
        or (p["answer_retention_rate_given_full"] < 0.99)
        or (p["mean_support_coverage"] < 0.98)
        for p in provence_points.values()
    )
    return "structural_gate_holds_floor" if floor_holds else "mixed_pareto"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-provence", action="store_true", help="ours-only dry run")
    args = parser.parse_args()

    cases = hotpot_cases() + wiki_cases()
    ours = run_ours(cases)
    ours_summary = {regime: summarize(rows) for regime, rows in ours.items()}
    print(json.dumps({"ours": ours_summary}, indent=2))

    result = {
        "experiment": "Research Task #1278 Trained Pruner (Provence) vs Structural Gates",
        "provence_model": "naver/provence-reranker-debertav3-v1 (CC BY-NC-ND 4.0; research comparison only)",
        "thresholds": THRESHOLDS,
        "decision_rules": "D1 dominate (all three metrics); D2 floor (no threshold matches our operating point within standard gates); else mixed_pareto",
        "ours": ours_summary,
        "mode": "ours_only" if args.skip_provence else "full",
    }
    if not args.skip_provence:
        provence = run_provence(cases, THRESHOLDS)
        provence_summary = {
            regime: {
                str(th): summarize(provence[(regime, th)]) for th in THRESHOLDS
            }
            for regime in ("hotpot", "2wiki")
        }
        decisions = {
            regime: decide(
                regime,
                ours_summary[regime],
                {th: provence_summary[regime][str(th)] for th in THRESHOLDS},
            )
            for regime in ("hotpot", "2wiki")
        }
        result["provence"] = provence_summary
        result["decisions"] = decisions
        result["claim_decision"] = ";".join(f"{r}:{d}" for r, d in decisions.items())
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "ours"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
