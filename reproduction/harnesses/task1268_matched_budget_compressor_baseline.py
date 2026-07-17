"""Research Task #1268: Token-budget-matched cheap compressor baseline ladder.

Question:
    At the SAME per-case token budget that lexical_top10 uses, is lexical_top10
    actually competitive with other cheap context-compression baselines on the
    123 frozen #1262 Hotpot-or cases? Or is the #1262-#1267 result just "any
    compressor at ~55% budget would have worked"?

This is the positioning test demanded by the related-work audit (#1253):
Prometheus claims "cheap admission / query-structure gating", so the cheap
selector must at least (a) beat naive baselines and (b) not be dominated by
another equally-cheap selector at matched budget.

Fully local, non-generative. Metrics are context-retention proxies
(answer-in-context, support coverage), NOT generation quality. #1264/#1266/
#1267 established that retention translated to generative semantic parity for
lexical_top10; this task compares selectors on the retention layer only.

Baselines (all matched to lexical_top10's per-case token count, +5% hard cap):
    lead_truncation    pages in original document order until budget
    random_pages       sha256-seeded random page order until budget
    bm25_pages         pages ranked by BM25(question terms) until budget
    lexical_sentences  single sentences ranked by the same lexical score
                       family, greedy until budget (finer granularity)
    oracle_support     supporting-fact sentences only (reference upper bound,
                       uses gold labels - NOT a baseline, excluded from gates)

Pre-registered gates (fixed before any metric is computed):
    G1 naive_margin:  lexical_top10 answer_retention_given_full >=
                      lead_truncation + 0.05 AND >= random_pages + 0.05
    G2 non_dominated: max(bm25_pages, lexical_sentences) retention
                      - lexical_top10 retention <= 0.02
    G3 budget_fair:   every baseline's mean token count <= lexical_top10's
                      mean token count * 1.05

Claim decisions:
    all pass          -> lexical_top10_competitive_at_matched_budget
    G2 fails          -> stronger_cheap_selector_found (named); this is a
                         selector-upgrade finding, not a Prometheus failure
    G1 fails          -> lexical_top10_not_better_than_naive_baselines
                         (would weaken #1262-#1267 positioning)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
SOURCE_1262 = ROOT / "task1262_frozen_hotpot_or_top10_second_holdout_results.json"
OUTPUT = ROOT / "task1268_matched_budget_compressor_baseline_results.json"

BUDGET_TOLERANCE = 1.05
NAIVE_MARGIN = 0.05
DOMINANCE_MARGIN = 0.02


def load_task1264_module():
    spec = importlib.util.spec_from_file_location(
        "task1264",
        Path(__file__).parent / "task1264_hotpot_local_semantic_generation_pack.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T1264 = load_task1264_module()


def page_sentence_keys(row: dict, page: dict, sentences: list[dict]) -> set[tuple[str, int]]:
    keys = set()
    for item in sentences:
        if item["text"] in page["text"]:
            keys.add((item["title"], item["sent_idx"]))
    return keys


def bm25_scores(question: str, pages: list[dict]) -> list[float]:
    q_terms = T1264.terms(question)
    doc_terms = [list(T1264.terms(page["text"]) & q_terms) for page in pages]
    doc_lens = [max(1, len(T1264.terms(page["text"]))) for page in pages]
    avg_len = mean(doc_lens)
    n_docs = len(pages)
    df: dict[str, int] = {}
    for term in q_terms:
        df[term] = sum(1 for page in pages if term in T1264.terms(page["text"]))
    k1, b = 1.5, 0.75
    scores = []
    for page, length in zip(pages, doc_lens):
        page_term_set = T1264.terms(page["text"])
        score = 0.0
        for term in q_terms:
            if term not in page_term_set:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            tf = 1.0  # binary tf at this granularity (2-sentence pages)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg_len))
        scores.append(score)
    return scores


def greedy_pick(order: list[int], pages: list[dict], budget: int) -> list[int]:
    chosen, used = [], 0
    for page_index in order:
        tokens = pages[page_index]["tokens"]
        if chosen and used + tokens > budget:
            continue
        chosen.append(page_index)
        used += tokens
        if used >= budget:
            break
    return chosen


def evaluate_selection(
    row: dict,
    dataset_row: dict,
    pages: list[dict],
    chosen: list[int],
    support_keys: set[tuple[str, int]],
    sentences: list[dict],
) -> dict:
    chosen_pages = [pages[i] for i in sorted(chosen)]
    text = "\n".join(page["text"] for page in chosen_pages)[: T1264.MAX_CONTEXT_CHARS]
    tokens = sum(page["tokens"] for page in chosen_pages)
    covered = set()
    for page in chosen_pages:
        covered |= page_sentence_keys(dataset_row, page, sentences)
    covered &= support_keys
    coverage = len(covered) / len(support_keys) if support_keys else 1.0
    return {
        "tokens": tokens,
        "answer_in_context": T1264.answer_match(row["gold_answer"], text),
        "support_coverage": coverage,
        "perfect_support": coverage >= 1.0,
    }


def sentence_selection(
    row: dict, sentences: list[dict], budget: int
) -> tuple[str, int, set[tuple[str, int]]]:
    scored = []
    for pos, item in enumerate(sentences):
        page_like = {"text": item["text"], "titles": [item["title"]]}
        scored.append((T1264.lexical_score(row["question"], page_like), -pos, pos))
    order = [pos for _s, _tie, pos in sorted(scored, key=lambda t: (-t[0], t[2]))]
    chosen, used = [], 0
    for pos in order:
        tokens = T1264.approx_tokens(sentences[pos]["text"])
        if chosen and used + tokens > budget:
            continue
        chosen.append(pos)
        used += tokens
        if used >= budget:
            break
    chosen.sort()
    text = "\n".join(sentences[pos]["text"] for pos in chosen)[: T1264.MAX_CONTEXT_CHARS]
    keys = {(sentences[pos]["title"], sentences[pos]["sent_idx"]) for pos in chosen}
    return text, used, keys


def main() -> None:
    hotpot_rows = json.loads(T1264.HOTPOT_DATASET.read_text(encoding="utf-8"))
    source_rows = json.loads(SOURCE_1262.read_text(encoding="utf-8"))["rows"]

    methods = [
        "lexical_top10",
        "lead_truncation",
        "random_pages",
        "bm25_pages",
        "lexical_sentences",
        "oracle_support",
    ]
    per_method: dict[str, list[dict]] = {name: [] for name in methods}
    answer_in_full_flags: list[bool] = []

    for row in sorted(source_rows, key=lambda r: r["index"]):
        dataset_row = hotpot_rows[row["index"]]
        sentences = T1264.sentence_rows(dataset_row)
        pages = T1264.build_pages(dataset_row)
        support_keys = {
            (title, sent_idx) for title, sent_idx in dataset_row["supporting_facts"]
        }
        full_text = "\n".join(page["text"] for page in pages)[: T1264.MAX_CONTEXT_CHARS]
        answer_in_full = T1264.answer_match(row["gold_answer"], full_text)
        answer_in_full_flags.append(answer_in_full)

        # lexical_top10: identical to #1262/#1264 (top 10 pages, no budget cut)
        scored = [
            (T1264.lexical_score(row["question"], page), page["page_index"])
            for page in pages
        ]
        top10 = [
            page_index
            for _score, page_index in sorted(scored, key=lambda t: (-t[0], t[1]))[
                : T1264.TOP_K
            ]
        ]
        lex_eval = evaluate_selection(
            row, dataset_row, pages, top10, support_keys, sentences
        )
        budget = lex_eval["tokens"]
        per_method["lexical_top10"].append({**lex_eval, "answer_in_full": answer_in_full})

        # lead truncation: document order
        lead_order = [page["page_index"] for page in pages]
        per_method["lead_truncation"].append(
            {
                **evaluate_selection(
                    row,
                    dataset_row,
                    pages,
                    greedy_pick(lead_order, pages, budget),
                    support_keys,
                    sentences,
                ),
                "answer_in_full": answer_in_full,
            }
        )

        # random pages: deterministic sha-seeded order
        rand_order = sorted(
            (page["page_index"] for page in pages),
            key=lambda i: hashlib.sha256(
                f"task1268-random:{row['index']}:{i}".encode()
            ).hexdigest(),
        )
        per_method["random_pages"].append(
            {
                **evaluate_selection(
                    row,
                    dataset_row,
                    pages,
                    greedy_pick(rand_order, pages, budget),
                    support_keys,
                    sentences,
                ),
                "answer_in_full": answer_in_full,
            }
        )

        # bm25 pages
        scores = bm25_scores(row["question"], pages)
        bm25_order = [
            page_index
            for _s, page_index in sorted(
                ((scores[i], i) for i in range(len(pages))),
                key=lambda t: (-t[0], t[1]),
            )
        ]
        per_method["bm25_pages"].append(
            {
                **evaluate_selection(
                    row,
                    dataset_row,
                    pages,
                    greedy_pick(bm25_order, pages, budget),
                    support_keys,
                    sentences,
                ),
                "answer_in_full": answer_in_full,
            }
        )

        # lexical sentences
        text, used, keys = sentence_selection(row, sentences, budget)
        covered = keys & support_keys
        coverage = len(covered) / len(support_keys) if support_keys else 1.0
        per_method["lexical_sentences"].append(
            {
                "tokens": used,
                "answer_in_context": T1264.answer_match(row["gold_answer"], text),
                "support_coverage": coverage,
                "perfect_support": coverage >= 1.0,
                "answer_in_full": answer_in_full,
            }
        )

        # oracle: supporting-fact sentences only (reference, uses gold labels)
        oracle_rows = [s for s in sentences if (s["title"], s["sent_idx"]) in support_keys]
        oracle_text = "\n".join(s["text"] for s in oracle_rows)
        per_method["oracle_support"].append(
            {
                "tokens": sum(T1264.approx_tokens(s["text"]) for s in oracle_rows),
                "answer_in_context": T1264.answer_match(row["gold_answer"], oracle_text),
                "support_coverage": 1.0 if support_keys else 1.0,
                "perfect_support": True,
                "answer_in_full": answer_in_full,
            }
        )

    def summarize(rows: list[dict]) -> dict:
        given_full = [r for r in rows if r["answer_in_full"]]
        return {
            "n": len(rows),
            "mean_tokens": round(mean(r["tokens"] for r in rows), 3),
            "answer_in_context_rate": round(
                mean(1.0 if r["answer_in_context"] else 0.0 for r in rows), 4
            ),
            "answer_retention_rate_given_full": round(
                mean(1.0 if r["answer_in_context"] else 0.0 for r in given_full), 4
            )
            if given_full
            else 0.0,
            "mean_support_coverage": round(mean(r["support_coverage"] for r in rows), 4),
            "perfect_support_rate": round(
                mean(1.0 if r["perfect_support"] else 0.0 for r in rows), 4
            ),
        }

    summaries = {name: summarize(rows) for name, rows in per_method.items()}
    lex = summaries["lexical_top10"]
    lex_ret = lex["answer_retention_rate_given_full"]

    g1 = (
        lex_ret >= summaries["lead_truncation"]["answer_retention_rate_given_full"] + NAIVE_MARGIN
        and lex_ret >= summaries["random_pages"]["answer_retention_rate_given_full"] + NAIVE_MARGIN
    )
    alt_names = ["bm25_pages", "lexical_sentences"]
    best_alt = max(alt_names, key=lambda n: summaries[n]["answer_retention_rate_given_full"])
    best_alt_ret = summaries[best_alt]["answer_retention_rate_given_full"]
    g2 = best_alt_ret - lex_ret <= DOMINANCE_MARGIN
    g3 = all(
        summaries[name]["mean_tokens"] <= lex["mean_tokens"] * BUDGET_TOLERANCE
        for name in ("lead_truncation", "random_pages", "bm25_pages", "lexical_sentences")
    )

    if g1 and g2 and g3:
        claim = "lexical_top10_competitive_at_matched_budget"
    elif not g2:
        claim = f"stronger_cheap_selector_found_{best_alt}"
    elif not g1:
        claim = "lexical_top10_not_better_than_naive_baselines"
    else:
        claim = "budget_fairness_violated_rerun_needed"

    result = {
        "experiment": "Research Task #1268 Token-Budget-Matched Cheap Compressor Baseline Ladder",
        "hypothesis": (
            "At matched per-case token budgets on the 123 frozen #1262 cases, "
            "lexical_top10 beats naive baselines and is not dominated by other "
            "cheap selectors on answer retention."
        ),
        "mode": "local_non_generative",
        "case_count": len(source_rows),
        "answer_in_full_count": sum(answer_in_full_flags),
        "budget_definition": "per-case token count of lexical_top10 selected context",
        "gates": {
            "G1_naive_margin": {"required_margin": NAIVE_MARGIN, "passed": g1},
            "G2_non_dominated": {
                "allowed_margin": DOMINANCE_MARGIN,
                "best_alternative": best_alt,
                "best_alternative_retention": best_alt_ret,
                "passed": g2,
            },
            "G3_budget_fair": {"tolerance": BUDGET_TOLERANCE, "passed": g3},
        },
        "summaries": summaries,
        "success": g1 and g2 and g3,
        "claim_decision": claim,
        "limitations": [
            "Retention-layer proxies only; no generation in this task.",
            "oracle_support uses gold labels and is a reference bound, not a baseline.",
            "No trained/perplexity compressor (LLMLingua) baseline; cheap selectors only.",
            "BM25 uses binary tf at 2-sentence-page granularity.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
