# Research Task #1268: Token-Budget-Matched Cheap Compressor Baseline Ladder

Date: 2026-07-15 KST

## Question

At the same per-case token budget, is lexical_top10 actually competitive with
other cheap context-compression baselines on the 123 frozen #1262 cases — or
would "any compressor at ~55% budget" have produced the #1262-#1267 result?

This is the positioning test demanded by the related-work audit (#1253):
Prometheus claims cheap admission/selection, so the selector must beat naive
baselines and not be dominated by an equally-cheap alternative.

## Method

Harness: `lab/experiments/task1268_matched_budget_compressor_baseline.py`
(fully local, non-generative; imports #1264 page/scoring functions so the
lexical_top10 arm is byte-identical to #1262/#1264).

Per case, budget = lexical_top10's selected-context token count. Baselines
fill greedily in their own rank order without exceeding the budget:

```text
lead_truncation    document order (classic truncation)
random_pages       sha256-seeded random order (lower bound)
bm25_pages         BM25-ranked pages (stronger cheap retriever)
lexical_sentences  sentence-granularity lexical ranking (finer variant)
oracle_support     gold supporting sentences only (reference bound, no gate)
```

Pre-registered gates (fixed in the script before metrics were computed):

```text
G1 naive_margin:  lexical retention >= truncation + 0.05 AND >= random + 0.05
G2 non_dominated: best cheap alternative retention - lexical retention <= 0.02
G3 budget_fair:   every baseline mean tokens <= lexical mean tokens * 1.05
```

## Result

```text
claim_decision: lexical_top10_competitive_at_matched_budget
success: true      (G1, G2, G3 all passed)
n = 123, answer_in_full = 119
```

| method | mean tokens | retention given full | support coverage | perfect support |
|---|---|---|---|---|
| lexical_top10 | 755.5 | **1.0000** | 0.9897 | 0.9675 |
| lexical_sentences | 748.2 | 0.9916 | **0.9932** | **0.9837** |
| bm25_pages | 739.6 | 0.9748 | 0.9379 | 0.8618 |
| random_pages | 738.6 | 0.8319 | 0.5227 | 0.2602 |
| lead_truncation | 740.9 | 0.7815 | 0.5889 | 0.3577 |
| oracle_support (ref) | **103.9** | 1.0000 | 1.0000 | 1.0000 |

## Interpretation

1. The #1262-#1267 result is NOT "any compressor would do": at the identical
   budget, truncation loses 22 points of retention and random loses 17.
   Support coverage roughly halves for both.
2. lexical_top10 also beats BM25 at matched budget (+0.025 retention,
   +0.05 coverage), so the specific cheap scorer is pulling weight.
3. lexical_sentences is a near-equal sibling: slightly lower retention
   (-0.0084) but higher support coverage and perfect-support rate. Sentence
   granularity is a plausible refinement candidate, not a refutation.
4. Oracle headroom is large: gold support sentences average 104 tokens vs
   lexical_top10's 755 (~86% reduction vs our ~45%). The current selector is
   competitive among cheap methods but nowhere near the information-theoretic
   floor. This is the honest upside argument for smarter (non-cheap)
   compression stages downstream of the admission gate.

## Warning Labels

```text
Moderate evidence: matched-budget retention-layer baseline ladder (this task).
Missing evidence: generation-layer baseline comparison (would need live API).
Missing evidence: trained compressor baseline (LLMLingua-family) at matched budget.
Note: oracle_support uses gold labels; reference bound only.
```

## Suggested Next Experiments

1. Latency-controlled interleaved live canary (tests #1267's suggestive 26%).
2. lexical_sentences refinement holdout: frozen replication of the sentence-
   granularity variant (it beat top10 on coverage at equal budget).
3. Optional generation-layer spot-check of lead_truncation on a 16-case
   subsample (~$0.02) to show retention loss translates to answer loss.
