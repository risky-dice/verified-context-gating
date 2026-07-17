# Research Task #1271: 2Wiki Entity Regime Frozen Holdout

Date: 2026-07-15 KST

## Question

Does the #1270 candidate (ent2_norel_noyn boundary + sent_full entity-doc
selector) pass the project-standard gates on data never touched by any
search? This is the clean test of everything tuned in #1270 v1 -> v2.

## Method

Harness: `lab/experiments/task1271_2wiki_entity_regime_frozen_holdout.py`

Holdout: ALL accepted rows with ODD dataset index. #1269 and #1270 searched
even indices only, so this split is untouched. Single frozen policy — no
sweep, no adjustment. Gates unchanged (n>=100, retention>=0.99,
coverage>=0.98, perfect>=0.94, reduction>=35).

## Result

```text
claim_decision: frozen_2wiki_entity_regime_holdout_passes
success: true
n: 981 (admission audit: comparison 976, compositional 4, inference 1)
answer_in_full_rate: 0.9990
answer_retention_rate_given_full: 0.9959
mean_support_coverage: 0.9980
perfect_support_rate: 0.9959
strict_safe_rate: 0.9949
mean_token_reduction_pct: 68.914
```

Every gate passed with margin, on 3x more cases than any prior holdout in
this project, including the search-split-tuned word lists.

## Interpretation

1. **Prometheus now has a second frozen regime**, and it is stronger than
   the first on every retention-layer axis:

```text
                     Hotpot-or + lexical_top10 (#1262)   2Wiki ent2 + sent_full (#1271)
   n                 123                                 981
   retention         1.0000                              0.9959
   coverage          0.9897                              0.9980
   reduction         45.2%                               68.9%
```

2. The regime-addition methodology is now demonstrably repeatable:
   diagnose failure mode (#1269) -> design selector against it (#1270) ->
   frozen holdout (#1271). The second regime took one day, not eleven tasks.
3. The two regimes use different admission signals (question contains " or "
   vs. >=2 verbatim entity titles + no relational word), supporting the
   portfolio architecture: cheap structural gates, each with its own
   selector, full-context fallback otherwise.
4. The admission predicate is deployable: it reads only the question and the
   candidate documents' titles. Leakage on the holdout was 5/981 (0.5%).

## Evidence level

```text
Moderate evidence: retention-layer frozen holdout, n=981.
Missing: live generative canary for THIS regime (the Hotpot regime needed
         #1264/#1266 before its retention result became a quality claim).
Missing: cross-model, latency, and external-judge evidence, as with Hotpot.
```

## Files

```text
lab/experiments/task1270_2wiki_entity_aware_selector_search.py
task1270_2wiki_entity_aware_selector_search_results.json
lab/experiments/task1271_2wiki_entity_regime_frozen_holdout.py
task1271_2wiki_entity_regime_frozen_holdout_results.json
```

## Suggested Next Experiments

1. #1272 live generative canary for the 2Wiki entity regime: 32-case
   stratified sample, selected vs full, gpt-4.1-mini, then blinded 3-pass
   judge — the exact #1264/#1266 recipe. Estimated ~$0.04; needs approval.
2. Latency-controlled interleaved canary (both regimes, ~$0.09; the 68.9%
   reduction regime is the better latency bet than Hotpot's 45%).
3. Hotpot back-transfer: does an entity-style selector also help Hotpot-or,
   or are the two admission signals genuinely regime-specific?
