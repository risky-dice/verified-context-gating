# Research Task #1272: 2Wiki Entity Regime Generative Canary + Blinded Judge

Date: 2026-07-16 KST

## Question

Does regime 2 (#1271: ent2_norel_noyn + sent_full, retention-verified at
n=981, 68.9% token reduction) survive the generation layer? The compression
here is far more aggressive than regime 1's (68.9% vs 45.2%), so retention
safety did not automatically imply generative parity.

## Method

Harness: `lab/experiments/task1272_2wiki_entity_generation_judge.py`
(the exact #1264/#1266 recipe applied to regime 2).

Generation:

```text
32-case stratified sample from the 981 frozen holdout rows
(retention_fail 4, low_reduction 6, high_reduction 6, ordinary 16 —
 deliberately over-weighting risk cases: retention_fail is 0.4% of the
 holdout but 12.5% of this sample, so the sample is pessimistic)
model gpt-4.1-mini, temperature 0; 64 requests, 0 failures
actual spend: $0.019429 (approved ceiling $0.05)
```

Judge: blinded Claude panel, identical protocol to #1266/#1267
(sha256("task1272-blind:{index}") A/B assignment, label-free request file,
3 independent passes, 2-of-3 majority).

Pre-registered gates: paired >= 30, semantic delta >= -0.05,
cost reduction >= 30%.

## Result

Generation layer (exact match):

```text
selected_exact_match_rate: 0.8438
full_exact_match_rate: 0.9062
delta: -0.0625 (passed the -0.10 generation gate but exceeded the -0.05
judge-gate threshold — the blinded judge was the decisive test)
latency observation: selected 1781 ms vs full 2006 ms (-11.2%, not a claim)
```

Judge layer (blinded, 3-pass, 100% pass agreement, 0 disagreements):

```text
claim_decision: 2wiki_entity_regime_blinded_judge_supports_selected_context
success: true
paired_case_count: 32
selected_semantic_correct_rate: 0.875
full_semantic_correct_rate: 0.875
selected_minus_full_semantic_rate: 0.0000
better_source_counts: selected 7, full 4, tie 19, neither 2
selected_cost_reduction_vs_full_pct: 61.195
```

Discordant pairs (2 vs 2, verified against raw answers after unblinding):

```text
selected-only correct:
  157: full context confused entities (named the wrong person with
       self-contradictory dates); selected answered correctly.
  477: full context made a date-comparison error (claimed 1963 earlier
       than 1940); selected answered correctly.
full-only correct:
  811: selected refused ("Insufficient context") — needed info beyond the
       named-entity documents.
  4819: selected named the wrong (historical) publisher — current-publisher
       info lived outside the named docs.
```

## Interpretation

1. Regime 2 is now quality-verified at canary level: exact semantic parity
   (0.875 vs 0.875) on a risk-over-weighted sample, with 61% generation
   cost reduction.
2. The exact-match delta (-0.0625) was proxy noise in the PESSIMISTIC
   direction this time — the third time exact-match has miscounted
   (optimistic in #1264, neutral-noise in #1267, pessimistic here).
   Blinded semantic judging is now clearly the project's decisive layer.
3. Failure modes trade rather than accumulate: full context induces
   distractor confusion and comparison errors (2 cases) at the same rate
   that selected context loses out-of-scope information (2 cases). This is
   a genuinely new observation — aggressive selection did not just "cost
   less at equal quality", it removed distractor-driven errors.
4. Both regimes have now cleared the same evidence ladder:
   frozen retention holdout -> live generation -> blinded semantic judge.

## Warning Labels

```text
Moderate evidence: blinded 3-pass judge over live answers, 32-case canary.
Note: sample deliberately over-weights risk buckets (pessimistic).
Suggestive only: 11% latency reduction (sequential, uncontrolled).
Missing: full-sample (981-case) generative scale-up for regime 2.
Missing: external judge model confirmation (same caveat as #1266/#1267).
Missing: latency-controlled measurement.
```

## Files

```text
lab/experiments/task1272_2wiki_entity_generation_judge.py
task1272_2wiki_entity_generation_results.json
task1272_2wiki_entity_judge_requests.jsonl
task1272_claude_blinded_judge_passes.json
task1272_claude_blinded_judge_verdicts.jsonl
task1272_2wiki_entity_semantic_judge_results.json
```

## Suggested Next Experiments

1. Latency-controlled interleaved canary across BOTH regimes (~$0.09):
   the last unclaimed axis ("accelerator" in the wall-clock sense).
2. Regime-2 full-sample scale-up (981 cases, ~$0.30 — larger; optional
   until a paper/demo needs it).
3. External gpt-4.1-mini judge over all three judge request files (~$0.05)
   to remove the same-model-family caveat in one run.
