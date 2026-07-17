# Research Task #1275: 2Wiki Entity Regime Large-Sample Generation + Blinded Judge

Date: 2026-07-16 KST

## Question

Does the #1272 canary result hold at scale on the #1271 frozen holdout?

## Budget note (honesty record)

The full 981 pairs cost ~$0.63 at gpt-4.1-mini prices; the earlier ~$0.30
estimate quoted to the user was wrong. Rather than exceed the approved
ceiling ($0.35), this task generated a deterministic index-order PREFIX of
the holdout until the ceiling: 595 complete pairs (61% of the holdout, 64
requests reused from #1272), 0 failures, $0.33 new spend. The remaining 386
pairs can be added later as a clean prefix extension (~$0.25).

## Method

Harness: `lab/experiments/task1275_2wiki_entity_large_sample_generation_judge.py`
(reuses the #1272 context builders; prompts byte-identical).
Blinding: sha256("task1275-blind:{index}"). Judge: blinded Claude panel,
3 passes x 595 cases via 15 subagents writing verdict files directly;
chunk index-sets validated against the request file; 2-of-3 majority.
Pre-registered gates: paired >= 500, semantic delta >= -0.05,
cost reduction >= 30%.

## Result

```text
claim_decision: 2wiki_entity_large_sample_blinded_judge_supports_selected_context
success: true (all gates)
paired_case_count: 595
exact match:    selected 0.9866 vs full 0.9866 (identical)
semantic judge: selected 0.9697 vs full 0.9059 (delta +0.0639)
discordant pairs: selected-only correct 44 vs full-only correct 6
                  sign test two-sided p = 3.2e-08
better counts: selected 64, full 23, tie 496, neither 12
generation cost reduction: 59.7%
judge agreement: 3/3 passes identical on all 1190 correctness labels
latency: -2.0% (consistent with the #1273 negative)
```

## Verified mechanism (raw-answer inspection of discordant cases)

Full-context failures are dominated by DISTRACTOR-INDUCED SELF-CONTRADICTION:
the model cites correct dates but flips the comparison direction, e.g.

```text
index 981 full:     "Imre Wampetich died earlier (1950) than Samson Occom (1792)."
index 981 selected: "Samson Occom died earlier (1792) than Imre Wampetich (1950)."
index 527 full:     "Circus Herman Renz was established first in 1911, while
                     BESIX has been active since 1909."
```

Exact-match scores these full-context answers CORRECT because the gold
entity string appears in the sentence — which is precisely why exact match
was identical (0.9866 = 0.9866) while blinded semantic judging separated
the arms. The reverse failure (selected loses out-of-scope info or makes
its own comparison error) occurred 6 times vs 44.

## Interpretation

1. The pre-registered non-inferiority gates PASS at n=595. That is the
   claimed result.
2. Post-hoc observation (flagged as post-hoc, not pre-registered): selected
   context is semantically SUPERIOR on this sample with p = 3.2e-08 and an
   inspected mechanism — removing the 8 distractor documents removes most
   comparison-direction errors. This amplifies #1272's L3 finding
   (failure modes trade) into: at 2Wiki scale, the trade is 44:6 in favor
   of selection.
3. gpt-4.1-mini exact-match at 2Wiki is saturated AND wrong: it hid a
   6.4-point semantic gap. Fourth consecutive exact-match miscount
   (L1 extended).

## Warning Labels

```text
Moderate evidence: blinded 3-pass judge, n=595 prefix of the frozen holdout.
Post-hoc (needs confirmatory framing): superiority claim; the pre-registered
  gate was non-inferiority. External-judge confirmation of THIS pack
  (~$0.10) is the natural next hardening step, given judges here were the
  Claude panel (external agreement was 96-97% on the three earlier packs).
Prefix covers 61% of the holdout; remaining 386 pairs ~= $0.25.
```

## Files

```text
lab/experiments/task1275_2wiki_entity_large_sample_generation_judge.py
task1275_2wiki_entity_large_sample_generation_results.json
task1275_2wiki_entity_large_sample_judge_requests.jsonl
task1275_claude_blinded_judge_passes.json
task1275_claude_blinded_judge_verdicts.jsonl
task1275_2wiki_entity_large_sample_judge_results.json
```
