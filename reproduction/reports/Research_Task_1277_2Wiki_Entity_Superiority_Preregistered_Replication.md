# Research Task #1277: Pre-Registered Superiority Replication (386 Fresh Pairs)

Date: 2026-07-16 KST

## Question

Does the #1275/#1276 superiority finding replicate on fresh data with the
hypothesis and gates fixed BEFORE generation? This closes the last
provenance concern: the hypothesis was born post-hoc on the 595-pair prefix.

## Method

Data: the 386 frozen-holdout pairs untouched by #1275 (complement of its
prefix). Hypothesis and all gates were pre-registered in the harness before
any generation:

```text
G0 paired >= 350
G1(each judge family) non-inferiority: delta >= -0.05
G2(each judge family) superiority: selected ahead, sign test p < 0.05
G3 generation cost reduction >= 30%
Families: Claude 3-pass blinded majority AND external gpt-4.1-mini.
```

Generation: gpt-4.1-mini temp 0, 772 requests, 0 failures, $0.2236.
Blinding: sha256("task1277-blind:{index}"). Claude panel: 12 chunk agents
(6 were interrupted mid-run by a session limit; their on-disk files were
schema/index-validated, and 4 missing chunks re-run after the reset —
recorded in the passes file). Pass agreement: 100%/99.87%/99.87%; 1 case
with any correctness disagreement, resolved by majority. External judge:
386 calls, $0.0649, 0 failures.

## Result

```text
claim_decision: preregistered_superiority_replication_passes
success: true (G0, G1x2, G2x2, G3 all pass)

                     Claude panel        external gpt-4.1-mini
selected rate        0.9663              0.9689
full rate            0.9093              0.9119
delta                +0.057              +0.057
discordant           29 : 7              29 : 7
sign test p          3.1e-04             3.1e-04
cross-family label agreement: 99.22%
generation cost reduction: 58.6%
```

## Interpretation

1. The superiority finding REPLICATES on fresh data under a fully
   pre-registered design, with near-identical effect size (+0.057 vs
   #1275's +0.059-0.064). The discovery chain is now textbook-complete:
   post-hoc observation (#1275) -> judge-family confirmation (#1276) ->
   pre-registered fresh-data replication (#1277).
2. With #1275 + #1277 the ENTIRE 981-pair frozen holdout is now generated
   and blind-judged. Pooled (Claude panel): selected 0.9684 vs full 0.9072;
   combined discordant 73:13.
3. Final claim for regime 2: named-entity-document selection cuts
   generation input cost ~59% AND improves semantic accuracy by ~6 points
   by removing distractor-driven comparison errors — verified across two
   judge families, with the effect pre-registered and replicated.

## Remaining caveats

```text
Single generation model (gpt-4.1-mini) and a single dataset family for
regime 2. External judge shares a family with the generator (symmetric
across arms). These are scope notes, not validity threats to this claim.
```

## Files

```text
lab/experiments/task1277_2wiki_entity_superiority_preregistered_replication.py
task1277_2wiki_entity_replication_generation_results.json
task1277_2wiki_entity_replication_judge_requests.jsonl
task1277_claude_blinded_judge_passes.json
task1277_claude_blinded_judge_verdicts.jsonl
task1277_external_judge_verdicts.jsonl
task1277_2wiki_entity_superiority_replication_results.json
```
