# Research Task #1267: Hotpot Full-Sample Live Generation + Blinded Semantic Judge

Date: 2026-07-15 KST

## Question

Does the #1264/#1266 diagnostic result (selected context preserves generated
answer quality at ~parity while cutting input cost) hold on ALL 123 #1262
accepted Hotpot-or cases, not just the 32-case diagnostic sample?

## Method

Harness: `lab/experiments/task1267_hotpot_full_sample_generation_judge.py`
(imports the #1264 context-construction and request-building functions
directly, so prompts are byte-identical to #1264).

Generation:

```text
model: gpt-4.1-mini, temperature 0
cases: all 123 #1262 accepted rows -> 246 selected/full requests
64 requests reused from the #1264 live backup (prompt-identical, same model/params)
182 requests newly generated live; 0 failures
new live spend from usage: $0.089408 (user-approved scale-up; budget ceiling $0.11)
```

Judge: blinded Claude panel via Claude Code subagents, same protocol as #1266:

```text
Blinding: sha256("task1267-blind:{index}") % 2; 0 -> A=selected, 1 -> A=full.
Request file task1267_hotpot_full_sample_judge_requests.jsonl has no
selected/full labels.
3 independent passes x 123 cases (12 subagents; each pass split 31/31/31/30).
Final labels: 2-of-3 majority vote. Zero external judge cost.
```

Pre-registered gates (fixed in the script before judging):

```text
paired_case_count >= 120
selected_semantic_correct_rate >= full_semantic_correct_rate - 0.05
selected_cost_reduction_vs_full >= 30%
```

## Result

```text
claim_decision: full_sample_blinded_judge_supports_hotpot_selected_context
success: true
paired_case_count: 123
selected_semantic_correct_rate: 0.8862
full_semantic_correct_rate: 0.8618
selected_minus_full_semantic_rate: +0.0244
better_source_counts: selected 12, full 5, tie 94, neither 12
selected_cost_reduction_vs_full_pct: 43.142
judge pass agreement: 3/3 passes identical on all 246 correctness labels
```

Discordant pairs and significance:

```text
selected-correct-only: 5 cases (4771, 4953, 4982, 5297, 5438)
full-correct-only:     2 cases (5458, 5470)
sign test 5 vs 2: two-sided p = 0.45 -> NOT significant.
The +0.0244 delta is parity-consistent noise; claim non-inferiority, not
superiority.
```

Exact-match vs semantic disagreement:

```text
selected: 9 exact-true-but-wrong, 19 semantic-true-but-exact-miss
full:     11 exact-true-but-wrong, 19 semantic-true-but-exact-miss
Exact-match mislabeled ~23-24% of answers per arm, again symmetrically.
```

Subset consistency:

```text
reused 32 diagnostic cases: selected 0.8125 vs full 0.8125 (matches #1266 exactly)
new 91 cases:               selected 0.9121 vs full 0.8791
```

Latency observation (same-session new-91 pairs only, sequential requests):

```text
mean selected 1292 ms vs full 1751 ms -> 26.2% reduction
selected faster in 54/91 pairs (59%)
#1264's diagnostic sample showed -0.4% on a different day.
Suggestive of a live latency benefit at larger contexts, but mean-gap is
tail-sensitive, this was not a pre-registered gate, and conditions were not
latency-controlled. Do NOT claim latency acceleration from this.
```

Blinding sanity checks: asymmetric cases 4953 (full refused, selected
answered) and 5470 (selected refused, full answered) were unblinded correctly
and charged one refusal to each arm; sha flips differ across the asymmetric
cases.

## Files

```text
lab/experiments/task1267_hotpot_full_sample_generation_judge.py
task1267_hotpot_full_sample_generation_results.json
task1267_hotpot_full_sample_judge_requests.jsonl
task1267_claude_blinded_judge_passes.json
task1267_claude_blinded_judge_verdicts.jsonl
task1267_hotpot_full_sample_semantic_judge_results.json
```

## Updated Claim (supersedes #1266's sample-level claim)

```text
On all 123 frozen #1262 HotpotQA-local "or"-comparison cases, lexical_top10
selected context is semantically non-inferior to full context for live
GPT-4.1-mini generation (0.8862 vs 0.8618, blinded 3-pass judge, sign test
p=0.45) while reducing generation input cost by ~43%.
```

## Warning Labels

```text
Weak evidence: local 1B exact-match generation.
Moderate evidence: live GPT-4.1-mini exact-match generation (now full-sample).
Moderate evidence: blinded 3-pass Claude judge, full 123-case sample (this task).
Suggestive only: 26% same-session latency reduction on new-91 pairs.
Missing evidence: independent external judge model confirmation.
Missing evidence: strong compressor baseline comparison.
Missing evidence: cross-dataset transfer.
Missing evidence: latency-controlled measurement.
```

Judge-provenance caveat unchanged from #1266: the Claude judge panel ran in
the same tool session; blinding was structural. External gpt-4.1-mini judge
over `task1267_hotpot_full_sample_judge_requests.jsonl` would cost ~$0.03.

## Suggested Next Experiments

1. #1268 compressor baseline: run a token-budget-matched truncation/LLMLingua-
   style baseline against lexical_top10 on the same 123 cases to show the
   cheap selector is competitive with real compressors (the positioning claim
   from the related-work audit).
2. Latency-controlled canary: interleaved A/B request ordering, multiple
   repeats per pair, to test the suggestive 26% latency reduction properly.
3. Optional external judge (~$0.03) to remove the same-model-family caveat
   for both #1266 and #1267 in one run.
