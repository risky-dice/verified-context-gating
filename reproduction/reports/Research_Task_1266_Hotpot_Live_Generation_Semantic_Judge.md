# Research Task #1266: Hotpot Live Generation Blinded Semantic Judge

Date: 2026-07-15 KST

## Question

Does the #1264 live exact-match result (HotpotQA-local selected context
preserves generated-answer quality vs full context) survive a blinded
semantic judge?

## Motivation

Exact-match can undercount semantically correct answers and can overcount
weak answers that merely contain the gold string. #1264 reported
selected 0.6875 vs full 0.6562 exact-match; that +0.0312 delta could be
exact-match noise rather than real quality parity.

## Method

Input: the 64 live GPT-4.1-mini generated answers (32 selected / 32 full
paired cases) preserved in
`task1264_hotpot_local_semantic_generation_pack_results.json`
(backed up to
`task1264_hotpot_local_semantic_generation_pack_results.live_backup_20260715.json`).

Harness: `lab/experiments/task1266_hotpot_live_generation_semantic_judge.py`

Blinding:

```text
Per case, A/B assignment = sha256("task1266-blind:{index}") % 2
0 -> A=selected, 1 -> A=full
The request file task1266_hotpot_live_generation_semantic_judge_requests.jsonl
contains only question, gold answer, Answer A, Answer B.
It contains no selected/full labels, token counts, or context.
```

Judge: instead of the originally proposed external OpenAI judge, the judging
was performed by a blinded Claude judge panel run locally through Claude Code
subagents (zero external API cost):

```text
3 independent full passes x 32 cases (6 subagents; each pass split 16/16).
Judges were instructed to read only the blinded request file.
Verdict schema per case: a_correct, b_correct, better (A|B|tie|neither), rationale.
Final labels: 2-of-3 majority vote.
```

Raw per-pass verdicts: `task1266_claude_blinded_judge_passes.json`
Majority verdicts: `task1266_claude_blinded_judge_verdicts.jsonl`
Scored output: `task1266_hotpot_live_generation_semantic_judge_results.json`

Pre-registered gates (from the #1265/#1266 handoff, unchanged):

```text
paired_case_count >= 32
selected_semantic_correct_rate >= full_semantic_correct_rate - 0.05
selected_cost_reduction_vs_full >= 30%
```

## Result

```text
claim_decision: blinded_semantic_judge_supports_hotpot_selected_context
success: true
judge_model: claude-fable-5-blinded-3pass-majority
paired_case_count: 32
selected_semantic_correct_rate: 0.8125
full_semantic_correct_rate: 0.8125
selected_minus_full_semantic_rate: 0.0000
better_source_counts: selected 3, full 3, tie 21, neither 5
selected_cost_reduction_vs_full_pct: 37.915
```

Judge reliability:

```text
pairwise correctness agreement across 3 passes: 1.0, 1.0, 1.0
cases with any correctness disagreement: 0
```

Exact-match vs semantic disagreement (per arm):

```text
selected: 4 exact-match-true but semantically wrong, 8 semantically correct but exact-match-false
full:     4 exact-match-true but semantically wrong, 9 semantically correct but exact-match-false
```

Blinding sanity check (asymmetric refusal cases):

```text
index 4982: selected answered correctly, full refused ("Insufficient context.")
index 5470: full answered correctly, selected refused ("Insufficient context.")
The sha flip differed between the two cases and judges scored both
consistently, one refusal charged to each arm.
```

## Interpretation

1. The #1264 quality-preservation result survives semantic judging: selected
   and full context are at exact semantic parity (0.8125 vs 0.8125) on this
   sample, with symmetric better/worse counts (3 vs 3, 21 ties).
2. The #1264 exact-match edge (+0.0312) does NOT survive: it was exact-match
   noise. The supported claim is parity, not improvement. Stop citing the
   positive delta.
3. Exact-match mislabeled roughly 12-13 of 32 answers per arm in both
   directions, but the errors were symmetric across arms, which is why the
   #1264 conclusion direction was still right.
4. All 5 "neither" cases had both arms giving the same wrong answer, and
   the truncation/self-contradiction failures were generator failures, not
   selection failures. Context selection is not the current quality
   bottleneck on this sample.

## Updated Claim

```text
For a HotpotQA-local subset of explicit comparison questions containing "or",
a lexical_top10 selected context preserves generated-answer semantic
correctness at parity with full context (blinded 3-pass judge, 32 paired live
cases) while reducing generation input cost by ~38%.
```

## Warning Labels

```text
Weak evidence: local 1B exact-match generation.
Moderate evidence: live GPT-4.1-mini exact-match generation canary.
Moderate evidence: blinded 3-pass Claude judge over live answers (this task).
Missing evidence: independent external judge model (e.g. GPT-4.1-mini judge).
Missing evidence: strong compressor baseline comparison.
Missing evidence: cross-dataset transfer.
Missing evidence: live API latency gain.
```

Judge-provenance caveat: the judge panel ran inside the same Claude Code
session that orchestrated the experiment. Blinding was enforced structurally
(assignment absent from the request file; deterministic pre-registered sha
coin), but the judge model family is not independent of the tooling. The
harness supports the original external OpenAI judge unchanged:

```text
python3 lab/experiments/task1266_hotpot_live_generation_semantic_judge.py \
  --live --budget-usd 0.03 --model gpt-4.1-mini
# requires explicit user approval + TASK1266_LIVE_APPROVED=yes + OPENAI_API_KEY
# estimated cost: ~$0.0076
```

## Suggested Next Experiments

1. #1267: scale the live generation + blinded judge from the 32-case
   diagnostic sample to all 123 #1262 accepted cases (est. ~$0.06 generation
   + judge, needs approval), to turn the canary into a full-sample result.
2. Optional cheap confirmation: rerun this judge with gpt-4.1-mini
   (~$0.0076) to remove the same-model-family caveat.
3. Compression baseline: compare lexical_top10 against a real compressor
   baseline (e.g. LLMLingua-style token pruning) at matched token budgets,
   since "cheap admission gating" must beat or complement it to matter.
