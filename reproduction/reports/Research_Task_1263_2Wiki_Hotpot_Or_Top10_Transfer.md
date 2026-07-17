# Research Task #1263

## 2Wiki Transfer of Frozen Hotpot-Or Top10

### Question

Does the #1262 HotpotQA-style explicit candidate comparison candidate transfer
to 2WikiMultiHopQA comparison rows?

### Frozen Candidate

```text
boundary: question contains " or "
selector: lexical_top10
```

### Hypothesis

If #1262 is not a HotpotQA-format artifact, the same cheap admission/selection
policy should preserve answer/support safety and token reduction on 2Wiki
comparison rows.

### Method

Dataset:

```text
data/external/2wikimultihop/dev.json
```

Slice:

```text
start_index: 0
comparison_case_count: 300
accepted_or_count: 207
```

No external API was called. No model was loaded. This is a non-generative
answer/support retention audit.

### Success Gate

```text
n >= 100
answer_retention_rate_given_full >= 0.99
mean_support_coverage >= 0.98
perfect_support_rate >= 0.94
mean_token_reduction_pct >= 35.0
```

### Result

The frozen Hotpot-or top10 candidate failed to transfer to 2Wiki.

```text
claim_decision: hotpot_or_top10_fails_2wiki_transfer
```

Metrics:

```text
n: 207
answer_in_full_rate: 0.9952
answer_in_selected_rate: 0.9807
answer_retention_rate_given_full: 0.9806
mean_support_coverage: 0.9807
perfect_support_rate: 0.9614
strict_safe_rate: 0.9614
mean_token_reduction_pct: 24.931
success: false
```

### Interpretation

This is useful negative evidence.

The #1262 candidate should not be described as a general comparison-question
accelerator. The safer current claim is:

```text
HotpotQA-style explicit candidate comparison may admit cheap lexical_top10
context reduction, but the same frozen policy does not transfer to 2Wiki.
```

The failure is not catastrophic for the whole direction because support metrics
remained fairly high. But the token reduction was too small and answer retention
missed the strict gate.

### Failure / Risk

Remaining weaknesses:

```text
1. Non-generative metrics only.
2. First 300 2Wiki comparison rows only.
3. No semantic judge.
4. No direct baseline against existing compressors.
5. The positive #1262 signal is now more likely dataset-format dependent.
```

### Evidence Update

Previous:

```text
#1262: Hotpot-or + lexical_top10 passed a second frozen HotpotQA holdout.
```

Updated:

```text
#1263: The same frozen policy failed 2Wiki transfer.
```

This changes the candidate from:

```text
non-generative holdout-supported candidate
```

to:

```text
HotpotQA-local non-generative holdout-supported candidate
```

### Suggested Next Experiment

Do not claim transfer yet.

Next legitimate options:

```text
1. Run semantic judge on #1262 only, keeping the claim HotpotQA-local.
2. Search a separately frozen 2Wiki-specific boundary, then require a disjoint
   2Wiki holdout.
3. Compare #1262 against one real compression baseline before any broader claim.
```

Preferred next step:

```text
#1264: HotpotQA-local semantic judge mini-canary
```

That test asks whether the current positive signal survives semantic answer
quality, not merely answer-string presence.
