# Research Task #1261

## Hotpot-Or Compression Depth Sweep

### Question

Can the #1260 frozen Hotpot-or boundary recover the token-reduction gate by
using a more aggressive lexical selector depth?

### Frozen Boundary

```text
dataset == hotpot and question contains " or "
```

### Hypothesis

The #1260 boundary preserved safety but missed the acceleration gate because
`lexical_top12` was too conservative. A slightly more aggressive selector depth
may preserve safety while exceeding 30% token reduction.

### Method

Holdout:

```text
start_index: 3400
comparison_case_count: 240
accepted_hotpot_or_count: 111
```

Swept only selector depth:

```text
lexical_top8
lexical_top10
lexical_top12
```

No model was loaded. No external API was called.

### Success Gate

```text
n >= 100
answer_retention_rate_given_full >= 0.99
mean_support_coverage >= 0.98
perfect_support_rate >= 0.94
mean_token_reduction_pct >= 30.0
```

### Result

Depth sweep found an acceleration candidate.

```text
claim_decision: hotpot_or_depth_sweep_finds_acceleration_candidate
```

Best strategy:

```text
lexical_top10
```

Metrics:

```text
n: 111
answer_retention_rate_given_full: 1.0000
mean_support_coverage: 0.9895
perfect_support_rate: 0.9730
strict_safe_rate: 0.9640
mean_token_reduction_pct: 39.528
mean_selected_page_count: 9.874
success: true
```

Other strategies:

```text
lexical_top8:
answer_retention_rate_given_full: 0.9909
mean_support_coverage: 0.9820
perfect_support_rate: 0.9550
strict_safe_rate: 0.9459
mean_token_reduction_pct: 50.755
success: true

lexical_top12:
answer_retention_rate_given_full: 1.0000
mean_support_coverage: 0.9970
perfect_support_rate: 0.9910
strict_safe_rate: 0.9820
mean_token_reduction_pct: 28.747
success: false
```

### Interpretation

`lexical_top10` is the best current tradeoff:

```text
top12: safest but not enough compression
top10: passes safety and compression
top8: more compression but closer to safety boundary
```

The current candidate is now:

```text
HotpotQA-style explicit candidate comparison
+ lexical_top10 cheap selector
```

### Failure / Risk

This is still not final validation.

Why:

```text
1. Selector depth was searched on the holdout.
2. This is non-generative only.
3. It is HotpotQA-only.
4. It may still be a dataset formatting artifact.
```

### Evidence Update

Previous:

```text
#1260: safety survived, token gate failed.
```

Updated:

```text
#1261: lexical_top10 recovers the token gate while preserving non-generative
safety on the #1260 holdout.
```

### Suggested Next Experiment

Run #1262:

```text
Frozen Hotpot-Or Top10 Second Holdout
```

Protocol:

- Freeze boundary:

```text
dataset == hotpot and question contains " or "
```

- Freeze selector:

```text
lexical_top10
```

- Test on a new disjoint HotpotQA comparison slice.
- Non-generative first.

Success gate:

```text
n >= 100
answer_retention_rate_given_full >= 0.99
mean_support_coverage >= 0.98
perfect_support_rate >= 0.94
mean_token_reduction_pct >= 35.0
```

If #1262 passes, then run local generation or semantic judge.
