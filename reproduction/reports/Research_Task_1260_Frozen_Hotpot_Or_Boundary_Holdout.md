# Research Task #1260

## Frozen Hotpot-Or Boundary Holdout

### Question

Does the #1259 `hotpot_or_question` admission boundary survive on a disjoint
HotpotQA comparison slice?

### Frozen Boundary

```text
dataset == hotpot and question contains " or "
```

### Hypothesis

If #1259 was not just search overfitting, the frozen boundary should preserve
answer/support safety and maintain at least 30% token reduction on a disjoint
HotpotQA comparison slice.

### Method

Dataset:

```text
data/external/hotpot_dev_distractor_v1.json
```

Holdout:

```text
start_index: 3400
comparison_case_count: 240
```

This is disjoint from the #1243/#1244/#1258 Hotpot range:

```text
previous Hotpot range: 1800..3356
holdout range starts: 3400
```

No model was loaded. No external API was called.

### Success Gate

For accepted rows:

```text
n >= 100
answer_retention_rate_given_full >= 0.99
mean_support_coverage >= 0.98
perfect_support_rate >= 0.94
mean_token_reduction_pct >= 30.0
```

### Result

Holdout failed.

```text
claim_decision: frozen_hotpot_or_boundary_holdout_fails
```

Accepted boundary rows:

```text
n: 111
answer_retention_rate_given_full: 1.0000
mean_support_coverage: 0.9970
perfect_support_rate: 0.9910
strict_safe_rate: 0.9820
mean_token_reduction_pct: 28.747
accepted_rate: 0.4625
```

The failure reason is specific:

```text
quality/safety gates: pass
token reduction gate: fail
required token reduction: >= 30.0
observed token reduction: 28.747
gap: -1.253
```

### Negative Control

Rejected Hotpot non-or rows:

```text
n: 129
answer_retention_rate_given_full: 0.9892
mean_support_coverage: 0.9845
perfect_support_rate: 0.9690
strict_safe_rate: 0.6822
mean_token_reduction_pct: 33.229
```

Accepted rows are safer, but reduce fewer tokens.

### Interpretation

The #1259 candidate did not survive the full gate.

However, it did preserve safety on holdout:

```text
answer retention: 1.0000
support coverage: 0.9970
perfect support: 0.9910
```

This suggests:

```text
hotpot_or_question may be a safety admission boundary,
but not yet a strong acceleration boundary.
```

### Failure / Risk

The current cheap selector is too conservative for the accepted Hotpot-or
boundary. It keeps enough context to preserve support, but token reduction falls
below the 30% gate.

This weakens the acceleration claim.

### Evidence Update

Previous:

```text
#1259 found hotpot_or_question candidate on reused diagnostic rows.
```

Updated:

```text
#1260 holdout shows safety generalizes, but token reduction does not clear the
acceleration gate.
```

### Suggested Next Experiment

Do not change the admission boundary yet.

Next legitimate test:

```text
#1261 Hotpot-Or Compression Depth Sweep
```

Freeze the boundary:

```text
dataset == hotpot and question contains " or "
```

Sweep only selector depth:

```text
lexical_top8
lexical_top10
lexical_top12
```

on the same #1260 holdout.

Purpose:

```text
Find whether a slightly more aggressive cheap selector can keep safety while
reaching >=30% token reduction.
```

If no depth passes, downgrade the track from acceleration candidate to
quality-preserving compression candidate.
