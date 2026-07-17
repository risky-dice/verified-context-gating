# Research Task #1258

## Non-Which Admission Negative Control

### Question

Is the frozen `Which` gate actually selecting safer cheap-context cases, or
would non-Which comparison rows look just as safe under the same cheap selector?

### Hypothesis

The `Which` gate is meaningful only if accepted `Which` rows are safer than
rejected non-Which rows under the same cheap context selector.

### Method

Applied the same frozen #1251 cheap selector to all rows from:

```text
#1241
#1242
#1243
#1244
```

Groups:

```text
which_accepted: rows where the frozen policy used selected context
nonwhich_rejected: rows where the frozen policy used full context
```

No model was loaded. No external API was called.

Metrics:

```text
answer_retention_rate_given_full
mean_support_coverage
perfect_support_rate
mean_token_reduction_pct
```

### Result

The negative control failed the `Which` gate.

```text
claim_decision: which_admission_not_distinguished_from_nonwhich_control
```

Overall:

```text
nonwhich_rejected:
n: 217
answer_retention_rate_given_full: 0.9935
mean_support_coverage: 0.9806
perfect_support_rate: 0.9447
mean_token_reduction_pct: 31.501

which_accepted:
n: 303
answer_retention_rate_given_full: 1.0000
mean_support_coverage: 0.9340
perfect_support_rate: 0.7954
mean_token_reduction_pct: 28.369
```

Comparative gaps:

```text
answer_retention_gap_accepted_minus_rejected: +0.0065
mean_support_coverage_gap_accepted_minus_rejected: -0.0466
perfect_support_gap_accepted_minus_rejected: -0.1493
mean_token_reduction_gap_accepted_minus_rejected: -3.132
```

### Interpretation

This weakens the `Which`-specific admission claim.

The `Which` gate does not clearly identify safer cheap-context rows. Non-Which
rows are nearly as good on answer retention and better on support coverage.

The strongest remaining positive signal is not:

```text
Which questions are uniquely safe.
```

It is narrower:

```text
Some comparison-style RAG rows may tolerate cheap context reduction, but the
surface-form Which gate is probably not the right admission boundary.
```

### Dataset Breakdown

HotpotQA is especially important:

```text
hotpot:nonwhich_rejected:
n: 194
answer_retention_rate_given_full: 0.9932
mean_support_coverage: 0.9948
perfect_support_rate: 0.9897
mean_token_reduction_pct: 32.167

hotpot:which_accepted:
n: 126
answer_retention_rate_given_full: 1.0000
mean_support_coverage: 1.0000
perfect_support_rate: 1.0000
mean_token_reduction_pct: 34.314
```

Hotpot still looks broadly compression-tolerant, but not specifically because
of the `Which` prefix.

2Wiki remains weak:

```text
2wiki:nonwhich_rejected:
n: 23
mean_support_coverage: 0.8609
perfect_support_rate: 0.5652

2wiki:which_accepted:
n: 177
mean_support_coverage: 0.8870
perfect_support_rate: 0.6497
```

The small non-Which 2Wiki count limits confidence.

### Failure / Risk

This is non-generative and string-level only.

However, as an admission-boundary test, it is damaging:

```text
The current gate is too crude.
```

### Evidence Update

Previous claim:

```text
Question-form-conditioned near-zero admission may identify safe comparison RAG
compression cases.
```

Updated claim:

```text
The Which-prefix gate is not currently distinguished from a non-Which control.
The useful boundary is likely not the literal question prefix.
```

### Suggested Next Experiment

Do not run more `Which`-prefix variants.

Next legitimate direction:

```text
Learn or derive a better admission boundary from cheap observable features:

- dataset/source family
- question comparison structure
- answer candidate count
- title/entity overlap
- support/answer presence proxy
- artifact risk proxy
```

Call this a new admission-boundary search only if it is tested against a
negative control from the start.
