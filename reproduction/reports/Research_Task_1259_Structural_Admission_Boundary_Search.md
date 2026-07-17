# Research Task #1259

## Structural Admission Boundary Search

### Question

If the literal `Which` prefix gate failed, can cheap observable structural
features find a better admission boundary for cheap context reduction?

### Hypothesis

Literal prefix is probably too crude. A better admission boundary may depend on
dataset/source family and comparison structure.

### Method

Searched cheap observable boundaries over #1258 diagnostic rows.

Source:

```text
task1258_nonwhich_admission_negative_control_results.json
```

Cases:

```text
520 total
2Wiki: 200
HotpotQA: 320
```

No model was loaded. No external API was called.

### Success Gate

```text
n >= 80
answer_retention_rate_given_full >= 0.99
mean_support_coverage >= 0.98
perfect_support_rate >= 0.94
mean_token_reduction_pct >= 30.0
```

### Result

Candidate found.

Best boundary:

```text
hotpot_or_question
```

Definition:

```text
dataset == "hotpot" and question contains " or "
```

Metrics:

```text
n: 168
accepted_rate: 0.3231
answer_retention_rate_given_full: 1.0000
mean_support_coverage: 1.0000
perfect_support_rate: 1.0000
strict_safe_rate: 1.0000
mean_token_reduction_pct: 33.948
success: true
```

### Comparison Against Failed Boundaries

```text
literal_which:
n: 303
answer_retention: 0.9967
support_coverage: 0.9340
perfect_support: 0.7954
token_reduction: 28.369
success: false

all_rows:
n: 520
answer_retention: 0.8750
support_coverage: 0.9535
perfect_support: 0.8577
token_reduction: 29.676
success: false

or_question_all:
n: 345
answer_retention: 0.9971
support_coverage: 0.9420
perfect_support: 0.8203
token_reduction: 28.915
success: false
```

### Interpretation

This is a stronger boundary than literal `Which`.

However, it may be a HotpotQA dataset artifact:

```text
HotpotQA comparison questions often encode candidate comparison as "A or B".
```

The current candidate is not a general question-form principle. It is:

```text
HotpotQA-style comparison-with-explicit-candidates admission.
```

### Failure / Risk

The boundary was found on the same diagnostic rows.

Therefore it is not validation evidence.

Risks:

```text
1. It may not transfer to other comparison datasets.
2. It may encode HotpotQA formatting artifacts.
3. It has not been tested with semantic answer judging.
4. It uses dataset/source family, which may not exist in production.
```

### Evidence Update

Previous:

```text
Which prefix gate failed negative control.
```

Updated:

```text
The useful boundary may be explicit candidate comparison within HotpotQA-style
RAG, not literal Which prefix.
```

### Suggested Next Experiment

Run #1260:

```text
Frozen Hotpot-Or Boundary Holdout
```

Protocol:

- Freeze `dataset == hotpot and question contains " or "`.
- Test on a disjoint HotpotQA comparison slice not used in #1243/#1244/#1258.
- Use the same cheap context selector.
- Non-generative first: answer retention, support coverage, token reduction.
- If it passes, then run local generation or semantic judge.

Success gate:

```text
n >= 100
answer_retention_rate_given_full >= 0.99
mean_support_coverage >= 0.98
perfect_support_rate >= 0.94
mean_token_reduction_pct >= 30.0
```
