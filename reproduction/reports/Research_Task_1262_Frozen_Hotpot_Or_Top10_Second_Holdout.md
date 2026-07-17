# Research Task #1262

## Frozen Hotpot-Or Top10 Second Holdout

### Question

Does the #1261 candidate survive when both boundary and selector depth are
frozen on a new disjoint HotpotQA comparison slice?

### Frozen Candidate

```text
boundary: dataset == hotpot and question contains " or "
selector: lexical_top10
```

### Hypothesis

If #1261 was not a selector-depth overfit, the frozen candidate should preserve
answer/support safety and token reduction on a new HotpotQA comparison slice.

### Method

Dataset:

```text
data/external/hotpot_dev_distractor_v1.json
```

Second holdout:

```text
start_index: 4200
comparison_case_count: 240
accepted_hotpot_or_count: 123
```

No model was loaded. No external API was called.

### Success Gate

```text
n >= 100
answer_retention_rate_given_full >= 0.99
mean_support_coverage >= 0.98
perfect_support_rate >= 0.94
mean_token_reduction_pct >= 35.0
```

### Result

Second holdout passed.

```text
claim_decision: frozen_hotpot_or_top10_second_holdout_passes
```

Metrics:

```text
n: 123
answer_in_full_rate: 0.9675
answer_in_selected_rate: 0.9756
answer_retention_rate_given_full: 1.0000
mean_support_coverage: 0.9897
perfect_support_rate: 0.9675
strict_safe_rate: 0.9431
mean_token_reduction_pct: 45.248
mean_selected_page_count: 9.837
success: true
```

### Interpretation

This is the strongest non-generative evidence so far.

The candidate survived:

```text
frozen boundary
frozen selector
new disjoint HotpotQA comparison slice
answer/support safety gate
token reduction gate
```

The current candidate is:

```text
HotpotQA-style explicit candidate comparison
+ lexical_top10 cheap selector
```

### Failure / Risk

This is still not proof of a general AI accelerator.

Remaining weaknesses:

```text
1. HotpotQA-only evidence.
2. Non-generative metrics only.
3. Answer-string retention is not semantic answer quality.
4. No real API latency measurement.
5. Boundary may still be dataset-format specific.
```

### Evidence Update

Previous:

```text
#1261: top10 chosen in first holdout depth sweep.
```

Updated:

```text
#1262: frozen hotpot_or + lexical_top10 passed a new second holdout.
```

This upgrades the candidate from:

```text
exploratory boundary
```

to:

```text
non-generative holdout-supported candidate
```

### Suggested Next Experiment

Run #1263:

```text
Hotpot-Or Top10 Local Generation Canary
```

Protocol:

- Sample from #1262 accepted rows.
- Generate selected vs full answers with a local model if available.
- If Qwen3B is unavailable, use a smaller available local model and mark as weak.
- Compare exact answer match first.

Alternative:

```text
#1264 Semantic Judge Mini-Canary
```

Use external OpenAI judge only after explicit approval.
