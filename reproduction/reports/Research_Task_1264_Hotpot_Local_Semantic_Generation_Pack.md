# Research Task #1264

## Hotpot-Local Semantic Generation Pack / Live Canary

### Question

Does the #1262 HotpotQA-local selected context preserve generated answer
quality compared with full context?

### Why This Test Exists

#1262 only proved non-generative answer/support retention. It did not prove that
a model can answer correctly from the selected context.

After #1263 failed 2Wiki transfer, the claim is now intentionally local:

```text
HotpotQA-local explicit comparison + lexical_top10
```

The next required test is semantic answer quality inside that local boundary.

### Hypothesis

The #1262 selected context preserves generated answer quality versus full
context on a diagnostic HotpotQA-local sample.

### Method

Dry-run only. No external API was called.

Source:

```text
task1262_frozen_hotpot_or_top10_second_holdout_results.json
```

Generated request file:

```text
task1264_hotpot_local_semantic_generation_requests.jsonl
```

Sample:

```text
case_count: 32
request_count: 64
selected/full requests per case
```

Risk-stratified sampling:

```text
support_below_095: 4
low_token_reduction: 6
high_token_reduction: 6
ordinary_pass: 16
```

### Dry-Run Result

```text
claim_decision: semantic_generation_package_prepared_not_run
case_count: 32
request_count: 64
estimated_input_tokens: 67852
estimated_output_tokens: 3072
estimated_total_tokens: 70924
mean_token_reduction_pct: 36.728
mean_selected_context_tokens: 730.969
mean_full_context_tokens: 1301.688
```

### Cost Estimate

Using the official GPT-4.1 mini token prices:

```text
input:  $0.40 / 1M tokens
output: $1.60 / 1M tokens
```

Estimated generation-stage cost:

```text
input  67,852 * 0.40 / 1,000,000 = $0.0271
output  3,072 * 1.60 / 1,000,000 = $0.0049
total approximately $0.032
```

The follow-up judge stage would add additional cost. A safe approval ceiling
for generation plus small judge follow-up is:

```text
$0.05
```

### Failure / Risk

This is not evidence yet.

Limitations:

```text
1. Dry-run only.
2. Requires generation outputs before semantic judging.
3. Diagnostic sample, not a full 123-case run.
4. No compressor baseline.
5. The claim is HotpotQA-local after #1263 failed transfer.
```

### Approval Phrase

If live execution is desired:

```text
#1264 Hotpot-local semantic generation canary의 외부 OpenAI API 전송과 최대 $0.05 비용을 승인한다. live 실행해.
```

### Suggested Next Experiment

Run the live generation canary after approval, then build a blinded selected vs
full semantic judge package from the generated answers.

---

## Live Generation Result

### Approval

The PI approved:

```text
#1264 Hotpot-local semantic generation canary의 외부 OpenAI API 전송과 최대 $0.05 비용을 승인한다. live 실행해.
```

### Live Method

Model:

```text
gpt-4.1-mini
```

Requests:

```text
case_count: 32
request_count: 64
selected/full generation per case
```

Budget ceiling:

```text
$0.05
```

### Result

The live generation canary passed the narrow quality/token gate.

```text
claim_decision: live_generation_canary_supports_hotpot_selected_context
success: true
```

Summary:

```text
generated_request_count: 64
paired_case_count: 32
selected_exact_match_rate: 0.6875
full_exact_match_rate: 0.6562
selected_minus_full_exact_match_rate: +0.0312
mean_token_reduction_pct: 36.728
actual_total_tokens: 65,855
actual_cost_usd: 0.028338
```

Token/cost split:

```text
selected_prompt_tokens: 23,896
full_prompt_tokens: 40,296
selected_input_token_reduction_vs_full: 40.699%

selected_estimated_cost_usd: 0.010854
full_estimated_cost_usd: 0.017483
selected_cost_reduction_vs_full: 37.915%
```

Latency did not improve:

```text
mean_selected_latency_ms: 1150.996
mean_full_latency_ms: 1145.971
latency_reduction_pct: -0.439
```

### Interpretation

This is meaningful but bounded.

Supported:

```text
On this diagnostic HotpotQA-local sample, selected context preserved or slightly
improved exact answer-match versus full context while reducing input tokens and
estimated cost.
```

Not supported:

```text
This does not prove real API latency acceleration.
```

The selected context was slightly slower on mean latency, likely because
single-request API latency is dominated by provider/network variance at this
small scale.

### Failure / Risk

Remaining weaknesses:

```text
1. Exact-match proxy, not semantic judge.
2. Diagnostic 32-case sample, not full 123-case run.
3. HotpotQA-local only.
4. No LongLLMLingua/SARA/AttnComp baseline.
5. No latency win in this live run.
```

### Evidence Update

Before:

```text
#1262: non-generative HotpotQA-local holdout support.
```

After:

```text
#1264: live GPT-4.1-mini generation canary supports selected-context quality
proxy and cost/token reduction, but not latency reduction.
```

### Suggested Next Experiment

Build #1266:

```text
Blinded semantic judge over the #1264 live selected/full generated answers.
```

This is needed because exact-match can undercount semantically correct answers
and can also overstate correctness when the answer string appears in a weak
explanation.
