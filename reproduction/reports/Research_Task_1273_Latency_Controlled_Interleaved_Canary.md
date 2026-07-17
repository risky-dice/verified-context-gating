# Research Task #1273: Latency-Controlled Interleaved Canary

Date: 2026-07-16 KST

## Question

Does selected context reduce live OpenAI wall-clock latency? Last unclaimed
axis. Prior signals were suggestive and methodologically weak: -0.4% (#1264,
blocked ordering), +26% (#1267) and +11% (#1272), both sequential,
uncontrolled, mean-based.

## Method

Harness: `lab/experiments/task1273_latency_controlled_interleaved_canary.py`

24 regime-1 pairs (prompts byte-identical to #1264) + 24 regime-2 pairs
(byte-identical to #1272). Per case ABBA interleaving (each arm timed twice,
order cancels within case), first arm alternating across cases, 2 untimed
warm-ups. gpt-4.1-mini, temperature 0. 192 timed requests, $0.074618 spend
(cap $0.09).

Pre-registered gates per regime: paired >= 20; sign test (selected faster)
two-sided p < 0.05; mean paired reduction >= 10%; completion-token ratio in
[0.80, 1.25] (output-length confound guard).

## Result

```text
claim_decision: latency_acceleration_not_supported
success: false (both regimes fail)

regime 1 (Hotpot-or, ~44% fewer prompt tokens: 780 vs 1384):
  selected faster 14/24, sign test p = 0.54
  mean per-case reduction 2.8%, median 1.4%
  completion ratio 0.998 (comparable outputs)

regime 2 (2Wiki entity, ~63% fewer prompt tokens: 359 vs 979):
  selected faster 12/24, sign test p = 1.0
  mean per-case reduction 6.9%, median 1.1%
  completion ratio 0.876 (comparable outputs)
```

Note the tail effect that fooled earlier tasks: regime 2's grand means
(1068 ms vs 1791 ms) suggest a 40% gap, but the win/loss split is exactly
12/12 and the median per-case reduction is 1.1% — a handful of slow-tail
full-arm calls dominate the mean. This is precisely the artifact the
interleaved paired design exists to expose.

## Interpretation

1. At this context scale (0.4-1.4k prompt tokens) on the OpenAI serving
   stack, cutting 44-63% of input tokens produces a ~1-3% typical latency
   change — indistinguishable from noise. Latency is dominated by fixed
   overhead and queue variance, not prefill.
2. The #1267 (+26%) and #1272 (+11%) signals are now explained as
   drift/tail artifacts of uncontrolled sequential measurement. They should
   never be cited again.
3. The original reviewer stance ("do not claim latency speedup unless
   measured latency improves") is now backed by a controlled measurement,
   not just caution. Prometheus's verified value proposition is COST/TOKEN
   reduction (43-61%) at semantic parity — not wall-clock acceleration.
4. Honest scope note: prefill-bound latency gains may exist at much larger
   contexts (tens of kilotokens) or on latency-transparent local serving
   (#1265 saw 32% locally). Neither is claimed; both are testable later.

## Warning Labels

```text
Moderate evidence (negative): order-controlled paired latency measurement,
  24+24 cases, output-length confound gated.
Retired: all prior "suggestive" latency signals (#1264, #1267, #1272).
Open: large-context latency; local-serving latency (out of current scope).
```

## Files

```text
lab/experiments/task1273_latency_controlled_interleaved_canary.py
task1273_latency_controlled_interleaved_canary_results.json
```
