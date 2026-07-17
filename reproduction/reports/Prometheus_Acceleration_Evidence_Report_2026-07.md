# Prometheus Acceleration: Consolidated Evidence Report

Date: 2026-07-16 KST
Scope: Research Tasks #1250-#1274
Status: evidence program for the current claim set complete

---

## 1. Executive Summary

Prometheus set out to find "AI acceleration" via cheap context handling. What
survived three weeks of adversarial evidence production is narrower and more
defensible than the original ambition:

**Supported claims**

```text
C1. Regime 1 (HotpotQA "or"-comparison questions + lexical_top10 selection):
    semantically non-inferior to full context under live GPT-4.1-mini
    generation with blinded judging, at ~43% generation input cost reduction.
    (n=123 frozen cases, full-sample live generation, 3-pass blinded judge.)

C2. Regime 2 (2Wiki entity questions + named-entity-document selection):
    retention-safe on a frozen 981-case holdout (retention 0.9959, support
    coverage 0.9980) at 68.9% token reduction; semantically NON-INFERIOR at
    scale — 595-pair deterministic prefix of the holdout, blinded 3-pass
    judge, selected 0.9697 vs full 0.9059 — at ~60% generation cost
    reduction (#1275, extending the #1272 canary).

C2+. Superiority — discovered, confirmed, and pre-registered-replicated
    (#1275 -> #1276 -> #1277), now covering the ENTIRE 981-pair frozen
    holdout: selected context is semantically SUPERIOR to full context.
    Discovery (595-pair prefix, post-hoc): Claude panel 0.9697 vs 0.9059
    (44:6, p=3.2e-08). Judge-family confirmation (#1276): external
    gpt-4.1-mini 0.9681 vs 0.9092 (43:8, p=6.9e-07; 99.24% label
    agreement). Pre-registered fresh-data replication (#1277, 386 pairs,
    hypothesis and gates fixed before generation): BOTH families
    independently found delta +0.057, discordant 29:7, p=3.1e-04.
    Pooled holdout (Claude panel): 0.9684 vs 0.9072, discordant 73:13.
    Verified mechanism by raw-answer inspection: full context induces
    distractor-driven self-contradictions (dates cited correctly,
    comparison direction flipped) that exact-match cannot see — exact-match
    was near-identical across arms while hiding the ~6-point semantic gap.

C3. The regime-addition methodology is repeatable: diagnose the failure mode
    of a rejected selector family, design an admission+selector pair against
    it, and confirm on a pre-reserved frozen split. Regime 2 was produced by
    one such loop in a single day.

C4. Neither regime's result is attributable to "any compressor would do":
    at matched per-case token budgets, truncation loses 22 points of answer
    retention, random selection 17, and BM25 2.5 (#1268). Against a trained
    pruner (#1278): at Provence's matched operating point on 2Wiki
    (62.7% vs our 66.8% reduction) the ICLR'25 pruner loses ~10 points of
    retention and ~19 points of support coverage versus the structural
    gate; across thresholds 0.01-0.5 no Provence point matches our
    operating point on either regime. Regime-scoped claim: inside admitted
    regimes, question-structure knowledge beats generic learned relevance;
    outside them we have no selector and the approaches are complementary.
```

**Rejected claims (measured, not just cautioned)**

```text
R1. Wall-clock latency acceleration: REJECTED at kilotoken scale on OpenAI
    serving. Order-controlled paired measurement (ABBA interleaving, sign
    test, output-length gate) found ~1-3% typical change (p=0.54 / p=1.0)
    despite 44-63% input reduction. Earlier +26%/+11% signals were
    drift/tail artifacts and are retired.
R2. Cross-dataset transfer of a fixed policy (#1263).
R3. Literal "Which"-prefix as an admission boundary (#1258).
R4. Lexical depth-shrinking as a universal selector family (#1269: 0/36
    cells on 2Wiki).
R5. Any general-purpose accelerator claim.
```

**Value proposition, one line**: structural admission gates that detect
regime membership from the question alone, cut generation input cost 43-61%
at blind-judged quality parity, and fall back to full context otherwise.

---

## 2. Architecture Thesis

Generality is not a property of a single technique; it is coverage
accumulated one verified regime at a time.

```text
request -> [cheap structural admission gates, O(question) cost]
             |- regime 1 matched -> lexical_top10 selected context
             |- regime 2 matched -> named-entity-document context
             |- no match         -> full context (zero-risk fallback)
```

The two verified gates read different structural signals (an "or"
comparison form vs. >=2 verbatim entity titles + no relational word),
supporting the portfolio design: each regime owns its own admission
predicate AND its own selector. #1263 showed policies do not transfer;
#1269-#1271 showed the loop that mints new regimes does.

## 3. Regime 1 Evidence Chain (Hotpot-or + lexical_top10)

```text
#1259 exploratory boundary search  -> candidate found
#1260 frozen holdout               -> safety held, reduction missed
#1261 depth sweep                  -> lexical_top10 selected
#1262 second frozen holdout        -> PASSED (n=123, ret 1.0, red 45.2%)
#1264 live generation (32 cases)   -> exact-match parity, cost -38%
#1266 blinded 3-pass judge         -> semantic parity 0.8125 = 0.8125
#1267 full-sample live scale-up    -> 0.8862 vs 0.8618 (sign test p=0.45
                                      -> parity, NOT superiority), cost -43%
```

## 4. Regime 2 Evidence Chain (2Wiki entity + sent_full)

```text
#1263 Hotpot policy transfer       -> FAILED (the honest starting point)
#1269 lexical boundary x depth     -> 0/36 cells; diagnosed: 2 supporting
                                      facts split across 2 short docs
#1270 entity-aware search          -> comparison rows keep 100% of support
                                      inside question-named docs (~34% of
                                      chars); bridge rows 0% -> role-word
                                      filter; candidate passes search split
#1271 frozen odd-split holdout     -> PASSED (n=981, ret 0.9959, cov 0.9980,
                                      red 68.9%; admission leak 0.5%)
#1272 live canary + blinded judge  -> semantic parity 0.875 = 0.875 on a
                                      risk-over-weighted sample, cost -61%
#1275 large-sample scale-up        -> PASSED (595-pair holdout prefix,
                                      0.9697 vs 0.9059, cost -60%; post-hoc
                                      superiority 44:6, p=3.2e-08 — see C2+)
```

## 5. Baselines and Bounds (#1268)

Matched per-case token budgets, 123 frozen cases, retention layer:

```text
lexical_top10      retention 1.0000  coverage 0.9897   (the candidate)
lexical_sentences  retention 0.9916  coverage 0.9932   (refinement candidate)
bm25_pages         retention 0.9748  coverage 0.9379
random_pages       retention 0.8319  coverage 0.5227
lead_truncation    retention 0.7815  coverage 0.5889
oracle_support     retention 1.0000  at 104 tokens (~86% reduction bound)
```

The oracle gap (86% possible vs 45-69% achieved) is the honest upside
argument for heavier compression stages downstream of the admission gate.

## 6. Measurement Lessons (transferable beyond this project)

```text
L1. Exact-match miscounted in all four live tests, in different
    directions: optimistic (#1264 +0.031 -> 0.0), neutral noise (#1267),
    pessimistic (#1272 -0.0625 -> 0.0), and blind-to-a-real-gap (#1275:
    identical 0.9866 across arms while the semantic gap was 6.4 points).
    A regime would have been wrongly rejected in #1272 and a superiority
    signal entirely missed in #1275 on exact-match alone. Blinded semantic
    judging is the decisive layer.
L2. Grand means lie about latency: #1273 regime-2 means differed 40% while
    the per-case win/loss split was exactly 12/12 — slow-tail calls dominate
    means. Paired interleaved design plus sign tests exposed it.
L3. Failure modes trade rather than accumulate — and at scale the trade
    favors selection: #1272 saw 2:2; #1275 (n=595) saw 44:6 in favor of
    selected. Full context induces distractor-driven comparison errors far
    more often than selection loses out-of-scope information. Aggressive
    selection REMOVED a class of errors, not only cost.
L4. Support-coverage gates caught yes/no-answer retention inflation twice
    (#1269 bridge rows, #1270 v1). Proxy metrics need adversarial gates.
```

## 7. Methodology

All positive claims passed, in order: pre-registered gates fixed before
metrics; exploratory search confined to a search split with the holdout
split reserved in advance; single frozen policy on the untouched holdout;
live generation at temperature 0; blinded A/B judging with deterministic
sha256 assignment absent from judge inputs; 3 independent judge passes with
2-of-3 majority (pass agreement was 100% on all 442 correctness labels
across #1266/#1267/#1272); unblinding spot-verified against raw answers.
Negative results (#1258, #1263, #1269, #1273) are retained with equal
status; v1 mistakes (#1270) are documented rather than overwritten.

Judge provenance: the panel runs on Claude via the orchestrating tool.
Blinding is structural, but the model family is shared — #1274 (external
gpt-4.1-mini judge over all three request packs) addresses this; see the
addendum below.

## 8. Related Work Positioning

LongLLMLingua/AttnComp-style compressors, SARA-style adaptive RAG, and
RouteLLM/FrugalGPT-style routing already cover general compression and
cost-quality routing. Prometheus's contribution is upstream of all of them:
O(question) structural admission that decides WHETHER cheap selection is
safe, with a verified-parity guarantee inside admitted regimes and full
context otherwise. The #1268 ladder shows the cheap selector is not
dominated within its cost class; no claim is made against trained
compressors (untested, see Limitations).

## 9. Limitations and Open Questions

```text
1. Two regimes, both on multi-hop QA benchmarks; no real-traffic evidence.
   The decisive product question is regime prevalence in actual workloads.
2. Regime 2 generative evidence covers a 595-pair deterministic prefix
   (61%) of the 981-case holdout; the remaining 386 pairs cost ~$0.25.
   The C2+ superiority observation is post-hoc and Claude-panel-judged;
   external-judge confirmation of the #1275 pack (~$0.10) is the natural
   hardening step.
3. Trained-pruner comparison covers Provence only (#1278, retention layer,
   out-of-domain for Provence); LLMLingua-family perplexity compressors
   remain untested.
4. Latency untested at tens-of-kilotokens contexts and on local serving
   (#1265 saw 32% locally with a 1B model; weak evidence).
5. Judge model family caveat until #1274 confirms.
6. Single generation model (gpt-4.1-mini) for all live evidence.
```

## 10. Reproduction

Workspace: `/Users/gosu/Desktop/Catleaf/Prometheus`. Each task has a script
under `lab/experiments/task12XX_*.py`, a results JSON at the repo root, and
a report in `docs/research/`. Live tasks require explicit approval env vars
and budget ceilings; total live spend across the entire program
(#1264-#1274) was under $0.40. Judge request packs
(`task126?_*judge_requests.jsonl`, `task1272_*`) are label-free and
reusable with any judge model.

---

## Addendum: #1274 External Judge Confirmation (PASSED)

An external gpt-4.1-mini judge re-judged all 187 blinded A/B cases from the
three label-free request packs ($0.031 spend, 0 failures):

```text
claim_decision: external_judge_confirms_all_blinded_results

pack                     selected  full    delta    agreement w/ Claude panel
#1266 Hotpot canary      0.8438    0.8438  0.0000   96.9% (64 labels)
#1267 Hotpot full        0.8943    0.8618  +0.0325  96.3% (246 labels)
#1272 2Wiki canary       0.8438    0.8438  0.0000   96.9% (64 labels)
```

Every pack passes the same non-inferiority floor (-0.05) under the external
judge; per-label agreement with the Claude 3-pass majority is 96-97%.
Absolute rates shift slightly with judge strictness, but every delta
conclusion is unchanged (parity; #1267's positive delta remains
non-significant and is still not claimed as superiority). The judge
model-family caveat in Section 7 is now closed: limitations item 5 is
resolved, and C1/C2 hold under two independent judge families.
