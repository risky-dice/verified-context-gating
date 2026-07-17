# Verified Context Gating: Training-Free Structural Admission for Context Selection, with Pre-Registered Quality Guarantees

Author: Yongsun Lee — Independent Researcher
Draft v0, 2026-07-17. Target: arXiv preprint (cs.CL).

---

## Abstract

Context compression for retrieval-augmented generation is usually shipped
on benchmark averages: a trained pruner or compressor reports aggregate
gains, and users must trust that the gains transfer to their workload. We
study the opposite contract. A *context gate* is a training-free admission
predicate that reads only the question and the candidate documents' titles,
admits a query into a narrowly defined *regime*, applies a regime-specific
selector, and falls back to the full context otherwise — and every gate
must pass a pre-registered evidence ladder (frozen-split retention holdout,
live generation, blinded LLM judging with two judge families) before it is
enabled. We instantiate two gates on multi-hop QA: an "A or B" comparison
gate on HotpotQA (lexical top-10 selection, 45% token reduction) and an
entity-title gate on 2WikiMultihopQA (named-entity-document selection, 69%
token reduction). Both pass the full ladder. Inside its admitted regime,
the 2Wiki gate is not merely quality-preserving: across the entire frozen
981-pair holdout, selected context is semantically *more* accurate than
full context (0.968 vs 0.907; discordant pairs 73:13), an effect discovered
post-hoc, confirmed under a second judge family, and then replicated on
held-out pairs with the hypothesis pre-registered (29:7, p=3.1e-4). At a
matched compression rate, the same training-free gate outperforms a
state-of-the-art trained pruner (Provence, ICLR 2025) by 10 points of
answer retention and 19 points of evidence coverage on this regime. Two
negative results sharpen the claims: token-matched exact-match scores were
blind to the entire quality gap (identical 0.9866 across arms while a
6-point semantic gap existed), and an order-controlled paired latency
canary found no wall-clock gain at kilotoken scale (~1-3% median), against
the acceleration narratives common in the compression literature. The
total API cost of the full evidence program was under one US dollar,
suggesting that per-deployment verification — not only better compressors —
is a practical missing layer in context engineering.

---

## 1 Introduction

Retrieval-augmented generation (RAG) pipelines routinely pay for context
they do not need. A line of work addresses this with compression: trained
context pruners and filters (FILCO, RECOMP, Provence), perplexity-based
prompt compressors (LLMLingua family), and adaptive variants. These systems
report strong aggregate results — LongLLMLingua reports accuracy *gains* at
4x compression — and are shipped plug-and-play. What they do not ship is a
per-deployment answer to the question a practitioner actually faces:
*is this optimization safe on my traffic?*

This paper explores a complementary contract built from three decisions.

**Decision 1: admission before compression.** Rather than compressing every
input, a *gate* first decides whether the query belongs to a regime where
cheap selection is known to be safe. Our gates are deliberately austere:
training-free predicates over the question string and candidate-document
titles, evaluable in microseconds, with full-context fallback on rejection.
The research question becomes: *how far can pure question structure go?*

**Decision 2: verification as a first-class artifact.** A gate is enabled
only after passing a pre-registered evidence ladder: (i) a retention-layer
holdout on a frozen split never touched during gate design, with gates on
answer retention, gold-evidence coverage, and token reduction; (ii) live
generation; (iii) blinded LLM judging — deterministic A/B blinding whose
assignment is absent from judge inputs — under a 3-pass self-consistent
panel and an external judge family. Search/design work is confined to a
disjoint split. Effects discovered post-hoc must survive a pre-registered
replication on fresh data before being claimed.

**Decision 3: report the negatives.** The evidence program retains its
failures: a plausible surface gate refuted by a control (Section 5.4), a
policy that did not transfer across datasets, a selector family that could
not pass gates on a second dataset, and a controlled null result on
latency.

### Contributions

1. **Two verified gates.** A comparison-form gate on HotpotQA (45% token
   reduction at blind-judged semantic parity on all 123 frozen holdout
   cases) and an entity-title gate on 2WikiMultihopQA (69% reduction,
   n=981 holdout), each with a deployable admission predicate that uses no
   dataset metadata and no trained model (Section 3).
2. **Accuracy *improvement* under selection, with a replicated chain of
   evidence.** On the 2Wiki regime, removing the eight non-named documents
   improves semantic accuracy by ~6 points (73:13 discordant pairs over
   the full 981-pair holdout). The effect was found post-hoc, survived a
   judge-family swap, and passed a fully pre-registered fresh-data
   replication (Section 5.2). Mechanistically, full context induces
   distractor-driven comparison errors (dates cited correctly, direction
   flipped) that selection removes.
3. **A training-free floor that beats a trained pruner in-regime.** At a
   matched compression rate on 2Wiki, Provence (ICLR 2025) loses 10 points
   of retention and 19 points of coverage against the entity gate; across
   its threshold range it reaches neither gate's operating point on either
   regime (Section 5.3).
4. **Two cautionary measurements.** Exact match was wrong in all four live
   evaluations, in different directions, and in the starkest case scored
   both arms identically (0.9866) while a 6-point semantic gap existed —
   i.e., the metric would have erased the deployment-relevant difference.
   An order-controlled paired latency canary (ABBA interleaving, sign
   tests, output-length gates) found no wall-clock gain at kilotoken scale
   despite 44-63% input reduction (Section 5.5).
5. **A micro-budget verification protocol.** The complete program — 14
   experiments, ~2,400 live generations and ~1,200 external judge calls —
   cost under $1 in API spend, with all harnesses released (Section 6).

We do not claim a general accelerator, cross-domain transfer of any single
policy, or wall-clock speedup. The claim is narrower and, we believe, more
useful: cheap structural admission plus per-regime verification is a
practical contract, and inside verified regimes it can beat both the full
context and a trained pruner.

## 2 Related Work

**Context compression and filtering.** LLMLingua and LongLLMLingua compress
prompts with a small LM, reporting up to +21.4 accuracy at 4x compression
and end-to-end speedups; FILCO and RECOMP train filters/extractors that
improve RAG quality while shortening prompts; AdaComp and FltLM adapt
compression per query; Provence (ICLR 2025) casts sentence pruning as
sequence labeling over a DeBERTa reranker and prunes dynamically,
plug-and-play. Our work differs on two axes: our selectors use no model
and no training — admission and selection read question structure and
titles — and every enabled gate carries a pre-registered, blinded,
replicated safety record rather than a benchmark average. We compare
directly against Provence in-regime (Section 5.3).

**Distraction and context sufficiency.** That irrelevant context degrades
LLM answers is established (Shi et al., ICML 2023; lost-in-the-middle and
distractor-position studies, including on 2WikiMultihopQA). Sufficient
Context (ICLR 2025) analyzes whether retrieved context suffices to answer
and uses an LLM autorater for selective abstention. Our gates are a
cheaper, structural sibling of the when-to-act idea: instead of judging
sufficiency per query with a model, we verify a structural regime once and
then admit by predicate. Our contribution to this literature is not the
phenomenon but its contractualization: a replicated, pre-registered
instance in which distractor removal *improves* accuracy at 60%+ cost
reduction, plus the observation that exact match cannot see it.

**Routing.** Query routing for RAG and model routing (RouteLLM, FrugalGPT,
cost-aware RAG routing) select among pipelines using trained predictors of
complexity or cost. Gating differs in granularity and guarantee: routes are
predictions; gates are verified regimes with fallback.

**Evaluation.** LLM-as-judge studies document exact match's misalignment
with semantic correctness. We contribute a sharp, decision-relevant
instance: EM identical across arms while a pre-registered +6pt semantic
difference existed, i.e., EM was not merely noisy but blind to the effect
being purchased.

## 3 Verified Context Gating

### 3.1 Gates

A gate G = (A, S) is an admission predicate A(q, D) over question q and
candidate documents D (titles only), plus a selector S(q, D) that returns a
sub-context. Both instantiated gates are training-free:

**Gate 1 (Hotpot-or).** A: q contains the token sequence " or " (explicit
candidate comparison form). S: rank 2-sentence pages by a lexical overlap
score (term overlap + title overlap + verbatim-title bonus) and keep the
top 10 pages.

**Gate 2 (2Wiki-entity).** A: at least two document titles appear verbatim
(lowercased) in q; q contains no relational-role or kinship word
(director, performer, ..., father, grandfather, husband, ...); q is not of
yes/no form (does not start with are/is/do/does/did/were/was/have/has).
S: keep every sentence of the matched-title documents; drop all other
documents. On the frozen holdout the predicate admits 981 cases with a
0.5% leak of non-comparison types; the role-word clause excludes 100% of
bridge_comparison rows (whose gold evidence lies outside the named
documents) at a 1.5% cost among true comparisons.

Rejected queries receive the full context. Neither predicate uses dataset
type labels, embeddings, or any learned component.

### 3.2 The evidence ladder

Every gate must pass, in order, with all thresholds fixed in the harness
before the data is touched:

```text
L1 Retention holdout (frozen split): answer-retention-given-full >= 0.99,
   mean gold-evidence coverage >= 0.98, perfect-coverage rate >= 0.94,
   mean token reduction >= 35%, n >= 100. Gate design and any word-list
   tuning happen on a disjoint search split (even dataset indices);
   the holdout (odd indices) is spent exactly once.
L2 Live generation: paired selected/full generations (temperature 0),
   exact-match non-inferiority as a tripwire only (Section 5.5 shows why
   EM cannot be the decision layer).
L3 Blinded semantic judging: per-case A/B assignment by
   sha256(task-salt || case-id) mod 2; the judge request file contains no
   arm labels, token counts, or contexts. A 3-pass panel judges
   independently (2-of-3 majority); gates: paired n, semantic
   non-inferiority (delta >= -0.05), cost reduction >= 30%. An external
   judge family re-judges the same blinded pack.
L4 (for any post-hoc effect) pre-registered replication on untouched data.
```

## 4 Experimental Setup

Datasets: HotpotQA distractor dev (regime 1; 123 frozen holdout cases
surviving two prior holdouts) and 2WikiMultihopQA dev (regime 2; even
indices for search, all 981 admitted odd-index cases as holdout).
Generation: gpt-4.1-mini, temperature 0, max 48 tokens, identical prompts
across arms except the context. Judging: blinded panels as in L3; external
judge gpt-4.1-mini in JSON mode (the external judge shares a family with
the generator; this affinity is symmetric across arms and cannot favor
either). Cheap baselines: lead truncation, seeded random pages, BM25
pages, sentence-granularity lexical selection, and a gold-evidence oracle,
all at per-case matched token budgets. Trained baseline: Provence
(provence-reranker-debertav3-v1) as shipped, per-document, thresholds
0.01-0.5. Total live API spend across the program: <$1.

## 5 Results

### 5.1 Both gates pass the ladder

```text
                         retention  coverage  reduction   n     (L1, frozen)
Gate 1 Hotpot-or          1.0000     0.9897     45.2%     123
Gate 2 2Wiki-entity       0.9959     0.9980     68.9%     981

Blind-judged semantic accuracy, selected vs full (L3):
Gate 1, all 123 cases:    0.8862 vs 0.8618  (sign test p=0.45 -> parity;
                          the +0.024 delta is not significant and the
                          gate's claim is non-inferiority at cost -43%)
Gate 2, all 981 cases:    0.9684 vs 0.9072  (Section 5.2; cost -59 to -61%)
External judge agreement with the panel: 96.3-99.2% per pack.
```

### 5.2 Selection beats full context on the 2Wiki regime

The discovery chain, in the order it actually happened:

```text
Stage 1 (595-pair holdout prefix, post-hoc): selected 0.9697 vs full
        0.9059; discordant 44:6; sign test p=3.2e-08. Exact match on the
        same pairs: identical (0.9866 both arms).
Stage 2 (same pairs, external judge family): 0.9681 vs 0.9092; 43:8;
        p=6.9e-07; 99.24% per-label agreement with the panel.
Stage 3 (remaining 386 pairs, hypothesis and gates pre-registered before
        generation): both judge families independently: delta +0.057,
        discordant 29:7, p=3.1e-04.
Pooled, all 981 pairs: 0.9684 vs 0.9072; discordant 73:13.
```

Raw-answer inspection of discordant pairs shows the mechanism: with ten
documents present, the generator cites correct dates but flips the
comparison direction ("X died earlier (1950) than Y (1792)") or confuses
entities across distractor documents; with only the two named documents,
these errors largely vanish. The reverse failure — selection missing
out-of-scope information — occurs at one-fifth the rate (13 vs 73).
Failure modes trade, and the trade favors selection in this regime.

### 5.3 Training-free gates vs a trained pruner

At matched per-case token budgets, cheap baselines lose badly on the
Hotpot regime (truncation -22, random -17, BM25 -2.5 retention points vs
Gate 1), and a gold-evidence oracle shows ~86% reduction is
information-theoretically available. Against Provence as shipped:

```text
              retention / coverage / reduction
2WIKI  gate    0.997 / 0.998 / 66.8%
       provence@0.01   0.900 / 0.810 / 62.7%   <- matched operating point
       provence@0.1    0.673 / 0.459 / 85.6%
HOTPOT gate    1.000 / 0.990 / 45.3%
       provence@0.01   0.925 / 0.735 / 67.4%   (its least aggressive point)
```

At the one directly comparable operating point, the trained pruner loses
10 points of retention and 19 points of coverage. Qualitatively, at its
most conservative shipped threshold Provence pruned a gold document named
verbatim in the question to the empty string: per-sentence relevance
scoring discards the attribute sentences (dates, places) that comparison
questions require. The gate encodes exactly the regime fact the learned
scorer misses — comparison questions need the named documents whole. The
claim is regime-scoped: Provence is out-of-domain here, and outside
admitted regimes the two approaches are complementary (the gate abstains;
a general pruner still applies).

### 5.4 Negative results that shaped the gates

(i) A plausible surface predicate ("question starts with Which") was
refuted by a matched negative control — rejected questions were equally
safe, so the boundary carried no information. (ii) The Hotpot policy
transferred verbatim to 2Wiki fails its gates (retention 0.981, reduction
24.9%). (iii) On 2Wiki, an exhaustive lexical boundary x depth sweep
(36 cells) has no passing cell: retention >= 0.99 and reduction >= 35% are
individually reachable but coverage collapses jointly — the geometry
(two gold sentences split across two short documents) defeats
depth-shrinking, which is what motivated the entity-title selector.

### 5.5 Two measurement cautions

**Exact match.** EM disagreed with blinded semantic judgment in all four
live evaluations, in different directions: optimistic (+0.031 -> 0.000
parity), neutral, pessimistic (-0.0625 -> 0.000, which would have wrongly
rejected a sound gate under a -0.05 EM gate), and blind (identical 0.9866
vs a 6.4-point semantic gap, hiding the paper's main effect). EM's errors
were also asymmetric with context length: self-contradictory full-context
answers contain the gold string and are scored correct. Deployment
decisions about context selection should not be made on EM.

**Latency.** An order-controlled paired canary (24+24 pairs, per-case ABBA
interleaving, alternating first arm, warm-ups, completion-length gates)
found median per-case latency changes of 1.1-1.4% (sign tests p=0.54 and
p=1.0) despite 44-63% prompt reduction — grand means differed by up to 40%
purely through slow-tail calls, the artifact that had produced +26% and
+11% "signals" in our own uncontrolled measurements, since retired. At
kilotoken scale on a commercial serving stack, fixed overhead and queue
variance dominate prefill; cost reduction, not wall-clock acceleration, is
the honest value proposition. We did not test tens-of-kilotoken contexts,
where prefill-bound gains may exist.

## 6 Reproducibility

All harnesses, per-experiment reports, raw judge verdicts (panel passes and
external), request packs (label-free), and result JSONs are released.
Blinding salts are deterministic and recorded; frozen splits are defined by
dataset index parity; every gate threshold appears in the harness source
predating the corresponding run. Total live spend: <$1 (gpt-4.1-mini).

## 7 Limitations

Two regimes, both multi-hop QA benchmarks; regime prevalence in real RAG
traffic is unmeasured (a protocol is included in the release). Single
generation model; the distractor-error mechanism may attenuate with
stronger generators. The external judge shares a model family with the
generator (symmetric across arms). Perplexity-based compressors
(LLMLingua family) were not run; the trained-pruner comparison covers
Provence only, out-of-domain. Admission predicates are English-specific.
Verbatim-title matching under-admits paraphrased questions.

## 8 Conclusion

Context selection can be shipped the way safety-critical changes are
shipped: behind a cheap gate, with pre-registered evidence, and with a
fallback. On two multi-hop QA regimes this contract is not a tax — inside
its admitted regime the cheapest possible selector beat both the full
context and a trained pruner, and the entire proof cost less than a
dollar. We suspect the scarce resource in context engineering is not
compression power but verified knowledge of when compression is safe.

---

## Appendix A: numbers index (internal draft aid, remove before submission)

Evidence report: docs/research/Prometheus_Acceleration_Evidence_Report_2026-07.md
Task index: #1258 #1259-#1262 (gate 1 ladder), #1263 (transfer fail),
#1264/#1266/#1267 (gate 1 generation+judging), #1268 (cheap baselines),
#1269-#1271 (gate 2 ladder), #1272/#1275 (gate 2 generation+judging),
#1276 (external judge), #1277 (pre-registered replication), #1273 (latency),
#1274 (external judge, packs 1-3), #1278 (Provence).
