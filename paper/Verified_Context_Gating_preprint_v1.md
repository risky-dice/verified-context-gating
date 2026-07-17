# Verified Context Gating: Training-Free Structural Admission for Context Selection, with Pre-Registered Quality Evidence

Author: Yongsun Lee — Independent Researcher (bmt216ays@gmail.com)
Version v1 (post-review revision), 2026-07-17. LaTeX source: latex/main.tex (authoritative).

## Abstract

Context compression for retrieval-augmented generation (RAG) is usually shipped on benchmark averages: a trained pruner or compressor reports aggregate gains, and users must trust that the gains transfer to their workload. We study the opposite contract. A *context gate* is a training-free admission predicate that reads only the question and the candidate documents' titles, admits a query into a narrowly defined *regime*, applies a regime-specific selector, and falls back to the full context otherwise — and every gate must pass an evidence ladder whose thresholds are pre-specified in released harness sources: frozen-split retention holdout, live generation, and blinded LLM judging under two model families. We instantiate two gates on multi-hop QA: an "A or B" comparison gate on HotpotQA (lexical top-10 selection, 45% token reduction at blind-judged semantic parity) and an entity-title gate on 2WikiMultihopQA (named-entity-document selection, 69% reduction). That irrelevant context can hurt LLM answers is well documented; our contribution is a verified, replicated instance under this contract. On a 595-pair holdout prefix we observed post hoc that selected context was semantically *more* accurate than full context; the effect survived a judge-family swap, and on the remaining 386 untouched pairs — with the hypothesis and thresholds fixed before generation — both judge families independently reproduced it (delta +0.057, discordant 29:7, p=3.1e-4). Descriptively, over the full 981-pair holdout: 0.968 vs 0.907, discordant 73:13. At the retention layer on a clean holdout sample, the same training-free gate outperforms the trained pruner Provence at its most conservative shipped setting by 8 points of answer retention and 19 points of evidence coverage while compressing more. Two negative results sharpen the claims: paired exact-match scores were blind to the entire quality gap (identical 0.9866 across arms while a 6-point semantic gap existed), and an order-controlled paired latency canary on a commercial serving stack found no evidence of the >=10% pre-specified wall-clock gain at kilotoken scale (median ~1%) — a caution for acceleration claims transplanted to API deployments. The metered API cost of the full evidence program was approximately one US dollar, suggesting that per-deployment verification — not only better compressors — is a practical missing layer in context engineering.

---

# Introduction

Retrieval-augmented generation pipelines routinely pay for context they
do not need. A line of work addresses this with compression: trained
context pruners and filters (Wang et al. 2023; Xu et al. 2024; Chirkova
et al. 2025; Zhang et al. 2024), perplexity-based prompt
compressors (Jiang et al. 2023, 2024), and adaptive variants. These
systems report strong aggregate results—LongLLMLingua reports accuracy
*gains* at $`4\times`$ compression (Jiang et al. 2024)—and are shipped
plug-and-play. What they do not ship is a per-deployment answer to the
question a practitioner actually faces: *is this optimization safe on my
traffic?*

This paper explores a complementary contract built from three decisions.

#### Decision 1: admission before compression.

Rather than compressing every input, a *gate* first decides whether the
query belongs to a regime where cheap selection is known to be safe. Our
gates are deliberately austere: training-free predicates over the
question string and candidate-document titles, evaluable in
microseconds, with full-context fallback on rejection. Sentence- and
passage-level selection for QA predates LLM-era compression (Choi et al.
2017; Min et al. 2018; Nie et al. 2019); our selectors are deliberately
in that lineage. The contribution is not the selector but the admission
predicate and the verification contract wrapped around it; the research
question is *how far pure question structure can go when it only has to
work inside a verified regime*.

#### Decision 2: verification as a first-class artifact.

A gate is enabled only after passing an evidence ladder whose thresholds
are fixed in the harness source before the data is touched: (i) a
retention-layer holdout on a frozen split never used during gate design;
(ii) live generation; (iii) blinded LLM judging—deterministic A/B
blinding whose assignment is absent from judge inputs—under a three-pass
panel and an external judge family. Search and design work is confined
to a disjoint split. Effects discovered post hoc must survive a
replication on fresh data with the hypothesis fixed in advance before
being claimed. We use “pre-specified” in the precise sense that
thresholds appear in the released harness sources that produced each
run; we did not use an independent timestamping service
(§<a href="#sec:limits" data-reference-type="ref"
data-reference="sec:limits">7</a>).

#### Decision 3: report the negatives.

The evidence program retains its failures: a plausible surface gate
refuted by a matched control
(§<a href="#sec:negatives" data-reference-type="ref"
data-reference="sec:negatives">5.4</a>), a policy that did not transfer
across datasets, a selector family that could not pass gates on a second
dataset, and a controlled null result on latency.

## Contributions

1.  **Two verified gates.** A comparison-form gate on HotpotQA (Yang et
    al. 2018) (45% token reduction at blind-judged semantic parity on
    all 123 frozen holdout cases) and an entity-title gate on
    2WikiMultihopQA (Ho et al. 2020) (69% reduction, $`n{=}981`$
    holdout), each with a deployable admission predicate that uses no
    dataset metadata and no trained model
    (§<a href="#sec:method" data-reference-type="ref"
    data-reference="sec:method">3</a>).

2.  **A pre-specified, judge-replicated instance of distraction-removal
    improving accuracy.** Distraction effects are known (Shi et al.
    2023; Liu et al. 2024); what we add is a replicated instance under
    the verification contract. On the 2Wiki regime, removing the eight
    non-named documents improves semantic accuracy by $`{\sim}6`$ points
    (descriptively $`73{:}13`$ discordant pairs over the full 981-pair
    holdout); the effect was found post hoc, survived a judge-family
    swap, and was reproduced on untouched pairs with hypothesis and
    thresholds fixed before generation
    (§<a href="#sec:superiority" data-reference-type="ref"
    data-reference="sec:superiority">5.2</a>). Mechanistically, full
    context induces distractor-driven comparison errors (dates cited
    correctly, direction flipped) that selection removes.

3.  **A training-free in-regime baseline that beats a trained pruner at
    the retention layer.** On a clean holdout sample of its admitted
    regime, the entity gate outperforms Provence (Chirkova et al. 2025)
    at Provence’s most conservative shipped setting by 8 retention and
    19 coverage points while compressing more; across its threshold
    range Provence reaches neither gate’s operating point on either
    regime. The comparison is a retention-layer proxy and Provence may
    be out-of-domain there; scope caveats in
    §<a href="#sec:provence" data-reference-type="ref"
    data-reference="sec:provence">5.3</a>.

4.  **Two cautionary measurements.** Exact match disagreed with blinded
    semantic judgment in all five paired live evaluations; in the
    decisive large-sample case it was blind to the effect entirely
    (identical $`0.9866`$ across arms vs. a 6.4-point semantic gap), and
    in the replication it understated the effect $`{\sim}7\times`$. An
    order-controlled paired latency canary found no evidence of the
    pre-specified $`\ge`$<!-- -->10% wall-clock gain at kilotoken scale
    (§<a href="#sec:cautions" data-reference-type="ref"
    data-reference="sec:cautions">5.5</a>).

5.  **A micro-budget verification protocol.** The program spans 20
    experiments, $`{\sim}2{,}400`$ live generations and
    $`{\sim}1{,}200`$ external judge calls, for approximately \$1 of
    metered API spend (cost accounting, including what ran on
    subscription agent compute, in
    §<a href="#sec:repro" data-reference-type="ref"
    data-reference="sec:repro">6</a>); all harnesses are released.

We do not claim a general accelerator, cross-domain transfer of any
single policy, or wall-clock speedup. The claim is narrower and, we
believe, more useful: cheap structural admission plus per-regime
verification is a practical contract, and inside verified regimes it can
beat both the full context and a trained pruner at the layers we
measured.

# Related Work

#### Context selection, filtering, and compression.

Selecting minimal context for QA predates LLMs: coarse-to-fine document
and sentence selection (Choi et al. 2017), minimal-context sentence
selection with robustness gains (Min et al. 2018), and semantic sentence
retrieval for multi-hop QA (Nie et al. 2019). LLM-era systems train
filters and pruners—FILCO (Wang et al. 2023), RECOMP (Xu et al. 2024),
AdaComp (Zhang et al. 2024), Provence (Chirkova et al. 2025)—or compress
with a small LM (Jiang et al. 2023, 2024). Our work differs on two axes:
our selectors use no model and no training—admission and selection read
question structure and titles—and every enabled gate carries a
pre-specified, blinded, replicated safety record rather than a benchmark
average. We compare directly against Provence in-regime
(§<a href="#sec:provence" data-reference-type="ref"
data-reference="sec:provence">5.3</a>).

#### Distraction and context sufficiency.

That irrelevant context degrades LLM answers is established (Shi et al.
2023; Liu et al. 2024; Cuconasu et al. 2024; Yoran et al. 2024).
Sufficient Context (Joren et al. 2025) analyzes whether retrieved
context suffices to answer and uses an LLM autorater for selective
abstention. Our gates are a cheaper, structural sibling of the
when-to-act idea: instead of judging sufficiency per query with a model,
we verify a structural regime once and then admit by predicate. Our
contribution to this literature is not the phenomenon but its
contractualization.

#### When-to-act decisions and formal guarantees.

Adaptive-RAG routes queries among retrieval strategies by predicted
complexity (Jeong et al. 2024); Self-RAG learns when to retrieve (Asai
et al. 2024); Self-Route lets the model self-reflect on whether a query
needs long context or focused retrieval (Z. Li et al. 2024); the Context
Awareness Gate decides retrieve-or-not from query–context embedding
statistics (Heydari et al. 2024); and model routing selects among
LLMs (Ong et al. 2024; Chen et al. 2023). These decide *whether to
retrieve or which pipeline to run*; a context gate instead decides
*whether cheap selection within the retrieved set is verified safe*, and
keeps the full context otherwise. The contrast is in *what is verified
and when*: per-query learned or statistical decisions carry
validation-set accuracy, whereas a gate is a per-regime structural
predicate with a pre-specified evidence record and full-context
fallback. Conformal methods provide formal coverage guarantees for
RAG (Kang et al. 2024; S. Li et al. 2024; Mohri and Hashimoto 2024); we
provide empirical pre-specified evidence, not distribution-free
bounds—the two are complementary, and our use of “evidence” rather than
“guarantees” is deliberate.

#### Evaluation.

Exact match misaligns with semantic correctness in the LLM era (Kamalloo
et al. 2023), and LLM judges are the standard remedy (Zheng et al.
2023), with known self-preference risks (Panickssery et al. 2024). We
contribute a sharp, decision-relevant instance: EM identical across arms
while a replicated $`+6`$pt semantic difference existed—i.e., EM was not
merely noisy but blind to the effect being purchased.

# Verified Context Gating

## Gates

A gate $`G=(A,S)`$ is an admission predicate $`A(q,D)`$ over question
$`q`$ and candidate documents $`D`$ (titles only), plus a selector
$`S(q,D)`$ that returns a sub-context. Both instantiated gates are
training-free.

#### Gate 1 (<span class="smallcaps">Hotpot-or</span>).

$`A`$: $`q`$ contains the token sequence “ or ” (explicit candidate
comparison form). $`S`$: rank two-sentence pages by a lexical overlap
score (term overlap $`+`$ title overlap $`+`$ verbatim-title bonus) and
keep the top 10 pages.

#### Gate 2 (<span class="smallcaps">2Wiki-entity</span>).

$`A`$: at least two document titles appear verbatim (lowercased) in
$`q`$; $`q`$ contains no relational-role or kinship word (*director,
performer, …, father, grandfather, husband, …*); $`q`$ is not of yes/no
form (does not start with *are/is/do/does/did/were/was/have/has*).
$`S`$: keep every sentence of the matched-title documents; drop all
other documents. On the search split used for word-list design, the
role-word clause excluded 100% of `bridge_comparison` rows (whose gold
evidence lies outside the named documents) at a 1.5% cost among true
comparisons; on the frozen holdout the predicate admits 981 cases with a
0.5% leak of non-comparison types.

Rejected queries receive the full context. Neither predicate uses
dataset type labels, embeddings, or any learned component.

## The evidence ladder

Every gate must pass, in order, with all thresholds fixed in the harness
source before the data is touched:

- **Retention holdout (frozen split).** Answer-retention-given-full
  $`\ge 0.99`$; mean gold-evidence coverage $`\ge
  0.98`$; perfect-coverage rate $`\ge 0.94`$; mean token reduction
  $`\ge 35\%`$; $`n \ge 100`$. Gate design and any word-list tuning
  happen on a disjoint search split (even dataset indices); the holdout
  (odd indices) is spent exactly once.

- **Live generation.** Paired selected/full generations (temperature 0);
  exact-match non-inferiority as a tripwire only
  (§<a href="#sec:cautions" data-reference-type="ref"
  data-reference="sec:cautions">5.5</a> shows why EM cannot be the
  decision layer).

- **Blinded semantic judging.** Per-case A/B assignment by
  $`\mathrm{sha256}(\text{task salt} \,\|\, \text{case id})
  \bmod 2`$; the judge request file contains no arm labels, token
  counts, or contexts. A panel of three prompt-varied blinded passes
  votes 2-of-3; in practice pass agreement was $`{\ge}99.9\%`$ (100% on
  most packs), so the panel functions as one deterministic judge—the
  meaningful independence check is the external judge family re-judging
  the same blinded pack. Gates: paired $`n`$, semantic non-inferiority
  ($`\Delta \ge -0.05`$), cost reduction $`\ge 30\%`$.

- **Replication with the hypothesis fixed in advance** on untouched
  data, for any effect discovered post hoc.

# Experimental Setup

Datasets: HotpotQA distractor dev (Yang et al. 2018) (Gate 1; the
123-case frozen slice on which the selection policy had previously
passed two independent holdouts) and 2WikiMultihopQA dev (Ho et al.
2020) (Gate 2; even indices for search, all 981 admitted odd-index cases
as holdout). Generation: `gpt-4.1-mini`, temperature 0, max 48 tokens,
identical prompts across arms except the context. Judging: the panel
judge is Claude (`claude-fable-5`), run as three prompt-varied blinded
passes by the orchestrating agent that also ran the experiments—blinding
is structural (sha256 assignment absent from judge inputs), and this
provenance is disclosed as a limitation
(§<a href="#sec:limits" data-reference-type="ref"
data-reference="sec:limits">7</a>); the external judge is `gpt-4.1-mini`
in JSON mode. The external judge shares a family with the generator; we
see no mechanism for arm-asymmetric bias, since self-preference effects
operate on a model’s own generations (Panickssery et al. 2024) and both
arms here are generated by the same model, and cross-family per-label
agreement of 96.3–99.2% bounds the possible effect of either judge’s
idiosyncrasy. Cheap baselines: lead truncation, seeded random pages,
BM25 pages, sentence-granularity lexical selection, and a gold-evidence
oracle, all at per-case matched token budgets. Trained baseline:
Provence (`provence-reranker-debertav3-v1`) as shipped: per-document
pruning over (question, passage, title) with `always_select_title`
enabled, thresholds 0.01–0.5. Metered API spend across the program:
$`{\approx}\$1`$ (itemized in
§<a href="#sec:repro" data-reference-type="ref"
data-reference="sec:repro">6</a>).

# Results

## Both gates pass the ladder

| Gate (L1, frozen holdout) | Retention | Coverage | Reduction | $`n`$ |
|:---|:--:|:--:|:--:|:--:|
| <span class="smallcaps">Hotpot-or</span> | 1.0000 | 0.9897 | 45.2% | 123 |
| <span class="smallcaps">2Wiki-entity</span> | 0.9959 | 0.9980 | 68.9% | 981 |

Retention-layer holdout results (L1).

Blind-judged semantic accuracy (L3), selected vs. full:
<span class="smallcaps">Hotpot-or</span>, all 123 cases: $`0.8862`$
vs. $`0.8618`$ (sign test $`p=0.45`$ $`\rightarrow`$ parity; the
$`+0.024`$ delta is not significant and the gate’s claim is
non-inferiority at cost $`-43\%`$).
<span class="smallcaps">2Wiki-entity</span>, all 981 cases: $`0.9684`$
vs. $`0.9072`$ (§<a href="#sec:superiority" data-reference-type="ref"
data-reference="sec:superiority">5.2</a>; generation cost $`-59`$ to
$`-61\%`$, $`-58.6\%`$ on the replication set). External judge agreement
with the panel: 96.3–99.2% per pack.

## Selection beats full context on the 2Wiki regime

The discovery chain, in the order it actually happened:

- **Stage 1** (595-pair holdout prefix, post hoc): selected $`0.9697`$
  vs. full $`0.9059`$; discordant $`44{:}6`$; sign test
  $`p=3.2\times10^{-8}`$. Exact match on the same pairs: identical
  ($`0.9866`$ both arms).

- **Stage 2** (same pairs, external judge family—a judge-bias check, not
  new data): $`0.9681`$ vs. $`0.9092`$; $`43{:}8`$;
  $`p=6.9\times10^{-7}`$; 99.24% per-label agreement with the panel.

- **Stage 3** (remaining 386 untouched pairs; hypothesis and thresholds
  fixed in the harness before generation): both judge families
  independently: $`\Delta=+0.057`$, discordant $`29{:}7`$,
  $`p=3.1\times10^{-4}`$. The confirmatory estimate agrees with the
  discovery estimate ($`+0.057`$ vs. $`+0.059`$–$`0.064`$).

- **Pooled (descriptive)**, all 981 pairs: $`0.9684`$ vs. $`0.9072`$;
  discordant $`73{:}13`$.

Raw-answer inspection of discordant pairs shows the mechanism: with ten
documents present, the generator cites correct dates but flips the
comparison direction (“X died earlier (1950) than Y (1792)”) or confuses
entities across distractor documents; with only the two named documents,
these errors largely vanish. The reverse failure—selection missing
out-of-scope information—occurs at one-fifth the rate (13 vs. 73).
Because full-context answers are longer, they hit the 48-token
generation cap more often (21 vs. 14 of 595); excluding every
cap-affected discordant pair leaves $`41{:}5`$ ($`p=4.4\times10^{-8}`$),
so truncation does not explain the effect. Failure modes trade, and the
trade favors selection in this regime.

## Training-free gates vs. a trained pruner

At matched per-case token budgets, cheap baselines lose badly on the
Hotpot regime (truncation $`-22`$, random $`-17`$, BM25 $`-2.5`$
retention points vs. Gate 1), and a gold-evidence oracle shows
$`{\sim}86\%`$ reduction is information-theoretically available.

Against Provence, on a clean sample of the 2Wiki *holdout* split (first
300 admitted odd-index cases, never used for gate design; the same
comparison on the search split gives the same decision) and the 123
frozen Hotpot cases:

| Regime | Method | Retention | Coverage | Reduction |
|:---|:---|:--:|:--:|:--:|
| 2Wiki ($`n{=}300`$, holdout) | <span class="smallcaps">2Wiki-entity</span> (ours) | **0.997** | **0.998** | 70.4% |
|  | Provence @0.01 | 0.920 | 0.811 | 62.2% |
|  | Provence @0.1 | 0.673 | 0.458 | 85.0% |
| Hotpot ($`n{=}123`$, frozen) | <span class="smallcaps">Hotpot-or</span> (ours) | **1.000** | **0.990** | 45.2% |
|  | Provence @0.01 | 0.925 | 0.735 | 67.4% |

Trained pruner vs. structural gates at the retention layer. Provence
@0.01 is its most conservative shipped setting.

At Provence’s most conservative setting the gate retains 8 more points
of answers and 19 more points of gold evidence while compressing more.
Qualitatively, at threshold 0.1 Provence pruned a gold document named
verbatim in the question to the empty string (with `always_select_title`
enabled; when no sentence crosses the threshold the document is
dropped): per-sentence relevance scoring discards the attribute
sentences (dates, places) that comparison questions require. The gate
encodes exactly the regime fact the learned scorer misses—comparison
questions need the named documents whole.

Three scope caveats. First, this is a retention-layer proxy comparison;
we did not run generation over Provence-pruned contexts, and
§<a href="#sec:cautions" data-reference-type="ref"
data-reference="sec:cautions">5.5</a> is precisely about proxy
metrics—though the deficit here (19 coverage points) is an order of
magnitude larger than any proxy-vs-semantic disagreement we observed.
Second, string-presence metrics structurally penalize aggressive
sentence pruners; retention given full and gold-sentence coverage are,
however, the same metrics our own gates are held to at L1. Third,
Provence’s training distribution is web-search RAG; its authors claim
domain robustness, but multi-hop comparison questions may still be
out-of-domain for it, and outside admitted regimes the two approaches
are complementary (the gate abstains; a general pruner still applies).

## Negative results that shaped the gates

\(i\) A plausible surface predicate (“question starts with *Which*”) was
refuted by a matched negative control—rejected questions were equally
safe, so the boundary carried no information. (ii) The Hotpot policy
transferred verbatim to 2Wiki fails its gates (retention 0.981,
reduction 24.9%). (iii) On 2Wiki, an exhaustive lexical boundary
$`\times`$ depth sweep (36 cells) has no passing cell: retention
$`\ge 0.99`$ and reduction $`\ge 35\%`$ are individually reachable but
coverage collapses jointly—the geometry (two gold sentences split across
two short documents) defeats depth-shrinking, which is what motivated
the entity-title selector.

## Two measurement cautions

#### Exact match.

EM and blinded semantic judgment disagreed in all five paired live
evaluations. In the two 32-case canaries the EM deltas ($`+0.031`$ and
$`-0.0625`$) rested on one and two discordant cases
respectively—noise-consistent, though the latter would have wrongly
rejected a sound gate under a $`-0.05`$ EM gate. At scale EM was
magnitude-blind: identical $`0.9866`$ across arms while a 6.4-point
semantic gap existed (Stage 1), and $`+0.008`$ vs. a $`+0.057`$ semantic
delta in the replication—a $`{\sim}7\times`$ understatement. EM’s errors
were also asymmetric with context length: self-contradictory
full-context answers contain the gold string and are scored correct.
Deployment decisions about context selection should not be made on EM.

#### Latency.

An order-controlled paired canary ($`24{+}24`$ pairs, per-case ABBA
interleaving, alternating first arm, warm-ups, completion-length gates)
found no evidence of the pre-specified $`\ge`$<!-- -->10% gain: sign
tests $`p=0.54`$ and $`p=1.0`$, median per-case reductions 1.4% and
1.1%, despite 44–63% prompt reduction. At this sample size only large,
consistent effects could have been detected; the point estimates (means
$`+2.8\%`$/$`+6.9\%`$, corroborated by uncontrolled batch means of
$`+2.0\%`$ at $`n{=}595`$ and $`+6.4\%`$ at $`n{=}386`$) suggest any
true effect at this scale is small. Grand means are untrustworthy here:
they differed by up to 40% purely through slow-tail calls, the artifact
that had produced $`+26\%`$ and $`+11\%`$ “signals” in our own
uncontrolled measurements, since retired. At kilotoken scale on a
commercial serving stack, fixed overhead and queue variance dominate
prefill; cost reduction, not wall-clock acceleration, is the honest
value proposition there. We did not test tens-of-kilotoken contexts,
where prefill-bound gains may exist.

# Reproducibility and Cost

All harnesses, per-experiment reports, raw judge verdicts (panel passes
and external), label-free request packs, and result JSONs are released
at <https://github.com/risky-dice/verified-context-gating>. Blinding
salts are deterministic and recorded; frozen splits are defined by
dataset index parity; every gate threshold appears in the harness source
that produced the corresponding run. Metered API spend (gpt-4.1-mini,
usage-based): $`{\approx}\$0.96`$ across the seven live-generation and
three external-judging experiments reported here. Panel judging ran on
subscription agent compute (Claude); replaying all $`{\sim}3{,}500`$
panel judgments through a metered API at gpt-4.1-mini prices would add
$`{\approx}\$0.60`$.

# Limitations

Two regimes, both multi-hop QA benchmarks. 2WikiMultihopQA questions are
template-generated from entity tuples (Ho et al. 2020), which makes
verbatim title matching unusually informative there; the general lesson
is the contract, not the predicate. Regime prevalence in real RAG
traffic is unmeasured (a measurement protocol is included in the
release). Single generation model; the distractor-error mechanism may
attenuate with stronger generators. The panel judge is the same agent
lineage that designed the experiments (blinding is structural, not
organizational), the external judge shares a family with the generator,
and the 86 discordant labels carrying the headline effect were not
human-adjudicated. Pre-specification is attested by released harness
sources, not by an independent timestamping service. Perplexity-based
compressors (LLMLingua family) were not run; the trained-pruner
comparison covers Provence only, at the retention layer. Admission
predicates are English-specific; verbatim-title matching under-admits
paraphrased questions.

# Conclusion

Context selection can be shipped the way safety-critical changes are
shipped: behind a cheap gate, with pre-specified evidence, and with a
fallback. On two multi-hop QA regimes this contract is not a tax—inside
its admitted regime the cheapest possible selector beat the full context
at the semantic layer and a trained pruner at the retention layer, and
the entire metered proof cost about a dollar. We suspect the scarce
resource in context engineering is not compression power but verified
knowledge of when compression is safe.

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-asai2024selfrag" class="csl-entry">

Asai, Akari, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh
Hajishirzi. 2024. “Self-RAG: Learning to Retrieve, Generate, and
Critique Through Self-Reflection.” *Proceedings of ICLR*.

</div>

<div id="ref-chen2023frugalgpt" class="csl-entry">

Chen, Lingjiao, Matei Zaharia, and James Zou. 2023. “FrugalGPT: How to
Use Large Language Models While Reducing Cost and Improving
Performance.” *arXiv Preprint arXiv:2305.05176*.

</div>

<div id="ref-chirkova2025provence" class="csl-entry">

Chirkova, Nadezhda, Thibault Formal, Vassilina Nikoulina, and Stéphane
Clinchant. 2025. “Provence: Efficient and Robust Context Pruning for
Retrieval-Augmented Generation.” *Proceedings of ICLR*.

</div>

<div id="ref-choi2017coarse" class="csl-entry">

Choi, Eunsol, Daniel Hewlett, Jakob Uszkoreit, Illia Polosukhin,
Alexandre Lacoste, and Jonathan Berant. 2017. “Coarse-to-Fine Question
Answering for Long Documents.” *Proceedings of ACL*.

</div>

<div id="ref-cuconasu2024power" class="csl-entry">

Cuconasu, Florin, Giovanni Trappolini, Federico Siciliano, et al. 2024.
“The Power of Noise: Redefining Retrieval for RAG Systems.” *Proceedings
of SIGIR*.

</div>

<div id="ref-heydari2024cag" class="csl-entry">

Heydari, Mohammad Hassan, Arshia Hemmat, Erfan Naman, and Afsaneh
Fatemi. 2024. “Context Awareness Gate for Retrieval Augmented
Generation.” *arXiv Preprint arXiv:2411.16133*.

</div>

<div id="ref-ho2020constructing" class="csl-entry">

Ho, Xanh, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
“Constructing a Multi-Hop QA Dataset for Comprehensive Evaluation of
Reasoning Steps.” *Proceedings of COLING*.

</div>

<div id="ref-jeong2024adaptive" class="csl-entry">

Jeong, Soyeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C.
Park. 2024. “Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large
Language Models Through Question Complexity.” *Proceedings of NAACL*.

</div>

<div id="ref-jiang2023llmlingua" class="csl-entry">

Jiang, Huiqiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu.
2023. “LLMLingua: Compressing Prompts for Accelerated Inference of Large
Language Models.” *Proceedings of EMNLP*.

</div>

<div id="ref-jiang2024longllmlingua" class="csl-entry">

Jiang, Huiqiang, Qianhui Wu, Xufang Luo, et al. 2024. “LongLLMLingua:
Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt
Compression.” *Proceedings of ACL*.

</div>

<div id="ref-joren2025sufficient" class="csl-entry">

Joren, Hailey, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly,
and Cyrus Rajagopal. 2025. “Sufficient Context: A New Lens on Retrieval
Augmented Generation Systems.” *Proceedings of ICLR*.

</div>

<div id="ref-kamalloo2023evaluating" class="csl-entry">

Kamalloo, Ehsan, Nouha Dziri, Charles L. A. Clarke, and Davood Rafiei.
2023. “Evaluating Open-Domain Question Answering in the Era of Large
Language Models.” *Proceedings of ACL*.

</div>

<div id="ref-kang2024crag" class="csl-entry">

Kang, Mintong, Nezihe Merve Gürel, Ning Yu, Dawn Song, and Bo Li. 2024.
“C-RAG: Certified Generation Risks for Retrieval-Augmented Language
Models.” *Proceedings of ICML*.

</div>

<div id="ref-li2024traq" class="csl-entry">

Li, Shuo, Sangdon Park, Insup Lee, and Osbert Bastani. 2024. “TRAQ:
Trustworthy Retrieval Augmented Question Answering via Conformal
Prediction.” *Proceedings of NAACL*.

</div>

<div id="ref-li2024selfroute" class="csl-entry">

Li, Zhuowan, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael
Bendersky. 2024. “Retrieval Augmented Generation or Long-Context LLMs? A
Comprehensive Study and Hybrid Approach.” *Proceedings of EMNLP
(Industry Track)*.

</div>

<div id="ref-liu2024lost" class="csl-entry">

Liu, Nelson F., Kevin Lin, John Hewitt, et al. 2024. “Lost in the
Middle: How Language Models Use Long Contexts.” *Transactions of the
Association for Computational Linguistics* 12.

</div>

<div id="ref-min2018efficient" class="csl-entry">

Min, Sewon, Victor Zhong, Richard Socher, and Caiming Xiong. 2018.
“Efficient and Robust Question Answering from Minimal Context over
Documents.” *Proceedings of ACL*.

</div>

<div id="ref-mohri2024conformal" class="csl-entry">

Mohri, Christopher, and Tatsunori Hashimoto. 2024. “Language Models with
Conformal Factuality Guarantees.” *Proceedings of ICML*.

</div>

<div id="ref-nie2019revealing" class="csl-entry">

Nie, Yixin, Songhe Wang, and Mohit Bansal. 2019. “Revealing the
Importance of Semantic Retrieval for Machine Reading at Scale.”
*Proceedings of EMNLP*.

</div>

<div id="ref-ong2024routellm" class="csl-entry">

Ong, Isaac, Amjad Almahairi, Vincent Wu, et al. 2024. “RouteLLM:
Learning to Route LLMs with Preference Data.” *arXiv Preprint
arXiv:2406.18665*.

</div>

<div id="ref-panickssery2024llm" class="csl-entry">

Panickssery, Arjun, Samuel R. Bowman, and Shi Feng. 2024. “LLM
Evaluators Recognize and Favor Their Own Generations.” *Proceedings of
NeurIPS*.

</div>

<div id="ref-shi2023distracted" class="csl-entry">

Shi, Freda, Xinyun Chen, Kanishka Misra, et al. 2023. “Large Language
Models Can Be Easily Distracted by Irrelevant Context.” *Proceedings of
ICML*.

</div>

<div id="ref-wang2023filco" class="csl-entry">

Wang, Zhiruo, Jun Araki, Zhengbao Jiang, Md Rizwan Parvez, and Graham
Neubig. 2023. “Learning to Filter Context for Retrieval-Augmented
Generation.” *arXiv Preprint arXiv:2311.08377*.

</div>

<div id="ref-xu2024recomp" class="csl-entry">

Xu, Fangyuan, Weijia Shi, and Eunsol Choi. 2024. “RECOMP: Improving
Retrieval-Augmented LMs with Compression and Selective Augmentation.”
*Proceedings of ICLR*.

</div>

<div id="ref-yang2018hotpotqa" class="csl-entry">

Yang, Zhilin, Peng Qi, Saizheng Zhang, et al. 2018. “HotpotQA: A Dataset
for Diverse, Explainable Multi-Hop Question Answering.” *Proceedings of
EMNLP*.

</div>

<div id="ref-yoran2024making" class="csl-entry">

Yoran, Ori, Tomer Wolfson, Ori Ram, and Jonathan Berant. 2024. “Making
Retrieval-Augmented Language Models Robust to Irrelevant Context.”
*Proceedings of ICLR*.

</div>

<div id="ref-zhang2024adacomp" class="csl-entry">

Zhang, Qianchi, Hainan Chen, Lei Cai, et al. 2024. “AdaComp: Extractive
Context Compression with Adaptive Predictor for Retrieval-Augmented
Large Language Models.” *arXiv Preprint arXiv:2409.01579*.

</div>

<div id="ref-zheng2023judging" class="csl-entry">

Zheng, Lianmin, Wei-Lin Chiang, Ying Sheng, et al. 2023. “Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena.” *Proceedings of NeurIPS
Datasets and Benchmarks*.

</div>

</div>
