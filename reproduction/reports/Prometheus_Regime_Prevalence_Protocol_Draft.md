# Prometheus Regime Prevalence Measurement — Protocol Draft (v0.1)

Date: 2026-07-16 KST
Status: DRAFT — protocol only; no measurement has been run. Written so the
product-track decision can be made with a pre-registered design instead of
an improvised one.

## 1. Question

What fraction of realistic retrieval-augmented (RAG) query traffic is
admitted by the verified Prometheus gates, and what input-cost saving does
that imply end-to-end?

```text
expected_saving = sum over regimes r of
    prevalence(r) x token_reduction(r) x (1 - quality_risk_discount(r))
with verified token_reduction: regime 1 ~43-45%, regime 2 ~61-69%.
```

## 2. The denominator problem (why naive measurement would be wrong)

The verified regimes are defined over QUESTION + CANDIDATE-DOCUMENT pairs:
regime 1 needs an explicit "A or B" comparison form; regime 2 needs >=2
retrieved-document titles appearing verbatim in the question. Raw chat
traffic (WildChat/LMSYS-style) mostly has NO attached documents, so gate
admission there is structurally ~0 — that would measure the wrong
denominator, not low prevalence. The correct denominator is queries entering
a RAG pipeline together with their retrieved candidate sets.

## 3. Candidate denominators (in preference order)

```text
D1. Real service logs (query + retrieved-docs pairs) from an actual RAG
    deployment. Gold standard; requires the user to supply or connect data.
    None currently available in this workspace.
D2. Public RAG/QA benchmark mix EXCLUDING HotpotQA and 2Wiki (circularity
    guard): e.g., Natural Questions, TriviaQA(+retrieval), MS MARCO dev,
    BEIR query sets with their corpora. Requires dataset downloads ->
    explicit user approval per project rules.
D3. Synthetic RAG traffic: sample questions from D2 corpora via an LLM.
    Cheapest but weakest; use only as a sanity check, never as the headline.
```

## 4. Measurement procedure (pre-registered shape)

```text
1. Freeze gate implementations exactly as shipped:
   regime 1: question contains " or " (Hotpot-or form)  [note: this
   admission was validated on HotpotQA distractor format; applying it
   elsewhere is itself a transfer experiment — report per-source]
   regime 2: >=2 candidate-doc titles verbatim in question AND no
   role/kinship word AND not yes/no-form
2. For each source in the mix: run both gates over (question, candidates).
3. Report per source and pooled:
   admission_rate(r1), admission_rate(r2), overlap, neither.
4. For admitted samples, compute the token reduction the regime selector
   would have achieved (retention-layer only; no generation cost).
5. Quality spot-check: 32-case generative canary per NEW source before any
   savings claim on that source (the #1263 lesson: policies do not transfer
   on trust).
```

## 5. Pre-registered decision framing (thresholds TBD by user)

```text
pooled admitted-traffic saving < 3%   -> paper-only result; stop product track
3-10%                                 -> niche accelerator; viable as a
                                         RAG-pipeline plugin, not a headline
> 10%                                 -> product-worthy; invest in regime
                                         mining loop as the core asset
```

The regime-mining loop (#1269 -> #1270 -> #1271 pattern) changes this
calculus: prevalence is not fixed — each new regime adds coverage. The real
product question is the MARGINAL COST OF A NEW REGIME (one day, ~$0.05 at
current cadence) versus its traffic share.

## 6. Known pitfalls to guard in the real run

```text
P1. Benchmark circularity: exclude Hotpot/2Wiki from the denominator.
P2. Title-matching convention dependence: regime 2's ENT2 predicate assumes
    docs carry meaningful titles; corpora with URL-ish titles will deflate
    admission. Report title-quality per source.
P3. Yes/no exclusion cost: the NOYN filter drops real comparison questions;
    count them separately as "recoverable with a better answer proxy".
P4. Gate word lists are English-only.
```

## 7. Cost estimate for the real run

D2 with ~5k queries across 3-4 sources: $0 generation (retention-layer
gates only) + one-time dataset downloads (user approval) + ~$0.10 for the
per-source 32-case generative canaries if any source is admitted at
meaningful rates.
