# Verified Context Gating

Reproduction package for the preprint:

> **Verified Context Gating: Training-Free Structural Admission for Context
> Selection, with Pre-Registered Quality Guarantees** — Yongsun Lee, 2026.
> (arXiv link forthcoming)

## TL;DR

A *context gate* is a training-free admission predicate (question string +
document titles only, microseconds, no model) that admits a query into a
verified regime, applies a regime-specific selector, and falls back to the
full context otherwise. Every gate must pass a pre-registered evidence
ladder before it is enabled: frozen-split retention holdout → live
generation → blinded LLM judging under two judge families → pre-registered
replication for any post-hoc effect.

Headline results (all reproducible from this package):

- **Gate 1 (Hotpot-or)**: 45% input token reduction at blind-judged
  semantic parity (123 frozen cases).
- **Gate 2 (2Wiki-entity)**: 69% reduction — and semantic accuracy
  **improves** by ~6 points over full context (0.968 vs 0.907, discordant
  73:13 over the full 981-pair holdout; discovered post-hoc, confirmed
  under a second judge family, replicated with the hypothesis
  pre-registered, 29:7, p=3.1e-4). Mechanism: full context induces
  distractor-driven comparison errors that selection removes.
- At a matched compression rate, the training-free gate beats the trained
  pruner **Provence** (ICLR 2025) by +10pt retention / +19pt coverage
  in-regime.
- **Cautions we measured**: exact match was blind to the entire effect
  (identical 0.9866 across arms vs a 6.4pt semantic gap), and an
  order-controlled paired latency canary found **no wall-clock speedup**
  at kilotoken scale (median ~1-3%). This package cuts cost, not latency.

Total API spend of the entire evidence program: **under $1** (gpt-4.1-mini).

## Honest scope

- Two regimes, both multi-hop QA benchmarks. Regime prevalence in real RAG
  traffic is unmeasured — see `reproduction/reports/Prometheus_Regime_Prevalence_Protocol_Draft.md`
  for the pre-registered measurement protocol.
- Single generation model (gpt-4.1-mini). Admission predicates are
  English-specific.
- Provence comparison is out-of-domain for Provence and regime-scoped;
  outside admitted regimes the approaches are complementary.

## Layout

```
paper/          preprint (markdown + LaTeX sources)
reproduction/
  harnesses/    one self-contained Python script per experiment (#1258-#1278)
  results/      result JSONs, blinded judge request packs (label-free),
                raw panel verdicts, external-judge verdicts
  reports/      per-experiment reports + consolidated evidence report +
                related-work audit + prevalence protocol draft
```

## Reproducing

Datasets (download separately): HotpotQA distractor dev
(`hotpot_dev_distractor_v1.json`), 2WikiMultihopQA dev (`dev.json`).
Place under `data/external/` as referenced at the top of each harness.

- Retention-layer experiments (`task1259-1263, 1268-1271, 1278`) are fully
  local and free. `task1278` additionally requires
  `pip install torch transformers nltk` and downloads
  `naver/provence-reranker-debertav3-v1` (CC BY-NC-ND 4.0 — research use;
  NOT bundled here).
- Live experiments (`task1264, 1267, 1272-1277`) call the OpenAI API and
  are guarded by explicit approval environment variables and budget
  ceilings (see each script's header). Blinding salts are deterministic;
  frozen splits are dataset-index parity; every gate threshold appears in
  harness source predating the corresponding run.

## License

Code and reports: MIT. Datasets and the Provence model belong to their
respective owners and licenses.
