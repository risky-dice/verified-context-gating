# Related Work Audit Update (post-#1277, pre-preprint)

Date: 2026-07-17 KST
Purpose: answer "does prior work overlap our headline results?" before the
preprint. Supersedes the scope of #1253 with the new C2+ finding in view.

## Verdict in one line

The PHENOMENON (removing irrelevant context can improve answer accuracy) is
already documented in several lines of work; our novelty is NOT the
discovery. What remains ours: training-free O(question) structural
admission with per-regime verified guarantees, the evidence methodology
(pre-registration, structural blinding, two judge families, fresh-data
replication), and the quantified exact-match masking effect tied to a
deployment decision.

## Overlap map

### A. "Less context, better answers" — the phenomenon is known

```text
LongLLMLingua (Jiang et al.)      question-aware prompt compression; reports
                                  up to +21.4 accuracy at 4x compression and
                                  2.1x end-to-end latency gain. CLOSEST
                                  headline overlap. Uses a small LM
                                  compressor (not training-free).
FILCO (Wang et al. 2023)          trained context filters; +8.6 EM, -64%
                                  prompt length.
RECOMP / AdaComp / FltLM          trained extractive compressors/filters
                                  improving RAG quality.
Provence (Chirkova et al.,        trained DeBERTa sentence-pruner, ICLR 2025,
  ICLR 2025)                      plug-and-play, dynamic pruning amount.
Shi et al. (ICML 2023)            LLMs easily distracted by irrelevant
                                  context — the mechanism, established.
Distractor-position studies       incl. on 2WikiMultihopQA (lost-in-the-
                                  middle family).
```

### B. "When to act" gating — adjacent, different granularity

```text
Sufficient Context (Joren et al., sufficiency-of-context lens; selective
  ICLR 2025, Google)              abstention. Judges context sufficiency,
                                  not selection safety; uses an LLM
                                  autorater, not structural predicates.
Query routing for RAG             route queries among retrieval pipelines by
  (RAGRouter-Bench 2026,          predicted complexity/cost. Routing among
  cost-aware routing 2026, etc.)  PIPELINES, not verified-safety context
                                  selection regimes; typically trained
                                  classifiers.
RouteLLM / FrugalGPT              model routing (as in #1253).
```

### C. Exact-match critique — genre exists, our instance is sharper

```text
Reassessing Extractive QA with LLM-as-a-Judge (2025), NLI-based metrics
(2025), CLEV, etc.: EM under/over-counts and LLM judges align better with
humans. Known genre. Our contribution is a sharp measured instance: EM
IDENTICAL across arms (0.9866 = 0.9866) while a pre-registered, replicated
+6pt semantic gap existed — i.e., EM would have erased the deployment-
relevant difference entirely, in a cost-decision context.
```

## What survives as our contribution (preprint positioning)

```text
1. Training-free floor: pure question-structure admission (verbatim titles,
   role-word exclusion, or-form) + entity-doc selection matches or beats
   the QUALITY results reported by trained pruners on its admitted regime,
   at zero training and O(question) admission cost. Framed as "how cheap
   can safe context selection be" — a baseline/floor claim, not SOTA.
2. Verified-regime architecture: admission gates carry frozen-holdout
   guarantees and fall back to full context; safety is verified per regime,
   not predicted by a classifier. No prior work we found does
   pre-registered per-gate verification.
3. Evidence methodology at micro-budget: frozen splits, structural sha
   blinding, 3-pass panels, two judge families, post-hoc -> confirmation ->
   pre-registered replication chain, total spend ~$1.1.包括 negative
   results (latency null; transfer failures) that the compression
   literature rarely reports — notably our controlled latency NULL at
   kilotoken scale contrasts with LongLLMLingua's claimed 2.1x speedup.
4. The EM-masking quantification (C above).
```

## Consequences for the preprint

```text
- Do NOT frame as discovery of "less context, better answers"; cite
  LongLLMLingua/FILCO/Provence/Shi et al. up front and position as floor +
  verification methodology.
- The #1268 baseline ladder now NEEDS a trained-pruner arm (Provence is
  downloadable open-source; LLMLingua-2 likewise) for the floor claim to be
  credible. This requires model-download approval (~0.4GB DeBERTa) and is
  the single highest-value pre-preprint experiment.
- "Sufficient Context" (ICLR 2025) should be cited as the nearest
  when-to-act work; our gates are a cheaper, structurally verified sibling.
```

## Sources

- LongLLMLingua: https://arxiv.org/abs/2310.06839 / https://llmlingua.com/longllmlingua.html
- LLMLingua project: https://www.microsoft.com/en-us/research/project/llmlingua/
- FILCO: https://www.semanticscholar.org/paper/7848d4b4e6ba0897a85cebb6467e94eb0b60d583
- AdaComp: https://arxiv.org/html/2409.01579
- Provence (ICLR 2025): https://arxiv.org/abs/2501.16214
- Sufficient Context (ICLR 2025): https://arxiv.org/abs/2411.06037
- Shi et al., Easily Distracted (ICML 2023): https://proceedings.mlr.press/v202/shi23a/shi23a.pdf
- Reassessing Extractive QA (LLM-as-judge): https://arxiv.org/abs/2504.11972
- RAG routing (2026): https://arxiv.org/pdf/2604.03455 , https://arxiv.org/html/2606.02581v1
- xRAG (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/c5cf13bfd3762821ef7607e63ee90075-Paper-Conference.pdf
