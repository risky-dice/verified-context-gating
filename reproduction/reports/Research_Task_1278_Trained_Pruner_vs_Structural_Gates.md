# Research Task #1278: Trained Pruner (Provence) vs. Structural Gates

Date: 2026-07-17 KST

## Question

Does a state-of-the-art trained context pruner (Provence, ICLR 2025,
DeBERTa-v3 430M, plug-and-play) dominate our training-free structural gates
on their admitted regimes? Load-bearing experiment for the preprint's
"training-free floor" claim, prompted by the 2026-07-17 related-work audit.

## Method

Harness: `lab/experiments/task1278_trained_pruner_vs_structural_gate.py`
(fully local; venv `.venv-provence`; model
naver/provence-reranker-debertav3-v1, CC BY-NC-ND 4.0 — research comparison
only, cannot be shipped commercially).

Arms: ours = lexical_top10 (Hotpot, 123 frozen #1262 cases) and
ent_sent_full (2Wiki, 300 even-split ent2_norel_noyn cases). Provence used
as shipped (per-document pruning, its intended retrieved-passage usage,
always_select_title default) at thresholds {0.01, 0.05, 0.1, 0.25, 0.5} —
0.01/0.05 added after the first run showed 0.1 already compresses ~86% on
these short documents, to close the "you skipped the comparable operating
point" objection. Our arm reproduces #1262/#1270 numbers exactly (harness
validation). Sentence-membership matching spot-verified against raw
Provence outputs. Decision rules pre-registered in the harness.

## Result

```text
decisions: hotpot structural_gate_holds_floor; 2wiki structural_gate_holds_floor
```

Retention layer, answer retention given full / support coverage / token reduction:

```text
HOTPOT  ours lexical_top10   1.000 / 0.990 / 45.3%
        provence 0.01        0.925 / 0.735 / 67.4%
        provence 0.05        0.792 / 0.588 / 81.4%
        provence 0.1         0.742 / 0.472 / 86.4%
        provence 0.25-0.5    <=0.58 / <=0.32 / >=91.6%

2WIKI   ours ent_sent_full   0.997 / 0.998 / 66.8%
        provence 0.01        0.900 / 0.810 / 62.7%   <- SAME operating point
        provence 0.05        0.767 / 0.601 / 79.3%
        provence 0.1         0.673 / 0.459 / 85.6%
        provence 0.25-0.5    <=0.46 / <=0.26 / >=92.3%
```

The decisive row is 2Wiki at threshold 0.01: at essentially the same
compression (62.7% vs 66.8%), the trained pruner loses ~10 points of answer
retention and ~19 points of support coverage against the structural gate.

Qualitative failure mode (spot-checked): at its most conservative shipped
setting, Provence pruned a gold document named verbatim in the question
("Summer Magic", asked about directly) to EMPTY. Per-sentence relevance
scoring drops the attribute sentences (dates, places) that comparison
questions need, and sometimes whole gold documents; the structural gate
keeps named documents whole, which is exactly what this regime requires.

## Interpretation

1. On regimes admitted by cheap structural gates, regime knowledge beats
   generic learned relevance: the gate encodes "comparison questions need
   the complete named-entity documents", which a per-sentence relevance
   model cannot recover.
2. The preprint's floor claim is upgraded to an offensive result: the
   training-free gate does not merely approach the trained pruner — it
   beats it at matched compression on its admitted regimes.
3. Fair scope limits: Provence was trained for web-search RAG passages;
   multi-hop Wikipedia comparison questions are out-of-domain for it.
   The claim is regime-scoped ("inside admitted regimes"), NOT "Provence
   is worse in general" — outside admitted regimes we have no selector at
   all and fall back to full context, where Provence still applies. The
   two approaches are complementary, which is the gating thesis.
4. Retention layer only; generative confirmation of the Provence arm was
   not run (its retention deficit is too large for generation to
   plausibly rescue, and gates are the project standard).

## Files

```text
lab/experiments/task1278_trained_pruner_vs_structural_gate.py
task1278_trained_pruner_vs_structural_gate_results.json
.venv-provence/ (torch, transformers, nltk; provence model in HF cache)
```
