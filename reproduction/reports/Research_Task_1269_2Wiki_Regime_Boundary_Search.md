# Research Task #1269: 2Wiki Regime Boundary x Selector-Depth Search

Date: 2026-07-15 KST

## Question

Does 2Wiki have its own safe admission regime (boundary + selector depth),
analogous to Hotpot-or + lexical_top10? This is the "is regime addition a
repeatable methodology?" test. #1263 already showed the verbatim Hotpot
policy fails on 2Wiki; the hypothesis here was that a 2Wiki-native depth
would fix it.

## Method

Harness: `lab/experiments/task1269_2wiki_regime_boundary_search.py`
(fully local, non-generative; imports #1263 functions so scoring is
identical).

Split discipline fixed in advance: even dataset indices = search split (this
task); odd indices reserved untouched for a #1270 frozen holdout.

Grid: 6 boundaries (cmp_or, cmp_all, bridge_or, cmp_which, cmp_or_which,
anycmp_or) x 6 selectors (page_top{4,6,8,10}, sent_top{8,12}); 300 accepted
rows per boundary; project-standard gates (n>=100, retention>=0.99,
coverage>=0.98, perfect>=0.94, reduction>=35).

## Result

```text
claim_decision: no_2wiki_regime_candidate_in_search_grid
success: false
cells: 36, passing: 0
```

The frontier (closest cells):

```text
cmp_which  x page_top8: ret 0.9833  cov 0.9650  perf 0.9300  red 40.53
anycmp_or  x page_top8: ret 0.9933  cov 0.8842  perf 0.6733  red 34.87
bridge_or  x page_top8: ret 0.9967  cov 0.8033  perf 0.4233  red 37.24
cmp_or     x page_top10: ret 0.9900 cov 0.9800  perf 0.9600  red 23.28
```

## Interpretation

1. There is a hard tradeoff wall: on 2Wiki, retention >= 0.99 and
   reduction >= 35% are individually reachable but support coverage
   collapses whenever both are pushed. No cell satisfies all gates.
2. Structural reason: 2Wiki rows have 10 short documents with exactly-two
   supporting facts usually split across two documents; cheap lexical
   scoring reliably finds one but drops the other once depth shrinks.
   Hotpot's longer documents let top10 keep ~45% reduction with full
   support; 2Wiki's geometry does not.
3. bridge_or retention (1.0 at top10) is inflated by yes/no-style short
   answers matching anywhere in the text — exactly the failure mode the
   support-coverage gates exist to catch. The gates worked as designed.
4. Verdict on the strategic question: regime addition is NOT free. The
   methodology correctly refused to fabricate a second regime rather than
   silently relaxing gates. #1270 (frozen holdout) is NOT warranted — there
   is no candidate to freeze.

## What could still rescue a 2Wiki regime (future options, in order)

```text
1. Generative check of the frontier cell (cmp_which x page_top8): the
   coverage gate is a proxy; Hotpot showed retention-layer safety translated
   to generative parity. A ~$0.05 live canary could test whether cov 0.965
   still yields semantic parity. Requires approval; do not relax proxy gates
   without it.
2. Entity-aware cheap selector: 2Wiki questions name two entities; a
   selector that forces >=K pages per named entity might hold coverage at
   higher reduction. New selector family, needs its own search + holdout.
3. Accept 2Wiki as out-of-regime and route it to full context. The admission
   gate architecture explicitly permits this.
```

## Warning Labels

```text
Moderate evidence (negative): 36-cell exploratory sweep, search split only.
Note: retention proxy inflated for yes/no answers; coverage gates caught it.
Missing: generative test of frontier cells (needs approval).
```
