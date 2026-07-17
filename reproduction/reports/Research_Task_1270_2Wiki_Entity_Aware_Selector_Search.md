# Research Task #1270: 2Wiki Entity-Aware Selector Search (Exploratory)

Date: 2026-07-15 KST

## Question

#1269 showed cheap lexical depth-shrinking cannot pass the gates on 2Wiki
(coverage collapses whenever retention and reduction are pushed together).
Can an entity-aware selector — keep only the documents named verbatim in the
question — plus a purely question-structural admission predicate form a
passing 2Wiki regime?

## Motivating diagnostic (search split, even indices only)

```text
comparison rows:        100% have >=2 context titles named in the question;
                        100% of supporting facts lie INSIDE those named docs;
                        named docs hold only ~34% of characters.
bridge_comparison rows: 100% have >=2 matched titles but 0% of supporting
                        facts inside them -> must be excluded by admission.
role-word predicate:    excludes 1346/1346 bridge rows (fn=0) at a cost of
                        23/1529 comparison rows (1.5%).
```

## Method

Harness: `lab/experiments/task1270_2wiki_entity_aware_selector_search.py` (v2)

Deployable admission predicate (uses NO dataset metadata):

```text
ENT2:  >= 2 context titles appear verbatim (lowercase) in the question
NOREL: no relational-role/kinship word in the question
       (director|performer|composer|...|father|mother|grandfather|husband|...)
NOYN:  question does not start with are/is/do/does/did/were/was/have/has
```

Selectors: sentence-granularity, matched-title documents only
(sent_full, sent_top{3,2,1}_per_doc). Grid = 2 boundaries x 4 selectors.
Search split only (even indices); 300 accepted rows per boundary;
project-standard gates.

## v1 lessons (recorded for honesty)

```text
1. Page-granularity selection failed spuriously (coverage 0.55): #1263 pages
   chunk sentences ACROSS document boundaries, so mixed-title pages were
   dropped. v2 selects at sentence granularity.
2. Kinship questions ("maternal grandfather of X") leaked through the
   role-word filter; kinship terms were added.
3. Yes/no answers ("no") caused substring retention artifacts against the
   longer full text; yes/no-form questions were excluded in the _noyn
   boundary.
v1 -> v2 tuning happened ON THE SEARCH SPLIT; the odd-index holdout (#1271)
is the only clean test of it.
```

## Result

```text
claim_decision: exploratory_2wiki_entity_regime_candidate_found
success: true
passing cells: 1/8
best: ent2_norel_noyn x sent_full
      ret 0.9967  cov 0.9983  perf 0.9967  red 66.75  n=300
admission audit (full search split): comparison 1036, compositional 2, inference 1
```

The per-doc-cap variants trade coverage for reduction and fail the gates
(top3: cov 0.968; top2: 0.942; top1: 0.819) — the regime needs whole named
documents, which are already only ~1/3 of the context.

## Interpretation

The #1269 wall was a selector-family limitation, not a dataset
impossibility. Named-entity document selection is qualitatively different
from lexical depth-shrinking: it uses the question's explicit entity
structure, which is exactly the "cheap admission / query-structure gating"
thesis. Candidate frozen for #1271.
