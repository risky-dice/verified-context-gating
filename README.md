# Verified Context Gating

**Cut RAG input cost only where you have proven it is safe — on your data, not on a benchmark average.**

This repo is two things:

1. **`contextgates`** — a small, dependency-free Python library: training-free
   context gates + the evidence ladder that decides whether to enable them.
2. **The reproduction package** for the preprint behind it:
   *Verified Context Gating: Training-Free Structural Admission for Context
   Selection, with Pre-Registered Quality Evidence* — Yongsun Lee, 2026
   (arXiv link forthcoming).

## The idea

Compression tools ship benchmark averages and ask for trust. A **context gate**
asks the opposite question: *is this query inside a regime where cheap selection
is verified safe?* A gate is an admission predicate over the question string and
document titles only — no model, no training, microseconds — plus a selector.
Rejected queries get the full context. Gates stay **disabled until you attach an
evidence report** produced on your own data.

```python
from contextgates import entity_title_gate, GateRegistry, run_ladder

gate = entity_title_gate()
report = run_ladder(gate, my_rows, generate=my_gen, judge=my_judge)  # your callables

registry = GateRegistry()
if report["passed"]:
    registry.register(gate, evidence=report)

r = registry.route(question, docs)      # falls back to full context if unsure
answer = my_llm(prompt(r.context_text)) # r.token_reduction_pct tells you what you saved
```

`generate` and `judge` are functions you write (~20 lines, any provider — see
[`examples/quickstart.py`](examples/quickstart.py)). The library never holds an
API key and never phones home.

## The evidence ladder

| Layer | What it does | Cost |
|---|---|---|
| **L1 retention** | Does the selected context still contain the answer and the gold evidence, at what token reduction? | free, local |
| **L2 generation** | Paired selected/full answers from your model | your API |
| **L3 blinded judging** | A/B judging with a deterministic sha256 arm assignment absent from judge inputs; non-inferiority gate | your API |
| **L4 replication** | For anything discovered post hoc: re-run with the hypothesis fixed first | your API |

L1 alone never enables a gate — the paper's sharpest finding is that proxy
metrics mislead in **both** directions (exact match was *identical* across arms
while a replicated 6-point semantic gap existed).

## Built-in gates (verified in the paper)

| Gate | Admits | Selects | Verified result |
|---|---|---|---|
| `or_comparison_gate()` | question contains `" or "` | lexical top-10 pages | HotpotQA regime: **45% fewer input tokens** at blind-judged semantic parity (n=123 frozen holdout) |
| `entity_title_gate()` | ≥2 document titles verbatim in the question, no role/kinship word, not yes/no form | the named documents, whole | 2WikiMultihopQA regime: **69% fewer tokens** and **~6 points *higher* semantic accuracy** (n=981 holdout; replicated, 73:13 discordant) |

Those numbers are for those regimes on those benchmarks. **Your mileage is
exactly what the ladder is for.** The library reproduces the paper's headline
retention numbers on the original data (verified: Gate 2 → n=981,
retention 0.9959, reduction 68.914%; coverage 0.9975 vs the harness's 0.998, a
0.0005 difference from set-based vs count-based sentence bookkeeping).

## Honest scope

- **Cost, not latency.** An order-controlled paired canary found no evidence of
  a ≥10% wall-clock gain at kilotoken scale on a commercial API (median ~1%).
  Prompt tokens fall 44–69%; the clock does not move. Don't buy this for speed.
- Both verified regimes are multi-hop QA benchmarks; 2Wiki's questions are
  template-generated, which makes verbatim title matching unusually informative
  there. Prevalence in real traffic is unmeasured — a measurement protocol is in
  [`reproduction/reports/`](reproduction/reports/).
- Gates are English-specific; verbatim-title matching under-admits paraphrases.
- The paper's judging used an LLM panel plus an external judge family; 86
  discordant labels carrying the headline effect were not human-adjudicated.

## Install

```bash
pip install contextgates                  # zero dependencies, Python >= 3.10
pip install "contextgates[llamaindex]"    # + GateNodePostprocessor
pip install "contextgates[langchain]"     # + GateDocumentCompressor
```

From source:

```bash
pip install -e .                          # use a venv (PEP 668 blocks system python)
python tests/test_contextgates.py         # 9 tests
python tests/test_integrations.py         # 5 tests
python examples/quickstart.py             # L1 only, free
```

### Framework integrations

```python
# LlamaIndex — drop verified gates into any query engine
from contextgates.integrations.llamaindex import GateNodePostprocessor
engine = index.as_query_engine(node_postprocessors=[GateNodePostprocessor(registry)])

# LangChain — as a document compressor
from contextgates.integrations.langchain import GateDocumentCompressor
retriever = ContextualCompressionRetriever(
    base_compressor=GateDocumentCompressor(registry=registry),
    base_retriever=my_retriever,
)
```

Adapters group retrieved chunks into titled documents, apply the registry, and
drop or trim chunks accordingly; a rejected query passes every chunk through
untouched. Neither framework is a dependency of the core package.

## Reproduction package

```
paper/          preprint (markdown + LaTeX sources)
reproduction/
  harnesses/    one self-contained script per experiment (#1258–#1278)
  results/      result JSONs, label-free judge request packs, raw panel and
                external-judge verdicts
  reports/      per-experiment reports, consolidated evidence report,
                related-work audit, prevalence-measurement protocol
```

Datasets (download separately): HotpotQA distractor dev, 2WikiMultihopQA dev →
`data/external/`. Retention-layer experiments are local and free; live ones are
guarded by approval env vars and budget ceilings. `task1278` compares against
[Provence](https://huggingface.co/naver/provence-reranker-debertav3-v1)
(CC BY-NC-ND — research use, downloaded by you, not bundled here). The entire
published evidence program cost **≈$1** of metered API spend.

## License

Code: MIT. Datasets and third-party models keep their own licenses.

## Citation

```bibtex
@misc{lee2026verifiedcontextgating,
  title  = {Verified Context Gating: Training-Free Structural Admission for
            Context Selection, with Pre-Registered Quality Evidence},
  author = {Lee, Yongsun},
  year   = {2026},
  note   = {arXiv preprint (forthcoming); https://github.com/risky-dice/verified-context-gating}
}
```
