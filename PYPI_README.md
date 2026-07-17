# contextgates

**Cut RAG input cost only where you have proven it is safe — on your data, not on a benchmark average.**

Context compression tools ship benchmark averages and ask for trust. A
**context gate** asks the opposite question: *is this query inside a regime
where cheap selection is verified safe?* A gate is a training-free admission
predicate over the question string and document titles only — no model, no
training, microseconds — plus a selector. Rejected queries get the full
context. **Gates stay disabled until you attach an evidence report produced on
your own data.**

```python
from contextgates import entity_title_gate, GateRegistry, run_ladder

gate = entity_title_gate()
report = run_ladder(gate, my_rows, generate=my_gen, judge=my_judge)  # your callables

registry = GateRegistry()
if report["passed"]:
    registry.register(gate, evidence=report)

r = registry.route(question, docs)      # full-context fallback when unsure
answer = my_llm(prompt(r.context_text)) # r.token_reduction_pct = what you saved
```

`generate` and `judge` are functions you write (~20 lines, any provider). The
library holds no API keys, has **zero dependencies**, and never phones home.

## The evidence ladder

| Layer | What it does | Cost |
|---|---|---|
| **L1 retention** | Does the selected context still hold the answer and the gold evidence, at what token reduction? | free, local |
| **L2 generation** | Paired selected/full answers from your model | your API |
| **L3 blinded judging** | A/B judging with a deterministic sha256 arm assignment absent from judge inputs; non-inferiority gate | your API |

L1 alone never enables a gate: in the accompanying research, exact match was
*identical* across arms while a replicated 6-point semantic gap existed. Proxy
metrics mislead in both directions.

## Built-in gates (verified in the paper)

| Gate | Admits | Verified result |
|---|---|---|
| `or_comparison_gate()` | question contains `" or "` | HotpotQA regime: **45% fewer input tokens** at blind-judged semantic parity (n=123 frozen holdout) |
| `entity_title_gate()` | ≥2 document titles verbatim in the question, no role/kinship word, not yes/no form | 2WikiMultihopQA regime: **69% fewer tokens** and **~6 points higher semantic accuracy** (n=981 holdout, replicated) |

## Framework integrations (optional)

```bash
pip install "contextgates[llamaindex]"   # GateNodePostprocessor
pip install "contextgates[langchain]"    # GateDocumentCompressor
```

## Honest scope

- **Cost, not latency.** An order-controlled paired canary found no evidence of
  a ≥10% wall-clock gain at kilotoken scale on a commercial API (median ~1%).
  Don't adopt this for speed.
- Both verified regimes are multi-hop QA benchmarks; prevalence on real traffic
  is unmeasured — that is what the ladder is for.
- Gates are English-specific; verbatim-title matching under-admits paraphrases.

Paper, reproduction package, and full limitations:
<https://github.com/risky-dice/verified-context-gating>

MIT licensed.
