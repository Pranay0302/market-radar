# MarketRadar

An OEM-native commercial *action layer*. It turns configuration-level competitor
moves into spec-mapped, constraint-checked recommendations for your own
portfolio. Not a price monitor, not a digital-shelf dashboard.

## What it does

1. **Maps** a competitor config to your closest own SKU by spec, via a
   knowledge-graph entity resolver that does no name matching, so it survives
   renames.
2. **Detects** which spec attributes are driving sell-out velocity at a stable
   price (e.g. 32GB + OLED).
3. **Recommends** a constrained action (reprice / promote / hold), where the
   margin and inventory math is done by a deterministic solver and the model
   only writes the rationale.

## Quickstart

```bash
pip install -r requirements.txt          # core deps are enough to run everything

python -m marketradar.pipeline --tenant acme-pc   # end-to-end demo (CLI)
python -m marketradar.evals                        # recommendation-correctness evals
pytest -q                                          # test suite
streamlit run app.py                               # minimal dashboard
```

The open-source ML models (MiniLM, distilbert, flan-t5, Chroma) are **optional**:
if they aren't installed, MarketRadar falls back to sklearn / lexicon / template
backends and still runs end-to-end. Install the optional lines in
`requirements.txt` to switch the real models on.

## Layout

```text
marketradar/
  schema.py       spec taxonomy + normalization
  data_gen.py     seeded synthetic market / portfolio / reviews
  embedding.py    MiniLM-or-TF-IDF embedder
  graph.py        portfolio knowledge graph
  resolver.py     KG entity resolution (survives renames)
  traction.py     velocity-at-stable-price detection + attribution
  rag.py          RAG few-shot aspect classifier
  sentiment.py    "why is it winning" mining
  generation.py   flan-t5-or-template rationale
  router.py       cost-aware model routing + call log
  constraints.py  deterministic feasible-action solver
  recommend.py    ranking + recommendation
  monitor.py      proactive windowed alerts
  audit.py        audit log + tenant isolation
  evals.py        golden-set correctness harness
  pipeline.py     end-to-end orchestration
app.py            Streamlit dashboard
```

See `PRD.md` and `TDD.md` for the product and technical write-ups.
