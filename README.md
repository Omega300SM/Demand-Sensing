# Demand Sensing POC — Upload-Driven (v3)

A laptop-runnable Streamlit app that proves whether a **sensed forecast beats
the demand plan** on your data — before any lake connectivity or IT tickets.
Data enters through **manual file uploads**; the engine runs behind one clean
interface so the later swap to lake ingestion (v2) is a one-layer change.

This repository is the **complete engine build (S1–S8)**: the app skeleton with
all five pages (S1), a hardened upload + column-mapper layer (S2), the full M2 QC
gate suite (S3), a **model-agnostic rolling-origin backtest harness** with
load-bearing as-of leakage guards (S4), real **signal repair** (S5), the real
**statistical baseline + blend** (S6), the real **sell-out → sell-in translation
engine** — channel-inventory projection + empirically-calibrated,
inventory-aware order model, with a distributed-lag transfer-function fallback
auto-selected per series, scored against shipments (sell-in) through the frozen
harness (S7) — and the **hardened alerts engine + optional ML tier** (S8): the
full design-§8 exception suite plus a global gradient-boosted quantile model
behind `RunConfig.ml_enabled`, judged through the same frozen harness by a
champion–challenger rule. The engine runs behind one interface,
`run_pipeline(workspace, as_of, config) -> RunResult`; the next move is not
another module but **graduation to the v2 lake-connected build** (a one-layer
swap of `ingest_ui.py` for lake pipelines). See `HANDOFF.md` for the frozen
contracts and the graduation guide.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

Then on the **Home** page click **⚡ Load demo dataset** and walk
Home → Run Sensing → Forecast Review → Alerts → Accuracy / FVA.

## Structure

```
demand-sensing-poc/
├── app/
│   ├── Home.py                    # workspace picker, status, one-click demo
│   ├── app_common.py              # shared helpers (path bootstrap, workspace state)
│   └── pages/
│       ├── 1_Upload_Data.py       # upload slots, column mapper, QC cards
│       ├── 2_Run_Sensing.py       # config + run button + progress + manifest
│       ├── 3_Forecast_Review.py   # fan chart, repaired demand, inventory projection, export
│       ├── 4_Alerts.py            # exception queue with driving-signal charts
│       └── 5_Accuracy_FVA.py      # FVA waterfall, WMAPE/bias by lag
├── sensing/
│   ├── config.py                  # RunConfig + CANONICAL_SCHEMAS
│   ├── workspace.py               # DuckDB workspace: snapshots, canonical, runs
│   ├── ingest_ui.py               # schema sniffing, column mapping, coercion
│   ├── quality.py                 # QC gates -> validation cards
│   ├── demo_data.py               # synthetic generator + template writer
│   └── engine.py                  # run_pipeline + RunResult (stub with real bones)
├── templates/                     # CSV templates + demo_dataset/ + messy export
├── workspace/                     # DuckDB files live here (created at runtime)
└── tests/                         # pytest: workspace + ingestion layers
```

## Tests

```bash
python -m pytest -q
```

## What the engine does today (S1–S8, complete)

It reads **only** canonical tables from the workspace and runs: **signal repair**
(inventory-driven + statistical stockout de-censoring, promo/baseline
decomposition, outlier/structural-break handling — S5) → **statistical forecast**
(a real damped-local-level ETS-family tier fit on the de-censored target, with a
genuine P10/P50/P90 fan and a guarded seasonal — S6), **optionally overridden per
series by a global gradient-boosted quantile ML tier** where it wins a
champion–challenger backtest (`RunConfig.ml_enabled`, off by default — S8) →
**sell-out→sell-in translation** (project channel inventory forward and order
back to a target cover, with order-cycle batching + reaction lag, the target WOS
and order response **calibrated empirically** per series, and a distributed-lag
transfer-function fallback where inventory is unusable — chosen per series by
backtest; scored against **shipments/sell-in** — S7) → **horizon-weighted blend**
with the plan (sensed-dominant near-in, decaying out — S6) → the **hardened
alerts engine** (cumulative sensed-vs-plan deviation, projected stockout,
overstock / order-cliff, promo mid-flight variance, censoring holds — each row
carrying its driving signal, S8) → a **genuine rolling-origin FVA backtest** with
load-bearing as-of leakage guards (S4). The statistical tier clears the S6 bar
(positive FVA vs. naïve), the translation beats a shipments-history-only model at
near lags, and the ML tier is judged through the **same frozen harness** and used
only where it demonstrably wins — all with `ml_enabled=False` reproducing the S7
pipeline byte-for-byte. The optional `statsforecast` and `lightgbm` wheels are
import-guarded, so the suite runs without them (**129 tests**). The next move is
the **v2 graduation**, not another module. See `HANDOFF.md` for the interface
contracts and graduation guide.

`statsforecast` (AutoETS) is an import-guarded, opt-in engine for the statistical
tier (`sensing.engine.USE_STATSFORECAST`); the default tier is dependency-free so
the suite runs with or without the wheel.
