# S4 → S5 Handoff

**Status: S4 complete.** The rolling-origin **backtest harness** is built —
model-agnostic, with load-bearing as-of leakage guards — and the **Accuracy /
FVA page** renders the waterfall + by-lag view with a one-click,
manifest-reproducible recompute. Per v2's rule, the harness landed *before* any
real model, so S5–S8 plug a forecaster callable in and are judged through it.
`pytest` is green (**69 tests**: the S1–S3 58 + 11 new in
`tests/test_backtest_s4.py`). One module this session — the backtest half of
`sensing/engine.py` plus `app/pages/5_Accuracy_FVA.py` (and a tiny
`app_common.run_manifest` helper). Everything else is untouched.

Earlier: **S3** — full M2 QC gate suite. **S2** — hardened upload layer.
**S1** — app skeleton + workspace/DuckDB + one-click demo + `run_pipeline` stub.

Carry this file into the next Claude session with the v3 design doc. It freezes
the contracts every later step builds against.

---

## The one interface everything runs behind

```python
from sensing import run_pipeline, RunConfig
result = run_pipeline(workspace, as_of, config)   # -> RunResult
```

**Rule that must not break:** every engine function reads ONLY from the
workspace DuckDB (canonical_* tables), never from file paths. That is what makes
the v3->v2 graduation a one-layer swap of `ingest_ui.py` for lake pipelines.

### `RunConfig` (sensing/config.py)
`as_of: date`, `horizon_weeks=8`, `target_wos=4.0`, `order_cadence_weeks=1`,
`reaction_lag_weeks=1`, `ml_enabled=False`, `deviation_threshold=0.15`,
`backtest_weeks=26`. `.to_dict()` / `.from_dict()` for manifests.

### `RunResult` (sensing/engine.py) — dataclass
| field | grain / columns |
|---|---|
| `forecast` | item_id, region, week, lag, p10, p50, p90, plan_units, blend_weight_sensed, blended |
| `repaired` | item_id, region, week, units_sold, units_unconstrained, is_censored, base_units, promo_uplift |
| `inventory_projection` | item_id, region, week, projected_on_hand, target_position, projected_order |
| `alerts` | severity, alert_type, item_id, region, week, message |
| `fva` | step, wmape, bias  (waterfall at lag 1-2) |
| `fva_by_lag` | lag, method, wmape, bias  (method in naive/statistical/translated/plan) |
| `manifest` | dict: run_id, as_of, config, inputs, snapshots |

`result.outputs()` returns the dict persisted to the workspace and re-read by
the pages via `ws.read_run_output(run_id, name)`.

---

## Canonical schemas (sensing/config.py `CANONICAL_SCHEMAS`) — FROZEN

Grain: **item_id x region x week**. Streams: `pos` (required),
`channel_inventory`, `shipments` (required), `demand_plan`, `promo`. Missing
optional streams degrade gracefully.

## Workspace / Ingestion / Quality APIs — FROZEN (unchanged in S4)

Workspace (`sensing/workspace.py`): `add_snapshot` · `read_canonical` ·
`canonical_status` · `rebuild_canonical` (stacks all snapshots, de-dups on
`_grain_keys`, last-write-wins; `demand_plan` keeps distinct
`plan_version_date` vintages) · `save_run` / `list_runs` / `read_run_output` ·
`save_mapping` / `get_mapping` · `snapshot_coverage`.
Ingestion (`sensing/ingest_ui.py`) and Quality (`sensing/quality.py`,
`run_qc(...) -> QCReport`) are exactly as S2/S3 left them.

New tiny app helper (S4): `app_common.run_manifest(ws, run_id) -> dict | None`
returns a run's persisted JSON manifest, so page 5 can recompute reproducibly.

---

## Backtest harness (sensing/engine.py) — **NEW in S4, now frozen surface**

The harness is model-agnostic: any forecaster with the standard signature is
scored without touching the harness. S5–S7 plug in here.

```python
# forecaster signature
Callable[[AsOfPanel, as_of, horizon], forecast_df]
#   forecast_df: columns [item_id, region, week, pred], weeks STRICTLY > as_of

backtest(
    forecaster, data, cfg, *,
    value_col="units_unconstrained",     # actuals target; falls back to units_sold
    lags=(1, 2, 4, 8),
    method_name="model",
    min_train=8,
) -> tidy_df   # item_id, region, lag, week, as_of, pred, actual, method
```

Helpers on the same surface (import from `sensing.engine`):

* **`AsOfPanel`** — the history view handed to a forecaster at one origin.
  `.history()` / `.series(item, region)` (both <= as_of), `.keys()`,
  `.future_weeks(horizon)` (target week labels, **no values**), and
  `.value_at(item, region, week)` which **raises `LeakageError` on a future
  week** — this is what catches a peeking forecaster.
* **`LeakageError(AssertionError)`** — raised on any as-of violation (a future
  read via `value_at`, a forecast dated <= as_of, or a future plan vintage).
* **Built-in forecaster adapters** (S1 placeholders; S5–S7 replace internals,
  not signatures): `naive_forecaster`, `statistical_forecaster`,
  `translated_forecaster`, and `plan_forecaster(plan)` (a factory that closes
  over the versioned plan and emits the lag-adjusted vintage).
* **`_plan_asof(plan, as_of)`** — per (item, region, week) the row with the
  latest `plan_version_date <= as_of`. Never the latest plan; asserts no future
  vintage survives.
* **`_aggregate_fva(tidy) -> (fva, fva_by_lag)`** — collapses per-cell scores
  into the two frozen shapes; orders methods by `FVA_METHODS` and lags ascending.
* **`_backtest(repaired, plan, cfg) -> (fva, fva_by_lag)`** — the assembly
  `run_pipeline` calls; scores naive/statistical/translated against the sell-out
  target and the plan lag-adjusted. **Signature and return contract unchanged.**

### Frozen output guarantees (page 5 + downstream depend on these)

* `fva` columns are exactly `step, wmape, bias` (waterfall, lag 1-2).
* `fva_by_lag` columns are exactly `lag, method, wmape, bias`.
* Method vocabulary is exactly `{naive, statistical, translated, plan}`
  (`FVA_METHODS`); waterfall step labels come from `FVA_STEP_LABELS`.
  **Extend by adding rows/lags/method-values, never by renaming columns.**
* WMAPE = sum|actual-pred| / sum|actual|; bias = sum(pred-actual) / sum|actual|
  (+ve = over-forecast). Pooled across SKU-regions at each (lag, method), so the
  aggregate is volume-weighted.

### As-of leakage — load-bearing, not decorative
1. A forecaster only ever receives an `AsOfPanel` truncated to <= as_of; the
   panel self-checks and `value_at` raises on future reads.
2. The harness rejects any returned forecast week <= as_of.
3. The plan is scored by the vintage available at each origin
   (`plan_version_date <= as_of`), never the latest plan.
`tests/test_backtest_s4.py` proves each of these fires.

### Reproducibility
The FVA numbers are a deterministic function of `canonical_*` + `RunConfig`
(no RNG in the backtest). The run manifest records config + snapshot dates, and
page 5's **Recompute backtest** button re-runs `run_pipeline` from that manifest.

---

## Where each remaining step plugs in

| Step | File(s) to replace/extend | Interface stays |
|---|---|---|
| ~~S2 Upload mapper~~ done | `ingest_ui.py`, `workspace.py`, page 1 | canonical schemas |
| ~~S3 QC cards~~ done | `quality.py`, page 1 | `run_qc(...) -> QCReport` |
| ~~S4 Backtest/FVA~~ done | `engine` backtest half, page 5 | `fva`, `fva_by_lag` shapes |
| **S5 Signal repair** <- next | `engine._repair` | `repaired` columns |
| S6 Statistical baseline + blend | `engine._seasonal_naive` / `statistical_forecaster`, `_forecast_and_translate` | `forecast` columns |
| S7 Translation engine | `engine._forecast_and_translate` + `translated_forecaster` | `inventory_projection` columns |
| S8 Alerts + LightGBM | `engine._alerts`; add `ml_model.py` | `alerts` columns; `RunConfig.ml_enabled` |

Swap internals freely; keep the column contracts and the pages keep working.

## Working-method rules (carried from v2/v3, still binding)

One module per session · tests are the memory between sessions · **as-of leakage
assertions everywhere** · run manifests on every run · engine reads only the
workspace, never file paths · QC reads canonical and never mutates.

## S4 test fixtures (reused by later steps)

`tests/test_backtest_s4.py` builds small known-signal series inline
(`_noisy_stationary`, a clean trend) and a two-vintage plan (`_versioned_plan`).
It proves: statistical beats naive on a noisy-stationary signal (positive
naive->statistical FVA) and is reproducible; naive WMAPE is monotonic in lag on a
trend; both leakage vectors are caught; the plan is scored by the available
vintage (not the future "perfect" one); the frozen shapes/vocabulary hold; and,
end-to-end on the demo, the plan is scored at lags 1/2/4 but **not** lag 8
(its only vintage there is dated after the origin).

## S4 -> S5 note

**S5 owns signal repair — `engine._repair`** (design section 4 "Demand Signal
Repair", v1 section 4 / v2 M3). Replace the S1 heuristic (low-stock +
depressed-sales) with the real thing: inventory-driven + statistical
zero-inflation stockout de-censoring, promo/baseline decomposition against the
promo calendar (incl. post-promo dip), and outlier/structural-break handling —
emitting the **frozen `repaired` columns** (`units_sold, units_unconstrained,
is_censored, base_units, promo_uplift`). The demo already seeds one stockout
(SKU-1002/East, weeks 38-42) and two promos, so S5's acceptance is: injected
stockouts/promos are recovered on synthetic data, with shaded-censoring visuals
on the Forecast Review page.

The harness is the judge: `_backtest` already scores against
`units_unconstrained` when present (falling back to `units_sold`). So better
repair -> a better sell-out target -> the naive/statistical/translated FVA
numbers move **without any harness change**. Do **not** touch the backtest
surface, `fva`/`fva_by_lag` shapes, the method vocabulary, page 5, or the frozen
workspace/ingest/quality contracts — extend `_repair` behind its existing
columns and let the frozen harness re-judge.

## Known stubs (deliberate, not bugs)

* Statistical tier is a damped-EWMA placeholder, not ETS/AutoETS -> on demo data
  it does **not** clear the >=15% FVA bar (it currently trails naive, which is
  the honest S1/S4 verdict). S5-S7 earn the number; S4 only makes it *trustworthy*.
* `translated_forecaster` is a naive/statistical blend scored against the
  sell-out target; S7 replaces it with the channel-inventory projection and
  re-targets scoring to shipments (sell-in) — the harness itself won't change.
* ML toggle is wired through config but does not yet train a model.
* Repair censoring detection is a heuristic; S5 replaces it.
