# S5 → S6 Handoff

**Status: S5 complete.** Real **signal repair** now lives in `engine._repair`
(+ helpers), replacing the S1 heuristic (design §4 "Demand Signal Repair", v1
§4, v2 M3): inventory-driven **and** statistical stockout de-censoring,
promo/baseline decomposition against the trade calendar (with the post-promo
dip left intact as real demand), and an outlier/structural-break pass. The
**Forecast Review page** (page 3) gains shaded-censoring bands and a
baseline-vs-promo-uplift view. Because the frozen backtest harness scores
against `units_unconstrained`, better repair moves the naïve/statistical/
translated FVA numbers **with no harness change** — the whole point of building
the harness first. `pytest` is green (**83 tests**: the S1–S4 69 + 14 new in
`tests/test_repair_s5.py`). One module this session — the repair half of
`sensing/engine.py` plus `app/pages/3_Forecast_Review.py`. Everything else is
untouched.

Earlier: **S4** — rolling-origin backtest harness + Accuracy/FVA page. **S3** —
full M2 QC gate suite. **S2** — hardened upload layer. **S1** — app skeleton +
workspace/DuckDB + one-click demo + `run_pipeline` stub.

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

## Workspace / Ingestion / Quality / Backtest APIs — FROZEN (unchanged in S5)

Workspace (`sensing/workspace.py`): `add_snapshot` · `read_canonical` ·
`canonical_status` · `rebuild_canonical` · `save_run` / `list_runs` /
`read_run_output` · `save_mapping` / `get_mapping` · `snapshot_coverage`.
Ingestion (`sensing/ingest_ui.py`) and Quality (`sensing/quality.py`,
`run_qc(...) -> QCReport`) are exactly as S2/S3 left them. The **backtest
harness** (`backtest` / `AsOfPanel` / `LeakageError` / `_backtest` /
`_plan_asof` / `_aggregate_fva` / the built-in forecaster adapters /
`FVA_METHODS` / `FVA_STEP_LABELS`) is exactly as S4 left it — **do not touch**.
`app_common.run_manifest(ws, run_id) -> dict | None` unchanged.

---

## Signal repair (sensing/engine.py) — **NEW in S5, now the repaired contract's owner**

`_repair(pos, inv, promo) -> DataFrame` emits the **frozen `repaired` columns**
`item_id, region, week, units_sold, units_unconstrained, is_censored,
base_units, promo_uplift` and is called by `run_pipeline` before forecasting;
its output feeds BOTH `_forecast_and_translate` AND `_backtest`. **Reads
canonical only; never mutates its inputs** (flags/imputes into fresh arrays).
`channel_inventory` and `promo` are OPTIONAL and may be empty.

### Frozen `repaired` guarantees (page 3 + the harness depend on these)
* Columns/dtypes exactly as above; `is_censored` is `bool`.
* Uncensored weeks: `units_unconstrained == units_sold` exactly.
* Censored weeks: `units_unconstrained` = imputed same-series velocity (never
  raw sales) — train-on-unconstrained so the model can't learn our stockouts.
* Identity everywhere: `base_units + promo_uplift == units_unconstrained`;
  `promo_uplift >= 0`, non-zero only on promo weeks.
* No promo calendar → `promo_uplift = 0`, `base_units = units_unconstrained`.

### New repair helper signatures (extend behind them; don't rename)
```python
REPAIR_PARAMS: dict           # tunable knobs (see below), referenced by tests
_robust_level(y, window) -> np.ndarray          # wide centered-median expected level
_mad(resid) -> float                            # robust scale (floored)
_detect_censored(sales, on_hand, expected, has_inv, p) -> bool[]   # (a)+(b) signals
_impute_unconstrained(sales, censored, window) -> np.ndarray       # two-pass clean-neighbour
_dampen_outliers(unconstrained, promo_flag, censored, expected, p) -> np.ndarray
_decompose_promo(unconstrained, promo_flag, window) -> (base_units, promo_uplift)
```
`REPAIR_PARAMS` defaults (calibrated on the demo): `expected_window=11`,
`censor_ratio=0.40`, `inv_low_frac=0.50`, `inv_trail_window=8`, `stat_z=3.5`,
`base_window=7`, `outlier_z=5.0`.

### How detection works (design §4, both signals wired)
A week is censored only if **depressed** (sell-out `< censor_ratio * expected`,
where `expected` is a wide *centered* rolling median that stays on the healthy
level even inside a 5-week hole) AND corroborated by either
**(a) inventory-driven** (`on_hand` below `inv_low_frac` of its own trailing
typical) or **(b) statistical** (a robust-MAD low-outlier / zero-inflation test,
used when a series has no usable inventory). Imputation is a two-pass
clean-neighbour median (mask the hole, re-estimate the level, fill). The
outlier pass pulls *isolated* one-off blips to the expected level but leaves
*sustained* same-direction shifts (structural breaks) alone.

### The harness re-judges automatically
`_backtest` already scores sell-out forecasters against `units_unconstrained`
(falling back to `units_sold` only if the column is absent). So on the demo's
injected stockout (SKU-1002/East, weeks 38–42), scoring on the de-censored
target beats scoring on raw sales at those weeks (WMAPE ~0.08 vs ~6.9 for the
statistical model) with **no harness change**. `tests/test_repair_s5.py` proves
this for both the statistical and translated forecasters.

---

## Where each remaining step plugs in

| Step | File(s) to replace/extend | Interface stays |
|---|---|---|
| ~~S2 Upload mapper~~ done | `ingest_ui.py`, `workspace.py`, page 1 | canonical schemas |
| ~~S3 QC cards~~ done | `quality.py`, page 1 | `run_qc(...) -> QCReport` |
| ~~S4 Backtest/FVA~~ done | `engine` backtest half, page 5 | `fva`, `fva_by_lag` shapes |
| ~~S5 Signal repair~~ done | `engine._repair` (+ helpers), page 3 | `repaired` columns |
| **S6 Statistical baseline + blend** <- next | `engine._seasonal_naive` / `statistical_forecaster`, `_forecast_and_translate` (forecast + blend half) | `forecast` columns |
| S7 Translation engine | `engine._forecast_and_translate` (translation half) + `translated_forecaster` | `inventory_projection` columns |
| S8 Alerts + LightGBM | `engine._alerts`; add `ml_model.py` | `alerts` columns; `RunConfig.ml_enabled` |

Swap internals freely; keep the column contracts and the pages keep working.

## Working-method rules (carried from v2/v3, still binding)

One module per session · tests are the memory between sessions · **as-of leakage
assertions everywhere** (repair may clean across a whole series, but every
feature/forecast still runs through the harness's as-of guards) · run manifests
on every run · engine reads only the workspace, never file paths · QC reads
canonical and never mutates · repair never mutates its inputs.

## S5 test fixtures (reused by later steps)

`tests/test_repair_s5.py` reads the clean demo (`demo_data.generate()`, seed 7)
and references its injected-event constants (`PROMO_WINDOWS`, `STOCKOUT`,
`START_MONDAY`) so the assertions track the generator, not magic dates. It
proves: the injected stockout is flagged and recovered to within 20% of the
surrounding healthy level (and >4x the depressed raw); no other series is
censored; the two promos give positive uplift over a sensible de-promoted base
with the identity holding; the post-promo dip is not mis-flagged; graceful
degradation with empty inv/promo (the statistical test alone still catches the
stockout); no input mutation; the outlier rule dampens an isolated blip but
preserves a structural shift and never touches promo/censored weeks; and the
FVA-moves check for both statistical and translated forecasters.

## S5 -> S6 note

**S6 owns the statistical baseline + blend** (design §5–§7, v2 M5/M8). Replace
the S1 damped-EWMA placeholder in `engine._seasonal_naive` /
`statistical_forecaster` with a real ETS/AutoETS tier (Nixtla `statsforecast`
is in the stack's plan; keep seasonal-naïve as the fallback for short/sparse
series), and firm up the horizon-weighted blend in `_forecast_and_translate`
(days 1–14 sensed-dominant, decaying to plan by the horizon; tune the crossover
to where sensed stops beating the lag-adjusted plan). **Fit on the improved
`units_unconstrained`** that S5 now produces — the de-censored stockout weeks
and de-promoted base are exactly what make the statistical tier trustworthy.

Keep the `forecast` columns frozen (`item_id, region, week, lag, p10, p50, p90,
plan_units, blend_weight_sensed, blended`) so page 3's fan chart and the CSV/XLSX
export keep working. Do **not** touch the backtest surface, `fva`/`fva_by_lag`
shapes, the method vocabulary, page 5, `_repair`, or the frozen workspace/
ingest/quality contracts — extend the forecaster internals behind their existing
signatures and let the frozen harness re-judge. S6's acceptance (design §5, S6):
**positive FVA vs. naïve on demo data, and the CSV export works.** The honest
S1–S5 verdict is that the placeholder statistical tier does *not* yet clear the
≥15% bar; S6 is the step that starts earning it.

## Known stubs (deliberate, not bugs)

* Statistical tier is still a damped-EWMA placeholder, not ETS/AutoETS -> on
  demo data it does not yet clear the >=15% FVA bar. **S6 replaces it.**
* `translated_forecaster` is a naïve/statistical blend scored against the
  sell-out target; S7 replaces it with the channel-inventory projection and
  re-targets scoring to shipments (sell-in) — the harness itself won't change.
* ML toggle is wired through config but does not yet train a model (S8).
