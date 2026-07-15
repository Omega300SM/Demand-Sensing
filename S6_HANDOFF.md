# S6 → S7 Handoff

**Status: S6 complete.** The real **statistical baseline + blend** now lives in
the forecast/blend half of `sensing/engine.py` (design §5 "Sell-Out Forecast",
§7 "Blending", v2 M5/M8), replacing the S1 damped-EWMA placeholder. The tier is
a self-contained **damped-local-level model** (an ETS-family forecaster) built
for the POC's regime — short, promo-spiked, ~1.5-cycle CPG series — and it is
fit on S5's **`units_unconstrained`** (de-censored, de-promoted) target, which is
the whole reason repair landed first. The **Forecast Review** fan is now a
**genuine P10/P50/P90** interval, and the CSV/XLSX export lands the improved
blended forecast unchanged. Because the tier is judged **through the frozen
harness**, the FVA numbers move with no harness change. `pytest` is green
(**98 tests**: the S1–S5 83 + 15 new in `tests/test_forecast_s6.py`). One module
this session — the forecast + blend half of `sensing/engine.py`; no page edits
were needed (the `forecast` columns are unchanged). Everything else is untouched.

Earlier: **S5** — real signal repair (`_repair` + helpers) + page-3 shaded
censoring. **S4** — rolling-origin backtest harness + Accuracy/FVA page. **S3** —
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

### `RunResult` (sensing/engine.py) — dataclass  **[shapes FROZEN]**
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

## Workspace / Ingestion / Quality / Backtest / Repair APIs — FROZEN (unchanged in S6)

Workspace (`sensing/workspace.py`): `add_snapshot` · `read_canonical` ·
`canonical_status` · `rebuild_canonical` · `save_run` / `list_runs` /
`read_run_output` · `save_mapping` / `get_mapping` · `snapshot_coverage`.
Ingestion (`sensing/ingest_ui.py`) and Quality (`sensing/quality.py`,
`run_qc(...) -> QCReport`) are exactly as S2/S3 left them. The **backtest
harness** (`backtest` / `AsOfPanel` / `LeakageError` / `_backtest` /
`_plan_asof` / `_aggregate_fva` / `naive_forecaster` / `plan_forecaster` /
`FVA_METHODS` / `FVA_STEP_LABELS` / `DEFAULT_LAGS`) is exactly as S4 left it —
**do not touch**. **Repair** (`_repair` + its helpers, the `repaired` contract)
is exactly as S5 left it — **do not touch**. `app_common.run_manifest` unchanged.

---

## Statistical baseline + blend (sensing/engine.py) — **NEW in S6, now the forecast contract's owner**

`_seasonal_naive(history, steps, season=52) -> np.ndarray` is the **point path**
of the statistical tier (name kept for the frozen call sites). It is a
**damped-local-level** forecaster: de-spike the history (cap highs at rolling
`median + k·MAD` so promo plateaus don't inflate the level — no promo flag is
read, so this is not peeking), a robust EWMA level + damped local trend, plus a
**horizon-decaying persistence** of the last observation (`phi**lag`) that tracks
the autocorrelated plateaus near-in and reverts far-out, and a **guarded, shrunk
seasonal** factor that only activates past `season_min_cycles` cycles (OFF on the
1.5-cycle demo, where the harness shows it doesn't earn its keep). Fit on
`units_unconstrained`.

### Frozen `forecast` guarantees (page 3 fan + export + page 5 harness depend on these)
* Columns/dtypes exactly `item_id, region, week, lag, p10, p50, p90, plan_units,
  blend_weight_sensed, blended`; `p10 <= p50 <= p90` and all `>= 0`.
* `blend_weight_sensed` in `[0, 1]`; sensed-dominant near-in (weeks 1–2 = 0.9),
  monotonically decaying toward the plan by the horizon (0.2 at week 8).
* `blended` = `w·P50 + (1-w)·plan_units` (falls back to P50 where the plan is
  missing) — a convex mix, so **`blended` is never worse than `max(sensed, plan)`
  cell-by-cell**, hence on any backtest aggregate.
* `p50` is exactly the tier's point path; the fan's interval is genuine (empirical
  quantiles of the model's one-step residuals, widened by √lag — not a fixed ±).

### New forecast helper signatures (extend behind them; don't rename)
```python
STAT_PARAMS: dict                     # tunable knobs (see below), referenced by tests
_despike(y, window, k) -> np.ndarray                      # robust high-side cap
_seasonal_index(despiked, season, shrink, min_cycles) -> np.ndarray  # guarded factors
_seasonal_naive(history, steps, season=52) -> np.ndarray  # point path (frozen name)
_statistical_forecast(history, steps, season=52) -> (p10, p50, p90)  # genuine fan
_blend_weight_sensed(lag, cfg) -> float                   # §7 blend schedule
_autoets_point(y, steps, season) -> np.ndarray            # optional statsforecast
```
`STAT_PARAMS` defaults (calibrated on the demo): `phi=0.75`, `level_span=6`,
`despike_window=9`, `despike_k=3.0`, `trend_window=10`, `trend_damp=0.9`,
`season=52`, `season_min_cycles=1.75`, `season_shrink=0.5`, `q_lo=0.10`,
`q_hi=0.90`.

### Optional statsforecast (import-guarded)
`statsforecast` is now in `requirements.txt`, but the **default tier is the
hand-rolled damped-level model** — deterministic, dependency-free, and the suite
runs with or without the wheel (`try/except` import → `_HAS_STATSFORECAST`).
Flip `sensing.engine.USE_STATSFORECAST = True` to route the point path through
`AutoETS`. On the ~1.5-cycle demo AutoETS(season_length=52) collapses to a flat
level and does **not** beat the damped-level tier through the harness (and is
~100× slower per backtest), so it stays opt-in — an FVA-driven "does this step
earn its keep?" call. Re-evaluate once ≥2 clean cycles of history are uploaded.

### The harness re-judges automatically (S6 verdict on the demo)
Scored through the **unchanged** harness against `units_unconstrained`, the
statistical tier now clears the S6 bar and then some:

| lag | naïve WMAPE | statistical WMAPE |
|---|---|---|
| 1–2 (pooled, the S6 bar) | 0.180 | **0.175** ✅ positive FVA |
| 4 | 0.353 | **0.257** |
| 8 | 0.297 | **0.147** |
| 1,2,4,8 pooled | 0.245 | **0.190** (≈ +22%) |

The wide long-lag margin is the tier reverting to a sane level where last-value
naïve carries a promo spike forward. `blended` aggregate (0.178) ≤
`max(sensed 0.190, plan 0.168)` on the backtest — the M8 acceptance holds.
The honest caveat: this clears **positive FVA vs. naïve** (design §5, S6). The
**≥15–20% vs. the lag-adjusted plan** graduation bar (design §7 / exit criteria)
is still open — on the demo the plan is competitive at lags 1–2; **S7's
translation engine (sell-in) is the step expected to open that gap**, since the
plan is a sell-in artifact and the translated forecaster re-targets shipments.

---

## Where each remaining step plugs in

| Step | File(s) to replace/extend | Interface stays |
|---|---|---|
| ~~S2 Upload mapper~~ done | `ingest_ui.py`, `workspace.py`, page 1 | canonical schemas |
| ~~S3 QC cards~~ done | `quality.py`, page 1 | `run_qc(...) -> QCReport` |
| ~~S4 Backtest/FVA~~ done | `engine` backtest half, page 5 | `fva`, `fva_by_lag` shapes |
| ~~S5 Signal repair~~ done | `engine._repair` (+ helpers), page 3 | `repaired` columns |
| ~~S6 Statistical baseline + blend~~ done | `engine._seasonal_naive` / `_statistical_forecast` / `_forecast_and_translate` (forecast+blend half) | `forecast` columns |
| **S7 Translation engine** ← next | `engine._forecast_and_translate` (translation half) + `translated_forecaster` | `inventory_projection` columns |
| S8 Alerts + LightGBM | `engine._alerts`; add `ml_model.py` | `alerts` columns; `RunConfig.ml_enabled` |

Swap internals freely; keep the column contracts and the pages keep working.

## Working-method rules (carried from v2/v3, still binding)

One module per session · tests are the memory between sessions · **as-of leakage
assertions everywhere** (the tier fits only on `<= as_of` history; forecasts are
invariant to future values — enforced by the harness's `AsOfPanel`) · run
manifests on every run · engine reads only the workspace, never file paths · QC
reads canonical and never mutates · repair never mutates its inputs.

## S6 test fixtures (reused by later steps)

`tests/test_forecast_s6.py` reads the clean demo (`demo_data.generate()`) and
scores the statistical tier **through the frozen harness**. It proves: positive
FVA vs. naïve at lags 1–2 and wide wins at lags 4/8; the waterfall's statistical
step improves on naïve; genuine quantiles (`p10<=p50<=p90`, interval widens with
lag, P50 == the point tier); the blend schedule is clipped/shaped, sensed-dominant
near-in and decaying, and `blended` ≤ `max(sensed, plan)` on the backtest
aggregate (a convex-blend property); CSV **and** XLSX export round-trip on the
frozen columns; the tier fits only on `<= as_of` history (forecast invariant to
corrupted future values); and the guarded seasonal is exactly all-ones on ~1.5
cycles but activates (shrunk) past the guard, with short/sparse series degrading
gracefully.

## S6 → S7 note

**S7 owns the sell-out → sell-in translation engine** (design §6 "the
differentiating layer", v2 M7) — the **translation half of
`_forecast_and_translate`** (the `inventory_projection` block: project channel
inventory forward, generate projected orders to restore target WOS with
order-cycle/batching + reaction-lag constraints, **calibrated empirically** from
historic shipments vs. POS+inventory), and the **`translated_forecaster`
adapter**, which S7 re-targets to score against **shipments (sell-in)** rather
than the sell-out target. Add the fallback path (a distributed-lag transfer
function from sell-out to sell-in) for series where channel inventory is poor,
auto-selected per series by backtest (design §6 / v2 M7 acceptance: beat a
"shipments-history-only" model, or auto-fall back).

Keep the `inventory_projection` columns frozen
(`item_id, region, week, projected_on_hand, target_position, projected_order`)
and the `forecast` columns frozen. Do **not** touch the backtest surface,
`fva`/`fva_by_lag` shapes, the method vocabulary, page 5, `_repair`, the
**S6 forecast+blend half** (`_seasonal_naive` / `_statistical_forecast` /
`_blend_weight_sensed` / the blend portion of `_forecast_and_translate`), or the
frozen workspace/ingest/quality contracts — extend the translation internals
behind their existing signatures and let the frozen harness re-judge. S7's
acceptance (design §5, S7): **the translated forecaster beats a
shipments-history-only model on the backtest, or auto-falls back** — and it is
the step expected to open the ≥15% gap vs. the lag-adjusted (sell-in) plan.

## Known stubs (deliberate, not bugs)

* `translated_forecaster` is still a naïve/statistical blend scored against the
  **sell-out** target (it now rides the improved `_seasonal_naive`). **S7
  replaces it** with the channel-inventory projection and re-targets scoring to
  shipments (sell-in) — the harness itself won't change.
* Seasonal factors stay OFF on the demo's ~1.5 cycles by design (the guard);
  they activate once ≥1.75 cycles of clean history exist. Re-tune
  `season_shrink` / `season_min_cycles` against the harness when that lands.
* `USE_STATSFORECAST` is opt-in (default off) pending ≥2 clean seasonal cycles.
* ML toggle is wired through config but does not yet train a model (S8).
