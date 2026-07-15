# S7 → S8 Handoff

**Status: S7 complete.** The real **sell-out → sell-in translation engine**
(design §6 "the differentiating layer", §3 L3, v2 M7) now lives in the
translation half of `sensing/engine.py`, replacing the S1 placeholder. It
projects channel inventory forward and orders back to a target cover:

```
Projected position(t) = On-hand(t-1) + In-transit arrivals(t) - Sell-out(t)
Target position(t)    = Target WOS x forward sell-out forecast(t)
Projected order(t)    = Sell-out(t) + [Target position(t) - Projected position(t)]
```

with order-cycle **batching** (`cfg.order_cadence_weeks`) and a **reaction lag**
(`cfg.reaction_lag_weeks`). The behavioural parameters (target WOS, cadence,
reaction lag) and the **order response** are **calibrated empirically** from each
series' own historic shipments vs. POS + inventory (cfg values are only the
prior/fallback). Where channel inventory is unusable (phantom-flat, no draw-down)
the projection is replaced by a **distributed-lag transfer function** (a
regression of shipments on lags of POS), **chosen per series** by whichever wins
on an as-of-safe holdout. The translated forecaster is now **re-targeted to score
against SHIPMENTS (sell-in)** — the thing it actually predicts — through the
**unchanged** harness. `pytest` is green (**110 tests**: the S1–S6 98 + 12 new in
`tests/test_translate_s7.py`). One module this session — the translation half of
`engine.py` (+ one new poor-inventory fixture in `demo_data.py`); no page edits
were needed (the `inventory_projection` columns are unchanged).

Earlier: **S6** — statistical baseline + blend. **S5** — real signal repair +
page-3 shaded censoring. **S4** — rolling-origin backtest harness + Accuracy/FVA
page. **S3** — full M2 QC gate suite. **S2** — hardened upload layer. **S1** —
app skeleton + workspace/DuckDB + one-click demo + `run_pipeline` stub.

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
optional streams degrade gracefully (no inventory → the translation seeds cover
from the sell-out forecast / uses the transfer function; no shipments → the
`translated` FVA row is simply absent, the rest of the pipeline is unaffected).

## Workspace / Ingestion / Quality / Backtest / Repair / S6-forecast — FROZEN (unchanged in S7)

Workspace (`sensing/workspace.py`): `add_snapshot` · `read_canonical` ·
`canonical_status` · `rebuild_canonical` · `save_run` / `list_runs` /
`read_run_output` · `save_mapping` / `get_mapping` · `snapshot_coverage`.
Ingestion (`sensing/ingest_ui.py`) and Quality (`sensing/quality.py`,
`run_qc(...) -> QCReport`) are exactly as S2/S3 left them. The **backtest
harness** (`backtest` / `AsOfPanel` / `LeakageError` / `_plan_asof` /
`_aggregate_fva` / `naive_forecaster` / `statistical_forecaster` /
`plan_forecaster` / `FVA_METHODS` / `FVA_STEP_LABELS` / `DEFAULT_LAGS`) is exactly
as S4 left it — **do not touch** (S7 only EXTENDED the private `_backtest`
*assembly* to pass a shipments frame + `value_col` for the translated method; the
harness surface and output shapes are unchanged). **Repair** (`_repair` + its
helpers, the `repaired` contract) is exactly as S5 left it. The **S6 forecast +
blend half** (`_seasonal_naive` / `_statistical_forecast` /
`_blend_weight_sensed` / the forecast + blend portion of
`_forecast_and_translate`, the `forecast` columns) is exactly as S6 left it —
**do not touch**. `app_common.run_manifest` unchanged.

---

## Translation engine (sensing/engine.py) — **NEW in S7, owns `inventory_projection` + the `translated` FVA row**

### Frozen `inventory_projection` guarantees (page 3 renders these; reaffirmed)
* Columns/dtypes exactly `item_id, region, week, projected_on_hand,
  target_position, projected_order`.
* `projected_on_hand >= 0` and `projected_order >= 0` **always** (an overstocked
  channel yields a *zero* order — the bullwhip-drain signal — never negative).
* `target_position` is always the target-WOS cover for the week
  (`target_wos * forward sell-out P50`).
* The projection is seeded from the **latest ACTUAL on-hand (+ in-transit) at
  as_of** and rolls forward by the design recurrence; the sell-out path it
  consumes is S6's point path (`p50`), read, not recomputed differently.

### Two paths, one contract
* **Display projection** (the `inventory_projection` output, built in the
  translation half of `_forecast_and_translate`): the **rigid design formula**
  with `target_wos` + recovery rate fitted from channel inventory. This function
  keeps its **frozen signature** `(repaired, inv, plan, as_of, cfg)` — it has no
  shipments, so cadence/reaction-lag take the cfg prior there.
* **Scored translation** (the `translated` FVA row, built by
  `make_translated_forecaster`, which closes over repaired + inventory +
  shipments): the design formula **generalized by an empirically-fitted order
  response** (`_fit_order_model`), or the **transfer function** where inventory
  is unusable — chosen per series by `_select_translation_method`. Scored against
  `units_shipped`.

### New translation helper signatures (extend behind them; don't rename)
```python
TRANSLATE_PARAMS: dict                       # tunable knobs (bounds, holdout, lags)
@dataclass TranslationParams(target_wos, order_cadence_weeks,
                             reaction_lag_weeks, usable_inv, source)  # per-series fit
_inventory_usable(on_hand, sell_out, p) -> bool         # phantom/no-drawdown check
_fit_target_wos(on_hand, sell_out, censored, p, fallback) -> float
_fit_order_cadence(shipments, p, fallback) -> int
_fit_reaction_lag(sell_out, shipments, p, fallback) -> int
_calibrate_translation(sell_out, on_hand, shipments, censored, cfg) -> TranslationParams
_projected_position_hist(sell_out, on_hand, in_transit) -> np.ndarray
_fit_order_model(sell_out, on_hand, in_transit, shipments) -> (b0,b1,b2) | None
_fit_replenishment(sell_out, on_hand) -> float           # on-hand recovery rate, [0,1.5]
_project_orders(sell_out_fc, seed_on_hand, seed_in_transit, params,
                order_model=None, replenish=1.0) -> (proj_on_hand, target_pos, orders)
_transfer_function(sell_out_hist, ship_hist, sell_out_fc, p=None) -> np.ndarray
_select_translation_method(sell_out, on_hand, in_transit, ship, params, p) -> "projection"|"transfer"
make_translated_forecaster(repaired, inv, ship, cfg) -> forecaster   # the scored adapter (factory)
translated_forecaster(panel, as_of, horizon) -> df   # bare single-series adapter (kept for S5)
shipments_history_only_forecaster(panel, as_of, horizon) -> df       # the benchmark to beat
_shipments_panel(repaired, ship) -> df               # long units_shipped frame for scoring
```
`TRANSLATE_PARAMS` defaults: `wos_min=1.0`, `wos_max=12.0`, `wos_min_weeks=12`,
`lag_max=4`, `cadence_max=4`, `inv_usable_finite=0.6`, `inv_usable_cv=0.02`,
`inv_drawdown_corr=0.0`, `tf_max_lag=3`, `select_holdout=8`,
`select_lags=(1,2,3,4)`.

### FVA re-targeting note (important, and by design)
`_backtest` now scores the **translated** method against `units_shipped`
(sell-in) on a shipments panel, while **naive / statistical / plan stay on
`units_unconstrained`** (sell-out) — exactly the split the S7 prompt specified.
Consequence: in the FVA waterfall / `fva_by_lag`, the `translated` row measures
**sell-in** accuracy and the other three measure **sell-out** accuracy. They are
all relative (WMAPE) so they render side by side, but they answer different
questions; don't read the translated row as directly comparable to the sell-out
rows. A future step could add a sell-in plan comparison if a sell-in plan is
uploaded — the method vocabulary is frozen, so that would be a scoring-frame
change inside `_backtest`, not a new method.

### The harness re-judges automatically (S7 verdict on the demo)
Scored through the **unchanged** harness against `units_shipped`, the translation
beats the shipments-history-only benchmark at every near-lag band:

| lags | shipments-only WMAPE | translated WMAPE | rel. |
|---|---|---|---|
| 1–2 | 0.152 | **0.149** | +1.7% |
| 1,2,4 | 0.176 | **0.173** | +2.2% |
| 4,8 | 0.251 | **0.236** | +6.0% |
| 1,2,4,8 | 0.199 | **0.191** | +4.3% |

On the demo the per-series selector keeps the **inventory projection** (channel
inventory is clean and draws down); the **transfer function** is exercised and
auto-selected on the new poor-inventory fixture. The near-lag win comes from the
exact actual on-hand seed at as_of; the wider long-lag win comes from the fitted
order response tracking the channel's persistent cover deficit that a
shipments-only carry-forward cannot see.

---

## Where each remaining step plugs in

| Step | File(s) to replace/extend | Interface stays |
|---|---|---|
| ~~S2 Upload mapper~~ done | `ingest_ui.py`, `workspace.py`, page 1 | canonical schemas |
| ~~S3 QC cards~~ done | `quality.py`, page 1 | `run_qc(...) -> QCReport` |
| ~~S4 Backtest/FVA~~ done | `engine` backtest half, page 5 | `fva`, `fva_by_lag` shapes |
| ~~S5 Signal repair~~ done | `engine._repair` (+ helpers), page 3 | `repaired` columns |
| ~~S6 Statistical baseline + blend~~ done | `engine` forecast+blend half | `forecast` columns |
| ~~S7 Translation engine~~ done | `engine` translation half + `make_translated_forecaster` + `_backtest` shipments wiring | `inventory_projection` columns |
| **S8 Alerts + LightGBM** ← next | `engine._alerts`; add `ml_model.py` | `alerts` columns; `RunConfig.ml_enabled` |

Swap internals freely; keep the column contracts and the pages keep working.

## Working-method rules (carried from v2/v3, still binding)

One module per session · tests are the memory between sessions · **as-of leakage
assertions everywhere** (the translated forecaster reads only `<= as_of`
POS/inventory/shipments — `make_translated_forecaster` truncates each frame at
`panel.as_of` and the harness's `AsOfPanel` re-asserts it) · run manifests on
every run · engine reads only the workspace, never file paths · QC reads
canonical and never mutates · repair/translation never mutate their inputs.

## S7 test fixtures (reused by later steps)

`tests/test_translate_s7.py` reads the clean demo (`demo_data.generate()`) and a
**new** `demo_data.poor_inventory_streams()` (a single series with phantom-flat,
unusable channel inventory — added as an optional generator; `generate()` is
untouched). It proves, through the frozen harness: the translation beats
shipments-history-only at lags 1–2 and 1–4; the projection identity + full
non-negativity + recurrence reconciliation; cadence batching (orders on the
cadence only) and reaction-lag delay of a seed in-transit; empirical calibration
returns sane, in-band params and is as-of safe (corrupting future weeks can't
change a `<= as_of` fit); the transfer function is auto-selected on the
poor-inventory series and stays finite + competitive; and the translated
forecaster is invariant to corrupted future POS/inventory/shipments.

## S7 → S8 note

**S8 owns alerts hardening + the optional ML tier** (design §8 S&OE alerts, v2
M9; §5 ML tier). Two pieces:

1. **Alerts** (`engine._alerts`, the `alerts` columns are frozen:
   `severity, alert_type, item_id, region, week, message`). Harden the exception
   classes against the now-real signals: cumulative sensed-vs-plan deviation,
   projected retailer stockout within lead time (reads the S7
   `inventory_projection`), channel overstock / order-cliff (S7 projected orders
   collapsing), promo mid-flight under/over-performance, QC/censoring holds. The
   acceptance (design §8, S8): **replay a historical event and confirm an alert
   fires early enough to act.** Page 4 already loops over the frozen `alerts`
   columns and shows the driving-signal chart, so sharper rules surface with
   no page change.
2. **Optional LightGBM tier** behind `RunConfig.ml_enabled` (currently wired
   through config but inert). Add `sensing/ml_model.py` — a single global
   gradient-boosted model across all SKU-regions, direct multi-horizon, quantile
   objective for P10/P50/P90, features per design §5 (lag/rolling velocities,
   calendar, promo mechanics, price, distribution, channel-inventory position,
   item attributes for cold-start). **It must be judged through the frozen
   harness** (positive FVA vs. the S6 statistical tier at lags 1–4, no future
   leakage — assert every feature is computable at `as_of`) and slot in behind
   `run_pipeline` with `ml_enabled=True` selecting it where it wins, the
   statistical tier as the fallback. `lightgbm` is commented in
   `requirements.txt`; import-guard it exactly like `statsforecast` so the suite
   still runs without the wheel.

Keep the `alerts` columns and every other RunResult contract frozen, the FVA
method vocabulary frozen, and **do not touch** the backtest harness surface,
page 5, `_repair`, the S6 forecast+blend half, the **S7 translation half**
(`_project_orders` / `_calibrate_translation` / `_fit_order_model` /
`make_translated_forecaster` / the `inventory_projection` output), ingest/quality/
workspace/config, or `demo_data.generate()`. Extend behind the frozen signatures
and let the harness re-judge.

## Known stubs (deliberate, not bugs)

* `RunConfig.ml_enabled` is wired through config + manifest but does **not** yet
  train a model — **S8** adds `ml_model.py` behind it.
* The **display** `inventory_projection` uses the rigid design formula (no
  shipments in that frozen signature); the **scored** translated forecaster
  additionally fits the order response to shipments. Both honour the frozen
  columns; the small divergence is documented above (two paths, one contract).
* Seasonal factors in the S6 tier stay OFF on the demo's ~1.5 cycles by design
  (the guard); they activate past 1.75 cycles. `USE_STATSFORECAST` is opt-in
  (default off) pending ≥2 clean seasonal cycles.
