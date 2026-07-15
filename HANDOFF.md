# S8 → v2 Handoff (build complete)

**Status: S8 complete — the engine build is done.** This session hardened the
**alerts** engine against the now-real S5–S7 signals and added the **optional ML
tier** (`sensing/ml_model.py`) behind `RunConfig.ml_enabled`, judged through the
**unchanged** backtest harness (champion–challenger vs. the S6 statistical tier).
`pytest` is green (**129 tests**: the S1–S7 110 + 19 new in
`tests/test_alerts_ml_s8.py`). Two surfaces touched, both behind frozen
contracts: `engine._alerts` (+ helpers) and the new `ml_model.py`, plus three
thin `ml_enabled` routing hooks in `run_pipeline` / `_forecast_and_translate` /
`_backtest`. No page edits (the `alerts` columns and every RunResult contract are
unchanged), and with `ml_enabled=False` the pipeline is **byte-for-byte S7**.

The alerts engine now emits the full design-§8 exception suite — cumulative
sensed-vs-plan deviation, projected retailer stockout within lead time (reads the
S7 `inventory_projection`), channel overstock / order-cliff, promo mid-flight
variance, and signal-repair / censoring holds — each row carrying the driving
signal in its `message`. The ML tier is a single global gradient-boosted quantile
model across all SKU-regions (direct multi-horizon, P10/P50/P90), import-guarded
exactly like `statsforecast`: with no `lightgbm` wheel it falls back to a
self-contained deterministic gradient-boosted-stumps learner, and the
champion–challenger backtest decides ML-vs-statistical **per series** either way,
routing the winner through the **frozen** `statistical` FVA step (no new method).

Earlier: **S7** — sell-out → sell-in translation engine. **S6** — statistical
baseline + blend. **S5** — real signal repair + page-3 shaded censoring. **S4** —
rolling-origin backtest harness + Accuracy/FVA page. **S3** — full M2 QC gate
suite. **S2** — hardened upload layer. **S1** — app skeleton + workspace/DuckDB +
one-click demo + `run_pipeline` stub.

This is the final engine handoff. The v3 POC build is complete; the remaining
road is **graduation to the v2 lake-connected build** (see the closing section).

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

## Alerts engine (sensing/engine.py) — **hardened in S8, owns the `alerts` output**

### Frozen `alerts` contract (page 4 renders these; unchanged)
Columns/dtypes exactly `severity, alert_type, item_id, region, week, message`,
`severity ∈ {high, medium, low}`. Page 4 routes the driving-signal chart by a
**substring of `alert_type`**: `"stockout"`/`"overstock"` → the
`inventory_projection` chart, `"deviation"` → forecast P50-vs-plan, else → the
repaired raw-vs-repaired chart. The `alert_type` strings below preserve those
substrings, so sharper rules surfaced with **no page change**.

### The five exception classes (each helper returns one row or `None`)
```python
_ALERT_COLS = ["severity","alert_type","item_id","region","week","message"]
ALERT_PARAMS: dict                                  # tunable thresholds (below)
_sev_by_magnitude(ratio, hi=2.0) -> "high"|"medium"|"low"
_alert_deviation(g, cfg, sku, region)  -> dict|None # "Sensed-vs-plan deviation"
_alert_stockout(gp, cfg, sku, region)  -> dict|None # "Projected retailer stockout"
_alert_order_cliff(gp, sku, region)    -> dict|None # "Channel overstock / order-cliff"
_alert_promo(rep_g, cfg, sku, region)  -> dict|None # "Promo mid-flight variance"
_alert_censoring(rep_g, cfg, sku, region) -> dict|None # "Signal repair / censoring"
_alerts(forecast, proj, repaired, cfg, promo=None) -> pd.DataFrame
```
`ALERT_PARAMS`: `stockout_cover_weeks=0.15`, `stockout_buffer=1`,
`cliff_order_frac=0.05`, `cliff_cover_frac=1.30`, `promo_window=6`,
`promo_dev=0.35`, `censor_window=8`. The deviation rule is **cumulative**
(sensed-vs-plan summed over the horizon, not single-week noise) and scales
severity by magnitude; promo/censoring read **only `repaired` weeks `<= cfg.as_of`**
so they never peek past the origin.

### One non-obvious tuning fact worth keeping (don't "fix" it away)
The **frozen display** `inventory_projection` is the rigid design-formula
projection (no fitted order response — that lives only in the *scored* translated
forecaster). On every demo series it decays cover to ~0 across the horizon, so a
naïve "% of target position" stockout floor **false-fires on all series always**.
`_alert_stockout` therefore keys off **weeks-of-cover**
(`projected_on_hand / (target_position / target_wos)`) with a `0.15`-week floor,
and **skips the reaction-lag seed ramp** (a week-0 `projected_on_hand=0` is a
reaction-lag artefact — a reorder physically can't land inside the lag, so it is
not actionable). It scans `gp.iloc[reaction_lag : reaction_lag+buffer+1]`. This is
why the stockout alert is quiet on clean series and only speaks when cover is
genuinely, actionably short.

### S8 alert verdict on the demo (replay = the design-§8 acceptance)
Driving the frozen demo (`demo_data.generate()`, seed 7 — stockout injected on
SKU-1002/East weeks 38–42, promos weeks 18–21 & 54–57) through `run_pipeline`:

| as_of | alerts fired | reading |
|---|---|---|
| week 30 (clean, pre-stockout) | **0** | silent when nothing is wrong |
| week 44 (just after the stockout) | **1** — `Signal repair / censoring`, high, SKU-1002/East | the real injected event, caught early enough to act |
| week 52 (promo-2 in the horizon) | 6 — `Sensed-vs-plan deviation`, high | the plan under-scopes the promo; sensing flags it |
| final week (no plan ahead, stockout out of window) | 0 | no false positives at the horizon edge |

These four as-ofs are the natural end-to-end replay anchors and are asserted in
`tests/test_alerts_ml_s8.py`.

---

## ML tier (sensing/ml_model.py) — **NEW in S8, optional, behind `RunConfig.ml_enabled`**

A single **global** gradient-boosted **quantile** model across all SKU-regions
(design §5 ML tier), direct multi-horizon (the horizon step `h` is a feature),
trained on a **scale-normalised ratio target** (`y[t+h] / level_t`, where `level`
is a robust mean of the last 13 weeks) so sparse and high-volume series share one
model. Import-guarded exactly like `statsforecast`.

### Public surface (extend behind these; don't rename)
```python
_HAS_LIGHTGBM: bool
ML_PARAMS: dict            # min_hist=16, level_window=13, val_weeks=6,
                           # n_estimators=60, learning_rate=0.12, max_bins=24,
                           # min_leaf=8, q_lo=0.10, q_hi=0.90, champion_lags=4,
                           # champion_margin=0.02
FEATURE_NAMES: list[str]   # 14 as-of-safe features (see below)
class _GBStumps            # deterministic gradient-boosted depth-1 stumps
                           #   (vectorised prefix-sum split search; the fallback)
class GlobalQuantileModel  # backend switch: 3 LightGBM quantile models OR the
                           #   fallback point + residual band widened by sqrt(h);
                           #   guarantees p10 <= p50 <= p90, non-negative
make_ml_forecaster(promo, inv, cfg, *, params=None) -> forecaster
    # PURE ML adapter for the frozen `backtest` harness (reads panel.history())
make_champion_forecaster(promo, inv, cfg, *, point_fn, params=None) -> forecaster
    # champion–challenger: ML where it wins an inner <=as_of holdout, else point_fn
sellout_quantiles(sellout, promo, inv, as_of, cfg, *, point_fn, params=None)
    -> (override: dict[(sku,region) -> (p10,p50,p90)], used_ml_keys: list)
    # production override: populated ONLY for series where ML is champion
```
`FEATURE_NAMES` (all computable at `as_of`): `r_last`, `r_mean2/4/8`, `r_std8`,
`r_slope4`, `r_yoy`, `zero_frac8`, `woy_sin/cos`, `h`, `promo_t`, `promo_tgt`,
`inv_ratio`. `promo_tgt` is legitimately known-future (from the trade calendar);
`inv_ratio` uses only `<= as_of` inventory. No feature reads past the origin.

### How it routes (three thin hooks, default path untouched)
* `_forecast_and_translate(..., sellout_override=None)` — the **one** hook line:
  where `(sku,region)` is in the override, the sell-out `p10/p50/p90` come from
  ML; otherwise from `_statistical_forecast`. `sellout_override=None` (the
  default) → **byte-for-byte S7**.
* `run_pipeline` — when `config.ml_enabled`, computes
  `sellout_quantiles(repaired, promo, inv, as_of, config, point_fn=_seasonal_naive)`,
  passes the override into `_forecast_and_translate`, and passes `promo=` into
  `_alerts` and `_backtest`. Manifest gains `ml_enabled`, `ml_backend`
  (`"lightgbm"` | `"fallback"` | `None`), and `ml_champion_series` (the
  `"sku|region"` list ML won).
* `_backtest(..., promo=None)` — when `cfg.ml_enabled`, the **statistical step is
  replaced by** `make_champion_forecaster(promo, inv, cfg, point_fn=_seasonal_naive)`
  — **method name stays `"statistical"`** (the FVA vocabulary is frozen; ML never
  adds a row, it competes for the existing step).

### Champion–challenger rule (the design-§5 / §9 discipline)
The decision is **aggregate WMAPE at lags 1..`champion_lags`** on an **inner
holdout drawn only from `<= as_of` data**, ML must beat statistical by
`champion_margin` (2%) to switch, and the pick is **per series and per origin** —
so a backtest origin can route some series through ML and others through the
statistical point path, all still scored as the single `statistical` FVA step.

### S8 ML verdict on the demo (honest, and the reason it stays optional)
With **no `lightgbm` wheel** in the environment the tier runs on the **fallback**
learner (`ml_backend="fallback"`). Through the **unchanged** harness the champion
switches to ML at *some* backtest origins (the `statistical` FVA row moves when
`ml_enabled` flips — e.g. on one demo run lag-2 WMAPE `0.194 → 0.164`, lag-4
`0.263 → 0.234`), but at the **final production `as_of` the aggregate winner is
the statistical tier**, so `ml_champion_series=[]` and the production override is
empty. That is the correct, conservative outcome on 1.5 seasonal cycles of demo
data: **ML is wired, judged, and only ever used where it demonstrably wins** — it
never silently degrades the default. On real data with ≥2 seasonal cycles and a
`lightgbm` wheel, the same harness re-judges automatically with no code change.

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
| ~~S8 Alerts + ML tier~~ **done** | `engine._alerts` (+ helpers); new `ml_model.py`; `ml_enabled` hooks | `alerts` columns; `RunConfig.ml_enabled` |

Every step is built. Swap internals freely behind the frozen column contracts
and the pages keep working; the next move is the v2 graduation, not another S-step.

## Test fixtures (reused across steps)

`tests/test_alerts_ml_s8.py` (19 tests) reads the clean demo
(`demo_data.generate()`) and builds small hand-made frames for the alert-helper
unit tests. It proves: each of the five exception classes fires on its seeded
scenario and stays silent on a clean one; the demo replay anchors above (week 44
censoring fires for SKU-1002/East, week 30 silent, final week no false
positives); the `alerts` columns/dtypes + severity vocabulary are exactly the
frozen shape; the `alert_type` strings still route the page-4 charts; the ML tier
beats-or-ties the statistical tier at lags 1–4 through the frozen harness (or is
correctly *not* selected, with the statistical fallback scored); the champion
never scores worse than statistical and only switches where it wins; the ML
forecast is invariant to corrupted future rows and forecasts strictly-future
weeks (as-of safe); `ml_enabled=False` reproduces the S7 forecast/FVA
byte-for-byte; and the dependency-free fallback learner trains, orders its
quantiles, and is scored end-to-end when `lightgbm` is absent.

Earlier fixtures still hold: `tests/test_translate_s7.py` uses
`demo_data.poor_inventory_streams()` (phantom-flat inventory, added as an
optional generator — `generate()` untouched) to exercise the transfer-function
fallback; S2's `messy_fixtures()` / S3's `defects_fixtures()` remain the ingest
and QC fixtures.

## Graduating to the v2 lake-connected build

The v3 POC exists to answer one question — *does sensing beat the plan on this
business's data?* — and the engine is now complete enough to answer it. When the
POC clears its exit criteria (design §7: ≥15–20% relative WMAPE improvement vs.
the lag-adjusted plan at 1–2 week lags with bias within ±5%, a planner completing
the weekly cycle unassisted, and ≥2 live alerts that led to action), graduation
to v2 is a **one-layer swap**, by construction:

1. **Replace `ingest_ui.py` with lake pipelines.** Every engine function already
   reads **only** the `canonical_*` DuckDB tables, never file paths — so the
   whole change is landing the same canonical schemas from scheduled lake
   extracts instead of manual uploads. Nothing downstream of ingestion moves.
2. **Schedule `run_pipeline`.** The one interface
   `run_pipeline(workspace, as_of, config) -> RunResult` becomes an overnight job;
   the run manifest (config hash, snapshot dates, ML backend/champions) is already
   written on every run for reproducibility.
3. **What carries forward unchanged:** the engine (repair → forecast → translate
   → blend → alerts), the frozen backtest harness and its accumulated FVA history,
   the empirically-calibrated retailer parameters (target WOS, cadence, reaction
   lag per series), and the ML tier — which re-judges itself through the same
   harness the moment real data and a `lightgbm` wheel are present, with no code
   change.
4. **What v2 adds on top** (deferred, not missing — design §8, v1 §10): live lake
   scheduling, planner overrides with reason codes + FVA tracking of the human
   touches, the S&OE workbench in place of the read-only Streamlit review, and —
   if series count demands — distributed ML training and hierarchical
   reconciliation.

The FVA waterfall is the graduation gate: if the POC's exit number falls short,
the waterfall shows **which layer** (statistical, ML, translation, or plan) isn't
earning its keep, so the v2 build starts by fixing or cutting that layer rather
than scaling everything.

## Working-method rules (carried from v2/v3, still binding)

One module per session · tests are the memory between sessions · **as-of leakage
assertions everywhere** (the ML tier reads only `<= as_of` history; every feature
is computable at the origin; `promo_tgt` is legitimately known-future) · run
manifests on every run · engine reads only the workspace, never file paths · QC
reads canonical and never mutates · repair / translation / alerts / ML never
mutate their inputs · **do not rename or add FVA methods** (ML competes for the
frozen `statistical` step) · import-guard optional wheels (`statsforecast`,
`lightgbm`) so the suite runs without them.

## Known stubs & honest limitations (deliberate, not bugs)

* **ML stays optional and, on the demo, unused in production.** On 1.5 seasonal
  cycles the statistical tier wins the aggregate champion decision at the final
  as_of, so `ml_champion_series=[]` there. This is correct conservatism, not a
  gap — the tier is wired, judged through the frozen harness, and switched on
  per-series only where it beats statistical by the `champion_margin`. Real data
  with ≥2 cycles (and a `lightgbm` wheel) re-judges it automatically.
* **Two projection paths, one contract.** The **display** `inventory_projection`
  uses the rigid design formula; the **scored** translated forecaster additionally
  fits the order response to shipments. Both honour the frozen columns; the
  divergence is why `_alert_stockout` keys off weeks-of-cover rather than the
  display floor (documented above).
* **Seasonal factors** in the S6 tier stay OFF on the demo's ~1.5 cycles by design
  (the guard activates past 1.75 cycles). `USE_STATSFORECAST` is opt-in (default
  off) pending ≥2 clean seasonal cycles.
* **POC scope caps** carry over from design §8: manual-upload freshness, a
  single-user local workspace with no auth, Streamlit slowing past ~500 series,
  and planner reactions via the exported CSV rather than in-app overrides — all
  deliberately deferred to v2.
