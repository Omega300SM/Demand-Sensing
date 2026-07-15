# S3 → S4 Handoff

**Status: S3 complete.** The QC validation cards are hardened to the full M2
gate suite (design §3.3, v1 §3.3) and proven: every seeded defect surfaces in
the UI. `pytest` is green (**58 tests**: the S1/S2 31 + 27 new in
`tests/test_quality_s3.py`). One module this session — `sensing/quality.py`
plus the QC section of `app/pages/1_Upload_Data.py`, with a defects-seeding
helper in `demo_data.py`.

Earlier: **S2 complete** — upload layer hardened (wide-format unpivot, saved
mappings, configurable week-calendar, multi-file stacking with grain de-dup).
**S1 complete** — app skeleton (5 pages), workspace + DuckDB wiring, one-click
synthetic demo, engine stubbed behind `run_pipeline`.

Carry this file into the next Claude session along with the v3 design doc. It
freezes the contracts every later step builds against.

---

## The one interface everything runs behind

```python
from sensing import run_pipeline, RunConfig
result = run_pipeline(workspace, as_of, config)   # -> RunResult
```

**Rule that must not break:** every engine function reads ONLY from the
workspace DuckDB (canonical_* tables), never from file paths. That is what makes
the v3→v2 graduation a one-layer swap of `ingest_ui.py` for lake pipelines.

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
| `fva` | step, wmape, bias  (waterfall at lag 1–2) |
| `fva_by_lag` | lag, method, wmape, bias  (method ∈ naive/statistical/translated/plan) |
| `manifest` | dict: run_id, as_of, config, inputs, snapshots |

`result.outputs()` returns the dict persisted to the workspace and re-read by
the pages via `ws.read_run_output(run_id, name)`.

---

## Canonical schemas (sensing/config.py `CANONICAL_SCHEMAS`)

Grain: **item_id × region × week** (weekly for the POC). Streams:
`pos` (required), `channel_inventory`, `shipments` (required), `demand_plan`,
`promo`. Missing optional streams degrade gracefully.

## Workspace API (sensing/workspace.py) — FROZEN

`add_snapshot` · `read_canonical` · `canonical_status` · `save_run` /
`list_runs` / `read_run_output` · `save_mapping` / `get_mapping` ·
`snapshot_coverage`. **`rebuild_canonical()`** stacks all snapshots per stream
and de-dups on the stream grain (`_grain_keys`) with last-write-wins by upload
time (`demand_plan` keeps distinct `plan_version_date` vintages).

## Ingestion API (sensing/ingest_ui.py) — FROZEN (unchanged in S3)

`read_upload` · `suggest_mapping` · `apply_mapping(..., unit_multiplier=1.0,
week_calendar=None)` · `is_wide_format` / `unpivot_wide` /
`suggest_wide_id_cols` · `WeekCalendar` · `missing_required` ·
`source_signature` · `looks_wm_labels`.

## Quality API (sensing/quality.py) — **CHANGED in S3**

```python
run_qc(stream, df, crosswalk=None, pos=None, *,
       as_of=None, sla_weeks=2, phantom_weeks=3,
       coverage_shift_threshold=0.30) -> QCReport
```

* **Backward-compatible.** `crosswalk` and `pos` keep their positions; the new
  behaviour is opt-in via keyword-only args that default to the S1/S2 semantics.
  Old positional call `run_qc(stream, df, crosswalk, pos)` still works.
* **FROZEN shapes (unchanged):** `QCReport(stream, rows, checks)` with `.worst`;
  `QCCheck(name, status ∈ {"pass","warn","fail"}, detail, data)`. New gates are
  added as extra checks/attachments — shapes and the three status values do not
  change.
* **Full M2 gate suite**, in render order (freshness first, because it decides
  whether weekly sensing is viable):
  1. `Freshness / SLA` — latest week vs. `as_of`; pass ≤ `sla_weeks`, warn ≤ 2×,
     fail beyond. When `as_of=None` it reports the latest week and passes.
  2. `Date coverage` — contiguous weeks; attaches the **missing weeks** list.
  3. `Negative values` — attaches the offending rows.
  4. `Impossible / range values` — `promo_flag ∉ {0,1}`; `demand_plan`
     `plan_version_date` on/after its target week (**leakage**); grossly
     implausible magnitudes (>100× column median). Attaches offenders + a
     `reason` column.
  5. `Duplicate rows` — on the stream **grain** (vintage-aware for demand_plan);
     attaches the colliding rows.
  6. `Coverage shift` — week-over-week adds/drops in the active item×region
     series set ≥ `coverage_shift_threshold`; attaches affected weeks with
     `n_added`/`n_dropped`/`added`/`dropped`.
  7. `Crosswalk match` — unmatched-items list (unchanged).
  8. `Phantom inventory` — an **N-consecutive-week** run of (on_hand>0 AND zero
     sell-out), `N=phantom_weeks`, per series; attaches series + run weeks.
* **Attachment convention (NEW):** *any* check may set `chk.data` to a DataFrame
  of offenders. Page 1 already renders a generic per-check download
  (`qc_<stream>_<check>.csv`) for whatever a check attaches — quarantine-and-flag,
  the planner's cleanup to-do. No page change is needed for future attachments.
* **Never mutates** its input (flag-don't-impute; `test_checks_never_mutate_input`).
* **Page 1** passes `as_of=date.today()` (a real weekly export should be recent)
  and renders freshness first.

**Note on duplicates + rebuild:** the duplicate gate fires on an uploaded
file/snapshot, but `rebuild_canonical()` de-dups on the grain, so an exact
grain-duplicate does **not** survive into `canonical_*`. The gate is therefore
most meaningful at the file level (it tells the planner a collision existed and
was resolved by last-write-wins). Post-rebuild, the other seven gates still
surface. See `test_qc_demo_files_surface_every_defect_class` (file view) vs.
`test_qc_demo_loads_into_workspace_and_survives_rebuild` (canonical view).

## Defects demo (NEW in S3) — entry points

`sensing/demo_data.defects_fixtures()` → the five streams of a compact
(2 base SKUs × 2 regions × 12 weeks) dataset seeded with **exactly one instance
of each defect class**: a date gap, a negative, a duplicate, an unmatched item,
a coverage shift (adds + drops), a phantom-inventory run, a stale feed, a leaky
plan vintage, and a bad promo flag. `defects_as_of()` returns the reference date
the freshness gate is meant to see (the last base week; only channel_inventory
trips it under that as-of). `write_defects_fixtures(dir)` writes them to CSV
(also emitted into `templates/demo_dataset/` by `write_templates`).
`load_qc_demo_into_workspace(ws)` lands them + rebuilds. The upload page has a
one-click **🧪 Load QC demo** button beside Rebuild. The engine's clean
`generate()` demo is untouched, so S1/S2 tests stay green.

---

## Where each remaining step plugs in

| Step | File(s) to replace/extend | Interface stays |
|---|---|---|
| ~~S2 Upload mapper polish~~ ✅ | `ingest_ui.py`, `workspace.py`, page 1 | canonical schemas |
| ~~S3 QC cards~~ ✅ | `quality.py`, page 1 | `run_qc(...) -> QCReport` (now with kwargs) |
| **S4 Backtest/FVA** ← next | `engine._backtest`, page 5 | `fva`, `fva_by_lag` shapes |
| S5 Signal repair | `engine._repair` | `repaired` columns |
| S6 Statistical baseline + blend | `engine._seasonal_naive`, `_forecast_and_translate` | `forecast` columns |
| S7 Translation engine | `engine._forecast_and_translate` (translation half) | `inventory_projection` columns |
| S8 Alerts + LightGBM | `engine._alerts`; add `ml_model.py` | `alerts` columns; `RunConfig.ml_enabled` |

Swap internals freely; keep the column contracts above and the pages keep
working untouched.

## Working-method rules (carried from v2/v3, still binding)

One module per session · tests are the memory between sessions · **as-of leakage
assertions everywhere** (no feature/forecast may read data after `as_of`) · run
manifests on every run · engine reads only the workspace, never file paths ·
QC reads canonical (page path) and never mutates (flag, don't impute).

## S3 test fixtures (reused by app + tests)

`sensing/demo_data.defects_fixtures()` / `write_defects_fixtures(dir)` /
`load_qc_demo_into_workspace(ws)` / `defects_as_of()`. `tests/test_quality_s3.py`
proves, for every gate, a fixture where it fires (correct status + attached
offenders) and a clean pass on `generate()` (no false positive), plus
signature back-compat, no-mutation, frozen statuses, and the headline
"all defects surface" acceptance.

## S3 → S4 note

S4 owns **the backtest harness + the FVA page** (design §4 Accuracy/FVA, v2 M4).
Build the harness *before* touching any model — everything later is judged
through it (v2: "M4 comes before models"). Work in `engine._backtest` (and the
helpers it calls) and page `5_Accuracy_FVA.py`; do **not** change `quality.py`,
`ingest_ui.py`, the workspace/canonical contracts, or the other pages.

* **Frozen output shapes** (page 5 already renders these): `RunResult.fva`
  (`step, wmape, bias` — the waterfall at lag 1–2) and `RunResult.fva_by_lag`
  (`lag, method, wmape, bias`, `method ∈ {naive, statistical, translated, plan}`).
  Keep these column names and the method vocabulary.
* **Rolling-origin, model-agnostic.** The backtester takes any forecaster
  callable and scores WMAPE + weighted bias at configurable lags over a holdout
  window (`RunConfig.backtest_weeks`). Score sell-out models against
  `units_unconstrained` (falls back to `units_sold` until S5 lands repair) and
  translated forecasts against shipments.
* **FVA waterfall** naïve → statistical → (ML, later) → translated → vs. the
  **lag-adjusted** demand plan, scored using `plan_version_date` so the plan is
  compared at the vintage that was actually available at each origin. The S3
  leakage gate exists precisely so a leaky vintage can't fake plan FVA.
* **As-of leakage assertions everywhere** — assert every forecast at origin `t`
  reads no data after `t`. Leakage is the #1 way a prototype fakes success.
* **Reproducible from a manifest** (already written per run). Add pytest that a
  known-signal synthetic series produces a stable, positive naïve→statistical
  FVA and that lag scoring is monotonic where expected.
* **Honest S1 caveat still stands:** the statistical tier is a damped-EWMA
  placeholder, so it may not clear the ≥15% FVA bar yet — S5–S7 earn that. S4's
  job is to make the number *trustworthy*, not to make it pass.

## Known stubs (deliberate, not bugs)

* Statistical tier is a damped-EWMA placeholder, not ETS/AutoETS → on demo data
  it does not clear the ≥15% FVA bar. That is the honest verdict; S5–S7 earn it.
* ML toggle is wired through config but does not yet train a model.
* Signal repair censoring detection is a heuristic (low stock + depressed sales),
  not the full inventory-driven + zero-inflation test from v2 M3.
