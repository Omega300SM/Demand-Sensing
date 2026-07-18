# S9 → v2 Handoff: Master Data (the item crosswalk as a dimension) + Upload-Page Honesty

**Status: S9 complete. 141 tests green (129 S1–S8 unchanged + 12 new in
`tests/test_master_data_s9.py`). `sensing/quality.py` has a zero-line diff.**

S9 was **not an engine session.** It closed the last gap between the v3 design
doc and the shipped app, entirely on the upload side. Three defects from the
S8→S9 walkthrough are fixed:

1. **The item crosswalk was never built** (design §3.1 lists it as required).
   It now exists as a **dimension**, not a sixth stream.
2. **The crosswalk QC gate was right for the wrong reason** — page 1 fabricated
   the crosswalk out of POS itself, so gate 7 was tautological on `pos` and
   measured stream-coverage (not master-data match) everywhere else. It now
   measures item match rate against **real** uploaded master data.
3. **The upload page reported success on the empty case.** Accept-state is now
   visible, unaccepted files block-and-warn above Rebuild, and an empty rebuild
   is a warning, not a green box.

Scope was exactly: `config.py` (additive), `workspace.py` (additive),
`demo_data.py` (additive), `app/pages/1_Upload_Data.py` (edits),
`tests/test_master_data_s9.py` (new), two new templates. The engine, the
backtest harness, `ml_model.py`, `ingest_ui.py`, pages 2–5, `RunResult` /
`RunConfig` / `run_pipeline`, `CANONICAL_SCHEMAS` / `STREAM_ORDER`, the
`item_id × region × week` grain, and `rebuild_canonical()` semantics were **not
touched.**

---

## Step 0 corrections — the frozen workspace API list was stale, the code is fine

The S9 spec flagged two calls in page 1 as possibly broken. Both are the doc's
fault, not the code's. **Do not "fix" the code to match the old docs.**

* **`ws.has_canonical(stream)` EXISTS** (`workspace.py`). It returns `bool` by
  querying `information_schema.tables` for `canonical_<stream>`. The page never
  threw. The frozen-API list repeated across the S1–S8 handoffs was simply
  **incomplete** — it omitted several real methods.
* **`add_snapshot`'s `suffix` has a default** (`suffix="a"`). So
  `ws.add_snapshot(stream, df, source_name=…)` with no `suffix` is valid — no
  `TypeError`. The "4-positional, no default" signature in the old docs was
  wrong.

### Corrected & complete `sensing/workspace.py` public API (as of S9)

```
# snapshots (append-only fact history)
add_snapshot(stream, df, source_name, suffix="a") -> snapshot_id
list_snapshots(stream=None) -> DataFrame
latest_snapshot(stream) -> table_name | None

# references / dimensions (current-state master data)   [NEW in S9]
add_reference(name, df, source_name) -> "dim_<name>"    # REPLACE-on-upload
read_reference(name) -> DataFrame | None                # None until uploaded
has_reference(name) -> bool

# canonical tables (the ONLY thing the engine reads)
rebuild_canonical() -> {stream: row_count}              # ignores dim_* tables
has_canonical(stream) -> bool
read_canonical(stream) -> DataFrame
canonical_status() -> {stream: {...}}
snapshot_coverage(stream) -> {snapshots, rows, date_min, date_max}

# runs
save_run(run_id, as_of, manifest, outputs) -> None
list_runs() -> DataFrame
read_run_output(run_id, name) -> DataFrame

# saved column mappings
save_mapping(stream, signature, payload) -> None
get_mapping(stream, signature) -> dict | None
```

New metadata table: `references_log(name, table_name, source_name, row_count,
uploaded_at)` — one row per reference, upserted on each replace.

---

## The design decision that must survive graduation: a crosswalk is a DIMENSION

**It is not in `STREAM_ORDER` / `CANONICAL_SCHEMAS`, and it must not become so.**
`config.py`'s field-`role` vocabulary is `id | date | numeric`, and **every
canonical stream carries a required `week` (date role)**. That is load-bearing:

* `workspace._grain_keys` derives the de-dup grain from id+date roles;
* `snapshot_coverage` reports `date_min` / `date_max`;
* QC gates 1, 2 and 6 (freshness, date coverage, coverage shift) all read `week`.

A crosswalk has **no week**. If it were stream six, page 1 would call
`run_qc("item_crosswalk", df, …)` on a dateless frame — freshness misfires,
coverage is nonsense, the card shows an empty date range. Keeping it out of the
stream contracts is what keeps the frozen contracts frozen.

So it lives in a **parallel** structure:

* `config.REFERENCE_SCHEMAS` / `config.REFERENCE_ORDER` — same `fields`/`role`
  shape as a stream (so `suggest_mapping` / `apply_mapping` / `missing_required`
  work unchanged), **with no `date` role anywhere**. Guarded by
  `test_reference_schemas_have_no_date_role`.
* Workspace `dim_<name>` tables via `add_reference` — **replace-on-upload, not
  stacked.** A crosswalk is current-state master data, not an append-only fact
  history; nobody wants last-write-wins grain de-dup on a dimension.
* `rebuild_canonical()` **never touches `dim_*` tables** (guarded by
  `test_rebuild_canonical_ignores_reference_tables`), and the **engine never
  reads them** (guarded by `test_engine_never_reads_dim_tables`:
  `run_pipeline` is green with no crosswalk uploaded). Only `quality.py`
  consumes the crosswalk, via the `crosswalk=` parameter that has existed since
  S2.

### Scope boundary — measure, don't remap

**The crosswalk measures match rate only. It never remaps `item_id` at rebuild.**
Producing the unmatched-items cleanup list (design §3.3) is the whole job in the
POC; the planner fixes their export. Real harmonization — predecessor/successor
chaining, pack-size conversion, item transitions — is **v2's `harmonize.py` /
M1**, which v3 deliberately traded away for the upload mapper. Silently remapping
would also break the flag-don't-impute rule S3 made binding. `units_per_case`
and `successor_item_id` are carried in `REFERENCE_SCHEMAS` for the v2 graduation
and are **unused in S9** — do not wire them into anything.

### The discriminating property the old code could never produce

With a **real** crosswalk (master data, not a POS-derived stand-in), gate 7
**passes on `pos`** and **fires on `shipments`**, and — the regression test that
matters — **catches an item present in both streams but absent from master
data** (`test_crosswalk_gate_catches_item_present_in_both_streams`: `SKU-9999`
seeded into both, absent from the crosswalk → fires on both). The old
POS-derived version fired on **neither**: seeding the item into POS put it *into*
the derived crosswalk, making it "known" on shipments and tautologically matched
on pos. The defect was structurally invisible.

`demo_data.defects_crosswalk()` is the demo's master list: every base SKU
(`SKU-A/B/C`) and **deliberately not** the unmatched `SKU-Z` seeded into
shipments — so on 🧪 Load QC demo, gate 7 fires on the shipments card **for a
real reason**, not by accident. `load_qc_demo_into_workspace` lands it as a
`dim_item_crosswalk` table before the rebuild.

**QC-demo note (pre-existing, not an S9 regression):** the page QCs the
*canonical* (post-rebuild) tables. `rebuild_canonical()` collapses the seeded
duplicate row on the grain (last-write-wins, frozen semantics), so the
**Duplicate rows** gate fires on the pre-rebuild *file* view
(`test_qc_demo_files_surface_every_defect_class`) but not post-rebuild — exactly
as the S3 test `test_qc_demo_loads_into_workspace_and_survives_rebuild` already
asserts (it lists seven gates, omitting Duplicate rows). The other seven fire on
canonical, with **no false freshness reds** (only `channel_inventory` trips it,
because the QC demo stashes `defects_as_of()` in session state instead of
defaulting freshness to `date.today()`).

---

## The standing rule this session establishes: **empty is not success**

> A gate with no data to check, a rebuild with nothing to rebuild, and a run
> with no inputs must all say so **in warning colour.** The derived-crosswalk
> gate and the green "nothing to build" box were the same bug wearing different
> hats: the app reported success on the empty case twice, and both times it cost
> a user real time.

Applied in page 1:

* The dead `st.success` before `st.rerun()` is gone — the accept confirmation is
  stashed in `st.session_state["upload_toast"]` and rendered as a `st.toast`
  after the rerun, so it actually shows.
* Accepted files render **"✅ Already snapshotted"** (matched by
  `source_signature`) instead of a live Accept button that looks identical to an
  un-accepted one.
* Files that are mapped-but-unaccepted are counted and **block-and-warn above
  Rebuild**: *"N file(s) are mapped but not accepted — click Accept on each
  before rebuilding."* This one line was the whole original bug.
* An empty `rebuild_canonical()` result is a **`st.warning`** that says why
  ("no snapshots found — accept your mapped files first"), never a green box.
* `_status_line` is promoted from the caption stack into the card header.

Apply the rule to pages 2–5 if a cheap opportunity presents itself; do not
refactor them.

### Other page-1 fixes (all small, all real)

* **Freshness default no longer drowns the QC demo in false reds.** The QC-demo
  button stashes `demo_data.defects_as_of()` in session state; the page exposes
  a `st.date_input("Evaluate freshness as of", …)` override defaulting to that
  (or `date.today()` normally).
* **Unit multiplier hidden where it is meaningless.** The eaches↔cases selector
  no longer renders for streams/references with no unit-bearing quantity
  (`promo` and both references — set `_NO_UNIT`). Note: `apply_mapping` only
  scales `units_sold/units_shipped/on_hand_units/in_transit_units`, so it never
  actually corrupted `promo_flag`; the selector was misleading, and now it's
  gone.
* **Lossless `WeekCalendar` round-trip.** The picker starts from
  `WeekCalendar.from_dict(saved_cal)` (which preserves `label_format` /
  `century`) and overlays only the two UI picks, instead of reconstructing a
  bare `WeekCalendar(...)` that dropped the saved fields.
* **Unused `import pandas as pd` removed** (the page no longer derives a frame),
  and the template guard regenerates when **any** expected template is missing,
  so a pre-S9 `templates/` dir picks up the two new reference templates.

---

## S9 → v2 graduation path

After S9 the v3 POC matches its design doc. Graduation to the v2 lake-connected
build: swap `ingest_ui.py` for lake pipelines behind the same canonical
contracts, and **promote `dim_item_crosswalk`** from an uploaded current-state
dimension to v2's versioned crosswalk with predecessor/successor chaining and
pack-size conversion (`harmonize.py` / M1). That is the point at which the
measure-only boundary above is deliberately lifted — with tests — and
`units_per_case` / `successor_item_id` finally get wired in. The engine,
backtest history, and calibrated retailer parameters all carry forward
unchanged.

---

## Working-method rules (carried from v2/v3, still binding)

One module per session · tests are the memory between sessions · as-of leakage
assertions everywhere · run manifests on every run · the engine reads only the
workspace (`canonical_*`), never file paths and never `dim_*` · QC reads
canonical and never mutates (flag, don't impute) · **empty is not success**
(new in S9).
