"""Synthetic demo dataset generator.

Produces a self-consistent set of the five upload streams for
**3 SKUs x 2 regions x 78 weeks**, with:

* annual seasonality + gentle trend + noise on base demand,
* **two promotions** (uplift + a post-promo pantry-loading dip),
* **one stockout** (censored, depressed sell-out on one SKU-region),
* channel inventory that builds ahead of promos and drains during the stockout,
* shipments (sell-in) as a lagged, order-batched function of sell-out,
* a versioned demand plan that is deliberately a little wrong (so sensing has
  something to beat).

Deterministic given a seed so tests and demos are reproducible. The generator
both returns DataFrames (for one-click loading into a workspace) and can write
CSV templates + a bundled demo dataset to ``templates/``.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

SKUS = ["SKU-1001", "SKU-1002", "SKU-1003"]
REGIONS = ["East", "West"]
N_WEEKS = 78
START_MONDAY = date(2025, 1, 6)  # a Monday

# Promo windows (week indices, 0-based) and the SKU-region that stocks out.
PROMO_WINDOWS = [(18, 21), (54, 57)]      # two ~4-week promos
STOCKOUT = {"sku": "SKU-1002", "region": "East", "start": 38, "end": 42}

# Per-series base level and seasonal amplitude.
_BASE = {
    ("SKU-1001", "East"): 1200, ("SKU-1001", "West"): 900,
    ("SKU-1002", "East"): 640,  ("SKU-1002", "West"): 720,
    ("SKU-1003", "East"): 300,  ("SKU-1003", "West"): 260,
}


def _weeks() -> list[date]:
    return [START_MONDAY + timedelta(weeks=i) for i in range(N_WEEKS)]


def _promo_flag(week_idx: int) -> int:
    return int(any(a <= week_idx <= b for a, b in PROMO_WINDOWS))


def generate(seed: int = 7) -> dict[str, pd.DataFrame]:
    """Return a dict of stream -> DataFrame in canonical column names."""
    rng = np.random.default_rng(seed)
    weeks = _weeks()

    pos_rows, inv_rows, ship_rows, plan_rows, promo_rows = [], [], [], [], []

    for sku in SKUS:
        for region in REGIONS:
            base = _BASE[(sku, region)]
            # unconstrained "true" demand series
            true_demand = np.zeros(N_WEEKS)
            for i in range(N_WEEKS):
                seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * (i % 52) / 52.0)
                trend = 1.0 + 0.0018 * i
                noise = rng.normal(1.0, 0.06)
                level = base * seasonal * trend * noise

                if _promo_flag(i):
                    level *= rng.uniform(1.8, 2.4)  # promo uplift
                # post-promo dip
                for _, end in PROMO_WINDOWS:
                    if end < i <= end + 2:
                        level *= 0.80
                true_demand[i] = max(0.0, level)

            # observed (censored) sell-out: stockout depresses one series
            observed = true_demand.copy()
            is_stockout = np.zeros(N_WEEKS, dtype=bool)
            if sku == STOCKOUT["sku"] and region == STOCKOUT["region"]:
                for i in range(STOCKOUT["start"], STOCKOUT["end"] + 1):
                    observed[i] = true_demand[i] * rng.uniform(0.05, 0.20)
                    is_stockout[i] = True

            # channel inventory: target cover ~4 weeks, builds pre-promo, drains in stockout
            on_hand = np.zeros(N_WEEKS)
            in_transit = np.zeros(N_WEEKS)
            cover = 4.0
            oh = base * cover
            for i in range(N_WEEKS):
                fwd = true_demand[i]
                # pre-promo build
                if any(a - 2 <= i < a for a, _ in PROMO_WINDOWS):
                    oh += 0.6 * fwd
                oh = max(0.0, oh - observed[i] + 0.9 * fwd)  # sell-out draws down, replen adds
                if is_stockout[i]:
                    oh = max(0.0, oh * 0.15)  # phantom / genuine depletion
                on_hand[i] = round(oh)
                in_transit[i] = round(0.4 * fwd) if not is_stockout[i] else 0

            # shipments (sell-in): lagged + smoothed reorder to restore cover
            ship = np.zeros(N_WEEKS)
            for i in range(N_WEEKS):
                lag = max(0, i - 1)
                target_pos = cover * true_demand[i]
                proj_pos = on_hand[max(0, i - 1)] - observed[i]
                order = observed[lag] + max(0.0, target_pos - proj_pos) * 0.5
                ship[i] = max(0.0, round(order * rng.normal(1.0, 0.05)))

            # demand plan: a smoothed, slightly-lagged, slightly-biased view.
            # Versioned "as of" 6 weeks before each week (a realistic plan age).
            smooth = pd.Series(true_demand).rolling(6, min_periods=1).mean().values
            for i in range(N_WEEKS):
                plan_units = smooth[i] * rng.uniform(0.90, 1.02)
                # plan misses promos (planned late / underscoped)
                if _promo_flag(i):
                    plan_units = smooth[i] * 1.15
                plan_version = weeks[i] - timedelta(weeks=6)
                plan_rows.append(
                    dict(item_id=sku, region=region, week=weeks[i],
                         plan_units=round(plan_units),
                         plan_version_date=plan_version)
                )

            for i in range(N_WEEKS):
                pos_rows.append(dict(item_id=sku, region=region, week=weeks[i],
                                     units_sold=int(round(observed[i]))))
                inv_rows.append(dict(item_id=sku, region=region, week=weeks[i],
                                     on_hand_units=int(on_hand[i]),
                                     in_transit_units=int(in_transit[i])))
                ship_rows.append(dict(item_id=sku, region=region, week=weeks[i],
                                      units_shipped=int(ship[i])))
                promo_rows.append(dict(item_id=sku, region=region, week=weeks[i],
                                       promo_flag=_promo_flag(i)))

    return {
        "pos": pd.DataFrame(pos_rows),
        "channel_inventory": pd.DataFrame(inv_rows),
        "shipments": pd.DataFrame(ship_rows),
        "demand_plan": pd.DataFrame(plan_rows),
        "promo": pd.DataFrame(promo_rows),
    }


# --------------------------------------------------------------------------- #
# Template + bundled-dataset writers
# --------------------------------------------------------------------------- #

def write_templates(templates_dir: str) -> dict[str, str]:
    """Write empty CSV templates (headers only) and a full demo dataset.

    Returns a dict describing what was written."""
    os.makedirs(templates_dir, exist_ok=True)
    data = generate()

    written: dict[str, str] = {}
    # Header-only templates the user can download and fill.
    for stream, df in data.items():
        tmpl_path = os.path.join(templates_dir, f"template_{stream}.csv")
        df.head(0).to_csv(tmpl_path, index=False)
        written[f"template_{stream}"] = tmpl_path
    # Full demo dataset — messy on purpose for the POS file to exercise the mapper.
    demo_dir = os.path.join(templates_dir, "demo_dataset")
    os.makedirs(demo_dir, exist_ok=True)
    for stream, df in data.items():
        out = df.copy()
        path = os.path.join(demo_dir, f"demo_{stream}.csv")
        out.to_csv(path, index=False)
        written[f"demo_{stream}"] = path

    # A deliberately messy POS export to demo the column mapper (renamed cols,
    # WM-week labels, thousands separators, wide-ish). Kept separate.
    messy = data["pos"].copy()
    messy = messy.rename(columns={
        "item_id": "Article Nbr",
        "region": "Dist Region",
        "week": "WM Week",
        "units_sold": "POS Qty",
    })
    messy["WM Week"] = pd.to_datetime(messy["WM Week"]).dt.strftime("%m/%d/%Y")
    messy["POS Qty"] = messy["POS Qty"].map(lambda v: f"{v:,}")
    messy_path = os.path.join(demo_dir, "messy_pos_export.csv")
    messy.to_csv(messy_path, index=False)
    written["messy_pos_export"] = messy_path

    # QC-defects demo streams (one instance of each defect class) for S3.
    for stream, path in write_defects_fixtures(demo_dir).items():
        written[f"defects_{stream}"] = path

    # Reference (dimension) templates — header-only, so a planner can fill and
    # upload master data under the Master-data cards (S9).
    reference_headers = {
        "item_crosswalk": ["source_item_id", "item_id",
                           "units_per_case", "successor_item_id"],
        "location_map": ["source_location_id", "region"],
    }
    for name, headers in reference_headers.items():
        tmpl_path = os.path.join(templates_dir, f"template_{name}.csv")
        pd.DataFrame(columns=headers).to_csv(tmpl_path, index=False)
        written[f"template_{name}"] = tmpl_path

    return written


def messy_fixtures() -> dict[str, pd.DataFrame]:
    """Three deliberately-messy POS exports exercising the S2 upload landmines.

    Small and hand-checkable (so tests can assert exact values), each targeting
    one failure mode the mapper must survive:

    * ``renamed``  renamed headers + thousands separators + mixed date formats
      (ISO and US) within one column.
    * ``wm_weeks`` retailer week labels ('WM Week 2501').
    * ``wide``     weeks-as-columns wide format needing an unpivot.

    All three describe the *same* 2-SKU x 2-week POS slice, so a test can cross
    check that every path lands the same canonical values.
    """
    renamed = pd.DataFrame({
        "Article Nbr": ["SKU-1001", "SKU-1001", "SKU-1002", "SKU-1002"],
        "Dist Region": ["East", "East", "West", "West"],
        # mixed formats in one column: ISO then US
        "Sales Week": ["2025-01-06", "01/13/2025", "2025-01-06", "2025-01-13"],
        "POS Qty": ["1,200", "1,340", "640", "720"],
    })

    wm_weeks = pd.DataFrame({
        "Item": ["SKU-1001", "SKU-1001", "SKU-1002", "SKU-1002"],
        "Region": ["East", "East", "West", "West"],
        "Fiscal Week": ["WM Week 2502", "WM Week 2503", "WM Week 2502", "WM Week 2503"],
        "Units": ["1200", "1340", "640", "720"],
    })

    wide = pd.DataFrame({
        "item_id": ["SKU-1001", "SKU-1002"],
        "region": ["East", "West"],
        "2025-01-06": [1200, 640],
        "2025-01-13": [1340, 720],
        "2025-01-20": [1290, 705],
        "2025-01-27": [1310, 715],
    })

    return {"renamed": renamed, "wm_weeks": wm_weeks, "wide": wide}


def write_messy_fixtures(out_dir: str) -> dict[str, str]:
    """Write the messy fixtures to CSV; reused by both the app and the tests."""
    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, str] = {}
    for name, df in messy_fixtures().items():
        path = os.path.join(out_dir, f"messy_pos_{name}.csv")
        df.to_csv(path, index=False)
        written[name] = path
    return written


def load_demo_into_workspace(workspace) -> dict[str, int]:
    """One-click: generate demo data, land it as snapshots, rebuild canonical."""
    data = generate()
    for stream, df in data.items():
        workspace.add_snapshot(stream, df, source_name="bundled_demo", suffix="demo")
    return workspace.rebuild_canonical()


# --------------------------------------------------------------------------- #
# S3 — DEFECTS demo: one instance of each QC defect class, for the gate suite
# --------------------------------------------------------------------------- #
#
# A compact, hand-checkable 5-stream dataset (2 base SKUs x 2 regions x 12
# weeks) into which we inject EXACTLY ONE instance of each defect the M2 gates
# must catch. Kept entirely separate from ``generate()`` so the engine's clean
# demo — and the S1/S2 tests that depend on it — never change.
#
# Seeded defects (check that should fire):
#   * date gap ............... a whole week removed from POS      -> Date coverage
#   * negative ............... one POS units_sold < 0            -> Negative values
#   * duplicate .............. one POS grain row repeated        -> Duplicate rows
#   * unmatched item ......... a shipments SKU absent from POS   -> Crosswalk match
#   * coverage shift ......... 2 series appear mid-history, 2    -> Coverage shift
#                              others drop out mid-history
#   * phantom-inventory run .. on_hand>0 & zero sell-out, 3 wks  -> Phantom inventory
#   * stale feed ............. channel_inventory ends 4 wks early-> Freshness / SLA
#   * leaky plan vintage ..... plan_version_date >= target week  -> Impossible/range
#   * bad promo flag ......... promo_flag = 5                    -> Impossible/range
# --------------------------------------------------------------------------- #

DEFECTS_START_MONDAY = date(2025, 1, 6)   # a Monday
DEFECTS_N_WEEKS = 12
_DEFECTS_BASE = {
    ("SKU-A", "East"): 100, ("SKU-A", "West"): 90,
    ("SKU-B", "East"): 80,  ("SKU-B", "West"): 70,
}
_DEFECTS_ADD = {("SKU-C", "East"): 60, ("SKU-C", "West"): 55}  # appear mid-history


def _defects_weeks() -> list[date]:
    return [DEFECTS_START_MONDAY + timedelta(weeks=i) for i in range(DEFECTS_N_WEEKS)]


def defects_fixtures() -> dict[str, pd.DataFrame]:
    """Return the five streams of the QC-defects demo (canonical columns).

    Deterministic and small enough to hand-check. See the module comment for
    the exact defect injected for each gate. ``as_of`` for the freshness gate is
    the last base week (weeks[-1]); the only stream that trips it is
    channel_inventory, which deliberately ends 4 weeks early.
    """
    weeks = _defects_weeks()
    pos, inv, ship, plan, promo = [], [], [], [], []

    # ----- base series: present every week -------------------------------- #
    for (sku, region), base in _DEFECTS_BASE.items():
        # coverage-shift DROP: SKU-A disappears after week 6
        drop_after = 6 if sku == "SKU-A" else None
        for i, wk in enumerate(weeks):
            if drop_after is not None and i > drop_after:
                continue
            units = base + (i % 3)  # gentle, positive, non-flagging variation
            pos.append(dict(item_id=sku, region=region, week=wk, units_sold=units))
            # channel inventory ends 4 weeks early -> stale feed (freshness)
            if i <= DEFECTS_N_WEEKS - 1 - 4:
                inv.append(dict(item_id=sku, region=region, week=wk,
                                on_hand_units=base * 3, in_transit_units=base))
            ship.append(dict(item_id=sku, region=region, week=wk,
                             units_shipped=units))
            # plan: valid vintage 4 weeks before the target week...
            pv = wk - timedelta(weeks=4)
            # ...except SKU-A/East, whose vintage LEAKS (dated on the target week)
            if sku == "SKU-A" and region == "East":
                pv = wk
            plan.append(dict(item_id=sku, region=region, week=wk,
                             plan_units=units, plan_version_date=pv))
            promo.append(dict(item_id=sku, region=region, week=wk, promo_flag=0))

    # ----- coverage-shift ADD: SKU-C appears from week 3 onward ------------ #
    for (sku, region), base in _DEFECTS_ADD.items():
        for i, wk in enumerate(weeks):
            if i < 3:
                continue
            units = base + (i % 3)
            pos.append(dict(item_id=sku, region=region, week=wk, units_sold=units))
            if i <= DEFECTS_N_WEEKS - 1 - 4:
                inv.append(dict(item_id=sku, region=region, week=wk,
                                on_hand_units=base * 3, in_transit_units=base))
            ship.append(dict(item_id=sku, region=region, week=wk,
                             units_shipped=units))
            plan.append(dict(item_id=sku, region=region, week=wk,
                             plan_units=units, plan_version_date=wk - timedelta(weeks=4)))
            promo.append(dict(item_id=sku, region=region, week=wk, promo_flag=0))

    pos = pd.DataFrame(pos)
    inv = pd.DataFrame(inv)
    ship = pd.DataFrame(ship)
    plan = pd.DataFrame(plan)
    promo = pd.DataFrame(promo)

    # ----- PHANTOM run: SKU-B/West on_hand>0 but zero sell-out, weeks 2-4 -- #
    for i in (2, 3, 4):
        wk = weeks[i]
        m = (pos["item_id"] == "SKU-B") & (pos["region"] == "West") & (pos["week"] == wk)
        pos.loc[m, "units_sold"] = 0
        # its inventory rows already have on_hand>0 (weeks <= 7), leave them.

    # ----- NEGATIVE: one POS row < 0 (SKU-A/East, week 2) ----------------- #
    m = (pos["item_id"] == "SKU-A") & (pos["region"] == "East") & (pos["week"] == weeks[2])
    pos.loc[m, "units_sold"] = -25

    # ----- DATE GAP: remove week 9 from POS entirely ---------------------- #
    pos = pos[pos["week"] != weeks[9]].reset_index(drop=True)

    # ----- DUPLICATE: repeat one POS grain row (SKU-B/East, week 4) -------- #
    dup = pos[(pos["item_id"] == "SKU-B") & (pos["region"] == "East") &
              (pos["week"] == weeks[4])]
    pos = pd.concat([pos, dup], ignore_index=True)

    # ----- UNMATCHED ITEM: a shipments SKU with no POS history ------------ #
    for i in (0, 1):
        ship = pd.concat([ship, pd.DataFrame([dict(
            item_id="SKU-Z", region="East", week=weeks[i], units_shipped=42)])],
            ignore_index=True)

    # ----- BAD PROMO FLAG: one promo_flag out of {0,1} -------------------- #
    m = (promo["item_id"] == "SKU-B") & (promo["region"] == "East") & (promo["week"] == weeks[5])
    promo.loc[m, "promo_flag"] = 5

    return {"pos": pos, "channel_inventory": inv, "shipments": ship,
            "demand_plan": plan, "promo": promo}


def defects_crosswalk() -> pd.DataFrame:
    """Master item list for the defects demo: every base SKU, and deliberately
    NOT the unmatched item (``SKU-Z``) seeded into shipments.

    A crosswalk is dateless, unit-less master data (``source_item_id`` ->
    ``item_id``); here the retailer item id and the internal SKU coincide,
    because the POC's crosswalk *measures* match rate and never remaps. Omitting
    SKU-Z is what makes gate 7 fire for a real reason on the shipments card —
    an item present in the data that master data has never heard of.
    """
    skus = sorted({sku for sku, _ in _DEFECTS_BASE} |
                  {sku for sku, _ in _DEFECTS_ADD})    # SKU-A, SKU-B, SKU-C
    return pd.DataFrame({"source_item_id": skus, "item_id": skus})


def defects_location_map() -> pd.DataFrame:
    """Location -> region map for the defects demo (optional reference).

    The demo already uses canonical region names, so the map is identity; it
    exists so the second reference slot has a concrete, downloadable template."""
    regions = sorted({region for _, region in _DEFECTS_BASE})
    return pd.DataFrame({"source_location_id": regions, "region": regions})


def defects_as_of() -> date:
    """The reference date the freshness gate should be evaluated against for the
    QC-defects demo (the last base week). Only channel_inventory trips it."""
    return _defects_weeks()[-1]


def write_defects_fixtures(out_dir: str) -> dict[str, str]:
    """Write the QC-defects demo streams to CSV; reused by the app and tests."""
    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, str] = {}
    for stream, df in defects_fixtures().items():
        path = os.path.join(out_dir, f"defects_{stream}.csv")
        df.to_csv(path, index=False)
        written[stream] = path
    return written


def load_qc_demo_into_workspace(workspace) -> dict[str, int]:
    """One-click: land the QC-defects demo as snapshots and rebuild canonical.

    Mirrors ``load_demo_into_workspace`` but uses the defects set so the upload
    page can show the full M2 gate suite firing. Does not touch the clean demo.
    """
    data = defects_fixtures()
    for stream, df in data.items():
        workspace.add_snapshot(stream, df, source_name="qc_defects_demo", suffix="qcdemo")
    # Land the item crosswalk as a dimension so gate 7 measures against REAL
    # master data (which omits SKU-Z) rather than a POS-derived stand-in.
    workspace.add_reference("item_crosswalk", defects_crosswalk(),
                            source_name="qc_demo")
    return workspace.rebuild_canonical()


# --------------------------------------------------------------------------- #
# S7 — POOR-INVENTORY fixture: proves the transfer-function fallback path
# --------------------------------------------------------------------------- #
#
# A NEW, optional generator (``generate()`` is deliberately left untouched so the
# S1–S6 suite stays green). It emits one series whose channel inventory is
# UNUSABLE — a phantom-flat perpetual on-hand with no in-transit — so the
# translation engine's per-series selector falls back to the distributed-lag
# transfer function (design §6 / v2 M7). Shipments are generated as a genuine
# distributed lag of sell-out (+ small noise) so the transfer function has real
# structure to fit and beats a shipments-history-only carry-forward.
# --------------------------------------------------------------------------- #

POOR_INV = {"sku": "SKU-9001", "region": "East", "base": 500, "n_weeks": 78}


def poor_inventory_streams(seed: int = 11) -> dict[str, pd.DataFrame]:
    """Return canonical pos / channel_inventory / shipments frames for a single
    series with PHANTOM-FLAT (unusable) inventory. Deterministic given seed.

    ``on_hand_units`` is a constant (no variation, no draw-down) -> the inventory
    signal fails the usability check -> the translated forecaster selects the
    transfer function. Shipments = 0.6*sell_out(t) + 0.4*sell_out(t-1) + noise."""
    rng = np.random.default_rng(seed)
    sku, region, base, n = (POOR_INV["sku"], POOR_INV["region"],
                            POOR_INV["base"], POOR_INV["n_weeks"])
    weeks = [START_MONDAY + timedelta(weeks=i) for i in range(n)]
    so = np.zeros(n)
    for i in range(n):
        seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (i % 52) / 52.0)
        so[i] = max(0.0, base * seasonal * rng.normal(1.0, 0.08))
    # shipments are a genuine distributed lag of sell-out with real noise, so a
    # POS-driven transfer function has structure to exploit that a shipments-only
    # carry-forward cannot (the lag term means last week's ship != this week's).
    ship = np.zeros(n)
    for i in range(n):
        p1 = so[i - 1] if i > 0 else so[i]
        p2 = so[i - 2] if i > 1 else p1
        ship[i] = max(0.0, (0.5 * so[i] + 0.3 * p1 + 0.2 * p2) * rng.normal(1.0, 0.10))

    pos_rows, inv_rows, ship_rows = [], [], []
    for i in range(n):
        pos_rows.append(dict(item_id=sku, region=region, week=weeks[i],
                             units_sold=int(round(so[i]))))
        # phantom-flat on-hand: constant, never draws down -> unusable inventory
        inv_rows.append(dict(item_id=sku, region=region, week=weeks[i],
                             on_hand_units=int(base * 3), in_transit_units=0))
        ship_rows.append(dict(item_id=sku, region=region, week=weeks[i],
                              units_shipped=int(round(ship[i]))))
    return {
        "pos": pd.DataFrame(pos_rows),
        "channel_inventory": pd.DataFrame(inv_rows),
        "shipments": pd.DataFrame(ship_rows),
    }


if __name__ == "__main__":  # manual generation
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    w = write_templates(os.path.join(here, "templates"))
    print(f"Wrote {len(w)} files to templates/")
