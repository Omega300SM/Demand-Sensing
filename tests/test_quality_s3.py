"""S3 tests — the full M2 quality-gate suite (design §3.3, v1 §3.3).

For EVERY gate we prove two things:
  * it FIRES on its seeded defect (correct status + attached offenders), and
  * it PASSES CLEAN on the engine's untouched demo (no false positive).

Plus: backward-compat of the ``run_qc`` signature, the quarantine-and-flag rule
(checks never mutate their input), and the headline acceptance from the design
(§5, S3): *seeded defects in the demo files all surface*.
"""

import os
import sys
from datetime import date, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensing import Workspace, quality, demo_data  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def defects():
    return demo_data.defects_fixtures()


@pytest.fixture(scope="module")
def clean():
    return demo_data.generate()


@pytest.fixture
def ws(tmp_path):
    return Workspace(str(tmp_path / "s3.duckdb"))


def _check(report, name):
    for c in report.checks:
        if c.name == name:
            return c
    raise AssertionError(f"check {name!r} not present; got {[c.name for c in report.checks]}")


def _clean_crosswalk(clean):
    return clean["pos"][["item_id"]].drop_duplicates()


def _defects_crosswalk(defects):
    return defects["pos"][["item_id"]].drop_duplicates()


# --------------------------------------------------------------------------- #
# 1. Freshness / SLA  — surfaced first
# --------------------------------------------------------------------------- #

def test_freshness_is_the_first_check():
    rep = quality.run_qc("pos", demo_data.generate()["pos"], as_of=date.today())
    assert rep.checks[0].name == "Freshness / SLA"


def test_freshness_fires_on_stale_stream(defects):
    # channel_inventory ends 4 weeks early vs the defects as-of.
    rep = quality.run_qc("channel_inventory", defects["channel_inventory"],
                         pos=defects["pos"], as_of=demo_data.defects_as_of())
    chk = _check(rep, "Freshness / SLA")
    assert chk.status in ("warn", "fail")


def test_freshness_passes_when_current(clean):
    latest = pd.to_datetime(clean["pos"]["week"]).max().date()
    rep = quality.run_qc("pos", clean["pos"], as_of=latest)
    assert _check(rep, "Freshness / SLA").status == "pass"


def test_freshness_hard_fails_when_very_stale(clean):
    latest = pd.to_datetime(clean["pos"]["week"]).max().date()
    rep = quality.run_qc("pos", clean["pos"], as_of=latest + timedelta(weeks=20))
    assert _check(rep, "Freshness / SLA").status == "fail"


# --------------------------------------------------------------------------- #
# 2. Date coverage & gaps
# --------------------------------------------------------------------------- #

def test_date_gap_fires_with_missing_weeks(defects):
    rep = quality.run_qc("pos", defects["pos"], as_of=demo_data.defects_as_of())
    chk = _check(rep, "Date coverage")
    assert chk.status == "warn"
    assert chk.data is not None and len(chk.data) == 1  # exactly one week removed


def test_date_coverage_passes_clean(clean):
    rep = quality.run_qc("pos", clean["pos"], as_of=pd.to_datetime(clean["pos"]["week"]).max().date())
    assert _check(rep, "Date coverage").status == "pass"


# --------------------------------------------------------------------------- #
# 3. Negative values (with offenders attached)
# --------------------------------------------------------------------------- #

def test_negative_fires_and_attaches_offender(defects):
    rep = quality.run_qc("pos", defects["pos"])
    chk = _check(rep, "Negative values")
    assert chk.status == "warn"
    assert chk.data is not None and (chk.data["units_sold"] < 0).any()


def test_negatives_pass_clean(clean):
    rep = quality.run_qc("pos", clean["pos"])
    assert _check(rep, "Negative values").status == "pass"


# --------------------------------------------------------------------------- #
# 4. Impossible / range values  (bad promo flag; leaky plan vintage; magnitude)
# --------------------------------------------------------------------------- #

def test_bad_promo_flag_fires(defects):
    rep = quality.run_qc("promo", defects["promo"])
    chk = _check(rep, "Impossible / range values")
    assert chk.status == "warn"
    assert chk.data["reason"].str.contains("promo_flag").any()


def test_leaky_plan_vintage_fires(defects):
    rep = quality.run_qc("demand_plan", defects["demand_plan"])
    chk = _check(rep, "Impossible / range values")
    assert chk.status == "warn"
    assert chk.data["reason"].str.contains("leakage").any()


def test_range_passes_clean_plan_and_promo(clean):
    assert _check(quality.run_qc("demand_plan", clean["demand_plan"]),
                  "Impossible / range values").status == "pass"
    assert _check(quality.run_qc("promo", clean["promo"]),
                  "Impossible / range values").status == "pass"


def test_magnitude_does_not_false_positive_on_promo_spikes(clean):
    # Clean POS contains legitimate promo spikes (~2.4x) — must NOT trip.
    assert _check(quality.run_qc("pos", clean["pos"]),
                  "Impossible / range values").status == "pass"


# --------------------------------------------------------------------------- #
# 5. Duplicate rows (vintage-aware grain; offenders attached)
# --------------------------------------------------------------------------- #

def test_duplicate_fires_and_attaches(defects):
    rep = quality.run_qc("pos", defects["pos"])
    chk = _check(rep, "Duplicate rows")
    assert chk.status == "warn"
    assert chk.data is not None and len(chk.data) >= 2  # both colliding rows


def test_duplicates_pass_clean(clean):
    assert _check(quality.run_qc("pos", clean["pos"]), "Duplicate rows").status == "pass"


def test_demand_plan_vintages_are_not_flagged_as_duplicates():
    # Two distinct vintages for the same item/region/week must NOT be duplicates.
    wk = date(2025, 1, 6)
    df = pd.DataFrame([
        dict(item_id="X", region="E", week=wk, plan_units=10,
             plan_version_date=date(2024, 12, 1)),
        dict(item_id="X", region="E", week=wk, plan_units=12,
             plan_version_date=date(2024, 12, 8)),
    ])
    assert _check(quality.run_qc("demand_plan", df), "Duplicate rows").status == "pass"


# --------------------------------------------------------------------------- #
# 6. Coverage-shift detection (adds AND drops)
# --------------------------------------------------------------------------- #

def test_coverage_shift_fires_on_adds_and_drops(defects):
    rep = quality.run_qc("pos", defects["pos"])
    chk = _check(rep, "Coverage shift")
    assert chk.status == "warn"
    assert (chk.data["n_added"] > 0).any()    # SKU-C appears mid-history
    assert (chk.data["n_dropped"] > 0).any()  # SKU-A drops mid-history


def test_coverage_shift_passes_clean(clean):
    assert _check(quality.run_qc("pos", clean["pos"]), "Coverage shift").status == "pass"


# --------------------------------------------------------------------------- #
# 7. Crosswalk match
# --------------------------------------------------------------------------- #

def test_unmatched_item_fires(defects):
    rep = quality.run_qc("shipments", defects["shipments"],
                         crosswalk=_defects_crosswalk(defects))
    chk = _check(rep, "Crosswalk match")
    assert chk.status == "warn"
    assert "SKU-Z" in set(chk.data["unmatched_item_id"])


def test_crosswalk_passes_clean(clean):
    rep = quality.run_qc("shipments", clean["shipments"], crosswalk=_clean_crosswalk(clean))
    assert _check(rep, "Crosswalk match").status == "pass"


# --------------------------------------------------------------------------- #
# 8. Phantom inventory (N-consecutive-week run; N configurable)
# --------------------------------------------------------------------------- #

def test_phantom_run_fires(defects):
    rep = quality.run_qc("channel_inventory", defects["channel_inventory"],
                         pos=defects["pos"], phantom_weeks=3)
    chk = _check(rep, "Phantom inventory")
    assert chk.status == "warn"
    row = chk.data.iloc[0]
    assert row["item_id"] == "SKU-B" and row["region"] == "West"
    assert int(row["run_weeks"]) >= 3


def test_phantom_threshold_is_configurable(defects):
    # The seeded run is exactly 3 weeks — with N=4 it must NOT fire.
    rep = quality.run_qc("channel_inventory", defects["channel_inventory"],
                         pos=defects["pos"], phantom_weeks=4)
    assert _check(rep, "Phantom inventory").status == "pass"


def test_phantom_passes_clean(clean):
    rep = quality.run_qc("channel_inventory", clean["channel_inventory"], pos=clean["pos"])
    assert _check(rep, "Phantom inventory").status == "pass"


# --------------------------------------------------------------------------- #
# Contract guarantees
# --------------------------------------------------------------------------- #

def test_run_qc_backward_compatible_positional_signature(clean):
    # Old S1/S2 call style: run_qc(stream, df, crosswalk, pos) with no kwargs.
    rep = quality.run_qc("channel_inventory", clean["channel_inventory"],
                         _clean_crosswalk(clean), clean["pos"])
    assert rep.worst in ("pass", "warn", "fail")


def test_checks_never_mutate_input(defects):
    df = defects["pos"].copy()
    before = df.copy()
    quality.run_qc("pos", df, crosswalk=_defects_crosswalk(defects),
                   pos=defects["pos"], as_of=demo_data.defects_as_of())
    pd.testing.assert_frame_equal(df, before)


def test_status_values_stay_frozen(defects):
    rep = quality.run_qc("pos", defects["pos"], as_of=demo_data.defects_as_of())
    assert all(c.status in ("pass", "warn", "fail") for c in rep.checks)


# --------------------------------------------------------------------------- #
# Headline acceptance (design §5, S3): all seeded defects surface via the app path
# --------------------------------------------------------------------------- #

def test_qc_demo_files_surface_every_defect_class(defects):
    # Design bar (§5, S3): "seeded defects in demo FILES all surface." Run the
    # gate suite over the uploaded files (the planner's view of each stream).
    pos_df = defects["pos"]
    crosswalk = pos_df[["item_id"]].drop_duplicates()
    as_of = demo_data.defects_as_of()

    fired = {}
    for stream, df in defects.items():
        rep = quality.run_qc(stream, df, crosswalk=crosswalk, pos=pos_df, as_of=as_of)
        for c in rep.checks:
            if c.status != "pass":
                fired[c.name] = c.status

    for name in ["Freshness / SLA", "Date coverage", "Negative values",
                 "Impossible / range values", "Duplicate rows", "Coverage shift",
                 "Crosswalk match", "Phantom inventory"]:
        assert name in fired, f"{name} did not surface on the QC demo files"


def test_qc_demo_loads_into_workspace_and_survives_rebuild(ws):
    # The one-click path: land the defects demo + rebuild. Rebuild de-dups the
    # exact duplicate on the grain (last-write-wins), so on canonical every
    # defect EXCEPT the collapsed duplicate still surfaces.
    demo_data.load_qc_demo_into_workspace(ws)
    pos_df = ws.read_canonical("pos")
    crosswalk = pos_df[["item_id"]].drop_duplicates()
    as_of = demo_data.defects_as_of()

    fired = set()
    for stream in ["pos", "channel_inventory", "shipments", "demand_plan", "promo"]:
        if not ws.has_canonical(stream):
            continue
        rep = quality.run_qc(stream, ws.read_canonical(stream),
                             crosswalk=crosswalk, pos=pos_df, as_of=as_of)
        fired |= {c.name for c in rep.checks if c.status != "pass"}

    for name in ["Freshness / SLA", "Date coverage", "Negative values",
                 "Impossible / range values", "Coverage shift",
                 "Crosswalk match", "Phantom inventory"]:
        assert name in fired, f"{name} did not surface post-rebuild"
