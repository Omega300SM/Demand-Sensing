"""S2 tests — upload layer hardened to production quality (design §3.2–§3.3).

Covers the five gaps closed this session:
  1. wide-format unpivot -> canonical snapshot round-trip
  2. saved mappings per stream+source (save/get, header signature)
  3. configurable retailer week-calendar
  4. multi-file stacking with grain de-dup (last-write-wins)
  5. sniffing/coercion robustness (mixed dates, numeric-ID codes,
     empty headers, required-but-unmapped block)

The headline acceptance is proved end-to-end for each messy fixture:
read -> (unpivot) -> map -> coerce -> snapshot -> rebuild_canonical, and the
canonical table has the right grain, dtypes, and values.
"""

import os
import sys
import warnings

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensing import Workspace, ingest_ui, demo_data  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


@pytest.fixture
def ws(tmp_path):
    return Workspace(str(tmp_path / "s2.duckdb"))


def _fixture_bytes(name: str) -> bytes:
    # Regenerate on the fly so the test never depends on a stale committed file.
    demo_data.write_messy_fixtures(FIXTURES)
    with open(os.path.join(FIXTURES, f"messy_pos_{name}.csv"), "rb") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Gap 5 — headline acceptance: each messy fixture reaches canonical intact
# --------------------------------------------------------------------------- #

def test_renamed_headers_roundtrip_to_canonical(ws):
    raw = ingest_ui.read_upload("weekly_pos_2025_01.csv", _fixture_bytes("renamed"))
    mapping = ingest_ui.suggest_mapping(raw, "pos")
    assert mapping["item_id"] == "Article Nbr"
    assert mapping["week"] == "Sales Week"
    canonical, warns = ingest_ui.apply_mapping(raw, "pos", mapping)
    assert warns == []
    ws.add_snapshot("pos", canonical, source_name="weekly_pos_2025_01.csv")
    built = ws.rebuild_canonical()
    assert built["pos"] == 4
    canon = ws.read_canonical("pos")
    # correct dtypes
    assert pd.api.types.is_datetime64_any_dtype(canon["week"])
    assert pd.api.types.is_numeric_dtype(canon["units_sold"])
    # mixed date formats (ISO + US) both parsed
    assert canon["week"].notna().all()
    # thousands separators stripped
    assert sorted(canon["units_sold"].tolist()) == [640, 720, 1200, 1340]


def test_wm_week_labels_roundtrip_to_canonical(ws):
    raw = ingest_ui.read_upload("wm_pos.csv", _fixture_bytes("wm_weeks"))
    mapping = ingest_ui.suggest_mapping(raw, "pos")
    canonical, warns = ingest_ui.apply_mapping(raw, "pos", mapping)
    assert warns == []
    ws.add_snapshot("pos", canonical, source_name="wm_pos.csv")
    ws.rebuild_canonical()
    canon = ws.read_canonical("pos").sort_values("week")
    weeks = canon["week"].dt.date.unique()
    # WM Week 2502/2503 with the default calendar land on Gregorian Mondays
    assert len(weeks) == 2
    assert str(weeks[0]) == "2025-01-06"
    assert str(weeks[1]) == "2025-01-13"


def test_wide_format_unpivot_roundtrip_to_canonical(ws):
    raw = ingest_ui.read_upload("wide_pos.csv", _fixture_bytes("wide"))
    assert ingest_ui.is_wide_format(raw)
    id_cols = ingest_ui.suggest_wide_id_cols(raw)
    assert id_cols == ["item_id", "region"]
    long = ingest_ui.unpivot_wide(raw, id_cols=id_cols, value_name="units_sold")
    assert len(long) == 8  # 2 series x 4 weeks
    mapping = ingest_ui.suggest_mapping(long, "pos")
    canonical, warns = ingest_ui.apply_mapping(long, "pos", mapping)
    assert warns == []
    ws.add_snapshot("pos", canonical, source_name="wide_pos.csv")
    built = ws.rebuild_canonical()
    assert built["pos"] == 8
    canon = ws.read_canonical("pos")
    assert pd.api.types.is_datetime64_any_dtype(canon["week"])
    assert canon["week"].dt.date.nunique() == 4
    assert sorted(canon["units_sold"].tolist()) == [640, 705, 715, 720, 1200, 1290, 1310, 1340]


# --------------------------------------------------------------------------- #
# Gap 4 — multi-file stacking with grain de-dup
# --------------------------------------------------------------------------- #

def _pos(weeks, units):
    return pd.DataFrame({
        "item_id": ["A"] * len(weeks), "region": ["East"] * len(weeks),
        "week": pd.to_datetime(weeks), "units_sold": units,
    })


def test_stacking_dedups_on_grain_last_write_wins(ws):
    ws.add_snapshot("pos", _pos(["2025-01-06", "2025-01-13"], [10, 20]),
                    source_name="jan.csv", suffix="a")
    # feb file overlaps 01-13 (corrected to 99) and extends to 01-20
    ws.add_snapshot("pos", _pos(["2025-01-13", "2025-01-20"], [99, 30]),
                    source_name="feb.csv", suffix="b")
    built = ws.rebuild_canonical()
    assert built["pos"] == 3  # 01-06, 01-13, 01-20 (not 4)
    canon = ws.read_canonical("pos").sort_values("week").reset_index(drop=True)
    assert canon.loc[canon["week"] == "2025-01-13", "units_sold"].iloc[0] == 99
    # both files preserved in lineage; nothing overwritten
    assert len(ws.list_snapshots("pos")) == 2


def test_snapshot_coverage_reports_combined_range(ws):
    ws.add_snapshot("pos", _pos(["2025-01-06", "2025-01-13"], [10, 20]),
                    source_name="jan.csv", suffix="a")
    ws.add_snapshot("pos", _pos(["2025-01-20"], [30]),
                    source_name="feb.csv", suffix="b")
    cov = ws.snapshot_coverage("pos")
    assert cov["snapshots"] == 2
    assert str(cov["date_min"]) == "2025-01-06"
    assert str(cov["date_max"]) == "2025-01-20"


def test_demand_plan_keeps_distinct_vintages_when_stacked(ws):
    # same item/region/week but two plan_version_dates must both survive
    def plan(vdate, units):
        return pd.DataFrame({
            "item_id": ["A"], "region": ["East"], "week": pd.to_datetime(["2025-02-03"]),
            "plan_units": [units], "plan_version_date": pd.to_datetime([vdate]),
        })
    ws.add_snapshot("demand_plan", plan("2025-01-01", 100), source_name="v1.csv", suffix="a")
    ws.add_snapshot("demand_plan", plan("2025-01-15", 120), source_name="v2.csv", suffix="b")
    built = ws.rebuild_canonical()
    assert built["demand_plan"] == 2  # vintages not collapsed


# --------------------------------------------------------------------------- #
# Gap 2 — saved mappings
# --------------------------------------------------------------------------- #

def test_source_signature_is_header_based_and_month_stable():
    jan = pd.DataFrame(columns=["Article Nbr", "Dist Region", "WM Week", "POS Qty"])
    feb = pd.DataFrame(columns=["POS Qty", "WM Week", "Dist Region", "Article Nbr"])  # reordered
    other = pd.DataFrame(columns=["sku", "region", "week", "units"])
    assert ingest_ui.source_signature("pos_jan.csv", jan) == \
           ingest_ui.source_signature("pos_feb.csv", feb)          # same headers -> same sig
    assert ingest_ui.source_signature("x.csv", jan) != \
           ingest_ui.source_signature("x.csv", other)              # different headers -> different


def test_save_and_get_mapping_roundtrip(ws):
    sig = "hdr:deadbeef"
    payload = {"mapping": {"item_id": "Article Nbr", "week": "WM Week"},
               "unit_multiplier": 12.0,
               "week_calendar": {"fiscal_year_start_month": 2}}
    ws.save_mapping("pos", sig, payload)
    assert ws.get_mapping("pos", sig) == payload
    assert ws.get_mapping("pos", "missing") is None
    # upsert: last write wins, no duplicate rows
    ws.save_mapping("pos", sig, {"unit_multiplier": 24.0})
    assert ws.get_mapping("pos", sig) == {"unit_multiplier": 24.0}


# --------------------------------------------------------------------------- #
# Gap 3 — configurable week calendar
# --------------------------------------------------------------------------- #

def test_week_calendar_default_matches_s1():
    s = pd.Series(["WM Week 2502", "WM Week 2503"])
    parsed = ingest_ui._coerce_week(s)  # no calendar arg == S1 behaviour
    assert str(parsed.iloc[0].date()) == "2025-01-06"
    assert parsed.iloc[1] > parsed.iloc[0]


def test_week_calendar_fiscal_start_shifts_labels():
    s = pd.Series(["WM Week 2502"])
    default = ingest_ui._coerce_week(s).iloc[0]
    feb = ingest_ui._coerce_week(s, ingest_ui.WeekCalendar(fiscal_year_start_month=2)).iloc[0]
    assert default != feb
    assert feb.month == 2  # anchored to the February fiscal start


def test_week_calendar_serialization_roundtrips():
    cal = ingest_ui.WeekCalendar(fiscal_year_start_month=2, fiscal_year_start_day=1)
    assert ingest_ui.WeekCalendar.from_dict(cal.to_dict()) == cal
    assert ingest_ui.WeekCalendar.from_dict(None) == ingest_ui.WeekCalendar()
    # tolerates extra keys in a persisted payload
    assert ingest_ui.WeekCalendar.from_dict({"fiscal_year_start_month": 2, "junk": 1}) \
        == ingest_ui.WeekCalendar(fiscal_year_start_month=2)


# --------------------------------------------------------------------------- #
# Gap 5 — robustness details
# --------------------------------------------------------------------------- #

def test_numeric_id_codes_stay_strings(ws):
    csv = b"item_id,region,week,units_sold\n001001,East,2025-01-06,10\n001002,East,2025-01-06,20\n"
    raw = ingest_ui.read_upload("x.csv", csv)
    mapping = ingest_ui.suggest_mapping(raw, "pos")
    out, _ = ingest_ui.apply_mapping(raw, "pos", mapping)
    assert out["item_id"].tolist() == ["001001", "001002"]  # leading zeros intact
    assert not pd.api.types.is_numeric_dtype(out["item_id"])


def test_empty_and_duplicate_headers_are_cleaned():
    csv = b"item_id,,week,item_id\nA,junk,2025-01-06,B\n"
    raw = ingest_ui.read_upload("x.csv", csv)
    # no blank header survives; duplicates disambiguated
    assert all(str(c).strip() for c in raw.columns)
    assert len(set(raw.columns)) == len(raw.columns)


def test_missing_required_helper_and_apply_agree():
    df = pd.DataFrame({"region": ["East"], "week": ["2025-01-06"], "units_sold": ["10"]})
    mapping = ingest_ui.suggest_mapping(df, "pos")
    mapping["item_id"] = None
    assert "item_id" in ingest_ui.missing_required("pos", mapping)
    _, warns = ingest_ui.apply_mapping(df, "pos", mapping)
    assert any("item_id" in w for w in warns)


def test_no_pandas_date_inference_warnings_leak():
    # the noisy 'Could not infer format' UserWarning must be muted internally
    df = pd.DataFrame({
        "item_id": ["A", "A"], "region": ["East", "East"],
        "week": ["2025-01-06", "01/13/2025"], "units_sold": ["10", "20"],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mapping = ingest_ui.suggest_mapping(df, "pos")
        ingest_ui.apply_mapping(df, "pos", mapping)
