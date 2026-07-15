"""Tests for the ingestion layer (schema sniffing, mapping, coercion)."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensing import ingest_ui  # noqa: E402
from sensing import demo_data  # noqa: E402


def test_read_csv_bytes():
    csv = b"item_id,region,week,units_sold\nA,East,2025-01-06,10\n"
    df = ingest_ui.read_upload("x.csv", csv)
    assert list(df.columns) == ["item_id", "region", "week", "units_sold"]
    assert len(df) == 1


def test_suggest_mapping_clean_headers():
    df = pd.DataFrame({
        "item_id": ["A"], "region": ["East"],
        "week": ["2025-01-06"], "units_sold": ["10"],
    })
    mapping = ingest_ui.suggest_mapping(df, "pos")
    assert mapping["item_id"] == "item_id"
    assert mapping["week"] == "week"
    assert mapping["units_sold"] == "units_sold"


def test_suggest_mapping_messy_headers():
    df = pd.DataFrame({
        "Article Nbr": ["SKU-1"], "Dist Region": ["East"],
        "WM Week": ["01/06/2025"], "POS Qty": ["1,234"],
    })
    mapping = ingest_ui.suggest_mapping(df, "pos")
    assert mapping["item_id"] == "Article Nbr"
    assert mapping["region"] == "Dist Region"
    assert mapping["week"] == "WM Week"
    assert mapping["units_sold"] == "POS Qty"


def test_apply_mapping_coerces_types_and_thousands():
    df = pd.DataFrame({
        "Article Nbr": ["SKU-1", "SKU-1"],
        "Dist Region": ["East", "East"],
        "WM Week": ["01/06/2025", "01/13/2025"],
        "POS Qty": ["1,234", "2,000"],
    })
    mapping = ingest_ui.suggest_mapping(df, "pos")
    out, warns = ingest_ui.apply_mapping(df, "pos", mapping)
    assert out["units_sold"].tolist() == [1234.0, 2000.0]
    assert pd.api.types.is_datetime64_any_dtype(out["week"])
    assert warns == [] or all("Dropped 0" not in w for w in warns)


def test_unit_multiplier_cases_to_eaches():
    df = pd.DataFrame({
        "item_id": ["A"], "region": ["East"],
        "week": ["2025-01-06"], "units_sold": ["10"],
    })
    mapping = ingest_ui.suggest_mapping(df, "pos")
    out, _ = ingest_ui.apply_mapping(df, "pos", mapping, unit_multiplier=12.0)
    assert out["units_sold"].iloc[0] == 120.0


def test_wm_week_label_parses():
    s = pd.Series(["WM Week 2501", "WM Week 2502"])
    parsed = ingest_ui._coerce_week(s)
    assert parsed.notna().all()
    assert parsed.iloc[1] > parsed.iloc[0]


def test_missing_required_field_warns():
    df = pd.DataFrame({"region": ["East"], "week": ["2025-01-06"], "units_sold": ["10"]})
    mapping = ingest_ui.suggest_mapping(df, "pos")
    mapping["item_id"] = None
    out, warns = ingest_ui.apply_mapping(df, "pos", mapping)
    assert any("item_id" in w for w in warns)


def test_messy_demo_export_maps_and_loads():
    """The bundled messy POS export should map end-to-end (S2 acceptance)."""
    data = demo_data.generate()
    messy = data["pos"].rename(columns={
        "item_id": "Article Nbr", "region": "Dist Region",
        "week": "WM Week", "units_sold": "POS Qty"})
    messy["WM Week"] = pd.to_datetime(messy["WM Week"]).dt.strftime("%m/%d/%Y")
    messy["POS Qty"] = messy["POS Qty"].map(lambda v: f"{v:,}")

    mapping = ingest_ui.suggest_mapping(messy, "pos")
    out, warns = ingest_ui.apply_mapping(messy, "pos", mapping)
    assert len(out) == len(messy)
    assert pd.api.types.is_datetime64_any_dtype(out["week"])
    assert (out["units_sold"] >= 0).all()


def test_wide_format_detection_and_unpivot():
    df = pd.DataFrame({
        "item_id": ["A", "B"],
        "2025-01-06": [10, 20], "2025-01-13": [11, 21],
        "2025-01-20": [12, 22], "2025-01-27": [13, 23],
    })
    assert ingest_ui.is_wide_format(df)
    long = ingest_ui.unpivot_wide(df, id_cols=["item_id"], value_name="units_sold")
    assert len(long) == 8
    assert set(long.columns) == {"item_id", "week", "units_sold"}
