"""Tests for the workspace layer (DuckDB snapshots, canonical rebuild, runs)."""

import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensing import Workspace, RunConfig, run_pipeline  # noqa: E402
from sensing import demo_data  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    return Workspace(str(tmp_path / "test.duckdb"))


def _tiny_pos():
    weeks = pd.date_range("2025-01-06", periods=5, freq="7D")
    return pd.DataFrame({
        "item_id": ["A"] * 5,
        "region": ["East"] * 5,
        "week": weeks,
        "units_sold": [10, 12, 0, 14, 11],
    })


def test_add_snapshot_records_lineage(ws):
    sid = ws.add_snapshot("pos", _tiny_pos(), source_name="hand.csv")
    snaps = ws.list_snapshots("pos")
    assert len(snaps) == 1
    row = snaps.iloc[0]
    assert row["snapshot_id"] == sid
    assert row["source_name"] == "hand.csv"
    assert row["row_count"] == 5
    assert str(row["date_min"]).startswith("2025-01-06")


def test_snapshots_are_immutable_and_stacked(ws):
    ws.add_snapshot("pos", _tiny_pos(), source_name="jan.csv", suffix="a")
    ws.add_snapshot("pos", _tiny_pos(), source_name="feb.csv", suffix="b")
    snaps = ws.list_snapshots("pos")
    # both kept; nothing overwritten
    assert len(snaps) == 2
    assert set(snaps["source_name"]) == {"jan.csv", "feb.csv"}


def test_rebuild_canonical_uses_latest(ws):
    ws.add_snapshot("pos", _tiny_pos(), source_name="old.csv", suffix="a")
    newer = _tiny_pos()
    newer["units_sold"] = [99, 99, 99, 99, 99]
    ws.add_snapshot("pos", newer, source_name="new.csv", suffix="b")
    built = ws.rebuild_canonical()
    assert built["pos"] == 5
    canon = ws.read_canonical("pos")
    assert (canon["units_sold"] == 99).all()  # latest wins


def test_missing_stream_degrades_gracefully(ws):
    ws.add_snapshot("pos", _tiny_pos(), source_name="pos.csv")
    built = ws.rebuild_canonical()
    assert "pos" in built
    assert "demand_plan" not in built  # never uploaded
    assert not ws.has_canonical("channel_inventory")
    assert ws.read_canonical("channel_inventory").empty


def test_canonical_status_reports_coverage(ws):
    ws.add_snapshot("pos", _tiny_pos(), source_name="pos.csv")
    ws.rebuild_canonical()
    status = ws.canonical_status()
    assert status["pos"]["loaded"] is True
    assert status["pos"]["rows"] == 5
    assert status["channel_inventory"]["loaded"] is False


def test_run_persists_outputs_and_manifest(ws):
    demo_data.load_demo_into_workspace(ws)
    cfg = RunConfig(as_of=date(2026, 5, 4))
    result = run_pipeline(ws, cfg.as_of, cfg)
    runs = ws.list_runs()
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == result.run_id
    # outputs round-trip out of the workspace
    fc = ws.read_run_output(result.run_id, "forecast")
    assert len(fc) > 0
    assert {"item_id", "region", "week", "p50"}.issubset(fc.columns)


def test_run_requires_pos(ws):
    with pytest.raises(ValueError):
        run_pipeline(ws, date(2026, 5, 4), RunConfig(as_of=date(2026, 5, 4)))
