"""S9 tests — the item crosswalk as a *dimension*, and the crosswalk QC gate.

S9 closes the last gap between the v3 design doc and the shipped app, all on the
upload side. The engine, harness and ``quality.py`` are untouched. These tests
pin down three things the S1–S8 suite could not:

  * **The discriminating property of gate 7.** With a *real* crosswalk (master
    data, not a POS-derived stand-in) the crosswalk-match gate PASSES on ``pos``
    and FIRES on ``shipments`` — and, crucially, catches an item present in
    *both* streams but absent from master data. The old page fabricated the
    crosswalk out of POS itself, so it forced pass-on-pos structurally and could
    never see the both-streams defect. That is THE regression this session fixes.
  * **A crosswalk is a dimension, not a sixth stream.** It has no ``week`` role;
    it lands in the workspace as a ``dim_*`` table via replace-on-upload
    (not stacked); ``rebuild_canonical`` never touches it and the engine never
    reads it.
  * **``quality.py`` is frozen.** Its ``run_qc`` signature is exactly what S3
    left — S9 supplies a real crosswalk frame, it does not change the gate.
"""

import inspect
import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensing import Workspace, RunConfig, run_pipeline  # noqa: E402
from sensing import quality, demo_data, config  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def ws(tmp_path):
    return Workspace(str(tmp_path / "s9.duckdb"))


@pytest.fixture(scope="module")
def defects():
    return demo_data.defects_fixtures()


def _check(report, name):
    for c in report.checks:
        if c.name == name:
            return c
    raise AssertionError(
        f"check {name!r} not present; got {[c.name for c in report.checks]}")


def _seed_item(df: pd.DataFrame, item_id: str) -> pd.DataFrame:
    """Append one row for ``item_id`` mirroring an existing row's shape."""
    row = df.iloc[0].to_dict()
    row["item_id"] = item_id
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


# --------------------------------------------------------------------------- #
# Gate 7 — the crosswalk match, measured against REAL master data
# --------------------------------------------------------------------------- #

def test_crosswalk_gate_passes_on_pos_and_fires_on_shipments(defects):
    """The property the POS-derived stand-in could never produce: a real
    crosswalk PASSES on pos (all POS items are known SKUs) and FIRES on
    shipments (which carries the unmatched SKU-Z the master list omits)."""
    xwalk = demo_data.defects_crosswalk()

    pos_rep = quality.run_qc("pos", defects["pos"], crosswalk=xwalk)
    ship_rep = quality.run_qc("shipments", defects["shipments"], crosswalk=xwalk)

    assert _check(pos_rep, "Crosswalk match").status == "pass"
    ship_chk = _check(ship_rep, "Crosswalk match")
    assert ship_chk.status == "warn"
    assert "SKU-Z" in set(ship_chk.data["unmatched_item_id"])


def test_crosswalk_gate_skips_cleanly_when_no_crosswalk(defects):
    """No crosswalk uploaded -> the gate is skipped honestly: status 'pass',
    detail says skipped. (The page shows a banner; the gate stays green-but-
    honest rather than fabricating a stand-in.)"""
    rep = quality.run_qc("shipments", defects["shipments"], crosswalk=None)
    chk = _check(rep, "Crosswalk match")
    assert chk.status == "pass"
    assert "skipped" in chk.detail.lower()


def test_crosswalk_gate_catches_item_present_in_both_streams(defects):
    """THE regression test.

    Seed SKU-9999 into BOTH pos and shipments, absent from the crosswalk. A real
    crosswalk fires gate 7 on BOTH cards — that unmapped-in-both item is exactly
    the business defect the gate exists to surface (design §3.3).

    Under the old page this fired on NEITHER: the crosswalk was
    ``pos[["item_id"]].drop_duplicates()``, so seeding SKU-9999 into pos put it
    *into* the crosswalk — making it 'known' on shipments and tautologically
    matched on pos. The defect was structurally invisible.
    """
    pos = _seed_item(defects["pos"], "SKU-9999")
    ship = _seed_item(defects["shipments"], "SKU-9999")
    xwalk = demo_data.defects_crosswalk()          # master data; lacks SKU-9999
    assert "SKU-9999" not in set(xwalk["item_id"].astype(str))

    pos_chk = _check(quality.run_qc("pos", pos, crosswalk=xwalk), "Crosswalk match")
    ship_chk = _check(quality.run_qc("shipments", ship, crosswalk=xwalk), "Crosswalk match")

    assert pos_chk.status == "warn"
    assert ship_chk.status == "warn"
    assert "SKU-9999" in set(pos_chk.data["unmatched_item_id"])
    assert "SKU-9999" in set(ship_chk.data["unmatched_item_id"])

    # And the old POS-derived stand-in demonstrably CANNOT catch it: with
    # SKU-9999 now in POS, the derived crosswalk knows it, so shipments passes.
    derived = pos[["item_id"]].drop_duplicates()
    derived_ship = _check(quality.run_qc("shipments", ship, crosswalk=derived),
                          "Crosswalk match")
    assert "SKU-9999" not in set(
        (derived_ship.data["unmatched_item_id"] if derived_ship.data is not None
         else pd.Series(dtype=str)))


# --------------------------------------------------------------------------- #
# The crosswalk as a workspace dimension (dim_* tables, replace-on-upload)
# --------------------------------------------------------------------------- #

def test_reference_upload_replaces_not_stacks(ws):
    """References are current-state master data: a re-upload REPLACES, it does
    not stack like a snapshot. Two uploads leave one dim table's worth of rows,
    reflecting the latest upload only."""
    first = pd.DataFrame({"source_item_id": ["A", "B"], "item_id": ["A", "B"]})
    second = pd.DataFrame({"source_item_id": ["A", "B", "C"],
                           "item_id": ["A", "B", "C"]})
    ws.add_reference("item_crosswalk", first, source_name="v1.csv")
    ws.add_reference("item_crosswalk", second, source_name="v2.csv")

    got = ws.read_reference("item_crosswalk")
    assert set(got["item_id"]) == {"A", "B", "C"}
    assert len(got) == 3               # replaced, not 5 stacked
    assert ws.has_reference("item_crosswalk")


def test_read_reference_is_none_until_uploaded(ws):
    assert ws.read_reference("item_crosswalk") is None
    assert ws.has_reference("item_crosswalk") is False


def test_rebuild_canonical_ignores_reference_tables(ws):
    """rebuild_canonical materialises canonical_* streams only. A dim_* table is
    invisible to it: not rebuilt, not dropped, and never a canonical stream."""
    demo_data.load_demo_into_workspace(ws)
    ws.add_reference("item_crosswalk",
                     pd.DataFrame({"source_item_id": ["A"], "item_id": ["A"]}),
                     source_name="x.csv")
    built = ws.rebuild_canonical()
    assert "item_crosswalk" not in built
    assert not ws.has_canonical("item_crosswalk")
    # the dim table survives the rebuild untouched
    assert ws.has_reference("item_crosswalk")
    assert set(ws.read_reference("item_crosswalk")["item_id"]) == {"A"}


def test_engine_never_reads_dim_tables(ws):
    """The engine reads ONLY canonical_* tables. run_pipeline is green with a
    full demo and NO crosswalk uploaded — proving the dim layer is not a
    required engine input."""
    demo_data.load_demo_into_workspace(ws)
    assert not ws.has_reference("item_crosswalk")
    cfg = RunConfig(as_of=date(2026, 5, 4))
    result = run_pipeline(ws, cfg.as_of, cfg)
    fc = ws.read_run_output(result.run_id, "forecast")
    assert len(fc) > 0


# --------------------------------------------------------------------------- #
# Dimension/fact boundary + quality.py freeze
# --------------------------------------------------------------------------- #

def test_reference_schemas_have_no_date_role():
    """Guards the dimension/fact boundary: a reference is dateless. Every
    canonical stream requires a 'week' (date role); no reference may carry a
    date role, or _grain_keys / snapshot_coverage / freshness would misfire."""
    assert config.REFERENCE_SCHEMAS
    for name, schema in config.REFERENCE_SCHEMAS.items():
        roles = {f: m["role"] for f, m in schema["fields"].items()}
        assert "date" not in roles.values(), f"{name} carries a date role"
    # and none of them are smuggled into the canonical stream contracts
    for name in config.REFERENCE_SCHEMAS:
        assert name not in config.STREAM_ORDER
        assert name not in config.CANONICAL_SCHEMAS


def test_reference_order_matches_schemas():
    assert set(config.REFERENCE_ORDER) == set(config.REFERENCE_SCHEMAS)
    assert "item_crosswalk" in config.REFERENCE_ORDER


def test_defects_crosswalk_omits_the_unmatched_item():
    """The defects master list contains every base SKU and deliberately NOT
    SKU-Z (the item seeded into shipments) — so gate 7 fires for a real reason."""
    xwalk = demo_data.defects_crosswalk()
    items = set(xwalk["item_id"].astype(str))
    assert {"SKU-A", "SKU-B", "SKU-C"} <= items
    assert "SKU-Z" not in items
    # every item actually present in the defects POS is covered (pass-on-pos)
    pos_items = set(demo_data.defects_fixtures()["pos"]["item_id"].astype(str))
    assert pos_items <= items


def test_load_qc_demo_lands_the_crosswalk_dimension(ws):
    """The one-click QC demo lands the crosswalk as a dim table so gate 7 has
    real master data to measure against."""
    demo_data.load_qc_demo_into_workspace(ws)
    assert ws.has_reference("item_crosswalk")
    xwalk = ws.read_reference("item_crosswalk")
    assert "SKU-Z" not in set(xwalk["item_id"].astype(str))


def test_quality_py_unchanged_signature():
    """Back-compat, as S3 froze it: run_qc keeps its exact parameter list. S9
    supplies a real crosswalk; it does not change the gate."""
    sig = inspect.signature(quality.run_qc)
    params = list(sig.parameters)
    assert params == ["stream", "df", "crosswalk", "pos", "as_of",
                      "sla_weeks", "phantom_weeks", "coverage_shift_threshold"]
    assert sig.parameters["crosswalk"].default is None
    assert sig.parameters["pos"].default is None
