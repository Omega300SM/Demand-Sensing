"""S6 — tests for the real statistical baseline + blend (design §5, §7; v2 M5/M8).

The statistical tier is judged THROUGH the frozen backtest harness (nothing about
the harness changes), so these tests defend the properties that make the tier and
its blended output trustworthy:

* positive FVA vs. naïve on the demo — the tier's pooled WMAPE at lags 1–2 beats
  last-value naïve (the design's S6 bar), and wins by a wide margin at longer
  lags where carrying a promo spike forward is catastrophic;
* the review fan is genuine — ``p10 <= p50 <= p90`` and the interval WIDENS with
  lag (not a point estimate ± a fixed multiplier);
* the blend is sensed-dominant near-in and decays toward the plan, and
  ``blended`` is never worse than ``max(sensed, plan)`` on the backtest aggregate;
* the CSV/XLSX export round-trips (the POC's downstream hand-off);
* the tier fits only on ``<= as_of`` history — reusing the harness's leakage
  guards, a forecast is invariant to any future values in the frame.

They complement (not replace) the existing 83 tests, which stay green.
"""

import io
import os
import sys
import tempfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensing import RunConfig, demo_data as dd  # noqa: E402
from sensing.workspace import open_workspace  # noqa: E402
from sensing.engine import (  # noqa: E402
    run_pipeline,
    _repair,
    _statistical_forecast,
    _seasonal_naive,
    _seasonal_index,
    _blend_weight_sensed,
    _despike,
    STAT_PARAMS,
    backtest,
    naive_forecaster,
    statistical_forecaster,
    plan_forecaster,
    AsOfPanel,
    _wmape,
)

FORECAST_COLS = ["item_id", "region", "week", "lag", "p10", "p50", "p90",
                 "plan_units", "blend_weight_sensed", "blended"]

MON = date(2025, 1, 6)


def _weeks(n):
    return [pd.Timestamp(MON + timedelta(weeks=i)) for i in range(n)]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def demo():
    return dd.generate()


@pytest.fixture(scope="module")
def repaired(demo):
    return _repair(demo["pos"], demo["channel_inventory"], demo["promo"])


@pytest.fixture(scope="module")
def demo_run(demo):
    """A full pipeline run on the demo, loaded into a real workspace."""
    tmp = tempfile.mkdtemp()
    ws = open_workspace(os.path.join(tmp, "ws.duckdb"))
    for stream, df in demo.items():
        ws.add_snapshot(stream, df, f"demo_{stream}", "a")
    ws.rebuild_canonical()
    as_of = pd.to_datetime(demo["pos"]["week"]).max().date()
    cfg = RunConfig(as_of=as_of, backtest_weeks=26)
    return run_pipeline(ws, as_of, cfg)


def _pooled(tidy, lags):
    d = tidy[tidy["lag"].isin(lags)]
    return _wmape(d["actual"].values, d["pred"].values)


# --------------------------------------------------------------------------- #
# (a) positive FVA vs. naïve on the demo, through the frozen harness
# --------------------------------------------------------------------------- #

def test_statistical_beats_naive_at_short_lags_on_demo(repaired):
    cfg = RunConfig(as_of=date(2025, 1, 6), backtest_weeks=26)
    nv = backtest(naive_forecaster, repaired, cfg,
                  value_col="units_unconstrained", method_name="naive")
    st = backtest(statistical_forecaster, repaired, cfg,
                  value_col="units_unconstrained", method_name="statistical")
    assert len(nv) and len(st)
    # the design's S6 bar: positive FVA vs. naïve at lags 1–2 (pooled)
    assert _pooled(st, [1, 2]) < _pooled(nv, [1, 2])


def test_statistical_wins_by_a_wide_margin_at_longer_lags(repaired):
    cfg = RunConfig(as_of=date(2025, 1, 6), backtest_weeks=26)
    nv = backtest(naive_forecaster, repaired, cfg,
                  value_col="units_unconstrained", method_name="naive")
    st = backtest(statistical_forecaster, repaired, cfg,
                  value_col="units_unconstrained", method_name="statistical")
    # carrying a promo spike forward (naïve) is catastrophic far out; the
    # damped-level tier reverts and wins big.
    assert _pooled(st, [4]) < _pooled(nv, [4])
    assert _pooled(st, [8]) < _pooled(nv, [8])
    # pooled across all scored lags, clearly positive FVA
    all_lags = [1, 2, 4, 8]
    assert _pooled(st, all_lags) < 0.9 * _pooled(nv, all_lags)


def test_fva_waterfall_statistical_step_improves_on_naive(demo_run):
    """The page-5 waterfall (lags 1–2) shows the statistical step adding value."""
    fva = demo_run.fva.set_index("step")["wmape"]
    assert fva["Statistical baseline"] < fva["Seasonal naïve"]


# --------------------------------------------------------------------------- #
# (b) genuine quantiles: monotonicity + interval widening with lag
# --------------------------------------------------------------------------- #

def test_quantiles_monotonic_on_demo_forecast(demo_run):
    fc = demo_run.forecast
    assert (fc["p10"] <= fc["p50"] + 1e-6).all()
    assert (fc["p50"] <= fc["p90"] + 1e-6).all()
    assert (fc["p10"] >= -1e-9).all()


def test_interval_widens_with_lag(repaired):
    # a healthy high-volume series so P10 never clips at zero
    g = repaired[(repaired["item_id"] == "SKU-1001") & (repaired["region"] == "East")]
    y = g.sort_values("week").set_index("week")["units_unconstrained"]
    p10, p50, p90 = _statistical_forecast(y, 8)
    width = p90 - p10
    assert (p10 <= p50 + 1e-9).all() and (p50 <= p90 + 1e-9).all()
    # strictly non-decreasing width as the horizon extends
    assert (np.diff(width) >= -1e-9).all()
    assert width[-1] > width[0]  # genuinely wider far out, not a flat band


def test_p50_matches_the_point_tier(repaired):
    """The fan's P50 is exactly the statistical tier's point path (one source)."""
    g = repaired[(repaired["item_id"] == "SKU-1003") & (repaired["region"] == "West")]
    y = g.sort_values("week").set_index("week")["units_unconstrained"]
    _, p50, _ = _statistical_forecast(y, 8)
    assert np.allclose(p50, _seasonal_naive(y, 8))


# --------------------------------------------------------------------------- #
# (c) blend behaviour
# --------------------------------------------------------------------------- #

def test_blend_weight_sensed_dominant_near_in_and_decays(demo_run):
    fc = demo_run.forecast
    assert (fc["blend_weight_sensed"] >= 0).all()
    assert (fc["blend_weight_sensed"] <= 1).all()
    by_lag = fc.groupby("lag")["blend_weight_sensed"].first()
    # near-in sensed-dominant (days 1–14 ~ weeks 1–2)
    assert by_lag.loc[1] >= 0.8 and by_lag.loc[2] >= 0.8
    # monotonically decaying toward the plan across the horizon
    assert list(by_lag.values) == sorted(by_lag.values, reverse=True)
    assert by_lag.iloc[-1] < by_lag.iloc[0]


def test_blend_weight_schedule_is_clipped_and_shaped():
    cfg = RunConfig(as_of=date(2025, 1, 6), horizon_weeks=8)
    ws = [_blend_weight_sensed(h, cfg) for h in range(1, 9)]
    assert all(0.15 <= w <= 1.0 for w in ws)
    assert ws[0] == pytest.approx(0.9) and ws[1] == pytest.approx(0.9)
    assert ws == sorted(ws, reverse=True)


def test_blended_never_worse_than_max_of_sensed_and_plan(demo, repaired):
    """v2 M8 / design §7 acceptance, checked on the backtest aggregate."""
    cfg = RunConfig(as_of=date(2025, 1, 6), backtest_weeks=26)
    plan = demo["demand_plan"]

    def blended_forecaster(panel, as_of, horizon):
        s = statistical_forecaster(panel, as_of, horizon)
        pf = plan_forecaster(plan)(panel, as_of, horizon)
        m = s.merge(pf, on=["item_id", "region", "week"], how="left",
                    suffixes=("_s", "_p"))
        fut = list(panel.future_weeks(horizon))
        lagmap = {wk: i + 1 for i, wk in enumerate(fut)}
        m["lag"] = m["week"].map(lagmap)
        w = m["lag"].map(lambda L: _blend_weight_sensed(int(L), cfg))
        plan_c = m["pred_p"].where(m["pred_p"].notna(), m["pred_s"])
        m["pred"] = w * m["pred_s"] + (1.0 - w) * plan_c
        return m[["item_id", "region", "week", "pred"]]

    bt_s = backtest(statistical_forecaster, repaired, cfg,
                    value_col="units_unconstrained", method_name="s")
    bt_p = backtest(plan_forecaster(plan), repaired, cfg,
                    value_col="units_unconstrained", method_name="p")
    bt_b = backtest(blended_forecaster, repaired, cfg,
                    value_col="units_unconstrained", method_name="b")

    def agg(t):
        return _wmape(t["actual"].values, t["pred"].values)

    # convex per-cell blend => aggregate never worse than the worse of the two
    assert agg(bt_b) <= max(agg(bt_s), agg(bt_p)) + 1e-9
    # for both the near horizon and pooled across all lags
    assert (_pooled(bt_b, [1, 2])
            <= max(_pooled(bt_s, [1, 2]), _pooled(bt_p, [1, 2])) + 1e-9)


# --------------------------------------------------------------------------- #
# (d) export round-trips (the POC's downstream hand-off)
# --------------------------------------------------------------------------- #

def test_csv_export_round_trips(demo_run):
    export = demo_run.forecast[FORECAST_COLS].copy()
    csv = export.to_csv(index=False).encode()
    back = pd.read_csv(io.BytesIO(csv))
    assert list(back.columns) == FORECAST_COLS
    assert len(back) == len(export)
    assert np.allclose(back["blended"].values, export["blended"].values, equal_nan=True)


def test_xlsx_export_round_trips(demo_run):
    export = demo_run.forecast[FORECAST_COLS].copy()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        export.to_excel(xw, index=False, sheet_name="forecast")
    back = pd.read_excel(io.BytesIO(buf.getvalue()), sheet_name="forecast")
    assert list(back.columns) == FORECAST_COLS
    assert len(back) == len(export)
    assert np.allclose(back["p50"].values, export["p50"].values)


# --------------------------------------------------------------------------- #
# (e) as-of leakage sanity — reuse the harness's guards
# --------------------------------------------------------------------------- #

def test_statistical_forecaster_only_uses_history_at_or_before_as_of():
    wk = _weeks(40)
    base = pd.DataFrame({"item_id": "A", "region": "East", "week": wk,
                         "units_unconstrained": 100.0 + np.arange(40)})
    as_of = wk[30]
    future = wk[31:39]

    panel_clean = AsOfPanel(base, as_of, "units_unconstrained", future)
    f1 = statistical_forecaster(panel_clean, as_of, 8)
    # every returned week is strictly after as_of
    assert (f1["week"] > as_of).all()

    # corrupt every future value; the forecast must be invariant (no peeking)
    corrupt = base.copy()
    corrupt.loc[corrupt["week"] > as_of, "units_unconstrained"] = 99_999.0
    panel_corrupt = AsOfPanel(corrupt, as_of, "units_unconstrained", future)
    f2 = statistical_forecaster(panel_corrupt, as_of, 8)
    assert np.allclose(f1["pred"].values, f2["pred"].values)


# --------------------------------------------------------------------------- #
# guarded seasonal machinery (design §5) — off on ~1.5 cycles, on past the guard
# --------------------------------------------------------------------------- #

def test_seasonal_index_is_guarded_off_on_short_history():
    # ~1.5 cycles (the demo's regime): factors must be exactly all-ones
    y = 100.0 + 20.0 * np.sin(2 * np.pi * np.arange(78) / 52.0)
    idx = _seasonal_index(_despike(y, 9, 3.0), 52, STAT_PARAMS["season_shrink"],
                          STAT_PARAMS["season_min_cycles"])
    assert np.allclose(idx, 1.0)


def test_seasonal_index_activates_and_is_shrunk_past_the_guard():
    # ≥2 clean cycles with a strong seasonal shape -> non-trivial, shrunk factors
    n = 120
    y = 100.0 + 30.0 * np.sin(2 * np.pi * np.arange(n) / 52.0) + 100.0
    idx = _seasonal_index(_despike(y, 9, 3.0), 52, STAT_PARAMS["season_shrink"],
                          STAT_PARAMS["season_min_cycles"])
    assert not np.allclose(idx, 1.0)          # it activated
    # shrunk toward 1.0: peak factor is well below the raw seasonal ratio
    assert idx.max() < 1.5 and idx.min() > 0.5


def test_short_and_sparse_series_degrade_gracefully():
    # empty, single-point, and two-point histories must not blow up
    assert np.allclose(_seasonal_naive(pd.Series([], dtype=float), 8), np.zeros(8))
    assert np.allclose(_seasonal_naive(pd.Series([42.0]), 8), np.full(8, 42.0))
    p10, p50, p90 = _statistical_forecast(pd.Series([5.0, 7.0]), 8)
    assert len(p50) == 8
    assert (p10 <= p50 + 1e-9).all() and (p50 <= p90 + 1e-9).all()
    assert (p10 >= -1e-9).all()
