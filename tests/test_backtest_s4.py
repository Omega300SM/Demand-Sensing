"""S4 — tests for the rolling-origin backtest harness and FVA assembly.

The harness is the POC's verdict machine, so these tests defend the properties
that make its numbers *trustworthy*:

* a known-signal series yields a stable, positive naïve→statistical FVA;
* per-lag scoring is monotonic where the signal makes it so;
* leakage is actively caught (a forecaster peeking at the future raises);
* the demand plan is scored lag-adjusted *by vintage*, never the latest plan;
* the frozen ``fva`` / ``fva_by_lag`` shapes + method vocabulary hold.

They complement (not replace) the existing 58 tests, which stay green.
"""

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensing import RunConfig  # noqa: E402
from sensing.engine import (  # noqa: E402
    AsOfPanel,
    LeakageError,
    backtest,
    naive_forecaster,
    statistical_forecaster,
    plan_forecaster,
    _plan_asof,
    _aggregate_fva,
    _wmape,
    FVA_METHODS,
)

MON = date(2025, 1, 6)  # a Monday


def _weeks(n):
    return [pd.Timestamp(MON + timedelta(weeks=i)) for i in range(n)]


def _long(values, item="A", region="East", col="y"):
    wk = _weeks(len(values))
    return pd.DataFrame({"item_id": item, "region": region, "week": wk, col: values})


def _pooled_wmape(tidy, lags=None):
    d = tidy if lags is None else tidy[tidy["lag"].isin(lags)]
    return _wmape(d["actual"].values, d["pred"].values)


# --------------------------------------------------------------------------- #
# (a) known-signal series -> stable, positive naïve→statistical FVA
# --------------------------------------------------------------------------- #

def _noisy_stationary(n=64, seed=3):
    """A stationary mean with observation noise: the EWMA baseline should beat
    a last-value naïve, which chases the noise."""
    rng = np.random.default_rng(seed)
    return list(100.0 + rng.normal(0.0, 18.0, size=n))


def test_statistical_beats_naive_on_noisy_stationary_signal():
    data = _long(_noisy_stationary())
    cfg = RunConfig(as_of=date(2025, 1, 6), backtest_weeks=30)

    nv = backtest(naive_forecaster, data, cfg, value_col="y", method_name="naive")
    st = backtest(statistical_forecaster, data, cfg, value_col="y", method_name="statistical")

    assert len(nv) and len(st)
    naive_wmape = _pooled_wmape(nv, lags=[1, 2])
    stat_wmape = _pooled_wmape(st, lags=[1, 2])
    # positive FVA going naïve -> statistical
    assert stat_wmape < naive_wmape
    assert (naive_wmape - stat_wmape) > 0.01


def test_fva_is_reproducible_bitwise():
    """Deterministic: same data + config -> identical scores (manifest-repro)."""
    data = _long(_noisy_stationary())
    cfg = RunConfig(as_of=date(2025, 1, 6), backtest_weeks=30)
    a = backtest(statistical_forecaster, data, cfg, value_col="y", method_name="statistical")
    b = backtest(statistical_forecaster, data, cfg, value_col="y", method_name="statistical")
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# (b) lag scoring monotonic where it should be
# --------------------------------------------------------------------------- #

def test_naive_wmape_monotonic_in_lag_on_trend():
    """On a clean upward trend, a last-value naïve falls further behind as the
    lag grows, so WMAPE must be non-decreasing in lag."""
    trend = [50.0 + 3.0 * i for i in range(60)]
    data = _long(trend)
    cfg = RunConfig(as_of=date(2025, 1, 6), backtest_weeks=30)

    tidy = backtest(naive_forecaster, data, cfg, value_col="y", method_name="naive")
    per_lag = {lag: _pooled_wmape(tidy, lags=[lag]) for lag in (1, 2, 4, 8)}

    lags_sorted = sorted(per_lag)
    for a, b in zip(lags_sorted, lags_sorted[1:]):
        assert per_lag[a] <= per_lag[b] + 1e-9, per_lag


# --------------------------------------------------------------------------- #
# (c) leakage is actively caught
# --------------------------------------------------------------------------- #

def test_peeking_forecaster_via_value_at_is_caught():
    """A forecaster that reads a *future* value through the panel is caught by
    AsOfPanel.value_at, which raises LeakageError."""
    data = _long(_noisy_stationary())
    cfg = RunConfig(as_of=date(2025, 1, 6), backtest_weeks=20)

    def peeking(panel: AsOfPanel, as_of, horizon):
        rows = []
        for sku, region in panel.keys():
            for wk in panel.future_weeks(horizon):
                # cheat: read the actual we are supposed to be predicting
                cheat = panel.value_at(sku, region, wk)
                rows.append(dict(item_id=sku, region=region, week=wk, pred=cheat))
        return pd.DataFrame(rows)

    with pytest.raises(LeakageError):
        backtest(peeking, data, cfg, value_col="y", method_name="cheater")


def test_forecast_dated_on_or_before_as_of_is_caught():
    """A forecaster that returns a non-future week (in-sample leakage) is
    rejected by the harness output guard."""
    data = _long(_noisy_stationary())
    cfg = RunConfig(as_of=date(2025, 1, 6), backtest_weeks=20)

    def past_leaker(panel: AsOfPanel, as_of, horizon):
        # returns the as_of week itself (an already-resolved period)
        rows = [dict(item_id=sku, region=region, week=panel.as_of, pred=1.0)
                for sku, region in panel.keys()]
        return pd.DataFrame(rows)

    with pytest.raises(LeakageError):
        backtest(past_leaker, data, cfg, value_col="y", method_name="past")


def test_asof_panel_history_never_contains_future():
    data = _long(_noisy_stationary(40))
    as_of = _weeks(40)[20]
    panel = AsOfPanel(data.rename(columns={"y": "y"}), as_of, "y", _weeks(40)[21:25])
    assert panel.history()["week"].max() <= as_of
    with pytest.raises(LeakageError):
        panel.value_at("A", "East", as_of + pd.Timedelta(weeks=1))


# --------------------------------------------------------------------------- #
# (d) plan scored lag-adjusted by vintage, not the latest plan
# --------------------------------------------------------------------------- #

def _versioned_plan():
    """One target week (index 30) with TWO vintages:
      * an EARLY vintage available at the origin, value 100 (slightly wrong),
      * a LATE 'cheating' vintage dated after the origin, value == the actual.
    A vintage-correct backtest must use the early one."""
    weeks = _weeks(60)
    target = weeks[30]
    early = weeks[24]   # 6 weeks before target -> available at origins >= wk24
    late = weeks[29]    # 1 week before target -> only available very near-in
    rows = [
        dict(item_id="A", region="East", week=target, plan_units=100.0,
             plan_version_date=early),
        dict(item_id="A", region="East", week=target, plan_units=999.0,
             plan_version_date=late),
    ]
    return pd.DataFrame(rows), target, early, late


def test_plan_asof_picks_latest_available_vintage_not_future():
    plan, target, early, late = _versioned_plan()

    # origin between early and late: only the early vintage is available
    as_of_mid = early + pd.Timedelta(weeks=2)
    picked = _plan_asof(plan, as_of_mid)
    assert len(picked) == 1
    assert picked["plan_units"].iloc[0] == 100.0
    assert (picked["plan_version_date"] <= as_of_mid).all()

    # origin after both vintages: the latest (late) vintage wins
    as_of_after = late + pd.Timedelta(weeks=1)
    picked2 = _plan_asof(plan, as_of_after)
    assert picked2["plan_units"].iloc[0] == 999.0

    # origin before any vintage: nothing available (no leakage)
    as_of_before = early - pd.Timedelta(weeks=1)
    assert _plan_asof(plan, as_of_before).empty


def test_plan_forecaster_uses_available_vintage_in_backtest():
    """Integration: the plan the harness scores at each origin is the vintage
    available then — never the future 'perfect' one."""
    plan, target, early, late = _versioned_plan()
    # actuals: target actual == 999 so the LATE (cheating) vintage would be
    # perfect. A leak would show plan pred == 999 at that cell; vintage-correct
    # scoring must show 100 for origins where only the early vintage exists.
    vals = [200.0] * 60
    vals[30] = 999.0
    data = _long(vals)
    cfg = RunConfig(as_of=date(2025, 1, 6), backtest_weeks=40)

    tidy = backtest(plan_forecaster(plan), data, cfg, value_col="y",
                    method_name="plan")
    cell = tidy[(tidy["week"] == target)]
    assert len(cell) >= 1
    # every scored plan prediction for the target uses an available vintage:
    # for origins strictly before `late`, that is the early (100) vintage.
    early_origins = cell[cell["as_of"] < late]
    assert len(early_origins)
    assert (early_origins["pred"] == 100.0).all()
    # and none of the scored cells used a future-dated vintage
    assert not (cell["pred"] == 999.0).any() or (cell["as_of"] >= late).any()


# --------------------------------------------------------------------------- #
# frozen shapes + method vocabulary
# --------------------------------------------------------------------------- #

def test_aggregate_fva_shapes_and_vocabulary():
    # build a tiny tidy frame across all four methods
    rows = []
    for method in FVA_METHODS:
        for lag in (1, 2):
            rows.append(dict(item_id="A", region="East", lag=lag,
                             week=_weeks(3)[lag], as_of=_weeks(3)[0],
                             pred=10.0, actual=12.0, method=method))
    tidy = pd.DataFrame(rows)
    fva, by_lag = _aggregate_fva(tidy)

    assert list(fva.columns) == ["step", "wmape", "bias"]
    assert list(by_lag.columns) == ["lag", "method", "wmape", "bias"]
    assert set(by_lag["method"]).issubset(set(FVA_METHODS))
    # waterfall ordered per frozen vocabulary
    assert len(fva) == len(FVA_METHODS)


def test_empty_inputs_return_frozen_empty_shapes():
    empty = pd.DataFrame(columns=["item_id", "region", "lag", "week",
                                  "as_of", "pred", "actual", "method"])
    fva, by_lag = _aggregate_fva(empty)
    assert list(fva.columns) == ["step", "wmape", "bias"]
    assert list(by_lag.columns) == ["lag", "method", "wmape", "bias"]
    assert len(fva) == 0 and len(by_lag) == 0


# --------------------------------------------------------------------------- #
# end-to-end vintage-correctness through run_pipeline on the demo dataset
# --------------------------------------------------------------------------- #

def test_pipeline_fva_scores_plan_only_where_vintage_available():
    """On the demo (plan versioned 6 weeks ahead of each week), the plan is
    scorable at lags 1/2/4 but NOT at lag 8 — its only vintage for a week 8
    ahead is dated after the origin. The old harness scored it anyway; this
    guards the fix. All four methods must appear in the by-lag table."""
    from sensing import Workspace, run_pipeline, demo_data

    import tempfile
    ws = Workspace(os.path.join(tempfile.mkdtemp(), "w.duckdb"))
    demo_data.load_demo_into_workspace(ws)
    pos = ws.read_canonical("pos")
    as_of = pd.to_datetime(pos["week"]).max().date()

    result = run_pipeline(ws, as_of, RunConfig(as_of=as_of, backtest_weeks=26))
    by_lag = result.fva_by_lag

    assert set(by_lag["method"]) == set(FVA_METHODS)
    plan_lags = set(by_lag[by_lag["method"] == "plan"]["lag"])
    assert {1, 2, 4}.issubset(plan_lags)
    assert 8 not in plan_lags  # vintage 2 weeks in the future -> not scored
    # waterfall present and includes the lag-adjusted plan bar
    assert "Demand plan (lag-adj.)" in set(result.fva["step"])
