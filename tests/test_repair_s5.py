"""S5 — tests for real signal repair (design §4, v1 §4, v2 M3).

Repair is judged THROUGH the frozen backtest harness, so these tests defend the
properties that make ``units_unconstrained`` a trustworthy sell-out target:

* the ONE injected stockout (SKU-1002/East, weeks 38–42) is flagged
  ``is_censored`` and its demand is recovered close to the surrounding healthy
  level — not left at the depressed raw sales;
* the two promo windows yield positive ``promo_uplift`` over a sensible
  de-promoted ``base_units``, and the post-promo dip is NOT mis-flagged as
  censoring;
* repair degrades gracefully with EMPTY channel_inventory and/or EMPTY promo,
  still emitting the frozen columns;
* repair never mutates its inputs;
* with real repair the harness re-judges automatically — scoring on the
  de-censored target beats scoring on raw ``units_sold`` at the stockout weeks,
  with NO change to the harness.

They complement (not replace) the existing 69 tests, which stay green.
"""

import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensing import RunConfig, Workspace, demo_data as dd  # noqa: E402
from sensing.engine import (  # noqa: E402
    _repair,
    _dampen_outliers,
    _robust_level,
    REPAIR_PARAMS,
    backtest,
    statistical_forecaster,
    translated_forecaster,
    _wmape,
)

REPAIRED_COLS = ["item_id", "region", "week", "units_sold",
                 "units_unconstrained", "is_censored", "base_units", "promo_uplift"]


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def demo_streams():
    """The clean demo streams, exactly as loaded into a workspace."""
    data = dd.generate()
    return data["pos"], data["channel_inventory"], data["promo"]


def _week_ts(idx: int) -> pd.Timestamp:
    return pd.Timestamp(dd.START_MONDAY + timedelta(weeks=idx))


def _stockout_week_set():
    return {_week_ts(i) for i in range(dd.STOCKOUT["start"], dd.STOCKOUT["end"] + 1)}


def _promo_week_set():
    wks = set()
    for a, b in dd.PROMO_WINDOWS:
        wks |= {_week_ts(i) for i in range(a, b + 1)}
    return wks


def _post_promo_dip_week_set():
    # the generator applies the dip for end < i <= end + 2
    wks = set()
    for _, b in dd.PROMO_WINDOWS:
        wks |= {_week_ts(i) for i in (b + 1, b + 2)}
    return wks


def _series(rep, sku, region):
    return rep[(rep["item_id"] == sku) & (rep["region"] == region)].copy()


# --------------------------------------------------------------------------- #
# (a) injected stockout is flagged and recovered
# --------------------------------------------------------------------------- #

def test_injected_stockout_is_flagged_censored(demo_streams):
    pos, inv, promo = demo_streams
    rep = _repair(pos, inv, promo)
    stk = _series(rep, dd.STOCKOUT["sku"], dd.STOCKOUT["region"])
    stk["week"] = pd.to_datetime(stk["week"])

    censored_weeks = set(stk.loc[stk["is_censored"], "week"])
    # exactly the injected stockout weeks are flagged on this series
    assert censored_weeks == _stockout_week_set()


def test_only_the_injected_series_is_censored(demo_streams):
    """No false positives: no other SKU-region has any censored week."""
    pos, inv, promo = demo_streams
    rep = _repair(pos, inv, promo)
    rep["week"] = pd.to_datetime(rep["week"])
    others = rep[~((rep["item_id"] == dd.STOCKOUT["sku"]) &
                   (rep["region"] == dd.STOCKOUT["region"]))]
    assert not others["is_censored"].any()


def test_stockout_demand_recovered_near_healthy_level(demo_streams):
    pos, inv, promo = demo_streams
    rep = _repair(pos, inv, promo)
    stk = _series(rep, dd.STOCKOUT["sku"], dd.STOCKOUT["region"]).sort_values("week")
    stk["week"] = pd.to_datetime(stk["week"])

    cens_mask = stk["is_censored"].to_numpy()
    recovered = stk.loc[cens_mask, "units_unconstrained"].to_numpy()
    raw = stk.loc[cens_mask, "units_sold"].to_numpy()

    # ground-truth proxy: the healthy sell-out just before/after the hole
    start, end = dd.STOCKOUT["start"], dd.STOCKOUT["end"]
    healthy_idx = list(range(start - 5, start)) + list(range(end + 1, end + 6))
    healthy_weeks = {_week_ts(i) for i in healthy_idx}
    healthy = stk.loc[stk["week"].isin(healthy_weeks), "units_sold"].to_numpy()
    healthy_level = float(np.median(healthy))

    # recovered demand tracks the healthy level (within 20%), not the raw hole
    assert abs(recovered.mean() - healthy_level) / healthy_level < 0.20
    # and is a large recovery over the depressed raw sales
    assert recovered.mean() > 4.0 * raw.mean()
    # uncensored weeks are left exactly equal to raw sales
    unc_rows = stk.loc[~cens_mask]
    assert np.allclose(unc_rows["units_unconstrained"], unc_rows["units_sold"])


# --------------------------------------------------------------------------- #
# (b) promo decomposition; post-promo dip not mis-flagged
# --------------------------------------------------------------------------- #

def test_promo_windows_yield_positive_uplift_over_sensible_base(demo_streams):
    pos, inv, promo = demo_streams
    rep = _repair(pos, inv, promo)
    rep["week"] = pd.to_datetime(rep["week"])
    promo_weeks = _promo_week_set()

    # invariant everywhere: base + uplift == unconstrained, uplift >= 0
    assert np.allclose(rep["base_units"] + rep["promo_uplift"], rep["units_unconstrained"])
    assert (rep["promo_uplift"] >= -1e-9).all()

    # every series sees positive uplift concentrated on promo weeks
    for (sku, region), g in rep.groupby(["item_id", "region"]):
        g = g.sort_values("week")
        on_promo = g[g["week"].isin(promo_weeks)]
        off_promo = g[~g["week"].isin(promo_weeks)]
        assert on_promo["promo_uplift"].sum() > 0
        # uplift lives on promo weeks only
        assert np.allclose(off_promo["promo_uplift"], 0.0)
        # de-promoted base sits well below the promoted demand on promo weeks
        assert (on_promo["base_units"] < on_promo["units_unconstrained"]).mean() > 0.5
        # base on promo weeks is near the surrounding non-promo level (within 35%)
        base_ref = float(off_promo["units_unconstrained"].median())
        assert abs(on_promo["base_units"].median() - base_ref) / base_ref < 0.35


def test_post_promo_dip_not_flagged_as_censoring(demo_streams):
    pos, inv, promo = demo_streams
    rep = _repair(pos, inv, promo)
    rep["week"] = pd.to_datetime(rep["week"])
    dip_weeks = _post_promo_dip_week_set()
    dip_rows = rep[rep["week"].isin(dip_weeks)]
    # the pantry-loading dip is real demand, never de-censored
    assert not dip_rows["is_censored"].any()


# --------------------------------------------------------------------------- #
# (c) graceful degradation with empty optional streams
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("drop_inv,drop_promo", [(True, False), (False, True), (True, True)])
def test_repair_degrades_gracefully(demo_streams, drop_inv, drop_promo):
    pos, inv, promo = demo_streams
    use_inv = inv.iloc[0:0] if drop_inv else inv
    use_promo = promo.iloc[0:0] if drop_promo else promo

    rep = _repair(pos, use_inv, use_promo)
    # frozen columns and dtypes hold regardless of which optional streams exist
    assert list(rep.columns) == REPAIRED_COLS
    assert rep["is_censored"].dtype == bool
    # no promo -> uplift is identically zero and base == unconstrained
    if drop_promo:
        assert (rep["promo_uplift"] == 0).all()
        assert np.allclose(rep["base_units"], rep["units_unconstrained"])
    # the stockout is still caught: with inventory dropped, the statistical
    # zero-inflation test alone recovers it (both signals wired, design §4)
    stk = _series(rep, dd.STOCKOUT["sku"], dd.STOCKOUT["region"])
    stk["week"] = pd.to_datetime(stk["week"])
    assert set(stk.loc[stk["is_censored"], "week"]) == _stockout_week_set()


def test_repair_on_empty_pos_returns_frozen_empty_frame():
    empty = pd.DataFrame(columns=["item_id", "region", "week", "units_sold"])
    rep = _repair(empty, pd.DataFrame(), pd.DataFrame())
    assert list(rep.columns) == REPAIRED_COLS
    assert len(rep) == 0


# --------------------------------------------------------------------------- #
# (d) no mutation of inputs
# --------------------------------------------------------------------------- #

def test_repair_does_not_mutate_inputs(demo_streams):
    pos, inv, promo = demo_streams
    pos_c, inv_c, promo_c = pos.copy(), inv.copy(), promo.copy()
    _repair(pos, inv, promo)
    pd.testing.assert_frame_equal(pos, pos_c)
    pd.testing.assert_frame_equal(inv, inv_c)
    pd.testing.assert_frame_equal(promo, promo_c)


# --------------------------------------------------------------------------- #
# (e) the frozen harness re-judges automatically with better repair
# --------------------------------------------------------------------------- #

def _wmape_at_weeks(tidy, week_set, lags=(1, 2, 4, 8)):
    t = tidy.copy()
    t["week"] = pd.to_datetime(t["week"])
    d = t[t["lag"].isin(lags) & t["week"].isin(week_set)]
    return _wmape(d["actual"].values, d["pred"].values), len(d)


@pytest.mark.parametrize("forecaster,name",
                         [(statistical_forecaster, "statistical"),
                          (translated_forecaster, "translated")])
def test_repair_moves_fva_without_touching_harness(demo_streams, forecaster, name):
    """Scoring on the de-censored target beats scoring on raw ``units_sold`` at
    the stockout weeks. Same forecaster, same harness — only the target column
    changes — so this proves better repair moves FVA with no harness change."""
    pos, inv, promo = demo_streams
    rep = _repair(pos, inv, promo)
    stk = _series(rep, dd.STOCKOUT["sku"], dd.STOCKOUT["region"])

    # wide backtest window so rolling origins actually forecast INTO the hole
    cfg = RunConfig(as_of=pd.to_datetime(pos["week"]).max().date(), backtest_weeks=60)
    stockout_weeks = _stockout_week_set()

    t_unc = backtest(forecaster, stk, cfg, value_col="units_unconstrained", method_name=name)
    t_raw = backtest(forecaster, stk, cfg, value_col="units_sold", method_name=name)

    wmape_unc, n_unc = _wmape_at_weeks(t_unc, stockout_weeks)
    wmape_raw, n_raw = _wmape_at_weeks(t_raw, stockout_weeks)

    assert n_unc > 0 and n_raw > 0
    # de-censored scoring is strictly better where the stockout lives
    assert wmape_unc < wmape_raw


# --------------------------------------------------------------------------- #
# outlier / structural-break rule: isolated blip corrected, shift preserved
# --------------------------------------------------------------------------- #

def test_isolated_outlier_dampened_structural_break_preserved():
    n = 40
    y = np.full(n, 100.0)
    # a sustained upward level shift for the back half -> structural break
    y[20:] = 200.0
    # an isolated one-off spike inside the first (flat) regime
    y[10] = 900.0

    expected = _robust_level(y, int(REPAIR_PARAMS["expected_window"]))
    promo_flag = np.zeros(n)
    censored = np.zeros(n, dtype=bool)
    out = _dampen_outliers(y, promo_flag, censored, expected, REPAIR_PARAMS)

    # isolated blip pulled toward the local level (far from its raw 900)
    assert out[10] < 300.0
    # sustained shift left intact — the forecaster should adapt, not have it erased
    assert np.allclose(out[25:], 200.0)
    assert np.allclose(out[:9], 100.0)


def test_dampen_never_touches_promo_or_censored_weeks():
    n = 20
    y = np.full(n, 100.0)
    y[5] = 900.0    # would be an isolated outlier...
    y[12] = 5.0     # ...and a low one
    expected = _robust_level(y, int(REPAIR_PARAMS["expected_window"]))
    promo_flag = np.zeros(n); promo_flag[5] = 1        # protect week 5 as promo
    censored = np.zeros(n, dtype=bool); censored[12] = True  # protect week 12 as censored
    out = _dampen_outliers(y, promo_flag, censored, expected, REPAIR_PARAMS)
    assert out[5] == 900.0     # promo spike preserved
    assert out[12] == 5.0      # censored value left for the imputer to own
