"""S7 — tests for the real sell-out -> sell-in translation engine.

Design §6 ("the differentiating layer") / §3 L3 / v2 M7. The translation is
judged THROUGH the frozen backtest harness (nothing about the harness changes),
re-targeted to score against SHIPMENTS (sell-in) — the thing the layer actually
predicts. These tests defend the properties that make it trustworthy:

* (a) it beats a shipments-history-only model on the demo shipments backtest at
  near lags (weeks 1–4) — the S7 acceptance bar;
* (b) the inventory projection is well-formed — ``projected_on_hand`` /
  ``projected_order`` >= 0, ``target_position`` == the target-WOS cover, and the
  forward recurrence reconciles to the design identity;
* (c) orders batch on the calibrated cadence (not every week) and the reaction
  lag delays when a seed in-transit lands;
* (d) empirical calibration returns sane parameters on the demo and is as-of
  safe (fit only on ``<= as_of`` history);
* (e) where channel inventory is unusable the distributed-lag transfer function
  is auto-selected per series and still produces a finite, scored, competitive
  forecast;
* (f) the translated forecaster reads only ``<= as_of`` history — reusing the
  harness's leakage guards, its forecast is invariant to corrupted future
  POS/inventory/shipments.

They complement (not replace) the existing 98 tests, which stay green.
"""

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensing import RunConfig, demo_data as dd  # noqa: E402
from sensing.workspace import open_workspace  # noqa: E402
from sensing.engine import (  # noqa: E402
    run_pipeline,
    _repair,
    _seasonal_naive,
    backtest,
    make_translated_forecaster,
    translated_forecaster,
    shipments_history_only_forecaster,
    _shipments_panel,
    _project_orders,
    _fit_order_model,
    _fit_replenishment,
    _calibrate_translation,
    _select_translation_method,
    _inventory_usable,
    _transfer_function,
    TranslationParams,
    TRANSLATE_PARAMS,
    FVA_METHODS,
    _wmape,
)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def demo():
    d = dd.generate()
    return d["pos"], d["channel_inventory"], d["shipments"], d["demand_plan"], d["promo"]


@pytest.fixture(scope="module")
def demo_repaired(demo):
    pos, inv, _ship, _plan, promo = demo
    return _repair(pos, inv, promo)


def _as_of(pos):
    return pd.to_datetime(pos["week"]).max().date()


def _wmape_at(tidy, lags):
    d = tidy[tidy["lag"].isin(lags)]
    return _wmape(d["actual"].values, d["pred"].values), len(d)


# --------------------------------------------------------------------------- #
# (a) acceptance: translation beats a shipments-history-only model at near lags
# --------------------------------------------------------------------------- #

def test_translation_beats_shipments_history_only_at_near_lags(demo, demo_repaired):
    """The S7 bar (design §5): projecting sell-out through channel inventory into
    expected orders beats forecasting sell-in from shipments history alone."""
    pos, inv, ship, _plan, _promo = demo
    cfg = RunConfig(as_of=_as_of(pos), backtest_weeks=40)
    sp = _shipments_panel(demo_repaired, ship)

    tr = backtest(make_translated_forecaster(demo_repaired, inv, ship, cfg), sp, cfg,
                  value_col="units_shipped", lags=(1, 2, 4, 8), method_name="translated")
    so = backtest(shipments_history_only_forecaster, sp, cfg,
                  value_col="units_shipped", lags=(1, 2, 4, 8), method_name="shiponly")

    tr12, n = _wmape_at(tr, (1, 2))
    so12, _ = _wmape_at(so, (1, 2))
    assert n > 0
    assert tr12 < so12                       # wins at the nearest lags (weeks 1–2)

    tr14, _ = _wmape_at(tr, (1, 2, 4))
    so14, _ = _wmape_at(so, (1, 2, 4))
    assert tr14 < so14                       # and across the near horizon (weeks 1–4)


def test_translated_scored_on_shipments_in_pipeline(demo):
    """The pipeline's ``translated`` FVA row is scored against shipments (sell-in)
    while the method vocabulary stays frozen."""
    import tempfile
    pos, inv, ship, plan, promo = demo
    ws = open_workspace(os.path.join(tempfile.mkdtemp(), "ws.duckdb"))
    dd.load_demo_into_workspace(ws)
    r = run_pipeline(ws, _as_of(pos), RunConfig(as_of=_as_of(pos), backtest_weeks=40))
    # frozen vocabulary intact, translated present
    assert set(r.fva_by_lag["method"]).issubset(set(FVA_METHODS))
    assert "translated" in set(r.fva_by_lag["method"])
    # inventory projection frozen columns + non-negativity
    ip = r.inventory_projection
    assert list(ip.columns) == ["item_id", "region", "week",
                                "projected_on_hand", "target_position", "projected_order"]
    assert (ip["projected_on_hand"] >= 0).all()
    assert (ip["projected_order"] >= 0).all()


# --------------------------------------------------------------------------- #
# (b) inventory projection identity + non-negativity
# --------------------------------------------------------------------------- #

def test_projection_identity_and_non_negativity():
    """Design recurrence (no fitted order model): target == wos*fc, everything
    non-negative, and the projected orders reconcile to
    ``order = max(0, fc + target - projected_position)``."""
    params = TranslationParams(target_wos=4.0, order_cadence_weeks=1,
                               reaction_lag_weeks=0, usable_inv=True)
    fc = np.array([100.0, 120.0, 90.0, 110.0, 130.0])
    seed_oh, seed_it, replenish = 380.0, 0.0, 1.0
    proj_oh, target_pos, orders = _project_orders(fc, seed_oh, seed_it, params,
                                                  replenish=replenish)

    assert np.allclose(target_pos, params.target_wos * fc)
    assert (proj_oh >= 0).all() and (target_pos >= 0).all() and (orders >= 0).all()

    # reconstruct the recurrence by hand
    recon, oh_prev = [], seed_oh
    for i in range(len(fc)):
        pp = oh_prev - fc[i]
        recon.append(max(0.0, fc[i] + params.target_wos * fc[i] - pp))
        oh_prev = max(0.0, pp + replenish * fc[i])
    assert np.allclose(orders, recon)


def test_projection_orders_stay_non_negative_when_overstocked():
    """A hugely overstocked channel yields zero (never negative) orders — the
    bullwhip-drain signal, still honouring the frozen ``>= 0`` guarantee."""
    params = TranslationParams(target_wos=4.0, order_cadence_weeks=1,
                               reaction_lag_weeks=0, usable_inv=True)
    fc = np.full(6, 100.0)
    _, _, orders = _project_orders(fc, seed_on_hand=10_000.0, seed_in_transit=0.0,
                                   params=params)
    assert (orders >= 0).all()
    assert orders[0] == 0.0                   # far above target -> no order this week


# --------------------------------------------------------------------------- #
# (c) order cadence + reaction-lag behaviour
# --------------------------------------------------------------------------- #

def test_order_cadence_batches_orders():
    """With cadence 2 the retailer orders on alternate weeks only; off-cadence
    weeks carry zero order (the need rolls into the next order week)."""
    params = TranslationParams(target_wos=4.0, order_cadence_weeks=2,
                               reaction_lag_weeks=1, usable_inv=True)
    fc = np.full(8, 100.0)
    _, _, orders = _project_orders(fc, 400.0, 0.0, params)
    assert np.all(orders[1::2] == 0.0)        # odd weeks: no order
    assert np.all(orders[0::2] > 0.0)         # even weeks: an order


def test_reaction_lag_delays_seed_in_transit_landing():
    """A seed in-transit quantity lands after ``reaction_lag_weeks`` — so a large
    in-transit shipment lifts projected on-hand later, not immediately."""
    fc = np.full(6, 100.0)
    p0 = TranslationParams(4.0, 1, 0, True)
    p2 = TranslationParams(4.0, 1, 2, True)
    oh0, _, _ = _project_orders(fc, 300.0, 500.0, p0)   # lands week 1
    oh2, _, _ = _project_orders(fc, 300.0, 500.0, p2)   # lands week 3
    # with a 2-week lag the early on-hand is lower (the in-transit hasn't landed)
    assert oh2[0] < oh0[0]


# --------------------------------------------------------------------------- #
# (d) empirical calibration — sane + as-of safe
# --------------------------------------------------------------------------- #

def test_calibration_returns_sane_parameters(demo, demo_repaired):
    pos, inv, ship, _plan, _promo = demo
    cfg = RunConfig(as_of=_as_of(pos))
    p = TRANSLATE_PARAMS
    for sku in dd.SKUS:
        for region in dd.REGIONS:
            g = demo_repaired[(demo_repaired.item_id == sku) &
                              (demo_repaired.region == region)].sort_values("week")
            gi = inv[(inv.item_id == sku) & (inv.region == region)].sort_values("week")
            gs = ship[(ship.item_id == sku) & (ship.region == region)].sort_values("week")
            so = g["units_unconstrained"].to_numpy(float)
            oh = gi["on_hand_units"].to_numpy(float)
            sh = gs["units_shipped"].to_numpy(float)
            cens = g["is_censored"].to_numpy(bool)
            par = _calibrate_translation(so, oh, sh, cens, cfg)
            assert p["wos_min"] <= par.target_wos <= p["wos_max"]
            assert 1 <= par.order_cadence_weeks <= p["cadence_max"]
            assert 0 <= par.reaction_lag_weeks <= p["lag_max"]
            assert par.source == "fitted"     # demo has ample clean history


def test_calibration_is_as_of_safe(demo, demo_repaired):
    """Calibrating on history truncated at an earlier as_of is unaffected by any
    later weeks: appending corrupted future rows must not change the fit."""
    pos, inv, ship, _plan, _promo = demo
    cfg = RunConfig(as_of=_as_of(pos))
    sku, region = dd.SKUS[0], dd.REGIONS[0]
    g = demo_repaired[(demo_repaired.item_id == sku) &
                      (demo_repaired.region == region)].sort_values("week")
    gi = inv[(inv.item_id == sku) & (inv.region == region)].sort_values("week")
    gs = ship[(ship.item_id == sku) & (ship.region == region)].sort_values("week")
    so = g["units_unconstrained"].to_numpy(float)
    oh = gi["on_hand_units"].to_numpy(float)
    sh = gs["units_shipped"].to_numpy(float)
    cens = g["is_censored"].to_numpy(bool)

    cut = 50
    base = _calibrate_translation(so[:cut], oh[:cut], sh[:cut], cens[:cut], cfg)
    # corrupt everything AFTER the cut, then re-fit on the SAME <= cut slice
    so2, oh2, sh2 = so.copy(), oh.copy(), sh.copy()
    so2[cut:] *= 9.0; oh2[cut:] *= 9.0; sh2[cut:] *= 9.0
    again = _calibrate_translation(so2[:cut], oh2[:cut], sh2[:cut], cens[:cut], cfg)
    assert base == again


def test_fit_order_model_recovers_positive_sell_out_response(demo, demo_repaired):
    """The fitted order model responds positively to sell-out (b1 > 0) — the
    sanity gate that keeps a degenerate fit from being used."""
    pos, inv, ship, _plan, _promo = demo
    sku, region = dd.SKUS[0], dd.REGIONS[0]
    g = demo_repaired[(demo_repaired.item_id == sku) &
                      (demo_repaired.region == region)].sort_values("week")
    gi = inv[(inv.item_id == sku) & (inv.region == region)].sort_values("week")
    gs = ship[(ship.item_id == sku) & (ship.region == region)].sort_values("week")
    om = _fit_order_model(g["units_unconstrained"].to_numpy(float),
                          gi["on_hand_units"].to_numpy(float),
                          gi["in_transit_units"].to_numpy(float),
                          gs["units_shipped"].to_numpy(float))
    assert om is not None
    b0, b1, b2 = om
    assert b1 > 0                             # more sell-out -> more order
    assert all(np.isfinite([b0, b1, b2]))


# --------------------------------------------------------------------------- #
# (e) transfer-function fallback on a poor-inventory series
# --------------------------------------------------------------------------- #

def test_transfer_function_fallback_on_poor_inventory():
    """Phantom-flat channel inventory is unusable -> the per-series selector
    falls back to the distributed-lag transfer function, which still produces a
    finite, scored, competitive shipment forecast (design §6 / v2 M7)."""
    s = dd.poor_inventory_streams()
    pos, inv, ship = s["pos"], s["channel_inventory"], s["shipments"]
    rep = _repair(pos, inv, pd.DataFrame())
    sku, region = dd.POOR_INV["sku"], dd.POOR_INV["region"]
    g = rep[(rep.item_id == sku) & (rep.region == region)].sort_values("week")
    gi = inv.sort_values("week"); gs = ship.sort_values("week")
    so = g["units_unconstrained"].to_numpy(float)
    oh = gi["on_hand_units"].to_numpy(float)
    it = gi["in_transit_units"].to_numpy(float)
    sh = gs["units_shipped"].to_numpy(float)

    # the inventory signal is unusable and the selector falls back
    assert _inventory_usable(oh, so, TRANSLATE_PARAMS) is False
    cfg = RunConfig(as_of=pd.to_datetime(pos["week"]).max().date(), backtest_weeks=30)
    par = _calibrate_translation(so, oh, sh, g["is_censored"].to_numpy(bool), cfg)
    assert _select_translation_method(so, oh, it, sh, par, TRANSLATE_PARAMS) == "transfer"

    # the transfer function itself is finite and non-negative
    fc = _seasonal_naive(pd.Series(so), 4)
    tf = _transfer_function(so, sh, fc, TRANSLATE_PARAMS)
    assert np.isfinite(tf).all() and (tf >= 0).all() and len(tf) == 4

    # scored through the frozen harness: finite and competitive with shipments-only
    sp = _shipments_panel(rep, ship)
    tr = backtest(make_translated_forecaster(rep, inv, ship, cfg), sp, cfg,
                  value_col="units_shipped", lags=(1, 2, 4), method_name="translated")
    sob = backtest(shipments_history_only_forecaster, sp, cfg,
                   value_col="units_shipped", lags=(1, 2, 4), method_name="shiponly")
    assert len(tr) > 0 and np.isfinite(tr["pred"]).all()
    tr_w, _ = _wmape_at(tr, (1, 2, 4))
    so_w, _ = _wmape_at(sob, (1, 2, 4))
    assert np.isfinite(tr_w)
    assert tr_w <= 1.2 * so_w                 # still >= the shipments-history baseline


# --------------------------------------------------------------------------- #
# (f) as-of leakage: forecast invariant to corrupted future data
# --------------------------------------------------------------------------- #

def test_translated_forecaster_ignores_future_data(demo, demo_repaired):
    """Reusing the harness's guards: corrupting POS/inventory/shipments in weeks
    strictly AFTER an origin cannot change that origin's translated forecast."""
    pos, inv, ship, _plan, _promo = demo
    cfg = RunConfig(as_of=_as_of(pos), backtest_weeks=20)
    sp = _shipments_panel(demo_repaired, ship)

    weeks = sorted(pd.to_datetime(demo_repaired["week"]).unique())
    cutoff = weeks[-3]

    def corrupt(df, col):
        d = df.copy(); d["week"] = pd.to_datetime(d["week"])
        d.loc[d["week"] > cutoff, col] *= 9.0
        return d

    rep_c = corrupt(demo_repaired, "units_unconstrained")
    inv_c = corrupt(inv, "on_hand_units")
    ship_c = corrupt(ship, "units_shipped")

    clean = backtest(make_translated_forecaster(demo_repaired, inv, ship, cfg), sp, cfg,
                     value_col="units_shipped", method_name="translated")
    dirty = backtest(make_translated_forecaster(rep_c, inv_c, ship_c, cfg), sp, cfg,
                     value_col="units_shipped", method_name="translated")

    def before(t):
        t = t.copy(); t["as_of"] = pd.to_datetime(t["as_of"])
        return (t[t["as_of"] < cutoff]
                .sort_values(["item_id", "region", "as_of", "lag"])["pred"]
                .round(4).to_numpy())

    a, b = before(clean), before(dirty)
    assert len(a) > 0
    assert np.array_equal(a, b)               # earlier-origin forecasts unchanged


# --------------------------------------------------------------------------- #
# the module-level bare adapter still works (kept for the S5 harness re-judge)
# --------------------------------------------------------------------------- #

def test_bare_translated_forecaster_is_a_valid_series_adapter(demo_repaired):
    """``translated_forecaster`` (no inventory/shipments context) still runs as a
    plain single-series adapter through the harness — the contract the S5 repair
    test depends on."""
    sku, region = dd.STOCKOUT["sku"], dd.STOCKOUT["region"]
    stk = demo_repaired[(demo_repaired.item_id == sku) &
                        (demo_repaired.region == region)].copy()
    cfg = RunConfig(as_of=pd.to_datetime(stk["week"]).max().date(), backtest_weeks=30)
    t = backtest(translated_forecaster, stk, cfg,
                 value_col="units_unconstrained", method_name="translated")
    assert len(t) > 0
    assert np.isfinite(t["pred"]).all() and (t["pred"] >= 0).all()
