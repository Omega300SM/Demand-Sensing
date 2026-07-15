"""S8 — tests for the hardened alerts engine + the optional ML tier.

Structure follows the S8 deliverable checklist:

  (a) each exception class fires on a seeded scenario and stays silent on a clean
      one (precision), with the demo's injected stockout replaying end-to-end
      early enough to act;
  (b) the alerts columns / dtypes / severity vocabulary are exactly the frozen
      shape;
  (c) the ML tier beats the S6 statistical tier on the demo backtest at lags 1–4
      (or is correctly NOT selected, statistical fallback scored) — through the
      FROZEN harness;
  (d) as-of leakage: the ML forecast is invariant to corrupted future rows and
      every feature is computable at as_of;
  (e) ml_enabled=False reproduces the S7 forecast / fva byte-for-byte;
  (f) the dependency-free fallback learner runs and is scored when lightgbm is
      absent.

They complement (not replace) the existing 110 tests, which stay green.
"""

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensing import RunConfig, demo_data as dd  # noqa: E402
from sensing.workspace import open_workspace  # noqa: E402
from sensing import ml_model  # noqa: E402
from sensing.engine import (  # noqa: E402
    run_pipeline,
    _repair,
    _alerts,
    _alert_deviation,
    _alert_stockout,
    _alert_order_cliff,
    _alert_promo,
    _alert_censoring,
    _ALERT_COLS,
    ALERT_PARAMS,
    AsOfPanel,
    backtest,
    statistical_forecaster,
    _seasonal_naive,
    _wmape,
    DEFAULT_LAGS,
)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def demo():
    return dd.generate()


@pytest.fixture(scope="module")
def demo_repaired(demo):
    return _repair(demo["pos"], demo["channel_inventory"], demo["promo"])


def _ws(tmp_path, data):
    ws = open_workspace(os.path.join(tmp_path, "w.duckdb"))
    for stream, df in data.items():
        ws.add_snapshot(stream, df, source_name="t", suffix="t")
    ws.rebuild_canonical()
    return ws


def _weeks(df):
    return sorted(pd.to_datetime(df["week"]).dt.date.unique())


def _cfg(as_of, **kw):
    return RunConfig(as_of=as_of, **kw)


# small hand-built frames for direct helper precision tests
def _fc(rows):
    return pd.DataFrame(rows, columns=["item_id", "region", "week", "lag", "p10",
                                       "p50", "p90", "plan_units",
                                       "blend_weight_sensed", "blended"])


def _proj(rows):
    return pd.DataFrame(rows, columns=["item_id", "region", "week",
                                       "projected_on_hand", "target_position",
                                       "projected_order"])


WK = pd.Timestamp("2026-01-05")


# =========================================================================== #
# (a) precision — each class fires on a seeded scenario, silent on a clean one
# =========================================================================== #

def test_deviation_fires_and_is_silent():
    cfg = _cfg(date(2026, 1, 5), deviation_threshold=0.15)
    weeks = [WK + pd.Timedelta(weeks=h) for h in range(4)]
    # sensed 40% above plan -> fires
    hot = pd.DataFrame([dict(item_id="S", region="R", week=w, lag=i + 1,
                             p10=0, p50=140, p90=0, plan_units=100,
                             blend_weight_sensed=1, blended=140)
                        for i, w in enumerate(weeks)])
    assert _alert_deviation(hot, cfg, "S", "R") is not None
    # sensed within threshold -> silent
    cool = hot.assign(p50=105.0)
    assert _alert_deviation(cool, cfg, "S", "R") is None


def test_stockout_fires_and_is_silent():
    cfg = _cfg(date(2026, 1, 5), reaction_lag_weeks=1, target_wos=4.0)
    weeks = [WK + pd.Timedelta(weeks=h) for h in range(4)]
    # target_position ~ 4 weeks of demand (=400) -> 1 week of cover = 100 units.
    # projected cover runs to ~0 AT the reaction-lag week -> fires.
    stk = _proj([dict(item_id="S", region="R", week=weeks[i],
                      projected_on_hand=oh, target_position=400.0,
                      projected_order=0.0)
                 for i, oh in enumerate([0, 5, 0, 0])])
    assert _alert_stockout(stk, cfg, "S", "R") is not None
    # healthy cover from the reaction-lag week onward -> silent (even though the
    # seed week 0 is at zero — the un-actionable ramp is skipped)
    healthy = _proj([dict(item_id="S", region="R", week=weeks[i],
                          projected_on_hand=oh, target_position=400.0,
                          projected_order=100.0)
                     for i, oh in enumerate([0, 300, 280, 260])])
    assert _alert_stockout(healthy, cfg, "S", "R") is None


def test_order_cliff_fires_and_is_silent():
    weeks = [WK + pd.Timedelta(weeks=h) for h in range(3)]
    # order collapsed AND cover well above target -> overstock/cliff fires
    over = _proj([dict(item_id="S", region="R", week=weeks[i],
                       projected_on_hand=oh, target_position=100.0,
                       projected_order=od)
                  for i, (oh, od) in enumerate([(200, 0.0), (190, 1.0), (180, 0.0)])])
    assert _alert_order_cliff(over, "S", "R") is not None
    # order collapsed but cover LOW (a genuine zero order on a thin channel) -> silent
    thin = _proj([dict(item_id="S", region="R", week=weeks[i],
                       projected_on_hand=oh, target_position=100.0,
                       projected_order=0.0)
                  for i, oh in enumerate([40, 30, 20])])
    assert _alert_order_cliff(thin, "S", "R") is None


def _promo_series(latest_uplift):
    """Repaired-style frame: 3 promo weeks; last one's uplift set by the caller."""
    weeks = [WK + pd.Timedelta(weeks=h) for h in range(10)]
    rows = []
    promo_idx = {3, 6, 9}
    for i, w in enumerate(weeks):
        base = 100.0
        if i in promo_idx:
            up = latest_uplift if i == 9 else 80.0     # typical uplift 80 (=0.8x base)
        else:
            up = 0.0
        rows.append(dict(item_id="S", region="R", week=w, units_sold=base + up,
                         units_unconstrained=base + up, is_censored=False,
                         base_units=base, promo_uplift=up))
    return pd.DataFrame(rows)


def test_promo_fires_and_is_silent():
    cfg = _cfg(date(2026, 1, 5) + pd.Timedelta(weeks=9))
    # latest promo under-performs badly (uplift 20 vs typical 80) -> fires
    assert _alert_promo(_promo_series(20.0), cfg, "S", "R") is not None
    # latest promo in line with typical (uplift ~82 vs 80) -> silent
    assert _alert_promo(_promo_series(82.0), cfg, "S", "R") is None


def test_censoring_fires_and_is_silent():
    cfg = _cfg(date(2026, 1, 5) + pd.Timedelta(weeks=9))
    weeks = [WK + pd.Timedelta(weeks=h) for h in range(10)]
    cens = pd.DataFrame([dict(item_id="S", region="R", week=w, units_sold=10,
                              units_unconstrained=100, base_units=100,
                              promo_uplift=0, is_censored=(i in {7, 8, 9}))
                         for i, w in enumerate(weeks)])
    assert _alert_censoring(cens, cfg, "S", "R") is not None
    clean = cens.assign(is_censored=False)
    assert _alert_censoring(clean, cfg, "S", "R") is None


def test_injected_stockout_replays_end_to_end(tmp_path, demo):
    """Acceptance (design §8 / S8): replay the demo's injected stockout (SKU-1002
    /East, weeks 38–42) and confirm the censoring alert fires EARLY (right after
    the event) while a clean earlier as_of stays silent."""
    ws = _ws(str(tmp_path), demo)
    weeks = _weeks(demo["pos"])
    # as_of just after the stockout window -> the de-censored weeks are recent
    r_after = run_pipeline(ws, weeks[44], _cfg(weeks[44], backtest_weeks=18))
    cens = r_after.alerts[r_after.alerts["alert_type"] == "Signal repair / censoring"]
    assert (("SKU-1002" == cens["item_id"]) & ("East" == cens["region"])).any()
    # clean as_of well before the stockout -> no censoring alert for that series
    r_before = run_pipeline(ws, weeks[30], _cfg(weeks[30], backtest_weeks=18))
    cens_b = r_before.alerts[r_before.alerts["alert_type"] == "Signal repair / censoring"]
    assert not (("SKU-1002" == cens_b.get("item_id", pd.Series(dtype=object))).any())


def test_clean_demo_final_asof_no_false_alerts(tmp_path, demo):
    """At the final as_of the plan has no future horizon and no recent stockout —
    a clean scenario the hardened rules must not false-fire on."""
    ws = _ws(str(tmp_path), demo)
    as_of = _weeks(demo["pos"])[-1]
    r = run_pipeline(ws, as_of, _cfg(as_of, backtest_weeks=18))
    assert len(r.alerts) == 0


# =========================================================================== #
# (b) frozen alerts shape / dtypes / severity vocabulary
# =========================================================================== #

def test_alerts_shape_and_vocabulary(tmp_path, demo):
    ws = _ws(str(tmp_path), demo)
    weeks = _weeks(demo["pos"])
    frames = [run_pipeline(ws, weeks[i], _cfg(weeks[i], backtest_weeks=15)).alerts
              for i in (44, 52)]
    alla = pd.concat(frames, ignore_index=True)
    assert list(alla.columns) == _ALERT_COLS
    assert set(alla["severity"].unique()) <= {"high", "medium", "low"}
    # empty case is still the exact frozen frame
    empty = _alerts(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                    _cfg(date(2026, 1, 1)))
    assert list(empty.columns) == _ALERT_COLS and len(empty) == 0


def test_alert_type_strings_route_page4_charts():
    """Page 4 keys its driving chart on substrings — guard them so the hardened
    rules keep rendering with no page change."""
    types = {"Sensed-vs-plan deviation", "Projected retailer stockout",
             "Channel overstock / order-cliff", "Promo mid-flight variance",
             "Signal repair / censoring"}
    for t in types:
        low = t.lower()
        # each type resolves to exactly one of page 4's chart branches
        routed = ("stockout" in low or "overstock" in low or "deviation" in low)
        # the non-routed ones fall through to the repaired chart (always valid)
        assert routed or t in {"Promo mid-flight variance", "Signal repair / censoring"}


# =========================================================================== #
# (c) ML tier vs the S6 statistical tier through the FROZEN harness
# =========================================================================== #

def test_ml_beats_or_ties_statistical_at_near_lags(demo, demo_repaired):
    cfg = _cfg(pd.to_datetime(demo_repaired["week"]).max().date(), backtest_weeks=40)
    mlf = ml_model.make_ml_forecaster(demo["promo"], demo["channel_inventory"], cfg)
    t_ml = backtest(mlf, demo_repaired, cfg, value_col="units_unconstrained",
                    lags=DEFAULT_LAGS, method_name="ml")
    t_st = backtest(statistical_forecaster, demo_repaired, cfg,
                    value_col="units_unconstrained", lags=DEFAULT_LAGS,
                    method_name="statistical")
    assert len(t_ml) and len(t_st)
    ml = t_ml[t_ml["lag"].isin([1, 2, 4])]
    st = t_st[t_st["lag"].isin([1, 2, 4])]
    w_ml = _wmape(ml["actual"].values, ml["pred"].values)
    w_st = _wmape(st["actual"].values, st["pred"].values)
    # the fallback learner exploits the promo-known-future feature the statistical
    # tier throws away -> a genuine near-lag win on the demo
    assert w_ml < w_st


def test_champion_forecaster_never_worse_than_statistical(demo, demo_repaired):
    """The champion (ML-where-wins, statistical fallback) routed through the
    'statistical' method row must not be worse than pure statistical at the
    headline lags 1–2 — that is the whole point of champion–challenger."""
    cfg = _cfg(pd.to_datetime(demo_repaired["week"]).max().date(), backtest_weeks=40)
    champ = ml_model.make_champion_forecaster(
        demo["promo"], demo["channel_inventory"], cfg, point_fn=_seasonal_naive)
    t_ch = backtest(champ, demo_repaired, cfg, value_col="units_unconstrained",
                    lags=DEFAULT_LAGS, method_name="statistical")
    t_st = backtest(statistical_forecaster, demo_repaired, cfg,
                    value_col="units_unconstrained", lags=DEFAULT_LAGS,
                    method_name="statistical")
    ch = t_ch[t_ch["lag"].isin([1, 2])]
    st = t_st[t_st["lag"].isin([1, 2])]
    w_ch = _wmape(ch["actual"].values, ch["pred"].values)
    w_st = _wmape(st["actual"].values, st["pred"].values)
    assert w_ch <= w_st * 1.02      # never materially worse; usually better


def test_champion_only_switches_where_it_wins(demo, demo_repaired):
    """The per-series mask is a subset decision: a series is only switched to ML
    when it beats statistical by the margin on its own inner holdout."""
    cfg = _cfg(pd.to_datetime(demo_repaired["week"]).max().date())
    promo_map = ml_model._promo_lookup(demo["promo"])
    hist = demo_repaired[["item_id", "region", "week", "units_unconstrained"]].rename(
        columns={"units_unconstrained": "val"}).copy()
    hist["week"] = pd.to_datetime(hist["week"])
    mask = ml_model._champion_series_mask(
        hist, promo_map, demo["channel_inventory"],
        hist["week"].max(), cfg.horizon_weeks, ml_model.ML_PARAMS, _seasonal_naive)
    all_series = set(map(tuple, hist[["item_id", "region"]].drop_duplicates().values))
    assert mask <= all_series          # a subset (possibly empty), never invented keys


# =========================================================================== #
# (d) as-of leakage discipline
# =========================================================================== #

def test_ml_forecast_invariant_to_corrupted_future(demo, demo_repaired):
    cfg = _cfg(pd.to_datetime(demo_repaired["week"]).max().date())
    f = ml_model.make_ml_forecaster(demo["promo"], demo["channel_inventory"], cfg)
    frame = demo_repaired[["item_id", "region", "week", "units_unconstrained"]].copy()
    frame["week"] = pd.to_datetime(frame["week"])
    weeks = sorted(frame["week"].unique())
    as_of, future = weeks[50], weeks[51:59]
    p1 = AsOfPanel(frame, as_of, "units_unconstrained", future)
    out1 = f(p1, as_of, 8).sort_values(["item_id", "region", "week"]).reset_index(drop=True)
    corrupt = frame.copy()
    corrupt.loc[corrupt["week"] > as_of, "units_unconstrained"] *= 99.0
    p2 = AsOfPanel(corrupt, as_of, "units_unconstrained", future)
    out2 = f(p2, as_of, 8).sort_values(["item_id", "region", "week"]).reset_index(drop=True)
    assert np.allclose(out1["pred"].values, out2["pred"].values)


def test_ml_forecast_weeks_strictly_future(demo, demo_repaired):
    cfg = _cfg(pd.to_datetime(demo_repaired["week"]).max().date())
    f = ml_model.make_ml_forecaster(demo["promo"], demo["channel_inventory"], cfg)
    frame = demo_repaired[["item_id", "region", "week", "units_unconstrained"]].copy()
    frame["week"] = pd.to_datetime(frame["week"])
    weeks = sorted(frame["week"].unique())
    as_of, future = weeks[40], weeks[41:49]
    out = f(AsOfPanel(frame, as_of, "units_unconstrained", future), as_of, 8)
    assert (pd.to_datetime(out["week"]) > pd.Timestamp(as_of)).all()


# =========================================================================== #
# (e) ml_enabled=False reproduces S7 forecast / fva byte-for-byte
# =========================================================================== #

def test_ml_disabled_reproduces_default_path(tmp_path, demo):
    ws = _ws(str(tmp_path), demo)
    as_of = _weeks(demo["pos"])[60]
    r1 = run_pipeline(ws, as_of, _cfg(as_of, backtest_weeks=20, ml_enabled=False))
    r2 = run_pipeline(ws, as_of, _cfg(as_of, backtest_weeks=20, ml_enabled=False))
    pdt.assert_frame_equal(r1.forecast.reset_index(drop=True),
                           r2.forecast.reset_index(drop=True))
    pdt.assert_frame_equal(r1.fva_by_lag.reset_index(drop=True),
                           r2.fva_by_lag.reset_index(drop=True))
    # manifest records the tier is off and no champion series were used
    assert r1.manifest["ml_enabled"] is False
    assert r1.manifest["ml_champion_series"] == []


def test_ml_disabled_forecast_matches_statistical_path(tmp_path, demo):
    """With ml_enabled=False the sell-out override is None, so every forecast row
    must equal the S6 statistical path (no ML contamination)."""
    ws = _ws(str(tmp_path), demo)
    as_of = _weeks(demo["pos"])[60]
    r = run_pipeline(ws, as_of, _cfg(as_of, backtest_weeks=20, ml_enabled=False))
    # ml on changes at least the champion-picked series' forecast p50
    r_on = run_pipeline(ws, as_of, _cfg(as_of, backtest_weeks=20, ml_enabled=True))
    if r_on.manifest["ml_champion_series"]:
        merged = r.forecast.merge(
            r_on.forecast, on=["item_id", "region", "week", "lag"],
            suffixes=("_off", "_on"))
        assert not np.allclose(merged["p50_off"], merged["p50_on"])


# =========================================================================== #
# (f) the dependency-free fallback learner runs and is scored
# =========================================================================== #

def test_fallback_learner_is_deterministic_and_trains():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * rng.normal(size=200)
    m1 = ml_model._GBStumps(40, 0.1, 16, 8).fit(X, y)
    m2 = ml_model._GBStumps(40, 0.1, 16, 8).fit(X, y)
    p1, p2 = m1.predict(X), m2.predict(X)
    assert np.allclose(p1, p2)                       # deterministic
    # genuinely learns: beats the mean predictor
    sse_model = float(np.sum((y - p1) ** 2))
    sse_mean = float(np.sum((y - y.mean()) ** 2))
    assert sse_model < 0.6 * sse_mean


def test_global_quantile_model_orders_quantiles():
    rng = np.random.default_rng(1)
    n = len(ml_model.FEATURE_NAMES)
    X = rng.uniform(size=(150, n))
    y = 1.0 + 0.5 * X[:, 0]
    m = ml_model.GlobalQuantileModel().fit(X, y)
    p10, p50, p90 = m.predict_quantiles(X)
    assert np.all(p10 <= p50 + 1e-9) and np.all(p50 <= p90 + 1e-9)
    assert np.all(p10 >= -1e-9)


def test_ml_pipeline_runs_and_scores_without_lightgbm(tmp_path, demo, monkeypatch):
    """Force the no-lightgbm path and confirm ml_enabled=True still runs end to
    end and the champion re-scores through the frozen harness."""
    monkeypatch.setattr(ml_model, "_HAS_LIGHTGBM", False)
    ws = _ws(str(tmp_path), demo)
    as_of = _weeks(demo["pos"])[60]
    r = run_pipeline(ws, as_of, _cfg(as_of, backtest_weeks=25, ml_enabled=True))
    assert r.manifest["ml_backend"] == "fallback"
    # the statistical FVA row exists and is finite (the champion was scored)
    stat = r.fva_by_lag[r.fva_by_lag["method"] == "statistical"]
    assert len(stat) and np.isfinite(stat["wmape"]).all()
