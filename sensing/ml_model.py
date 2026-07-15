"""ML tier (S8) — a single GLOBAL gradient-boosted model across all SKU-regions.

Design §5 ("ML global model", v2 M6): one model pooled across every SKU-region,
direct multi-horizon (the horizon step ``h`` is a feature), quantile output for
P10/P50/P90. It is an OPTIONAL tier, toggled by ``RunConfig.ml_enabled``; the
default S6 statistical tier is untouched when it is off.

Two hard rules make this tier honest rather than a leakage machine:

  1. **As-of discipline.** Every feature is computable at the origin week ``t``:
     sell-out features read only ``<= t`` history, the promo flag is *known-future*
     (a planned trade calendar — legitimately available at ``t`` for a future
     target week), and the channel-inventory position feature reads the latest
     ``<= t`` on-hand. The forecaster is judged THROUGH the frozen backtest
     harness exactly like ``statistical_forecaster`` (scored on
     ``units_unconstrained``), so a peeking feature is caught by the harness's
     ``LeakageError`` guards, not silently rewarded.

  2. **Champion–challenger.** The ML tier only *replaces* the statistical
     sell-out point path where it actually wins. ``make_champion_forecaster``
     decides ML-vs-statistical at each rolling origin using an inner holdout of
     the origin's own ``<= as_of`` history (an aggregate decision across series,
     documented) — never using the scoring actuals. Where ML loses, the
     statistical tier is the fallback.

Dependency posture mirrors ``statsforecast`` in ``engine.py``: ``lightgbm`` is
IMPORT-GUARDED (``_HAS_LIGHTGBM``). When the wheel is absent the tier falls back
to a **deterministic, dependency-free** gradient-boosted-stumps learner, so
``ml_enabled=True`` is fully testable in CI. The public surface is identical
either way.

Public surface (imported by ``engine.py``):
    _HAS_LIGHTGBM
    ML_PARAMS
    make_ml_forecaster(promo, inv, cfg, *, point_fn)        -> harness adapter (pure ML)
    make_champion_forecaster(promo, inv, cfg, *, point_fn)  -> harness adapter (ML-where-wins)
    sellout_quantiles(sellout, promo, inv, as_of, cfg)      -> (override_dict, used_ml_keys)
"""

from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Optional acceleration: LightGBM. Import-guarded exactly like statsforecast.
# --------------------------------------------------------------------------- #
try:                                     # pragma: no cover - optional dependency
    import lightgbm as _lgb              # type: ignore
    _HAS_LIGHTGBM = True
except Exception:                        # pragma: no cover
    _lgb = None
    _HAS_LIGHTGBM = False


# Tunable ML-tier knobs (documented; referenced by tests + the handoff).
ML_PARAMS: dict[str, float] = {
    "min_hist": 16,          # weeks of history before a series contributes rows
    "level_window": 13,      # robust-mean window used to scale the ratio target
    "val_weeks": 6,          # inner-holdout length for the champion decision
    "n_estimators": 60,      # boosting rounds (fallback learner)
    "learning_rate": 0.12,   # shrinkage (fallback learner)
    "max_bins": 24,          # candidate thresholds per feature (fallback learner)
    "min_leaf": 8,           # min rows either side of a stump split (fallback)
    "q_lo": 0.10,            # residual quantile for P10
    "q_hi": 0.90,            # residual quantile for P90
    "champion_lags": 4,      # decide ML-vs-stat on WMAPE at lags 1..this
    "champion_margin": 0.02, # ML must beat statistical by this (relative) to switch
}

# Order of the feature vector (documented so the handoff/tests can reason about it).
FEATURE_NAMES = [
    "r_last",        # y[t] / level
    "r_mean2", "r_mean4", "r_mean8",   # window means / level
    "r_std8",        # window std / level
    "r_slope4",      # 4-week slope / level
    "r_yoy",         # y[t-52] / level (1.0 if unavailable)
    "zero_frac8",    # fraction of zeros in the last 8 weeks
    "woy_sin", "woy_cos",   # week-of-year of the TARGET week
    "h",             # horizon step (direct multi-horizon)
    "promo_t",       # promo flag at the origin week
    "promo_tgt",     # promo flag at the TARGET week (known-future)
    "inv_ratio",     # on_hand[t] / trailing-mean(on_hand)  (1.0 if no inventory)
]


# --------------------------------------------------------------------------- #
# Deterministic, dependency-free fallback learner: gradient-boosted stumps.
# --------------------------------------------------------------------------- #

class _GBStumps:
    """A small, deterministic gradient-boosting regressor of depth-1 trees.

    Squared-error boosting; each round fits one axis-aligned stump on the current
    residual. No randomness, no external wheels — so it is reproducible in CI and
    a genuine (if modest) learner that can exploit the promo-known-future feature
    the statistical tier throws away when it de-spikes."""

    def __init__(self, n_estimators: int, learning_rate: float,
                 max_bins: int, min_leaf: int):
        self.n_estimators = int(n_estimators)
        self.lr = float(learning_rate)
        self.max_bins = int(max_bins)      # retained for API compatibility
        self.min_leaf = int(min_leaf)
        self.base_ = 0.0
        self.stumps_: list[tuple[int, float, float, float]] = []  # feat, thr, left, right

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_GBStumps":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.base_ = float(np.mean(y)) if n else 0.0
        if n == 0:
            return self
        pred = np.full(n, self.base_)
        # Sort order + sorted values per feature, computed ONCE. Each round then
        # finds the exact best split for every feature via prefix sums in O(n)
        # (variance-reduction ∝ L²/nL + R²/nR), which is what keeps the fallback
        # cheap enough to run twice per rolling origin in CI.
        order = np.argsort(X, axis=0)                      # (n, d)
        Xs = np.take_along_axis(X, order, axis=0)          # sorted feature values
        ml = self.min_leaf
        for _ in range(self.n_estimators):
            resid = y - pred
            best = None  # (gain, feat, thr, left, right)
            for j in range(d):
                oj = order[:, j]
                rs = resid[oj]
                csum = np.cumsum(rs)
                total = csum[-1]
                xs = Xs[:, j]
                # valid split after position k (0-based) requires a value change
                # between k and k+1 and min_leaf on both sides.
                lo, hi = ml - 1, n - ml - 1
                if hi < lo:
                    continue
                k = np.arange(lo, hi + 1)
                boundary = xs[k] < xs[k + 1]               # only real cut points
                if not boundary.any():
                    continue
                nl = (k + 1).astype(float)
                nr = (n - (k + 1)).astype(float)
                lsum = csum[k]
                rsum = total - lsum
                gain = np.where(boundary, lsum * lsum / nl + rsum * rsum / nr, -np.inf)
                bi = int(np.argmax(gain))
                if not np.isfinite(gain[bi]):
                    continue
                g = float(gain[bi])
                if best is None or g > best[0]:
                    kk = int(k[bi])
                    thr = float((xs[kk] + xs[kk + 1]) / 2.0)
                    left = float(lsum[bi] / nl[bi])
                    right = float(rsum[bi] / nr[bi])
                    best = (g, j, thr, left, right)
            if best is None:
                break
            _, j, thr, left, right = best
            self.stumps_.append((j, thr, left, right))
            pred = pred + self.lr * np.where(X[:, j] <= thr, left, right)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.base_)
        for j, thr, left, right in self.stumps_:
            out = out + self.lr * np.where(X[:, j] <= thr, left, right)
        return out


# --------------------------------------------------------------------------- #
# Global quantile model — a thin backend switch over LightGBM / the fallback.
# --------------------------------------------------------------------------- #

class GlobalQuantileModel:
    """Fit once on pooled (scale-normalised) ratio targets, predict P10/P50/P90.

    Targets are ``y[t+h] / level_t`` (scale-free) so a single global model pools
    cleanly across SKU-regions of very different volume; predictions are rescaled
    by the series level at prediction time. LightGBM path fits three quantile
    models; the fallback path fits one point model and widens an empirical
    residual band by ``sqrt(h)`` (matching the statistical tier's fan logic)."""

    def __init__(self, params: dict | None = None):
        self.p = dict(ML_PARAMS if params is None else params)
        self._backend = "lightgbm" if _HAS_LIGHTGBM else "fallback"
        self._models: dict = {}
        self._q_lo = 0.0
        self._q_hi = 0.0
        self._h_col = FEATURE_NAMES.index("h")

    def fit(self, X: np.ndarray, y_ratio: np.ndarray) -> "GlobalQuantileModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y_ratio, dtype=float)
        if _HAS_LIGHTGBM:                     # pragma: no cover - optional path
            for a in (self.p["q_lo"], 0.5, self.p["q_hi"]):
                m = _lgb.LGBMRegressor(
                    objective="quantile", alpha=float(a),
                    n_estimators=200, learning_rate=0.05,
                    num_leaves=31, min_child_samples=int(self.p["min_leaf"]),
                    verbosity=-1, deterministic=True, force_row_wise=True)
                m.fit(X, y)
                self._models[a] = m
        else:
            m = _GBStumps(self.p["n_estimators"], self.p["learning_rate"],
                          self.p["max_bins"], self.p["min_leaf"]).fit(X, y)
            self._models["point"] = m
            resid = y - m.predict(X)
            if len(resid) >= 5:
                self._q_lo = min(float(np.quantile(resid, self.p["q_lo"])), 0.0)
                self._q_hi = max(float(np.quantile(resid, self.p["q_hi"])), 0.0)
            else:
                sd = float(np.std(y)) if len(y) else 0.0
                self._q_lo, self._q_hi = -1.2816 * sd, 1.2816 * sd
        return self

    def predict_ratio(self, X: np.ndarray) -> np.ndarray:
        """Point (P50) ratio prediction."""
        X = np.asarray(X, dtype=float)
        if _HAS_LIGHTGBM:                     # pragma: no cover - optional path
            return np.asarray(self._models[0.5].predict(X), dtype=float)
        return self._models["point"].predict(X)

    def predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        h = X[:, self._h_col]
        p50 = self.predict_ratio(X)
        if _HAS_LIGHTGBM:                     # pragma: no cover - optional path
            p10 = np.asarray(self._models[self.p["q_lo"]].predict(X), dtype=float)
            p90 = np.asarray(self._models[self.p["q_hi"]].predict(X), dtype=float)
        else:
            grow = np.sqrt(np.maximum(h, 1.0))
            p10 = p50 + self._q_lo * grow
            p90 = p50 + self._q_hi * grow
        # enforce ordering + non-negativity on the ratio scale
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)
        p10 = np.maximum(p10, 0.0)
        return p10, p50, p90


# --------------------------------------------------------------------------- #
# Feature engineering (all as-of-safe)
# --------------------------------------------------------------------------- #

def _woy(ts: pd.Timestamp) -> tuple[float, float]:
    w = int(pd.Timestamp(ts).isocalendar().week)
    ang = 2 * np.pi * (w % 52) / 52.0
    return float(np.sin(ang)), float(np.cos(ang))


def _level(y: np.ndarray, window: int) -> float:
    tail = y[-window:] if len(y) >= 1 else y
    lvl = float(np.mean(tail)) if len(tail) else 0.0
    return lvl if lvl > 1e-9 else 1.0


def _feat_at(y: np.ndarray, origin_wk: pd.Timestamp, tgt_wk: pd.Timestamp, h: int,
             level: float, promo_t: float, promo_tgt: float, inv_ratio: float
             ) -> list[float]:
    """Feature row for a single (origin, target, horizon). ``y`` is the series'
    values up to AND INCLUDING the origin week (i.e. strictly ``<= origin_wk``)."""
    n = len(y)
    last = float(y[-1]) if n else 0.0

    def wmean(k):
        return float(np.mean(y[-k:])) if n else 0.0

    def wstd(k):
        return float(np.std(y[-k:])) if n >= 2 else 0.0

    slope = 0.0
    if n >= 4:
        w = min(4, n)
        slope = float(np.polyfit(np.arange(w), y[-w:], 1)[0])
    yoy = float(y[-52]) if n >= 52 else level
    zero_frac = float(np.mean(y[-8:] <= 1e-9)) if n else 0.0
    ws, wc = _woy(tgt_wk)
    return [
        last / level,
        wmean(2) / level, wmean(4) / level, wmean(8) / level,
        wstd(8) / level,
        slope / level,
        yoy / level,
        zero_frac,
        ws, wc,
        float(h),
        float(promo_t), float(promo_tgt),
        float(inv_ratio),
    ]


def _promo_lookup(promo: pd.DataFrame | None):
    if promo is None or promo.empty or "promo_flag" not in promo.columns:
        return {}
    p = promo[["item_id", "region", "week", "promo_flag"]].copy()
    p["week"] = pd.to_datetime(p["week"])
    return {(r.item_id, r.region, pd.Timestamp(r.week)): float(r.promo_flag)
            for r in p.itertuples(index=False)}


def _inv_ratio_series(inv: pd.DataFrame | None, sku: str, region: str,
                      as_of: pd.Timestamp) -> dict[pd.Timestamp, float]:
    """week -> on_hand / trailing-mean(on_hand), only for weeks <= as_of."""
    if inv is None or inv.empty:
        return {}
    g = inv[(inv["item_id"] == sku) & (inv["region"] == region)].copy()
    if g.empty:
        return {}
    g["week"] = pd.to_datetime(g["week"])
    g = g[g["week"] <= as_of].sort_values("week")
    if g.empty:
        return {}
    oh = g["on_hand_units"].astype(float).to_numpy()
    trail = pd.Series(oh).rolling(8, min_periods=1).mean().to_numpy()
    trail = np.where(trail > 1e-9, trail, 1.0)
    return {pd.Timestamp(w): float(o / t) for w, o, t in zip(g["week"], oh, trail)}


def _build_training(hist: pd.DataFrame, promo_map: dict, inv: pd.DataFrame | None,
                    as_of: pd.Timestamp, horizon: int, p: dict
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Pool training rows across all series. Every row's features are computable
    at its origin week; every target week is ``<= as_of`` (so actuals exist and
    nothing reads the future)."""
    X_rows, y_rows = [], []
    min_hist = int(p["min_hist"])
    lw = int(p["level_window"])
    for (sku, region), g in hist.groupby(["item_id", "region"]):
        g = g.sort_values("week")
        wk = list(pd.to_datetime(g["week"]))
        y = g["val"].to_numpy(dtype=float)
        n = len(y)
        if n < min_hist + 1:
            continue
        inv_r = _inv_ratio_series(inv, sku, region, as_of)
        for c in range(min_hist, n):            # c = # observed points; origin idx c-1
            oi = c - 1
            level = _level(y[:c], lw)
            promo_t = promo_map.get((sku, region, wk[oi]), 0.0)
            inv_ratio = inv_r.get(wk[oi], 1.0)
            for h in range(1, horizon + 1):
                ti = oi + h
                if ti >= n:
                    break
                tgt_wk = wk[ti]
                promo_tgt = promo_map.get((sku, region, tgt_wk), 0.0)
                X_rows.append(_feat_at(y[:c], wk[oi], tgt_wk, h, level,
                                       promo_t, promo_tgt, inv_ratio))
                y_rows.append(y[ti] / level)
    if not X_rows:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0)
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float)


def _predict_future(model: GlobalQuantileModel, hist: pd.DataFrame, promo_map: dict,
                    inv: pd.DataFrame | None, as_of: pd.Timestamp, horizon: int,
                    future_weeks: list[pd.Timestamp], p: dict, quantiles: bool
                    ):
    """Predict the future weeks for every series. Returns a dict keyed by
    (sku, region) -> point path (quantiles=False) or (p10, p50, p90)."""
    lw = int(p["level_window"])
    out: dict[tuple[str, str], object] = {}
    for (sku, region), g in hist.groupby(["item_id", "region"]):
        g = g.sort_values("week")
        wk = list(pd.to_datetime(g["week"]))
        y = g["val"].to_numpy(dtype=float)
        if len(y) == 0:
            continue
        oi = len(y) - 1
        level = _level(y, lw)
        promo_t = promo_map.get((sku, region, wk[oi]), 0.0)
        inv_r = _inv_ratio_series(inv, sku, region, as_of)
        inv_ratio = inv_r.get(wk[oi], 1.0)
        rows = []
        for h, tw in enumerate(future_weeks[:horizon], start=1):
            promo_tgt = promo_map.get((sku, region, pd.Timestamp(tw)), 0.0)
            rows.append(_feat_at(y, wk[oi], pd.Timestamp(tw), h, level,
                                 promo_t, promo_tgt, inv_ratio))
        X = np.asarray(rows, dtype=float)
        if quantiles:
            r10, r50, r90 = model.predict_quantiles(X)
            out[(sku, region)] = (np.maximum(r10 * level, 0.0),
                                  np.maximum(r50 * level, 0.0),
                                  np.maximum(r90 * level, 0.0))
        else:
            out[(sku, region)] = np.maximum(model.predict_ratio(X) * level, 0.0)
    return out


# --------------------------------------------------------------------------- #
# Harness adapters
# --------------------------------------------------------------------------- #

def _hist_from_panel(panel) -> pd.DataFrame:
    """Pull the ``<= as_of`` history from an AsOfPanel into the internal frame
    (columns item_id, region, week, val). Uses only the panel's own truncated
    view, so this is leakage-safe by construction."""
    h = panel.history().copy()
    vc = panel.value_col
    h = h.rename(columns={vc: "val"})
    h["week"] = pd.to_datetime(h["week"])
    return h[["item_id", "region", "week", "val"]]


def make_ml_forecaster(promo: pd.DataFrame | None, inv: pd.DataFrame | None,
                       cfg, *, params: dict | None = None) -> Callable:
    """PURE ML forecaster adapter for the frozen ``backtest`` harness.

    Fits the global model on the panel's ``<= as_of`` history at each origin and
    returns the ML point path. Judged on ``units_unconstrained`` exactly like
    ``statistical_forecaster`` — the harness leakage guards apply unchanged."""
    p = dict(ML_PARAMS if params is None else params)
    promo_map = _promo_lookup(promo)

    def _f(panel, as_of, horizon: int) -> pd.DataFrame:
        hist = _hist_from_panel(panel)
        X, y = _build_training(hist, promo_map, inv, pd.Timestamp(as_of), horizon, p)
        future = panel.future_weeks(horizon)
        rows = []
        if len(X) == 0:
            return pd.DataFrame(columns=["item_id", "region", "week", "pred"])
        model = GlobalQuantileModel(p).fit(X, y)
        preds = _predict_future(model, hist, promo_map, inv, pd.Timestamp(as_of),
                                horizon, future, p, quantiles=False)
        for (sku, region), path in preds.items():
            for wk, pv in zip(future, path):
                rows.append(dict(item_id=sku, region=region, week=wk, pred=float(pv)))
        return pd.DataFrame(rows)

    return _f


def _wmape_pool(av, fv) -> float:
    av, fv = np.asarray(av, float), np.asarray(fv, float)
    denom = np.sum(np.abs(av))
    return float(np.sum(np.abs(av - fv)) / denom) if denom > 0 else np.inf


def _champion_series_mask(hist: pd.DataFrame, promo_map: dict, inv,
                          as_of: pd.Timestamp, horizon: int, p: dict,
                          point_fn: Callable) -> set:
    """PER-SERIES ML-vs-statistical decision using an inner holdout of the
    ``<= as_of`` history ONLY.

    Split at ``val_weeks`` before the last observed week; fit ONE global ML on
    the earlier part; score ML and the statistical ``point_fn`` on the held-out
    tail (still ``<= as_of`` — never the scoring actuals) at lags
    1..``champion_lags``. A series joins the ML mask only if ML beats statistical
    by more than ``champion_margin`` (relative) on its own holdout — hysteresis
    so a noise-level tie keeps the robust statistical tier. Fully as-of-safe;
    returns the set of (sku, region) that should use the ML point path."""
    val = int(p["val_weeks"])
    clags = int(p["champion_lags"])
    margin = float(p.get("champion_margin", 0.02))
    weeks = sorted(pd.to_datetime(hist["week"].unique()))
    if len(weeks) < int(p["min_hist"]) + val + 1:
        return set()
    inner_cut = weeks[-(val + 1)]
    train = hist[hist["week"] <= inner_cut]
    X, y = _build_training(train, promo_map, inv, inner_cut, horizon, p)
    if len(X) == 0:
        return set()
    model = GlobalQuantileModel(p).fit(X, y)

    val_future = [w for w in weeks if w > inner_cut][:clags]
    if not val_future:
        return set()
    actual = {(r.item_id, r.region, pd.Timestamp(r.week)): float(r.val)
              for r in hist.itertuples(index=False)}
    ml_pred = _predict_future(model, train, promo_map, inv, inner_cut,
                              horizon, val_future, p, quantiles=False)

    winners: set = set()
    for (sku, region), g in train.groupby(["item_id", "region"]):
        g = g.sort_values("week")
        st_path = point_fn(pd.Series(g["val"].to_numpy(dtype=float)), len(val_future))
        ml_path = ml_pred.get((sku, region))
        if ml_path is None:
            continue
        ml_a, ml_f, st_a, st_f = [], [], [], []
        for i, wk in enumerate(val_future):
            a = actual.get((sku, region, pd.Timestamp(wk)))
            if a is None:
                continue
            ml_a.append(a); ml_f.append(float(ml_path[i]))
            st_a.append(a); st_f.append(float(st_path[i]))
        if not ml_a:
            continue
        ml_w, st_w = _wmape_pool(ml_a, ml_f), _wmape_pool(st_a, st_f)
        if np.isfinite(st_w) and ml_w < st_w * (1.0 - margin):
            winners.add((sku, region))
    return winners


def make_champion_forecaster(promo: pd.DataFrame | None, inv: pd.DataFrame | None,
                             cfg, *, point_fn: Callable,
                             params: dict | None = None) -> Callable:
    """Champion–challenger adapter: ML point path where it wins on an inner
    ``<= as_of`` holdout, else the statistical ``point_fn`` fallback. Routed
    through the harness's existing ``statistical`` method row when
    ``ml_enabled=True`` (no new FVA vocabulary). ``point_fn`` is the statistical
    tier's point path (``_seasonal_naive``), injected to avoid importing engine."""
    p = dict(ML_PARAMS if params is None else params)
    promo_map = _promo_lookup(promo)

    def _f(panel, as_of, horizon: int) -> pd.DataFrame:
        hist = _hist_from_panel(panel)
        future = panel.future_weeks(horizon)
        mask = _champion_series_mask(hist, promo_map, inv, pd.Timestamp(as_of),
                                     horizon, p, point_fn)
        ml_preds = {}
        if mask:
            X, y = _build_training(hist, promo_map, inv, pd.Timestamp(as_of), horizon, p)
            if len(X):
                model = GlobalQuantileModel(p).fit(X, y)
                ml_preds = _predict_future(model, hist, promo_map, inv,
                                           pd.Timestamp(as_of), horizon, future, p,
                                           quantiles=False)
        rows = []
        for (sku, region), g in hist.groupby(["item_id", "region"]):
            g = g.sort_values("week")
            if (sku, region) in mask and (sku, region) in ml_preds:
                path = ml_preds[(sku, region)]
            else:
                path = point_fn(pd.Series(g["val"].to_numpy(dtype=float)), horizon)
            for wk, pv in zip(future, path):
                rows.append(dict(item_id=sku, region=region, week=wk, pred=float(pv)))
        return pd.DataFrame(rows)

    return _f


# --------------------------------------------------------------------------- #
# Production path: per-series sell-out quantile override for _forecast_and_translate
# --------------------------------------------------------------------------- #

def sellout_quantiles(sellout: pd.DataFrame, promo: pd.DataFrame | None,
                      inv: pd.DataFrame | None, as_of: date, cfg, *,
                      point_fn: Callable, params: dict | None = None
                      ) -> tuple[dict, list]:
    """Build the sell-out P10/P50/P90 override the pipeline hands to
    ``_forecast_and_translate`` when ``ml_enabled=True``.

    ``sellout`` is the repaired frame (``item_id, region, week,
    units_unconstrained``). Runs the SAME aggregate champion decision as the
    harness adapter at the single production ``as_of`` (using only ``<= as_of``
    history). Returns ``(override, used_ml_keys)``: ``override`` maps
    (sku, region) -> (p10, p50, p90) ONLY where ML is the champion; series where
    statistical wins are absent, so the caller keeps S6's ``_statistical_forecast``
    for them. When ML never wins the override is empty and behaviour is identical
    to S7."""
    p = dict(ML_PARAMS if params is None else params)
    as_of_ts = pd.Timestamp(as_of)
    promo_map = _promo_lookup(promo)

    hist = sellout[["item_id", "region", "week", "units_unconstrained"]].copy()
    hist = hist.rename(columns={"units_unconstrained": "val"})
    hist["week"] = pd.to_datetime(hist["week"])
    hist = hist[hist["week"] <= as_of_ts]              # as-of guard
    if hist.empty:
        return {}, []

    horizon = int(cfg.horizon_weeks)
    mask = _champion_series_mask(hist, promo_map, inv, as_of_ts, horizon, p, point_fn)
    if not mask:
        return {}, []

    X, y = _build_training(hist, promo_map, inv, as_of_ts, horizon, p)
    if len(X) == 0:
        return {}, []
    model = GlobalQuantileModel(p).fit(X, y)

    override: dict[tuple[str, str], tuple] = {}
    used: list[tuple[str, str]] = []
    for (sku, region), g in hist.groupby(["item_id", "region"]):
        if (sku, region) not in mask:
            continue                                     # statistical fallback
        g = g.sort_values("week")
        last_wk = g["week"].max()
        future = [last_wk + pd.Timedelta(weeks=h) for h in range(1, horizon + 1)]
        pred = _predict_future(model, g.assign(item_id=sku, region=region),
                               promo_map, inv, as_of_ts, horizon, future, p,
                               quantiles=True)
        if (sku, region) in pred:
            override[(sku, region)] = pred[(sku, region)]
            used.append((sku, region))
    return override, used
