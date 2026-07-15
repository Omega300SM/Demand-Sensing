"""Accuracy / FVA — the POC's verdict.

Rolling-origin backtest over the uploaded history: the FVA waterfall
(naïve → statistical → translated → vs. lag-adjusted plan) and WMAPE / bias
by lag. This page either earns the v2 build or it doesn't.
"""

from datetime import date

from app_common import (bootstrap, get_workspace, page_header, require_canonical,
                        latest_run_id, run_manifest)

bootstrap()
import altair as alt
import pandas as pd
import streamlit as st
from sensing import RunConfig, run_pipeline

st.set_page_config(page_title="Accuracy / FVA", page_icon="📊", layout="wide")
page_header("Accuracy / FVA", "The POC's verdict: does sensing beat the lag-adjusted plan?")

ws = get_workspace()
if not require_canonical(ws, "pos"):
    st.stop()

run_id = st.session_state.get("last_run_id") or latest_run_id(ws)
if not run_id:
    st.warning("No runs yet — open **Run Sensing** first.")
    st.stop()

# --------------------------------------------------------------------------- #
# One-click rolling-origin recompute — reproducible from the run's manifest
# (config + data snapshot dates), so the FVA number can always be regenerated.
# --------------------------------------------------------------------------- #
top_l, top_r = st.columns([3, 1])
with top_r:
    if st.button("🔄 Recompute backtest", use_container_width=True,
                 help="Re-run the rolling-origin backtest from this run's manifest config."):
        man = run_manifest(ws, run_id)
        if man and man.get("config"):
            cfg = RunConfig.from_dict(man["config"])
            as_of = cfg.as_of
        else:  # fall back to the last complete POS week
            as_of = pd.to_datetime(ws.read_canonical("pos")["week"]).max().date()
            cfg = RunConfig(as_of=as_of)
        with st.spinner("Rolling through the holdout window…"):
            fresh = run_pipeline(ws, as_of, cfg)
        st.session_state["last_run_id"] = fresh.run_id
        st.rerun()

fva = ws.read_run_output(run_id, "fva")
by_lag = ws.read_run_output(run_id, "fva_by_lag")
with top_l:
    st.caption(f"Showing run `{run_id}` · the backtest is reproducible from its "
               "manifest (config + snapshot dates).")

if not len(fva):
    st.warning("Backtest produced no results — not enough history. Load more weeks.")
    st.stop()

# --------------------------------------------------------------------------- #
# Headline: exit-criteria check
# --------------------------------------------------------------------------- #
def _wmape_at(method, lags):
    d = by_lag[(by_lag["method"] == method) & (by_lag["lag"].isin(lags))]
    return d["wmape"].mean() if len(d) else float("nan")

sensed_short = _wmape_at("translated", [1, 2])
plan_short = _wmape_at("plan", [1, 2])
improvement = (plan_short - sensed_short) / plan_short if plan_short and plan_short == plan_short else float("nan")
bias_short = by_lag[(by_lag["method"] == "translated") & (by_lag["lag"].isin([1, 2]))]["bias"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sensed WMAPE (lag 1–2)", f"{sensed_short:.1%}" if sensed_short == sensed_short else "—")
m2.metric("Plan WMAPE (lag 1–2)", f"{plan_short:.1%}" if plan_short == plan_short else "—")
m3.metric("Relative improvement", f"{improvement:+.1%}" if improvement == improvement else "—")
m4.metric("Sensed bias (lag 1–2)", f"{bias_short:+.1%}" if bias_short == bias_short else "—")

target_met = (improvement == improvement and improvement >= 0.15 and
              abs(bias_short) <= 0.05)
if target_met:
    st.success("✅ Exit criterion met on this run: ≥15% WMAPE improvement vs. plan "
               "at 1–2 week lags with bias within ±5%.")
else:
    st.info("Exit bar is ≥15–20% WMAPE improvement vs. the lag-adjusted plan at "
            "1–2 week lags with bias within ±5%. The stub engine won't clear it — "
            "the FVA waterfall below shows exactly which layer must improve.")

st.divider()

# --------------------------------------------------------------------------- #
# FVA waterfall
# --------------------------------------------------------------------------- #
st.markdown("#### FVA waterfall (lag 1–2 weeks)")
wf = fva.copy()
wf["wmape_pct"] = (wf["wmape"] * 100).round(1)
chart = alt.Chart(wf).mark_bar().encode(
    x=alt.X("step:N", sort=list(wf["step"]), title=None),
    y=alt.Y("wmape_pct:Q", title="WMAPE (%)"),
    color=alt.Color("step:N", legend=None),
    tooltip=["step", "wmape_pct", alt.Tooltip("bias:Q", format=".1%")],
).properties(height=320)
labels = alt.Chart(wf).mark_text(dy=-8).encode(
    x=alt.X("step:N", sort=list(wf["step"])), y="wmape_pct:Q",
    text=alt.Text("wmape_pct:Q", format=".1f"))
st.altair_chart(chart + labels, use_container_width=True)
st.caption("Each step should add value (lower WMAPE) or be removed. Compared against "
           "the lag-adjusted demand plan on the right-most bar.")

# --------------------------------------------------------------------------- #
# WMAPE + bias by lag
# --------------------------------------------------------------------------- #
st.markdown("#### WMAPE by lag")
method_names = {"naive": "Seasonal naïve", "statistical": "Statistical",
                "translated": "Translated sell-in", "plan": "Demand plan"}
bl = by_lag.copy()
bl["method_name"] = bl["method"].map(method_names).fillna(bl["method"])
wmape_chart = alt.Chart(bl).mark_line(point=True).encode(
    x=alt.X("lag:O", title="Forecast lag (weeks)"),
    y=alt.Y("wmape:Q", title="WMAPE", axis=alt.Axis(format="%")),
    color=alt.Color("method_name:N", title="Method"),
).properties(height=300)
st.altair_chart(wmape_chart, use_container_width=True)

st.markdown("#### Bias by lag")
bias_chart = alt.Chart(bl).mark_line(point=True).encode(
    x=alt.X("lag:O", title="Forecast lag (weeks)"),
    y=alt.Y("bias:Q", title="Weighted bias", axis=alt.Axis(format="%")),
    color=alt.Color("method_name:N", title="Method"),
).properties(height=260)
band = alt.Chart(pd.DataFrame({"y": [0.05, -0.05]})).mark_rule(
    strokeDash=[4, 4], color="#c0392b").encode(y="y:Q")
st.altair_chart(bias_chart + band, use_container_width=True)
st.caption("Dashed red lines mark the ±5% bias tolerance.")

st.divider()
with st.expander("Backtest detail table"):
    show = bl[["lag", "method_name", "wmape", "bias"]].rename(
        columns={"method_name": "method"})
    st.dataframe(show.style.format({"wmape": "{:.1%}", "bias": "{:+.1%}"}),
                 use_container_width=True, hide_index=True)
st.download_button("⬇ Backtest results (CSV)", data=by_lag.to_csv(index=False).encode(),
                   file_name=f"fva_by_lag_{run_id}.csv", mime="text/csv")
