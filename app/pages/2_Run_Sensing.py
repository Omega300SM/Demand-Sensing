"""Run Sensing — config, run button with progress, and run manifest.

The engine runs behind a single call: run_pipeline(ws, as_of, config).
Progress here is illustrative of the pipeline stages; each run is versioned.
"""

import time
from datetime import date

from app_common import bootstrap, get_workspace, page_header, require_canonical

bootstrap()
import pandas as pd
import streamlit as st
from sensing import RunConfig, run_pipeline

st.set_page_config(page_title="Run Sensing", page_icon="▶️", layout="wide")
page_header("Run Sensing", "Pick an as-of date and horizon, tune retailer params, and run.")

ws = get_workspace()
if not require_canonical(ws, "pos"):
    st.stop()

pos = ws.read_canonical("pos")
last_week = pd.to_datetime(pos["week"]).max().date()

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
c1, c2, c3 = st.columns(3)
with c1:
    as_of = st.date_input("As-of date (last complete period)", value=last_week)
    horizon = st.slider("Horizon (weeks)", 1, 16, 8)
with c2:
    target_wos = st.number_input("Target weeks-of-supply", 1.0, 12.0, 4.0, 0.5,
                                 help="Pre-fill from empirical calibration once enough history exists.")
    reaction_lag = st.number_input("Reaction lag (weeks)", 0, 6, 1)
with c3:
    deviation = st.slider("Sensed-vs-plan alert threshold", 0.05, 0.50, 0.15, 0.05)
    backtest_weeks = st.slider("Backtest holdout (weeks)", 8, 52, 26)

ml_enabled = st.toggle("Enable ML tier (LightGBM)", value=False,
                       help="Optional. The statistical tier alone is enough to prove FVA first.")
if ml_enabled:
    st.caption("ML tier is a placeholder in S1 — the statistical tier drives the run.")

cfg = RunConfig(
    as_of=as_of, horizon_weeks=horizon, target_wos=target_wos,
    reaction_lag_weeks=int(reaction_lag), ml_enabled=ml_enabled,
    deviation_threshold=deviation, backtest_weeks=backtest_weeks,
)

st.divider()

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
STAGES = ["Signal repair", "Sell-out forecast", "Sell-out→sell-in translation",
          "Blend with plan", "Alerts & backtest"]

if st.button("▶ Run sensing", type="primary"):
    progress = st.progress(0.0, text="Starting…")
    for i, stage in enumerate(STAGES):
        progress.progress(i / len(STAGES), text=f"{stage}…")
        time.sleep(0.25)  # visual pacing; the real work happens in run_pipeline
    try:
        result = run_pipeline(ws, as_of, cfg)
    except ValueError as e:
        progress.empty()
        st.error(str(e))
        st.stop()
    progress.progress(1.0, text="Done")
    st.session_state["last_run_id"] = result.run_id

    st.success(f"Run `{result.run_id}` complete.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecast rows", f"{len(result.forecast):,}")
    m2.metric("Censored weeks repaired", int(result.repaired["is_censored"].sum()))
    m3.metric("Alerts", len(result.alerts))
    fva = result.fva
    headline = fva[fva["step"] == "Statistical baseline"]["wmape"]
    m4.metric("Statistical WMAPE (lag 1–2)", f"{headline.iloc[0]:.1%}" if len(headline) else "—")

    st.caption("Next: **Forecast Review**, **Alerts**, and **Accuracy / FVA**.")

    with st.expander("Run manifest"):
        st.json(result.manifest)

st.divider()
runs = ws.list_runs()
if len(runs):
    st.markdown("#### Previous runs")
    st.dataframe(runs[["run_id", "as_of", "created_at"]], use_container_width=True, hide_index=True)
