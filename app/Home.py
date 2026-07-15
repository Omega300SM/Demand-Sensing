"""Home — workspace picker, status overview, and next-step guidance.

Entry point: `streamlit run app/Home.py`.
"""

import os

from app_common import (
    bootstrap, get_workspace, page_header, kfmt, DEFAULT_WS, project_root,
)

bootstrap()
import streamlit as st
from sensing import STREAM_ORDER, CANONICAL_SCHEMAS, Workspace
from sensing import demo_data

st.set_page_config(page_title="Demand Sensing POC", page_icon="📈", layout="wide")

page_header(
    "Demand Sensing POC",
    "Upload → validate → run → review. Prove sensing beats the plan on your "
    "data before building the lake-connected pipeline.",
)

# --------------------------------------------------------------------------- #
# Workspace picker
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Workspace")
    ws_path = st.text_input("DuckDB workspace file", value=st.session_state.get("ws_path", DEFAULT_WS))
    st.session_state["ws_path"] = ws_path
    st.caption("One file per project. Uploads and runs persist between sessions.")

ws = get_workspace()

# --------------------------------------------------------------------------- #
# One-click demo
# --------------------------------------------------------------------------- #
col_a, col_b = st.columns([2, 1])
with col_a:
    st.markdown("#### Get started")
    st.write(
        "New workspace? Load the bundled synthetic dataset "
        "(**3 SKUs × 2 regions × 78 weeks**, two promos and one stockout) to see "
        "every page render immediately."
    )
with col_b:
    st.write("")
    st.write("")
    if st.button("⚡ Load demo dataset", type="primary", use_container_width=True):
        with st.spinner("Generating synthetic data and rebuilding canonical tables…"):
            built = demo_data.load_demo_into_workspace(ws)
        st.success(f"Loaded demo: {sum(built.values()):,} rows across {len(built)} streams.")
        st.rerun()

st.divider()

# --------------------------------------------------------------------------- #
# Status overview
# --------------------------------------------------------------------------- #
st.markdown("#### Data status")
status = ws.canonical_status()
cols = st.columns(len(STREAM_ORDER))
for col, stream in zip(cols, STREAM_ORDER):
    info = status.get(stream, {"loaded": False, "rows": 0})
    label = CANONICAL_SCHEMAS[stream]["label"]
    required = CANONICAL_SCHEMAS[stream]["required"]
    with col:
        badge = "🟢" if info["loaded"] else ("🔴" if required else "⚪")
        st.markdown(f"**{badge} {label}**")
        if info["loaded"]:
            st.metric("rows", kfmt(info["rows"]))
            if "week_min" in info:
                st.caption(f"{info['week_min']:%Y-%m-%d} → {info['week_max']:%Y-%m-%d}")
        else:
            st.caption("Required" if required else "Optional")
            st.caption("not loaded")

st.divider()

# --------------------------------------------------------------------------- #
# Runs + next steps
# --------------------------------------------------------------------------- #
left, right = st.columns(2)
with left:
    st.markdown("#### Recent runs")
    runs = ws.list_runs()
    if len(runs):
        st.dataframe(
            runs[["run_id", "as_of", "created_at"]].head(8),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("No runs yet. Load data, then open **Run Sensing**.")

with right:
    st.markdown("#### Next step")
    pos_loaded = status.get("pos", {}).get("loaded", False)
    if not pos_loaded:
        st.info("① Load the demo dataset above, or go to **Upload Data**.")
    elif not len(ws.list_runs()):
        st.info("② Data is loaded. Open **Run Sensing** to produce a forecast.")
    else:
        st.success("③ You have a run. Explore **Forecast Review**, **Alerts**, "
                   "and **Accuracy / FVA** — the FVA page is the POC's verdict.")

    with st.expander("What this POC answers"):
        st.write(
            "Does a sensed forecast beat the current demand plan at 1–2 week "
            "lags, with bias within ±5%, on *this* business's data? "
            "The engine here is a working stub (statistical baseline + "
            "sell-out→sell-in translation + a real rolling-origin backtest); "
            "the production forecasting modules slot in behind the same "
            "`run_pipeline` interface."
        )

st.caption(f"Workspace file: `{os.path.abspath(ws.path)}`")
