"""Alerts — the S&OE exception queue with the driving signal shown inline.

Same exception classes as v2: sensed-vs-plan deviation, projected retailer
stockout, channel overstock / order-cliff, signal-repair/censoring, QC holds.
"""

from app_common import bootstrap, get_workspace, page_header, require_canonical, latest_run_id, SEVERITY_COLORS

bootstrap()
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Alerts", page_icon="🚨", layout="wide")
page_header("Alerts", "Exception queue for the pilot retailer — each row carries the *why*.")

ws = get_workspace()
if not require_canonical(ws, "pos"):
    st.stop()

run_id = st.session_state.get("last_run_id") or latest_run_id(ws)
if not run_id:
    st.warning("No runs yet — open **Run Sensing** first.")
    st.stop()

alerts = ws.read_run_output(run_id, "alerts")
forecast = ws.read_run_output(run_id, "forecast")
proj = ws.read_run_output(run_id, "inventory_projection")
repaired = ws.read_run_output(run_id, "repaired")

st.caption(f"Showing run `{run_id}`")

if not len(alerts):
    st.success("No exceptions fired for this run.")
    st.stop()

# --------------------------------------------------------------------------- #
# Filters + summary
# --------------------------------------------------------------------------- #
sev_order = {"high": 0, "medium": 1, "low": 2}
alerts = alerts.assign(_o=alerts["severity"].map(sev_order)).sort_values(["_o", "item_id"])

c1, c2, c3 = st.columns(3)
c1.metric("Total alerts", len(alerts))
c2.metric("High severity", int((alerts["severity"] == "high").sum()))
c3.metric("Alert types", alerts["alert_type"].nunique())

types = ["All"] + sorted(alerts["alert_type"].unique())
sevs = ["All", "high", "medium", "low"]
f1, f2 = st.columns(2)
type_filter = f1.selectbox("Filter by type", types)
sev_filter = f2.selectbox("Filter by severity", sevs)

view = alerts
if type_filter != "All":
    view = view[view["alert_type"] == type_filter]
if sev_filter != "All":
    view = view[view["severity"] == sev_filter]

st.divider()

# --------------------------------------------------------------------------- #
# Alert rows with inline driving-signal chart
# --------------------------------------------------------------------------- #
def _driving_chart(alert_type: str, sku: str, region: str):
    if "stockout" in alert_type.lower() or "overstock" in alert_type.lower():
        gp = proj[(proj["item_id"] == sku) & (proj["region"] == region)].sort_values("week")
        if not len(gp):
            return None
        long = gp.melt(id_vars=["week"], value_vars=["projected_on_hand", "target_position"],
                       var_name="s", value_name="units")
        return alt.Chart(long).mark_line().encode(
            x=alt.X("week:T", title=None), y=alt.Y("units:Q", title="Units"),
            color=alt.Color("s:N", title="")).properties(height=160)
    if "deviation" in alert_type.lower():
        fc = forecast[(forecast["item_id"] == sku) & (forecast["region"] == region)].sort_values("week")
        if not len(fc):
            return None
        long = fc.melt(id_vars=["week"], value_vars=["p50", "plan_units"],
                       var_name="s", value_name="units").dropna()
        long["s"] = long["s"].map({"p50": "Sensed P50", "plan_units": "Plan"})
        return alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("week:T", title=None), y=alt.Y("units:Q", title="Units"),
            color=alt.Color("s:N", title="")).properties(height=160)
    # censoring / repair
    h = repaired[(repaired["item_id"] == sku) & (repaired["region"] == region)].sort_values("week")
    if not len(h):
        return None
    long = h.melt(id_vars=["week"], value_vars=["units_sold", "units_unconstrained"],
                  var_name="s", value_name="units")
    long["s"] = long["s"].map({"units_sold": "Raw", "units_unconstrained": "Repaired"})
    return alt.Chart(long).mark_line().encode(
        x=alt.X("week:T", title=None), y=alt.Y("units:Q", title="Units"),
        color=alt.Color("s:N", title="")).properties(height=160)


for _, a in view.iterrows():
    color = SEVERITY_COLORS.get(a["severity"], "#7d8a99")
    with st.container(border=True):
        left, right = st.columns([2, 3])
        with left:
            st.markdown(
                f"<span style='color:{color};font-weight:700'>"
                f"● {a['severity'].upper()}</span> · **{a['alert_type']}**",
                unsafe_allow_html=True)
            st.markdown(f"**{a['item_id']} · {a['region']}**")
            wk = pd.to_datetime(a["week"])
            st.caption(f"Week: {wk:%Y-%m-%d}" if pd.notna(wk) else "")
            st.write(a["message"])
        with right:
            chart = _driving_chart(a["alert_type"], a["item_id"], a["region"])
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)

st.divider()
st.download_button("⬇ Export alert queue (CSV)",
                   data=view.drop(columns=["_o"], errors="ignore").to_csv(index=False).encode(),
                   file_name=f"alerts_{run_id}.csv", mime="text/csv")
