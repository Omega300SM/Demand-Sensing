"""Forecast Review — per-series drill-down and the POC's downstream hand-off.

Shows repaired demand vs. raw sales (censored periods shaded), the forecast fan
(P10/P50/P90), and projected channel inventory vs. target. The blended forecast
export at sku×region×week *is* the POC's downstream integration.
"""

from app_common import bootstrap, get_workspace, page_header, require_canonical, latest_run_id

bootstrap()
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forecast Review", page_icon="🔎", layout="wide")
page_header("Forecast Review", "Drill into a series; export the blended forecast for DRP/planning.")

ws = get_workspace()
if not require_canonical(ws, "pos"):
    st.stop()

run_id = st.session_state.get("last_run_id") or latest_run_id(ws)
if not run_id:
    st.warning("No runs yet — open **Run Sensing** first.")
    st.stop()

forecast = ws.read_run_output(run_id, "forecast")
repaired = ws.read_run_output(run_id, "repaired")
proj = ws.read_run_output(run_id, "inventory_projection")
alerts = ws.read_run_output(run_id, "alerts")

st.caption(f"Showing run `{run_id}`")

# --------------------------------------------------------------------------- #
# Series picker (ranked by volume or alert severity)
# --------------------------------------------------------------------------- #
vol = (repaired.groupby(["item_id", "region"])["units_unconstrained"].sum()
       .reset_index().rename(columns={"units_unconstrained": "volume"}))
sev_rank = {"high": 2, "medium": 1, "low": 0}
if len(alerts):
    asev = alerts.assign(sev=alerts["severity"].map(sev_rank)).groupby(
        ["item_id", "region"])["sev"].max().reset_index()
    vol = vol.merge(asev, on=["item_id", "region"], how="left")
    vol["sev"] = vol["sev"].fillna(-1)
else:
    vol["sev"] = -1

rank_by = st.radio("Rank by", ["Volume", "Alert severity"], horizontal=True)
vol = vol.sort_values("volume" if rank_by == "Volume" else ["sev", "volume"], ascending=False)

labels = [f"{r.item_id} · {r.region}  ({int(r.volume):,})" for r in vol.itertuples()]
choice = st.selectbox("Series", labels)
sel = vol.iloc[labels.index(choice)]
sku, region = sel["item_id"], sel["region"]

# --------------------------------------------------------------------------- #
# Chart 1: repaired demand vs raw sales + forecast fan
# --------------------------------------------------------------------------- #
hist = repaired[(repaired["item_id"] == sku) & (repaired["region"] == region)].sort_values("week")
fc = forecast[(forecast["item_id"] == sku) & (forecast["region"] == region)].sort_values("week")

hist_long = hist.melt(id_vars=["week", "is_censored"],
                      value_vars=["units_sold", "units_unconstrained"],
                      var_name="series", value_name="units")
name_map = {"units_sold": "Raw sales", "units_unconstrained": "Repaired demand"}
hist_long["series"] = hist_long["series"].map(name_map)

base_lines = alt.Chart(hist_long).mark_line().encode(
    x=alt.X("week:T", title="Week"),
    y=alt.Y("units:Q", title="Units"),
    color=alt.Color("series:N", title="",
                    scale=alt.Scale(domain=["Raw sales", "Repaired demand"],
                                    range=["#95a5a6", "#2c3e50"])),
)

# shade censored weeks as proper vertical bands (week ± half a week)
cens = hist[hist["is_censored"]].copy()
shade = None
if len(cens):
    cens["w0"] = pd.to_datetime(cens["week"]) - pd.Timedelta(days=3, hours=12)
    cens["w1"] = pd.to_datetime(cens["week"]) + pd.Timedelta(days=3, hours=12)
    shade = alt.Chart(cens).mark_rect(opacity=0.18, color="#c0392b").encode(
        x="w0:T", x2="w1:T")

fan = None
if len(fc):
    band = alt.Chart(fc).mark_area(opacity=0.20, color="#2980b9").encode(
        x="week:T", y="p10:Q", y2="p90:Q")
    p50 = alt.Chart(fc).mark_line(color="#2980b9", strokeDash=[4, 2]).encode(
        x="week:T", y="p50:Q")
    fan = band + p50

layers = [base_lines]
if shade is not None:
    layers.insert(0, shade)
if fan is not None:
    layers.append(fan)

st.markdown(f"#### {sku} · {region} — repaired demand & forecast fan")
st.altair_chart(alt.layer(*layers).resolve_scale(y="shared").properties(height=320),
                use_container_width=True)
if len(cens):
    st.caption(f"🔴 shaded = {len(cens)} censored week(s) de-censored before forecasting; "
               "dashed line + band = P50 with P10–P90 fan. Repaired demand imputes the "
               "unconstrained level on stockout weeks, so the model never learns to predict "
               "our own stockouts.")
else:
    st.caption("No censored weeks detected on this series; repaired demand tracks raw sales. "
               "Dashed line + band = P50 with P10–P90 fan.")

# --------------------------------------------------------------------------- #
# Chart 1b: baseline / promo-uplift decomposition (where a promo calendar exists)
# --------------------------------------------------------------------------- #
if "promo_uplift" in hist.columns and float(hist["promo_uplift"].sum()) > 0:
    st.markdown("#### Baseline vs. promo uplift")
    decomp = hist[["week", "base_units", "promo_uplift"]].melt(
        id_vars=["week"], value_vars=["base_units", "promo_uplift"],
        var_name="component", value_name="units")
    decomp["component"] = decomp["component"].map(
        {"base_units": "Base demand", "promo_uplift": "Promo uplift"})
    decomp_chart = alt.Chart(decomp).mark_area().encode(
        x=alt.X("week:T", title="Week"),
        y=alt.Y("units:Q", title="Units", stack="zero"),
        color=alt.Color("component:N", title="",
                        scale=alt.Scale(domain=["Base demand", "Promo uplift"],
                                        range=["#2c3e50", "#e67e22"])),
        order=alt.Order("component:N", sort="descending"),
    ).properties(height=240)
    st.altair_chart(decomp_chart, use_container_width=True)
    n_promo = int((hist["promo_uplift"] > 0).sum())
    st.caption(f"Decomposition splits repaired demand into a de-promoted baseline and the "
               f"incremental promo lift on {n_promo} promo week(s) "
               f"(base + uplift = repaired demand). The post-promo dip stays in the baseline "
               f"as real demand, not a signal to explain away.")
else:
    st.caption("No promo calendar for this series (or no uplift) — base demand equals repaired "
               "demand and promo uplift is zero.")

# --------------------------------------------------------------------------- #
# Chart 2: projected channel inventory vs target + projected orders vs plan
# --------------------------------------------------------------------------- #
gp = proj[(proj["item_id"] == sku) & (proj["region"] == region)].sort_values("week")
if len(gp):
    st.markdown("#### Projected channel inventory vs. target cover")
    inv_long = gp.melt(id_vars=["week"], value_vars=["projected_on_hand", "target_position"],
                       var_name="series", value_name="units")
    inv_long["series"] = inv_long["series"].map(
        {"projected_on_hand": "Projected on-hand", "target_position": "Target position"})
    inv_chart = alt.Chart(inv_long).mark_line().encode(
        x=alt.X("week:T", title="Week"), y=alt.Y("units:Q", title="Units"),
        color=alt.Color("series:N", title="",
                        scale=alt.Scale(range=["#27ae60", "#e67e22"])),
        strokeDash=alt.condition(alt.datum.series == "Target position",
                                 alt.value([5, 3]), alt.value([0])),
    ).properties(height=260)
    st.altair_chart(inv_chart, use_container_width=True)

    st.markdown("#### Projected orders (sell-in) vs. demand plan")
    fc_plan = fc[["week", "p50", "plan_units", "blended"]].merge(
        gp[["week", "projected_order"]], on="week", how="left")
    order_long = fc_plan.melt(id_vars=["week"],
                              value_vars=["projected_order", "plan_units", "blended"],
                              var_name="series", value_name="units").dropna()
    order_long["series"] = order_long["series"].map(
        {"projected_order": "Projected order", "plan_units": "Demand plan", "blended": "Blended"})
    order_chart = alt.Chart(order_long).mark_line(point=True).encode(
        x=alt.X("week:T", title="Week"), y=alt.Y("units:Q", title="Units"),
        color=alt.Color("series:N", title=""),
    ).properties(height=260)
    st.altair_chart(order_chart, use_container_width=True)

# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
st.divider()
st.markdown("#### Export blended forecast (POC downstream hand-off)")
export = forecast[["item_id", "region", "week", "lag", "p10", "p50", "p90",
                   "plan_units", "blend_weight_sensed", "blended"]].copy()
c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇ Blended forecast (CSV)", data=export.to_csv(index=False).encode(),
                       file_name=f"blended_forecast_{run_id}.csv", mime="text/csv",
                       use_container_width=True)
with c2:
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        export.to_excel(xw, index=False, sheet_name="forecast")
    st.download_button("⬇ Blended forecast (XLSX)", data=buf.getvalue(),
                       file_name=f"blended_forecast_{run_id}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
st.caption("Hand this file to DRP/planning as a manual feed — that *is* the POC's downstream integration.")
