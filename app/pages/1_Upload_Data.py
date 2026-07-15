"""Upload Data — per-stream upload, interactive column mapper, QC cards.

This page is the v3 core UX: it replaces v2's fixed data contracts with an
upload-and-map layer. Accepted files land as immutable snapshots in the
workspace; "Rebuild dataset" materialises the canonical tables the engine reads.

S2 hardening (design §3.2–§3.3):
  * wide-format (weeks-as-columns) exports are unpivoted in the UI,
  * a your-column->canonical mapping (plus unit + week-calendar + wide config)
    is saved per stream+source and pre-filled on a matching re-upload,
  * a configurable retailer week-calendar appears next to WM-week columns,
  * multiple files stack per slot; rebuild de-dups on the grain,
  * required-but-unmapped fields block the snapshot with a clear message.
"""

import os
from datetime import date

from app_common import bootstrap, get_workspace, page_header, STATUS_EMOJI

bootstrap()
import pandas as pd
import streamlit as st
from sensing import CANONICAL_SCHEMAS, STREAM_ORDER
from sensing import ingest_ui, quality, demo_data

st.set_page_config(page_title="Upload Data", page_icon="📤", layout="wide")
page_header("Upload Data", "One card per stream. Missing optional streams degrade gracefully.")

ws = get_workspace()

# Ensure templates + messy fixtures exist on disk for download / demo.
templates_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "templates")
if not os.path.exists(os.path.join(templates_dir, "template_pos.csv")):
    demo_data.write_templates(templates_dir)

_UNIT_OPTIONS = {"eaches (×1)": 1.0, "cases (×12)": 12.0, "cases (×24)": 24.0}


def _template_bytes(stream: str) -> bytes:
    path = os.path.join(templates_dir, f"template_{stream}.csv")
    with open(path, "rb") as f:
        return f.read()


def _numeric_fields(stream: str) -> list[str]:
    return [c for c, m in CANONICAL_SCHEMAS[stream]["fields"].items()
            if m["role"] == "numeric"]


def _status_line(stream: str) -> str:
    cov = ws.snapshot_coverage(stream)
    if cov["snapshots"] == 0:
        return "No uploads yet."
    rng = ""
    if cov["date_min"] is not None and cov["date_max"] is not None:
        rng = f" · {cov['date_min']} → {cov['date_max']}"
    canon = ""
    if ws.has_canonical(stream):
        canon = f" · canonical {len(ws.read_canonical(stream)):,} rows"
    return (f"{cov['snapshots']} file(s) stacked · "
            f"{cov['rows']:,} rows uploaded{rng}{canon}")


def _render_file(stream: str, schema: dict, uf) -> None:
    """Render the mapper + accept flow for one uploaded file."""
    content = uf.getvalue()
    try:
        raw = ingest_ui.read_upload(uf.name, content)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read {uf.name}: {e}")
        return

    sig = ingest_ui.source_signature(uf.name, raw)
    saved = ws.get_mapping(stream, sig)
    kbase = f"{stream}_{uf.name}"

    st.markdown(f"**Mapping `{uf.name}`** — {len(raw):,} rows, {len(raw.columns)} columns"
                + ("  ·  ↩ saved mapping applied" if saved else ""))

    # ---- wide-format unpivot -------------------------------------------- #
    saved_wide = (saved or {}).get("wide")
    work = raw
    wide_cfg = None
    if ingest_ui.is_wide_format(raw) or saved_wide:
        with st.container(border=True):
            st.caption("📐 Wide format detected (weeks as columns). "
                       "Pick the ID columns to keep; the rest are melted into rows.")
            default_ids = (saved_wide or {}).get("id_cols") or ingest_ui.suggest_wide_id_cols(raw)
            default_ids = [c for c in default_ids if c in raw.columns]
            id_cols = st.multiselect("ID columns to keep", list(raw.columns),
                                     default=default_ids, key=f"wide_ids_{kbase}")
            num_fields = _numeric_fields(stream)
            vf_default = (saved_wide or {}).get("value_field") or (num_fields[0] if num_fields else "value")
            vf_idx = num_fields.index(vf_default) if vf_default in num_fields else 0
            value_field = st.selectbox("Melt values into", num_fields, index=vf_idx,
                                       key=f"wide_val_{kbase}")
            if id_cols:
                work = ingest_ui.unpivot_wide(raw, id_cols=id_cols, value_name=value_field)
                wide_cfg = {"id_cols": id_cols, "value_field": value_field}
                st.caption(f"→ unpivoted to {len(work):,} long rows.")

    # ---- column mapping -------------------------------------------------- #
    suggested = (saved or {}).get("mapping") if saved else ingest_ui.suggest_mapping(work, stream)
    if not suggested:
        suggested = ingest_ui.suggest_mapping(work, stream)
    src_options = ["— none —"] + list(work.columns)

    mapping: dict[str, str | None] = {}
    mcols = st.columns(min(4, len(schema["fields"])))
    for i, (canon, meta) in enumerate(schema["fields"].items()):
        with mcols[i % len(mcols)]:
            default = suggested.get(canon)
            idx = src_options.index(default) if default in src_options else 0
            pick = st.selectbox(
                f"{canon}{' *' if meta['required'] else ''}",
                src_options, index=idx, key=f"map_{kbase}_{canon}",
            )
            mapping[canon] = None if pick == "— none —" else pick

    # ---- week-calendar picker (only when WM-style labels are present) ---- #
    week_calendar = None
    wk_src = mapping.get("week")
    if wk_src and wk_src in work.columns and ingest_ui.looks_wm_labels(work[wk_src]):
        saved_cal = (saved or {}).get("week_calendar") or {}
        with st.expander("🗓 Retailer week-calendar (WM/fiscal labels detected)", expanded=False):
            months = list(range(1, 13))
            m_default = int(saved_cal.get("fiscal_year_start_month", 1))
            fy_month = st.selectbox("Fiscal year starts in month", months,
                                    index=months.index(m_default) if m_default in months else 0,
                                    key=f"cal_m_{kbase}",
                                    help="e.g. Walmart's fiscal year starts in February.")
            fy_day = st.number_input("…on day", min_value=1, max_value=28,
                                     value=int(saved_cal.get("fiscal_year_start_day", 1)),
                                     key=f"cal_d_{kbase}")
            week_calendar = ingest_ui.WeekCalendar(
                fiscal_year_start_month=int(fy_month), fiscal_year_start_day=int(fy_day))

    # ---- units ---------------------------------------------------------- #
    unit_labels = list(_UNIT_OPTIONS)
    u_default = (saved or {}).get("unit_label", unit_labels[0])
    u_idx = unit_labels.index(u_default) if u_default in unit_labels else 0
    unit = st.selectbox("Units in this file", unit_labels, index=u_idx, key=f"unit_{kbase}")
    mult = _UNIT_OPTIONS[unit]

    # ---- required-but-unmapped block ------------------------------------ #
    missing = ingest_ui.missing_required(stream, mapping)
    if missing:
        st.error("Required field(s) not mapped: " + ", ".join(f"`{m}`" for m in missing)
                 + ". Map them above before this file can be accepted.")

    canonical, warns = ingest_ui.apply_mapping(
        work, stream, mapping, unit_multiplier=mult, week_calendar=week_calendar)
    for w in warns:
        st.caption(f"⚠ {w}")

    st.caption("Preview (first 20 parsed rows):")
    st.dataframe(canonical.head(20), use_container_width=True, hide_index=True)

    disabled = bool(missing) or canonical.empty
    if st.button(f"✅ Accept & snapshot `{uf.name}`", key=f"accept_{kbase}",
                 type="primary", disabled=disabled):
        sid = ws.add_snapshot(stream, canonical, source_name=uf.name)
        ws.save_mapping(stream, sig, {
            "mapping": mapping,
            "unit_label": unit,
            "week_calendar": week_calendar.to_dict() if week_calendar else None,
            "wide": wide_cfg,
            "source_name": uf.name,
        })
        st.success(f"Snapshot saved: `{sid}` · mapping remembered for this export.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Per-stream cards
# --------------------------------------------------------------------------- #
for stream in STREAM_ORDER:
    schema = CANONICAL_SCHEMAS[stream]
    req = "Required" if schema["required"] else "Optional"
    with st.container(border=True):
        head, dl = st.columns([4, 1])
        with head:
            st.markdown(f"### {schema['label']}  \n*{req}*")
            st.caption("Canonical fields: " + ", ".join(schema["fields"].keys()))
            st.caption(_status_line(stream))
        with dl:
            st.download_button(
                "⬇ CSV template", data=_template_bytes(stream),
                file_name=f"template_{stream}.csv", mime="text/csv",
                use_container_width=True, key=f"tmpl_{stream}",
            )

        uploaded = st.file_uploader(
            "Drag & drop CSV / XLSX / parquet (multi-file to stack months)",
            type=["csv", "xlsx", "xls", "parquet"],
            accept_multiple_files=True, key=f"up_{stream}",
        )

        for uf in (uploaded or []):
            _render_file(stream, schema, uf)

st.divider()

# --------------------------------------------------------------------------- #
# QC validation cards (against current snapshots) + Rebuild
# --------------------------------------------------------------------------- #
st.markdown("### Validation & rebuild")
rebuild_col, qcdemo_col, _ = st.columns([1, 1, 2])
with rebuild_col:
    if st.button("🔁 Rebuild dataset", type="primary", use_container_width=True):
        built = ws.rebuild_canonical()
        st.success("Rebuilt canonical tables: " +
                   (", ".join(f"{k} ({v:,})" for k, v in built.items()) or "nothing to build."))
with qcdemo_col:
    if st.button("🧪 Load QC demo", use_container_width=True,
                 help="Load a small dataset seeded with one instance of each QC "
                      "defect class, so every gate below fires."):
        built = demo_data.load_qc_demo_into_workspace(ws)
        st.success("Loaded the QC-defects demo: " +
                   ", ".join(f"{k} ({v:,})" for k, v in built.items()) +
                   ". Expand the cards below — each gate should flag its defect.")
        st.rerun()

# QC cards run against the current canonical tables (post-rebuild).
# Freshness is evaluated against *today* — a real weekly export should be recent.
as_of = date.today()
crosswalk = None
pos_df = ws.read_canonical("pos") if ws.has_canonical("pos") else None
if pos_df is not None and len(pos_df):
    crosswalk = pos_df[["item_id"]].drop_duplicates()

for stream in STREAM_ORDER:
    if not ws.has_canonical(stream):
        continue
    df = ws.read_canonical(stream)
    report = quality.run_qc(stream, df, crosswalk=crosswalk, pos=pos_df, as_of=as_of)
    with st.expander(f"{STATUS_EMOJI[report.worst]} {CANONICAL_SCHEMAS[stream]['label']} "
                     f"— {report.rows:,} rows", expanded=(report.worst != "pass")):
        for chk in report.checks:
            st.markdown(f"{STATUS_EMOJI[chk.status]} **{chk.name}** — {chk.detail}")
            # Any check may attach offending rows — surface them as a download
            # (quarantine-and-flag: this frame is the planner's cleanup to-do).
            if chk.data is not None and len(chk.data):
                slug = chk.name.lower().replace(" ", "_").replace("/", "_")
                st.download_button(
                    f"⬇ {chk.name} — offenders ({len(chk.data):,})",
                    data=chk.data.to_csv(index=False).encode(),
                    file_name=f"qc_{stream}_{slug}.csv", mime="text/csv",
                    key=f"qc_dl_{stream}_{chk.name}",
                )
