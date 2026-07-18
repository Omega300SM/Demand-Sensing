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

S9 (design §3.1, §3.3):
  * the item crosswalk is a *dimension* (master data), rendered in its own cards
    after the streams and landed as a ``dim_*`` table (replace-on-upload);
  * gate 7 measures item match rate against that real crosswalk (``None`` +
    a skipped-banner when none is uploaded) — never a POS-derived stand-in;
  * **empty is not success**: accept-state is visible per file, unaccepted files
    block-and-warn above Rebuild, and an empty rebuild is a warning, not a
    green box.
"""

import os
from datetime import date

from app_common import bootstrap, get_workspace, page_header, STATUS_EMOJI

bootstrap()
import streamlit as st
from sensing import (CANONICAL_SCHEMAS, STREAM_ORDER,
                     REFERENCE_SCHEMAS, REFERENCE_ORDER)
from sensing import ingest_ui, quality, demo_data

st.set_page_config(page_title="Upload Data", page_icon="📤", layout="wide")
page_header("Upload Data", "One card per stream. Missing optional streams degrade gracefully.")

ws = get_workspace()

# Ensure templates + messy fixtures exist on disk for download / demo. Regenerate
# whenever ANY expected template is missing — a workspace created before S9 has
# the stream templates but not the new reference (master-data) ones.
templates_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "templates")
_expected_templates = [f"template_{n}.csv" for n in STREAM_ORDER + REFERENCE_ORDER]
if not all(os.path.exists(os.path.join(templates_dir, t)) for t in _expected_templates):
    demo_data.write_templates(templates_dir)

_UNIT_OPTIONS = {"eaches (×1)": 1.0, "cases (×12)": 12.0, "cases (×24)": 24.0}

# Streams / references with no unit-bearing quantity: the eaches↔cases selector
# is meaningless (and misleading) for them, so it is not rendered. `promo` only
# carries a 0/1 flag; references carry ids and a pack size that is not a per-file
# unit choice.
_NO_UNIT = {"promo"} | set(REFERENCE_ORDER)

# Confirmation of the last accepted snapshot survives the post-accept rerun.
if _msg := st.session_state.pop("upload_toast", None):
    st.toast(_msg, icon="✅")


def _template_bytes(name: str) -> bytes:
    path = os.path.join(templates_dir, f"template_{name}.csv")
    with open(path, "rb") as f:
        return f.read()


def _numeric_fields(name: str, schema: dict) -> list[str]:
    return [c for c, m in schema["fields"].items() if m["role"] == "numeric"]


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


def _is_snapshotted(stream: str, sig: str) -> bool:
    """True if a file with this source signature already has a saved mapping —
    i.e. it was accepted before. The signature is header-set based, so the same
    export re-dropped is recognised as already snapshotted."""
    return ws.get_mapping(stream, sig) is not None


def _render_file(name: str, schema: dict, uf, *, is_reference: bool) -> bool:
    """Render the mapper + accept flow for one uploaded file.

    Returns True if the file is dropped-but-not-yet-accepted (so the page can
    block-and-warn above Rebuild). ``is_reference`` toggles off the unit and
    week-calendar controls and swaps QC for a two-line self-check.
    """
    content = uf.getvalue()
    try:
        raw = ingest_ui.read_upload(uf.name, content)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read {uf.name}: {e}")
        return False

    sig = ingest_ui.source_signature(uf.name, raw)
    saved = ws.get_mapping(name, sig)
    kbase = f"{name}_{uf.name}"

    already = _is_snapshotted(name, sig)
    st.markdown(f"**Mapping `{uf.name}`** — {len(raw):,} rows, {len(raw.columns)} columns"
                + ("  ·  ↩ saved mapping applied" if saved else ""))
    if already:
        st.success(f"✅ Already snapshotted — this export matches a file already "
                   f"in the workspace. Re-accept only to overwrite its mapping.")

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
            num_fields = _numeric_fields(name, schema)
            vf_default = (saved_wide or {}).get("value_field") or (num_fields[0] if num_fields else "value")
            vf_idx = num_fields.index(vf_default) if vf_default in num_fields else 0
            value_field = st.selectbox("Melt values into", num_fields, index=vf_idx,
                                       key=f"wide_val_{kbase}")
            if id_cols:
                work = ingest_ui.unpivot_wide(raw, id_cols=id_cols, value_name=value_field)
                wide_cfg = {"id_cols": id_cols, "value_field": value_field}
                st.caption(f"→ unpivoted to {len(work):,} long rows.")

    # ---- column mapping -------------------------------------------------- #
    suggested = (saved or {}).get("mapping") if saved else ingest_ui.suggest_mapping(work, name)
    if not suggested:
        suggested = ingest_ui.suggest_mapping(work, name)
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

    # ---- week-calendar picker (streams only; references are dateless) ---- #
    week_calendar = None
    if not is_reference:
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
                # Start from the saved calendar (keeps label_format / century) and
                # overlay just the two UI picks — .from_dict tolerates extra keys.
                week_calendar = ingest_ui.WeekCalendar.from_dict(saved_cal)
                week_calendar.fiscal_year_start_month = int(fy_month)
                week_calendar.fiscal_year_start_day = int(fy_day)

    # ---- units (only for streams that carry a unit-bearing quantity) ---- #
    mult = 1.0
    unit = None
    if name not in _NO_UNIT:
        unit_labels = list(_UNIT_OPTIONS)
        u_default = (saved or {}).get("unit_label", unit_labels[0])
        u_idx = unit_labels.index(u_default) if u_default in unit_labels else 0
        unit = st.selectbox("Units in this file", unit_labels, index=u_idx, key=f"unit_{kbase}")
        mult = _UNIT_OPTIONS[unit]

    # ---- required-but-unmapped block ------------------------------------ #
    missing = ingest_ui.missing_required(name, mapping)
    if missing:
        st.error("Required field(s) not mapped: " + ", ".join(f"`{m}`" for m in missing)
                 + ". Map them above before this file can be accepted.")

    canonical, warns = ingest_ui.apply_mapping(
        work, name, mapping, unit_multiplier=mult, week_calendar=week_calendar)
    for w in warns:
        st.caption(f"⚠ {w}")

    # ---- reference self-check (no week/units → no QC card) --------------- #
    if is_reference:
        _reference_self_check(name, schema, canonical)

    st.caption("Preview (first 20 parsed rows):")
    st.dataframe(canonical.head(20), use_container_width=True, hide_index=True)

    disabled = bool(missing) or canonical.empty
    if st.button(f"✅ Accept & snapshot `{uf.name}`", key=f"accept_{kbase}",
                 type="primary", disabled=disabled):
        if is_reference:
            ws.add_reference(name, canonical, source_name=uf.name)
            confirm = f"Master data saved: `{name}` ({len(canonical):,} rows)."
        else:
            sid = ws.add_snapshot(name, canonical, source_name=uf.name)
            confirm = f"Snapshot saved: `{sid}`."
        ws.save_mapping(name, sig, {
            "mapping": mapping,
            "unit_label": unit,
            "week_calendar": week_calendar.to_dict() if week_calendar else None,
            "wide": wide_cfg,
            "source_name": uf.name,
        })
        # st.rerun() unwinds the script, so an inline st.success here would never
        # render — stash the confirmation and show it as a toast after the rerun.
        st.session_state["upload_toast"] = confirm + " Mapping remembered."
        st.rerun()

    # Dropped but not accepted (and not already snapshotted) -> counts toward
    # the block-and-warn above Rebuild.
    return not already and not disabled


def _reference_self_check(name: str, schema: dict, df) -> None:
    """A crosswalk has no week/units to QC — do a two-line integrity check:
    duplicate source ids, and blank canonical ids."""
    id_fields = [c for c, m in schema["fields"].items()
                 if m["role"] == "id" and m["required"]]
    src = id_fields[0] if id_fields else None
    if src and src in df.columns:
        dups = int(df.duplicated(subset=[src]).sum())
        if dups:
            st.warning(f"{dups} duplicate `{src}` value(s) — master data should be "
                       f"one row per source id. Last one wins on read.")
        else:
            st.caption(f"✓ `{src}` is unique.")
    # blank canonical target (e.g. item_id) — unmappable master rows
    tgt = "item_id" if "item_id" in df.columns else (id_fields[1] if len(id_fields) > 1 else None)
    if tgt and tgt in df.columns:
        blanks = int(df[tgt].astype(str).str.strip().isin(["", "nan", "None"]).sum())
        if blanks:
            st.warning(f"{blanks} row(s) with a blank `{tgt}` — these map nothing.")
        else:
            st.caption(f"✓ no blank `{tgt}`.")


# --------------------------------------------------------------------------- #
# Per-stream cards
# --------------------------------------------------------------------------- #
pending = 0  # files dropped-but-unaccepted across all cards

for stream in STREAM_ORDER:
    schema = CANONICAL_SCHEMAS[stream]
    req = "Required" if schema["required"] else "Optional"
    with st.container(border=True):
        head, dl = st.columns([4, 1])
        with head:
            st.markdown(f"### {schema['label']}  \n*{req}*")
            # Accept-state promoted into the card header (was buried in captions).
            st.markdown(f"**{_status_line(stream)}**")
            st.caption("Canonical fields: " + ", ".join(schema["fields"].keys()))
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
            if _render_file(stream, schema, uf, is_reference=False):
                pending += 1

# --------------------------------------------------------------------------- #
# Master-data (reference / dimension) cards — rendered after the streams.
# --------------------------------------------------------------------------- #
st.markdown("## Master data")
st.caption("Dateless, current-state reference tables. Replace-on-upload (not "
           "stacked). The item crosswalk drives the crosswalk-match gate below.")

for name in REFERENCE_ORDER:
    schema = REFERENCE_SCHEMAS[name]
    req = "Required" if schema["required"] else "Optional"
    with st.container(border=True):
        head, dl = st.columns([4, 1])
        with head:
            st.markdown(f"### {schema['label']}  \n*{req}*")
            if ws.has_reference(name):
                n = len(ws.read_reference(name))
                st.markdown(f"**✅ Uploaded · {n:,} rows (current state).**")
            else:
                st.markdown("**No master data uploaded yet.**")
            st.caption("Fields: " + ", ".join(schema["fields"].keys()))
        with dl:
            st.download_button(
                "⬇ CSV template", data=_template_bytes(name),
                file_name=f"template_{name}.csv", mime="text/csv",
                use_container_width=True, key=f"tmpl_{name}",
            )

        uploaded = st.file_uploader(
            "Drag & drop CSV / XLSX / parquet (replaces current master data)",
            type=["csv", "xlsx", "xls", "parquet"],
            accept_multiple_files=True, key=f"up_ref_{name}",
        )
        for uf in (uploaded or []):
            if _render_file(name, schema, uf, is_reference=True):
                pending += 1

st.divider()

# --------------------------------------------------------------------------- #
# QC validation cards (against current snapshots) + Rebuild
# --------------------------------------------------------------------------- #
st.markdown("### Validation & rebuild")

# Empty is not success: warn before Rebuild when files are mapped-but-unaccepted.
if pending:
    st.warning(f"{pending} file(s) are mapped but **not accepted** — click "
               f"**✅ Accept & snapshot** on each before rebuilding. A rebuild "
               f"only sees accepted files.")

rebuild_col, qcdemo_col, _ = st.columns([1, 1, 2])
with rebuild_col:
    if st.button("🔁 Rebuild dataset", type="primary", use_container_width=True):
        built = ws.rebuild_canonical()
        if built:
            st.success("Rebuilt canonical tables: " +
                       ", ".join(f"{k} ({v:,})" for k, v in built.items()))
        else:
            # Empty is not success — say why, in warning colour.
            st.warning("Nothing to build — no snapshots found. Accept your mapped "
                       "files first (✅ Accept & snapshot on each card above), then "
                       "rebuild.")
with qcdemo_col:
    if st.button("🧪 Load QC demo", use_container_width=True,
                 help="Load a small dataset seeded with one instance of each QC "
                      "defect class, so every gate below fires."):
        built = demo_data.load_qc_demo_into_workspace(ws)
        # The demo's freshness is calibrated against defects_as_of(), not today —
        # stash it so the QC cards evaluate freshness against the right date and
        # only channel_inventory trips it (rather than every stream going red).
        st.session_state["qc_as_of"] = demo_data.defects_as_of()
        st.success("Loaded the QC-defects demo: " +
                   ", ".join(f"{k} ({v:,})" for k, v in built.items()) +
                   ". Expand the cards below — each gate should flag its defect.")
        st.rerun()

# QC cards run against the current canonical tables (post-rebuild).
# Freshness defaults to *today* (a real weekly export should be recent), but the
# QC demo stashes its own as-of so its calibrated single freshness red survives;
# either way the planner can override the reference date here.
default_as_of = st.session_state.get("qc_as_of", date.today())
as_of = st.date_input("Evaluate freshness as of", value=default_as_of,
                      help="The reference date the freshness gate compares the "
                           "latest week against. The QC demo sets this to its "
                           "calibrated as-of automatically.")

# The crosswalk is real master data now — NOT derived from POS. When none is
# uploaded the gate is skipped honestly (banner + status stays green-but-honest);
# a POS-derived stand-in was worse than no gate because it reported success.
xwalk = ws.read_reference("item_crosswalk")
pos_df = ws.read_canonical("pos") if ws.has_canonical("pos") else None
if xwalk is None:
    st.info("No item crosswalk uploaded — the crosswalk-match gate is skipped. "
            "Upload one under **Master data** above to get your unmatched-items list.")

for stream in STREAM_ORDER:
    if not ws.has_canonical(stream):
        continue
    df = ws.read_canonical(stream)
    report = quality.run_qc(stream, df, crosswalk=xwalk, pos=pos_df, as_of=as_of)
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
