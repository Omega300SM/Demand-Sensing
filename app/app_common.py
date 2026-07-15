"""Shared helpers for the Streamlit pages.

Handles project-root path bootstrap (so `import sensing` works regardless of
how streamlit is launched), workspace session state, and small formatting
utilities. Pages call ``bootstrap()`` first, then the rest.
"""

from __future__ import annotations

import os
import sys


def project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = here
    for _ in range(5):
        if os.path.isdir(os.path.join(root, "sensing")):
            return root
        root = os.path.dirname(root)
    return here


def bootstrap() -> None:
    root = project_root()
    app_dir = os.path.join(root, "app")
    for p in (root, app_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


bootstrap()

import streamlit as st  # noqa: E402
from sensing import Workspace  # noqa: E402

DEFAULT_WS = os.path.join(project_root(), "workspace", "poc.duckdb")

SEVERITY_COLORS = {"high": "#c0392b", "medium": "#d68910", "low": "#7d8a99"}
STATUS_EMOJI = {"pass": "🟢", "warn": "🟡", "fail": "🔴"}


def get_workspace() -> Workspace:
    """Return the active workspace, creating the default one on first use."""
    path = st.session_state.get("ws_path", DEFAULT_WS)
    st.session_state["ws_path"] = path
    return Workspace(path)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def kfmt(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"


def require_canonical(ws, stream: str = "pos") -> bool:
    """Guard used by run/review pages; returns True if data is present."""
    if not ws.has_canonical(stream):
        st.warning(
            "No data loaded yet. Go to **Home** to load the demo dataset, "
            "or **Upload Data** to add your own exports."
        )
        return False
    return True


def latest_run_id(ws) -> str | None:
    runs = ws.list_runs()
    return runs.iloc[0]["run_id"] if len(runs) else None


def run_manifest(ws, run_id: str) -> dict | None:
    """Return the persisted JSON manifest for a run (config + snapshot dates),
    so a page can recompute the run reproducibly from exactly what it recorded."""
    import json

    runs = ws.list_runs()
    row = runs[runs["run_id"] == run_id]
    if not len(row):
        return None
    raw = row.iloc[0]["manifest"]
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
