"""Workspace — the POC's single-file "lake".

A workspace is one DuckDB file plus a small set of conventions:

* **Snapshots**  Each accepted upload lands as an immutable table named
  ``snap_<stream>_<timestamp>_<suffix>`` and a row is written to
  ``snapshots`` recording its lineage. Nothing is ever silently overwritten.
* **Canonical tables**  A "Rebuild dataset" action materialises the latest
  snapshot per stream into ``canonical_<stream>`` at ``item_id x region x week``.
  The engine reads ONLY these tables (never file paths) so the later swap to
  lake ingestion is a one-layer change.
* **Runs**  Each ``run_pipeline`` execution persists its outputs and a JSON
  manifest under ``runs`` / ``run_outputs`` for full reproducibility.

Everything here is deliberately dependency-light: duckdb + pandas.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, date
from typing import Any

import duckdb
import pandas as pd

from .config import STREAM_ORDER, CANONICAL_SCHEMAS


class Workspace:
    """Thin wrapper over a DuckDB file with snapshot/canonical/run helpers."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._init_metadata()

    # ---- connection -------------------------------------------------------- #
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Open a fresh connection. Callers should use it as a context manager
        or close it; DuckDB single-file access is single-writer."""
        return duckdb.connect(self.path)

    def _init_metadata(self) -> None:
        with self.connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id   VARCHAR PRIMARY KEY,
                    stream        VARCHAR,
                    table_name    VARCHAR,
                    source_name   VARCHAR,
                    row_count     BIGINT,
                    uploaded_at   TIMESTAMP,
                    date_min      DATE,
                    date_max      DATE
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id      VARCHAR PRIMARY KEY,
                    as_of       DATE,
                    created_at  TIMESTAMP,
                    manifest    VARCHAR
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS mappings (
                    stream      VARCHAR,
                    signature   VARCHAR,
                    payload     VARCHAR,
                    updated_at  TIMESTAMP,
                    PRIMARY KEY (stream, signature)
                )
                """
            )

    # ---- snapshots --------------------------------------------------------- #
    def add_snapshot(
        self,
        stream: str,
        df: pd.DataFrame,
        source_name: str,
        suffix: str = "a",
    ) -> str:
        """Persist ``df`` as an immutable snapshot table and record lineage.

        Returns the snapshot_id. ``df`` is expected to already be in canonical
        column names (the ingestion layer conforms it before calling this)."""
        ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        snapshot_id = f"{stream}_{ts}_{suffix}"
        table_name = f"snap_{snapshot_id}"

        date_min = date_max = None
        if "week" in df.columns and len(df):
            wk = pd.to_datetime(df["week"], errors="coerce")
            date_min = wk.min().date() if wk.notna().any() else None
            date_max = wk.max().date() if wk.notna().any() else None

        with self.connect() as con:
            con.register("incoming", df)
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM incoming')
            con.unregister("incoming")
            con.execute(
                "INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?)",
                [
                    snapshot_id,
                    stream,
                    table_name,
                    source_name,
                    int(len(df)),
                    datetime.now(),
                    date_min,
                    date_max,
                ],
            )
        return snapshot_id

    def list_snapshots(self, stream: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM snapshots"
        params: list[Any] = []
        if stream:
            q += " WHERE stream = ?"
            params.append(stream)
        q += " ORDER BY uploaded_at DESC"
        with self.connect() as con:
            return con.execute(q, params).df()

    def latest_snapshot(self, stream: str) -> str | None:
        df = self.list_snapshots(stream)
        return df.iloc[0]["table_name"] if len(df) else None

    # ---- canonical tables -------------------------------------------------- #
    @staticmethod
    def _grain_keys(stream: str) -> list[str]:
        """De-duplication grain for a stream: its id + date role fields.

        For demand_plan this includes ``plan_version_date`` so distinct plan
        vintages for the same week survive stacking rather than colliding.
        """
        fields = CANONICAL_SCHEMAS.get(stream, {}).get("fields", {})
        return [f for f, m in fields.items() if m.get("role") in ("id", "date")]

    def rebuild_canonical(self) -> dict[str, int]:
        """Materialise ALL snapshots per stream into canonical_<stream>.

        Snapshots are stacked (multi-file: months uploaded across several files)
        and de-duplicated on the stream grain with **last-write-wins** by upload
        time — so a corrected re-upload of the same weeks overrides the old rows
        while genuinely new weeks simply extend coverage.

        Returns a dict of stream -> row_count for streams that were rebuilt.
        Streams without any snapshot are dropped (they degrade gracefully)."""
        built: dict[str, int] = {}
        with self.connect() as con:
            for stream in STREAM_ORDER:
                con.execute(f"DROP TABLE IF EXISTS canonical_{stream}")
                snaps = self.list_snapshots(stream)
                if not len(snaps):
                    continue
                # oldest first so a higher __ord means a more recent upload
                snaps = snaps.sort_values(["uploaded_at", "snapshot_id"])
                tables = list(snaps["table_name"])

                # keys must exist in the snapshot tables; fall back defensively
                cols = set(
                    con.execute(f'SELECT * FROM "{tables[-1]}" LIMIT 0').df().columns
                )
                keys = [k for k in self._grain_keys(stream) if k in cols]
                if not keys:
                    keys = list(cols)

                union = " UNION ALL BY NAME ".join(
                    f'SELECT *, {i} AS __ord FROM "{t}"' for i, t in enumerate(tables)
                )
                part = ", ".join(f'"{k}"' for k in keys)
                con.execute(
                    f"""
                    CREATE TABLE canonical_{stream} AS
                    SELECT * EXCLUDE (__ord, __rn) FROM (
                        SELECT *, ROW_NUMBER() OVER (
                            PARTITION BY {part} ORDER BY __ord DESC
                        ) AS __rn
                        FROM ({union})
                    ) WHERE __rn = 1
                    """
                )
                n = con.execute(
                    f"SELECT COUNT(*) FROM canonical_{stream}"
                ).fetchone()[0]
                built[stream] = int(n)
        return built

    def snapshot_coverage(self, stream: str) -> dict[str, Any]:
        """Combined coverage across all snapshots in a slot (pre-rebuild view)."""
        snaps = self.list_snapshots(stream)
        if not len(snaps):
            return {"snapshots": 0, "rows": 0, "date_min": None, "date_max": None}
        dmin = pd.to_datetime(snaps["date_min"], errors="coerce").min()
        dmax = pd.to_datetime(snaps["date_max"], errors="coerce").max()
        return {
            "snapshots": int(len(snaps)),
            "rows": int(snaps["row_count"].sum()),
            "date_min": None if pd.isna(dmin) else dmin.date(),
            "date_max": None if pd.isna(dmax) else dmax.date(),
        }

    def has_canonical(self, stream: str) -> bool:
        with self.connect() as con:
            rows = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name = ?",
                [f"canonical_{stream}"],
            ).fetchall()
        return len(rows) > 0

    def read_canonical(self, stream: str) -> pd.DataFrame:
        if not self.has_canonical(stream):
            return pd.DataFrame()
        with self.connect() as con:
            df = con.execute(f"SELECT * FROM canonical_{stream}").df()
        if "week" in df.columns:
            df["week"] = pd.to_datetime(df["week"], errors="coerce")
        return df

    def canonical_status(self) -> dict[str, dict[str, Any]]:
        """Summary used by the Home page: which streams are loaded + coverage."""
        out: dict[str, dict[str, Any]] = {}
        for stream in STREAM_ORDER:
            if not self.has_canonical(stream):
                out[stream] = {"loaded": False, "rows": 0}
                continue
            df = self.read_canonical(stream)
            info: dict[str, Any] = {"loaded": True, "rows": int(len(df))}
            if "week" in df.columns and len(df):
                info["week_min"] = df["week"].min()
                info["week_max"] = df["week"].max()
            out[stream] = info
        return out

    # ---- runs -------------------------------------------------------------- #
    def save_run(self, run_id: str, as_of: date, manifest: dict, outputs: dict[str, pd.DataFrame]) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
            con.execute(
                "INSERT INTO runs VALUES (?,?,?,?)",
                [run_id, as_of, datetime.now(), json.dumps(manifest, default=str)],
            )
            for name, df in outputs.items():
                table = f"run_{run_id}_{name}"
                con.execute(f'DROP TABLE IF EXISTS "{table}"')
                con.register("incoming", df)
                con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM incoming')
                con.unregister("incoming")

    def list_runs(self) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(
                "SELECT * FROM runs ORDER BY created_at DESC"
            ).df()

    # ---- saved column mappings -------------------------------------------- #
    def save_mapping(self, stream: str, signature: str, payload: dict) -> None:
        """Persist a your-column->canonical-field mapping (plus unit / week
        calendar / wide config) keyed by stream + source signature, so a
        matching re-upload is one click. Upserts (last write wins)."""
        with self.connect() as con:
            con.execute(
                "DELETE FROM mappings WHERE stream = ? AND signature = ?",
                [stream, signature],
            )
            con.execute(
                "INSERT INTO mappings VALUES (?,?,?,?)",
                [stream, signature, json.dumps(payload, default=str), datetime.now()],
            )

    def get_mapping(self, stream: str, signature: str) -> dict | None:
        """Return a previously saved mapping payload, or None."""
        with self.connect() as con:
            rows = con.execute(
                "SELECT payload FROM mappings WHERE stream = ? AND signature = ?",
                [stream, signature],
            ).fetchall()
        if not rows:
            return None
        return json.loads(rows[0][0])

    def read_run_output(self, run_id: str, name: str) -> pd.DataFrame:
        table = f"run_{run_id}_{name}"
        with self.connect() as con:
            exists = con.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
                [table],
            ).fetchall()
            if not exists:
                return pd.DataFrame()
            df = con.execute(f'SELECT * FROM "{table}"').df()
        if "week" in df.columns:
            df["week"] = pd.to_datetime(df["week"], errors="coerce")
        return df


def open_workspace(path: str) -> Workspace:
    """Convenience factory."""
    return Workspace(path)
