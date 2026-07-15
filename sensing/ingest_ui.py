"""Ingestion layer — the v3 upload-and-map replacement for v2's fixed contracts.

Responsibilities:
  * read an uploaded CSV/XLSX/parquet into a DataFrame (with header hygiene),
  * sniff likely mappings (your column -> canonical field) by header name and
    content type, keeping ID-like codes out of the numeric bucket,
  * coerce types (dates including configurable retailer week labels, thousands
    separators, unit selection) onto the canonical schema,
  * flag wide-format exports and help the UI unpivot them.

The engine never sees any of this — it only reads canonical tables from the
workspace. Swapping this file for lake pipelines is the v3 -> v2 graduation.

Public surface (kept backward-compatible across S1 -> S2):
  read_upload, suggest_mapping, apply_mapping, is_wide_format, unpivot_wide,
  _coerce_week  (existing)
  WeekCalendar, suggest_wide_id_cols, missing_required, source_signature,
  looks_wm_labels  (new in S2)
"""

from __future__ import annotations

import hashlib
import io
import re
import warnings
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Any

import pandas as pd

from .config import CANONICAL_SCHEMAS


# --------------------------------------------------------------------------- #
# File reading + header hygiene
# --------------------------------------------------------------------------- #

def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Rename empty/whitespace/duplicate headers to safe, unique names.

    Real exports ship blank columns ("", " ", "Unnamed: 3") and the occasional
    duplicate header. Left alone these break selectboxes and mapping. We rename
    them deterministically to ``column_<n>`` without touching good headers.
    """
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for i, c in enumerate(df.columns):
        name = "" if c is None else str(c).strip()
        if name == "" or name.lower().startswith("unnamed:"):
            name = f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df = df.copy()
    df.columns = new_cols
    return df


def read_upload(name: str, content: bytes) -> pd.DataFrame:
    """Read raw upload bytes into a DataFrame based on the file extension."""
    lower = name.lower()
    if lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content), dtype=object)
    elif lower.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(content))
    else:
        # default: CSV, keep everything as string first so we can coerce cleanly
        # (this preserves ID codes with leading zeros).
        df = pd.read_csv(io.BytesIO(content), dtype=str)
    return _clean_headers(df)


# --------------------------------------------------------------------------- #
# Header-name aliases used by the sniffer
# --------------------------------------------------------------------------- #

_ALIASES: dict[str, list[str]] = {
    "item_id": ["item", "sku", "article", "upc", "gtin", "product", "material"],
    "region": ["region", "location", "market", "dist", "dc", "store", "geo"],
    "week": ["week", "date", "period", "day", "wk", "fiscal"],
    "units_sold": ["units_sold", "pos", "sold", "sales_units", "qty", "sell_out"],
    "on_hand_units": ["on_hand", "onhand", "inventory", "stock", "oh"],
    "in_transit_units": ["in_transit", "intransit", "transit", "pipeline"],
    "units_shipped": ["shipped", "ship_qty", "sell_in", "orders", "shipment"],
    "plan_units": ["plan", "forecast", "consensus", "demand_plan"],
    "plan_version_date": ["version", "snapshot", "asof", "plan_date"],
    "promo_flag": ["promo", "deal", "feature", "display", "event"],
}

# Retailer week label like "WM Week 2226" / "WK 2501" / "Wm2226".
_WMLABEL_RE = re.compile(r"w(?:m|k)?\s*(?:week\s*)?\d{3,4}", re.I)
_WM_RE = re.compile(r"w(?:m|k)?\s*(?:week\s*)?(\d{2})(\d{2})", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def _to_datetime_quiet(values, **kwargs):
    """pd.to_datetime with the noisy 'Could not infer format' UserWarning muted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce", **kwargs)


def looks_wm_labels(series: pd.Series) -> bool:
    """True if a column's values look like retailer week labels (WM Week 2226)."""
    sample = series.dropna().astype(str).str.strip().head(30)
    if sample.empty:
        return False
    return bool(sample.str.contains(_WMLABEL_RE).mean() > 0.5)


def _looks_date(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).str.strip().head(30)
    if sample.empty:
        return False
    if sample.str.contains(_WMLABEL_RE).mean() > 0.5:
        return True  # retailer week labels
    parsed = _to_datetime_quiet(sample)
    return bool(parsed.notna().mean() > 0.7)


def _looks_id_code(series: pd.Series) -> bool:
    """Numeric-looking values that are really identifiers, not quantities.

    Leading-zero digit strings ('001234') and hyphen/alpha-mixed codes must not
    be treated as numeric — coercing them to numbers destroys the identifier.
    """
    sample = series.dropna().astype(str).str.strip().head(30)
    if sample.empty:
        return False
    # leading-zero pure-digit strings are codes (a real quantity isn't '007')
    if sample.str.match(r"0\d+").mean() > 0.3:
        return True
    return False


def _looks_numeric(series: pd.Series) -> bool:
    if _looks_id_code(series):
        return False
    sample = (series.dropna().astype(str).str.strip()
              .str.replace(",", "", regex=False).head(30))
    if sample.empty:
        return False
    parsed = pd.to_numeric(sample, errors="coerce")
    return bool(parsed.notna().mean() > 0.7)


def _role_of(series: pd.Series) -> str:
    if _looks_date(series):
        return "date"
    if _looks_id_code(series):
        return "id"
    if _looks_numeric(series):
        return "numeric"
    return "id"


# --------------------------------------------------------------------------- #
# Auto-mapping
# --------------------------------------------------------------------------- #

def suggest_mapping(df: pd.DataFrame, stream: str) -> dict[str, str | None]:
    """Return canonical_field -> source_column best guesses (None if unknown)."""
    fields = CANONICAL_SCHEMAS[stream]["fields"]
    cols = list(df.columns)
    norm_cols = {c: _norm(c) for c in cols}
    roles = {c: _role_of(df[c]) for c in cols}
    used: set[str] = set()
    mapping: dict[str, str | None] = {}

    for canon, meta in fields.items():
        want_role = meta["role"]
        best: str | None = None
        best_score = 0.0
        for col in cols:
            if col in used:
                continue
            score = 0.0
            nc = norm_cols[col]
            # exact/alias header match (header intent beats content sniffing)
            if nc == canon:
                score += 3.0
            for alias in _ALIASES.get(canon, []):
                if alias in nc:
                    score += 1.5
                    break
            # role agreement
            if roles[col] == want_role:
                score += 1.0
            elif want_role == "id" and roles[col] != "date":
                score += 0.2
            if score > best_score:
                best_score, best = score, col
        if best is not None and best_score >= 1.0:
            mapping[canon] = best
            used.add(best)
        else:
            mapping[canon] = None
    return mapping


def missing_required(stream: str, mapping: dict[str, str | None]) -> list[str]:
    """Canonical fields that are required but currently unmapped/blank."""
    fields = CANONICAL_SCHEMAS[stream]["fields"]
    return [c for c, m in fields.items() if m["required"] and not mapping.get(c)]


# --------------------------------------------------------------------------- #
# Wide-format handling
# --------------------------------------------------------------------------- #

def is_wide_format(df: pd.DataFrame) -> bool:
    """Heuristic: many columns whose *names* look like week/date labels."""
    date_like = sum(1 for c in df.columns if _looks_date(pd.Series([c])))
    return date_like >= 4


def suggest_wide_id_cols(df: pd.DataFrame) -> list[str]:
    """Columns that are NOT date-like headers — the melt id_vars candidates."""
    return [c for c in df.columns if not _looks_date(pd.Series([c]))]


def unpivot_wide(df: pd.DataFrame, id_cols: list[str], value_name: str) -> pd.DataFrame:
    """Melt a wide (weeks-as-columns) export to long format."""
    value_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(id_vars=id_cols, value_vars=value_cols,
                   var_name="week", value_name=value_name)
    return long


# --------------------------------------------------------------------------- #
# Week calendar (configurable retailer fiscal weeks)
# --------------------------------------------------------------------------- #

@dataclass
class WeekCalendar:
    """How to interpret retailer week labels like 'WM Week 2226'.

    Defaults reproduce S1 behaviour exactly (label ``yyww`` -> 20YY, week WW
    counted from the Monday on/of Jan 1). Retailers whose fiscal year starts
    elsewhere (e.g. Walmart's ~February start) set ``fiscal_year_start_month``.
    """

    label_format: str = "yyww"        # first two digits = fiscal year, last two = week
    century: int = 2000               # 25 -> 2025
    fiscal_year_start_month: int = 1
    fiscal_year_start_day: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "WeekCalendar":
        if not d:
            return cls()
        allowed = {f for f in cls().__dict__}
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def week1_monday(self, year: int) -> date:
        anchor = date(year, self.fiscal_year_start_month, self.fiscal_year_start_day)
        return anchor - timedelta(days=anchor.weekday())


# --------------------------------------------------------------------------- #
# Coercion
# --------------------------------------------------------------------------- #

def _coerce_week(series: pd.Series, week_calendar: WeekCalendar | None = None) -> pd.Series:
    """Coerce a date-ish column to Monday-anchored weekly timestamps.

    Handles ISO dates, US dates, *mixed formats within one column*, and retailer
    week labels like 'WM Week 2226'. ``week_calendar`` controls fiscal-year
    interpretation of the labels; its default reproduces S1 behaviour.
    """
    cal = week_calendar or WeekCalendar()

    def one(v: Any):
        if pd.isna(v):
            return pd.NaT
        s = str(v).strip()
        m = _WM_RE.search(s)
        if m and "w" in s.lower():
            yy, ww = int(m.group(1)), int(m.group(2))
            year = cal.century + yy
            try:
                monday = cal.week1_monday(year)
                return pd.Timestamp(monday + timedelta(weeks=max(0, ww - 1)))
            except ValueError:
                return pd.NaT
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(s, errors="coerce")

    parsed = series.map(one)
    # snap to Monday of the week (W-SUN period start)
    return _to_datetime_quiet(parsed).dt.to_period("W-SUN").dt.start_time


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def apply_mapping(
    df: pd.DataFrame,
    stream: str,
    mapping: dict[str, str | None],
    unit_multiplier: float = 1.0,
    week_calendar: WeekCalendar | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Produce a canonical DataFrame from a source frame + a mapping.

    ``unit_multiplier`` converts eaches<->cases per file (e.g. 12).
    ``week_calendar`` controls retailer-week interpretation (optional).
    Returns (canonical_df, warnings)."""
    fields = CANONICAL_SCHEMAS[stream]["fields"]
    warns: list[str] = []
    out = pd.DataFrame()

    for canon, meta in fields.items():
        src = mapping.get(canon)
        if src is None or src not in df.columns:
            if meta["required"]:
                warns.append(f"Required field '{canon}' is unmapped.")
            continue
        col = df[src]
        role = meta["role"]
        if role == "date":
            out[canon] = _coerce_week(col, week_calendar)
        elif role == "numeric":
            vals = _coerce_numeric(col)
            if canon in ("units_sold", "units_shipped", "on_hand_units",
                         "in_transit_units"):
                vals = vals * unit_multiplier
            out[canon] = vals
        else:  # id
            out[canon] = col.astype(str).str.strip()

    # drop rows missing any required field
    req = [c for c, m in fields.items() if m["required"] and c in out.columns]
    before = len(out)
    if req:
        out = out.dropna(subset=req)
    if len(out) < before:
        warns.append(f"Dropped {before - len(out)} rows missing required fields.")

    return out.reset_index(drop=True), warns


# --------------------------------------------------------------------------- #
# Saved-mapping source signature
# --------------------------------------------------------------------------- #

def source_signature(name: str, df: pd.DataFrame) -> str:
    """Stable key for "this kind of export", so re-uploads pre-fill their mapping.

    Based on the *normalized header set* (order-independent) so that the same
    export template re-uploaded for a different month — regardless of filename —
    reuses its saved mapping. The filename is not part of identity, only the
    columns are.
    """
    headers = "|".join(sorted(_norm(c) for c in df.columns))
    return "hdr:" + hashlib.sha1(headers.encode("utf-8")).hexdigest()[:12]
