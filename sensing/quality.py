"""Quality gates (M2), surfaced as upload validation cards.

Every accepted stream gets a QC card summarising the full M2 gate suite
(design §3.3, v1 §3.3):

  * **Freshness / SLA** — is the latest week recent enough vs. a reference date?
    This is the check that decides whether weekly sensing is viable at all, so
    it is surfaced first.
  * **Date coverage & gaps** — contiguous weeks, with the missing weeks listed.
  * **Negative values** — impossible measures, with the offending rows attached.
  * **Impossible / range values** — promo_flag ∉ {0,1}; a demand-plan vintage
    dated on/after its target week (a leakage red flag); grossly implausible
    magnitudes.
  * **Duplicate rows** — repeats on the stream grain.
  * **Coverage shift** — a retailer adding/dropping stores or SKUs looks exactly
    like a demand change; sudden week-over-week changes in the active
    item×region series set are flagged with the affected weeks.
  * **Crosswalk match** — item match-rate vs. the crosswalk, with a downloadable
    unmatched-items list.
  * **Phantom inventory** — an N-consecutive-week run of (on_hand > 0 AND zero
    sell-out), attributed per series.

Gates **quarantine-and-flag rather than silently impute**: every firing check
that identifies bad rows attaches the specific offenders (``chk.data``) as the
planner's cleanup to-do. Nothing here mutates the data — rebuild remains the
sole materialisation step.

``run_qc`` returns a structured report the UI renders. The ``QCReport`` /
``QCCheck`` shapes and the three status values ("pass"/"warn"/"fail") are frozen
— new gates are added as extra checks/attachments, never by changing shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from .config import CANONICAL_SCHEMAS


@dataclass
class QCCheck:
    name: str
    status: str          # "pass" | "warn" | "fail"
    detail: str
    data: Any = None     # optional attachment (offending rows / cleanup list)


@dataclass
class QCReport:
    stream: str
    rows: int
    checks: list[QCCheck] = field(default_factory=list)

    @property
    def worst(self) -> str:
        order = {"pass": 0, "warn": 1, "fail": 2}
        return max((c.status for c in self.checks), key=lambda s: order[s],
                   default="pass")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _grain_keys(stream: str, df: pd.DataFrame) -> list[str]:
    """Stream de-dup grain (id + date role fields) present in ``df``.

    Mirrors workspace._grain_keys so demand_plan keeps distinct
    ``plan_version_date`` vintages rather than reading them as duplicates.
    """
    fields = CANONICAL_SCHEMAS.get(stream, {}).get("fields", {})
    keys = [f for f, m in fields.items() if m.get("role") in ("id", "date")]
    return [k for k in keys if k in df.columns]


def _weeks(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["week"], errors="coerce")


def _measure_cols(stream: str, df: pd.DataFrame) -> list[str]:
    """Numeric *measure* columns for a stream (excludes flags/ids)."""
    fields = CANONICAL_SCHEMAS.get(stream, {}).get("fields", {})
    out = []
    for f, m in fields.items():
        if m.get("role") != "numeric" or f not in df.columns:
            continue
        if f == "promo_flag":          # a flag, not a magnitude
            continue
        out.append(f)
    # fall back to any numeric column if the schema is unknown
    if not out:
        out = [c for c in df.select_dtypes("number").columns]
    return out


# --------------------------------------------------------------------------- #
# 1. Freshness / SLA  (surfaced first — decides whether sensing is viable)
# --------------------------------------------------------------------------- #

def _freshness(df: pd.DataFrame, as_of: date | None, sla_weeks: int) -> QCCheck:
    if "week" not in df.columns:
        return QCCheck("Freshness / SLA", "pass", "No week column to check.")
    wk = _weeks(df).dropna()
    if wk.empty:
        return QCCheck("Freshness / SLA", "fail", "No parseable weeks — cannot assess freshness.")
    latest = wk.max().date()
    if as_of is None:
        return QCCheck("Freshness / SLA", "pass",
                       f"Latest week {latest:%Y-%m-%d}; no reference date supplied "
                       f"— freshness not assessed.")
    if isinstance(as_of, datetime):
        as_of = as_of.date()
    gap_days = (as_of - latest).days
    gap_weeks = gap_days / 7.0
    if gap_weeks <= sla_weeks:
        return QCCheck("Freshness / SLA", "pass",
                       f"Latest week {latest:%Y-%m-%d} is {gap_weeks:.0f} week(s) "
                       f"behind {as_of:%Y-%m-%d} — within the {sla_weeks}-week SLA.")
    status = "warn" if gap_weeks <= 2 * sla_weeks else "fail"
    return QCCheck("Freshness / SLA", status,
                   f"Latest week {latest:%Y-%m-%d} is {gap_weeks:.0f} week(s) behind "
                   f"{as_of:%Y-%m-%d} (SLA {sla_weeks}w) — feed is stale; weekly "
                   f"sensing on this stream may not be viable until refreshed.")


# --------------------------------------------------------------------------- #
# 2. Date coverage & gaps
# --------------------------------------------------------------------------- #

def _date_coverage(df: pd.DataFrame) -> QCCheck:
    if "week" not in df.columns or df["week"].isna().all():
        return QCCheck("Date coverage", "fail", "No parseable dates found.")
    wk = _weeks(df).dropna()
    present = pd.Index(sorted(wk.dt.normalize().unique()))
    if len(present) <= 1:
        return QCCheck("Date coverage", "pass", f"{len(present)} week present.")
    full = pd.date_range(present.min(), present.max(), freq="7D")
    missing = full.difference(present)
    if len(missing):
        frame = pd.DataFrame({"missing_week": missing.strftime("%Y-%m-%d")})
        return QCCheck("Date coverage", "warn",
                       f"{len(present)} weeks present, {len(missing)} week(s) "
                       f"missing between {present.min():%Y-%m-%d} and "
                       f"{present.max():%Y-%m-%d} — download the gap list.",
                       data=frame)
    return QCCheck("Date coverage", "pass",
                   f"{len(present)} contiguous weeks "
                   f"({present.min():%Y-%m-%d} to {present.max():%Y-%m-%d}).")


# --------------------------------------------------------------------------- #
# 3. Negative values  (attaches offenders)
# --------------------------------------------------------------------------- #

def _negatives(stream: str, df: pd.DataFrame) -> QCCheck:
    cols = [c for c in df.select_dtypes("number").columns]
    if not cols:
        return QCCheck("Negative values", "pass", "No numeric columns to check.")
    mask = (df[cols] < 0).any(axis=1)
    n = int(mask.sum())
    if n:
        return QCCheck("Negative values", "warn",
                       f"{n} row(s) with a negative measure — quarantine & fix.",
                       data=df[mask].copy())
    return QCCheck("Negative values", "pass", "No negative values.")


# --------------------------------------------------------------------------- #
# 4. Impossible / range values  (attaches offenders with a reason)
# --------------------------------------------------------------------------- #

def _range_values(stream: str, df: pd.DataFrame) -> QCCheck:
    offenders = []

    # promo_flag must be 0/1
    if "promo_flag" in df.columns:
        bad = df[~df["promo_flag"].isin([0, 1])]
        for _, r in bad.iterrows():
            offenders.append({**r.to_dict(), "reason": "promo_flag not in {0,1}"})

    # demand-plan vintage dated on/after its target week => leakage
    if stream == "demand_plan" and {"week", "plan_version_date"} <= set(df.columns):
        wk = _weeks(df)
        pv = pd.to_datetime(df["plan_version_date"], errors="coerce")
        leak = df[pv >= wk]
        for _, r in leak.iterrows():
            offenders.append({**r.to_dict(),
                              "reason": "plan_version_date on/after target week (leakage)"})

    # grossly implausible magnitudes (very conservative: > 100x column median)
    for col in _measure_cols(stream, df):
        s = pd.to_numeric(df[col], errors="coerce")
        med = s[s > 0].median()
        if pd.notna(med) and med > 0:
            big = df[s > med * 100]
            for _, r in big.iterrows():
                offenders.append({**r.to_dict(),
                                  "reason": f"{col} implausibly large (>100x median)"})

    if offenders:
        frame = pd.DataFrame(offenders)
        return QCCheck("Impossible / range values", "warn",
                       f"{len(frame)} row(s) failed a range/validity rule "
                       f"— download the offenders.", data=frame)
    return QCCheck("Impossible / range values", "pass",
                   "All values within valid ranges.")


# --------------------------------------------------------------------------- #
# 5. Duplicate rows on the grain  (attaches offenders)
# --------------------------------------------------------------------------- #

def _duplicates(stream: str, df: pd.DataFrame) -> QCCheck:
    keys = _grain_keys(stream, df)
    if not keys:
        return QCCheck("Duplicate rows", "pass", "No grain keys to check.")
    mask = df.duplicated(subset=keys, keep=False)
    n_extra = int(df.duplicated(subset=keys).sum())
    if n_extra:
        return QCCheck("Duplicate rows", "warn",
                       f"{n_extra} duplicate row(s) on {'+'.join(keys)} "
                       f"— download the colliding rows.",
                       data=df[mask].sort_values(keys).copy())
    return QCCheck("Duplicate rows", "pass", "No duplicates on the grain.")


# --------------------------------------------------------------------------- #
# 6. Coverage-shift detection  (adds/drops in the active-series set)
# --------------------------------------------------------------------------- #

def _coverage_shift(df: pd.DataFrame, threshold: float) -> QCCheck:
    need = {"item_id", "region", "week"}
    if not need <= set(df.columns):
        return QCCheck("Coverage shift", "pass", "Not applicable to this stream.")
    wk = _weeks(df)
    tmp = df.assign(_wk=wk.dt.normalize()).dropna(subset=["_wk"])
    if tmp["_wk"].nunique() <= 1:
        return QCCheck("Coverage shift", "pass", "Too few weeks to compare.")
    tmp["_series"] = tmp["item_id"].astype(str) + "|" + tmp["region"].astype(str)
    by_week = {w: set(g["_series"]) for w, g in tmp.groupby("_wk")}
    weeks_sorted = sorted(by_week)

    rows = []
    prev_w = weeks_sorted[0]
    for w in weeks_sorted[1:]:
        prev, cur = by_week[prev_w], by_week[w]
        added, dropped = cur - prev, prev - cur
        churn = len(added) + len(dropped)
        base = max(1, len(prev))
        if churn and churn / base >= threshold:
            rows.append({
                "week": pd.Timestamp(w).strftime("%Y-%m-%d"),
                "n_added": len(added), "n_dropped": len(dropped),
                "added": ", ".join(sorted(added)) or "—",
                "dropped": ", ".join(sorted(dropped)) or "—",
            })
        prev_w = w

    if rows:
        frame = pd.DataFrame(rows)
        return QCCheck("Coverage shift", "warn",
                       f"{len(frame)} week(s) with a sudden change in the active "
                       f"series set (≥{threshold:.0%}) — a store/SKU add or drop "
                       f"can masquerade as a demand change. Download the affected weeks.",
                       data=frame)
    return QCCheck("Coverage shift", "pass",
                   "Active series set is stable week to week.")


# --------------------------------------------------------------------------- #
# 7. Crosswalk match
# --------------------------------------------------------------------------- #

def _crosswalk_match(df: pd.DataFrame, crosswalk: pd.DataFrame | None) -> QCCheck:
    if crosswalk is None or "item_id" not in df.columns:
        return QCCheck("Crosswalk match", "pass",
                       "No crosswalk supplied — skipped.")
    known = set(crosswalk["item_id"].astype(str))
    present = set(df["item_id"].astype(str))
    unmatched = sorted(present - known)
    rate = 1 - len(unmatched) / max(1, len(present))
    if unmatched:
        frame = pd.DataFrame({"unmatched_item_id": unmatched})
        return QCCheck("Crosswalk match", "warn",
                       f"{rate:.0%} matched; {len(unmatched)} unmatched item(s) "
                       f"— download the cleanup list.", data=frame)
    return QCCheck("Crosswalk match", "pass", "100% of items matched.")


# --------------------------------------------------------------------------- #
# 8. Phantom inventory  (N-consecutive-week run of on_hand>0 & zero sell-out)
# --------------------------------------------------------------------------- #

def _phantom_inventory(df: pd.DataFrame, pos: pd.DataFrame | None,
                       n_weeks: int) -> QCCheck:
    if "on_hand_units" not in df.columns or pos is None:
        return QCCheck("Phantom inventory", "pass", "Not applicable to this stream.")
    merged = df.merge(
        pos[["item_id", "region", "week", "units_sold"]],
        on=["item_id", "region", "week"], how="left")
    merged["week"] = pd.to_datetime(merged["week"], errors="coerce")
    merged = merged.sort_values(["item_id", "region", "week"])
    merged["_suspect"] = ((merged["on_hand_units"] > 0) &
                          (merged["units_sold"].fillna(0) == 0))

    runs = []
    for (item, region), g in merged.groupby(["item_id", "region"]):
        flags = g["_suspect"].tolist()
        weeks = g["week"].tolist()
        i = 0
        while i < len(flags):
            if flags[i]:
                j = i
                while j + 1 < len(flags) and flags[j + 1]:
                    j += 1
                length = j - i + 1
                if length >= n_weeks:
                    runs.append({
                        "item_id": item, "region": region,
                        "run_weeks": length,
                        "start_week": pd.Timestamp(weeks[i]).strftime("%Y-%m-%d"),
                        "end_week": pd.Timestamp(weeks[j]).strftime("%Y-%m-%d"),
                    })
                i = j + 1
            else:
                i += 1

    if runs:
        frame = pd.DataFrame(runs)
        return QCCheck("Phantom inventory", "warn",
                       f"{len(frame)} series with ≥{n_weeks} consecutive weeks of "
                       f"stock-on-hand but zero sell-out — treat availability as "
                       f"suspect. Download the offending series/weeks.", data=frame)
    return QCCheck("Phantom inventory", "pass",
                   f"No run of ≥{n_weeks} weeks with stock but zero sell-out.")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_qc(
    stream: str,
    df: pd.DataFrame,
    crosswalk: pd.DataFrame | None = None,
    pos: pd.DataFrame | None = None,
    *,
    as_of: date | None = None,
    sla_weeks: int = 2,
    phantom_weeks: int = 3,
    coverage_shift_threshold: float = 0.30,
) -> QCReport:
    """Run all applicable M2 gates for a stream and return a report.

    Backward-compatible: ``crosswalk`` and ``pos`` keep their positions; the new
    gates (freshness, coverage-shift, hardened phantom, range checks) are opt-in
    via keyword-only args that default to sensible values. Nothing here mutates
    ``df`` — checks flag and attach offenders only.
    """
    report = QCReport(stream=stream, rows=len(df))
    # Freshness first — it decides whether weekly sensing is viable.
    report.checks.append(_freshness(df, as_of, sla_weeks))
    report.checks.append(_date_coverage(df))
    report.checks.append(_negatives(stream, df))
    report.checks.append(_range_values(stream, df))
    report.checks.append(_duplicates(stream, df))
    report.checks.append(_coverage_shift(df, coverage_shift_threshold))
    report.checks.append(_crosswalk_match(df, crosswalk))
    if stream == "channel_inventory":
        report.checks.append(_phantom_inventory(df, pos, phantom_weeks))
    return report
