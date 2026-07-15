"""Configuration objects for the demand-sensing POC.

Everything the engine needs to run one as-of pass is captured in ``RunConfig``.
Canonical field definitions for each upload stream live in ``CANONICAL_SCHEMAS``
so the ingestion layer (ingest_ui) and the engine agree on column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any


# --------------------------------------------------------------------------- #
# Canonical schemas — the target column names every stream is mapped onto.
# The ingestion layer maps a messy upload's columns onto these; the engine
# reads only these names from the workspace DuckDB.
# --------------------------------------------------------------------------- #

CANONICAL_SCHEMAS: dict[str, dict[str, Any]] = {
    "pos": {
        "label": "Retail POS (sell-out)",
        "required": True,
        "fields": {
            "item_id": {"role": "id", "required": True},
            "region": {"role": "id", "required": True},
            "week": {"role": "date", "required": True},
            "units_sold": {"role": "numeric", "required": True},
        },
    },
    "channel_inventory": {
        "label": "Channel inventory",
        "required": False,
        "fields": {
            "item_id": {"role": "id", "required": True},
            "region": {"role": "id", "required": True},
            "week": {"role": "date", "required": True},
            "on_hand_units": {"role": "numeric", "required": True},
            "in_transit_units": {"role": "numeric", "required": False},
        },
    },
    "shipments": {
        "label": "Shipments (sell-in)",
        "required": True,
        "fields": {
            "item_id": {"role": "id", "required": True},
            "region": {"role": "id", "required": True},
            "week": {"role": "date", "required": True},
            "units_shipped": {"role": "numeric", "required": True},
        },
    },
    "demand_plan": {
        "label": "Demand plan snapshots",
        "required": False,
        "fields": {
            "item_id": {"role": "id", "required": True},
            "region": {"role": "id", "required": True},
            "week": {"role": "date", "required": True},
            "plan_units": {"role": "numeric", "required": True},
            "plan_version_date": {"role": "date", "required": True},
        },
    },
    "promo": {
        "label": "Promo calendar",
        "required": False,
        "fields": {
            "item_id": {"role": "id", "required": True},
            "region": {"role": "id", "required": True},
            "week": {"role": "date", "required": True},
            "promo_flag": {"role": "numeric", "required": True},
        },
    },
}

STREAM_ORDER = ["pos", "channel_inventory", "shipments", "demand_plan", "promo"]


@dataclass
class RunConfig:
    """Parameters for a single sensing run."""

    as_of: date
    horizon_weeks: int = 8
    target_wos: float = 4.0            # target weeks-of-supply for translation
    order_cadence_weeks: int = 1       # how often the retailer places orders
    reaction_lag_weeks: int = 1        # lag between sell-out shift and reorder
    ml_enabled: bool = False           # optional ML tier toggle
    deviation_threshold: float = 0.15  # cumulative sensed-vs-plan alert threshold
    backtest_weeks: int = 26           # rolling-origin holdout length

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        d = dict(d)
        if isinstance(d.get("as_of"), str):
            d["as_of"] = date.fromisoformat(d["as_of"])
        return cls(**d)
