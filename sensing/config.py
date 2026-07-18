"""Configuration objects for the demand-sensing POC.

Everything the engine needs to run one as-of pass is captured in ``RunConfig``.
Canonical field definitions for each upload stream live in ``CANONICAL_SCHEMAS``
so the ingestion layer (ingest_ui) and the engine agree on column names.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
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


# --------------------------------------------------------------------------- #
# Reference (dimension) schemas — parallel to CANONICAL_SCHEMAS, NOT streams.
#
# A reference is current-state master data: dateless, unit-less, keyed by a
# source id -> canonical id. It lands in the workspace as a ``dim_*`` table
# (replace-on-upload), is never part of ``STREAM_ORDER`` / ``CANONICAL_SCHEMAS``,
# and the engine never reads it — only ``quality.py`` consumes the crosswalk, to
# measure item match rate. Deliberately **no ``date`` role anywhere**: every
# canonical stream requires a ``week``, and _grain_keys / snapshot_coverage /
# freshness all depend on that; a dimension has no week, so it must stay out of
# the stream contracts. Same ``fields``/``role`` shape as a stream, so the
# ingest mapper (suggest_mapping / apply_mapping / missing_required) works
# against it unchanged.
#
# The crosswalk MEASURES match rate only — it never remaps ``item_id`` at
# rebuild. ``units_per_case`` / ``successor_item_id`` are carried for the v2
# graduation (harmonize.py / M1) and are unused in S9.
# --------------------------------------------------------------------------- #

REFERENCE_SCHEMAS: dict[str, dict[str, Any]] = {
    "item_crosswalk": {
        "label": "Item crosswalk (retailer item → SKU)",
        "required": True,          # design §3.1: crosswalk is required
        "fields": {
            "source_item_id":    {"role": "id", "required": True},
            "item_id":           {"role": "id", "required": True},
            "units_per_case":    {"role": "numeric", "required": False},
            "successor_item_id": {"role": "id", "required": False},
        },
    },
    "location_map": {
        "label": "Location → region map",
        "required": False,
        "fields": {
            "source_location_id": {"role": "id", "required": True},
            "region":             {"role": "id", "required": True},
        },
    },
}

REFERENCE_ORDER = ["item_crosswalk", "location_map"]


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
