"""Demand-sensing POC engine + ingestion package.

Public surface used by the app pages:
    Workspace, open_workspace     -- workspace.py
    RunConfig, CANONICAL_SCHEMAS  -- config.py
    run_pipeline, RunResult       -- engine.py
    demo_data, ingest_ui, quality -- submodules
"""

from .config import (
    RunConfig, CANONICAL_SCHEMAS, STREAM_ORDER,
    REFERENCE_SCHEMAS, REFERENCE_ORDER,
)
from .workspace import Workspace, open_workspace
from .engine import run_pipeline, RunResult

__all__ = [
    "RunConfig", "CANONICAL_SCHEMAS", "STREAM_ORDER",
    "REFERENCE_SCHEMAS", "REFERENCE_ORDER",
    "Workspace", "open_workspace",
    "run_pipeline", "RunResult",
]
