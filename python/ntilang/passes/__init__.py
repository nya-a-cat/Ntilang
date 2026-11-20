"""Middle-end compiler analysis passes shared across frontends and backends."""

from __future__ import annotations

from .layout_inference import (
    analyze_mmas,
    collect_workspaces,
    copy_snapshots,
    infer_fragment_layouts,
    parallel_accesses,
    validate_shared_memory_budget,
    walk,
)
from .smem_hazards import HazardTracker, smem_footprint

__all__ = [
    "HazardTracker",
    "analyze_mmas",
    "collect_workspaces",
    "copy_snapshots",
    "infer_fragment_layouts",
    "parallel_accesses",
    "smem_footprint",
    "validate_shared_memory_budget",
    "walk",
]
