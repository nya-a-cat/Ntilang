"""TileLang-compatible layout descriptor helpers (ntilang.layout / T.layout)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwizzledLayout:
    """Explicit swizzled shared-memory layout hint for T.annotate_layout."""

    buffer: object
    is_a: bool = False
    is_b: bool = False


def make_swizzled_layout(buffer, is_a: bool = False, is_b: bool = False) -> SwizzledLayout:
    """Create a swizzled layout annotation compatible with tilelang.layout.make_swizzled_layout."""
    return SwizzledLayout(buffer=buffer, is_a=is_a, is_b=is_b)


__all__ = ["SwizzledLayout", "make_swizzled_layout"]
