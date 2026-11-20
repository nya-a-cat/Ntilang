"""Drop-in TileLang compatibility helpers: target auto-detection, out_idx, and sys.modules patching."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from .ir import CompileError, ScalarParameter


def resolve_target(target: str | None = "auto") -> str:
    """Resolve TileLang-style target strings ('auto', 'cuda', 'cuda:sm_80', 'sm_80') without adding deps."""
    if target is None or target in ("auto", "cuda"):
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    major, minor = torch.cuda.get_device_capability()
                    return f"sm_{major}{minor}"
            except Exception:
                pass
        return "sm_80"
    if isinstance(target, str) and target.startswith("cuda:"):
        return target[5:]
    return target


def normalize_out_idx(out_idx: int | Sequence[int] | None, parameters) -> tuple[int, ...]:
    """Normalize TileLang out_idx (int or sequence of ints, supporting negative indices)."""
    if out_idx is None:
        return ()
    raw = (out_idx,) if isinstance(out_idx, int) else tuple(out_idx)
    n_params = len(parameters)
    normalized: list[int] = []
    seen: set[int] = set()
    for index in raw:
        if type(index) is not int:
            raise TypeError(f"out_idx entries must be integers, got {index!r}")
        resolved = index + n_params if index < 0 else index
        if not 0 <= resolved < n_params:
            raise ValueError(f"out_idx {index} is out of range for {n_params} kernel parameters")
        if isinstance(parameters[resolved], ScalarParameter):
            raise ValueError(
                f"out_idx {index} cannot refer to scalar parameter {parameters[resolved].name!r}"
            )
        if resolved in seen:
            raise ValueError(f"Duplicate out_idx {index}")
        seen.add(resolved)
        normalized.append(resolved)
    return tuple(normalized)


class _TileLangPatchContext:
    """Allows ntilang.patch_tilelang() to be used both directly and as a context manager."""

    def __init__(self):
        import ntilang
        import ntilang.language as language
        import ntilang.layout as layout

        self._keys = ("tilelang", "tilelang.language", "tilelang.layout")
        self._previous = {key: sys.modules.get(key) for key in self._keys}
        sys.modules["tilelang"] = ntilang
        sys.modules["tilelang.language"] = language
        sys.modules["tilelang.layout"] = layout
        self._active = True

    def restore(self) -> None:
        if not self._active:
            return
        for key, prev in self._previous.items():
            if prev is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = prev
        self._active = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.restore()
        return False


def patch_tilelang() -> _TileLangPatchContext:
    """Redirect `import tilelang` / `import tilelang.language` to Ntilang (supports `with` context)."""
    return _TileLangPatchContext()


def torch_dtype_for(dtype_name: str, torch_module):
    """Map an Ntilang canonical dtype name to a PyTorch dtype."""
    mapping = {
        "bool": torch_module.bool,
        "int8": torch_module.int8,
        "int16": torch_module.int16,
        "int32": torch_module.int32,
        "int64": torch_module.int64,
        "uint8": torch_module.uint8,
        "uint16": getattr(torch_module, "uint16", None),
        "uint32": getattr(torch_module, "uint32", None),
        "uint64": getattr(torch_module, "uint64", None),
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "float64": torch_module.float64,
    }
    resolved = mapping.get(dtype_name)
    if resolved is None:
        raise CompileError(f"PyTorch does not expose dtype {dtype_name!r} for out_idx allocation")
    return resolved


__all__ = ["normalize_out_idx", "patch_tilelang", "resolve_target", "torch_dtype_for"]
