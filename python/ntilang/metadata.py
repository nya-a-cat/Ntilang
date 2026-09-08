"""Source-level metadata for compiler-owned buffers and scalar types."""

from dataclasses import dataclass

from .ir import Buffer, Expr
from .language import DType


@dataclass(frozen=True)
class ScopeQuery:
    buffer: Buffer

    @property
    def value(self):
        return (
            self.buffer.source_scope
            or {"global": "global", "shared": "shared.dyn", "fragment": "local.fragment"}[self.buffer.space]
        )


def buffer_attribute(buffer, name):
    if name in ("shape", "strides"):
        values = buffer.type.shape if name == "shape" else buffer.strides
        return tuple(Expr("const", value=value) for value in values)
    if name == "dtype":
        return DType(buffer.type.dtype)
    if name == "scope":
        return ScopeQuery(buffer)
    if name == "elem_offset":
        return Expr("const", value=0)
    defaults = {"data_alignment": 64, "offset_factor": 1, "buffer_type": 1, "axis_separators": ()}
    if name in defaults:
        return defaults[name]
    raise ValueError(f"Buffer metadata attribute {name!r} requires further parser integration")


def dtype_attribute(dtype, name):
    if name in ("bits", "bytes", "itemsize", "lanes", "type_code"):
        return getattr(dtype, name)
    raise ValueError(f"Unsupported dtype metadata attribute {name!r}")
