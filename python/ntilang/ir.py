"""Typed, backend-independent tile IR used by the Python compiler."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

DTYPES = {
    "bool": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}
DTYPE_ALIASES = {
    "short": "int16",
    "int": "int32",
    "uint": "uint32",
    "long": "int64",
    "half": "float16",
    "float": "float32",
    "double": "float64",
}


def integer_limits(dtype):
    if dtype == "bool":
        return 0, 1
    bits = DTYPES[dtype] * 8
    if dtype.startswith("uint"):
        return 0, 2**bits - 1
    if dtype.startswith("int"):
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    raise ValueError(f"{dtype} is not an integer dtype")


@dataclass(frozen=True)
class SourceLocation:
    filename: str
    line: int
    column: int = 0


class CompileError(ValueError):
    def __init__(self, message: str, location: SourceLocation | None = None):
        self.location = location
        prefix = f"{location.filename}:{location.line}:{location.column + 1}: " if location else ""
        super().__init__(prefix + message)


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        object.__setattr__(self, "dtype", str(DTYPE_ALIASES.get(self.dtype, self.dtype)))
        if not self.shape or any(type(n) is not int or n <= 0 for n in self.shape):
            raise CompileError("Tensor dimensions must be positive static integers")
        if self.dtype not in DTYPES:
            raise CompileError(f"Unsupported dtype {self.dtype!r}; supported: {', '.join(DTYPES)}")
        if prod(self.shape) > 2**31 - 1:
            raise CompileError("Tensor sizes exceeding signed 32-bit indexing are not supported")


@dataclass(frozen=True)
class Buffer:
    name: str
    type: TensorType
    space: str = "global"


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple[Expr, ...] = ()
    value: int | float | str | bool | None = None


@dataclass(frozen=True)
class Region:
    buffer: str
    origin: tuple[Expr, ...]
    shape: tuple[int, ...]
    axes: tuple[int | None, ...] = ()

    def __post_init__(self):
        if not self.axes:
            object.__setattr__(self, "axes", tuple(range(len(self.origin))))

    @property
    def extents(self):
        return tuple(1 if axis is None else self.shape[axis] for axis in self.axes)

    def is_full(self, buffer_shape):
        return self.extents == buffer_shape and all(x == Expr("const", value=0) for x in self.origin)


@dataclass(frozen=True)
class Statement:
    op: str
    args: tuple
    location: SourceLocation
    annotations: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class Kernel:
    name: str
    parameters: tuple[Buffer, ...]
    buffers: tuple[Buffer, ...]
    grid: tuple[int, ...]
    block_vars: tuple[str, ...]
    threads: int
    body: tuple[Statement, ...]
    source: str

    @property
    def buffer_map(self) -> dict[str, Buffer]:
        return {b.name: b for b in (*self.parameters, *self.buffers)}


@dataclass(frozen=True)
class Partition:
    """Row-major ownership: flat index = thread + register_slot * threads."""

    shape: tuple[int, ...]
    threads: int

    def __post_init__(self):
        TensorType(self.shape, "int32")
        if type(self.threads) is not int or not 1 <= self.threads <= 1024:
            raise CompileError("threads must be an integer in [1, 1024]")
        if self.slots * self.threads - 1 > 2**31 - 1:
            raise CompileError("Rounded tile partition can overflow signed 32-bit arithmetic")

    @property
    def slots(self) -> int:
        return (prod(self.shape) + self.threads - 1) // self.threads

    def coordinates(self, thread: int, slot: int) -> tuple[int, ...] | None:
        if not 0 <= thread < self.threads or not 0 <= slot < self.slots:
            raise IndexError("Invalid thread or register slot")
        flat = thread + slot * self.threads
        if flat >= prod(self.shape):
            return None
        coords = []
        for dim in reversed(self.shape):
            coords.append(flat % dim)
            flat //= dim
        return tuple(reversed(coords))
