"""TileLang-style syntax markers. Decorated kernel bodies are parsed, never executed."""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from dataclasses import dataclass
from functools import wraps
from typing import Callable

from .ir import DTYPE_ALIASES, DTYPES, TensorType


class DType(str):
    """A dtype name that also denotes a scalar cast inside parsed kernels."""

    def __new__(cls, value):
        for builtin, name in ((builtins.bool, "bool"), (builtins.int, "int32"), (builtins.float, "float32")):
            if value is builtin:
                value = name
                break
        if type(value) not in (str, DType):
            raise TypeError("dtype requires a supported name or Python scalar type")
        value = DTYPE_ALIASES.get(value, value)
        if value not in DTYPES:
            raise ValueError(f"Unsupported dtype {value!r}")
        return super().__new__(cls, value)

    @property
    def bits(self):
        return DTYPES[self] * 8

    @property
    def bytes(self):
        return DTYPES[self]

    @property
    def itemsize(self):
        return self.bytes

    @property
    def lanes(self):
        return 1

    @property
    def type_code(self):
        if self == "bool":
            return 6
        if self == "bfloat16":
            return 4
        return 1 if self.startswith("uint") else 0 if self.startswith("int") else 2

    def __call__(self, value):
        raise RuntimeError(f"T.{self}(value) is a scalar cast inside a @T.prim_func body")


DTYPE_NAMES = {**{name: name for name in DTYPES}, **DTYPE_ALIASES}
for _dtype_name, _canonical_dtype in DTYPE_NAMES.items():
    globals()[_dtype_name] = DType(_canonical_dtype)
del _dtype_name, _canonical_dtype


dtype = DType


def get_tvm_dtype(value):
    return value if type(value) is DType else DType(value)


def Tensor(shape, dtype="float32", data=None, scope=None) -> TensorType:
    if data is not None or scope not in (None, "global"):
        raise ValueError("Tensor pointer bindings and non-global parameter scopes require further lowering")
    return TensorType((shape,) if type(shape) is builtins.int else tuple(shape), DType(dtype))


def ceildiv(lhs: int, rhs: int, span=None) -> int:
    if type(lhs) is not builtins.int or type(rhs) is not builtins.int:
        raise TypeError("ceildiv specialization arguments must be integers")
    if span is not None:
        raise ValueError("Explicit source span objects require parser integration")
    if rhs == 0:
        raise ValueError("ceildiv requires a nonzero divisor")
    return (lhs + rhs - 1) // rhs


cdiv = ceildiv


def align_up(x: int, y: int) -> int:
    return ceildiv(x, y) * y


def index_to_coordinates(index, shape):
    """Convert an integer index to row-major coordinates using floor arithmetic."""
    if type(index) is not builtins.int:
        raise TypeError("index_to_coordinates requires an integer index")
    if type(shape) not in (tuple, list):
        raise TypeError("Coordinate shape requires a tuple or list")
    if any(type(size) is not builtins.int or size <= 0 for size in shape):
        raise ValueError("Coordinate extents must be positive integers")
    result = []
    for size in reversed(shape):
        result.append(index % size)
        index //= size
    return list(reversed(result))


@dataclass(frozen=True)
class PrimFunc:
    function: Callable
    annotation_locals: tuple = ()

    @property
    def __name__(self):
        return self.function.__name__

    def __call__(self, *args, **kwargs):
        raise TypeError("Compile a prim_func with ntilang.compile() before calling it")


@dataclass(frozen=True, eq=False)
class Macro:
    function: Callable
    annotation_locals: tuple = ()

    @property
    def __name__(self):
        return self.function.__name__

    def __call__(self, *args, **kwargs):
        raise TypeError("T.macro calls are expanded inside a compiled prim_func or another macro")


class Ref:
    """Macro parameter annotation for a mutable scalar, element, or buffer region."""


def _annotation_bindings(function, caller):
    # Deferred annotations may mention factory arguments that are absent from
    # the function's bytecode closure. Capture just those referenced names.
    names = set()
    for annotation in function.__annotations__.values():
        if isinstance(annotation, str):
            names.update(
                n.id for n in ast.walk(ast.parse(annotation, mode="eval")) if isinstance(n, ast.Name)
            )
    try:
        source = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(source):
            if isinstance(node, ast.AnnAssign):
                names.update(n.id for n in ast.walk(node.annotation) if isinstance(n, ast.Name))
    except (OSError, TypeError):
        pass  # The frontend reports source availability when compilation is requested.
    return tuple((name, caller[name]) for name in names if name in caller)


def prim_func(function: Callable) -> PrimFunc:
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_locals if frame is not None and frame.f_back is not None else {}
        bindings = _annotation_bindings(function, caller)
    finally:
        del frame
    return PrimFunc(function, bindings)


def macro(func=None):
    def decorate(function):
        frame = inspect.currentframe()
        try:
            caller = frame.f_back.f_locals if frame is not None and frame.f_back is not None else {}
            return Macro(function, _annotation_bindings(function, caller))
        finally:
            del frame

    if func is None:
        return decorate
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_locals if frame is not None and frame.f_back is not None else {}
        return Macro(func, _annotation_bindings(func, caller))
    finally:
        del frame


def _syntax_operation(name):
    def operation(*args, **kwargs):
        raise RuntimeError(f"T.{name} is syntax inside a @T.prim_func body")

    operation.__name__ = name
    return operation


_MARKER_NAMES = {}
for _name in (
    "Kernel",
    "Parallel",
    "grid",
    "serial",
    "Serial",
    "Pipelined",
    "unroll",
    "Unroll",
    "alloc_shared",
    "alloc_fragment",
    "alloc_var",
    "loop_break",
    "break_loop",
    "continue_loop",
    "print",
    "device_assert",
    "Assert",
    "likely",
    "copy",
    "transpose",
    "clamp",
    "clear",
    "fill",
    "gemm",
    "reduce",
    "reduce_sum",
    "reduce_abssum",
    "reduce_max",
    "reduce_absmax",
    "reduce_min",
    "reduce_bitand",
    "reduce_bitor",
    "reduce_bitxor",
    "cumsum",
    "cummax",
    "cumsum_fragment",
    "cummax_fragment",
    "exp",
    "exp2",
    "sqrt",
    "pow",
    "fmod",
    "atan2",
    "copysign",
    "hypot",
    "nextafter",
    "ldexp",
    "ieee_add",
    "ieee_sub",
    "ieee_mul",
    "ieee_fmaf",
    "ieee_frcp",
    "ieee_fsqrt",
    "ieee_frsqrt",
    "ieee_fdiv",
    "fma",
    "fmul",
    "__exp",
    "__exp10",
    "__log",
    "__log2",
    "__log10",
    "__sin",
    "__cos",
    "__tan",
    "fast_rcp",
    "exp10",
    "log",
    "log2",
    "log10",
    "log1p",
    "rsqrt",
    "erf",
    "sigmoid",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "abs",
    "floor",
    "ceil",
    "trunc",
    "round",
    "nearbyint",
    "isnan",
    "isinf",
    "isfinite",
    "maximum",
    "minimum",
    "max",
    "min",
    "cast",
    "reinterpret",
    "popcount",
    "clz",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "shift_left",
    "shift_right",
    "floordiv",
    "floormod",
    "truncdiv",
    "truncmod",
    "Select",
    "if_then_else",
):
    _marker = _syntax_operation(_name)
    globals()[_name] = _marker
    _MARKER_NAMES[_marker] = _name
del _name, _marker


def jit(function=None, *, target="sm_80"):
    """Compile a kernel factory when it is called with specialization constants."""
    from .compiler import compile

    def decorate(factory):
        @wraps(factory)
        def wrapped(*args, **kwargs):
            return compile(factory(*args, **kwargs), target=target)

        return wrapped

    return decorate(function) if function is not None else decorate
