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

    @property
    def bits(self):
        return 1 if self == "bool" else DTYPES[self] * 8

    @property
    def bytes(self):
        return DTYPES[self]

    def __call__(self, value):
        raise RuntimeError(f"T.{self}(value) is a scalar cast inside a @T.prim_func body")


DTYPE_NAMES = {**{name: name for name in DTYPES}, **DTYPE_ALIASES}
for _dtype_name, _canonical_dtype in DTYPE_NAMES.items():
    globals()[_dtype_name] = DType(_canonical_dtype)
del _dtype_name, _canonical_dtype


def Tensor(shape, dtype="float32") -> TensorType:
    return TensorType(tuple(shape), dtype)


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


@dataclass(frozen=True)
class PrimFunc:
    function: Callable
    annotation_locals: tuple = ()

    @property
    def __name__(self):
        return self.function.__name__

    def __call__(self, *args, **kwargs):
        raise TypeError("Compile a prim_func with ntilang.compile() before calling it")


def prim_func(function: Callable) -> PrimFunc:
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
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_locals if frame is not None and frame.f_back is not None else {}
        bindings = tuple((name, caller[name]) for name in names if name in caller)
    finally:
        del frame
    return PrimFunc(function, bindings)


def _syntax_operation(name):
    def operation(*args, **kwargs):
        raise RuntimeError(f"T.{name} is syntax inside a @T.prim_func body")

    operation.__name__ = name
    return operation


_MARKER_NAMES = {}
for _name in (
    "Kernel",
    "Parallel",
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
    "copy",
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
    "exp",
    "exp2",
    "sqrt",
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
