"""TileLang-style syntax markers. Decorated kernel bodies are parsed, never executed."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Callable

from .ir import TensorType

float16 = "float16"
bfloat16 = "bfloat16"
float32 = "float32"
int32 = "int32"


def Tensor(shape, dtype="float32") -> TensorType:
    return TensorType(tuple(shape), dtype)


def ceildiv(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("ceildiv requires a positive divisor")
    return -(-a // b)


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
    "copy",
    "clear",
    "fill",
    "gemm",
    "exp",
    "exp2",
    "sqrt",
    "maximum",
    "minimum",
    "cast",
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
