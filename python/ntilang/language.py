"""TileLang-style syntax markers. Decorated kernel bodies are parsed, never executed."""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from dataclasses import dataclass
from functools import wraps
from typing import Callable

from . import layout as layout
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


@dataclass(frozen=True)
class SymbolicDim:
    """Symbolic dimension placeholder created by T.dynamic(name, dtype)."""

    name: str
    dtype: str = "int32"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("T.dynamic requires a non-empty dimension name")
        object.__setattr__(self, "dtype", str(DType(self.dtype)))


@dataclass(frozen=True)
class SymbolicTensorSpec:
    """Tensor annotation containing one or more T.dynamic symbolic dimensions."""

    shape: tuple[int | SymbolicDim, ...]
    dtype: str


class GemmWarpPolicy:
    """TileLang-compatible GEMM warp partitioning policy constants."""

    Square = 0
    FullRow = 1
    FullCol = 2


def dynamic(name: str, dtype: str = "int32") -> SymbolicDim:
    """Create a symbolic tensor dimension specialized automatically at @jit runtime."""
    return SymbolicDim(name, dtype)


def infinity(dtype: str = "float32") -> float:
    """Return positive infinity for a floating-point dtype."""
    canonical = str(DType(dtype))
    if canonical not in ("float16", "bfloat16", "float32", "float64"):
        raise ValueError(f"T.infinity requires a floating-point dtype, got {canonical!r}")
    return float("inf")


def Tensor(shape, dtype="float32", data=None, scope=None) -> TensorType | SymbolicTensorSpec:
    if data is not None or scope not in (None, "global"):
        raise ValueError("Tensor pointer bindings and non-global parameter scopes require further lowering")
    dims = (shape,) if isinstance(shape, (builtins.int, SymbolicDim)) else tuple(shape)
    resolved_dtype = DType(dtype)
    if any(isinstance(d, SymbolicDim) for d in dims):
        for dim in dims:
            if not isinstance(dim, SymbolicDim) and (type(dim) is not builtins.int or dim <= 0):
                raise ValueError("Static dimensions alongside T.dynamic must be positive integers")
        return SymbolicTensorSpec(dims, str(resolved_dtype))
    return TensorType(dims, resolved_dtype)


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
    specializations: tuple = ()

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
    "alloc_local",
    "alloc_var",
    "use_swizzle",
    "annotate_layout",
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


def _is_direct_kernel(function: Callable) -> bool:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return False
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    body = tree.body[0].body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body or not isinstance(body[-1], ast.With) or len(body[-1].items) != 1:
        return False
    ctx = body[-1].items[0].context_expr
    if not isinstance(ctx, ast.Call):
        return False
    func = ctx.func
    return (isinstance(func, ast.Attribute) and func.attr == "Kernel") or (
        isinstance(func, ast.Name) and func.id == "Kernel"
    )


def _has_symbolic_annotations(program: PrimFunc) -> bool:
    for ann in program.function.__annotations__.values():
        if isinstance(ann, SymbolicTensorSpec):
            return True
    for _, val in program.annotation_locals:
        if isinstance(val, (SymbolicDim, SymbolicTensorSpec)):
            return True
    if program.function.__closure__:
        for cell in program.function.__closure__:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, (SymbolicDim, SymbolicTensorSpec)):
                return True
    return False


def jit(
    function=None,
    *,
    out_idx=None,
    target="auto",
    target_host=None,
    execution_backend=None,
    verbose=False,
    pass_configs=None,
    compile_flags=None,
):
    """Compile a TileLang-style kernel or kernel factory directly to standalone CuTe DSL."""
    from .compat import resolve_target
    from .compiler import DynamicJitKernel, compile

    del target_host, execution_backend, verbose, compile_flags
    frame = inspect.currentframe()
    caller_locals = frame.f_back.f_locals.copy() if frame is not None and frame.f_back is not None else {}
    del frame

    def decorate(factory):
        if _is_direct_kernel(factory):
            program = PrimFunc(factory, _annotation_bindings(factory, caller_locals))
            if _has_symbolic_annotations(program):
                return DynamicJitKernel(program, target=target, out_idx=out_idx, pass_configs=pass_configs)
            return compile(program, target=resolve_target(target), out_idx=out_idx, pass_configs=pass_configs)

        cache = {}

        @wraps(factory)
        def wrapped(*args, **kwargs):
            resolved = resolve_target(target)
            key = (
                args,
                tuple(sorted(kwargs.items())),
                resolved,
                out_idx
                if isinstance(out_idx, builtins.int)
                else tuple(out_idx)
                if out_idx is not None
                else None,
            )
            if key in cache:
                return cache[key]
            program = factory(*args, **kwargs)
            if isinstance(program, PrimFunc) and _has_symbolic_annotations(program):
                kernel = DynamicJitKernel(
                    program, target=resolved, out_idx=out_idx, pass_configs=pass_configs
                )
            else:
                kernel = compile(program, target=resolved, out_idx=out_idx, pass_configs=pass_configs)
            cache[key] = kernel
            return kernel

        return wrapped

    return decorate(function) if function is not None else decorate
