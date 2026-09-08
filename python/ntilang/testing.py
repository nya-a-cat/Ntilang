"""Serial NumPy evaluation of the IR for debugging its mathematical semantics.

This evaluator does not simulate GPU scheduling, races, or Tensor Core rounding.
NumPy is an optional testing dependency and is imported only when reference runs.
"""

from __future__ import annotations

import itertools
import math
import operator

from .compiler import CompiledKernel
from .ir import Expr, integer_limits
from .scalar import (
    BINARY_MATH_OPS,
    CHOICE_OPS,
    INTEGER_DIVISION_OPS,
    ROUNDING_OPS,
    TRANSCENDENTAL_OPS,
    body_types,
    expression_dtype,
    operand_dtype,
)


def reference(kernel: CompiledKernel, *arrays):
    """Execute the tile IR serially, writing the supplied output arrays in place."""
    import numpy as np

    if len(arrays) != len(kernel.ir.parameters):
        raise TypeError("Incorrect number of arrays")
    if any(b.type.dtype == "bfloat16" for b in (*kernel.ir.parameters, *kernel.ir.buffers)):
        raise TypeError("The NumPy evaluator does not support bfloat16")
    buffers = {}
    for param, array in zip(kernel.ir.parameters, arrays):
        if array.shape != param.type.shape or array.dtype != np.dtype(param.type.dtype):
            raise ValueError(f"{param.name} requires {param.type.shape} {param.type.dtype}")
        buffers[param.name] = array
    variables = {}
    variable_types = {}
    buffer_types = kernel.ir.buffer_map
    ops = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "%": operator.mod,
        "&": operator.and_,
        "|": operator.or_,
        "^": operator.xor,
        "invert": operator.invert,
        "<<": operator.lshift,
        ">>": operator.rshift,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
        "neg": operator.neg,
        "pos": operator.pos,
        "not": operator.not_,
        "exp": np.exp,
        "exp2": np.exp2,
        "sqrt": np.sqrt,
        "pow": np.power,
        "fmod": np.fmod,
        "atan2": np.arctan2,
        "copysign": np.copysign,
        "exp10": lambda x: np.power(type(x)(10), x),
        "log": np.log,
        "log2": np.log2,
        "log10": np.log10,
        "log1p": np.log1p,
        "rsqrt": lambda x: type(x)(1) / np.sqrt(x),
        "erf": math.erf,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "asinh": np.arcsinh,
        "acosh": np.arccosh,
        "atanh": np.arctanh,
        "abs": np.abs,
        "floor": np.floor,
        "ceil": np.ceil,
        "trunc": np.trunc,
        "round": np.rint,
        "nearbyint": np.rint,
        "isnan": np.isnan,
        "isinf": np.isinf,
        "isfinite": np.isfinite,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "max": np.fmax,
        "min": np.fmin,
    }

    class LoopBreak(Exception):
        pass

    class LoopContinue(Exception):
        pass

    def read(name, indices):
        data = buffers[name]
        return data[indices] if all(0 <= i < s for i, s in zip(indices, data.shape)) else data.dtype.type(0)

    def cast(value, dtype):
        return np.asarray(value).astype(dtype, casting="unsafe", copy=False)[()]

    def write(name, indices, value):
        data = buffers[name]
        if all(0 <= i < s for i, s in zip(indices, data.shape)):
            data[indices] = cast(value, data.dtype)

    def expr(e):
        if e.op == "const":
            return e.value
        if e.op == "var":
            return variables[e.value]
        if e.op == "load":
            return read(e.value, tuple(expr(x) for x in e.args))
        if e.op in CHOICE_OPS:
            dtype = expression_dtype(e, buffer_types, variable_types)
            condition = expr(e.args[0])
            if e.op == "if_then_else":
                return cast(expr(e.args[1] if condition else e.args[2]), dtype)
            when_true, when_false = expr(e.args[1]), expr(e.args[2])
            return cast(when_true if condition else when_false, dtype)
        args = [expr(x) for x in e.args]
        if e.op == "cast":
            return cast(args[0], e.value)
        if e.op == "and":
            return all(args)
        if e.op == "or":
            return any(args)
        dtype = expression_dtype(e, buffer_types, variable_types)
        arg_dtype = operand_dtype(e, buffer_types, variable_types)
        args = [cast(value, arg_dtype) for value in args]
        if e.op == "pow_integer":
            result = cast(1, dtype) if e.value == 0 else args[0]
            for _ in range(1, e.value):
                result = cast(result * args[0], dtype)
            return result
        if e.op in BINARY_MATH_OPS:
            compute_dtype = "float32" if dtype in ("float16", "bfloat16") else dtype
            return cast(ops[e.op](*(cast(value, compute_dtype) for value in args)), dtype)
        if e.op in TRANSCENDENTAL_OPS:
            value = cast(args[0], "float32" if dtype in ("float16", "bfloat16") else dtype)
            if e.op == "sigmoid":
                exponential = cast(np.exp(-value), dtype)
                denominator = cast(cast(1, dtype) + exponential, dtype)
                return cast(cast(1, dtype) / denominator, dtype)
            return cast(ops[e.op](value), dtype)
        if e.op in ROUNDING_OPS:
            if arg_dtype.startswith(("int", "uint")) or arg_dtype == "bool":
                return args[0]
            if e.op == "round_away":
                value = args[0]
                magnitude = np.abs(value)
                integral = np.floor(magnitude)
                if np.isfinite(magnitude) and magnitude - integral >= 0.5:
                    integral += 1
                return cast(np.copysign(integral, value), dtype)
        if e.op in INTEGER_DIVISION_OPS:
            a, b = map(int, args)
            if b == 0:
                raise ValueError("Integer division requires a nonzero divisor")
            low, high = integer_limits(dtype)
            if a == low and b == -1:
                raise ValueError("Signed minimum divided by -1 overflows the integer dtype")
            if e.op == "ceildiv":
                numerator = cast(cast(a + b, dtype) - cast(1, dtype), dtype)
                result = int(numerator) // b
            elif e.op == "//":
                result = a // b
            elif e.op == "%":
                result = a % b
            else:
                quotient = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                result = quotient if e.op == "truncdiv" else a - quotient * b
            return cast(result, dtype)
        return cast(ops[e.op](*args), dtype)

    def statements(body):
        nonlocal variable_types
        for stmt in body:
            op, args = stmt.op, stmt.args
            if op == "alloc":
                b = kernel.ir.buffer_map[args[0]]
                buffers[b.name] = np.empty(b.type.shape, dtype=b.type.dtype)
            elif op == "pass":
                continue
            elif op == "break":
                raise LoopBreak
            elif op == "continue":
                raise LoopContinue
            elif op == "if":
                condition, then_body, else_body = args
                merged = body_types((stmt,), variable_types, buffer_types)
                variable_types = merged.copy()
                statements(then_body if expr(condition) else else_body)
                variable_types = merged
            elif op == "fill":
                buffers[args[0]].fill(cast(expr(args[1]), buffers[args[0]].dtype))
            elif op == "let":
                dtype = variable_types.get(args[0]) or expression_dtype(args[1], buffer_types, variable_types)
                variables[args[0]] = cast(expr(args[1]), dtype)
                variable_types[args[0]] = dtype
            elif op == "declare":
                if args[1] == "bfloat16":
                    raise TypeError("The NumPy evaluator does not support bfloat16")
                variables[args[0]] = expr(args[2])
                variable_types[args[0]] = args[1]
            elif op == "assign":
                variables[args[0]] = expr(args[1])
            elif op == "while":
                condition, inner = args
                before_types = variable_types.copy()
                while expr(condition):
                    try:
                        statements(inner)
                    except LoopContinue:
                        continue
                    except LoopBreak:
                        break
                    finally:
                        variable_types = before_types.copy()
            elif op == "store":
                write(args[0], tuple(expr(x) for x in args[1]), expr(args[2]))
            elif op == "copy":
                src, dst = args
                src_origin, dst_origin = (tuple(expr(x) for x in r.origin) for r in (src, dst))

                def indices(origin, axes, coord):
                    return tuple(o + (0 if axis is None else coord[axis]) for o, axis in zip(origin, axes))

                # Materialize the source before modifying a destination.
                values = [
                    (coord, read(src.buffer, indices(src_origin, src.axes, coord)))
                    for coord in np.ndindex(src.shape)
                ]
                for coord, value in values:
                    write(dst.buffer, indices(dst_origin, dst.axes, coord), value)
            elif op == "gemm":
                a, b, c, ta, tb = args
                av, bv = buffers[a], buffers[b]
                av, bv = av.T if ta else av, bv.T if tb else bv
                buffers[c] += av.astype(np.float32) @ bv.astype(np.float32)
            elif op == "reduce":
                src, dst, kind, dim, clear, nan_propagate = args
                output = buffers[dst]
                values = buffers[src].astype(output.dtype)
                if kind in ("abssum", "absmax") and output.dtype.kind not in ("u", "b"):
                    values = np.fmax(values, -values)
                propagate = nan_propagate and output.dtype == np.dtype("float16")
                combine = {
                    "sum": np.add,
                    "abssum": np.add,
                    "max": np.maximum if propagate else np.fmax,
                    "absmax": np.maximum if propagate else np.fmax,
                    "min": np.minimum if propagate else np.fmin,
                    "bitand": np.bitwise_and,
                    "bitor": np.bitwise_or,
                    "bitxor": np.bitwise_xor,
                }[kind]
                identity = 0
                integer = output.dtype.kind in ("i", "u", "b")
                if kind == "max":
                    identity = integer_limits(output.dtype.name)[0] if integer else -np.inf
                elif kind == "min":
                    identity = integer_limits(output.dtype.name)[1] if integer else np.inf
                elif kind == "bitand":
                    identity = -1 if output.dtype.kind == "i" else integer_limits(output.dtype.name)[1]
                reduced = combine.reduce(
                    values,
                    axis=dim,
                    dtype=output.dtype,
                    initial=identity,
                    keepdims=output.ndim == values.ndim,
                )
                output[...] = reduced if clear else combine(output, reduced)
            elif op == "parallel":
                names, shape, inner = args
                before_types = variable_types.copy()
                for coord in np.ndindex(shape):
                    variable_types = {**before_types, **{name: "int32" for name in names}}
                    variables.update(zip(names, coord))
                    statements(inner)
                variable_types = before_types
            elif op in ("serial", "unroll"):
                names, extent, inner = args
                before_types = variable_types.copy()
                domain = tuple(expr(value) if isinstance(value, Expr) else value for value in extent)
                for value in range(*domain):
                    variable_types = {**before_types, names[0]: "int32"}
                    variables[names[0]] = value
                    try:
                        statements(inner)
                    except LoopContinue:
                        continue
                    except LoopBreak:
                        break
                variable_types = before_types
            else:
                raise ValueError(f"Unknown IR operation {op}")

    for block in itertools.product(*(range(n) for n in kernel.ir.grid)):
        variables.clear()
        variable_types = {name: "int32" for name in kernel.ir.block_vars}
        variables.update(zip(kernel.ir.block_vars, block))
        statements(kernel.ir.body)
