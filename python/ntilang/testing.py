"""Serial NumPy evaluation of the IR for debugging its mathematical semantics.

This evaluator does not simulate GPU scheduling, races, or Tensor Core rounding.
NumPy is an optional testing dependency and is imported only when reference runs.
"""

from __future__ import annotations

import itertools
import operator

from .compiler import CompiledKernel


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
    ops = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "%": operator.mod,
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
        "maximum": np.maximum,
        "minimum": np.minimum,
    }

    def read(name, indices):
        data = buffers[name]
        return data[indices] if all(0 <= i < s for i, s in zip(indices, data.shape)) else data.dtype.type(0)

    def write(name, indices, value):
        data = buffers[name]
        if all(0 <= i < s for i, s in zip(indices, data.shape)):
            data[indices] = value

    def expr(e):
        if e.op == "const":
            return e.value
        if e.op == "var":
            return variables[e.value]
        if e.op == "load":
            return read(e.value, tuple(expr(x) for x in e.args))
        args = [expr(x) for x in e.args]
        if e.op == "cast":
            return np.dtype(e.value).type(args[0])
        if e.op == "and":
            return all(args)
        if e.op == "or":
            return any(args)
        return ops[e.op](*args)

    def statements(body):
        for stmt in body:
            op, args = stmt.op, stmt.args
            if op == "alloc":
                b = kernel.ir.buffer_map[args[0]]
                buffers[b.name] = np.empty(b.type.shape, dtype=b.type.dtype)
            elif op == "fill":
                buffers[args[0]].fill(expr(args[1]))
            elif op == "let":
                variables[args[0]] = expr(args[1])
            elif op == "store":
                write(args[0], tuple(expr(x) for x in args[1]), expr(args[2]))
            elif op == "copy":
                src, dst = args
                src_origin, dst_origin = (tuple(expr(x) for x in r.origin) for r in (src, dst))
                # Materialize the source before modifying a destination.
                values = [
                    (coord, read(src.buffer, tuple(o + i for o, i in zip(src_origin, coord))))
                    for coord in np.ndindex(src.shape)
                ]
                for coord, value in values:
                    write(dst.buffer, tuple(o + i for o, i in zip(dst_origin, coord)), value)
            elif op == "gemm":
                a, b, c, ta, tb = args
                av, bv = buffers[a], buffers[b]
                av, bv = av.T if ta else av, bv.T if tb else bv
                buffers[c] += av.astype(np.float32) @ bv.astype(np.float32)
            elif op == "parallel":
                names, shape, inner = args
                for coord in np.ndindex(shape):
                    variables.update(zip(names, coord))
                    statements(inner)
            elif op in ("serial", "unroll"):
                names, extent, inner = args
                for value in range(*extent):
                    variables[names[0]] = value
                    statements(inner)
            else:
                raise ValueError(f"Unknown IR operation {op}")

    for block in itertools.product(*(range(n) for n in kernel.ir.grid)):
        variables.clear()
        variables.update(zip(kernel.ir.block_vars, block))
        statements(kernel.ir.body)
