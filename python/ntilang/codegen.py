"""Emit standalone NVIDIA CuTe DSL; no Ntilang runtime appears in the output."""

from __future__ import annotations

import ast
import re
from math import isfinite, isnan, prod

from .ir import DTYPES, CompileError, Expr, Kernel, Partition, integer_limits, loop_controls
from .scalar import (
    BINARY_NUMERIC_OPS,
    BITWISE_OPS,
    CHOICE_OPS,
    CLASSIFICATION_OPS,
    INTEGER_DIVISION_OPS,
    TRANSCENDENTAL_OPS,
    UNARY_MATH_OPS,
    body_types,
    expression_dtype,
    operand_dtype,
)

CUTLASS_TYPES = {
    "float16": "Float16",
    "bfloat16": "BFloat16",
    "float32": "Float32",
    "int32": "Int32",
    "int8": "Int8",
    "int16": "Int16",
    "int64": "Int64",
    "uint8": "Uint8",
    "uint16": "Uint16",
    "uint32": "Uint32",
    "uint64": "Uint64",
    "float64": "Float64",
    "bool": "Boolean",
}


def walk(body):
    for stmt in body:
        yield stmt
        if stmt.op in ("serial", "parallel", "unroll"):
            yield from walk(stmt.args[2])
        elif stmt.op == "if":
            yield from walk(stmt.args[1])
            yield from walk(stmt.args[2])
        elif stmt.op == "while":
            yield from walk(stmt.args[1])


def check_target(target):
    if not isinstance(target, str) or not re.fullmatch(
        r"sm_(80|86|87|89|90a?|100a?|101a?|103a?|110a?|120a?|121a?)", target
    ):
        raise CompileError("Use an explicit supported CUDA target, for example sm_80, sm_90a, or sm_100a")
    return target


class Emitter:
    def __init__(self, kernel: Kernel, target: str):
        self.kernel = kernel
        self.target = check_target(target)
        self.buffers = kernel.buffer_map
        self.lines: list[str] = []
        self.depth = 0
        self.counter = 0
        self.mmas = {}
        self.parallel = None
        self.control = None
        self.scalar_types = {name: "int32" for name in kernel.block_vars}
        self.active_snapshots = set()
        for stmt in walk(kernel.body):
            if stmt.op == "gemm":
                a, b, c, ta, tb = stmt.args
                ab, bb, cb = (self.buffers[n] for n in (a, b, c))
                if ab.space != "shared" or bb.space != "shared" or cb.space != "fragment":
                    raise CompileError("T.gemm requires shared A/B and a fragment accumulator", stmt.location)
                if any(len(x.type.shape) != 2 for x in (ab, bb, cb)):
                    raise CompileError("T.gemm requires rank-two tiles", stmt.location)
                am, ak = ab.type.shape[::-1] if ta else ab.type.shape
                bk, bn = bb.type.shape[::-1] if tb else bb.type.shape
                if ak != bk or cb.type.shape != (am, bn):
                    raise CompileError("T.gemm tile shapes do not match C += A @ B", stmt.location)
                if (
                    ab.type.dtype != bb.type.dtype
                    or ab.type.dtype not in ("float16", "bfloat16")
                    or cb.type.dtype != "float32"
                ):
                    raise CompileError(
                        "T.gemm supports float16/bfloat16 operands with float32 accumulation", stmt.location
                    )
                if kernel.threads % 32 or kernel.threads not in (32, 64, 128, 256):
                    raise CompileError("Tensor Core GEMM supports 32, 64, 128, or 256 threads", stmt.location)
                warps = kernel.threads // 32
                candidates = [
                    wm
                    for wm in (1, 2, 4, 8)
                    if wm <= warps and am % (16 * wm) == 0 and bn % (8 * (warps // wm)) == 0
                ]
                if not candidates or ak % 16:
                    raise CompileError(
                        "GEMM M/N must cover the warp arrangement and K must be a multiple of 16",
                        stmt.location,
                    )
                wm = max(candidates)
                plan = (am, bn, ak, wm, warps // wm, ab.type.dtype)
                if c in self.mmas and self.mmas[c] != plan:
                    raise CompileError(
                        "All GEMMs sharing an accumulator require the same MMA layout", stmt.location
                    )
                self.mmas[c] = plan
        self.fragment_layouts = self.infer_fragment_layouts()
        self.snapshots = set()
        self.reduction_workspaces = {}
        for stmt in walk(kernel.body):
            if stmt.op == "parallel":
                _, snapshots, _ = self.parallel_accesses(stmt)
                self.snapshots.update(snapshots)
            elif stmt.op == "copy":
                self.snapshots.update(self.copy_snapshots(*stmt.args))
            elif stmt.op == "reduce":
                src, dst = stmt.args[:2]
                key = (src, self.buffers[dst].type.dtype)
                if key not in self.reduction_workspaces:
                    self.reduction_workspaces[key] = f"_nt_reduce_workspace_{len(self.reduction_workspaces)}"
        shared_bytes = 0
        for buffer in kernel.buffers:
            for needed in (buffer.name in self.snapshots, buffer.space == "shared"):
                if not needed:
                    continue
                shared_bytes = (shared_bytes + 15) // 16 * 16
                shared_bytes += prod(buffer.type.shape) * DTYPES[buffer.type.dtype]
        for src, dtype in self.reduction_workspaces:
            shared_bytes = (shared_bytes + 15) // 16 * 16
            shared_bytes += prod(self.buffers[src].type.shape) * DTYPES[dtype]
        if shared_bytes > 48 * 1024:
            raise CompileError("Shared buffers and fragment communication exceed 48 KiB per block")

    def parallel_accesses(self, stmt):
        names, shape, body = stmt.args
        expected = tuple(Expr("var", value=n) for n in names)
        pointwise, snapshots, written = set(), set(), set()
        shared_used, shared_written, shared_cross_reads = set(), set(), set()

        def loads(value):
            if isinstance(value, Expr):
                if value.op == "load" and self.buffers[value.value].space == "fragment":
                    if self.buffers[value.value].type.shape == shape and value.args == expected:
                        pointwise.add(value.value)
                    else:
                        snapshots.add(value.value)
                elif value.op == "load" and self.buffers[value.value].space == "shared":
                    shared_used.add(value.value)
                    if self.buffers[value.value].type.shape != shape or value.args != expected:
                        shared_cross_reads.add(value.value)
                for arg in value.args:
                    loads(arg)
            elif isinstance(value, (tuple, list)):
                for arg in value:
                    loads(arg)

        for inner in walk(body):
            loads(inner.args)
            if inner.op == "store" and self.buffers[inner.args[0]].space == "fragment":
                written.add(inner.args[0])
            elif inner.op == "store" and self.buffers[inner.args[0]].space == "shared":
                shared_used.add(inner.args[0])
                shared_written.add(inner.args[0])
        if snapshots & written:
            raise CompileError(
                "Cross-element fragment reads require a separate source fragment from parallel writes",
                stmt.location,
            )
        if shared_cross_reads & shared_written:
            raise CompileError(
                "Cross-element shared reads require a separate source tile from parallel writes",
                stmt.location,
            )
        return pointwise | written, snapshots, bool(shared_used)

    def infer_fragment_layouts(self):
        # Connected pointwise operations and whole-tile copies share ownership.
        # Propagate before allocation so a fragment's earlier writes use the same
        # mapping as its later MMA consumer.
        parents = {b.name: b.name for b in self.kernel.buffers if b.space == "fragment"}

        def root(name):
            while parents[name] != name:
                name = parents[name]
            return name

        for stmt in walk(self.kernel.body):
            names = set()
            if stmt.op == "parallel":
                names, _, _ = self.parallel_accesses(stmt)
            elif stmt.op == "copy":
                if all(r.is_full(self.buffers[r.buffer].type.shape) for r in stmt.args):
                    names = {r.buffer for r in stmt.args if self.buffers[r.buffer].space == "fragment"}
            names = sorted(names)
            for name in names[1:]:
                parents[root(name)] = root(names[0])
        seeds = {}
        for name, plan in self.mmas.items():
            group = root(name)
            if group in seeds:
                previous = self.mmas[seeds[group]]
                if (plan[:2], plan[3:5]) != (previous[:2], previous[3:5]):
                    raise CompileError("Connected MMA fragments require a register layout conversion")
            else:
                seeds[group] = name
        return {name: seeds.get(root(name)) for name in parents}

    def fragment_slot(self, name, indices):
        buf = self.buffers[name]
        if self.parallel is None or self.parallel[0] != buf.type.shape:
            raise CompileError("Fragment indexing requires a matching T.Parallel shape")
        expected = tuple(Expr("var", value=n) for n in self.parallel[1])
        if indices != expected:
            raise CompileError("Fragment indexing requires the exact T.Parallel induction variables")
        return f"{self.buf(name)}[{self.parallel[2]}]"

    def loop_fragment(self, name):
        layout = self.fragment_layouts[name]
        if layout is None:
            slot, coords = self.loop_tile(self.buffers[name].type.shape)
            return slot, coords, 2
        slot = self.unique("mma_slot")
        self.emit(f"for {slot} in cutlass.range_constexpr(cute.size({self.buf(name)})):")
        self.depth += 1
        physical = iter(f"_nt_coords_{layout}[{slot}][{i}]" for i in range(2))
        coords = ["0" if size == 1 else next(physical) for size in self.buffers[name].type.shape]
        return slot, coords, 1

    def materialize_fragment(self, name):
        self.emit("cute.arch.sync_threads()")
        if self.buffers[name].space == "fragment":
            slot, coords, depth = self.loop_fragment(name)
            value = f"{self.buf(name)}[{slot}]"
        else:
            _, coords = self.loop_tile(self.buffers[name].type.shape)
            depth = 2
            value = self.access(name, coords)
        self.emit(f"_nt_snapshot_{name}[{', '.join(coords)}] = {value}")
        self.depth -= depth
        self.emit("cute.arch.sync_threads()")

    def emit(self, line=""):
        self.lines.append("    " * self.depth + line if line else "")

    def unique(self, label):
        self.counter += 1
        return f"_nt_{label}_{self.counter}"

    def dtype(self, name):
        return f"cutlass.{CUTLASS_TYPES[self.buffers[name].type.dtype]}"

    @staticmethod
    def buf(name):
        return f"_b_{name}"

    @staticmethod
    def var(name):
        return f"_v_{name}"

    @staticmethod
    def tuple_text(items):
        return "(" + ", ".join(map(str, items)) + ("," if len(items) == 1 else "") + ")"

    def access(self, name, indices):
        return f"{self.buf(name)}[{', '.join(indices)}]"

    def predicate(self, name, indices):
        shape = self.buffers[name].type.shape
        return " and ".join(f"(0 <= ({idx}) and ({idx}) < {dim})" for idx, dim in zip(indices, shape))

    def index_expression(self, expr):
        value = self.expression(expr)
        dtype = expression_dtype(expr, self.buffers, self.scalar_types)
        # CuTe coordinates accept i32/i64. Widen after source arithmetic so
        # narrow integer operations retain their original dtype semantics.
        if dtype in ("int8", "uint8", "int16", "uint16"):
            return f"cutlass.Int32({value})"
        return value

    def expression(self, expr: Expr):
        op = expr.op
        if op in CHOICE_OPS:
            return self.conditional_expression(expr)
        if op == "const":
            if type(expr.value) not in (int, float, bool):
                raise CompileError("Only numeric and boolean values can appear in scalar expressions")
            if type(expr.value) is float and not isfinite(expr.value):
                value = "nan" if isnan(expr.value) else "inf" if expr.value > 0 else "-inf"
                return f"float('{value}')"
            return repr(expr.value)
        if op == "var":
            return self.var(expr.value)
        if op == "load":
            name = expr.value
            buf = self.buffers[name]
            indices = [self.index_expression(x) for x in expr.args]
            temp = self.unique("load")
            if buf.space == "fragment":
                expected = (
                    () if self.parallel is None else tuple(Expr("var", value=n) for n in self.parallel[1])
                )
                if self.parallel is not None and self.parallel[0] == buf.type.shape and expr.args == expected:
                    return self.fragment_slot(name, expr.args)
                if name not in self.active_snapshots:
                    raise CompileError("Fragment access requires a parallel coordinate mapping")
                self.emit(f"{temp} = {self.dtype(name)}(0)")
                self.emit(f"if {self.predicate(name, indices)}:")
                self.depth += 1
                self.emit(f"{temp} = _nt_snapshot_{name}[{', '.join(indices)}]")
                self.depth -= 1
                return temp
            self.emit(f"{temp} = {self.dtype(name)}(0)")
            self.emit(f"if {self.predicate(name, indices)}:")
            self.depth += 1
            self.emit(f"{temp} = {self.access(name, indices)}")
            self.depth -= 1
            return temp
        values = [self.expression(x) for x in expr.args]
        if op in UNARY_MATH_OPS:
            result_dtype = expression_dtype(expr, self.buffers, self.scalar_types)
            dtype = (
                result_dtype
                if op in TRANSCENDENTAL_OPS
                else expression_dtype(expr.args[0], self.buffers, self.scalar_types)
            )
            type_name = f"cutlass.{CUTLASS_TYPES[dtype]}"
            value = f"{type_name}({values[0]})"
            if dtype.startswith(("int", "uint")) or dtype == "bool":
                if op in CLASSIFICATION_OPS:
                    return f"cutlass.Boolean({op == 'isfinite'})"
                if op == "abs" and dtype.startswith("int"):
                    result = self.unique("abs")
                    self.emit(f"{result} = {type_name}({value}.ir_value())")
                    self.emit(f"if {result} < {type_name}(0):")
                    self.depth += 1
                    self.emit(f"{result} = -{result}")
                    self.depth -= 1
                    return result
                return value
            if dtype in ("float16", "bfloat16"):
                value = f"cutlass.Float32({value})"
            if op == "exp10":
                compute_type = "cutlass.Float32" if dtype in ("float16", "bfloat16") else type_name
                return f"{type_name}(cute.math.pow({compute_type}(10.0), {value}))"
            if op == "sigmoid":
                exponential = self.unique("sigmoid_exp")
                denominator = self.unique("sigmoid_denominator")
                self.emit(f"{exponential} = {type_name}(cute.math.exp(-({value})))")
                self.emit(f"{denominator} = {type_name}({type_name}(1) + {exponential})")
                return f"{type_name}({type_name}(1) / {denominator})"
            function = {"round": "roundeven", "nearbyint": "roundeven", "round_away": "round"}.get(op, op)
            result_type = "cutlass.Boolean" if op in CLASSIFICATION_OPS else type_name
            return f"{result_type}(cute.math.{function}({value}))"
        if op in BITWISE_OPS | BINARY_NUMERIC_OPS:
            expression_dtype(expr, self.buffers, self.scalar_types)
            dtype = operand_dtype(expr, self.buffers, self.scalar_types)
            values = [f"cutlass.{CUTLASS_TYPES[dtype]}({value})" for value in values]
        if op in INTEGER_DIVISION_OPS:
            return self.integer_division(op, values, dtype)
        if op in BITWISE_OPS:
            if op == "invert":
                return f"(~{values[0]})"
            return "(" + f" {op} ".join(values) + ")"
        if op in ("neg", "pos", "not"):
            symbol = {"neg": "-", "pos": "+", "not": "not "}[op]
            return f"({symbol}{values[0]})"
        if op == "cast":
            return f"cutlass.{CUTLASS_TYPES[expr.value]}({values[0]})"
        if op in ("maximum", "minimum", "max", "min"):
            function = "max" if op in ("maximum", "max") else "min"
            return f"cute.math.{function}({', '.join(values)}, propagate_nan={op in ('maximum', 'minimum')})"
        return "(" + f" {op} ".join(values) + ")"

    def conditional_expression(self, expr):
        condition, when_true, when_false = expr.args
        dtype = expression_dtype(expr, self.buffers, self.scalar_types)
        type_name = f"cutlass.{CUTLASS_TYPES[dtype]}"
        predicate = self.expression(condition)
        eager = None
        if expr.op == "select":
            eager = []
            for branch in (when_true, when_false):
                value = self.expression(branch)
                name = self.unique("select_value")
                self.emit(f"{name} = {type_name}({value})")
                eager.append(name)
        result = self.unique("choice")
        self.emit(f"{result} = {type_name}(0)")
        self.emit(f"if {predicate}:")
        self.depth += 1
        value = eager[0] if eager is not None else self.expression(when_true)
        self.emit(f"{result} = {type_name}({value})")
        self.depth -= 1
        self.emit("else:")
        self.depth += 1
        value = eager[1] if eager is not None else self.expression(when_false)
        self.emit(f"{result} = {type_name}({value})")
        self.depth -= 1
        return result

    def integer_division(self, op, values, dtype):
        # Materialize MLIR values so `%` always has CuTe's runtime remainder
        # semantics, including when both source operands are constants.
        type_name = f"cutlass.{CUTLASS_TYPES[dtype]}"
        lhs, rhs = self.unique("div_lhs"), self.unique("div_rhs")
        self.emit(f"{lhs} = {type_name}({values[0]}.ir_value())")
        self.emit(f"{rhs} = {type_name}({values[1]}.ir_value())")
        if op == "ceildiv":
            return f"(({lhs} + {rhs} - {type_name}(1)) // {rhs})"
        if op == "//":
            return f"({lhs} // {rhs})"
        remainder = self.unique("remainder")
        self.emit(f"{remainder} = {lhs} % {rhs}")
        if op == "truncmod":
            return remainder
        if dtype.startswith("uint"):
            return remainder if op == "%" else f"({lhs} // {rhs})"
        correction = f"{type_name}(({remainder} != 0) and (({lhs} < 0) != ({rhs} < 0)))"
        if op == "%":
            return f"({remainder} + {correction} * {rhs})"
        return f"(({lhs} // {rhs}) + {correction})"

    def coordinates(self, flat, shape):
        return [f"(({flat} // {prod(shape[i + 1 :])}) % {dim})" for i, dim in enumerate(shape)]

    def loop_tile(self, shape):
        plan = Partition(shape, self.kernel.threads)
        slot, flat = self.unique("slot"), self.unique("flat")
        self.emit(f"for {slot} in cutlass.range_constexpr({plan.slots}):")
        self.depth += 1
        self.emit(f"{flat} = _nt_tid + {slot} * {plan.threads}")
        self.emit(f"if {flat} < {prod(shape)}:")
        self.depth += 1
        return slot, self.coordinates(flat, shape)

    def region_indices(self, region, coords):
        return [
            f"({self.index_expression(origin)} + {coords[axis] if axis is not None else '0'})"
            for origin, axis in zip(region.origin, region.axes)
        ]

    def region_coordinates(self, region, buffer_coords):
        coords = ["0"] * len(region.shape)
        for index, origin, axis in zip(buffer_coords, region.origin, region.axes):
            if axis is not None:
                coords[axis] = f"({index} - {self.index_expression(origin)})"
        return coords

    def region_predicate(self, region, buffer_coords):
        predicates = []
        for index, origin, size in zip(buffer_coords, region.origin, region.extents):
            start = self.index_expression(origin)
            predicates.append(
                f"({start} <= {index} and cutlass.Int64({index}) < (cutlass.Int64({start}) + {size}))"
            )
        return " and ".join(predicates)

    def copy_snapshots(self, src, dst):
        if src.buffer == dst.buffer and src != dst:
            return {src.buffer}
        if self.buffers[src.buffer].space == "fragment" and (
            not src.is_full(self.buffers[src.buffer].type.shape)
            or self.buffers[dst.buffer].space == "fragment"
            and not dst.is_full(self.buffers[dst.buffer].type.shape)
        ):
            return {src.buffer}
        return set()

    def copy(self, src, dst):
        snapshots = self.copy_snapshots(src, dst)
        for name in sorted(snapshots):
            self.materialize_fragment(name)
        # A uniform barrier also protects shared tiles reused after readers finish.
        shared = any(self.buffers[r.buffer].space == "shared" for r in (src, dst))
        if shared:
            self.emit("cute.arch.sync_threads()")
        if self.buffers[dst.buffer].space == "fragment":
            slot, buffer_coords, depth = self.loop_fragment(dst.buffer)
            coords = self.region_coordinates(dst, buffer_coords)
            if not dst.is_full(self.buffers[dst.buffer].type.shape):
                self.emit(f"if {self.region_predicate(dst, buffer_coords)}:")
                self.depth += 1
                depth += 1
        elif self.buffers[src.buffer].space == "fragment" and src.buffer not in snapshots:
            slot, buffer_coords, depth = self.loop_fragment(src.buffer)
            coords = self.region_coordinates(src, buffer_coords)
        else:
            slot, coords = self.loop_tile(src.shape)
            depth = 2
        src_idx, dst_idx = self.region_indices(src, coords), self.region_indices(dst, coords)
        if self.buffers[src.buffer].space == "fragment" and src.buffer not in snapshots:
            value = f"{self.buf(src.buffer)}[{slot}]"
        else:
            value = self.unique("copy_value")
            self.emit(f"{value} = {self.dtype(src.buffer)}(0)")
            self.emit(f"if {self.predicate(src.buffer, src_idx)}:")
            self.depth += 1
            source = (
                f"_nt_snapshot_{src.buffer}[{', '.join(src_idx)}]"
                if src.buffer in snapshots
                else self.access(src.buffer, src_idx)
            )
            self.emit(f"{value} = {source}")
            self.depth -= 1
        if self.buffers[dst.buffer].space == "fragment":
            self.emit(f"{self.buf(dst.buffer)}[{slot}] = {self.dtype(dst.buffer)}({value})")
        else:
            self.emit(f"if {self.predicate(dst.buffer, dst_idx)}:")
            self.depth += 1
            self.emit(f"{self.access(dst.buffer, dst_idx)} = {self.dtype(dst.buffer)}({value})")
            self.depth -= 1
        self.depth -= depth
        if shared or snapshots:
            self.emit("cute.arch.sync_threads()")

    def gemm(self, a, b, c, ta, tb):
        am, bn, ak, _, _, _ = self.mmas[c]
        a_stride = (1, self.buffers[a].type.shape[1]) if ta else (ak, 1)
        # CuTe MMA B has (N, K) modes; source language B has (K, N).
        b_stride = (ak, 1) if tb else (1, bn)
        av, bv, ar, br = (self.unique(s) for s in ("a_part", "b_part", "a_reg", "b_reg"))
        self.emit("cute.arch.sync_threads()")
        self.emit(
            f"{av} = _nt_thr_{c}.partition_A(cute.make_tensor({self.buf(a)}.iterator, cute.make_layout(({am}, {ak}), stride={a_stride})))"
        )
        self.emit(
            f"{bv} = _nt_thr_{c}.partition_B(cute.make_tensor({self.buf(b)}.iterator, cute.make_layout(({bn}, {ak}), stride={b_stride})))"
        )
        self.emit(f"{ar} = _nt_mma_{c}.make_fragment_A({av})")
        self.emit(f"{br} = _nt_mma_{c}.make_fragment_B({bv})")
        # Scalar register copies retain the MMA's physical register layout.
        for reg, part in ((ar, av), (br, bv)):
            idx = self.unique("mma_load")
            self.emit(f"for {idx} in cutlass.range_constexpr(cute.size({reg})):")
            self.depth += 1
            self.emit(f"{reg}[{idx}] = {part}[{idx}]")
            self.depth -= 1
        k = self.unique("mma_k")
        self.emit(f"for {k} in cutlass.range_constexpr(cute.size({ar}, mode=[2])):")
        self.depth += 1
        self.emit(
            f"cute.gemm(_nt_mma_{c}, {self.buf(c)}, {ar}[None, None, {k}], {br}[None, None, {k}], {self.buf(c)})"
        )
        self.depth -= 1
        self.emit("cute.arch.sync_threads()")

    def reduction(self, src, dst, kind, dim, clear, nan_propagate):
        shape = self.buffers[src].type.shape
        out_shape = self.buffers[dst].type.shape
        dtype = self.buffers[dst].type.dtype
        workspace = self.reduction_workspaces[src, dtype]
        propagate = nan_propagate and dtype in ("float16", "bfloat16")

        def access(coords):
            return f"{workspace}[{', '.join(coords)}]"

        def combine(left, right):
            if kind in ("sum", "abssum"):
                return f"({left} + {right})"
            if kind in ("max", "absmax", "min"):
                operation = "min" if kind == "min" else "max"
                return f"cute.math.{operation}({left}, {right}, propagate_nan={propagate})"
            return f"({left} { {'bitand': '&', 'bitor': '|', 'bitxor': '^'}[kind] } {right})"

        self.emit("cute.arch.sync_threads()")
        if self.buffers[src].space == "fragment":
            slot, coords, depth = self.loop_fragment(src)
            value = f"{self.buf(src)}[{slot}]"
        else:
            _, coords = self.loop_tile(shape)
            depth = 2
            value = self.access(src, coords)
        value_name = self.unique("reduce_value")
        self.emit(f"{value_name} = {self.dtype(dst)}({value})")
        if kind in ("abssum", "absmax") and not (dtype.startswith("uint") or dtype == "bool"):
            value_name = f"cute.math.max({value_name}, -{value_name}, propagate_nan=False)"
        self.emit(f"{access(coords)} = {self.dtype(dst)}({value_name})")
        self.depth -= depth
        self.emit("cute.arch.sync_threads()")
        # Disjoint pairs in each tree level: right operands are never written
        # at that level, and the block reconverges before the following level.
        stride = 1
        while stride < shape[dim]:
            _, coords = self.loop_tile(shape)
            self.emit(
                f"if ({coords[dim]} % {2 * stride} == 0) and ({coords[dim]} + {stride} < {shape[dim]}):"
            )
            self.depth += 1
            partner = coords.copy()
            partner[dim] = f"({coords[dim]} + {stride})"
            result = combine(access(coords), access(partner))
            self.emit(f"{access(coords)} = {self.dtype(dst)}({result})")
            self.depth -= 3
            self.emit("cute.arch.sync_threads()")
            stride *= 2
        if self.buffers[dst].space == "fragment":
            slot, coords, depth = self.loop_fragment(dst)
            destination = f"{self.buf(dst)}[{slot}]"
        else:
            _, coords = self.loop_tile(out_shape)
            depth = 2
            destination = self.access(dst, coords)
        source_coords = coords.copy()
        if len(out_shape) == len(shape):
            source_coords[dim] = "0"
        else:
            source_coords.insert(dim, "0")
        if clear:
            identity = "0"
            integer = dtype.startswith(("int", "uint")) or dtype == "bool"
            if kind == "max":
                identity = str(integer_limits(dtype)[0]) if integer else "float('-inf')"
            elif kind == "min":
                identity = str(integer_limits(dtype)[1]) if integer else "float('inf')"
            elif kind == "bitand":
                identity = "-1" if dtype.startswith("int") else str(integer_limits(dtype)[1])
            initial = f"{self.dtype(dst)}({identity})"
        else:
            initial = destination
        self.emit(f"{destination} = {self.dtype(dst)}({combine(initial, access(source_coords))})")
        self.depth -= depth
        self.emit("cute.arch.sync_threads()")

    def loop_control(self, body):
        if not loop_controls(body):
            return None
        control = self.unique("loop_alive"), self.unique("loop_skip")
        self.emit(f"{control[0]} = cutlass.Boolean(True)")
        return control

    def loop_body(self, body, control):
        previous = self.control
        if control is not None:
            self.emit(f"{control[1]} = cutlass.Boolean(False)")
            types = body_types(body, self.scalar_types, self.buffers)
            for name in sorted(types.keys() - self.scalar_types.keys()):
                self.emit(f"{self.var(name)} = cutlass.{CUTLASS_TYPES[types[name]]}(0)")
            self.scalar_types = types
        self.control = control
        self.statements(body)
        self.control = previous

    def statements(self, body):
        for stmt in body:
            depth = self.depth
            if self.control is not None:
                alive, skip = self.control
                self.emit(f"if {alive} & ~{skip}:")
                self.depth += 1
            self.statement(stmt)
            self.depth = depth

    def statement(self, stmt):
        op, args = stmt.op, stmt.args
        self.emit(f"# {self.kernel.name}:{stmt.location.line} {op}")
        if op == "alloc":
            return
        if op == "pass":
            self.emit("pass")
        elif op in ("break", "continue"):
            alive, skip = self.control
            self.emit(f"{skip} = cutlass.Boolean(True)")
            if op == "break":
                self.emit(f"{alive} = cutlass.Boolean(False)")
        elif op == "if":
            condition, then_body, else_body = args
            value = self.expression(condition)
            merged_types = body_types((stmt,), self.scalar_types, self.buffers)
            for name in sorted(merged_types.keys() - self.scalar_types.keys()):
                self.emit(f"{self.var(name)} = cutlass.{CUTLASS_TYPES[merged_types[name]]}(0)")
            self.scalar_types = merged_types.copy()
            self.emit(f"if {value}:")
            self.depth += 1
            self.statements(then_body)
            self.depth -= 1
            self.scalar_types = merged_types.copy()
            if else_body:
                self.emit("else:")
                self.depth += 1
                self.statements(else_body)
                self.depth -= 1
            self.scalar_types = merged_types
        elif op == "fill":
            name, value = args
            text = self.expression(value)
            if self.buffers[name].space == "fragment":
                self.emit(f"{self.buf(name)}.fill({self.dtype(name)}({text}))")
            else:
                self.emit("cute.arch.sync_threads()")
                _, coords = self.loop_tile(self.buffers[name].type.shape)
                self.emit(f"{self.access(name, coords)} = {self.dtype(name)}({text})")
                self.depth -= 2
                self.emit("cute.arch.sync_threads()")
        elif op == "while":
            condition, inner = args
            if expression_dtype(condition, self.buffers, self.scalar_types) != "bool":
                raise CompileError("While conditions require Boolean expressions", stmt.location)
            predicate = self.unique("while_condition")
            control = self.loop_control(inner)
            self.emit(f"{predicate} = cutlass.Boolean({self.expression(condition)})")
            self.emit(f"while {predicate}:")
            self.depth += 1
            old_types = self.scalar_types.copy()
            self.loop_body(inner, control)
            self.scalar_types = old_types
            if control is not None:
                self.emit(f"if {control[0]}:")
                self.depth += 1
            self.emit(f"{predicate} = cutlass.Boolean({self.expression(condition)})")
            if control is not None:
                self.depth -= 1
                self.emit("else:")
                self.depth += 1
                self.emit(f"{predicate} = cutlass.Boolean(False)")
                self.depth -= 1
            self.depth -= 1
        elif op == "copy":
            self.copy(*args)
        elif op == "gemm":
            self.gemm(*args)
        elif op == "reduce":
            self.reduction(*args)
        elif op == "let":
            value = self.expression(args[1])
            dtype = self.scalar_types.get(args[0]) or expression_dtype(
                args[1], self.buffers, self.scalar_types
            )
            self.scalar_types[args[0]] = dtype
            self.emit(f"{self.var(args[0])} = cutlass.{CUTLASS_TYPES[dtype]}({value})")
        elif op == "declare":
            name, dtype, initializer = args
            value = self.expression(initializer)
            self.scalar_types[name] = dtype
            self.emit(f"{self.var(name)} = {value}")
        elif op == "assign":
            self.emit(f"{self.var(args[0])} = {self.expression(args[1])}")
        elif op == "store":
            name, indices, value = args
            coords = [self.index_expression(x) for x in indices]
            value = self.expression(value)
            if self.buffers[name].space == "fragment":
                slot = self.fragment_slot(name, indices)
                self.emit(f"{slot} = {self.dtype(name)}({value})")
                return
            self.emit(f"if {self.predicate(name, coords)}:")
            self.depth += 1
            self.emit(f"{self.access(name, coords)} = {self.dtype(name)}({value})")
            self.depth -= 1
        elif op in ("serial", "unroll"):
            names, extents, inner = args
            if all(type(value) is int for value in extents):
                trip_count = len(range(*extents))
                if not trip_count:
                    self.emit("pass  # empty static iteration domain")
                    return
                start = str(extents[0])
            else:
                captured = []
                for bound in extents[:2]:
                    value = self.expression(bound) if isinstance(bound, Expr) else str(bound)
                    captured.append(self.unique("loop_bound"))
                    self.emit(f"{captured[-1]} = cutlass.Int64({value})")
                start, stop = captured
                difference = f"({stop} - {start})" if extents[2] > 0 else f"({start} - {stop})"
                count = f"(({difference} + {abs(extents[2]) - 1}) // {abs(extents[2])})"
                trip_count = self.unique("trip_count")
                self.emit(f"{trip_count} = cutlass.Int32(cute.math.max(cutlass.Int64(0), {count}))")
            ordinal = self.unique("iteration")
            control = self.loop_control(inner)
            annotations = dict(stmt.annotations)
            loop = f"range({trip_count})"
            if op == "unroll":
                if annotations.get("pragma_unroll_explicit", False):
                    loop = f"cutlass.range_constexpr({trip_count})"
                elif "pragma_unroll_factor" in annotations:
                    factor = annotations["pragma_unroll_factor"]
                    loop = f"cutlass.range({trip_count}, unroll={max(1, factor)})"
                else:
                    loop = f"cutlass.range({trip_count}, unroll_full=True)"
            self.emit(f"for {ordinal} in {loop}:")
            self.depth += 1
            self.emit(
                f"{self.var(names[0])} = cutlass.Int32(cutlass.Int64({start}) + cutlass.Int64({ordinal}) * {extents[2]})"
            )
            old_types = self.scalar_types.copy()
            self.scalar_types[names[0]] = "int32"
            self.loop_body(inner, control)
            self.scalar_types = old_types
            self.depth -= 1
        elif op == "parallel":
            names, shape, inner = args
            fragments, snapshots, shared = self.parallel_accesses(stmt)
            if shared:
                self.emit("cute.arch.sync_threads()")
            for name in sorted(snapshots):
                self.materialize_fragment(name)
            self.active_snapshots = snapshots
            fragments = sorted(fragments)
            if fragments:
                if any(self.buffers[n].type.shape != shape for n in fragments):
                    raise CompileError(
                        "Fragment indexing requires a matching T.Parallel shape", stmt.location
                    )
                slot, coords, depth = self.loop_fragment(fragments[0])
            else:
                slot, coords = self.loop_tile(shape)
                depth = 2
            for name, coord in zip(names, coords):
                self.emit(f"{self.var(name)} = {coord}")
            old_types = self.scalar_types.copy()
            self.scalar_types.update({name: "int32" for name in names})
            self.parallel = (shape, names, slot)
            previous_control = self.control
            self.control = None
            self.statements(inner)
            self.control = previous_control
            self.scalar_types = old_types
            self.parallel = None
            self.active_snapshots = set()
            self.depth -= depth
            if shared:
                self.emit("cute.arch.sync_threads()")
        else:
            raise CompileError(f"Unsupported IR operation {op}", stmt.location)

    def generate(self):
        self.emit(f'"""Generated by Ntilang 0.1.0: {self.kernel.name}; target {self.target}."""')
        self.emit("import cutlass")
        self.emit("import cutlass.cute as cute")
        self.emit("import cutlass.utils as utils")
        self.emit()
        params = ", ".join(f"{self.buf(p.name)}: cute.Tensor" for p in self.kernel.parameters)
        self.emit("@cute.kernel")
        self.emit(f"def _nt_kernel({params}):")
        self.depth = 1
        self.emit("_nt_tid, _, _ = cute.arch.thread_idx()")
        block = [self.var(v) for v in self.kernel.block_vars] + ["_"] * (3 - len(self.kernel.block_vars))
        self.emit(f"{', '.join(block)} = cute.arch.block_idx()")
        if (
            self.snapshots
            or self.reduction_workspaces
            or any(b.space == "shared" for b in self.kernel.buffers)
        ):
            self.emit("_nt_smem = utils.SmemAllocator()")
        for name, (m, n, _, wm, wn, dtype) in self.mmas.items():
            self.emit(
                f"_nt_mma_{name} = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(cutlass.{CUTLASS_TYPES[dtype]}, cutlass.Float32, (16, 8, 16)), cute.make_layout(({wm}, {wn}, 1)))"
            )
            self.emit(f"_nt_thr_{name} = _nt_mma_{name}.get_slice(_nt_tid)")
            self.emit(
                f"_nt_coords_{name} = _nt_thr_{name}.partition_C(cute.make_identity_tensor(({m}, {n})))"
            )
        for b in self.kernel.buffers:
            if b.name in self.snapshots:
                strides = tuple(prod(b.type.shape[i + 1 :]) for i in range(len(b.type.shape)))
                self.emit(
                    f"_nt_snapshot_{b.name} = _nt_smem.allocate_tensor({self.dtype(b.name)}, cute.make_layout({b.type.shape}, stride={strides}), byte_alignment=16)"
                )
            if b.space == "shared":
                strides = tuple(prod(b.type.shape[i + 1 :]) for i in range(len(b.type.shape)))
                self.emit(
                    f"{self.buf(b.name)} = _nt_smem.allocate_tensor({self.dtype(b.name)}, cute.make_layout({b.type.shape}, stride={strides}), byte_alignment=16)"
                )
            elif b.name in self.mmas:
                self.emit(f"{self.buf(b.name)} = _nt_mma_{b.name}.make_fragment_C(_nt_coords_{b.name}.shape)")
            elif self.fragment_layouts[b.name] is not None:
                layout = self.fragment_layouts[b.name]
                template = self.unique("fragment_layout")
                self.emit(f"{template} = _nt_mma_{layout}.make_fragment_C(_nt_coords_{layout}.shape)")
                self.emit(
                    f"{self.buf(b.name)} = cute.make_rmem_tensor({template}.layout, {self.dtype(b.name)})"
                )
            else:
                self.emit(
                    f"{self.buf(b.name)} = cute.make_rmem_tensor(({Partition(b.type.shape, self.kernel.threads).slots},), {self.dtype(b.name)})"
                )
        for (src, dtype), workspace in self.reduction_workspaces.items():
            shape = self.buffers[src].type.shape
            strides = tuple(prod(shape[i + 1 :]) for i in range(len(shape)))
            self.emit(
                f"{workspace} = _nt_smem.allocate_tensor(cutlass.{CUTLASS_TYPES[dtype]}, cute.make_layout({shape}, stride={strides}), byte_alignment=16)"
            )
        self.statements(self.kernel.body)
        self.depth = 0
        self.emit()
        self.emit("@cute.jit")
        self.emit(f"def run({params}):")
        self.depth = 1
        actual = ", ".join(self.buf(p.name) for p in self.kernel.parameters)
        self.emit(
            f"_nt_kernel({actual}).launch(grid={self.kernel.grid + (1,) * (3 - len(self.kernel.grid))}, block=({self.kernel.threads}, 1, 1))"
        )
        self.depth = 0
        self.emit()
        self.emit("def compile_kernel():")
        self.depth = 1
        self.emit("from cutlass.cute.runtime import make_fake_compact_tensor")
        names = []
        for p in self.kernel.parameters:
            name = self.buf(p.name)
            names.append(name)
            order = tuple(reversed(range(len(p.type.shape))))
            self.emit(
                f"{name} = make_fake_compact_tensor({self.dtype(p.name)}, {p.type.shape}, stride_order={order}, assumed_align=16)"
            )
        self.emit(
            f'return cute.compile(run, {", ".join(names)}, options="--enable-tvm-ffi --gpu-arch={self.target}")'
        )
        result = "\n".join(self.lines) + "\n"
        ast.parse(result)
        return result


def generate(kernel: Kernel, target: str) -> str:
    return Emitter(kernel, target).generate()
