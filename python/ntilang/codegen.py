"""Emit standalone NVIDIA CuTe DSL; no Ntilang runtime appears in the output."""

from __future__ import annotations

import ast
import re
from math import prod

from .ir import CompileError, Expr, Kernel, Partition

CUTLASS_TYPES = {
    "float16": "Float16",
    "bfloat16": "BFloat16",
    "float32": "Float32",
    "int32": "Int32",
    "int64": "Int64",
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
        self.scalar_types = {name: "int32" for name in kernel.block_vars}
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

    def fragment_references(self, value):
        if isinstance(value, Expr):
            names = {value.value} if value.op == "load" else set()
            for arg in value.args:
                names.update(self.fragment_references(arg))
            return {n for n in names if self.buffers[n].space == "fragment"}
        if isinstance(value, (tuple, list)):
            names = set()
            for arg in value:
                names.update(self.fragment_references(arg))
            return names
        return set()

    def parallel_fragments(self, body):
        names = set()
        for stmt in walk(body):
            names.update(self.fragment_references(stmt.args))
            if stmt.op == "store" and self.buffers[stmt.args[0]].space == "fragment":
                names.add(stmt.args[0])
        return names

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
                names = self.parallel_fragments(stmt.args[2])
            elif stmt.op == "copy":
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
        return slot, [f"_nt_coords_{layout}[{slot}][{i}]" for i in range(2)], 1

    def emit(self, line=""):
        self.lines.append("    " * self.depth + line if line else "")

    @staticmethod
    def common_type(left, right):
        if left == right:
            return left
        if left == "bool":
            return right
        if right == "bool":
            return left
        widths = {"int32": 32, "int64": 64, "float16": 16, "bfloat16": 16, "float32": 32, "float64": 64}
        width = max(widths[left], widths[right])
        if left.startswith("int") and right.startswith("int"):
            return f"int{width}"
        return f"float{width}"

    def expression_type(self, expr, types):
        if expr.op == "const":
            if type(expr.value) is int and not -(2**31) <= expr.value < 2**31:
                if not -(2**63) <= expr.value < 2**63:
                    raise CompileError("Scalar integer constants must fit signed 64-bit arithmetic")
                return "int64"
            return {bool: "bool", int: "int32", float: "float32"}[type(expr.value)]
        if expr.op == "var":
            return types[expr.value]
        if expr.op == "load":
            return self.buffers[expr.value].type.dtype
        if expr.op == "cast":
            return expr.value
        if expr.op in ("<", "<=", ">", ">=", "==", "!=", "and", "or", "not"):
            return "bool"
        result = self.expression_type(expr.args[0], types)
        for arg in expr.args[1:]:
            result = self.common_type(result, self.expression_type(arg, types))
        return "float32" if expr.op == "/" and result in ("int32", "int64", "bool") else result

    def body_types(self, body, types):
        types = types.copy()
        for stmt in body:
            if stmt.op == "let":
                types[stmt.args[0]] = self.expression_type(stmt.args[1], types)
            elif stmt.op == "if":
                then_types = self.body_types(stmt.args[1], types)
                else_types = self.body_types(stmt.args[2], types)
                for name in then_types.keys() & else_types.keys():
                    types[name] = self.common_type(then_types[name], else_types[name])
        return types

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

    def expression(self, expr: Expr):
        op = expr.op
        if op == "const":
            if type(expr.value) not in (int, float, bool):
                raise CompileError("Only numeric and boolean values can appear in scalar expressions")
            return repr(expr.value)
        if op == "var":
            return self.var(expr.value)
        if op == "load":
            name = expr.value
            buf = self.buffers[name]
            indices = [self.expression(x) for x in expr.args]
            temp = self.unique("load")
            if buf.space == "fragment":
                return self.fragment_slot(name, expr.args)
            self.emit(f"{temp} = {self.dtype(name)}(0)")
            self.emit(f"if {self.predicate(name, indices)}:")
            self.depth += 1
            self.emit(f"{temp} = {self.access(name, indices)}")
            self.depth -= 1
            return temp
        values = [self.expression(x) for x in expr.args]
        if op in ("neg", "pos", "not"):
            symbol = {"neg": "-", "pos": "+", "not": "not "}[op]
            return f"({symbol}{values[0]})"
        if op == "cast":
            return f"cutlass.{CUTLASS_TYPES[expr.value]}({values[0]})"
        if op in ("exp", "exp2", "sqrt"):
            return f"cute.math.{op}({values[0]})"
        if op in ("maximum", "minimum"):
            function = "max" if op == "maximum" else "min"
            return f"cute.math.{function}({', '.join(values)}, propagate_nan=True)"
        return "(" + f" {op} ".join(values) + ")"

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
        return [f"({self.expression(origin)} + {coord})" for origin, coord in zip(region.origin, coords)]

    def copy(self, src, dst):
        # A uniform barrier also protects shared tiles reused after readers finish.
        shared = any(self.buffers[r.buffer].space == "shared" for r in (src, dst))
        if shared:
            self.emit("cute.arch.sync_threads()")
        fragments = [r.buffer for r in (src, dst) if self.buffers[r.buffer].space == "fragment"]
        if fragments:
            slot, coords, depth = self.loop_fragment(fragments[0])
        else:
            slot, coords = self.loop_tile(src.shape)
            depth = 2
        src_idx, dst_idx = self.region_indices(src, coords), self.region_indices(dst, coords)
        if self.buffers[src.buffer].space == "fragment":
            value = f"{self.buf(src.buffer)}[{slot}]"
        else:
            value = self.unique("copy_value")
            self.emit(f"{value} = {self.dtype(src.buffer)}(0)")
            self.emit(f"if {self.predicate(src.buffer, src_idx)}:")
            self.depth += 1
            self.emit(f"{value} = {self.access(src.buffer, src_idx)}")
            self.depth -= 1
        if self.buffers[dst.buffer].space == "fragment":
            self.emit(f"{self.buf(dst.buffer)}[{slot}] = {self.dtype(dst.buffer)}({value})")
        else:
            self.emit(f"if {self.predicate(dst.buffer, dst_idx)}:")
            self.depth += 1
            self.emit(f"{self.access(dst.buffer, dst_idx)} = {self.dtype(dst.buffer)}({value})")
            self.depth -= 1
        self.depth -= depth
        if shared:
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

    def statements(self, body):
        for stmt in body:
            op, args = stmt.op, stmt.args
            self.emit(f"# {self.kernel.name}:{stmt.location.line} {op}")
            if op == "alloc":
                continue
            if op == "pass":
                self.emit("pass")
            elif op == "if":
                condition, then_body, else_body = args
                value = self.expression(condition)
                merged_types = self.body_types((stmt,), self.scalar_types)
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
            elif op == "copy":
                self.copy(*args)
            elif op == "gemm":
                self.gemm(*args)
            elif op == "let":
                value = self.expression(args[1])
                dtype = self.scalar_types.get(args[0]) or self.expression_type(args[1], self.scalar_types)
                self.scalar_types[args[0]] = dtype
                self.emit(f"{self.var(args[0])} = cutlass.{CUTLASS_TYPES[dtype]}({value})")
            elif op == "store":
                name, indices, value = args
                coords = [self.expression(x) for x in indices]
                value = self.expression(value)
                if self.buffers[name].space == "fragment":
                    slot = self.fragment_slot(name, indices)
                    self.emit(f"{slot} = {self.dtype(name)}({value})")
                    continue
                self.emit(f"if {self.predicate(name, coords)}:")
                self.depth += 1
                self.emit(f"{self.access(name, coords)} = {self.dtype(name)}({value})")
                self.depth -= 1
            elif op in ("serial", "unroll"):
                names, extents, inner = args
                trip_count = len(range(*extents))
                if not trip_count:
                    self.emit("pass  # empty static iteration domain")
                    continue
                ordinal = self.unique("iteration")
                loop = "cutlass.range_constexpr" if op == "unroll" else "range"
                self.emit(f"for {ordinal} in {loop}({trip_count}):")
                self.depth += 1
                self.emit(
                    f"{self.var(names[0])} = cutlass.Int32(cutlass.Int64({extents[0]}) + cutlass.Int64({ordinal}) * {extents[2]})"
                )
                old_types = self.scalar_types.copy()
                self.scalar_types[names[0]] = "int32"
                self.statements(inner)
                self.scalar_types = old_types
                self.depth -= 1
            elif op == "parallel":
                names, shape, inner = args
                fragments = sorted(self.parallel_fragments(inner))
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
                self.statements(inner)
                self.scalar_types = old_types
                self.parallel = None
                self.depth -= depth
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
        if any(b.space == "shared" for b in self.kernel.buffers):
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
