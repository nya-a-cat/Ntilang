"""Emit standalone NVIDIA CuTe DSL; no Ntilang runtime appears in the output."""

from __future__ import annotations

import ast
import re
from math import prod

from .ir import CompileError, Expr, Kernel, Partition

CUTLASS_TYPES = {"float16": "Float16", "bfloat16": "BFloat16", "float32": "Float32", "int32": "Int32"}


def walk(body):
    for stmt in body:
        yield stmt
        if stmt.op in ("serial", "parallel", "unroll"):
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
        for stmt in walk(kernel.body):
            if stmt.op == "copy":
                src, dst = stmt.args
                if dst.buffer in self.mmas:
                    raise CompileError(
                        "Initialize MMA accumulators with T.clear/T.fill; copy into them is not supported",
                        stmt.location,
                    )
                if src.buffer in self.mmas and self.buffers[dst.buffer].space != "global":
                    raise CompileError(
                        "MMA accumulator copies currently require a global destination", stmt.location
                    )

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
                if name in self.mmas or self.parallel is None or self.parallel[0] != buf.type.shape:
                    raise CompileError(
                        "Fragment indexing requires a matching T.Parallel shape and linear layout"
                    )
                expected = tuple(Expr("var", value=n) for n in self.parallel[1])
                if expr.args != expected:
                    raise CompileError("Fragment loads require the exact T.Parallel induction variables")
                return f"{self.buf(name)}[{self.parallel[2]}]"
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
        if src.buffer in self.mmas:
            slot = self.unique("mma_store")
            coords_name = f"_nt_coords_{src.buffer}"
            self.emit(f"for {slot} in cutlass.range_constexpr(cute.size({self.buf(src.buffer)})):")
            self.depth += 1
            coords = [f"{coords_name}[{slot}][{i}]" for i in range(2)]
            out = self.region_indices(dst, coords)
            self.emit(f"if {self.predicate(dst.buffer, out)}:")
            self.depth += 1
            self.emit(
                f"{self.access(dst.buffer, out)} = {self.dtype(dst.buffer)}({self.buf(src.buffer)}[{slot}])"
            )
            self.depth -= 2
            return
        # A uniform barrier also protects shared tiles reused after readers finish.
        shared = any(self.buffers[r.buffer].space == "shared" for r in (src, dst))
        if shared:
            self.emit("cute.arch.sync_threads()")
        slot, coords = self.loop_tile(src.shape)
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
        self.depth -= 2
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
            if op == "fill":
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
                self.emit(f"{self.var(args[0])} = {value}")
            elif op == "store":
                name, indices, value = args
                coords = [self.expression(x) for x in indices]
                value = self.expression(value)
                if self.buffers[name].space == "fragment":
                    if name in self.mmas:
                        raise CompileError(
                            "MMA fragment element stores require its inferred coordinate layout",
                            stmt.location,
                        )
                    self.emit(f"{self.buf(name)}[{self.parallel[2]}] = {self.dtype(name)}({value})")
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
                self.statements(inner)
                self.depth -= 1
            elif op == "parallel":
                names, shape, inner = args
                slot, coords = self.loop_tile(shape)
                for name, coord in zip(names, coords):
                    self.emit(f"{self.var(name)} = {coord}")
                self.parallel = (shape, names, slot)
                self.statements(inner)
                self.parallel = None
                self.depth -= 2
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
