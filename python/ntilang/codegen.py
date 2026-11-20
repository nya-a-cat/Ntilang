"""Emit standalone NVIDIA CuTe DSL; no Ntilang runtime appears in the output."""

from __future__ import annotations

import ast
import re
from hashlib import sha256
from math import gcd, isfinite, isnan, prod

from . import scan
from .cuda_math import low_precision_asm, minmax_asm
from .debug import print_format
from .ir import (
    DTYPES,
    CompileError,
    Expr,
    Kernel,
    Partition,
    ScalarParameter,
    Statement,
    integer_limits,
    loop_controls,
)
from .passes import (
    HazardTracker,
    analyze_mmas,
    collect_workspaces,
    expr_smem_reads,
    infer_fragment_layouts,
    smem_footprint,
    validate_shared_memory_budget,
    walk,
)
from .passes import copy_snapshots as _copy_snapshots
from .passes import parallel_accesses as _parallel_accesses
from .runtime import HOST_ERROR_TYPES
from .scalar import (
    BINARY_MATH_OPS,
    BINARY_NUMERIC_OPS,
    BIT_COUNT_OPS,
    BITWISE_OPS,
    CHOICE_OPS,
    CLASSIFICATION_OPS,
    FAST_MATH_OPS,
    IEEE_MATH_OPS,
    INTEGER_DIVISION_OPS,
    TRANSCENDENTAL_OPS,
    UNARY_MATH_OPS,
    body_types,
    expression_dtype,
    operand_dtype,
)
from .validation import affine, interval, predicate_bounds

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


def requests_math_header(value):
    """Match the pinned CUDA emitter's module-level math header requests."""
    if isinstance(value, Expr):
        return value.op in ("__exp", "__log", "__sin", "__cos", "fast_rcp") or any(
            requests_math_header(arg) for arg in value.args
        )
    if isinstance(value, (tuple, list)):
        return any(requests_math_header(arg) for arg in value)
    return False


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
        self.math_externs = {}
        self.math_ptx = {}
        self.minmax_ptx = {}
        self.bitcasts = {}
        self.debug_prints = {}
        self.device_asserts = False
        self.emitting_host_checks = False
        self.math_header = any(requests_math_header(stmt.args) for stmt in walk(kernel.body))
        self.parallel = None
        self.control = None
        self.scalar_types = {name: "int32" for name in kernel.block_vars}
        self.scalar_types.update(
            {p.name: p.dtype for p in kernel.parameters if isinstance(p, ScalarParameter)}
        )

        def host_names(value):
            names = {value.value} if value.op == "var" else set()
            for child in value.args:
                names.update(host_names(child))
            return names

        host_inputs = set().union(*(host_names(check.args[0]) for check in kernel.host_checks))
        self.host_parameters = tuple(
            (index, p)
            for index, p in enumerate(kernel.parameters)
            if isinstance(p, ScalarParameter) and p.name in host_inputs
        )
        self.active_snapshots = set()
        self.mmas = analyze_mmas(kernel, self.buffers)
        self.fragment_layouts = infer_fragment_layouts(kernel, self.buffers, self.mmas)
        self.snapshots, self.reduction_workspaces, self.scan_workspaces = collect_workspaces(
            kernel, self.buffers
        )
        self.swizzled_buffers = {
            buf_name
            for stmt in walk(kernel.body)
            if stmt.op == "annotate_layout"
            for buf_name, kind in stmt.args[0]
            if kind == "swizzled"
        }
        self.hazards = HazardTracker()
        self.bounds: dict[str, tuple[int, int]] = {
            name: (0, extent - 1) for name, extent in zip(kernel.block_vars, kernel.grid)
        }
        self.definitions: dict[str, Expr] = {}
        self.active_stage: dict[str, str] = {}
        self.shared_swizzles: dict[str, tuple[int, int, int]] = self._plan_shared_swizzles()
        self.pipelined_plans, self.staged_buffers = self._plan_pipelined_loops()
        validate_shared_memory_budget(
            kernel,
            self.buffers,
            self.snapshots,
            self.reduction_workspaces,
            self.scan_workspaces,
            limit=self._shared_memory_limit(),
            stage_multipliers=self.staged_buffers,
        )

    def _shared_memory_limit(self) -> int:
        if not self.staged_buffers and len(self.mmas) < 2:
            return 48 * 1024
        if self.target == "sm_80":
            return 163 * 1024
        if self.target in ("sm_86", "sm_87", "sm_89"):
            return 99 * 1024
        return 227 * 1024

    def _plan_pipelined_loops(self):
        plans: dict[int, tuple[int, list, list]] = {}
        staged: dict[str, int] = {}

        def _buf_refs(stmts):
            reads, writes = smem_footprint(
                stmts, self.buffers, self.reduction_workspaces, self.scan_workspaces
            )
            return reads | writes

        all_stmts = list(walk(self.kernel.body))
        for stmt in all_stmts:
            if stmt.op != "pipelined":
                continue
            names, extents, inner, num_stages = stmt.args
            if num_stages < 2 or not all(type(v) is int for v in extents):
                continue
            static_range = range(*extents)
            if not static_range:
                continue
            loop_var = names[0]
            old_bounds = self.bounds.copy()
            self.bounds[loop_var] = (
                min(static_range[0], static_range[-1]),
                max(static_range[0], static_range[-1]),
            )
            try:
                inner_ids = {id(s) for s in walk(inner)}
                outside_stmts = [s for s in all_stmts if id(s) != id(stmt) and id(s) not in inner_ids]
                outside_refs = _buf_refs(outside_stmts)
                producers = []
                producer_bufs: set[str] = set()
                idx = 0
                while idx < len(inner):
                    child = inner[idx]
                    if child.op != "copy":
                        break
                    src, dst = child.args
                    src_buf, dst_buf = self.buffers[src.buffer], self.buffers[dst.buffer]
                    if (
                        src_buf.space != "global"
                        or dst_buf.space != "shared"
                        or not dst.is_full(dst_buf.type.shape)
                        or len(dst_buf.type.shape) != 2
                        or dst.buffer in producer_bufs
                        or dst.buffer in staged
                        or dst.buffer in self.snapshots
                        or dst.buffer in outside_refs
                    ):
                        break
                    vec_plan = self._vectorized_copy_plan(src, dst)
                    if vec_plan is None or vec_plan[0] * vec_plan[1] * 8 not in (32, 64, 128):
                        break
                    producers.append((src, dst))
                    producer_bufs.add(dst.buffer)
                    idx += 1
                consumers = list(inner[idx:])
                if not producers or not consumers:
                    continue
                _, consumer_writes = smem_footprint(
                    consumers, self.buffers, self.reduction_workspaces, self.scan_workspaces
                )
                if consumer_writes & producer_bufs:
                    continue
                for buf_name in producer_bufs:
                    if buf_name in self.shared_swizzles:
                        b_bits, m_bits, s_bits = self.shared_swizzles[buf_name]
                        if prod(self.buffers[buf_name].type.shape) % (1 << (b_bits + m_bits + s_bits)) != 0:
                            del self.shared_swizzles[buf_name]
                    staged[buf_name] = num_stages
                plans[id(stmt)] = (num_stages, producers, consumers)
            finally:
                self.bounds = old_bounds
        return plans, staged

    def _plan_shared_swizzles(self) -> dict[str, tuple[int, int, int]]:
        mma_operands: set[str] = set()
        block_copy_shared: set[str] = set()
        for stmt in walk(self.kernel.body):
            if stmt.op == "gemm":
                a, b, _, _, _ = stmt.args
                mma_operands.update((a, b))
            elif stmt.op == "copy":
                src, dst = stmt.args
                if len(src.shape) in (1, 2) and prod(src.shape) >= 2:
                    for reg in (src, dst):
                        if self.buffers[reg.buffer].space == "shared":
                            block_copy_shared.add(reg.buffer)

        swizzles: dict[str, tuple[int, int, int]] = {}
        for buf in self.kernel.buffers:
            if len(buf.type.shape) != 2:
                continue
            is_rs_gemm_snap = (
                buf.space == "fragment" and buf.name in mma_operands and buf.name in self.snapshots
            )
            if buf.space != "shared" and not is_rs_gemm_snap:
                continue
            dtype = buf.type.dtype
            if dtype not in ("float16", "bfloat16", "int16", "uint16", "float32", "int32", "uint32"):
                continue
            _, cols = buf.type.shape
            elem_bytes = DTYPES[dtype]
            needs_16b_contiguity = (
                buf.name in mma_operands or buf.name in block_copy_shared or buf.name in self.swizzled_buffers
            )
            key = f"_nt_snapshot_{buf.name}" if is_rs_gemm_snap else buf.name
            if needs_16b_contiguity:
                m_bits = 3 if elem_bytes == 2 else 2
                row_bytes = cols * elem_bytes
                if row_bytes % 128 == 0:
                    tz = (cols & -cols).bit_length() - 1
                    swizzles[key] = (3, m_bits, tz - m_bits)
                elif row_bytes % 64 == 0:
                    swizzles[key] = (2, m_bits, 3)
                elif row_bytes % 32 == 0:
                    swizzles[key] = (1, m_bits, 3)
            elif elem_bytes == 4 and cols >= 32 and cols % 32 == 0:
                tz = (cols & -cols).bit_length() - 1
                swizzles[key] = (min(5, tz), 0, tz)
        return swizzles

    def _affine_tile_coord(self, origin: Expr, tile_dim: int) -> str | None:
        try:
            const, coeffs = affine(origin, self.definitions, self.bounds, self.buffers)
        except CompileError:
            return None
        if const % tile_dim != 0 or any(c % tile_dim != 0 for c in coeffs.values()):
            return None
        if any(name not in self.bounds for name in coeffs):
            return None
        terms: list[str] = []
        if const != 0:
            terms.append(str(const // tile_dim))
        for name, coeff in coeffs.items():
            q = coeff // tile_dim
            var_name = self.var(name)
            if q == 1:
                terms.append(var_name)
            elif q == -1:
                terms.append(f"(-{var_name})")
            else:
                terms.append(f"({q} * {var_name})")
        return " + ".join(terms) if terms else "0"

    def _vectorized_copy_plan(self, src, dst):
        if self.parallel is not None or src.buffer == dst.buffer:
            return None
        src_buf, dst_buf = self.buffers[src.buffer], self.buffers[dst.buffer]
        if src_buf.type.dtype != dst_buf.type.dtype or src_buf.type.dtype == "bool":
            return None
        if {src_buf.space, dst_buf.space} not in ({"global", "shared"}, {"shared"}):
            return None
        rank = len(src.shape)
        if rank not in (1, 2) or src.shape != dst.shape or prod(src.shape) < 2:
            return None

        def _check_region(reg, buf):
            non_none = [ax for ax in reg.axes if ax is not None]
            if tuple(non_none) != tuple(range(rank)) or reg.axes[-1] != rank - 1:
                return None
            if len(buf.type.shape) < rank:
                return None
            tile_coords = []
            for origin, axis, dim in zip(reg.origin, reg.axes, buf.type.shape):
                iv = self.expr_interval(origin)
                if axis is None:
                    if iv is None or iv[0] < 0 or iv[1] >= dim:
                        return None
                else:
                    if iv is None or iv[0] < 0 or iv[1] + reg.shape[axis] - 1 >= dim:
                        return None
                    if dim % reg.shape[axis] != 0:
                        return None
                    tc = self._affine_tile_coord(origin, reg.shape[axis])
                    if tc is None:
                        return None
                    tile_coords.append(tc)
            return tile_coords

        src_coords = _check_region(src, src_buf)
        dst_coords = _check_region(dst, dst_buf)
        if src_coords is None or dst_coords is None:
            return None

        elem_bytes = DTYPES[src_buf.type.dtype]
        if elem_bytes not in (1, 2, 4, 8):
            return None
        max_vec = 16 // elem_bytes
        for reg in (src, dst):
            if reg.buffer in self.shared_swizzles:
                _, m_bits, _ = self.shared_swizzles[reg.buffer]
                max_vec = min(max_vec, 1 << m_bits)

        threads = self.kernel.threads
        tile_cols = src.shape[-1]
        for vec in (16 // elem_bytes, 8 // elem_bytes, 4 // elem_bytes):
            if vec < 2 or vec > max_vec or tile_cols % vec != 0:
                continue
            col_vecs = tile_cols // vec
            if rank == 1:
                if col_vecs % threads == 0:
                    return (vec, elem_bytes, (threads,), (1,), (vec,), src_coords, dst_coords)
            else:
                tile_rows = src.shape[0]
                thr_cols = gcd(threads, col_vecs)
                thr_rows = threads // thr_cols
                if col_vecs % thr_cols == 0 and tile_rows % thr_rows == 0:
                    return (
                        vec,
                        elem_bytes,
                        (thr_rows, thr_cols),
                        (thr_cols, 1),
                        (1, vec),
                        src_coords,
                        dst_coords,
                    )
        return None

    def _region_tile_tensor(self, reg, tile_coords):
        buf = self.buffers[reg.buffer]
        if reg.is_full(buf.type.shape):
            return self.buf(reg.buffer)
        base = self.buf(reg.buffer)
        if any(ax is None for ax in reg.axes):
            slice_items = [
                self.index_expression(o) if ax is None else "None" for o, ax in zip(reg.origin, reg.axes)
            ]
            sliced = self.unique("vec_slice")
            self.emit(f"{sliced} = {base}[{', '.join(slice_items)}]")
            base = sliced
        name = self.unique("vec_tile")
        self.emit(f"{name} = cute.local_tile({base}, {reg.shape}, {self.tuple_text(tile_coords)})")
        return name

    def _emit_vectorized_copy(self, src, dst, plan):
        vec, elem_bytes, thr_shape, thr_stride, val_shape, src_coords, dst_coords = plan
        bits = vec * elem_bytes * 8
        atom = self.unique("vec_atom")
        tc = self.unique("vec_tc")
        thr = self.unique("vec_thr")
        self.emit(
            f"{atom} = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {self.dtype(src.buffer)}, num_bits_per_copy={bits})"
        )
        self.emit(
            f"{tc} = cute.make_tiled_copy_tv({atom}, cute.make_layout({thr_shape}, stride={thr_stride}), cute.make_layout({val_shape}))"
        )
        self.emit(f"{thr} = {tc}.get_slice(_nt_tid)")
        src_tile = self._region_tile_tensor(src, src_coords)
        dst_tile = self._region_tile_tensor(dst, dst_coords)
        self.emit(f"cute.copy({tc}, {thr}.partition_S({src_tile}), {thr}.partition_D({dst_tile}))")

    def _emit_async_copy(self, src, dst, plan, stage_expr: str):
        vec, elem_bytes, thr_shape, thr_stride, val_shape, src_coords, _ = plan
        bits = vec * elem_bytes * 8
        cache_mode = (
            "cute.nvgpu.cpasync.LoadCacheMode.GLOBAL"
            if bits == 128
            else "cute.nvgpu.cpasync.LoadCacheMode.ALWAYS"
        )
        atom = self.unique("async_atom")
        tc = self.unique("async_tc")
        thr = self.unique("async_thr")
        self.emit(
            f"{atom} = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(cache_mode={cache_mode}), {self.dtype(src.buffer)}, num_bits_per_copy={bits})"
        )
        self.emit(
            f"{tc} = cute.make_tiled_copy_tv({atom}, cute.make_layout({thr_shape}, stride={thr_stride}), cute.make_layout({val_shape}))"
        )
        self.emit(f"{thr} = {tc}.get_slice(_nt_tid)")
        src_tile = self._region_tile_tensor(src, src_coords)
        dst_tile = f"{self.buf(dst.buffer)}[None, None, {stage_expr}]"
        self.emit(f"cute.copy({tc}, {thr}.partition_S({src_tile}), {thr}.partition_D({dst_tile}))")

    def sync_threads(self):
        self.emit("cute.arch.sync_threads()")
        self.hazards.clear()

    def sync_if_hazard(self, reads, writes):
        if self.hazards.has_hazard(reads, writes):
            self.sync_threads()

    def expr_interval(self, expr: Expr):
        try:
            return interval(expr, self.bounds, self.definitions, self.buffers)
        except CompileError:
            return None

    def parallel_accesses(self, stmt):
        return _parallel_accesses(stmt, self.buffers)

    def copy_snapshots(self, src, dst):
        return _copy_snapshots(src, dst, self.buffers)

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
            return self.loop_tile(self.buffers[name].type.shape, constexpr=True)
        slot = self.unique("mma_slot")
        self.emit(f"for {slot} in cutlass.range_constexpr(cute.size({self.buf(name)})):")
        self.depth += 1
        physical = iter(f"_nt_coords_{layout}[{slot}][{i}]" for i in range(2))
        coords = ["0" if size == 1 else next(physical) for size in self.buffers[name].type.shape]
        return slot, coords, 1

    def materialize_fragment(self, name):
        snap = f"_nt_snapshot_{name}"
        reads = {name} if self.buffers[name].space == "shared" else set()
        writes = {snap}
        self.sync_if_hazard(reads, writes)
        if self.buffers[name].space == "fragment":
            slot, coords, depth = self.loop_fragment(name)
            value = f"{self.buf(name)}[{slot}]"
        else:
            _, coords, depth = self.loop_tile(self.buffers[name].type.shape)
            value = self.access(name, coords)
        self.emit(f"{snap}[{', '.join(coords)}] = {value}")
        self.depth -= depth
        self.hazards.record(reads, writes)

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
        if name in self.active_stage:
            return f"{self.buf(name)}[{', '.join(indices)}, {self.active_stage[name]}]"
        return f"{self.buf(name)}[{', '.join(indices)}]"

    def predicate(self, name, indices):
        shape = self.buffers[name].type.shape
        return " and ".join(f"(0 <= ({idx}) and ({idx}) < {dim})" for idx, dim in zip(indices, shape))

    def predicate_for_exprs(self, name, exprs, indices):
        shape = self.buffers[name].type.shape
        clauses = []
        for expr, idx, dim in zip(exprs, indices, shape):
            iv = self.expr_interval(expr)
            if iv is not None and 0 <= iv[0] and iv[1] < dim:
                continue
            clauses.append(f"(0 <= ({idx}) and ({idx}) < {dim})")
        return " and ".join(clauses) if clauses else None

    def region_in_bounds_predicate(self, region, indices):
        shape = self.buffers[region.buffer].type.shape
        clauses = []
        for origin, axis, idx, dim in zip(region.origin, region.axes, indices, shape):
            iv = self.expr_interval(origin)
            max_offset = 0 if axis is None else region.shape[axis] - 1
            if iv is not None and 0 <= iv[0] and iv[1] + max_offset < dim:
                continue
            clauses.append(f"(0 <= ({idx}) and ({idx}) < {dim})")
        return " and ".join(clauses) if clauses else None

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
        if op == "likely":
            # The hint preserves its operand and dtype. Branch/range checks
            # also look through it; device branch weighting is not forced.
            return self.expression(expr.args[0])
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
            if buf.space == "fragment":
                expected = (
                    () if self.parallel is None else tuple(Expr("var", value=n) for n in self.parallel[1])
                )
                if self.parallel is not None and self.parallel[0] == buf.type.shape and expr.args == expected:
                    return self.fragment_slot(name, expr.args)
                if name not in self.active_snapshots:
                    raise CompileError("Fragment access requires a parallel coordinate mapping")
                pred = self.predicate_for_exprs(name, expr.args, indices)
                target = f"_nt_snapshot_{name}[{', '.join(indices)}]"
                if pred is None:
                    return target
                temp = self.unique("load")
                self.emit(f"{temp} = {self.dtype(name)}(0)")
                self.emit(f"if {pred}:")
                self.depth += 1
                self.emit(f"{temp} = {target}")
                self.depth -= 1
                return temp
            pred = self.predicate_for_exprs(name, expr.args, indices)
            target = self.access(name, indices)
            if pred is None:
                return target
            temp = self.unique("load")
            self.emit(f"{temp} = {self.dtype(name)}(0)")
            self.emit(f"if {pred}:")
            self.depth += 1
            self.emit(f"{temp} = {target}")
            self.depth -= 1
            return temp
        values = [self.expression(x) for x in expr.args]
        if op in BIT_COUNT_OPS | {"reinterpret"}:
            source = expression_dtype(expr.args[0], self.buffers, self.scalar_types)
            dtype = expression_dtype(expr, self.buffers, self.scalar_types)
            typ = f"cutlass.{CUTLASS_TYPES[dtype]}"
            value = f"cutlass.{CUTLASS_TYPES[source]}({values[0]})"
            if op == "reinterpret":
                if source == dtype or "bool" in (source, dtype):
                    # Valid C++ Boolean byte representations are 0 and 1. CuTe
                    # stores the Boolean scalar in i1, so adapt that representation.
                    return f"{typ}({value})"
                helper = f"_nt_reinterpret_{source}_{dtype}"
                self.bitcasts[helper] = (source, dtype)
                return f"{helper}({value})"
            operation = "popc" if op == "popcount" else "clz"
            return f"{typ}(cute.arch.{operation}({value}))"
        if op in FAST_MATH_OPS:
            dtype = expression_dtype(expr, self.buffers, self.scalar_types)
            if op == "fast_rcp":
                return f"cute.math.rcp(cutlass.Float32({values[0]}), approx=True, ftz=True)"
            operation = op[2:]
            if dtype == "float32":
                return self.math_external(f"__nv_fast_{operation}f", dtype, values, (dtype,))
            if dtype == "float64":
                return self.math_external(f"__nv_{operation}", dtype, values, (dtype,))
            return self.low_precision_unary(operation, dtype, values[0])
        if op in IEEE_MATH_OPS:
            dtype = expression_dtype(expr, self.buffers, self.scalar_types)
            operation, arity = IEEE_MATH_OPS[op]
            if dtype in ("float16", "bfloat16"):
                if operation == "sqrt" and self.math_header:
                    widened = self.math_external("__nv_sqrtf", "float32", values, ("float32",))
                    return f"cutlass.{CUTLASS_TYPES[dtype]}({widened})"
                return self.math_low_precision(operation, dtype, values)
            stem = (
                ("fmaf" if dtype == "float32" else "fma")
                if operation == "fma"
                else (("f" if dtype == "float32" else "d") + operation)
            )
            return self.math_external(f"__nv_{stem}_{expr.value}", dtype, values, (dtype,) * arity)
        if op == "pow_integer":
            dtype = expression_dtype(expr, self.buffers, self.scalar_types)
            type_name = f"cutlass.{CUTLASS_TYPES[dtype]}"
            if expr.value == 0:
                return f"{type_name}(1)"
            base, result = self.unique("pow_base"), self.unique("pow_result")
            self.emit(f"{base} = {type_name}({values[0]})")
            self.emit(f"{result} = {base}")
            if expr.value > 1:
                index = self.unique("pow_step")
                self.emit(f"for {index} in cutlass.range({expr.value - 1}):")
                self.depth += 1
                self.emit(f"{result} = {type_name}({result} * {base})")
                self.depth -= 1
            return result
        if op in BINARY_MATH_OPS:
            dtype = expression_dtype(expr, self.buffers, self.scalar_types)
            type_name = f"cutlass.{CUTLASS_TYPES[dtype]}"
            if op in ("hypot", "nextafter", "ldexp"):
                symbol = f"__nv_{op}{'f' if dtype == 'float32' else ''}"
                dtypes = (dtype, "int32" if op == "ldexp" else dtype)
                return self.math_external(symbol, dtype, values, dtypes)
            values = [f"{type_name}({value})" for value in values]
            if dtype in ("float16", "bfloat16"):
                values = [f"cutlass.Float32({value})" for value in values]
            function = "rem" if op == "fmod" else op
            return f"{type_name}(cute.math.{function}({', '.join(values)}))"
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
                if op in (
                    "exp",
                    "exp2",
                    "exp10",
                    "log",
                    "log2",
                    "log10",
                    "sin",
                    "cos",
                    "sqrt",
                    "rsqrt",
                    "tanh",
                    "tan",
                    "sinh",
                    "cosh",
                    "atan",
                    "erf",
                    "sigmoid",
                ):
                    return self.low_precision_unary(op, dtype, value)
                value = f"cutlass.Float32({value})"
            if op == "isinf" and self.emitting_host_checks:
                # The pinned CPU pipeline leaves math.isinf untranslated.
                # Ordered comparisons preserve both infinity signs and reject NaN.
                saved = self.unique("host_isinf")
                self.emit(f"{saved} = {value}")
                compare_type = "cutlass.Float32" if dtype in ("float16", "bfloat16") else type_name
                positive = f"{compare_type}(float('inf'))"
                negative = f"{compare_type}(float('-inf'))"
                return f"cutlass.Boolean(({saved} == {positive}) | ({saved} == {negative}))"
            if op == "exp10":
                symbol = "__nv_exp10f" if dtype == "float32" else "__nv_exp10"
                return self.math_external(symbol, dtype, [value], (dtype,))
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
            return self.float_minmax(function, dtype, values[0], values[1], op in ("maximum", "minimum"))
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

    def loop_tile(self, shape, *, constexpr: bool = False):
        plan = Partition(shape, self.kernel.threads)
        slot, flat = self.unique("slot"), self.unique("flat")
        if constexpr or plan.slots <= 16:
            self.emit(f"for {slot} in cutlass.range_constexpr({plan.slots}):")
        else:
            self.emit(f"for {slot} in cutlass.range({plan.slots}, unroll=4):")
        self.depth += 1
        self.emit(f"{flat} = _nt_tid + {slot} * {plan.threads}")
        total = prod(shape)
        if total % plan.threads == 0:
            return slot, self.coordinates(flat, shape), 1
        self.emit(f"if {flat} < {total}:")
        self.depth += 1
        return slot, self.coordinates(flat, shape), 2

    def region_indices(self, region, coords):
        result = []
        for origin, axis in zip(region.origin, region.axes):
            if origin.op == "const" and origin.value == 0 and axis is not None:
                result.append(coords[axis])
            elif axis is None:
                result.append(self.index_expression(origin))
            else:
                result.append(f"({self.index_expression(origin)} + {coords[axis]})")
        return result

    def region_coordinates(self, region, buffer_coords):
        coords = ["0"] * len(region.shape)
        for index, origin, axis in zip(buffer_coords, region.origin, region.axes):
            if axis is not None:
                if origin.op == "const" and origin.value == 0:
                    coords[axis] = index
                else:
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

    def copy(self, src, dst):
        snapshots = self.copy_snapshots(src, dst)
        for name in sorted(snapshots):
            self.materialize_fragment(name)
        reads = expr_smem_reads(src.origin, self.buffers) | expr_smem_reads(dst.origin, self.buffers)
        writes: set[str] = set()
        if src.buffer in snapshots:
            reads.add(f"_nt_snapshot_{src.buffer}")
        elif self.buffers[src.buffer].space == "shared":
            reads.add(src.buffer)
        if self.buffers[dst.buffer].space == "shared":
            writes.add(dst.buffer)
        self.sync_if_hazard(reads, writes)
        vec_plan = None if snapshots else self._vectorized_copy_plan(src, dst)
        if vec_plan is not None:
            self._emit_vectorized_copy(src, dst, vec_plan)
            self.hazards.record(reads, writes)
            return
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
            slot, coords, depth = self.loop_tile(src.shape)
        src_idx, dst_idx = self.region_indices(src, coords), self.region_indices(dst, coords)
        if self.buffers[src.buffer].space == "fragment" and src.buffer not in snapshots:
            value = f"{self.buf(src.buffer)}[{slot}]"
        else:
            source = (
                f"_nt_snapshot_{src.buffer}[{', '.join(src_idx)}]"
                if src.buffer in snapshots
                else self.access(src.buffer, src_idx)
            )
            src_pred = self.region_in_bounds_predicate(src, src_idx)
            if src_pred is None:
                value = source
            else:
                value = self.unique("copy_value")
                self.emit(f"{value} = {self.dtype(src.buffer)}(0)")
                self.emit(f"if {src_pred}:")
                self.depth += 1
                self.emit(f"{value} = {source}")
                self.depth -= 1
        if self.buffers[dst.buffer].space == "fragment":
            self.emit(f"{self.buf(dst.buffer)}[{slot}] = {self.dtype(dst.buffer)}({value})")
        else:
            dst_pred = self.region_in_bounds_predicate(dst, dst_idx)
            if dst_pred is None:
                self.emit(f"{self.access(dst.buffer, dst_idx)} = {self.dtype(dst.buffer)}({value})")
            else:
                self.emit(f"if {dst_pred}:")
                self.depth += 1
                self.emit(f"{self.access(dst.buffer, dst_idx)} = {self.dtype(dst.buffer)}({value})")
                self.depth -= 1
        self.depth -= depth
        self.hazards.record(reads, writes)

    def _shared_mma_view(
        self,
        name: str,
        shape: tuple[int, int],
        stride: tuple[int, int],
        native: bool,
        *,
        tensor_expr: str | None = None,
        swizzle_key: str | None = None,
    ):
        expr = self.buf(name) if tensor_expr is None else tensor_expr
        swz_key = name if swizzle_key is None else swizzle_key
        stage = self.active_stage.get(name) if tensor_expr is None else None
        if native:
            return f"{expr}[None, None, {stage}]" if stage is not None else expr
        view = self.unique("mma_view")
        swz = self.shared_swizzles.get(swz_key)
        if stage is not None:
            stages = self.staged_buffers[name]
            rows, cols = self.buffers[name].type.shape
            base = f"cute.make_layout(({shape[0]}, {shape[1]}, {stages}), stride=({stride[0]}, {stride[1]}, {rows * cols}))"
            layout_expr = (
                f"cute.make_composed_layout(cute.make_swizzle{swz}, 0, {base})" if swz is not None else base
            )
            self.emit(f"{view} = cute.make_tensor({expr}.iterator, {layout_expr})[None, None, {stage}]")
            return view
        base = f"cute.make_layout({shape}, stride={stride})"
        layout_expr = (
            f"cute.make_composed_layout(cute.make_swizzle{swz}, 0, {base})" if swz is not None else base
        )
        self.emit(f"{view} = cute.make_tensor({expr}.iterator, {layout_expr})")
        return view

    def gemm(self, a, b, c, ta, tb):
        am, bn, ak, wm, wn, mma_dtype = self.mmas[c]
        cutlass_ty = f"cutlass.{CUTLASS_TYPES[mma_dtype]}"
        if self.buffers[a].space == "fragment":
            self.materialize_fragment(a)
            snap_a = f"_nt_snapshot_{a}"
            reads = {snap_a, b}
            self.sync_if_hazard(reads, set())
            a_view = self._shared_mma_view(
                a,
                (am, ak),
                (1, self.buffers[a].type.shape[1]),
                native=not ta,
                tensor_expr=snap_a,
                swizzle_key=snap_a,
            )
        else:
            reads = {a, b}
            self.sync_if_hazard(reads, set())
            a_view = self._shared_mma_view(a, (am, ak), (1, self.buffers[a].type.shape[1]), native=not ta)
        # CuTe MMA B has (N, K) modes; source language B has (K, N) unless transpose_B=True.
        b_view = self._shared_mma_view(b, (bn, ak), (1, bn), native=tb)

        b_num_mat = 4 if (bn // wn) % 16 == 0 else 2
        atom_a, atom_b = self.unique("ldm_a"), self.unique("ldm_b")
        tc_a, tc_b = self.unique("tc_s2r_a"), self.unique("tc_s2r_b")
        thr_a, thr_b = self.unique("thr_s2r_a"), self.unique("thr_s2r_b")
        ts_a, ts_b = self.unique("tCsA"), self.unique("tCsB")
        ar, br = self.unique("a_reg"), self.unique("b_reg")
        ar_view, br_view = self.unique("a_reg_v"), self.unique("b_reg_v")

        self.emit(
            f"{atom_a} = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose={ta}, num_matrices=4), {cutlass_ty})"
        )
        self.emit(
            f"{atom_b} = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose={not tb}, num_matrices={b_num_mat}), {cutlass_ty})"
        )
        self.emit(f"{tc_a} = cute.make_tiled_copy_A({atom_a}, _nt_mma_{c})")
        self.emit(f"{tc_b} = cute.make_tiled_copy_B({atom_b}, _nt_mma_{c})")
        self.emit(f"{thr_a} = {tc_a}.get_slice(_nt_tid)")
        self.emit(f"{thr_b} = {tc_b}.get_slice(_nt_tid)")
        self.emit(f"{ts_a} = {thr_a}.partition_S({a_view})")
        self.emit(f"{ts_b} = {thr_b}.partition_S({b_view})")
        self.emit(f"{ar} = _nt_mma_{c}.make_fragment_A(_nt_thr_{c}.partition_A({a_view}))")
        self.emit(f"{br} = _nt_mma_{c}.make_fragment_B(_nt_thr_{c}.partition_B({b_view}))")
        self.emit(f"{ar_view} = {thr_a}.retile({ar})")
        self.emit(f"{br_view} = {thr_b}.retile({br})")

        k = self.unique("mma_k")
        self.emit(f"for {k} in cutlass.range_constexpr(cute.size({ar}, mode=[2])):")
        self.depth += 1
        self.emit(f"cute.copy({tc_a}, {ts_a}[None, None, {k}], {ar_view}[None, None, {k}])")
        self.emit(f"cute.copy({tc_b}, {ts_b}[None, None, {k}], {br_view}[None, None, {k}])")
        self.emit(
            f"cute.gemm(_nt_mma_{c}, {self.buf(c)}, {ar}[None, None, {k}], {br}[None, None, {k}], {self.buf(c)})"
        )
        self.depth -= 1
        self.hazards.record(reads, set())

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
                return self.float_minmax(operation, dtype, left, right, propagate)
            return f"({left} { {'bitand': '&', 'bitor': '|', 'bitxor': '^'}[kind] } {right})"

        stage0_reads = {src} if self.buffers[src].space == "shared" else set()
        stage0_writes = {workspace}
        self.sync_if_hazard(stage0_reads, stage0_writes)

        can_warp_shuffle = (
            dim == len(shape) - 1
            and self.kernel.threads % 32 == 0
            and shape[dim] >= 32
            and shape[dim] % 32 == 0
            and dtype in ("float32", "int32", "uint32")
        )

        if can_warp_shuffle:
            groups = shape[dim] // 32
            outer_shape = shape[:-1]
            outer_count = prod(outer_shape)
            is_mma_src = self.buffers[src].space == "fragment" and self.fragment_layouts[src] is not None
            if is_mma_src:
                slot, coords, depth = self.loop_fragment(src)
                value_name = self.unique("reduce_value")
                self.emit(f"{value_name} = {self.dtype(dst)}({self.buf(src)}[{slot}])")
                if kind in ("abssum", "absmax") and not (dtype.startswith("uint") or dtype == "bool"):
                    value_name = self.float_minmax("max", dtype, value_name, f"-{value_name}", False)
                self.emit(f"{access(coords)} = {self.dtype(dst)}({value_name})")
                self.depth -= depth
                self.hazards.record(stage0_reads, stage0_writes)
                self.sync_threads()

            slot, coords, depth = self.loop_tile(
                shape, constexpr=(self.buffers[src].space == "fragment" and not is_mma_src)
            )
            if is_mma_src:
                value = access(coords)
            elif self.buffers[src].space == "fragment":
                value = f"{self.buf(src)}[{slot}]"
            else:
                value = self.access(src, coords)
            val = self.unique("wred_val")
            self.emit(f"{val} = {self.dtype(dst)}({value})")
            if (
                not is_mma_src
                and kind in ("abssum", "absmax")
                and not (dtype.startswith("uint") or dtype == "bool")
            ):
                val = self.float_minmax("max", dtype, val, f"-{val}", False)
                val_var = self.unique("wred_abs")
                self.emit(f"{val_var} = {self.dtype(dst)}({val})")
                val = val_var
            for offset in (1, 2, 4, 8, 16):
                other = self.unique("wred_other")
                self.emit(f"{other} = cute.arch.shuffle_sync_down({val}, {offset})")
                self.emit(f"{val} = {self.dtype(dst)}({combine(val, other)})")
            self.emit("if (_nt_tid % 32) == 0:")
            self.depth += 1
            if is_mma_src:
                # Write in-place at group leader (32 * g) to avoid WAR with other warps.
                self.emit(f"{access(coords)} = {val}")
            else:
                group_coords = [*coords[:-1], f"({coords[-1]} // 32)"]
                self.emit(f"{access(group_coords)} = {val}")
            self.depth -= depth + 1
            self.hazards.record({workspace} if is_mma_src else stage0_reads, stage0_writes)

            group_stride = 32 if is_mma_src else 1
            if 2 <= groups <= 32:
                self.sync_threads()
                num_warps = self.kernel.threads // 32
                row_slots = (outer_count + num_warps - 1) // num_warps
                r_slot = self.unique("wred_row_slot")
                r_flat = self.unique("wred_row")
                lane = self.unique("wred_lane")
                self.emit(f"{lane} = _nt_tid % 32")
                loop_hdr = (
                    f"cutlass.range_constexpr({row_slots})"
                    if row_slots <= 16
                    else f"cutlass.range({row_slots}, unroll=4)"
                )
                self.emit(f"for {r_slot} in {loop_hdr}:")
                self.depth += 1
                self.emit(f"{r_flat} = (_nt_tid // 32) + {r_slot} * {num_warps}")
                r_depth = 1
                if outer_count % num_warps != 0:
                    self.emit(f"if {r_flat} < {outer_count}:")
                    self.depth += 1
                    r_depth += 1
                row_coords = self.coordinates(r_flat, outer_shape)
                val2 = self.unique("wred_gval")
                self.emit(f"{val2} = {self.dtype(dst)}(0)")
                self.emit(f"if {lane} < {groups}:")
                self.depth += 1
                g_idx = f"({lane} * {group_stride})" if group_stride > 1 else lane
                self.emit(f"{val2} = {access([*row_coords, g_idx])}")
                self.depth -= 1
                stride = 1
                while stride < groups:
                    other2 = self.unique("wred_gother")
                    self.emit(f"{other2} = cute.arch.shuffle_sync_down({val2}, {stride})")
                    self.emit(f"if ({lane} % {2 * stride} == 0) and ({lane} + {stride} < {groups}):")
                    self.depth += 1
                    self.emit(f"{val2} = {self.dtype(dst)}({combine(val2, other2)})")
                    self.depth -= 1
                    stride *= 2
                self.emit(f"if {lane} == 0:")
                self.depth += 1
                self.emit(f"{access([*row_coords, '0'])} = {val2}")
                self.depth -= r_depth + 1
                self.hazards.record({workspace}, {workspace})
            elif groups > 32:
                reduced_shape = (*outer_shape, groups)
                stride = 1
                while stride < groups:
                    self.sync_threads()
                    _, coords, depth = self.loop_tile(reduced_shape)
                    self.emit(
                        f"if ({coords[dim]} % {2 * stride} == 0) and ({coords[dim]} + {stride} < {groups}):"
                    )
                    self.depth += 1
                    cur_coords = (
                        [*coords[:-1], f"({coords[-1]} * {group_stride})"] if group_stride > 1 else coords
                    )
                    partner = coords.copy()
                    partner[dim] = (
                        f"(({coords[dim]} + {stride}) * {group_stride})"
                        if group_stride > 1
                        else f"({coords[dim]} + {stride})"
                    )
                    result = combine(access(cur_coords), access(partner))
                    self.emit(f"{access(cur_coords)} = {self.dtype(dst)}({result})")
                    self.depth -= depth + 1
                    self.hazards.record({workspace}, {workspace})
                    stride *= 2
        else:
            if self.buffers[src].space == "fragment":
                slot, coords, depth = self.loop_fragment(src)
                value = f"{self.buf(src)}[{slot}]"
            else:
                _, coords, depth = self.loop_tile(shape)
                value = self.access(src, coords)
            value_name = self.unique("reduce_value")
            self.emit(f"{value_name} = {self.dtype(dst)}({value})")
            if kind in ("abssum", "absmax") and not (dtype.startswith("uint") or dtype == "bool"):
                value_name = self.float_minmax("max", dtype, value_name, f"-{value_name}", False)
            self.emit(f"{access(coords)} = {self.dtype(dst)}({value_name})")
            self.depth -= depth
            self.hazards.record(stage0_reads, stage0_writes)
            # Disjoint pairs in each tree level: right operands are never written
            # at that level, and the block reconverges before the following level.
            stride = 1
            while stride < shape[dim]:
                self.sync_threads()
                _, coords, depth = self.loop_tile(shape)
                self.emit(
                    f"if ({coords[dim]} % {2 * stride} == 0) and ({coords[dim]} + {stride} < {shape[dim]}):"
                )
                self.depth += 1
                partner = coords.copy()
                partner[dim] = f"({coords[dim]} + {stride})"
                result = combine(access(coords), access(partner))
                self.emit(f"{access(coords)} = {self.dtype(dst)}({result})")
                self.depth -= depth + 1
                self.hazards.record({workspace}, {workspace})
                stride *= 2
        final_reads = {workspace}
        final_writes: set[str] = set()
        if self.buffers[dst].space == "shared":
            final_writes.add(dst)
            if not clear:
                final_reads.add(dst)
        self.sync_if_hazard(final_reads, final_writes)
        if self.buffers[dst].space == "fragment":
            slot, coords, depth = self.loop_fragment(dst)
            destination = f"{self.buf(dst)}[{slot}]"
        else:
            _, coords, depth = self.loop_tile(out_shape)
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
        self.hazards.record(final_reads, final_writes)

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
                before = self.hazards.clone()
                alive, skip = self.control
                self.emit(f"if {alive} & ~{skip}:")
                self.depth += 1
                self.statement(stmt)
                self.hazards.merge_from(before)
            else:
                self.statement(stmt)
            self.depth = depth

    def statement(self, stmt):
        op, args = stmt.op, stmt.args
        self.emit(f"# {self.kernel.name}:{stmt.location.line} {op}")
        if op in ("alloc", "annotate_layout"):
            return
        if op == "use_swizzle":
            panel_size, order, enable = args
            if enable and panel_size > 1 and len(self.kernel.grid) >= 2:
                gx, gy = self.kernel.grid[0], self.kernel.grid[1]
                bx_v = self.var(self.kernel.block_vars[0])
                by_v = self.var(self.kernel.block_vars[1])
                pid = self.unique("swz_pid")
                gid = self.unique("swz_gid")
                first = self.unique("swz_first")
                gsize = self.unique("swz_gsize")
                rem = self.unique("swz_rem")
                self.emit(f"{pid} = cutlass.Int32({bx_v}) + cutlass.Int32({by_v}) * cutlass.Int32({gx})")
                if order == "row":
                    panel_width = panel_size * gy
                    self.emit(f"{gid} = {pid} // {panel_width}")
                    self.emit(f"{first} = {gid} * {panel_size}")
                    self.emit(f"{gsize} = cutlass.Int32({panel_size})")
                    self.emit(f"if (cutlass.Int32({gx}) - {first}) < cutlass.Int32({panel_size}):")
                    self.depth += 1
                    self.emit(f"{gsize} = cutlass.Int32({gx}) - {first}")
                    self.depth -= 1
                    self.emit(f"{rem} = {pid} % {panel_width}")
                    self.emit(f"{bx_v} = {first} + ({rem} % {gsize})")
                    self.emit(f"{by_v} = {rem} // {gsize}")
                else:
                    panel_width = panel_size * gx
                    self.emit(f"{gid} = {pid} // {panel_width}")
                    self.emit(f"{first} = {gid} * {panel_size}")
                    self.emit(f"{gsize} = cutlass.Int32({panel_size})")
                    self.emit(f"if (cutlass.Int32({gy}) - {first}) < cutlass.Int32({panel_size}):")
                    self.depth += 1
                    self.emit(f"{gsize} = cutlass.Int32({gy}) - {first}")
                    self.depth -= 1
                    self.emit(f"{rem} = {pid} % {panel_width}")
                    self.emit(f"{by_v} = {first} + ({rem} % {gsize})")
                    self.emit(f"{bx_v} = {rem} // {gsize}")
            return
        if op == "pass":
            self.emit("pass")
        elif op == "evaluate":
            reads = expr_smem_reads(args[0], self.buffers)
            if self.parallel is None and reads:
                self.sync_if_hazard(reads, set())
            value = self.expression(args[0])
            self.emit(f"{self.unique('evaluate')} = {value}")
            if self.parallel is None and reads:
                self.hazards.record(reads, set())
        elif op == "print":
            self.debug_print(*args)
        elif op == "device_assert":
            reads = expr_smem_reads(args[0], self.buffers)
            if self.parallel is None and reads:
                self.sync_if_hazard(reads, set())
            self.device_asserts = True
            value = self.expression(args[0])
            condition = self.unique("assert_condition")
            self.emit(f"{condition} = cutlass.Boolean({value})")
            if args[1]:
                self.emit(f"if not {condition}:")
                self.depth += 1
                self.emit(f"cute.printf('Device assert failed: %s\\n', {self.debug_string(args[1])}, end='')")
                self.depth -= 1
            self.emit(f"_nt_device_assert({condition}, 'Device assertion failed')")
            if self.parallel is None and reads:
                self.hazards.record(reads, set())
        elif op in ("break", "continue"):
            alive, skip = self.control
            self.emit(f"{skip} = cutlass.Boolean(True)")
            if op == "break":
                self.emit(f"{alive} = cutlass.Boolean(False)")
        elif op == "if":
            condition, then_body, else_body = args
            cond_reads = expr_smem_reads(condition, self.buffers)
            if self.parallel is None and cond_reads:
                self.sync_if_hazard(cond_reads, set())
            value = self.expression(condition)
            if self.parallel is None and cond_reads:
                self.hazards.record(cond_reads, set())
            merged_types = body_types((stmt,), self.scalar_types, self.buffers)
            for name in sorted(merged_types.keys() - self.scalar_types.keys()):
                self.emit(f"{self.var(name)} = cutlass.{CUTLASS_TYPES[merged_types[name]]}(0)")
            old_bounds = self.bounds.copy()
            old_defs = self.definitions.copy()
            pre_hazards = self.hazards.clone()
            self.scalar_types = merged_types.copy()
            then_bounds = predicate_bounds(condition, True, old_bounds, old_defs, self.buffers)
            self.bounds = old_bounds.copy() if then_bounds is None else then_bounds
            self.definitions = old_defs.copy()
            self.hazards = pre_hazards.clone()
            self.emit(f"if {value}:")
            self.depth += 1
            self.statements(then_body)
            self.depth -= 1
            then_hazards = self.hazards
            self.scalar_types = merged_types.copy()
            self.hazards = pre_hazards.clone()
            if else_body:
                else_bounds = predicate_bounds(condition, False, old_bounds, old_defs, self.buffers)
                self.bounds = old_bounds.copy() if else_bounds is None else else_bounds
                self.definitions = old_defs.copy()
                self.emit("else:")
                self.depth += 1
                self.statements(else_body)
                self.depth -= 1
            else_hazards = self.hazards
            then_hazards.merge_from(else_hazards)
            self.hazards = then_hazards
            self.bounds = old_bounds
            self.definitions = old_defs
            self.scalar_types = merged_types
        elif op == "fill":
            name, value = args
            reads = expr_smem_reads(value, self.buffers)
            writes = {name} if self.buffers[name].space == "shared" else set()
            if self.parallel is None:
                self.sync_if_hazard(reads, writes)
            text = self.expression(value)
            if self.buffers[name].space == "fragment":
                self.emit(f"{self.buf(name)}.fill({self.dtype(name)}({text}))")
            else:
                _, coords, depth = self.loop_tile(self.buffers[name].type.shape)
                self.emit(f"{self.access(name, coords)} = {self.dtype(name)}({text})")
                self.depth -= depth
            if self.parallel is None:
                self.hazards.record(reads, writes)
        elif op == "while":
            condition, inner = args
            if expression_dtype(condition, self.buffers, self.scalar_types) != "bool":
                raise CompileError("While conditions require Boolean expressions", stmt.location)
            cond_reads = expr_smem_reads(condition, self.buffers)
            if self.parallel is None and cond_reads:
                self.sync_if_hazard(cond_reads, set())
            predicate = self.unique("while_condition")
            control = self.loop_control(inner)
            self.emit(f"{predicate} = cutlass.Boolean({self.expression(condition)})")
            if self.parallel is None and cond_reads:
                self.hazards.record(cond_reads, set())
            pre_loop_hazards = self.hazards.clone()
            old_bounds = self.bounds.copy()
            old_defs = self.definitions.copy()
            self.emit(f"while {predicate}:")
            self.depth += 1
            old_types = self.scalar_types.copy()
            self.loop_body(inner, control)
            self.scalar_types = old_types
            self.bounds = old_bounds.copy()
            self.definitions = old_defs.copy()
            if self.parallel is None and cond_reads:
                self.sync_if_hazard(cond_reads, set())
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
            if self.parallel is None:
                if cond_reads:
                    self.hazards.record(cond_reads, set())
                body_reads, body_writes = smem_footprint(
                    inner, self.buffers, self.reduction_workspaces, self.scan_workspaces
                )
                body_reads |= cond_reads
                self.sync_if_hazard(body_reads, body_writes)
            self.depth -= 1
            if self.parallel is None:
                self.hazards.merge_from(pre_loop_hazards)
            self.bounds = old_bounds
            self.definitions = old_defs
        elif op == "copy":
            self.copy(*args)
        elif op == "gemm":
            self.gemm(*args)
        elif op == "reduce":
            self.reduction(*args)
        elif op == "scan":
            scan.emit(self, *args)
        elif op == "let":
            reads = expr_smem_reads(args[1], self.buffers)
            if self.parallel is None and reads:
                self.sync_if_hazard(reads, set())
            value = self.expression(args[1])
            dtype = self.scalar_types.get(args[0]) or expression_dtype(
                args[1], self.buffers, self.scalar_types
            )
            self.scalar_types[args[0]] = dtype
            self.definitions[args[0]] = args[1]
            self.emit(f"{self.var(args[0])} = cutlass.{CUTLASS_TYPES[dtype]}({value})")
            if self.parallel is None and reads:
                self.hazards.record(reads, set())
        elif op == "declare":
            name, dtype, initializer = args
            reads = expr_smem_reads(initializer, self.buffers)
            if self.parallel is None and reads:
                self.sync_if_hazard(reads, set())
            value = self.expression(initializer)
            self.scalar_types[name] = dtype
            self.definitions.pop(name, None)
            self.bounds.pop(name, None)
            self.emit(f"{self.var(name)} = {value}")
            if self.parallel is None and reads:
                self.hazards.record(reads, set())
        elif op == "assign":
            reads = expr_smem_reads(args[1], self.buffers)
            if self.parallel is None and reads:
                self.sync_if_hazard(reads, set())
            self.definitions.pop(args[0], None)
            self.bounds.pop(args[0], None)
            self.emit(f"{self.var(args[0])} = {self.expression(args[1])}")
            if self.parallel is None and reads:
                self.hazards.record(reads, set())
        elif op == "store":
            name, indices, value = args
            reads = expr_smem_reads(indices, self.buffers) | expr_smem_reads(value, self.buffers)
            writes = {name} if self.buffers[name].space == "shared" else set()
            if self.parallel is None:
                self.sync_if_hazard(reads, writes)
            coords = [self.index_expression(x) for x in indices]
            value_text = self.expression(value)
            if self.buffers[name].space == "fragment":
                slot = self.fragment_slot(name, indices)
                self.emit(f"{slot} = {self.dtype(name)}({value_text})")
                if self.parallel is None and reads:
                    self.hazards.record(reads, set())
                return
            pred = self.predicate_for_exprs(name, indices, coords)
            if pred is None:
                self.emit(f"{self.access(name, coords)} = {self.dtype(name)}({value_text})")
            else:
                self.emit(f"if {pred}:")
                self.depth += 1
                self.emit(f"{self.access(name, coords)} = {self.dtype(name)}({value_text})")
                self.depth -= 1
            if self.parallel is None:
                self.hazards.record(reads, writes)
        elif op in ("serial", "unroll"):
            names, extents, inner = args
            extent_reads = expr_smem_reads(extents[:2], self.buffers)
            if self.parallel is None and extent_reads:
                self.sync_if_hazard(extent_reads, set())
                self.hazards.record(extent_reads, set())
            static_domain = all(type(value) is int for value in extents)
            if static_domain:
                static_range = range(*extents)
                trip_count = len(static_range)
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
            pre_loop_hazards = self.hazards.clone()
            old_bounds = self.bounds.copy()
            old_defs = self.definitions.copy()
            self.emit(f"for {ordinal} in {loop}:")
            self.depth += 1
            self.emit(
                f"{self.var(names[0])} = cutlass.Int32(cutlass.Int64({start}) + cutlass.Int64({ordinal}) * {extents[2]})"
            )
            old_types = self.scalar_types.copy()
            self.scalar_types[names[0]] = "int32"
            if static_domain:
                self.bounds[names[0]] = (
                    min(static_range[0], static_range[-1]),
                    max(static_range[0], static_range[-1]),
                )
            else:
                self.bounds.pop(names[0], None)
            self.definitions.pop(names[0], None)
            self.loop_body(inner, control)
            self.scalar_types = old_types
            self.bounds = old_bounds
            self.definitions = old_defs
            if self.parallel is None and (not static_domain or trip_count > 1):
                body_reads, body_writes = smem_footprint(
                    inner, self.buffers, self.reduction_workspaces, self.scan_workspaces
                )
                self.sync_if_hazard(body_reads, body_writes)
            self.depth -= 1
            if self.parallel is None and not static_domain:
                self.hazards.merge_from(pre_loop_hazards)
        elif op == "pipelined":
            names, extents, inner, _ = args
            if id(stmt) not in self.pipelined_plans or self.parallel is not None:
                fallback = Statement("serial", (names, extents, inner), stmt.location, stmt.annotations)
                self.statement(fallback)
                return
            num_stages, producers, consumers = self.pipelined_plans[id(stmt)]
            static_range = range(*extents)
            trip_count = len(static_range)
            start, step = extents[0], extents[2]
            loop_var = names[0]
            staged_names = {dst.buffer for _, dst in producers}

            old_types = self.scalar_types.copy()
            old_bounds = self.bounds.copy()
            old_defs = self.definitions.copy()
            self.scalar_types[loop_var] = "int32"
            self.bounds[loop_var] = (
                min(static_range[0], static_range[-1]),
                max(static_range[0], static_range[-1]),
            )
            self.definitions.pop(loop_var, None)

            # Prologue: asynchronously prefetch the first (num_stages - 1) tiles.
            p_var = self.unique("pipe_pre")
            self.emit(f"for {p_var} in cutlass.range_constexpr({num_stages - 1}):")
            self.depth += 1
            guarded_pro = trip_count < num_stages - 1
            if guarded_pro:
                self.emit(f"if {p_var} < {trip_count}:")
                self.depth += 1
            self.emit(
                f"{self.var(loop_var)} = cutlass.Int32(cutlass.Int64({start}) + cutlass.Int64({p_var}) * {step})"
            )
            for src, dst in producers:
                plan = self._vectorized_copy_plan(src, dst)
                self._emit_async_copy(src, dst, plan, p_var)
            if guarded_pro:
                self.depth -= 1
            self.emit("cute.arch.cp_async_commit_group()")
            self.depth -= 1

            # Mainloop: wait oldest stage, overlap next prefetch with consumer compute.
            it = self.unique("pipe_it")
            r_stage = self.unique("rstage")
            w_stage = self.unique("wstage")
            pref_it = self.unique("pref_it")
            self.emit(f"for {it} in range({trip_count}):")
            self.depth += 1
            self.emit(f"{r_stage} = {it} % {num_stages}")
            self.emit(f"{w_stage} = ({it} + {num_stages - 1}) % {num_stages}")
            self.emit(f"cute.arch.cp_async_wait_group({num_stages - 2})")
            self.sync_threads()
            self.emit(f"{pref_it} = {it} + {num_stages - 1}")
            self.emit(f"if {pref_it} < {trip_count}:")
            self.depth += 1
            self.emit(
                f"{self.var(loop_var)} = cutlass.Int32(cutlass.Int64({start}) + cutlass.Int64({pref_it}) * {step})"
            )
            for src, dst in producers:
                plan = self._vectorized_copy_plan(src, dst)
                self._emit_async_copy(src, dst, plan, w_stage)
            self.depth -= 1
            self.emit("cute.arch.cp_async_commit_group()")
            self.emit(
                f"{self.var(loop_var)} = cutlass.Int32(cutlass.Int64({start}) + cutlass.Int64({it}) * {step})"
            )

            old_stage = self.active_stage.copy()
            for name in staged_names:
                self.active_stage[name] = r_stage
            self.statements(consumers)
            self.active_stage = old_stage
            self.hazards.reads -= staged_names
            self.hazards.writes -= staged_names
            self.depth -= 1

            # Epilogue: drain any empty commit groups and leave shared state clean.
            self.emit("cute.arch.cp_async_wait_group(0)")
            self.sync_threads()
            self.scalar_types = old_types
            self.bounds = old_bounds
            self.definitions = old_defs
        elif op == "parallel":
            names, shape, inner = args
            fragments, snapshots, _ = self.parallel_accesses(stmt)
            for name in sorted(snapshots):
                self.materialize_fragment(name)
            p_reads, p_writes = smem_footprint(
                inner, self.buffers, self.reduction_workspaces, self.scan_workspaces
            )
            p_reads.update(f"_nt_snapshot_{name}" for name in snapshots)
            self.sync_if_hazard(p_reads, p_writes)
            self.active_snapshots = snapshots
            fragments = sorted(fragments)
            if fragments:
                if any(self.buffers[n].type.shape != shape for n in fragments):
                    raise CompileError(
                        "Fragment indexing requires a matching T.Parallel shape", stmt.location
                    )
                slot, coords, depth = self.loop_fragment(fragments[0])
            else:
                slot, coords, depth = self.loop_tile(shape)
            for name, coord in zip(names, coords):
                self.emit(f"{self.var(name)} = {coord}")
            old_types = self.scalar_types.copy()
            old_bounds = self.bounds.copy()
            old_defs = self.definitions.copy()
            self.scalar_types.update({name: "int32" for name in names})
            for name, dim in zip(names, shape):
                self.bounds[name] = (0, dim - 1)
                self.definitions.pop(name, None)
            self.parallel = (shape, names, slot)
            previous_control = self.control
            self.control = None
            self.statements(inner)
            self.control = previous_control
            self.scalar_types = old_types
            self.bounds = old_bounds
            self.definitions = old_defs
            self.parallel = None
            self.active_snapshots = set()
            self.depth -= depth
            self.hazards.record(p_reads, p_writes)
        else:
            raise CompileError(f"Unsupported IR operation {op}", stmt.location)

    @staticmethod
    def debug_string(text):
        text = text.split("\0", 1)[0]
        symbol = "_nt_debug_text_" + sha256(text.encode("utf-8")).hexdigest()
        return f"_nt_debug_string({text!r}, {symbol!r})"

    def debug_call(self, message, dtype=None, value=None, buffer=None, index=None):
        key = (message, dtype, buffer)
        if key not in self.debug_prints:
            self.debug_prints[key] = f"_nt_debug_print_{len(self.debug_prints)}"
        coords = [self.var(name) for name in self.kernel.block_vars]
        coords += ["0"] * (3 - len(coords))
        arguments = [f"cutlass.Int32({coord})" for coord in (*coords, "_nt_tid")]
        if buffer is not None:
            arguments.append(f"cutlass.Int32({index})")
        if dtype is not None:
            arguments.append(f"cutlass.{CUTLASS_TYPES[dtype]}({value})")
        self.emit(f"{self.debug_prints[key]}({', '.join(arguments)})")

    def debug_print(self, obj, message, main_lane):
        if isinstance(obj, Expr):
            reads = expr_smem_reads(obj, self.buffers)
            if self.parallel is None and reads:
                self.sync_if_hazard(reads, set())
            self.debug_call(
                message, expression_dtype(obj, self.buffers, self.scalar_types), self.expression(obj)
            )
            if self.parallel is None and reads:
                self.hazards.record(reads, set())
            return
        if obj is None:
            self.debug_call(message)
            return
        print_reads: set[str] = set()
        if obj.space == "fragment":
            self.materialize_fragment(obj.name)
            print_reads.add(f"_nt_snapshot_{obj.name}")
        elif obj.space == "shared":
            print_reads.add(obj.name)
        if self.parallel is None and print_reads:
            self.sync_if_hazard(print_reads, set())
        if obj.space != "global":
            # No matching lane means no output, including out-of-range selectors.
            self.emit(f"if _nt_tid == {main_lane}:")
            self.depth += 1
        index = self.unique("print_index")
        self.emit(f"for {index} in cutlass.range({prod(obj.type.shape)}):")
        self.depth += 1
        coords = self.coordinates(index, obj.type.shape)
        value = (
            f"_nt_snapshot_{obj.name}[{', '.join(coords)}]"
            if obj.space == "fragment"
            else self.access(obj.name, coords)
        )
        self.debug_call(message, obj.type.dtype, value, obj.source_name or obj.name, index)
        self.depth -= 1
        if obj.space != "global":
            self.depth -= 1
        if self.parallel is None and print_reads:
            self.hazards.record(print_reads, set())

    def emit_debug_helpers(self):
        if self.debug_prints or self.device_asserts:
            # Use a real C string operand. CuTe's brace substitution does not
            # escape {{}}; embedding user text in its format can consume numeric
            # operands. Globals also keep the argument count independent of text.
            self.emit("@dsl_user_op")
            self.emit("def _nt_debug_string(text, symbol, *, loc=None, ip=None):")
            self.depth = 1
            self.emit("owner = ir.InsertionPoint.current.block.owner")
            self.emit("while str(owner.name) not in ('gpu.module', 'builtin.module'):")
            self.depth += 1
            self.emit("owner = getattr(owner, 'parent_op', None) or owner.parent")
            self.depth -= 1
            self.emit("space = 4 if str(owner.name) == 'gpu.module' else 0")
            self.emit("body = owner.regions[0].blocks[0]")
            self.emit(
                "exists = any('sym_name' in op.attributes and ir.StringAttr(op.attributes['sym_name']).value == symbol for op in body)"
            )
            self.emit("if not exists:")
            self.depth += 1
            self.emit("with ir.InsertionPoint.at_block_begin(body):")
            self.depth += 1
            self.emit(
                "llvm.GlobalOp(sym_name=symbol, global_type=ir.Type.parse(f'!llvm.array<{len(text.encode(\"utf-8\")) + 1} x i8>'), linkage=ir.Attribute.parse('#llvm.linkage<internal>'), constant=True, addr_space=space, value=ir.StringAttr.get(text + '\\0'))"
            )
            self.depth -= 2
            self.emit(
                "pointer = llvm.AddressOfOp(llvm.PointerType.get(space), symbol, loc=loc, ip=ip).result"
            )
            self.emit(
                "pointer = llvm.addrspacecast(llvm.PointerType.get(0), pointer, loc=loc, ip=ip) if space else pointer"
            )
            self.emit("return cute.make_ptr(cutlass.Int8, pointer, assumed_align=1, loc=loc, ip=ip)")
            self.depth = 0
            self.emit()
        if self.device_asserts:
            self.emit("from cutlass.cute.testing import assert_ as _nt_device_assert")
            self.emit()
        for (message, dtype, buffer), helper in self.debug_prints.items():
            arguments = [f"{name}: cutlass.Int32" for name in ("bx", "by", "bz", "tx")]
            values = [self.debug_string(message), "bx", "by", "bz", "tx"]
            if buffer is not None:
                arguments.append("index: cutlass.Int32")
                values.extend([self.debug_string(buffer), "index"])
            if dtype is not None:
                arguments.append(f"value: cutlass.{CUTLASS_TYPES[dtype]}")
            self.emit("@cute.jit")
            self.emit(f"def {helper}({', '.join(arguments)}):")
            self.depth = 1
            if dtype == "bool":
                self.emit("if value:")
                self.depth += 1
                fmt = print_format(dtype, buffer, boolean=True)
                self.emit(f"cute.printf({fmt!r}, {', '.join(values)}, end='')")
                self.depth -= 1
                self.emit("else:")
                self.depth += 1
                fmt = print_format(dtype, buffer, boolean=False)
            else:
                if dtype is not None:
                    typ = (
                        "Float32"
                        if dtype in ("float16", "bfloat16")
                        else "Int32"
                        if dtype in ("int8", "int16")
                        else "Uint32"
                        if dtype in ("uint8", "uint16")
                        else CUTLASS_TYPES[dtype]
                    )
                    values.append(f"cutlass.{typ}(value)")
                fmt = print_format(dtype, buffer)
            self.emit(f"cute.printf({fmt!r}, {', '.join(values)}, end='')")
            self.depth = 0
            self.emit()

    def math_external(self, symbol, dtype, values, dtypes):
        self.math_externs[symbol] = (dtype, dtypes)
        arguments = ", ".join(f"cutlass.{CUTLASS_TYPES[typ]}({value})" for typ, value in zip(dtypes, values))
        return f"cutlass.{CUTLASS_TYPES[dtype]}(_nt_math{symbol}({arguments}))"

    def emit_host_checks(self, names):
        """Compile typed scalar predicates on CPU and retain Python error propagation."""
        parameters = ["_nt_output: cute.Pointer"]
        parameters.extend(
            f"{self.var(p.name)}: cutlass.{CUTLASS_TYPES[p.dtype]}" for _, p in self.host_parameters
        )
        self.emit("@cute.jit")
        self.emit(f"def _nt_check_preconditions({', '.join(parameters)}):")
        self.depth = 1
        self.emit("_nt_failed = cutlass.Int32(0)")
        self.emitting_host_checks = True
        for index, check in enumerate(self.kernel.host_checks, 1):
            # A failed assertion stops evaluation of all following predicates.
            self.emit("if _nt_failed == 0:")
            self.depth += 1
            condition = self.expression(check.args[0])
            self.emit(f"if not ({condition}):")
            self.depth += 1
            self.emit(f"_nt_failed = cutlass.Int32({index})")
            self.depth -= 2
        self.emitting_host_checks = False
        self.emit("_nt_output[0] = _nt_failed")
        self.depth = 0
        self.emit()
        errors = tuple(
            (kind.split("\0", 1)[0], "".join(part.split("\0", 1)[0] for part in parts))
            for _, parts, kind in (check.args for check in self.kernel.host_checks)
        )
        self.emit(f"_nt_host_errors = {errors!r}")
        self.emit(
            "_nt_host_error_types = {" + ", ".join(f"{name!r}: {name}" for name in HOST_ERROR_TYPES) + "}"
        )
        self.emit()
        self.emit("class _nt_CheckedExecutable:")
        self.depth = 1
        self.emit("def __init__(self, executable, checker):")
        self.depth = 2
        self.emit("self._executable = executable")
        self.emit("self._checker = checker")
        self.depth = 1
        self.emit()
        self.emit("def __getattr__(self, name):")
        self.depth = 2
        self.emit(
            "if name not in ('artifacts', 'function_name', 'has_gpu_module', '__ptx__', '__cubin__', '__sass__', '__mlir__'):"
        )
        self.depth += 1
        self.emit(
            "raise TypeError('Native callable conversion and export require separate host-check integration')"
        )
        self.depth -= 1
        self.emit("return getattr(self._executable, name)")
        self.depth = 1
        self.emit()
        signature = ", ".join(["self", *names])
        self.emit(f"def _check_arguments({signature}):")
        self.depth = 2
        self.emit("import ctypes")
        self.emit("from cutlass.cute.runtime import make_ptr")
        self.emit("_nt_status = ctypes.c_int32(0)")
        self.emit(
            "_nt_pointer = make_ptr(cutlass.Int32, ctypes.addressof(_nt_status), cute.AddressSpace.generic, assumed_align=4)"
        )
        checked = ["_nt_pointer", *(names[index] for index, _ in self.host_parameters)]
        self.emit(f"self._checker({', '.join(checked)})")
        self.emit("if _nt_status.value:")
        self.depth += 1
        self.emit("kind, message = _nt_host_errors[_nt_status.value - 1]")
        self.emit("raise _nt_host_error_types.get(kind, RuntimeError)(message)")
        self.depth = 1
        self.emit()
        self.emit(f"def __call__({signature}):")
        self.depth = 2
        self.emit(f"self._check_arguments({', '.join(names)})")
        self.emit(f"return self._executable({', '.join(names)})")
        self.depth = 0
        self.emit()

    def float_minmax(self, operation, dtype, left, right, propagate_nan):
        """Emit PTX min/max on device to preserve +0.0 > -0.0 and NaN propagation contracts."""
        assert operation in ("max", "min")
        if self.emitting_host_checks or dtype not in ("float16", "bfloat16", "float32", "float64"):
            return f"cute.math.{operation}({left}, {right}, propagate_nan={propagate_nan})"
        helper = f"_nt_minmax_{operation}_{'nan' if propagate_nan else 'num'}_{dtype}"
        self.minmax_ptx[helper] = (operation, dtype, propagate_nan)
        typ = f"cutlass.{CUTLASS_TYPES[dtype]}"
        return f"{helper}({typ}({left}), {typ}({right}))"

    def math_low_precision(self, operation, dtype, values):
        helper = f"_nt_{operation}_{dtype}"
        self.math_ptx[helper] = (operation, dtype, len(values))
        arguments = ", ".join(f"cutlass.{CUTLASS_TYPES[dtype]}({value})" for value in values)
        return f"{helper}({arguments})"

    def low_precision_unary(self, operation, dtype, value):
        """Resolve native CUDA wrappers and module-level CUTLASS aliases."""
        typ = f"cutlass.{CUTLASS_TYPES[dtype]}"
        if operation == "sigmoid":
            exponential = self.unique("sigmoid_exp")
            denominator = self.unique("sigmoid_denominator")
            self.emit(f"{exponential} = {self.low_precision_unary('exp', dtype, f'-{typ}({value})')}")
            addition = self.math_low_precision("add", dtype, [f"{typ}(1)", exponential])
            self.emit(f"{denominator} = {addition}")
            return self.math_low_precision("div", dtype, [f"{typ}(1)", denominator])
        widened = (
            operation in ("tan", "sinh", "cosh", "atan", "erf")
            or (operation == "tanh" and not self.math_header)
            or (operation in ("sin", "cos") and dtype == "bfloat16")
            or (self.math_header and operation in ("log", "sin", "cos", "sqrt"))
            or (self.math_header and operation == "exp" and dtype == "bfloat16")
        )
        if widened:
            result = self.math_external(f"__nv_{operation}f", "float32", [value], ("float32",))
            return f"{typ}({result})"
        if operation in ("sqrt", "rsqrt", "tanh"):
            return self.math_low_precision(operation, dtype, [value])
        return self.math_low_precision(f"unary_{operation}", dtype, [value])

    def generate(self):
        self.emit(f'"""Generated by Ntilang 0.1.0: {self.kernel.name}; target {self.target}."""')
        self.emit("import cutlass")
        self.emit("import cutlass.cute as cute")
        self.emit("import cutlass.utils as utils")
        self.emit()

        def parameter_name(p):
            return self.var(p.name) if isinstance(p, ScalarParameter) else self.buf(p.name)

        params = ", ".join(
            f"{parameter_name(p)}: "
            + (f"cutlass.{CUTLASS_TYPES[p.dtype]}" if isinstance(p, ScalarParameter) else "cute.Tensor")
            for p in self.kernel.parameters
        )
        self.emit("@cute.kernel")
        self.emit(f"def _nt_kernel({params}):")
        self.depth = 1
        self.emit("_nt_tid, _, _ = cute.arch.thread_idx()")
        block = [self.var(v) for v in self.kernel.block_vars] + ["_"] * (3 - len(self.kernel.block_vars))
        self.emit(f"{', '.join(block)} = cute.arch.block_idx()")
        if (
            self.snapshots
            or self.reduction_workspaces
            or self.scan_workspaces
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
                base = f"cute.make_layout({b.type.shape}, stride={strides})"
                snap_key = f"_nt_snapshot_{b.name}"
                snap_swz = self.shared_swizzles.get(snap_key)
                snap_layout = (
                    f"cute.make_composed_layout(cute.make_swizzle{snap_swz}, 0, {base})"
                    if snap_swz is not None
                    else base
                )
                self.emit(
                    f"{snap_key} = _nt_smem.allocate_tensor({self.dtype(b.name)}, {snap_layout}, byte_alignment=16)"
                )
            if b.space == "shared":
                if b.name in self.staged_buffers:
                    stages = self.staged_buffers[b.name]
                    rows, cols = b.type.shape
                    base = f"cute.make_layout(({rows}, {cols}, {stages}), stride=({cols}, 1, {rows * cols}))"
                else:
                    strides = tuple(prod(b.type.shape[i + 1 :]) for i in range(len(b.type.shape)))
                    base = f"cute.make_layout({b.type.shape}, stride={strides})"
                swz = self.shared_swizzles.get(b.name)
                layout_expr = (
                    f"cute.make_composed_layout(cute.make_swizzle{swz}, 0, {base})"
                    if swz is not None
                    else base
                )
                self.emit(
                    f"{self.buf(b.name)} = _nt_smem.allocate_tensor({self.dtype(b.name)}, {layout_expr}, byte_alignment=16)"
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
        for (shape, dtype), workspaces in self.scan_workspaces.items():
            strides = tuple(prod(shape[i + 1 :]) for i in range(len(shape)))
            for workspace in workspaces:
                self.emit(
                    f"{workspace} = _nt_smem.allocate_tensor(cutlass.{CUTLASS_TYPES[dtype]}, cute.make_layout({shape}, stride={strides}), byte_alignment=16)"
                )
        self.statements(self.kernel.body)
        self.depth = 0
        self.emit()
        if self.math_ptx or self.minmax_ptx or self.bitcasts or self.debug_prints or self.device_asserts:
            self.emit("from cutlass.cutlass_dsl import dsl_user_op")
            self.emit("from cutlass._mlir.dialects import llvm")
            self.emit()
        if self.debug_prints or self.device_asserts:
            self.emit("from cutlass._mlir import ir")
            self.emit()
        self.emit_debug_helpers()
        for helper, (source, dtype) in sorted(self.bitcasts.items()):
            typ = f"cutlass.{CUTLASS_TYPES[dtype]}"
            self.emit("@dsl_user_op")
            self.emit(f"def {helper}(arg: cutlass.{CUTLASS_TYPES[source]}, *, loc=None, ip=None):")
            self.depth = 1
            value = "arg.ir_value(loc=loc, ip=ip)"
            if not (source.startswith(("int", "uint")) and dtype.startswith(("int", "uint"))):
                value = f"llvm.bitcast({typ}.mlir_type, {value}, loc=loc, ip=ip)"
            self.emit(f"return {typ}({value})")
            self.depth = 0
            self.emit()
        for helper, (operation, dtype, arity) in sorted(self.math_ptx.items()):
            typ = f"cutlass.{CUTLASS_TYPES[dtype]}"
            self.emit("@dsl_user_op")
            arguments = ", ".join(f"arg{i}: {typ}" for i in range(arity))
            self.emit(f"def {helper}({arguments}, *, loc=None, ip=None):")
            self.depth = 1
            self.emit("operands = [")
            self.depth += 1
            for i in range(arity):
                self.emit(
                    f"llvm.bitcast(cutlass.Uint16.mlir_type, arg{i}.ir_value(loc=loc, ip=ip), loc=loc, ip=ip),"
                )
            self.depth -= 1
            self.emit("]")
            assembly = low_precision_asm(operation, dtype, self.target)
            constraints = "=h" + ",h" * arity
            self.emit(
                f"result = llvm.inline_asm(cutlass.Uint16.mlir_type, operands, {assembly!r}, {constraints!r}, has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)"
            )
            self.emit(f"return {typ}(llvm.bitcast({typ}.mlir_type, result, loc=loc, ip=ip))")
            self.depth = 0
            self.emit()
        for helper, (operation, dtype, propagate_nan) in sorted(self.minmax_ptx.items()):
            typ = f"cutlass.{CUTLASS_TYPES[dtype]}"
            assembly = minmax_asm(operation, dtype, propagate_nan)
            self.emit("@dsl_user_op")
            self.emit(f"def {helper}(arg0: {typ}, arg1: {typ}, *, loc=None, ip=None):")
            self.depth = 1
            if dtype in ("float16", "bfloat16"):
                self.emit("operands = [")
                self.depth += 1
                for i in range(2):
                    self.emit(
                        f"llvm.bitcast(cutlass.Uint16.mlir_type, arg{i}.ir_value(loc=loc, ip=ip), loc=loc, ip=ip),"
                    )
                self.depth -= 1
                self.emit("]")
                self.emit(
                    f"result = llvm.inline_asm(cutlass.Uint16.mlir_type, operands, {assembly!r}, '=h,h,h', has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)"
                )
                self.emit(f"return {typ}(llvm.bitcast({typ}.mlir_type, result, loc=loc, ip=ip))")
            else:
                constraint = "=f,f,f" if dtype == "float32" else "=d,d,d"
                self.emit("operands = [arg0.ir_value(loc=loc, ip=ip), arg1.ir_value(loc=loc, ip=ip)]")
                self.emit(
                    f"result = llvm.inline_asm({typ}.mlir_type, operands, {assembly!r}, {constraint!r}, has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)"
                )
                self.emit(f"return {typ}(result)")
            self.depth = 0
            self.emit()
        for symbol, (dtype, dtypes) in sorted(self.math_externs.items()):
            self.emit(f'@cute.extern(name="{symbol}", overloaded=False)')
            arguments = ", ".join(f"arg{i}: cutlass.{CUTLASS_TYPES[typ]}" for i, typ in enumerate(dtypes))
            self.emit(f"def _nt_math{symbol}({arguments}) -> cutlass.{CUTLASS_TYPES[dtype]}:")
            self.depth = 1
            self.emit("...")
            self.depth = 0
            self.emit()
        self.emit("@cute.jit")
        self.emit(f"def run({params}):")
        self.depth = 1
        actual = ", ".join(parameter_name(p) for p in self.kernel.parameters)
        self.emit(
            f"_nt_kernel({actual}).launch(grid={self.kernel.grid + (1,) * (3 - len(self.kernel.grid))}, block=({self.kernel.threads}, 1, 1))"
        )
        self.depth = 0
        self.emit()
        parameter_names = [parameter_name(p) for p in self.kernel.parameters]
        if self.kernel.host_checks:
            self.emit_host_checks(parameter_names)
        self.emit("def compile_kernel():")
        self.depth = 1
        self.emit("from cutlass.cute.runtime import make_fake_compact_tensor")
        names = []
        for p in self.kernel.parameters:
            name = parameter_name(p)
            names.append(name)
            if isinstance(p, ScalarParameter):
                self.emit(f"{name} = cutlass.{CUTLASS_TYPES[p.dtype]}(0)")
                continue
            order = tuple(reversed(range(len(p.type.shape))))
            self.emit(
                f"{name} = make_fake_compact_tensor({self.dtype(p.name)}, {p.type.shape}, stride_order={order}, assumed_align=16)"
            )
        arguments = ", ".join(["run", *names, f'options="--enable-tvm-ffi --gpu-arch={self.target}"'])
        if self.kernel.host_checks:
            self.emit(f"executable = cute.compile({arguments})")
            self.emit("from cutlass.cute.runtime import nullptr")
            self.emit("_nt_output = nullptr(cutlass.Int32, cute.AddressSpace.generic, assumed_align=4)")
            check_arguments = ", ".join(
                [
                    "_nt_check_preconditions",
                    "_nt_output",
                    *(names[index] for index, _ in self.host_parameters),
                    f'options="--enable-tvm-ffi --gpu-arch={self.target}"',
                ]
            )
            self.emit(f"checker = cute.compile({check_arguments})")
            self.emit("return _nt_CheckedExecutable(executable, checker)")
        else:
            self.emit(f"return cute.compile({arguments})")
        result = "\n".join(self.lines) + "\n"
        ast.parse(result)
        return result


def generate(kernel: Kernel, target: str) -> str:
    return Emitter(kernel, target).generate()
