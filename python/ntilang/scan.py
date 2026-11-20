"""Inclusive scans: source binding, shared-memory lowering, and numeric reference.

The 32-element segment tree and carry order follow the pinned TileLang CUDA
scan.h. All working values retain the source dtype; destination conversion is a
separate store. Shared staging makes overlapping regions and MMA fragments safe.
"""

from __future__ import annotations

from math import prod

from .ir import Partition, Region, Statement, integer_limits
from .macros import ValueNode
from .passes.smem_hazards import expr_smem_reads


def workspace_shape(shape, dim):
    return shape[:dim] + ((shape[dim] + 31) // 32 * 32,) + shape[dim + 1 :]


def workspace_key(args, buffers):
    src, _, _, dim, _ = args
    return workspace_shape(src.shape, dim), buffers[src.buffer].type.dtype


def identity(kind, dtype):
    if kind == "sum":
        return 0
    return integer_limits(dtype)[0] if dtype.startswith(("int", "uint")) or dtype == "bool" else -float("inf")


def parse(parser, call, name, location):
    if parser.before_launch:
        parser.fail(call, "Scan operations must appear inside T.Kernel")
    explicit_fragment = name.endswith("_fragment")
    defaults = {"annotations": None}
    if not explicit_fragment:
        defaults.update(dst=None, dim=0, reverse=False)
    arguments = parser.bind_call(call, ["src", "dst", "dim", "reverse", "annotations"], defaults)
    # Bind each expression exactly once, in call-site argument evaluation order.
    values = {key: parser.macro_value(value) for key, value in arguments.items()}

    def region(key):
        spec = parser.region_spec(ValueNode(values[key], arguments[key]))
        buffer, origin, extents = spec
        # A BufferLoad denotes one element in every dimension. Unit dimensions
        # stay present: scan axes refer to source rank, unlike copy broadcasting.
        return Region(buffer, origin, (1,) * len(origin) if extents is None else extents)

    src = region("src")
    if values["dst"] is None:
        if explicit_fragment:
            parser.fail(call, "Fragment scan helpers require a destination buffer")
        dst = src
    else:
        dst = region("dst")
    if src.shape != dst.shape:
        parser.fail(call, "Scan destination shape must match the source shape, including unit dimensions")
    dim, reverse = (parser.static(ValueNode(values[key], arguments[key])) for key in ("dim", "reverse"))
    if type(dim) is not int or not -len(src.shape) <= dim < len(src.shape):
        parser.fail(call, "Scan dimension is outside the source rank")
    dim %= len(src.shape)
    if type(reverse) is not bool:
        parser.fail(call, "Scan reverse must be a construction-time bool")
    annotations = parser.static(ValueNode(values["annotations"], arguments["annotations"]))
    if annotations is not None and type(annotations) is not dict:
        parser.fail(call, "Scan annotations must be a static dictionary or None")
    if annotations:
        parser.fail(call, "Scan lowering annotations require further implementation")
    source, destination = (parser.buffers[r.buffer] for r in (src, dst))
    if not explicit_fragment and source.space != "fragment":
        if source.space != "shared" or destination.space != "shared":
            parser.fail(call, "Direct scans require shared-to-shared buffers or a fragment source")
        if source.type.dtype != destination.type.dtype:
            parser.fail(call, "Direct shared scans require matching source and destination dtypes")
    Partition(workspace_shape(src.shape, dim), parser.threads)
    if source.space != "global" and src.buffer not in parser.initialized:
        parser.fail(call, f"Buffer {src.buffer} is read before initialization")
    if dst.is_full(destination.type.shape):
        parser.initialized.add(dst.buffer)
    elif destination.space != "global" and dst.buffer not in parser.initialized:
        parser.fail(call, "Partial temporary scans require an initialized destination")
    kind = "sum" if name.startswith("cumsum") else "max"
    return Statement("scan", (src, dst, kind, dim, reverse), location)


def emit(emitter, src, dst, kind, dim, reverse):
    e = emitter
    shape, dtype = workspace_key((src, dst, kind, dim, reverse), e.buffers)
    first, second = e.scan_workspaces[shape, dtype]
    typ = e.dtype(src.buffer)
    initial = (
        f"{typ}({identity(kind, dtype)!r})"
        if kind == "sum" or dtype.startswith(("int", "uint")) or dtype == "bool"
        else f"{typ}(float('-inf'))"
    )

    def access(workspace, coords):
        return f"{workspace}[{', '.join(coords)}]"

    def combine(left, right):
        if dtype == "bool":
            return f"{typ}({left} | {right})"
        if kind == "max" and dtype == "float64":
            # The pinned fast_max has a float specialization only. Its double
            # template uses ordered comparison and retains the left operand on
            # ties/unordered inputs, including NaNs and signed zero.
            result = e.unique("scan_max")
            e.emit(f"{result} = {left}")
            e.emit(f"if {left} < {right}:")
            e.depth += 1
            e.emit(f"{result} = {right}")
            e.depth -= 1
            return result
        result = f"({left} + {right})" if kind == "sum" else e.float_minmax("max", dtype, left, right, False)
        return f"{typ}({result})"

    can_warp_scan = (
        dim == len(shape) - 1
        and e.kernel.threads % 32 == 0
        and dtype in ("float32", "int32", "uint32")
        and (
            e.buffers[src.buffer].space != "fragment"
            or (
                e.fragment_layouts[src.buffer] is None
                and src.shape == shape
                and src.is_full(e.buffers[src.buffer].type.shape)
            )
        )
    )

    if can_warp_scan:
        src_reads = expr_smem_reads(src.origin, e.buffers)
        if e.buffers[src.buffer].space == "shared":
            src_reads.add(src.buffer)
        e.sync_if_hazard(src_reads, {first})
        slot, coords, depth = e.loop_tile(shape, constexpr=(e.buffers[src.buffer].space == "fragment"))
        lane = e.unique("wscan_lane")
        val = e.unique("wscan_val")
        e.emit(f"{lane} = _nt_tid % 32")
        if e.buffers[src.buffer].space == "fragment":
            e.emit(f"{val} = {typ}({e.buf(src.buffer)}[{slot}])")
        else:
            indices = e.region_indices(src, coords)
            src_pred = e.region_in_bounds_predicate(src, indices)
            if src.shape == shape:
                if src_pred is None:
                    e.emit(f"{val} = {typ}({e.access(src.buffer, indices)})")
                else:
                    e.emit(f"{val} = {typ}(0)")
                    e.emit(f"if {src_pred}:")
                    e.depth += 1
                    e.emit(f"{val} = {typ}({e.access(src.buffer, indices)})")
                    e.depth -= 1
            else:
                e.emit(f"{val} = {initial}")
                e.emit(f"if {coords[-1]} < {src.shape[-1]}:")
                e.depth += 1
                if src_pred is None:
                    e.emit(f"{val} = {typ}({e.access(src.buffer, indices)})")
                else:
                    e.emit(f"{val} = {typ}(0)")
                    e.emit(f"if {src_pred}:")
                    e.depth += 1
                    e.emit(f"{val} = {typ}({e.access(src.buffer, indices)})")
                    e.depth -= 1
                e.depth -= 1
        shfl_fn = "cute.arch.shuffle_sync_down" if reverse else "cute.arch.shuffle_sync_up"
        for offset in (1, 2, 4, 8, 16):
            other = e.unique("wscan_other")
            e.emit(f"{other} = {shfl_fn}({val}, {offset})")
            pred = f"{lane} < {32 - offset}" if reverse else f"{lane} >= {offset}"
            e.emit(f"if {pred}:")
            e.depth += 1
            e.emit(f"{val} = {combine(val, other)}")
            e.depth -= 1
        e.emit(f"{access(first, coords)} = {val}")
        e.depth -= depth
        e.hazards.record(src_reads, {first})
        e.sync_threads()

        segments = shape[dim] // 32
        if segments > 1:
            outer_shape = shape[:-1]
            outer_count = prod(outer_shape)
            row_slots = (outer_count + e.kernel.threads - 1) // e.kernel.threads
            r_slot = e.unique("wscan_rslot")
            r_flat = e.unique("wscan_row")
            loop_hdr = (
                f"cutlass.range_constexpr({row_slots})"
                if row_slots <= 16
                else f"cutlass.range({row_slots}, unroll=4)"
            )
            e.emit(f"for {r_slot} in {loop_hdr}:")
            e.depth += 1
            e.emit(f"{r_flat} = _nt_tid + {r_slot} * {e.kernel.threads}")
            r_depth = 1
            if outer_count % e.kernel.threads != 0:
                e.emit(f"if {r_flat} < {outer_count}:")
                e.depth += 1
                r_depth += 1
            row_coords = e.coordinates(r_flat, outer_shape)
            carry = e.unique("wscan_carry")
            seg = e.unique("wscan_seg")
            e.emit(f"{carry} = {initial}")
            e.emit(f"for {seg} in cutlass.range({segments}):")
            e.depth += 1
            seg_idx = f"({segments - 1} - {seg})" if reverse else seg
            e.emit(f"{access(second, [*row_coords, seg_idx])} = {carry}")
            end_col = f"({seg_idx} * 32)" if reverse else f"({seg_idx} * 32 + 31)"
            e.emit(f"{carry} = {combine(access(first, [*row_coords, end_col]), carry)}")
            e.depth -= r_depth + 1
            e.hazards.record({first}, {second})
            e.sync_threads()

        dst_reads = ({first, second} if segments > 1 else {first}) | expr_smem_reads(dst.origin, e.buffers)
        dst_writes = {dst.buffer} if e.buffers[dst.buffer].space == "shared" else set()
        e.sync_if_hazard(dst_reads, dst_writes)
        if e.buffers[dst.buffer].space == "fragment":
            slot, physical, depth = e.loop_fragment(dst.buffer)
            coords = e.region_coordinates(dst, physical)
            if not dst.is_full(e.buffers[dst.buffer].type.shape):
                e.emit(f"if {e.region_predicate(dst, physical)}:")
                e.depth += 1
                depth += 1
            target = f"{e.buf(dst.buffer)}[{slot}]"
        else:
            _, coords, depth = e.loop_tile(dst.shape)
            indices = e.region_indices(dst, coords)
            dst_pred = e.region_in_bounds_predicate(dst, indices)
            if dst_pred is not None:
                e.emit(f"if {dst_pred}:")
                e.depth += 1
                depth += 1
            target = e.access(dst.buffer, indices)
        carry_val = initial if segments == 1 else access(second, [*coords[:-1], f"({coords[-1]} // 32)"])
        e.emit(f"{target} = {e.dtype(dst.buffer)}({combine(access(first, coords), carry_val)})")
        e.depth -= depth
        e.hazards.record(dst_reads, dst_writes)
        return

    # Padding has the reducer identity when the scan axis is not a multiple of 32.
    if src.shape != shape:
        e.sync_if_hazard(set(), {first})
        _, coords, depth = e.loop_tile(shape)
        e.emit(f"{access(first, coords)} = {initial}")
        e.depth -= depth
        e.hazards.record(set(), {first})

    src_reads = expr_smem_reads(src.origin, e.buffers)
    if e.buffers[src.buffer].space == "shared":
        src_reads.add(src.buffer)
    e.sync_if_hazard(src_reads, {first})
    if e.buffers[src.buffer].space == "fragment":
        slot, physical, depth = e.loop_fragment(src.buffer)
        coords = e.region_coordinates(src, physical)
        if not src.is_full(e.buffers[src.buffer].type.shape):
            e.emit(f"if {e.region_predicate(src, physical)}:")
            e.depth += 1
            depth += 1
        value = f"{e.buf(src.buffer)}[{slot}]"
    else:
        _, coords, depth = e.loop_tile(src.shape)
        indices = e.region_indices(src, coords)
        src_pred = e.region_in_bounds_predicate(src, indices)
        if src_pred is None:
            value = e.access(src.buffer, indices)
        else:
            value = e.unique("scan_input")
            e.emit(f"{value} = {typ}(0)")
            e.emit(f"if {src_pred}:")
            e.depth += 1
            e.emit(f"{value} = {e.access(src.buffer, indices)}")
            e.depth -= 1
    e.emit(f"{access(first, coords)} = {value}")
    e.depth -= depth
    e.hazards.record(src_reads, {first})
    e.sync_threads()

    # Each stage reads only the preceding bank, then reconverges before the
    # banks swap. This holds even when a thread owns several register slots.
    for offset in (1, 2, 4, 8, 16):
        _, coords, depth = e.loop_tile(shape)
        lane = f"({coords[dim]} % 32)"
        predicate = f"{lane} < {32 - offset}" if reverse else f"{lane} >= {offset}"
        neighbor = coords.copy()
        neighbor[dim] = f"({coords[dim]} {'+' if reverse else '-'} {offset})"
        value = e.unique("scan_value")
        e.emit(f"{value} = {access(first, coords)}")
        e.emit(f"if {predicate}:")
        e.depth += 1
        e.emit(f"{value} = {combine(value, access(first, neighbor))}")
        e.depth -= 1
        e.emit(f"{access(second, coords)} = {value}")
        e.depth -= depth
        e.sync_threads()
        first, second = second, first

    # Propagate segment carries in the same order as InclusiveScanLine. The
    # first segment still combines with identity (observable for NaNs and -0).
    segments = shape[dim] // 32
    segment = e.unique("scan_segment")
    e.emit(f"for {segment} in cutlass.range({segments}):")
    e.depth += 1
    chunk_shape = shape[:dim] + (32,) + shape[dim + 1 :]
    _, coords, depth = e.loop_tile(chunk_shape)
    segment_index = f"({segments - 1} - {segment})" if reverse else segment
    coords[dim] = f"({segment_index} * 32 + {coords[dim]})"
    carry = e.unique("scan_carry")
    e.emit(f"{carry} = {initial}")
    e.emit(f"if {segment} > 0:")
    e.depth += 1
    previous = coords.copy()
    previous[dim] = f"(({segment_index} + 1) * 32)" if reverse else f"({segment_index} * 32 - 1)"
    e.emit(f"{carry} = {access(first, previous)}")
    e.depth -= 1
    e.emit(f"{access(first, coords)} = {combine(access(first, coords), carry)}")
    e.depth -= depth
    e.sync_threads()
    e.depth -= 1

    # All input has been captured before any destination write. A fragment
    # destination uses its own physical ownership, including MMA layouts.
    dst_reads = {first} | expr_smem_reads(dst.origin, e.buffers)
    dst_writes = {dst.buffer} if e.buffers[dst.buffer].space == "shared" else set()
    e.sync_if_hazard(dst_reads, dst_writes)
    if e.buffers[dst.buffer].space == "fragment":
        slot, physical, depth = e.loop_fragment(dst.buffer)
        coords = e.region_coordinates(dst, physical)
        if not dst.is_full(e.buffers[dst.buffer].type.shape):
            e.emit(f"if {e.region_predicate(dst, physical)}:")
            e.depth += 1
            depth += 1
        target = f"{e.buf(dst.buffer)}[{slot}]"
    else:
        _, coords, depth = e.loop_tile(dst.shape)
        indices = e.region_indices(dst, coords)
        dst_pred = e.region_in_bounds_predicate(dst, indices)
        if dst_pred is not None:
            e.emit(f"if {dst_pred}:")
            e.depth += 1
            depth += 1
        target = e.access(dst.buffer, indices)
    e.emit(f"{target} = {e.dtype(dst.buffer)}({access(first, coords)})")
    e.depth -= depth
    e.hazards.record(dst_reads, dst_writes)


def evaluate(values, kind, dim, reverse):
    """Reproduce the segment tree with rounding after every combine.

    This checks arithmetic and source capture. It does not simulate CUDA thread
    scheduling, barrier execution, or instruction-level NaN payload behavior.
    """
    import numpy as np

    padded_shape = workspace_shape(values.shape, dim)
    initial = identity(kind, values.dtype.name)
    work = np.full(padded_shape, initial, dtype=values.dtype)
    region = tuple(slice(0, size) for size in values.shape)
    work[region] = values
    lines = np.moveaxis(work, dim, -1)
    segments = lines.reshape(*lines.shape[:-1], lines.shape[-1] // 32, 32)

    def operation(left, right):
        if kind == "sum":
            return np.add(left, right)
        if values.dtype == np.dtype("float64"):
            return np.where(left < right, right, left)
        result = np.fmax(left, right)
        if values.dtype.kind == "f":
            # CUDA fmaxf/__hmax prefer +0 when both zeros have different signs.
            zero = np.where(np.signbit(left) & np.signbit(right), -0.0, 0.0).astype(values.dtype)
            result = np.where((left == 0) & (right == 0), zero, result)
        return result

    with np.errstate(over="ignore", invalid="ignore"):
        for offset in (1, 2, 4, 8, 16):
            old = segments.copy()
            if reverse:
                segments[..., :-offset] = operation(old[..., :-offset], old[..., offset:])
            else:
                segments[..., offset:] = operation(old[..., offset:], old[..., :-offset])
        carry = np.full(lines.shape[:-1], initial, dtype=values.dtype)
        order = range(segments.shape[-2] - 1, -1, -1) if reverse else range(segments.shape[-2])
        for index in order:
            segments[..., index, :] = operation(segments[..., index, :], carry[..., None])
            carry = segments[..., index, 0 if reverse else 31].copy()
    return work[region].copy()
