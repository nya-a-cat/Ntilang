"""Inclusive scans: source binding, shared-memory lowering, and numeric reference.

The 32-element segment tree and carry order follow the pinned TileLang CUDA
scan.h. All working values retain the source dtype; destination conversion is a
separate store. Shared staging makes overlapping regions and MMA fragments safe.
"""

from __future__ import annotations

from .ir import Partition, Region, Statement, integer_limits
from .macros import ValueNode


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
        result = (
            f"({left} + {right})" if kind == "sum" else f"cute.math.max({left}, {right}, propagate_nan=False)"
        )
        return f"{typ}({result})"

    # Padding has the reducer identity. The barriers also protect workspace
    # reuse across multiple scans, branches, and loop iterations.
    e.emit("cute.arch.sync_threads()")
    _, coords = e.loop_tile(shape)
    e.emit(f"{access(first, coords)} = {initial}")
    e.depth -= 2
    e.emit("cute.arch.sync_threads()")
    if e.buffers[src.buffer].space == "fragment":
        slot, physical, depth = e.loop_fragment(src.buffer)
        coords = e.region_coordinates(src, physical)
        if not src.is_full(e.buffers[src.buffer].type.shape):
            e.emit(f"if {e.region_predicate(src, physical)}:")
            e.depth += 1
            depth += 1
        value = f"{e.buf(src.buffer)}[{slot}]"
    else:
        _, coords = e.loop_tile(src.shape)
        depth = 2
        indices = e.region_indices(src, coords)
        value = e.unique("scan_input")
        e.emit(f"{value} = {typ}(0)")
        e.emit(f"if {e.predicate(src.buffer, indices)}:")
        e.depth += 1
        e.emit(f"{value} = {e.access(src.buffer, indices)}")
        e.depth -= 1
    e.emit(f"{access(first, coords)} = {value}")
    e.depth -= depth
    e.emit("cute.arch.sync_threads()")

    # Each stage reads only the preceding bank, then reconverges before the
    # banks swap. This holds even when a thread owns several register slots.
    for offset in (1, 2, 4, 8, 16):
        _, coords = e.loop_tile(shape)
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
        e.depth -= 2
        e.emit("cute.arch.sync_threads()")
        first, second = second, first

    # Propagate segment carries in the same order as InclusiveScanLine. The
    # first segment still combines with identity (observable for NaNs and -0).
    segments = shape[dim] // 32
    segment = e.unique("scan_segment")
    e.emit(f"for {segment} in cutlass.range({segments}):")
    e.depth += 1
    chunk_shape = shape[:dim] + (32,) + shape[dim + 1 :]
    _, coords = e.loop_tile(chunk_shape)
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
    e.depth -= 2
    e.emit("cute.arch.sync_threads()")
    e.depth -= 1

    # All input has been captured before any destination write. A fragment
    # destination uses its own physical ownership, including MMA layouts.
    if e.buffers[dst.buffer].space == "fragment":
        slot, physical, depth = e.loop_fragment(dst.buffer)
        coords = e.region_coordinates(dst, physical)
        if not dst.is_full(e.buffers[dst.buffer].type.shape):
            e.emit(f"if {e.region_predicate(dst, physical)}:")
            e.depth += 1
            depth += 1
        target = f"{e.buf(dst.buffer)}[{slot}]"
    else:
        _, coords = e.loop_tile(dst.shape)
        indices = e.region_indices(dst, coords)
        e.emit(f"if {e.predicate(dst.buffer, indices)}:")
        e.depth += 1
        depth = 3
        target = e.access(dst.buffer, indices)
    e.emit(f"{target} = {e.dtype(dst.buffer)}({access(first, coords)})")
    e.depth -= depth
    e.emit("cute.arch.sync_threads()")


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
