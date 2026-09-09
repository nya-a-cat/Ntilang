"""Collective syntax lowering into the existing copy and parallel-loop IR."""

from __future__ import annotations

from . import macros
from .ir import Buffer, Expr, Partition, Region, Statement, TensorType


def _const(value):
    return Expr("const", value=value)


def _region(parser, node):
    name, origin, shape = parser.region_spec(node)
    if shape is None:
        parser.fail(node, "This operation requires a buffer or an explicitly sized region")
    return Region(name, origin, shape)


def _annotations(parser, node):
    value = parser.static(node)
    if value is not None and type(value) is not dict:
        parser.fail(node, "Tile annotations must be a static dictionary")
    if value:
        parser.fail(node, "Nonempty scan/transpose annotations require further lowering")


def _check_access(parser, call, src, dst):
    if src.buffer not in parser.initialized and parser.buffers[src.buffer].space != "global":
        parser.fail(call, f"Buffer {src.buffer} is read before initialization")
    if dst.is_full(parser.buffers[dst.buffer].type.shape):
        parser.initialized.add(dst.buffer)
    elif dst.buffer not in parser.initialized and parser.buffers[dst.buffer].space != "global":
        parser.fail(call, "Partial temporary writes require an initialized destination")


def transpose(parser, call, node):
    args = parser.bind_call(call, ["src", "dst", "annotations"], {"annotations": None})
    src, dst = (_region(parser, args[name]) for name in ("src", "dst"))
    _annotations(parser, args["annotations"])
    rank = len(src.shape)
    if rank < 2 or len(dst.shape) != rank:
        parser.fail(call, "Transpose requires matching ranks of at least two")
    permutation = (*range(rank - 2), rank - 1, rank - 2)
    if dst.shape != tuple(src.shape[axis] for axis in permutation):
        parser.fail(call, "Transpose destination must swap the source's final two dimensions")
    if any(parser.buffers[r.buffer].space != "shared" for r in (src, dst)):
        parser.fail(call, "Transpose requires shared-memory source and destination regions")
    Partition(src.shape, parser.threads)
    _check_access(parser, call, src, dst)
    # Copy already implements guarded loads, dtype conversion, overlapping-source
    # snapshots, synchronization, and write ownership for arbitrary region axes.
    destination = Region(dst.buffer, dst.origin, src.shape, permutation)
    return Statement("copy", (src, destination), parser.location(node))


def _workspace(parser, shape, dtype, dim, loc):
    key = (shape, dtype, dim)
    if key not in parser.scan_workspaces:
        buffers = []
        for label, sizes in (
            ("scan_a", shape),
            ("scan_b", shape),
            ("scan_carry", tuple(1 if i == dim else size for i, size in enumerate(shape))),
        ):
            name = parser.fresh(label)
            buf = Buffer(name, TensorType(sizes, dtype), "shared", source_scope="shared.dyn")
            parser.buffers[name] = buf
            parser.allocated.append(buf)
            parser.prologue.append(Statement("alloc", (name,), loc))
            buffers.append(name)
        parser.scan_workspaces[key] = tuple(buffers)
    return parser.scan_workspaces[key]


def scan(parser, call, node, kind):
    """Preserve the pinned CUDA scan's 32-lane tree and sequential segment carry.

    Shared ping-pong tiles implement lane exchange without depending on the
    fragment's physical layout. Barriers are supplied by ordinary IR lowering.
    Each tree edge and carry combination is rounded to the source dtype.
    """
    args = parser.bind_call(
        call,
        ["src", "dst", "dim", "reverse", "annotations"],
        {"dst": None, "dim": 0, "reverse": False, "annotations": None},
    )
    src = _region(parser, args["src"])
    destination_value = parser.macro_value(args["dst"])
    dst = (
        src
        if destination_value is None
        else _region(parser, macros.ValueNode(destination_value, args["dst"]))
    )
    dim, reverse = (parser.static(args[name]) for name in ("dim", "reverse"))
    _annotations(parser, args["annotations"])
    rank = len(src.shape)
    if rank not in (1, 2):
        parser.fail(call, "Scans currently require one- or two-dimensional regions")
    if type(dim) is not int or not -rank <= dim < rank:
        parser.fail(call, "Scan dim must be a static integer within the source rank")
    dim %= rank
    if type(reverse) is not bool:
        parser.fail(call, "Scan reverse must be a static bool")
    if src.shape != dst.shape:
        parser.fail(call, "Scan source and destination shapes must match")
    source, destination = (parser.buffers[r.buffer] for r in (src, dst))
    if source.space not in ("shared", "fragment"):
        parser.fail(call, "Scan sources must be shared or fragment regions")
    if source.space == "shared" and (
        destination.space != "shared" or source.type.dtype != destination.type.dtype
    ):
        parser.fail(call, "A shared scan requires a shared destination with the same dtype")
    dtype = source.type.dtype
    if dtype == "bool":
        parser.fail(call, "Boolean scans require further lowering")
    if parser.threads not in (32, 64, 128, 256, 512, 1024):
        parser.fail(call, "Scans require 32, 64, 128, 256, 512, or 1024 threads")
    _check_access(parser, call, src, dst)
    loc = parser.location(node)
    segments = (src.shape[dim] + 31) // 32
    padded = tuple(segments * 32 if i == dim else size for i, size in enumerate(src.shape))
    Partition(padded, parser.threads)
    left, right, carry = _workspace(parser, padded, dtype, dim, loc)
    zeros = tuple(_const(0) for _ in padded)
    if kind == "cumsum":
        identity = 0
    elif dtype.startswith("uint"):
        identity = 0
    elif dtype.startswith("int"):
        identity = -(1 << (int(dtype[3:]) - 1))
    else:
        identity = float("-inf")
    identity = Expr("cast", (_const(identity),), dtype)

    def combine(x, y):
        return Expr("cast", (Expr("+" if kind == "cumsum" else "max", (x, y)),), dtype)

    result = [
        Statement("fill", (left, identity), loc),
        Statement("fill", (carry, identity), loc),
        Statement("copy", (src, Region(left, zeros, src.shape)), loc),
    ]
    # Every segment uses precisely five shuffle distances, including padded lanes.
    # Keep the operand order val OP neighbor used by the CUDA template.
    for distance in (1, 2, 4, 8, 16):
        names = tuple(parser.fresh("scan_i") for _ in padded)
        coords = tuple(Expr("var", value=name) for name in names)
        neighbor = list(coords)
        neighbor[dim] = Expr("+" if reverse else "-", (coords[dim], _const(distance)))
        lane = Expr("%", (coords[dim], _const(32)))
        predicate = Expr("<" if reverse else ">=", (lane, _const(32 - distance if reverse else distance)))
        value = Expr("load", coords, left)
        shifted = Expr("load", tuple(neighbor), left)
        selected = Expr("if_then_else", (predicate, combine(value, shifted), value))
        body = (Statement("store", (right, coords, selected), loc),)
        result.append(Statement("parallel", (names, padded, body), loc))
        left, right = right, left

    # Segment carries are serialized along the scan axis; different lines and
    # all 32 lanes of each segment are still assigned by a parallel domain.
    segment_name = parser.fresh("scan_segment")
    segment = Expr("var", value=segment_name)
    segment_offset = Expr("*", (segment, _const(32)))
    line_shape = tuple(32 if i == dim else size for i, size in enumerate(padded))
    names = tuple(parser.fresh("scan_lane") for _ in padded)
    coords = tuple(Expr("var", value=name) for name in names)
    indices = tuple(
        Expr("+", (segment_offset, value)) if i == dim else value for i, value in enumerate(coords)
    )
    carry_indices = tuple(_const(0) if i == dim else value for i, value in enumerate(coords))
    value = combine(Expr("load", indices, left), Expr("load", carry_indices, carry))
    body = (Statement("store", (right, indices, value), loc),)
    carry_shape = parser.buffers[carry].type.shape
    edge_origin = tuple(
        Expr("+", (segment_offset, _const(0 if reverse else 31))) if i == dim else _const(0)
        for i in range(rank)
    )
    update = Statement(
        "copy", (Region(right, edge_origin, carry_shape), Region(carry, zeros, carry_shape)), loc
    )
    bounds = (segments - 1, -1, -1) if reverse else (0, segments, 1)
    result.append(
        Statement(
            "serial",
            ((segment_name,), bounds, (Statement("parallel", (names, line_shape, body), loc), update)),
            loc,
        )
    )
    result.append(Statement("copy", (Region(right, zeros, src.shape), dst), loc))
    return tuple(result)
