"""Shared transpose lowering through the existing copy IR."""

from __future__ import annotations

from .ir import Partition, Region, Statement
from .macros import ValueNode


def _region(parser, node):
    name, origin, shape = parser.region_spec(node)
    if shape is None:
        parser.fail(node, "This operation requires a buffer or an explicitly sized region")
    return Region(name, origin, shape)


def transpose(parser, call, node):
    args = parser.bind_call(call, ["src", "dst", "annotations"], {"annotations": None})
    values = {key: ValueNode(parser.macro_value(value), value) for key, value in args.items()}
    src, dst = (_region(parser, values[name]) for name in ("src", "dst"))
    annotations = parser.static(values["annotations"])
    if annotations is not None and type(annotations) is not dict:
        parser.fail(call, "Tile annotations must be a static dictionary")
    if annotations:
        parser.fail(call, "Nonempty transpose annotations require further lowering")
    rank = len(src.shape)
    if rank < 2 or len(dst.shape) != rank:
        parser.fail(call, "Transpose requires matching ranks of at least two")
    permutation = (*range(rank - 2), rank - 1, rank - 2)
    if dst.shape != tuple(src.shape[axis] for axis in permutation):
        parser.fail(call, "Transpose destination must swap the source's final two dimensions")
    if any(parser.buffers[r.buffer].space != "shared" for r in (src, dst)):
        parser.fail(call, "Transpose requires shared-memory source and destination regions")
    Partition(src.shape, parser.threads)
    if src.buffer not in parser.initialized:
        parser.fail(call, f"Buffer {src.buffer} is read before initialization")
    if dst.is_full(parser.buffers[dst.buffer].type.shape):
        parser.initialized.add(dst.buffer)
    elif dst.buffer not in parser.initialized:
        parser.fail(call, "Partial temporary writes require an initialized destination")
    # Copy handles conversions, source snapshots, barriers and ownership.
    destination = Region(dst.buffer, dst.origin, src.shape, permutation)
    return Statement("copy", (src, destination), parser.location(node))
