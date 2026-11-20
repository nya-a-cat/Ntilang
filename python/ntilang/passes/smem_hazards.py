"""Shared-memory RAW/WAR/WAW hazard tracking for provably safe barrier elision."""

from __future__ import annotations

from .. import scan
from ..ir import Buffer, Expr
from .layout_inference import copy_snapshots, parallel_accesses, walk


class HazardTracker:
    """Tracks shared-memory objects read or written since the last uniform CTA barrier."""

    def __init__(self, reads: set[str] | None = None, writes: set[str] | None = None):
        self.reads: set[str] = set() if reads is None else set(reads)
        self.writes: set[str] = set() if writes is None else set(writes)

    def has_hazard(self, reads: set[str] | frozenset[str], writes: set[str] | frozenset[str]) -> bool:
        return bool(
            (reads & self.writes)  # RAW
            or (writes & self.reads)  # WAR
            or (writes & self.writes)  # WAW
        )

    def clear(self) -> None:
        self.reads.clear()
        self.writes.clear()

    def record(self, reads: set[str] | frozenset[str], writes: set[str] | frozenset[str]) -> None:
        self.reads.update(reads)
        self.writes.update(writes)

    def clone(self) -> HazardTracker:
        return HazardTracker(self.reads, self.writes)

    def merge_from(self, other: HazardTracker) -> None:
        self.reads.update(other.reads)
        self.writes.update(other.writes)


def expr_smem_reads(value, buffers: dict[str, Buffer]) -> set[str]:
    """Return shared-memory buffers read anywhere inside a scalar expression tree."""
    reads: set[str] = set()

    def visit(node):
        if isinstance(node, Expr):
            if node.op == "load" and buffers[node.value].space == "shared":
                reads.add(node.value)
            for arg in node.args:
                visit(arg)
        elif isinstance(node, (tuple, list)):
            for arg in node:
                visit(arg)

    visit(value)
    return reads


def smem_footprint(
    body,
    buffers: dict[str, Buffer],
    reduction_workspaces: dict[tuple[str, str], str],
    scan_workspaces: dict[tuple[tuple[int, ...], str], tuple[str, str]],
) -> tuple[set[str], set[str]]:
    """Return conservative (reads, writes) of all shared-memory objects touched in body."""
    reads: set[str] = set()
    writes: set[str] = set()

    def expr_reads(value):
        reads.update(expr_smem_reads(value, buffers))

    for stmt in walk(body):
        op, args = stmt.op, stmt.args
        if op == "store":
            name, indices, value = args
            expr_reads(indices)
            expr_reads(value)
            if buffers[name].space == "shared":
                writes.add(name)
        elif op == "fill":
            name, value = args
            expr_reads(value)
            if buffers[name].space == "shared":
                writes.add(name)
        elif op == "copy":
            src, dst = args
            expr_reads(src.origin)
            expr_reads(dst.origin)
            snaps = copy_snapshots(src, dst, buffers)
            for snap_name in snaps:
                snap = f"_nt_snapshot_{snap_name}"
                reads.add(snap)
                writes.add(snap)
                if buffers[snap_name].space == "shared":
                    reads.add(snap_name)
            if buffers[src.buffer].space == "shared" and src.buffer not in snaps:
                reads.add(src.buffer)
            if buffers[dst.buffer].space == "shared":
                writes.add(dst.buffer)
        elif op == "gemm":
            a, b, _, _, _ = args
            reads.add(a)
            reads.add(b)
        elif op == "reduce":
            src, dst = args[:2]
            ws = reduction_workspaces[src, buffers[dst].type.dtype]
            reads.add(ws)
            writes.add(ws)
            if buffers[src].space == "shared":
                reads.add(src)
            if buffers[dst].space == "shared":
                writes.add(dst)
        elif op == "scan":
            src, dst = args[:2]
            key = scan.workspace_key(args, buffers)
            ws_a, ws_b = scan_workspaces[key]
            reads.update((ws_a, ws_b))
            writes.update((ws_a, ws_b))
            if buffers[src.buffer].space == "shared":
                reads.add(src.buffer)
            if buffers[dst.buffer].space == "shared":
                writes.add(dst.buffer)
        elif op == "parallel":
            _, stmt_snaps, _ = parallel_accesses(stmt, buffers)
            for snap_name in stmt_snaps:
                snap = f"_nt_snapshot_{snap_name}"
                reads.add(snap)
                writes.add(snap)
        elif op == "print":
            target = args[0]
            if isinstance(target, Buffer):
                if target.space == "shared":
                    reads.add(target.name)
                elif target.space == "fragment":
                    snap = f"_nt_snapshot_{target.name}"
                    reads.add(snap)
                    writes.add(snap)
            else:
                expr_reads(args)
        else:
            expr_reads(args)

    return reads, writes
