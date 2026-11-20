"""MMA layout inference, fragment ownership propagation, and shared-memory planning passes."""

from __future__ import annotations

from math import prod

from .. import scan
from ..ir import DTYPES, Buffer, CompileError, Expr, Kernel


def walk(body):
    """Yield every IR statement in pre-order across nested control flow."""
    for stmt in body:
        yield stmt
        if stmt.op in ("serial", "parallel", "unroll", "pipelined"):
            yield from walk(stmt.args[2])
        elif stmt.op == "if":
            yield from walk(stmt.args[1])
            yield from walk(stmt.args[2])
        elif stmt.op == "while":
            yield from walk(stmt.args[1])


def analyze_mmas(kernel: Kernel, buffers: dict[str, Buffer]) -> dict[str, tuple]:
    """Validate T.gemm statements and select sm_80+ warp MMA tilings per accumulator."""
    mmas: dict[str, tuple] = {}
    for stmt in walk(kernel.body):
        if stmt.op != "gemm":
            continue
        a, b, c, ta, tb = stmt.args
        ab, bb, cb = (buffers[n] for n in (a, b, c))
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
            wm for wm in (1, 2, 4, 8) if wm <= warps and am % (16 * wm) == 0 and bn % (8 * (warps // wm)) == 0
        ]
        if not candidates or ak % 16:
            raise CompileError(
                "GEMM M/N must cover the warp arrangement and K must be a multiple of 16",
                stmt.location,
            )
        wm = max(candidates)
        plan = (am, bn, ak, wm, warps // wm, ab.type.dtype)
        if c in mmas and mmas[c] != plan:
            raise CompileError("All GEMMs sharing an accumulator require the same MMA layout", stmt.location)
        mmas[c] = plan
    return mmas


def parallel_accesses(stmt, buffers: dict[str, Buffer]) -> tuple[set[str], set[str], bool]:
    """Classify fragment/shared reads and writes inside a T.Parallel block."""
    names, shape, body = stmt.args
    expected = tuple(Expr("var", value=n) for n in names)
    pointwise, snapshots, written = set(), set(), set()
    shared_used, shared_written, shared_cross_reads = set(), set(), set()

    def loads(value):
        if isinstance(value, Expr):
            if value.op == "load" and buffers[value.value].space == "fragment":
                if buffers[value.value].type.shape == shape and value.args == expected:
                    pointwise.add(value.value)
                else:
                    snapshots.add(value.value)
            elif value.op == "load" and buffers[value.value].space == "shared":
                shared_used.add(value.value)
                if buffers[value.value].type.shape != shape or value.args != expected:
                    shared_cross_reads.add(value.value)
            for arg in value.args:
                loads(arg)
        elif isinstance(value, (tuple, list)):
            for arg in value:
                loads(arg)

    for inner in walk(body):
        loads(inner.args)
        if inner.op == "print" and isinstance(inner.args[0], Buffer) and inner.args[0].space == "shared":
            shared_used.add(inner.args[0].name)
            shared_cross_reads.add(inner.args[0].name)
        if inner.op == "store" and buffers[inner.args[0]].space == "fragment":
            written.add(inner.args[0])
        elif inner.op == "store" and buffers[inner.args[0]].space == "shared":
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


def copy_snapshots(src, dst, buffers: dict[str, Buffer]) -> set[str]:
    """Determine whether a T.copy requires materializing src into a synchronized snapshot."""
    if src.buffer == dst.buffer and src != dst:
        return {src.buffer}
    if buffers[src.buffer].space == "fragment" and (
        not src.is_full(buffers[src.buffer].type.shape)
        or buffers[dst.buffer].space == "fragment"
        and not dst.is_full(buffers[dst.buffer].type.shape)
    ):
        return {src.buffer}
    return set()


def infer_fragment_layouts(
    kernel: Kernel,
    buffers: dict[str, Buffer],
    mmas: dict[str, tuple],
) -> dict[str, str | None]:
    """Propagate MMA accumulator ownership across connected pointwise loops and full-tile copies."""
    parents = {b.name: b.name for b in kernel.buffers if b.space == "fragment"}

    def root(name):
        while parents[name] != name:
            name = parents[name]
        return name

    for stmt in walk(kernel.body):
        names = set()
        if stmt.op == "parallel":
            names, _, _ = parallel_accesses(stmt, buffers)
        elif stmt.op == "copy":
            if all(r.is_full(buffers[r.buffer].type.shape) for r in stmt.args):
                names = {r.buffer for r in stmt.args if buffers[r.buffer].space == "fragment"}
        ordered = sorted(names)
        for name in ordered[1:]:
            parents[root(name)] = root(ordered[0])
    seeds: dict[str, str] = {}
    for name, plan in mmas.items():
        group = root(name)
        if group in seeds:
            previous = mmas[seeds[group]]
            if (plan[:2], plan[3:5]) != (previous[:2], previous[3:5]):
                raise CompileError("Connected MMA fragments require a register layout conversion")
        else:
            seeds[group] = name
    return {name: seeds.get(root(name)) for name in parents}


def collect_workspaces(
    kernel: Kernel,
    buffers: dict[str, Buffer],
) -> tuple[set[str], dict[tuple[str, str], str], dict[tuple[tuple[int, ...], str], tuple[str, str]]]:
    """Collect required shared-memory snapshot, reduction, and scan workspaces."""
    snapshots: set[str] = set()
    reduction_workspaces: dict[tuple[str, str], str] = {}
    scan_workspaces: dict[tuple[tuple[int, ...], str], tuple[str, str]] = {}
    for stmt in walk(kernel.body):
        if stmt.op == "print" and isinstance(stmt.args[0], Buffer):
            if stmt.args[0].space == "fragment":
                snapshots.add(stmt.args[0].name)
        elif stmt.op == "parallel":
            _, stmt_snapshots, _ = parallel_accesses(stmt, buffers)
            snapshots.update(stmt_snapshots)
        elif stmt.op == "copy":
            snapshots.update(copy_snapshots(*stmt.args, buffers))
        elif stmt.op == "reduce":
            src, dst = stmt.args[:2]
            key = (src, buffers[dst].type.dtype)
            if key not in reduction_workspaces:
                reduction_workspaces[key] = f"_nt_reduce_workspace_{len(reduction_workspaces)}"
        elif stmt.op == "scan":
            key = scan.workspace_key(stmt.args, buffers)
            if key not in scan_workspaces:
                label = f"_nt_scan_workspace_{len(scan_workspaces)}"
                scan_workspaces[key] = (label + "_a", label + "_b")
    return snapshots, reduction_workspaces, scan_workspaces


def validate_shared_memory_budget(
    kernel: Kernel,
    buffers: dict[str, Buffer],
    snapshots: set[str],
    reduction_workspaces: dict[tuple[str, str], str],
    scan_workspaces: dict[tuple[tuple[int, ...], str], tuple[str, str]],
    limit: int = 48 * 1024,
) -> int:
    """Compute 16-byte aligned static shared-memory usage and enforce the block limit."""
    shared_bytes = 0
    for buffer in kernel.buffers:
        for needed in (buffer.name in snapshots, buffer.space == "shared"):
            if not needed:
                continue
            shared_bytes = (shared_bytes + 15) // 16 * 16
            shared_bytes += prod(buffer.type.shape) * DTYPES[buffer.type.dtype]
    for src, dtype in reduction_workspaces:
        shared_bytes = (shared_bytes + 15) // 16 * 16
        shared_bytes += prod(buffers[src].type.shape) * DTYPES[dtype]
    for (shape, dtype), workspaces in scan_workspaces.items():
        for _ in workspaces:
            shared_bytes = (shared_bytes + 15) // 16 * 16
            shared_bytes += prod(shape) * DTYPES[dtype]
    if shared_bytes > limit:
        raise CompileError("Shared buffers and fragment communication exceed 48 KiB per block")
    return shared_bytes
