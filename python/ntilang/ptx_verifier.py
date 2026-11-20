"""Static PTX/MLIR contract verifier for CPU-only CI and hardware-in-the-loop regression checks."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_SHARED_MEM_RE = re.compile(
    r"\b(?:ld\.shared|st\.shared|ldmatrix\.|cp\.async\.|atom\.shared|red\.shared|prefetch\.shared)\b"
)
_BARRIER_RE = re.compile(r"\bbar\.sync\b")
_LOCAL_DECL_RE = re.compile(r"^\s*\.local\b", re.MULTILINE)
_LOCAL_SPILL_RE = re.compile(r"\b(?:ld|st)\.local\b")
_MMA_RE = re.compile(r"\bmma\.sync\b")
_LDMATRIX_RE = re.compile(r"\bldmatrix\.")
_CP_ASYNC_RE = re.compile(r"\bcp\.async\.")
_CONTROL_FLOW_RE = re.compile(r"(?:^\s*\$L|\b(?:bra|ret|exit|call)\b|^\s*[\w$]+:)")


@dataclass(frozen=True)
class PtxReport:
    """Summary of key hardware-level instructions extracted from compiled PTX."""

    ptx: str
    barriers: int
    consecutive_barriers: int
    has_local_decl: bool
    local_spill_ops: int
    mma_ops: int
    ldmatrix_ops: int
    cp_async_ops: int

    def contains(self, pattern: str) -> bool:
        return re.search(pattern, self.ptx) is not None


def _compile_with_ptx(compiled_kernel):
    if importlib.util.find_spec("cutlass") is None:
        raise RuntimeError("CuTe DSL is required to compile and extract PTX")
    with tempfile.TemporaryDirectory(prefix="ntilang_ptx_") as tmpdir:
        dump_dir = Path(tmpdir)
        patched = compiled_kernel.source.replace(
            'options="--enable-tvm-ffi ',
            f'options="--enable-tvm-ffi --keep-ptx --dump-dir={dump_dir.as_posix()} ',
            1,
        )
        module_path = dump_dir / "kernel_ptx.py"
        module_path.write_text(patched, encoding="utf-8", newline="\n")
        module_name = f"_ntilang_ptx_{compiled_kernel.cache_key[:16]}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            executable = module.compile_kernel()
        finally:
            sys.modules.pop(module_name, None)
        ptx = _read_ptx_attr(executable)
        if ptx is not None:
            return ptx
        ptx_files = sorted(dump_dir.glob("*.ptx"))
        if ptx_files:
            return "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in ptx_files)
    raise RuntimeError("CuTe compilation completed but emitted no PTX output")


def _read_ptx_attr(executable) -> str | None:
    ptx = getattr(executable, "__ptx__", None)
    if isinstance(ptx, bytes):
        ptx = ptx.decode("utf-8", errors="replace")
    if isinstance(ptx, str) and ptx.strip():
        return ptx
    artifacts = getattr(executable, "artifacts", None)
    if artifacts is not None:
        ptx_attr = getattr(artifacts, "PTX", None) or getattr(artifacts, "ptx", None)
        if isinstance(ptx_attr, bytes):
            ptx_attr = ptx_attr.decode("utf-8", errors="replace")
        if isinstance(ptx_attr, str) and ptx_attr.strip():
            return ptx_attr
    return None


def extract_ptx(target) -> str:
    """Extract PTX text from either a CompiledKernel or a CuTe compiled executable."""
    ptx = _read_ptx_attr(target)
    if ptx is not None:
        return ptx
    if hasattr(target, "source") and hasattr(target, "cache_key"):
        return _compile_with_ptx(target)
    raise RuntimeError("Target did not expose PTX text via __ptx__/artifacts and is not a CompiledKernel.")


def analyze_ptx(ptx: str) -> PtxReport:
    """Parse PTX text for barrier redundancy, register spilling, and key hardware instructions."""
    barriers = 0
    consecutive_barriers = 0
    saw_barrier_without_smem = False

    for raw_line in ptx.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        if _CONTROL_FLOW_RE.search(line):
            saw_barrier_without_smem = False
        if _SHARED_MEM_RE.search(line):
            saw_barrier_without_smem = False
        if _BARRIER_RE.search(line):
            barriers += 1
            if saw_barrier_without_smem:
                consecutive_barriers += 1
            saw_barrier_without_smem = True

    return PtxReport(
        ptx=ptx,
        barriers=barriers,
        consecutive_barriers=consecutive_barriers,
        has_local_decl=bool(_LOCAL_DECL_RE.search(ptx)),
        local_spill_ops=len(_LOCAL_SPILL_RE.findall(ptx)),
        mma_ops=len(_MMA_RE.findall(ptx)),
        ldmatrix_ops=len(_LDMATRIX_RE.findall(ptx)),
        cp_async_ops=len(_CP_ASYNC_RE.findall(ptx)),
    )


def inspect_kernel_ptx(target) -> PtxReport:
    """Extract and statically verify PTX emitted for a CompiledKernel or CuTe executable."""
    return analyze_ptx(extract_ptx(target))
