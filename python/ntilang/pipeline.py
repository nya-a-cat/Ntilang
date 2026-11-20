"""Multi-frontend, multi-stage Compiler API facade (import_program -> lower -> compile -> load)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .codegen import generate
from .compat import normalize_out_idx, resolve_target
from .compiler import CompiledKernel, DynamicJitKernel
from .frontend import parse
from .ir import CompileError, Kernel
from .language import PrimFunc, SymbolicTensorSpec
from .validation import validate


@dataclass(frozen=True)
class Artifact:
    format: str
    content: str

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.content, encoding="utf-8", newline="\n")
        return target


@dataclass(frozen=True)
class Program:
    ir: Kernel
    frontend: str
    name: str

    def as_text(self) -> str:
        params = ", ".join(p.name for p in self.ir.parameters)
        return f"Program(frontend={self.frontend!r}, name={self.name!r}, params=({params}), threads={self.ir.threads})"


@dataclass(frozen=True)
class LoweredProgram:
    program: Program
    backend: str
    target: str
    source: str
    out_idx: tuple[int, ...] = ()

    def emit(self, *, format: str = "python") -> Artifact:
        if format != "python":
            raise CompileError(f"Unsupported emit format {format!r} for backend {self.backend!r}")
        return Artifact(format=format, content=self.source)


@dataclass(frozen=True)
class ArtifactBundle:
    lowered: LoweredProgram
    kernel: CompiledKernel


class RuntimeModule:
    """Loaded executable container exposing entry points by kernel name."""

    def __init__(self, bundle: ArtifactBundle, device: str = "cuda:0"):
        self.bundle = bundle
        self.device = device
        self._entries = {bundle.lowered.program.name: bundle.kernel}

    def __getitem__(self, name: str) -> CompiledKernel:
        return self._entries[name]

    def __contains__(self, name: str) -> bool:
        return name in self._entries


class Compiler:
    """Extensible staged compiler supporting import_program -> lower -> compile -> load -> jit."""

    def __init__(self):
        self._frontends: dict[str, Callable[[PrimFunc], Kernel]] = {"tilelang": self._tilelang_frontend}
        self._backends: dict[str, Callable[[Kernel, str], str]] = {"cute": generate}

    @staticmethod
    def _tilelang_frontend(source: PrimFunc) -> Kernel:
        ir = parse(source)
        validate(ir)
        return ir

    def register_frontend(self, name: str, parser_fn: Callable) -> None:
        self._frontends[name] = parser_fn

    def register_backend(self, name: str, emitter_fn: Callable[[Kernel, str], str]) -> None:
        self._backends[name] = emitter_fn

    def import_program(self, source: PrimFunc, *, frontend: str = "tilelang") -> Program:
        if frontend not in self._frontends:
            raise CompileError(
                f"Unknown frontend {frontend!r}; registered: {', '.join(sorted(self._frontends))}"
            )
        ir = self._frontends[frontend](source)
        return Program(ir=ir, frontend=frontend, name=ir.name)

    def lower(
        self,
        program: Program,
        *,
        backend: str = "cute",
        target: str = "auto",
        out_idx=None,
    ) -> LoweredProgram:
        if backend not in self._backends:
            raise CompileError(
                f"Unknown backend {backend!r}; registered: {', '.join(sorted(self._backends))}"
            )
        resolved = resolve_target(target)
        source = self._backends[backend](program.ir, resolved)
        sym_names = {f"_nt_sym_{s.name}" for s in program.ir.sym_dims}
        user_params = tuple(p for p in program.ir.parameters if p.name not in sym_names)
        normalized_out = normalize_out_idx(out_idx, user_params)
        return LoweredProgram(
            program=program,
            backend=backend,
            target=resolved,
            source=source,
            out_idx=normalized_out,
        )

    def compile(self, lowered: LoweredProgram) -> ArtifactBundle:
        kernel = CompiledKernel(
            lowered.program.ir,
            lowered.target,
            lowered.source,
            out_idx=lowered.out_idx,
        )
        return ArtifactBundle(lowered=lowered, kernel=kernel)

    def load(self, bundle: ArtifactBundle, *, device: str = "cuda:0") -> RuntimeModule:
        bundle.kernel.build()
        return RuntimeModule(bundle, device=device)

    def jit(
        self,
        source: PrimFunc,
        *,
        frontend: str = "tilelang",
        backend: str = "cute",
        target: str = "auto",
        out_idx=None,
    ) -> CompiledKernel | DynamicJitKernel:
        if isinstance(source, PrimFunc):
            has_symbolic = any(
                isinstance(ann, SymbolicTensorSpec) for ann in source.function.__annotations__.values()
            )
            if has_symbolic and frontend == "tilelang" and backend == "cute":
                return DynamicJitKernel(source, target=target, out_idx=out_idx)
        program = self.import_program(source, frontend=frontend)
        lowered = self.lower(program, backend=backend, target=target, out_idx=out_idx)
        return self.compile(lowered).kernel
