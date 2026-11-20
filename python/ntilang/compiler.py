"""Public compilation API with lazy, optional NVIDIA compiler integration."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .codegen import generate
from .compat import normalize_out_idx, resolve_target, torch_dtype_for
from .frontend import parse
from .ir import DTYPES, CompileError, Kernel, ScalarParameter, SymDim
from .language import PrimFunc, SymbolicDim, SymbolicTensorSpec
from .runtime import scalar_ffi_argument
from .validation import validate


@dataclass
class CompiledKernel:
    ir: Kernel
    target: str
    source: str
    out_idx: tuple[int, ...] = ()
    _executable: object = field(default=None, init=False, repr=False)

    @property
    def cache_key(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    def get_kernel_source(self) -> str:
        """Return the standalone CuTe DSL Python module source (TileLang API compatibility)."""
        return self.source

    def get_host_source(self) -> str:
        """Return the host/launch CuTe DSL Python module source."""
        return self.source

    def save(self, path: str | Path) -> Path:
        """Save a standalone CuTe DSL module with run() and compile_kernel()."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source, encoding="utf-8", newline="\n")
        return path

    def build(self):
        """Run NVIDIA's compiler for the explicit target, without launching the kernel."""
        if self._executable is not None:
            return self._executable
        if importlib.util.find_spec("cutlass") is None:
            raise RuntimeError(
                "CuTe DSL is required for build(). Source generation is available without it. "
                "Install ntilang[cuda] in a supported Linux environment; see README.md."
            )
        cache_root = Path(os.environ.get("NTILANG_CACHE_DIR", Path.home() / ".cache" / "ntilang"))
        path = self.save(cache_root / f"{self.cache_key}.py")
        module_name = f"_ntilang_generated_{self.cache_key}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            self._executable = module.compile_kernel()
        finally:
            sys.modules.pop(module_name, None)
        return self._executable

    @property
    def user_parameters(self):
        if not self.ir.sym_dims:
            return self.ir.parameters
        sym_param_names = {f"_nt_sym_{sym.name}" for sym in self.ir.sym_dims}
        return tuple(p for p in self.ir.parameters if p.name not in sym_param_names)

    def _extract_sym_bindings(self, user_params, arguments, out_set) -> dict[str, int]:
        bindings: dict[str, int] = {}
        in_params = [p for i, p in enumerate(user_params) if i not in out_set]
        pairs = (
            zip(user_params, arguments) if len(arguments) == len(user_params) else zip(in_params, arguments)
        )
        for param, arg in pairs:
            if isinstance(param, ScalarParameter):
                continue
            if not hasattr(arg, "shape"):
                continue
            if len(arg.shape) != len(param.type.shape):
                raise ValueError(f"{param.name}: expected rank {len(param.type.shape)}, got {len(arg.shape)}")
            for spec_dim, actual in zip(param.type.shape, arg.shape):
                if isinstance(spec_dim, SymDim):
                    actual_int = int(actual)
                    if actual_int <= 0:
                        raise ValueError(f"Symbolic dimension {spec_dim.name!r} must be positive")
                    prev = bindings.get(spec_dim.name)
                    if prev is not None and prev != actual_int:
                        raise ValueError(
                            f"Symbolic dimension {spec_dim.name!r} mismatch: {prev} vs {actual_int}"
                        )
                    bindings[spec_dim.name] = actual_int
        return bindings

    def _resolve_shape(self, shape, sym_bindings: dict[str, int]) -> tuple[int, ...]:
        return tuple(sym_bindings[d.name] if isinstance(d, SymDim) else d for d in shape)

    def _allocate_outputs(self, inputs: tuple, sym_bindings: dict[str, int]) -> list:
        torch = sys.modules.get("torch")
        if torch is None:
            torch = importlib.import_module("torch")
        device = None
        for arg in inputs:
            if (
                hasattr(arg, "device")
                and hasattr(arg, "__dlpack_device__")
                and arg.__dlpack_device__()[0] == 2
            ):
                device = arg.device
                break
        if device is None:
            device = torch.device("cuda")
        outputs = []
        user_params = self.user_parameters
        for idx in self.out_idx:
            param = user_params[idx]
            concrete_shape = self._resolve_shape(param.type.shape, sym_bindings)
            outputs.append(
                torch.empty(concrete_shape, dtype=torch_dtype_for(param.type.dtype, torch), device=device)
            )
        return outputs

    def __call__(self, *arguments):
        user_params = self.user_parameters
        out_set = set(self.out_idx)
        sym_bindings = self._extract_sym_bindings(user_params, arguments, out_set) if self.ir.sym_dims else {}
        allocated_outputs = None
        if self.out_idx and len(arguments) == len(user_params) - len(self.out_idx):
            allocated_outputs = self._allocate_outputs(arguments, sym_bindings)
            out_map = dict(zip(self.out_idx, allocated_outputs))
            input_iter = iter(arguments)
            arguments = tuple(
                out_map[i] if i in out_map else next(input_iter) for i in range(len(user_params))
            )
        if len(arguments) != len(user_params):
            raise TypeError(f"Expected {len(user_params)} arguments, received {len(arguments)}")
        if self.ir.sym_dims:
            sym_bindings = self._extract_sym_bindings(user_params, arguments, set())
            missing = [s.name for s in self.ir.sym_dims if s.name not in sym_bindings]
            if missing:
                raise ValueError(f"Unbound symbolic dimensions: {', '.join(missing)}")
            full_arguments = (*arguments, *(sym_bindings[s.name] for s in self.ir.sym_dims))
        else:
            full_arguments = arguments
        devices = set()
        ranges = []
        ffi_arguments = []
        for tensor, param in zip(full_arguments, self.ir.parameters):
            if isinstance(param, ScalarParameter):
                ffi_arguments.append(scalar_ffi_argument(tensor, param))
                continue
            ffi_arguments.append(tensor)
            if not hasattr(tensor, "__dlpack_device__") or tensor.__dlpack_device__()[0] != 2:
                raise TypeError(f"{param.name} must be a CUDA tensor supporting DLPack")
            devices.add(tensor.__dlpack_device__()[1])
            expected_shape = self._resolve_shape(param.type.shape, sym_bindings)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{param.name}: expected shape {expected_shape}, got {tuple(tensor.shape)}")
            if str(tensor.dtype).split(".")[-1] != param.type.dtype:
                raise ValueError(f"{param.name}: expected dtype {param.type.dtype}, got {tensor.dtype}")
            contiguous = (
                tensor.is_contiguous()
                if hasattr(tensor, "is_contiguous")
                else bool(getattr(getattr(tensor, "flags", None), "c_contiguous", False))
            )
            if not contiguous:
                raise ValueError(f"{param.name} must be contiguous in row-major order")
            pointer = (
                tensor.data_ptr()
                if hasattr(tensor, "data_ptr")
                else getattr(getattr(tensor, "data", None), "ptr", None)
            )
            if pointer is None or pointer % 16:
                raise ValueError(
                    f"{param.name} requires a tensor with an inspectable, 16-byte-aligned data pointer"
                )
            size = DTYPES[param.type.dtype]
            for dimension in expected_shape:
                size *= dimension
            for start, end, name in ranges:
                if pointer < end and start < pointer + size:
                    raise ValueError(f"Tensor arguments {name} and {param.name} must have disjoint storage")
            ranges.append((pointer, pointer + size, param.name))
        if len(devices) > 1:
            raise ValueError("All tensor arguments must be on the same CUDA device")
        result = self.build()(*ffi_arguments)
        if allocated_outputs is not None:
            return allocated_outputs[0] if len(allocated_outputs) == 1 else tuple(allocated_outputs)
        return result


class DynamicJitKernel:
    """Dynamic-shape JIT dispatcher: prefers single-compile CuTe SymInt (Phase B),
    falling back seamlessly to per-shape specialization caching (Phase A)."""

    def __init__(
        self,
        program: PrimFunc,
        *,
        target: str | None = "auto",
        out_idx: int | Sequence[int] | None = None,
        pass_configs: dict | None = None,
        symbolic: bool = True,
    ):
        self.program = program
        self.target = target
        self.out_idx = out_idx
        self.pass_configs = pass_configs
        self.symbolic = symbolic
        self._cache: dict[tuple, CompiledKernel] = {}
        self._symbolic_cache: dict[str, CompiledKernel | None] = {}
        self._param_names = tuple(inspect.signature(program.function).parameters.keys())
        self._annotations = self._resolve_symbolic_annotations()

    @property
    def compile_count(self) -> int:
        sym_count = sum(1 for k in self._symbolic_cache.values() if k is not None)
        return sym_count + len(self._cache)

    def _try_symbolic_kernel(self, resolved_target: str) -> CompiledKernel | None:
        if not self.symbolic:
            return None
        if resolved_target not in self._symbolic_cache:
            try:
                self._symbolic_cache[resolved_target] = compile(
                    self.program,
                    target=resolved_target,
                    out_idx=self.out_idx,
                    pass_configs=self.pass_configs,
                )
            except CompileError:
                self._symbolic_cache[resolved_target] = None
        return self._symbolic_cache[resolved_target]

    def _resolve_symbolic_annotations(self) -> dict[str, object]:
        closure = inspect.getclosurevars(self.program.function)
        env = {
            **self.program.function.__globals__,
            **dict(self.program.annotation_locals),
            **closure.nonlocals,
        }
        resolved = {}
        for name in self._param_names:
            ann = self.program.function.__annotations__.get(name)
            if isinstance(ann, str):
                try:
                    ann = eval(ann, env)
                except Exception:
                    pass
            resolved[name] = ann
        return resolved

    def _normalized_out_set(self) -> set[int]:
        if self.out_idx is None:
            return set()
        raw = (self.out_idx,) if isinstance(self.out_idx, int) else tuple(self.out_idx)
        n = len(self._param_names)
        return {idx + n if idx < 0 else idx for idx in raw}

    def specialize(self, *arguments) -> CompiledKernel:
        """Bind T.dynamic dimensions from concrete input tensors and return a cached CompiledKernel."""
        n_params = len(self._param_names)
        out_set = self._normalized_out_set()
        if len(arguments) == n_params:
            arg_pairs = list(zip(self._param_names, arguments))
        elif out_set and len(arguments) == n_params - len(out_set):
            in_names = [name for i, name in enumerate(self._param_names) if i not in out_set]
            arg_pairs = list(zip(in_names, arguments))
        else:
            expected = n_params - len(out_set) if out_set else n_params
            raise TypeError(f"Expected {expected} arguments, received {len(arguments)}")

        bindings: dict[str, int] = {}
        for name, arg in arg_pairs:
            ann = self._annotations.get(name)
            if not isinstance(ann, SymbolicTensorSpec):
                continue
            if not hasattr(arg, "shape"):
                raise TypeError(f"{name}: expected tensor argument with .shape")
            shape = tuple(int(d) for d in arg.shape)
            if len(shape) != len(ann.shape):
                raise ValueError(f"{name}: expected rank {len(ann.shape)}, got {len(shape)}")
            for spec_dim, actual_dim in zip(ann.shape, shape):
                if isinstance(spec_dim, SymbolicDim):
                    prev = bindings.get(spec_dim.name)
                    if prev is not None and prev != actual_dim:
                        raise ValueError(
                            f"Symbolic dimension {spec_dim.name!r} mismatch: {prev} vs {actual_dim}"
                        )
                    bindings[spec_dim.name] = actual_dim
                elif spec_dim != actual_dim:
                    raise ValueError(f"{name}: expected static dim {spec_dim}, got {actual_dim}")

        resolved_target = resolve_target(self.target)
        key = (tuple(sorted(bindings.items())), resolved_target)
        if key not in self._cache:
            # Map both the symbolic dim's name and any Python variable holding that SymbolicDim.
            closure = inspect.getclosurevars(self.program.function)
            env = {
                **self.program.function.__globals__,
                **dict(self.program.annotation_locals),
                **closure.nonlocals,
            }
            spec_items = dict(bindings)
            for var_name, val in env.items():
                if isinstance(val, SymbolicDim) and val.name in bindings:
                    spec_items[var_name] = bindings[val.name]
            specialized = PrimFunc(
                self.program.function,
                self.program.annotation_locals,
                tuple(sorted(spec_items.items())),
            )
            self._cache[key] = compile(
                specialized,
                target=resolved_target,
                out_idx=self.out_idx,
                pass_configs=self.pass_configs,
            )
        return self._cache[key]

    def __call__(self, *arguments):
        resolved_target = resolve_target(self.target)
        sym_kernel = self._try_symbolic_kernel(resolved_target)
        if sym_kernel is not None:
            return sym_kernel(*arguments)
        return self.specialize(*arguments)(*arguments)


def compile(
    program: PrimFunc,
    *,
    target: str | None = "sm_80",
    out_idx: int | Sequence[int] | None = None,
    pass_configs: dict | None = None,
) -> CompiledKernel:
    """Parse, check, and generate source on any host. CUDA compilation is lazy."""
    del pass_configs
    resolved_target = resolve_target(target)
    ir = parse(program)
    validate(ir)
    sym_param_names = {f"_nt_sym_{sym.name}" for sym in ir.sym_dims}
    user_params = tuple(p for p in ir.parameters if p.name not in sym_param_names)
    normalized_out = normalize_out_idx(out_idx, user_params)
    return CompiledKernel(ir, resolved_target, generate(ir, resolved_target), out_idx=normalized_out)
