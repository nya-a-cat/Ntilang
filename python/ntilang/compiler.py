"""Public compilation API with lazy, optional NVIDIA compiler integration."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .codegen import generate
from .frontend import parse
from .ir import DTYPES, Kernel, ScalarParameter
from .language import PrimFunc
from .runtime import scalar_ffi_argument
from .validation import validate


@dataclass
class CompiledKernel:
    ir: Kernel
    target: str
    source: str
    _executable: object = field(default=None, init=False, repr=False)

    @property
    def cache_key(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

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

    def __call__(self, *arguments):
        if len(arguments) != len(self.ir.parameters):
            raise TypeError(f"Expected {len(self.ir.parameters)} arguments, received {len(arguments)}")
        devices = set()
        ranges = []
        ffi_arguments = []
        for tensor, param in zip(arguments, self.ir.parameters):
            if isinstance(param, ScalarParameter):
                ffi_arguments.append(scalar_ffi_argument(tensor, param))
                continue
            ffi_arguments.append(tensor)
            if not hasattr(tensor, "__dlpack_device__") or tensor.__dlpack_device__()[0] != 2:
                raise TypeError(f"{param.name} must be a CUDA tensor supporting DLPack")
            devices.add(tensor.__dlpack_device__()[1])
            if tuple(tensor.shape) != param.type.shape:
                raise ValueError(
                    f"{param.name}: expected shape {param.type.shape}, got {tuple(tensor.shape)}"
                )
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
            for dimension in param.type.shape:
                size *= dimension
            for start, end, name in ranges:
                if pointer < end and start < pointer + size:
                    raise ValueError(f"Tensor arguments {name} and {param.name} must have disjoint storage")
            ranges.append((pointer, pointer + size, param.name))
        if len(devices) != 1:
            raise ValueError("All tensor arguments must be on the same CUDA device")
        return self.build()(*ffi_arguments)


def compile(program: PrimFunc, *, target: str = "sm_80") -> CompiledKernel:
    """Parse, check, and generate source on any host. CUDA compilation is lazy."""
    ir = parse(program)
    validate(ir)
    return CompiledKernel(ir, target, generate(ir, target))
