# Ntilang

采用 TileLang 风格语法、直接编译为纯 CuTe DSL 的 Python 语言。

Ntilang is a pure-Python tile language with a Python AST frontend, an independent
tile IR, static checks, and a standalone NVIDIA CuTe DSL backend. Its core package
has zero runtime dependencies. CuTe DSL is an optional dependency for CUDA
compilation and execution.

The first release implements synchronous tile copies, elementwise expressions,
register fragments, and warp-level Tensor Core GEMM. The frontend syntax is
inspired by TileLang. The compiler and IR are implemented in this repository.

## Installation

From this checkout, on Windows or Linux:

```sh
uv sync
uv run python examples/vector_add.py
```

The second command prints generated CuTe DSL source. It works without a GPU,
CUDA driver, or NVIDIA Python packages.

For NVIDIA compilation on Linux, use Python 3.12 and the pinned compiler extra:

```sh
uv sync --python 3.12 --extra cuda
```

The checked compiler version is `nvidia-cutlass-dsl==4.7.1`, with
`apache-tvm-ffi==0.1.4` for its Python tensor ABI. NVIDIA's compiler extra includes
native compiler libraries. Ntilang's own implementation and wheel are pure Python.

## Quick start

Save this kernel in a Python file so its source can be inspected:

```python
import ntilang
import ntilang.language as T


@T.prim_func
def add(
    A: T.Tensor((1000,), "float32"),
    B: T.Tensor((1000,), "float32"),
    C: T.Tensor((1000,), "float32"),
):
    with T.Kernel(T.ceildiv(1000, 128), threads=128) as bx:
        for i in T.Parallel(128):
            C[bx * 128 + i] = A[bx * 128 + i] + B[bx * 128 + i]


kernel = ntilang.compile(add, target="sm_80")
kernel.save("add_cute.py")
print(kernel.source)
```

`ntilang.compile()` parses and checks the program, then generates source.
`kernel.build()` invokes NVIDIA's compiler for the requested architecture. This
compilation can run on a CPU-only Linux host. `kernel(a, b, c)` builds lazily and
launches with CUDA tensors.

The saved module imports only NVIDIA's `cutlass` packages. It exposes
`run()` and `compile_kernel()` and can be compiled independently of Ntilang:

```python
from add_cute import compile_kernel

executable = compile_kernel()
# On a compatible NVIDIA GPU: executable(a, b, c)
```

Choose a target matching the GPU that will execute the kernel. The default
`sm_80` makes source generation independent of local GPU detection.

## Examples

- [Vector addition](examples/vector_add.py): non-divisible dimensions and predicated loads/stores.
- [Fragment affine transform](examples/fragment_affine.py): global-to-register tile copy and elementwise use.
- [Tiled matrix multiplication](examples/matmul.py): shared tiles, FP16 inputs, FP32 accumulation, and CuTe's `MmaF16BF16Op`.

Example factories accept `target=`. A shape-specialized factory can also be
decorated with `@ntilang.jit(target="sm_80")` and return a `@T.prim_func`.

On a compatible NVIDIA machine with CUDA-enabled PyTorch installed:

```python
import torch
from examples.matmul import matmul

major, minor = torch.cuda.get_device_capability()
kernel = matmul(target=f"sm_{major}{minor}")
a = torch.randn(65, 37, device="cuda", dtype=torch.float16)
b = torch.randn(37, 71, device="cuda", dtype=torch.float16)
c = torch.empty(65, 71, device="cuda", dtype=torch.float32)
kernel(a, b, c)
```

Arguments must have the declared shapes and dtypes, contiguous row-major layout,
16-byte-aligned data pointers, disjoint storage, and a common CUDA device.
The wrapper checks this metadata before launch. The standalone CuTe module uses
the same argument contract.

## Language support

- `T.Tensor`, `T.Kernel`, `T.Parallel`, `T.serial`, and `T.ceildiv`.
- `T.alloc_shared`, `T.alloc_fragment`, `T.copy`, `T.clear`, and `T.fill`.
- `T.gemm(A, B, accumulator, transpose_A=False, transpose_B=False)`.
- Arithmetic, comparisons, `T.cast`, `T.exp`, `T.exp2`, `T.sqrt`, `T.maximum`, and `T.minimum`.
- Static shapes; `float16`, `bfloat16`, `float32`, and `int32` buffers.

Global tile loads outside the tensor return zero; global stores outside the
tensor are masked. The initial write checker accepts disjoint affine tile
indices. A parameter has one global write site and cannot also be read. Buffers
are allocated at kernel scope, and collective operations execute outside
parallel element loops.

Tensor Core GEMM uses FP16/BF16 operands and FP32 accumulation, with a K tile
multiple of 16 and a supported warp arrangement. This version uses synchronous
shared-memory copies. Dynamic shapes, arbitrary Python control flow, data-dependent
indices, reductions, atomics, asynchronous pipelines, TMA, and WGMMA are outside
the implemented subset. `T.Pipelined` accepts only synchronous stage counts 0 or 1.
Unsupported constructs produce compilation errors.

See [the language semantics](docs/semantics.md) for the precise restrictions and
[the compiler architecture](docs/architecture.md) for the implementation.

## Testing

```sh
uv run pytest
uv run ruff check python examples tests
uv run ruff format --check python examples tests
uv build
```

CPU tests cover parsing, diagnostics, write ownership, indexing, tail tiles,
NumPy IR evaluation, and generated-module independence. With the CUDA extra
installed, the suite also runs the actual NVIDIA compiler and checks embedded
CUDA ELF output. GPU tests are explicitly selected with:

```sh
uv run pytest -m gpu
```

For mathematical debugging on CPU:

```python
import numpy as np
from examples.vector_add import vector_add
from ntilang.testing import reference

a = np.arange(1000, dtype=np.float32)
b = np.ones_like(a)
c = np.empty_like(a)
reference(vector_add(), a, b, c)
np.testing.assert_array_equal(c, a + b)
```

The NumPy evaluator executes the IR serially. GPU scheduling, Tensor Core
rounding, and device performance require hardware validation. Full compilation
chain formal verification remains future work.

## References

- [TileLang](https://github.com/tile-ai/tilelang) for the source-language style.
- [NVIDIA CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) for the target language.
- [CuTe DSL TVM FFI compilation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/tvm_ffi_compilation.html) for fake-tensor compilation and the tensor ABI.
- [NVIDIA CUTLASS examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) for CuTe tensor and MMA APIs.

## License

[MIT](LICENSE).
