<div align="center">

# Ntilang

**TileLang-style Python, compiled to standalone CuTe DSL.**

[![CI](https://github.com/nya-a-cat/Ntilang/actions/workflows/ci.yml/badge.svg)](https://github.com/nya-a-cat/Ntilang/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Quick start](#quick-start) · [Examples](#examples) · [Documentation](#documentation)

</div>

Ntilang is a Python language and compiler for writing tiled GPU kernels. Express
computation with TileLang-style loops, buffers, and operators, then inspect and
export the NVIDIA CuTe DSL source it generates.

Use it to write custom kernels, study how tile operations map to CuTe, or
experiment with a compiler whose frontend, IR, and code generator are all Python.

## Highlights

- **Program with tiles:** compose shared-memory copies, register fragments,
  reductions, and FP16/BF16 Tensor Core GEMM with FP32 accumulation. Add elementwise
  epilogues in the same kernel.
- **Keep the generated source:** export a standalone CuTe Python module that
  compiles independently of Ntilang. Read the generated layouts, operations,
  and launch code directly.
- **Check kernels before execution:** static checks cover shapes, dtypes,
  initialization, and supported write-ownership patterns. Scalar assertions check
  preconditions before launch, including in exported modules. A serial NumPy
  evaluator helps check the mathematics on CPU.
- **Start with Python:** the core package has zero runtime dependencies and
  generates source on Windows and Linux. NVIDIA's compiler is an optional
  dependency for CUDA compilation and execution.

## Installation

Clone the repository and run the first example with [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/nya-a-cat/Ntilang.git
cd Ntilang
uv sync --frozen
uv run --frozen python examples/vector_add.py
```

This prints a complete CuTe DSL module. Source generation works without a GPU
or CUDA driver.

For CUDA compilation, use Linux with Python 3.12 and the pinned compiler extra:

```sh
uv sync --frozen --python 3.12 --extra cuda
```

The tested toolchain is NVIDIA CuTe DSL 4.7.1 with TVM FFI 0.1.11. Native
compilation can run on a CPU-only Linux host; launching a kernel requires a
compatible NVIDIA GPU and CUDA tensor inputs.

## Quick start

Save the following as `add.py` so Ntilang can inspect the function's source:

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

Run `uv run --frozen python add.py` to generate `add_cute.py`. The final tile
handles the remaining elements through guarded loads and masked stores.

`ntilang.compile()` parses and checks the kernel, then generates source.
`kernel.build()` invokes NVIDIA's compiler. Calling `kernel(a, b, c)` builds
lazily and launches the kernel with your CUDA tensors. Select a `target` matching
the GPU that will execute it.

The exported module exposes its own compilation entry point:

```python
from add_cute import compile_kernel

executable = compile_kernel()
# On a compatible NVIDIA GPU: executable(a, b, c)
```

The generated module uses NVIDIA's `cutlass` packages and can be compiled in a
separate environment with the same NVIDIA toolchain. Tensor arguments follow
the declared shapes and dtypes, use contiguous row-major storage, and have
disjoint, 16-byte-aligned buffers on one CUDA device. See the
[argument contract](docs/semantics.md) for details.

## Examples

Each example includes a kernel factory that accepts `target=`. Start with vector
addition, then follow the data movement and accumulator layout in matrix
multiplication.

| Example | What it demonstrates |
| --- | --- |
| [Vector addition](examples/vector_add.py) | Parallel tiles and boundary handling |
| [Tiled matrix multiplication](examples/matmul.py) | Shared-memory tiles and warp-level Tensor Core GEMM |
| [Matrix multiplication + ReLU](examples/matmul_relu.py) | Scale, bias, and activation in the accumulator layout |
| [Softmax](examples/softmax.py) | Maximum and sum reductions with fragment broadcasts |
| [Cumulative sum](examples/cumsum.py) | Inclusive row scans, reverse traversal, and tail tiles |
| [Runtime window sum](examples/window_sum.py) | Runtime grid bounds, integer clamping, and tail batches |
| [Tiled transpose](examples/transpose.py) | Sliced copies and shared-memory communication |
| [Fragment affine transform](examples/fragment_affine.py) | Elementwise operations on register fragments |
| [Piecewise transform](examples/piecewise.py) | Data-dependent branches and guarded stores |

## Documentation

- [Language semantics](docs/semantics.md): syntax, scalar types, macros,
  control flow, memory operations, and numerical behavior.
- [Compiler architecture](docs/architecture.md): the Python AST frontend,
  independent tile IR, validation, and CuTe lowering.
- [TileLang compatibility](docs/compatibility.md): supported behavior and
  the remaining language surface.
- [Roadmap](ROADMAP.md): implementation stages and the verified baseline.

## Project status

Ntilang is experimental and implements a subset of TileLang-style syntax.
Current kernels use static shapes and synchronous data movement. Dynamic shapes,
atomics, asynchronous pipelines, TMA, and WGMMA require further implementation.
Unsupported constructs produce compilation errors.

[CI](https://github.com/nya-a-cat/Ntilang/actions/workflows/ci.yml) checks Python
semantics on Windows and Linux, native CuTe compilation, and standalone generated
modules. GPU execution and performance remain unverified; the NumPy evaluator
models serial mathematical behavior. Detailed restrictions and validation
results are recorded in the documentation above.

## Development

Bug reports and focused contributions are welcome. Include a minimal kernel,
the target architecture, and the compiler versions when reporting a problem.

The repository uses uv, pytest, and Ruff:

```sh
uv sync --frozen
uv run --frozen pytest
uv run --frozen ruff check python examples tests
uv run --frozen ruff format --check python examples tests
uv build
```

The default test selection excludes GPU tests. With the CUDA extra and a
compatible GPU, select them with `uv run --frozen pytest -m gpu`.

## Acknowledgments

Ntilang's source syntax is inspired by [TileLang](https://github.com/tile-ai/tilelang).
Its backend uses [NVIDIA CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html),
with [CUTLASS examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL)
as references for tensor and MMA APIs. Ntilang's frontend and IR are implemented
in this repository.

## License

[MIT](LICENSE).
