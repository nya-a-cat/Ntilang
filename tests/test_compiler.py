import ast
import importlib.util
import itertools
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ir import Partition
from ntilang.testing import reference

from examples.fragment_affine import fragment_affine
from examples.matmul import matmul
from examples.vector_add import vector_add


@pytest.mark.parametrize(
    "shape,threads", [((1,), 32), ((257,), 128), ((7, 19), 64), ((3, 5, 7), 96), ((32, 32), 128)]
)
def test_partition_unique_and_complete(shape, threads):
    plan = Partition(shape, threads)
    visited = [
        c for t in range(threads) for s in range(plan.slots) if (c := plan.coordinates(t, s)) is not None
    ]
    expected = set(itertools.product(*(range(n) for n in shape)))
    assert len(visited) == len(set(visited))
    assert set(visited) == expected


@pytest.mark.parametrize("n,block", [(1, 1), (127, 96), (1000, 128), (513, 257)])
def test_vector_add_tail_semantics(n, block):
    rng = np.random.default_rng(7)
    a, b = (rng.normal(size=n).astype(np.float32) for _ in range(2))
    c = np.full(n, np.nan, np.float32)
    reference(vector_add(n, block), a, b, c)
    np.testing.assert_array_equal(c, a + b)


def test_fragment_affine_tail_semantics():
    a = np.linspace(-2, 2, 257, dtype=np.float32)
    b = np.full_like(a, np.nan)
    reference(fragment_affine(), a, b)
    np.testing.assert_array_equal(b, a * 2 + 1)


@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (32, 32, 32), (65, 71, 37)])
def test_gemm_tile_and_padding_semantics(m, n, k):
    rng = np.random.default_rng(19)
    a = rng.normal(size=(m, k)).astype(np.float16)
    b = rng.normal(size=(k, n)).astype(np.float16)
    c = np.full((m, n), np.nan, np.float32)
    reference(matmul(m, n, k), a, b, c)
    np.testing.assert_allclose(c, a.astype(np.float32) @ b.astype(np.float32), rtol=2e-5, atol=2e-5)


def test_output_standalone_and_deterministic(tmp_path):
    first, second = matmul(), matmul()
    assert first.source == second.source
    assert first.cache_key == second.cache_key
    assert vector_add(31).cache_key != vector_add(32).cache_key
    path = first.save(tmp_path / "generated.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [
        n.module if isinstance(n, ast.ImportFrom) else n.names[0].name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    assert all(name.startswith("cutlass") for name in imports)
    assert "cute.gemm(" in first.source
    assert "MmaF16BF16Op" in first.source


def test_core_import_is_dependency_free():
    code = "import sys, ntilang; print(','.join(sorted(set(sys.modules) & {'cutlass','tvm','tilelang','torch','numpy'})))"
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == ""


def test_invalid_threads():
    with pytest.raises(ntilang.CompileError, match="threads"):
        Partition((8,), 0)


def test_invalid_shape():
    with pytest.raises(ntilang.CompileError, match="positive"):
        T.Tensor((0,), "float32")


def test_missing_initialization_has_location():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            x = T.alloc_fragment((32,), "float32")
            T.copy(x, A[0])

    with pytest.raises(ntilang.CompileError, match="before initialization") as error:
        ntilang.compile(bad)
    assert error.value.location.filename.endswith("test_compiler.py")
    assert error.value.location.line > 1


def test_collective_in_parallel_is_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            x = T.alloc_shared((32,), "float32")
            for i in T.Parallel(32):
                T.copy(A, x)

    with pytest.raises(ntilang.CompileError, match="Collective"):
        ntilang.compile(bad)


def test_unsupported_pipeline_is_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for k in T.Pipelined(2, num_stages=3):
                for i in T.Parallel(32):
                    A[i] = 0.0

    with pytest.raises(ntilang.CompileError, match="multi-stage"):
        ntilang.compile(bad)


def test_gemm_shape_error():
    with pytest.raises(ntilang.CompileError, match="multiple of 16"):
        matmul(block_k=17)


def test_python_side_effects_are_rejected(tmp_path):
    path = str(tmp_path / "should-not-exist")

    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            open(path, "w")

    with pytest.raises(ntilang.CompileError, match="Only ntilang.language"):
        ntilang.compile(bad)
    assert not (tmp_path / "should-not-exist").exists()


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("factory", [vector_add, fragment_affine, matmul])
def test_real_cute_compilation_without_gpu(factory):
    compiled = factory().build()
    assert compiled.has_gpu_module
    binary = next(
        op.attributes["value"].value_bytes
        for op in compiled.ir_module.body.operations
        if op.name == "llvm.mlir.global" and str(op.attributes["sym_name"]) == '"kernels_binary"'
    )
    offset = binary.find(b"\x7fELF")
    assert offset >= 0
    # ELF e_machine = EM_CUDA (190), inside the NVIDIA fatbinary wrapper.
    assert int.from_bytes(binary[offset + 18 : offset + 20], "little") == 190


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_generated_module_compiles_without_ntilang(tmp_path):
    path = vector_add().save(tmp_path / "standalone.py")
    code = (
        "import runpy, sys; "
        "module = runpy.run_path(sys.argv[1]); "
        "f = module['compile_kernel'](); "
        "assert f.has_gpu_module; "
        "assert 'ntilang' not in sys.modules; "
        "print('standalone compiler check passed')"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)
