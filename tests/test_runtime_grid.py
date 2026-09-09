import importlib.util
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ir import CompileError
from ntilang.testing import reference

CUDA = pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe compiler unavailable")


def runtime_grid(dtype="int32", splat=False, target="sm_80"):
    @T.prim_func
    def kernel(A: T.Tensor((4, 3, 5), "float32"), B: T.Tensor((4,), "float32"), n: dtype, m: dtype):
        with T.Kernel(1, threads=32):
            for row in T.Parallel(4):
                total = T.alloc_var("float32", init=0)
                if splat:
                    for i, j in T.grid(*(T.max(0, T.min(n, 3)), T.max(0, T.min(m, 5)))):
                        total += A[row, i, j]
                else:
                    for i, j in T.grid(T.max(0, T.min(n, 3)), T.max(0, T.min(m, 5))):
                        total += A[row, i, j]
                B[row] = total
    return ntilang.compile(kernel, target=target)


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"])
@pytest.mark.parametrize("splat", [False, True])
@pytest.mark.parametrize("n,m", [(0, 5), (3, 0), (2, 4), (7, 9)])
def test_runtime_scalar_grid(dtype, splat, n, m):
    a = np.arange(60, dtype=np.float32).reshape(4, 3, 5)
    b = np.empty(4, dtype=np.float32)
    reference(runtime_grid(dtype, splat), a, b, n, m)
    np.testing.assert_array_equal(b, a[:, :min(n, 3), :min(m, 5)].sum(axis=(1, 2)))


def ragged_grid():
    @T.prim_func
    def kernel(A: T.Tensor((7, 3, 5), "int32"), Length: T.Tensor((7, 2), "int32"), B: T.Tensor((7,), "int32")):
        with T.Kernel(1, threads=32):
            for row in T.Parallel(8):
                total = T.alloc_var("int32", init=0)
                for i, j in T.grid(T.max(0, T.min(Length[row, 0], 3)), T.max(0, T.min(Length[row, 1], 5))):
                    if j == 3:
                        break
                    if i == 1:
                        continue
                    total += A[row, i, j]
                B[row] = total
    return ntilang.compile(kernel)


def test_ragged_grid_and_innermost_early_exits():
    a = np.arange(105, dtype=np.int32).reshape(7, 3, 5)
    lengths = np.array([[-1, 5], [3, 0], [2, 4], [3, 5], [9, 9], [1, 2], [0, -1]], dtype=np.int32)
    b = np.empty(7, dtype=np.int32)
    reference(ragged_grid(), a, lengths, b)
    expected = [sum(a[row, i, j] for i in range(max(0, min(n, 3))) for j in range(max(0, min(m, 3))) if i != 1) for row, (n, m) in enumerate(lengths)]
    np.testing.assert_array_equal(b, expected)


def mutable_grid():
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            for row in T.Parallel(1):
                n = T.alloc_var("int32", init=2)
                total = T.alloc_var("int32", init=0)
                for i, j in T.grid(3, T.max(0, T.min(n, 2))):
                    n = 0
                    total += 1
                B[row] = total
    return ntilang.compile(kernel)


def test_each_nested_runtime_bound_is_captured_at_its_own_loop_entry():
    b = np.empty(1, dtype=np.int32)
    reference(mutable_grid(), b)
    assert b[0] == 2


def macro_grid():
    @T.macro
    def dimensions(counter: T.Ref):
        counter += 1
        return (2, 3)
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            for row in T.Parallel(1):
                counter = T.alloc_var("int32", init=0)
                total = T.alloc_var("int32", init=0)
                for i, j in T.grid(*dimensions(counter)):
                    total += 1
                B[row] = total + counter * 100
    return ntilang.compile(kernel)


def test_starred_macro_runs_once_before_the_nest():
    b = np.empty(1, dtype=np.int32)
    reference(macro_grid(), b)
    assert b[0] == 106


def metadata_grid():
    @T.prim_func
    def kernel(A: T.Tensor((2, 3), "int32"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            for row in T.Parallel(1):
                total = T.alloc_var("int32", init=0)
                for i, j in T.grid(*A.shape):
                    total += A[i, j]
                B[row] = total
    return ntilang.compile(kernel)


def test_metadata_shape_splat():
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    b = np.empty(1, dtype=np.int32)
    reference(metadata_grid(), a, b)
    assert b[0] == 15


def test_runtime_grid_does_not_establish_initialization():
    @T.prim_func
    def kernel(B: T.Tensor((2,), "float32"), n: T.int32):
        with T.Kernel(1):
            tile = T.alloc_shared((2,), "float32")
            for i in T.grid(T.max(0, T.min(n, 2))):
                T.fill(tile, 1)
            T.copy(tile, B)
    with pytest.raises(CompileError, match="initializ"):
        ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["float32", "float64", "bool"])
def test_runtime_grid_rejects_noninteger_bounds(dtype):
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32"), n: dtype):
        with T.Kernel(1):
            for row in T.Parallel(1):
                total = T.alloc_var("int32", init=0)
                for i in T.grid(n):
                    total += 1
                B[row] = total
    with pytest.raises(CompileError, match="integer"):
        ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "uint32", "int64", "uint64"])
@CUDA
@pytest.mark.cuda
def test_runtime_grid_native(dtype):
    assert runtime_grid(dtype, splat=True).build().has_gpu_module


@pytest.mark.parametrize("factory", [ragged_grid, mutable_grid, macro_grid, metadata_grid])
@CUDA
@pytest.mark.cuda
def test_runtime_grid_composition_native(factory):
    assert factory().build().has_gpu_module


@CUDA
@pytest.mark.cuda
def test_runtime_grid_export_is_standalone(tmp_path):
    path = runtime_grid(splat=True).save(tmp_path / "grid.py")
    code = "import runpy,sys; m=runpy.run_path(sys.argv[1]); assert m['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)


@pytest.mark.gpu
def test_runtime_grid_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA device unavailable")
    major, minor = torch.cuda.get_device_capability()
    a = torch.arange(60, dtype=torch.float32, device="cuda").reshape(4, 3, 5)
    b = torch.empty(4, dtype=torch.float32, device="cuda")
    runtime_grid(target=f"sm_{major}{minor}")(a, b, 2, 4)
    torch.testing.assert_close(b, a[:, :2, :4].sum(dim=(1, 2)), rtol=0, atol=0)


@pytest.mark.parametrize("dtype,n,m", [("int64", -(2**63), 2**63-1), ("uint64", 2**64-1, 2**64-1), ("uint32", 2**32-1, 2**32-1)])
def test_full_width_data_can_be_clipped_before_indexing(dtype, n, m):
    a = np.arange(60, dtype=np.float32).reshape(4, 3, 5)
    b = np.empty(4, dtype=np.float32)
    reference(runtime_grid(dtype), a, b, n, m)
    np.testing.assert_array_equal(b, a[:, :max(0, min(n, 3)), :max(0, min(m, 5))].sum(axis=(1, 2)))


@pytest.mark.parametrize("case", ["arithmetic", "narrowing", "signedness"])
def test_clipping_keeps_arithmetic_and_conversion_checks(case):
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32"), n: T.int64):
        with T.Kernel(1):
            for row in T.Parallel(1):
                total = T.alloc_var("int32", init=0)
                if case == "arithmetic":
                    bound = T.max(0, T.min(n + 1, 3))
                elif case == "narrowing":
                    bound = T.max(0, T.min(T.cast(n, "int32"), 3))
                else:
                    bound = T.min(n, T.uint64(3))
                for i in T.grid(bound):
                    total += 1
                B[row] = total
    with pytest.raises(CompileError, match="overflow|conversion|cast"):
        ntilang.compile(kernel)


@pytest.mark.parametrize("rows,columns", [(-1, 5), (0, 0), (2, 4), (10, 10)])
def test_window_sum_example(rows, columns):
    from examples.window_sum import window_sum
    a = np.arange(105, dtype=np.float32).reshape(7, 3, 5)
    b = np.empty(7, dtype=np.float32)
    reference(window_sum(), a, b, rows, columns)
    np.testing.assert_array_equal(b, a[:, :max(0, min(rows, 3)), :max(0, min(columns, 5))].sum(axis=(1, 2)))


@CUDA
@pytest.mark.cuda
def test_window_sum_native():
    from examples.window_sum import window_sum
    assert window_sum().build().has_gpu_module
