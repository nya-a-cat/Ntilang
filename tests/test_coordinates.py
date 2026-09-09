import importlib.util
import math
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ir import CompileError
from ntilang.testing import reference

CUDA = pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe compiler unavailable")


@pytest.mark.parametrize("shape", [(1,), (2, 3), (2, 1, 5), (2, 3, 4, 5), ()])
@pytest.mark.parametrize("index", [-100, -1, 0, 1, 47, 1234])
def test_host_coordinate_arithmetic(shape, index):
    result = T.index_to_coordinates(index, shape)
    assert isinstance(result, list)
    if shape:
        assert result == list(np.unravel_index(index % math.prod(shape), shape))
    else:
        assert result == []


def coordinate_kernel(dtype="int32", dynamic=False, target="sm_80"):
    @T.prim_func
    def kernel(Index: T.Tensor((19,), dtype), B: T.Tensor((19, 2), "int32"), rows: T.int32, columns: T.int32):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                if dynamic:
                    shape = (T.max(1, T.min(rows, 3)), T.max(1, T.min(columns, 7)))
                else:
                    shape = (3, 7)
                x, y = T.index_to_coordinates(Index[i], shape)
                B[i, 0] = x
                B[i, 1] = y

    return ntilang.compile(kernel, target=target)


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("rows,columns", [(3, 7), (2, 4), (0, -1)])
def test_runtime_index_and_dynamic_shape(dtype, dynamic, rows, columns):
    index = (np.arange(19, dtype=np.int32) - 5).astype(dtype)
    b = np.empty((19, 2), dtype=np.int32)
    reference(coordinate_kernel(dtype, dynamic), index, b, rows, columns)
    shape = (max(1, min(rows, 3)), max(1, min(columns, 7))) if dynamic else (3, 7)
    expected = [np.unravel_index(int(value) % math.prod(shape), shape) for value in index]
    np.testing.assert_array_equal(b, expected)


def gather_coordinates():
    convert = T.index_to_coordinates

    @T.prim_func
    def kernel(
        A: T.Tensor((3, 7), "float32"), Index: T.Tensor((19,), "int32"), B: T.Tensor((19,), "float32")
    ):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                coordinates = convert(shape=A.shape, index=Index[i])
                B[i] = A[coordinates[0], coordinates[1]]

    return ntilang.compile(kernel)


def test_coordinates_compose_with_guarded_gathers_and_metadata():
    a = np.arange(21, dtype=np.float32).reshape(3, 7)
    index = np.arange(19, dtype=np.int32) - 7
    b = np.empty(19, dtype=np.float32)
    reference(gather_coordinates(), a, index, b)
    np.testing.assert_array_equal(b, a.ravel()[index % 21])


def macro_coordinates():
    @T.macro
    def get_shape(counter: T.Ref):
        counter += 1
        return (2, 3)

    @T.macro
    def get_index(counter: T.Ref):
        counter += 1
        return counter

    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            for i in T.Parallel(1):
                counter = T.alloc_var("int32", init=0)
                row, col = T.index_to_coordinates(shape=get_shape(counter), index=get_index(counter))
                B[i] = row * 3 + col + 100 * counter

    return ntilang.compile(kernel)


def test_coordinate_keywords_expand_once_in_source_order():
    b = np.empty(1, dtype=np.int32)
    reference(macro_coordinates(), b)
    assert b[0] == 202


def test_clamp_keyword_macro_order():
    @T.macro
    def next_value(counter: T.Ref):
        counter += 1
        snapshot = counter + 0
        return snapshot

    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            for i in T.Parallel(1):
                count = T.alloc_var("int32", init=0)
                value = T.clamp(max_val=next_value(count), dst=next_value(count), min_val=next_value(count))
                B[i] = value + count * 100

    b = np.empty(1, dtype=np.int32)
    reference(ntilang.compile(kernel), b)
    assert b[0] == 301


@pytest.mark.parametrize(
    "shape,match",
    [
        ((0, 3), "positive"),
        ((-1, 3), "positive"),
        ((True, 3), "integer"),
        ((2.5, 3), "integer"),
        (3, "tuple or list"),
    ],
)
def test_coordinate_invalid_shapes(shape, match):
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            for i in T.Parallel(1):
                coordinates = T.index_to_coordinates(1, shape)
                B[i] = coordinates[0]

    with pytest.raises(CompileError, match=match):
        ntilang.compile(kernel)


def test_zero_dynamic_divisor_is_rejected():
    @T.prim_func
    def kernel(A: T.Tensor((3,), "int32"), B: T.Tensor((1,), "int32"), n: T.int32):
        with T.Kernel(1):
            for i in T.Parallel(1):
                coordinates = T.index_to_coordinates(i, (T.min(n, 3),))
                B[i] = A[coordinates[0]]

    with pytest.raises(CompileError, match="zero|nonzero"):
        ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32"])
@CUDA
@pytest.mark.cuda
def test_coordinates_native(dtype):
    assert coordinate_kernel(dtype, dynamic=True).build().has_gpu_module


@pytest.mark.parametrize("factory", [gather_coordinates, macro_coordinates])
@CUDA
@pytest.mark.cuda
def test_coordinates_composition_native(factory):
    assert factory().build().has_gpu_module


@CUDA
@pytest.mark.cuda
def test_coordinates_standalone(tmp_path):
    path = coordinate_kernel(dynamic=True).save(tmp_path / "coordinates.py")
    code = "import runpy,sys; m=runpy.run_path(sys.argv[1]); assert m['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)


@pytest.mark.gpu
def test_coordinates_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA device unavailable")
    major, minor = torch.cuda.get_device_capability()
    indices = torch.arange(-5, 14, dtype=torch.int32, device="cuda")
    b = torch.empty((19, 2), dtype=torch.int32, device="cuda")
    coordinate_kernel(dynamic=True, target=f"sm_{major}{minor}")(indices, b, 2, 4)
    expected = torch.stack((torch.div(indices, 4, rounding_mode="floor") % 2, indices % 4), dim=1)
    torch.testing.assert_close(b, expected, rtol=0, atol=0)
