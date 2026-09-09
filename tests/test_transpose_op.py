import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ir import CompileError
from ntilang.testing import reference


def transpose_kernel(shape=(3, 5), dtype="float32", out_dtype=None, inplace=False):
    result_shape = (*shape[:-2], shape[-1], shape[-2])
    out_dtype = dtype if out_dtype is None else out_dtype

    @T.prim_func
    def kernel(A: T.Tensor(shape, dtype), B: T.Tensor(result_shape, out_dtype)):
        with T.Kernel(1, threads=64):
            src = T.alloc_shared(shape, dtype)
            dst = T.alloc_shared(result_shape, out_dtype)
            T.copy(A, src)
            if inplace:
                T.transpose(src, src)
                T.copy(src, B)
            else:
                T.transpose(src=src, dst=dst, annotations={})
                T.copy(dst, B)

    return ntilang.compile(kernel)


@pytest.mark.parametrize(
    "shape", [(3, 5), (1, 5), (5, 1), (1, 1), (2, 3, 5), (2, 1, 5), (2, 5, 1), (1, 1, 1), (2, 3, 1, 7)]
)
@pytest.mark.parametrize(
    "dtype,out_dtype", [("float32", "float32"), ("float16", "float32"), ("int16", "uint8"), ("bool", "int32")]
)
def test_transpose_batched_and_unit_axes(shape, dtype, out_dtype):
    a = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    b = np.empty((*shape[:-2], shape[-1], shape[-2]), dtype=out_dtype)
    reference(transpose_kernel(shape, dtype, out_dtype), a, b)
    np.testing.assert_array_equal(b, np.swapaxes(a, -1, -2).astype(out_dtype))


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (2, 7, 7)])
def test_transpose_inplace_snapshot(shape):
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    b = np.empty_like(a)
    reference(transpose_kernel(shape, inplace=True), a, b)
    np.testing.assert_array_equal(b, np.swapaxes(a, -1, -2))


def region_kernel(overlap=False):
    @T.prim_func
    def kernel(A: T.Tensor((8, 9), "float32"), B: T.Tensor((8, 9), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((8, 9), "float32")
            dst = T.alloc_shared((8, 9), "float32")
            T.copy(A, src)
            T.fill(dst, -7)
            if overlap:
                T.transpose(src[1:4, 2:7], src[2:7, 3:6])
                T.copy(src, B)
            else:
                T.transpose(src[1:4, 2:7], dst[2:7, 3:6])
                T.copy(dst, B)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("overlap", [False, True])
def test_transpose_regions(overlap):
    a = np.arange(72, dtype=np.float32).reshape(8, 9)
    b = np.empty_like(a)
    expected = a.copy() if overlap else np.full_like(a, -7)
    expected[2:7, 3:6] = a[1:4, 2:7].T
    reference(region_kernel(overlap), a, b)
    np.testing.assert_array_equal(b, expected)


@pytest.mark.parametrize(
    "initialize,shape,scope,annotations,match",
    [
        (False, (5, 3), "shared", None, "initialization"),
        (True, (3, 5), "shared", None, "swap"),
        (True, (5, 3), "fragment", None, "shared"),
        (True, (5, 3), "shared", {"unknown": 1}, "annotations"),
        (True, (5, 3), "shared", 1, "dictionary"),
    ],
)
def test_transpose_invalid(initialize, shape, scope, annotations, match):
    allocate = T.alloc_shared if scope == "shared" else T.alloc_fragment

    @T.prim_func
    def kernel(A: T.Tensor((3, 5), "float32"), B: T.Tensor(shape, "float32")):
        with T.Kernel(1):
            src = allocate((3, 5), "float32")
            dst = allocate(shape, "float32")
            if initialize:
                T.copy(A, src)
            T.transpose(src, dst, annotations)
            T.copy(dst, B)

    with pytest.raises(CompileError, match=match):
        ntilang.compile(kernel)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "shape,dtype,out_dtype",
    [
        ((3, 5), "float32", "float32"),
        ((2, 1, 5), "float16", "float32"),
        ((2, 5, 1), "bfloat16", "float32"),
        ((2, 3, 5), "uint64", "int64"),
    ],
)
def test_transpose_cute_compilation(shape, dtype, out_dtype):
    assert transpose_kernel(shape, dtype, out_dtype).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_transpose_overlap_cute_compilation():
    assert region_kernel(True).build().has_gpu_module
    assert transpose_kernel((2, 5, 5), inplace=True).build().has_gpu_module
