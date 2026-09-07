import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def broadcast_rows():
    @T.prim_func
    def kernel(
        A: T.Tensor((7, 19), "float32"),
        Bias: T.Tensor((7,), "float32"),
        B: T.Tensor((7, 19), "float32"),
    ):
        with T.Kernel(1, threads=64) as _bx:
            bias = T.alloc_fragment((7,), "float32")
            tile = T.alloc_fragment((7, 19), "float32")
            T.copy(Bias, bias)
            for i, j in T.Parallel(7, 19):
                tile[i, j] = A[i, j] + bias[i]
            T.copy(tile, B)

    return ntilang.compile(kernel)


def transpose_fragment():
    @T.prim_func
    def kernel(A: T.Tensor((7, 19), "float32"), B: T.Tensor((19, 7), "float32")):
        with T.Kernel(1, threads=64) as _bx:
            source = T.alloc_fragment((7, 19), "float32")
            T.copy(A, source)
            for i, j in T.Parallel(19, 7):
                B[i, j] = source[j, i]

    return ntilang.compile(kernel)


def shifted_fragment():
    @T.prim_func
    def kernel(A: T.Tensor((97,), "float32"), B: T.Tensor((97,), "float32")):
        with T.Kernel(1, threads=64) as _bx:
            source = T.alloc_fragment((97,), "float32")
            T.copy(A, source)
            for i in T.Parallel(97):
                B[i] = source[i - 1] + source[i + 1]

    return ntilang.compile(kernel)


def test_broadcast_reference():
    a = np.arange(133, dtype=np.float32).reshape(7, 19)
    bias = np.arange(7, dtype=np.float32)
    b = np.full_like(a, np.nan)
    reference(broadcast_rows(), a, bias, b)
    np.testing.assert_array_equal(b, a + bias[:, None])


def test_transpose_reference():
    a = np.arange(133, dtype=np.float32).reshape(7, 19)
    b = np.full((19, 7), np.nan, dtype=np.float32)
    reference(transpose_fragment(), a, b)
    np.testing.assert_array_equal(b, a.T)


def test_shifted_reference():
    a = np.arange(97, dtype=np.float32)
    b = np.full_like(a, np.nan)
    reference(shifted_fragment(), a, b)
    padded = np.pad(a, (1, 1))
    np.testing.assert_array_equal(b, padded[:-2] + padded[2:])


def test_cross_element_inplace_update_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((32,), "float32")
            T.clear(tile)
            for i in T.Parallel(32):
                tile[i] = tile[31 - i] + 1.0
            T.copy(tile, A)

    with pytest.raises(ntilang.CompileError, match="separate source fragment"):
        ntilang.compile(bad)


def test_communication_shared_memory_limit():
    @T.prim_func
    def bad(A: T.Tensor((13000,), "float32")):
        with T.Kernel(1, threads=128) as _bx:
            tile = T.alloc_fragment((13000,), "float32")
            T.clear(tile)
            for i in T.Parallel(13000):
                A[i] = tile[12999 - i]

    with pytest.raises(ntilang.CompileError, match="communication exceed 48 KiB"):
        ntilang.compile(bad)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("factory", [broadcast_rows, transpose_fragment, shifted_fragment])
def test_fragment_communication_cute_compilation(factory):
    assert factory().build().has_gpu_module
