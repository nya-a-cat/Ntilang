import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def strided_accumulate(start, stop, step, *, unroll=False):
    loop = T.unroll if unroll else T.serial

    @T.prim_func
    def kernel(A: T.Tensor((97,), "float32"), B: T.Tensor((97,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            tile = T.alloc_fragment((64,), "float32")
            T.copy(A[bx * 64], tile)
            for i in T.Parallel(64):
                for k in loop(start, stop, step):
                    tile[i] += T.cast(k, "float32")
            T.copy(tile, B[bx * 64])

    return ntilang.compile(kernel)


def fragment_initialization():
    @T.prim_func
    def kernel(A: T.Tensor((7, 19), "float32"), B: T.Tensor((7, 19), "float32")):
        with T.Kernel(1, threads=64) as _bx:
            tile = T.alloc_fragment((7, 19), "float32")
            for i, j in T.Parallel(7, 19):
                tile[i, j] = A[i, j] * 2.0
                tile[i, j] += 1.0
            T.copy(tile, B)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("extent", [(1, 8, 2), (7, -2, -3), (4, 4, 1), (0, 3, -1)])
@pytest.mark.parametrize("unroll", [False, True])
def test_strided_loop_reference(extent, unroll):
    a = np.arange(97, dtype=np.float32)
    b = np.full_like(a, np.nan)
    reference(strided_accumulate(*extent, unroll=unroll), a, b)
    np.testing.assert_array_equal(b, a + sum(range(*extent)))


def test_fragment_write_reference():
    a = np.arange(7 * 19, dtype=np.float32).reshape(7, 19)
    b = np.full_like(a, np.nan)
    reference(fragment_initialization(), a, b)
    np.testing.assert_array_equal(b, a * 2 + 1)


def test_empty_loop_does_not_initialize_fragment():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((32,), "float32")
            for k in T.serial(0):
                T.clear(tile)
            T.copy(tile, A)

    with pytest.raises(ntilang.CompileError, match="before initialization"):
        ntilang.compile(bad)


def test_zero_step_rejected():
    with pytest.raises(ntilang.CompileError, match="nonzero"):
        strided_accumulate(0, 5, 0)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("unroll", [False, True])
def test_strided_loop_cute_compilation(unroll):
    compiled = strided_accumulate(7, -2, -3, unroll=unroll).build()
    assert compiled.has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_fragment_write_cute_compilation():
    assert fragment_initialization().build().has_gpu_module
