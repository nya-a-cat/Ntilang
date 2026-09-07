import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference

from examples.piecewise import piecewise


def conditional_fragment():
    @T.prim_func
    def kernel(A: T.Tensor((93,), "float32"), B: T.Tensor((93,), "float32")):
        with T.Kernel(3, threads=32) as bx:
            tile = T.alloc_fragment((32,), "float32")
            for i in T.Parallel(32):
                x = A[bx * 32 + i]
                if x > 0.0:
                    tile[i] = x
                else:
                    tile[i] = -x
                if i < 16:
                    tile[i] += 1.0
            T.copy(tile, B[bx * 32])

    return ntilang.compile(kernel)


def uniform_collective():
    @T.prim_func
    def kernel(A: T.Tensor((93,), "float32"), B: T.Tensor((93,), "float32")):
        with T.Kernel(3, threads=32) as bx:
            tile = T.alloc_shared((32,), "float32")
            if bx % 2 == 0:
                T.copy(A[bx * 32], tile)
            else:
                T.fill(tile, -2.0)
            T.copy(tile, B[bx * 32])

    return ntilang.compile(kernel)


def scalar_join():
    @T.prim_func
    def kernel(A: T.Tensor((93,), "float32"), B: T.Tensor((93,), "float32")):
        with T.Kernel(3, threads=32) as bx:
            for i in T.Parallel(32):
                x = A[bx * 32 + i]
                if x > 0.0:
                    offset = bx * 32
                    index = offset + i
                    value = x * 2.0
                else:
                    index = bx * 32 + i
                    value = -x
                B[index] = value

    return ntilang.compile(kernel)


@pytest.mark.parametrize("factory", [piecewise, conditional_fragment, uniform_collective, scalar_join])
def test_conditional_reference(factory):
    a = np.linspace(-2, 2, 93, dtype=np.float32)
    b = np.full_like(a, np.nan)
    reference(factory(), a, b)
    expected = {
        piecewise: np.where(a < 0, -a, np.where(a < 1, a * a, a + 2)),
        conditional_fragment: np.abs(a) + (np.arange(93) % 32 < 16),
        uniform_collective: np.where(np.arange(93) // 32 % 2 == 0, a, -2),
        scalar_join: np.where(a > 0, a * 2, -a),
    }[factory]
    np.testing.assert_array_equal(b, expected)


def test_partial_conditional_initialization_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((32,), "float32")
            for i in T.Parallel(32):
                if i < 16:
                    tile[i] = 1.0
            T.copy(tile, A)

    with pytest.raises(ntilang.CompileError, match="before initialization"):
        ntilang.compile(bad)


def test_divergent_collective_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_shared((32,), "float32")
            for i in T.Parallel(32):
                if i < 16:
                    T.clear(tile)
            T.copy(tile, A)

    with pytest.raises(ntilang.CompileError, match="Collective"):
        ntilang.compile(bad)


def test_conditional_cross_lane_collision_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                if i < 16:
                    A[i] = 1.0
                else:
                    A[i - 16] = 2.0

    with pytest.raises(ntilang.CompileError, match="identical ownership"):
        ntilang.compile(bad)


def test_single_branch_local_does_not_escape():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                if i < 16:
                    value = 2.0
                A[i] = value

    with pytest.raises(ntilang.CompileError, match="static specialization"):
        ntilang.compile(bad)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("factory", [piecewise, conditional_fragment, uniform_collective, scalar_join])
def test_conditional_cute_compilation(factory):
    assert factory().build().has_gpu_module
