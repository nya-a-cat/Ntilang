import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ir import CompileError
from ntilang.testing import reference


def grid_kernel(shape=(3, 5), limit=False):
    n, m = shape
    @T.prim_func
    def kernel(A: T.Tensor((4, 3, 5), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1):
            for row in T.Parallel(4):
                total = T.alloc_var("float32", init=0)
                for i, j in T.grid(n, m):
                    if limit:
                        if j == 2:
                            break
                    total += A[row, i, j]
                B[row] = T.clamp(dst=total, min_val=-7., max_val=8.)
    return ntilang.compile(kernel)


@pytest.mark.parametrize("shape", [(3, 5), (0, 5), (3, 0), (1, 1)])
@pytest.mark.parametrize("limit", [False, True])
def test_grid_nest_and_innermost_break(shape, limit):
    a = np.arange(60, dtype=np.float32).reshape(4, 3, 5) / 10 - 2
    b = np.empty(4, dtype=np.float32)
    reference(grid_kernel(shape, limit), a, b)
    n, m = shape
    expected = a[:, :n, :min(m, 2) if limit else m].sum(axis=(1, 2))
    np.testing.assert_allclose(b, np.fmin(np.fmax(expected, -7), 8), atol=2e-6)


def test_grid_captures_extents_before_rebinding():
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            for row in T.Parallel(1):
                total = T.alloc_var("int32", init=0)
                i = 3
                for i, j in T.grid(i, i + 1):
                    total += i * 10 + j
                B[row] = total
    b = np.empty(1, dtype=np.int32)
    reference(ntilang.compile(kernel), b)
    assert b[0] == sum(i * 10 + j for i in range(3) for j in range(4))


@pytest.mark.parametrize("shape", [(-1, 3), (True, 3), (2.5, 3)])
def test_grid_rejects_invalid_extents(shape):
    with pytest.raises(CompileError, match="nonnegative static"):
        grid_kernel(shape)


def clamp_kernel(dtype="float32", lower=-2, upper=3):
    @T.prim_func
    def kernel(A: T.Tensor((8,), dtype), B: T.Tensor((8,), dtype)):
        with T.Kernel(1):
            for i in T.Parallel(8):
                B[i] = T.clamp(A[i], T.cast(lower, dtype), T.cast(upper, dtype))
    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int8", "int64"])
@pytest.mark.parametrize("lower,upper", [(-2, 3), (3, -2)])
def test_clamp_typed_operands(dtype, lower, upper):
    a = np.array([-8, -3, -2, 0, 1, 3, 4, 7], dtype=dtype)
    b = np.empty_like(a)
    reference(clamp_kernel(dtype, lower, upper), a, b)
    np.testing.assert_array_equal(b, np.fmin(np.fmax(a, lower), upper))


def test_clamp_nan_prefers_non_nan_operand():
    a = np.array([np.nan, np.inf, -np.inf, -0., 0., -2., 3., 4.], dtype=np.float32)
    b = np.empty_like(a)
    reference(clamp_kernel(), a, b)
    np.testing.assert_array_equal(b, np.fmin(np.fmax(a, -2), 3))


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_grid_clamp_cute_compilation():
    assert grid_kernel().build().has_gpu_module
    assert grid_kernel(limit=True).build().has_gpu_module
    assert clamp_kernel("bfloat16").build().has_gpu_module
