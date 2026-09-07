import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def math_kernel():
    @T.prim_func
    def math_ops(A: T.Tensor((39,), "float32"), B: T.Tensor((39,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                x = A[bx * 32 + i]
                y = T.sqrt(T.maximum(x, 0.0))
                B[bx * 32 + i] = T.minimum(T.exp(-y) + T.exp2(-y), 2.0)

    return ntilang.compile(math_ops)


def transposed_matmul(dtype="float16", *, transpose_a=False):
    m, n, k = 19, 23, 17

    if transpose_a:

        @T.prim_func
        def gemm_a(A: T.Tensor((k, m), dtype), B: T.Tensor((k, n), dtype), C: T.Tensor((m, n), "float32")):
            with T.Kernel(1, 1, threads=128) as (bx, by):
                sa = T.alloc_shared((32, 32), dtype)
                sb = T.alloc_shared((32, 32), dtype)
                acc = T.alloc_fragment((32, 32), "float32")
                T.clear(acc)
                T.copy(A[0, bx * 32], sa)
                T.copy(B[0, by * 32], sb)
                T.gemm(sa, sb, acc, transpose_A=True)
                T.copy(acc, C[bx * 32, by * 32])

        return ntilang.compile(gemm_a)

    @T.prim_func
    def gemm_b(A: T.Tensor((m, k), dtype), B: T.Tensor((n, k), dtype), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            sa = T.alloc_shared((32, 32), dtype)
            sb = T.alloc_shared((32, 32), dtype)
            acc = T.alloc_fragment((32, 32), "float32")
            T.clear(acc)
            T.copy(A[bx * 32, 0], sa)
            T.copy(B[by * 32, 0], sb)
            T.gemm(sa, sb, acc, transpose_B=True)
            T.copy(acc, C[bx * 32, by * 32])

    return ntilang.compile(gemm_b)


def test_math_reference_and_nan_contract():
    a = np.linspace(-1, 3, 39, dtype=np.float32)
    a[5] = np.nan
    b = np.empty_like(a)
    reference(math_kernel(), a, b)
    y = np.sqrt(np.maximum(a, 0.0))
    np.testing.assert_allclose(b, np.minimum(np.exp(-y) + np.exp2(-y), 2.0), equal_nan=True)


@pytest.mark.parametrize("transpose_a", [False, True])
def test_transposed_gemm_reference(transpose_a):
    rng = np.random.default_rng(7)
    a = rng.normal(size=(17, 19) if transpose_a else (19, 17)).astype(np.float16)
    b = rng.normal(size=(17, 23) if transpose_a else (23, 17)).astype(np.float16)
    c = np.empty((19, 23), dtype=np.float32)
    reference(transposed_matmul(transpose_a=transpose_a), a, b, c)
    expected = (a.T if transpose_a else a).astype(np.float32) @ (b if transpose_a else b.T).astype(np.float32)
    np.testing.assert_allclose(c, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("kind", ["math", "transpose_a", "transpose_b", "bfloat16"])
def test_operations_compile_to_cute(kind):
    kernel = (
        math_kernel()
        if kind == "math"
        else transposed_matmul(
            "bfloat16" if kind == "bfloat16" else "float16", transpose_a=kind == "transpose_a"
        )
    )
    assert kernel.build().has_gpu_module
