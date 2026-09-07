import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference

from examples.matmul_relu import matmul_relu


def mma_copy_chain(threads=128):
    @T.prim_func
    def kernel(
        A: T.Tensor((29, 17), "float16"),
        B: T.Tensor((17, 57), "float16"),
        Bias: T.Tensor((29, 57), "float32"),
        C: T.Tensor((29, 57), "float32"),
    ):
        with T.Kernel(1, threads=threads) as _bx:
            sa = T.alloc_shared((32, 32), "float16")
            sb = T.alloc_shared((32, 64), "float16")
            shared_out = T.alloc_shared((32, 64), "float16")
            bias = T.alloc_fragment((32, 64), "float32")
            acc = T.alloc_fragment((32, 64), "float32")
            narrow = T.alloc_fragment((32, 64), "float16")
            T.copy(Bias, bias)
            T.copy(bias, acc)
            T.copy(A, sa)
            T.copy(B, sb)
            T.gemm(sa, sb, acc)
            for i, j in T.Parallel(32, 64):
                acc[i, j] *= 0.5
                narrow[i, j] = T.maximum(acc[i, j] + bias[i, j], 0.0)
            T.copy(narrow, shared_out)
            T.copy(shared_out, C)

    return ntilang.compile(kernel)


def inputs(m, n, k):
    rng = np.random.default_rng(61)
    a = rng.normal(size=(m, k)).astype(np.float16)
    b = rng.normal(size=(k, n)).astype(np.float16)
    bias = rng.normal(size=(m, n)).astype(np.float32)
    return a, b, bias, np.full((m, n), np.nan, dtype=np.float32)


def test_matmul_relu_reference():
    a, b, bias, c = inputs(65, 71, 37)
    reference(matmul_relu(), a, b, bias, c)
    expected = np.maximum((a.astype(np.float32) @ b.astype(np.float32)) * 0.5 + bias, 0)
    np.testing.assert_allclose(c, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("threads", [32, 64, 128, 256])
def test_mma_copy_chain_reference(threads):
    a, b, bias, c = inputs(29, 57, 17)
    reference(mma_copy_chain(threads), a, b, bias, c)
    expected = np.maximum((bias + a.astype(np.float32) @ b.astype(np.float32)) * 0.5 + bias, 0)
    np.testing.assert_allclose(c, expected.astype(np.float16).astype(np.float32), rtol=1e-3, atol=1e-3)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("threads", [32, 64, 128, 256])
def test_mma_copy_chain_cute_compilation(threads):
    assert mma_copy_chain(threads).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_matmul_relu_cute_compilation():
    assert matmul_relu().build().has_gpu_module
