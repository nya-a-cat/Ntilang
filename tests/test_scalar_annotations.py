import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.codegen import walk
from ntilang.testing import reference


def annotated_values(dtype=T.int32):
    @T.prim_func
    def kernel(A: T.Tensor((39,), "float32"), B: T.Tensor((39,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            base: T.int32 = bx * 32
            for i in T.Parallel(32):
                value: dtype = A[base + i] + 0.5
                value: T.float64
                B[base + i] = value

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", [T.int8, T.float16, T.int32, T.float64])
def test_eager_annotations_preserve_value_type(dtype):
    kernel = annotated_values(dtype)
    a = np.linspace(-3, 2, 39, dtype=np.float32)
    b = np.empty_like(a)
    reference(kernel, a, b)
    np.testing.assert_array_equal(b, a + np.float32(0.5))
    binding = next(stmt for stmt in walk(kernel.ir.body) if stmt.op == "let" and stmt.args[1].op == "+")
    assert dict(binding.annotations)["scalar_annotation"] == dtype


def annotated_branch():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                if i < 16:
                    value: T.int32 = A[i] + 0.5
                    B[i] = value
                else:
                    value: T.float64 = A[i] * 2.5
                    B[i] = value

    return ntilang.compile(kernel)


def test_annotated_branch_local_values():
    a = np.arange(32, dtype=np.float32)
    b = np.empty_like(a)
    reference(annotated_branch(), a, b)
    np.testing.assert_array_equal(b, np.where(np.arange(32) < 16, a + 0.5, a * 2.5))


def test_annotation_does_not_initialize_a_value():
    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                _value: T.int32
                B[i] = i

    with pytest.raises(ntilang.CompileError, match="does not initialize"):
        ntilang.compile(bad)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("factory", [annotated_values, annotated_branch])
def test_annotated_scalars_compile(factory):
    assert factory().build().has_gpu_module
