import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def row_accumulation(dtype="float32", explicit=False):
    @T.prim_func
    def kernel(A: T.Tensor((39, 7), dtype), B: T.Tensor((39,), dtype)):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                acc = T.alloc_var(dtype)
                for k in T.unroll(start=0, stop=7, explicit=explicit):
                    acc += A[i, k]
                if i % 2 == 0:
                    acc = acc * 2
                B[i] = acc

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int8", "uint8", "int64"])
@pytest.mark.parametrize("explicit", [False, True])
def test_per_element_loop_carried_accumulator(dtype, explicit):
    np_dtype = np.dtype(dtype)
    a = (np.arange(39 * 7).reshape(39, 7) % 13).astype(np_dtype)
    b = np.empty(39, dtype=np_dtype)
    reference(row_accumulation(dtype, explicit), a, b)
    expected = np.zeros(39, dtype=np_dtype)
    for k in range(7):
        expected = (expected.astype(np.int64) + a[:, k]).astype(np_dtype)
    expected[::2] = (expected[::2].astype(np.int64) * 2).astype(np_dtype)
    np.testing.assert_array_equal(b, expected)


def uniform_accumulation():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            acc = T.alloc_var(dtype=T.int32, init=3)
            for k in T.serial(7, -2, -2):
                acc += k
            frozen = acc
            acc += 10
            for i in T.Parallel(32):
                B[i] = acc + frozen

    return ntilang.compile(kernel)


def test_uniform_accumulator_and_immutable_snapshot():
    out = np.empty(32, dtype=np.int32)
    reference(uniform_accumulation(), out)
    np.testing.assert_array_equal(out, (3 + sum(range(7, -2, -2))) * 2 + 10)


def constructor_forms():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                zero = T.alloc_var("float32", "local.var")
                one = T.alloc_var("float32", 1, "local.var")
                two = T.alloc_var("float32", init=2, scope="local.var")
                loaded = T.alloc_var("float32", A[i], scope="local.var")
                no_init = T.alloc_var("float32", init=None)
                B[i] = zero + one + two + loaded + no_init

    return ntilang.compile(kernel)


def test_allocation_initialization_forms():
    a = np.arange(32, dtype=np.float32)
    b = np.empty_like(a)
    reference(constructor_forms(), a, b)
    np.testing.assert_array_equal(b, a + 3)


def conditional_declaration():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                acc = T.alloc_var("int32")
                if i < 16:
                    acc = 2
                else:
                    acc = 5
                acc += i
                B[i] = acc

    return ntilang.compile(kernel)


def test_conditionally_updated_mutable_declaration():
    out = np.empty(32, dtype=np.int32)
    reference(conditional_declaration(), out)
    np.testing.assert_array_equal(out, np.where(np.arange(32) < 16, 2, 5) + np.arange(32))


def test_cross_parallel_scope_updates_require_state_mapping():
    @T.prim_func
    def bad(B: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            acc = T.alloc_var("int32")
            for i in T.Parallel(64):
                acc += 1
                B[i] = acc

    with pytest.raises(ntilang.CompileError, match="inside the parallel loop"):
        ntilang.compile(bad)


def test_mutable_output_ownership_is_not_inferred_from_initializer():
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                index = T.alloc_var("int32", 0)
                for k in T.serial(4):
                    index += k
                B[index] = A[i]

    with pytest.raises(ntilang.CompileError, match="affine indices"):
        ntilang.compile(bad)


def test_conditional_declaration_cannot_escape_its_frame():
    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                if i < 16:
                    acc = T.alloc_var("int32")
                else:
                    acc = 0
                B[i] = acc

    with pytest.raises(ntilang.CompileError, match="defining region"):
        ntilang.compile(bad)


def test_duplicate_initializer_rejected():
    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            acc = T.alloc_var("int32", 1, init=2)
            for i in T.Parallel(32):
                B[i] = acc

    with pytest.raises(ntilang.CompileError, match="multiple times"):
        ntilang.compile(bad)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int8", "uint8", "int64", "bfloat16"])
@pytest.mark.parametrize("explicit", [False, True])
def test_mutable_scalar_compilation(dtype, explicit):
    assert row_accumulation(dtype, explicit).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("factory", [uniform_accumulation, constructor_forms, conditional_declaration])
def test_mutable_scalar_scope_compilation(factory):
    assert factory().build().has_gpu_module
