import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.scalar import promote
from ntilang.testing import reference


def mixed_arithmetic(float_dtype="float16", integer_dtype="int64"):
    @T.prim_func
    def kernel(
        A: T.Tensor((33,), float_dtype),
        Integers: T.Tensor((33,), integer_dtype),
        Output: T.Tensor((33,), "float64"),
    ):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                value = A[bx * 32 + i] + Integers[bx * 32 + i]
                Output[bx * 32 + i] = T.float64(value)

    return ntilang.compile(kernel)


def mixed_comparison():
    @T.prim_func
    def kernel(
        A: T.Tensor((33,), "float16"), Integers: T.Tensor((33,), "int64"), Output: T.Tensor((33,), "bool")
    ):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                Output[bx * 32 + i] = A[bx * 32 + i] == Integers[bx * 32 + i]

    return ntilang.compile(kernel)


def mixed_conditional_value():
    @T.prim_func
    def kernel(
        A: T.Tensor((33,), "float16"), Integers: T.Tensor((33,), "int64"), Output: T.Tensor((33,), "float64")
    ):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                value = T.if_then_else(i % 2 == 0, A[bx * 32 + i], Integers[bx * 32 + i])
                Output[bx * 32 + i] = T.float64(value)

    return ntilang.compile(kernel)


def contextual_shift(dtype="int8", amount=1):
    @T.prim_func
    def kernel(A: T.Tensor((33,), dtype), Output: T.Tensor((33,), "int64")):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                Output[bx * 32 + i] = A[bx * 32 + i] << amount

    return ntilang.compile(kernel)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("float16", "int64", "float16"),
        ("uint64", "float16", "float16"),
        ("bfloat16", "int32", "bfloat16"),
        ("int64", "float32", "float32"),
        ("float32", "float64", "float64"),
        ("bfloat16", "float16", "float16"),
        ("int32", "uint32", "uint32"),
        ("uint8", "int16", "int16"),
    ],
)
def test_pinned_tir_type_matching(left, right, expected):
    assert promote(left, right) == expected
    assert promote(right, left) == expected


def test_mixed_arithmetic_rounds_in_the_source_float_dtype():
    a = np.ones(33, dtype=np.float16)
    i = np.full(33, 2049, dtype=np.int64)
    out = np.empty(33, dtype=np.float64)
    reference(mixed_arithmetic(), a, i, out)
    np.testing.assert_array_equal(out, np.full(33, 2048.0))


def test_mixed_comparison_converts_before_comparing():
    a = np.full(33, 2048, dtype=np.float16)
    i = np.full(33, 2049, dtype=np.int64)
    out = np.empty(33, dtype=np.bool_)
    reference(mixed_comparison(), a, i, out)
    np.testing.assert_array_equal(out, np.ones(33, dtype=np.bool_))


def test_conditional_value_preserves_the_selected_float_type():
    a = np.full(33, 2048, dtype=np.float16)
    i = np.full(33, 2049, dtype=np.int64)
    out = np.empty(33, dtype=np.float64)
    reference(mixed_conditional_value(), a, i, out)
    np.testing.assert_array_equal(out, np.full(33, 2048.0))


def test_integer_bitwise_literal_preserves_narrow_operand_width():
    a = np.full(33, 64, dtype=np.int8)
    out = np.empty(33, dtype=np.int64)
    reference(contextual_shift(), a, out)
    np.testing.assert_array_equal(out, np.full(33, -128, dtype=np.int64))


@pytest.mark.parametrize("amount", [8, 15])
def test_shift_literal_is_checked_against_contextual_width(amount):
    with pytest.raises(ntilang.CompileError, match="Shift count"):
        contextual_shift(amount=amount)


def test_bitwise_literal_must_fit_contextual_dtype():
    with pytest.raises(ntilang.CompileError, match="literal must fit"):
        contextual_shift(amount=128)


def test_integer_true_division_requires_explicit_conversion():
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), Output: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                Output[i] = A[i] / 2

    with pytest.raises(ntilang.CompileError, match="Integer '/' is ambiguous"):
        ntilang.compile(bad)


def test_logical_operation_requires_boolean_operand():
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), Output: T.Tensor((32,), "bool")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                Output[i] = A[i] and True

    with pytest.raises(ntilang.CompileError, match="Logical operations require Boolean"):
        ntilang.compile(bad)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "factory", [mixed_arithmetic, mixed_comparison, mixed_conditional_value, contextual_shift]
)
def test_scalar_semantic_compilation(factory):
    assert factory().build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_mixed_bfloat16_compilation():
    assert mixed_arithmetic("bfloat16", "int32").build().has_gpu_module
