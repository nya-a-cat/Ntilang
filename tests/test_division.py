import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ir import Expr
from ntilang.testing import reference
from ntilang.validation import interval

KINDS = ("floordiv", "floormod", "truncdiv", "truncmod", "ceildiv")


def division_kernel(kind, dtype="int32"):
    operation = getattr(T, kind)

    @T.prim_func
    def kernel(A: T.Tensor((33,), dtype), B: T.Tensor((33,), dtype), C: T.Tensor((33,), dtype)):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                if bx * 32 + i < 33:
                    C[bx * 32 + i] = operation(A[bx * 32 + i], B[bx * 32 + i])

    return ntilang.compile(kernel)


def constant_remainders():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int64"), B: T.Tensor((32,), "int64")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                A[i] = T.truncmod(a=-7, b=3, span=None)
                B[i] = T.floormod(-7, 3)

    return ntilang.compile(kernel)


def signed_indices():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                source = (i - 16) // 4 + 4
                residual = (i - 16) % 8
                B[T.floordiv(i * -2, -2)] = A[source] + A[residual] + A[i // (i % 3 + 1)]

    return ntilang.compile(kernel)


def aligned_indices():
    round_up = T.align_up
    ceiling = T.cdiv

    @T.prim_func
    def kernel(A: T.Tensor((33,), "int32"), B: T.Tensor((33,), "int32")):
        with T.Kernel(1, threads=64) as _bx:
            for i in T.Parallel(33):
                B[i] = A[round_up(i, 4)] + ceiling(lhs=i, rhs=4, span=None)

    return ntilang.compile(kernel)


def expected_integer(kind, a, b):
    if kind in ("floordiv", "floormod"):
        return a // b if kind == "floordiv" else a % b
    if kind == "ceildiv":
        return (a + b - 1) // b
    quotient = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        quotient = -quotient
    return quotient if kind == "truncdiv" else a - quotient * b


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "int32", "int64", "uint64"])
def test_division_signs_and_integer_precision(kind, dtype):
    a = np.arange(1, 34, dtype=np.int64)
    b = np.resize(np.array([1, 2, 3, 7], dtype=np.int64), 33)
    if dtype.startswith("int"):
        a[::2] *= -1
        b[1::3] *= -1
    if dtype == "int64":
        a *= 2**53 + 1
    if dtype == "uint64":
        a = a.astype(np.uint64) + np.uint64(2**63)
    a, b = a.astype(dtype), b.astype(dtype)
    out = np.empty_like(a)
    reference(division_kernel(kind, dtype), a, b, out)
    expected = np.array([expected_integer(kind, int(x), int(y)) for x, y in zip(a, b)], dtype=dtype)
    np.testing.assert_array_equal(out, expected)


def test_constant_truncating_remainder_matches_dynamic_semantics():
    a, b = np.empty(32, dtype=np.int64), np.empty(32, dtype=np.int64)
    reference(constant_remainders(), a, b)
    np.testing.assert_array_equal(a, -np.ones(32, dtype=np.int64))
    np.testing.assert_array_equal(b, np.full(32, 2, dtype=np.int64))


def test_signed_and_variable_divisor_indices():
    a, out = np.arange(32, dtype=np.int32), np.empty(32, dtype=np.int32)
    reference(signed_indices(), a, out)
    i = np.arange(32)
    np.testing.assert_array_equal(out, a[(i - 16) // 4 + 4] + a[(i - 16) % 8] + a[i // (i % 3 + 1)])


def test_expression_ceildiv_and_alignment():
    a, out = np.arange(33, dtype=np.int32), np.empty(33, dtype=np.int32)
    reference(aligned_indices(), a, out)
    i = np.arange(33)
    np.testing.assert_array_equal(out, a[((i + 3) // 4) * 4] + (i + 3) // 4)
    assert T.ceildiv(7, -3) == -1
    assert T.cdiv(lhs=7, rhs=3) == 3
    assert T.align_up(7, 4) == 8


@pytest.mark.parametrize("kind", KINDS)
def test_zero_divisor_is_diagnosed(kind):
    operation = getattr(T, kind)

    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = operation(A[i], 0)

    with pytest.raises(ntilang.CompileError, match="nonzero divisor"):
        ntilang.compile(bad)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", ["float32", "bool"])
def test_integer_division_requires_integer_types(kind, dtype):
    with pytest.raises(ntilang.CompileError, match="require integer operands"):
        division_kernel(kind, dtype)


def test_signed_division_overflow_is_diagnosed():
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                A[i] = T.truncdiv(-2147483648, -1)

    with pytest.raises(ntilang.CompileError, match="overflow"):
        ntilang.compile(bad)


def test_ceildiv_index_numerator_must_fit_its_dtype():
    value = Expr("ceildiv", (Expr("const", value=2**31 - 1), Expr("const", value=2)))
    with pytest.raises(ntilang.CompileError, match="numerator can overflow"):
        interval(value, {}, {})


def test_narrow_index_arithmetic_cannot_wrap():
    value = Expr("+", tuple(Expr("cast", (Expr("const", value=x),), "int8") for x in (127, 1)))
    with pytest.raises(ntilang.CompileError, match="overflow its integer dtype"):
        interval(value, {}, {})


@pytest.mark.parametrize(
    "kind,op", [("floordiv", "//"), ("floormod", "%"), ("truncdiv", "truncdiv"), ("truncmod", "truncmod")]
)
@pytest.mark.parametrize("a_bounds", [(-7, -3), (-4, 5), (0, 9)])
@pytest.mark.parametrize("b_bounds", [(-5, -1), (1, 4), (2, 2)])
def test_integer_division_bounds_contain_the_full_domain(kind, op, a_bounds, b_bounds):
    value = Expr(op, (Expr("var", value="a"), Expr("var", value="b")))
    low, high = interval(value, {"a": a_bounds, "b": b_bounds}, {})
    for a in range(a_bounds[0], a_bounds[1] + 1):
        for b in range(b_bounds[0], b_bounds[1] + 1):
            assert low <= expected_integer(kind, a, b) <= high


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", ["int8", "int64", "uint64"])
def test_division_compilation(kind, dtype):
    assert division_kernel(kind, dtype).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("factory", [constant_remainders, signed_indices, aligned_indices])
def test_division_index_and_constant_compilation(factory):
    assert factory().build().has_gpu_module
