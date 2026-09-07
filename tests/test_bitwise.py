import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ir import Expr
from ntilang.scalar import expression_dtype
from ntilang.testing import reference
from ntilang.validation import interval

INTEGER_DTYPES = [f"{sign}{bits}" for sign in ("int", "uint") for bits in (8, 16, 32, 64)]
OPERATIONS = {
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
    "shift_left": np.left_shift,
    "shift_right": np.right_shift,
}


def binary_kernel(name, dtype="int32"):
    operation = getattr(T, name)

    @T.prim_func
    def kernel(A: T.Tensor((39,), dtype), B: T.Tensor((39,), dtype), C: T.Tensor((39,), dtype)):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                C[bx * 32 + i] = operation(x=A[bx * 32 + i], y=B[bx * 32 + i], span=None)

    return ntilang.compile(kernel)


def unary_kernel(dtype):
    @T.prim_func
    def kernel(A: T.Tensor((39,), dtype), B: T.Tensor((39,), dtype)):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                B[bx * 32 + i] = T.bitwise_not(x=~A[bx * 32 + i])

    return ntilang.compile(kernel)


def syntax_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((64,), "int32"), B: T.Tensor((64,), "int32")):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                value = A[(bx << 5) + i]
                B[(bx << 5) + i] = ((value & 31) | 128) ^ ((value >> 3) << 1)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("name", OPERATIONS)
def test_integer_bit_patterns(dtype, name):
    a = np.arange(39).astype(dtype)
    if dtype.startswith("int"):
        a[::2] *= -1
    b = (np.arange(39) % 4).astype(dtype)
    c = np.empty_like(a)
    reference(binary_kernel(name, dtype), a, b, c)
    np.testing.assert_array_equal(c, OPERATIONS[name](a, b))


@pytest.mark.parametrize("dtype", ["bool", *INTEGER_DTYPES])
def test_double_inversion_preserves_width(dtype):
    a = np.arange(39).astype(dtype)
    b = np.empty_like(a)
    reference(unary_kernel(dtype), a, b)
    np.testing.assert_array_equal(b, a)


@pytest.mark.parametrize("name", ["bitwise_and", "bitwise_or", "bitwise_xor"])
def test_boolean_bitwise(name):
    a, b = (np.arange(39) % modulus == 0 for modulus in (2, 3))
    c = np.empty_like(a)
    reference(binary_kernel(name, "bool"), a, b, c)
    np.testing.assert_array_equal(c, OPERATIONS[name](a, b))


def test_bitwise_syntax_and_shifted_ownership():
    a = np.arange(-32, 32, dtype=np.int32)
    b = np.empty_like(a)
    reference(syntax_kernel(), a, b)
    np.testing.assert_array_equal(b, ((a & 31) | 128) ^ ((a >> 3) << 1))


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("int8", "int32", "int32"),
        ("uint32", "int32", "uint32"),
        ("int64", "uint32", "int64"),
        ("bool", "uint8", "uint8"),
    ],
)
def test_bitwise_type_join(left, right, expected):
    expression = Expr(
        "&", (Expr("cast", (Expr("const", value=1),), left), Expr("cast", (Expr("const", value=1),), right))
    )
    assert expression_dtype(expression, {}, {}) == expected


@pytest.mark.parametrize(
    "dtype,name", [("float32", "bitwise_and"), ("float64", "shift_right"), ("bool", "shift_left")]
)
def test_invalid_bitwise_type_rejected(dtype, name):
    with pytest.raises(ntilang.CompileError, match="require integer"):
        binary_kernel(name, dtype)


@pytest.mark.parametrize("shift", [-1, 32, 63])
def test_invalid_constant_shift_rejected(shift):
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = A[i] << shift

    with pytest.raises(ntilang.CompileError, match="Shift count"):
        ntilang.compile(bad)


def test_masked_indices_stay_bounded():
    @T.prim_func
    def kernel(A: T.Tensor((8,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = A[i & 7]

    a = np.arange(8, dtype=np.int32)
    b = np.empty(32, dtype=np.int32)
    reference(ntilang.compile(kernel), a, b)
    np.testing.assert_array_equal(b, np.tile(a, 4))


def test_unsigned_inversion_index_cannot_hide_overflow():
    value = Expr("invert", (Expr("cast", (Expr("const", value=0),), "uint32"),))
    with pytest.raises(ntilang.CompileError, match="overflow signed 32-bit"):
        interval(value, {}, {})


def test_unsigned_shift_uses_logical_right_shift_for_bounds():
    value = Expr(">>", (Expr("const", value=-1), Expr("cast", (Expr("const", value=1),), "uint32")))
    assert interval(value, {}, {}) == (2**31 - 1, 2**31 - 1)


def test_narrow_index_shift_cannot_wrap():
    value = Expr("<<", tuple(Expr("cast", (Expr("const", value=n),), "int8") for n in (127, 1)))
    with pytest.raises(ntilang.CompileError, match="overflow its integer dtype"):
        interval(value, {}, {})


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "uint64"])
@pytest.mark.parametrize("name", OPERATIONS)
def test_bitwise_compilation(dtype, name):
    assert binary_kernel(name, dtype).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_inversion_and_index_compilation():
    for dtype in ("bool", "int8", "uint64"):
        assert unary_kernel(dtype).build().has_gpu_module
    assert syntax_kernel().build().has_gpu_module
    for name in ("bitwise_and", "bitwise_or", "bitwise_xor"):
        assert binary_kernel(name, "bool").build().has_gpu_module
