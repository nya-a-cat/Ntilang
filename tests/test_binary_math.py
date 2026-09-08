import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference

FLOAT_TYPES = ["float16", "float32", "float64"]
INTEGER_TYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]


def binary_math(operation, left_dtype="float32", right_dtype="float32", output_dtype="float32"):
    intrinsic = getattr(T, operation)

    @T.prim_func
    def kernel(
        A: T.Tensor((39,), left_dtype), B: T.Tensor((39,), right_dtype), C: T.Tensor((39,), output_dtype)
    ):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                C[i] = intrinsic(A[i], B[i])

    return ntilang.compile(kernel)


@pytest.mark.parametrize("operation", ["pow", "fmod", "atan2", "copysign"])
@pytest.mark.parametrize("dtype", FLOAT_TYPES)
def test_binary_float_special_values(operation, dtype):
    if operation == "pow":
        left = [-2, -2, -2, -0.0, -0.0, 0, 1, np.nan, 2, 2, np.inf, np.inf, -np.inf]
        right = [3, 2, 0.5, 3, -3, 0, np.nan, 0, -2, 0.5, 0, -1, 3]
    else:
        left = [-5.5, 5.5, -5.5, -0.0, 0.0, np.inf, 2, np.nan, 2, -2, 1, -1, -np.inf]
        right = [2, -2, -2, 2, -2, 2, 0, 1, np.inf, -np.inf, np.nan, -0.0, np.inf]
    a = np.resize(np.array(left, dtype=dtype), 39)
    b = np.resize(np.array(right, dtype=dtype), 39)
    c = np.empty_like(a)
    oracle = {"pow": np.power, "fmod": np.fmod, "atan2": np.arctan2, "copysign": np.copysign}[operation]
    with np.errstate(all="ignore"):
        reference(binary_math(operation, dtype, dtype, dtype), a, b, c)
        expected = oracle(a.astype(np.float64), b.astype(np.float64)).astype(dtype)
    tolerance = 0.001 if dtype == "float16" else 2e-7 if dtype == "float32" else 1e-15
    np.testing.assert_allclose(c, expected, rtol=tolerance, atol=0)
    np.testing.assert_array_equal(np.signbit(c[c == 0]), np.signbit(expected[expected == 0]))


@pytest.mark.parametrize("operation", ["fmod", "atan2", "copysign"])
def test_binary_intrinsics_convert_to_the_first_argument_dtype(operation):
    a = np.resize(np.array([2.5, -2.5, 0.0, -0.0], dtype=np.float32), 39)
    b = np.resize(np.array([2.00000001, -2.00000001, -1e-300, 1e300], dtype=np.float64), 39)
    c = np.empty_like(a)
    with np.errstate(all="ignore"):
        reference(binary_math(operation, "float32", "float64"), a, b, c)
        oracle = {"fmod": np.fmod, "atan2": np.arctan2, "copysign": np.copysign}[operation]
        expected = oracle(a, b.astype(np.float32))
    np.testing.assert_allclose(c, expected, rtol=2e-7, atol=0)
    np.testing.assert_array_equal(np.signbit(c[c == 0]), np.signbit(expected[expected == 0]))


def constant_power(dtype, exponent):
    @T.prim_func
    def kernel(A: T.Tensor((39,), dtype), B: T.Tensor((39,), dtype)):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                n = T.int32(exponent)
                alias = n + 0
                B[i] = T.pow(x=A[i], y=alias, span=None)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", INTEGER_TYPES + FLOAT_TYPES)
@pytest.mark.parametrize("exponent", [0, 1, 3])
def test_constant_integer_power_preserves_base_dtype(dtype, exponent):
    a = (np.arange(39) % 4).astype(dtype)
    if dtype.startswith(("int", "float")):
        a -= np.array(2, dtype=dtype)
    b = np.empty_like(a)
    reference(constant_power(dtype, exponent), a, b)
    expected = np.array([int(x) ** exponent for x in a], dtype=dtype)
    np.testing.assert_array_equal(b, expected)


def test_half_constant_power_follows_sequential_rounding():
    a = np.linspace(0.25, 1.75, 39, dtype=np.float16)
    b = np.empty_like(a)
    reference(constant_power("float16", 7), a, b)
    expected = a.copy()
    for _ in range(6):
        expected = np.multiply(expected, a, dtype=np.float16)
    np.testing.assert_array_equal(b, expected)


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
def test_negative_constant_integer_exponent_uses_floating_power(dtype):
    a = np.arange(1, 40, dtype=dtype)
    b = np.empty_like(a)
    reference(constant_power(dtype, -2), a, b)
    expected = (1 / a.astype(np.float64) ** 2).astype(dtype)
    np.testing.assert_allclose(b, expected, rtol=0.001 if dtype == "float16" else 2e-7)


def test_power_promotes_both_operands():
    a = np.arange(1, 40, dtype=np.int32)
    b = np.full(39, 0.5, dtype=np.float64)
    c = np.empty(39, dtype=np.float64)
    reference(binary_math("pow", "int32", "float64", "float64"), a, b, c)
    np.testing.assert_allclose(c, np.sqrt(a.astype(np.float64)), rtol=2e-15)


def mutable_exponent():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "float32"), B: T.Tensor((39,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                n = T.alloc_var("int32", 3)
                n = -2
                B[i] = T.pow(A[i], n)

    return ntilang.compile(kernel)


def test_power_uses_current_mutable_exponent():
    a = np.arange(1, 40, dtype=np.float32)
    b = np.empty_like(a)
    reference(mutable_exponent(), a, b)
    np.testing.assert_allclose(b, 1 / (a * a), rtol=2e-7)


@pytest.mark.parametrize("dtype", ["int32", "bfloat16"])
def test_dynamic_power_rejects_non_float_promoted_dtype(dtype):
    with pytest.raises(ntilang.CompileError, match="floating result"):
        binary_math("pow", dtype, dtype, dtype)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "operation,left,right,output",
    [(op, dtype, dtype, dtype) for op in ["pow", "fmod", "atan2", "copysign"] for dtype in FLOAT_TYPES]
    + [(op, "bfloat16", "bfloat16", "bfloat16") for op in ["fmod", "atan2", "copysign"]]
    + [(op, "float32", "float64", "float32") for op in ["fmod", "atan2", "copysign"]]
    + [("pow", "float16", "float32", "float32"), ("pow", "float32", "int64", "float32")],
)
def test_binary_math_native_compilation(operation, left, right, output):
    assert binary_math(operation, left, right, output).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("dtype", INTEGER_TYPES + FLOAT_TYPES + ["bfloat16"])
@pytest.mark.parametrize("exponent", [0, 3])
def test_constant_power_native_compilation(dtype, exponent):
    assert constant_power(dtype, exponent).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_mutable_power_native_compilation():
    assert mutable_exponent().build().has_gpu_module
