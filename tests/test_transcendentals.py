import importlib.util
import math

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference

FUNCTIONS = [
    "exp",
    "exp2",
    "exp10",
    "log",
    "log2",
    "log10",
    "log1p",
    "sqrt",
    "rsqrt",
    "erf",
    "sigmoid",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
]
INTEGER_TYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]


def math_kernel(operation, dtype):
    intrinsic = getattr(T, operation)
    output_dtype = "float32" if operation == "exp" and dtype in INTEGER_TYPES else dtype

    @T.prim_func
    def kernel(A: T.Tensor((39,), dtype), B: T.Tensor((39,), output_dtype)):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                B[i] = intrinsic(x=A[i])

    return ntilang.compile(kernel)


@pytest.mark.parametrize("operation", FUNCTIONS)
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_transcendental_finite_values_against_scalar_math(operation, dtype):
    if operation in ("asin", "acos", "atanh", "log1p"):
        lo, hi = -0.875, 0.875
    elif operation == "acosh":
        lo, hi = 1, 4
    elif operation in ("log", "log2", "log10", "sqrt", "rsqrt"):
        lo, hi = 0.125, 4
    else:
        lo, hi = -2, 2
    a = np.linspace(lo, hi, 39, dtype=dtype)
    b = np.empty_like(a)
    oracle = {
        "exp2": lambda x: 2.0**x,
        "exp10": lambda x: 10.0**x,
        "rsqrt": lambda x: 1.0 / math.sqrt(x),
        "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
    }.get(operation, getattr(math, operation, None))
    expected = np.array([oracle(float(x)) for x in a], dtype=dtype)
    with np.errstate(all="ignore"):
        reference(math_kernel(operation, dtype), a, b)
    tolerance = 0.002 if dtype == "float16" else 4e-7 if dtype == "float32" else 4e-15
    np.testing.assert_allclose(b, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", INTEGER_TYPES)
def test_integer_exp_promotes_before_evaluation(dtype):
    values = np.arange(39) % 8
    if dtype.startswith("int"):
        values -= 4
    a = values.astype(dtype)
    b = np.empty(39, dtype=np.float32)
    reference(math_kernel("exp", dtype), a, b)
    np.testing.assert_allclose(b, np.exp(a.astype(np.float32)), rtol=1e-7)


def test_half_sigmoid_preserves_intermediate_rounding_and_overflow():
    a = np.resize(np.array([-12, 0, 12], dtype=np.float16), 39)
    b = np.empty_like(a)
    with np.errstate(all="ignore"):
        reference(math_kernel("sigmoid", "float16"), a, b)
    np.testing.assert_array_equal(b, np.resize(np.array([0, 0.5, 1], dtype=np.float16), 39))


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("log", [np.nan, -np.inf, np.inf, np.nan]),
        ("sqrt", [np.nan, -0.0, np.inf, np.nan]),
        ("rsqrt", [np.nan, -np.inf, 0, np.nan]),
        ("exp", [0, 1, np.inf, np.nan]),
        ("sigmoid", [0, 0.5, 1, np.nan]),
        ("tanh", [-1, -0.0, 1, np.nan]),
    ],
)
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_transcendental_special_values(operation, expected, dtype):
    a = np.resize(np.array([-np.inf, -0.0, np.inf, np.nan], dtype=dtype), 39)
    b = np.empty_like(a)
    target = np.resize(np.array(expected, dtype=dtype), 39)
    with np.errstate(all="ignore"):
        reference(math_kernel(operation, dtype), a, b)
    np.testing.assert_array_equal(b, target)
    np.testing.assert_array_equal(np.signbit(b[b == 0]), np.signbit(target[target == 0]))


@pytest.mark.parametrize(
    "operation",
    ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"],
)
def test_trigonometric_integer_inputs_match_upstream_rejection(operation):
    with pytest.raises(ntilang.CompileError, match="floating inputs"):
        math_kernel(operation, "int32")


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "operation,dtype",
    [(op, dt) for op in FUNCTIONS for dt in ("float16", "bfloat16", "float32", "float64")]
    + [("exp", dt) for dt in INTEGER_TYPES],
)
def test_transcendental_native_compilation(operation, dtype):
    assert math_kernel(operation, dtype).build().has_gpu_module
