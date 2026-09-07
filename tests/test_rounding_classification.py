import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference

OPERATIONS = [
    "abs",
    "floor",
    "ceil",
    "trunc",
    "round",
    "round_away",
    "nearbyint",
    "isnan",
    "isinf",
    "isfinite",
]
DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
]
CLASSIFICATION = {"isnan", "isinf", "isfinite"}


def unary_kernel(operation, dtype):
    intrinsic = getattr(T, "round" if operation == "round_away" else operation)
    output_dtype = "bool" if operation in CLASSIFICATION else dtype
    if operation == "round_away":

        @T.prim_func
        def kernel(A: T.Tensor((13,), dtype), B: T.Tensor((13,), output_dtype)):
            with T.Kernel(1, threads=32) as _bx:
                for i in T.Parallel(32):
                    B[i] = intrinsic(x=A[i], rounding_mode="ties-away-from-zero", span=None)

    else:

        @T.prim_func
        def kernel(A: T.Tensor((13,), dtype), B: T.Tensor((13,), output_dtype)):
            with T.Kernel(1, threads=32) as _bx:
                for i in T.Parallel(32):
                    B[i] = intrinsic(x=A[i], span=None)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rounding_and_classification_values(operation, dtype):
    if dtype.startswith("float"):
        a = np.array(
            [-np.inf, -2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, np.inf, np.nan, 0.25, -0.25], dtype=dtype
        )
        values = {
            "abs": [np.inf, 2.5, 1.5, 0.5, 0.0, 0.0, 0.5, 1.5, 2.5, np.inf, np.nan, 0.25, 0.25],
            "floor": [-np.inf, -3, -2, -1, -0.0, 0.0, 0, 1, 2, np.inf, np.nan, 0, -1],
            "ceil": [-np.inf, -2, -1, -0.0, -0.0, 0.0, 1, 2, 3, np.inf, np.nan, 1, -0.0],
            "trunc": [-np.inf, -2, -1, -0.0, -0.0, 0.0, 0, 1, 2, np.inf, np.nan, 0, -0.0],
            "round": [-np.inf, -2, -2, -0.0, -0.0, 0.0, 0, 2, 2, np.inf, np.nan, 0, -0.0],
            "round_away": [-np.inf, -3, -2, -1, -0.0, 0.0, 1, 2, 3, np.inf, np.nan, 0, -0.0],
            "isnan": [False] * 10 + [True, False, False],
            "isinf": [True] + [False] * 8 + [True, False, False, False],
            "isfinite": [False] + [True] * 8 + [False, False, True, True],
        }
        expected = np.array(
            values["round" if operation == "nearbyint" else operation],
            dtype="bool" if operation in CLASSIFICATION else dtype,
        )
    else:
        if dtype == "bool":
            a = np.array([False, True] * 6 + [False])
        else:
            limits = np.iinfo(dtype)
            samples = [limits.min, limits.max, 0, 1, 2, 3, 4]
            samples += [-1, -2, -3, -4, -5, -6] if dtype.startswith("int") else [5, 6, 7, 8, 9, 10]
            a = np.array(samples, dtype=dtype)
        if operation in CLASSIFICATION:
            expected = np.full(13, operation == "isfinite", dtype="bool")
        elif operation == "abs" and dtype.startswith("int"):
            expected = np.array(
                [int(x) if int(x) == np.iinfo(dtype).min else abs(int(x)) for x in a], dtype=dtype
            )
        else:
            expected = a.copy()
    b = np.empty_like(expected)
    reference(unary_kernel(operation, dtype), a, b)
    np.testing.assert_array_equal(b, expected)
    if dtype.startswith("float") and operation not in CLASSIFICATION:
        np.testing.assert_array_equal(np.signbit(b[b == 0]), np.signbit(expected[expected == 0]))


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_rounding_around_halfway_values(dtype):
    half = np.array(0.5, dtype=dtype)
    below = np.nextafter(half, np.array(0, dtype=dtype))
    above = np.nextafter(half, np.array(1, dtype=dtype))
    a = np.array([below, half, above, -below, -half, -above, 0, -0.0, 1, -1, 2, -2, 3], dtype=dtype)
    expected = np.array([0, 1, 1, -0.0, -1, -1, 0, -0.0, 1, -1, 2, -2, 3], dtype=dtype)
    b = np.empty_like(a)
    reference(unary_kernel("round_away", dtype), a, b)
    np.testing.assert_array_equal(b, expected)
    np.testing.assert_array_equal(np.signbit(b), np.signbit(expected))


@pytest.mark.parametrize("operation", sorted(CLASSIFICATION))
def test_classification_bfloat16_matches_upstream_diagnostic(operation):
    with pytest.raises(ntilang.CompileError, match="classification.*bfloat16"):
        unary_kernel(operation, "bfloat16")


def test_round_invalid_mode():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = T.round(A[i], "toward-zero")

    with pytest.raises(ntilang.CompileError, match="rounding_mode"):
        ntilang.compile(kernel)


def finite_abs_gather():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), Index: T.Tensor((13,), "int32"), B: T.Tensor((13,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                value = A[T.abs(Index[i])]
                if T.isfinite(value):
                    B[i] = T.round(value, None, None)
                else:
                    B[i] = -1

    return ntilang.compile(kernel)


def test_classification_control_flow_and_signed_absolute_index():
    a = np.arange(32, dtype=np.float32) + 0.5
    a[:3] = [np.inf, -np.inf, np.nan]
    index = np.array([-(2**31), -31, -20, -10, -2, -1, 0, 1, 2, 3, 20, 31, 2**31 - 1], dtype=np.int32)
    b = np.empty(13, dtype=np.float32)
    reference(finite_abs_gather(), a, index, b)
    np.testing.assert_array_equal(b, [0, 32, 20, 10, -1, -1, -1, -1, -1, 4, 20, 32, 0])


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_classification_control_flow_native_compilation():
    assert finite_abs_gather().build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "operation,dtype",
    [
        (op, dt)
        for op in OPERATIONS
        for dt in DTYPES + ["bfloat16"]
        if dt != "bfloat16" or op not in CLASSIFICATION
    ],
)
def test_rounding_and_classification_native_compilation(operation, dtype):
    assert unary_kernel(operation, dtype).build().has_gpu_module
