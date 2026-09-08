import importlib.util
import math
import subprocess
import sys
from fractions import Fraction

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.floating import FORMATS, evaluate, round_exact
from ntilang.scalar import IEEE_MATH_OPS
from ntilang.testing import reference


def math_kernel(operation, dtype="float32", mode="rn", other_dtype=None, target="sm_80"):
    intrinsic = getattr(T, operation)
    other_dtype = other_dtype or dtype
    arity = IEEE_MATH_OPS[operation][1]

    if operation == "ieee_frsqrt":

        @T.prim_func
        def kernel(
            A: T.Tensor(9, dtype), B: T.Tensor(9, other_dtype), D: T.Tensor(9, dtype), C: T.Tensor(9, dtype)
        ):
            with T.Kernel(1, threads=32) as _bx:
                for i in T.Parallel(32):
                    C[i] = intrinsic(x=A[i])

    elif operation == "fma":

        @T.prim_func
        def kernel(
            A: T.Tensor(9, dtype), B: T.Tensor(9, other_dtype), D: T.Tensor(9, dtype), C: T.Tensor(9, dtype)
        ):
            with T.Kernel(1, threads=32) as _bx:
                for i in T.Parallel(32):
                    C[i] = intrinsic(z=D[i], x=A[i], y=B[i])

    elif operation == "fmul":

        @T.prim_func
        def kernel(
            A: T.Tensor(9, dtype), B: T.Tensor(9, other_dtype), D: T.Tensor(9, dtype), C: T.Tensor(9, dtype)
        ):
            with T.Kernel(1, threads=32) as _bx:
                for i in T.Parallel(32):
                    C[i] = intrinsic(y=B[i], x=A[i])

    elif arity == 1:

        @T.prim_func
        def kernel(
            A: T.Tensor(9, dtype), B: T.Tensor(9, other_dtype), D: T.Tensor(9, dtype), C: T.Tensor(9, dtype)
        ):
            with T.Kernel(1, threads=32) as _bx:
                for i in T.Parallel(32):
                    C[i] = intrinsic(rounding_mode=mode, x=A[i])

    elif arity == 2:

        @T.prim_func
        def kernel(
            A: T.Tensor(9, dtype), B: T.Tensor(9, other_dtype), D: T.Tensor(9, dtype), C: T.Tensor(9, dtype)
        ):
            with T.Kernel(1, threads=32) as _bx:
                for i in T.Parallel(32):
                    C[i] = intrinsic(y=B[i], rounding_mode=mode, x=A[i])

    else:

        @T.prim_func
        def kernel(
            A: T.Tensor(9, dtype), B: T.Tensor(9, other_dtype), D: T.Tensor(9, dtype), C: T.Tensor(9, dtype)
        ):
            with T.Kernel(1, threads=32) as _bx:
                for i in T.Parallel(32):
                    C[i] = intrinsic(x=A[i], y=B[i], z=D[i], rounding_mode=mode)

    return ntilang.compile(kernel, target=target)


def run_math(operation, dtype, mode, values, other_dtype=None):
    arrays = [
        np.full(9, x, dtype=other_dtype if i == 1 and other_dtype else dtype) for i, x in enumerate(values)
    ]
    while len(arrays) < 3:
        arrays.append(np.zeros(9, dtype=dtype))
    output = np.empty(9, dtype=dtype)
    reference(math_kernel(operation, dtype, mode, other_dtype), *arrays, output)
    return output[0]


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("operation", ["ieee_fmaf", "fma"])
def test_fused_cancellation_retains_the_exact_product_residual(dtype, operation):
    epsilon = float(np.finfo(dtype).eps)
    assert run_math(operation, dtype, "rn", (1 + epsilon, 1 - epsilon, -1)) == -(epsilon**2)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["rn", "rz", "ru", "rd"])
def test_explicit_addition_rounds_positive_and_negative_midpoints(dtype, mode):
    half_ulp = float(np.finfo(dtype).eps) / 2
    above = np.nextafter(np.dtype(dtype).type(1), np.dtype(dtype).type(2))
    assert run_math("ieee_add", dtype, mode, (1, half_ulp)) == (above if mode == "ru" else 1)
    assert run_math("ieee_add", dtype, mode, (-1, -half_ulp)) == (-above if mode == "rd" else -1)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["rn", "rz", "ru", "rd"])
def test_multiply_keeps_the_requested_rounding_boundary(dtype, mode):
    epsilon = float(np.finfo(dtype).eps)
    below = np.nextafter(np.dtype(dtype).type(1), np.dtype(dtype).type(0))
    expected = below if mode in ("rz", "rd") else 1
    assert run_math("ieee_mul", dtype, mode, (1 + epsilon, 1 - epsilon)) == expected


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["rn", "rz", "ru", "rd"])
@pytest.mark.parametrize(
    "operation,values", [("ieee_fdiv", (1, 3)), ("ieee_frcp", (3,)), ("ieee_fsqrt", (2,))]
)
def test_division_and_roots_round_to_the_adjacent_representable_value(dtype, mode, operation, values):
    typ = np.dtype(dtype).type
    nearest = np.sqrt(typ(2)) if operation == "ieee_fsqrt" else typ(1) / typ(3)
    exact = Fraction(2) if operation == "ieee_fsqrt" else Fraction(1, 3)
    candidate = Fraction(float(nearest)) ** (2 if operation == "ieee_fsqrt" else 1)
    expected = nearest
    if mode == "ru" and candidate < exact:
        expected = np.nextafter(nearest, typ(np.inf))
    elif mode in ("rd", "rz") and candidate > exact:
        expected = np.nextafter(nearest, typ(0))
    assert run_math(operation, dtype, mode, values) == expected


@pytest.mark.parametrize("dtype", FORMATS)
@pytest.mark.parametrize("mode", ["rn", "rz", "ru", "rd"])
def test_oracle_subnormal_ties_overflow_and_signed_zero(dtype, mode):
    precision, minimum, maximum = FORMATS[dtype]
    tiny = math.ldexp(1.0, minimum - precision + 1)
    largest = math.ldexp((1 << precision) - 1, maximum - precision + 1)
    assert evaluate("mul", (tiny, 0.5), dtype, mode) == (tiny if mode == "ru" else 0)
    negative = evaluate("mul", (-tiny, 0.5), dtype, mode)
    assert negative == (-tiny if mode == "rd" else -0.0) and math.copysign(1, negative) < 0
    assert evaluate("mul", (tiny, 1.5), dtype, mode) == tiny * (2 if mode in ("rn", "ru") else 1)
    assert evaluate("mul", (largest, 2), dtype, mode) == (math.inf if mode in ("rn", "ru") else largest)
    assert evaluate("mul", (-largest, 2), dtype, mode) == (-math.inf if mode in ("rn", "rd") else -largest)
    for operation, args in [("add", (1, -1)), ("sub", (0, 0)), ("fma", (1, 1, -1))]:
        assert math.copysign(1, evaluate(operation, args, dtype, mode)) == (-1 if mode == "rd" else 1)
    for operation, args in [("add", (-0.0, -0.0)), ("sub", (-0.0, 0.0)), ("fma", (-0.0, 1, -0.0))]:
        assert math.copysign(1, evaluate(operation, args, dtype, mode)) == -1


@pytest.mark.parametrize("dtype", FORMATS)
def test_oracle_fma_does_not_overflow_the_intermediate_product(dtype):
    precision, _, maximum = FORMATS[dtype]
    largest = math.ldexp((1 << precision) - 1, maximum - precision + 1)
    assert evaluate("fma", (largest, 2, -largest), dtype) == largest


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_nearest_oracle_matches_numpy_over_random_bits(dtype):
    rng = np.random.default_rng(4789)
    typ = np.dtype(dtype)
    unsigned = np.dtype(f"uint{typ.itemsize * 8}")
    x = rng.integers(0, np.iinfo(unsigned).max, size=256, dtype=unsigned).view(typ)
    y = rng.integers(0, np.iinfo(unsigned).max, size=256, dtype=unsigned).view(typ)
    with np.errstate(all="ignore"):
        for operation, function in [("add", np.add), ("mul", np.multiply), ("div", np.divide)]:
            actual = np.array([evaluate(operation, pair, dtype) for pair in zip(x, y)], dtype=typ)
            expected = function(x, y)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                np.signbit(actual[actual == 0]), np.signbit(expected[expected == 0])
            )
        np.testing.assert_array_equal([evaluate("sqrt", (float(a),), dtype) for a in x], np.sqrt(x))


@pytest.mark.parametrize(
    "operation,values,expected",
    [
        ("fma", (math.inf, 0, 1), math.nan),
        ("fma", (math.inf, 2, -math.inf), math.nan),
        ("fma", (math.inf, 2, math.inf), math.inf),
        ("div", (0, 0), math.nan),
        ("div", (1, -0.0), -math.inf),
        ("sqrt", (-1,), math.nan),
        ("sqrt", (-0.0,), -0.0),
        ("rsqrt", (-0.0,), -math.inf),
        ("rsqrt", (math.inf,), 0.0),
    ],
)
def test_special_values(operation, values, expected):
    result = evaluate(operation, values, "float32")
    assert math.isnan(result) if math.isnan(expected) else result == expected
    if expected == 0:
        assert math.copysign(1, result) == math.copysign(1, expected)


def test_rsqrt_rounds_once_and_fma_ties_use_the_mode():
    # Select a representable input for which rounding sqrt before reciprocal changes the answer.
    inputs = np.arange(1, 256, dtype=np.float32)
    nearest = (1 / np.sqrt(inputs.astype(np.float64))).astype(np.float32)
    composed = np.float32(1) / np.sqrt(inputs)
    different = np.flatnonzero(nearest != composed)
    assert different.size
    index = different[0]
    x = inputs[index]
    assert run_math("ieee_frsqrt", "float32", "rn", (x,)) == nearest[index]
    for mode in ("rn", "rz", "ru", "rd"):
        assert evaluate("fma", (1.0, 1.0, 2**-24), "float32", mode) == (1 + 2**-23 if mode == "ru" else 1)
    # An exact rational midpoint square checks the integer root tie decision independently.
    midpoint = Fraction(1) + Fraction(1, 2**24)
    assert round_exact(midpoint**2, "float32", root=True) == 1
    assert round_exact(midpoint**2, "float32", "ru", root=True) == 1 + 2**-23


@pytest.mark.parametrize("operation", ["ieee_add", "ieee_mul", "ieee_fmaf", "ieee_fdiv"])
def test_ieee_operands_convert_to_first_dtype(operation):
    values = (1, 1 + 2**-30, -1)
    assert run_math(operation, "float32", "rn", values, "float64") == evaluate(
        IEEE_MATH_OPS[operation][0], (1, 1, -1), "float32"
    )


@pytest.mark.parametrize("operation", ["fma", "fmul"])
def test_cuda_fma_and_multiply_require_identical_types(operation):
    with pytest.raises(ntilang.CompileError, match="identical floating operand dtypes"):
        math_kernel(operation, other_dtype="float64")


@pytest.mark.parametrize("mode", ["rp", "rm", "", 1, None])
def test_ieee_mode_must_be_a_supported_static_string(mode):
    with pytest.raises(ntilang.CompileError, match="IEEE rounding_mode"):
        math_kernel("ieee_add", mode=mode)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("mode", ["rz", "ru", "rd"])
def test_half_precision_requires_nearest_rounding(dtype, mode):
    with pytest.raises(ntilang.CompileError, match="supports only rn"):
        math_kernel("ieee_add", dtype, mode)


def test_pinned_cuda_ieee_type_limits():
    with pytest.raises(ntilang.CompileError, match="requires a floating first operand"):
        math_kernel("ieee_add", "int32")
    with pytest.raises(ntilang.CompileError, match="does not support float64"):
        math_kernel("ieee_frsqrt", "float64")


requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)
NATIVE_CASES = [
    (operation, dtype, mode, "sm_80")
    for operation in IEEE_MATH_OPS
    for dtype in ("float16", "bfloat16", "float32", "float64")
    for mode in (
        ("rn", "rz", "ru", "rd")
        if dtype in ("float32", "float64") and operation not in ("fma", "fmul", "ieee_frsqrt")
        else ("rn",)
    )
    if (operation, dtype) != ("ieee_frsqrt", "float64")
]
NATIVE_CASES += [
    (operation, "bfloat16", "rn", "sm_90a") for operation in ("ieee_add", "ieee_sub", "ieee_mul", "fma")
]


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("operation,dtype,mode,target", NATIVE_CASES)
def test_explicit_math_native_compilation(operation, dtype, mode, target):
    assert math_kernel(operation, dtype, mode, target=target).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize(
    "operation,dtype", [("ieee_fdiv", "float16"), ("ieee_add", "bfloat16"), ("ieee_fmaf", "float64")]
)
def test_explicit_math_standalone_module(operation, dtype, tmp_path):
    path = math_kernel(operation, dtype).save(tmp_path / "explicit_math.py")
    code = "import runpy, sys; module = runpy.run_path(sys.argv[1]); assert module['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)
