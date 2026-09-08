import importlib.util
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def native_binary(operation, dtype="float32", other_dtype=None):
    intrinsic = getattr(T, operation)
    other_dtype = other_dtype or dtype

    @T.prim_func
    def kernel(A: T.Tensor(39, dtype), B: T.Tensor(39, other_dtype), C: T.Tensor(39, dtype)):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                C[i] = intrinsic(x1=A[i], x2=B[i])

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_hypot_avoids_intermediate_overflow_and_underflow(dtype):
    info = np.finfo(dtype)
    a = np.resize(np.array([info.max / 2, info.tiny, -0.0, 3, -5, np.inf, np.nan], dtype=dtype), 39)
    b = np.resize(np.array([info.max / 2, info.tiny, 0.0, 4, 12, np.nan, np.inf], dtype=dtype), 39)
    c = np.empty_like(a)
    reference(native_binary("hypot", dtype), a, b, c)
    np.testing.assert_array_equal(c, np.hypot(a, b))
    assert np.isfinite(c[0]) and c[1] > 0 and not np.signbit(c[2])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nextafter_preserves_adjacent_values_and_signed_zero(dtype):
    info = np.finfo(dtype)
    a = np.resize(
        np.array([0.0, -0.0, 0.0, -0.0, 1, -1, info.max, np.inf, -np.inf, np.nan, info.tiny], dtype=dtype), 39
    )
    b = np.resize(np.array([-0.0, 0.0, -1, 1, 2, 0, np.inf, 0, 0, 1, 0], dtype=dtype), 39)
    c = np.empty_like(a)
    with np.errstate(all="ignore"):
        reference(native_binary("nextafter", dtype), a, b, c)
        expected = np.nextafter(a, b)
    np.testing.assert_array_equal(c, expected)
    np.testing.assert_array_equal(np.signbit(c[c == 0]), np.signbit(expected[expected == 0]))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ldexp_uses_integer_exponents_and_preserves_special_values(dtype):
    info = np.finfo(dtype)
    a = np.resize(np.array([1, -1, 0.0, -0.0, np.inf, -np.inf, np.nan, info.tiny, info.max], dtype=dtype), 39)
    b = np.resize(np.array([3, -3, 100, -100, 1, -1, 0, -1, 1], dtype=np.int32), 39)
    c = np.empty_like(a)
    with np.errstate(all="ignore"):
        reference(native_binary("ldexp", dtype, "int32"), a, b, c)
        expected = np.ldexp(a, b)
    np.testing.assert_array_equal(c, expected)
    np.testing.assert_array_equal(np.signbit(c[c == 0]), np.signbit(expected[expected == 0]))


@pytest.mark.parametrize("other_dtype", ["int64", "uint64", "float64"])
def test_ldexp_converts_exponent_directly_to_int32(other_dtype):
    a = np.ones(39, dtype=np.float32)
    exponent = 2**32 + 1 if other_dtype != "float64" else 1.75
    b = np.full(39, exponent, dtype=other_dtype)
    c = np.empty_like(a)
    reference(native_binary("ldexp", "float32", other_dtype), a, b, c)
    np.testing.assert_array_equal(c, 2)


@pytest.mark.parametrize("operation", ["hypot", "nextafter"])
def test_external_binary_math_uses_the_first_argument_type(operation):
    a = np.ones(39, dtype=np.float32)
    b = np.full(39, 1 + 2**-30, dtype=np.float64)
    c = np.empty_like(a)
    reference(native_binary(operation, "float32", "float64"), a, b, c)
    np.testing.assert_array_equal(c, getattr(np, operation)(a, b.astype(np.float32)))


@pytest.mark.parametrize("operation", ["hypot", "nextafter", "ldexp"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int32"])
def test_external_binary_math_requires_the_pinned_float_dispatch_types(operation, dtype):
    with pytest.raises(ntilang.CompileError, match="float32 or float64 first operand"):
        native_binary(operation, dtype)


requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)
NATIVE_CASES = [
    (op, dtype, dtype) for op in ["hypot", "nextafter", "ldexp"] for dtype in ["float32", "float64"]
]
NATIVE_CASES += [("ldexp", "float32", dtype) for dtype in ["int32", "int64", "uint64"]]
NATIVE_CASES += [(op, "float32", "float64") for op in ["hypot", "nextafter", "ldexp"]]


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("operation,dtype,other_dtype", NATIVE_CASES)
def test_libdevice_native_compilation(operation, dtype, other_dtype):
    assert native_binary(operation, dtype, other_dtype).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
def test_libdevice_generated_module_is_standalone(tmp_path):
    path = native_binary("hypot", "float64").save(tmp_path / "libdevice_math.py")
    code = "import runpy, sys; module = runpy.run_path(sys.argv[1]); assert module['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)
