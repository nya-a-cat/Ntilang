import importlib.util
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.scalar import FAST_MATH_OPS
from ntilang.testing import reference

OPERATIONS = sorted(FAST_MATH_OPS - {"fast_rcp"})


def fast_kernel(operation, dtype="float32"):
    intrinsic = getattr(T, operation)

    @T.prim_func
    def kernel(A: T.Tensor(39, dtype), C: T.Tensor(39, dtype)):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                C[i] = intrinsic(x=A[i])

    return ntilang.compile(kernel)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_fast_math_reference_supplies_ideal_values(operation, dtype):
    values = [0.0, -0.0, 0.5, 1, 2, -1, np.inf, -np.inf, np.nan, np.finfo(dtype).tiny]
    a = np.resize(np.array(values, dtype=dtype), 39)
    c = np.empty_like(a)
    functions = {"__exp10": lambda x: np.power(type(x)(10), x)}
    fn = functions.get(operation, getattr(np, operation[2:], None))
    with np.errstate(all="ignore"):
        reference(fast_kernel(operation, dtype), a, c)
        compute = a.astype(np.float32) if dtype == "float16" else a
        expected = np.array([fn(x) for x in compute], dtype=dtype)
    np.testing.assert_array_equal(c, expected)
    np.testing.assert_array_equal(np.signbit(c[c == 0]), np.signbit(expected[expected == 0]))


def test_fast_reciprocal_reference_retains_ideal_subnormal_values():
    a = np.resize(np.array([0, -0.0, 1, -1, np.inf, -np.inf, np.nan, 2**127, 3], dtype=np.float32), 39)
    c = np.empty_like(a)
    with np.errstate(all="ignore"):
        reference(fast_kernel("fast_rcp"), a, c)
        expected = np.float32(1) / a
    np.testing.assert_array_equal(c, expected)
    np.testing.assert_array_equal(np.signbit(c[c == 0]), np.signbit(expected[expected == 0]))
    assert c[7] > 0  # The reference does not simulate rcp.approx.ftz hardware flushing.


@pytest.mark.parametrize("operation", OPERATIONS)
def test_float32_fast_math_selects_the_cuda_intrinsic(operation):
    assert (
        f'@cute.extern(name="__nv_fast_{operation[2:]}f", overloaded=False)' in fast_kernel(operation).source
    )


@pytest.mark.parametrize("operation", OPERATIONS)
def test_float64_fast_names_keep_the_double_precision_library_call(operation):
    source = fast_kernel(operation, "float64").source
    assert f'@cute.extern(name="__nv_{operation[2:]}", overloaded=False)' in source
    assert "cutlass.Float32(" not in source


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("dtype", ["int32", "uint64", "bool"])
def test_fast_transcendentals_require_floating_inputs(operation, dtype):
    with pytest.raises(ntilang.CompileError, match="requires floating inputs"):
        fast_kernel(operation, dtype)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float64", "int32"])
def test_fast_reciprocal_requires_scalar_float32(dtype):
    with pytest.raises(ntilang.CompileError, match="requires a scalar float32"):
        fast_kernel("fast_rcp", dtype)


@T.macro
def fast_log_macro(x):
    return T.__log(x)


@T.prim_func
def fast_math_in_macro_and_condition(A: T.Tensor(39, "float32"), C: T.Tensor(39, "float32")):
    with T.Kernel(1, threads=32) as _bx:
        for i in T.Parallel(64):
            if A[i] > 0:
                C[i] = T.__exp(fast_log_macro(A[i]))
            else:
                C[i] = T.__sin(A[i])


def test_direct_names_macro_results_and_runtime_control_flow():
    a = np.linspace(-2, 2, 39, dtype=np.float32)
    c = np.empty_like(a)
    reference(ntilang.compile(fast_math_in_macro_and_condition), a, c)
    expected = np.sin(a)
    expected[a > 0] = np.exp(np.log(a[a > 0]))
    np.testing.assert_array_equal(c, expected)


def paired_sqrt_kernel(operation, dtype, order):
    intrinsic = getattr(T, operation)

    @T.prim_func
    def kernel(
        A: T.Tensor(39, dtype), B: T.Tensor(39, "float32"), C: T.Tensor(39, dtype), D: T.Tensor(39, "float32")
    ):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                if order == "sqrt_first":
                    C[i] = T.ieee_fsqrt(A[i])
                    D[i] = intrinsic(B[i])
                else:
                    D[i] = intrinsic(B[i])
                    C[i] = T.ieee_fsqrt(A[i])

    return ntilang.compile(kernel)


@pytest.mark.parametrize("operation", ["__exp", "__log", "__sin", "__cos", "fast_rcp", "__log2"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("order", ["sqrt_first", "fast_first"])
def test_math_header_sqrt_dispatch_is_shared_by_the_whole_kernel(operation, dtype, order):
    source = paired_sqrt_kernel(operation, dtype, order).source
    header = operation != "__log2"
    assert ('@cute.extern(name="__nv_sqrtf", overloaded=False)' in source) == header
    assert ("sqrt.approx" in source) != header


requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)
NATIVE_CASES = [
    (operation, dtype) for operation in OPERATIONS for dtype in ("float16", "bfloat16", "float32", "float64")
]
NATIVE_CASES += [("fast_rcp", "float32")]


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("operation,dtype", NATIVE_CASES)
def test_fast_math_native_compilation(operation, dtype):
    assert fast_kernel(operation, dtype).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("order", ["sqrt_first", "fast_first"])
def test_fast_math_and_ieee_sqrt_compile_together(dtype, order):
    assert paired_sqrt_kernel("__log", dtype, order).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
def test_fast_math_macro_and_control_flow_native_compilation():
    assert ntilang.compile(fast_math_in_macro_and_condition).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize(
    "operation,dtype", [("__exp", "float16"), ("__log2", "bfloat16"), ("fast_rcp", "float32")]
)
def test_fast_math_standalone_module(operation, dtype, tmp_path):
    path = fast_kernel(operation, dtype).save(tmp_path / "fast_math.py")
    code = "import runpy, sys; module = runpy.run_path(sys.argv[1]); assert module['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)
