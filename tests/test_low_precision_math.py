import importlib.util
import subprocess
import sys

import ntilang
import ntilang.language as T
import pytest

NATIVE_UNARY = [
    "exp",
    "exp2",
    "exp10",
    "log",
    "log2",
    "log10",
    "sin",
    "cos",
    "sqrt",
    "rsqrt",
    "tanh",
    "sigmoid",
]


def mixed_math_kernel(operation, dtype, header=False):
    intrinsic = getattr(T, operation)

    @T.prim_func
    def kernel(
        A: T.Tensor(39, dtype), B: T.Tensor(39, "float32"), C: T.Tensor(39, dtype), D: T.Tensor(39, "float32")
    ):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                C[i] = intrinsic(A[i])
                if header:
                    D[i] = T.fast_rcp(B[i])

    return ntilang.compile(kernel)


@pytest.mark.parametrize("operation", ["exp", "log", "sin", "cos", "sqrt", "tanh"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("header", [False, True])
def test_ordinary_half_math_uses_the_pinned_header_dispatch(operation, dtype, header):
    source = mixed_math_kernel(operation, dtype, header).source
    expected_library = (
        (operation in ("sin", "cos") and dtype == "bfloat16")
        or (header and operation in ("log", "sin", "cos", "sqrt"))
        or (header and operation == "exp" and dtype == "bfloat16")
        or (not header and operation == "tanh")
    )
    assert (f'@cute.extern(name="__nv_{operation}f", overloaded=False)' in source) == expected_library
    if operation == "tanh" and header:
        assert ("tanh.approx.f16" if dtype == "float16" else "tanh.approx.f32") in source
    if operation == "exp" and not expected_library:
        # CUDA deliberately uses different nearest FP32 log2(e) constants.
        assert ("0f3fb8aa3b" if dtype == "float16" else "0f3fb8aa3c") in source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("header", [False, True])
def test_sigmoid_retains_typed_exponential_addition_and_division(dtype, header):
    source = mixed_math_kernel("sigmoid", dtype, header).source
    assert f"def _nt_add_{dtype}" in source
    assert f"def _nt_div_{dtype}" in source
    assert ('@cute.extern(name="__nv_expf", overloaded=False)' in source) == (dtype == "bfloat16" and header)
    if dtype == "float16":
        assert "rcp.approx.ftz.f32" in source and "@refine fma.rn.f32" in source
    else:
        assert "div.approx.f32" in source and "@scale fma.rn.f32" in source


@pytest.mark.parametrize("header", [False, True])
def test_half_exp2_retains_its_preconversion_fma_correction(header):
    source = mixed_math_kernel("exp2", "float16", header).source
    assert "ex2.approx.ftz.f32" in source
    assert "fma.rn.f32 value, value, 0f33800000, value;" in source


@pytest.mark.parametrize("operation", ["sin", "cos"])
def test_half_trigonometric_helpers_include_reduction_polynomial_and_corrections(operation):
    source = mixed_math_kernel(operation, "float16").source
    assert "mov.b32 quadrant, rounded_index;" in source
    assert "sub.rn.f32 index_value" in source
    assert "selp.f32 c8" in source and "fma.rn.f32 result, result, linear, constant;" in source
    assert "@patch add.rn.f16" in source
    if operation == "sin":
        assert "or.b16 rounded, rounded, sign;" in source
    else:
        assert "add.u32 quadrant, quadrant, 1;" in source


requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)


@pytest.mark.parametrize("dtype,symbol", [("float32", "__nv_exp10f"), ("float64", "__nv_exp10")])
def test_ordinary_exp10_uses_the_base_ten_library_function(dtype, symbol):
    source = mixed_math_kernel("exp10", dtype).source
    assert f'@cute.extern(name="{symbol}", overloaded=False)' in source
    assert "cute.math.pow" not in source


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("operation", NATIVE_UNARY)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("header", [False, True])
def test_ordinary_low_precision_math_native_compilation(operation, dtype, header):
    assert mixed_math_kernel(operation, dtype, header).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize(
    "operation,dtype", [("sin", "float16"), ("cos", "float16"), ("exp", "bfloat16"), ("sigmoid", "float16")]
)
def test_native_low_precision_math_standalone_module(operation, dtype, tmp_path):
    path = mixed_math_kernel(operation, dtype).save(tmp_path / "low_precision_math.py")
    code = "import runpy, sys; module = runpy.run_path(sys.argv[1]); assert module['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)
