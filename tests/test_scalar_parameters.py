import importlib.util
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.codegen import CUTLASS_TYPES
from ntilang.ir import DTYPES, ScalarParameter
from ntilang.runtime import scalar_ffi_argument
from ntilang.testing import reference


def scalar_store(dtype):
    @T.prim_func
    def kernel(value: dtype, B: T.Tensor((39,), dtype)):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                B[bx * 32 + i] = value

    return ntilang.compile(kernel)


def scalar_values(dtype):
    if dtype == "bool":
        return [False, True]
    if dtype.startswith(("int", "uint")):
        limits = np.iinfo(dtype)
        return [int(limits.min), 0, int(limits.max)]
    return [-2.25, 1 + 2**-40, float("inf"), float("nan"), -0.0]


@pytest.mark.parametrize("dtype", [d for d in DTYPES if d != "bfloat16"])
def test_runtime_scalar_types_and_values_survive_across_blocks(dtype):
    kernel = scalar_store(dtype)
    output = np.empty(39, dtype=dtype)
    assert [p.name for p in kernel.ir.parameters] == ["value", "B"]
    assert set(kernel.ir.buffer_map) == {"B"}
    for value in scalar_values(dtype):
        reference(kernel, value, output)
        expected = np.full(39, value, dtype=dtype)
        np.testing.assert_array_equal(output, expected)
        if dtype.startswith("float"):
            np.testing.assert_array_equal(np.signbit(output), np.signbit(expected))


def scaled_sum():
    @T.prim_func
    def kernel(
        A: T.Tensor((39, 7), "float32"),
        alpha: T.float32,
        B: T.Tensor((39,), "float32"),
        count: T.int32,
        enabled: T.bool,
        bias: T.float32,
    ):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                row = bx * 32 + i
                acc = T.alloc_var("float32", bias)
                if enabled:
                    for k in T.serial(T.max(0, T.min(count, 7))):
                        acc += A[row, k] * alpha
                B[row] = acc

    return ntilang.compile(kernel)


@pytest.mark.parametrize("count,enabled", [(-7, True), (0, False), (3, True), (2**31 - 1, True)])
def test_mixed_parameter_order_dynamic_bounds_and_conditions(count, enabled):
    a = np.arange(39 * 7, dtype=np.float32).reshape(39, 7)
    b = np.empty(39, dtype=np.float32)
    kernel = scaled_sum()
    for alpha, bias in [(0.25, -2.0), (-0.5, 1.0)]:
        reference(kernel, a, alpha, b, count, enabled, bias)
        n = max(0, min(count, 7)) if enabled else 0
        np.testing.assert_array_equal(b, a[:, :n].sum(axis=1) * alpha + bias)


def scalar_gather(dtype):
    @T.prim_func
    def kernel(A: T.Tensor((300,), "float32"), index: dtype, B: T.Tensor((39,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                B[bx * 32 + i] = A[index]

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32"])
def test_scalar_index_conversion_and_tail_guards(dtype):
    a = np.arange(300, dtype=np.float32) + 0.25
    b = np.empty(39, dtype=np.float32)
    kernel = scalar_gather(dtype)
    for index in scalar_values(dtype) + [7]:
        reference(kernel, a, index, b)
        np.testing.assert_array_equal(b, a[index] if 0 <= index < 300 else 0)


def runtime_power():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), exponent: T.int32, B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = T.pow(A[i], exponent)

    return ntilang.compile(kernel)


def test_scalar_exponent_is_runtime_state():
    kernel = runtime_power()
    a = np.arange(1, 33, dtype=np.float32)
    b = np.empty_like(a)
    for exponent in [0, 3, -2]:
        reference(kernel, a, exponent, b)
        np.testing.assert_allclose(b, np.power(a, np.float32(exponent)), rtol=1e-6)
    assert "cute.math.pow" in kernel.source


def test_scalar_dtype_alias():
    alias = T.short

    @T.prim_func
    def kernel(value: alias, B: T.Tensor((32,), "int16")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = value

    compiled = ntilang.compile(kernel)
    b = np.empty(32, dtype=np.int16)
    reference(compiled, -123, b)
    np.testing.assert_array_equal(b, -123)
    assert compiled.ir.parameters[0].dtype == "int16"


def test_runtime_parameter_cannot_shadow_a_static_grid_constant():
    extent = 2

    @T.prim_func
    def bad(extent: T.int32, B: T.Tensor((32,), "float32")):
        with T.Kernel(extent, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = 1

    assert extent == 2
    with pytest.raises(ntilang.CompileError, match="runtime variable.*static"):
        ntilang.compile(bad)


def test_scalar_parameter_cannot_be_overwritten_by_a_block_name():
    @T.prim_func
    def bad(bx: T.int32, B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as bx:
            for i in T.Parallel(32):
                B[i] = bx

    with pytest.raises(ntilang.CompileError, match="block variable"):
        ntilang.compile(bad)


class MetadataTensor:
    """Only exercises argument checks; no storage allocation or kernel execution."""

    shape = (39,)

    def __init__(self, dtype):
        self.dtype = dtype

    def __dlpack_device__(self):
        return (2, 0)

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return 4096


@pytest.mark.parametrize("dtype", list(DTYPES))
def test_launch_wrapper_preserves_argument_order_and_scalar_wire_values(dtype):
    kernel = scalar_store(dtype)
    tensor = MetadataTensor(dtype)
    kernel._executable = lambda *args: args
    values = [1.5] if dtype == "bfloat16" else scalar_values(dtype)
    for value in values:
        actual, actual_tensor = kernel(value, tensor)
        expected = scalar_ffi_argument(value, kernel.ir.parameters[0])
        if isinstance(expected, float) and np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == expected
        assert actual_tensor is tensor


@pytest.mark.parametrize(
    "dtype,value,error",
    [
        ("int8", 128, ValueError),
        ("int8", -129, ValueError),
        ("uint8", -1, ValueError),
        ("uint64", 2**64, ValueError),
        ("int64", 2**63, ValueError),
        ("int32", 1.5, TypeError),
        ("int32", True, TypeError),
        ("float32", "1.5", TypeError),
        ("float32", 1 + 2j, TypeError),
        ("bool", 1, TypeError),
    ],
)
def test_scalar_input_errors_precede_compilation(dtype, value, error):
    kernel = scalar_store(dtype)
    with pytest.raises(error, match="value"):
        kernel(value, MetadataTensor(dtype))
    with pytest.raises(error, match="value"):
        reference(kernel, value, np.empty(39, dtype=dtype))


requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_native_scalar_kernel_compilation(dtype):
    assert scalar_store(dtype).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize(
    "factory",
    [scaled_sum, runtime_power]
    + [lambda d=d: scalar_gather(d) for d in ["int8", "uint8", "int16", "uint16", "int32"]],
)
def test_native_scalar_control_math_and_index_compilation(factory):
    assert factory().build().has_gpu_module


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("dtype_name", list(DTYPES))
def test_actual_scalar_ffi_with_cpu_only_function(dtype_name):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor

    dtype = getattr(cutlass, CUTLASS_TYPES[dtype_name])
    if dtype_name == "bool":
        expected = True
    elif dtype_name.startswith(("int", "uint")):
        expected = int(np.iinfo(dtype_name).max)
    else:
        expected = 1 + 2**-40 if dtype_name == "float64" else 1.5

    @cute.jit
    def check(value: dtype, output: cute.Tensor):
        output[0] = cutlass.Int32(value == dtype(expected))

    # This function contains only host scalar arithmetic and never launches a kernel.
    # Compiled CuTe entrypoints expose results through arguments, not Python return values.
    fake_output = make_fake_compact_tensor(
        cutlass.Int32, (1,), memspace=cute.AddressSpace.generic, assumed_align=4
    )
    executable = cute.compile(check, dtype(0), fake_output, options="--enable-tvm-ffi --gpu-arch=sm_80")
    assert not executable.has_gpu_module
    output = np.empty(1, dtype=np.int32)
    parameter = ScalarParameter("value", dtype_name)
    executable(scalar_ffi_argument(expected, parameter), output)
    assert output[0] == 1
    executable(scalar_ffi_argument(False if dtype_name == "bool" else 0, parameter), output)
    assert output[0] == 0


@pytest.mark.cuda
@requires_cute
def test_scalar_generated_module_is_standalone(tmp_path):
    path = scaled_sum().save(tmp_path / "scalars.py")
    code = (
        "import runpy, sys; "
        "module = runpy.run_path(sys.argv[1]); "
        "assert module['compile_kernel']().has_gpu_module; "
        "assert 'ntilang' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)
