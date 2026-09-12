import ast
import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ir import DTYPES, ScalarParameter
from ntilang.runtime import scalar_ffi_argument
from ntilang.testing import reference

requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)


def checked_store(message="positive", error_kind="ValueError"):
    @T.prim_func
    def kernel(B: T.Tensor((4,), "int32"), value: T.int32):
        T.Assert(value > 0, message, error_kind=error_kind)
        with T.Kernel(2, threads=2) as bx:
            for i in T.Parallel(2):
                B[bx * 2 + i] = value

    return ntilang.compile(kernel)


def equality_check(dtype):
    @T.prim_func
    def kernel(value: dtype, expected: dtype):
        T.Assert(value == expected, "scalar equality")
        with T.Kernel(threads=1):
            pass

    return ntilang.compile(kernel)


def typed_value(dtype):
    if dtype == "bool":
        return True
    if dtype.startswith(("int", "uint")):
        return int(np.iinfo(dtype).max)
    return 1 + 2**-40 if dtype == "float64" else 1.5


def test_preconditions_run_before_any_output_write():
    compiled = checked_store()
    assert len(compiled.ir.host_checks) == 1
    assert compiled.ir.host_checks[0].op == "host_assert"
    assert compiled.ir.host_checks[0].args[1:] == (("positive",), "ValueError")
    output = np.full(4, -123, dtype=np.int32)
    with pytest.raises(ValueError, match="positive"):
        reference(compiled, output, 0)
    np.testing.assert_array_equal(output, -123)
    reference(compiled, output, 7)
    np.testing.assert_array_equal(output, 7)


def checked_window_sum(batch=7, height=3, width=5):
    @T.prim_func
    def kernel(
        A: T.Tensor((batch, height, width), "float32"),
        B: T.Tensor((batch,), "float32"),
        n: T.int32,
        columns: T.int32,
    ):
        T.Assert((n >= 0) and (n <= height), "row count outside window", error_kind="ValueError")
        with T.Kernel(T.ceildiv(batch, 32), threads=32) as bx:
            for item in T.Parallel(32):
                total = T.alloc_var("float32", init=0)
                for i, j in T.grid(T.clamp(n, 0, height), T.clamp(columns, 0, width)):
                    total += A[bx * 32 + item, i, j]
                B[bx * 32 + item] = total

    return ntilang.compile(kernel)


def test_preconditions_integrate_with_runtime_grid_and_clamp():
    compiled = checked_window_sum()
    source = np.arange(7 * 3 * 5, dtype=np.float32).reshape(7, 3, 5)
    output = np.full(7, -123, dtype=np.float32)
    for n in (-1, 4):
        with pytest.raises(ValueError, match="row count outside window"):
            reference(compiled, source, output, n, 5)
        np.testing.assert_array_equal(output, -123)
    for n, columns in [(0, 5), (2, 3), (3, 9), (1, -2)]:
        reference(compiled, source, output, n, columns)
        expected = source[:, :n, : max(0, min(columns, 5))].sum(axis=(1, 2))
        np.testing.assert_array_equal(output, expected)


def ordered_checks():
    @T.prim_func
    def kernel(value: T.int32):
        T.Assert(value > 0, "first", error_kind="ValueError")
        T.Assert(value > 3, "second", error_kind="IndexError")
        with T.Kernel(threads=1):
            pass

    return ntilang.compile(kernel)


def test_first_failing_precondition_retains_source_order():
    compiled = ordered_checks()
    with pytest.raises(ValueError, match="first"):
        reference(compiled, 0)
    with pytest.raises(IndexError, match="second"):
        reference(compiled, 2)
    reference(compiled, 4)


def guarded_following_check():
    @T.prim_func
    def kernel(value: T.int32):
        T.Assert(value != 0, "first divisor guard", error_kind="ValueError")
        T.Assert(100 // value > 0, "following quotient", error_kind="IndexError")
        with T.Kernel(threads=1):
            pass

    return ntilang.compile(kernel)


def test_failed_assertion_skips_later_undefined_arithmetic():
    compiled = guarded_following_check()
    with pytest.raises(ValueError, match="first divisor guard"):
        reference(compiled, 0)
    with pytest.raises(IndexError, match="following quotient"):
        reference(compiled, -2)
    reference(compiled, 2)


def test_generated_python_wrapper_limits_native_attribute_access():
    source = ast.parse(checked_store().source)
    wrapper = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "_nt_CheckedExecutable"
    )
    namespace = {}
    # Exercise the actual generated Python class without importing CuTe or running native code.
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), "<generated-wrapper>", "exec"), namespace)
    allowed = ("artifacts", "function_name", "has_gpu_module", "__ptx__", "__cubin__", "__sass__", "__mlir__")
    blocked = ("to", "engine", "export_to_c", "capi_func", "__tvm_ffi_object__")
    attributes = {name: object() for name in (*allowed, *blocked)}
    executable = namespace["_nt_CheckedExecutable"](SimpleNamespace(**attributes), object())
    for name in allowed:
        assert getattr(executable, name) is attributes[name]
    for name in blocked:
        with pytest.raises(TypeError, match="host-check integration"):
            getattr(executable, name)


def test_assert_alias_accepts_keyword_arguments_and_frames():
    require = T.Assert

    @T.prim_func
    def kernel(value: T.int32):
        require(message="alias first", condition=value > 0, error_kind="ValueError")
        with require(condition=value > 2, message="alias frame", error_kind="IndexError"):
            require(value < 5, "alias nested")
        with T.Kernel(threads=1):
            pass

    compiled = ntilang.compile(kernel)
    reference(compiled, 3)
    with pytest.raises(ValueError, match="alias first"):
        reference(compiled, 0)
    with pytest.raises(IndexError, match="alias frame"):
        reference(compiled, 1)
    with pytest.raises(RuntimeError, match="alias nested"):
        reference(compiled, 5)


@pytest.mark.parametrize(
    "mode",
    [
        "missing_condition",
        "missing_message",
        "duplicate_condition",
        "duplicate_message",
        "duplicate_kind",
        "extra_positional",
        "unknown_keyword",
    ],
)
def test_assert_argument_binding_rejects_invalid_calls(mode):
    require = T.Assert

    @T.prim_func
    def kernel(value: T.int32):
        if mode == "missing_condition":
            require(message="binding")
        elif mode == "missing_message":
            require(value > 0)
        elif mode == "duplicate_condition":
            require(value > 0, "binding", condition=value > 1)
        elif mode == "duplicate_message":
            require(value > 0, "binding", message="duplicate")
        elif mode == "duplicate_kind":
            require(value > 0, "binding", "RuntimeError", error_kind="ValueError")
        elif mode == "extra_positional":
            require(value > 0, "binding", "RuntimeError", "extra")
        else:
            require(value > 0, "binding", unknown=1)
        with T.Kernel(threads=1):
            pass

    with pytest.raises(ntilang.CompileError):
        ntilang.compile(kernel)


ERROR_CLASSES = [
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    IndexError,
    AssertionError,
    MemoryError,
]


@pytest.mark.parametrize("error_class", ERROR_CLASSES)
def test_registered_error_kind(error_class):
    compiled = checked_store(error_kind=error_class.__name__)
    with pytest.raises(error_class, match="positive"):
        reference(compiled, np.zeros(4, dtype=np.int32), 0)


def test_unknown_error_kind_falls_back_to_runtime_error():
    with pytest.raises(RuntimeError, match="positive"):
        reference(checked_store(error_kind="CustomError"), np.zeros(4, dtype=np.int32), 0)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("", ""),
        ("literal %n {value} ' \\\n猫", "literal %n {value} ' \\\n猫"),
        (["first", " second"], "first second"),
        (("ValueError", " text"), "ValueError text"),
        (["a\0discarded", "b\0discarded"], "ab"),
    ],
)
def test_message_parts_remain_literal_and_concatenate(message, expected):
    compiled = checked_store(message=message)
    with pytest.raises(ValueError) as caught:
        reference(compiled, np.zeros(4, dtype=np.int32), 0)
    assert str(caught.value) == expected


def test_python_assert_runtime_and_construction_semantics():
    @T.prim_func
    def runtime(value: T.int32):
        assert value > 0, "runtime condition"
        with T.Kernel(threads=1):
            pass

    compiled = ntilang.compile(runtime)
    reference(compiled, 1)
    with pytest.raises(RuntimeError, match="runtime condition"):
        reference(compiled, 0)

    @T.prim_func
    def static():
        assert False, "construction condition"
        with T.Kernel(threads=1):
            pass

    with pytest.raises(AssertionError, match="construction condition"):
        ntilang.compile(static)


@pytest.mark.parametrize("omit", [False, True])
def test_python_assert_default_message(omit):
    @T.prim_func
    def kernel(value: T.int32):
        if omit:
            assert value > 0
        else:
            assert value > 0, None
        with T.Kernel(threads=1):
            pass

    with pytest.raises(RuntimeError, match="Assertion failed"):
        reference(ntilang.compile(kernel), 0)


def test_explicit_static_failure_is_checked_at_invocation():
    @T.prim_func
    def kernel():
        T.Assert(False, "explicit static", error_kind="ValueError")
        with T.Kernel(threads=1):
            pass

    compiled = ntilang.compile(kernel)
    with pytest.raises(ValueError, match="explicit static"):
        reference(compiled)


def framed_check():
    @T.prim_func
    def kernel(value: T.int32, B: T.Tensor((1,), "int32")):
        with T.Assert(value > 0, "outer"):
            shifted = value + 1
            with T.Assert(shifted > 2, "inner", error_kind="ValueError"):
                result = shifted * 2
        with T.Kernel(threads=1):
            for i in T.Parallel(1):
                B[i] = result

    return ntilang.compile(kernel)


def test_assert_frames_keep_prelaunch_bindings_visible():
    compiled = framed_check()
    output = np.full(1, -1, dtype=np.int32)
    with pytest.raises(RuntimeError, match="outer"):
        reference(compiled, 0, output)
    with pytest.raises(ValueError, match="inner"):
        reference(compiled, 1, output)
    np.testing.assert_array_equal(output, -1)
    reference(compiled, 3, output)
    np.testing.assert_array_equal(output, 8)


@pytest.mark.parametrize("dtype", [d for d in DTYPES if d != "bfloat16"])
def test_scalar_preconditions_preserve_declared_types(dtype):
    compiled = equality_check(dtype)
    expected = typed_value(dtype)
    reference(compiled, expected, expected)
    with pytest.raises(RuntimeError, match="scalar equality"):
        reference(compiled, False if dtype == "bool" else 0, expected)


def uint64_high_check():
    @T.prim_func
    def kernel(value: T.uint64):
        T.Assert(value > T.uint64(2**63 - 1), "high half")
        with T.Kernel(threads=1):
            pass

    return ntilang.compile(kernel)


def test_uint64_comparison_keeps_unsigned_high_half():
    compiled = uint64_high_check()
    for value in [2**63, 2**64 - 1]:
        reference(compiled, value)
    with pytest.raises(RuntimeError, match="high half"):
        reference(compiled, 2**63 - 1)


def basic_math_check(dtype="float32"):
    @T.prim_func
    def kernel(value: dtype):
        T.Assert(T.min(T.max(T.abs(value), 1.0), 4.0) == 2.0, "bounded magnitude")
        with T.Kernel(threads=1):
            pass

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_portable_basic_math_preconditions(dtype):
    compiled = basic_math_check(dtype)
    reference(compiled, -2.0)
    with pytest.raises(RuntimeError, match="bounded magnitude"):
        reference(compiled, 0.0)


ROUNDING_CASES = [
    ("floor", -2.5, -3.0),
    ("ceil", -2.5, -2.0),
    ("trunc", -2.5, -2.0),
    ("round", -2.5, -2.0),
    ("round_away", -2.5, -3.0),
    ("nearbyint", -2.5, -2.0),
]
CLASSIFICATION_CASES = [
    ("isnan", float("nan"), True),
    ("isinf", float("-inf"), True),
    ("isfinite", float("inf"), False),
]
HOST_UNARY_CASES = [
    (operation, dtype, value, expected)
    for dtype in ("float32", "float64")
    for operation, value, expected in (*ROUNDING_CASES, *CLASSIFICATION_CASES)
] + [(operation, "float16", value, expected) for operation, value, expected in ROUNDING_CASES]
HOST_UNARY_CASES += [
    ("isinf", dtype, value, expected)
    for dtype in ("float16", "float32", "float64")
    for value, expected in ((float("inf"), True), (float("nan"), False), (-0.0, False), (65504.0, False))
] + [("isinf", "float16", float("-inf"), True)]


def unary_math_check(operation, dtype):
    intrinsic = getattr(T, "round" if operation == "round_away" else operation)
    output_dtype = "bool" if operation in ("isnan", "isinf", "isfinite") else dtype

    @T.prim_func
    def kernel(value: dtype, expected: output_dtype):
        if operation == "round_away":
            result = intrinsic(value, rounding_mode="ties-away-from-zero")
        else:
            result = intrinsic(value)
        T.Assert(result == expected, "unary scalar result")
        with T.Kernel(threads=1):
            pass

    return ntilang.compile(kernel)


@pytest.mark.parametrize("operation,dtype,value,expected", HOST_UNARY_CASES)
def test_host_rounding_and_classification_semantics(operation, dtype, value, expected):
    compiled = unary_math_check(operation, dtype)
    reference(compiled, value, expected)
    incorrect = not expected if isinstance(expected, bool) else expected + 1.0
    with pytest.raises(RuntimeError, match="unary scalar result"):
        reference(compiled, value, incorrect)


@pytest.mark.parametrize("operation", ["isnan", "isinf", "isfinite"])
def test_bfloat16_classification_retains_source_language_boundary(operation):
    with pytest.raises(ntilang.CompileError, match="bfloat16"):
        unary_math_check(operation, "bfloat16")


def lazy_check():
    @T.prim_func
    def kernel(value: T.int32):
        T.Assert(T.if_then_else(value != 0, value // value, 1) == 1, "lazy division")
        T.Assert(T.Select(T.likely(value >= 0), value, -value) < 8, "selected magnitude")
        with T.Kernel(threads=1):
            pass

    return ntilang.compile(kernel)


def test_lazy_predicate_skips_undefined_unselected_division():
    compiled = lazy_check()
    for value in [0, 3, -3]:
        reference(compiled, value)
    with pytest.raises(RuntimeError, match="selected magnitude"):
        reference(compiled, -8)


@pytest.mark.parametrize("message", [[], (), 7, ["valid", 7], None])
def test_explicit_assert_rejects_invalid_messages(message):
    with pytest.raises(ntilang.CompileError):
        checked_store(message=message)


@pytest.mark.parametrize("error_kind", [None, 7, ["ValueError"]])
def test_assert_rejects_nonstring_error_kinds(error_kind):
    with pytest.raises(ntilang.CompileError):
        checked_store(error_kind=error_kind)


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_assert_requires_boolean_condition(dtype):
    @T.prim_func
    def kernel(value: dtype):
        T.Assert(value, "must be Boolean")
        with T.Kernel(threads=1):
            pass

    with pytest.raises(ntilang.CompileError, match="[Bb]ool"):
        ntilang.compile(kernel)


@pytest.mark.parametrize("mode", ["tensor", "device", "frame_binding", "frame_kernel"])
def test_assert_rejects_unsupported_execution_locations(mode):
    @T.prim_func
    def kernel(value: T.int32, A: T.Tensor((1,), "int32")):
        if mode == "tensor":
            T.Assert(A[0] > 0, "tensor data")
        elif mode == "frame_binding":
            with T.Assert(value > 0, "frame") as _frame:
                T.Assert(value > 1, "nested")
        elif mode == "frame_kernel":
            with T.Assert(value > 0, "frame"):
                with T.Kernel(threads=1):
                    A[0] = value
        with T.Kernel(threads=1):
            if mode == "device":
                T.Assert(value > 0, "inside device")
            A[0] = value

    with pytest.raises(ntilang.CompileError):
        ntilang.compile(kernel)


@pytest.mark.parametrize("operation", ["fast_rcp", "__exp", "ieee_fsqrt", "popcount"])
def test_preconditions_reject_cuda_specific_intrinsics(operation):
    intrinsic = getattr(T, operation)
    dtype = "uint32" if operation == "popcount" else "float32"

    @T.prim_func
    def kernel(value: dtype):
        T.Assert(intrinsic(value) > 0, "CUDA intrinsic")
        with T.Kernel(threads=1):
            pass

    with pytest.raises(ntilang.CompileError):
        ntilang.compile(kernel)


@pytest.mark.cuda
@requires_cute
def test_native_checker_output_pointer_needs_no_cuda_module():
    import ctypes

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr, nullptr

    @cute.jit
    def check(value: cutlass.Int32, output: cute.Pointer):
        output[0] = cutlass.Int32(value > 3)

    executable = cute.compile(
        check,
        cutlass.Int32(0),
        nullptr(cutlass.Int32, mem_space=cute.AddressSpace.generic, assumed_align=4),
        options="--enable-tvm-ffi --gpu-arch=sm_80",
    )
    assert not executable.has_gpu_module
    output = ctypes.c_int32(-1)
    pointer = make_ptr(
        cutlass.Int32, ctypes.addressof(output), mem_space=cute.AddressSpace.generic, assumed_align=4
    )
    executable(4, pointer)
    assert output.value == 1
    executable(0, pointer)
    assert output.value == 0


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("condition", [False, True])
def test_native_zero_scalar_input_checker(condition):
    @T.prim_func
    def kernel():
        T.Assert(condition, "zero argument check", error_kind="ValueError")
        with T.Kernel(threads=1):
            pass

    executable = ntilang.compile(kernel).build()
    assert not executable._checker.has_gpu_module
    if condition:
        executable._check_arguments()
    else:
        with pytest.raises(ValueError, match="zero argument check"):
            executable()


@pytest.mark.cuda
@requires_cute
def test_native_assert_frame_bindings_and_prelaunch_failure():
    executable = framed_check().build()
    with pytest.raises(RuntimeError, match="outer"):
        executable(0, object())
    with pytest.raises(ValueError, match="inner"):
        executable(1, object())
    executable._check_arguments(3, object())


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("dtype", DTYPES)
def test_native_cpu_checker_preserves_every_scalar_ffi_type(dtype):
    compiled = equality_check(dtype)
    executable = compiled.build()
    value = typed_value(dtype)
    parameter = ScalarParameter("value", dtype)
    expected = scalar_ffi_argument(value, parameter)
    zero = scalar_ffi_argument(False if dtype == "bool" else 0, parameter)
    executable._check_arguments(expected, expected)
    with pytest.raises(RuntimeError, match="scalar equality"):
        executable._check_arguments(zero, expected)
    executable._check_arguments(expected, expected)
    assert executable.has_gpu_module


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_native_cpu_checker_uses_declared_float_rounding(dtype):
    executable = equality_check(dtype).build()
    if dtype == "float64":
        with pytest.raises(RuntimeError, match="scalar equality"):
            executable._check_arguments(1 + 2**-30, 1.0)
    else:
        executable._check_arguments(1 + 2**-30, 1.0)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_native_cpu_checker_low_precision_basic_arithmetic(dtype):
    @T.prim_func
    def kernel(value: dtype):
        T.Assert((value + value) * value == T.cast(4.5, dtype), "low precision arithmetic")
        with T.Kernel(threads=1):
            pass

    executable = ntilang.compile(kernel).build()
    executable._check_arguments(1.5)
    with pytest.raises(RuntimeError, match="low precision arithmetic"):
        executable._check_arguments(1.0)


@pytest.mark.cuda
@requires_cute
def test_native_cpu_checker_keeps_uint64_unsigned_comparisons():
    compiled = uint64_high_check()
    executable = compiled.build()
    for value in [2**63, 2**64 - 1]:
        executable._check_arguments(scalar_ffi_argument(value, compiled.ir.parameters[0]))
    with pytest.raises(RuntimeError, match="high half"):
        executable._check_arguments(2**63 - 1)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize(
    "factory", [lazy_check, lambda: basic_math_check("float32"), lambda: basic_math_check("float64")]
)
def test_native_cpu_checker_runs_portable_expression_lowering(factory):
    executable = factory().build()
    executable._check_arguments(-2)
    if factory is lazy_check:
        executable._check_arguments(0)
        with pytest.raises(RuntimeError, match="selected magnitude"):
            executable._check_arguments(8)
    else:
        with pytest.raises(RuntimeError, match="bounded magnitude"):
            executable._check_arguments(0)


@pytest.mark.cuda
@requires_cute
def test_native_first_failure_and_wrapper_precede_tensor_ffi():
    executable = ordered_checks().build()
    with pytest.raises(ValueError, match="first"):
        executable._check_arguments(0)
    with pytest.raises(IndexError, match="second"):
        executable._check_arguments(2)
    executable._check_arguments(4)
    # A plain object would fail the device callable's tensor FFI immediately.
    # Observing ValueError here verifies the wrapper checks scalar inputs first.
    with pytest.raises(ValueError, match="positive"):
        checked_store().build()(object(), 0)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("operation,dtype,value,expected", HOST_UNARY_CASES)
def test_native_cpu_checker_rounding_and_classification(operation, dtype, value, expected):
    executable = unary_math_check(operation, dtype).build()
    executable._check_arguments(value, expected)
    incorrect = not expected if isinstance(expected, bool) else expected + 1.0
    with pytest.raises(RuntimeError, match="unary scalar result"):
        executable._check_arguments(value, incorrect)


@pytest.mark.cuda
@requires_cute
def test_native_failure_skips_later_division_in_isolated_process(tmp_path):
    path = guarded_following_check().save(tmp_path / "guarded_host_assert.py")
    code = """
import runpy
import sys
module = runpy.run_path(sys.argv[1])
executable = module['compile_kernel']()
assert not executable._checker.has_gpu_module
for value in (0, 2, 0, -2):
    try:
        executable._check_arguments(value)
    except ValueError as error:
        assert value == 0 and str(error) == 'first divisor guard'
    except IndexError as error:
        assert value == -2 and str(error) == 'following quotient'
    else:
        assert value == 2
assert 'ntilang' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)


@pytest.mark.cuda
@requires_cute
def test_standalone_checked_runtime_grid_compiles_and_checks_on_cpu(tmp_path):
    path = checked_window_sum().save(tmp_path / "checked_window_sum.py")
    code = """
import runpy
import sys
module = runpy.run_path(sys.argv[1])
executable = module['compile_kernel']()
assert executable.has_gpu_module
assert not executable._checker.has_gpu_module
for n, columns in ((0, 5), (2, 3), (3, 9), (1, -2)):
    executable._check_arguments(object(), object(), n, columns)
for n in (-1, 4):
    try:
        executable(object(), object(), n, 5)
    except ValueError as error:
        assert str(error) == 'row count outside window'
    else:
        raise AssertionError('invalid runtime row count was accepted')
assert 'ntilang' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)


@pytest.mark.cuda
@requires_cute
def test_standalone_host_checker_uses_no_ntilang_or_array_adapter(tmp_path):
    compiled = checked_store(message=["literal %n {x} 猫\0discarded", " tail"])
    assert "import numpy" not in compiled.source
    path = compiled.save(tmp_path / "host_assert.py")
    code = """
import runpy
import sys
module = runpy.run_path(sys.argv[1])
executable = module['compile_kernel']()
assert executable.has_gpu_module
assert executable.artifacts is not None
try:
    executable(object(), 0)
except ValueError as error:
    assert str(error) == 'literal %n {x} 猫 tail', str(error)
else:
    raise AssertionError('host precondition was not enforced')
executable._check_arguments(object(), 1)
assert 'ntilang' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)
