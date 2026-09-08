import ctypes
import importlib.util
import runpy
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.codegen import CUTLASS_TYPES
from ntilang.ir import DTYPES, Expr, ScalarParameter
from ntilang.runtime import scalar_ffi_argument
from ntilang.scalar import expression_dtype
from ntilang.testing import reference

requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)


def message_kernel(message="hello"):
    @T.prim_func
    def kernel():
        with T.Kernel(2, 2, threads=2):
            result = T.print(msg=message)
            if result:
                T.print(msg="unexpected return value")

    return ntilang.compile(kernel)


def scalar_kernel(dtype, message="scalar"):
    @T.prim_func
    def kernel(A: T.Tensor((4,), dtype)):
        with T.Kernel(threads=4):
            for i in T.Parallel(4):
                T.print(A[i], message)

    return ntilang.compile(kernel)


def buffer_kernel(dtype, space, group=0, warp=0, threads=4, message=""):
    @T.prim_func
    def kernel(A: T.Tensor((2, 3), dtype)):
        with T.Kernel(threads=threads):
            if space == "shared":
                tile = T.alloc_shared((2, 3), dtype)
                T.copy(A, tile)
            elif space == "fragment":
                tile = T.alloc_fragment((2, 3), dtype)
                T.copy(A, tile)
            else:
                tile = A
            T.print(obj=tile, msg=message, warp_group_id=group, warp_id=warp)

    return ntilang.compile(kernel)


def assert_kernel(no_stack=False, message="positive", dtype="int32"):
    @T.prim_func
    def kernel(value: dtype):
        with T.Kernel(threads=4):
            T.device_assert(value, msg=message, no_stack_info=no_stack)

    return ntilang.compile(kernel)


def likely_kernel(dtype):
    @T.prim_func
    def kernel(A: T.Tensor((4,), dtype), B: T.Tensor((4,), dtype)):
        with T.Kernel(threads=4):
            for i in T.Parallel(4):
                value = T.likely(A[i], None, dtype="bool")
                B[T.likely(i)] = value

    return ntilang.compile(kernel)


def test_message_only_kernel_and_none_return(capsys):
    compiled = message_kernel()
    assert compiled.ir.parameters == ()
    assert compiled.ir.grid == (2, 2)
    reference(compiled)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 8
    assert lines[0] == "msg='hello' BlockIdx=(0, 0, 0), ThreadIdx=(0, 0, 0)"
    assert lines[-1] == "msg='hello' BlockIdx=(1, 1, 0), ThreadIdx=(1, 0, 0)"


@pytest.mark.parametrize("dtype", [d for d in DTYPES if d != "bfloat16"])
def test_scalar_prints_follow_logical_thread_mapping(dtype, capsys):
    a = np.arange(4).astype(dtype)
    reference(scalar_kernel(dtype), a)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert all(f"ThreadIdx=({i}, 0, 0)" in line for i, line in enumerate(lines))
    assert "value=false" in lines[0] if dtype == "bool" else "value=0" in lines[0]
    assert "value=true" in lines[-1] if dtype == "bool" else "value=3" in lines[-1]


@pytest.mark.parametrize("space,count", [("global", 24), ("shared", 6), ("fragment", 6)])
@pytest.mark.parametrize("dtype", ["bool", "int16", "uint16", "float32"])
def test_full_buffer_prints_have_scope_specific_multiplicity(space, count, dtype, capsys):
    a = np.arange(6).astype(dtype).reshape(2, 3)
    reference(buffer_kernel(dtype, space), a)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == count
    expected_name = "A" if space == "global" else "tile"
    assert all(f"buffer={expected_name}, index=" in line for line in lines)
    assert all("ThreadIdx=(0, 0, 0)" in line for line in lines[:6])
    if space != "global":
        assert lines[0].startswith(f"msg='buffer<tile, {dtype}>'")
    if dtype == "uint16":
        assert "dtype=uint16_t" in lines[0]


@pytest.mark.parametrize("space", ["shared", "fragment"])
@pytest.mark.parametrize("group,warp,lane", [(0, 1, 32), (1, 0, 128), (1, 1, None), (-1, 0, None)])
def test_print_warp_selectors(space, group, warp, lane, capsys):
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    reference(buffer_kernel("int32", space, group, warp, threads=160), a)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == (0 if lane is None else 6)
    if lane is not None:
        assert all(f"ThreadIdx=({lane}, 0, 0)" in line for line in lines)


def test_global_print_ignores_warp_selection(capsys):
    reference(buffer_kernel("int32", "global", -1, 1000, threads=2), np.zeros((2, 3), dtype=np.int32))
    assert len(capsys.readouterr().out.splitlines()) == 12


def test_printing_output_does_not_relax_numeric_alias_checks(capsys):
    @T.prim_func
    def kernel(B: T.Tensor((4,), "int32")):
        with T.Kernel(threads=4):
            for i in T.Parallel(4):
                B[i] = i + 7
                T.print(B[i], "result")
                T.device_assert(B[i] >= 7, "written result", no_stack_info=True)

    b = np.empty(4, dtype=np.int32)
    reference(ntilang.compile(kernel), b)
    np.testing.assert_array_equal(b, np.arange(7, 11))
    assert "value=10" in capsys.readouterr().out

    @T.prim_func
    def bad(B: T.Tensor((4,), "int32")):
        with T.Kernel(threads=4):
            for i in T.Parallel(4):
                B[i] = B[i] + 7
                T.print(B[i], "result")

    with pytest.raises(ntilang.CompileError, match="both read and written"):
        ntilang.compile(bad)


@T.macro
def diagnostic_value(counter: T.Ref):
    counter += 1
    return T.int32(7)


@T.macro
def diagnostic_message(counter: T.Ref):
    counter = counter * 10 + 2
    return "macro"


def test_print_keyword_construction_order_and_macro_return(capsys):
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32")):
        with T.Kernel(threads=1):
            for i in T.Parallel(1):
                counter = T.alloc_var("int32", init=0)
                T.print(obj=diagnostic_value(counter), msg=diagnostic_message(counter))
                B[i] = counter

    b = np.empty(1, dtype=np.int32)
    reference(ntilang.compile(kernel), b)
    assert b[0] == 12
    assert "msg='macro'" in capsys.readouterr().out


@T.macro
def nested_assert(value):
    T.device_assert(value > 0, "nested failure")


def test_assert_message_and_source_stack():
    @T.prim_func
    def kernel(value: T.int32):
        with T.Kernel(threads=1):
            nested_assert(value)

    compiled = ntilang.compile(kernel)
    reference(compiled, 1)
    with pytest.raises(AssertionError, match="nested failure") as error:
        reference(compiled, 0)
    assert "test_debug_operations.py:" in str(error.value)
    assert "in nested_assert" in str(error.value)
    assert "in kernel" in str(error.value)


@pytest.mark.parametrize("no_stack,message", [(True, ""), (True, "custom"), (False, "custom")])
def test_assert_truth_conversion_and_stack_option(no_stack, message):
    compiled = assert_kernel(no_stack, message)
    reference(compiled, -1)
    with pytest.raises(AssertionError) as error:
        reference(compiled, 0)
    assert ("  at " not in str(error.value)) == no_stack
    assert message in str(error.value)


def test_construction_assertions():
    def make(width):
        @T.prim_func
        def kernel():
            assert width % 2 == 0, "width must be even"
            with T.Kernel(threads=1):
                T.print(msg="constructed")

        return ntilang.compile(kernel)

    assert make(2).source
    with pytest.raises(AssertionError, match="width must be even"):
        make(3)


@pytest.mark.parametrize("dtype", [d for d in DTYPES if d != "bfloat16"])
def test_likely_preserves_dtype_and_values(dtype):
    a = np.arange(4).astype(dtype)
    b = np.empty_like(a)
    reference(likely_kernel(dtype), a, b)
    np.testing.assert_array_equal(b, a)
    assert expression_dtype(Expr("likely", (Expr("parameter", value=dtype),)), {}, {}) == dtype


def test_likely_refines_lazy_integer_predicate():
    @T.prim_func
    def kernel(B: T.Tensor((4,), "int32")):
        with T.Kernel(threads=4):
            for i in T.Parallel(4):
                B[i] = T.if_then_else(T.likely(i > 0), i // i, 0)

    b = np.empty(4, dtype=np.int32)
    reference(ntilang.compile(kernel), b)
    np.testing.assert_array_equal(b, [0, 1, 1, 1])


@pytest.mark.parametrize(
    "mode", ["empty", "python_value", "message", "selector", "fragment", "runtime_assert"]
)
def test_diagnostic_boundaries(mode):
    @T.prim_func
    def bad(A: T.Tensor((4,), "int32")):
        with T.Kernel(threads=4):
            if mode == "empty":
                T.print()
            elif mode == "python_value":
                T.print(7)
            elif mode == "message":
                T.print(msg=7)
            elif mode == "selector":
                T.print(A, warp_id=0.5)
            elif mode == "runtime_assert":
                assert A[0] > 0, "runtime"
            else:
                tile = T.alloc_fragment((4,), "int32")
                T.copy(A, tile)
                for i in T.Parallel(4):
                    T.print(tile)

    with pytest.raises(ntilang.CompileError, match="nonempty|expects|strings|integers|collective|host error"):
        ntilang.compile(bad)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("space", ["global", "shared", "fragment"])
def test_native_full_buffer_print(dtype, space):
    assert buffer_kernel(dtype, space).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("dtype", DTYPES)
def test_native_scalar_print_and_likely(dtype):
    assert scalar_kernel(dtype).build().has_gpu_module
    assert likely_kernel(dtype).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("dtype", ["bool", "int32", "uint64", "float16", "float32", "float64"])
def test_native_device_assert(dtype):
    assert assert_kernel(dtype=dtype).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
def test_native_zero_argument_and_empty_message_assert():
    assert message_kernel().build().has_gpu_module
    assert assert_kernel(no_stack=True, message="").build().has_gpu_module
    assert message_kernel("{}" * 40 + "%n 猫").build().has_gpu_module


@pytest.mark.cuda
@requires_cute
def test_standalone_diagnostics(tmp_path):
    path = buffer_kernel("float32", "fragment").save(tmp_path / "debug.py")
    code = (
        "import runpy, sys; module = runpy.run_path(sys.argv[1]); "
        "assert module['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_actual_generated_print_helper_on_cpu(dtype_name, tmp_path, capfd):
    import cutlass
    import cutlass.cute as cute

    dtype = getattr(cutlass, CUTLASS_TYPES[dtype_name])
    message = "literal %n %lld {x} {{}} ' \" \\ 猫" + "{}" * 40 + "\0ignored"
    path = scalar_kernel(dtype_name, message).save(tmp_path / "print.py")
    helper = runpy.run_path(path)["_nt_debug_print_0"]
    expected = (
        True
        if dtype_name == "bool"
        else 1.5
        if dtype_name.startswith(("float", "bfloat"))
        else int(np.iinfo(dtype_name).max)
    )

    @cute.jit
    def check(value: dtype):
        helper(cutlass.Int32(1), cutlass.Int32(2), cutlass.Int32(3), cutlass.Int32(7), value)

    executable = cute.compile(check, dtype(0), options="--enable-tvm-ffi --gpu-arch=sm_80")
    assert not executable.has_gpu_module
    executable(scalar_ffi_argument(expected, ScalarParameter("value", dtype_name)))
    ctypes.CDLL(None).fflush(None)
    output = capfd.readouterr().out
    assert "literal %n %lld {x} {{}} ' \" \\ 猫" in output
    assert "{}" * 40 in output
    assert "ignored" not in output
    assert "BlockIdx=(1, 2, 3), ThreadIdx=(7, 0, 0)" in output
    value_text = (
        "true" if dtype_name == "bool" else "1.500000" if isinstance(expected, float) else str(expected)
    )
    assert f"value={value_text}\n" in output
