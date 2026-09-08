import importlib.util
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def scalar_rebinding(dtype="float16"):
    @T.prim_func
    def kernel(A: T.Tensor((39,), dtype), B: T.Tensor((39,), "float64"), C: T.Tensor((39,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                value = A[i]
                saved = value
                value = value + 1
                value *= 3
                converted = T.float64(value)
                value = T.int32(7)
                value += i
                B[i] = converted + T.float64(saved)
                C[i] = value

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_rebinding_preserves_snapshots_and_changes_value_dtype(dtype):
    a = np.linspace(-3, 5, 39, dtype=dtype)
    b, c = np.empty(39, dtype=np.float64), np.empty(39, dtype=np.int32)
    reference(scalar_rebinding(dtype), a, b, c)
    expected = ((a + 1).astype(dtype) * 3).astype(dtype).astype(np.float64) + a.astype(np.float64)
    np.testing.assert_array_equal(b, expected)
    np.testing.assert_array_equal(c, np.arange(39) + 7)


@T.macro
def rebound_macro(value):
    original = value
    value += 3
    value *= 2
    return original, value


def macro_parameter_rebinding():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                value = A[i]
                before, after = rebound_macro(value)
                B[i] = before + after + value

    return ntilang.compile(kernel)


def test_macro_parameter_rebinding_keeps_caller_value():
    a = np.arange(32, dtype=np.int32)
    b = np.empty_like(a)
    reference(macro_parameter_rebinding(), a, b)
    np.testing.assert_array_equal(b, a * 4 + 6)


def scalar_parameter_rebinding():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "float32"), alpha: T.float32, B: T.Tensor((39,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            saved = alpha
            alpha = alpha * 2
            alpha += 1
            for i in T.Parallel(64):
                B[i] = A[i] * alpha + saved

    return ntilang.compile(kernel)


def test_kernel_parameter_rebinding_preserves_input_abi():
    a = np.arange(39, dtype=np.float32) / 4
    b = np.empty_like(a)
    reference(scalar_parameter_rebinding(), a, 1.5, b)
    np.testing.assert_array_equal(b, a * 4 + 1.5)


def construction_constants():
    @T.prim_func
    def kernel(B: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            width = 8
            width *= 2
            dtype = T.float32
            allocate = T.alloc_fragment
            mode = "selected"
            enabled = width == 16 and mode in ("selected", "other")
            if enabled:
                shape = (width,)
                ratio = 1 / 8
                tile = allocate(shape, dtype)
                for i in T.Parallel(width):
                    tile[i] = i * ratio
                T.copy(tile, B)
            else:
                T.unavailable_operation(B)

    return ntilang.compile(kernel)


def test_construction_constants_control_shapes_aliases_and_selected_source():
    b = np.empty(16, dtype=np.float32)
    reference(construction_constants(), b)
    np.testing.assert_array_equal(b, np.arange(16) / 8)


def construction_in_runtime_frames(count=3):
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            value = 1
            for _k in T.serial(count):
                value += 2
            remaining = T.alloc_var("int32", count)
            while remaining > 0:
                value += 4
                remaining -= 1
            for i in T.Parallel(32):
                if i < 16:
                    value += 8
                else:
                    value += 16
                B[i] = value

    return ntilang.compile(kernel)


@pytest.mark.parametrize("count", [0, 3])
def test_python_updates_follow_construction_order_across_runtime_frames(count):
    b = np.empty(32, dtype=np.int32)
    reference(construction_in_runtime_frames(count), b)
    np.testing.assert_array_equal(b, 31)


def loop_name_rebinding():
    @T.prim_func
    def kernel(B: T.Tensor((39,), "int32"), C: T.Tensor((39,), "int32")):
        with T.Kernel(2, threads=32) as bx:
            base = bx * 32
            bx = 41
            for i in T.Parallel(32):
                original = i
                i = i + 1
                B[base + original] = i + bx
            for i in T.Parallel(32):
                C[base + i] = i

    return ntilang.compile(kernel)


def test_loop_name_rebinding_preserves_ownership_and_allows_reuse():
    b, c = np.empty(39, dtype=np.int32), np.empty(39, dtype=np.int32)
    reference(loop_name_rebinding(), b, c)
    np.testing.assert_array_equal(b, np.arange(39) % 32 + 42)
    np.testing.assert_array_equal(c, np.arange(39) % 32)


def buffer_rebinding():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((32,), "int32")
            T.copy(A, tile)
            saved = tile
            tile = T.alloc_fragment((32,), "int32")
            T.fill(tile, 4)
            for i in T.Parallel(32):
                B[i] = saved[i] + tile[i]

    return ntilang.compile(kernel)


def test_reallocated_buffer_name_preserves_existing_alias():
    a = np.arange(32, dtype=np.int32)
    b = np.empty_like(a)
    reference(buffer_rebinding(), a, b)
    np.testing.assert_array_equal(b, a + 4)


def mutable_rebinding():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                value = T.alloc_var("int32", 2)
                saved = value
                value = T.alloc_var("int32", value + 3)
                value += i
                B[i] = value + saved

    return ntilang.compile(kernel)


def test_reallocated_mutable_name_reads_previous_initializer_and_snapshot():
    b = np.empty(32, dtype=np.int32)
    reference(mutable_rebinding(), b)
    np.testing.assert_array_equal(b, np.arange(32) + 7)


def chained_assignment():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32"), C: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                narrow = T.alloc_var("int8")
                narrow = B[i] = T.int32(200) + i
                C[i] = narrow

    return ntilang.compile(kernel)


def test_chained_assignment_captures_rhs_before_target_conversions():
    b, c = np.empty(32, dtype=np.int32), np.empty(32, dtype=np.int32)
    reference(chained_assignment(), b, c)
    np.testing.assert_array_equal(b, np.arange(32) + 200)
    np.testing.assert_array_equal(c, np.arange(32) - 56)


@T.macro
def escaped_binding(kind, source, output, index):
    if kind == "if":
        if index < 16:
            value = source[index]
        else:
            value = source[index] + 1
        output[index] = value
    if kind == "for":
        for _k in T.serial(2):
            value = source[index]
        output[index] = value
    if kind == "while":
        remaining = T.alloc_var("int32", 1)
        while remaining > 0:
            value = source[index]
            remaining -= 1
        output[index] = value
    if kind == "rebind":
        value = source[index]
        if index < 16:
            value = value + 1
        output[index] = value
    if kind == "buffer":
        if index < 16:
            alias = source
        output[index] = alias[index]


@pytest.mark.parametrize("kind", ["if", "for", "while", "rebind", "buffer"])
def test_runtime_binding_cannot_escape_defining_region(kind):
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                escaped_binding(kind, A, B, i)

    with pytest.raises(ntilang.CompileError, match="defining region"):
        ntilang.compile(bad)


shadowed_value = 29


def test_unbound_local_does_not_read_same_named_global():
    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = shadowed_value  # noqa: F823 - Exercise source-level unbound-local detection.
                shadowed_value = i  # noqa: F841 - The later assignment makes the name local.

    with pytest.raises(ntilang.CompileError, match="not bound"):
        ntilang.compile(bad)


requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize(
    "factory",
    [
        scalar_rebinding,
        macro_parameter_rebinding,
        scalar_parameter_rebinding,
        construction_constants,
        construction_in_runtime_frames,
        loop_name_rebinding,
        buffer_rebinding,
        mutable_rebinding,
        chained_assignment,
    ],
)
def test_native_binding_compilation(factory):
    assert factory().build().has_gpu_module


@pytest.mark.cuda
@requires_cute
def test_generated_binding_module_is_standalone(tmp_path):
    path = scalar_rebinding().save(tmp_path / "bindings.py")
    code = "import runpy, sys; module = runpy.run_path(sys.argv[1]); assert module['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)
