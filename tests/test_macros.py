from __future__ import annotations

import importlib.util
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


@T.macro
def shifted_square(value: T.int32, shift=1):
    temporary = value + shift
    return temporary * temporary


@T.macro()
def nested_scalar(value, /, *extra, factor=2, **options):
    temporary = shifted_square(value, shift=extra[0])
    return temporary * factor + options["bias"]


def scalar_macros():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "float32"), B: T.Tensor((39,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                temporary = 7
                value = A[bx * 32 + i]
                first = nested_scalar(value, 3, factor=2, bias=temporary)
                second = nested_scalar(*(value, 1), **{"factor": 4, "bias": 2})
                B[bx * 32 + i] = first + second

    return ntilang.compile(kernel)


def test_macro_names_arguments_and_annotations():
    a = np.linspace(-2, 2, 39, dtype=np.float32)
    b = np.empty_like(a)
    reference(scalar_macros(), a, b)
    np.testing.assert_allclose(b, 2 * (a + 3) ** 2 + 7 + 4 * (a + 1) ** 2 + 2, rtol=2e-7)


def closure_macro(dtype="float32"):
    factor = 3

    @T.macro
    def scale(value):
        local: dtype = value * factor
        return local

    @T.prim_func
    def kernel(A: T.Tensor((32,), dtype), B: T.Tensor((32,), dtype)):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                factor = 19
                B[i] = scale(A[i]) + factor

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_macro_definition_closure_and_deferred_dtype(dtype):
    a = np.arange(32, dtype=dtype) / 4
    b = np.empty_like(a)
    reference(closure_macro(dtype), a, b)
    np.testing.assert_array_equal(b, a * 3 + 19)


@T.macro
def tile_scale(source, shape, dtype, factor):
    tile = T.alloc_fragment(shape, dtype)
    T.copy(source[0], tile)
    for i in T.Parallel(shape[0]):
        tile[i] *= factor
    return tile


@T.macro
def two_tiles(source):
    first = tile_scale(source, (64,), "float32", 2)
    second = tile_scale(source, (64,), "float32", 3)
    return first, second


def buffer_returns():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "float32"), B: T.Tensor((39,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            left, right = two_tiles(A)
            alias = left
            for i in T.Parallel(64):
                B[i] = alias[i] + right[i]

    return ntilang.compile(kernel)


def test_macro_buffer_allocation_return_and_tuple_unpacking():
    a = np.arange(39, dtype=np.float32) * 0.5
    b = np.empty_like(a)
    kernel = buffer_returns()
    reference(kernel, a, b)
    np.testing.assert_array_equal(b, 5 * a)
    assert len(kernel.ir.buffers) == 2
    assert len({buffer.name for buffer in kernel.ir.buffers}) == 2


@T.macro
def increment(value: T.Ref, amount=1):
    value += amount
    return value


@T.macro
def value_and_reference(value, target: T.Ref):
    target += 7
    return value, target


def scalar_references():
    @T.prim_func
    def kernel(B: T.Tensor((39, 3), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i, j in T.Parallel(64, 3):
                local = T.alloc_var("int32", i)
                before, after = value_and_reference(local, local)
                again = increment(local, amount=2)
                B[i, j] = T.Select(j == 0, before, T.Select(j == 1, after, again))

    return ntilang.compile(kernel)


def test_macro_scalar_value_capture_reference_updates_and_tuple_results():
    b = np.empty((39, 3), dtype=np.int32)
    reference(scalar_references(), b)
    np.testing.assert_array_equal(b, np.arange(39)[:, None] + [0, 7, 9])


@T.macro
def element_update(target: T.Ref, index: T.Ref):
    index += 1
    target += 10


@T.macro
def element_write(target: T.Ref, value):
    target = value  # noqa: F841 - Ref assignment emits a store to the caller's element.


def element_references():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "int32"), B: T.Tensor((39,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((64,), "int32")
            T.copy(A[0], tile)
            for i in T.Parallel(64):
                index = T.alloc_var("int32", i)
                element_update(tile[i], index)
                element_write(B[i], tile[i] + index)

    return ntilang.compile(kernel)


def test_macro_fragment_reference_and_global_output_reference():
    a = np.arange(39, dtype=np.int32) * 2
    b = np.empty_like(a)
    reference(element_references(), a, b)
    np.testing.assert_array_equal(b, a + np.arange(39) + 11)


@T.macro
def copy_region(source: T.Ref, destination, index: T.Ref):
    index += 4
    T.copy(source, destination)


def region_references():
    @T.prim_func
    def kernel(A: T.Tensor((96,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            start = T.alloc_var("int16", 8)
            tile = T.alloc_fragment((32,), "int32")
            copy_region(A[start : start + 32], tile, start)
            for i in T.Parallel(32):
                B[i] = tile[i] + start

    return ntilang.compile(kernel)


def test_macro_region_reference_captures_origin_before_mutation():
    a = np.arange(96, dtype=np.int32)
    b = np.empty(32, dtype=np.int32)
    reference(region_references(), a, b)
    np.testing.assert_array_equal(b, a[8:40] + 12)


@T.macro
def loop_accumulate(value):
    total = T.alloc_var("int32")
    for k in T.serial(7):
        if k >= value:
            break
        if k == 2:
            continue
        total += k
    return total


def macro_control_flow():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "int32"), B: T.Tensor((39,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                result = loop_accumulate(A[i])
                B[i] = result

    return ntilang.compile(kernel)


def test_macro_mutable_return_and_internal_early_exits():
    a = np.arange(39, dtype=np.int32) % 10
    b = np.empty_like(a)
    reference(macro_control_flow(), a, b)
    np.testing.assert_array_equal(b, [sum(k for k in range(min(n, 7)) if k != 2) for n in a])


@T.macro
def bounded_expansion(value, depth):
    if depth == 0:
        return value
    return bounded_expansion(value + 1, depth - 1)


def recursive_macro():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = bounded_expansion(A[i], 4)

    return ntilang.compile(kernel)


def test_static_macro_branches_and_bounded_recursive_expansion():
    a = np.arange(32, dtype=np.float32) * 0.5
    b = np.empty_like(a)
    reference(recursive_macro(), a, b)
    np.testing.assert_array_equal(b, a + 4)


@T.macro
def condition_once(counter: T.Ref, bound: T.Ref):
    counter += 1
    return bound > 0


def macro_while_condition():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                counter = T.alloc_var("int32")
                bound = T.alloc_var("int32", i % 5)
                while condition_once(counter, bound):
                    bound -= 1
                B[i] = counter

    return ntilang.compile(kernel)


def test_macro_expansion_precedes_while_condition():
    b = np.empty(32, dtype=np.int32)
    reference(macro_while_condition(), b)
    np.testing.assert_array_equal(b, 1)


@T.macro
def choose_value(counter: T.Ref, amount):
    counter += amount
    return counter


def macro_eager_arguments():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                counter = T.alloc_var("int32")
                result = T.if_then_else(i % 2 == 0, choose_value(counter, 1), choose_value(counter, 10))
                B[i] = result + counter

    return ntilang.compile(kernel)


def test_macro_call_bodies_expand_before_scalar_intrinsic():
    b = np.empty(32, dtype=np.int32)
    reference(macro_eager_arguments(), b)
    # Each return is an unbound local.var load; both read the final counter value.
    np.testing.assert_array_equal(b, 22)


@T.macro
def record_buffer(counter: T.Ref, buffer):
    counter = counter * 10 + 1
    return buffer


@T.macro
def record_index(counter: T.Ref, index):
    counter = counter * 10 + 2
    return index


@T.macro
def record_value(counter: T.Ref):
    counter = counter * 10 + 3
    return counter


@T.macro
def ordered_store(mode, output, index, counter: T.Ref):
    if mode == "direct":
        record_buffer(counter, output)[record_index(counter, index)] = record_value(counter)
    else:
        record_buffer(counter, output)[record_index(counter, index)] = record_value(counter) + 0


def macro_store_order(mode="direct"):
    @T.prim_func
    def kernel(B: T.Tensor((39,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                counter = T.alloc_var("int32")
                ordered_store(mode, B, i, counter)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("mode", ["direct", "expression"])
def test_macro_store_expands_buffer_then_index_then_value(mode):
    b = np.empty(39, dtype=np.int32)
    reference(macro_store_order(mode), b)
    np.testing.assert_array_equal(b, 123)


def macro_tuple_store_order():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32"), C: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                counter = T.alloc_var("int32")
                counter, B[record_index(counter, i)] = record_value(counter), counter
                C[i] = counter

    return ntilang.compile(kernel)


def test_macro_tuple_snapshots_precede_ordered_target_updates():
    b = np.empty(32, dtype=np.int32)
    c = np.empty_like(b)
    reference(macro_tuple_store_order(), b, c)
    # The RHS first sets and snapshots 3; target binding then sets 3 and records 2.
    np.testing.assert_array_equal(b, 3)
    np.testing.assert_array_equal(c, 32)


@T.macro
def read_element_and_step(source: T.Ref, index: T.Ref):
    index += 1
    return source


def macro_element_index_capture():
    @T.prim_func
    def kernel(A: T.Tensor((64,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                index = T.alloc_var("int16", i)
                value = read_element_and_step(A[index], index)
                B[i] = value + index * 100

    return ntilang.compile(kernel)


def test_macro_element_reference_snapshots_mutable_index():
    a = np.arange(64, dtype=np.int32) ** 2
    b = np.empty(32, dtype=np.int32)
    reference(macro_element_index_capture(), a, b)
    np.testing.assert_array_equal(b, a[:32] + (np.arange(32) + 1) * 100)


def test_runtime_boolean_macro_branch_is_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "bool")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = A[i] > 0 and shifted_square(A[i]) > 2

    with pytest.raises(ntilang.CompileError, match="runtime Boolean branches"):
        ntilang.compile(bad)


def test_runtime_macro_return_inside_control_flow_is_rejected():
    @T.macro
    def invalid(value):
        if value > 0:
            return value
        return 0

    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = invalid(A[i])

    with pytest.raises(ntilang.CompileError, match="returns inside control flow"):
        ntilang.compile(bad)


@pytest.mark.parametrize(
    "call,pattern",
    [("missing", "missing a required"), ("extra", "too many positional"), ("reference", "T.Ref arguments")],
)
def test_macro_argument_diagnostics(call, pattern):
    @T.macro
    def dispatch(kind, value):
        if kind == "missing":
            return shifted_square()
        if kind == "extra":
            return shifted_square(value, 1, 2)
        return increment(value)

    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = dispatch(call, i)

    with pytest.raises(ntilang.CompileError, match=pattern):
        ntilang.compile(bad)


def test_macro_python_body_is_never_executed(tmp_path):
    path = str(tmp_path / "must-not-exist")

    @T.macro
    def invalid(value):
        open(path, "w")
        return value

    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = invalid(i)

    with pytest.raises(ntilang.CompileError, match="Only ntilang.language"):
        ntilang.compile(bad)
    assert not (tmp_path / "must-not-exist").exists()


requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize(
    "factory",
    [
        scalar_macros,
        closure_macro,
        buffer_returns,
        scalar_references,
        element_references,
        region_references,
        macro_control_flow,
        recursive_macro,
        macro_while_condition,
        macro_eager_arguments,
        macro_store_order,
        macro_tuple_store_order,
        macro_element_index_capture,
    ],
)
def test_native_macro_compilation(factory):
    assert factory().build().has_gpu_module


@pytest.mark.cuda
@requires_cute
def test_macro_generated_module_is_standalone(tmp_path):
    path = buffer_returns().save(tmp_path / "macros.py")
    code = (
        "import runpy, sys; module = runpy.run_path(sys.argv[1]); "
        "assert module['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)
