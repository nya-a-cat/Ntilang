import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def segmented_sum(unroll=False):
    loop = T.unroll if unroll else T.serial

    @T.prim_func
    def kernel(
        A: T.Tensor((39, 7), "float32"), Length: T.Tensor((39,), "int32"), B: T.Tensor((39,), "float32")
    ):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                stop = T.max(0, T.min(Length[i], 7))
                acc = T.alloc_var("float32")
                for k in loop(stop):
                    acc += A[i, k]
                B[i] = acc

    return ntilang.compile(kernel)


@pytest.mark.parametrize("unroll", [False, True])
def test_segmented_runtime_trip_counts(unroll):
    a = np.arange(39 * 7, dtype=np.float32).reshape(39, 7)
    length = np.arange(39, dtype=np.int32) % 12 - 2
    b = np.empty(39, dtype=np.float32)
    reference(segmented_sum(unroll), a, length, b)
    np.testing.assert_array_equal(b, [row[: max(0, min(n, 7))].sum() for row, n in zip(a, length)])


def descending_runtime_range():
    @T.prim_func
    def kernel(A: T.Tensor((39, 7), "int32"), Start: T.Tensor((39,), "int32"), B: T.Tensor((39,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                acc = T.alloc_var("int32")
                for k in T.serial(start=T.min(Start[i], 6), stop=-1, step=-2):
                    acc += A[i, k]
                B[i] = acc

    return ntilang.compile(kernel)


def test_dynamic_negative_step_and_empty_range():
    a = np.arange(39 * 7, dtype=np.int32).reshape(39, 7)
    start = np.arange(39, dtype=np.int32) % 12 - 3
    start[0] = -(2**31)
    b = np.empty(39, dtype=np.int32)
    reference(descending_runtime_range(), a, start, b)
    np.testing.assert_array_equal(
        b, [sum(row[k] for k in range(min(n, 6), -1, -2)) for row, n in zip(a, start)]
    )


def captured_mutable_bound():
    @T.prim_func
    def kernel(Length: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                stop = T.alloc_var("int32", Length[i])
                acc = T.alloc_var("int32")
                for k in T.serial(T.max(0, T.min(stop, 7))):
                    stop = 0
                    acc += k
                B[i] = acc

    return ntilang.compile(kernel)


def test_runtime_bounds_are_captured_before_iteration():
    length = np.arange(32, dtype=np.int32) % 12 - 2
    b = np.empty_like(length)
    reference(captured_mutable_bound(), length, b)
    n = np.clip(length, 0, 7)
    np.testing.assert_array_equal(b, n * (n - 1) // 2)


def masked_gather(index_dtype="int32"):
    @T.prim_func
    def kernel(
        A: T.Tensor((32,), "float32"), Index: T.Tensor((39,), index_dtype), B: T.Tensor((39,), "float32")
    ):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                B[i] = A[Index[i]]

    return ntilang.compile(kernel)


def test_integer_gather_is_guarded_at_both_load_levels():
    a = np.arange(32, dtype=np.float32) + 0.5
    index = np.arange(39, dtype=np.int32) - 3
    index[:2] = [-(2**31), 2**31 - 1]
    b = np.empty(39, dtype=np.float32)
    reference(masked_gather(), a, index, b)
    np.testing.assert_array_equal(b, [a[k] if 0 <= k < 32 else 0 for k in index])


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16"])
def test_narrow_integer_gather(dtype):
    a = np.arange(32, dtype=np.float32) + 0.5
    index = (np.arange(39, dtype=np.int32) - 3).astype(dtype)
    b = np.empty(39, dtype=np.float32)
    reference(masked_gather(dtype), a, index, b)
    np.testing.assert_array_equal(b, [a[int(k)] if 0 <= int(k) < 32 else 0 for k in index])


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16"])
def test_narrow_integer_gather_compilation(dtype):
    assert masked_gather(dtype).build().has_gpu_module


def mutable_gather():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                index = T.alloc_var("int32", 0)
                for k in T.serial(4):
                    index += k
                B[i] = A[index]

    return ntilang.compile(kernel)


def test_mutable_gather_uses_current_value():
    a = np.arange(32, dtype=np.float32)
    b = np.empty_like(a)
    reference(mutable_gather(), a, b)
    np.testing.assert_array_equal(b, a[6])


def test_dynamic_iteration_count_overflow_is_rejected():
    @T.prim_func
    def bad(Stop: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                acc = T.alloc_var("int32")
                for k in T.serial(-(2**31), Stop[i]):
                    acc += k
                B[i] = acc

    with pytest.raises(ntilang.CompileError, match="iteration count"):
        ntilang.compile(bad)


def test_explicit_unroll_requires_static_bounds():
    @T.prim_func
    def bad(Stop: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                acc = T.alloc_var("int32")
                for k in T.unroll(T.min(Stop[i], 7), explicit=True):
                    acc += k
                B[i] = acc

    with pytest.raises(ntilang.CompileError, match="static loop bounds"):
        ntilang.compile(bad)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "factory",
    [
        segmented_sum,
        lambda: segmented_sum(True),
        descending_runtime_range,
        captured_mutable_bound,
        masked_gather,
        mutable_gather,
    ],
)
def test_dynamic_loops_and_gathers_compile(factory):
    assert factory().build().has_gpu_module
