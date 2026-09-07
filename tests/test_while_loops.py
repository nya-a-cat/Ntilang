import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def triangular_counts():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "int32"), B: T.Tensor((39,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                remaining = T.alloc_var("int32", A[i])
                total = T.alloc_var("int32")
                while remaining > 0:
                    total += remaining
                    remaining -= 1
                B[i] = total

    return ntilang.compile(kernel)


@pytest.mark.parametrize("offset", [-3, 0, 2])
def test_lane_dependent_iteration_counts(offset):
    a = (np.arange(39, dtype=np.int32) % 8) + offset
    b = np.empty_like(a)
    reference(triangular_counts(), a, b)
    positive = np.maximum(a, 0)
    np.testing.assert_array_equal(b, positive * (positive + 1) // 2)


def fragment_predicate():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "int32"), B: T.Tensor((39,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((64,), "int32")
            T.copy(A[0], tile)
            for i in T.Parallel(64):
                while tile[i] > 0:
                    tile[i] -= 2
            T.copy(tile, B[0])

    return ntilang.compile(kernel)


def test_fragment_predicate_is_reloaded_after_updates():
    a = np.arange(39, dtype=np.int32) % 7 - 2
    b = np.empty_like(a)
    reference(fragment_predicate(), a, b)
    np.testing.assert_array_equal(b, np.where(a > 0, -(a % 2), a))


def nested_uniform_while():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            remaining = T.alloc_var("int32", 3)
            total = T.alloc_var("int32")
            while remaining > 0:
                inner = T.alloc_var("int32", 2)
                while inner > 0:
                    total += remaining
                    inner -= 1
                remaining -= 1
            for i in T.Parallel(32):
                B[i] = total

    return ntilang.compile(kernel)


def test_nested_loop_state():
    b = np.empty(32, dtype=np.int32)
    reference(nested_uniform_while(), b)
    np.testing.assert_array_equal(b, 12)


def uniform_collectives():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            shared = T.alloc_shared((32,), "int32")
            remaining = T.alloc_var("int32", A[0])
            T.clear(shared)
            while remaining > 0:
                T.fill(shared, remaining)
                remaining -= 1
            T.copy(shared, B)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("count", [0, 1, 5])
def test_uniform_collectives_inside_while(count):
    a = np.full(32, count, dtype=np.int32)
    b = np.empty_like(a)
    reference(uniform_collectives(), a, b)
    np.testing.assert_array_equal(b, int(count > 0))


def test_zero_iteration_does_not_initialize_fragment():
    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((32,), "int32")
            remaining = T.alloc_var("int32")
            while remaining > 0:
                T.clear(tile)
                remaining -= 1
            T.copy(tile, B)

    with pytest.raises(ntilang.CompileError, match="before initialization"):
        ntilang.compile(bad)


def test_static_true_while_is_rejected():
    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            while True:
                pass
            for i in T.Parallel(32):
                B[i] = 0

    with pytest.raises(ntilang.CompileError, match="infinite loop"):
        ntilang.compile(bad)


def test_while_requires_boolean_condition():
    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            remaining = T.alloc_var("int32", 2)
            while remaining:
                remaining -= 1
            for i in T.Parallel(32):
                B[i] = 0

    with pytest.raises(ntilang.CompileError, match="Boolean"):
        ntilang.compile(bad)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "factory", [triangular_counts, fragment_predicate, nested_uniform_while, uniform_collectives]
)
def test_while_compilation(factory):
    assert factory().build().has_gpu_module
