import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def early_for(unroll=False):
    loop = T.unroll if unroll else T.serial

    @T.prim_func
    def kernel(A: T.Tensor((39,), "int32"), B: T.Tensor((39,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                acc = T.alloc_var("int32")
                for k in loop(9):
                    if k >= A[i]:
                        break
                    if k % 2 == 0:
                        continue
                    term = k * 2
                    acc += term
                B[i] = acc

    return ntilang.compile(kernel)


@pytest.mark.parametrize("unroll", [False, True])
def test_for_break_continue_and_following_bindings(unroll):
    a = np.arange(39, dtype=np.int32) % 12 - 1
    b = np.empty_like(a)
    reference(early_for(unroll), a, b)
    np.testing.assert_array_equal(b, [sum(k * 2 for k in range(9) if k < limit and k % 2) for limit in a])


def early_while():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "int32"), B: T.Tensor((39,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(64):
                remaining = T.alloc_var("int32", A[i])
                total = T.alloc_var("int32")
                while remaining > 0:
                    current = remaining
                    remaining -= 1
                    if current % 2 == 0:
                        continue
                    total += current
                    if total > 10:
                        break
                B[i] = total

    return ntilang.compile(kernel)


def test_while_break_continue():
    a = np.arange(39, dtype=np.int32) % 12
    b = np.empty_like(a)
    reference(early_while(), a, b)
    expected = []
    for count in a:
        total = 0
        while count > 0:
            current = count
            count -= 1
            if current % 2 == 0:
                continue
            total += current
            if total > 10:
                break
        expected.append(total)
    np.testing.assert_array_equal(b, expected)


def no_predicate_after_break():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                divisor = T.alloc_var("int32", 2)
                while T.truncdiv(8, divisor) > 0:
                    divisor = 0
                    break
                B[i] = divisor

    return ntilang.compile(kernel)


def test_break_skips_next_condition_evaluation():
    b = np.empty(32, dtype=np.int32)
    reference(no_predicate_after_break(), b)
    np.testing.assert_array_equal(b, 0)


def nested_targets():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                acc = T.alloc_var("int32")
                for j in T.unroll(5, explicit=True):
                    if j == 2:
                        continue
                    for k in T.serial(6):
                        if k >= j:
                            break
                        acc += j + k
                B[i] = acc

    return ntilang.compile(kernel)


def test_nested_targets_with_explicit_outer_unroll():
    b = np.empty(32, dtype=np.int32)
    reference(nested_targets(), b)
    np.testing.assert_array_equal(b, sum(j + k for j in range(5) if j != 2 for k in range(j)))


def aliases(spelling="loop_break"):
    stop = T.loop_break if spelling == "loop_break" else T.break_loop

    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                acc = T.alloc_var("int32")
                for k in T.serial(5):
                    if k == 1:
                        T.continue_loop(span=None)
                    if k == 3:
                        stop()
                    acc += k
                B[i] = acc

    return ntilang.compile(kernel)


@pytest.mark.parametrize("spelling", ["loop_break", "break_loop"])
def test_control_intrinsic_statement_forms(spelling):
    b = np.empty(32, dtype=np.int32)
    reference(aliases(spelling), b)
    np.testing.assert_array_equal(b, 2)


def test_unreachable_statements_are_ignored():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                acc = T.alloc_var("int32", 7)
                for k in T.serial(3):
                    break
                    T.fill(B, 99)
                B[i] = acc

    b = np.empty(32, dtype=np.int32)
    reference(ntilang.compile(kernel), b)
    np.testing.assert_array_equal(b, 7)


def test_early_exit_does_not_initialize_fragment():
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((32,), "int32")
            for k in T.serial(3):
                if A[0] > 0:
                    break
                T.clear(tile)
            T.copy(tile, B)

    with pytest.raises(ntilang.CompileError, match="before initialization"):
        ntilang.compile(bad)


def test_explicit_unroll_rejects_targeted_break():
    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                for k in T.unroll(3, explicit=True):
                    if k > i:
                        break
                B[i] = i

    with pytest.raises(ntilang.CompileError, match="explicitly expanded"):
        ntilang.compile(bad)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "factory", [early_for, early_while, no_predicate_after_break, nested_targets, aliases]
)
def test_loop_control_compilation(factory):
    assert factory().build().has_gpu_module
