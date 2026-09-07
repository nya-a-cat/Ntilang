import itertools
import random

import ntilang
import ntilang.language as T
import pytest
from ntilang.ir import Expr, Partition
from ntilang.validation import check_ownership


def test_duplicate_writes_are_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                A[0] = 1.0

    with pytest.raises(ntilang.CompileError, match="unique global write"):
        ntilang.compile(bad)


def test_overlapping_blocks_are_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(2, threads=32) as _bx:
            for i in T.Parallel(32):
                A[i] = 1.0

    with pytest.raises(ntilang.CompileError, match="unique global write"):
        ntilang.compile(bad)


def test_index_overflow_is_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(3, threads=32) as bx:
            for i in T.Parallel(32):
                A[bx * 2_000_000_000 + i] = 1.0

    with pytest.raises(ntilang.CompileError, match="overflow"):
        ntilang.compile(bad)


def test_data_dependent_index_is_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = A[A[i]]

    with pytest.raises(ntilang.CompileError, match="Data-dependent"):
        ntilang.compile(bad)


def test_input_output_read_write_is_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                A[i] = A[i] + 1.0

    with pytest.raises(ntilang.CompileError, match="both read and written"):
        ntilang.compile(bad)


def test_reassigned_loop_variable_is_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            _bx = 1
            for i in T.Parallel(32):
                A[i] = 1.0

    with pytest.raises(ntilang.CompileError, match="Cannot assign"):
        ntilang.compile(bad)


def test_alias_indices_can_be_proved():
    @T.prim_func
    def good(A: T.Tensor((63,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                idx = bx * 32 + i
                A[idx] = 1.0

    assert ntilang.compile(good).source


def test_partition_rounding_cannot_overflow():
    with pytest.raises(ntilang.CompileError, match="Rounded tile partition"):
        Partition((2**31 - 1,), 96)


def test_shared_memory_limit_includes_alignment():
    @T.prim_func
    def bad(A: T.Tensor((1,), "float16")):
        with T.Kernel(1, threads=32) as _bx:
            small = T.alloc_shared((1,), "float16")
            large = T.alloc_shared((24575,), "float16")
            T.clear(small)
            T.clear(large)
            T.copy(small, A[0])

    with pytest.raises(ntilang.CompileError, match="48 KiB"):
        ntilang.compile(bad)


def test_write_ownership_rule_against_exhaustive_integer_maps():
    rng = random.Random(127)
    bounds = {"x": (0, 2), "y": (0, 3), "z": (0, 1)}
    names = tuple(bounds)
    accepted = 0
    for _ in range(500):
        coefficients = [[rng.randint(-12, 12) for _ in names] for _ in range(2)]
        indices = []
        for row in coefficients:
            e = Expr("const", value=0)
            for name, coefficient in zip(names, row):
                e = Expr("+", (e, Expr("*", (Expr("const", value=coefficient), Expr("var", value=name)))))
            indices.append(e)
        try:
            check_ownership(indices, bounds, {})
        except ntilang.CompileError:
            continue
        accepted += 1
        addresses = [
            tuple(sum(c * x for c, x in zip(row, point)) for row in coefficients)
            for point in itertools.product(range(3), range(4), range(2))
        ]
        assert len(addresses) == len(set(addresses))
    assert accepted > 10
