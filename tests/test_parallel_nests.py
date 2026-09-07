import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def nested_pointwise(rank=2):
    if rank == 2:

        @T.prim_func
        def kernel(A: T.Tensor((7, 13), "float32"), B: T.Tensor((7, 13), "float32")):
            with T.Kernel(1, threads=32) as _bx:
                for i in T.Parallel(8):
                    for j in T.Parallel(16):
                        B[i, j] = A[i, j] * 2 + i - j

    else:
        parallel = T.Parallel

        @T.prim_func
        def kernel(A: T.Tensor((3, 5, 7), "float32"), B: T.Tensor((3, 5, 7), "float32")):
            with T.Kernel(1, threads=32) as _bx:
                for i in parallel(4):
                    for j, k in parallel(8, 8):
                        B[i, j, k] = A[i, j, k] * 2 + i - j + k

    return ntilang.compile(kernel)


@pytest.mark.parametrize("rank", [2, 3])
def test_nested_parallel_coordinates_and_tail_masks(rank):
    shape = (7, 13) if rank == 2 else (3, 5, 7)
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    b = np.empty_like(a)
    reference(nested_pointwise(rank), a, b)
    coords = np.indices(shape)
    expected = a * 2 + coords[0] - coords[1]
    if rank == 3:
        expected += coords[2]
    np.testing.assert_array_equal(b, expected)


def nested_temporaries(space="fragment"):
    allocate = T.alloc_fragment if space == "fragment" else T.alloc_shared

    @T.prim_func
    def kernel(A: T.Tensor((8, 8), "float32"), B: T.Tensor((8, 8), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = allocate((8, 8), "float32")
            for i in T.Parallel(8):
                for j in T.Parallel(8):
                    acc = T.alloc_var("float32", A[i, j])
                    for k in T.serial(3):
                        acc += k
                    tile[i, j] = acc
            for i in T.Parallel(8):
                for j in T.Parallel(8):
                    B[i, j] = tile[j, i]

    return ntilang.compile(kernel)


@pytest.mark.parametrize("space", ["fragment", "shared"])
def test_nested_parallel_temporaries_mutation_and_cross_thread_reads(space):
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    b = np.empty_like(a)
    reference(nested_temporaries(space), a, b)
    np.testing.assert_array_equal(b, a.T + 3)


def test_nested_parallel_extent_cannot_capture_shadowed_induction_name():
    i = 8

    @T.prim_func
    def kernel(B: T.Tensor((i, i), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(8):
                for j in T.Parallel(i):
                    B[i, j] = 1

    with pytest.raises(ntilang.CompileError, match="static rectangular"):
        ntilang.compile(kernel)


def test_nested_parallel_duplicate_targets_are_rejected():
    @T.prim_func
    def kernel(B: T.Tensor((8, 8), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(8):
                for i in T.Parallel(8):
                    B[i, i] = 1

    with pytest.raises(ntilang.CompileError, match="unique"):
        ntilang.compile(kernel)


def test_imperfect_parallel_nest_remains_explicitly_rejected():
    @T.prim_func
    def kernel(B: T.Tensor((8, 8), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(8):
                value = i + 1
                for j in T.Parallel(8):
                    B[i, j] = value

    with pytest.raises(ntilang.CompileError, match="contiguous rectangular"):
        ntilang.compile(kernel)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "factory",
    [nested_pointwise, lambda: nested_pointwise(3), nested_temporaries, lambda: nested_temporaries("shared")],
)
def test_nested_parallel_native_compilation(factory):
    assert factory().build().has_gpu_module
