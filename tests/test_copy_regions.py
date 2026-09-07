import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference

from examples.transpose import transpose


def global_regions():
    @T.prim_func
    def kernel(A: T.Tensor((97,), "float32"), B: T.Tensor((100,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            T.copy(A[3:40], B[2:39], prefer_instruction="sync")
            T.copy(A[40:77], B[60:97], disable_tma=True)

    return ntilang.compile(kernel)


def block_slices():
    @T.prim_func
    def kernel(A: T.Tensor((97,), "float32"), B: T.Tensor((97,), "float32")):
        with T.Kernel(4, threads=32) as bx:
            T.copy(A[bx * 32 : (bx + 1) * 32], B[bx * 32 : (bx + 1) * 32])

    return ntilang.compile(kernel)


def row_regions():
    @T.prim_func
    def kernel(A: T.Tensor((5, 19), "float32"), B: T.Tensor((5, 19), "float32")):
        with T.Kernel(5, threads=32) as bx:
            row = T.alloc_fragment((19,), "float32")
            T.copy(A[bx : bx + 1, :], row)
            T.copy(row, B[bx, :])

    return ntilang.compile(kernel)


def partial_tile(scope="fragment"):
    allocate = T.alloc_fragment if scope == "fragment" else T.alloc_shared

    @T.prim_func
    def kernel(
        A: T.Tensor((9, 13), "float32"), B: T.Tensor((11, 17), "float32"), C: T.Tensor((5, 7), "float32")
    ):
        with T.Kernel(1, threads=64) as _bx:
            tile = allocate((11, 17), "float32")
            T.fill(tile, -7.0)
            T.copy(A[1:6, 2:9], tile[3:8, 4:11])
            T.copy(tile, B)
            T.copy(tile[3:8, 4:11], C[:, :])

    return ntilang.compile(kernel)


def overlapping_temporary(scope="fragment"):
    allocate = T.alloc_fragment if scope == "fragment" else T.alloc_shared

    @T.prim_func
    def kernel(A: T.Tensor((97,), "float32"), B: T.Tensor((97,), "float32")):
        with T.Kernel(1, threads=64) as _bx:
            tile = allocate((97,), "float32")
            T.copy(A, tile)
            T.copy(tile[0:64], tile[16:80])
            T.copy(tile, B)

    return ntilang.compile(kernel)


def mma_regions():
    @T.prim_func
    def kernel(
        A: T.Tensor((32, 32), "float16"),
        B: T.Tensor((32, 32), "float16"),
        C: T.Tensor((5, 7), "float32"),
        D: T.Tensor((1, 32, 32), "float32"),
    ):
        with T.Kernel(1, threads=128) as _bx:
            sa = T.alloc_shared((32, 32), "float16")
            sb = T.alloc_shared((32, 32), "float16")
            acc = T.alloc_fragment((32, 32), "float32")
            small = T.alloc_fragment((5, 7), "float32")
            T.copy(A, sa)
            T.copy(B, sb)
            T.clear(acc)
            T.gemm(sa, sb, acc)
            T.copy(acc[2:7, 4:11], small)
            T.copy(small, acc[9:14, 11:18])
            T.copy(small, C)
            T.copy(acc, D[0, :, :])

    return ntilang.compile(kernel)


def scalar_copy():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "int32"), B: T.Tensor((39,), "float64")):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                T.copy(src=A[bx * 32 + i], dst=B[bx * 32 + i])

    return ntilang.compile(kernel)


def test_global_disjoint_regions_reference():
    a = np.arange(97, dtype=np.float32)
    b = np.full(100, -9, dtype=np.float32)
    reference(global_regions(), a, b)
    expected = np.full_like(b, -9)
    expected[2:39], expected[60:97] = a[3:40], a[40:77]
    np.testing.assert_array_equal(b, expected)


@pytest.mark.parametrize("factory,shape", [(block_slices, (97,)), (row_regions, (5, 19))])
def test_block_and_row_region_reference(factory, shape):
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    b = np.full_like(a, np.nan)
    reference(factory(), a, b)
    np.testing.assert_array_equal(b, a)


@pytest.mark.parametrize("scope", ["shared", "fragment"])
def test_partial_temporary_reference(scope):
    a = np.arange(117, dtype=np.float32).reshape(9, 13)
    b = np.empty((11, 17), dtype=np.float32)
    c = np.empty((5, 7), dtype=np.float32)
    reference(partial_tile(scope), a, b, c)
    expected = np.full_like(b, -7)
    expected[3:8, 4:11] = a[1:6, 2:9]
    np.testing.assert_array_equal(b, expected)
    np.testing.assert_array_equal(c, a[1:6, 2:9])


@pytest.mark.parametrize("scope", ["shared", "fragment"])
def test_overlapping_temporary_reference(scope):
    a = np.arange(97, dtype=np.float32)
    b = np.empty_like(a)
    reference(overlapping_temporary(scope), a, b)
    expected = a.copy()
    expected[16:80] = a[:64]
    np.testing.assert_array_equal(b, expected)


def test_mma_regions_reference():
    rng = np.random.default_rng(917)
    a, b = (rng.normal(size=(32, 32)).astype(np.float16) for _ in range(2))
    c = np.empty((5, 7), dtype=np.float32)
    d = np.empty((1, 32, 32), dtype=np.float32)
    reference(mma_regions(), a, b, c, d)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    selected = expected[2:7, 4:11].copy()
    expected[9:14, 11:18] = selected
    np.testing.assert_allclose(c, selected, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(d[0], expected, rtol=3e-5, atol=3e-5)


def test_scalar_copy_reference():
    a = np.arange(39, dtype=np.int32)
    b = np.empty(39, dtype=np.float64)
    reference(scalar_copy(), a, b)
    np.testing.assert_array_equal(b, a.astype(np.float64))


def test_partial_uninitialized_destination_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((32,), "float32")
            T.copy(A[0:16], tile[0:16])
            T.copy(tile, B)

    with pytest.raises(ntilang.CompileError, match="initialized destination"):
        ntilang.compile(bad)


def test_overlapping_global_write_sites_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            T.copy(A[0:16], B[0:16])
            T.copy(A[0:16], B[8:24])

    with pytest.raises(ntilang.CompileError, match="multiple write sites"):
        ntilang.compile(bad)


def test_narrowing_cast_cannot_change_slice_extent():
    @T.prim_func
    def bad(A: T.Tensor((256,), "float32"), B: T.Tensor((130,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            T.copy(A[0 : T.int8(130)], B[:])

    with pytest.raises(ntilang.CompileError, match="slice extents"):
        ntilang.compile(bad)


def test_shared_parallel_transpose_reference():
    a = np.arange(65 * 71, dtype=np.float32).reshape(65, 71)
    b = np.full((71, 65), np.nan, dtype=np.float32)
    reference(transpose(), a, b)
    np.testing.assert_array_equal(b, a.T)


def test_shared_cross_element_write_dependency_rejected():
    @T.prim_func
    def bad(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_shared((32,), "float32")
            T.clear(tile)
            for i in T.Parallel(32):
                tile[i] = tile[31 - i] + 1.0
            T.copy(tile, A)

    with pytest.raises(ntilang.CompileError, match="Cross-element shared reads"):
        ntilang.compile(bad)


def test_copy_annotation_precedence():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            T.copy(A, B, prefer_instruction="tma", annotations={"prefer_instruction": "sync"})

    assert ntilang.compile(kernel).source


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "factory",
    [
        global_regions,
        block_slices,
        row_regions,
        partial_tile,
        overlapping_temporary,
        mma_regions,
        scalar_copy,
        transpose,
    ],
)
def test_copy_regions_cute_compilation(factory):
    assert factory().build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_shared_partial_regions_cute_compilation():
    assert partial_tile("shared").build().has_gpu_module
    assert overlapping_temporary("shared").build().has_gpu_module
