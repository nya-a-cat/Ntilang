"""Tests for M1 zero-dependency drop-in TileLang compatibility (import ntilang as tilelang)."""

from __future__ import annotations

import importlib.util

import ntilang
import ntilang as tilelang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def test_patch_tilelang_context_manager():
    with ntilang.patch_tilelang():
        import tilelang as patched_tl
        import tilelang.language as patched_lang
        import tilelang.layout as patched_layout

        assert patched_tl is ntilang
        assert patched_lang is T
        assert patched_layout is ntilang.layout


def test_direct_jit_and_get_kernel_source():
    @tilelang.jit(out_idx=[-1], target="auto")
    def direct_add(
        A: T.Tensor((128,), "float32"),
        B: T.Tensor((128,), "float32"),
        C: T.Tensor((128,), "float32"),
    ):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(128):
                C[bx * 128 + i] = A[bx * 128 + i] + B[bx * 128 + i]

    assert direct_add.out_idx == (2,)
    assert "def compile_kernel():" in direct_add.get_kernel_source()
    a = np.arange(128, dtype=np.float32)
    b = np.full(128, 3.0, dtype=np.float32)
    c = np.zeros_like(a)
    reference(direct_add, a, b, c)
    np.testing.assert_array_equal(c, a + b)


def test_factory_jit_caches_and_supports_swizzle_clear_accum_and_annotate_layout():
    @tilelang.jit(out_idx=[-1], target="cuda:sm_80")
    def tiled_gemm(m, n, k, block_m=32, block_n=32, block_k=32):
        @T.prim_func
        def main(
            A: T.Tensor((m, k), "float16"),
            B: T.Tensor((k, n), "float16"),
            C: T.Tensor((m, n), "float32"),
        ):
            with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_m, block_k), "float16")
                B_shared = T.alloc_shared((block_k, block_n), "float16")
                C_local = T.alloc_fragment((block_m, block_n), "float32")
                T.use_swizzle(panel_size=2, order="row")
                T.annotate_layout({A_shared: tilelang.layout.make_swizzled_layout(A_shared)})
                T.clear(C_local)
                for ko in T.serial(T.ceildiv(k, block_k)):
                    T.copy(A[bx * block_m, ko * block_k], A_shared)
                    T.copy(B[ko * block_k, by * block_n], B_shared)
                    T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.Square)
                T.copy(C_local, C[bx * block_m, by * block_n])

        return main

    k1 = tiled_gemm(65, 71, 37)
    k2 = tiled_gemm(65, 71, 37)
    assert k1 is k2
    assert k1.out_idx == (2,)
    assert "_nt_swz_pid_" in k1.source
    rng = np.random.default_rng(11)
    a = rng.normal(size=(65, 37)).astype(np.float16)
    b = rng.normal(size=(37, 71)).astype(np.float16)
    c = np.full((65, 71), np.nan, dtype=np.float32)
    reference(k1, a, b, c)
    np.testing.assert_allclose(c, a.astype(np.float32) @ b.astype(np.float32), rtol=3e-5, atol=3e-5)


def test_gemm_clear_accum_and_infinity_and_alloc_local():
    @T.prim_func
    def kernel(
        A: T.Tensor((32, 32), "float16"),
        B: T.Tensor((32, 32), "float16"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1, 1, threads=128):
            sa = T.alloc_shared((32, 32), "float16")
            sb = T.alloc_shared((32, 32), "float16")
            acc = T.alloc_local((32, 32), "float32")
            T.copy(A[0, 0], sa)
            T.copy(B[0, 0], sb)
            T.gemm(sa, sb, acc, clear_accum=True)
            for i, j in T.Parallel(32, 32):
                acc[i, j] = T.max(acc[i, j], -T.infinity("float32"))
            T.copy(acc, C[0, 0])

    compiled = ntilang.compile(kernel)
    rng = np.random.default_rng(12)
    a = rng.normal(size=(32, 32)).astype(np.float16)
    b = rng.normal(size=(32, 32)).astype(np.float16)
    c = np.full((32, 32), np.nan, dtype=np.float32)
    reference(compiled, a, b, c)
    np.testing.assert_allclose(c, a.astype(np.float32) @ b.astype(np.float32), rtol=3e-5, atol=3e-5)


def test_dynamic_dim_phase_a_specialization():
    M = T.dynamic("m")

    @tilelang.jit(out_idx=[-1], target="sm_80")
    def dyn_scale(
        A: T.Tensor((M, 32), "float32"),
        B: T.Tensor((M, 32), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, 4), threads=128) as bx:
            for i, j in T.Parallel(4, 32):
                B[bx * 4 + i, j] = A[bx * 4 + i, j] * 2.0

    a1 = np.ones((10, 32), dtype=np.float32)
    a2 = np.ones((18, 32), dtype=np.float32)
    k1 = dyn_scale.specialize(a1)
    k1_cached = dyn_scale.specialize(a1)
    k2 = dyn_scale.specialize(a2)
    assert k1 is k1_cached
    assert k1 is not k2
    out1 = np.empty_like(a1)
    reference(k1, a1, out1)
    np.testing.assert_array_equal(out1, a1 * 2.0)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_swizzled_gemm_compiles_natively():
    @tilelang.jit(out_idx=[-1], target="sm_80")
    def swz_gemm(m, n, k):
        @T.prim_func
        def main(
            A: T.Tensor((m, k), "float16"),
            B: T.Tensor((k, n), "float16"),
            C: T.Tensor((m, n), "float32"),
        ):
            with T.Kernel(T.ceildiv(m, 32), T.ceildiv(n, 32), threads=128) as (bx, by):
                sa = T.alloc_shared((32, 32), "float16")
                sb = T.alloc_shared((32, 32), "float16")
                acc = T.alloc_fragment((32, 32), "float32")
                T.use_swizzle(panel_size=2, order="col")
                T.clear(acc)
                for ko in T.serial(T.ceildiv(k, 32)):
                    T.copy(A[bx * 32, ko * 32], sa)
                    T.copy(B[ko * 32, by * 32], sb)
                    T.gemm(sa, sb, acc)
                T.copy(acc, C[bx * 32, by * 32])

        return main

    assert swz_gemm(65, 71, 37).build().has_gpu_module


@pytest.mark.gpu
def test_dropin_tilelang_gemm_and_dynamic_on_gpu(torch_cuda):
    torch = torch_cuda
    with ntilang.patch_tilelang():
        import tilelang as tl
        import tilelang.language as TL

        @tl.jit(out_idx=[-1], target="auto")
        def gemm_dropin(m, n, k, bm=32, bn=32, bk=32):
            @TL.prim_func
            def main(
                A: TL.Tensor((m, k), "float16"),
                B: TL.Tensor((k, n), "float16"),
                C: TL.Tensor((m, n), "float32"),
            ):
                with TL.Kernel(TL.ceildiv(m, bm), TL.ceildiv(n, bn), threads=128) as (bx, by):
                    sa = TL.alloc_shared((bm, bk), "float16")
                    sb = TL.alloc_shared((bk, bn), "float16")
                    acc = TL.alloc_fragment((bm, bn), "float32")
                    TL.use_swizzle(panel_size=2, order="row")
                    TL.clear(acc)
                    for ko in TL.serial(TL.ceildiv(k, bk)):
                        TL.copy(A[bx * bm, ko * bk], sa)
                        TL.copy(B[ko * bk, by * bn], sb)
                        TL.gemm(sa, sb, acc)
                    TL.copy(acc, C[bx * bm, by * bn])

            return main

        a = torch.randn(65, 37, device="cuda", dtype=torch.float16)
        b = torch.randn(37, 71, device="cuda", dtype=torch.float16)
        kernel = gemm_dropin(65, 71, 37)
        c = kernel(a, b)
        assert c.shape == (65, 71) and c.dtype == torch.float32
        torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-4, atol=1e-4)

        # Phase-A dynamic shape auto-allocation & caching on real GPU
        M = TL.dynamic("m")

        @tl.jit(out_idx=[-1], target="auto")
        def dyn_softmax(A: TL.Tensor((M, 128), "float32"), B: TL.Tensor((M, 128), "float32")):
            with TL.Kernel(TL.ceildiv(M, 4), threads=128) as bx:
                tile = TL.alloc_fragment((4, 128), "float32")
                row_max = TL.alloc_fragment((4,), "float32")
                row_sum = TL.alloc_fragment((4,), "float32")
                for i, j in TL.Parallel(4, 128):
                    tile[i, j] = A[bx * 4 + i, j]
                TL.reduce_max(tile, row_max, dim=1)
                for i, j in TL.Parallel(4, 128):
                    tile[i, j] = TL.exp(tile[i, j] - row_max[i])
                TL.reduce_sum(tile, row_sum, dim=1)
                for i, j in TL.Parallel(4, 128):
                    B[bx * 4 + i, j] = tile[i, j] / row_sum[i]

        x1 = torch.randn(13, 128, device="cuda", dtype=torch.float32)
        x2 = torch.randn(29, 128, device="cuda", dtype=torch.float32)
        y1 = dyn_softmax(x1)
        y2 = dyn_softmax(x2)
        torch.testing.assert_close(y1, torch.softmax(x1, dim=-1), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(y2, torch.softmax(x2, dim=-1), rtol=2e-5, atol=2e-6)


@pytest.mark.gpu
def test_pipelined_multistage_gemm_on_gpu():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")

    from examples.matmul import matmul

    for m, n, k, bm, bn, bk, stages in (
        (256, 256, 128, 128, 128, 32, 3),
        (256, 256, 128, 128, 128, 32, 4),
        (128, 128, 32, 64, 64, 32, 3),  # trip_count=1 < num_stages
        (65, 71, 37, 32, 32, 32, 3),  # non-divisible fallback
    ):
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        c = torch.empty(m, n, device="cuda", dtype=torch.float32)
        kernel = matmul(m, n, k, bm, bn, bk, num_stages=stages, swizzle_panel=4)
        kernel(a, b, c)
        torch.cuda.synchronize()
        torch.testing.assert_close(c, a.float() @ b.float(), rtol=2e-3, atol=2e-3)
