"""Tests for M3.3 (Phase-B SymInt), M4.1 (atomics), M4.2 (@autotune), and M4.3 (Compiler API)."""

from __future__ import annotations

import importlib.util

import ntilang
import ntilang as tilelang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ptx_verifier import inspect_kernel_ptx
from ntilang.testing import reference


def test_symint_single_compile_generates_cute_sym_int32():
    M = T.dynamic("m")

    @T.prim_func
    def dyn_add(A: T.Tensor((M, 64), "float32"), B: T.Tensor((M, 64), "float32")):
        with T.Kernel(T.ceildiv(M, 4), threads=128) as bx:
            for i, j in T.Parallel(4, 64):
                B[bx * 4 + i, j] = A[bx * 4 + i, j] * 3.0 + 1.0

    compiled = ntilang.compile(dyn_add, target="sm_80", out_idx=[-1])
    assert "cute.sym_int32(symbol='m')" in compiled.source
    assert len(compiled.ir.sym_dims) == 1
    assert compiled.ir.sym_dims[0].name == "m"


def test_atomic_add_and_max_reference_and_source():
    @T.prim_func
    def hist_kernel(
        X: T.Tensor((64,), "int32"), Out: T.Tensor((8,), "int32"), MaxOut: T.Tensor((8,), "int32")
    ):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(64):
                bucket = X[i] % 8
                T.atomic_add(Out[bucket], 1)
                T.atomic_max(MaxOut[bucket], i)

    compiled = ntilang.compile(hist_kernel, target="sm_80")
    assert "atom.global.add.s32" in compiled.source
    assert "atom.global.max.s32" in compiled.source

    x = np.arange(64, dtype=np.int32)
    out = np.zeros(8, dtype=np.int32)
    max_out = np.full(8, -1, dtype=np.int32)
    reference(compiled, x, out, max_out)
    np.testing.assert_array_equal(out, np.full(8, 8, dtype=np.int32))
    np.testing.assert_array_equal(max_out, np.arange(56, 64, dtype=np.int32))


def test_staged_compiler_api_and_artifact_save(tmp_path):
    @T.prim_func
    def scale(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(64):
                B[i] = A[i] * 2.0

    compiler = ntilang.Compiler()
    program = compiler.import_program(scale, frontend="tilelang")
    assert "scale" in program.as_text()
    lowered = compiler.lower(program, backend="cute", target="cuda:sm_80", out_idx=[-1])
    saved = lowered.emit(format="python").save(tmp_path / "scale_cute.py")
    assert saved.exists()
    assert "def compile_kernel" in saved.read_text(encoding="utf-8")
    bundle = compiler.compile(lowered)
    assert bundle.kernel.out_idx == (1,)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_atomic_and_symint_ptx_contracts():
    @T.prim_func
    def atomic_f32(X: T.Tensor((128,), "float32"), Out: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                T.atomic_add(Out[i % 4], X[i])

    report = inspect_kernel_ptx(ntilang.compile(atomic_f32, target="sm_80"))
    assert report.contains(r"\batom\.global\.add\.f32\b")
    assert report.local_spill_ops == 0


@pytest.mark.gpu
def test_symint_single_compile_and_atomics_and_autotune_on_gpu():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")

    # 1. Phase-B SymInt: compile ONCE and run across 3 distinct dynamic M values!
    M = T.dynamic("m")

    @tilelang.jit(out_idx=[-1], target="auto")
    def dyn_softmax(A: T.Tensor((M, 128), "float32"), B: T.Tensor((M, 128), "float32")):
        with T.Kernel(T.ceildiv(M, 4), threads=128) as bx:
            tile = T.alloc_fragment((4, 128), "float32")
            row_max = T.alloc_fragment((4,), "float32")
            row_sum = T.alloc_fragment((4,), "float32")
            for i, j in T.Parallel(4, 128):
                tile[i, j] = A[bx * 4 + i, j]
            T.reduce_max(tile, row_max, dim=1)
            for i, j in T.Parallel(4, 128):
                tile[i, j] = T.exp(tile[i, j] - row_max[i])
            T.reduce_sum(tile, row_sum, dim=1)
            for i, j in T.Parallel(4, 128):
                B[bx * 4 + i, j] = tile[i, j] / row_sum[i]

    for rows in (7, 19, 131):
        x = torch.randn(rows, 128, device="cuda", dtype=torch.float32)
        y = dyn_softmax(x)
        torch.cuda.synchronize()
        assert y.shape == (rows, 128)
        torch.testing.assert_close(y, torch.softmax(x, dim=-1), rtol=2e-5, atol=2e-6)
    assert dyn_softmax.compile_count == 1

    # 2. T.atomic_add (float32 + int32) and T.atomic_max (int32) on real GPU!
    @T.prim_func
    def atomic_reduce(
        X: T.Tensor((256,), "float32"),
        SumOut: T.Tensor((4,), "float32"),
        MaxOut: T.Tensor((4,), "int32"),
    ):
        with T.Kernel(4, threads=64) as bx:
            for i in T.Parallel(64):
                idx = bx * 64 + i
                T.atomic_add(SumOut[idx % 4], X[idx])
                T.atomic_max(MaxOut[idx % 4], idx)

    k_atom = ntilang.compile(atomic_reduce, target="auto")
    x_a = torch.randn(256, device="cuda", dtype=torch.float32)
    sum_out = torch.zeros(4, device="cuda", dtype=torch.float32)
    max_out = torch.full((4,), -1, device="cuda", dtype=torch.int32)
    k_atom(x_a, sum_out, max_out)
    torch.cuda.synchronize()
    expected_sum = x_a.view(64, 4).sum(dim=0)
    torch.testing.assert_close(sum_out, expected_sum, rtol=1e-4, atol=1e-4)
    assert max_out.tolist() == [252, 253, 254, 255]

    # 3. @tilelang.autotune selecting fastest GEMM config on real A100!
    @tilelang.autotune(
        configs=[
            {"block_m": 64, "block_n": 64, "block_k": 32, "num_stages": 2},
            {"block_m": 128, "block_n": 128, "block_k": 32, "num_stages": 3},
        ],
        warmup=3,
        rep=10,
    )
    @tilelang.jit(out_idx=[-1], target="auto")
    def tuned_gemm(m, n, k, block_m=64, block_n=64, block_k=32, num_stages=2):
        @T.prim_func
        def main(
            A: T.Tensor((m, k), "float16"),
            B: T.Tensor((k, n), "float16"),
            C: T.Tensor((m, n), "float32"),
        ):
            with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), threads=128) as (bx, by):
                sa = T.alloc_shared((block_m, block_k), "float16")
                sb = T.alloc_shared((block_k, block_n), "float16")
                acc = T.alloc_fragment((block_m, block_n), "float32")
                T.use_swizzle(panel_size=8, order="row")
                T.clear(acc)
                for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(A[bx * block_m, ko * block_k], sa)
                    T.copy(B[ko * block_k, by * block_n], sb)
                    T.gemm(sa, sb, acc)
                T.copy(acc, C[bx * block_m, by * block_n])

        return main

    best_k = tuned_gemm(512, 512, 512)
    assert tuned_gemm.best_config is not None
    assert tuned_gemm.best_latency_ms > 0.0
    a_t = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b_t = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c_t = best_k(a_t, b_t)
    torch.cuda.synchronize()
    torch.testing.assert_close(c_t, a_t.float() @ b_t.float(), rtol=1e-2, atol=1e-2)

    # 4. Multi-frontend Compiler.load() -> RuntimeModule["scale"] execution on GPU!
    @T.prim_func
    def gpu_scale(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = A[i] + 5.0

    compiler = ntilang.Compiler()
    prog = compiler.import_program(gpu_scale, frontend="tilelang")
    lowered = compiler.lower(prog, backend="cute", target="auto", out_idx=[-1])
    module = compiler.load(compiler.compile(lowered), device="cuda:0")
    x_s = torch.randn(128, device="cuda", dtype=torch.float32)
    y_s = module["gpu_scale"](x_s)
    torch.cuda.synchronize()
    torch.testing.assert_close(y_s, x_s + 5.0)
