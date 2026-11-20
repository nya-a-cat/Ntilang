"""LLM operator pack tests: FlashAttention-2 (RS-GEMM), RMSNorm, and LayerNorm."""

from __future__ import annotations

import importlib.util

import pytest
from ntilang.ptx_verifier import inspect_kernel_ptx

from examples.flash_attention import flash_attention
from examples.layernorm import layernorm
from examples.rmsnorm import rmsnorm


def test_llm_operators_emit_expected_ir_and_cute_source():
    fa = flash_attention(batch=2, heads=4, seq_len=256, dim=64, is_causal=False)
    assert "CopyG2SOp" in fa.source
    assert "LdMatrix8x8x16bOp" in fa.source
    assert "cute.arch.shuffle_sync_down" in fa.source

    fa_causal = flash_attention(batch=1, heads=2, seq_len=128, dim=64, is_causal=True)
    assert "LdMatrix8x8x16bOp" in fa_causal.source

    rms = rmsnorm(m=16, n=256)
    assert "cute.arch.shuffle_sync_down" in rms.source

    ln = layernorm(m=16, n=256)
    assert "cute.arch.shuffle_sync_down" in ln.source


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_llm_operators_ptx_contracts():
    fa_report = inspect_kernel_ptx(flash_attention(batch=1, heads=2, seq_len=256, dim=64))
    assert fa_report.mma_ops >= 8
    assert fa_report.ldmatrix_ops >= 4
    assert fa_report.cp_async_ops >= 4
    assert fa_report.consecutive_barriers == 0
    assert fa_report.local_spill_ops == 0

    rms_report = inspect_kernel_ptx(rmsnorm(m=64, n=512))
    assert rms_report.contains(r"\bshfl\.sync\.down\.b32\b")
    assert rms_report.barriers <= 3
    assert rms_report.local_spill_ops == 0

    ln_report = inspect_kernel_ptx(layernorm(m=64, n=512))
    assert ln_report.contains(r"\bshfl\.sync\.down\.b32\b")
    assert ln_report.barriers <= 6
    assert ln_report.local_spill_ops == 0


@pytest.mark.gpu
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
def test_flash_attention_matches_pytorch_sdpa_on_gpu(is_causal, dim):
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")

    batch, heads, seq_len = 2, 4, 256
    q = torch.randn(batch, heads, seq_len, dim, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(batch, heads, seq_len, dim, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(batch, heads, seq_len, dim, device="cuda", dtype=torch.float16) * 0.5
    out = torch.empty_like(q)

    kernel = flash_attention(batch, heads, seq_len, dim, is_causal=is_causal)
    kernel(q, k, v, out)
    torch.cuda.synchronize()

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [256, 512, 768])
def test_rmsnorm_and_layernorm_on_gpu(n):
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")

    m = 131
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    y_rms = torch.empty_like(x)
    rmsnorm(m, n, eps=1e-6)(x, w, y_rms)
    torch.cuda.synchronize()
    ref_rms = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(y_rms, ref_rms, rtol=1e-4, atol=1e-5)

    y_ln = torch.empty_like(x)
    layernorm(m, n, eps=1e-5)(x, w, b, y_ln)
    torch.cuda.synchronize()
    ref_ln = F.layer_norm(x, (n,), weight=w, bias=b, eps=1e-5)
    torch.testing.assert_close(y_ln, ref_ln, rtol=1e-4, atol=1e-5)
