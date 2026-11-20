"""CPU-runnable PTX contract tests and static instruction regression guards."""

from __future__ import annotations

import importlib.util

import ntilang
import ntilang.language as T
import pytest
from ntilang.ptx_verifier import analyze_ptx, inspect_kernel_ptx

from examples.matmul import matmul
from examples.vector_add import vector_add


def _minmax_kernel(kind: str, dtype: str, target: str = "sm_80"):
    operation = getattr(T, kind)

    @T.prim_func
    def kernel(A: T.Tensor((8,), dtype), B: T.Tensor((8,), dtype), C: T.Tensor((8,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(8):
                C[i] = operation(A[i], B[i])

    return ntilang.compile(kernel, target=target)


@pytest.mark.parametrize(
    ("kind", "dtype", "expected_ptx"),
    [
        ("max", "float16", "max.f16"),
        ("min", "float16", "min.f16"),
        ("maximum", "float16", "max.NaN.f16"),
        ("minimum", "float16", "min.NaN.f16"),
        ("max", "bfloat16", "max.bf16"),
        ("maximum", "bfloat16", "max.NaN.bf16"),
        ("max", "float32", "max.f32"),
        ("min", "float32", "min.f32"),
        ("maximum", "float32", "max.NaN.f32"),
        ("minimum", "float32", "min.NaN.f32"),
        ("max", "float64", "max.f64"),
        ("min", "float64", "min.f64"),
        ("maximum", "float64", "testp.notanumber.f64"),
        ("minimum", "float64", "testp.notanumber.f64"),
    ],
)
def test_minmax_emits_inline_ptx_in_generated_source(kind, dtype, expected_ptx):
    compiled = _minmax_kernel(kind, dtype)
    assert expected_ptx in compiled.source
    assert "cute.math.max" not in compiled.source
    assert "cute.math.min" not in compiled.source


def test_host_assertion_minmax_stays_cpu_portable():
    @T.prim_func
    def kernel(x: T.float32, y: T.float32, out: T.Tensor((4,), "float32")):
        assert T.max(x, y) >= 0.0
        with T.Kernel(1, threads=32):
            for i in T.Parallel(4):
                out[i] = x

    compiled = ntilang.compile(kernel)
    assert "_nt_check_preconditions" in compiled.source
    assert "_nt_minmax_" not in compiled.source


def test_analyze_ptx_detects_barriers_spills_and_mma():
    sample_ptx = """
    .visible .entry demo() {
        .local .align 4 .b8 __local_depot0[16];
        st.local.f32 [__local_depot0], %f1;
        bar.sync 0;
        bar.sync 0;
        ld.shared.b32 %r1, [%r2];
        bar.sync 0;
        mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};
        max.NaN.f32 %f5, %f1, %f2;
        ret;
    }
    """
    report = analyze_ptx(sample_ptx)
    assert report.barriers == 3
    assert report.consecutive_barriers == 1
    assert report.has_local_decl is True
    assert report.local_spill_ops == 1
    assert report.mma_ops == 1
    assert report.contains(r"\bmax\.NaN\.f32\b")


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    ("kind", "dtype", "ptx_regex"),
    [
        ("max", "float16", r"\bmax\.f16\b"),
        ("maximum", "float16", r"\bmax\.NaN\.f16\b"),
        ("max", "bfloat16", r"\bmax\.bf16\b"),
        ("max", "float32", r"\bmax\.f32\b"),
        ("maximum", "float32", r"\bmax\.NaN\.f32\b"),
        ("max", "float64", r"\bmax\.f64\b"),
        ("maximum", "float64", r"\btestp\.notanumber\.f64\b"),
    ],
)
def test_minmax_ptx_survives_cute_compilation(kind, dtype, ptx_regex):
    kernel = _minmax_kernel(kind, dtype)
    assert kernel.build().has_gpu_module
    report = inspect_kernel_ptx(kernel)
    assert report.contains(ptx_regex)
    assert report.local_spill_ops == 0


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_core_kernels_ptx_contracts():
    vec_report = inspect_kernel_ptx(vector_add())
    assert vec_report.local_spill_ops == 0
    assert vec_report.barriers == 0

    gemm_report = inspect_kernel_ptx(matmul())
    assert gemm_report.mma_ops >= 1
    assert gemm_report.local_spill_ops == 0
