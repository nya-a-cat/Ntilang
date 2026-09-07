import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.codegen import walk
from ntilang.testing import reference


def unrolled_accumulation(*, explicit=False, factor=None, annotations=None, start=9, stop=-2, step=-2):
    @T.prim_func
    def kernel(A: T.Tensor((39,), "int32"), B: T.Tensor((39,), "int32")):
        with T.Kernel(2, threads=32) as bx:
            tile = T.alloc_fragment((32,), "int32")
            T.copy(A[bx * 32], tile)
            for i in T.Parallel(32):
                for k in T.Unroll(
                    start=start,
                    stop=stop,
                    step=step,
                    explicit=explicit,
                    unroll_factor=factor,
                    annotations=annotations,
                ):
                    tile[i] += k
            T.copy(tile, B[bx * 32])

    return ntilang.compile(kernel)


OPTIONS = [
    {},
    {"explicit": True},
    {"factor": 0},
    {"factor": 1},
    {"factor": 4},
    {"factor": 13},
    {"annotations": {"pragma_unroll_explicit": True}},
    {"annotations": {"pragma_unroll_factor": 3}},
    {"factor": 2, "annotations": {"pragma_unroll_factor": 5}},
    {"explicit": True, "annotations": {"pragma_unroll_explicit": False}},
]


@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("domain", [(9, -2, -2), (0, 7, 1), (3, 3, 1)])
def test_unroll_scheduling_preserves_iteration(options, domain):
    kernel = unrolled_accumulation(**options, start=domain[0], stop=domain[1], step=domain[2])
    a = np.arange(39, dtype=np.int32)
    b = np.empty_like(a)
    reference(kernel, a, b)
    np.testing.assert_array_equal(b, a + sum(range(*domain)))


@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, "unroll_full=True"),
        ({"explicit": True}, "range_constexpr(6)"),
        ({"factor": 0}, "unroll=1"),
        ({"factor": 1}, "unroll=1"),
        ({"factor": 4}, "unroll=4"),
        ({"factor": 2, "annotations": {"pragma_unroll_factor": 5}}, "unroll=2"),
        ({"explicit": False, "annotations": {"pragma_unroll_explicit": True}}, "range_constexpr(6)"),
        ({"explicit": True, "annotations": {"pragma_unroll_explicit": False}}, "range_constexpr(6)"),
    ],
)
def test_unroll_request_reaches_the_backend(options, expected):
    kernel = unrolled_accumulation(**options)
    assert expected in kernel.source
    loop = next(stmt for stmt in walk(kernel.ir.body) if stmt.op == "unroll")
    assert dict(loop.annotations)["pragma_unroll_explicit"] == ("range_constexpr" in expected)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"explicit": True, "factor": 2}, "mutually exclusive"),
        ({"factor": 2, "annotations": {"pragma_unroll_explicit": True}}, "mutually exclusive"),
        ({"explicit": 1}, "Boolean"),
        ({"factor": -1}, "nonnegative"),
        ({"factor": 2**31}, "nonnegative"),
        ({"factor": True}, "nonnegative"),
        ({"annotations": {"unknown": 1}}, "Unsupported loop annotations"),
        ({"annotations": (1,)}, "dictionary"),
    ],
)
def test_invalid_loop_options(options, message):
    with pytest.raises(ntilang.CompileError, match=message):
        unrolled_accumulation(**options)


def serial_keywords():
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((32,), "int32")
            T.clear(tile)
            for i in T.Parallel(32):
                for k in T.Serial(start=7, step=2, annotations={}):
                    tile[i] += k
                for j in T.serial(start=3, stop=None, step=None, annotations=None):
                    tile[i] += j
            T.copy(tile, B)

    return ntilang.compile(kernel)


def test_serial_keyword_defaults():
    out = np.empty(32, dtype=np.int32)
    reference(serial_keywords(), out)
    np.testing.assert_array_equal(out, sum(range(0, 7, 2)) + sum(range(3)))


def test_unroll_options_are_keyword_only():
    @T.prim_func
    def bad(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                for k in T.unroll(0, 3, 1, True):
                    B[i] = k

    with pytest.raises(ntilang.CompileError, match="keyword-only"):
        ntilang.compile(bad)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("options", OPTIONS)
def test_unroll_options_compile(options):
    assert unrolled_accumulation(**options).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_serial_keywords_compile():
    assert serial_keywords().build().has_gpu_module
