import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference

from examples.softmax import softmax


def reduction_kernel(
    shape=(5, 37),
    dim=-1,
    kind="sum",
    dtype="float32",
    out_dtype=None,
    keepdims=False,
    clear=True,
    scopes=("fragment", "fragment"),
    nan_propagate=False,
):
    out_dtype = dtype if out_dtype is None else out_dtype
    axis = dim % len(shape)
    out_shape = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1 :]
    alloc_in = T.alloc_shared if scopes[0] == "shared" else T.alloc_fragment
    alloc_out = T.alloc_shared if scopes[1] == "shared" else T.alloc_fragment

    @T.prim_func
    def kernel(A: T.Tensor(shape, dtype), B: T.Tensor(out_shape, out_dtype)):
        with T.Kernel(1, threads=64) as _bx:
            src = alloc_in(shape, dtype)
            dst = alloc_out(out_shape, out_dtype)
            T.copy(A, src)
            T.fill(dst, 3)
            T.reduce(src, dst, kind, dim=dim, clear=clear, nan_propagate=nan_propagate)
            T.copy(dst, B)

    return ntilang.compile(kernel)


def wrapper_kernel(kind, *, target="sm_80"):
    dtype = "int32" if kind.startswith("bit") else "float32"
    operation = getattr(T, "reduce_" + kind)

    @T.prim_func
    def kernel(A: T.Tensor((3, 19), dtype), B: T.Tensor((3,), dtype)):
        with T.Kernel(1, threads=32) as _bx:
            src = T.alloc_fragment((3, 19), dtype)
            dst = T.alloc_fragment((3,), dtype)
            T.copy(A, src)
            operation(src, dst)
            T.copy(dst, B)

    return ntilang.compile(kernel, target=target)


def mma_reduction():
    @T.prim_func
    def kernel(
        A: T.Tensor((32, 32), "float16"),
        B: T.Tensor((32, 32), "float16"),
        C: T.Tensor((32,), "float32"),
    ):
        with T.Kernel(1, threads=128) as _bx:
            sa = T.alloc_shared((32, 32), "float16")
            sb = T.alloc_shared((32, 32), "float16")
            acc = T.alloc_fragment((32, 32), "float32")
            row = T.alloc_fragment((32,), "float32")
            T.copy(A, sa)
            T.copy(B, sb)
            T.clear(acc)
            T.gemm(sa, sb, acc)
            T.reduce_sum(acc, row)
            T.copy(row, C)

    return ntilang.compile(kernel)


def expected_reduce(a, kind, dim, keepdims=False):
    values = np.abs(a) if kind.startswith("abs") else a
    operation = {
        "sum": np.add,
        "abssum": np.add,
        "max": np.maximum,
        "absmax": np.maximum,
        "min": np.minimum,
        "bitand": np.bitwise_and,
        "bitor": np.bitwise_or,
        "bitxor": np.bitwise_xor,
    }[kind]
    return operation.reduce(values, axis=dim, keepdims=keepdims, dtype=a.dtype)


@pytest.mark.parametrize("kind", ["sum", "abssum", "max", "absmax", "min", "bitand", "bitor", "bitxor"])
def test_reduction_wrappers_reference(kind):
    dtype = np.int32 if kind.startswith("bit") else np.float32
    a = (np.arange(57).reshape(3, 19) - 28).astype(dtype)
    b = np.zeros(3, dtype=dtype)
    reference(wrapper_kernel(kind), a, b)
    np.testing.assert_array_equal(b, expected_reduce(a, kind, -1))


@pytest.mark.parametrize(
    "scopes", [("fragment", "fragment"), ("shared", "fragment"), ("fragment", "shared"), ("shared", "shared")]
)
@pytest.mark.parametrize("dim,keepdims", [(0, False), (1, True), (-1, False)])
def test_reduction_dimensions_and_scopes(scopes, dim, keepdims):
    a = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
    expected = a.sum(axis=dim, keepdims=keepdims, dtype=np.float32) + 3
    b = np.zeros_like(expected)
    reference(reduction_kernel(a.shape, dim, scopes=scopes, keepdims=keepdims, clear=False), a, b)
    np.testing.assert_array_equal(b, expected)


@pytest.mark.parametrize("kind", ["sum", "max", "absmax", "min", "bitand", "bitor", "bitxor"])
def test_reduction_combines_initial_value_once(kind):
    dtype = "int32" if kind.startswith("bit") else "float32"
    a = (np.arange(185).reshape(5, 37) % 13 - 6).astype(dtype)
    b = np.empty(5, dtype=dtype)
    reference(reduction_kernel(kind=kind, dtype=dtype, clear=False), a, b)
    value = expected_reduce(a, kind, -1)
    expected = {
        "sum": lambda: value + 3,
        "max": lambda: np.maximum(value, 3),
        "absmax": lambda: np.maximum(value, 3),
        "min": lambda: np.minimum(value, 3),
        "bitand": lambda: value & 3,
        "bitor": lambda: value | 3,
        "bitxor": lambda: value ^ 3,
    }[kind]()
    np.testing.assert_array_equal(b, expected)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("nan_propagate", [False, True])
def test_reduction_nan_contract(dtype, nan_propagate):
    a = np.array([[np.nan, 1, 2], [np.nan, np.nan, np.nan]], dtype=dtype)
    b = np.empty(2, dtype=dtype)
    reference(reduction_kernel(a.shape, kind="max", dtype=dtype, nan_propagate=nan_propagate), a, b)
    expected = [np.nan, np.nan] if dtype == "float16" and nan_propagate else [2, -np.inf]
    np.testing.assert_array_equal(b, np.asarray(expected, dtype=dtype))


def test_softmax_reference():
    rng = np.random.default_rng(491)
    a = rng.normal(size=(9, 113)).astype(np.float32) * 100
    b = np.full_like(a, np.nan)
    reference(softmax(), a, b)
    exp = np.exp(a - a.max(axis=1, keepdims=True))
    np.testing.assert_allclose(b, exp / exp.sum(axis=1, keepdims=True), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(b.sum(axis=1), 1, rtol=2e-6, atol=2e-6)


def test_invalid_reduction_dimension():
    with pytest.raises(ntilang.CompileError, match="dimension"):
        reduction_kernel(dim=2)


def test_bitwise_reduction_requires_integer_output():
    with pytest.raises(ntilang.CompileError, match="integer output"):
        reduction_kernel(kind="bitand")


def test_reduction_requires_initialized_accumulation():
    @T.prim_func
    def bad(A: T.Tensor((32, 17), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=64) as _bx:
            src = T.alloc_fragment((32, 17), "float32")
            dst = T.alloc_fragment((32,), "float32")
            T.copy(A, src)
            T.reduce_sum(src, dst, clear=False)
            T.copy(dst, B)

    with pytest.raises(ntilang.CompileError, match="before initialization"):
        ntilang.compile(bad)


def test_mma_reduction_reference():
    rng = np.random.default_rng(83)
    a, b = (rng.normal(size=(32, 32)).astype(np.float16) for _ in range(2))
    c = np.empty(32, dtype=np.float32)
    reference(mma_reduction(), a, b, c)
    expected = (a.astype(np.float32) @ b.astype(np.float32)).sum(axis=1)
    np.testing.assert_allclose(c, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("kind", ["sum", "abssum", "max", "absmax", "min", "bitand", "bitor", "bitxor"])
def test_reduction_wrappers_cute_compilation(kind):
    assert wrapper_kernel(kind).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduction_nan_cute_compilation(dtype):
    assert reduction_kernel(dtype=dtype, kind="max", nan_propagate=True).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_shared_mixed_precision_reduction_cute_compilation():
    kernel = reduction_kernel(
        (3, 5, 7),
        dim=0,
        keepdims=True,
        dtype="float16",
        out_dtype="float32",
        scopes=("shared", "shared"),
        clear=False,
    )
    assert kernel.build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_softmax_cute_compilation():
    assert softmax().build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_mma_and_single_axis_reduction_cute_compilation():
    assert mma_reduction().build().has_gpu_module
    assert reduction_kernel((37,), keepdims=True).build().has_gpu_module


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["sum", "abssum", "max", "absmax", "min", "bitand", "bitor", "bitxor"])
def test_reduction_kinds_on_gpu(kind):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA GPU available")
    major, minor = torch.cuda.get_device_capability()
    dtype = torch.int32 if kind.startswith("bit") else torch.float32
    a = (torch.arange(57, device="cuda", dtype=dtype) - 28).reshape(3, 19)
    b = torch.empty(3, device="cuda", dtype=dtype)
    wrapper_kernel(kind, target=f"sm_{major}{minor}")(a, b)
    expected = expected_reduce(a.cpu().numpy(), kind, -1)
    torch.testing.assert_close(b.cpu(), torch.from_numpy(expected), rtol=0, atol=0)
