import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.codegen import CUTLASS_TYPES
from ntilang.ir import DTYPES
from ntilang.scalar import promote
from ntilang.testing import reference


def typed_copy(dtype, *, target="sm_80"):
    @T.prim_func
    def kernel(A: T.Tensor((39,), dtype), B: T.Tensor((39,), dtype)):
        with T.Kernel(1, threads=32) as _bx:
            fragment = T.alloc_fragment((64,), dtype)
            shared = T.alloc_shared((64,), dtype)
            T.copy(A[0], fragment)
            T.copy(fragment, shared)
            T.copy(shared, B[0])

    return ntilang.compile(kernel, target=target)


def typed_reduce(dtype, kind):
    @T.prim_func
    def kernel(A: T.Tensor((3, 19), dtype), B: T.Tensor((3,), dtype)):
        with T.Kernel(1, threads=32) as _bx:
            source = T.alloc_fragment((3, 19), dtype)
            output = T.alloc_fragment((3,), dtype)
            T.copy(A, source)
            T.reduce(source, output, kind, 1, True)
            T.copy(output, B)

    return ntilang.compile(kernel)


def scalar_constructors():
    narrow = T.short

    @T.prim_func
    def kernel(A: T.Tensor((39,), T.int8), B: T.Tensor((39,), T.double)):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                value = A[bx * 32 + i] + 1
                temporary = narrow(value)
                B[T.int32(bx * 32 + i)] = T.float64(temporary)

    return ntilang.compile(kernel)


def unsigned_cast():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "int32"), B: T.Tensor((39,), "uint32")):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                B[bx * 32 + i] = T.uint32(A[bx * 32 + i])

    return ntilang.compile(kernel)


def upstream_minmax():
    @T.prim_func
    def kernel(A: T.Tensor((39,), "float32"), B: T.Tensor((39,), "float32")):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                B[bx * 32 + i] = T.min(T.max(A[bx * 32 + i], 0.0), 1.0)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", [dtype for dtype in DTYPES if dtype != "bfloat16"])
def test_typed_copy_reference(dtype):
    a = np.arange(39).astype(dtype)
    if dtype == "uint64":
        a += np.uint64(2**63)
    b = np.zeros_like(a)
    reference(typed_copy(dtype), a, b)
    np.testing.assert_array_equal(b, a)


def test_scalar_constructor_and_promotion_reference():
    a = np.full(39, 127, dtype=np.int8)
    b = np.empty(39, dtype=np.float64)
    reference(scalar_constructors(), a, b)
    np.testing.assert_array_equal(b, np.full(39, 128.0))


def test_unsigned_integer_cast_reference():
    a = np.arange(-20, 19, dtype=np.int32)
    b = np.empty(39, dtype=np.uint32)
    reference(unsigned_cast(), a, b)
    np.testing.assert_array_equal(b, a.astype(np.uint32))


def test_upstream_minmax_nan_reference():
    a = np.linspace(-1, 2, 39, dtype=np.float32)
    a[0] = np.nan
    b = np.empty_like(a)
    reference(upstream_minmax(), a, b)
    np.testing.assert_array_equal(b, np.fmin(np.fmax(a, 0), 1))


def test_dtype_alias_metadata():
    assert T.short == "int16" and T.short.bits == 16 and T.short.bytes == 2
    assert T.bool.bits == 8 and T.bool.bytes == 1 and T.bool.type_code == 6
    assert T.Tensor((3,), "double").dtype == "float64"
    assert T.dtype(int) == "int32" and T.get_tvm_dtype("ulong") == "uint64"
    assert T.dtype("float16").lanes == 1 and T.dtype("float16").itemsize == 2


REDUCTION_CASES = [(dtype, kind) for dtype in DTYPES for kind in ("sum", "max", "min")]
REDUCTION_CASES += [
    (dtype, kind)
    for dtype in ("int8", "uint8", "int64", "uint64", "bool")
    for kind in ("bitand", "bitor", "bitxor")
]


@pytest.mark.parametrize("dtype,kind", [(d, k) for d, k in REDUCTION_CASES if d != "bfloat16"])
def test_typed_reduction_reference(dtype, kind):
    a = (np.arange(57) % 4).astype(dtype).reshape(3, 19)
    b = np.empty(3, dtype=dtype)
    reference(typed_reduce(dtype, kind), a, b)
    operation = {
        "sum": np.add,
        "max": np.maximum,
        "min": np.minimum,
        "bitand": np.bitwise_and,
        "bitor": np.bitwise_or,
        "bitxor": np.bitwise_xor,
    }[kind]
    expected = operation.reduce(a, axis=1, dtype=a.dtype)
    np.testing.assert_array_equal(b, expected)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("dtype", DTYPES)
def test_typed_copy_cute_compilation(dtype):
    assert typed_copy(dtype).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("dtype,kind", REDUCTION_CASES)
def test_typed_reduction_cute_compilation(dtype, kind):
    assert typed_reduce(dtype, kind).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_explicit_promotion_produces_the_selected_nvidia_type():
    import cutlass

    for left in DTYPES:
        for right in DTYPES:
            expected = getattr(cutlass, CUTLASS_TYPES[promote(left, right)])
            actual = expected(getattr(cutlass, CUTLASS_TYPES[left])(1)) + expected(
                getattr(cutlass, CUTLASS_TYPES[right])(1)
            )
            assert type(actual) is expected, (left, right, type(actual), expected)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("factory", [scalar_constructors, unsigned_cast, upstream_minmax])
def test_scalar_dtype_cute_compilation(factory):
    assert factory().build().has_gpu_module


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_cuda_tensor_roundtrip(dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA GPU available")
    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        pytest.skip(f"The installed PyTorch does not expose {dtype}")
    major, minor = torch.cuda.get_device_capability()
    a = torch.arange(39, dtype=torch.int64).to(dtype=torch_dtype, device="cuda")
    b = torch.empty_like(a)
    typed_copy(dtype, target=f"sm_{major}{minor}")(a, b)
    if dtype == "bfloat16":
        torch.testing.assert_close(b.float(), a.float(), rtol=0, atol=0)
    else:
        np.testing.assert_array_equal(b.cpu().numpy(), a.cpu().numpy())
