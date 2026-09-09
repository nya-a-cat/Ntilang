import importlib.util

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


def minmax_kernel(kind, dtype, target="sm_80"):
    operation = getattr(T, kind)

    @T.prim_func
    def kernel(A: T.Tensor((8,), dtype), B: T.Tensor((8,), dtype), C: T.Tensor((8,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(8):
                C[i] = operation(A[i], B[i])

    return ntilang.compile(kernel, target=target)


def operands(dtype):
    a = np.array([-0.0, 0.0, -0.0, 0.0, np.nan, -2.0, np.nan, np.inf], dtype=dtype)
    b = np.array([-0.0, -0.0, 0.0, 0.0, 2.0, np.nan, np.nan, -np.inf], dtype=dtype)
    return a, b


def expected_values(kind, dtype):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max
    largest = kind in ("max", "maximum")
    propagate = kind in ("maximum", "minimum")
    zeros = [-0.0, 0.0, 0.0, 0.0] if largest else [-0.0, -0.0, -0.0, 0.0]
    tail = [np.nan, np.nan] if propagate else [2.0, -2.0]
    return np.array([*zeros, *tail, np.nan, np.inf if largest else -np.inf], dtype=dtype)


@pytest.mark.parametrize("kind", ["max", "min", "maximum", "minimum"])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_minmax_zero_order_and_nan_contract(kind, dtype):
    a, b = operands(dtype)
    output = np.empty_like(a)
    reference(minmax_kernel(kind, dtype), a, b, output)
    expected = expected_values(kind, dtype)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(np.signbit(output[:4]), np.signbit(expected[:4]))


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("kind", ["max", "min", "maximum", "minimum"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_minmax_zero_native_compilation(kind, dtype):
    assert minmax_kernel(kind, dtype).build().has_gpu_module


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["max", "min", "maximum", "minimum"])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_minmax_zero_on_gpu(kind, dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA GPU available")
    major, minor = torch.cuda.get_device_capability()
    a, b = (torch.from_numpy(value).cuda() for value in operands(dtype))
    output = torch.empty_like(a)
    minmax_kernel(kind, dtype, target=f"sm_{major}{minor}")(a, b, output)
    actual = output.cpu().numpy()
    expected = expected_values(kind, dtype)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual[:4]), np.signbit(expected[:4]))
