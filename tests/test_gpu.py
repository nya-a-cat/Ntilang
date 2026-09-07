"""Hardware checks are opt-in: uv run pytest -m gpu on an NVIDIA machine."""

import pytest

from examples.fragment_affine import fragment_affine
from examples.matmul import matmul
from examples.matmul_relu import matmul_relu
from examples.piecewise import piecewise
from examples.softmax import softmax
from examples.vector_add import vector_add

pytestmark = pytest.mark.gpu


@pytest.fixture
def torch_cuda():
    torch = pytest.importorskip("torch", reason="Install a CUDA-enabled PyTorch for GPU checks")
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA GPU available")
    return torch


def target_for(torch):
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def test_vector_add_on_gpu(torch_cuda):
    torch = torch_cuda
    a, b = (torch.randn(1000, device="cuda") for _ in range(2))
    c = torch.empty_like(a)
    kernel = vector_add(target=target_for(torch))
    kernel(a, b, c)
    torch.testing.assert_close(c, a + b, rtol=0, atol=0)


def test_fragment_affine_on_gpu(torch_cuda):
    torch = torch_cuda
    a = torch.randn(257, device="cuda")
    b = torch.empty_like(a)
    fragment_affine(target=target_for(torch))(a, b)
    torch.testing.assert_close(b, a * 2 + 1, rtol=1e-6, atol=1e-6)


def test_gemm_on_gpu(torch_cuda):
    torch = torch_cuda
    a = torch.randn(65, 37, device="cuda", dtype=torch.float16)
    b = torch.randn(37, 71, device="cuda", dtype=torch.float16)
    c = torch.empty(65, 71, device="cuda", dtype=torch.float32)
    matmul(target=target_for(torch))(a, b, c)
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-4, atol=1e-4)


def test_gemm_relu_on_gpu(torch_cuda):
    torch = torch_cuda
    a = torch.randn(65, 37, device="cuda", dtype=torch.float16)
    b = torch.randn(37, 71, device="cuda", dtype=torch.float16)
    bias = torch.randn(65, 71, device="cuda", dtype=torch.float32)
    c = torch.empty_like(bias)
    matmul_relu(target=target_for(torch))(a, b, bias, c)
    torch.testing.assert_close(c, torch.relu((a.float() @ b.float()) * 0.5 + bias), rtol=1e-4, atol=1e-4)


def test_piecewise_on_gpu(torch_cuda):
    torch = torch_cuda
    a = torch.linspace(-2, 2, 93, device="cuda")
    b = torch.empty_like(a)
    piecewise(target=target_for(torch))(a, b)
    expected = torch.where(a < 0, -a, torch.where(a < 1, a * a, a + 2))
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


def test_softmax_on_gpu(torch_cuda):
    torch = torch_cuda
    a = torch.randn(9, 113, device="cuda", dtype=torch.float32) * 30
    b = torch.empty_like(a)
    softmax(target=target_for(torch))(a, b)
    torch.testing.assert_close(b, torch.softmax(a, dim=1), rtol=2e-5, atol=2e-6)


def test_nondefault_stream_on_gpu(torch_cuda):
    torch = torch_cuda
    kernel = vector_add(target=target_for(torch))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = torch.full((1000,), 2.0, device="cuda")
        b = torch.full_like(a, 3.0)
        c = torch.empty_like(a)
        kernel(a, b, c)
        expected = a + b
    stream.synchronize()
    torch.testing.assert_close(c, expected, rtol=0, atol=0)
