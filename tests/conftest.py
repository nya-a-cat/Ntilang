"""Shared pytest fixtures and capability guards for Ntilang tests."""

from __future__ import annotations

import importlib.util

import pytest

HAS_CUTE_COMPILER = importlib.util.find_spec("cutlass") is not None

requires_cute = pytest.mark.skipif(
    not HAS_CUTE_COMPILER,
    reason="CuTe DSL compiler is not installed",
)


@pytest.fixture
def torch_cuda():
    torch = pytest.importorskip("torch", reason="Install a CUDA-enabled PyTorch for GPU checks")
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA GPU available")
    return torch


def target_for(torch_module) -> str:
    major, minor = torch_module.cuda.get_device_capability()
    return f"sm_{major}{minor}"
