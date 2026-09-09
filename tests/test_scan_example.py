import importlib.util

import numpy as np
import pytest
from ntilang.testing import reference

from examples.scan import scan


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("m,n", [(1, 1), (9, 113), (5, 65)])
def test_scan_example_reference(m, n, reverse):
    a = (np.arange(m * n).reshape(m, n) % 7 - 3).astype(np.float32)
    prefix, maximum = np.empty_like(a), np.empty_like(a)
    kernel = scan(m, n, reverse=reverse)
    reference(kernel, a, prefix, maximum)
    values = a[:, ::-1] if reverse else a
    expected_prefix = np.cumsum(values, axis=1)
    expected_maximum = np.maximum.accumulate(values, axis=1)
    if reverse:
        expected_prefix = expected_prefix[:, ::-1]
        expected_maximum = expected_maximum[:, ::-1]
    np.testing.assert_array_equal(prefix, expected_prefix)
    np.testing.assert_array_equal(maximum, expected_maximum)
    assert kernel.source.count("_nt_smem.allocate_tensor") == 2


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_example_native_compilation(reverse):
    assert scan(reverse=reverse).build().has_gpu_module
