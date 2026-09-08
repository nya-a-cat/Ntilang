from __future__ import annotations

import ntilang
import ntilang.language as T
import numpy as np
from ntilang.testing import reference


def test_factory_constants_in_deferred_annotations():
    def make(size):
        @T.prim_func
        def init(A: T.Tensor((size,), "float32")):
            with T.Kernel(1, threads=32) as _bx:
                for i in T.Parallel(32):
                    A[i] = 0.0

        return ntilang.compile(init)

    assert make(48).ir.parameters[0].type.shape == (48,)


def test_jit_factory():
    @ntilang.jit(target="sm_80")
    def make(size):
        @T.prim_func
        def init(A: T.Tensor((size,), "float32")):
            with T.Kernel(T.ceildiv(size, 32), threads=32) as bx:
                for i in T.Parallel(32):
                    A[bx * 32 + i] = 1.0

        return init

    kernel = make(79)
    assert isinstance(kernel, ntilang.CompiledKernel)
    assert kernel.ir.grid == (3,)
    assert make.__name__ == "make"


def test_factory_scalar_dtype_in_deferred_annotations():
    def make(dtype):
        @T.prim_func
        def init(value: dtype, A: T.Tensor((48,), dtype)):
            with T.Kernel(2, threads=32) as bx:
                for i in T.Parallel(32):
                    A[bx * 32 + i] = value

        return ntilang.compile(init)

    kernel = make(T.int16)
    output = np.empty(48, dtype=np.int16)
    reference(kernel, -321, output)
    np.testing.assert_array_equal(output, -321)
    assert kernel.ir.parameters[0].dtype == "int16"
