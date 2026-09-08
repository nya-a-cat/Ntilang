from __future__ import annotations

import importlib.util
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference


@T.macro
def generic_affine(source, destination, factor):
    tile = T.alloc_fragment(shape=source.shape, dtype=source.dtype)
    shared = T.alloc_shared(tile.shape, tile.dtype, scope="shared")
    T.copy(source, tile)
    rows, columns = source.shape
    for i, j in T.Parallel(rows, columns):
        tile[i, j] = tile[i, j] * factor + source.dtype(1)
    T.copy(tile, shared)
    T.copy(shared, destination)


def metadata_macro(dtype="float32"):
    @T.prim_func
    def kernel(A: T.Tensor(shape=(3, 11), dtype=dtype), B: T.Tensor((3, 11), dtype)):
        with T.Kernel(1, threads=32) as _bx:
            generic_affine(A, B, 2)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["int16", "float16", "float32", "float64"])
def test_macro_infers_buffers_and_conversions_from_metadata(dtype):
    a = np.arange(33, dtype=dtype).reshape(3, 11)
    b = np.empty_like(a)
    kernel = metadata_macro(dtype)
    reference(kernel, a, b)
    np.testing.assert_array_equal(b, a * 2 + 1)
    assert kernel.ir.parameters[0].strides == (11, 1)
    assert [buffer.strides for buffer in kernel.ir.buffers] == [(), ()]
    assert [buffer.source_scope for buffer in kernel.ir.buffers] == ["local.fragment", "shared"]


def buffer_metadata():
    @T.prim_func
    def kernel(B: T.Tensor(32, "int32", data=None, scope="global")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment(B.shape, B.dtype)
            shared = T.alloc_shared(B.shape, "bool", scope="shared.dyn")
            dims = tuple(B.shape)
            dtype = T.get_tvm_dtype(B.dtype)
            scope = shared.scope
            if (
                B.scope() == "global"
                and scope() == "shared"
                and tile.scope() == "local.fragment"
                and len(tile.strides) == 0
                and len(B.strides) == 1
                and len(dims) == 1
                and dtype == "int32"
                and T.dtype(int) == T.int32
                and str(dtype) == "int32"
                and dtype.bits == 32
                and dtype.bytes == 4
                and dtype.itemsize == 4
                and dtype.lanes == 1
                and dtype.type_code == 0
                and shared.dtype.bits == 8
                and shared.dtype.type_code == 6
                and B.data_alignment == 64
                and B.offset_factor == 1
                and B.buffer_type == 1
                and len(B.axis_separators) == 0
            ):
                size = B.shape[0].value
                stride = int(B.strides[0])
                offset = B.elem_offset.value
                for i in T.Parallel(size):
                    B[i] = size + stride + offset + i
            else:
                T.unavailable_operation()

    return ntilang.compile(kernel)


def test_buffer_metadata_preserves_source_fields_and_bool_scope_override():
    b = np.empty(32, dtype=np.int32)
    reference(buffer_metadata(), b)
    np.testing.assert_array_equal(b, 33 + np.arange(32))


def prelude_bindings():
    @T.prim_func
    def kernel(A: T.Tensor(39), alpha: T.float32, B: T.Tensor(39)):
        width = A.shape[0]
        shape = A.shape
        dtype = A.dtype
        scale = alpha * 2 + 1
        bx = 17
        prior_block_name = bx
        if dtype == "float32" and len(shape) == 1:
            block_size = 32
        else:
            block_size = 16
        with T.Kernel(T.ceildiv(width, block_size), threads=32) as bx:
            for i in T.Parallel(block_size):
                B[bx * block_size + i] = A[bx * block_size + i] * scale + prior_block_name

    return ntilang.compile(kernel)


def test_prelaunch_metadata_and_scalar_parameters_inline_into_device_body():
    a, b = np.arange(39, dtype=np.float32), np.empty(39, dtype=np.float32)
    kernel = prelude_bindings()
    reference(kernel, a, 1.5, b)
    np.testing.assert_array_equal(b, a * 4 + 17)
    assert kernel.ir.grid == (2,)
    assert kernel.ir.block_vars != ("bx",)
    assert kernel.ir.body[0].op == "parallel"


def construction_phases():
    @T.prim_func
    def kernel(B: T.Tensor(32, "int32")):
        before = B.shape[0]
        (unpacked_before,) = B.shape
        with T.Kernel(1, threads=32) as _bx:
            after = B.shape[0]
            (unpacked_after,) = B.shape
            marker = 0
            if before == 32:
                marker += 1
            else:
                marker += 2
            if unpacked_before == 32:
                marker += 4
            else:
                marker += 8
            if after == 32 and unpacked_after == 32:
                marker += 16
            else:
                T.unavailable_operation()
            for i in T.Parallel(32):
                B[i] = marker

    return ntilang.compile(kernel)


def test_prelaunch_primexprs_preserve_eager_construction_phase():
    b = np.empty(32, dtype=np.int32)
    kernel = construction_phases()
    reference(kernel, b)
    np.testing.assert_array_equal(b, 31)
    assert sum(statement.op == "if" for statement in kernel.ir.body) == 2


def scalar_metadata(dtype="float16"):
    @T.prim_func
    def kernel(A: T.Tensor(32, dtype), count: T.uint8, B: T.Tensor(32, "float64")):
        with T.Kernel(1, threads=32) as _bx:
            temp = T.alloc_fragment(A.shape, A.dtype)
            mutable = T.alloc_var("uint16")
            if temp[0].dtype == dtype and mutable.dtype == "uint16" and count.dtype == "uint8":
                for i in T.Parallel(32):
                    original = A[i]
                    original_type = original.dtype
                    computed_type = (original + 1).dtype
                    original = T.float64(original)
                    if original_type == dtype and computed_type == dtype and original.dtype == "float64":
                        B[i] = original + count.dtype(2) + i.dtype(3)
                    else:
                        T.unavailable_operation()
            else:
                T.unavailable_operation()

    return ntilang.compile(kernel)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_scalar_metadata_tracks_rebinding_and_does_not_read_uninitialized_buffers(dtype):
    a = np.arange(32, dtype=dtype)
    b = np.empty(32, dtype=np.float64)
    reference(scalar_metadata(dtype), a, 7, b)
    np.testing.assert_array_equal(b, a.astype(np.float64) + 5)


@T.macro
def typed_macro_receiver(counter: T.Ref):
    counter += 1
    return T.float64(7)


def metadata_receiver_effects():
    @T.prim_func
    def kernel(B: T.Tensor(32, "float64")):
        with T.Kernel(1, threads=32) as _bx:
            counter = T.alloc_var("int32")
            value = typed_macro_receiver(counter).dtype(3)
            for i in T.Parallel(32):
                B[i] = value + counter

    return ntilang.compile(kernel)


def test_metadata_call_receiver_expands_once():
    b = np.empty(32, dtype=np.float64)
    reference(metadata_receiver_effects(), b)
    np.testing.assert_array_equal(b, 4)


def test_prelaunch_buffer_read_is_diagnosed():
    @T.prim_func
    def bad(A: T.Tensor(32), B: T.Tensor(32)):
        value = A[0]
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = value

    with pytest.raises(ntilang.CompileError, match="before T.Kernel"):
        ntilang.compile(bad)


def test_dynamic_python_int_conversion_is_diagnosed():
    @T.prim_func
    def bad(A: T.Tensor(32, "int32"), B: T.Tensor(32, "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                value = int(i)
                B[i] = A[i] + value

    with pytest.raises(ntilang.CompileError, match="constant integer IR"):
        ntilang.compile(bad)


def test_unsupported_buffer_metadata_is_diagnosed():
    @T.prim_func
    def bad(B: T.Tensor(32, "int32")):
        with T.Kernel(1, threads=32) as _bx:
            pointer = B.data
            for i in T.Parallel(32):
                B[i] = pointer

    with pytest.raises(ntilang.CompileError, match="metadata attribute 'data'"):
        ntilang.compile(bad)


requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize(
    "factory",
    [
        metadata_macro,
        buffer_metadata,
        prelude_bindings,
        construction_phases,
        scalar_metadata,
        metadata_receiver_effects,
    ],
)
def test_metadata_native_compilation(factory):
    assert factory().build().has_gpu_module


@pytest.mark.cuda
@requires_cute
def test_generated_metadata_module_is_standalone(tmp_path):
    path = metadata_macro().save(tmp_path / "metadata.py")
    code = "import runpy, sys; module = runpy.run_path(sys.argv[1]); assert module['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)
