import importlib.util
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ir import CompileError
from ntilang.testing import reference


def scan_kernel(
    shape=(3, 37),
    dim=-1,
    reverse=False,
    kind="cumsum",
    dtype="float32",
    scope="shared",
    inplace=True,
    threads=64,
    out_dtype=None,
    target="sm_80",
):
    operation = getattr(T, kind)
    allocate = T.alloc_shared if scope == "shared" else T.alloc_fragment
    out_dtype = dtype if out_dtype is None else out_dtype

    @T.prim_func
    def kernel(A: T.Tensor(shape, dtype), B: T.Tensor(shape, out_dtype)):
        with T.Kernel(1, threads=threads):
            src = allocate(shape, dtype)
            dst = allocate(shape, out_dtype)
            T.copy(A, src)
            if inplace:
                operation(src=src, dim=dim, reverse=reverse)
                T.copy(src, B)
            else:
                operation(src, dst, dim, reverse, annotations={})
                T.copy(dst, B)

    return ntilang.compile(kernel, target=target)


def segmented_reference(a, dim, reverse, kind):
    """Independent transcription of pinned CUDA InclusiveScanLine arithmetic."""
    values = np.moveaxis(a, dim, -1)
    output = np.empty_like(values)
    identity = (
        0
        if kind == "cumsum" or a.dtype.kind == "u"
        else np.iinfo(a.dtype).min
        if a.dtype.kind == "i"
        else -np.inf
    )

    def operation(x, y):
        if kind == "cumsum":
            return np.add(x, y)
        result = np.fmax(x, y)
        if a.dtype.kind == "f":
            # The device max instruction ranks +0 above -0 independently
            # of host SIMD tie-breaking. Both negative inputs preserve -0.
            both_zero = (x == 0) & (y == 0)
            zero = np.where(np.signbit(x) & np.signbit(y), -0.0, 0.0)
            result = np.where(both_zero, zero, result).astype(a.dtype)
        return result

    for index in np.ndindex(values.shape[:-1]):
        row = values[index]
        carry = a.dtype.type(identity)
        segments = range((row.size + 31) // 32)
        for segment in reversed(segments) if reverse else segments:
            lanes = np.full(32, identity, dtype=a.dtype)
            valid = min(32, row.size - 32 * segment)
            lanes[:valid] = row[32 * segment : 32 * segment + valid]
            for distance in (1, 2, 4, 8, 16):
                before = lanes.copy()
                if reverse:
                    lanes[:-distance] = operation(before[:-distance], before[distance:])
                else:
                    lanes[distance:] = operation(before[distance:], before[:-distance])
            lanes = operation(lanes, carry).astype(a.dtype)
            output[index][32 * segment : 32 * segment + valid] = lanes[:valid]
            carry = lanes[0 if reverse else 31]
    return np.moveaxis(output, -1, dim)


@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((1,), 0),
        ((31,), -1),
        ((32,), 0),
        ((33,), 0),
        ((65,), 0),
        ((3, 37), -1),
        ((37, 3), 0),
        ((1, 7), 0),
        ((7, 1), 1),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "scope,inplace", [("shared", True), ("shared", False), ("fragment", True), ("fragment", False)]
)
def test_scan_reference(shape, dim, reverse, kind, scope, inplace):
    a = (np.arange(np.prod(shape)).reshape(shape) % 19 - 9).astype(np.float32)
    b = np.empty_like(a)
    kernel = scan_kernel(shape, dim, reverse, kind, scope=scope, inplace=inplace)
    reference(kernel, a, b)
    np.testing.assert_array_equal(b, segmented_reference(a, dim, reverse, kind))
    ordered = np.flip(a, axis=dim) if reverse else a
    expected = (np.add if kind == "cumsum" else np.fmax).accumulate(ordered, axis=dim)
    np.testing.assert_array_equal(b, np.flip(expected, axis=dim) if reverse else expected)


@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_typed_rounding(dtype, reverse, kind):
    a = np.random.default_rng(13).uniform(-2, 2, (2, 65)).astype(dtype)
    b = np.empty_like(a)
    with np.errstate(over="ignore", invalid="ignore"):
        reference(scan_kernel(a.shape, 1, reverse, kind, dtype), a, b)
        expected = segmented_reference(a, 1, reverse, kind)
    np.testing.assert_array_equal(b, expected)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
def test_scan_special_values(reverse, kind):
    a = np.resize(np.array([-0.0, 0.0, np.nan, -np.inf, np.inf, 1.0], dtype=np.float32), 65)
    b = np.empty_like(a)
    with np.errstate(invalid="ignore"):
        reference(scan_kernel(a.shape, 0, reverse, kind), a, b)
        expected = segmented_reference(a, 0, reverse, kind)
    np.testing.assert_array_equal(b, expected)
    np.testing.assert_array_equal(np.signbit(b[b == 0]), np.signbit(expected[expected == 0]))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"dim": 2}, "dim"),
        ({"dim": True}, "dim"),
        ({"reverse": 1}, "reverse"),
    ],
)
def test_scan_invalid_arguments(kwargs, match):
    with pytest.raises(CompileError, match=match):
        scan_kernel(**kwargs)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    ],
)
@pytest.mark.parametrize("dim,reverse,scope", [(0, False, "shared"), (1, True, "fragment")])
def test_scan_cute_compilation(kind, dtype, dim, reverse, scope):
    assert scan_kernel((3, 37), dim, reverse, kind, dtype, scope).build().has_gpu_module


def context_kernel(context="serial", kind="cumsum"):
    operation = getattr(T, kind)

    @T.macro
    def apply(tile):
        operation(tile, dim=-1)

    @T.prim_func
    def kernel(A: T.Tensor((3, 37), "float32"), B: T.Tensor((3, 37), "float32"), flag: T.int32):
        with T.Kernel(1):
            src = T.alloc_fragment((3, 37), "float32")
            T.copy(A, src)
            if context == "serial":
                for k in T.serial(2):
                    apply(src)
            elif context == "while":
                k = T.alloc_var("int32", init=0)
                while k < 2:
                    apply(src)
                    k += 1
            elif context == "conditional":
                if flag > 0:
                    apply(src)
                else:
                    operation(src, reverse=True, dim=-1)
            else:
                apply(src)
                apply(src)
            T.copy(src, B)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("context", ["serial", "while", "conditional", "macro"])
@pytest.mark.parametrize("flag", [0, 1])
@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
def test_scan_control_flow_workspace_reuse(context, flag, kind):
    kernel = context_kernel(context, kind)
    # Canonical scan lowering pools two workspaces outside control flow.
    for bank in ("a", "b"):
        assert kernel.source.count(f"_nt_scan_workspace_0_{bank} = _nt_smem.allocate_tensor(") == 1
    assert "_nt_scan_workspace_1_a" not in kernel.source
    a = (np.arange(111, dtype=np.float32).reshape(3, 37) % 7) - 3
    b = np.empty_like(a)
    reference(kernel, a, b, flag)
    expected = segmented_reference(a, 1, context == "conditional" and flag == 0, kind)
    if context != "conditional":
        expected = segmented_reference(expected, 1, False, kind)
    np.testing.assert_array_equal(b, expected)


def scan_region_kernel(kind="cumsum", reverse=False, partial_init=True):
    operation = getattr(T, kind)

    @T.prim_func
    def kernel(A: T.Tensor((4, 72), "float32"), B: T.Tensor((4, 72), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((4, 72), "float32")
            dst = T.alloc_shared((4, 72), "float32")
            T.copy(A, src)
            if partial_init:
                T.fill(dst, -5)
            operation(src[1:3, 2:67], dst[0:2, 4:69], dim=1, reverse=reverse)
            T.copy(dst, B)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_regions(kind, reverse):
    a = np.arange(288, dtype=np.float32).reshape(4, 72) % 7 - 3
    b = np.empty_like(a)
    expected = np.full_like(a, -5)
    expected[0:2, 4:69] = segmented_reference(a[1:3, 2:67], 1, reverse, kind)
    reference(scan_region_kernel(kind, reverse), a, b)
    np.testing.assert_array_equal(b, expected)


def test_scan_partial_destination_requires_initialization():
    with pytest.raises(CompileError, match="initialized destination"):
        scan_region_kernel(partial_init=False)


def test_scan_workspace_counts_towards_shared_limit():
    with pytest.raises(CompileError, match="48 KiB"):
        scan_kernel((4096,), dim=0)


def test_scan_fragment_output_conversion():
    a = (np.arange(111, dtype=np.float16).reshape(3, 37) % 7) - 3
    b = np.empty_like(a, dtype=np.float32)
    reference(scan_kernel(dtype="float16", out_dtype="float32", scope="fragment", inplace=False), a, b)
    np.testing.assert_array_equal(b, segmented_reference(a, 1, False, "cumsum").astype(np.float32))


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("context", ["serial", "while", "conditional", "macro"])
def test_scan_context_cute_compilation(context):
    assert context_kernel(context).build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_scan_regions_cute_compilation():
    assert scan_region_kernel().build().has_gpu_module


def mma_scan_kernel():
    @T.prim_func
    def kernel(
        A: T.Tensor((32, 32), "float16"), B: T.Tensor((32, 32), "float16"), C: T.Tensor((32, 32), "float32")
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((32, 32), "float16")
            b = T.alloc_shared((32, 32), "float16")
            acc = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.clear(acc)
            T.gemm(a, b, acc)
            T.cumsum(acc, dim=-1)
            T.copy(acc, C)

    return ntilang.compile(kernel)


def test_scan_preserves_mma_fragment_mapping():
    a = (np.arange(1024).reshape(32, 32) % 5).astype(np.float16)
    b = (np.arange(1024).reshape(32, 32) % 3).astype(np.float16)
    c = np.empty((32, 32), dtype=np.float32)
    reference(mma_scan_kernel(), a, b, c)
    expected = segmented_reference(a.astype(np.float32) @ b.astype(np.float32), 1, False, "cumsum")
    np.testing.assert_array_equal(c, expected)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
def test_scan_mma_cute_compilation():
    assert mma_scan_kernel().build().has_gpu_module


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
def test_scan_generated_module_is_standalone(kind, tmp_path):
    path = scan_kernel(kind=kind).save(tmp_path / "scan.py")
    code = (
        "import runpy, sys; module = runpy.run_path(sys.argv[1]); "
        "assert module['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)


def test_cumsum_example_tail_rows():
    from examples.cumsum import cumsum

    a = np.arange(7 * 65, dtype=np.float32).reshape(7, 65) % 11 - 5
    for reverse in (False, True):
        b = np.empty_like(a)
        reference(cumsum(reverse=reverse), a, b)
        np.testing.assert_array_equal(b, segmented_reference(a, 1, reverse, "cumsum"))


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
@pytest.mark.parametrize("reverse", [False, True])
def test_scans_on_gpu(kind, reverse):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA GPU available")
    major, minor = torch.cuda.get_device_capability()
    a = (torch.arange(3 * 65, device="cuda", dtype=torch.float32).reshape(3, 65) % 11) - 5
    b = torch.empty_like(a)
    scan_kernel((3, 65), 1, reverse, kind, target=f"sm_{major}{minor}")(a, b)
    expected = segmented_reference(a.cpu().numpy(), 1, reverse, kind)
    torch.testing.assert_close(b.cpu(), torch.from_numpy(expected), rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"shape": (2, 2, 2)}, {"threads": 48}, {"dtype": "bool"}])
def test_scan_extensions_remain_available(kwargs):
    shape, dtype = kwargs.get("shape", (3, 37)), kwargs.get("dtype", "float32")
    a = (np.arange(np.prod(shape)).reshape(shape) % 3).astype(dtype)
    b = np.empty_like(a)
    reference(scan_kernel(**kwargs), a, b)
    np.testing.assert_array_equal(b, segmented_reference(a, -1, False, "cumsum"))
