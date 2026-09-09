import importlib.util
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.testing import reference

CUDA = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)
DTYPES = (
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
)


def scan_kernel(
    shape=(3, 67),
    kind="cumsum",
    dim=-1,
    reverse=False,
    scopes=("fragment", "fragment"),
    dtype="float32",
    out_dtype=None,
    threads=64,
    inplace=False,
    target="sm_80",
):
    operation = getattr(T, kind)
    allocate_src = T.alloc_fragment if scopes[0] == "fragment" else T.alloc_shared
    allocate_dst = T.alloc_fragment if scopes[1] == "fragment" else T.alloc_shared
    out_dtype = dtype if out_dtype is None else out_dtype

    @T.prim_func
    def kernel(A: T.Tensor(shape, dtype), B: T.Tensor(shape, out_dtype)):
        with T.Kernel(1, threads=threads):
            src = allocate_src(shape, dtype)
            dst = allocate_dst(shape, out_dtype)
            T.copy(A, src)
            if inplace:
                operation(src, dim=dim, reverse=reverse)
                T.copy(src, B)
            else:
                operation(src, dst, dim, reverse, {})
                T.copy(dst, B)

    return ntilang.compile(kernel, target=target)


def mathematical_scan(a, kind, dim, reverse):
    value = np.flip(a, dim) if reverse else a
    operation = np.add if kind == "cumsum" else np.fmax
    value = operation.accumulate(value, axis=dim, dtype=a.dtype)
    return np.flip(value, dim) if reverse else value


@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((1,), 0),
        ((31,), -1),
        ((32,), 0),
        ((33,), 0),
        ((97,), -1),
        ((3, 67), 1),
        ((67, 3), 0),
        ((1, 65), -1),
        ((65, 1), 0),
        ((2, 5, 37), 2),
        ((3, 35, 2), 1),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "scopes,inplace",
    [
        (("fragment", "fragment"), False),
        (("fragment", "fragment"), True),
        (("fragment", "shared"), False),
        (("shared", "shared"), False),
        (("shared", "shared"), True),
    ],
)
def test_scans_shapes_scopes_and_directions(kind, shape, dim, reverse, scopes, inplace):
    a = (np.arange(np.prod(shape)).reshape(shape) % 11 - 5).astype(np.float32)
    b = np.full_like(a, np.nan)
    reference(scan_kernel(shape, kind, dim, reverse, scopes, inplace=inplace), a, b)
    np.testing.assert_array_equal(b, mathematical_scan(a, kind, dim, reverse))


@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_numeric_dtypes(kind, dtype, reverse):
    a = (np.arange(105).reshape(3, 35) % 7).astype(dtype)
    b = np.zeros_like(a)
    reference(scan_kernel(a.shape, kind, reverse=reverse, dtype=dtype), a, b)
    with np.errstate(over="ignore", invalid="ignore"):
        np.testing.assert_array_equal(b, mathematical_scan(a, kind, -1, reverse))


def segmented_oracle(a, kind, axis, reverse):
    # Independent per-line transcription of InclusiveScanLine in upstream
    # src/tl_templates/cuda/scan.h, including padded lanes and identity carry.
    result = np.empty_like(a)
    axes = a.shape[:axis] + a.shape[axis + 1 :]
    initial = a.dtype.type(0 if kind == "cumsum" else -np.inf)

    def operation(left, right):
        if kind == "cumsum":
            return np.add(left, right)
        if a.dtype == np.dtype("float64"):
            return right if left < right else left
        if left == 0 and right == 0:
            return a.dtype.type(-0.0 if np.signbit(left) and np.signbit(right) else 0.0)
        return np.fmax(left, right)

    for coordinate in np.ndindex(axes):
        index = coordinate[:axis] + (slice(None),) + coordinate[axis:]
        line = a[index]
        out = np.empty_like(line)
        carry = initial
        count = (len(line) + 31) // 32
        order = range(count - 1, -1, -1) if reverse else range(count)
        for segment in order:
            begin = segment * 32
            lanes = np.full(32, initial, dtype=a.dtype)
            lanes[: len(line[begin : begin + 32])] = line[begin : begin + 32]
            for offset in (1, 2, 4, 8, 16):
                before = lanes.copy()
                for lane in range(32):
                    other = lane + offset if reverse else lane - offset
                    if 0 <= other < 32:
                        lanes[lane] = operation(before[lane], before[other])
            for lane in range(32):
                lanes[lane] = operation(lanes[lane], carry)
                if begin + lane < len(line):
                    out[begin + lane] = lanes[lane]
            carry = lanes[0 if reverse else 31]
        result[index] = out
    return result


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_scan_segment_tree_rounding_and_special_values(dtype, kind, reverse, axis):
    rng = np.random.default_rng(407)
    a = (rng.normal(size=(3, 97)) * 30).astype(dtype)
    a[0, :6] = [0.0, -0.0, np.inf, -np.inf, np.nan, np.nan]
    a[1, :] = np.nan
    a = a.T.copy() if axis == 0 else a
    b = np.empty_like(a)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = segmented_oracle(a, kind, axis, reverse)
        reference(scan_kernel(a.shape, kind, axis, reverse, dtype=dtype), a, b)
    np.testing.assert_array_equal(b, expected)
    np.testing.assert_array_equal(np.signbit(b[b == 0]), np.signbit(expected[expected == 0]))


def sliced_scan(scope="fragment", kind="cumsum", reverse=False, target="sm_80"):
    allocate = T.alloc_fragment if scope == "fragment" else T.alloc_shared
    operation = getattr(T, kind)

    @T.prim_func
    def kernel(A: T.Tensor((7, 80), "float32"), B: T.Tensor((7, 80), "float32")):
        with T.Kernel(1, threads=96):
            tile = allocate((7, 80), "float32")
            T.copy(A, tile)
            # Unequal row pitch, nonzero row/column origins, overlapping rows
            # and columns, multiple segments, and a non-power-of-two block.
            operation(tile[1:5, 3:70], tile[2:6, 9:76], dim=1, reverse=reverse)
            T.copy(tile, B)

    return ntilang.compile(kernel, target=target)


@pytest.mark.parametrize("scope", ["fragment", "shared"])
@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_pitched_subregions(scope, kind, reverse):
    a = (np.arange(560).reshape(7, 80) % 17 - 8).astype(np.float32)
    b = np.empty_like(a)
    expected = a.copy()
    expected[2:6, 9:76] = mathematical_scan(a[1:5, 3:70].copy(), kind, 1, reverse)
    reference(sliced_scan(scope, kind, reverse), a, b)
    np.testing.assert_array_equal(b, expected)


def helper_kernel(kind="cumsum_fragment", dtype="float16", out_dtype="float32"):
    operation = getattr(T, kind)

    @T.prim_func
    def kernel(A: T.Tensor((97,), dtype), B: T.Tensor((97,), out_dtype)):
        with T.Kernel(2, threads=96) as bx:
            operation(A[bx * 64 : bx * 64 + 64], B[bx * 64 : bx * 64 + 64], 0, False, None)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("kind", ["cumsum_fragment", "cummax_fragment"])
def test_explicit_fragment_helpers_global_tiles_and_tail(kind):
    a = (np.arange(97) % 9 - 4).astype(np.float16)
    b = np.empty(97, np.float32)
    reference(helper_kernel(kind), a, b)
    expected = np.concatenate(
        [mathematical_scan(part, kind.removesuffix("_fragment"), 0, False) for part in (a[:64], a[64:])]
    ).astype(np.float32)
    np.testing.assert_array_equal(b, expected)


def control_flow_scan():
    @T.macro
    def prefix(src, dst):
        T.cumsum(src=src, dst=dst, dim=-1)

    @T.prim_func
    def kernel(A: T.Tensor((3, 37), "float32"), B: T.Tensor((3, 37), "float32"), flag: T.bool):
        with T.Kernel(1, threads=64):
            src = T.alloc_fragment((3, 37), "float32")
            dst = T.alloc_fragment((3, 37), "float32")
            T.copy(A, src)
            T.clear(dst)
            for iteration in T.serial(2):
                if flag:
                    prefix(src, dst)
                else:
                    T.cummax(src, dst, reverse=True)
                T.copy(dst, src)
            T.copy(dst, B)

    return ntilang.compile(kernel)


@pytest.mark.parametrize("flag", [False, True])
def test_scan_macro_uniform_branches_and_serial_workspace_reuse(flag):
    a = (np.arange(111).reshape(3, 37) % 9 - 4).astype(np.float32)
    b = np.empty_like(a)
    reference(control_flow_scan(), a, b, flag)
    expected = a.copy()
    for _ in range(2):
        expected = mathematical_scan(expected, "cumsum" if flag else "cummax", -1 if flag else 0, not flag)
    np.testing.assert_array_equal(b, expected)


def mma_scan():
    @T.prim_func
    def kernel(
        A: T.Tensor((32, 32), "float16"), B: T.Tensor((32, 32), "float16"), C: T.Tensor((32, 32), "float32")
    ):
        with T.Kernel(1, threads=128):
            sa = T.alloc_shared((32, 32), "float16")
            sb = T.alloc_shared((32, 32), "float16")
            acc = T.alloc_fragment((32, 32), "float32")
            T.copy(A, sa)
            T.copy(B, sb)
            T.clear(acc)
            T.gemm(sa, sb, acc)
            T.cumsum(acc[1:30, 2:29], acc[2:31, 3:30], dim=1)
            T.copy(acc, C)

    return ntilang.compile(kernel)


def test_mma_scan_subregion_capture():
    a = (np.arange(1024).reshape(32, 32) % 3).astype(np.float16)
    b = (np.arange(1024).reshape(32, 32) % 5).astype(np.float16)
    c = np.empty((32, 32), np.float32)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    expected[2:31, 3:30] = np.cumsum(expected[1:30, 2:29].copy(), axis=1)
    reference(mma_scan(), a, b, c)
    np.testing.assert_array_equal(c, expected)


@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
def test_scalar_load_regions_preserve_unit_axes(kind):
    operation = getattr(T, kind)

    @T.prim_func
    def kernel(A: T.Tensor((3, 17), "int32"), B: T.Tensor((3, 17), "int32")):
        with T.Kernel(1, threads=32):
            tile = T.alloc_shared((3, 17), "int32")
            T.copy(A, tile)
            operation(tile[1, 2], tile[2, 3], dim=-1)
            operation(tile[0, :], tile[1, :], dim=1)
            T.copy(tile, B)

    a = np.arange(51, dtype=np.int32).reshape(3, 17)
    b = np.empty_like(a)
    reference(ntilang.compile(kernel), a, b)
    expected = a.copy()
    expected[2, 3] = a[1, 2]
    expected[1, :] = mathematical_scan(a[0, :], kind, 0, False)
    np.testing.assert_array_equal(b, expected)


@pytest.mark.parametrize("dim", [-3, 2, True, 0.5])
def test_invalid_scan_dimensions(dim):
    with pytest.raises(ntilang.CompileError, match="dimension"):
        scan_kernel(dim=dim)


@pytest.mark.parametrize("reverse", [1, None, "yes"])
def test_invalid_scan_reverse(reverse):
    with pytest.raises(ntilang.CompileError, match="reverse"):
        scan_kernel(reverse=reverse)


def invalid_scan(case):
    @T.prim_func
    def kernel(A: T.Tensor((3, 37), "float32"), B: T.Tensor((3, 37), "float32")):
        with T.Kernel(1, threads=64):
            src = T.alloc_shared((3, 37), "float32")
            dst = T.alloc_fragment((3, 37), "float32")
            if case == "uninitialized_source":
                T.cumsum(src)
            T.copy(A, src)
            if case == "partial_destination":
                T.cumsum_fragment(src[1:3, :], dst[1:3, :], 1, False)
            elif case == "shape":
                T.cumsum(src[1:3, :], src, dim=1)
            elif case == "bounds":
                T.cumsum(src[2:4, :], dim=1)
            elif case == "stride":
                T.cumsum(src[:, ::2])
            elif case == "annotations":
                T.cumsum(src, annotations={"unknown": 1})
            elif case == "annotation_type":
                T.cumsum(src, annotations=1)
            elif case == "parallel":
                for i in T.Parallel(3):
                    T.cumsum(src)
            elif case == "global":
                T.cumsum(A, B)
            elif case == "scope":
                T.cumsum(src, dst)
            elif case == "missing":
                T.cumsum_fragment(src, dst)
            elif case == "duplicate":
                T.cumsum(src, src=src)
            elif case == "unknown_keyword":
                T.cumsum(src, axis=1)
            T.copy(src, B)

    return ntilang.compile(kernel)


@pytest.mark.parametrize(
    "case,match",
    [
        ("uninitialized_source", "before initialization"),
        ("partial_destination", "initialized destination"),
        ("shape", "destination shape"),
        ("bounds", "provably in bounds"),
        ("stride", "unit stride"),
        ("annotations", "annotations"),
        ("annotation_type", "annotations"),
        ("parallel", "Collective"),
        ("global", "shared-to-shared"),
        ("scope", "shared-to-shared"),
        ("missing", "Missing required"),
        ("duplicate", "duplicate"),
        ("unknown_keyword", "argument"),
    ],
)
def test_scan_diagnostics(case, match):
    with pytest.raises(ntilang.CompileError, match=match):
        invalid_scan(case)


def test_direct_shared_scan_dtype_mismatch():
    with pytest.raises(ntilang.CompileError, match="matching.*dtypes"):
        scan_kernel(scopes=("shared", "shared"), out_dtype="float16")


def test_scan_padded_workspaces_count_toward_shared_memory():
    with pytest.raises(ntilang.CompileError, match="48 KiB"):
        scan_kernel((6145,), threads=128)
    assert scan_kernel((6144,), threads=128).source.count("_nt_smem.allocate_tensor") == 2


def test_scan_workspace_reuse_and_standalone_imports():
    source = control_flow_scan().source
    assert source.count("_nt_smem.allocate_tensor") == 4  # dim=1 and dim=0 have different padded shapes
    assert "from ntilang" not in source and "import ntilang" not in source
    compile(source, "scan_cute.py", "exec")


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("dtype", (*DTYPES, "bfloat16"))
@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
def test_scan_all_basic_dtypes_cute_compilation(dtype, kind):
    assert scan_kernel((3, 37), dtype=dtype, kind=kind, reverse=True).build().has_gpu_module


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("scope", ["fragment", "shared"])
@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
def test_scan_subregions_cute_compilation(scope, kind):
    assert sliced_scan(scope, kind, reverse=True).build().has_gpu_module


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("factory", [control_flow_scan, mma_scan, helper_kernel])
def test_scan_compositions_cute_compilation(factory):
    assert factory().build().has_gpu_module


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("dim,threads", [(0, 32), (1, 96), (2, 257)])
def test_scan_rank_three_arbitrary_threads_cute_compilation(dim, threads):
    assert scan_kernel((3, 5, 37), dim=dim, threads=threads).build().has_gpu_module


@pytest.mark.cuda
@CUDA
def test_scan_single_thread_cute_compilation():
    assert scan_kernel((33,), threads=1).build().has_gpu_module


def test_scan_accumulates_in_source_dtype_before_output_conversion():
    a = np.ones(65, np.float16)
    a[0] = 2048
    b = np.empty(65, np.float32)
    reference(scan_kernel(a.shape, dtype="float16", out_dtype="float32"), a, b)
    expected = segmented_oracle(a, "cumsum", 0, False).astype(np.float32)
    np.testing.assert_array_equal(b, expected)
    assert np.any(b != segmented_oracle(a.astype(np.float32), "cumsum", 0, False))


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "uint32", "int64", "uint64"])
def test_cumsum_integer_wrap_and_full_width_values(dtype):
    info = np.iinfo(dtype)
    a = np.full((3, 65), info.max, dtype=dtype)
    b = np.empty_like(a)
    reference(scan_kernel(a.shape, dtype=dtype), a, b)
    modulus = 1 << info.bits
    expected = []
    for index in range(65):
        value = (int(info.max) * (index + 1)) % modulus
        if info.min < 0 and value > info.max:
            value -= modulus
        expected.append(value)
    np.testing.assert_array_equal(b, np.tile(np.asarray(expected, dtype=dtype), (3, 1)))


def test_scan_macro_arguments_are_evaluated_once_in_keyword_order():
    @T.macro
    def mark(tile, count: T.Ref):
        count += 1
        T.fill(tile, count)
        return tile

    @T.prim_func
    def kernel(B: T.Tensor((37,), "int32"), C: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=32):
            src = T.alloc_fragment((37,), "int32")
            dst = T.alloc_fragment((37,), "int32")
            count = T.alloc_var("int32", init=0)
            T.cumsum(dst=mark(dst, count), src=mark(src, count))
            T.copy(dst, B)
            for i in T.Parallel(1):
                C[i] = count

    b, c = np.empty(37, np.int32), np.empty(1, np.int32)
    reference(ntilang.compile(kernel), b, c)
    np.testing.assert_array_equal(b, np.arange(1, 38) * 2)
    assert c[0] == 2


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("kind,dtype", [("cumsum", "float16"), ("cummax", "float64")])
def test_scan_standalone_module_compiles_without_ntilang(tmp_path, kind, dtype):
    path = scan_kernel(kind=kind, dtype=dtype).save(tmp_path / "scan_cute.py")
    code = (
        "import importlib.util, sys; "
        "spec = importlib.util.spec_from_file_location('scan_cute', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "assert module.compile_kernel().has_gpu_module; "
        "assert 'ntilang' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["cumsum", "cummax"])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_overlapping_subregions_on_gpu(kind, reverse):
    torch = pytest.importorskip("torch", reason="Install CUDA-enabled PyTorch for GPU checks")
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA GPU available")
    major, minor = torch.cuda.get_device_capability()
    a = (torch.arange(560, device="cuda").reshape(7, 80) % 17 - 8).float()
    b = torch.empty_like(a)
    sliced_scan("fragment", kind, reverse, target=f"sm_{major}{minor}")(a, b)
    values = a[1:5, 3:70].clone()
    if reverse:
        values = values.flip(1)
    values = values.cumsum(1) if kind == "cumsum" else values.cummax(1).values
    expected = a.clone()
    expected[2:6, 9:76] = values.flip(1) if reverse else values
    torch.testing.assert_close(b, expected, rtol=0, atol=0)
