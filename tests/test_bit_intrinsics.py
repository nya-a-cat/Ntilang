import importlib.util
import itertools
import runpy
import subprocess
import sys

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.codegen import CUTLASS_TYPES
from ntilang.ir import DTYPES, Expr
from ntilang.scalar import expression_dtype
from ntilang.testing import reference
from ntilang.validation import affine, interval

requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed"
)
PAIRS = [(a, b) for a, b in itertools.product(DTYPES, repeat=2) if DTYPES[a] == DTYPES[b]]
COUNTS = [("popcount", d) for d in ("uint32", "uint64")] + [
    ("clz", d) for d in ("int32", "uint32", "int64", "uint64")
]


def words(bits):
    # Include both zeros, subnormals, infinities and quiet/signaling NaN payloads
    # when viewed as a float; integer cases also exercise both signed endpoints.
    specials = {
        8: [0, 1, 127, 128, 255],
        16: [0, 1, 0x8000, 0x7C00, 0xFC00, 0x7E31, 0x7C01, 0xFFFF, 0x7F80, 0x7FC1],
        32: [0, 1, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F800001, 0xFFFFFFFF],
        64: [
            0,
            1,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000001234,
            0x7FF0000000000001,
            0xFFFFFFFFFFFFFFFF,
        ],
    }[bits]
    data = np.frombuffer(np.random.default_rng(714).bytes(39 * bits // 8), dtype=f"uint{bits}").copy()
    data[: len(specials)] = specials
    return data


def reinterpret_kernel(source, target, roundtrip=False):
    @T.prim_func
    def kernel(A: T.Tensor((39,), source), B: T.Tensor((39,), source if roundtrip else target)):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                value = T.reinterpret(dtype=target, value=A[bx * 32 + i], span=None)
                if roundtrip:
                    B[bx * 32 + i] = T.reinterpret(source, value)
                else:
                    B[bx * 32 + i] = value

    return ntilang.compile(kernel)


def count_kernel(name, dtype, gather=False):
    operation = getattr(T, name)
    output = dtype if name == "popcount" else "int32"
    width = DTYPES[dtype] * 8

    @T.prim_func
    def kernel(A: T.Tensor((39,), dtype), B: T.Tensor((39,), output), LUT: T.Tensor((width + 1,), output)):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                count = operation(x=A[bx * 32 + i], dtype="float16")
                if gather:
                    B[bx * 32 + i] = LUT[count]
                else:
                    B[bx * 32 + i] = count

    return ntilang.compile(kernel)


@pytest.mark.parametrize("source,target", [p for p in PAIRS if "bfloat16" not in p])
def test_reinterpret_preserves_storage_bits(source, target):
    data = words(DTYPES[source] * 8)
    if "bool" in (source, target):
        data &= 1
    a = data.view(source)
    b = np.empty(39, dtype=target)
    reference(reinterpret_kernel(source, target), a, b)
    np.testing.assert_array_equal(b.view(data.dtype), data)
    assert expression_dtype(Expr("reinterpret", (Expr("parameter", value=source),), target), {}, {}) == target


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_reinterpret_roundtrip_preserves_nan_payloads(dtype):
    source = f"uint{DTYPES[dtype] * 8}"
    a = words(DTYPES[dtype] * 8)
    b = np.empty_like(a)
    reference(reinterpret_kernel(source, dtype, roundtrip=True), a, b)
    np.testing.assert_array_equal(b, a)


@pytest.mark.parametrize("name,dtype", COUNTS)
@pytest.mark.parametrize("gather", [False, True])
def test_counts_and_full_width_gathers(name, dtype, gather):
    width = DTYPES[dtype] * 8
    raw = words(width)
    a = raw.view(dtype)
    output = dtype if name == "popcount" else "int32"
    b = np.empty(39, dtype=output)
    lut = (np.arange(width + 1) * 7 + 3).astype(output)
    reference(count_kernel(name, dtype, gather), a, b, lut)
    expected = [int(x).bit_count() if name == "popcount" else width - int(x).bit_length() for x in raw]
    np.testing.assert_array_equal(b, lut[expected] if gather else expected)
    assert expression_dtype(Expr(name, (Expr("parameter", value=dtype),)), {}, {}) == output


@pytest.mark.parametrize(
    "source,target", [(a, b) for a, b in itertools.product(DTYPES, repeat=2) if DTYPES[a] != DTYPES[b]]
)
def test_reinterpret_width_mismatch(source, target):
    with pytest.raises(ntilang.CompileError, match="identical.*bit widths"):
        reinterpret_kernel(source, target)


@pytest.mark.parametrize(
    "name,dtype", [(n, d) for n, d in itertools.product(("clz", "popcount"), DTYPES) if (n, d) not in COUNTS]
)
def test_unsupported_count_types(name, dtype):
    with pytest.raises(ntilang.CompileError, match="pinned CUDA lowering"):
        count_kernel(name, dtype)


def test_invalid_boolean_object_representation():
    with pytest.raises(ValueError, match="valid byte representation"):
        reference(
            reinterpret_kernel("uint8", "bool"), np.full(39, 2, dtype=np.uint8), np.empty(39, dtype=bool)
        )


def test_bfloat16_reference_boundary():
    a = words(16)
    with pytest.raises(TypeError, match="bfloat16 reinterpretation"):
        reference(reinterpret_kernel("uint16", "bfloat16", roundtrip=True), a, np.empty_like(a))


@T.macro
def record(counter: T.Ref, digit):
    counter = counter * 10 + digit
    return T.uint32(digit)


@pytest.mark.parametrize("reverse,expected", [(False, 13), (True, 22)])
def test_discarded_dtype_keyword_preserves_construction_order(reverse, expected):
    @T.prim_func
    def kernel(B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                counter = T.alloc_var("int32", init=0)
                if reverse:
                    count = T.popcount(dtype=record(counter, 2), x=record(counter, 1))
                else:
                    count = T.popcount(record(counter, 1), dtype=record(counter, 2))
                B[i] = T.int32(count) + counter

    b = np.empty(32, dtype=np.int32)
    reference(ntilang.compile(kernel), b)
    np.testing.assert_array_equal(b, expected)


@pytest.mark.parametrize("mode", ["extra", "span", "duplicate", "missing"])
def test_count_call_diagnostics(mode):
    @T.prim_func
    def bad(A: T.Tensor((32,), "uint32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                if mode == "extra":
                    B[i] = T.clz(A[i], "int32")
                elif mode == "span":
                    B[i] = T.clz(A[i], span=None)
                elif mode == "duplicate":
                    B[i] = T.clz(A[i], x=A[i])
                else:
                    B[i] = T.clz(dtype="int32")

    with pytest.raises(ntilang.CompileError, match="positional|argument"):
        ntilang.compile(bad)


def test_constant_alias_counts_and_affine_reinterpret():
    bindings = {"word": Expr("reinterpret", (Expr("const", value=-1),), "uint32")}
    value = Expr("popcount", (Expr("var", value="word"),))
    assert interval(value, {}, bindings) == (32, 32)
    assert affine(value, bindings) == (32, {})
    value = Expr("reinterpret", (Expr("var", value="i"),), "uint32")
    assert affine(value, {}, {"i": (0, 31)}) == (0, {"i": 1})
    with pytest.raises(ntilang.CompileError, match="overflow signed 32-bit"):
        interval(value, {"i": (-1, 31)}, {})
    count = Expr("clz", (Expr("parameter", value="uint64"),))
    with pytest.raises(ntilang.CompileError, match="affine"):
        affine(count, {})


def bit_gather_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), LUT: T.Tensor((8,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                bits = T.reinterpret("int32", A[i])
                B[T.reinterpret("uint32", i)] = LUT[bits & 7]

    return ntilang.compile(kernel)


def test_masked_float_bit_gather():
    raw = words(32)[:32]
    lut = np.arange(8, dtype=np.int32) * 11
    output = np.empty(32, dtype=np.int32)
    reference(bit_gather_kernel(), raw.view("float32"), lut, output)
    np.testing.assert_array_equal(output, lut[raw & 7])


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("source,target", PAIRS)
def test_native_reinterpret_compilation(source, target):
    assert reinterpret_kernel(source, target).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("name,dtype", COUNTS)
@pytest.mark.parametrize("gather", [False, True])
def test_native_count_compilation(name, dtype, gather):
    assert count_kernel(name, dtype, gather).build().has_gpu_module


@pytest.mark.cuda
@requires_cute
def test_standalone_bit_gather(tmp_path):
    path = bit_gather_kernel().save(tmp_path / "bits.py")
    code = (
        "import runpy, sys; module = runpy.run_path(sys.argv[1]); "
        "assert module['compile_kernel']().has_gpu_module; assert 'ntilang' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], check=True, capture_output=True, text=True)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("floating", ["float16", "bfloat16", "float32", "float64"])
def test_actual_generated_bitcast_roundtrip_on_cpu(floating, tmp_path):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor

    integer = f"uint{DTYPES[floating] * 8}"
    module = runpy.run_path(reinterpret_kernel(integer, floating, roundtrip=True).save(tmp_path / "cast.py"))
    to_float = module[f"_nt_reinterpret_{integer}_{floating}"]
    to_integer = module[f"_nt_reinterpret_{floating}_{integer}"]
    dtype = getattr(cutlass, CUTLASS_TYPES[integer])

    @cute.jit
    def check(A: cute.Tensor, B: cute.Tensor):
        for i in cutlass.range_constexpr(39):
            B[i] = to_integer(to_float(A[i]))

    fake = make_fake_compact_tensor(dtype, (39,), memspace=cute.AddressSpace.generic, assumed_align=8)
    executable = cute.compile(check, fake, fake, options="--enable-tvm-ffi --gpu-arch=sm_80")
    assert not executable.has_gpu_module
    a = words(DTYPES[integer] * 8)
    b = np.empty_like(a)
    executable(a, b)
    np.testing.assert_array_equal(b, a)


@pytest.mark.cuda
@requires_cute
@pytest.mark.parametrize("floating", ["float16", "float32", "float64"])
@pytest.mark.parametrize("reverse", [False, True])
def test_actual_generated_one_way_bitcast_on_cpu(floating, reverse, tmp_path):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor

    integer = f"uint{DTYPES[floating] * 8}"
    source, target = (floating, integer) if reverse else (integer, floating)
    module = runpy.run_path(reinterpret_kernel(source, target).save(tmp_path / "cast.py"))
    convert = module[f"_nt_reinterpret_{source}_{target}"]

    @cute.jit
    def check(A: cute.Tensor, B: cute.Tensor):
        for i in cutlass.range_constexpr(39):
            B[i] = convert(A[i])

    fakes = [
        make_fake_compact_tensor(
            getattr(cutlass, CUTLASS_TYPES[d]), (39,), memspace=cute.AddressSpace.generic, assumed_align=8
        )
        for d in (source, target)
    ]
    executable = cute.compile(check, *fakes, options="--enable-tvm-ffi --gpu-arch=sm_80")
    assert not executable.has_gpu_module
    raw = words(DTYPES[source] * 8)
    a = raw.view(source)
    b = np.empty(39, dtype=target)
    executable(a, b)
    np.testing.assert_array_equal(b.view(integer), raw)
