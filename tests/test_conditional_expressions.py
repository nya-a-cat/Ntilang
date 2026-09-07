import importlib.util
import operator

import ntilang
import ntilang.language as T
import numpy as np
import pytest
from ntilang.ir import Expr
from ntilang.testing import reference
from ntilang.validation import predicate_bounds


def guarded_division(lazy=True):
    choose = T.if_then_else if lazy else T.Select

    @T.prim_func
    def kernel(A: T.Tensor((39,), "int32"), B: T.Tensor((39,), "int32"), C: T.Tensor((39,), "int32")):
        with T.Kernel(2, threads=32) as bx:
            for i in T.Parallel(32):
                C[bx * 32 + i] = choose(
                    B[bx * 32 + i] != 0,
                    T.truncdiv(A[bx * 32 + i], B[bx * 32 + i]),
                    0,
                )

    return ntilang.compile(kernel)


def conditional_index(lazy=True):
    choose = T.if_then_else if lazy else T.Select

    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = A[choose(i != 0, i // i, 0)]

    return ntilang.compile(kernel)


def nested_choice():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = T.Select(
                    condition=A[i] < 0,
                    true_value=-A[i],
                    false_value=T.if_then_else(cond=A[i] < 1, t=A[i] * A[i], f=A[i] + 2, span=None),
                    span=None,
                )

    return ntilang.compile(kernel)


def mixed_choice(lazy=True):
    choose = T.if_then_else if lazy else T.Select

    @T.prim_func
    def kernel(A: T.Tensor((32,), "float16"), B: T.Tensor((32,), "int64"), C: T.Tensor((32,), "float64")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                C[i] = choose(i < 16, A[i], B[i])

    return ntilang.compile(kernel)


def fragment_selection():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32) as _bx:
            tile = T.alloc_fragment((32,), "float32")
            output = T.alloc_fragment((32,), "float32")
            T.copy(A, tile)
            for i in T.Parallel(32):
                output[i] = T.if_then_else(i < 16, tile[i + 16], tile[i - 16])
            T.copy(output, B)

    return ntilang.compile(kernel)


def mma_selection():
    @T.prim_func
    def kernel(
        A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 8), "float16"), C: T.Tensor((16, 8), "float32")
    ):
        with T.Kernel(1, threads=32) as _bx:
            a = T.alloc_shared((16, 16), "float16")
            b = T.alloc_shared((16, 8), "float16")
            acc = T.alloc_fragment((16, 8), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.clear(acc)
            T.gemm(a, b, acc)
            for i, j in T.Parallel(16, 8):
                acc[i, j] = T.if_then_else(acc[i, j] >= 0, acc[i, j], 0.0)
            T.copy(acc, C)

    return ntilang.compile(kernel)


def test_lazy_branch_guards_division_and_padded_elements():
    a = np.arange(-19, 20, dtype=np.int32)
    b = np.resize(np.array([0, 1, -2, 3], dtype=np.int32), 39)
    out = np.empty_like(a)
    reference(guarded_division(), a, b, out)
    expected = np.array([int(int(x) / int(y)) if y else 0 for x, y in zip(a, b)], dtype=np.int32)
    np.testing.assert_array_equal(out, expected)


def test_select_does_not_guard_unselected_division():
    a = np.ones(39, dtype=np.int32)
    b, out = np.zeros_like(a), np.empty_like(a)
    with pytest.raises(ValueError, match="nonzero divisor"):
        reference(guarded_division(False), a, b, out)


def test_lazy_predicate_refines_index_divisor_bounds():
    a, out = np.arange(32, dtype=np.int32), np.empty(32, dtype=np.int32)
    reference(conditional_index(), a, out)
    np.testing.assert_array_equal(out, np.array([0, *([1] * 31)], dtype=np.int32))
    with pytest.raises(ntilang.CompileError, match="index divisor can be zero"):
        conditional_index(False)


def test_nested_conditional_values():
    a = np.linspace(-2, 2, 32, dtype=np.float32)
    out = np.empty_like(a)
    reference(nested_choice(), a, out)
    np.testing.assert_array_equal(out, np.where(a < 0, -a, np.where(a < 1, a * a, a + 2)))


def test_lazy_branch_type_matching_and_select_type_constraint():
    a = np.full(32, 2048, dtype=np.float16)
    b, out = np.full(32, 2049, dtype=np.int64), np.empty(32, dtype=np.float64)
    reference(mixed_choice(), a, b, out)
    np.testing.assert_array_equal(out, np.full(32, 2048.0))
    with pytest.raises(ntilang.CompileError, match="identical true and false"):
        mixed_choice(False)


def test_conditional_fragment_communication():
    a, out = np.arange(32, dtype=np.float32), np.empty(32, dtype=np.float32)
    reference(fragment_selection(), a, out)
    np.testing.assert_array_equal(out, np.roll(a, 16))


def test_conditional_mma_epilogue():
    rng = np.random.default_rng(911)
    a, b = rng.normal(size=(16, 16)).astype(np.float16), rng.normal(size=(16, 8)).astype(np.float16)
    out = np.empty((16, 8), dtype=np.float32)
    reference(mma_selection(), a, b, out)
    np.testing.assert_allclose(
        out, np.maximum(a.astype(np.float32) @ b.astype(np.float32), 0), rtol=3e-5, atol=3e-5
    )


def test_conditional_expression_requires_boolean_predicate():
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = T.Select(A[i], 1, 0)

    with pytest.raises(ntilang.CompileError, match="Boolean condition"):
        ntilang.compile(bad)


@pytest.mark.parametrize(
    "relation,operation",
    [
        ("<", operator.lt),
        ("<=", operator.le),
        (">", operator.gt),
        (">=", operator.ge),
        ("==", operator.eq),
        ("!=", operator.ne),
    ],
)
@pytest.mark.parametrize("coefficient", [-3, 1, 2])
@pytest.mark.parametrize("truth", [False, True])
def test_predicate_bounds_preserve_all_matching_values(relation, operation, coefficient, truth):
    lhs = Expr(
        "+", (Expr("*", (Expr("const", value=coefficient), Expr("var", value="i"))), Expr("const", value=2))
    )
    condition = Expr(relation, (lhs, Expr("const", value=5)))
    result = predicate_bounds(condition, truth, {"i": (-8, 8)}, {})
    for value in range(-8, 9):
        if operation(coefficient * value + 2, 5) == truth:
            assert result is not None and result["i"][0] <= value <= result["i"][1]


def test_unsigned_conversion_does_not_enable_signed_predicate_refinement():
    condition = Expr("<", (Expr("var", value="i"), Expr("cast", (Expr("const", value=0),), "uint32")))
    bounds = {"i": (-4, 3)}
    assert predicate_bounds(condition, False, bounds, {}) == bounds


def test_tensor_predicate_can_choose_bounded_indices():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                B[i] = A[T.if_then_else(A[0] > 0, i, 31 - i)]

    a, out = np.arange(32, dtype=np.int32), np.empty(32, dtype=np.int32)
    reference(ntilang.compile(kernel), a, out)
    np.testing.assert_array_equal(out, a[::-1])


def test_output_index_predicates_participate_in_read_tracking():
    @T.prim_func
    def bad(A: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32) as _bx:
            for i in T.Parallel(32):
                A[T.if_then_else(A[i] == 0, i, i)] = 1

    with pytest.raises(ntilang.CompileError, match="both read and written"):
        ntilang.compile(bad)


@pytest.mark.cuda
@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="CuTe DSL compiler is not installed")
@pytest.mark.parametrize(
    "factory",
    [guarded_division, conditional_index, nested_choice, mixed_choice, fragment_selection, mma_selection],
)
def test_conditional_expression_compilation(factory):
    assert factory().build().has_gpu_module
