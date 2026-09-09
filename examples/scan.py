"""Row-wise inclusive sum and maximum with shared workspace reuse."""

import ntilang
import ntilang.language as T


def scan(m=9, n=113, *, reverse=False, target="sm_80"):
    @T.prim_func
    def kernel(
        A: T.Tensor((m, n), "float32"),
        Prefix: T.Tensor((m, n), "float32"),
        Maximum: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(T.ceildiv(m, 4), threads=128) as bx:
            values = T.alloc_fragment((4, n), "float32")
            prefix = T.alloc_fragment((4, n), "float32")
            T.copy(A[bx * 4 : bx * 4 + 4, :], values)
            T.cumsum(values, prefix, dim=-1, reverse=reverse)
            T.cummax(values, dim=-1, reverse=reverse)
            T.copy(prefix, Prefix[bx * 4 : bx * 4 + 4, :])
            T.copy(values, Maximum[bx * 4 : bx * 4 + 4, :])

    return ntilang.compile(kernel, target=target)


if __name__ == "__main__":
    print(scan().source)
