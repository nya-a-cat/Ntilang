"""Inclusive row scans with standalone CuTe DSL output."""

import ntilang
import ntilang.language as T


def cumsum(rows=7, columns=65, *, reverse=False, target="sm_80"):
    @T.prim_func
    def kernel(A: T.Tensor((rows, columns), "float32"), B: T.Tensor((rows, columns), "float32")):
        with T.Kernel(T.ceildiv(rows, 4), threads=128) as bx:
            tile = T.alloc_fragment((4, columns), "float32")
            T.copy(A[bx * 4 : bx * 4 + 4, :], tile)
            T.cumsum(tile, dim=-1, reverse=reverse)
            T.copy(tile, B[bx * 4 : bx * 4 + 4, :])

    return ntilang.compile(kernel, target=target)


if __name__ == "__main__":
    print(cumsum().source)
