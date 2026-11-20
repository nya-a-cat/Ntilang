"""RMSNorm operator written in idiomatic TileLang syntax."""

from __future__ import annotations

import ntilang
import ntilang.language as T


def rmsnorm(m=4096, n=1024, block_m=4, *, eps=1e-6, target="sm_80"):
    block_n = 1 << (n - 1).bit_length()
    inv_n = 1.0 / n

    @T.prim_func
    def main(
        X: T.Tensor((m, n), "float32"),
        W: T.Tensor((n,), "float32"),
        Y: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(T.ceildiv(m, block_m), threads=128) as bx:
            x_frag = T.alloc_fragment((block_m, block_n), "float32")
            sq_frag = T.alloc_fragment((block_m, block_n), "float32")
            row_sq = T.alloc_fragment((block_m,), "float32")
            inv_rms = T.alloc_fragment((block_m,), "float32")
            for i, j in T.Parallel(block_m, block_n):
                if j < n:
                    val = X[bx * block_m + i, j]
                    x_frag[i, j] = val
                    sq_frag[i, j] = val * val
                else:
                    x_frag[i, j] = 0.0
                    sq_frag[i, j] = 0.0
            T.reduce_sum(sq_frag, row_sq, dim=1)
            for i in T.Parallel(block_m):
                inv_rms[i] = T.rsqrt(row_sq[i] * inv_n + eps)
            for i, j in T.Parallel(block_m, block_n):
                if j < n:
                    Y[bx * block_m + i, j] = x_frag[i, j] * inv_rms[i] * W[j]

    return ntilang.compile(main, target=target)


if __name__ == "__main__":
    print(rmsnorm().source)
