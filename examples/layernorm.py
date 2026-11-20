"""LayerNorm operator written in idiomatic TileLang syntax."""

from __future__ import annotations

import ntilang
import ntilang.language as T


def layernorm(m=4096, n=1024, block_m=4, *, eps=1e-5, target="sm_80"):
    block_n = 1 << (n - 1).bit_length()
    inv_n = 1.0 / n

    @T.prim_func
    def main(
        X: T.Tensor((m, n), "float32"),
        W: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        Y: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(T.ceildiv(m, block_m), threads=128) as bx:
            x_frag = T.alloc_fragment((block_m, block_n), "float32")
            diff_sq = T.alloc_fragment((block_m, block_n), "float32")
            row_sum = T.alloc_fragment((block_m,), "float32")
            row_var = T.alloc_fragment((block_m,), "float32")
            row_mean = T.alloc_fragment((block_m,), "float32")
            row_rstd = T.alloc_fragment((block_m,), "float32")
            for i, j in T.Parallel(block_m, block_n):
                x_frag[i, j] = X[bx * block_m + i, j] if j < n else 0.0
            T.reduce_sum(x_frag, row_sum, dim=1)
            for i in T.Parallel(block_m):
                row_mean[i] = row_sum[i] * inv_n
            for i, j in T.Parallel(block_m, block_n):
                if j < n:
                    d = x_frag[i, j] - row_mean[i]
                    diff_sq[i, j] = d * d
                else:
                    diff_sq[i, j] = 0.0
            T.reduce_sum(diff_sq, row_var, dim=1)
            for i in T.Parallel(block_m):
                row_rstd[i] = T.rsqrt(row_var[i] * inv_n + eps)
            for i, j in T.Parallel(block_m, block_n):
                if j < n:
                    Y[bx * block_m + i, j] = (x_frag[i, j] - row_mean[i]) * row_rstd[i] * W[j] + B[j]

    return ntilang.compile(main, target=target)


if __name__ == "__main__":
    print(layernorm().source)
