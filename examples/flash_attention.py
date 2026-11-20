"""FlashAttention-2 forward pass written in idiomatic TileLang syntax."""

from __future__ import annotations

import math

import ntilang
import ntilang.language as T


def flash_attention(
    batch=1,
    heads=4,
    seq_len=256,
    dim=64,
    block_m=64,
    block_n=64,
    *,
    is_causal=False,
    num_stages=2,
    target="sm_80",
):
    scale = (1.0 / math.sqrt(dim)) * 1.44269504  # log2(e) / sqrt(d)
    neg_inf = -1e30

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, seq_len, dim), "float16"),
        K: T.Tensor((batch, heads, seq_len, dim), "float16"),
        V: T.Tensor((batch, heads, seq_len, dim), "float16"),
        Out: T.Tensor((batch, heads, seq_len, dim), "float16"),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_m, dim), "float16")
            K_shared = T.alloc_shared((block_n, dim), "float16")
            V_shared = T.alloc_shared((block_n, dim), "float16")
            acc_s = T.alloc_fragment((block_m, block_n), "float32")
            acc_s_cast = T.alloc_fragment((block_m, block_n), "float16")
            acc_o = T.alloc_fragment((block_m, dim), "float32")
            scores_max = T.alloc_fragment((block_m,), "float32")
            scores_max_prev = T.alloc_fragment((block_m,), "float32")
            scores_scale = T.alloc_fragment((block_m,), "float32")
            scores_sum = T.alloc_fragment((block_m,), "float32")
            logsum = T.alloc_fragment((block_m,), "float32")

            T.copy(Q[bz, by, bx * block_m : (bx + 1) * block_m, :], Q_shared)
            T.fill(acc_o, 0.0)
            T.fill(logsum, 0.0)
            T.fill(scores_max, -T.infinity("float32"))

            loop_range = T.ceildiv((bx + 1) * block_m, block_n) if is_causal else T.ceildiv(seq_len, block_n)
            for k in T.Pipelined(loop_range, num_stages=1 if is_causal else num_stages):
                T.copy(K[bz, by, k * block_n : (k + 1) * block_n, :], K_shared)
                T.copy(V[bz, by, k * block_n : (k + 1) * block_n, :], V_shared)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, clear_accum=True)
                if is_causal:
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_m + i >= k * block_n + j,
                            acc_s[i, j],
                            neg_inf,
                        )
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_m):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_m):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o)

            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Out[bz, by, bx * block_m : (bx + 1) * block_m, :])

    return ntilang.compile(main, target=target)


if __name__ == "__main__":
    print(flash_attention().source)
