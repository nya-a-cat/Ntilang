import ntilang
import ntilang.language as T


def softmax(m=9, n=113, *, target="sm_80"):
    block_n = 1 << (n - 1).bit_length()
    negative_inf = -float("inf")

    @T.prim_func
    def kernel(A: T.Tensor((m, n), "float32"), B: T.Tensor((m, n), "float32")):
        with T.Kernel(T.ceildiv(m, 4), threads=128) as bx:
            tile = T.alloc_fragment((4, block_n), "float32")
            row_max = T.alloc_fragment((4,), "float32")
            row_sum = T.alloc_fragment((4,), "float32")
            for i, j in T.Parallel(4, block_n):
                if j < n:
                    tile[i, j] = A[bx * 4 + i, j]
                else:
                    tile[i, j] = negative_inf
            T.reduce_max(tile, row_max, dim=1)
            for i, j in T.Parallel(4, block_n):
                tile[i, j] = T.exp(tile[i, j] - row_max[i])
            T.reduce_sum(tile, row_sum, dim=1)
            for i, j in T.Parallel(4, block_n):
                B[bx * 4 + i, j] = tile[i, j] / row_sum[i]

    return ntilang.compile(kernel, target=target)
