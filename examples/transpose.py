import ntilang
import ntilang.language as T


def transpose(m=65, n=71, *, target="sm_80"):
    @T.prim_func
    def kernel(A: T.Tensor((m, n), "float32"), B: T.Tensor((n, m), "float32")):
        with T.Kernel(T.ceildiv(m, 32), T.ceildiv(n, 32), threads=128) as (bx, by):
            tile = T.alloc_shared((32, 32), "float32")
            for i, j in T.Parallel(32, 32):
                T.copy(A[bx * 32 + i, by * 32 + j], tile[i, j])
            for i, j in T.Parallel(32, 32):
                B[by * 32 + i, bx * 32 + j] = tile[j, i]

    return ntilang.compile(kernel, target=target)
