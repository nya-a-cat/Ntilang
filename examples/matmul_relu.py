import ntilang
import ntilang.language as T


def matmul_relu(
    m=65,
    n=71,
    k=37,
    block_m=32,
    block_n=32,
    block_k=32,
    *,
    num_stages=1,
    swizzle_panel=1,
    target="sm_80",
):
    @T.prim_func
    def kernel(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        Bias: T.Tensor((m, n), "float32"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), threads=128) as (bx, by):
            sa = T.alloc_shared((block_m, block_k), "float16")
            sb = T.alloc_shared((block_k, block_n), "float16")
            acc = T.alloc_fragment((block_m, block_n), "float32")
            bias = T.alloc_fragment((block_m, block_n), "float32")
            T.use_swizzle(panel_size=swizzle_panel, enable=(swizzle_panel > 1))
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                T.copy(A[bx * block_m, ko * block_k], sa)
                T.copy(B[ko * block_k, by * block_n], sb)
                T.gemm(sa, sb, acc)
            T.copy(Bias[bx * block_m, by * block_n], bias)
            for i, j in T.Parallel(block_m, block_n):
                acc[i, j] = T.maximum(acc[i, j] * 0.5 + bias[i, j], 0.0)
            T.copy(acc, C[bx * block_m, by * block_n])

    return ntilang.compile(kernel, target=target)
