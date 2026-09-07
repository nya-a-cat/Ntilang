import ntilang
import ntilang.language as T


def matmul_relu(m=65, n=71, k=37, *, target="sm_80"):
    @T.prim_func
    def kernel(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        Bias: T.Tensor((m, n), "float32"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(T.ceildiv(m, 32), T.ceildiv(n, 32), threads=128) as (bx, by):
            sa = T.alloc_shared((32, 32), "float16")
            sb = T.alloc_shared((32, 32), "float16")
            acc = T.alloc_fragment((32, 32), "float32")
            bias = T.alloc_fragment((32, 32), "float32")
            T.copy(Bias[bx * 32, by * 32], bias)
            T.clear(acc)
            for ko in T.serial(T.ceildiv(k, 32)):
                T.copy(A[bx * 32, ko * 32], sa)
                T.copy(B[ko * 32, by * 32], sb)
                T.gemm(sa, sb, acc)
            for i, j in T.Parallel(32, 32):
                acc[i, j] = T.maximum(acc[i, j] * 0.5 + bias[i, j], 0.0)
            T.copy(acc, C[bx * 32, by * 32])

    return ntilang.compile(kernel, target=target)
