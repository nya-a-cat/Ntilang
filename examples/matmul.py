import ntilang
import ntilang.language as T


def matmul(m=65, n=71, k=37, block_m=32, block_n=32, block_k=32, *, target="sm_80"):
    @T.prim_func
    def gemm(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), "float16")
            B_shared = T.alloc_shared((block_k, block_n), "float16")
            C_local = T.alloc_fragment((block_m, block_n), "float32")
            T.clear(C_local)
            for ko in T.serial(T.ceildiv(k, block_k)):
                T.copy(A[bx * block_m, ko * block_k], A_shared)
                T.copy(B[ko * block_k, by * block_n], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * block_m, by * block_n])

    return ntilang.compile(gemm, target=target)


if __name__ == "__main__":
    print(matmul().source)
