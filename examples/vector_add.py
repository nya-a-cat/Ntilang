import ntilang
import ntilang.language as T


def vector_add(n=1000, block=128, *, target="sm_80"):
    @T.prim_func
    def add(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, block), threads=128) as bx:
            for i in T.Parallel(block):
                C[bx * block + i] = A[bx * block + i] + B[bx * block + i]

    return ntilang.compile(add, target=target)


if __name__ == "__main__":
    kernel = vector_add()
    print(kernel.source)
