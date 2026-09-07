import ntilang
import ntilang.language as T


def fragment_affine(n=257, block=96, *, target="sm_80"):
    @T.prim_func
    def affine(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, block), threads=64) as bx:
            tile = T.alloc_fragment((block,), "float32")
            T.copy(A[bx * block], tile)
            for i in T.Parallel(block):
                B[bx * block + i] = tile[i] * 2.0 + 1.0

    return ntilang.compile(affine, target=target)


if __name__ == "__main__":
    print(fragment_affine().source)
