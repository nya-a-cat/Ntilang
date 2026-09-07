import ntilang
import ntilang.language as T


def piecewise(n=93, *, target="sm_80"):
    @T.prim_func
    def kernel(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 32), threads=32) as bx:
            for i in T.Parallel(32):
                x = A[bx * 32 + i]
                if x < 0.0:
                    B[bx * 32 + i] = -x
                elif x < 1.0:
                    B[bx * 32 + i] = x * x
                else:
                    B[bx * 32 + i] = x + 2.0

    return ntilang.compile(kernel, target=target)
