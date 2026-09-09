"""Sum runtime-sized windows within statically shaped batches."""

import ntilang
import ntilang.language as T


def window_sum(batch=7, height=3, width=5, *, target="sm_80"):
    @T.prim_func
    def kernel(
        A: T.Tensor((batch, height, width), "float32"),
        B: T.Tensor((batch,), "float32"),
        rows: T.int32,
        columns: T.int32,
    ):
        with T.Kernel(T.ceildiv(batch, 128), threads=128) as bx:
            for item in T.Parallel(128):
                total = T.alloc_var("float32", init=0)
                for i, j in T.grid(T.clamp(rows, 0, height), T.clamp(columns, 0, width)):
                    total += A[bx * 128 + item, i, j]
                B[bx * 128 + item] = total

    return ntilang.compile(kernel, target=target)


if __name__ == "__main__":
    print(window_sum().source)
