# Language semantics

This document describes the implemented Ntilang 0.1 subset.

## Parameters and shapes

Every parameter is a `T.Tensor(shape, dtype)` with positive static dimensions.
The dtype is one of `float16`, `bfloat16`, `float32`, or `int32`. Tensor size and
every intermediate integer index expression must fit signed 32-bit arithmetic.
Runtime tensors are contiguous in row-major order, 16-byte aligned, on the same
CUDA device, and have pairwise disjoint storage.

`@T.prim_func` captures a function whose source is available in a Python file.
The body contains one `with T.Kernel(..., threads=...)` block. Grid dimensions
are positive static integers within CUDA limits. Block and loop variables and
scalar local assignments use fresh names. Names beginning with `_nt_` are reserved.

## Tiles and iteration

`T.Parallel(d0, ..., dn)` defines a logical row-major tile. Its logical elements
are distributed over threads as `flat = thread + slot * threads`; excess slots
do not execute the body. This ownership is shared by linear register fragments.
Fragment indexing uses the exact variables of a parallel loop with the same
shape. MMA fragments have their own CuTe-defined distribution.

`T.serial(stop)` and `T.serial(start, stop)` have nonnegative static bounds and a
positive iteration count. Global outputs are written after serial accumulation
loops. `T.Pipelined(..., num_stages=0 or 1)` has the same synchronous semantics.

Shared and fragment buffers are allocated directly inside the kernel block.
Initialization is required before reads. `T.clear(tile)` and `T.fill(tile, value)`
initialize the complete logical tile. Shared allocation is limited to 48 KiB per
block in this version.

## Memory and copies

`T.copy(source, destination)` copies a whole temporary tile. At least one operand
is a named shared or fragment buffer, which determines the copy extent. An
expression such as `A[block_row, block_col]` identifies the origin of a global
tile, with the same rank as the temporary tile. Slices and partial temporary
copies are outside this grammar.

Each out-of-bounds global read returns zero. Each out-of-bounds global write is
suppressed. The implementation emits control-flow guards around reads, so an
invalid pointer is never dereferenced to compute the masked value.

Collective operations execute at block scope or inside uniform serial loops.
Shared copies, shared fills, and GEMM include synchronization around shared
accesses and reuse. Collectives inside `T.Parallel` are rejected.

Global write indices must satisfy the compiler's conservative affine ownership
rule. Every global output has a single write site and is not read by the kernel.
Aliasing between parameters is rejected at launch. Input tensors may be read
by multiple threads. Grid coverage remains part of the source program: positions
that the source does not write retain their prior contents.

## Arithmetic and GEMM

Scalar arithmetic lowers to CuTe's numeric operations with explicit casts on
stores and copies. `T.cast(value, dtype)` requests a conversion. Division and
remainder used with `//` and `%` are restricted to statically bounded nonnegative
integer expressions and positive constant divisors. Data-dependent indexing is
outside the supported subset.

`T.maximum` and `T.minimum` propagate NaNs. The generator does not request fast
math for exponential and square-root operations. Floating-point operations
follow the target compiler's CUDA semantics; bitwise identity with NumPy is not
specified.

`T.gemm(A, B, C)` means `C += A @ B`, after optional transposition of A and/or B.
A and B are complete shared tiles with matching FP16 or BF16 dtype. C is an FP32
fragment initialized by `T.clear` or `T.fill`. GEMM supports 32, 64, 128, or 256
threads, compatible M/N warp tiling, and K tile sizes divisible by 16. Multiple
GEMMs can accumulate into C with the same layout. The final copy from C writes
global memory through the accumulator's CuTe coordinate partition.

The current backend uses the SM80 warp MMA instruction family. Its shared copies
and register operand loads are synchronous. GPU numerical tests use tolerances
because Tensor Core accumulation order differs from the serial reference.

## Scope of verification

The compiler checks a restricted source contract, then delegates instruction
lowering and binary generation to NVIDIA CuTe DSL. The Python checker, CuTe
compiler, CUDA runtime, and hardware are components whose behavior must be
trusted and tested. An end-to-end machine-checked semantic-preservation theorem
is future work.

The NumPy evaluator implements serial mathematical tile semantics. It supports
FP16, FP32, and INT32 buffers and uses FP32 matrix multiplication for GEMM. It
does not simulate physical lane scheduling, register allocation, Tensor Core
rounding, or performance. GPU tests are the next validation layer for executable
behavior.
