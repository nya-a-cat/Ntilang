# Language semantics

This document describes the implemented Ntilang 0.1 subset.

## Parameters and shapes

Every parameter is a `T.Tensor(shape, dtype)` with positive static dimensions.
The dtype is Boolean, signed or unsigned 8/16/32/64-bit integer, or
`float16`, `bfloat16`, `float32`, or `float64`. Tensor size and
every intermediate integer index expression must fit signed 32-bit arithmetic.
Runtime tensors are contiguous in row-major order, 16-byte aligned, on the same
CUDA device, and have pairwise disjoint storage.

`@T.prim_func` captures a function whose source is available in a Python file.
The body contains one `with T.Kernel(..., threads=...)` block. Grid dimensions
are positive static integers within CUDA limits. Block and loop variables and
scalar local assignments use fresh names. Names beginning with `_nt_` are reserved.

Dtype symbols work as tensor dtype names and scalar casts within parsed bodies,
for example `T.Tensor((32,), T.float32)` and `T.float32(value)`. Common aliases
include `short`, `int`, `uint`, `long`, `half`, `float`, and `double`. Dtype
descriptors expose `bits` and `bytes`; Boolean has one logical bit and one byte
of tensor storage. Sub-byte and vector dtype variants remain open compatibility
work. Calling a dtype constructor outside a parsed body is currently unsupported.

## Tiles and iteration

`T.Parallel(d0, ..., dn)` defines a logical tile. Without an MMA layout constraint,
its elements are distributed as `flat = thread + slot * threads` in row-major
order; excess slots do not execute the body.
Pointwise fragment indexing uses the exact variables of a parallel loop with the
same shape. Broadcasts and other cross-element reads materialize a synchronized
shared-memory view before the parallel loop. Those source fragments cannot also
be written in that loop. Out-of-bounds cross-element reads return zero.
Parallel operations connected to an MMA accumulator use its CuTe-defined
coordinate partition. Whole-fragment copies and pointwise operations propagate
that partition through connected fragments, including earlier initialization
and copies. Connections between incompatible MMA partitions currently require
a layout conversion and produce a compilation error.

`T.serial(stop)`, `T.serial(start, stop)`, and `T.serial(start, stop, step)` follow
Python's static integer range semantics, including negative steps and empty
domains. A zero step is rejected. Induction arithmetic is formed in 64 bits
before conversion to its checked 32-bit range. `T.unroll` has the same iteration
domain and uses CuTe compile-time iteration. Its tuning annotations remain open
compatibility work. Global outputs are currently written after serial
accumulation loops. `T.Pipelined(..., num_stages=0 or 1)` is synchronous.

Fragment elements can be assigned or updated with operators such as
`+=` inside the matching parallel tile. An unconditional full parallel write
initializes the fragment. Empty loops do not initialize their body allocations
or buffers. MMA epilogues support scalar expressions and additional fragments
with the same logical shape, including fragments with a different storage dtype.

Shared and fragment buffers are allocated directly inside the kernel block.
Initialization is required before reads. `T.clear(tile)` and `T.fill(tile, value)`
initialize the complete logical tile. Shared allocation is limited to 48 KiB per
block in this version, including storage introduced for fragment communication.

## Memory and copies

`T.copy(source, destination)` accepts whole buffers, positive unit-stride slices,
and fixed indices within sliced regions. Slice extents must simplify to static
integers. Unit dimensions are matched through the remaining non-unit extents.
For example, `T.copy(A[row, :], tile)` copies a matrix row to a vector tile.
An operand containing only indices is a tile origin whose extent is inferred
from the other operand. Two whole buffers require equal shapes. Two indexed
elements inside a parallel loop form a scalar copy with dtype conversion.

Global-to-global copies and partial shared/fragment copies are supported. A
partial temporary destination must already be initialized so its untouched
elements have defined values. A full-region copy initializes its destination.
Copies between overlapping regions of one temporary buffer capture source
values before writing the destination. Fragment regions use their physical
owners or a synchronized shared view when communication is required.

The synchronous lowering accepts `prefer_instruction="sync"`, `disable_tma`,
and the normal eviction policy. Supported annotation values override individual
keyword options. Explicit vector widths, custom copy layouts, other eviction
policies, and asynchronous instruction preferences require further lowering.

Each out-of-bounds global read returns zero. Each out-of-bounds global write is
suppressed. The implementation emits control-flow guards around reads, so an
invalid pointer is never dereferenced to compute the masked value.

Collective operations execute at block scope or inside uniform serial loops.
Shared copies, shared fills, and GEMM include synchronization around shared
accesses and reuse. Parallel loops can initialize or update shared elements at
their matching logical indices; barriers surround these loops. Cross-element
reads from a shared tile that is also written in the same parallel loop are
rejected. Collective tile copies inside `T.Parallel` are rejected.

Global write indices must satisfy the compiler's conservative affine ownership
rule. Multiple write sites to an output require proven disjoint address ranges
or mutually exclusive branches with the same affine ownership mapping.
This restriction prevents branches taken by
different lanes from writing the same location. Outputs are not read by the kernel.
Aliasing between parameters is rejected at launch. Input tensors may be read
by multiple threads. Grid coverage remains part of the source program: positions
that the source does not write retain their prior contents.

## Conditional statements

`if`, `elif`, `else`, and `pass` preserve conditional execution. A scalar first
defined in a branch is available after the conditional only when both paths
define it. Joined scalars receive a common numeric type and a declaration before
the generated CuTe branch. Fragment initialization must hold on both paths;
conditionally updating an already initialized fragment is allowed.

Branches inside `T.Parallel` can depend on element values. Collective operations
remain outside parallel loops. At block scope, predicates use block-uniform
indices and values, so every thread participates in any selected shared-memory
barriers. Branch-dependent global write mappings remain outside the currently
implemented ownership proof. `while`, scalar mutation, and loop exits still need
implementation.

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

The upstream `T.max` and `T.min` spellings prefer a non-NaN operand. The two
longer spellings are Ntilang extensions. Scalar promotion follows the checked
CuTe numeric rules, including integer signedness and width, promotion of narrow
integers with default `int32` constants, and floating/integer width conversion.
The reference evaluator applies these rules explicitly before arithmetic.

`T.gemm(A, B, C)` means `C += A @ B`, after optional transposition of A and/or B.
A and B are complete shared tiles with matching FP16 or BF16 dtype. C is an FP32
fragment initialized by fill, copy, or a complete parallel assignment. GEMM supports 32, 64, 128, or 256
threads, compatible M/N warp tiling, and K tile sizes divisible by 16. Multiple
GEMMs can accumulate into C with the same layout. Copies involving C use the
accumulator's CuTe coordinate partition, with shared-memory synchronization
when appropriate. Elementwise epilogues preserve that coordinate mapping.

The current backend uses the SM80 warp MMA instruction family. Its shared copies
and register operand loads are synchronous. GPU numerical tests use tolerances
because Tensor Core accumulation order differs from the serial reference.

## Reductions

`T.reduce(buffer, out, reduce_type, dim, clear)` supports `sum`, `abssum`,
`max`, `absmax`, `min`, `bitand`, `bitor`, and `bitxor`. The corresponding
`T.reduce_*` wrappers preserve their upstream defaults and argument names.
Source and destination may be shared or fragment buffers. The selected axis
is static; the output removes that axis or keeps it with extent one. A rank-one
input can use a `(1,)` output. Inputs are converted to the output dtype before
combining. Bitwise reductions require an integer output.

With `clear=True`, the output starts from the operation's identity. With
`clear=False`, its initialized value is combined once with the reduction result.
`nan_propagate=True` affects FP16/BF16 max/min/absmax, following the upstream
signature. Other floating max/min reductions prefer non-NaN operands; an
all-NaN maximum with clearing returns negative infinity. Sum propagates NaNs.

The backend materializes a typed shared workspace, then combines disjoint pairs
in tree levels with uniform barriers between levels. The input buffer is
preserved. MMA input fragments retain their coordinate mapping when entering
the workspace. Tree order can change floating-point rounding relative to other
implementations. `batch=1` and empty lowering annotations are supported; batched
AllReduce scheduling, packed arithmetic controls, and reducer epochs remain open.
Reduction workspace counts toward the shared-memory limit.

## Scope of verification

The compiler checks a restricted source contract, then delegates instruction
lowering and binary generation to NVIDIA CuTe DSL. The Python checker, CuTe
compiler, CUDA runtime, and hardware are components whose behavior must be
trusted and tested. An end-to-end machine-checked semantic-preservation theorem
is future work.

The NumPy evaluator implements serial mathematical tile semantics. It supports
the basic numeric and Boolean buffer types except BF16, and uses FP32 matrix multiplication for GEMM. It
does not simulate physical lane scheduling, register allocation, Tensor Core
rounding, or performance. GPU tests are the next validation layer for executable
behavior.
