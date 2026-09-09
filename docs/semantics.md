# Language semantics

This document describes the implemented Ntilang 0.1 subset.

## Parameters and shapes

Parameters use `T.Tensor(shape, dtype="float32", data=None, scope=None)` with positive static dimensions or a basic
scalar dtype annotation. The dtype is Boolean, signed or unsigned 8/16/32/64-bit integer, or
`float16`, `bfloat16`, `float32`, or `float64`. Tensor size and
every intermediate integer index expression must fit signed 32-bit arithmetic.
Runtime tensors are contiguous in row-major order, 16-byte aligned, on the same
CUDA device, and have pairwise disjoint storage.
Scalar arguments remain runtime inputs, interspersed with tensors in declaration
order. Rebinding a parameter inside a kernel preserves its original input ABI.
An integer shape denotes a one-dimensional tensor. Tensor arguments accept
positional and keyword forms; explicit data pointers and parameter scopes other
than `None` or `"global"` require further lowering.

`@T.prim_func` captures a function whose source is available in a Python file.
The body ends with one `with T.Kernel(..., threads=...)` block. Pure construction
bindings may precede it, including buffer aliases, shape/type queries, static
Python conditionals, and scalar parameter expressions. These scalar expressions
are inlined into their device uses. Host buffer reads, allocations, loops, and
runtime control flow preceding the launch remain unsupported. Grid dimensions
are positive static integers within CUDA limits. Runtime scalar definitions,
local allocations, and loop variables receive distinct internal identities.
Source names can be rebound; existing values and buffer aliases keep their
earlier identities. Names beginning with `_nt_` are reserved.
The block-variable `as ...` binding is optional. Omitting grid dimensions uses
one block. Kernels may have only scalar parameters or no parameters, and may
perform diagnostics without writing a tensor output.

Dtype symbols work as tensor dtype names and scalar casts within parsed bodies,
for example `T.Tensor((32,), T.float32)` and `T.float32(value)`. Common aliases
include `short`, `int`, `uint`, `long`, `ulong`, `half`, `float`, and `double`.
`T.dtype(value)` and `T.get_tvm_dtype(value)` accept these supported names, dtype
objects, and the Python `int`, `float`, and `bool` types. Descriptors expose
`bits`, `bytes`, `itemsize`, `lanes`, and `type_code`. The pinned TVM version
represents Boolean metadata with 8 bits, one storage byte, and DLPack type code 6.
Sub-byte/vector descriptors and NumPy/PyTorch dtype conversion remain open.
Scalar cast syntax such as `T.float32(value)` requires a parsed body.

## Construction metadata

Buffers expose `shape`, `dtype`, `strides`, `scope()`, `elem_offset`,
`data_alignment`, `offset_factor`, `buffer_type`, and `axis_separators`.
Tensor parameters have explicit row-major strides; shared/fragment allocations
retain the upstream empty stride list. Shape, stride, and element-offset entries
are integer IR constants. The supported declarations have element offset zero,
offset factor 1, buffer type 1, and no axis separators. The source alignment
metadata is 64 bytes; generated pointer operations use the conservative 16-byte
alignment checked by the current runtime.

`T.alloc_shared(shape, dtype, scope="shared.dyn")` and
`T.alloc_fragment(shape, dtype, scope="local.fragment")` accept argument names
and supported scopes `shared`, `shared.dyn`, and `local.fragment`. Shared Boolean
allocation follows the upstream override to `shared`. Other memory scopes remain
open. Buffer pointer/name/span objects and general buffer construction APIs
require further integration.

IR scalars and buffer elements expose their resolved `dtype`, including scalar
parameters and mutable values. An element dtype query inspects its buffer
declaration without reading its contents. Dtype values can select static source
branches or perform scalar casts, for example `A.dtype(value)`.

Construction supports `len` and `tuple` on built-in containers, `str` on primitive
values/dtypes, and `int` on primitive values or constant integer IR values.
Integer metadata constants also expose `value`. Each of these conversion calls
currently requires one positional argument.

The eager binding phase is preserved: `n = A.shape[0]` before the launch keeps
an integer IR constant. The same assignment inside the Kernel frame converts
an int32 constant into a Python integer. A subsequent condition on the former
builds both source arms; a Python condition on the latter selects one source arm.

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
Python's integer range semantics, including negative steps and empty
domains. Start and stop may also be integer expressions with proven 32-bit
ranges; the step remains static. Runtime bounds are captured once on loop entry.
Trip-count arithmetic is formed in 64 bits, clamped at zero for empty domains,
and checked against the 32-bit loop-count limit. A zero step is rejected.
Induction arithmetic also uses 64 bits before conversion to its checked 32-bit
range. `T.unroll` has the same iteration
domain. `explicit=True` uses CuTe compile-time iteration; the default emits a
full-unroll compiler hint. `unroll_factor` emits a factor hint, with 0 and 1
disabling unrolling. These hints preserve the iteration domain, including tails
when the trip count is not divisible by the factor. The compiler may optimize
the final loop according to its backend rules. `pragma_unroll_explicit` and
`pragma_unroll_factor` annotations follow the upstream precedence: a true
`explicit` argument and a non-None factor override their annotation values.
Explicit expansion and a factor are mutually exclusive. Expansion requires static bounds;
dynamic loops retain compiler unroll hints. Dynamic loops do not establish new
buffer initialization after the loop. Other scheduling options remain open
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

`if`, `elif`, `else`, and `pass` follow the default eager parser. A Python
construction-time condition selects the source branch to expand. A runtime
condition creates a conditional IR statement, and both source branches are
constructed in order. Runtime values and buffer aliases created inside a branch
retain that defining region and cannot be read after it. Fragment initialization
must hold on both paths; conditionally updating an initialized fragment is allowed.

Python scalar constants, tuples, and strings update during construction. For
example, assigning `value = 1` and `value = 2` in the two arms of a runtime
conditional leaves the Python value `2` after constructing those arms. Declare
`value = T.alloc_var("int32")` before the conditional to make those assignments
update a runtime scalar. Rebinding a runtime value to another expression creates
a fresh immutable definition; ordinary aliases preserve their earlier values.
Reallocating a buffer or local.var under the same source name similarly creates
a separate allocation. Chained assignments capture their right-hand side before
target updates and target-specific dtype conversions.

Whitelisted construction expressions include built-in scalar arithmetic,
comparisons, Boolean operations, tuples/dictionaries, indexing, basic metadata, dtype/operation
aliases, and conditional expressions. Macro-returned Python values participate in
these expressions. User Python function bodies and arbitrary object operators
are not executed by the source parser. General container mutation, comprehensions,
and remaining object/metadata APIs still need implementation.

Branches inside `T.Parallel` can depend on element values. Collective operations
remain outside parallel loops. At block scope, predicates use block-uniform
indices and values, so every thread participates in any selected shared-memory
barriers. Global writes use the disjoint-region and exclusive-branch ownership
checks described above.

`T.Select(condition, true_value, false_value, span=None)` selects between values
with identical dtypes and a Boolean condition. Its value expressions can both
be evaluated; it does not provide a guard for division or memory operations.
`T.if_then_else(cond, t, f, span=None)` evaluates the selected value expression
and matches the two branch dtypes using the scalar promotion rules. Generated
code places branch-local loads and arithmetic inside the corresponding branch.
Both forms compose with fragment communication and MMA epilogues.
Chained Python comparisons follow the pinned eager AST rewrite: adjacent
comparisons are combined through Boolean operations, and middle source
expressions are repeated. A macro in a repeated runtime Boolean branch receives
the upstream macro restriction. Static Boolean short-circuiting selects the
construction expressions that execute.

Local `name: T.dtype = expression` bindings follow the default upstream eager
frontend: the expression determines the value dtype. The annotation is retained
as source metadata and does not insert a conversion. Use an explicit dtype
constructor or `T.cast` for conversion. An annotation without a value can refer
to an existing scalar; it does not initialize a new variable. Legacy TIR parser
annotation behavior and non-scalar type annotations require separate support.

`T.alloc_var(dtype, ..., scope="local.var", init=None)` creates a mutable scalar.
Its default value is zero, following the pinned CUDA allocation lowering.
The supported positional initializer/scope forms and keyword initializer cast
to the declared dtype. Assignments and augmented assignments preserve that
dtype, including narrowing after each loop update. An ordinary scalar binding
such as `saved = accumulator` captures its value at that point.

Mutable values can carry state through serial/unrolled loops and conditional
updates. Scalars declared inside a parallel loop are initialized separately for
each logical element. Scalars declared outside parallel loops can be updated
by uniform statements and read inside parallel loops. Updating them inside a
parallel loop requires additional per-thread state mapping and is currently
diagnosed. Mutable integer reads use their dtype range in index analysis;
their initializer is never substituted as a bound for later reads. Proving
more precise mutable ranges and ownership still requires loop-state analysis.

`while` evaluates a Boolean condition before the first iteration and again
after every body execution. Condition-local tensor loads and conditional
expressions are regenerated at both evaluation sites, so updates remain visible.
Mutable scalars retain their values through nested loops. The body may run zero
times; its new runtime bindings and buffer initialization do not escape the loop.
Python constant updates happen once while constructing the loop body.
Uniform while loops may contain collective tile operations. Per-element loops
inside `T.Parallel` retain the restriction against collective operations.
Termination and absence of arithmetic overflow are caller preconditions for
data-dependent loops. Statically true conditions receive the upstream eager
infinite-loop diagnostic. Loop `else` remains open.

`break` and `continue` target the nearest serial, unroll, or while loop. The
statement forms `T.loop_break()`, `T.break_loop(span=None)`, and
`T.continue_loop(span=None)` have the same control effect. Direct unconditional
exits discard the following statements in that block. Nested loops carry
independent control flags. A while-loop break skips the next condition
evaluation; continue reevaluates it normally. For loops predicate the remaining
iterations after break, preserving the iteration bound and requested unroll hint.
Explicit expansion rejects a break targeting that loop, following the upstream
unroll pass; breaks targeting a nested loop remain valid. Direct exits from
`T.Parallel` and source span objects require further lowering. A loop with early
exits does not establish new buffer initialization after the loop.

For lazy branches, the index checker refines single-variable affine integer
comparisons, their negation, true conjunctions, and false disjunctions. This
allows guards such as `i != 0` to establish a nonzero divisor when zero is an
interval endpoint. Conditions requiring disjoint ranges or general relational
constraints still need broader analysis. Index predicates participate in global
read tracking, even when both selected address expressions are identical.

## Scalar arithmetic

Scalar arithmetic lowers to CuTe's numeric operations with explicit casts on
operands, stores, and copies. `T.cast(value, dtype)` requests a conversion.
Integer data-dependent reads use the same masked loads as other indices.
Type ranges, integer min/max, and bit masks can bound their addresses. All index
intermediates must still fit the checked 32-bit domain and their own dtypes.
Data-dependent scatter writes require additional ownership analysis.

`T.hypot(x1, x2)`, `T.nextafter(x1, x2)`, and `T.ldexp(x1, x2)` preserve the
first operand's `float32` or `float64` result type and accept positional or named
operands. They use typed `cute.extern` declarations for CUDA libdevice calls.
The generated module contains these declarations and compiles independently
of Ntilang. Half, BF16, integer, and vector result forms require further lowering.

`hypot` uses the library's scaled hypotenuse calculation to avoid undue
intermediate overflow/underflow. `nextafter` selects the adjacent representable
value in its result precision, including signed-zero and subnormal transitions.
Their second argument is converted to the first argument's type. `ldexp`
converts its exponent directly to signed int32; integer narrowing retains the
low 32 bits. A floating exponent must be finite and fit int32 after truncation.
This is the CUDA/C++ conversion precondition. Exponent conversion happens
independently of the floating input's type. See the CUDA libdevice documentation
for [hypot](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hypotf.html),
[nextafter](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nextafterf.html),
and [ldexp](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ldexpf.html).
Reference checks cover these boundary cases; GPU numerical behavior remains unverified.

`T.ieee_add(x, y)`, `ieee_sub`, `ieee_mul`, `ieee_fdiv`, `ieee_fmaf(x, y, z)`,
`ieee_frcp(x)`, and `ieee_fsqrt(x)` accept a static `rounding_mode`, defaulting
to `"rn"`. The modes are nearest with ties to even (`rn`), toward zero (`rz`),
toward positive infinity (`ru`), and toward negative infinity (`rd`). FP32 and
FP64 use the corresponding typed CUDA libdevice calls. Arguments convert to
the first operand's dtype before the operation. `T.ieee_frsqrt(x)` accepts one
argument and uses the correctly rounded FP32 reciprocal-square-root intrinsic;
the pinned TileLang CUDA lowering rejects FP64 for this operation.
`T.fma(x, y, z)` and `T.fmul(x, y)` require identical floating argument dtypes
and use nearest rounding. FMA rounds its product-plus-sum once; fmul preserves
an explicit multiply boundary. These signatures and restrictions follow
[TileLang's scalar intrinsics](https://github.com/tile-ai/tilelang/blob/62bba8d20ddb232e29050770472cb2649dd3e718/tilelang/language/math_intrinsics.py)
and [CUDA math extension](https://github.com/tile-ai/tilelang/blob/62bba8d20ddb232e29050770472cb2649dd3e718/tilelang/cuda/language/math.py).

FP16/BF16 accept `rn` only for these interfaces. Generated standalone helpers
use native half arithmetic and FMA instructions. On SM80 through SM89, BF16
add/subtract/multiply use the CUDA header's BF16 FMA identities; SM90 and newer
use direct BF16 instructions. With the common CUDA header, FP16 reciprocal, square root, and reciprocal
square root widen to FP32 approximate instructions with FTZ, then convert back.
Their BF16 counterparts use FP32 approximate instructions without FTZ.
FP16 division retains the approximate reciprocal and the two FMA corrections
for a nonzero rounded result below half bit pattern `0x008f`. BF16 division
scales denominators with magnitude at least `2**126`, performs approximate
division, and rescales with FMA. These instruction choices follow the CUDA
12.9 device headers; [PTX instruction semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
define the approximation and rounding behavior. Half/BF16 root and division
paths inherit those approximations. Their numerical results have not been
measured on hardware.

The pinned TileLang emitter includes `math.h` when an emitted expression uses
`__exp`, `__log`, `__sin`, `__cos`, or `fast_rcp`. That header aliases `hsqrt`
to CUTLASS's FP32 `sqrtf` wrapper. In such kernels, FP16/BF16 `ieee_fsqrt`
therefore uses FP32 library square root followed by conversion. Ntilang retains
this kernel-wide interaction, including when the square root appears before
the triggering expression. FP32/FP64 IEEE roots and the low-precision reciprocal
and reciprocal-square-root paths keep their existing dispatch. The source of
this behavior is [TileLang's math header](https://github.com/tile-ai/tilelang/blob/62bba8d20ddb232e29050770472cb2649dd3e718/src/tl_templates/cuda/math.h).

The reference evaluator uses exact rational arithmetic and integer square-root
comparisons for explicit rounding. Its checks include FMA cancellation and
intermediate overflow, rounding midpoints, directed underflow/overflow, signed
zeros, infinities, and NaNs. For the approximate low-precision operations it
supplies ideal mathematical reference values. It does not simulate approximate
instruction results or NaN payloads. BF16 buffer evaluation remains unsupported;
the internal scalar oracle can represent BF16 values for arithmetic checks.

Fast scalar calls are `T.__exp(x)`, `__exp10`, `__log`, `__log2`, `__log10`,
`__sin`, `__cos`, `__tan`, and `T.fast_rcp(x)`. They accept positional or named
`x` and retain its floating dtype. `fast_rcp` accepts scalar FP32 only and emits
`rcp.approx.ftz.f32`. The other calls use CUDA's `__nv_fast_*f` library functions
for FP32 and the ordinary double-precision functions for FP64. In particular,
`__tan` uses the fast FP32 tangent intrinsic in this explicit fast family.
CUDA documents their approximation and denormal behavior in the
[libdevice fast-function reference](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_logf.html).

Low-precision fast calls follow the pinned TileLang template wrappers and
CUTLASS revision `b2dd65dc864e09688245b316ac46c4a6cd07e15c`:

| Calls | FP16 | BF16 |
| --- | --- | --- |
| `__log`, `__sin`, `__cos`, `__tan` | FP32 ordinary library call, then conversion | FP32 ordinary library call, then conversion |
| `__exp` | CUDA half exponential instruction sequence and corrections | FP32 ordinary exponential, then conversion |
| `__exp10`, `__log2`, `__log10` | CUDA half instruction sequences and corrections | CUDA BF16 instruction sequences and the exp10 special case |

The CUDA half sequences use FP32 approximate log2/exp2 instructions with FTZ,
typed scaling, and the header's representable-result corrections. BF16 uses
the corresponding approximate instructions without FTZ. Exported generated
modules include every required helper and library declaration. The reference
evaluator supplies ideal mathematical values and preserves the declared type;
it does not simulate approximation errors, flush-to-zero, or NaN payloads.
Native compilation checks cover scalar dispatch for all four floating types.
Device numerical accuracy and performance remain unmeasured. Vector forms and
global compiler flags remain part of the compatibility work.

Ordinary FP16/BF16 `exp`, `exp2`, `exp10`, `log`, `log2`, `log10`, `sin`,
`cos`, `sqrt`, `rsqrt`, and `tanh` also follow the native CUDA wrappers and
module math-header aliases. Without the math header, BF16 `exp` uses its native
FP32 approximate exp2 sequence with an upward-rounded log2(e) coefficient;
FP16 uses the distinct coefficient and correction points from its own header.
FP16 `exp2` retains the FMA adjustment before conversion. Ordinary low-precision
logarithms retain the native approximate log2 instruction, typed scaling, and
applicable correction points. Ordinary FP32/FP64 `exp10` calls CUDA's dedicated
base-ten exponential function.

For FP16 sine and cosine without the math header, the generated helper performs
the CUDA half-range argument reduction, selects the sine/cosine polynomial by
quadrant, evaluates with explicit FP32 FMAs, and applies the final FP16
corrections. BF16 sine/cosine use ordinary FP32 library calls. With the math
header, FP16 sine/cosine and both low-precision logarithm/square-root types use
the widened library wrappers. The header also selects `tanh.approx.f16` for
FP16 tanh and widened `tanh.approx.f32` for BF16 tanh; without it, tanh uses the
ordinary FP32 library wrapper. These dispatch rules follow
[TileLang's common header](https://github.com/tile-ai/tilelang/blob/62bba8d20ddb232e29050770472cb2649dd3e718/src/tl_templates/cuda/common.h)
and [pinned CUTLASS fast math](https://github.com/NVIDIA/cutlass/blob/b2dd65dc864e09688245b316ac46c4a6cd07e15c/include/cutlass/fast_math.h).

Low-precision sigmoid evaluates `1 / (1 + exp(-x))` with a destination-typed
exponential and addition, followed by the native CUDA half/BF16 division path.
Intermediate overflow and rounding are retained, including the exponential's
math-header dispatch. The mathematical reference preserves those intermediate
types while supplying ideal values for approximate operations. The remaining
inverse-function/dtype combinations and global fast-math compiler options need
further upstream lowering work. Hardware parity remains unverified.

Integer expressions support `&`, `|`, `^`, `~`, `<<`, and `>>`, together with
`T.bitwise_and`, `T.bitwise_or`, `T.bitwise_xor`, `T.bitwise_not`, `T.shift_left`,
and `T.shift_right`. Function spellings accept their upstream operand keyword
names and the default `span=None`. A bare integer operand in a bitwise operation
adopts the other operand's integer dtype and must fit its range. Binary operands
are explicitly converted to their common integer type before lowering. Boolean bitwise operations preserve
one-bit semantics; shifts require integer operands excluding Boolean. Right
shift is arithmetic for signed types and logical for unsigned types.

A shift count must be nonnegative and smaller than the promoted operand width.
The compiler rejects invalid statically bounded counts. Data-dependent counts
retain this source-program precondition. Index analysis tracks unsigned
conversion, inversion, and shift width; a potentially wrapping index shift is
rejected. A constant left shift can participate in the affine ownership proof.
Masks can bound input indices, while general bitwise output permutations still
require additional ownership analysis.

`T.reinterpret(dtype, value, span=None)` preserves the scalar's storage bits and
requires equal source and destination widths. The 13 basic dtypes form 43 legal
same-width pairs, including identities and FP16/BF16 pairs. Generated helpers
use LLVM bitcasts for floating representations and reuse integer SSA bits for
signedness changes. The Boolean dtype has 8-bit source metadata and an i1 CuTe
scalar representation; conversions use valid C++ Boolean bytes (0 or 1).
An integer-to-Boolean reinterpretation requires one of those two byte values.
The NumPy evaluator rejects invalid Boolean representations and preserves
negative zero and NaN payloads through typed byte views. It currently rejects
BF16 reinterpretation; generated BF16 helpers have separate native checks.
Vector, sub-byte, pointer, and non-default span forms remain open.

`T.popcount(x)` accepts uint32/uint64 and preserves the input dtype.
`T.clz(x)` accepts int32/uint32/int64/uint64 and returns int32. These are the
pinned CUDA intrinsic dispatch rules. Both accept the TileLang wrapper's
keyword-only `dtype` argument, which is evaluated and discarded. Argument
construction order, including macro effects in that keyword, is preserved.
The generated code uses typed CuTe `arch.popc` and `arch.clz` operations.
For this CUDA target, `clz(0)` returns 32 or 64; negative signed inputs have
zero leading zeros. The generic TIR documentation leaves zero unspecified,
whereas the [CUDA integer intrinsic contract](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html)
defines the word-width result.

Bit counts have bounds from zero through the operand width. They can index
33/65-element lookup tables even when their input words span all 64 bits.
Underlying buffer addresses still undergo the normal index checks. Integer
reinterpretation tracks signedness changes modulo the word width; a conversion
with a single constant offset preserves affine ownership. Counts of runtime
data retain the general restriction on non-affine output ownership.
CPU-only CuTe tests execute generated one-way FP16/FP32/FP64 helpers and
FP16/BF16/FP32/FP64 roundtrips against raw bit patterns. These checks establish
host execution behavior; GPU execution remains deferred.

`T.maximum` and `T.minimum` propagate NaNs. Floating-point operations follow
their explicit CUDA library or instruction path above, including the native
low-precision approximations. Bitwise identity with NumPy is not specified.

The upstream `T.max` and `T.min` spellings prefer a non-NaN operand. The two
longer spellings are Ntilang extensions. Scalar promotion follows the pinned
TIR numeric matching rules. Mixed floating/integer operands use the floating
dtype; mixed floating types use the wider floating dtype. Integer pairs use the
wider type, with unsigned winning at equal width. Boolean converts to the other
numeric operand's type. Bare arithmetic constants default to `int32`/`int64`
or `float32`. Bitwise integer literals use the contextual rule above.

These conversions apply before runtime arithmetic, comparisons, and conditional
expressions. For example, FP16 plus INT64 first converts the integer
to FP16, so rounding can occur before addition. Both generated source and the
reference evaluator apply the same conversion. Runtime integer `/` requires an explicit
division choice or a floating cast, following the source language's ambiguity
check. Logical operations require Boolean operands.

## Integer division and remainder

`//` and `T.floordiv` round the quotient toward negative infinity; `%` and
`T.floormod` return the corresponding remainder, whose sign follows the divisor.
`T.truncdiv` rounds toward zero, and `T.truncmod` returns the corresponding
remainder, whose sign follows the dividend. Operands may be signed or unsigned
integer expressions, including tensor values. The function forms accept `a`,
`b`, and default `span=None`.

The backend materializes typed operands before division so the CuTe runtime
operators are used consistently for constants and variables. Its native integer
`//` already rounds down. Given the native truncating remainder `r`, the floor
remainder is `r + b` when `r != 0` and the operand signs differ, and `r` otherwise.
The truncating quotient adds one to the floor quotient under the same condition.
Unsigned division needs no sign correction. These transformations use integer
arithmetic throughout, including for 64-bit operands.

Integer divisors must be nonzero. Signed minimum divided by `-1` is excluded
because the quotient is unrepresentable. Proven constant violations are rejected;
data-dependent operands retain these preconditions on every executed logical
element. Source programs must guard any padded elements that could violate them.
Index bounds additionally require a divisor interval excluding zero and track
both signs. An exactly divisible variable part can preserve affine ownership.

`T.ceildiv` and its alias `T.cdiv` accept expressions and specialization integers.
They preserve the pinned upstream formula `(lhs + rhs - 1) // rhs`, including
its behavior for negative divisors. With a positive divisor and representable
intermediates, this equals the mathematical ceiling. `T.align_up(x, y)` is
`T.cdiv(x, y) * y`. The index checker rejects numerator or final-result overflow;
data-value arithmetic retains the source dtype's overflow constraints.

## Diagnostics and hints

`T.print(obj=None, msg="", warp_group_id=0, warp_id=0)` accepts a scalar IR
expression, a global/shared/fragment buffer, or a nonempty message with no
object. A Python integer, float, or Boolean object must first be constructed
as an IR scalar, for example `T.int32(7)`. Messages and warp selectors are
construction-time values. Argument effects retain source evaluation order;
the call returns the construction value `None`.

Scalar and message prints run on every active thread. A global-buffer print
iterates through every flattened element on each active thread, following the
[pinned CUDA print implementation](https://github.com/tile-ai/tilelang/blob/62bba8d20ddb232e29050770472cb2649dd3e718/tilelang/cuda/language/print.py).
Shared and fragment buffer prints use thread
`warp_group_id * 128 + warp_id * 32`; an absent selected thread produces no
output. Fragment buffers are first materialized into synchronized shared
storage using their current ownership layout, including MMA partitions.
That temporary storage counts toward the 48 KiB shared-memory limit. Full
fragment prints require uniform execution outside `T.Parallel`. Existing
initialization and cross-element shared-read checks still apply.

Output includes block/thread coordinates, dtype, value, and buffer/index fields
for buffer prints. Boolean values print as `true`/`false`; integer output
preserves full width, and low-precision floats widen for decimal printing.
Buffer labels preserve their source allocation name. Default scalar labels
use Ntilang source text and may differ from TileLang's printed TIR expression.
Message contents are literal, including percent signs and braces; embedded NUL
ends the message as in the CUDA string argument. Device-wide output ordering,
printf-buffer exhaustion, and flushing follow CUDA runtime behavior.

`T.device_assert(condition, msg="", no_stack_info=False)` converts the scalar
condition to Boolean and emits a device assertion. Nonempty messages print on
failure; by default they include source locations through nested macros.
`no_stack_info=True` omits that stack. Ntilang emits these checks for its explicit
CUDA target independently of whether the source-generation host has a GPU.
The serial reference raises `AssertionError` on a failed condition. Diagnostic
reads may inspect output buffers; ordinary numeric read/write alias restrictions
remain enforced. Concurrent diagnostic observations do not establish memory
synchronization or deterministic ordering.

Ordinary Python assertions with construction-time conditions execute during
parsing, including before `T.Kernel`. A false condition raises `AssertionError`.
Runtime Python assertions, `T.Assert` frames and host exception lowering remain
open; use `T.device_assert` for the implemented device-side check.

`T.likely(cond, span=None)` preserves the operand and its dtype. The wrapper's
`dtype` keyword is evaluated and ignored. Index bounds, affine ownership and
lazy conditional predicates look through the hint. This implementation does
not force backend branch weights. Non-default spans remain unsupported.

Reference print events model linear row-major thread ownership and execute in
serial order. They do not model the physical MMA lane mapping or GPU diagnostic
ordering. Native CPU helper tests exercise decimal formatting, message escaping,
Unicode and unsigned 64-bit values. GPU printf and assertion execution remain
unverified. Local-thread buffers, vector/sub-byte values, pointer diagnostics,
and the remaining assertion/assumption forms require further implementation.

## GEMM

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

## Inclusive scans and shared transpose

`T.cumsum(src, dst=None, dim=0, reverse=False, annotations=None)` and
`T.cummax` accept one- or two-dimensional shared/fragment buffers and explicit
unit-stride regions. Negative axes are normalized; omitted destinations update
the source in place. Shapes must match. Shared sources require shared destinations
of the same dtype. Fragment sources stage through shared storage and permit final
copy dtype conversion. Boolean scans and nonempty annotations remain unsupported.
The launch uses 32, 64, 128, 256, 512, or 1024 threads.

The lowering preserves the pinned CUDA `InclusiveScanLine` arithmetic schedule:
32-element segments, shuffle distances 1/2/4/8/16, identity-padded tail lanes,
source-dtype rounding after each combine, and sequential segment carries. Reverse
scans traverse segments backwards and combine with higher-index neighbors.
`cummax` uses non-NaN-preferring maximum, including the identity carry. The CPU
reference executes this schedule; NVIDIA device numerical parity and performance
remain unverified. Two shared ping-pong tiles, padded to 32 along the scan axis,
and a per-line carry tile implement communication with block barriers. Workspace
is reused by compatible calls and counted in the 48 KiB block memory limit.

`T.transpose(src, dst, annotations=None)` swaps the final two axes of shared
buffers or explicit regions, preserving batch axes and singleton dimensions.
The destination shape must match the permutation. Copies apply destination dtype
conversion and guard tensor bounds. Aliased/overlapping temporary regions capture
the source before writes. Nonempty annotations and other memory scopes remain
unsupported. These collectives may appear in uniform branches or serial loops;
placing them inside a `T.Parallel` body is rejected.

`T.grid(*extents)` constructs nested serial loops in argument order. This version
accepts nonnegative static integer extents, captures all extents before binding
induction variables, and requires a distinct variable per dimension. Zero extents
produce empty domains. Existing serial-loop scope, initialization, and early-exit
rules apply. Dynamic grid extents remain unsupported.

`T.clamp(dst, min_val, max_val)` composes `T.min(T.max(dst, min_val), max_val)`.
Operands are evaluated once in argument order and follow existing promotion and
non-NaN preference rules; reversed bounds retain the same composition.
