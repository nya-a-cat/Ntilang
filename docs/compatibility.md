# TileLang language compatibility

The objective is the complete TileLang language surface with an independent
pure-Python Ntilang frontend and pure CuTe DSL output. Compatibility work covers
supported-program semantics, compiler transformations, and static checks.
GPU execution is deferred; hardware behavior remains unverified. Design
arguments must state their assumptions and the properties they establish.

The comparison baseline is TileLang commit
[`62bba8d20ddb232e29050770472cb2649dd3e718`](https://github.com/tile-ai/tilelang/tree/62bba8d20ddb232e29050770472cb2649dd3e718).
Its default `tilelang.language` facade exports the common language, the TIR script
surface, and CUDA extensions. Exported names, signatures, annotations, and
interactions between operations are all in scope. A familiar function name alone
does not establish compatibility.

The upstream TIR parser is provided by the pinned `TileLang/tvm` submodule at
`907a88c8791ccf33b9874821bc875e7abf624367`; its exports are included in this scope.

The [export inventory](tilelang-api.json) records 706 distinct names from that
default CUDA facade, including dtype variants, parser constructors, common tile
operations, and CUDA-specific APIs. It follows the source `__all__` unions,
removes backend-only parser exports from the common layer, and adds the CUDA
layer. The inventory includes source hashes and overlapping export groups.
This is a name inventory; implementation status requires separate signature,
behavior, interaction, and hardware checks. Ntilang currently implements only
parts of this surface. Its `maximum` and `minimum` spellings are extensions;
the upstream facade exports `max` and `min`.
Both upstream spellings are now implemented with their non-NaN preference.

## Current work

The frontend expands `@T.macro` and `@T.macro()` from available Python source.
Expansion isolates definition-side closures and local names, binds positional,
keyword, default, and tuple/dictionary argument forms, and supports scalar,
buffer, region, and tuple results. Basic `T.Ref` annotations preserve mutable
scalar and buffer-element updates and capture region origins at macro entry.
Expanded statements participate in the existing ownership and initialization
checks. Scalar annotations on ordinary macro arguments preserve the argument's
dtype, following the default eager builder.
Scalar stores expand the target buffer and indices before the value expression.
Tuple assignment captures scalar values before binding targets and preserves
the order of target updates and any macro expansions within those targets.
Element references capture their indices before updates in the called macro.

Macro statements are emitted at the call's construction position. This includes
expansion before a while loop and expansion of both macro arguments to a scalar
`if_then_else` call. Runtime Boolean branches and returns inside runtime control
flow retain upstream diagnostics. Static branches can select recursive expansion
with a depth limit of 128. Macro exits into caller loops, arbitrary object
constructors, and full buffer metadata/parser forms remain open.

The source environment separates Python construction values from runtime IR
values. Basic scalar rebinding creates fresh IR identities and preserves earlier
snapshots. Buffer aliases retain their allocation when a source name is rebound;
`alloc_var` reallocation also preserves the prior scalar used in an initializer.
Loop-variable reuse, ordinary augmented assignments, tuple unpacking, and chained
assignments follow this environment. Whitelisted Python expressions can determine
static branches, allocation shapes, dtype aliases, and operation aliases.
Macro-returned Python constants preserve their construction phase through scalar
operators. Chained comparisons follow the pinned AST mutator, including its
repeated evaluation of middle expressions and runtime Boolean-frame restrictions.
Buffer metadata supports shape, dtype, source strides/scope, and the default
offset/alignment fields; scalar dtype metadata follows expression promotion and
rebinding. Tensor parameters retain explicit contiguous strides and allocations
retain empty source strides. Basic dtype descriptors expose bits/bytes/itemsize,
lanes, and DLPack type code; Boolean uses 8-bit metadata in the pinned TVM version.
Basic dtype/get_tvm_dtype conversions and primitive len/tuple/int/str calls are
implemented. Pure bindings can precede the single final Kernel frame, preserving
the upstream distinction between PrimFunc-level IR constants and Kernel-level
Python int bindings. Host operations before the launch, general Python objects,
container mutation, comprehensions, and the remaining constructor/metadata forms
still require frontend work. See [semantics.md](semantics.md#construction-metadata)
for supported fields and conversion restrictions.

The compiler currently handles static tensor kernels, linear register fragments,
shared copies, scalar arithmetic, bounded strided serial loops, unrolled loops,
and warp MMA GEMM. Fragment element assignment and augmented assignment use the
same per-thread ownership as the enclosing parallel tile. CPU semantic tests and
real CuTe compilation tests are kept separately from GPU execution tests.

Contiguous rectangular nests of `T.Parallel` loops share the flattened logical
domain used by the equivalent multidimensional `T.Parallel` spelling. This
preserves fragment/shared ownership, guarded global accesses, deepest-body local
scalars, and cross-element temporary reads. Statements between parallel levels,
dependent extents, and explicit nested layout annotations need further lowering.

Conditional statements support mutable scalar updates, conditional fragment
updates, and uniform collective branches. Runtime bindings retain their defining
region; fresh values created inside a branch or loop cannot be read after it.
Python constant assignments execute in construction order, including both sides
of a runtime conditional. Earlier implicit branch-result joins were removed to
match the pinned eager builder. Initialization is intersected across paths.
Multiple global stores are accepted for proven disjoint ranges or in
exclusive branches with the same ownership mapping; general path-dependent
write analysis remains open.

Conditional expressions implement eager `Select` and lazy `if_then_else`,
including their differing type rules. Lazy branch checks refine simple affine
integer predicates. Tensor predicates may select bounded index values; index
predicate reads are included in alias/read-write checks. General disjunctive
range reasoning and source span objects remain open.

Copy regions support static positive unit-stride slices, fixed dimensions,
global-to-global transfers, partial temporary regions, and scalar element copies.
Overlapping temporary copies capture the source before destination writes.
Shared parallel stores preserve logical ownership and uniform synchronization.
Explicit vector widths, custom copy layouts, asynchronous instructions, and
remaining copy annotations still require implementation.

MMA coordinate partitions propagate through pointwise operations and whole-tile
fragment copies. Accumulators support elementwise epilogues, initialization from
global/shared/fragment copies, and copies through shared output tiles. Connections
between incompatible register partitions still need explicit layout conversion.
Broadcasts and other cross-element fragment reads use synchronized shared views.
The additional storage participates in the block's shared-memory limit. Direct
shuffle-based communication and in-place cross-element transformations need
further lowering and dependency analysis.

All eight basic reduction kinds lower through synchronized shared-memory trees.
Shared/fragment scope combinations, static axes, kept dimensions, output dtype
conversion, accumulation, and FP16/BF16 NaN controls have source and compiler
coverage. Stable Softmax exercises reductions with broadcast communication.
Batch scheduling, packed reduction annotations, local scopes, regions, and
reducer epoch APIs remain open compatibility work.

Basic Boolean/integer/floating dtypes and scalar cast constructors are implemented.
The 169 pairwise basic-type additions use explicit operand conversions before
NVIDIA's numeric operators. Vector and sub-byte dtype variants, external constructor
behavior, and the remaining dtype utility APIs still need implementation.

Kernel signatures accept these basic scalar dtypes alongside static tensors,
preserving positional argument order. Scalars remain runtime values in generated
CuTe signatures and support arithmetic, predicates, mutable initializers,
captured serial bounds, and guarded integer gathers. The host wrapper checks
integer ranges and accepts real floating values, including infinities and NaNs;
Boolean parameters require Python `bool`. Full-width `uint64` inputs use a
bit-preserving signed TVM FFI payload. Tensor dimensions and launch grids remain
static. Scalar-index checks retain the current signed-32-bit intermediate limit;
keyword/default arguments, symbolic shapes, and vector scalar ABIs remain open.
CPU-only native FFI tests write comparison results into CPU output tensors for
each basic scalar dtype. CuTe 4.7.1's callable wrapper requires the pinned TVM FFI
0.1.11 interface; earlier source-compilation checks with 0.1.4 had not exercised
live callable construction.

Bitwise operators and their six upstream function spellings are implemented for
basic integer types, with Boolean support for non-shift operations. Lowering
uses explicit common-type conversions. Static shift counts, unsigned index
inversion, and narrow index overflow have dedicated checks. General bitwise
output permutations and non-default source span objects remain open.

Numeric promotion follows the pinned TIR matching source across arithmetic,
comparisons, and conditional expressions. Floating/integer pairs preserve the floating
operand's dtype. Bitwise integer literals adopt the other operand's integer
type. Integer `/` and non-Boolean logical operands are rejected according to
the source contract. General constant folding, non-default constructor forms,
and remaining scalar signatures still need implementation.

Scalar `abs`, `floor`, `ceil`, `trunc`, `round`, `nearbyint`, `isnan`, `isinf`,
and `isfinite` preserve the pinned type rules, including integer identities,
signed-minimum absolute values, and Boolean classification results. Round's
ties-to-even and ties-away-from-zero modes lower to separate math operations.
Half-precision rounding widens before the operation and converts back. Tests
cover halfway neighbors, signed zero, NaN, infinity, and integer extrema.
The pinned classification operators reject bfloat16 inputs; other math families
and explicit source span objects remain open.

The floating scalar family includes exponential, logarithmic, trigonometric,
inverse trigonometric, hyperbolic, inverse hyperbolic, square-root, reciprocal
square-root, error-function, and sigmoid operations. Integer `exp` inputs convert
to float32 before evaluation. Half/bfloat inputs use widened math with a typed
result; sigmoid additionally preserves the source formula's intermediate result
types. `exp10` currently uses CuTe's power operation with base ten. Mathematical
reference comparisons and native compilation cover these paths; hardware ULP
accuracy and equivalence to upstream CUDA library implementations remain unverified.

Binary scalar math includes `pow`, `fmod`, `atan2`, `copysign`, `hypot`,
`nextafter`, and `ldexp`. Constant
nonnegative integer powers retain the base dtype and sequential multiplication;
zero powers produce a typed one. Immutable integer aliases and basic constant
integer arithmetic participate in selecting that path. Dynamic and negative
exponents use the promoted floating power contract. The remaining binary
intrinsics use the first argument's result dtype. `hypot`/`nextafter`/`ldexp`
currently require float32 or float64 results and call CUDA libdevice through
typed CuTe extern declarations. `ldexp` converts its exponent directly to int32;
the other binary arguments convert to the result type. Special-value reference
checks and native/standalone compilation cover this path.
General constant folding, power expressions in integer index analysis, other
binary functions, and device-level numerical parity remain open.

Scalar `ieee_add`, `ieee_sub`, `ieee_mul`, `ieee_fmaf`, `ieee_frcp`,
`ieee_fsqrt`, and `ieee_fdiv` preserve four FP32/FP64 rounding modes and
first-operand type conversion. `ieee_frsqrt` supplies its one-argument FP32
rounded path. FP16/BF16 support `rn`, retaining CUDA's native arithmetic and
approximate root/reciprocal/division instruction sequences. `fma` and `fmul`
require matching operand types and preserve fused or explicit multiply
boundaries. An exact arithmetic oracle checks single rounding and special
values. For low-precision approximate operations the oracle gives ideal values;
hardware approximation behavior and numerical parity remain unverified.
See [scalar arithmetic](semantics.md#scalar-arithmetic) for the target rules,
FTZ behavior, and reference limits. Vector forms and fast-math APIs remain open.

Local scalar annotations preserve the expression dtype according to the default
eager frontend, including branch-local values and captured specialization dtype names.
Legacy TIR annotation semantics and buffer annotations remain open.

`alloc_var` supports basic scalar dtypes in the local.var scope, initialization,
assignment, augmented assignment, serial/unrolled loop-carried values, and
conditional updates. Cross-parallel persistent scalar state, mutable index
ownership/range refinement, alternative allocation scopes, and buffer-style scalar access
remain open.

Boolean while loops support mutable loop-carried values, nested loops,
per-element iteration counts, and uniform collective bodies. Conditions reload
their inputs each iteration; zero-iteration initialization is preserved.
Break/continue and their exported statement spellings use independent nested
loop state, including condition evaluation after early exits and the upstream
restriction on explicitly expanded break targets. General termination proofs,
parallel-loop exits, and more precise initialization joins remain open.

Floor and truncating integer division/remainder support signed and unsigned
data operands. The index analysis handles either divisor sign and bounded
nonzero variable divisors. Expression `ceildiv`, `cdiv`, and `align_up` preserve
the pinned upstream formula and participate in static shape specialization.
Zero-divisor and signed-overflow preconditions remain explicit. Additional
remainder signatures and broader path-sensitive arithmetic analysis remain open.

Loop steps and empty iteration domains are preserved. Serial/unroll loops accept
keyword start/stop/step arguments. Unroll's explicit expansion, full-unroll hint,
factor hint, and corresponding annotation precedence are implemented separately.
Integer runtime start/stop bounds use captured values, static steps, 64-bit
trip-count arithmetic, and checked induction ranges. Data-dependent gathers
use guarded loads with dtype/min/max/bit-mask interval reasoning. More general
typed index domains, runtime steps, other loop annotations, and vectorized
loop lowering require further implementation.

## Remaining language families

- Python/TIR syntax: general branch write analysis, broader early exits and scalar mutation, remaining macro forms,
  function attributes, assertions, lets, eager definitions, and full scalar argument forms.
- Tensor declarations: symbolic/dynamic dimensions, strides, local/global
  allocations, scalar variables, general buffer regions and slicing, views, reshape,
  reinterpretation, pointers, and dtype coverage.
- Iteration and scheduling: all loop options, general nested parallel layout inference,
  vectorization, persistent scheduling, pipeline stages/order/group/sync metadata,
  swizzles, and warp-specialization schedules.
- Tile operations: complete copy signatures, transpose, im2col, all reduction
  types and reducer epochs, scan/cumulative operations, fragment broadcasting,
  layout conversion, GEMM policies, sparse GEMM, and MMA layout propagation.
- Scalar intrinsics: the full arithmetic, math, comparison, logical, bitwise,
  conversion, random, assertion, and printing surfaces.
- CUDA operations: warp votes/shuffles/reductions, atomics and their memory
  semantics, named/cluster barriers, asynchronous copies, TMA, WGMMA, tensor
  memory, descriptors, cluster kernels, PDL, register allocation, external
  source integration, and target-specific intrinsics.
- Compiler/runtime integration: full launch signatures, architecture checks,
  ABI and aliasing behavior, supported source decorators and parser forms, and
  importable standalone output for every implemented lowering.

These families remain open until their individual signatures and semantics have
been compared against the pinned upstream sources and relevant test programs.
The current initial-subset restrictions must be replaced as the corresponding
analyses and lowerings are implemented. They do not redefine the full objective.

## Evidence and deferred execution checks

Each operation needs source-level positive and negative examples, explicit IR
semantics, and a lowering argument that accounts for ownership, initialization,
and synchronization. Layout and scheduling operations additionally need evidence
that their requested mapping or scheduling behavior is honored. Existing compiler
checks remain documented with their toolchain versions.

GPU numerical and concurrency checks are retained as optional follow-up work.
No GPU execution is scheduled in the current development scope. Source inspection
and static checks do not establish measured hardware behavior or an end-to-end
machine-checked proof; those evidence boundaries remain explicit.
