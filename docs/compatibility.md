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

The compiler currently handles static tensor kernels, linear register fragments,
shared copies, scalar arithmetic, bounded strided serial loops, unrolled loops,
and warp MMA GEMM. Fragment element assignment and augmented assignment use the
same per-thread ownership as the enclosing parallel tile. CPU semantic tests and
real CuTe compilation tests are kept separately from GPU execution tests.

Conditional statements support branch-defined scalar joins, conditional fragment
updates, and uniform collective branches. Initialization is intersected across
paths. Multiple global stores are accepted for proven disjoint ranges or in
exclusive branches with the same ownership mapping; general path-dependent
write analysis remains open.

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
The 169 pairwise basic-type addition promotions are checked against NVIDIA's
numeric implementation. Vector and sub-byte dtype variants, external constructor
behavior, and the remaining dtype utility APIs still need implementation.

Loop steps and empty iteration domains are preserved. The supported `unroll`
form emits CuTe compile-time iteration. Loop annotations, partial unroll factors,
dynamic bounds, and vectorized loop lowering require further implementation.

## Remaining language families

- Python/TIR syntax: general branch write analysis, while, break, continue, scalar mutation, macros,
  function attributes, assertions, lets, eager definitions, and scalar arguments.
- Tensor declarations: symbolic/dynamic dimensions, strides, local/global
  allocations, scalar variables, general buffer regions and slicing, views, reshape,
  reinterpretation, pointers, and dtype coverage.
- Iteration and scheduling: all loop options, nested parallel layout inference,
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
