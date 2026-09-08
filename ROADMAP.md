# Roadmap

Status: development resumed at the user's request on 2026-09-08.
Work continues from stage 1. The stages below describe remaining work and
dependencies; they do not schedule recurring background jobs.

## Objective

Implement the complete TileLang language surface using an independent,
pure-Python frontend and compiler that emits standalone NVIDIA CuTe DSL.
Compatibility includes signatures, type rules, program semantics, annotations,
layouts, scheduling, and interactions between operations.

The comparison baseline is TileLang
[`62bba8d20ddb232e29050770472cb2649dd3e718`](https://github.com/tile-ai/tilelang/tree/62bba8d20ddb232e29050770472cb2649dd3e718)
and its TVM submodule `907a88c8791ccf33b9874821bc875e7abf624367`.
The [export inventory](docs/tilelang-api.json) records 706 names from the default
CUDA language facade. This count measures inventory size. It does not measure
implemented or validated compatibility.

## Current baseline

Implementation revision: `806a3ab3af7980b6cf49cebe5295caaad56a34d5`.
[GitHub Actions run 34187693666](https://github.com/nya-a-cat/Ntilang/actions/runs/34187693666)
passed with NVIDIA CuTe DSL 4.7.1 and TVM FFI 0.1.11:

- Linux: 1,395 semantic, native compilation, and host FFI checks passed.
- Windows: 863 checks passed; 532 compiler-dependent checks were skipped.
- Both jobs deselected 29 GPU tests. No GPU execution was performed.
- Source and wheel distributions built successfully.

Implemented areas, within the restrictions in
[compatibility.md](docs/compatibility.md):

- Static tensor declarations, basic runtime scalar parameters, kernel launches, independent IR, source generation,
  fake-tensor compilation, and standalone generated modules.
- Shared/register fragments, guarded global accesses, sliced synchronous copies,
  temporary copy snapshots, and shared communication for cross-element reads.
- Multidimensional and contiguous nested parallel loops; serial/unrolled loops
  with runtime start/stop and static steps; while, break, continue, conditionals,
  scalar joins, and typed mutable local scalars.
- Basic integer, Boolean, and floating types; numeric promotion, bitwise
  operations, integer division/remainder, conditional expressions, rounding,
  absolute value, classification, 23 transcendental operations, power, floating
  remainder, `atan2`, and `copysign`.
- Eight basic reduction kinds, broadcast reads, warp MMA GEMM, accumulator
  epilogues, and propagation of compatible fragment ownership layouts.
- Conservative ownership, initialization, alias, index-range, and shared-memory
  checks, with a serial NumPy reference evaluator.

The reference evaluator does not model device scheduling or Tensor Core
rounding. Native compilation establishes compiler acceptance and binary
generation. CPU-only native FFI checks observe scalar conversions through CPU
output storage. They cover Boolean and all basic integer/floating argument types,
including full-width unsigned inputs. Hardware numerical accuracy, concurrency behavior, performance,
and full compilation-chain formal verification remain unverified.

## 1. Complete scalar semantics and the source frontend

- [ ] Implement `T.pow` with its distinct constant-integer and dynamic-exponent
  paths, preserving repeated-multiplication rounding and the zero-exponent rule.
- [ ] Implement floating remainder, `atan2`, `copysign`, `hypot`, `nextafter`,
  `ldexp`, remaining arithmetic intrinsics, and their exact argument/type rules.
- [ ] Implement fused operations, explicit IEEE rounding modes, fast-math
  intrinsics, and target/dtype restrictions without dropping requested modes.
- [ ] Complete scalar constructors, vector/sub-byte dtypes, reinterpretation,
  bit utilities, random operations, assertions, printing, and source spans.
- [ ] Complete function attributes, decorators, macros, eager/TIR parser forms,
  lexical bindings, scalar arguments, and remaining exported constructors.
- [ ] Extend constant evaluation and branch/range analysis while preserving
  overflow, initialization, and mutable-state semantics.

Basic scalar `T.pow`, `fmod`, `atan2`, and `copysign` paths are implemented and
covered by the current CI baseline. Power recognizes immutable integer aliases
and basic integer constant arithmetic, preserves sequential multiplication for
nonnegative integer exponents, and uses floating promotion for other exponents.
Remaining work includes full constant evaluation, integer-index integration,
vector/sub-byte forms, and the other binary math operations listed above.
Basic runtime scalar parameters now pass through mixed tensor/scalar signatures,
typed CuTe arguments, host validation, and reference evaluation. Further argument
work includes symbolic dimensions, defaults/keywords, vector types, and broader
index-range analysis.
Source macro expansion now covers hygienic calls, ordinary and reference
arguments, nested calls, static recursive branches, and scalar/buffer/tuple
returns. Broader Python bindings, object forms, and exits into caller scopes
remain in the frontend work above.
The upstream `pow_of_int` helper returns the base for nonpositive template
exponents; the public `T.pow` adapter handles zero and negative exponents
separately. This source distinction remains relevant to further parser work.

## 2. Generalize tensors, indexing, and layouts

Depends on the scalar and binding contracts established in stage 1.

- [ ] Add symbolic/dynamic dimensions, explicit strides, broader scalar parameter ABI,
  and specialization rules.
- [ ] Add local/global allocation forms and general buffer regions, slices,
  views, reshape, pointer access, and storage reinterpretation.
- [ ] Extend typed index domains beyond the current signed-32-bit checks and
  prove ownership for additional data-dependent and branch-dependent writes.
- [ ] Support parallel nests with intervening statements or dependent domains,
  explicit layouts, vectorization, and the full loop signatures/annotations.
- [ ] Implement fragment layout conversion and general broadcasting, including
  dependencies between in-place reads and writes.
- [ ] Replace current global read/write and alias restrictions as the required
  dependency and memory-effect analysis becomes available.

## 3. Complete tile operations

Depends on general buffer regions, layouts, and memory-effect analysis.

- [ ] Complete copy options, coalescing/vector-width controls, transpose, and im2col.
- [ ] Complete reduction regions, local scopes, batching, packed annotations,
  reducer epochs, scans, and cumulative operations.
- [ ] Complete GEMM signatures and policies, additional operand/accumulator
  types, sparse GEMM, architecture-specific MMA forms, and layout conversion.
- [ ] Verify compositions such as copy-to-reduction, reduction-to-broadcast,
  GEMM-to-epilogue, and overlapping temporary updates.

## 4. Implement CUDA synchronization and scheduling

Depends on explicit memory effects, layouts, and target capability checks.

- [ ] Add warp votes, shuffles, reductions, atomics, and their memory semantics.
- [ ] Add named/cluster barriers, asynchronous copies, pipeline stages and
  order/group/synchronization metadata.
- [ ] Add persistent schedules, swizzles, and warp specialization.
- [ ] Add TMA, WGMMA, tensor memory, descriptors, clusters, PDL, register
  allocation controls, and remaining target-specific intrinsics.
- [ ] Add external source/intrinsic integration and complete launch contracts.

Each asynchronous feature requires an explicit account of participants,
ownership, ordering, visibility, completion, and resource lifetime. Scheduling
annotations need a corresponding lowering that honors their requested behavior.

## 5. Complete integration and compatibility verification

Maintain this work alongside every implementation stage.

- [ ] Track every baseline export against its source definition, full signature,
  supported types, parser/IR semantics, lowering, and unresolved limitations.
- [ ] Port representative upstream programs and negative cases for each family;
  include interactions between already implemented operations.
- [ ] Verify generated modules independently of Ntilang, including ABI,
  specialization, architecture checks, diagnostics, and package installation.
- [ ] Keep README and semantics documentation synchronized with the detailed
  compatibility record; several introductory descriptions still reflect the
  initial subset.
- [ ] Record numerical and concurrency questions requiring hardware evidence.
  GPU execution requires a later change to the user's current no-GPU instruction.
- [ ] Audit the entire pinned language surface before claiming full compatibility.
  Passing existing tests leaves unimplemented signatures and semantics open.

## Workflow when resumed

Use the existing `uv.lock`, pure-Python implementation, and pinned CuTe compiler.
Run semantic and native compilation checks in GitHub Actions. Continue without
Docker or GPU execution under the current instructions.

For each implementation unit: inspect upstream behavior, update frontend/IR and
lowering together, add meaningful semantic and compiler checks, inspect the exact
CI revision, update compatibility documentation, and create a focused commit.
Keep detailed development and experiment notes in the ignored `process.md`.

If five substantially different approaches fail to improve a complex issue,
record the source context, attempted approaches, results, and open question for
further review before continuing that issue.
