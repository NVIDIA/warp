<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Target patterns

The catalogue of workload patterns worth porting to Warp, what each is decided
by, and the ones that reliably are not worth it. Use it to audit a codebase for
candidates and to screen a candidate once you have one.

A pattern match tells you what to measure and what usually kills it. It is
never itself evidence that this codebase will benefit.

## Workload classes

The workload class predicts the outcome far better than the source framework
does. Classify before choosing an API.

### Class 1 — low-dimensional spatial neighborhood construction

**Signals:** 2D/3D coordinates already resident on an NVIDIA GPU; the current
path scans all points, materializes dense candidates, or holds a large static
cell-capacity tensor; output is naturally sparse and can be bounded by a
documented capacity; construction repeats so a grid can be reused or refitted;
per-query work is irregular enough that cell-local traversal skips most pairs.

**Anti-signals:** high-dimensional KNN, exact nearest-`K` ordering, or
distances are the real contract; small sets, CPU-origin data, or a one-shot
call; triclinic/multi-image periodicity, dynamic masks, fractional coordinates,
float64 or dense output required; a mature indexed CUDA baseline already
supplies the exact contract.

**Seam:** an opt-in backend object with explicit dimension, cutoff, capacity,
box and output format. Return padded indices **plus actual count and an
overflow flag**; make compaction a separate convenience path because it can
synchronize and create dynamic shapes.

### Class 2 — persistent BVH point/mesh queries

**Signals:** many points query one static mesh, or a mesh deforms with fixed
topology and supports refit; the current path scans every face, transfers
through the CPU, or rebuilds a CPU tree per call; the result can be
reconstructed from a BVH-selected candidate face.

**Anti-signals:** very small warm query batches; one-shot cold execution;
rebuild-per-query; highly overlapping triangle AABBs or many degenerate/small
faces forcing a fallback scan; exact brute-force tie-breaking, rich distance
types, float64, all-hits or containment semantics are contractual; a mature
persistent implementation already wins.

**Seam:** a caller-owned query/cache object. Separate `build`, `refit`,
`rebuild`, `query`, and state which mutations trigger which. Limit the first
API to the proven subset.

### Class 3 — branchy graph and label algorithms

**Signals:** the current tensor formulation runs a fixed number of global
propagation passes, masks divergent work, or launches many full-domain ops; the
algorithm is naturally block-local with union/find, a sparse frontier,
compaction, or early termination; outputs are discrete and no gradient is
needed; batch parallelism can offset per-item contention.

**Anti-signals:** a mature CUDA library already wins the target density/batch
distribution; dense single-item contention dominates; legacy dtype, autograd,
connectivity or label-order semantics are required.

**Seam:** a separate opt-in function with explicit connectivity, compactness,
output dtype and a documented non-differentiable contract. Leave the existing
function untouched.

### Class 4 — fused per-element loops that remove batch×domain intermediates

**Signals:** the tensor path expands a domain across cameras, samples, stencil
directions, graph edges or another batch axis; intermediate storage is far
larger than final state; each output element can loop locally while keeping
state in registers and writing once; update order is fixed and reproducible in
one kernel.

Message passing and graph convolution belong here when the framework path
materializes a per-edge intermediate that a fused gather-transform-accumulate
kernel would keep in registers. The value is the removed `[edges, features]`
tensor, not the traversal itself — traversal alone is what scatter/segment
reductions and maintained GNN libraries already do well.

**Anti-signals:** the batch loop is enormous or highly divergent; the
framework's compiler, Triton or Pallas fuses it correctly with less custom
code; exact sampler rounding/padding, gradients or topology cannot be
reproduced; a one-shot tool cannot amortize cold compile and the memory saving
is not operationally important.

### Class 5 — regular dense-grid stencil and geometry

**Default expectation: stay in the framework.** Regular stencils, dense arrays
and static control flow are core framework strengths. Consider Warp only after
profiling shows fusion or memory planning genuinely fails *and* chunking,
tiling, rematerialization, `scan`/`fori_loop`, Pallas, Triton or `torch.compile`
cannot bound memory while preserving the ecosystem. A narrow export-only kernel
with no solver or autodiff obligation may still be reasonable.

### Class 6 — framework FFI with stateful data structures

**Signals:** a persistent mutable structure gives a measured benefit
unavailable from the pure framework; inputs/outputs can have static shapes with
bounded dynamic content; the operation can be serialized inside one executable
and is not differentiated through its mutable state.

**Anti-signals:** independent threads or executables must share one builder;
automatic sharding, multi-host, TPU/AMD, `vmap` over hidden state or
higher-order transforms are central; small complete workloads already fuse
well.

**This class carries the highest lifecycle risk.** Hidden mutable state,
pointer-keyed graph capture and unsafe concurrent use routinely hold a
candidate at INCONCLUSIVE for production even when the measured wins are
large — solve ownership before quoting throughput.

### Class 7 — existing mature CUDA APIs with richer semantics

Migration is credible only if Warp supplies a genuinely missing capability or a
narrow measurable sub-operation while the mature API stays intact and
authoritative. If exact IDs, types, degeneracy handling or gradients are
user-visible and cannot be reproduced, stop at the first reproducible contract
failure.

### Class 8 — many independent small problems, none filling the GPU

**Signals:** the workload is *N* independent systems — candidates, molecules,
scenes, episodes — each too small to occupy the device on its own; they are
currently orchestrated serially from Python; per-system state could stay
resident across the work instead of being marshalled in and out.

**The signal alone is not enough.** Padding the systems into a batch axis and
running ordinary tensor ops is the default answer, and on uniform systems it
usually wins on simplicity. Warp earns this only when the per-system work is
**ragged or independently terminating** — different sizes, different neighbour
counts, per-system convergence — so a padded formulation either wastes most of
its lanes or needs masking that consumes the gain.

**Anti-signals:** systems are uniform enough to pad cheaply; `vmap` or a batched
library call already expresses it; the per-system work is dense tensor algebra;
the count of systems is small enough that occupancy was never the problem.

**Seam:** a batched entry point taking packed per-system state plus offsets,
returning per-system results and a per-system status. Keep the single-system
path as the oracle.

**Measure carefully:** batching changes the workload *and* the backend at once.
Report the single-system comparison separately from the batched one, or the
ratio conflates two independent effects.

### Class 9 — iterative loops that cross the host boundary every step

**Signals:** a solver, optimizer, relaxation or integrator running hundreds to
thousands of steps; each step launches from Python, synchronizes, or moves
state across the host boundary; the per-step kernel time is small relative to
that overhead.

**The default answer is not Warp.** CUDA graph capture removes per-step launch
and host overhead without changing language, and `torch.compile`'s
reduce-overhead mode does it for you. Price that first — it is far cheaper than
a port.

Warp earns this when the loop body **cannot be captured as a fixed program**:
per-item convergence tests, structures rebuilt mid-loop, topology that changes
between steps, or early termination that a static graph cannot express — and
when fusing the update and the convergence check into one kernel removes the
intermediates along with the round trip.

**Anti-signals:** the step body is a fixed sequence of tensor ops; a captured
graph already recovers most of the overhead; the loop runs few enough times
that setup dominates; gradients are required through the whole trajectory.

**Seam:** a resident state object plus a `step`/`run_until` entry point, with
the host loop retained as the reference. State ownership and what invalidates
resident state are part of the contract.

## What each pattern is usually decided by

Match a candidate to its class, then go straight to the evidence that class
normally turns on. This is where to spend the measurement budget.

| Pattern | Usual source of value | Usually decided by |
|---|---|---|
| Spatial neighbourhood construction | Replacing all-pairs or dense-candidate work with cell-local traversal; bounded sparse output instead of a materialized candidate tensor | Whether output capacity can be bounded and overflow surfaced; whether a mature indexed CUDA library already supplies the exact contract |
| Persistent BVH point/mesh queries | Reusing one acceleration structure across many queries; refit instead of rebuild for deforming geometry | Query count per build, and whether tie-breaking / distance-type semantics are contractual |
| Branchy graph and label algorithms | Replacing fixed-iteration global propagation with block-local union-find and early termination | Contention at high density; whether a mature CUDA library wins the target density and batch mix |
| Fused per-element loops | Eliminating a batch × domain intermediate by looping locally and writing once | Whether the framework compiler already fuses it; whether the memory saving is operationally material |
| Regular dense-grid stencils | Rarely a win — assume the framework owns this | Whether chunking/tiling in the incumbent framework already bounds memory |
| Framework FFI with stateful structures | Persistent mutable state that pure framework code cannot express | Ownership: concurrency, pointer-keyed graph capture, teardown — not throughput |
| Mature CUDA API with rich semantics | Only a genuinely missing capability, never duplication | Whether exact IDs, types, degeneracy handling and gradients can be reproduced |
| Many independent small problems | Filling the device with work that individually underuses it, and keeping per-system state resident | Whether the per-system work is ragged enough to defeat a padded batched tensor formulation |
| Host-bound iterative loops | Removing the host from the loop and fusing update with convergence | Whether CUDA graph capture already recovers the overhead without a port |

Two cross-cutting expectations worth carrying into any candidate:

- **Persistence is usually the product.** The valuable abstraction is normally a
  cached grid, mesh BVH, fixed output buffer or captured graph *with an explicit
  lifetime* — not a stateless replacement function. Build, refit, rebuild and
  query are separate operations with separate costs; refit is often what makes
  a deforming-geometry case viable.
- **A memory reduction can outrank a speedup.** Removing a large intermediate
  changes which problem sizes are reachable at all. Treat a measured reduction
  as a first-class result, and account for retained structures, outputs, graph
  pools and framework allocators before claiming it.

## Reliably not worth porting

Recognize these early and stop; they do not need a prototype to reject:

- dense linear algebra, convolution, FFT, reductions and neural-network layers
  already lowered onto vendor libraries or a framework compiler;
- work whose data starts and ends on the CPU, when the boundary cannot be
  widened;
- small or infrequent calls where cold start and launch overhead dominate;
- one-shot processes that cannot amortize import, init and JIT compilation;
- a mature CUDA implementation that already meets the contract, where a port
  would duplicate maintained code without an algorithmic advantage;
- anything requiring float64 throughout, exact tie ordering, or portability to
  non-NVIDIA accelerators;
- a hot path that profiling shows is too small a share of the objective to move
  it.

## Common false positives

- Replacing tuned dense algebra with a naive per-thread kernel.
- Timing one GPU kernel against a Python loop while ignoring the vectorized or
  library baseline.
- Crediting the candidate with overhead that belongs to the incumbent's
  *wrapper* rather than its kernel. Per-call dispatch, argument marshalling and
  array re-wrapping are flat in problem size, so they dominate small calls and
  can manufacture a several-fold "backend win" that evaporates once the
  incumbent hoists them. Hoist them first, then re-measure, then attribute what
  is left — and measure each component separately rather than assigning the
  whole gap to whichever one you guessed.
- Timing a stage that never ran on the device: identical host and device
  timings with no device allocation mean the "GPU path" converted to host
  arrays or silently downgraded the request.
- Timing launch enqueue instead of completion.
- Excluding compilation from a product where every process compiles.
- Calling a conversion "free" because it aliases storage, while ignoring
  descriptor construction and stream waits.
- Excluding output allocation because the microbenchmark reuses a buffer that
  production cannot reuse.
- Excluding a BVH/hash-grid build or refit that every real input requires.
- Comparing forward-only Warp against a baseline that includes backward.
- Comparing different precision, fast-math, or determinism settings.
- Claiming a universal benefit from one GPU, one shape, or one friendly
  distribution.

## Common false negatives

Rejections go wrong in mirror-image ways, and they are harder to notice because
nothing downstream contradicts them:

- **Screening a candidate against the study's headline metric instead of the
  one its pattern exhausts.** A memory/capacity candidate dismissed on a
  latency share is the standard case (see Gate F).
- **Generalizing a rejection from one device's budget.** A multi-GiB transient
  is immaterial on a 48 GiB card and decisive on the 24 GiB card the users
  actually have. One GPU can hide a NO-GO exactly as easily as it can
  manufacture a GO — state the memory budget any materiality judgement assumes.
- **Measuring only the configuration you happened to run.** Library defaults,
  CLI defaults and documented examples often differ; a stage that is trivial at
  the CLI default can dominate at the library default or at a user-supplied
  size. Check the defaults on every reachable entry point, not just the one you
  invoked.
- **Rejecting on a share of a total that your own environment inflated.** If a
  missing optional accelerator makes some *other* stage artificially slow, every
  share you compute against that total understates the rest. Say so, and say in
  which direction it biases the conclusion.
- **Treating "too small to matter" as settled without the arithmetic.** Gate F
  requires a ceiling you can state, not an impression.
- **Accepting a disappointing measurement without auditing the harness.** The
  skill warns repeatedly against tuning to rescue a *promising* benchmark. The
  mirror failure is invisible: a result that says "no" ends the investigation,
  so nothing downstream ever contradicts it. When a measurement contradicts a
  strong structural prior — an O(N·M) scan losing to an indexed traversal, a
  fused kernel losing to a materialized intermediate — **audit the harness
  before you record the verdict.** Check the workload pairing, the candidate's
  unset defaults, and which branch actually executed. A NO-GO earns the same
  scrutiny as a GO; it is simply cheaper to get wrong.
- **Screening a seam on the aggregate when the product has more than one
  execution regime.** A seam that is unreachable noise behind a 4.5 s import in
  a one-shot command can be 69 % of an iteration in a long-lived loop of the
  same product. Answer per regime.
