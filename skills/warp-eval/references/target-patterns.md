# Target patterns

Workload patterns potentially relevant to Warp, the evidence each requires, and
the ones that fire an early gate. **A pattern match tells you what to measure
and what usually disqualifies it — it is never itself evidence that this
codebase will benefit.** Workload class is more informative than source
framework, so classify before choosing an API.

## Workload classes

### Class 1 — low-dimensional spatial neighborhood construction

**Signals:** 2D/3D coordinates already resident on an NVIDIA GPU; the current
path scans all points, materializes dense candidates, or holds a large static
cell-capacity tensor; output is naturally sparse and boundable by a documented
capacity; construction repeats so a grid can be reused or refitted; per-query
work is irregular enough that cell-local traversal skips most pairs.

**Anti-signals:** high-dimensional KNN, exact nearest-`K` ordering, or distances
are the real contract; small sets, CPU-origin data, or a one-shot call;
triclinic/multi-image periodicity, dynamic masks, fractional coordinates,
float64 or dense output required; a mature indexed CUDA baseline already
supplies the exact contract.

**Seam:** an opt-in backend object with explicit dimension, cutoff, capacity,
box and output format. Return padded indices **plus actual count and an overflow
flag**; make compaction a separate convenience path, since it can synchronize
and create dynamic shapes.

**Decided by** whether output capacity can be bounded and overflow surfaced, and
whether a mature indexed CUDA library already supplies the exact contract.

### Class 2 — persistent BVH point/mesh queries

**Signals:** many points query one static mesh, or a mesh deforms with fixed
topology and supports refit; the current path scans every face, transfers
through the CPU, or rebuilds a CPU tree per call; the result can be
reconstructed from a BVH-selected candidate face.

**Anti-signals:** very small warm query batches; one-shot cold execution;
rebuild-per-query; highly overlapping triangle AABBs or many degenerate faces
forcing a fallback scan; exact brute-force tie-breaking, rich distance types,
float64, all-hits or containment semantics are contractual; a mature persistent
implementation is already faster.

**Seam:** a caller-owned query/cache object with separate `build`, `refit`,
`rebuild` and `query`, stating which mutations trigger which. Limit the first
API to the proven subset.

**Decided by** query count per build, and whether tie-breaking and
distance-type semantics are contractual.

### Class 3 — branchy graph and label algorithms

**Signals:** the current tensor formulation runs a fixed number of global
propagation passes, masks divergent work, or launches many full-domain ops; the
algorithm is naturally block-local with union/find, a sparse frontier,
compaction or early termination; outputs are discrete and no gradient is needed;
batch parallelism can offset per-item contention.

**Anti-signals:** a mature CUDA library is already faster for the target
density/batch distribution; dense single-item contention dominates; legacy
dtype, autograd, connectivity or label-order semantics are required.

**Seam:** a separate opt-in function with explicit connectivity, compactness,
output dtype and a documented non-differentiable contract. Leave the existing
function untouched.

**Decided by** contention at high density, and whether a mature CUDA library is
faster for the target density and batch mix.

### Class 4 — fused per-element loops that remove batch×domain intermediates

**Signals:** the tensor path expands a domain across cameras, samples, stencil
directions, graph edges or another batch axis; intermediate storage far exceeds
final state; each output element can loop locally keeping state in registers and
writing once; update order is fixed and reproducible in one kernel.

Message passing and graph convolution belong here when the framework path
materializes a per-edge intermediate that a fused gather-transform-accumulate
kernel would keep in registers. **The value is the removed `[edges, features]`
tensor, not the traversal** — traversal alone is what scatter/segment reductions
and maintained GNN libraries already do well.

**Anti-signals:** the batch loop is enormous or highly divergent; the
framework's compiler, Triton or Pallas fuses it correctly with less custom code;
exact sampler rounding/padding, gradients or topology cannot be reproduced; a
one-shot tool cannot amortize cold compile and the memory saving is not
operationally important.

**Decided by** whether the framework compiler already fuses it, and whether the
memory saving is operationally material.

### Class 5 — regular dense-grid stencil and geometry

Regular stencils, dense arrays and static control flow are core framework
strengths, so the framework implementation is the required baseline. Evaluate
Warp only after profiling shows fusion or memory planning genuinely fails *and*
chunking, tiling, rematerialization, `scan`/`fori_loop`, Pallas, Triton or
`torch.compile` cannot bound memory while preserving the ecosystem.

**Decided by** whether chunking or tiling in the incumbent framework already
bounds memory.

### Class 5b — compute-bound regular work with high data reuse

**Signals:** many independent evaluations of the same arithmetic over
overlapping data — a patch/window/neighborhood per output, a per-block
reduction, scan, sort or small matmul — with no acceleration structure, no
irregular traversal, and a working set that fits a tile. Often transcendental-
or FMA-heavy and already on a CPU JIT.

**Anti-signals:** a plain elementwise or dense linear-algebra call a vendor
library already owns — that is gate C, not this class.

**Decided by the achievable read amplification, not the flop count.** The naive
formulation — one thread per output, each re-reading its own neighborhood —
reads every input `(window)` times and is memory bound *by construction* on any
hardware. Assess this class only once a cooperative formulation has been
considered; see *Formulations to consider* below. Expect a hand-written CUDA or
CuPy kernel to be competitive here, and measure one.

**Do not file it under Class 4.** There may be no intermediate to remove and no
memory saving at all; the value here is arithmetic throughput and load reuse, so
Class 4's metric sends you to the wrong evidence.

### Class 6 — framework FFI with stateful data structures

**Signals:** a persistent mutable structure gives a measured benefit unavailable
from pure framework code; inputs/outputs can have static shapes with bounded
dynamic content; the operation can be serialized inside one executable and is
not differentiated through its mutable state.

**Anti-signals:** independent threads or executables must share one builder;
automatic sharding, multi-host, TPU/AMD, `vmap` over hidden state or
higher-order transforms are central; small complete workloads already fuse well.

**Decided by ownership, not throughput:** concurrency, pointer-keyed graph
capture, teardown. This class carries the highest lifecycle risk — hidden
mutable state and unsafe concurrent use routinely hold a candidate at `unknown`
for production even when measured throughput differences are large. Report
ownership evidence before quoting throughput.

### Class 7 — existing mature CUDA APIs with richer semantics

Continue the evaluation only if Warp supplies a genuinely missing capability, or
a narrow measurable sub-operation while the mature API stays intact and
authoritative. If exact IDs, types, degeneracy handling or gradients are
user-visible and cannot be reproduced, abort at the first reproducible contract
failure.

**Decided by** whether exact IDs, types, degeneracy handling and gradients can
be reproduced.

### Class 8 — many independent small problems, none filling the GPU

**Signals:** *N* independent systems — candidates, molecules, scenes, episodes —
each too small to occupy the device alone, currently orchestrated serially from
Python, with per-system state that could stay resident.

**The signal alone is not enough.** Padding into a batch axis and running
ordinary tensor ops is the required baseline for uniform systems. Warp is
relevant only when the per-system work is **ragged or independently
terminating** — different sizes, different neighbor counts, per-system
convergence — so a padded formulation wastes most of its lanes or needs masking
that consumes the gain.

**Anti-signals:** systems uniform enough to pad cheaply; `vmap` or a batched
library call already expresses it; per-system work is dense tensor algebra; the
system count is small enough that occupancy was never the problem.

**Seam:** a batched entry point taking packed per-system state plus offsets,
returning per-system results and status. Keep the single-system path as the
oracle. Batching changes the workload *and* the backend at once, so report the
single-system comparison separately from the batched one.

**Decided by** whether per-system work is ragged enough to defeat a padded
batched formulation.

### Class 9 — iterative loops that cross the host boundary every step

**Signals:** a solver, optimizer, relaxation or integrator running hundreds to
thousands of steps, each launching from Python, synchronizing, or moving state
across the host boundary, with per-step kernel time small relative to that
overhead.

CUDA graph capture removes per-step launch and host overhead without changing
language, and `torch.compile`'s reduce-overhead mode does it automatically, so
measure that baseline first. Warp is relevant when the loop body **cannot be
captured as a fixed program** — per-item convergence tests, structures rebuilt
mid-loop, topology changing between steps, early termination a static graph
cannot express — and when fusing the update and the convergence check into one
kernel removes the intermediates along with the round trip.

**Anti-signals:** the step body is a fixed sequence of tensor ops; a captured
graph already recovers most of the overhead; the loop runs few enough times that
setup dominates; gradients are required through the whole trajectory.

**Seam:** a resident state object plus a `step`/`run_until` entry point, with
the host loop retained as the reference. State ownership and what invalidates
resident state are part of the contract.

**Decided by** whether CUDA graph capture already recovers the overhead without
a port.

## Two expectations that cut across every class

- **Persistence is usually the product.** The valuable abstraction is normally a
  cached grid, mesh BVH, fixed output buffer or captured graph *with an explicit
  lifetime*, not a stateless replacement function. Build, refit, rebuild and
  query are separate operations with separate costs, and refit is often what
  makes a deforming-geometry case viable.
- **Report memory independently from time.** Removing a large intermediate
  changes which problem sizes are reachable at all. Account for retained
  structures, outputs, graph pools and framework allocators.

## Patterns that fire an early gate

Recognize these early and abort; they need no prototype. Apply the exact
boundaries from the stage-1 gate reference named by `SKILL.md`.

- dense linear algebra, convolution, FFT, reductions and neural-network layers
  already lowered onto vendor libraries or a framework compiler (gate C);
- data that starts and ends on the CPU where the boundary cannot be widened;
  small or infrequent calls where cold start and launch overhead dominate;
  one-shot processes that cannot amortize import, init and JIT (gate B);
- a mature CUDA implementation that already meets the contract, where a port
  duplicates maintained code without an algorithmic advantage (gate D);
- a hot path profiling shows is too small a share of the objective (gate F);
- anything **requiring** float64 throughout, exact tie ordering, or portability
  to non-NVIDIA accelerators — requiring in the sense that the product promises
  it. A second target that merely exists is a cost to price in; a target the
  repository never commits to either way is a question to ask (gate A).

## Common false positives

- Replacing tuned dense algebra with a naive per-thread kernel.
- Timing one GPU kernel against a Python loop while ignoring the vectorized or
  library baseline.
- **Crediting the candidate with overhead that belongs to the incumbent's
  *wrapper* rather than its kernel.** Per-call dispatch, argument marshalling
  and array re-wrapping are flat in problem size, so they dominate small calls
  and can manufacture an apparent speedup that evaporates once the incumbent
  hoists them. Hoist first, re-measure, then attribute what is left component by
  component.
- Timing a stage that never ran on the device: identical host and device timings
  with no device allocation mean the "GPU path" converted to host arrays or
  silently downgraded the request.
- Timing launch enqueue instead of completion.
- Excluding compilation from a product where every process compiles.
- Calling a conversion "free" because it aliases storage, while ignoring
  descriptor construction and stream waits.
- Excluding output allocation because the microbenchmark reuses a buffer
  production cannot reuse.
- Excluding a BVH/hash-grid build or refit that every real input requires.
- Comparing forward-only Warp against a baseline that includes backward.
- Comparing different precision, fast-math or determinism settings.
- Claiming a universal benefit from one GPU, one shape, or one friendly
  distribution.

## Common false negatives

Rejections are harder to notice than false positives, because nothing downstream
contradicts them:

- **Screening a candidate against the study's headline metric instead of the one
  its pattern exhausts.** A memory/capacity candidate screened on a latency
  share is the standard case (gate F).
- **Generalizing a screen from one device's budget.** A transient immaterial on
  a large card can be decisive on the card the users actually have. State the
  memory budget the materiality calculation used.
- **Measuring only the configuration you happened to run.** Library defaults,
  CLI defaults and documented examples often differ, and a stage trivial at one
  default can dominate at another. Check the defaults on every reachable entry
  point.
- **Rejecting on a share of a total your own environment inflated.** If a
  missing optional accelerator makes some *other* stage artificially slow, every
  share computed against that total understates the rest. Say which direction it
  biases the measurement.
- **Treating "too small to matter" as settled without the arithmetic.** Gate F
  requires a ceiling you can state.
- **Accepting a disappointing measurement without auditing the harness.** When a
  measurement contradicts a strong structural prior — a linear scan beating an
  indexed traversal, a fused kernel losing to a materialized intermediate —
  audit the workload pairing, the candidate's unset defaults, and which branch
  actually executed before recording the result. A disappointing result requires
  the same scrutiny as a favorable one.
- **Screening a seam on the aggregate when the product has more than one
  execution regime.** A seam that is unreachable noise behind a slow import in a
  one-shot command can dominate an iteration of a long-lived loop in the same
  product. Report per regime.
- **Generalizing from the first formulation measured.** See below.

## Formulations to consider

A one-thread-per-output kernel that re-reads its whole neighborhood inherits
that read amplification on any hardware, so its result describes that prototype
rather than Warp generally. Before interpreting a compute-bound or reuse-heavy
seam, state which formulation you measured and which you considered:

- **tile / cooperative primitives** (`wp.tile_load`, `wp.tile_matmul`,
  `wp.tile_reduce`, `wp.tile_scan_*`, `wp.tile_sort`, and the tiled BVH/mesh
  queries) where a block can load a tile plus its halo once and reuse it, or
  where the seam is a per-block reduction, scan, sort or small matmul;
- **fusing adjacent kernels** so an intermediate never reaches memory;
- **one launch over all items** instead of a Python loop over per-item launches;
- **CUDA graph capture** where the seam is launch-bound rather than
  compute-bound;
- **caller-owned persistent structures** so a BVH or hash grid is built once and
  refit rather than rebuilt per call.

These are examples, not a checklist: the right formulation is a property of the
seam. Name the mechanism and check it against the release the project pins.
