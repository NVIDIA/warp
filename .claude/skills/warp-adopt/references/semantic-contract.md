<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Semantic contract and correctness hazards

Write the observable contract before building anything; the hazard catalogue
below is what adversarial testing must cover once you do.

## Write the contract before the prototype

Tolerances chosen **after** seeing Warp's output are not tolerances; they are
excuses. Write all of this down first, and get it reviewed if the API is
public.

### Observable behavior

- Values, dtypes, shapes, devices, error conditions, and what may be mutated.
- Ordering, tie-breaking, capacities, overflow behavior, partitions, topology,
  and degeneracy handling.
- Which outputs are **semantic** and which are **implementation artifacts**.
  Arbitrary IDs may be free to vary; a partition or a selected face may not.
- For an API that already ships, its docstrings and tests **are** the contract.
  Read them as binding rather than inferring a looser one that the prototype
  happens to satisfy.
- Numerical, geometric and task-level tolerances — each with a number.
- Gradients: required only where they are **already** required or the user
  explicitly asked. If the legacy path is non-differentiable export code, say
  so explicitly rather than silently preserving a float dtype that implies
  autograd support.

### Lifecycle

- Stream: which stream does work launch on, and who synchronizes?
- Ownership: who owns inputs, outputs, caches and acceleration structures?
- Aliasing: does any returned array view internal storage?
- Cache invalidation: what mutation triggers refit versus rebuild?
- Concurrency: how many builders/caches may exist per device, per stream, per
  thread? What happens on concurrent misuse — isolation or a clear error?
- Graph capture: what must be static, and what must stay alive through replay?
- Teardown and fallback: what happens when the GPU is absent, compilation
  fails, or an unsupported dtype/device arrives?

**Correctness and required gradients veto performance.** Never weaken a
contract after a mismatch to save a promising benchmark. If the mismatch is
acceptable, that is a *product decision* someone must make explicitly — record
it as such.

## Hazard catalogue

Each of these has, on its own, decided a migration.

### Cutoff ties and floating-point classification

Specify `<` versus `<=` and the arithmetic precision exactly. Test a point
exactly on the cutoff and one representable step inside and outside.

Expect a handful of disagreements per millions of pairs at large scale, all of
them within floating-point noise of the cutoff — and expect two implementations
of the *same* incumbent framework to disagree with each other too. Decide up
front whether production needs bit-identical sets, a geometric tolerance, or
only invariant downstream quantities.

### Truncation and overflow

A capacity-sized count cannot distinguish "exactly full" from "truncated".
Return the unclamped count or an explicit overflow bit, so that exactly-full
(`overflow=false`) is distinguishable from truncated (`overflow=true`).

Traversal order is usually unspecified. Above capacity, test that retained
entries are valid, unique members of the true set — do not promise nearest or
reference ordering unless you implemented it. Size capacity from clustered and
adversarial distributions plus telemetry, not from one average-density sample.

### Face and distance-type ties

Equal distances can select different valid faces. Whether that is acceptable
depends entirely on the API: for a scalar distance it is usually harmless; when
the selected face and a multi-way distance type feed the backward pass and are
user-visible, rare mismatches are fatal. Test exact duplicate faces,
neighbouring coplanar faces, shared edges/vertices, degenerate triangles, and
re-test on every version upgrade — tie order is rarely a documented guarantee.

**Measure how often the ambiguity is even reachable in the production data
before you judge a mismatch rate.** The same operation can look almost-clean or
hopeless depending on the input topology, and both readings can be correct
measurements of different things. On a closed mesh, the closest point lands on a
shared edge or vertex far more often than in a face interior — one real mesh put
only 5,959 of 100,000 queries strictly inside a face, so ~94 % of face indices
were *legitimately* ambiguous and two correct implementations disagreed on most
of them. The same comparison on synthetic triangle sets produced a mismatch rate
four orders of magnitude lower. Report the reachability statistic (how many
queries land in the unambiguous case) next to the mismatch count, or the
mismatch count means nothing on its own.

Note also that a raw disagreement rate measures the *implementation you built*,
not the seam's potential: selecting whatever face the acceleration structure
returns will disagree constantly, while a kernel that re-implements the
incumbent's branch order can drive the same comparison to near zero. Say which
you measured before concluding the contract is unreproducible.

### Gradients at edges, vertices, seams and medial axes

The closest face and its normal are not unique at nonsmooth locations. Expect
large numbers of raw selected-normal differences even when distances, signs and
smooth-region gradients agree closely: an all-point maximum gradient difference
is usually dominated entirely by the nondifferentiable region and says nothing
about smooth-region accuracy.

Report smooth-region gradients separately from all-point maxima. Use finite
differences **away** from topology changes and nonsmooth sets. Test repeated
backward and every gradient operand. A shared existing backward is safe only if
the IDs and types you feed it satisfy its original assumptions.

### Topology drift

Tiny field differences near zero or near sampling boundaries can change
marching-cubes face indices. A result can preserve exact vertex, face and
component counts and exact bounds while still differing in a couple of voxel
observations, leaving face-index arrays that are not byte-identical.

Validate the **downstream artifact**: component counts, largest component,
surface area, bounds, watertightness if relevant, and bidirectional geometric
distance. Define what "practically equivalent" means before rollout.

### Float32 scale behavior

Warp mesh queries operate in float32/int32 even behind a public float64 API.
Relative error stays tiny while absolute error grows with coordinate
magnitude — at scale 1e9, float32 spacing alone puts absolute point error in
the tens, at a relative error still around 1e-7. Sweep realistic **and** extreme scales, state the
supported coordinate envelope, and never silently downcast a float64 contract.

### Nondeterministic atomics and reductions

Atomic append or union order can change edge/label order, which changes
floating-point reduction rounding downstream. Compare semantic sets and
partitions, and deterministic public labels where the contract requires them.
Over a trajectory, expect per-component differences at the level of float32
reduction noise and small energy-drift differences even while positions stay
bitwise equal. Define an acceptable physical-invariant tolerance rather than
demanding bit equality.

**Measure the incumbent's own run-to-run spread before you write the tolerance.**
Call the incumbent repeatedly on identical inputs and record the spread of every
output. That number is the **floor** for any equality contract: a tolerance
tighter than it describes a test the incumbent itself fails, so a prototype
scored against it is being graded on the baseline's nondeterminism.

This is easy to get backwards, because the pre-registration rule says to fix
tolerances before seeing prototype output — and "bitwise equal" looks like the
safest possible choice to pre-register. It is not safe if the incumbent
accumulates gradients with atomics: one such backward kernel was deterministic
in one output and reproducible only to ~1.5e-05 in another, so a pre-registered
bitwise contract on the second was unsatisfiable by anything, including itself.
Establishing the floor is a *measurement of the baseline*, not an observation of
the prototype, so it belongs before the contract and does not compromise it.

When the two implementations select identical indices, any residual difference
in an atomically-accumulated output is ordering noise by construction — assert
the index equality, then bound the value difference by the measured floor.

### Output aliasing

A builder that returns a view into its internal storage will overwrite previous
outputs on the next call. Either name the alias as a borrowed view or return
fresh framework-owned arrays. Test two live outputs across a subsequent call
and a garbage collection.

### Mutable external state

A functional framework model does not make an external hash grid immutable. A
side-effect token orders calls **within one executable**; it does not serialize
independent host threads or executables. Scope builders per stream/executable,
document non-shareability, and either test or reject concurrent use.

### Stream and lifetime ordering

Zero-copy pointers require both storage lifetime **and** CUDA ordering. Record
a readiness event after structure construction when another stream may query.
Keep input storage, Warp objects, cache wrappers, captured graphs and output
owners alive through final stream completion. Test non-default streams,
producer/consumer events, deletion before completion, and repeated
create/destroy.

### Pointer-keyed graph caches

Graph capture can be keyed by call descriptor **and buffer pointers**. Capture
counts typically stabilize after a handful of distinct pointer sets, but
allocator churn can exceed the graph-cache limit and cause repeated recapture
and eviction. Measure capture count and
latency under realistic churn; stable warm samples alone are insufficient. A
captured graph may retain raw pointers without retaining the Python object that
owns the underlying structure — releasing the owner before replay leaves a
stale handle.

**Probe retention explicitly; it is not visible any other way.** Take a weak
reference to the acceleration structure and to its owning cache, capture the
graph, drop every strong reference the *caller* would plausibly drop, force a
collection, and assert the weak references are still alive. A real audit found
that neither the captured graph nor the returned output tensors retained the
Warp mesh, so a replay after the caller released its cache would run against a
freed handle — while every eager test passed and every warm replay looked
correct. Do this even when the seam does not itself capture graphs: a
persistent-index API invites a caller to capture around it, so the retention
contract has to be stated either way. If a convenience path constructs a cache
internally and does not return it, say plainly that the path is not
capture-safe.

### Bounds, pruning and early exit

Any argument that lets an implementation **skip work** — an admissible lower
bound, a monotonicity claim, a conservative filter, a hierarchical cull, an
early-exit condition — is a correctness proof obligation, not an optimization
detail. If the bound is not actually admissible, the result is a *plausible,
fast, wrong* answer: it will look like the best-performing variant you have.

Derivations of this kind are easy to get subtly wrong and impossible to catch by
inspection or by timing. Before benchmarking such a variant, brute-force it
against the unpruned implementation on randomized inputs at scale, and confirm
the pruned path never selects a worse result under the contract's own metric.
Prefer to state the bound explicitly in the report so a reader can check the
algebra. A pruned variant that has not been validated this way must not be
quoted as a performance result.

### Optional dependencies and version drift

Pin or qualify Warp and every compiled dependency across the supported matrix.
Lazy imports must preserve the default path when Warp is absent. Re-run tie,
graph, stream and build tests on every Warp/framework/CUDA upgrade —
undocumented traversal and tie choices can change between versions.

## Validate the oracle, not just the prototype

Every check below compares the prototype against the incumbent. That is only
evidence if the incumbent you are comparing against is the real one, running
correctly, in the configuration you think.

- **Assert an independent invariant on the incumbent's own output** — a quantity
  you can derive without either implementation (the number of in-range rows, a
  conservation law, a known total). Print it next to the comparison rather than
  reporting only the diff count.
- **Prefer a third, independent reference implementation.** Two-against-one
  localises the defect; agreement between two implementations tells you nothing
  when both are built from the same wrong assumption. When the third
  implementation disagrees with both accelerated ones, suspect the reference
  first — unfused host arithmetic and fused device arithmetic legitimately
  disagree near boundaries.
- **Re-derive the incumbent from the project itself** where you can, instead of
  transcribing it. Where you must extract it, check that you extracted the
  configuration you meant: a kernel builder that emits both a CPU and a device
  variant will happily hand you the device variant carrying the *host* reduction
  body, which silently loses work under contention and makes any prototype look
  correct and fast by comparison.
- A mutation test on the prototype does **not** cover this. The mutant dies
  against a broken oracle exactly as it does against a good one.

**Score every implementation against the oracle, including the incumbent, and
report their errors separately.** Scoring only the candidate turns shared,
benign numerical noise into an apparent defect of the new code. The typical
case: a float64 oracle disagrees with the float32 candidate on a handful of
near-tied points, which reads as a semantic failure — until the incumbent is
scored against the same oracle and disagrees on the *identical* points, because
the disagreement belongs to float32, not to either implementation. Report
"candidate wrong on N₁, incumbent wrong on N₂" rather than "candidate differs
from oracle on N". `differs` is not a finding; the pair of error counts is.

## Adversarial test checklist

Run before any performance measurement:

- empty, singleton, and all-identical inputs;
- duplicates and exact ties;
- values exactly at, one step inside, and one step outside every threshold;
- capacity below, equal to, and above demand; overflow surfaced correctly;
- degenerate geometry and exact degeneracies;
- extreme scales, both very small and very large;
- noncontiguous arrays and unsupported dtypes hitting the fallback;
- tracked mutation, then refit and rebuild;
- non-default streams; repeated calls; repeated backward;
- graph capture, replay **with changed input**, and recapture;
- a large random audit at production scale, capable of exposing rare failures;
- every bound, pruning rule or early exit brute-forced against the unpruned
  path before it is benchmarked;
- the incumbent's own run-to-run spread on identical inputs, before any equality
  tolerance is fixed;
- both implementations scored against the oracle, with errors reported
  separately;
- weak-reference retention of any acceleration structure across graph capture
  and caller release;
- the unchanged upstream test suite — and if none of it covers the seam, say
  so, because then it protects nothing here. Check *what* it covers, not just
  that it passes: a suite whose seam tests are all float64 and a few dozen
  elements will route every case to the fallback and exercise none of a
  float32 fast path, so it protects nothing while appearing green.

**Never write a test that asserts a hazard reproduces.** "The stale cache must
return wrong answers", "the unsynchronised stream must corrupt output" — these
assert a *failure mode* as the pass condition, and they fail whenever the
implementation is better than you assumed or the race simply does not fire.
Characterise the frequency instead (`0/20 runs disagreed`) and record it as an
observation. Two consequences worth expecting:

- A hazard that does not reproduce is **not** a cleared hazard; it is an
  unquantified one. Report the trial count, keep the mitigation, and do not
  promote a single non-reproducing observation into a measured defect.
- A safety property stronger than you assumed is a **finding**, not a failed
  test. One audit expected a stale spatial index to produce wrong answers and
  found it produced correct ones, because the admissibility check refused the
  unproven candidates and fell through to the exact path. That converted cache
  invalidation from a correctness contract into a performance contract — a
  materially different, much safer API — and it would have been recorded as a
  test failure by a suite that insisted the hazard fire.

Finally: **deliberately break a key branch and confirm the suite catches it.**
A suite that cannot fail has not validated anything.
