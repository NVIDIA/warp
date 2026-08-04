# Semantic contract and correctness hazards

Write the contract before building anything; the hazard catalog is what
adversarial testing must cover once you do. Each hazard here has, on its own,
decided a migration.

## Write the contract before the prototype

Tolerances chosen **after** seeing Warp's output are not tolerances. Write down
first:

**Observable behavior** — values, dtypes, shapes, devices, error conditions,
what may be mutated; ordering, tie-breaking, capacities, overflow, partitions,
topology, degeneracy; which outputs are **semantic** and which are
implementation artifacts (arbitrary IDs may be free to vary, a partition or a
selected face may not); numerical, geometric and task-level tolerances, each
with a number; gradients only where already required or explicitly requested.

**Lifecycle** — which stream work launches on and who synchronizes; who owns
inputs, outputs, caches and acceleration structures; whether any returned array
views internal storage; what mutation triggers refit versus rebuild; how many
builders/caches may exist per device, stream and thread, and what concurrent
misuse does; what must stay static and alive across graph capture and replay;
what happens when the GPU is absent, compilation fails, or an unsupported
dtype/device arrives.

Two rules over both:

- **For an API that already ships, its docstrings and tests *are* the
  contract.** Read them as binding rather than inferring a looser one the
  prototype happens to satisfy.
- **Correctness and required gradients veto performance.** Never weaken a
  contract after a mismatch. If the mismatch is acceptable, that is a *product
  decision* someone makes explicitly — record it as one.

## Hazard catalog

**Cutoff ties and float classification.** Specify `<` versus `<=` and the
arithmetic precision exactly. Test a point exactly on the cutoff and one
representable step either side. Expect a handful of disagreements per millions
of pairs, and expect two implementations of the *same* incumbent framework to
disagree with each other. Decide up front whether production needs bit-identical
sets, a geometric tolerance, or only invariant downstream quantities.

**Truncation and overflow.** A capacity-sized count cannot distinguish "exactly
full" from "truncated" — return the unclamped count or an explicit overflow bit.
Traversal order is usually unspecified, so above capacity test only that
retained entries are valid unique members of the true set; do not promise
nearest or reference ordering unless you implemented it. Size capacity from
clustered and adversarial distributions plus telemetry, never one
average-density sample.

**Face and distance-type ties.** Equal distances can select different valid
faces. Harmless for a scalar distance; fatal when the selected face and a
multi-way distance type feed the backward pass and are user-visible. Test exact
duplicate faces, neighboring coplanar faces, shared edges/vertices and
degenerate triangles, and re-test on every version upgrade — tie order is rarely
a documented guarantee.

- **Measure how often the ambiguity is reachable before judging a mismatch
  rate.** On a closed mesh the closest point lands on a shared edge or vertex
  far more often than in a face interior, so most face indices can be
  *legitimately* ambiguous — while the same comparison on synthetic triangle
  sets runs orders of magnitude cleaner. Report the reachability statistic next
  to the mismatch count, or the count means nothing.
- A raw disagreement rate measures **the implementation you built**, not the
  seam's potential: taking whatever face the acceleration structure returns will
  disagree constantly, while a kernel that reproduces the incumbent's branch
  order can drive the same comparison near zero. Say which you measured.

**Gradients at edges, vertices, seams and medial axes.** The closest face and
its normal are not unique at nonsmooth locations. An all-point maximum gradient
difference is usually dominated entirely by the nondifferentiable region and
says nothing about smooth-region accuracy — report the two separately. Use
finite differences **away** from topology changes. Test repeated backward and
every gradient operand. A shared existing backward is safe only if the IDs and
types you feed it satisfy its original assumptions.

**Topology drift.** Tiny field differences near zero or near sampling boundaries
change marching-cubes face indices, so a result can preserve exact vertex, face
and component counts and exact bounds and still not be byte-identical. Validate
the **downstream artifact**: component counts, largest component, surface area,
bounds, watertightness, bidirectional geometric distance.

**Float32 scale behavior.** Warp mesh queries operate in float32/int32 even
behind a float64 public API. Relative error stays tiny while absolute error
grows with coordinate magnitude — at large scales, float32 spacing alone can put
absolute point error in the tens at a relative error still around 1e-7. Sweep
realistic **and** extreme scales, state the supported coordinate envelope, and
never silently downcast a float64 contract.

**Nondeterministic atomics and reductions.** Atomic append or union order
changes edge/label order, which changes downstream floating-point rounding.
Compare semantic sets and partitions, and deterministic public labels where the
contract requires them. Define a physical-invariant tolerance rather than
demanding bit equality.

- **Measure the incumbent's own run-to-run spread first.** Call it repeatedly on
  identical inputs and record the spread of every output: that is the **floor**
  for any equality contract. A tolerance tighter than the floor describes a test
  the incumbent itself fails.
- "Bitwise equal" looks like the safest thing to pre-register and is not: a
  backward kernel accumulating with atomics can be reproducible only to a few
  parts in 1e5, making a bitwise contract unsatisfiable by anything, including
  itself. Establishing the floor measures the *baseline*, so it belongs before
  the contract rather than compromising it.
- When both implementations select identical indices, any residual difference in
  an atomically-accumulated output is ordering noise by construction: assert the
  index equality, then bound the value difference by the measured floor.

**Output aliasing.** A builder returning a view into internal storage overwrites
previous outputs on the next call. Either name the alias as a borrowed view or
return fresh framework-owned arrays. Test two live outputs across a subsequent
call and a garbage collection.

**Mutable external state.** A functional framework model does not make an
external hash grid immutable. A side-effect token orders calls **within one
executable**; it does not serialize independent host threads or executables.
Scope builders per stream/executable, document non-shareability, and either test
or reject concurrent use.

**Stream and lifetime ordering.** Zero-copy pointers need both storage lifetime
**and** CUDA ordering. Record a readiness event after structure construction
when another stream may query. Keep input storage, Warp objects, cache wrappers,
captured graphs and output owners alive through final stream completion. Test
non-default streams, producer/consumer events, deletion before completion, and
repeated create/destroy.

**Pointer-keyed graph caches.** Graph capture can be keyed by call descriptor
**and buffer pointers**. Capture counts usually stabilize after a few distinct
pointer sets, but allocator churn can exceed the graph-cache limit and cause
repeated recapture and eviction — measure capture count and latency under
realistic churn, since stable warm samples hide it.

**Probe retention explicitly; it is not visible any other way.** Take a weak
reference to the acceleration structure and its owning cache, capture the graph,
drop every strong reference the *caller* would plausibly drop, force a
collection, and assert the weak references are still alive. A captured graph can
retain raw pointers without retaining the Python object that owns the underlying
structure, so a replay after the caller released its cache runs against a freed
handle — while every eager test passes and every warm replay looks correct. Do
this even when the seam does not itself capture graphs: a persistent-index API
invites a caller to capture around it. If a convenience path builds a cache
internally and does not return it, say plainly that the path is not
capture-safe.

**Bounds, pruning and early exit.** Any argument that lets an implementation
**skip work** — an admissible lower bound, a monotonicity claim, a conservative
filter, a hierarchical cull, an early exit — is a correctness proof obligation,
not an optimization detail. An inadmissible bound produces a *plausible, fast,
wrong* answer that will look like your best-performing variant, and it is
impossible to catch by inspection or by timing. Before benchmarking such a
variant, brute-force it against the unpruned implementation on randomized inputs
at scale and confirm it never selects a worse result under the contract's own
metric. State the bound in the report so a reader can check the algebra. **An
unvalidated pruned variant must not be quoted as a performance result.**

**Optional dependencies and version drift.** Pin or qualify Warp and every
compiled dependency across the supported matrix. Lazy imports must preserve the
default path when Warp is absent. Re-run tie, graph, stream and build tests on
every Warp/framework/CUDA upgrade.

## Validate the oracle, not just the prototype

Every check compares the prototype against the incumbent, which is evidence only
if the incumbent is the real one, running correctly, in the configuration you
think.

- **Assert an independent invariant on the incumbent's own output** — a quantity
  derivable without either implementation (in-range row count, a conservation
  law, a known total). Print it next to the comparison.
- **Prefer a third, independent reference.** Agreement between two
  implementations tells you nothing when both share a wrong assumption. When the
  third disagrees with both accelerated ones, suspect the reference first —
  unfused host and fused device arithmetic legitimately disagree near
  boundaries.
- **Re-derive the incumbent from the project** rather than transcribing it, and
  check you extracted the configuration you meant: a kernel builder emitting
  both CPU and device variants will hand you the device variant carrying the
  *host* reduction body, which silently loses work under contention and makes
  any prototype look correct and fast.
- A mutation test on the prototype does **not** cover this — the mutant dies
  against a broken oracle exactly as against a good one.
- **Score every implementation against the oracle, including the incumbent, and
  report their errors separately.** Scoring only the candidate turns shared
  benign numerical noise into an apparent defect of the new code: a float64
  oracle disagrees with a float32 candidate on near-tied points, which reads as
  a semantic failure until the incumbent is scored against the same oracle and
  disagrees on the *identical* points. `differs on N` is not a finding;
  "candidate wrong on N₁, incumbent wrong on N₂" is.

## Adversarial test checklist

Run before any performance measurement:

- empty, singleton, and all-identical inputs;
- duplicates and exact ties;
- values exactly at, one step inside, and one step outside every threshold;
- capacity below, equal to, and above demand; overflow surfaced correctly;
- degenerate geometry and exact degeneracies;
- extreme scales, very small and very large;
- noncontiguous arrays and unsupported dtypes hitting the fallback;
- tracked mutation, then refit and rebuild;
- non-default streams; repeated calls; repeated backward;
- graph capture, replay **with changed input**, and recapture;
- a large random audit at production scale, sized to the defect rate you need to
  detect;
- every bound, pruning rule or early exit brute-forced against the unpruned path
  before it is benchmarked;
- the incumbent's own run-to-run spread, before any equality tolerance is fixed;
- both implementations scored against the oracle, errors reported separately;
- weak-reference retention across graph capture and caller release;
- the unchanged upstream suite — and check *what* it covers, not just that it
  passes. A suite whose seam tests are all float64 and a few dozen elements
  routes every case to the fallback and exercises none of a float32 fast path,
  protecting nothing while appearing green.

## Never write a test that asserts a hazard reproduces

"The stale cache must return wrong answers" makes a *failure mode* the pass
condition, and fails whenever the implementation is better than you assumed or
the race does not fire. Characterize the frequency instead
(`0/20 runs disagreed`) and record it as an observation.

- A hazard that does not reproduce is **not cleared**, it is unquantified.
  Report the trial count and keep the mitigation.
- A safety property stronger than you assumed is a **finding**, not a failed
  test. An admissibility check that refuses unproven candidates and falls
  through to the exact path converts cache invalidation from a correctness
  contract into a performance one — a materially safer API that a suite
  insisting the hazard fire would have recorded as a failure.

Finally: **deliberately break a key branch and confirm the suite catches it.** A
suite that cannot fail has validated nothing.
