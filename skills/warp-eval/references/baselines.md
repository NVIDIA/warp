# Baselines — algorithm first, backend second

The most common way a Warp evaluation goes wrong is beating a baseline nobody
would defend.

## The ladder

Work down it in order, then measure Warp. An in-project route meeting the
objective is an observed fact, not a reason to omit the Warp comparison from an
authorized `warp-eval`.

1. **A better algorithm** — asymptotically better, or output-sensitive so it
   stops doing work proportional to a quantity that does not matter.
2. **Representation and memory strategy** — chunking, tiling, slabs, sparse
   output, layout change, rematerialization.
3. **The incumbent framework's compiler and native primitives.**
4. **What the project already depends on** — its own accelerator backend, a
   parallel idiom it ships but leaves off, or a capability an existing
   dependency exposes and nobody wired up.
5. **Only then** a narrow Warp implementation.

Rungs 1–4 are limited to what the project can reach **without a new
dependency**. Warp is the only addition this skill puts on the table. Rungs 1
and 2 routinely end the study on their own.

**The dependency set is what the manifests declare**, not what happens to be
installed or imported: `pyproject.toml` (including
`[project.optional-dependencies]`), `setup.cfg`/`setup.py`, `requirements*.txt`,
`environment.yml`, and the lockfile if one is committed. An optional extra is
in-project — a capability it ships is a rung-4 route, not a new dependency.

### Rung 1 is mandatory when the diagnosis is algorithmic

If the cause you wrote down is that the incumbent uses the **wrong algorithm** —
a linear scan where an index belongs, a rebuild where a refit belongs, a
materialized candidate set where a traversal belongs — rung 1 means implementing
that algorithm **on the incumbent's own hardware**, not tuning the existing one.
Tuning the wrong algorithm's constants measures the tuning, and leaves the
backend comparison unable to separate *the algorithm was wrong* from *the
hardware was slow*.

Until the right algorithm has run on the incumbent's hardware, **no speedup may
be attributed to the accelerator**: measure it, evidence it impossible, or get
it waived on the record. An honest, unoptimized implementation is enough to
separate the two effects, so no external project is needed. If writing one is
impractical, mark the backend comparison `not measured`.

## Rung 4 supplies the direct comparison

If the project already ships a GPU path — `numba.cuda`, CuPy, its own CUDA
kernels — the baseline for a new-backend proposal is **that backend running the
same algorithm**, not the CPU path it currently uses. Port the candidate kernel
to the dependency the project already has, measure it, and only then ask what a
new dependency adds. If the existing backend closes the gap, record the observed
parity or difference and still run the Warp comparison.

Skipping this rung turns a CPU-versus-GPU result into an apparent
Warp-versus-everything result, and hides the common real finding: that nobody
had moved the loop to the GPU at all.

### When the project has no accelerator backend

The ordinary case, so rung 4 needs a substitute rather than a "not applicable",
in this order:

1. **The project's own parallel idiom, turned on** — an OpenMP pragma in a
   sibling kernel, a `numba.prange`, an `n_jobs`/`n_threads` left at its
   default, a chunked path used elsewhere. This costs no new dependency and is
   frequently the whole finding.
2. **A capability an existing dependency exposes but the project never wired
   up.** Bindings routinely cover only part of a native API. Read the
   dependency's surface before concluding it cannot help; the fix may be a
   binding addition rather than a new dependency.

Record the substitution explicitly: "the project ships no accelerator backend;
rung 4 is its own OpenMP idiom applied to this seam". A rung answered "no" and
left there is how a serial baseline becomes the thing a GPU is measured against.

**Do not substitute a library outside the project's declared dependencies.** The
question is whether Warp beats what the project can already reach, not whether
it beats the ecosystem. If no existing dependency can express the better
algorithm, say so — "no in-project route to a BVH query; the only options are a
binding addition or Warp" is the finding.

**Measure the in-project route even when it is the harder one to measure.** Do
not benchmark whatever installs in one command while deferring the route that
needs a binding written, a flag plumbed through or a kernel ported. If reaching
the existing dependency's capability takes an afternoon of implementation, that
afternoon is part of the evaluation. Defer it only with the user's agreement,
and say plainly that the direct comparison is missing.

### Parallelism is part of the baseline's definition

A baseline is not "the incumbent" but "the incumbent at the parallelism the
deployment actually has". **A CPU baseline measured below the deployment's
thread count has not been compared.** State the thread count of every timed
configuration and the core counts of both the measuring host and the target.

## Preserve the oracle

Keep the strongest correct implementation selectable and use it as the
correctness oracle for every later comparison. If you improve the incumbent
here, the improved version — not the original — is what the prototype must beat.

## Strongest practical baseline by incumbent

| Incumbent | Baseline you must beat |
|---|---|
| NumPy | Vectorized formulation with tuned linked libraries and realistic CPU threading |
| CuPy | Built-in/library op, default accelerator backend, fusion, and a reasonable custom kernel |
| Numba | Optimized launch configuration, device-resident arrays, cached specialization |
| JAX | Outermost practical `jit`, correct static specialization, device-resident inputs, blocked completion; plus `scan`/`fori_loop` chunking and Pallas where justified |
| PyTorch | Built-in/fused op and `torch.compile`; plus any existing custom op that is the real production baseline |
| C++ | Optimized compiler settings, representative threading/vectorization, relevant native libraries |
| CUDA | Tuned kernel or library, realistic streams and graphs, production compiler settings |

Domain libraries can supply correctness references, but are performance
baselines only when the project already declares them: Embree or Open3D for CPU
BVH point/ray queries; a maintained CUDA BVH library; an indexed CUDA
neighbor-search library; cuCIM/CuPy for image and label operations; libigl for
geometry references; OptiX where full traversal semantics matter.

**Include any approximate cached structure the project already supports.** A
voxel SDF, a precomputed grid or a coarse index answers a cheaper question and
routinely beats an exact GPU path by a wide margin. It is not a drop-in
competitor and its error must be stated — but if the product tolerates that
error, report its measured time, memory and error beside Warp.

## Equivalent work

Hold these constant, or a comparison means nothing: accepted input domain and
filtering; outputs and ordering where contractually required; dtype, precision
and fast-math policy; convergence criterion or iteration count; initialization
and seeds; structure quality, or an explicitly justified difference;
forward/backward requirement; synchronization semantics; inclusion of pre- and
post-processing.

**The invoked public operation and configuration are part of the work.** A
benchmark of a helper, fallback or sibling operation cannot support a row or
screening claim about another operation. Hold input domain, option values,
output cardinality, ordering, duplicate handling and missing-value
representation constant. If a prototype implements only a subset, scope every
result to that subset and leave the rest unmeasured.

Where APIs genuinely differ, **state exactly what extra work each performs**. An
indexed CUDA neighbor library typically returns nearest-`K` neighbors *and*
squared distances *and* includes self, while a fixed-capacity Warp path returns
arbitrary-order neighbors plus an overflow flag. Both facts belong in the
report.

## Compare optimized with optimized

- **Default-versus-default comparisons inflate ratios dramatically**, and
  collapse once both sides are tuned. State which comparison you ran; if you
  only had time for default/default, say so rather than presenting it as a
  backend speedup.
- **The rule is symmetric: a candidate that loses on its own unset defaults has
  not been compared, it has been misconfigured.** Name the knobs before
  benchmarking, sweep the ones with defensible physical meaning, and report the
  tuned comparison alongside the default. Watch for a default heuristic wrong
  for the data's intrinsic dimension — a spatial-index cell size derived from
  bounding-box volume is far too coarse for points lying on a surface, and the
  candidate then loses to brute force. A knob that cannot be set from data the
  caller already has is a finding about the seam's API, not a license to
  hand-pick it per measurement.
- **Transferable optimizations belong to both sides.** Reformulations,
  precomputation, blocking, fusing a division into a reciprocal and hoisting
  invariants are almost always backend-neutral. A speedup found while tuning the
  Warp kernel and never tried on the baseline is a *formulation* result reported
  as a *backend* result, and it routinely reverses. Port the idea, re-measure
  both, report the pair; if the incumbent gains as much, the idea belongs in the
  "improve the incumbent" finding. Re-run the correctness audit — a
  reformulation can change numerics.
- **Do not change the backend and the workload in one measurement.** A headline
  ratio from porting *and* from processing many systems concurrently is both
  effects multiplied. Report the single-item comparison separately from the
  batched one; the gap is what the port bought.

## A fast wrong baseline is not a baseline

An incumbent that does not satisfy its own contract cannot be a performance
competitor. A fixed-iteration solver that has not converged, a truncated
neighbor list, or a tolerance-collapsed geometric test all look fast while
producing wrong answers — exclude them and mark them failing. Conversely, an
intentionally weak baseline is evidence only that the baseline needs improving.

### When the incumbent is the defective one

It also stops being a usable oracle, and "% agreement with the incumbent"
**inverts**: the more correct an implementation is, the worse it scores. The
candidate disagreeing on more elements may be the only one that is right.

1. Construct a reference satisfying the *intended* contract and make it the
   oracle. Say how you built it and why you believe it.
2. Classify every disagreement as a **correction** or a **regression**, judged
   by the incumbent's own metric, and report the two counts separately. "Differs
   on N" is not a finding; "corrects N₁, regresses N₂" is.
3. Compare each candidate against the constructed reference *as well as* against
   the incumbent. Comparing only against a defective incumbent conflates the bug
   being fixed with the candidate's own tie behavior.
4. Report the defect as a finding in its own right and abort the performance
   comparison until a valid oracle exists (hard rule 2 in `SKILL.md`).

**A green test is not evidence the fast path is correct — check what its fixture
exercises.** A parity test between a project's reference and accelerated path
can pass because the fixture is degenerate in exactly the variable the two paths
disagree on: a straight rod where the dropped term is proportional to curvature,
a zero mask, an empty batch, a single element. The guard exists, passes, and is
structurally incapable of failing. Before trusting an accelerated incumbent as
an oracle, confirm its fixture is non-degenerate in the quantities that
distinguish the implementations; if it is not, build one that is and re-run
there.

### When the *alternative* is the defective one

An in-project alternative that turns out to be wrong — a backend that does not
run with its own driver, or whose results diverge from the reference on
non-degenerate input — is not a baseline, however attractive its time. Name the
defect in its row's note, keep its raw results so the exclusion is auditable,
and do not let its number stand as something the candidate had to beat.
