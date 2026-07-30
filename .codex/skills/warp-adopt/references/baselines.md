<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Baselines — algorithm first, backend second

The single most common way a Warp assessment goes wrong is beating a baseline
nobody would defend.

## The ladder

Work down it. Stop as soon as something meets the objective.

1. **A better algorithm.** Asymptotically better, or output-sensitive so it
   stops doing work proportional to a quantity that does not matter.
2. **Representation and memory strategy.** Chunking, tiling, slabs, sparse
   output, layout change, rematerialization.
3. **The incumbent framework's compiler and native primitives.**
4. **Mature domain libraries and existing CUDA implementations.**
5. **Only then** a narrow Warp implementation.

Two source cases make the point. In one, a bounded chunked implementation in
the *existing* framework produced identical outputs, fit a size the materialized
path could not, ran several times faster than the Warp adapter, used less
process GPU memory, and kept JIT/autodiff — reversing an earlier narrow Warp
recommendation. In another, replacing a quadratic formulation with its
linear-time equivalent cut the working set by more than thirty-fold before any
backend question arose.

## The project's own accelerator backend is rung 4, and it is usually decisive

If the project already ships a GPU path — `numba.cuda`, CuPy, its own CUDA
kernels — then the baseline for a new-backend proposal is **that backend
running the same algorithm**, not the CPU path it currently uses and not the
array library's elementwise ops. Port the candidate kernel to the backend the
project already depends on, measure it, and only then ask what a new dependency
adds. If the existing backend closes the gap, the assessment is over and the
recommendation is "use what you already have".

Skipping this rung is how a CPU-versus-GPU result gets mistaken for a
Warp-versus-everything result. A source case: a serial CPU loop at 842.8 ms, the
same loop expressed in the array library at 213.3 ms, in the project's existing
GPU backend at 2.47 ms, and in Warp at 2.46 ms with bitwise-identical output.
Measured against the incumbent CPU path, the new backend looks like a 341x win;
measured against the dependency already in the project, it is parity — and the
real finding was that nobody had moved the loop to the GPU at all. Cite this as
an illustration of the failure mode, never as a forecast for your own case.

## Preserve the oracle

Whatever the strongest correct implementation turns out to be, keep it
selectable and use it as the correctness oracle for every later comparison. If
you improve the incumbent during this stage, the improved version — not the
original — is the baseline the prototype must beat.

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

Domain libraries worth installing before concluding Warp wins: Embree or
Open3D for CPU BVH point/ray queries; a maintained CUDA BVH library; an
indexed CUDA neighbor-search library; cuCIM/CuPy for image and label
operations; libigl for geometry references; OptiX where full traversal
semantics matter; and any approximate cached structure the project already
supports.

That last one matters more than it looks. An existing **approximate cache** —
a voxel SDF, a precomputed grid, a coarse index — routinely beats an exact GPU
path by a wide margin, because it answers a cheaper question. It is not a
drop-in competitor, and its error must be stated. But if the product tolerates
that error, the cheap cache wins and Warp is unnecessary. Compare like with
like, and state plainly what extra work each side performs.

## Equivalent work

Implementations must do the same job before a comparison means anything:

- same accepted input domain and filtering;
- same outputs and ordering where contractually required;
- same dtype, precision and fast-math policy;
- same convergence criterion or iteration count;
- same initialization and seeds;
- same structure quality, or an explicitly justified difference;
- same forward/backward requirement;
- same synchronization semantics;
- same inclusion of pre- and post-processing.

Where APIs genuinely differ, **state exactly what extra work each performs**.
A comparison against an indexed CUDA neighbour library, for instance, is
credible but rarely identical-work: such libraries typically return nearest-`K`
neighbours *and* squared distances *and* include self, while a fixed-capacity
Warp path returns arbitrary-order neighbours plus an overflow flag. Both facts
belong in the report.

## Compare optimized with optimized

Default-versus-default comparisons inflate ratios dramatically — published
backend comparisons have shown two-orders-of-magnitude ratios collapse to
single-digit ones once both sides are tuned. Always state which comparison you
ran. If you only had time for default/default, say so and do not present it as
a backend speedup.

The same trap has a second form: **changing the backend and the workload in the
same measurement**. A headline ratio that comes from porting *and* from
processing many systems concurrently is not a backend result — it is both
effects multiplied. Report the single-item comparison separately from the
batched one; the gap between them is what the port actually bought.

### The rule is symmetric — an untuned candidate proves nothing either

Everything above is written against leaving the *baseline* untuned, because
that is how ratios get inflated. Read it in both directions: **a candidate that
loses on its own unset defaults has not been compared, it has been
misconfigured**, and the resulting NO-GO is as unfounded as the inflated GO.

The usual form is a default heuristic that is wrong for the data's intrinsic
dimension or distribution. A spatial-index cell size derived from *bounding-box
volume* is ~6× too coarse for points lying on a 2-D surface in 3-D — a
completely ordinary case in geometry, graphics and simulation — and the
resulting candidate loses to brute force. Name the candidate's knobs before you
benchmark (see the stage-4 pre-registration block), sweep the ones with a
defensible physical meaning, and report the tuned comparison alongside the
default one. If a knob cannot be set from data the caller already has, that is
a finding about the seam's API, not a licence to hand-pick it per measurement.

## Transferable optimizations belong to both sides

**If a tuning idea is expressible in the incumbent framework, implement it
there before crediting it to Warp.** Reformulations, precomputation, blocking,
fusing a division into a reciprocal, hoisting invariants out of a loop — these
are almost always backend-neutral. A speedup you found while tuning the Warp
kernel and never tried on the baseline is a *formulation* result being reported
as a *backend* result, and it routinely reverses when you do try it.

The test is cheap and decisive: port the idea, re-measure both, and report the
pair. If the incumbent gains as much or more, the idea belongs in the
"improve the incumbent" recommendation, not in the Warp case. Note also that a
reformulation can change numerics — re-run the correctness audit on it rather
than assuming an algebraically equivalent rewrite is numerically equivalent.

## A fast wrong baseline is not a baseline

If the incumbent does not satisfy its own contract, it cannot be a performance
competitor. A fixed-iteration solver that has not converged, a truncated
neighbour list, or a tolerance-collapsed geometric test will all look fast
while producing wrong answers — exclude them and mark them as failing.
Compiling an incorrect algorithm preserves its incorrectness.

Conversely, an intentionally weak baseline is evidence only that the baseline
needs improvement. If you find one, the honest recommendation is often "improve
the incumbent" — and that may end the Warp path entirely.

### When the incumbent is the defective one, stop scoring against it

If the incumbent fails its own contract, it also stops being a usable
correctness oracle, and "% agreement with the incumbent" **inverts**: the more
correct an implementation is, the *worse* it scores. Expect this to be
counter-intuitive in the moment — a candidate that disagrees on more elements
may be the only one that is right.

Do this instead:

1. Construct a reference that satisfies the *intended* contract, and make it
   the oracle. Say how you built it and why you believe it.
2. Classify every disagreement with the incumbent as a **correction** or a
   **regression**, judged by the incumbent's own metric — the quantity its code
   is trying to compute — and report the two counts separately. "Differs on N
   elements" is not a finding; "corrects N₁, regresses N₂" is.
3. Isolate each candidate's own error by comparing it against the constructed
   reference as well as against the incumbent. Comparing only against a
   defective incumbent conflates the bug being fixed with the candidate's own
   tie behaviour, and hides both.
4. Report the defect to the user as a finding in its own right. Fixing it is
   normally the first recommendation and is **independent of the backend
   question** (skill rule 3).
