<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rejection gates

Every gate here answers one question: **is Warp simply wrong for this?** Each is
a property of the environment, the workload shape or the incumbent — never a
statement about how much evidence you happen to have. Missing measurements are
not a gate; they surface downstream in profiling and produce `INCONCLUSIVE`.

All of them resolve to `ABORT`.

Run these **before** profiling or prototyping. Any one can end the study. A
strong gate justifies `ABORT` with **no Warp prototype at all** — a
speculative port is not owed to anyone.

## The gates

### Gate A — deployment excludes NVIDIA

Production is CPU-only, or requires accelerated non-NVIDIA portability, with no
acceptable optional CUDA path. **Warp's CPU kernels are serial**, so "run it
on CPU instead" is not a performance fallback. → `ABORT`.

### Gate B — the boundary dominates

Data must cross host/device per small or infrequent call and the boundary
cannot be widened. Moving one low-intensity operation to the GPU and copying
the result straight back is a classic false positive. → `ABORT` unless the
caller can be restructured to keep data resident.

### Gate C — it is ordinary dense tensor algebra

Dense linear algebra, convolution, FFT, reductions, standard neural-network
layers, or anything already lowered efficiently by the framework or a vendor
library. This is what tensor frameworks are for. → `ABORT`; recommend the
framework's own primitives, its compiler, or the vendor library.

### Gate D — a mature CUDA implementation already satisfies the contract

If a maintained CUDA path already meets the semantics and approaches hardware
limits, Warp would duplicate maintained code. → `ABORT` unless the user
explicitly requested an allowed non-performance objective *and* defined an
acceptable performance-regression budget.

### Gate E — the project cannot own the obligations

Warp adds an optional dependency, runtime compilation, a kernel cache, a
version-qualification matrix, platform-specific tests and a fallback path.
If the project cannot accept these, the best War kernel in the world does
not help. → `ABORT`.

### Gate F — the region provably cannot matter

The candidate stage is too small a share of the requested metric for any
backend to move it. This gate only fires when you can already tell — a profile
the user supplied, an obvious structural bound, or arithmetic on figures they
quoted. → `ABORT`, stating the ceiling.

If you cannot tell without measuring, this is **not** a gate. Let it through
and settle it in profiling, which is what that stage is for.

**Apply it against the metric that candidate's pattern exhausts — not the
study's headline objective.** Classify the candidate first, read its row in
[target-patterns.md](target-patterns.md), and use the resource named in the
"usually decided by" column. A study whose stated objective is latency will
still contain candidates whose whole value is memory or capacity; carrying one
global metric into every screen is the most common way a real opportunity is
rejected. A Class-4 candidate dismissed on a latency share, or a Class-1
candidate dismissed without checking output capacity, has not been screened —
it has been mismeasured. **State which metric the gate was applied against**,
and state the device memory budget the judgement assumes.

## Naming the better route

When you reject, say what to do instead, if anything is evident:

| Situation | Better route |
|---|---|
| Dense array math, CPU | NumPy/SciPy, vectorized formulation, tuned BLAS |
| Dense array math, GPU, already in a framework | The framework's own ops; `torch.compile`/Inductor; `jax.jit` |
| One custom elementwise/reduction kernel in a Torch program | Triton, or `torch.compile` |
| Custom kernel inside JAX | Pallas, `lax.scan`/`fori_loop`, or a custom primitive |
| CUDA-resident array work, no custom control flow | CuPy; cuTile for dense tiles |
| Existing CPU numeric kernel, modest scale | Numba |
| Image/vision primitives on GPU | cuCIM |
| Classical ML on GPU | cuML |
| Geospatial ops | cuSpatial |
| Ray tracing / all-hits / complex traversal | OptiX |
| CPU BVH closest-point and ray queries | Embree (directly or via a wrapper) |
| A maintained CUDA library already does it | Use it |
| The bottleneck is algorithmic | Fix the algorithm; re-profile |
| Nothing is materially wrong | **No change.** This is a valid, common result |

Recommending any of these — or recommending nothing at all — is a **successful
outcome** of this skill.
