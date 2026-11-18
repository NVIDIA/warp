# Closest-pair normalization review

Adopt float32 triangle-normal rescaling. Correct the reviewed segment rescaling
to divide the second direction by its **length**, not its squared length.
Float64 arithmetic retains its original expressions. Public signatures are unchanged.

## Newton rationale

Newton's pinned [face-contact path](https://github.com/newton-physics/newton/blob/8a6560683ca99d17a5b5369a27f6fb886e2dcd23/newton/_src/geometry/soft_contacts_sdf.py#L590)
rejects AABB misses, transforms triangles to capsule-local coordinates, and
rejects pairs whose centroid SDF exceeds the contact threshold plus the maximum
centroid-to-vertex distance. The compacted optimizer consumes survivors of this
classification. Large triangles can survive this cull without being close, so
this is not a claim that every surviving pair is near contact.

The [DAT cloth example](https://github.com/newton-physics/newton/blob/8a6560683ca99d17a5b5369a27f6fb886e2dcd23/newton/examples/multiphysics/example_vbd_dat_rigid_soft.py#L70)
uses 1.6/40 and 1.5/25 metre sheet-grid spacings, capsule radii 0.05–0.10 m, and
capsule half-height 0.16 m. These are small, structured triangles compared with
our unrestricted random workload. This supports prioritizing local near-contact
performance, but no actual surviving-query distribution or end-to-end solver
profile was captured in this review. Parallel cases are useful stress tests,
not evidence of their frequency in Newton.

Extreme scales are not typical of that example. Nevertheless, the public
closest-point query should not silently miss a piercing interior when inexpensive
rescaling fixes it. A roughly 1.7% random-workload cost is a reasonable tradeoff
for that correctness improvement, with gains on near and parallel workloads.

## Regression found in the reviewed segment fix

For endpoints

```text
p0 = (-1e10, 0, 1), p1 = (1e10, 0, 1)
q0 = (0, -1e-10, 0), q1 = (0, 1e-10, 0)
```

the exact parameters are `(0.5, 0.5)` and squared distance is `1`. The reviewed
`v / dot(v,v)` formulation returned `(0, 0.5)` and approximately `1e20` on CPU and
CUDA. It normalized uniform scale but made the cross product too large for
mixed segment lengths. Using `v / sqrt(dot(v,v))` fixes this fixture and its
swapped order. Tests cover length contrasts through `1e15` versus `1e-15`, using
an absolute distance check so a tolerance relative to the longest edge cannot
hide the error.

## Paired capsule timings

Same RTX PRO 6000 Blackwell MIG 1g.24gb and 262,144 float32 pairs as the original
report. Each round is the median of 21 alternating CUDA-event samples, with
30 launches per graph; the table reports the median of three rounds. Both sides
use the same packed output record and downstream SDF/normal evaluation. The
baseline is the other agent's reviewed code, including its segment rescaling.
These compare analytic implementations, not the Newton iterative optimizer.

| Triangle workload | Reviewed (µs) | Rescaled (µs) | Time change |
| --- | ---: | ---: | ---: |
| random | 15.906 | 16.172 | +1.67% |
| near | 17.869 | 16.845 | −5.73% |
| parallel | 16.436 | 15.315 | −6.82% |
| separated | 17.869 | 16.841 | −5.75% |
| degenerate | 17.750 | 17.657 | −0.53% |

Capsule segment timings changed by at most 0.11% across these workloads. The
capsule axis has unit length in this benchmark, so that result does not establish
the cost of length normalization for arbitrary segment endpoints. The separate
general-segment benchmark measured 9.131 µs for the new float32 random query versus
9.073 µs for the native routine; other float32 rows differ by at most 0.21%.
Near-parallel maximum sampled distance error remains `4.91e-8`, versus `5.04e-6`
for native. Float64 timings and sampled errors remain consistent with the initial
report. These are machine-specific measurements, not performance guarantees.

Raw data: [paired capsule comparison](normalization-review.json) and
[general segment comparison](segments-review.json). The paired comparison uses
the existing ASV benchmark's kernel factory with the query functions bound to
either the reviewed snapshot or the current implementation. Both source hashes
are recorded; its baseline snapshot and comparison driver are embedded in the JSON.

The embedded driver is recorded exactly as it ran, so it is not portable as is.
It reads the benchmark from the measurement machine's checkout
(`/home/horde/Repos/warp.intersection`), so set that path to your own checkout
before rerunning it. It also imports the baseline from
`/tmp/closest_review_baseline.py` but hashes `/tmp/closest-review-baseline.py`.
To check `baseline_sha256`, hash the embedded `baseline_source` directly rather
than relying on either file path.

## Validation and limits

106 selected geometry tests pass on CPU and CUDA, covering both precisions.
Pre-commit passes, the Towncrier draft renders, and the docs build completes
without warnings.
The closest-pair module contributes 22 tests, including the new mixed-length
regression and triangle scale cases. Finite differences still pass; additional
tests verify gradient scaling for parameters and squared distance between
`1e-6` and `1e6`. Forward triangle tests span `1e-15` through `1e15`, including
interior piercing and independent reference distances.

At more extreme scales, reverse-mode intermediate products can overflow even
when forward values are finite. This was observed in squared-distance gradients
at scale `1e15` and is now documented. The normal rescaling does not make gradients
globally finite, remove geometric ill-conditioning, or define unique derivatives
at feature transitions. Newton integration and its axis-normal fallback remain
separate work.
