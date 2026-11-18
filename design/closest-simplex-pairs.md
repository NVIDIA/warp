# Closest pairs for segments and triangles

## Motivation

Follow-up to [NVIDIA/warp#2150](https://github.com/NVIDIA/warp/issues/2150), tracked
on the fork in [#25](https://github.com/alecjacobson/warp/issues/25), under the
[suite roadmap](https://github.com/alecjacobson/warp/issues/24).

Newton currently minimizes its capsule SDF over a soft triangle using up to 12
Frank–Wolfe steps, each with a 12-step golden-section line search. Soft edges use
golden-section search directly. For a capsule with axis segment S and radius r,
`min(phi(x), x in T) = distance(T,S)-r`: an analytic closest-pair query computes
the same objective, including negative distances, without iterative search.

This is Newton's opt-in full-surface rigid–soft SolverVBD path. Its separate
rigid mesh/heightfield GJK/MPR contact path has different manifold and penetration
semantics. A closest pair is not a replacement for all contact manifolds or a
minimum translation distance.

## Design

Two pure Warp, inline, float32/float64 functions:

- `closest_points_seg_seg(p0,p1,q0,q1)` returns `(s,t,distance_squared)`.
- `closest_points_tri_seg(a,b,c,p,q)` returns `(bary,t,distance_squared)`.

The parameters reconstruct witnesses without adding redundant coordinate outputs.
They remain differentiable for consumers that need more than distance. No
intersection plane tolerance or absolute segment-length cutoff is used. Exactly
collapsed segments are points; zero-area triangles reduce to their edges.
Deterministic ties follow input order, so permutations need not preserve the
chosen witness at a nonunique minimum.

The segment query uses constrained line parameters with endpoint correction.
Its line determinant is the squared cross-product norm, avoiding cancellation
in `dot(u,u)*dot(v,v)-dot(u,v)**2` for almost parallel lines. Its residual is formed
in a translated frame to avoid subtracting reconstructed world-space witnesses.
For float32, the second direction is divided by its length before forming the
cross products. This reduces fourth-order products to second order without
changing the line parameter. Dividing by its squared length instead was rejected:
it overflows for a long first segment and a short second segment.

The triangle query first minimizes distance from the segment to the triangle's
supporting plane. If that point's projection is inside the triangle, it attains
the global plane-distance lower bound. Otherwise a closest pair exists on a
triangle edge, so three segment–segment queries suffice. To see this, an interior
face optimum permits movement in both tangent directions, forcing a normal
residual; optimization along the segment then either minimizes plane distance
or reaches a triangle boundary. This also covers parallel and coplanar inputs:
if the selected endpoint projection is outside but another portion projects
inside, it must cross the triangle boundary. Explicitly testing the plane case
is essential: triangle boundary tests alone miss interior piercing.
For float32, divide the normal by the larger squared edge length and restore
that factor in the barycentric denominator. This avoids losing valid interiors
when the unscaled squared normal underflows or overflows. Float64 retains the
unscaled expressions. See the [normalization review](../reports/closest-points/normalization-review.md)
for the measured tradeoff and Newton-specific rationale.

These are floating-point algorithms, not exact arithmetic. Inputs and
intermediates must fit the precision. Extreme dynamic ranges and ill-conditioned
coordinates may still need rescaling or higher precision. No universal
geometric-error bound is advertised.

### Derivatives

Warp differentiates the selected branch. Finite-difference tests cover distance
and parameter outputs away from feature transitions, including triangle-interior,
edge, and piercing cases. Derivatives at ties are not unique. Unlike the native
edge–edge adjoint, this implementation does not damp near-parallel parameter
derivatives; they can be large because the underlying problem is ill-conditioned.
Consumers must not silently assume identical gradient regularization.
Forward and reverse computations have different intermediate ranges: finite
forward values at extreme scales do not imply finite gradients. Scale-covariance
tests cover parameter and squared-distance gradients from 1e-6 to 1e6 on smooth
fixtures; the forward scale tests extend from 1e-15 to 1e15. Neither is a universal
error or representability guarantee.

### Capsule adapter

For Newton's local Z-axis capsule, use endpoints `(0,0,±half_height)`. Reconstruct
the soft witness and retain Newton's existing SDF/gradient evaluation, contact
margin, radius, and projection convention. The benchmark does exactly this so
the minimizing-point search is the only algorithmic substitution. An axis hit
has distance `-radius` but no unique normal. The benchmark preserves Newton's
fallback; choosing better deep-penetration normals is separate integration work.

## Source audit

The audit searched Python/C++/CUDA source for closest/distance functions and
segment/edge/triangle combinations, then inspected candidates. Absence below
means no equivalent direct pair query was found in these pinned revisions, not
a statement about all versions or every downstream project.

| Project / revision | Findings and comparison |
| --- | --- |
| Warp, branch base `eb5d91725` | Native float32 `wp.closest_point_edge_edge` returns `(s,t,distance)` and has an analytic adjoint, including near-parallel damping. Benchmarked directly with epsilon zero and equivalent outputs. Native `closest_point_to_triangle` is point–triangle, not triangle–segment. |
| [Newton `8a656068`](https://github.com/newton-physics/newton/tree/8a6560683ca99d17a5b5369a27f6fb886e2dcd23) | Capsule segment/face SDF optimization; native edge–edge callers in soft mesh contact, self-contact, and finite-plane contact; analytic capsule–capsule manifold routine. No direct triangle–segment closest-pair API found. |
| [PhysicsNeMo `64bc1d96`](https://github.com/NVIDIA/physicsnemo/tree/64bc1d96e7c0614512e2b71173236e5eacfdecf4) | `nn/functional/geometry/deform/_shrinkwrap_torch_impl.py:closest_points_on_triangles` is paired point–triangle, with scale-normalized math and degenerate edge fallback. Mesh SDF/shrinkwrap uses Warp mesh queries. Experimental ISLA also has point–segment/triangle helpers. No equivalent segment–segment or triangle–segment closest-pair API found. |
| [Kaolin `ee8b7eca`](https://github.com/NVIDIAGameWorks/kaolin/tree/ee8b7eca1e56b1b9fd1509fc406f8d8ccbc9240b) | `metrics/trianglemesh.py` and `csrc/metrics/unbatched_triangle_distance_cuda.cu` implement point-to-mesh distance through point–triangle feature tests. No equivalent segment–segment or triangle–segment closest-pair API found. |

Point–mesh routines are not timed as if they performed the same query: they add
search/reduction over triangles but do not minimize over a segment. No speedup
against PhysicsNeMo or Kaolin is claimed. Their treatment of precision and
degeneracy informs the tests.

## Validation and performance

The unit tests enumerate active sets of an independent convex quadratic program.
Small and ill-conditioned cases use rational Gaussian elimination; larger random
sets use NumPy least squares over active sets. Tests also check reconstructed
witnesses, parameter domains, permutations, reversals, scale variation,
degeneracies, thin triangles, almost-parallel segments, and autodiff on CPU/CUDA.
Capsule witness tests exercise cap/side contacts, penetration, axis crossings,
zero half-height, rigid transforms, margins, and unique-normal surface projections.

`asv/benchmarks/closest_points.py` provides maintained ASV cases plus a standalone
CUDA-graph runner. The standalone runner can import the pinned Newton source
directly. The included capsule-specialized reproduction preserves the algorithm
but may differ in compiler rounding, selected iterates, and timings; original
Newton measurements are authoritative when reporting speedup. Compilation,
allocation, transfers, and reference evaluation are excluded from timed graphs.

See [the measured report](../reports/closest-points/README.md) for exact commands,
hardware, workloads, accuracy results, and limitations. No timing assertions are
added to tests. These measurements assess narrow-phase queries and a shared
initial SDF cull; they do not establish an end-to-end SolverVBD speedup.
