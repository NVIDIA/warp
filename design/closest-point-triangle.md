# Point–triangle closest point and feature classification

**Status**: In Progress

**Issue**: [GH-2150](https://github.com/NVIDIA/warp/issues/2150)

## Motivation

Newton's mesh contacts use a pure Warp point–triangle query that returns both
barycentrics and a vertex/edge/face identifier. Warp's native mesh-query helper
returns two barycentric weights, but is not a standalone public Warp function.
A public pure Warp query should provide the same useful information, support
both precisions and automatic differentiation, and improve behavior on thin,
collapsed, very small, and very large triangles.

## Design

`closest_point_point_tri(p, v0, v1, v2)` returns `(bary, distance_squared, feature)`.
The name, argument order, and parameter names follow `closest_point_seg_tri` and
`intersect_seg_tri`. `TriangleFeature` is an `IntEnum` with vertices `VERTEX_0`,
`VERTEX_1`, and `VERTEX_2` numbered 0–2, edges `EDGE_01`, `EDGE_02`, and `EDGE_12`
numbered 3–5, and `FACE` numbered 6. These values match Newton's feature
constants, which name the vertices A, B, and C.

The feature names the relative interior of the returned barycentric support.
Exactly zero weights identify the lower-dimensional features; there is no
geometric tolerance that erases small positive weights. Degenerate triangles
can have several representations of the same point. Their selected feature
is deterministic for a fixed input order, not invariant under vertex
permutations. A shared vertex ID/topological adjacency model is irrelevant to
this one-triangle query.

Well-shaped triangles use the established dot-product region determinants.
On slivers, cross-product region tests avoid subtracting nearly equal Gram
products. The shape threshold selects arithmetic; it never discards a feature. Fixed
power-of-two scaling reduces underflow/overflow without introducing rounding
from arbitrary normalization. The face projection is tested before endpoint
regions, so underflow in endpoint dot products does not misclassify a face
interior on a very thin triangle. On slivers, a dot projection recovers the
coordinate along the longer edge to avoid inconsistent independently rounded
area weights. Region selection produces unnormalized nonnegative weights;
all regions share a single reciprocal. The largest final weight is recovered
by partition of unity, preserving small weights and exact boundary zeros.

Gradients follow the selected continuous branch. Integer features, transitions,
and nonunique minima do not have unique geometric derivatives. Representability
of intermediate calculations still matters; this is not an exact-predicate API.

## Validation

The unit tests enumerate convex-hull active sets using exact rational arithmetic
on the *quantized* input floats, independently of the region-test implementation.
They check witnesses, distances, nonnegative barycentrics, partition of unity,
feature consistency, and the exact feature for nondegenerate inputs. Fixtures
cover all features, permutations, coplanarity, one-ULP boundary offsets, exact
oblique edge projections, degenerate triangles, scales from 1e-15 to 1e15,
rotated slivers, translation, finite-difference gradients in all seven regions,
and the squared-distance envelope gradient at multiple scales on CPU and CUDA.

`asv/benchmarks/closest_point_tri.py` compares against the actual native
`wp::closest_point_to_triangle` header and a verbatim copy of Newton's helper
at `f39d8199f18d5117f9dd8073c1eb71c9a196f020`. The Newton helper is unchanged at
`a1903cdc48f349d0de7d5e19f1838e736c4d1cf3`. Timings use GPU events captured inside the graph to exclude host scheduling
gaps. Record cgroup CPU quota and throttling before and after each run. Use
exclusive GPU admission and inspect activity inside the lock before measurement.
Timings exclude compilation, transfers, and BVH traversal; all implementations write the same outputs. Native feature
IDs are inferred from its returned weights, since it does not provide one.
This inference can expose rounding in the reconstructed third weight and must
not be described as a native feature-ID API failure.

The benchmark can omit feature output to verify that native timing does not
benefit the new query by imposing extra classification work. No timing assertions
are used in unit tests. Accuracy is reported separately for each implementation
against the rational oracle, including nonfinite results and feature mismatches.

## Performance status

The strict goal of being at least as fast on every workload remains open.
On this machine, the final three-run medians at 262,144 float32 queries are
approximately at parity on ordinary workloads, but thin triangles take about
4.2% longer than Newton and degenerate triangles about 2.9% longer. Thin-triangle
feature classification and extreme-scale behavior improve substantially.
At 1,048,576 queries all measured workloads are within 0.15% of both references.
These are machine-specific observations, not performance guarantees.

The [measurement report and raw results](https://github.com/alecjacobson/warp/blob/alecjacobson/closest-points-reports/reports/closest-points/point-triangle.md)
live on the separate reports branch, keeping measurement artifacts out of this
feature branch. The report also documents residual classification errors for
inputs within rounding distance of a boundary; this API is not an exact predicate.
