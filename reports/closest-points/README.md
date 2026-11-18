# Closest-pair implementation and benchmark report

These measurements describe the initial implementation. The subsequent
[normalization review](normalization-review.md) records the reviewed source,
additional regression fixes, and updated timings. The original JSON source
checksums are retained as historical provenance, not checksums of the current code.

Measured on an **NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb**
(46 SMs, approximately 24 GiB), not the full GPU. Driver 595.91.07, CUDA toolkit
12.9, Python 3.12, Warp 1.20.0.dev0. Results are specific to this machine and workload.

Each timing is microseconds per launch for **262,144 pairs**, the median of 21
CUDA-event samples of a graph containing 30 launches. Method order alternates.
Block size is 128. Compilation, transfers, allocation, broad phase, and oracle
evaluation are excluded. Inputs are seeded with 25. Capsule comparisons are
float32; the general segment benchmark also covers float64.

## Newton capsule queries

The authoritative baseline imports the actual functions from Newton commit
`8a6560683ca99d17a5b5369a27f6fb886e2dcd23`. It runs with the same local Warp build
and GPU launch configuration as the new implementation. It preserves Newton's
centroid initialization, 12 outer/line-search iterations, capsule gap early exit,
and final SDF/normal calculation. Inputs are already in capsule-local space, with
radius and half-height both 0.5. The benchmark writes the same packed record of
barycentrics, signed distance, normal, and cull flag for every implementation.

The new implementation changes only how the minimizing point is found. Downstream
SDF/normal evaluation, including Newton's nonunique-normal fallback, is shared.
This is an adapter benchmark, not a complete collision pipeline or SolverVBD run.

| Query | Workload | Original Newton (µs) | Pure Warp (µs) | Speedup |
| --- | --- | ---: | ---: | ---: |
| seg–capsule | random | 48.66 | 22.11 | 2.20× |
| seg–capsule | near | 25.52 | 22.99 | 1.11× |
| seg–capsule | parallel | 38.45 | 22.38 | 1.72× |
| seg–capsule | separated | 25.18 | 22.65 | 1.11× |
| seg–capsule | degenerate | 32.34 | 22.10 | 1.46× |
| tri–capsule | random | 501.27 | 16.13 | 31.08× |
| tri–capsule | near | 203.27 | 17.76 | 11.44× |
| tri–capsule | parallel | 400.56 | 16.30 | 24.58× |
| tri–capsule | separated | 148.10 | 17.74 | 8.35× |
| tri–capsule | degenerate | 394.27 | 17.86 | 22.08× |

The included capsule-only reproduction is also timed in the JSON as `newton`;
the unmodified imported implementation is `original`. They are **not claimed
bit-identical or timing-identical**. Specializing dispatch changes compiler
rounding and can change later line-search decisions. The recorded
`specialization_max_sdf_delta` exposes this difference. Use `original`, as above,
for reported speedups. The reproduction makes ASV self-contained and preserves
the algorithm's work; direct-source mode is the stronger comparison.

### Accuracy

For each workload, 256 evenly spaced input pairs are checked against an independent
float64 active-set convex quadratic-program oracle, using the actual quantized
float32 inputs. The following errors concern **minimum signed-distance values**,
not accuracy of the minimizing location or normal. Nonunique locations can differ.

| Query | Workload | Newton max absolute SDF error | Pure Warp max absolute SDF error | Confirmed missed contacts by Newton |
| --- | --- | ---: | ---: | ---: |
| seg–capsule | random | 0.0021 | 9.55e-08 | 26 |
| seg–capsule | near | 0.000249 | 4.74e-08 | 81 |
| seg–capsule | parallel | 0.00145 | 1.14e-07 | 18 |
| seg–capsule | separated | 0.000455 | 1.91e-07 | 0 |
| seg–capsule | degenerate | 1.14e-07 | 1.14e-07 | 0 |
| tri–capsule | random | 0.183 | 2.98e-07 | 4908 |
| tri–capsule | near | 8.67e-05 | 6.56e-08 | 3 |
| tri–capsule | parallel | 0.00585 | 1.27e-07 | 24 |
| tri–capsule | separated | 2.47e-05 | 1.91e-07 | 0 |
| tri–capsule | degenerate | 3.15e-05 | 1.25e-07 | 1 |

Missed-contact counts cover the full 262,144 pairs at threshold 0.05. Every case
where Newton rejects but the new query accepts is independently re-evaluated by
the oracle. Only oracle distances below `0.05 - 3e-6` are counted as confirmed;
borderline disagreements are separately recorded in the JSON. This is measured
evidence for these inputs, not an exhaustive correctness proof. Random triangles
can be large relative to the capsule; the `near` workload uses small triangles
near its side to represent easier local contact searches.

### Including Newton's initial midpoint/centroid cull

The same SDF center/radius cull is placed ahead of both queries. This excludes
world transforms, AABB culling, queue compaction, contact-buffer atomics, and solver
work. On the `separated` workload all pairs are rejected before either query;
the numbers then reflect generated-kernel overhead and output writes, not the
closest-point algorithm. Zero accuracy error on an all-culled JSON row means no
active query was tested in that row.

| Query | Workload | Pairs reaching query | Newton (µs) | Pure Warp (µs) |
| --- | --- | ---: | ---: | ---: |
| seg–capsule | random | 227490 | 50.29 | 21.83 |
| seg–capsule | near | 233250 | 27.30 | 21.97 |
| seg–capsule | parallel | 72256 | 35.15 | 21.77 |
| seg–capsule | separated | 0 | 17.61 | 19.92 |
| seg–capsule | degenerate | 15930 | 22.80 | 20.88 |
| tri–capsule | random | 259699 | 507.15 | 19.10 |
| tri–capsule | near | 252412 | 206.45 | 17.39 |
| tri–capsule | parallel | 72360 | 302.24 | 18.44 |
| tri–capsule | separated | 0 | 17.75 | 17.87 |
| tri–capsule | degenerate | 127862 | 395.66 | 16.90 |

## General segment–segment versus Warp native

Both kernels consume arbitrary segment endpoints and return `(s,t,distance)`;
the new squared-distance result is square-rooted to match the native contract.
The native epsilon is zero, so short segments are not deliberately collapsed.
An independent exact-rational active-set oracle checks 256 input pairs per row.
This test is separate from capsule-specific code generation and output records.

| Precision | Workload | Native (µs) | Pure Warp (µs) | Native max distance error | Pure Warp max distance error |
| --- | --- | ---: | ---: | ---: | ---: |
| float32 | random | 9.08 | 9.01 | 1.26e-07 | 2.15e-07 |
| float32 | parallel | 9.13 | 9.04 | 7.99e-08 | 1.18e-07 |
| float32 | near_parallel | 9.08 | 9.07 | 5.04e-06 | 4.91e-08 |
| float32 | degenerate | 9.21 | 9.21 | 1.79e-07 | 1.79e-07 |
| float64 | random | — | 144.82 | — | 2.22e-16 |
| float64 | parallel | — | 129.66 | — | 1.57e-16 |
| float64 | near_parallel | — | 114.47 | — | 7.31e-17 |
| float64 | degenerate | — | 65.80 | — | 4.44e-16 |

Float32 throughput is effectively tied on these memory-bound batches. The new
cross-product formulation improves the tested near-parallel maximum error by
about two orders of magnitude. Ordinary-case rounding errors are comparable,
not uniformly smaller for every input. Float64 has no native edge–edge counterpart
and runs on hardware with much lower float64 throughput. The native adjoint damps
near-parallel parameter gradients; the new query differentiates its selected
branch without that regularization, so derivative semantics are not identical.

## Source audit and scope

See [the design/source audit](../../design/closest-simplex-pairs.md) for pinned
Warp, Newton, PhysicsNeMo, and Kaolin sources. Newton and Warp provide comparable
segment/capsule operations and are measured here. PhysicsNeMo and Kaolin contain
point–triangle/point–mesh operations, but no equivalent direct pair API was found;
we do not claim a speedup over a different query.

## Reproduction

From the Warp checkout (with its native library built):

```bash
git clone https://github.com/newton-physics/newton.git /tmp/newton-closest-reference
git -C /tmp/newton-closest-reference checkout 8a6560683ca99d17a5b5369a27f6fb886e2dcd23
uv run asv/benchmarks/closest_points.py --newton-source /tmp/newton-closest-reference --output /tmp/capsule.json
uv run asv/benchmarks/closest_points.py --newton-source /tmp/newton-closest-reference --initial-cull --output /tmp/capsule-cull.json
uv run asv/benchmarks/closest_points.py --segments --output /tmp/segments.json
uv run -m warp.tests.geometry.test_closest
```

ASV discovers `closest_points.ClosestPairs` and `closest_points.SegmentPairs`.
ASV timing methods measure a whole 30-launch graph; the standalone report divides
by 30. Raw measurements include source checksums and are retained in
[capsule.json](capsule.json), [capsule-cull.json](capsule-cull.json), and
[segments.json](segments.json).

The new unit tests pass on CPU/CUDA in both precisions; the selected geometry
regression run passed 98 tests. Capsule witness checks cover side/cap contacts,
penetration, axis crossings, zero half-height, rigid transforms, contact margins,
and surface projection where the normal is unique. At an axis hit the normal is
nonunique; Newton's fallback need not yield a valid surface projection, so its
solver behavior remains integration work.

Pre-commit and ASV discovery checks passed. Docs built successfully; a timeout
fetching NumPy's intersphinx inventory caused nine external-reference warnings.
No timing assertions are added to CI. Full Warp and Newton simulation suites
have not been run. Newton itself has not been modified.

Authored by Codex from measured results and source inspection.
