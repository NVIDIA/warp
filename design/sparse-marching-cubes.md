# Sparse Marching Cubes

**Status**: Implemented

**Issue**: No tracking issue; prompted by an external user request (see Motivation).

## Motivation

:class:`warp.geometry.IsoSurfaceMarchingCubes` extracts an isosurface from a **dense** 3D scalar
field: the caller must materialize an `nx x ny x nz` array and pay `O(R^3)` in
both memory and field evaluations, where `R` is the per-axis resolution. For
high resolutions this is wasteful, because a surface is a 2D object: only
`O(R^2)` cells actually straddle it, and the remaining `~R^3` interior/exterior
samples contribute nothing to the output.

An external user (surfacing very high-resolution custom sparse data structures)
asked for isosurface extraction that takes an **implicit function** -- something
that can evaluate the field at a point -- rather than a pre-filled dense grid,
so the extractor itself decides where to sample. This is the common shape for
signed distance functions (SDFs), neural implicit (NeRF-style occupancy/SDF
networks), and mesh distance queries, where evaluating a full dense grid at the
target resolution is expensive or infeasible.

This is a well-trodden approach in geometry processing. libigl provides
`igl::lipschitz_octree` (adaptively find the cells near the level set of a
1-Lipschitz function) followed by the sparse-voxel overload of
`igl::marching_cubes`. The libigl tutorial `1001_LipschitzOctree` demonstrates
the asymptotic win over a dense grid. This feature re-creates that pipeline in
pure Warp, running entirely on the GPU.

A closely related request came up in discussion: teams building sparse variants
of dense mesh-extraction methods (e.g. Flexicubes) for training 3D generative
models often *already have* a marked set of occupied voxels near the object and
just want to extract a mesh on those cells. So the extraction stage should be
callable on an explicit cell set, independent of how the cells were chosen.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R1  | Extract an isosurface from an implicit function without materializing a dense grid | Must | The core ask |
| R2  | Run cell selection, corner de-duplication, and extraction on the GPU | Must | These stages never round-trip to host |
| R3  | Accept the implicit function as a batched Python callable `evaluate(points) -> values` | Must | Batches evaluate meshes, neural implicit, and NumPy/PyTorch fields uniformly; see "Batched-callable-only contract" below |
| R4  | Produce a watertight, manifold mesh matching dense marching cubes at equal resolution | Must | Correctness / fair comparison |
| R5  | Expose the extraction stage on an explicit `(cells, corner_values)` list | Should | Vision/genAI "marked voxels" workflow |
| R6  | Expose the cell-selection stage (`lipschitz_octree`) on its own | Should | Custom extractors, visualization |
| R7  | Asymptotically beat the dense grid in time and evaluations as resolution grows | Should | The performance justification |
| R8  | Support backward-mode autodiff w.r.t. field values, without differentiating cell selection | Should | Matches dense `IsoSurfaceMarchingCubes`; see "Backward-mode autodiff" below |

**Non-goals**: Adaptive/octree *output* meshes (the output is a uniform-resolution
mesh, like dense marching cubes; the octree is only used to prune work).
Dual contouring or feature-preserving extraction. Native (C++/CUDA) code -- this
is implemented entirely in the pure-Python Warp layer.

## Design

### Approach

Two stages, each exposed as a public function, composed by a third:

1. **`wp.geometry.lipschitz_octree(sdf, origin, root_width, max_depth, ...)`** -- build a
   sparse set of leaf cells that provably bracket the level set of a 1-Lipschitz
   field, top-down, level by level.
2. **`wp.geometry.sparse_marching_cubes_from_cells(cells, corner_values, ...)`** -- run
   marching cubes on an explicit list of occupied cells and their sampled corner
   values, sharing vertices between neighbors so the result is watertight.
3. **`wp.geometry.sparse_marching_cubes(sdf, ...)`** -- chain the two: run the octree,
   sample the field at the surviving cells' corners, and extract.

This mirrors libigl's decomposition and, as suggested in review discussion, is
written "extraction first, cell-selection on top," so the extractor is usable
standalone (R5, R6).

**Lipschitz pruning bound.** A cell of width `h` centered at `c` can contain the
`t`-level set of a field `f` with Lipschitz constant `L` only if
`|f(c) - t| <= L * (sqrt(3)/2) * h`. The factor `sqrt(3)/2` is the half-diagonal
of the cube: the farthest any interior point can be from the center. If the
center value is farther than that from the isovalue, `f` cannot reach `t`
anywhere in the cell, so the whole subtree can be discarded. For a true SDF,
`L = 1`. Pruning at every level (not just the leaves) is what makes the octree
cheap: an entire coarse subtree far from the surface is culled in one test. This
is exactly the bound used by `igl::lipschitz_octree_prune`.

### Batched-callable-only contract

The public API accepts `sdf` only as a batched callable,
`evaluate(points: wp.array[wp.vec3]) -> wp.array[wp.float32]` --
not a bare single-point `@wp.func`. An earlier revision auto-wrapped a
`@wp.func` in a generated, cached `eval_sdf_kernel` (closing over the function
object) so a per-point signature could be passed directly. Review feedback
was that this convenience hid a kernel launch behind what looked like a
single-point call, and made it easy to reach for the (materially slower,
though still correct) fallback semantics without noticing. It also collapsed
two different implementer intents -- "evaluate on the GPU" vs. "evaluate
however you like, host round trip included" -- into one call shape that
looked identical either way.

Removing that path makes the two supported styles explicit at the call site:

- **All on-device.** Write a small `@wp.kernel` that evaluates the field over
  the batch (see `warp/examples/geometry/example_sparse_marching_cubes.py` and
  the `test_sparse_mc_mesh_minus_sphere_on_device` test, which composes a mesh
  query with an analytic sphere via CSG subtraction, `max(d_mesh, -d_sphere)`,
  in one kernel). Field evaluation then never leaves the GPU.
- **Host round trip, caller's choice.** A callable that wraps NumPy, PyTorch,
  or any other host library is equally valid -- `_make_evaluator` only
  requires the *returned* array to be a `wp.array[wp.float32]` back on
  the query points' device. This costs a device/host sync on every call (once
  per octree level, plus once for the corners), which is the caller's to pay
  knowingly rather than something the library defaults into. Exercised by
  `test_sparse_mc_numpy_evaluator`.

Passing a `wp.Function` (a bare `@wp.func`) now raises `TypeError` with a
worked example of how to batch it.

### Alternatives Considered

- **A dense hash grid / `wp.HashGrid` for corner de-duplication.** `wp.HashGrid`
  is built for spatial neighbor queries on float positions, not exact integer
  de-duplication. We instead pack corner subscripts into int64 codes and use the
  existing `radix_sort_pairs` + scan primitives, which are exact and already
  available.

- **Reusing native `igl`/OpenVDB.** VDB does support similar extraction, but
  pulling it into Warp's build and Python surface was deemed heavier than a pure
  Warp implementation, which also keeps the field evaluation on-device and
  differentiable-friendly.

- **Emitting an adaptive (octree) mesh.** Out of scope; the goal was a drop-in
  sparse analogue of dense marching cubes with identical output.

### Key Implementation Details

Module: `warp/_src/geometry/sparse_marching_cubes.py` (public re-exports in
`warp/geometry.py`).

**GPU octree construction** (`_build_lipschitz_octree`). Cells are stored as a
flat `wp.array[wp.vec3i]` of integer subscripts. Each level: evaluate the
field at cell centers, mark cells within the Lipschitz band, stream-compact the
survivors with `wp._src.utils.array_scan`, and subdivide each survivor into 8
children. The only host synchronizations are the per-level compaction counts
(`max_depth + 1` of them). The finest survivors are the leaf cells at resolution
`2^max_depth`.

**Corner de-duplication** (`_dedupe_corners`). Each leaf has 8 corners; adjacent
leaves share corners. We pack each corner's integer subscript into an int64 code
(relative to the minimum subscript, so arbitrary/negative/non-contiguous
subscripts are supported), `radix_sort_pairs` the `8N` codes, mark run
boundaries, and scan to assign a compact unique id per corner. This yields
`cell_corners` (`N x 8` indices into the unique corners) plus the unique corner
positions -- exactly the input a sparse marching cubes needs.

**Field evaluation.** The implicit function is evaluated only at the `O(R^2)`
unique corners (and, during pruning, at cell centers). For the octree-driven
path this is a single batched call; for the explicit-cells path the caller
supplies per-cell corner values directly.

**Sparse marching cubes core** (`_extract_from_dedup`). Reuses the *exact*
lookup tables of the dense `warp.geometry.IsoSurfaceMarchingCubes` (`MC_CASE_TO_TRI_RANGE`,
`MC_TRI_LOCAL_INDICES`, `MC_CUBE_CORNER_OFFSETS`) so the output matches the dense
extractor case for case: identical triangulation and vertex/triangle counts, with
vertex positions agreeing to floating-point tolerance (see the equivalence test
below). Vertices are de-duplicated by giving every crossed
edge a **canonical slot** `owner_corner_unique_id * 3 + axis`, where the owner is
the edge endpoint with the lower coordinate on that axis. Because both endpoints
of a cell edge are shared unique corners, all cells incident to an edge compute
the same slot and the same interpolated vertex is emitted exactly once. This is
the watertightness mechanism, and it is the part unique to the sparse method.
The extraction runs in the same count/scan/emit passes as dense marching cubes
(vertices, then faces).

**Benign races.** Marking active edges and recording each edge's upper endpoint
has multiple incident cells writing identical values to the same slot; the race
is deterministic. Confirmed on both CPU (multithreaded) and CUDA.

### Public API

```python
# Full pipeline: implicit function -> mesh. Parameterized exactly like
# warp.geometry.IsoSurfaceMarchingCubes.extract, so the two are interchangeable.
verts, indices = wp.geometry.sparse_marching_cubes(
    sdf,  # evaluate(points: wp.array[wp.vec3]) -> wp.array[wp.float32]
    nx,
    ny,
    nz,
    lower=None,
    upper=None,
    threshold=0.0,
    lipschitz_bound=1.0,
    device=None,
    return_stats=False,
)

# Stage 1: choose occupied cells (cubic root box; lower-level primitive, not
# required to mirror a dense grid).
cells, cell_width = wp.geometry.lipschitz_octree(sdf, origin, root_width, max_depth, ...)

# Stage 2: extract on an explicit cell set (e.g. marked voxels from a model).
verts, indices = wp.geometry.sparse_marching_cubes_from_cells(
    cells,  # (N, 3) int subscripts
    corner_values,  # (N, 8) field values, in IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS order
    origin,
    cell_width,
    threshold=0.0,
    device=None,
)
```

The `lipschitz_bound` parameter widens the retained band for fields that are
`L`-Lipschitz with `L > 1` (i.e. not unit-speed SDFs), trading work for the
bracketing guarantee.

### Matching the dense grid parameterization

Review feedback on the initial `(origin, root_width, max_depth)` signature of
`sparse_marching_cubes` asked that it accept the same
grid description as `IsoSurfaceMarchingCubes.extract` -- `nx, ny, nz` node
counts plus `lower`/`upper` -- so that calling either extractor with the
same arguments yields the same surface. This is nontrivial because the
octree needs a single cubic-ish root box subdivided by 2 on every axis,
while the requested grid can be anisotropic and need not have a
power-of-two cell count on any axis.

The reconciliation:

1. Compute `dx, dy, dz` (the per-axis cell size) exactly as
   `resolve_domain_bounds` does for the dense path, from
   `(nx, ny, nz)` and the corner bounds.
2. Let `ncells = (nx - 1, ny - 1, nz - 1)` and
   `max_depth = (max(ncells) - 1).bit_length()` -- the smallest depth such
   that `2**max_depth >= max(ncells)`.
3. Build the octree on an **anisotropic root box** with per-axis width
   `dx * 2**max_depth, dy * 2**max_depth, dz * 2**max_depth`, so the leaf
   cell size matches the dense cell size exactly on every axis. Any axis
   whose `ncells` is not itself a power of two gets a box that extends past
   `upper` on that axis (this can happen on every axis, including the
   longest one, unless its cell count is already a power of two).
4. The Lipschitz pruning bound generalizes from the cubic
   `L * (sqrt(3)/2) * h` to the box half-diagonal
   `L * 0.5 * ||(dx, dy, dz)||` at each depth, which reduces to the cubic
   formula when isotropic.
5. After the octree finishes, **cull leaf cells whose subscript is `>=
   ncells` on any axis** -- the cells that exist only because of the
   power-of-two padding -- before dedup/extraction, so the output is
   identical to a dense grid over exactly `[lower, upper]` at `nx, ny, nz`.

`max_depth` is fully derived and is not part of the public signature.
`lipschitz_octree` and `sparse_marching_cubes_from_cells` keep their
existing cubic/scalar-width signatures: they are documented as general
low-level primitives (R5/R6), not required to mirror a dense grid.

### Backward-mode autodiff

Dense `IsoSurfaceMarchingCubes.extract` already supports backward-mode
autodiff (its kernels default to `enable_backward=True`, `verts` is
allocated with `requires_grad=field.requires_grad`, and the interpolation is
`wp.lerp`). Every kernel in `sparse_marching_cubes.py` originally had
`enable_backward=False`. Review feedback (R8) asked for
`sparse_marching_cubes_from_cells` to support backward-mode autodiff
w.r.t. `corner_values`, while deliberately *not* differentiating
`lipschitz_octree` -- cell selection is a discrete,
threshold-based search, not a smooth function of the field.

**The scatter-write problem.** `sparse_marching_cubes_from_cells` takes
`corner_values` as one entry per `(cell, corner)` pair, redundant at shared
corners by design (the "benign race" the old `_scatter_corner_values_kernel`
relied on: every cell touching a corner writes the same value to the same
slot). An empirical check of Warp's autodiff on exactly this pattern (`wp.Tape`
over a 2-writer, 1-slot scatter) showed the backward pass gives the *entire*
downstream gradient to whichever write happened to run last in program order,
and *zero* to the others -- correct per forward semantics, but the winning
write is not guaranteed deterministic across launches on the GPU, so enabling
backward on it naively would make gradients correct-looking but silently
non-reproducible.

**Fix: scatter to gather.** `_dedupe_corners` already computes, via
`radix_sort_pairs` + `is_first`/`unique_scan`, the "first occurrence in sorted
order" structure needed to pick one canonical `(cell, corner)` source per
unique corner. A new opt-in step (`compute_unique_source=True`, used only by
`sparse_marching_cubes_from_cells` -- the octree-driven path never scatters,
since it evaluates `field` directly on already-unique corner positions) records
`unique_source[unique_id]`, the flat `cell*8+corner` index of that canonical
source. `_gather_corner_values_kernel` then does a pure 1:1 gather --
`unique_values[i] = per_cell_values_flat[unique_source[i]]` -- with no
aliasing in either direction, so it is trivially and deterministically
differentiable. This is also the mathematically correct choice, not just a
convenient one: when duplicate corner entries are independent evaluations of
the same underlying point function, only one gradient path should be counted,
matching how dense marching cubes evaluates each grid node exactly once
(rather than summing or splitting gradient across up to 8 redundant cells).

**Which kernels are backward-enabled.** Warp warns whenever a kernel with
`enable_backward=False` is recorded on a tape whose `.backward()` gets
called, regardless of whether that specific kernel's own arrays require
grad -- so every kernel in `sparse_marching_cubes_from_cells`'s call graph
(corner de-duplication plus `_extract_from_dedup`) has backward enabled, even
where the actual gradient computation is a no-op (e.g. `_emit_faces_kernel`'s
discrete triangle indices), mirroring the precedent dense marching cubes
already set for its own faces kernel. Octree-*construction* kernels
(`_compute_cell_centers_kernel`, `_mark_active_cells_kernel`,
`_subdivide_cells_kernel`, `_compact_cells_kernel`, `_cull_out_of_bounds_kernel`
-- used only by `lipschitz_octree`/`_build_lipschitz_octree`)
stay `enable_backward=False`: they are never reached by
`sparse_marching_cubes_from_cells`, and by design should not be
differentiated. `sparse_marching_cubes` reuses the same `_extract_from_dedup`
core directly on already-deduplicated values, so it inherits differentiability
w.r.t. whatever `field` returns, as a natural side effect, with zero changes
to the octree machinery. Recording it on a `wp.Tape()` would otherwise print
benign-but-alarming "may produce incorrect gradients" warnings for every
octree-construction kernel launch, since Warp warns on any `enable_backward=False`
kernel recorded on a tape whose `.backward()` runs, regardless of whether
that launch's own arrays require grad. Rather than accept that noise (or ask
callers to split their call across two tape scopes to avoid it), every
octree-construction launch (`_compute_cell_centers_kernel`,
`_mark_active_cells_kernel`, `_subdivide_cells_kernel`, `_compact_cells_kernel`,
`_cull_out_of_bounds_kernel`) uses `wp.launch(..., record_tape=False)`, the
same mechanism Warp's own `_launch_adj_copy_add()` and `render_opengl.py`
use for launches that must never appear on a tape. They are simply never
recorded, so no warning fires and there is nothing for callers to work around.

**Caller responsibility.** As with any Warp kernel output that should carry
gradient, the caller's `field` evaluator (for `sparse_marching_cubes`) or
`corner_values` array (for `sparse_marching_cubes_from_cells`) must itself
be allocated with `requires_grad=True` -- this can't be inferred on the
caller's behalf, and forgetting it fails silently (zero gradient, no error).

**Performance.** `_dedupe_corners`'s `unique_source` computation is opt-in and
skipped entirely by the octree-driven hot path. The scatter-to-gather
rework and the `enable_backward` flips on already-cheap discrete-output
kernels are not expected to change `enable_backward=False` (forward-only)
performance; see `warp/examples/benchmarks/benchmark_sparse_marching_cubes.py`'s
dedicated `sparse_marching_cubes_from_cells` timing section for the ongoing
check.

## Testing Strategy

Tests live in `warp/tests/geometry/test_sparse_marching_cubes.py` and run across
`get_test_devices()` (CPU + CUDA).

- **Correctness vs. ground truth** -- vertices of a sphere SDF lie on the sphere;
  surface area matches the analytic value.
- **Equivalence to dense** -- for sphere and torus SDFs at several depths, the
  sparse mesh matches `wp.geometry.IsoSurfaceMarchingCubes` on the equivalent dense grid: identical
  vertex/triangle counts and a tolerance-based two-sided Hausdorff match. (Exact
  position equality is intentionally *not* asserted, because the batched-evaluator
  kernel and the dense field kernel compile the same arithmetic with slightly
  different floating-point contraction.) This equivalence is the fairness
  anchor for the benchmark's speedup claims.
- **Watertightness / manifoldness** -- for closed SDFs strictly inside the
  domain, the mesh has zero boundary edges (a boundary edge is a hole) and zero
  edges shared by more than two faces, and the Euler characteristic recovers the
  genus (sphere -> 2, torus -> 0). This directly guards the sparse-specific
  vertex de-duplication.
- **Octree bracketing guarantee** -- every dense grid cell that contains a sign
  change is present in the octree leaves (a superset check), which is the
  invariant that prevents holes.
- **Explicit-cells path** -- `sparse_marching_cubes_from_cells` fed the octree's
  own cells plus independently sampled corner values reproduces the octree-driven
  result, and is invariant to a large negative subscript shift with a
  compensating origin (exercises the offset-relative corner packing).
- **Interfaces and edge cases** -- an all-on-device Warp evaluator vs. a NumPy
  round-tripping one (`test_sparse_mc_numpy_evaluator`), a mesh SDF composed
  with an analytic sphere via CSG subtraction entirely on-device
  (`test_sparse_mc_mesh_minus_sphere_on_device`), a mesh-based SDF via
  `wp.mesh_query_point_sign_normal`, non-zero isovalue, empty output, and
  argument validation -- including that a bare `@wp.func` is rejected with a
  `TypeError`.
- **Anisotropic grid matches dense** -- for unequal, non-power-of-two `nx, ny,
  nz` and anisotropic corner bounds, the sparse mesh matches
  `IsoSurfaceMarchingCubes` on the same grid, exercising the padding + cull
  reconciliation described above.
- **No padding for exact power-of-two grids (regression guard)** -- for
  `nx = ny = nz = 2**depth + 1`, `culled_cells` is exactly zero and
  `leaf_cells`/`field_evaluations` exactly match fixed values recorded for that
  depth. This is a deterministic stand-in for a performance-regression test
  (unit tests must not assert on timing, per `AGENTS.md`): if the generalized,
  anisotropic-capable code path ever added overhead for the common isotropic
  case, these exact counts would change.
- **Backward-mode autodiff** -- `test_sparse_mc_from_cells_differentiable`
  mirrors `test_marching_cubes_differentiable` in `test_marching_cubes.py`
  exactly (`d(surface area)/d(sphere radius)` via `wp.Tape`, checked against
  the analytic value), but with `corner_values` for every cell of a dense
  grid (no octree, so cell-selection effects can't confound the extraction
  math). `test_sparse_mc_gradient_matches_dense` runs the identical grid
  through both `sparse_marching_cubes_from_cells` and
  `IsoSurfaceMarchingCubes.extract` and asserts the two gradients agree --
  possible only because of the earlier nx/ny/nz-matching work.
  `test_sparse_mc_gradient_deterministic` checks two runs with identical
  inputs match closely (not exactly zero -- ordinary GPU floating-point
  non-associativity is expected -- but far tighter than the arbitrary-winner
  divergence the old scatter would have risked).
  `test_sparse_mc_via_lipschitz_pruning_differentiable` confirms the
  side-effect differentiability of the octree-driven entry point.

Beyond unit tests, `warp/examples/benchmarks/benchmark_sparse_marching_cubes.py`
compares sparse vs. dense on both an analytic SDF and the bunny mesh SDF,
asserting equal triangle counts at each depth before reporting timings, so the
reported speedups (roughly an order of magnitude by depth 9, with the dense grid
exhausting memory beyond that) are honest. It also has a dedicated,
forward-only (no `wp.Tape`) timing section for
`sparse_marching_cubes_from_cells` in isolation from cell selection, as the
ongoing check that adding backward-mode support did not slow down the
`enable_backward=False` route.
