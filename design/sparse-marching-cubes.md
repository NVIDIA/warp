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
signed distance functions (SDFs), neural implicits (NeRF-style occupancy/SDF
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
| R2  | Run the whole pipeline (cell selection, field evaluation, extraction) on the GPU | Must | Field eval must not round-trip to host |
| R3  | Accept the implicit function as a `@wp.func` or a batched Python callable | Must | Callable form covers mesh queries, neural implicits |
| R4  | Produce a watertight, manifold mesh matching dense marching cubes at equal resolution | Must | Correctness / fair comparison |
| R5  | Expose the extraction stage on an explicit `(cells, corner_values)` list | Should | Vision/genAI "marked voxels" workflow |
| R6  | Expose the cell-selection stage (`lipschitz_octree`) on its own | Should | Custom extractors, visualization |
| R7  | Asymptotically beat the dense grid in time and evaluations as resolution grows | Should | The performance justification |

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

### Alternatives Considered

- **Passing a `wp.Function` as a kernel launch argument.** Warp resolves function
  targets at codegen/hash time, not as runtime values, so a top-level launched
  kernel cannot receive a `@wp.func` through its `inputs` list. Instead, when the
  caller passes a `@wp.func`, we generate (and cache, keyed by the function
  object) one tiny `eval_sdf_kernel` that closes over it. Everything else is
  generic and operates on plain arrays. The public contract is the more general
  *batched callable* `points -> values`; the `@wp.func` path is sugar over it.

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

Module: `warp/_src/sparse_marching_cubes.py` (public re-exports in
`warp/geometry.py`).

**GPU octree construction** (`_build_lipschitz_octree`). Cells are stored as a
flat `wp.array(dtype=wp.vec3i)` of integer subscripts. Each level: evaluate the
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
# Full pipeline: implicit function -> mesh.
verts, indices = wp.geometry.sparse_marching_cubes(
    sdf,                 # @wp.func (p: wp.vec3) -> float, or callable(points)->values
    origin, root_width, max_depth,
    threshold=0.0, lipschitz_bound=1.0, device=None, return_stats=False,
)

# Stage 1: choose occupied cells.
cell_origins, cell_width = wp.geometry.lipschitz_octree(sdf, origin, root_width, max_depth, ...)

# Stage 2: extract on an explicit cell set (e.g. marked voxels from a model).
verts, indices = wp.geometry.sparse_marching_cubes_from_cells(
    cells,               # (N, 3) int subscripts
    corner_values,       # (N, 8) field values, in IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS order
    origin, cell_width, threshold=0.0, device=None,
)
```

The `lipschitz_bound` parameter widens the retained band for fields that are
`L`-Lipschitz with `L > 1` (i.e. not unit-speed SDFs), trading work for the
bracketing guarantee.

## Testing Strategy

Tests live in `warp/tests/geometry/test_sparse_marching_cubes.py` and run across
`get_test_devices()` (CPU + CUDA).

- **Correctness vs. ground truth** -- vertices of a sphere SDF lie on the sphere;
  surface area matches the analytic value.
- **Equivalence to dense** -- for sphere and torus SDFs at several depths, the
  sparse mesh matches `wp.geometry.IsoSurfaceMarchingCubes` on the equivalent dense grid: identical
  vertex/triangle counts and a tolerance-based two-sided Hausdorff match. (Exact
  position equality is intentionally *not* asserted, because the `@wp.func` and
  the dense field kernel compile the same arithmetic with slightly different
  floating-point contraction.) This equivalence is the fairness anchor for the
  benchmark's speedup claims.
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
- **Interfaces and edge cases** -- `@wp.func` vs. batched-callable equivalence, a
  mesh-based SDF via `wp.mesh_query_point_sign_normal`, non-zero isovalue, empty
  output, and argument validation.

Beyond unit tests, `warp/examples/benchmarks/benchmark_sparse_marching_cubes.py`
compares sparse vs. dense on both an analytic SDF and the bunny mesh SDF,
asserting equal triangle counts at each depth before reporting timings, so the
reported speedups (roughly an order of magnitude by depth 9, with the dense grid
exhausting memory beyond that) are honest.
