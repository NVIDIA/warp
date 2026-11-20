# Constrained Basis Spaces and Periodic Boundary Conditions in Warp FEM

**Status**: Implemented (R1–R9); R10 and R11 remain follow-ups

**Issue**: None yet

Builds on [hanging-node-elimination.md](hanging-node-elimination.md), which
introduced `ConstrainedBasisSpace` for conforming spaces on
`AdaptiveNanogrid`.

## Motivation

`warp.fem` has no way to impose periodic boundary conditions. Examples that
need periodicity work around it: `example_taylor_green.py` wraps advected
positions by hand and solves its linear systems with Dirichlet conditions, and
`example_kelvin_helmholtz.py` looks up the opposite boundary explicitly.
Periodic conditions are standard in homogenization (computing effective
properties of a periodic material from one unit cell), turbulence and flow
benchmarks, crystal and lattice simulations, and any problem posed on a torus.

The conforming adaptive spaces already contain the machinery periodicity needs.
A `ConstrainedBasisSpace` is defined by a sparse prolongation `P` that expresses
every node of an ordinary ("raw") basis in terms of a smaller set of independent
degrees of freedom, `u_raw = P u`. Forms assemble directly on the independent
degrees of freedom, fields evaluate through `P`, and Dirichlet projectors and
nodal values are restricted through it. A periodic condition is just another
`P`: each node on a "high" face takes the value of its partner on the opposite
"low" face.

Before this design, `P` could only be produced internally, for Q1 hanging
nodes on `AdaptiveNanogrid`. This design lets users supply their own `P`,
generalizes the space construction and the hanging-node producer to any
interpolatory basis and element type, and provides periodic constraints as the
first built-in producer besides hanging nodes.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | ----------- | -------- | ----- |
| R1  | Build a constrained basis from any raw basis and a user-supplied scalar prolongation `P`. | Must | `make_constrained_basis_space` |
| R2  | Build periodic prolongations for one, two or three translation directions, including edge and corner nodes tied by several translations. | Must | `periodic_prolongation` |
| R3  | Support grids (`Grid2D`, `Grid3D`), simplex meshes (`Trimesh`, `Tetmesh`), quad/hex meshes and nanogrids, at any polynomial degree supported by the raw basis. | Must | |
| R4  | Combine periodicity with hanging-node elimination when refinement matches on paired faces. | Must | By composing prolongations |
| R5  | Keep `make_conforming_basis_space` behavior and performance unchanged. | Must | Becomes a wrapper |
| R6  | Report inconsistent constraints and cycles with clear errors. | Must | Unmatched periodic nodes stay independent |
| R7  | Construction stays on the device, with readbacks limited to sizes and error status. | Must | Same rule as the conforming builder |
| R8  | Accept a constrained basis as the base of another constraint and flatten it to the raw basis. | Should | Avoids nested spaces |
| R9  | Ship a periodic example that is part of the example test suite. | Must | Homogenization |
| R10 | Degrees of freedom without a raw-node anchor (mean value, rigid modes). | Could | Follow-up |
| R11 | Periodic seams with nonmatching refinement on the two sides. | Could | Follow-up |

**Non-goals**:

- Constraints that mix vector components, such as slip on a slanted wall or
  rigid links with rotations. A basis is scalar and applies the same `P` to every
  component; mixing components needs a constrained function space with block `P`.
- Jumps and DG terms across a periodic seam. The seam stays a geometric boundary.
- Wrapping positions in lookups, interpolation at arbitrary points, or particle
  quadratures that cross the seam.
- Rebuilding a constrained space in place after the geometry changes.

## Design

### Approach

Construction is split into three stages, each usable on its own:

```text
constraint producers  ->  shared resolution  ->  P -> space builder
(hanging nodes,           (chains, canonical      (independent numbering, cell
 periodic pairs, ...)      choice, consistency)    and side node lists, anchors)
```

1. A **constraint producer** emits candidate constraints on the nodes of a base
   basis: "node `a` equals `sum_k w_k * node(m_k)`". The hanging-node builder
   already emits these from side-facet incidence (up to four masters each). The
   periodic producer emits "node `x` equals node `x - T`" with weight one.
2. The **shared resolution** stage already exists in the GPU conforming builder:
   it picks one canonical candidate per constrained node (smallest key), expands
   chains of constraints round by round until every row refers only to
   unconstrained nodes, and checks that all other candidates for a node agree
   with the resolved row. Its output is `P` in CSR form. Only the candidate
   format needs to be shared; it already handles chains of any depth.
3. The **P -> space builder** turns a final `P` into a `ConstrainedBasisSpace`:
   independent numbering, per-cell node lists and coefficients, side node
   lists, and anchors for node samples. This is the second half of the GPU
   pipeline, independent of the element type and degree.

### Public API

```python
fem.make_constrained_basis_space(
    base_basis: BasisSpace,
    prolongation: BsrMatrix,
    independent_raw_nodes: wp.array | None = None,
    max_nodes_per_element: int = 256,
) -> ConstrainedBasisSpace

fem.periodic_prolongation(
    basis: BasisSpace,
    translations: Sequence[vec],
    tolerance: float | None = None,
) -> BsrMatrix

fem.hanging_node_prolongation(raw_basis: BasisSpace) -> BsrMatrix

fem.make_conforming_basis_space(raw_basis)  # unchanged; same stages as the two functions above
```

- `make_constrained_basis_space` takes a scalar BSR matrix with one row per node
  of `base_basis` and one column per independent degree of freedom. If
  `base_basis` is itself a `ConstrainedBasisSpace`, the prolongation is composed
  with the base's, `P_raw = P_base @ P`, and the result is built on the base's
  raw basis, so constrained spaces never nest.
- `periodic_prolongation` matches nodes of any basis that provides
  `node_positions()`, including a `ConstrainedBasisSpace`, so periodicity can be
  layered on top of hanging-node elimination.
- `hanging_node_prolongation` exposes the existing hanging-node `P` on its own.
- `make_conforming_basis_space` runs the hanging-node producer and the
  P -> space builder directly on the resolved CSR arrays. It skips the
  `BsrMatrix` round trip and the column validation, whose result is known, and
  applies no `max_nodes_per_element` limit, so its behavior is unchanged.
- When `base_basis` is constrained, `independent_raw_nodes` holds node indices
  of `base_basis`; they are mapped to raw nodes through its anchors.

Typical uses:

```python
# Periodic in x and y on a regular grid, quadratic elements
geo = fem.Grid2D(res=wp.vec2i(32), bounds_lo=wp.vec2(0.0), bounds_hi=wp.vec2(1.0))
raw = fem.make_polynomial_basis_space(geo, degree=2)
P = fem.periodic_prolongation(raw, translations=[wp.vec2(1.0, 0.0), wp.vec2(0.0, 1.0)])
basis = fem.make_constrained_basis_space(raw, P)
space = fem.make_collocated_function_space(basis)

# Periodic adaptive grid with matching refinement on paired faces
conforming = fem.make_conforming_basis_space(fem.make_polynomial_basis_space(adaptive_geo, degree=1))
P = fem.periodic_prolongation(conforming, translations=[wp.vec3(L, 0.0, 0.0)])
basis = fem.make_constrained_basis_space(conforming, P)  # composed with the hanging-node P
```

### Prolongation contract

- **Scalar blocks.** `P` has `1x1` blocks of a floating-point type. Vector
  function spaces apply it per component through `make_collocated_function_space`.
- **Anchored columns.** Each independent degree of freedom `j` has an anchor raw
  node `r_j` whose row of `P` is exactly the unit vector `e_j`. Anchors give
  `node_positions()`, `at_node()` samples, interpolation into the space, and
  Dirichlet projector reduction their meaning. They are detected on the device from
  identity rows; when several identity rows refer to the same column, the
  smallest raw node index is the anchor unless `independent_raw_nodes` names
  them explicitly. A column without an identity row raises a `ValueError` until
  unanchored degrees of freedom are supported (R10). An identity row is one
  stored nonzero entry equal to exactly one.
- **No other structural requirement.** Rows need not sum to one (general
  multi-point constraints do not), and columns may be referenced from anywhere
  in the mesh. Empty columns and rows referring to out-of-range columns raise.
  Explicit zeros are dropped and repeated columns in a row are summed, so any
  compact or padded scalar BSR matrix is accepted.
- **Locality.** A cell's node list is the union of the columns referenced by its
  raw nodes, and the largest list becomes `MAX_NODES_PER_ELEMENT`, which is
  compiled into assembly kernels. Local constraints keep it small (hanging
  nodes: at most a few tens; periodicity: unchanged). The builder reports the
  maximum in the space name and raises if it exceeds a limit
  (`max_nodes_per_element`, default 256) so that very wide lists are deliberate.
  The limit applies to cell lists; side lists are at most the union of the two
  adjacent cell lists.

### Periodic constraints

For each translation `T`, every node at position `x` that has a partner node
at `x - T` receives the candidate `node(x) = 1.0 * node(x - T)`. In a cell that
tiles space with the given translations, only the nodes on the face that `T`
maps onto have such a partner, whatever the angle between `T` and the faces, so
oblique cells such as parallelograms work. The producer only pairs nodes: it
does not check that paired faces carry matching nodes, and nodes without a
partner stay independent. Checking the faces would need to know which face
each translation maps, which a position-based helper cannot tell in general
(oblique cells with fewer translations than dimensions, non-planar periodic
faces). On geometries with several environments, which share local
coordinates, nodes only match nodes of the same environment.

A first version only constrained the nodes on the extreme plane. That is
correct for boxes but silently wrong for oblique translations, where the plane
reduces to a single vertex.

- **Matching** uses `wp.HashGrid` on node positions with a radius of
  `tolerance`, so it works on any mesh whose paired faces carry matching nodes.
  The default tolerance is `1e-6` times the bounding-box diagonal, in the
  geometry's scalar type. The hash grid uses the geometry's scalar type, and
  its cell size is `max(2 * tolerance, 1e-5 * diagonal)`. The node bounding box
  is computed on the device and read back (`2 * dimension` scalars), because
  the default tolerance and the hash grid cell size are host parameters; this
  is the only readback besides sizes and error status. Tolerances must be
  positive and smaller than half of every translation length.
- **Edges and corners** are tied by several translations: a 2D corner
  `x + T1 + T2` has candidates toward `x + T2` and `x + T1`, which are themselves
  constrained toward `x`. The shared resolution stage expands these chains, and
  its consistency check confirms that all routes lead to the same master.
- **Masters** are the nodes on the low faces: the columns of the periodic `P`
  are numbered by their unconstrained low copies, in node order. Every copy has
  an identity row, so `make_constrained_basis_space` anchors each class at its
  smallest node index. On grids and meshes numbered in increasing coordinate
  order, this is the lowest-coordinate copy; otherwise pass
  `independent_raw_nodes` to choose the anchors.
- **Cycles.** Giving both `T` and `-T` makes every paired node constrained to
  its partner; resolution then reports cyclic constraints.
- The producer only needs node positions, so it works on raw bases, constrained
  bases (through their anchors), and every geometry.

### Behavior on periodic faces

The geometry does not know that faces are paired; they remain boundary sides.

- **Continuous Galerkin forms work unchanged.** With periodic trial and test
  functions, the boundary flux contributions of paired faces cancel, so users
  apply no boundary condition on periodic faces. Boundary integrals over those
  faces should be avoided; they are evaluated one face at a time.
- **No jumps across the seam.** `fem.jump` and DG terms see no outer cell on a
  periodic face. This needs geometric side pairing and is out of scope.
- **Pure natural conditions leave a constant null space** for scalar diffusion
  (and rigid translations for elasticity). Users pin one independent node with a
  diagonal projector, or rely on a compatible right-hand side with CG and remove
  the mean afterwards. A mean-value constraint is an unanchored degree of
  freedom (R10).
- **Positions do not wrap.** Lookups and semi-Lagrangian advection still wrap
  positions themselves.

### Sampling nodes with `at_node()`

`at_node()` must give a sample at the node of a local slot, for nodal reads
such as `U(at_node(U, s, k))`, APIC-style offsets `domain(at_node(v, s)) -
domain(s)`, and forms that evaluate one function at another's node, such as
`u(at_node(v, s)) * v(s)`.

- **Stay local when possible.** At construction, each cell slot records the
  coordinates of a raw node of the cell that carries the independent node: its
  anchor if the cell holds it, otherwise any raw node whose row of `P` is the
  node's unit vector, such as a periodic copy. Sides record the same for raw
  nodes lying on the side. `node_sample()` returns the current element with
  those coordinates. Periodic spaces then never leave the current cell, and
  offsets point to the nearby copy rather than one period away.
- **Otherwise move to an owner element.** A slot without a copy, such as a
  coarse master seen from a fine cell, falls back to the first cell (or side)
  holding the anchor. This keeps the hanging-node behavior.
- **Other fields do not follow the move.** The other test or trial slot
  refers to the original element in that field's own topology, which
  `at_node()` does not know, so when the sample changes element the other
  descriptor becomes `NULL_DOF_INDEX` and that field evaluates to zero.
  Evaluating a test function at a trial node (or the reverse) is therefore
  only meaningful while the sample stays in its element, under generic
  assembly. Discrete fields' `at_node()` clears both descriptors: test and
  trial functions are not defined at another field's nodes (GH-2116).
- **No negative reads.** A sample without a degree of freedom (`NULL_DOF_INDEX`,
  for instance from `lookup()`) evaluates test and trial functions to zero
  (GH-2116): evaluators zero the basis weight of the null slot, and the
  dispatch assembly path maps such descriptors to an unused Taylor slot.

Traces list every node of both cells (see below). On a side slot whose node
does not lie on the side, `at_node()` moves to the owner side of the node, or
returns `OUTSIDE` coordinates when the anchor lies on no side, as on
unconstrained traces.

### Generic builder and hanging-node producer

Neither the builder nor the hanging-node producer depends on the element type
or the polynomial degree. They only use the generic geometry and basis
interfaces, and the geometry-specific knowledge is isolated in
`Geometry.side_facet_incidence()`, which tells which local facet of each cell
a side covers and which sides cover each facet.

1. **Raw slot loops.** Constrained weights and gradients sum over raw slots
   up to `raw_basis.topology.MAX_NODES_PER_ELEMENT`.
2. **Anchor coordinates.** Anchors are located with the raw basis's
   `node_coords_in_element` in their owner cell, and their side coordinates
   with `side_from_cell_coords`, instead of inverting geometry maps.
3. **Side node lists.** A side lists the independent nodes of all raw nodes
   of both adjacent cells, mapped through `P` and merged so that a node shared
   by both cells appears once. Listing only the nodes lying on the side would
   be narrower, but a function that vanishes on the side can still have a
   nonzero normal derivative there, such as the function of a Q1 node on the
   opposite face, so side forms using `fem.grad()` would miss entries.
4. **Hanging nodes.** Facets of neighboring cells are nested, so a side's finer
   neighbor is the one whose facet is covered only by sides leading to the
   other cell; sides of equal facets are conforming. Each node of the finer
   cell lying on such a side takes the value of the coarser cell's
   interpolant there, evaluated with the raw basis functions of the coarser
   cell. This requires an interpolatory basis (each function is one at its own
   node and zero at the others), which shape functions declare with
   `ShapeFunction.INTERPOLATORY`; weights are kept above a small multiple of
   the machine epsilon of the basis scalar type.
   Chains across several levels are resolved by the shared resolution stage.

### Combining with hanging nodes

Periodicity and hanging-node elimination compose through `P_raw = P_hanging @
P_periodic`, with the periodic producer matching the conforming space's anchors.
This requires matching refinement on paired faces: if one face is refined and
its partner is not, the fine anchors have no partner and stay independent, so
the seam is not periodic there. Nonmatching periodic seams (R11) need the hanging-node producer to treat
the seam as an interior interface, which requires pairing facets in the
side-facet incidence; the shared resolution stage would then resolve hanging
and periodic candidates together.

### Alternatives Considered

- **Periodic geometry.** Grids could wrap node indices and treat seam cells as
  neighbors. This gives jumps across the seam and wrapping lookups, but it must
  be implemented separately for each geometry and does not combine with
  hanging-node constraints. It remains a good follow-up for structured grids
  that need DG across the seam.
- **Weak enforcement** (penalty or Lagrange multipliers tying paired nodes).
  Penalties are inexact and hurt conditioning; multipliers produce saddle-point
  systems. Elimination keeps symmetric positive definite systems smaller.
- **Nested constrained spaces.** Wrapping a constrained basis in another
  constrained basis would double the indirection in every kernel. Composing
  prolongations keeps one level.

### Key Implementation Details

- **Candidate format.** The hanging-node builder stores candidates as
  `(raw node, masters[4], weights[4], key)` with a canonical candidate per node
  chosen by the smallest key. Share this format with a configurable master
  capacity; periodic candidates use one master. The resolution round bound
  becomes "level count plus number of translations plus one", and the existing
  "continue while progress is made" fallback keeps arbitrary producers safe.
- **User prolongations skip resolution.** A user `P` is already final. The
  builder validates it (shape, scalar blocks, column range, anchors) and feeds
  its rows directly into the P -> space stage.
- **Composition** uses `bsr_mm` once at construction.
- **Readbacks.** Sizes (column count, maximum cell and side list lengths) and
  error status only, as in the current builder, plus the periodic node bounding
  box. Splitting resolution from the P -> space builder adds one readback to
  the conforming construction (the prolongation size before the cell and side
  counts); on a 433k-cell graded grid construction still takes about 8 ms,
  against 9 ms before the split.
- **Files.** The producers and the shared resolution move to a new
  `warp/_src/fem/space/constraints.py`; the P -> space builder stays in
  `constrained_basis_space.py`. Public names are exported from `warp.fem` and
  listed in the API reference.

## Example: periodic homogenization

`warp/examples/fem/example_periodic_homogenization.py` computes the effective
conductivity tensor of a periodic composite from one unit cell. For each
direction `e_i`, it solves for a periodic corrector `w_i` with

```text
integral_Y k(x) (grad w_i + e_i) . grad v dx = 0    for all periodic v,
```

then forms `K_eff[i, j] = 1/|Y| integral_Y k (e_i + grad w_i) . (e_j + grad w_j) dx`.
The material is a matrix with a circular (2D) or spherical (3D) inclusion of a
different conductivity, or a laminate. The laminate has an exact answer: the
arithmetic mean of the layer conductivities along the layers and the harmonic
mean across them, which the example test checks. Options select the geometry
(grid, triangle mesh, adaptive nanogrid refined around the inclusion), the
polynomial degree, the resolution and the conductivity contrast. One
independent node is pinned to remove the constant null space. The example
renders a corrector field and prints the tensor.

On the grid, triangle and adaptive meshes the inclusion boundary cuts through
cells, so the step in conductivity is sampled at quadrature points and the
tensor converges at first order whatever the element degree (about 1% error at
h = 1/128 for contrast 10 and radius 0.25). The `fitted` mesh shows that the
constrained spaces work on curved geometry: a triangulated grid is deformed
with a quadratic position field (`DeformedGeometry`) so that a centered block of
cells covers the inclusion and the cell boundary stays fixed, leaving periodic
faces unchanged. Triangles are needed because a quad grid fitted to a smooth
closed curve flattens the four corner cells of the block to a 180 degree angle,
which limits convergence; splitting cells along diagonals that bisect those
corners avoids it. Measured against Rayleigh's multipole formula for a square
array of cylinders (Perrins, McKenzie and McPhedran, 1979):

| Relative error | h = 1/8 | 1/16 | 1/32 | 1/64 | Order |
| --- | --- | --- | --- | --- | --- |
| Grid, P2 | 3.4e-2 | 1.4e-2 | 8.1e-3 | 4.3e-3 | ~1 |
| Fitted quads, P2 | 5.7e-4 | 1.2e-4 | 1.8e-5 | 4.3e-6 | ~2.3 |
| Fitted triangles, P1 | 3.9e-2 | 1.1e-2 | 3.4e-3 | 8.9e-4 | ~1.9 |
| Fitted triangles, P2 | 6.9e-4 | 6.4e-5 | 4.1e-6 | 2.7e-7 | ~3.9 |

The example test checks the fitted P2 result at h = 1/16 against the Rayleigh
value to a relative tolerance of 1e-4.

## Testing Strategy

- **Hanging nodes of any degree.** Q2 and Q3 on a two-cell grid and on a
  three-level grid with missing cells: prolongation rows sum to one,
  polynomials of the basis degree are reproduced at every raw node, fields do
  not jump across sides, and assembly on cells and boundary sides, and of a
  side form using normal derivatives of both traces, matches `P^T A P`. Discontinuous bases and geometries without
  side-facet incidence are rejected.
- **Periodic prolongation.** On `Grid2D` and `Grid3D` (degrees 1 and 2) and
  `Trimesh`: every high node maps to the right low node; corner and edge nodes
  resolve to one master; independent node count equals the raw count minus the
  high nodes; unmatched nodes (perturbed mesh) stay independent; errors on bad
  tolerance; environments are matched separately.
- **Periodic solve.** Poisson with a manufactured periodic solution
  `sin(2 pi x) sin(2 pi y)` on Q1, Q2 and P2 triangles: error decreases at the
  expected rate; assembling with periodic fields matches a dense reference
  `P^T K P`.
- **User prolongation.** Identity `P` reproduces the raw space; a hand-built `P`
  tying two nodes matches the dense reference; errors for non-scalar blocks,
  wrong shapes, empty columns, missing anchors, and exceeding
  `max_nodes_per_element`.
- **Composition.** Periodic on top of `make_conforming_basis_space` on a
  two-level `AdaptiveNanogrid` with matching refinement: trace continuity across
  interfaces, and periodic values equal across the seam; error on nonmatching
  refinement.
- **Conforming regression.** `TestFemConstrainedSpace` unchanged, and
  `make_conforming_basis_space` construction time unchanged on a large grid.
- **Example.** Registered in `test_fem_examples.py`; the laminate case checks the
  exact effective tensor.
- Device coverage: CUDA for `AdaptiveNanogrid` cases, CPU and CUDA otherwise.
