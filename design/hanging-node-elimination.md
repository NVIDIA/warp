# Conforming Adaptive Spaces in Warp FEM

**Status**: Implemented for continuous Lagrange spaces of any degree

**Issue**: None yet

User-supplied prolongations and periodic boundary conditions, which build on the
constrained spaces described here, are designed in
[constrained-basis-spaces.md](constrained-basis-spaces.md). Since then, the
construction runs in three stages: the hanging-node producer
(`hanging_node_prolongation`), shared constraint resolution, and a builder
that turns any final prolongation into a `ConstrainedBasisSpace`
(`make_constrained_basis_space`). `make_conforming_basis_space` chains them
with unchanged outputs. The producer and the builder were then generalized from
Q1 to any interpolatory basis, with the geometry-specific knowledge isolated in
`Geometry.side_facet_incidence()`; see "Generic builder and hanging-node
producer" in that document. The Q1 details below record the first version.

## Motivation

`AdaptiveNanogrid` supports cells at different resolutions. Its current polynomial
spaces share degrees of freedom at matching mesh entities, but T-junctions leave
the resulting field discontinuous across coarse-to-fine interfaces. Different
polynomial degrees on adjacent cells create the same trace-continuity problem,
even when their geometric faces match. The FEM guide currently recommends DG or
similar methods for coarse-to-fine interfaces. Users should be able to request
a conforming nodal space and use ordinary forms and fields for h and p refinement.

## Current implementation scope

The implementation provides a static constrained basis on `AdaptiveNanogrid`
for continuous Lagrange spaces of any degree (first Q1 only), explicit construction with prebuilt side node lists,
independent node enumeration and `at_node()` anchors, generic and dispatch
assembly, and full-domain interpolation. It reduces raw Dirichlet projectors
to independent nodes and rejects nodal assembly on the constrained basis.
The constrained basis is built entirely on the geometry device: footprint
validation, per-side constraints, chained-constraint resolution, independent
numbering, the prolongation rows, and the cell and side node lists run as
kernels, with the host reading back only error status and array sizes.
`ConstrainedBasisSpace.prolongation_matrix()` exposes the raw-from-independent
prolongation `P` as a BSR matrix (scalar blocks, or `weight * identity` blocks
for vector degrees of freedom), built directly from the stored CSR arrays.
`ConstrainedBasisSpace.reduce_dirichlet_projector()` restricts a raw nodal
projector to the independent nodes by gathering the diagonal block of each
independent node's raw node, and `gather_independent_values()` gathers raw
nodal values (such as Dirichlet values) the same way. Projector blocks of
dependent nodes are dropped, so a Dirichlet region should contain the masters
of the hanging nodes it constrains. The result is not normalized.

The explicit rebuildable variant, variable-order and combined h/p spaces,
filtered raw-side nodal quadrature for constrained forms, and restricted-domain
interpolation checks remain follow-ups. `AdaptiveNanogrid` sides cover every cell facet, including
exposed portions of sparse grids; the constrained builder validates that
coverage.

## Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| R1 | Eliminate dependent hanging-node degrees of freedom for supported continuous spaces. | Must |
| R2 | Keep the original space available for unconstrained nodal integration on boundaries. | Must |
| R3 | Assemble linear and bilinear forms directly into independent degrees of freedom through the existing generic and dispatch paths. | Must |
| R4 | Interpolate into and evaluate fields in the constrained space using independent values. | Must |
| R5 | Apply strong boundary values to independent degrees of freedom and derive hanging values from them. | Must |
| R6 | Reject raw projectors containing active off-diagonal blocks when reducing them to independent nodes. Ignore inactive BSR row-capacity slots. Leave `normalize_dirichlet_projector()` unchanged. | Must |
| R7 | Preserve existing unconstrained and DG behavior. | Must |
| R8 | Map each side to one or two full cell facets and expose the sides covering each facet, including nonmatching interfaces. | Must |
| R9 | Keep polynomial order and trace-basis information in the space; use the same interface contract for h and p constraints. | Must |
| R10 | Expose each independent node acting on either adjacent cell once per constrained side, without duplicating a shared global node across the two traces. | Must |
| R11 | Provide an explicit rebuild path for a rebuildable adaptive geometry; refresh constraints, cell maps, and side node lists together. | Follow-up |

## Design

### Side and facet interfaces

A **facet** is the full codimension-one face of one cell, identified by
`(cell, local_facet)`. It remains one facet even if several smaller cells
touch it. A **side** is a geometry element used for integration over one
piece of a facet. It refers to one facet on the boundary, or to one facet
from each of two neighboring cells at an interior interface. Every
`AdaptiveNanogrid` cell facet is covered by sides without gaps or overlap;
one facet may have several sides. At a dense one-level refinement interface,
the coarse cell has one facet covered by four sides, each of which also refers
to one fine-cell facet. If fine cells cover only part of the coarse facet,
the remaining sides are boundary sides.

Expose this relationship as `SideFacetIncidence`: each side identifies its
one or two adjacent `(cell, local_facet)` pairs, while a CSR inverse lists
the sides covering each full cell facet. This does not introduce a separate
half-side element or a quadrature domain over facets. The `(cell, local_facet)`
pair already identifies the facet; it needs no new geometry storage. Existing
side-to-cell coordinate maps locate side points in the adjacent cells and
therefore on their facets. Build the incidence arrays on the geometry device
from side neighbors and local facet IDs. The geometry reports coverage and
orientation; it does not depend on a downstream basis space or its order.

Keep quadrature and interpolation samples attached to **sides**. A side
quadrature point has one `Sample.element_index`, one `Sample.qp_index`, and
one side-local coordinate, just as it does today. On an interior side, the
adjacent facet pair and side-to-cell coordinate maps identify both cell
traces at that point. Thus `fem.average(f, s)` and other two-sided operators
require neither a sample containing two element indices nor two quadrature
evaluations writing the same point index. Splitting the integration domain
into sides also resolves which fine facet supplies the outer trace at a
coarse-to-fine interface.

Interpolation at side quadrature points is therefore unambiguous. Nodal
interpolation into a side space follows its existing reduction rule when a
destination node occurs on multiple sides. If the quantity being sampled is
discontinuous across those sides, one shared nodal value cannot represent
all of its traces; use a discontinuous side space or retain values at side
quadrature points. The incidence contract does not choose a fine-cell trace
for a point specified only on an unsplit coarse facet.

The geometry reports overlap and orientation only. The space supplies the
cell's polynomial degree, local trace degrees of freedom, and trace basis
evaluation. Thus the same side describes both h mismatch (partial facet
coverage) and p mismatch (full facet coverage with different trace bases).
The interface contract must allow both properties on one side. The initial
constraint builder handles h refinement at uniform order; p refinement and
combined h/p constraints remain follow-ups.

Current `ShapeBasisSpace` has one shape and one `ORDER` for every cell; p refinement
will also require a variable-order raw basis and topology. The existing
`element_node_count()` can vary by cell, but that alone does not select a
different local shape.

### Space representation

Start with a polynomial basis and topology as the **raw space**. Build a sparse
prolongation map `P` from independent values to raw nodal values. For the initial
h-only case, evaluate the coarse facet basis at fine trace nodes to obtain the
dependent-node weights. Matching nodes already shared by the raw topology need
no additional constraint. Resolve chains and assign compact independent indices.

For p refinement on matching geometric facets, require equality of the two
traces. The shared trace must lie in the intersection of the adjacent trace
spaces. For nested nodal Lagrange spaces, the lower-order trace is the natural
master. Constraint construction must also reconcile edges and vertices where
several faces meet.

Combined h/p refinement is a later phase. Its sides need no new
geometric concept, but its constraints need a more general common-trace rule:
choosing the coarse facet as master may be wrong when its trace order exceeds
what the fine facets can represent. Do not make the initial geometry contract
encode a coarse-master or lower-order-master policy.

`ConstrainedSpaceTopology` stores, for each cell, the independent nodes whose
basis functions affect that cell. `ConstrainedBasisSpace` evaluates their weights
and gradients as weighted sums of the raw shape functions:

```text
constrained_basis[e, j](x) = sum_i local_prolongation[e, i, j] * raw_basis[e, i](x)
```

The number of contributing independent nodes may vary by cell and exceed the raw
shape function's node count. The topology exposes the maximum required count to
existing assembly code. `node_count()` and field `dof_values` refer to independent
nodes. Retain a reference to the raw space for boundary operations and for
expanding a solution to raw nodal values when needed.

The existing `SpaceTopology.element_node_count()` and `element_node_index()`
already abstract element-local node enumeration. For a constrained element,
their local slots must enumerate the distinct independent degrees of freedom
with nonzero transformed basis functions on that element. The corresponding
`BasisSpace.element_inner_weight()` and gradient use the same slots. A hanging
node whose value is a weighted combination of masters cannot be represented by
replacing its index with one master index; it disappears from the list, while
each contributing master appears with its transformed basis weight. Thus
`fem.node_count()` and `fem.node_index()` expose concrete independent nodes,
without adding a separate public enumeration path. The current generated
trace topology concatenates all inner and outer cell slots, with two slots
for a global node shared by both cells. A constrained trace should expose
each distinct independent node of the two cells once. Nodes with a zero trace
on the side must stay listed: their normal derivative there is generally
nonzero, and side forms using `fem.grad()` depend on it.

Begin with scalar and componentwise vector H1 Lagrange spaces on
`AdaptiveNanogrid`, initially Q1. Higher-order H1 spaces need coarse-facet
interpolation weights and coincident-node handling. Variable-order H1 spaces
need per-cell shape selection and a topology that permits differing local
trace nodes. H(div) and H(curl) spaces need separate trace constraint rules.

### Constrained trace elements

Build each constrained side's node list as the union of independent global
nodes whose transformed basis is nonzero on either adjacent cell. For each
side-local slot, store its global node index and
the corresponding inner and outer cell-local slots, using a null slot where
the node has no contribution from that cell. `TraceBasisSpace` evaluates the
inner and outer weights from those two references. A shared node has one
side-local slot with both weights; a DG node has only one because the two
cells use different global indices. This keeps jump and average forms tied
to the same independent unknowns that cell assembly uses.

Do not restrict the list to nodes with a nonzero trace on the side. A
function vanishing on the side can have a nonzero normal derivative there,
such as the function of a Q1 node on the opposite face, so a list of the
side's own nodes misses entries of any side form using gradients. For H1
nodal spaces, start from all raw nodes of both cells, then follow their
prolongation entries to the independent masters. Merge matching global
indices across the inner and outer lists. The side-local count may vary;
retain a static maximum for assembly, and verify that partitions and
restrictions consume the merged list.

The current trace topology assumes each slot belongs exclusively to the
inner or outer cell. Its `neighbor_cell_index()` and nodal quadrature-weight
paths also assume a single neighboring cell. Introduce the paired mapping
for constrained traces first; keep the raw trace path for unconstrained
boundary nodal integration. Generalizing paired traces to other spaces can
follow once their trace-support rules and nodal assembly behavior are clear.

### Construction and rebuild lifecycle

`make_conforming_basis_space(raw_basis)` is the explicit construction point.
It obtains the geometry's side-facet incidence, builds `P`, the
cell-local transform, master owner samples, and the side node lists, then
creates the constrained topology and basis with their trace wrappers. Both
`ConstrainedSpaceTopology.trace()` and `ConstrainedBasisSpace.trace()` return
the prebuilt wrappers. Side restrictions and partitions can therefore use
the trace without triggering hidden construction or host synchronization.

A later rebuildable variant should use separate explicit calls, in order:

```python
geometry.rebuild(...)
raw_basis.topology.rebuild()
constrained_basis.rebuild()
partition.rebuild()
restriction.rebuild()
# Recreate fields/matrices or transfer field values into the new numbering.
```

`AdaptiveNanogrid.rebuild()` updates only geometric data. It does not retain
or discover downstream basis spaces, partitions, fields, or restrictions.
The caller is responsible for rebuilding each dependent object after its
inputs have been refreshed. `constrained_basis.rebuild()` is the only point
that reconstructs its constrained cell and trace data.

This call must refresh the incidence, `P`, cell-local and side node maps,
owner samples, and active independent-node count as one coherent generation.
It must invalidate cached topology/basis arguments. The cell and trace
`MAX_NODES_PER_ELEMENT` bounds are compiled into assembly kernels, so a
rebuildable instance needs fixed bounds chosen at construction and must
reject a rebuild that exceeds them, requiring construction of a larger
space. No trace work is deferred to a later `trace()` call.

The follow-up public mode can be `make_conforming_basis_space(raw_basis,
rebuildable=True)`, backed by a rebuildable `AdaptiveNanogrid`. It should
expose or derive fixed local cell and side slot bounds at construction.
`AdaptiveNanogrid` currently lacks the rebuildable construction and topology
refresh path that `Nanogrid` has. Supporting the variant therefore includes
rebuilding its cell levels, node and stacked-face grids, completed side
grid, and side-facet incidence from the new cells. Independent numbering may
change even when capacities do not. After `constrained_basis.rebuild()`,
partitions and restrictions must be rebuilt, while fields and matrices need
new storage or an explicit value-transfer step. Captured graphs using the
old maps must be recaptured unless a later capacity-stable implementation
guarantees their device pointers and launch bounds. Stale dependents should
fail clearly rather than silently use the old generation.

### Form integration and boundary nodal integration

The generic and dispatch integration kernels already ask a topology for the
element's global nodes and a basis for their weights. With the transformed
topology and basis, they should produce the reduced vector and matrix directly:

```text
b_reduced = P^T b_raw
A_reduced = P^T A_raw P
```

Keep `assembly="nodal"` on the raw space, including `BoundarySides`, unchanged.
That path assumes a nodal basis and produces a diagonal matrix. Do not silently
apply it to a constrained test or trial space. When nodal boundary *quadrature*
must act on constrained unknowns, take the sample points from the raw boundary
basis and evaluate the constrained test and trial basis using generic or dispatch
assembly. The current `NodalQuadrature` enumerates all trace slots, including
ones whose coordinates are `OUTSIDE`, and Q1 gives those slots nonzero trace
quadrature weights. Filter to valid raw side nodes before using that path.
The resulting reduced boundary matrix can have off-diagonal entries.

### Interpolation and node locations

Each independent H1 node retains its raw master position as its physical
anchor. The local basis slot and global node index identify that anchor even
when it lies outside the current cell or side. Today,
`NodalFieldBase.at_node()` and the test/trial variant simply replace a
sample's coordinates with `BasisSpace.node_coords_in_element()` and keep its
element index. For a constrained space, this operation must instead use the
master associated with the local slot. `fem.node_index(f, s, k)` returns that
master's independent global index, and `fem.at_node(f, s, k)` returns a sample
at the same master's physical position. It must never return a sample at an
eliminated hanging node. This preserves APIC's
`domain(at_node(u, s))` node position and nodal reads such as
`U(at_node(U, s, k))`.

The master need not lie inside the element or side from which `k` was
enumerated. `at_node()` may return a sample on another element containing
the master. Associate each independent node with a representative cell and,
for an active trace node, a representative side containing its anchor. The
returned sample uses the representative element's index and in-element
coordinates. This works for geometries without relying on extrapolation
outside an element's reference domain.

Relocation must not leave stale element-local data in the returned `Sample`.
When `at_node()` selects a different element, keep `qp_index` and
`qp_weight`, which identify the integrand's quadrature point rather than an
element. Map the selected test or trial
field's local DOF descriptor to the same master on the representative element.
Other test/trial descriptors must not reuse their old local indices: they keep
a reference to their slot in the original element and are looked up again on
the representative element when evaluated (see the `at_node()` section of
`constrained-basis-spaces.md`). When the sample stays on its original element,
retain the existing `at_node()` behavior. The constrained polynomial basis
must evaluate at the returned coordinates. Verify that
`domain(at_node(u, s))` gives the master's position and that evaluating an H1
nodal field at its returned sample gives the master's value.

Use a canonical representative to populate `BasisSpace.node_positions()` once
per independent node. Interpolation into a constrained field should sample
each independent node from an element in the interpolation domain that
contains its anchor. For restricted domains, this may differ from the
canonical representative used by `at_node()`. Return `OUTSIDE` for an
incidence without a valid local interpolation sample; do not extrapolate the
integrand from every incident element. Check full, boundary, and restricted
domains, as well as interpolation Jacobians.

### Strong Dirichlet conditions

Raw boundary nodal assembly remains useful for finding boundary samples and
values. A raw projector has raw-space dimensions and cannot be applied directly
to a reduced system. Reducing a diagonal raw boundary matrix with `P^T D P`
can produce off-diagonal entries; it is a boundary operator, not a diagonal
selector of independent boundary degrees of freedom. Construct the strong
Dirichlet selector from independent boundary nodes, apply values there, and
derive dependent values through `P`.

`reduce_dirichlet_projector()` only gathers the diagonal block of each
independent node's anchor. It does not validate the projector, so it needs no
readback and can run during graph capture. A projector assembled with
`assembly="nodal"` is block-diagonal by construction.

## Implementation Plan

1. Define the geometry-independent side-facet incidence contract;
   adapt `AdaptiveNanogrid` to it and verify that sides cover every cell facet
   exactly once. Include matching facets so p refinement
   uses the same contract.
2. Define the raw-to-independent constraint map and build its first h-only
   version from that contract. Detect conflicting constraints, cycles, and
   cross-environment links.
3. Add `ConstrainedSpaceTopology` and `ConstrainedBasisSpace`, including
   concrete element-node enumeration, trace evaluation, and canonical node
   positions. Expose the raw space. Build all cell and side maps at the
   explicit space construction point.
4. Add constrained trace elements with one slot per independent global node
   of the adjacent cells and paired inner/outer references. Update trace basis weights,
   restrictions, and partitions to use those slots.
5. Expose a first-phase `make_conforming_basis_space(raw_basis)` factory.
   A convenience option on polynomial space creation can follow once the
   wrapper's behavior is established. Keep `discontinuous=True` separate.
6. Validate generic and dispatch integration. Guard the constrained
   `assembly="nodal"` case; preserve raw boundary nodal assembly. Add valid
   raw-side-node filtering before using nodal quadrature with constrained
   forms.
7. Make `at_node()` resolve the independent master for each local slot,
   including relocation to a representative cell or side and consistent
   sample metadata. Validate APIC-style positions and nodal-field reads.
   Validate interpolation and its Jacobians on cell and side domains.
8. Add independent-node Dirichlet selection with a structural check of the
   raw projector.
9. Add a rebuildable `AdaptiveNanogrid` variant and explicit constrained
   basis rebuild. Keep local cell and trace bounds fixed per instance;
   rebuild or replace dependent partitions, restrictions, fields, matrices,
   and captured graphs according to their storage and generation rules.
10. Extend the raw basis and topology to per-cell polynomial orders. Build
   constraints for differing orders on matching geometric facets, using the
   same side-facet contract and constrained-space assembly path.
11. Later, extend constraint construction to combined h/p refinement without
   changing the side-facet contract.

## Proposed Code Changes and Impact

This is a review draft centered on scalar and componentwise vector Q1 H1
spaces on `AdaptiveNanogrid`. It also records follow-up work. The current
implementation does not include p refinement, H(div), H(curl), in-place
updates after changing the geometry, independent-node strong Dirichlet
selection, or filtered raw-side nodal quadrature for constrained forms.

### Public construction and ownership

Keep existing polynomial factories and their defaults unchanged. Add a space
transformation with the following proposed use:

```python
raw_basis = fem.make_polynomial_basis_space(geo, degree=1)
basis = fem.make_conforming_basis_space(raw_basis)
space = fem.make_collocated_function_space(basis, dtype=wp.vec3)
```

The factory accepts only a supported nodal H1 basis on an
`AdaptiveNanogrid` in the first phase and raises `NotImplementedError` for
other inputs. `basis.raw_basis` retains the original space for unconstrained
boundary nodal operations. Its topology numbers only independent nodes;
field `dof_values` and assembled matrices use those numbers. The constraint
map provides a way to expand independent values to raw nodal values for
inspection and boundary workflows. A convenience keyword on
`make_polynomial_space()` is deferred; it is not needed to define the core
behavior.

### Files and interfaces

| File | Proposed change | Existing users |
| --- | --- | --- |
| `warp/_src/fem/geometry/geometry.py` | Add the optional `SideFacetIncidence` contract. Keep the existing side-to-cell coordinate maps. | Other geometries keep their existing behavior until they implement the contract. |
| `warp/_src/fem/geometry/adaptive_nanogrid.py` | Build side-to-local-facet IDs and the inverse facet-to-side CSR map on the geometry device. Complete fine-level boundary sides beside refinement interfaces. | Previously missing boundary sides may increase side counts on sparse grids. |
| `warp/_src/fem/space/constrained_basis_space.py` (new) | Build Q1 constraints and define `ConstrainedSpaceTopology`, `ConstrainedBasisSpace`, `ConstrainedTraceSpaceTopology`, and `ConstrainedTraceBasisSpace`. | Opt-in basis only. |
| `warp/_src/fem/space/topology.py` and `basis_space.py` | Add a default basis node-sample hook that returns the current element and local coordinates. Let the constrained basis override it. | Existing bases retain the current result. |
| `warp/_src/fem/field/nodal_field.py` and `virtual.py` | Route `at_node()` through the node-sample hook. Remap the selected local DOF when the element changes; clear `qp_index` and `qp_weight` on relocation. | Behavior changes only for constrained spaces. |
| `warp/_src/fem/integrate.py` and `quadrature/quadrature.py` | Reject constrained `assembly="nodal"` and constrained `NodalQuadrature`. Filter `OUTSIDE` raw side nodes before offering raw nodal quadrature with constrained generic or dispatch forms. | Raw nodal boundary integration remains available. |
| `warp/_src/fem/space/partition.py` and `restriction.py` | Consume the merged trace returned by `ConstrainedSpaceTopology.trace()`; adjust code only if an existing assumption of two cell-sized halves remains. | Raw spaces use the current trace topology. |
| `warp/_src/fem/space/__init__.py`, `warp/fem/__init__.py`, and `warp/__init__.py` | Export the factory and register any new internal module with the public module source map. | One new opt-in API. |
| `warp/tests/fem/`, `warp/tests/unittest_suites.py`, docs, and `changelog/` | Add focused `unittest` coverage, register any new test module, document construction and boundary behavior, and add a user-facing changelog fragment. | No runtime impact. |

### Side-facet incidence data

`SideFacetIncidence` holds arrays equivalent to:

```text
inner_cell[side], outer_cell[side]
inner_facet[side], outer_facet[side]
facet_side_offsets[cell * facets_per_cell + local_facet]
facet_side_indices[facet_side_offsets[i] : facet_side_offsets[i + 1]]
```

Each side maps to one full facet on each adjacent cell and appears once in
each corresponding inverse list. Its geometric footprint may cover only part
of either facet. Boundary sides have one cell contribution and
``outer_cell = outer_facet = -1``. The inverse list may have several sides
for a coarse facet, but the sides cover every facet exactly once, including
exposed parts of sparse grids. Existing `side_inner_cell_coords()` and
`side_outer_cell_coords()` map side points into the adjacent cells, so the
geometry does not store polynomial degrees or constraint coefficients.
Construction of the incidence arrays and inverse CSR map stays on the GPU;
the constraint builder separately reads those arrays when it constructs
the space. Other geometries can implement the same contract.

### Constraint and element data

Store the global prolongation `P` in CSR form:

```text
raw_row_offsets[raw_node + 1]
reduced_node_ids[entry], coefficients[entry]
```

An independent raw node has one entry with weight one; a hanging raw node
has the coarse-facet master nodes and their Q1 interpolation weights. Number
independent nodes in raw-node order. For each fine trace node on a split
side, evaluate the coarse facet basis at its physical point. Gather candidate
rows from every touching side, resolve dependencies in coarse-to-fine order,
and compare duplicate rows after expansion. Reject conflicting rows, cycles,
and cross-environment dependencies rather than choosing one silently.

Build an element-local sparse transform from `P` and the raw element-node
indices. It stores each cell's distinct reduced node IDs and, for every
`(cell, local reduced slot)`, a short list of `(raw local slot, weight)` pairs.
Then the device methods have the following behavior:

```text
ConstrainedSpaceTopology.element_node_count(cell) -> count of reduced slots
ConstrainedSpaceTopology.element_node_index(cell, j) -> reduced node ID
ConstrainedBasisSpace.element_inner_weight(cell, x, j)
    -> sum((weight * raw_basis(cell, x, raw_slot)) for that slot)
```

Apply the same transform to weight gradients. Choose
`MAX_NODES_PER_ELEMENT` from the built map's maximum, because assembly sizes
its temporary sparse buffers from this constant. Keep raw topology and basis
arguments available inside the wrappers. Record one containing owner cell,
local reduced slot, and coordinates per independent node for `at_node()` and
`node_positions()`. Reconstruct these arrays and the constrained space after
the adaptive geometry changes in the first phase.

### Trace data and evaluation

For each side, enumerate the raw slots of both adjacent cells, follow their
local transform entries, and take the sorted union of the resulting reduced
node IDs from both cells. Build these arrays:

```text
side_offsets[side + 1]
side_reduced_node_ids[slot]
side_inner_cell_slot[slot]  # or NULL_NODE_INDEX
side_outer_cell_slot[slot]  # or NULL_NODE_INDEX
```

`ConstrainedTraceSpaceTopology.element_node_count()` and
`element_node_index()` read this list. The trace basis evaluates the inner
weight from `side_inner_cell_slot` and outer weight from
`side_outer_cell_slot`, returning zero for a null slot. The same global node
can therefore have both traces in one side-local slot. A Q1 interior side
between conforming cells exposes twelve nodes rather than the current
sixteen; a DG side would expose sixteen distinct nodes if this scheme were
later extended to DG. Raw traces remain unchanged for now.

Build the constraint map, cell-local transform, master owner-cell and
owner-side data, side slot arrays above, and their maximum slot count in
`make_conforming_basis_space()`. Create and retain the trace topology and
basis there. `ConstrainedSpaceTopology.trace()` and
`ConstrainedBasisSpace.trace()` return these prebuilt objects; neither call
allocates maps or computes side counts. `SpaceRestriction` on a side domain
and `NodePartition` with side halos can then call `topology.trace()` without
triggering construction. A rebuildable variant refreshes the side node lists
only at its explicit `rebuild()` call. No assembly kernel constructs the
trace. Have the trace topology's `full_space_topology()` return the parent
topology so restrictions, partitions, and test/trial compatibility checks
share the mapping.

Do not use the master's physical location on the current side to decide
trace support: its transformed basis can be nonzero there while its anchor
lies on another side. `SpaceRestriction` and `NodePartition` should see the
merged list through `topology.trace()`; generic and dispatch integration
continue to ask the topology for side-local counts and indices and the basis
for inner/outer weights.

### Node samples, interpolation, and boundaries

Add a `BasisSpace.node_sample(element, local_slot)` device hook returning an
element index, coordinates, and the slot of the same node on that element.
The default implementation returns the current element and existing local
node coordinates. A constrained cell or trace implementation returns the
recorded owner cell or side when the master lies elsewhere. `at_node()`
builds its `Sample` from that result. If the element changes, set
`qp_index=NULL_QP_INDEX` and `qp_weight=0`, remap the selected test or trial
DOF, and invalidate the other element-local DOF descriptor. If the element
does not change, retain the current behavior. This preserves APIC-style
`domain(at_node(u, s))` and nodal reads through `U(at_node(U, s, k))`.

Keep `node_coords_in_element()` for interpolation separate from the
relocating node-sample hook. Return `OUTSIDE` when the master's anchor is not
in that incident element, so `interpolate()` samples only from containing
elements in its domain. If a restricted domain contains no valid anchor
sample for a node it otherwise includes, reject that interpolation rather
than silently extrapolating. `BasisSpace.node_positions()` uses one recorded
owner instead of writing a position from every incident cell.

For boundary nodal integration, use the raw basis with
`assembly="nodal"`. To use raw nodal points with constrained test/trial
fields and generic or dispatch assembly, first pack only valid raw side
nodes into a filtered quadrature; the current `NodalQuadrature` includes
off-face slots. A reduced boundary matrix can have off-diagonal entries
and is not a diagonal strong-Dirichlet selector. Build that selector from
independent master anchors on the
selected boundary. If the boundary subset contains a dependent point but
not the masters needed to represent its value, report that strong nodal
selection is unsupported for that subset; retain weak boundary forms as an
alternative. Before projector normalization, inspect only active BSR row
entries, including padded matrices, and raise on any off-diagonal column.

### Expected impact

| Area | Impact |
| --- | --- |
| Existing code | Default spaces, raw traces, and raw boundary nodal assembly are unchanged. Constrained spaces are opt-in and have fewer global DOFs. |
| Forms | Generic and dispatch kernels should require no new form API. They assemble directly into reduced rows and columns through the wrapper topology and basis. Explicit nodal assembly on a constrained space raises. |
| Performance | Space construction builds constraints, cell-local maps, and side node lists together. Cell basis evaluation sums short raw-basis stencils. Side lists merge nodes shared by both cells, but a side's node count is close to the sum of its cells' counts, and the cell's maximum active-node count may increase and thus enlarge temporary sparse buffers. |
| Memory | Store global and cell-local CSR maps, side slot maps, and owner samples. No dense prolongation or post-assembly `P^T A P` matrix product is required. |
| Device and capture | Build the constrained space before capture; its trace is already available. Reconstruct after geometry changes in the first version. The follow-up rebuildable variant refreshes maps explicitly and must define recapture and dependent-object rules. |
| Compatibility | Code that indexes `dof_values` by raw node ID must use the reduced numbering or expand through `P`. On constrained spaces, `at_node()` may return another element and clears quadrature metadata when it does. |

## Testing Strategy

- Reproduce constants and linear fields across coarse-to-fine facets; check
  zero jumps in values and continuity at edges and corners.
- Check that sides cover every cell-local facet without gaps or overlaps,
  including coarse-to-fine interfaces and exposed boundary facets.
- Compare reduced linear and bilinear forms against explicit `P^T b` and
  `P^T A P` references on small grids, including boundary forms.
- Check interpolation into scalar and vector constrained fields and expansion
  back to raw nodal values.
- Check `fem.node_count()` and `fem.node_index()` against the independent
  element-node list. Include masters outside the current fine cell or side.
  Check `domain(at_node(u, s))` against each master's physical position
  for APIC-style transfers, and check that `U(at_node(U, s, k))` returns the
  corresponding independent nodal value. When `at_node()` changes elements,
  check that quadrature metadata is cleared and local DOF indices are
  remapped or invalidated correctly.
- On side elements, check that every exposed node has nonzero inner or outer
  trace, shared global nodes occur once, and inner/outer weights and jump
  forms agree with an uncompressed reference.
- Verify unconstrained boundary nodal integration still works and that reduced
  boundary operators are not treated as diagonal Dirichlet projectors.
- Check that projector normalization rejects active off-diagonal blocks in
  compact and padded BSR matrices without changing either input, while
  ignoring padded slack.
- Exercise CPU and CUDA, partition boundaries, multiple environments, and
  graph capture where supported.

When p refinement is implemented later, add p1/p2 and p2/p1 neighbors on
matching faces, then mixed h/p interfaces; check exact trace agreement at
more points than the interpolation nodes.

## Open Questions

- Should conforming behavior become the default for supported adaptive spaces
  after an opt-in period?
- What public option should request continuity constraints for both h and p
  refinement, and how should it interact with `discontinuous=True`?
- Which adaptive grading conditions and polynomial orders belong in the first
  implementation?
- How stable should side and cell-facet identifiers remain across explicit
  geometry rebuilds and partitions?
- How should representative interpolation samples be selected for restricted
  or boundary-only domains when the anchor lies outside the current element?
- How should strong Dirichlet conditions behave when a selected boundary
  subset contains dependent nodes but not all of their master anchors? A
  diagonal independent-node selector may not represent that condition.
- What validation strategy gives a synchronous projector error on CUDA
  without imposing unacceptable cost or breaking graph capture?
