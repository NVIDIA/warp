# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constraint producers and shared constraint resolution for constrained basis spaces.

A producer emits candidate constraints of the form "node ``a`` equals
``sum_k w_k * node(m_k)``" on the nodes of a basis. The shared resolution
picks one canonical candidate per constrained node, expands chains of
constraints until every row refers only to unconstrained nodes, checks that
the other candidates agree, and numbers the unconstrained nodes. Its output is
the prolongation ``P`` in CSR form.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import warp as wp
from warp._src.fem import cache
from warp._src.fem.types import NULL_QP_INDEX
from warp._src.sparse import BsrMatrix, bsr_zeros
from warp._src.utils import array_scan

from .basis_space import BasisSpace, ShapeBasisSpace
from .shape import ShapeFunction
from .topology import RegularDiscontinuousSpaceTopologyMixin

_NO_KEY = wp.constant((1 << 31) - 1)
"""Empty value for ``atomic_min`` slots; larger than any valid key."""

# Slots of the constraint status array. Error slots hold the smallest
# offending key (``_NO_KEY`` when there is no error); the others hold flags,
# counts and sizes read back by the host.
_PROJECTION_ERROR = wp.constant(0)
_EXPANSION_OVERFLOW = wp.constant(1)
_UNRESOLVED_COUNT = wp.constant(2)
_INCONSISTENT_ERROR = wp.constant(3)
_INDEPENDENT_COUNT = wp.constant(4)
_PROLONGATION_COUNT = wp.constant(5)
_HANGING_SIDE_COUNT = wp.constant(6)
_SIDE_NODE_MAX = wp.constant(7)
_STATUS_SIZE = 8

_INITIAL_EXPANSION_CAPACITY = 8
"""Initial number of independent nodes stored per raw node; doubled on overflow."""


def _new_status(device) -> wp.array:
    """Allocate a status array with empty error slots and zero counters."""
    status = wp.empty(_STATUS_SIZE, dtype=int, device=device)
    status.fill_(_NO_KEY)
    status[_EXPANSION_OVERFLOW:_INCONSISTENT_ERROR].zero_()
    status[_INDEPENDENT_COUNT:_STATUS_SIZE].zero_()
    return status


@dataclass
class _Candidates:
    """Candidate constraints, one per key, with the canonical key of each constrained node."""

    count: wp.array
    """Number of masters of each candidate; zero for unused keys."""
    raw: wp.array
    """Constrained node of each candidate."""
    masters: wp.array
    """Master nodes of each candidate, sorted, in the first ``count`` columns."""
    weights: wp.array
    """Double-precision weight of each master."""
    canonical: wp.array
    """Smallest candidate key per node, or ``_NO_KEY`` for unconstrained nodes."""


@dataclass
class _Prolongation:
    """Prolongation in CSR form, with one row per node and one column per independent node."""

    offsets: wp.array
    indices: wp.array
    weights: wp.array
    independent_raw_nodes: wp.array
    """Row of the unit entry of each column, in increasing order."""
    column_count: int
    nnz: int


@wp.func
def _find_sorted_range(keys: wp.array[int], begin: int, end: int, key: int):
    """Return the position of ``key`` in the sorted range ``[begin, end)`` of ``keys``, or -1."""
    lower = int(begin)
    upper = int(end)
    while lower < upper:
        middle = (lower + upper) // 2
        if keys[middle] < key:
            lower = middle + 1
        else:
            upper = middle
    if lower < end:
        if keys[lower] == key:
            return lower
    return -1


@wp.func
def _add_sorted_range(
    keys: wp.array[int], weights: wp.array[wp.float64], begin: int, count: int, key: int, weight: wp.float64
):
    """Add ``weight`` to ``key`` in the sorted range starting at ``begin``; return the new count."""
    position = int(0)
    for entry in range(count):
        if keys[begin + entry] == key:
            weights[begin + entry] += weight
            return count
        if keys[begin + entry] < key:
            position = entry + 1
    cursor = int(count)
    while cursor > position:
        keys[begin + cursor] = keys[begin + cursor - 1]
        weights[begin + cursor] = weights[begin + cursor - 1]
        cursor -= 1
    keys[begin + position] = key
    weights[begin + position] = weight
    return count + 1


@wp.func
def _find_sorted(keys: wp.array2d[int], row: int, count: int, key: int):
    """Return the position of ``key`` in the sorted first ``count`` entries of ``row``, or -1."""
    lower = int(0)
    upper = int(count)
    while lower < upper:
        middle = (lower + upper) // 2
        if keys[row, middle] < key:
            lower = middle + 1
        else:
            upper = middle
    if lower < count:
        if keys[row, lower] == key:
            return lower
    return -1


@wp.func
def _insert_sorted_key(keys: wp.array2d[int], row: int, count: int, key: int):
    """Insert ``key`` in a sorted row unless present; return the new count."""
    position = int(0)
    for entry in range(count):
        if keys[row, entry] == key:
            return count
        if keys[row, entry] < key:
            position = entry + 1
    cursor = int(count)
    while cursor > position:
        keys[row, cursor] = keys[row, cursor - 1]
        cursor -= 1
    keys[row, position] = key
    return count + 1


@wp.func
def _add_sorted_entry(
    keys: wp.array2d[int], weights: wp.array2d[wp.float64], row: int, count: int, key: int, weight: wp.float64
):
    """Add ``weight`` to ``key`` in a sorted row; return the new count, or -1 when the row is full."""
    position = int(0)
    for entry in range(count):
        if keys[row, entry] == key:
            weights[row, entry] += weight
            return count
        if keys[row, entry] < key:
            position = entry + 1
    if count == keys.shape[1]:
        return -1
    cursor = int(count)
    while cursor > position:
        keys[row, cursor] = keys[row, cursor - 1]
        weights[row, cursor] = weights[row, cursor - 1]
        cursor -= 1
    keys[row, position] = key
    weights[row, position] = weight
    return count + 1


@wp.kernel(enable_backward=False)
def _init_expansion_kernel(
    canonical: wp.array[int],
    expansion_count: wp.array[int],
    expansion_keys: wp.array2d[int],
    expansion_weights: wp.array2d[wp.float64],
    resolved: wp.array[int],
):
    node = wp.tid()
    if canonical[node] == _NO_KEY:
        expansion_count[node] = 1
        expansion_keys[node, 0] = node
        expansion_weights[node, 0] = wp.float64(1.0)
        resolved[node] = 1
    else:
        expansion_count[node] = 0
        resolved[node] = 0


@wp.kernel(enable_backward=False)
def _resolve_round_kernel(
    canonical: wp.array[int],
    candidate_count: wp.array[int],
    candidate_masters: wp.array2d[int],
    candidate_weights: wp.array2d[wp.float64],
    resolved_in: wp.array[int],
    resolved_out: wp.array[int],
    expansion_count: wp.array[int],
    expansion_keys: wp.array2d[int],
    expansion_weights: wp.array2d[wp.float64],
    status: wp.array[int],
):
    node = wp.tid()
    if resolved_in[node] != 0:
        resolved_out[node] = 1
        return

    # Wait until every master of the node's constraint was resolved in an
    # earlier round, so the rows read here are complete.
    candidate = canonical[node]
    master_count = candidate_count[candidate]
    for i in range(master_count):
        if resolved_in[candidate_masters[candidate, i]] == 0:
            resolved_out[node] = 0
            return

    count = int(0)
    for i in range(master_count):
        master = candidate_masters[candidate, i]
        weight = candidate_weights[candidate, i]
        for j in range(expansion_count[master]):
            count = _add_sorted_entry(
                expansion_keys,
                expansion_weights,
                node,
                count,
                expansion_keys[master, j],
                weight * expansion_weights[master, j],
            )
            if count < 0:
                wp.atomic_max(status, _EXPANSION_OVERFLOW, 1)
                expansion_count[node] = 0
                resolved_out[node] = 1
                return

    kept = int(0)
    for j in range(count):
        if wp.abs(expansion_weights[node, j]) > wp.float64(1.0e-12):
            expansion_keys[node, kept] = expansion_keys[node, j]
            expansion_weights[node, kept] = expansion_weights[node, j]
            kept += 1
    expansion_count[node] = kept
    resolved_out[node] = 1


@wp.kernel(enable_backward=False)
def _count_unresolved_kernel(resolved: wp.array[int], status: wp.array[int]):
    node = wp.tid()
    if resolved[node] == 0:
        wp.atomic_add(status, _UNRESOLVED_COUNT, 1)


@wp.func
def _candidate_value(
    candidate_count: wp.array[int],
    candidate_masters: wp.array2d[int],
    candidate_weights: wp.array2d[wp.float64],
    expansion_count: wp.array[int],
    expansion_keys: wp.array2d[int],
    expansion_weights: wp.array2d[wp.float64],
    candidate: int,
    key: int,
):
    """Weight of independent raw node ``key`` in a candidate expanded with resolved masters."""
    value = wp.float64(0.0)
    for i in range(candidate_count[candidate]):
        master = candidate_masters[candidate, i]
        position = _find_sorted(expansion_keys, master, expansion_count[master], key)
        if position >= 0:
            value += candidate_weights[candidate, i] * expansion_weights[master, position]
    return value


@wp.kernel(enable_backward=False)
def _consistency_kernel(
    canonical: wp.array[int],
    candidate_count: wp.array[int],
    candidate_raw: wp.array[int],
    candidate_masters: wp.array2d[int],
    candidate_weights: wp.array2d[wp.float64],
    expansion_count: wp.array[int],
    expansion_keys: wp.array2d[int],
    expansion_weights: wp.array2d[wp.float64],
    tolerance: wp.float64,
    status: wp.array[int],
):
    candidate = wp.tid()
    if candidate_count[candidate] == 0:
        return
    node = candidate_raw[candidate]
    if canonical[node] == candidate:
        return

    # Same nonzero keys as the resolved node, with matching weights
    node_count = expansion_count[node]
    for j in range(node_count):
        value = _candidate_value(
            candidate_count,
            candidate_masters,
            candidate_weights,
            expansion_count,
            expansion_keys,
            expansion_weights,
            candidate,
            expansion_keys[node, j],
        )
        if wp.abs(value) <= wp.float64(1.0e-12) or wp.abs(value - expansion_weights[node, j]) > tolerance:
            wp.atomic_min(status, _INCONSISTENT_ERROR, node)
            return

    for i in range(candidate_count[candidate]):
        master = candidate_masters[candidate, i]
        for j in range(expansion_count[master]):
            key = expansion_keys[master, j]
            if _find_sorted(expansion_keys, node, node_count, key) < 0:
                value = _candidate_value(
                    candidate_count,
                    candidate_masters,
                    candidate_weights,
                    expansion_count,
                    expansion_keys,
                    expansion_weights,
                    candidate,
                    key,
                )
                if wp.abs(value) > wp.float64(1.0e-12):
                    wp.atomic_min(status, _INCONSISTENT_ERROR, node)
                    return


@wp.kernel(enable_backward=False)
def _row_count_kernel(
    canonical: wp.array[int],
    expansion_count: wp.array[int],
    master_mask: wp.array[int],
    row_counts: wp.array[int],
):
    # One extra trailing entry so that exclusive scans also produce totals
    node = wp.tid()
    if node == canonical.shape[0]:
        master_mask[node] = 0
        row_counts[node] = 0
        return
    master_mask[node] = wp.where(canonical[node] == _NO_KEY, 1, 0)
    row_counts[node] = expansion_count[node]


@wp.kernel(enable_backward=False)
def _store_totals_kernel(independent_index: wp.array[int], offsets: wp.array[int], status: wp.array[int]):
    last = offsets.shape[0] - 1
    status[_INDEPENDENT_COUNT] = independent_index[last]
    status[_PROLONGATION_COUNT] = offsets[last]


def _resolve_constraints(
    candidates: _Candidates,
    round_count: int,
    status: wp.array,
    scalar_type: type,
    device,
    kind: str,
    node_name: str = "node",
    tolerance: float = 1.0e-7,
) -> _Prolongation:
    """Resolve candidate constraints into a prolongation in CSR form.

    Unconstrained nodes become the independent nodes, numbered in node order.
    Each constrained node is expanded through its canonical candidate until it
    refers only to independent nodes, then all of its other candidates must
    expand to the same row.

    Args:
        candidates: Candidate constraints and canonical keys.
        round_count: Expected length of the longest chain of constraints. More
            rounds run while they make progress.
        status: Constraint status array.
        scalar_type: Scalar type of the returned weights.
        device: Device of the candidate arrays.
        kind: Kind of constraint, used in error messages.
        node_name: Name of the constrained nodes, used in error messages.
        tolerance: Largest difference between the weights of two candidates of a node.
    """
    node_count = candidates.canonical.shape[0]
    capacity = _INITIAL_EXPANSION_CAPACITY
    while True:
        expansions = _expand_constraints(candidates, round_count, status, capacity, device, kind)
        if expansions is not None:
            break
        if capacity >= node_count:
            raise RuntimeError(f"Expansion of {kind} constraints exceeds the node count")
        capacity = min(2 * capacity, node_count)

    # All constraints of a node must agree once expanded
    wp.launch(
        _consistency_kernel,
        dim=candidates.count.shape[0],
        inputs=[
            candidates.canonical,
            candidates.count,
            candidates.raw,
            candidates.masters,
            candidates.weights,
            *expansions,
            wp.float64(tolerance),
            status,
        ],
        device=device,
    )

    # Independent nodes are the nodes without constraints, in node order
    master_mask = wp.empty(node_count + 1, dtype=int, device=device)
    row_counts = wp.empty(node_count + 1, dtype=int, device=device)
    independent_index = wp.empty(node_count + 1, dtype=int, device=device)
    offsets = wp.empty(node_count + 1, dtype=int, device=device)
    wp.launch(
        _row_count_kernel,
        dim=node_count + 1,
        inputs=[candidates.canonical, expansions[0], master_mask, row_counts],
        device=device,
    )
    array_scan(master_mask, independent_index, inclusive=False)
    array_scan(row_counts, offsets, inclusive=False)
    wp.launch(_store_totals_kernel, dim=1, inputs=[independent_index, offsets, status], device=device)

    values = status.numpy()
    if values[_INCONSISTENT_ERROR] != _NO_KEY:
        raise ValueError(f"Inconsistent {kind} constraints for {node_name} {int(values[_INCONSISTENT_ERROR])}")
    independent_count = int(values[_INDEPENDENT_COUNT])
    nnz = int(values[_PROLONGATION_COUNT])

    independent_raw_nodes = wp.empty(independent_count, dtype=int, device=device)
    indices = wp.empty(nnz, dtype=int, device=device)
    weights = wp.empty(nnz, dtype=scalar_type, device=device)
    wp.launch(
        _emit_prolongation_kernel(scalar_type),
        dim=node_count,
        inputs=[
            candidates.canonical,
            independent_index,
            offsets,
            *expansions,
            independent_raw_nodes,
            indices,
            weights,
        ],
        device=device,
    )
    return _Prolongation(offsets, indices, weights, independent_raw_nodes, independent_count, nnz)


def _expand_constraints(candidates: _Candidates, round_count: int, status, capacity: int, device, kind: str):
    """Run resolution rounds with fixed row capacity; return ``None`` if a row overflows."""
    canonical = candidates.canonical
    node_count = canonical.shape[0]
    expansion_count = wp.empty(node_count, dtype=int, device=device)
    expansion_keys = wp.empty((node_count, capacity), dtype=int, device=device)
    expansion_weights = wp.empty((node_count, capacity), dtype=wp.float64, device=device)
    resolved = [wp.empty(node_count, dtype=int, device=device), wp.empty(node_count, dtype=int, device=device)]
    status[_EXPANSION_OVERFLOW : _UNRESOLVED_COUNT + 1].zero_()
    wp.launch(
        _init_expansion_kernel,
        dim=node_count,
        inputs=[canonical, expansion_count, expansion_keys, expansion_weights, resolved[0]],
        device=device,
    )

    # Producers give the expected chain length; extra rounds only run if
    # chains are longer, and stop when a round makes no progress.
    remaining = node_count + 1
    while True:
        for _ in range(round_count):
            wp.launch(
                _resolve_round_kernel,
                dim=node_count,
                inputs=[
                    canonical,
                    candidates.count,
                    candidates.masters,
                    candidates.weights,
                    resolved[0],
                    resolved[1],
                    expansion_count,
                    expansion_keys,
                    expansion_weights,
                    status,
                ],
                device=device,
            )
            resolved.reverse()

        status[_UNRESOLVED_COUNT : _UNRESOLVED_COUNT + 1].zero_()
        wp.launch(_count_unresolved_kernel, dim=node_count, inputs=[resolved[0], status], device=device)
        values = status.numpy()
        if values[_EXPANSION_OVERFLOW] != 0:
            return None
        previous = remaining
        remaining = int(values[_UNRESOLVED_COUNT])
        if remaining == 0:
            return expansion_count, expansion_keys, expansion_weights
        if remaining == previous:
            raise ValueError(f"Cyclic {kind} constraints")
        round_count = 1


def _emit_prolongation_kernel(scalar_type):
    @cache.dynamic_kernel(suffix=scalar_type.__name__, kernel_options={"enable_backward": False})
    def emit_prolongation(
        canonical: wp.array[int],
        independent_index: wp.array[int],
        offsets: wp.array[int],
        expansion_count: wp.array[int],
        expansion_keys: wp.array2d[int],
        expansion_weights: wp.array2d[wp.float64],
        independent_raw_nodes: wp.array[int],
        indices: wp.array[int],
        weights: wp.array[scalar_type],
    ):
        node = wp.tid()
        if canonical[node] == _NO_KEY:
            independent_raw_nodes[independent_index[node]] = node
        start = offsets[node]
        for j in range(expansion_count[node]):
            indices[start + j] = independent_index[expansion_keys[node, j]]
            weights[start + j] = scalar_type(expansion_weights[node, j])

    return emit_prolongation


def _prolongation_matrix(prolongation: _Prolongation, row_count: int) -> BsrMatrix:
    """Wrap a CSR prolongation in a compact BSR matrix with scalar blocks; the matrix owns copies."""
    device = prolongation.offsets.device
    matrix = bsr_zeros(row_count, prolongation.column_count, block_type=prolongation.weights.dtype, device=device)
    wp.copy(dest=matrix.offsets, src=prolongation.offsets, count=row_count + 1)
    matrix.columns = wp.clone(prolongation.indices)
    matrix.values = wp.clone(prolongation.weights)
    matrix.notify_nnz_changed(nnz=prolongation.nnz)
    return matrix


# Hanging nodes


def _check_hanging_node_basis(raw_basis: BasisSpace):
    if not isinstance(raw_basis, ShapeBasisSpace) or raw_basis.value != ShapeFunction.Value.Scalar:
        raise NotImplementedError("Hanging-node elimination requires a scalar polynomial basis")
    if isinstance(raw_basis.topology, RegularDiscontinuousSpaceTopologyMixin):
        raise NotImplementedError("Hanging-node elimination requires a continuous basis")
    if not raw_basis.shape.INTERPOLATORY:
        raise NotImplementedError(
            "Hanging-node elimination requires an interpolatory basis, whose functions are one at their own node "
            "and zero at the others"
        )


def _hanging_node_tolerance(scalar_type) -> float:
    """Tolerance on interpolation weights evaluated in the basis scalar type.

    Functions that vanish at a point only pick up rounding errors there, while
    genuine weights between levels several apart can be small, so the tolerance
    stays a small multiple of the machine epsilon.
    """
    return float(100.0 * np.finfo(wp.dtype_to_numpy(scalar_type)).eps)


@wp.func
def _facet_within(
    inner_cell: wp.array[int],
    outer_cell: wp.array[int],
    facet_side_offsets: wp.array[int],
    facet_side_indices: wp.array[int],
    facet_key: int,
    cell: int,
    other: int,
):
    """Whether every side covering facet ``facet_key`` of ``cell`` leads to ``other``."""
    for i in range(facet_side_offsets[facet_key], facet_side_offsets[facet_key + 1]):
        side = facet_side_indices[i]
        neighbor = inner_cell[side]
        if neighbor == cell:
            neighbor = outer_cell[side]
        if neighbor != other:
            return False
    return True


def _hanging_node_kernels(raw_basis: ShapeBasisSpace):
    """Kernels finding hanging nodes and their interpolation weights.

    Besides the side-facet incidence of the geometry, they only rely on the
    generic geometry and basis interfaces: a node lies on a side when
    ``side_from_cell_coords`` does not return ``OUTSIDE``, and weights come from
    the basis functions of the neighboring cell.
    """
    geometry = raw_basis.geometry
    topology = raw_basis.topology
    node_coords_in_element = raw_basis.node_coords_in_element
    element_inner_weight = raw_basis.element_inner_weight

    @cache.dynamic_func(suffix=raw_basis.name)
    def side_coords_of_node(
        side_arg: geometry.SideArg,
        topo_arg: topology.TopologyArg,
        basis_arg: raw_basis.BasisArg,
        side: int,
        cell: int,
        slot: int,
    ):
        coords = node_coords_in_element(geometry.side_to_cell_arg(side_arg), topo_arg, basis_arg, cell, slot)
        return geometry.side_from_cell_coords(side_arg, side, cell, coords)

    @cache.dynamic_func(suffix=raw_basis.name)
    def side_node_count(
        side_arg: geometry.SideArg,
        topo_arg: topology.TopologyArg,
        basis_arg: raw_basis.BasisArg,
        side: int,
        cell: int,
    ):
        count = int(0)
        for slot in range(topology.element_node_count(geometry.side_to_cell_arg(side_arg), topo_arg, cell)):
            if side_coords_of_node(side_arg, topo_arg, basis_arg, side, cell, slot)[0] > -1.0e7:
                count += 1
        return count

    @cache.dynamic_kernel(suffix=raw_basis.name, kernel_options={"enable_backward": False})
    def classify_sides(
        side_arg: geometry.SideArg,
        topo_arg: topology.TopologyArg,
        basis_arg: raw_basis.BasisArg,
        inner_cell: wp.array[int],
        outer_cell: wp.array[int],
        inner_facet: wp.array[int],
        outer_facet: wp.array[int],
        facet_side_offsets: wp.array[int],
        facet_side_indices: wp.array[int],
        facets_per_cell: int,
        fine_cell: wp.array[int],
        coarse_cell: wp.array[int],
        hanging: wp.array[int],
        status: wp.array[int],
    ):
        # Facets of neighboring cells are nested: all the sides covering the
        # finer neighbor's facet lead to the coarser neighbor, while the
        # coarser neighbor's facet also has sides leading to other cells
        side = wp.tid()
        hanging[side] = 0
        if side == fine_cell.shape[0]:
            return
        fine_cell[side] = -1
        inner = inner_cell[side]
        outer = outer_cell[side]
        if outer < 0:
            return
        inner_within = _facet_within(
            inner_cell,
            outer_cell,
            facet_side_offsets,
            facet_side_indices,
            inner * facets_per_cell + inner_facet[side],
            inner,
            outer,
        )
        outer_within = _facet_within(
            inner_cell,
            outer_cell,
            facet_side_offsets,
            facet_side_indices,
            outer * facets_per_cell + outer_facet[side],
            outer,
            inner,
        )
        if inner_within == outer_within:
            return
        if inner_within:
            fine_cell[side] = inner
            coarse_cell[side] = outer
        else:
            fine_cell[side] = outer
            coarse_cell[side] = inner
        hanging[side] = 1
        wp.atomic_max(status, _SIDE_NODE_MAX, side_node_count(side_arg, topo_arg, basis_arg, side, fine_cell[side]))

    @cache.dynamic_kernel(suffix=raw_basis.name, kernel_options={"enable_backward": False})
    def hanging_candidates(
        side_arg: geometry.SideArg,
        topo_arg: topology.TopologyArg,
        basis_arg: raw_basis.BasisArg,
        raw_nodes: wp.array2d[int],
        fine_cell: wp.array[int],
        coarse_cell: wp.array[int],
        hanging_index: wp.array[int],
        tolerance: wp.float64,
        candidate_count: wp.array[int],
        candidate_raw: wp.array[int],
        candidate_masters: wp.array2d[int],
        candidate_weights: wp.array2d[wp.float64],
        canonical: wp.array[int],
        status: wp.array[int],
    ):
        # One candidate per node of the finer cell lying on the side: the value of
        # the coarser cell's interpolant at that node
        side, ordinal = wp.tid()
        fine = fine_cell[side]
        if fine < 0:
            return
        key = hanging_index[side] * candidate_masters.shape[1] + ordinal
        candidate_count[key] = 0

        cell_arg = geometry.side_to_cell_arg(side_arg)
        slot = int(-1)
        seen = int(0)
        for k in range(topology.element_node_count(cell_arg, topo_arg, fine)):
            if slot < 0 and side_coords_of_node(side_arg, topo_arg, basis_arg, side, fine, k)[0] > -1.0e7:
                if seen == ordinal:
                    slot = k
                seen += 1
        if slot < 0:
            return

        side_coords = side_coords_of_node(side_arg, topo_arg, basis_arg, side, fine, slot)
        coarse = coarse_cell[side]
        if coarse == geometry.side_inner_cell_index(side_arg, side):
            coarse_coords = geometry.side_inner_cell_coords(side_arg, side, side_coords)
        else:
            coarse_coords = geometry.side_outer_cell_coords(side_arg, side, side_coords)

        raw_node = raw_nodes[fine, slot]
        count = int(0)
        total = wp.float64(0.0)
        for k in range(topology.element_node_count(cell_arg, topo_arg, coarse)):
            weight = wp.float64(
                element_inner_weight(cell_arg, topo_arg, basis_arg, coarse, coarse_coords, k, NULL_QP_INDEX)
            )
            if wp.abs(weight) > tolerance:
                total += weight
                count = _add_sorted_entry(
                    candidate_masters, candidate_weights, key, count, raw_nodes[coarse, k], weight
                )
                if count < 0:
                    wp.atomic_min(status, _PROJECTION_ERROR, side)
                    return

        if wp.abs(total - wp.float64(1.0)) > tolerance:
            wp.atomic_min(status, _PROJECTION_ERROR, side)
            return
        # Nodes shared with the coarser cell are not constrained
        if count == 1 and candidate_masters[key, 0] == raw_node:
            return

        candidate_count[key] = count
        candidate_raw[key] = raw_node
        # The candidate with the smallest key defines the node
        wp.atomic_min(canonical, raw_node, key)

    return classify_sides, hanging_candidates


@wp.kernel(enable_backward=False)
def _store_hanging_side_count(hanging_index: wp.array[int], status: wp.array[int]):
    status[_HANGING_SIDE_COUNT] = hanging_index[hanging_index.shape[0] - 1]


def _hanging_node_constraints(raw_basis: BasisSpace) -> tuple[_Prolongation, wp.array]:
    """Build the hanging-node prolongation of a continuous Lagrange basis.

    Returns the prolongation and the raw node indices of each cell.
    """
    _check_hanging_node_basis(raw_basis)

    geometry = raw_basis.geometry
    topology = raw_basis.topology
    incidence = geometry.side_facet_incidence()
    device = incidence.inner_cell.device
    cell_count = geometry.cell_count()
    side_count = geometry.side_count()
    raw_count = topology.node_count()
    max_nodes = topology.MAX_NODES_PER_ELEMENT
    tolerance = _hanging_node_tolerance(geometry.scalar_type)
    classify_sides, hanging_candidates = _hanging_node_kernels(raw_basis)

    side_arg = geometry.side_arg_value(device)
    topo_arg = topology.topo_arg_value(device)
    basis_arg = raw_basis.basis_arg_value(device)
    raw_nodes = topology.element_node_indices(out=wp.empty((cell_count, max_nodes), dtype=int, device=device))
    status = _new_status(device)

    # Sides between cells of different sizes, and their finer neighbor
    fine_cell = wp.empty(side_count, dtype=int, device=device)
    coarse_cell = wp.empty(side_count, dtype=int, device=device)
    hanging = wp.empty(side_count + 1, dtype=int, device=device)
    hanging_index = wp.empty(side_count + 1, dtype=int, device=device)
    wp.launch(
        classify_sides,
        dim=side_count + 1,
        inputs=[
            side_arg,
            topo_arg,
            basis_arg,
            incidence.inner_cell,
            incidence.outer_cell,
            incidence.inner_facet,
            incidence.outer_facet,
            incidence.facet_side_offsets,
            incidence.facet_side_indices,
            incidence.facets_per_cell,
            fine_cell,
            coarse_cell,
            hanging,
            status,
        ],
        device=device,
    )
    array_scan(hanging, hanging_index, inclusive=False)
    wp.launch(_store_hanging_side_count, dim=1, inputs=[hanging_index, status], device=device)

    values = status.numpy()
    hanging_count = int(values[_HANGING_SIDE_COUNT])
    side_node_max = max(int(values[_SIDE_NODE_MAX]), 1)

    # Constraints of the finer cell's nodes on hanging sides, one per (side, node)
    candidate_total = hanging_count * side_node_max
    candidates = _Candidates(
        count=wp.empty(candidate_total, dtype=int, device=device),
        raw=wp.empty(candidate_total, dtype=int, device=device),
        masters=wp.empty((candidate_total, side_node_max), dtype=int, device=device),
        weights=wp.empty((candidate_total, side_node_max), dtype=wp.float64, device=device),
        canonical=wp.empty(raw_count, dtype=int, device=device),
    )
    candidates.canonical.fill_(_NO_KEY)
    wp.launch(
        hanging_candidates,
        dim=(side_count, side_node_max),
        inputs=[
            side_arg,
            topo_arg,
            basis_arg,
            raw_nodes,
            fine_cell,
            coarse_cell,
            hanging_index,
            wp.float64(tolerance),
            candidates.count,
            candidates.raw,
            candidates.masters,
            candidates.weights,
            candidates.canonical,
            status,
        ],
        device=device,
    )
    projection_error = int(status.numpy()[_PROJECTION_ERROR])
    if projection_error != _NO_KEY:
        raise ValueError(
            f"Invalid hanging-node interpolation across side {projection_error}: the sides of the geometry must "
            "cover the facets of their cells"
        )

    # A master of a hanging node lies on a strictly coarser cell, so chains are
    # short; more rounds run while they make progress
    prolongation = _resolve_constraints(
        candidates,
        round_count=2,
        status=status,
        scalar_type=geometry.scalar_type,
        device=device,
        kind="hanging-node",
        node_name="raw node",
        tolerance=tolerance,
    )
    return prolongation, raw_nodes


def hanging_node_prolongation(raw_basis: BasisSpace) -> BsrMatrix:
    """Return the prolongation that eliminates the hanging nodes of a continuous Lagrange basis.

    The geometry describes which sides cover each cell facet through
    :meth:`Geometry.side_facet_incidence`, as :class:`AdaptiveNanogrid` does.
    A side that covers a whole facet of one neighbor but only part of a facet
    of the other is a coarse-to-fine interface. Each node of the finer cell on
    such a side takes the value of the coarser cell's interpolant at that node;
    chains of such constraints across several levels are resolved. Other raw
    nodes are independent. Passing the result to
    :func:`make_constrained_basis_space` gives the same space as
    :func:`make_conforming_basis_space`.

    The basis must be interpolatory: each function is one at its own node and
    zero at the others, as for Lagrange polynomials of any degree. Nothing else
    depends on the element type.

    Args:
        raw_basis: Continuous scalar interpolatory basis on a static geometry
            that implements :meth:`Geometry.side_facet_incidence`.

    Returns:
        A compact BSR matrix with scalar blocks of the geometry scalar type,
        one row per node of ``raw_basis`` and one column per independent node.
        Independent nodes are numbered in increasing raw node order, and each
        has a unit row.

    Raises:
        NotImplementedError: If ``raw_basis`` is not a continuous, scalar and
            interpolatory basis, or if its geometry does not implement
            :meth:`Geometry.side_facet_incidence`.
        ValueError: If the interpolation from a coarser cell is invalid, which
            happens when the geometry's sides do not cover cell facets, or if
            constraints are inconsistent or cyclic.
    """
    prolongation, _raw_nodes = _hanging_node_constraints(raw_basis)
    return _prolongation_matrix(prolongation, raw_basis.topology.node_count())


# Periodic constraints


_HASH_GRID_DIM = 128
"""Hash grid resolution along each axis for periodic node matching."""

_DEFAULT_PERIODIC_TOLERANCE = 1.0e-6
"""Default matching tolerance, relative to the bounding-box diagonal of the nodes."""


def _periodic_kernels(vec_type):
    """Kernels for bounds and periodic matching of node positions of type ``vec_type``."""
    dimension = vec_type._length_
    scalar = vec_type._wp_scalar_type_
    vec3 = cache.cached_vec_type(length=3, dtype=scalar)

    @cache.dynamic_func(suffix=f"{dimension}_{scalar.__name__}")
    def to_vec3(x: vec_type):
        if wp.static(dimension == 2):
            return vec3(x[0], x[1], scalar(0.0))
        return vec3(x[0], x[1], x[2])

    @cache.dynamic_kernel(suffix=f"{dimension}_{scalar.__name__}", kernel_options={"enable_backward": False})
    def node_bounds(
        positions: wp.array[vec_type],
        points: wp.array[vec3],
        lower: wp.array[scalar],
        upper: wp.array[scalar],
    ):
        node = wp.tid()
        x = positions[node]
        points[node] = to_vec3(x)
        for axis in range(dimension):
            wp.atomic_min(lower, axis, x[axis])
            wp.atomic_max(upper, axis, x[axis])

    @cache.dynamic_func(suffix=f"{dimension}_{scalar.__name__}")
    def closest_node(
        grid: wp.uint64,
        positions: wp.array[vec_type],
        node_environments: wp.array[int],
        environment: int,
        target: vec_type,
        tolerance: scalar,
    ):
        partner = int(-1)
        best = tolerance * tolerance
        query = wp.hash_grid_query(grid, to_vec3(target), tolerance)
        other = int(0)
        while wp.hash_grid_query_next(query, other):
            # Environments share local coordinates, so only nodes of the same one match
            if node_environments[other] != environment:
                continue
            offset = positions[other] - target
            distance = wp.dot(offset, offset)
            # Closest node within tolerance; ties go to the smallest index
            if distance < best or (distance == best and (partner < 0 or other < partner)):
                best = distance
                partner = other
        return partner

    @cache.dynamic_kernel(suffix=f"{dimension}_{scalar.__name__}", kernel_options={"enable_backward": False})
    def match_periodic_nodes(
        grid: wp.uint64,
        positions: wp.array[vec_type],
        node_environments: wp.array[int],
        translations: wp.array[vec_type],
        tolerance: scalar,
        candidate_count: wp.array[int],
        candidate_raw: wp.array[int],
        candidate_masters: wp.array2d[int],
        candidate_weights: wp.array2d[wp.float64],
        canonical: wp.array[int],
    ):
        node, t = wp.tid()
        translation_count = translations.shape[0]
        key = node * translation_count + t
        candidate_count[key] = 0

        # A node is constrained by a translation exactly when a node exists at its
        # translated position. In a cell that tiles space with the translations, only
        # nodes on the face that the translation maps onto have such a partner, whatever
        # the direction of the translation relative to the faces.
        x = positions[node]
        translation = translations[t]
        partner = closest_node(grid, positions, node_environments, node_environments[node], x - translation, tolerance)
        if partner < 0:
            return

        candidate_count[key] = 1
        candidate_raw[key] = node
        candidate_masters[key, 0] = partner
        candidate_weights[key, 0] = wp.float64(1.0)
        # The candidate of the first translation defines the node
        wp.atomic_min(canonical, node, key)

    return node_bounds, match_periodic_nodes


def _node_environments(basis: BasisSpace, device) -> wp.array:
    """Return the environment index of each node of ``basis``."""
    node_count = basis.topology.node_count()
    if basis.geometry.environment_count() == 1:
        return wp.zeros(node_count, dtype=int, device=device)

    geometry = basis.geometry
    topology = basis.topology

    @cache.dynamic_kernel(suffix=basis.name, kernel_options={"enable_backward": False})
    def fill_node_environments(
        geo_cell_arg: geometry.CellArg,
        topo_arg: topology.TopologyArg,
        node_environments: wp.array[int],
    ):
        cell = wp.tid()
        environment = geometry.cell_environment_index(geo_cell_arg, cell)
        for n in range(topology.element_node_count(geo_cell_arg, topo_arg, cell)):
            node_environments[topology.element_node_index(geo_cell_arg, topo_arg, cell, n)] = environment

    node_environments = wp.empty(node_count, dtype=int, device=device)
    wp.launch(
        fill_node_environments,
        dim=geometry.cell_count(),
        inputs=[geometry.cell_arg_value(device=device), topology.topo_arg_value(device=device), node_environments],
        device=device,
    )
    return node_environments


def _basis_device(basis: BasisSpace):
    """Device holding the arrays of ``basis`` or of its geometry, or the current device for grids."""
    independent_raw_nodes = getattr(basis, "independent_raw_nodes", None)
    if independent_raw_nodes is not None:
        return independent_raw_nodes.device

    geometry = basis.geometry
    # Deformed geometries read their positions from a field
    field = getattr(geometry, "field", None)
    if field is not None:
        return field.dof_values.device
    # Meshes hold vertex positions, Nanogrids a sparse volume
    for attr in ("positions", "cell_grid"):
        data = getattr(geometry, attr, None)
        if data is not None:
            return data.device
    return wp.get_device()


def periodic_prolongation(
    basis: BasisSpace,
    translations: Sequence[Any],
    tolerance: float | None = None,
) -> BsrMatrix:
    """Return a prolongation that ties the nodes of ``basis`` to their periodic copies.

    For each translation ``T``, every node ``x`` that has a partner node at
    ``x - T`` takes the value of that partner. When the domain is a periodic
    cell that tiles space with the given translations, these are exactly the
    nodes on the faces that the translations map onto, whether or not the
    translations are perpendicular to the faces (for instance, the two edge
    vectors of a parallelogram). Nodes tied by several translations, such as
    edges and corners, resolve to a single copy. Nodes are matched by position,
    so any basis that provides :meth:`BasisSpace.node_positions` works,
    including a constrained basis returned by
    :func:`make_conforming_basis_space`, whose nodes are its independent nodes.
    On geometries with several environments, nodes only match nodes of the
    same environment.

    Only nodes with a partner are tied: this function does not check that
    paired faces carry matching nodes, and nodes without a partner, for
    instance on a face refined more than its partner, stay independent.

    Pass the result to :func:`make_constrained_basis_space` with the same
    ``basis`` to build the periodic space. The geometry itself is unchanged:
    paired faces remain boundary sides, so jumps across the seam are not
    available and positions are not wrapped.

    Matching runs with a :class:`warp.HashGrid` on the device holding the
    geometry's arrays (or those of a constrained ``basis``), and on the
    current device for :class:`.Grid2D` and :class:`.Grid3D`; only the node
    bounding box and error status are read back.

    Args:
        basis: Basis whose nodes are made periodic.
        translations: One to three translation vectors with the dimension of
            the geometry, such as ``wp.vec2(1.0, 0.0)``. Each one maps the low
            face of the periodic domain to its high face.
        tolerance: Distance under which two nodes match. ``None`` uses
            ``1e-6`` times the diagonal of the node bounding box.

    Returns:
        A compact BSR matrix with scalar blocks of the geometry scalar type,
        one row per node of ``basis`` and one column per independent node.
        Independent nodes are numbered in increasing node order, and each has
        a unit row.

    Raises:
        ValueError: If a translation is zero or has the wrong dimension, if
            ``tolerance`` is not positive or not smaller than half of every
            translation length, or if the translations lead to cyclic or
            inconsistent constraints, for instance when both ``T`` and ``-T``
            are given.
    """
    geometry = basis.geometry
    dimension = geometry.dimension
    scalar_type = geometry.scalar_type
    vec_type = cache.cached_vec_type(length=dimension, dtype=scalar_type)

    translation_values = []
    for translation in translations:
        values = [float(component) for component in translation]
        if len(values) != dimension:
            raise ValueError(f"Translations must have {dimension} components, got {values}")
        translation_values.append(values)
    translation_values = np.array(translation_values, dtype=np.float64).reshape(-1, dimension)
    translation_count = translation_values.shape[0]
    if translation_count == 0:
        raise ValueError("At least one translation is required")
    lengths = np.linalg.norm(translation_values, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("Translations must be nonzero")

    device = _basis_device(basis)
    node_count = basis.topology.node_count()
    with wp.ScopedDevice(device):
        positions = basis.node_positions(out=wp.empty(node_count, dtype=vec_type, device=device))
    node_environments = _node_environments(basis, device)

    node_bounds, match_periodic_nodes = _periodic_kernels(vec_type)
    vec3 = cache.cached_vec_type(length=3, dtype=scalar_type)
    translation_array = wp.array(translation_values, dtype=vec_type, device=device)
    points = wp.empty(node_count, dtype=vec3, device=device)
    lowest = float(np.finfo(wp.dtype_to_numpy(scalar_type)).min)
    lower = wp.full(dimension, value=-lowest, dtype=scalar_type, device=device)
    upper = wp.full(dimension, value=lowest, dtype=scalar_type, device=device)
    wp.launch(
        node_bounds,
        dim=node_count,
        inputs=[positions, points, lower, upper],
        device=device,
    )

    diagonal = float(np.linalg.norm(upper.numpy().astype(np.float64) - lower.numpy().astype(np.float64)))
    if tolerance is None:
        tolerance = _DEFAULT_PERIODIC_TOLERANCE * diagonal
    tolerance = float(tolerance)
    if not tolerance > 0.0:
        raise ValueError(f"Periodic matching tolerance must be positive, got {tolerance}")
    if tolerance >= 0.5 * float(np.min(lengths)):
        raise ValueError(
            f"Periodic matching tolerance {tolerance} must be smaller than half of the shortest translation length"
        )

    # Cells larger than the tolerance keep the grid coordinates of distant nodes small
    grid = wp.HashGrid(_HASH_GRID_DIM, _HASH_GRID_DIM, _HASH_GRID_DIM, device=device, dtype=scalar_type)
    grid.build(points, radius=max(2.0 * tolerance, 1.0e-5 * diagonal))

    status = _new_status(device)
    candidate_total = node_count * translation_count
    candidates = _Candidates(
        count=wp.empty(candidate_total, dtype=int, device=device),
        raw=wp.empty(candidate_total, dtype=int, device=device),
        masters=wp.empty((candidate_total, 1), dtype=int, device=device),
        weights=wp.empty((candidate_total, 1), dtype=wp.float64, device=device),
        canonical=wp.empty(node_count, dtype=int, device=device),
    )
    candidates.canonical.fill_(_NO_KEY)
    wp.launch(
        match_periodic_nodes,
        dim=(node_count, translation_count),
        inputs=[
            grid.id,
            positions,
            node_environments,
            translation_array,
            scalar_type(tolerance),
            candidates.count,
            candidates.raw,
            candidates.masters,
            candidates.weights,
            candidates.canonical,
        ],
        device=device,
    )

    # Edge and corner nodes chain through one constraint per translation
    prolongation = _resolve_constraints(
        candidates,
        round_count=translation_count + 1,
        status=status,
        scalar_type=scalar_type,
        device=device,
        kind="periodic",
    )
    return _prolongation_matrix(prolongation, node_count)
