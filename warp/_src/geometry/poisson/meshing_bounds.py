# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative convex-hull certificates for the reference degree-one field.

Selector cells own (lower, upper] on each axis. Within one selector leaf the
field is the sum of its correction, the parent's cumulative polynomial, and
the next level's correction. Subdivision at the next level's knots makes this
sum trilinear; its Bernstein coefficients are its eight corner values. Bounds
are reduced up the selector hierarchy. Closed extraction cells additionally
include their lower-face neighbors to respect the evaluator's knot ownership.
"""

import warp as wp
from warp.utils import array_scan

from .field import basis_key, reference_evaluation_function
from .keyed import coordinates

wp.set_module_options({"fuse_fp": False, "enable_backward": False})

_coefficients_type = wp.types.vector(length=8, dtype=wp.float64)
_child_indices_type = wp.types.vector(length=8, dtype=int)
_evaluate_reference = reference_evaluation_function(wp.float32)


@wp.kernel
def sample_cube_kernel(
    keys: wp.array[wp.int64],
    coefficients: wp.array[float],
    cumulative: wp.array[float],
    points: wp.array[wp.vec3],
    depth: int,
    iso: wp.array[float],
    values: wp.array[float],
):
    index = wp.tid()
    value, _gradient, _valid = _evaluate_reference(
        keys,
        coefficients,
        cumulative,
        points[index],
        wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        wp.vec3(0.0),
        depth,
    )
    values[index] = value - iso[0]


@wp.struct
class _KeyLookup:
    keys: wp.array[wp.int64]
    slots: wp.array[int]


@wp.func
def _hash_slot(key: wp.int64, mask: int):
    value = wp.uint64(key)
    value = (value ^ (value >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    return int((value ^ (value >> wp.uint64(31))) & wp.uint64(mask))


@wp.func
def _find_key(lookup: _KeyLookup, key: wp.int64):
    mask = lookup.slots.shape[0] - 1
    slot = _hash_slot(key, mask)
    while True:
        index = lookup.slots[slot]
        if index < 0:
            return -1
        if lookup.keys[index] == key:
            return index
        slot = (slot + 1) & mask


@wp.kernel
def _build_lookup_kernel(lookup: _KeyLookup):
    index = wp.tid()
    mask = lookup.slots.shape[0] - 1
    slot = _hash_slot(lookup.keys[index], mask)
    while wp.atomic_cas(lookup.slots, slot, -1, index) != -1:
        slot = (slot + 1) & mask


@wp.func
def _offset(index: int):
    return wp.vec3i(index // 4, (index // 2) % 2, index % 2)


@wp.func
def _ancestor(cell: wp.vec3i, levels: int):
    return wp.vec3i(cell[0] >> levels, cell[1] >> levels, cell[2] >> levels)


@wp.func
def _load_coefficients(lookup: _KeyLookup, values: wp.array[float], depth: int, cell: wp.vec3i):
    coefficients = _coefficients_type(wp.float64(0))
    for corner in range(8):
        node = cell + _offset(corner)
        index = _find_key(lookup, basis_key(depth, node[0], node[1], node[2]))
        if index >= 0:
            coefficients[corner] = wp.float64(values[index])
    return coefficients


@wp.func
def _interpolate(coefficients: _coefficients_type, fraction: wp.vec3d):
    return wp.lerp(
        wp.lerp(
            wp.lerp(coefficients[0], coefficients[1], fraction[2]),
            wp.lerp(coefficients[2], coefficients[3], fraction[2]),
            fraction[1],
        ),
        wp.lerp(
            wp.lerp(coefficients[4], coefficients[5], fraction[2]),
            wp.lerp(coefficients[6], coefficients[7], fraction[2]),
            fraction[1],
        ),
        fraction[0],
    )


@wp.func
def _rounded_bounds(lower: wp.float64, upper: wp.float64, magnitude: wp.float64):
    # Enclose float32 weight/product/sum rounding (three eight-term sums),
    # double bound construction, and subnormal flushing. 128 float32 eps is
    # larger than gamma_64; the positive weights sum to one in each piece.
    if magnitude == wp.float64(0):
        return wp.vec2d(wp.float64(0))
    error = magnitude * wp.float64(128.0 * 1.1920928955078125e-7) + wp.float64(1.0e-35)
    if magnitude + error > wp.float64(3.4028234663852886e38):
        return wp.vec2d(wp.float64(-wp.inf), wp.float64(wp.inf))
    return wp.vec2d(lower - error, upper + error)


@wp.func
def _finest_children_bounds(
    lookup: _KeyLookup, coefficients: wp.array[float], cumulative: wp.array[float], depth: int, cell: wp.vec3i
):
    parent = _load_coefficients(lookup, cumulative, depth, cell)
    parent_magnitude = wp.max(wp.abs(parent))
    fine_magnitude = wp.float64(0)
    lower = wp.float64(wp.inf)
    upper = wp.float64(-wp.inf)
    # The parent and fine correction each have a convex-hull certificate.
    # Adding these intervals avoids interpolating 27 overlapping polynomials;
    # cancellation may leave the result uncertain, never falsely certified.
    for index in range(27):
        offset = wp.vec3i(index // 9, (index // 3) % 3, index % 3)
        node = cell * 2 + offset
        node_index = _find_key(lookup, basis_key(depth + 1, node[0], node[1], node[2]))
        value = wp.float64(0)
        if node_index >= 0:
            value = wp.float64(coefficients[node_index])
        fine_magnitude = wp.max(fine_magnitude, wp.abs(value))
        lower = wp.min(lower, value)
        upper = wp.max(upper, value)
    return _rounded_bounds(wp.min(parent) + lower, wp.max(parent) + upper, parent_magnitude + fine_magnitude)


@wp.func
def _polynomial_bounds(
    lookup: _KeyLookup,
    coefficients: wp.array[float],
    cumulative: wp.array[float],
    leaf: int,
    cell: wp.vec3i,
    cell_depth: int,
    field_depth: int,
):
    leaf_cell = _ancestor(cell, cell_depth - leaf)
    current = _load_coefficients(lookup, coefficients, leaf, leaf_cell)
    parent = _coefficients_type(wp.float64(0))
    if leaf > 0:
        parent = _load_coefficients(lookup, cumulative, leaf - 1, _ancestor(leaf_cell, 1))
    fine = _coefficients_type(wp.float64(0))
    fine_cell = wp.vec3i(0)
    if leaf < field_depth and cell_depth > leaf:
        fine_cell = _ancestor(cell, cell_depth - leaf - 1)
        fine = _load_coefficients(lookup, coefficients, leaf + 1, fine_cell)

    # Restrict the coarse polynomial to eight corners. A next-level correction
    # that cuts the cell has its own 27-value hull, added as an interval below.
    lower = wp.float64(wp.inf)
    upper = wp.float64(-wp.inf)
    coarse_magnitude = wp.max(wp.abs(current)) + wp.max(wp.abs(parent))
    magnitude = coarse_magnitude + wp.max(wp.abs(fine))
    for index in range(8):
        offset = wp.vec3d(_offset(index))
        point = (wp.vec3d(cell) + offset) / wp.float64(1 << cell_depth)
        value = _interpolate(current, point * wp.float64(1 << leaf) - wp.vec3d(leaf_cell))
        if cell_depth == leaf:
            value = current[index]
        if leaf > 0:
            value += _interpolate(parent, point * wp.float64(1 << (leaf - 1)) - wp.vec3d(_ancestor(leaf_cell, 1)))
        if leaf < field_depth and cell_depth > leaf:
            value += _interpolate(fine, point * wp.float64(1 << (leaf + 1)) - wp.vec3d(fine_cell))
        lower = wp.min(lower, value)
        upper = wp.max(upper, value)

    if leaf < field_depth and cell_depth == leaf:
        fine_lower = wp.float64(wp.inf)
        fine_upper = wp.float64(-wp.inf)
        for index in range(27):
            node = cell * 2 + wp.vec3i(index // 9, (index // 3) % 3, index % 3)
            node_index = _find_key(lookup, basis_key(leaf + 1, node[0], node[1], node[2]))
            value = wp.float64(0)
            if node_index >= 0:
                value = wp.float64(coefficients[node_index])
            magnitude = wp.max(magnitude, coarse_magnitude + wp.abs(value))
            fine_lower = wp.min(fine_lower, value)
            fine_upper = wp.max(fine_upper, value)
        lower += fine_lower
        upper += fine_upper

    # Boundary nudges remain inside the first/last finest cell (depth <= 15).
    return _rounded_bounds(lower, upper, magnitude)


@wp.kernel
def _build_bounds_kernel(
    lookup: _KeyLookup,
    coefficients: wp.array[float],
    cumulative: wp.array[float],
    field_depth: int,
    bound_depth: int,
    depth: int,
    bounds: wp.array[wp.vec2d],
    status: wp.array[int],
):
    index = wp.tid()
    key = lookup.keys[index]
    key_depth = int(key >> wp.int64(48))
    cell = coordinates(key)
    if depth == bound_depth:
        if not wp.isfinite(coefficients[index]) or not wp.isfinite(cumulative[index]):
            wp.atomic_or(status, 1, 2)
        if key_depth > 0 and wp.max(cell) < 1 << key_depth:
            parent = _ancestor(cell, 1)
            if _find_key(lookup, basis_key(key_depth - 1, parent[0], parent[1], parent[2])) < 0:
                wp.atomic_or(status, 1, 1)
    if key_depth != depth:
        return
    if wp.max(cell) >= 1 << depth:
        return
    lower = wp.float64(wp.inf)
    upper = wp.float64(-wp.inf)
    children = int(0)
    child_indices = _child_indices_type(-1)
    if depth < field_depth:
        for corner in range(8):
            child = cell * 2 + _offset(corner)
            child_index = _find_key(lookup, basis_key(depth + 1, child[0], child[1], child[2]))
            child_indices[corner] = child_index
            if child_index >= 0:
                children += 1
        if depth == field_depth - 1 and bound_depth < field_depth and children == 8:
            bound = _finest_children_bounds(lookup, coefficients, cumulative, depth, cell)
            lower, upper = bound[0], bound[1]
        else:
            for corner in range(8):
                if child_indices[corner] >= 0:
                    bound = wp.vec2d(wp.float64(0))
                    if depth == field_depth - 1 and bound_depth < field_depth:
                        child = cell * 2 + _offset(corner)
                        bound = _polynomial_bounds(
                            lookup, coefficients, cumulative, depth + 1, child, depth + 1, field_depth
                        )
                    else:
                        bound = bounds[child_indices[corner]]
                    lower = wp.min(lower, bound[0])
                    upper = wp.max(upper, bound[1])
    if children != 8:
        bound = _polynomial_bounds(lookup, coefficients, cumulative, depth, cell, depth, field_depth)
        lower = wp.min(lower, bound[0])
        upper = wp.max(upper, bound[1])
    bounds[index] = wp.vec2d(lower, upper)
    if not wp.isfinite(lower) or not wp.isfinite(upper):
        wp.atomic_or(status, 1, 2)


@wp.kernel
def _classify_bounds_kernel(
    lookup: _KeyLookup,
    coefficients: wp.array[float],
    cumulative: wp.array[float],
    field_depth: int,
    bounds: wp.array[wp.vec2d],
    cells: wp.array[wp.vec3i],
    depth: int,
    iso: wp.array[float],
    counts: wp.array[int],
    status: wp.array[int],
):
    index = wp.tid()
    lower = wp.float64(wp.inf)
    upper = wp.float64(-wp.inf)
    for neighbor in range(8):
        cell = cells[index] - _offset(neighbor)
        if wp.min(cell) >= 0:
            leaf = depth
            ancestor = cell
            node_index = _find_key(lookup, basis_key(leaf, ancestor[0], ancestor[1], ancestor[2]))
            while node_index < 0 and leaf > 0:
                leaf -= 1
                ancestor = _ancestor(ancestor, 1)
                node_index = _find_key(lookup, basis_key(leaf, ancestor[0], ancestor[1], ancestor[2]))
            bound = wp.vec2d(wp.float64(-wp.inf), wp.float64(wp.inf))
            if node_index >= 0:
                if leaf == depth:
                    bound = bounds[node_index]
                else:
                    bound = _polynomial_bounds(lookup, coefficients, cumulative, leaf, cell, depth, field_depth)
            lower = wp.min(lower, bound[0])
            upper = wp.max(upper, bound[1])
    # MC requires both a strictly negative and a nonnegative centered corner.
    uncertain = lower < wp.float64(iso[0]) and upper >= wp.float64(iso[0])
    counts[index] = 8 * int(uncertain)
    if not wp.isfinite(iso[0]):
        wp.atomic_or(status, 1, 2)


@wp.kernel
def _count_kernel(offsets: wp.array[int], status: wp.array[int]):
    status[0] = offsets[offsets.shape[0] - 1]


@wp.kernel
def _children_kernel(cells: wp.array[wp.vec3i], counts: wp.array[int], offsets: wp.array[int], out: wp.array[wp.vec3i]):
    index = wp.tid()
    if counts[index]:
        start = offsets[index] - 8
        for child in range(8):
            out[start + child] = cells[index] * 2 + _offset(child)


def candidate_cells(field, depth, iso, *, statistics=None):
    """Refine only cells whose closed bounds cannot exclude the isovalue."""
    device = field.coefficients.device
    capacity = 1 << (2 * len(field.keys) - 1).bit_length()
    if capacity > 2**31 - 1:
        raise ValueError("field lookup capacity exceeds int32 indexing")
    lookup = _KeyLookup()
    lookup.keys = field.keys
    lookup.slots = wp.full(capacity, -1, dtype=int, device=device)
    wp.launch(_build_lookup_kernel, len(field.keys), inputs=[lookup], device=device, record_tape=False)
    bounds = wp.empty(len(field.keys), dtype=wp.vec2d, device=device)
    status = wp.zeros(2, dtype=int, device=device)
    bound_depth = field.depth if depth > field.depth else max(0, field.depth - 1)
    for level in range(bound_depth, -1, -1):
        wp.launch(
            _build_bounds_kernel,
            len(field.keys),
            inputs=[lookup, field.coefficients, field.cumulative, field.depth, bound_depth, level, bounds, status],
            device=device,
            record_tape=False,
        )
    cells = wp.zeros(1, dtype=wp.vec3i, device=device)
    visits = []
    for level in range(depth):
        visits.append(len(cells))
        if len(cells) > (2**31 - 1) // 8:
            raise ValueError("candidate child count exceeds int32 indexing")
        counts = wp.empty(len(cells), dtype=int, device=device)
        offsets = wp.empty_like(counts)
        wp.launch(
            _classify_bounds_kernel,
            len(cells),
            inputs=[
                lookup,
                field.coefficients,
                field.cumulative,
                field.depth,
                bounds,
                cells,
                level,
                iso,
                counts,
                status,
            ],
            device=device,
            record_tape=False,
        )
        array_scan(counts, offsets, inclusive=True)
        wp.launch(_count_kernel, 1, inputs=[offsets, status], device=device, record_tape=False)
        count, invalid = map(int, status.numpy())
        if invalid & 1:
            raise ValueError("Poisson selector hierarchy must contain every cell's parent")
        if invalid & 2:
            raise ValueError("Poisson field bounds and isovalue must be finite")
        children = wp.empty(count, dtype=wp.vec3i, device=device)
        if count:
            wp.launch(
                _children_kernel,
                len(cells),
                inputs=[cells, counts, offsets, children],
                device=device,
                record_tape=False,
            )
        cells = children
        if not count:
            break
    if statistics is not None:
        statistics.update(visited=visits, candidates=len(cells), bound_nodes=len(field.keys), bound_depth=bound_depth)
    return cells
