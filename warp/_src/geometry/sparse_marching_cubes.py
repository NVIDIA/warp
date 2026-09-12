# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse marching cubes driven by an implicit function.

This module extracts an isosurface directly from a callable implicit function
(for example a signed distance function) without ever materializing a dense
grid. It first builds a *Lipschitz octree*, a sparse set of small voxels that
provably bracket the level set of a 1-Lipschitz function, and then runs
marching cubes only on those voxels.

The approach mirrors ``igl::lipschitz_octree`` plus the sparse-voxel overload of
``igl::marching_cubes`` from libigl. Octree construction, corner
de-duplication, and triangle extraction run on the GPU in pure Warp; field
evaluation calls a caller-supplied batched implicit function, which stays on
the GPU too as long as it is implemented in Warp.

For a dense field already resident in memory, use
:class:`warp.geometry.IsoSurfaceMarchingCubes` instead.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

import warp as wp
from warp._src import utils as _wp_utils
from warp._src.geometry.iso_surface import resolve_domain_bounds
from warp._src.geometry.marching_cubes import (
    MC_CUBE_CORNER_OFFSETS,
    MC_EDGE_TO_CORNERS,
    _get_mc_case_to_tri_range_table,
    _get_mc_tri_local_inds_table,
)

# Half the diagonal of a (possibly anisotropic) cell box: the maximum distance
# from a cell center to any point in the cell is half the box's diagonal
# length. A 1-Lipschitz function whose magnitude at the center exceeds that
# distance cannot reach its level set anywhere inside the cell.


# =============================================================================
# Static edge geometry, derived from the shared marching cubes tables
# =============================================================================


# For each of the 12 cube edges, determine (owner_corner, upper_corner, axis):
#   * ``axis`` is the coordinate (0=x, 1=y, 2=z) along which the edge runs.
#   * ``owner_corner`` is the edge endpoint with the smaller coordinate on that
#     axis; ``upper_corner`` is the other endpoint.
# The owner corner gives every edge a canonical identity shared by all cells
# that touch it, which is how we de-duplicate marching cubes vertices.
def _build_edge_geometry() -> tuple[tuple[int, int, int], ...]:
    geometry = []
    for corner_a, corner_b in MC_EDGE_TO_CORNERS:
        offset_a = MC_CUBE_CORNER_OFFSETS[corner_a]
        offset_b = MC_CUBE_CORNER_OFFSETS[corner_b]
        axis = next(a for a in range(3) if offset_a[a] != offset_b[a])
        if offset_a[axis] < offset_b[axis]:
            owner, upper = corner_a, corner_b
        else:
            owner, upper = corner_b, corner_a
        geometry.append((owner, upper, axis))
    return tuple(geometry)


_MC_EDGE_GEOMETRY: tuple[tuple[int, int, int], ...] = _build_edge_geometry()

# Flattened views suitable for wp.array creation: for edge e,
# owner = _MC_EDGE_OWNER[e], upper = _MC_EDGE_UPPER[e], axis = _MC_EDGE_AXIS[e].
_MC_EDGE_OWNER: tuple[int, ...] = tuple(g[0] for g in _MC_EDGE_GEOMETRY)
_MC_EDGE_UPPER: tuple[int, ...] = tuple(g[1] for g in _MC_EDGE_GEOMETRY)
_MC_EDGE_AXIS: tuple[int, ...] = tuple(g[2] for g in _MC_EDGE_GEOMETRY)

_mc_edge_owner_cache: dict[str, wp.array] = {}
_mc_edge_upper_cache: dict[str, wp.array] = {}
_mc_edge_axis_cache: dict[str, wp.array] = {}


def _get_edge_owner_table(device) -> wp.array:
    device = str(device)
    if device not in _mc_edge_owner_cache:
        _mc_edge_owner_cache[device] = wp.array(_MC_EDGE_OWNER, dtype=wp.int32, device=device)
    return _mc_edge_owner_cache[device]


def _get_edge_upper_table(device) -> wp.array:
    device = str(device)
    if device not in _mc_edge_upper_cache:
        _mc_edge_upper_cache[device] = wp.array(_MC_EDGE_UPPER, dtype=wp.int32, device=device)
    return _mc_edge_upper_cache[device]


def _get_edge_axis_table(device) -> wp.array:
    device = str(device)
    if device not in _mc_edge_axis_cache:
        _mc_edge_axis_cache[device] = wp.array(_MC_EDGE_AXIS, dtype=wp.int32, device=device)
    return _mc_edge_axis_cache[device]


# =============================================================================
# Implicit-function evaluation
# =============================================================================


def _make_evaluator(field, device) -> Callable[[wp.array], wp.array]:
    """Return a callable mapping a ``wp.array[wp.vec3]`` batch to a ``wp.array[wp.float32]``.

    ``field`` must already be a batched callable; a bare ``@wp.func`` taking a
    single point is rejected with a pointer to how to batch it, since silently
    wrapping it here would hide a per-point kernel launch inside what looks
    like a single call.
    """
    if isinstance(field, wp.Function):
        raise TypeError(
            "`field` must be a batched callable with signature "
            "evaluate(points: wp.array[wp.vec3]) -> wp.array[wp.float32], not a single-point "
            "@wp.func. Wrap it in a small kernel that evaluates it over the batch, e.g.:\n\n"
            "    @wp.kernel\n"
            "    def eval_kernel(points: wp.array[wp.vec3], values: wp.array[wp.float32]):\n"
            "        i = wp.tid()\n"
            "        values[i] = my_sdf(points[i])\n\n"
            "    def evaluate(points):\n"
            "        values = wp.empty(points.shape[0], dtype=wp.float32, device=points.device)\n"
            "        wp.launch(eval_kernel, dim=points.shape[0], inputs=[points], outputs=[values], "
            "device=points.device)\n"
            "        return values\n\n"
            "This keeps evaluation on the GPU; see warp/examples/geometry/example_sparse_marching_cubes.py."
        )

    if not callable(field):
        raise TypeError(
            "`field` must be a callable mapping a warp.array[wp.vec3] batch of query points to a "
            f"warp.array[wp.float32] of distances, but got {type(field)}."
        )

    def evaluate(points: wp.array) -> wp.array:
        values = field(points)
        if not isinstance(values, wp.array):
            raise TypeError(
                "The implicit function callable must return a warp.array of float32 distance values, "
                f"but returned {type(values)}."
            )
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                f"The implicit function returned {values.shape[0]} values for {points.shape[0]} query points."
            )
        # Reject rather than silently copy: the evaluator is called once per
        # octree level and once for the corners, so a hidden transfer here
        # would quietly dominate the runtime of an otherwise GPU-only pipeline.
        # A callable that itself round-trips through the host (e.g. wrapping a
        # NumPy or PyTorch evaluation) is still valid -- it just pays that sync
        # cost on every call, since the *returned* array must land back on
        # ``points.device``.
        if values.device != points.device:
            raise ValueError(
                f"The implicit function returned values on device '{values.device}', but the query points are "
                f"on device '{points.device}'. Evaluate the field on the device of the points it is given."
            )
        if values.dtype != wp.float32:
            raise TypeError(
                f"The implicit function must return a warp.array of float32 distance values, got {values.dtype}."
            )
        return values

    return evaluate


# =============================================================================
# Octree construction kernels
# =============================================================================


@wp.kernel(enable_backward=False)
def _compute_cell_centers_kernel(
    cells: wp.array[wp.vec3i],
    origin: wp.vec3,
    cell_width: wp.vec3,
    centers: wp.array[wp.vec3],
):
    """Compute the geometric center of each octree cell at the current depth."""
    tid = wp.tid()
    c = cells[tid]
    centers[tid] = origin + wp.cw_mul(
        cell_width,
        wp.vec3(wp.float32(c[0]) + 0.5, wp.float32(c[1]) + 0.5, wp.float32(c[2]) + 0.5),
    )


@wp.kernel(enable_backward=False)
def _mark_active_cells_kernel(
    values: wp.array[wp.float32],
    isovalue: wp.float32,
    band: wp.float32,
    keep: wp.array[wp.int32],
):
    """Flag cells whose subtree can still bracket the level set.

    A 1-Lipschitz field changes by at most ``band`` between the cell center and
    any point in the cell, so a center value farther than ``band`` from the
    isovalue rules out a crossing anywhere inside.
    """
    tid = wp.tid()
    keep[tid] = wp.where(wp.abs(values[tid] - isovalue) <= band, 1, 0)


@wp.kernel
def _cell_subscript_bounds_kernel(
    cells: wp.array[wp.vec3i],
    lo: wp.array[wp.int32],
    hi: wp.array[wp.int32],
):
    """Reduce the per-axis minimum and maximum of the cell subscripts.

    Backward stays enabled, even though this only ever touches the (integer,
    never grad-requiring) ``cells`` array, purely so Warp doesn't warn when
    this kernel is recorded on a tape alongside :func:`sparse_marching_cubes_from_cells`'s
    genuinely differentiable kernels; see ``_mark_active_edges_kernel``.
    """
    tid = wp.tid()
    c = cells[tid]
    for a in range(3):
        wp.atomic_min(lo, a, c[a])
        wp.atomic_max(hi, a, c[a])


@wp.kernel(enable_backward=False)
def _compact_cells_kernel(
    cells: wp.array[wp.vec3i],
    keep: wp.array[wp.int32],
    scan: wp.array[wp.int32],
    out_cells: wp.array[wp.vec3i],
):
    """Stream-compact the kept cells using an inclusive scan of ``keep``."""
    tid = wp.tid()
    if keep[tid] == 1:
        out_cells[scan[tid] - 1] = cells[tid]


@wp.kernel(enable_backward=False)
def _cull_out_of_bounds_kernel(
    cells: wp.array[wp.vec3i],
    ncells: wp.vec3i,
    keep: wp.array[wp.int32],
):
    """Flag leaf cells whose subscript lies outside the requested grid.

    The octree root box is padded up to a power-of-two cell count per axis so
    every axis can share a single ``max_depth``; this discards the cells that
    only exist because of that padding.
    """
    tid = wp.tid()
    c = cells[tid]
    inside = c[0] < ncells[0] and c[1] < ncells[1] and c[2] < ncells[2]
    keep[tid] = wp.where(inside, 1, 0)


@wp.kernel(enable_backward=False)
def _subdivide_cells_kernel(
    cells: wp.array[wp.vec3i],
    out_cells: wp.array[wp.vec3i],
):
    """Split each surviving cell into its 8 children at the next-finer depth."""
    tid = wp.tid()
    c = cells[tid]
    base_i = c[0] * 2
    base_j = c[1] * 2
    base_k = c[2] * 2
    for child in range(8):
        di = child & 1
        dj = (child >> 1) & 1
        dk = (child >> 2) & 1
        out_cells[tid * 8 + child] = wp.vec3i(base_i + di, base_j + dj, base_k + dk)


# =============================================================================
# Corner de-duplication kernels
# =============================================================================


@wp.kernel
def _compute_corner_codes_kernel(
    cells: wp.array[wp.vec3i],
    offset: wp.vec3i,
    stride_x: wp.int64,
    stride_y: wp.int64,
    codes: wp.array[wp.int64],
):
    """Emit the 8 linear corner codes of each cell, in cube-corner order.

    Codes are packed relative to ``offset`` (the minimum corner subscript) so
    arbitrary, possibly negative, cell subscripts stay within the packing range.

    Backward stays enabled (a no-op -- ``cells``/``codes`` are integer, never
    grad-requiring) purely to avoid Warp's warning when this corner
    de-duplication machinery is recorded on a tape alongside
    :func:`sparse_marching_cubes_from_cells`'s genuinely differentiable
    kernels; see ``_mark_active_edges_kernel``.
    """
    tid = wp.tid()
    c = cells[tid]
    for corner in range(8):
        ci = c[0] + wp.static(MC_CUBE_CORNER_OFFSETS[corner][0]) - offset[0]
        cj = c[1] + wp.static(MC_CUBE_CORNER_OFFSETS[corner][1]) - offset[1]
        ck = c[2] + wp.static(MC_CUBE_CORNER_OFFSETS[corner][2]) - offset[2]
        codes[tid * 8 + corner] = wp.int64(ci) * stride_x + wp.int64(cj) * stride_y + wp.int64(ck)


@wp.kernel
def _mark_first_occurrence_kernel(
    sorted_codes: wp.array[wp.int64],
    is_first: wp.array[wp.int32],
):
    """Flag the first element of each run of equal (sorted) corner codes."""
    tid = wp.tid()
    if tid == 0:
        is_first[tid] = 1
    else:
        is_first[tid] = wp.where(sorted_codes[tid] != sorted_codes[tid - 1], 1, 0)


@wp.kernel
def _scatter_inverse_kernel(
    sorted_perm: wp.array[wp.int32],
    unique_scan: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
):
    """Write, for each original corner, the index of its unique representative."""
    tid = wp.tid()
    inverse[sorted_perm[tid]] = unique_scan[tid] - 1


@wp.kernel
def _record_unique_codes_kernel(
    sorted_codes: wp.array[wp.int64],
    is_first: wp.array[wp.int32],
    unique_scan: wp.array[wp.int32],
    unique_codes: wp.array[wp.int64],
):
    """Gather one representative code per unique corner."""
    tid = wp.tid()
    if is_first[tid] == 1:
        unique_codes[unique_scan[tid] - 1] = sorted_codes[tid]


@wp.kernel
def _record_unique_source_kernel(
    sorted_perm: wp.array[wp.int32],
    is_first: wp.array[wp.int32],
    unique_scan: wp.array[wp.int32],
    unique_source: wp.array[wp.int32],
):
    """Record, for each unique corner, the flat ``cell*8+corner`` index of its
    canonical (first-in-sorted-order) source.

    Only computed when a caller needs to *gather* per-cell values onto unique
    corners (:func:`sparse_marching_cubes_from_cells`); the octree-driven path
    never needs it, since it evaluates the implicit function directly on
    already-unique corner positions.
    """
    tid = wp.tid()
    if is_first[tid] == 1:
        unique_source[unique_scan[tid] - 1] = sorted_perm[tid]


@wp.kernel
def _decode_corner_positions_kernel(
    unique_codes: wp.array[wp.int64],
    stride_x: wp.int64,
    stride_y: wp.int64,
    base: wp.vec3,
    cell_width: wp.vec3,
    positions: wp.array[wp.vec3],
):
    """Recover world-space positions of the unique corners from their codes.

    The codes hold subscripts *relative* to the minimum corner, and ``base`` is
    the world position of that minimum corner (folded in on the host, in double
    precision). Decoding relatively keeps the float32 multiplier small: adding
    the absolute subscript instead would round adjacent corners onto the same
    position once subscripts pass float32's 2**24 exact-integer limit.
    """
    tid = wp.tid()
    code = unique_codes[tid]
    ci = code / stride_x
    rem = code - ci * stride_x
    cj = rem / stride_y
    ck = rem - cj * stride_y
    positions[tid] = base + wp.cw_mul(
        cell_width,
        wp.vec3(wp.float32(wp.int32(ci)), wp.float32(wp.int32(cj)), wp.float32(wp.int32(ck))),
    )


@wp.kernel
def _fill_iota_kernel(out: wp.array[wp.int32]):
    tid = wp.tid()
    out[tid] = tid


@wp.kernel
def _gather_corner_values_kernel(
    unique_source: wp.array[wp.int32],
    per_cell_values_flat: wp.array[wp.float32],
    unique_values: wp.array[wp.float32],
):
    """Gather each unique corner's value from its one canonical per-cell source.

    A corner shared by several cells has several redundant, and expected to
    agree, entries in ``per_cell_values_flat``; ``unique_source`` (built in
    :func:`_dedupe_corners`) picks one deterministic representative per unique
    corner. Unlike a many-to-one scatter, this 1:1 gather has no write races,
    so it is safe and deterministic to differentiate: gradient flows from each
    unique corner back to exactly the one per-cell entry that produced it,
    matching how dense marching cubes evaluates each grid node exactly once.
    """
    tid = wp.tid()
    unique_values[tid] = per_cell_values_flat[unique_source[tid]]


# =============================================================================
# Sparse marching cubes kernels
# =============================================================================


@wp.func
def _cell_case_code(
    cell_corners: wp.array2d[wp.int32],
    corner_values: wp.array[wp.float32],
    isovalue: wp.float32,
    cell: wp.int32,
):
    """Compute the 8-bit marching cubes case index for a cell."""
    case_code = wp.int32(0)
    for corner in range(8):
        if corner_values[cell_corners[cell, corner]] >= isovalue:
            case_code += wp.static(wp.int32(1)) << wp.int32(corner)
    return case_code


@wp.kernel
def _mark_active_edges_kernel(
    cell_corners: wp.array2d[wp.int32],
    corner_values: wp.array[wp.float32],
    isovalue: wp.float32,
    case_to_tri_range: wp.array[wp.int32],
    tri_local_inds: wp.array[wp.int32],
    edge_owner: wp.array[wp.int32],
    edge_upper: wp.array[wp.int32],
    edge_axis: wp.array[wp.int32],
    edge_active: wp.array[wp.int32],
    edge_upper_corner: wp.array[wp.int32],
):
    """Mark every edge that a cell's triangles reference as crossing the surface.

    Each crossing edge is identified canonically by ``owner_corner * 3 + axis``.
    Multiple incident cells write the same values to the same slot, so races are
    benign and the result is deterministic.

    Backward is left enabled (rather than ``enable_backward=False``), even
    though its ``edge_active``/``edge_upper_corner`` outputs are discrete and
    have no useful adjoint, purely so Warp doesn't warn about differentiating
    a kernel that reads a ``requires_grad`` array
    (``corner_values``) with backward disabled -- the same reasoning as dense
    marching cubes' ``extract_faces_kernel``.
    """
    cell = wp.tid()
    case_code = _cell_case_code(cell_corners, corner_values, isovalue, cell)
    tri_start = case_to_tri_range[case_code]
    tri_end = case_to_tri_range[case_code + 1]

    for i in range(tri_start, tri_end):
        local_edge = tri_local_inds[i]
        owner_uid = cell_corners[cell, edge_owner[local_edge]]
        upper_uid = cell_corners[cell, edge_upper[local_edge]]
        slot = owner_uid * 3 + edge_axis[local_edge]
        edge_active[slot] = 1
        edge_upper_corner[slot] = upper_uid


@wp.kernel
def _emit_vertices_kernel(
    edge_active: wp.array[wp.int32],
    edge_vertex_index: wp.array[wp.int32],
    edge_upper_corner: wp.array[wp.int32],
    corner_positions: wp.array[wp.vec3],
    corner_values: wp.array[wp.float32],
    isovalue: wp.float32,
    verts_out: wp.array[wp.vec3],
):
    """Place one interpolated vertex on each active edge.

    This is the differentiable core of sparse marching cubes: ``wp.lerp``
    between the two corner positions, weighted by ``t``, a smooth function of
    the two corner values. Gradient flows from ``verts_out`` back to
    ``corner_values`` through here.
    """
    slot = wp.tid()
    if edge_active[slot] == 0:
        return

    owner_uid = slot / 3
    upper_uid = edge_upper_corner[slot]

    val_lo = corner_values[owner_uid]
    val_hi = corner_values[upper_uid]
    t = (isovalue - val_lo) / (val_hi - val_lo)
    t = wp.clamp(t, 0.0, 1.0)

    verts_out[edge_vertex_index[slot] - 1] = wp.lerp(corner_positions[owner_uid], corner_positions[upper_uid], t)


@wp.kernel
def _count_faces_kernel(
    cell_corners: wp.array2d[wp.int32],
    corner_values: wp.array[wp.float32],
    isovalue: wp.float32,
    case_to_tri_range: wp.array[wp.int32],
    face_count: wp.array[wp.int32],
):
    """Count the triangles each cell will emit.

    Backward stays enabled (a no-op, since ``face_count`` is discrete) only to
    avoid Warp's warning about a ``requires_grad`` input on a
    backward-disabled kernel; see ``_mark_active_edges_kernel``.
    """
    cell = wp.tid()
    case_code = _cell_case_code(cell_corners, corner_values, isovalue, cell)
    face_count[cell] = (case_to_tri_range[case_code + 1] - case_to_tri_range[case_code]) // 3


@wp.kernel
def _emit_faces_kernel(
    cell_corners: wp.array2d[wp.int32],
    corner_values: wp.array[wp.float32],
    isovalue: wp.float32,
    case_to_tri_range: wp.array[wp.int32],
    tri_local_inds: wp.array[wp.int32],
    edge_owner: wp.array[wp.int32],
    edge_axis: wp.array[wp.int32],
    edge_vertex_index: wp.array[wp.int32],
    face_scan: wp.array[wp.int32],
    indices_out: wp.array[wp.int32],
):
    """Emit triangle index triples, referencing the de-duplicated vertices.

    Backward stays enabled (a no-op, since ``indices_out`` is discrete) only
    to avoid Warp's warning about a ``requires_grad`` input on a
    backward-disabled kernel; see ``_mark_active_edges_kernel``.
    """
    cell = wp.tid()
    case_code = _cell_case_code(cell_corners, corner_values, isovalue, cell)
    tri_start = case_to_tri_range[case_code]
    tri_end = case_to_tri_range[case_code + 1]
    n_tri = (tri_end - tri_start) // 3
    if n_tri == 0:
        return

    if cell == 0:
        out_base = wp.int32(0)
    else:
        out_base = face_scan[cell - 1]

    for tri in range(n_tri):
        for s in range(3):
            local_edge = tri_local_inds[tri_start + 3 * tri + s]
            owner_uid = cell_corners[cell, edge_owner[local_edge]]
            slot = owner_uid * 3 + edge_axis[local_edge]
            indices_out[3 * (out_base + tri) + s] = edge_vertex_index[slot] - 1


# =============================================================================
# Host driver
# =============================================================================


def _scan_total(keep: wp.array, scan: wp.array) -> int:
    """Inclusive-scan ``keep`` into ``scan`` and return the total (a sync point)."""
    _wp_utils.array_scan(keep, scan, inclusive=True)
    return int(scan[-1:].numpy()[0])


def _as_cell_subscripts(cells, device) -> wp.array:
    """Return ``cells`` as a contiguous ``wp.array[wp.vec3i]`` on ``device``.

    A ``wp.array`` of ``vec3i`` or ``int32`` already on ``device`` is
    reinterpreted in place, so cells produced on the GPU are never round-tripped
    through the host. Anything else goes through NumPy.
    """
    if isinstance(cells, wp.array):
        if cells.device != device:
            cells = cells.to(device)
        if not cells.is_contiguous:
            cells = cells.contiguous()
        if cells.dtype == wp.vec3i:
            return cells.flatten() if cells.ndim > 1 else cells
        if cells.dtype == wp.int32:
            return cells.reshape((-1, 3)).view(wp.vec3i)
        cells = cells.numpy()

    cells_np = np.ascontiguousarray(cells, dtype=np.int32).reshape(-1, 3)
    return wp.array(cells_np, dtype=wp.vec3i, device=device)


def _cell_subscript_bounds(cells: wp.array, device) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the per-axis ``(min, max)`` of the cell subscripts on ``device``.

    Only the six resulting integers cross the bus, rather than the whole cell
    array, which matters when the caller already holds the cells on the GPU.
    """
    lo = wp.full(3, value=np.iinfo(np.int32).max, dtype=wp.int32, device=device)
    hi = wp.full(3, value=np.iinfo(np.int32).min, dtype=wp.int32, device=device)
    wp.launch(
        _cell_subscript_bounds_kernel,
        dim=cells.shape[0],
        inputs=[cells],
        outputs=[lo, hi],
        device=device,
    )
    return lo.numpy(), hi.numpy()


def _build_lipschitz_octree(evaluate, origin, root_width, max_depth, isovalue, lipschitz_bound, device):
    """Build the sparse set of leaf cells that may contain the level set.

    ``root_width`` is a ``wp.vec3`` giving the (possibly anisotropic) side
    lengths of the root box. Returns a ``wp.array[wp.vec3i]`` of
    leaf-cell minimum-corner subscripts at resolution ``2**max_depth`` (or
    ``None`` if the level set is not bracketed).
    """
    cells = wp.array([[0, 0, 0]], dtype=wp.vec3i, device=device)
    n_cells = 1

    for depth in range(max_depth + 1):
        cell_width = root_width * (1.0 / float(1 << depth))
        # Half the diagonal of the (possibly anisotropic) cell box: the
        # farthest any interior point can be from the center.
        band = lipschitz_bound * 0.5 * wp.length(cell_width)

        centers = wp.empty(n_cells, dtype=wp.vec3, device=device)
        wp.launch(
            _compute_cell_centers_kernel,
            dim=n_cells,
            inputs=[cells, wp.vec3(origin), cell_width],
            outputs=[centers],
            device=device,
            record_tape=False,
        )
        values = evaluate(centers)

        keep = wp.empty(n_cells, dtype=wp.int32, device=device)
        wp.launch(
            _mark_active_cells_kernel,
            dim=n_cells,
            inputs=[values, wp.float32(isovalue), wp.float32(band)],
            outputs=[keep],
            device=device,
            record_tape=False,
        )

        scan = wp.empty(n_cells, dtype=wp.int32, device=device)
        n_keep = _scan_total(keep, scan)
        if n_keep == 0:
            return None

        survivors = wp.empty(n_keep, dtype=wp.vec3i, device=device)
        wp.launch(
            _compact_cells_kernel,
            dim=n_cells,
            inputs=[cells, keep, scan],
            outputs=[survivors],
            device=device,
            record_tape=False,
        )

        if depth == max_depth:
            return survivors

        children = wp.empty(8 * n_keep, dtype=wp.vec3i, device=device)
        wp.launch(
            _subdivide_cells_kernel,
            dim=n_keep,
            inputs=[survivors],
            outputs=[children],
            device=device,
            record_tape=False,
        )
        cells = children
        n_cells = 8 * n_keep

    # Unreachable: the ``depth == max_depth`` branch always returns on the final
    # iteration, and ``max_depth >= 0`` guarantees at least one iteration.
    raise AssertionError("Lipschitz octree loop exited without reaching max_depth.")


def _cull_out_of_bounds(cells, ncells, device):
    """Drop leaf cells whose subscript falls outside the requested ``(nx-1, ny-1, nz-1)`` grid.

    The octree root box is padded up to a shared power-of-two cell count per
    axis; this removes the cells that only exist because of that padding.
    Returns ``(culled_cells, n_removed)``.
    """
    n_cells = cells.shape[0]
    keep = wp.empty(n_cells, dtype=wp.int32, device=device)
    wp.launch(
        _cull_out_of_bounds_kernel,
        dim=n_cells,
        inputs=[cells, wp.vec3i(ncells)],
        outputs=[keep],
        device=device,
        record_tape=False,
    )
    scan = wp.empty(n_cells, dtype=wp.int32, device=device)
    n_keep = _scan_total(keep, scan)
    if n_keep == n_cells:
        return cells, 0

    culled = wp.empty(n_keep, dtype=wp.vec3i, device=device)
    wp.launch(
        _compact_cells_kernel,
        dim=n_cells,
        inputs=[cells, keep, scan],
        outputs=[culled],
        device=device,
        record_tape=False,
    )
    return culled, n_cells - n_keep


def _dedupe_corners(cells, n_cells, offset, axis_stride, origin, cell_width, device, compute_unique_source=False):
    """De-duplicate cell corners.

    ``offset`` is the minimum corner subscript and ``axis_stride`` the number of
    distinct corner nodes per axis; together they pack corner subscripts into a
    unique int64 code. ``cell_width`` is a ``wp.vec3`` of the per-axis leaf cell
    size. Returns ``(cell_corners, corner_positions, n_unique, unique_source)``
    where ``cell_corners`` is an ``(n_cells, 8)`` int32 array of indices into
    the ``n_unique`` unique corners, ordered by :data:`MC_CUBE_CORNER_OFFSETS`.

    ``unique_source`` is ``None`` unless ``compute_unique_source`` is set: it is
    an ``(n_unique,)`` int32 array giving, for each unique corner, the flat
    ``cell*8+corner`` index of its canonical source, for
    :func:`sparse_marching_cubes_from_cells` to gather per-cell values with.
    The octree-driven path never needs this (it evaluates the implicit
    function directly on the already-unique ``corner_positions``), so it's
    opt-in to avoid adding a kernel launch to that hot path.
    """
    m = 8 * n_cells
    if axis_stride**3 >= 2**63:
        raise ValueError(
            f"Cell subscript range spans {axis_stride} nodes per axis, too large to pack into 64-bit corner codes."
        )
    stride_y = wp.int64(axis_stride)
    stride_x = wp.int64(axis_stride * axis_stride)
    offset_subscript = (int(offset[0]), int(offset[1]), int(offset[2]))
    offset = wp.vec3i(offset)

    codes = wp.empty(m, dtype=wp.int64, device=device)
    wp.launch(
        _compute_corner_codes_kernel,
        dim=n_cells,
        inputs=[cells, offset, stride_x, stride_y],
        outputs=[codes],
        device=device,
    )

    # radix_sort_pairs sorts in place and needs 2*count storage for keys/values.
    sort_keys = wp.empty(2 * m, dtype=wp.int64, device=device)
    sort_vals = wp.empty(2 * m, dtype=wp.int32, device=device)
    wp.copy(sort_keys[:m], codes)
    wp.launch(_fill_iota_kernel, dim=m, inputs=[sort_vals[:m]], device=device)
    _wp_utils.radix_sort_pairs(sort_keys, sort_vals, m)
    sorted_codes = sort_keys[:m]
    sorted_perm = sort_vals[:m]

    is_first = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(_mark_first_occurrence_kernel, dim=m, inputs=[sorted_codes], outputs=[is_first], device=device)

    unique_scan = wp.empty(m, dtype=wp.int32, device=device)
    n_unique = _scan_total(is_first, unique_scan)

    inverse = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        _scatter_inverse_kernel,
        dim=m,
        inputs=[sorted_perm, unique_scan],
        outputs=[inverse],
        device=device,
    )

    unique_codes = wp.empty(n_unique, dtype=wp.int64, device=device)
    wp.launch(
        _record_unique_codes_kernel,
        dim=m,
        inputs=[sorted_codes, is_first, unique_scan],
        outputs=[unique_codes],
        device=device,
    )

    unique_source = None
    if compute_unique_source:
        unique_source = wp.empty(n_unique, dtype=wp.int32, device=device)
        wp.launch(
            _record_unique_source_kernel,
            dim=m,
            inputs=[sorted_perm, is_first, unique_scan],
            outputs=[unique_source],
            device=device,
        )

    # Fold the subscript offset into the origin here, in double precision, so the
    # kernel only ever converts small relative subscripts to float32.
    base = wp.vec3(
        float(origin[0]) + float(cell_width[0]) * offset_subscript[0],
        float(origin[1]) + float(cell_width[1]) * offset_subscript[1],
        float(origin[2]) + float(cell_width[2]) * offset_subscript[2],
    )

    corner_positions = wp.empty(n_unique, dtype=wp.vec3, device=device)
    wp.launch(
        _decode_corner_positions_kernel,
        dim=n_unique,
        inputs=[unique_codes, stride_x, stride_y, base, wp.vec3(cell_width)],
        outputs=[corner_positions],
        device=device,
    )

    cell_corners = inverse.reshape((n_cells, 8))
    return cell_corners, corner_positions, n_unique, unique_source


def _empty_mesh(device, requires_grad=False):
    verts = wp.empty(0, dtype=wp.vec3, device=device, requires_grad=requires_grad)
    indices = wp.empty(0, dtype=wp.int32, device=device)
    return verts, indices


def _extract_from_dedup(cell_corners, corner_positions, corner_values, threshold, device):
    """Run marching cubes on de-duplicated cells.

    This is the sparse marching cubes core: given each cell's 8 unique corner
    indices, the unique corner positions, and the field value at each unique
    corner, it emits a watertight triangle mesh. Both the Lipschitz-octree driver
    and the explicit-cell entry point funnel through here.

    Differentiable w.r.t. ``corner_values``: the output ``verts`` array is
    allocated with ``requires_grad=corner_values.requires_grad`` (matching
    dense marching cubes' ``requires_grad=field.requires_grad`` convention),
    and gradient flows through the ``wp.lerp`` vertex interpolation in
    ``_emit_vertices_kernel``. ``indices`` (triangle topology) is discrete and
    carries no gradient, same as dense.
    """
    n_cells = cell_corners.shape[0]
    n_unique = corner_positions.shape[0]

    case_table = _get_mc_case_to_tri_range_table(device)
    tri_table = _get_mc_tri_local_inds_table(device)
    edge_owner = _get_edge_owner_table(device)
    edge_upper = _get_edge_upper_table(device)
    edge_axis = _get_edge_axis_table(device)

    # Vertex de-duplication: one slot per (owner corner, axis) edge.
    n_slots = 3 * n_unique
    edge_active = wp.zeros(n_slots, dtype=wp.int32, device=device)
    edge_upper_corner = wp.empty(n_slots, dtype=wp.int32, device=device)
    wp.launch(
        _mark_active_edges_kernel,
        dim=n_cells,
        inputs=[
            cell_corners,
            corner_values,
            wp.float32(threshold),
            case_table,
            tri_table,
            edge_owner,
            edge_upper,
            edge_axis,
        ],
        outputs=[edge_active, edge_upper_corner],
        device=device,
    )

    edge_vertex_index = wp.empty(n_slots, dtype=wp.int32, device=device)
    n_verts = _scan_total(edge_active, edge_vertex_index)
    if n_verts == 0:
        return _empty_mesh(device, requires_grad=corner_values.requires_grad)

    verts_out = wp.empty(n_verts, dtype=wp.vec3, device=device, requires_grad=corner_values.requires_grad)
    wp.launch(
        _emit_vertices_kernel,
        dim=n_slots,
        inputs=[
            edge_active,
            edge_vertex_index,
            edge_upper_corner,
            corner_positions,
            corner_values,
            wp.float32(threshold),
        ],
        outputs=[verts_out],
        device=device,
    )

    face_count = wp.empty(n_cells, dtype=wp.int32, device=device)
    wp.launch(
        _count_faces_kernel,
        dim=n_cells,
        inputs=[cell_corners, corner_values, wp.float32(threshold), case_table],
        outputs=[face_count],
        device=device,
    )
    face_scan = wp.empty(n_cells, dtype=wp.int32, device=device)
    n_faces = _scan_total(face_count, face_scan)
    if n_faces == 0:
        return verts_out, wp.empty(0, dtype=wp.int32, device=device)

    indices_out = wp.empty(3 * n_faces, dtype=wp.int32, device=device)
    wp.launch(
        _emit_faces_kernel,
        dim=n_cells,
        inputs=[
            cell_corners,
            corner_values,
            wp.float32(threshold),
            case_table,
            tri_table,
            edge_owner,
            edge_axis,
            edge_vertex_index,
            face_scan,
        ],
        outputs=[indices_out],
        device=device,
    )

    return verts_out, indices_out


def lipschitz_octree(
    field: Callable[[wp.array], wp.array],
    origin: wp.vec3 | tuple[float, float, float],
    root_width: float,
    max_depth: int,
    threshold: float = 0.0,
    lipschitz_bound: float = 1.0,
    device: wp.DeviceLike = None,
) -> tuple[wp.array, float]:
    """Return max-depth octree cells that may intersect a level set of a Lipschitz-bounded implicit field.

    Builds a sparse octree top-down, keeping only cells whose subtree can still
    reach the ``threshold`` level set of a 1-Lipschitz field: a cell of width
    ``h`` centered at ``c`` survives when ``|field(c) - threshold| <= lipschitz_bound * (sqrt(3)/2) * h``.
    The surviving leaves at ``max_depth`` form a thin shell around the surface.

    This mirrors ``igl::lipschitz_octree`` from libigl. It is the pruning stage
    used by :func:`sparse_marching_cubes`, exposed separately so callers can
    build their own extractors, visualize the adaptive grid, or reuse the cells.

    This function does not support backward-mode automatic differentiation,
    by design: cell selection is a discrete, threshold-based search, not a
    smooth function of ``field``'s values, so there is no gradient to carry
    from ``field`` to the output ``cells``. Recording this function on a
    ``wp.Tape()`` is silent -- no warning, no gradient; see
    :func:`sparse_marching_cubes_from_cells` for the differentiable
    extraction stage.

    Args:
        field: The implicit function, in the batched form accepted by
            :func:`sparse_marching_cubes`.
        origin: The minimum corner of the cubic root cell. Set ``origin`` and
            ``root_width`` so that the box ``[origin, origin + root_width]``
            covers the entire level set to be found; cells outside it are never
            visited.
        root_width: The side length of the cubic root cell.
        max_depth: The octree depth. Leaf cells have width ``root_width / 2**max_depth``.
        threshold: The isovalue defining the surface.
        lipschitz_bound: An upper bound on the Lipschitz constant of ``field``.
        device: The Warp device to run on. Defaults to the current device.

    Returns:
        A tuple ``(cells, cell_width)`` where ``cells`` is a
        ``wp.array[wp.vec3i]`` of the leaf cells' integer minimum-corner
        subscripts, in the convention :func:`sparse_marching_cubes_from_cells`
        expects (cell ``(i, j, k)`` covers
        ``[origin + cell_width * (i, j, k), origin + cell_width * (i + 1, j + 1, k + 1)]``),
        and ``cell_width`` is their common side length. ``cells`` is empty if
        the level set is not bracketed.
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}.")
    if root_width <= 0.0:
        raise ValueError(f"root_width must be positive, got {root_width}.")
    if lipschitz_bound < 0.0:
        raise ValueError(f"lipschitz_bound must be non-negative, got {lipschitz_bound}.")

    device = wp.get_device(device)
    origin = wp.vec3(origin)
    root_width = float(root_width)
    cell_width = root_width / float(1 << max_depth)
    root_width_vec = wp.vec3(root_width, root_width, root_width)

    evaluate = _make_evaluator(field, device)
    cells = _build_lipschitz_octree(evaluate, origin, root_width_vec, max_depth, threshold, lipschitz_bound, device)
    if cells is None:
        return wp.empty(0, dtype=wp.vec3i, device=device), cell_width

    return cells, cell_width


def sparse_marching_cubes_from_cells(
    cells: wp.array | npt.ArrayLike,
    corner_values: wp.array | npt.ArrayLike,
    origin: wp.vec3 | tuple[float, float, float] = (0.0, 0.0, 0.0),
    cell_width: float = 1.0,
    threshold: float = 0.0,
    device: wp.DeviceLike = None,
) -> tuple[wp.array, wp.array]:
    """Extract an isosurface from an explicit set of occupied cells and corner values.

    This is the sparse marching cubes core: it runs marching cubes on a
    caller-provided list of voxels (rather than cells discovered by a Lipschitz
    octree), sharing vertices between adjacent cells so the output is
    watertight. It is useful when the occupied cells are already known, such as
    a marked band of voxels around an object from a vision or generative model,
    and the implicit field has already been sampled at their corners. ``cells``
    are expected to be unique (duplicate cells may result in
    duplicate/intersecting output mesh faces).

    :func:`lipschitz_octree` returns cell subscripts in this function's
    convention, but not sampled corner values, which the caller must supply.

    This function supports backward-mode automatic differentiation: if
    ``corner_values`` has ``requires_grad=True`` and the call is wrapped in a
    ``wp.Tape()``, gradient flows from the output ``verts`` back to
    ``corner_values`` through the marching-cubes vertex interpolation. When a
    corner is shared by several cells, ``corner_values`` is expected to agree
    at that corner (see above); gradient flows back through exactly one of
    the agreeing entries (a fixed, deterministic choice, not an arbitrary
    one), which is the correct behavior when the redundant entries are
    independent evaluations of the same underlying point function. ``cells``
    (and any cell-selection step that produced them, such as
    :func:`lipschitz_octree`) is not differentiated -- it is
    integer subscript data with no gradient to carry.

    .. note::

        The subscript convention here differs from the VDB-style one used by
        :class:`warp.Volume`, so subscripts are not interchangeable between the
        two. In :class:`warp.Volume`, a subscript names a *voxel* and the sample
        lives at that voxel's center, so voxel ``(i, j, k)`` covers
        ``[ijk - 1/2, ijk + 1/2]`` in index space. Here a subscript names a
        *cell* by its minimum corner, so cell ``(i, j, k)`` covers
        ``[ijk, ijk + 1]`` and the samples live at its 8 corners.

        Both place samples on the integer lattice, so what shifts by half a cell
        is the region a subscript refers to, not the data. Concretely, a
        :class:`warp.Volume` voxel subscript corresponds to a *corner* subscript
        here rather than a cell subscript: the corners of cell ``(i, j, k)`` are
        the volume voxels ``(i, j, k)`` through ``(i + 1, j + 1, k + 1)``.
        ``origin`` matches ``min_world`` in
        :meth:`warp.Volume.load_from_numpy` and ``cell_width`` matches
        ``voxel_size``, since both give the world position of subscript
        ``(0, 0, 0)`` and the lattice spacing.

    Args:
        cells: An ``(N, 3)`` array of integer cell minimum-corner subscripts, as a
            ``wp.array[wp.vec3i]``, a ``wp.array[wp.int32]`` of shape
            ``(N, 3)``, or any array-like convertible to one. A cell at subscript
            ``(i, j, k)`` occupies the box with minimum corner
            ``origin + cell_width * (i, j, k)``. Subscripts may be negative and
            need not be contiguous. A ``wp.array`` already on ``device`` is
            consumed in place, without a host round trip.
        corner_values: An ``(N, 8)`` array of the field value at each cell's 8
            corners, ordered by :attr:`warp.geometry.IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS`
            (corner ``c`` is at subscript ``(i, j, k) + CUBE_CORNER_OFFSETS[c]``).
            Values at corners shared between cells are expected to agree.
        origin: The world-space position of cell subscript ``(0, 0, 0)``.
        cell_width: The side length of a cell.
        threshold: The isovalue defining the surface.
        device: The Warp device to run on. Defaults to the current device.

    Returns:
        A tuple ``(vertices, indices)`` as in :func:`sparse_marching_cubes`.

    Raises:
        ValueError: If ``cell_width`` is not positive, the shapes of ``cells`` and
            ``corner_values`` are inconsistent, or the subscript range is too
            large to pack into 64-bit corner codes.
    """
    device = wp.get_device(device)

    if cell_width <= 0.0:
        raise ValueError(f"cell_width must be positive, got {cell_width}.")

    cells_wp = _as_cell_subscripts(cells, device)
    n_cells = cells_wp.shape[0]

    values_wp = (
        corner_values
        if isinstance(corner_values, wp.array)
        else wp.array(np.ascontiguousarray(corner_values, dtype=np.float32), device=device)
    )
    if values_wp.device != device:
        values_wp = values_wp.to(device)
    if values_wp.dtype != wp.float32:
        raise ValueError(f"corner_values must be float32, got {values_wp.dtype}.")
    if values_wp.size != 8 * n_cells:
        raise ValueError(f"corner_values must have {8 * n_cells} entries for {n_cells} cells, got {values_wp.size}.")

    if n_cells == 0:
        return _empty_mesh(device)

    # The gather below indexes the flat (8*n_cells,) layout directly, but a
    # caller pulling corner values out of a larger structure can easily hand
    # us a strided view or a differently-shaped array, so normalize to a flat,
    # contiguous array before reshaping.
    if not values_wp.is_contiguous:
        values_wp = wp.clone(values_wp)
    if values_wp.ndim != 1:
        values_wp = values_wp.reshape((8 * n_cells,))

    # Pack corner subscripts relative to their minimum, using a per-axis stride
    # wide enough to cover the whole subscript range (plus the +1 corner).
    lo, hi = _cell_subscript_bounds(cells_wp, device)
    offset = (int(lo[0]), int(lo[1]), int(lo[2]))
    axis_stride = int((hi.astype(np.int64) - lo.astype(np.int64)).max()) + 2

    cell_width_vec = wp.vec3(float(cell_width), float(cell_width), float(cell_width))
    cell_corners, corner_positions, n_unique, unique_source = _dedupe_corners(
        cells_wp, n_cells, offset, axis_stride, wp.vec3(origin), cell_width_vec, device, compute_unique_source=True
    )

    unique_values = wp.empty(n_unique, dtype=wp.float32, device=device, requires_grad=values_wp.requires_grad)
    wp.launch(
        _gather_corner_values_kernel,
        dim=n_unique,
        inputs=[unique_source, values_wp],
        outputs=[unique_values],
        device=device,
    )

    return _extract_from_dedup(cell_corners, corner_positions, unique_values, float(threshold), device)


def sparse_marching_cubes(
    field: Callable[[wp.array], wp.array],
    nx: int,
    ny: int,
    nz: int,
    *,
    lower: wp.vec3 | tuple[float, float, float] | None = None,
    upper: wp.vec3 | tuple[float, float, float] | None = None,
    threshold: float = 0.0,
    lipschitz_bound: float = 1.0,
    device: wp.DeviceLike = None,
    return_stats: bool = False,
) -> tuple[wp.array, wp.array] | tuple[wp.array, wp.array, dict]:
    """Extract an isosurface from an implicit function using a Lipschitz octree.

    Rather than sampling a dense grid, this routine builds a sparse octree that
    provably brackets the level set of a 1-Lipschitz implicit function (such as a
    signed distance function) and runs marching cubes only on the near-surface
    cells, at a cost that scales with the surface area rather than the volume.

    The grid is specified exactly as for
    :meth:`warp.geometry.IsoSurfaceMarchingCubes.extract`: ``nx, ny, nz`` grid
    nodes over the box ``[lower, upper]``, which may be anisotropic. Calling
    this function and
    :meth:`~warp.geometry.IsoSurfaceMarchingCubes.extract` with the same
    ``nx, ny, nz`` and bounds produces the same surface.

    Both the pruning pass (at cell centers) and the extraction pass (at cell
    corners) query ``field`` in batches, never one point at a time. If ``field`` is
    implemented entirely with Warp (a kernel launch, with no host round trip),
    the whole pipeline stays on the GPU; see
    ``warp/examples/geometry/example_sparse_marching_cubes.py`` for a mesh-query
    implicit function that does this. A callable that wraps a host library
    (NumPy, PyTorch, ...) is equally valid -- it just pays a device/host sync
    on every call, since the values it returns must still land back on the
    query points' device.

    This function supports backward-mode automatic differentiation: if
    ``field``'s output array has ``requires_grad=True`` (the caller's
    responsibility to allocate) and the call is wrapped in a ``wp.Tape()``,
    gradient flows from the output ``verts`` back through the corner values
    ``field`` produced. Cell selection (:func:`lipschitz_octree`) is not
    differentiated -- it is a discrete, threshold-based search -- and does
    not print tape warnings either.

    Args:
        field: The implicit function, as a batched callable with the contract
            ``evaluate(points: wp.array[wp.vec3]) -> wp.array[wp.float32]``,
            returning the signed distance (or any 1-Lipschitz field whose
            ``threshold`` level set is the surface) at each query point. A bare
            single-point ``@wp.func`` is not accepted directly -- wrap it in a
            kernel that evaluates it over the batch (see above), which keeps
            evaluation on the GPU instead of hiding a per-point kernel launch
            behind what looks like a single call.
        nx: Number of grid nodes in the x-direction.
        ny: Number of grid nodes in the y-direction.
        nz: Number of grid nodes in the z-direction.
        lower: The 3D coordinate that the grid's corner at index
            ``(0, 0, 0)`` maps to. Defaults to ``(0.0, 0.0, 0.0)`` if
            ``None``. Anything outside ``[lower, upper]`` is never visited,
            so parts of the level set that leave it are simply missing from
            the output, leaving the mesh open where it exits.
        upper: The 3D coordinate that the grid's corner at index
            ``(nx - 1, ny - 1, nz - 1)`` maps to. Defaults to align with the
            grid's maximal indices if ``None``.
        threshold: The isovalue defining the surface.
        lipschitz_bound: An upper bound on the Lipschitz constant of ``field``. Use
            ``1.0`` for a true signed distance function. Larger values widen the
            retained band, trading speed for a stronger guarantee when the field
            varies faster than unit rate.
        device: The Warp device to run on. Defaults to the current device.
        return_stats: If ``True``, also return a dictionary of diagnostics
            (leaf-cell count, unique-corner count, implicit-function evaluation
            count, culled-cell count) useful for benchmarking against a dense
            grid.

    Returns:
        A tuple ``(vertices, indices)`` where ``vertices`` is a
        ``wp.array[wp.vec3]`` and ``indices`` is a flat
        ``wp.array[wp.int32]`` with three consecutive entries per triangle.
        If ``return_stats`` is ``True``, returns ``(vertices, indices, stats)``.

    Raises:
        ValueError: If ``nx``, ``ny``, or ``nz`` is less than 2, or
            ``lipschitz_bound`` is negative.
        TypeError: If ``field`` is not callable (including a bare
            single-point ``@wp.func``, which must be wrapped in a batched
            kernel first).
    """
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError(f"nx, ny, nz must each be at least 2, got ({nx}, {ny}, {nz}).")
    if lipschitz_bound < 0.0:
        raise ValueError(f"lipschitz_bound must be non-negative, got {lipschitz_bound}.")

    device = wp.get_device(device)
    lower_corner, grid_delta = resolve_domain_bounds((nx, ny, nz), lower, upper)

    ncells_x, ncells_y, ncells_z = nx - 1, ny - 1, nz - 1
    max_depth = (max(ncells_x, ncells_y, ncells_z) - 1).bit_length()
    resolution = 1 << max_depth
    root_width = grid_delta * float(resolution)
    ncells = (ncells_x, ncells_y, ncells_z)

    evaluate = _make_evaluator(field, device)

    stats = {
        "leaf_cells": 0,
        "unique_corners": 0,
        "field_evaluations": 0,
        "resolution": resolution,
        "requested_ncells": ncells,
        "culled_cells": 0,
    }

    def finish(result):
        if return_stats:
            verts, indices = result
            return verts, indices, stats
        return result

    # -- Octree construction -------------------------------------------------
    # Track evaluations by wrapping the evaluator.
    eval_count = [0]

    def counting_evaluate(points):
        eval_count[0] += points.shape[0]
        return evaluate(points)

    cells = _build_lipschitz_octree(
        counting_evaluate, lower_corner, root_width, max_depth, threshold, lipschitz_bound, device
    )
    if cells is None:
        stats["field_evaluations"] = eval_count[0]
        return finish(_empty_mesh(device))

    # -- Discard cells outside the requested (padded) grid --------------------
    # Leaf-cell subscripts are guaranteed < resolution on every axis, so
    # culling can only ever remove cells when some axis needed padding to
    # reach the shared power-of-two resolution. Skip the extra kernel launch
    # and host-syncing scan entirely otherwise -- this is the common case.
    if ncells_x == resolution and ncells_y == resolution and ncells_z == resolution:
        n_culled = 0
    else:
        cells, n_culled = _cull_out_of_bounds(cells, ncells, device)
    stats["culled_cells"] = n_culled
    n_cells = cells.shape[0]
    if n_cells == 0:
        stats["field_evaluations"] = eval_count[0]
        return finish(_empty_mesh(device))
    stats["leaf_cells"] = n_cells

    # -- Corner de-duplication and field evaluation --------------------------
    # Octree subscripts live in [0, resolution], so pack from a zero offset.
    cell_width = root_width * (1.0 / float(resolution))
    cell_corners, corner_positions, n_unique, _unique_source = _dedupe_corners(
        cells, n_cells, (0, 0, 0), resolution + 1, lower_corner, cell_width, device
    )
    stats["unique_corners"] = n_unique

    corner_values = counting_evaluate(corner_positions)
    stats["field_evaluations"] = eval_count[0]

    # -- Sparse marching cubes -----------------------------------------------
    return finish(_extract_from_dedup(cell_corners, corner_positions, corner_values, threshold, device))
