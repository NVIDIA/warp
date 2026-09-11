# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing utilities for triangle meshes.

The public entry points are the topology builder :func:`tri_tri_adjacency`,
its single-pair counterpart :func:`find_triangle_neighbor_edge_index`, and the
:func:`delaunay_edge_flip` operation, plus :func:`swept_volume_mesh`, which extracts
the motion envelope of animated rigid meshes; the predicates :func:`in_circle`
and :func:`signed_area` are reusable but stay internal until their naming
settles.
Everything runs on the Warp device (CPU or CUDA); the Delaunay convergence loop is
driven on-device with :func:`warp.capture_while`, so the routines are CUDA-graph
capturable.

Algorithm-specific kernels that are not meant for reuse are grouped in private
classes: :class:`_EdgeFlipper` holds the parts of the parallel flip loop that are
independent of the flip criterion (claim staking, topology mutation, pass
bookkeeping), and :class:`_DelaunayFlipper` subclasses it to add the Delaunay
criterion itself. This keeps their names tied to the algorithm and out of the
module namespace.
"""

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING

import numpy as np

import warp as wp
from warp._src.marching_cubes import MarchingCubes
from warp._src.types import type_repr, types_equal
from warp._src.utils import array_scan

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from warp._src.context import DeviceLike

__all__ = [
    "SweptVolumeSignMode",
    "delaunay_edge_flip",
    "find_triangle_neighbor_edge_index",
    "in_circle",
    "signed_area",
    "swept_volume_bounds",
    "swept_volume_field",
    "swept_volume_mesh",
    "swept_volume_sdf",
    "tri_tri_adjacency",
]


# ---------------------------------------------------------------------------
# Reusable geometric predicates
# ---------------------------------------------------------------------------


@wp.func
def signed_area(a: wp.vec2, b: wp.vec2, c: wp.vec2) -> float:
    """Return the signed area of triangle ``abc`` (positive when counterclockwise)."""
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


@wp.func
def _in_circle_det(a: wp.vec2, b: wp.vec2, c: wp.vec2, d: wp.vec2) -> float:
    """Return the in-circle determinant for the circumcircle of triangle ``abc`` and point ``d``.

    For a counterclockwise triangle ``abc``, the result is positive when ``d``
    lies strictly inside the circumcircle, zero when the four points are
    cocircular, and negative when ``d`` lies outside.
    """
    ad = a - d
    bd = b - d
    cd = c - d
    ad2 = wp.dot(ad, ad)
    bd2 = wp.dot(bd, bd)
    cd2 = wp.dot(cd, cd)
    return (
        ad[0] * (bd[1] * cd2 - cd[1] * bd2)
        - ad[1] * (bd[0] * cd2 - cd[0] * bd2)
        + ad2 * (bd[0] * cd[1] - bd[1] * cd[0])
    )


@wp.func
def in_circle(a: wp.vec2, b: wp.vec2, c: wp.vec2, d: wp.vec2) -> bool:
    """Return ``True`` if point ``d`` lies inside the circumcircle of triangle ``abc``.

    The triangle ``abc`` is assumed to be counterclockwise. This uses the
    standard floating-point in-circle determinant, which is accurate for
    well-conditioned configurations but is not an exact geometric predicate.
    """
    return _in_circle_det(a, b, c, d) > 0.0


# ---------------------------------------------------------------------------
# Triangle-triangle adjacency
# ---------------------------------------------------------------------------


@wp.kernel
def _reduce_max_vertex_index(tri: wp.array2d[wp.int32], max_index: wp.array[wp.int32]):
    t = wp.tid()
    for j in range(3):
        wp.atomic_max(max_index, 0, tri[t, j])


@wp.kernel
def _count_vertex_edges(tri: wp.array2d[wp.int32], counts: wp.array[wp.int32]):
    # Bucket each half-edge under its lower-indexed endpoint.
    t = wp.tid()
    for j in range(3):
        v1 = tri[t, (j + 1) % 3]
        v2 = tri[t, (j + 2) % 3]
        wp.atomic_add(counts, wp.min(v1, v2), 1)


@wp.kernel
def _scatter_vertex_edges(
    tri: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    fill: wp.array[wp.int32],
    bucket_hi: wp.array[wp.int32],
    bucket_he: wp.array[wp.int32],
):
    # Scatter each half-edge into its lower-endpoint bucket, recording the upper
    # endpoint and the packed half-edge id ``tri * 3 + local_edge``.
    t = wp.tid()
    for j in range(3):
        v1 = tri[t, (j + 1) % 3]
        v2 = tri[t, (j + 2) % 3]
        lo = wp.min(v1, v2)
        hi = wp.max(v1, v2)
        slot = offsets[lo] + wp.atomic_add(fill, lo, 1)
        bucket_hi[slot] = hi
        bucket_he[slot] = t * 3 + j


@wp.kernel
def _match_vertex_buckets(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    bucket_hi: wp.array[wp.int32],
    bucket_he: wp.array[wp.int32],
    triangle_neighbors: wp.array2d[wp.int32],
):
    # Each thread owns one vertex's bucket (a disjoint range), so it can pair
    # half-edges that share the same upper endpoint without synchronization. The
    # bucket holds only edges incident to this vertex (average degree ~6), so the
    # quadratic inner scan is over a handful of entries.
    v = wp.tid()
    beg = offsets[v]
    end = beg + counts[v]
    for a in range(beg, end):
        he_a = bucket_he[a]
        if he_a < 0:  # already paired
            continue
        hi_a = bucket_hi[a]
        for b in range(a + 1, end):
            if bucket_he[b] >= 0 and bucket_hi[b] == hi_a:
                he_b = bucket_he[b]
                triangle_neighbors[he_a // 3, he_a % 3] = he_b // 3
                triangle_neighbors[he_b // 3, he_b % 3] = he_a // 3
                bucket_he[a] = -1
                bucket_he[b] = -1
                break


@wp.kernel
def _match_vertex_buckets_with_neighbor_edge_indices(
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    bucket_hi: wp.array[wp.int32],
    bucket_he: wp.array[wp.int32],
    triangle_neighbors: wp.array2d[wp.int32],
    neighbor_edge_indices: wp.array2d[wp.int32],
):
    # As _match_vertex_buckets, but also records each edge's local index within its neighbor.
    v = wp.tid()
    beg = offsets[v]
    end = beg + counts[v]
    for a in range(beg, end):
        he_a = bucket_he[a]
        if he_a < 0:
            continue
        hi_a = bucket_hi[a]
        for b in range(a + 1, end):
            if bucket_he[b] >= 0 and bucket_hi[b] == hi_a:
                he_b = bucket_he[b]
                ta = he_a // 3
                ja = he_a % 3
                tb = he_b // 3
                jb = he_b % 3
                triangle_neighbors[ta, ja] = tb
                neighbor_edge_indices[ta, ja] = jb
                triangle_neighbors[tb, jb] = ta
                neighbor_edge_indices[tb, jb] = ja
                bucket_he[a] = -1
                bucket_he[b] = -1
                break


def _bucket_half_edges(indices: wp.array2d[wp.int32], vertex_count: int, device, num_tris: int):
    """Counting-sort the half-edges into per-vertex buckets keyed by their lower endpoint.

    Returns ``(offsets, counts, bucket_hi, bucket_he)``: for vertex ``v`` the slice
    ``[offsets[v] : offsets[v] + counts[v]]`` of ``bucket_hi`` / ``bucket_he`` lists
    the upper endpoint and packed half-edge id ``tri * 3 + local_edge`` of every
    half-edge whose lower endpoint is ``v``. Uses :func:`warp.utils.array_scan`
    rather than a global key sort, and needs no host synchronization.
    """
    num_half_edges = 3 * num_tris

    counts = wp.zeros(shape=vertex_count, dtype=wp.int32, device=device)
    offsets = wp.empty(shape=vertex_count, dtype=wp.int32, device=device)
    fill = wp.zeros(shape=vertex_count, dtype=wp.int32, device=device)
    bucket_hi = wp.empty(shape=num_half_edges, dtype=wp.int32, device=device)
    bucket_he = wp.empty(shape=num_half_edges, dtype=wp.int32, device=device)

    wp.launch(_count_vertex_edges, dim=num_tris, inputs=[indices, counts], device=device)
    array_scan(counts, offsets, inclusive=False)
    wp.launch(
        _scatter_vertex_edges,
        dim=num_tris,
        inputs=[indices, offsets, fill, bucket_hi, bucket_he],
        device=device,
    )
    return offsets, counts, bucket_hi, bucket_he


def tri_tri_adjacency(
    indices: wp.array2d[wp.int32], vertex_count: int | None = None, return_neighbor_edge_indices: bool = True
):
    """Build triangle-triangle adjacency for a triangle mesh. Assumes
    edge-manifold with possible boundary (i.e., exactly one or two triangles per
    edge). Consistent triangle orientation is not required.

    Args:
        indices: A ``(num_tris, 3)`` :class:`warp.array` of triangle vertex
            indices (``int32``).
        vertex_count: Number of vertices in the mesh. If ``None``, inferred as one
            plus the maximum vertex index. Must not be ``None`` when used in a
            CUDA graph capture context.
        return_neighbor_edge_indices: If ``True`` (default), also compute and return
            each neighboring triangle's local edge index for the shared edge.
            Pass ``False`` to skip this and build only the adjacency array,
            which is faster and uses less memory (use
            ``find_triangle_neighbor_edge_index`` to recover a single such
            index on demand).

    Returns:
        If ``return_neighbor_edge_indices`` is ``True``, a tuple ``(triangle_neighbors, neighbor_edge_indices)`` of
        ``(num_tris, 3)`` ``int32`` arrays; otherwise the single array ``triangle_neighbors``.
        ``triangle_neighbors[t, j]`` is the triangle adjacent to triangle ``t`` across the edge
        opposite local vertex ``j`` (the edge joining local vertices
        ``(j + 1) % 3`` and ``(j + 2) % 3``), or ``-1`` on a boundary edge.
        ``neighbor_edge_indices[t, j]`` is the local edge index of that shared edge within the
        neighboring triangle.

    """
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must be a (num_tris, 3) array of triangle vertex indices")

    device = indices.device
    num_tris = indices.shape[0]

    triangle_neighbors = wp.full(shape=(num_tris, 3), value=-1, dtype=wp.int32, device=device)
    neighbor_edge_indices = (
        wp.full(shape=(num_tris, 3), value=-1, dtype=wp.int32, device=device) if return_neighbor_edge_indices else None
    )

    if num_tris == 0:
        return (triangle_neighbors, neighbor_edge_indices) if return_neighbor_edge_indices else triangle_neighbors

    if vertex_count is None:
        if device.is_capturing:
            raise RuntimeError("`tri_tri_adjacency` requires `vertex_count` to be set for use in graph capture")
        max_index = wp.zeros(shape=1, dtype=wp.int32, device=device)
        wp.launch(_reduce_max_vertex_index, dim=num_tris, inputs=[indices, max_index], device=device)
        vertex_count = int(max_index.numpy()[0]) + 1

    offsets, counts, bucket_hi, bucket_he = _bucket_half_edges(indices, vertex_count, device, num_tris)
    if return_neighbor_edge_indices:
        wp.launch(
            _match_vertex_buckets_with_neighbor_edge_indices,
            dim=vertex_count,
            inputs=[offsets, counts, bucket_hi, bucket_he, triangle_neighbors, neighbor_edge_indices],
            device=device,
        )
        return triangle_neighbors, neighbor_edge_indices

    wp.launch(
        _match_vertex_buckets,
        dim=vertex_count,
        inputs=[offsets, counts, bucket_hi, bucket_he, triangle_neighbors],
        device=device,
    )
    return triangle_neighbors


# ---------------------------------------------------------------------------
# Parallel Delaunay edge flipping
# ---------------------------------------------------------------------------


@wp.func
def find_triangle_neighbor_edge_index(
    triangle_neighbors: wp.array2d[wp.int32], triangle: wp.int32, neighbor: wp.int32
) -> wp.int32:
    """Return the local edge of ``triangle`` shared with ``neighbor``.

    Args:
        triangle_neighbors: The ``(num_tris, 3)`` adjacency array built by
            :func:`tri_tri_adjacency` (its ``triangle_neighbors`` output).
        triangle: Index of the triangle to search.
        neighbor: Index of the triangle to find among ``triangle``'s neighbors.

    Returns:
        The local edge index ``j`` such that ``triangle_neighbors[triangle, j]
        == neighbor``, or -1 if ``triangle`` and ``neighbor`` are not adjacent.
        Calling this with ``triangle`` and ``neighbor`` swapped recovers a
        single entry -- the value :func:`tri_tri_adjacency` would have stored
        in ``neighbor_edge_indices`` had it been called with
        ``return_neighbor_edge_indices=True`` -- without having to keep that array around.
    """
    if triangle_neighbors[triangle, 0] == neighbor:
        return 0
    if triangle_neighbors[triangle, 1] == neighbor:
        return 1
    if triangle_neighbors[triangle, 2] == neighbor:
        return 2
    return -1


class _EdgeFlipper:
    """Criterion-independent machinery for a parallel, priority-based edge-flip loop.

    Subclasses (for example :class:`_DelaunayFlipper`) supply the flip criterion
    -- typically built on top of :meth:`_link_condition_ok` -- and drive these
    kernels from a :func:`warp.capture_while` convergence loop. Grouped in a
    private class -- not part of the public API -- so these names stay tied to
    the flip algorithm and do not clutter the module namespace.
    """

    @wp.func
    def _link_condition_ok(pc: wp.vec2, pa: wp.vec2, pb: wp.vec2, pd: wp.vec2, area_eps: float) -> bool:
        """Return whether flipping edge ``ab`` to ``cd`` yields two valid triangles.

        ``a``, ``b``, ``c``, ``d`` are as in :meth:`_DelaunayFlipper._edge_should_flip`:
        flipping replaces edge ``ab`` with ``cd``, producing triangles ``(c, a, d)``
        and ``(c, d, b)``. This holds only when quad ``a, d, b, c`` is convex, i.e.
        both resulting triangles are counterclockwise -- the link condition that
        any flip criterion must satisfy before its own test is meaningful.
        """
        if signed_area(pc, pa, pd) <= area_eps:
            return False
        if signed_area(pc, pd, pb) <= area_eps:
            return False
        return True

    @wp.func
    def _reverse_bits32(x: wp.uint32) -> wp.uint32:
        # Standard SWAR bit-reversal, self-inverse and therefore a bijection on
        # uint32. See _edge_priority for why that matters.
        x = ((x >> wp.uint32(1)) & wp.uint32(0x55555555)) | ((x & wp.uint32(0x55555555)) << wp.uint32(1))
        x = ((x >> wp.uint32(2)) & wp.uint32(0x33333333)) | ((x & wp.uint32(0x33333333)) << wp.uint32(2))
        x = ((x >> wp.uint32(4)) & wp.uint32(0x0F0F0F0F)) | ((x & wp.uint32(0x0F0F0F0F)) << wp.uint32(4))
        x = ((x >> wp.uint32(8)) & wp.uint32(0x00FF00FF)) | ((x & wp.uint32(0x00FF00FF)) << wp.uint32(8))
        x = (x >> wp.uint32(16)) | (x << wp.uint32(16))
        return x

    @wp.func
    def _edge_priority(t: wp.int32, j: wp.int32) -> wp.uint32:
        """Return a claim priority for edge ``j`` of triangle ``t``, unique across all edges.

        ``t * 3 + j`` alone is already unique, but it is monotonic in ``t`` --
        for mesh generators that number triangles with any spatial locality
        (grids, marching cubes, subdivision, ...), that makes the claim in
        :meth:`_stake_claim` always favor one geometric direction over another,
        so a chain of edges that mutually conflict (their claimed rows overlap)
        collapses to a single winner per pass instead of resolving in
        parallel: convergence degrades from the expected ``O(log n)`` passes to
        ``O(n)``. Reversing the bits of the (nonzero) key breaks that
        correlation while keeping the mapping a bijection -- so uniqueness,
        and therefore correctness, is unaffected -- and keeps 0 reserved as the
        claim array's "unclaimed" sentinel.
        """
        key = wp.uint32(t * 3 + j) + wp.uint32(1)
        return _EdgeFlipper._reverse_bits32(key)

    @wp.func
    def _stake_claim(claim: wp.array[wp.uint32], t: wp.int32, prio: wp.uint32):
        # Contend for exclusive ownership of triangle row ``t`` this pass; the
        # highest-priority edge touching the row wins the atomic_max.
        if t >= 0:
            wp.atomic_max(claim, t, prio)

    @wp.kernel
    def _apply_won_flips(
        tri: wp.array2d[wp.int32],
        triangle_neighbors: wp.array2d[wp.int32],
        claim: wp.array[wp.uint32],
        num_flips: wp.array[wp.int32],
    ):
        t = wp.tid()
        for j in range(3):
            prio = _EdgeFlipper._edge_priority(t, j)

            # Check ownership before reading any mutable topology: if this edge did
            # not win row t, another concurrent flip may be rewriting it. An edge
            # only staked a claim if it passed the flip predicate, and no winning
            # conflicting flip can touch its four protected rows, so there is no
            # need to re-run the predicate here.
            if claim[t] != prio:
                continue
            n = triangle_neighbors[t, j]
            if n < 0 or t >= n:
                continue
            if claim[n] != prio:
                continue

            jn = find_triangle_neighbor_edge_index(triangle_neighbors, n, t)
            if jn < 0:
                continue

            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            jn1 = (jn + 1) % 3
            jn2 = (jn + 2) % 3

            n_bc = triangle_neighbors[t, j1]
            n_ad = triangle_neighbors[n, jn1]
            if n_bc >= 0 and claim[n_bc] != prio:
                continue
            if n_ad >= 0 and claim[n_ad] != prio:
                continue

            # This edge owns {t, n, n_bc, n_ad}; their rows are stable for this pass.
            #   t: (c, a, b) -> (c, a, d)   n: (d, b, a) -> (d, b, c)
            # so the new diagonal is c-d and only slots j2 and jn2 change vertex.
            c = tri[t, j]
            d = tri[n, jn]
            tri[t, j2] = d
            tri[n, jn2] = c

            # t keeps slot j2 (edge c-a); slot j (edge a-d) inherits n's a-d
            # neighbor; slot j1 (edge d-c) is the new diagonal to n.
            triangle_neighbors[t, j] = n_ad
            triangle_neighbors[t, j1] = n
            # n keeps slot jn2 (edge d-b); slot jn (edge b-c) inherits t's b-c
            # neighbor; slot jn1 (edge c-d) is the new diagonal to t.
            triangle_neighbors[n, jn] = n_bc
            triangle_neighbors[n, jn1] = t

            # Only these two outer neighbors change ownership; re-point them.
            if n_ad >= 0:
                triangle_neighbors[n_ad, find_triangle_neighbor_edge_index(triangle_neighbors, n_ad, n)] = t
            if n_bc >= 0:
                triangle_neighbors[n_bc, find_triangle_neighbor_edge_index(triangle_neighbors, n_bc, t)] = n

            wp.atomic_add(num_flips, 0, 1)

    @wp.kernel
    def _record_pass(
        num_flips: wp.array[wp.int32],
        max_passes: wp.int32,
        total_flips: wp.array[wp.int32],
        pass_count: wp.array[wp.int32],
        condition: wp.array[wp.int32],
    ):
        # Single-thread bookkeeping between passes: accumulate the running total
        # and decide whether another pass is warranted. Written to a device array
        # so the convergence loop needs no host synchronization under graph capture.
        total_flips[0] += num_flips[0]
        pass_count[0] += 1
        if num_flips[0] > 0 and pass_count[0] < max_passes:
            condition[0] = 1
        else:
            condition[0] = 0


class _DelaunayFlipper(_EdgeFlipper):
    """Adds the Delaunay in-circle criterion to the :class:`_EdgeFlipper` loop."""

    @wp.func
    def _edge_should_flip(
        tri: wp.array2d[wp.int32],
        pos: wp.array[wp.vec2],
        ref: wp.array[wp.vec2],
        has_ref: wp.int32,
        area_eps: float,
        ref_eps: float,
        t: wp.int32,
        j: wp.int32,
        n: wp.int32,
        jn: wp.int32,
    ) -> bool:
        """Return whether the interior edge between triangles ``t`` and ``n`` should flip.

        ``j`` is the local edge of ``t`` opposite apex ``c`` and ``jn`` the local
        edge of ``n`` opposite apex ``d``. The shared edge joins vertices ``a`` and
        ``b``. A flip replaces edge ``ab`` with edge ``cd``, producing triangles
        ``(c, a, d)`` and ``(c, d, b)``.
        """
        c = tri[t, j]
        a = tri[t, (j + 1) % 3]
        b = tri[t, (j + 2) % 3]
        d = tri[n, jn]

        pa = pos[a]
        pb = pos[b]
        pc = pos[c]
        pd = pos[d]

        if not _EdgeFlipper._link_condition_ok(pc, pa, pb, pd, area_eps):
            return False

        # Delaunay test: flip only if the opposite apex is inside the circumcircle.
        # Triangle (c, a, b) is counterclockwise, matching the input winding.
        if _in_circle_det(pc, pa, pb, pd) <= 0.0:
            return False

        # Reject flips that would create degenerate triangles in a reference config.
        if has_ref != 0:
            ra = ref[a]
            rb = ref[b]
            rc = ref[c]
            rd = ref[d]
            if wp.abs(signed_area(rc, ra, rd)) <= ref_eps:
                return False
            if wp.abs(signed_area(rc, rd, rb)) <= ref_eps:
                return False

        return True

    @wp.kernel
    def _stake_flip_claims(
        tri: wp.array2d[wp.int32],
        triangle_neighbors: wp.array2d[wp.int32],
        pos: wp.array[wp.vec2],
        ref: wp.array[wp.vec2],
        has_ref: wp.int32,
        area_eps: float,
        ref_eps: float,
        claim: wp.array[wp.uint32],
    ):
        t = wp.tid()
        for j in range(3):
            n = triangle_neighbors[t, j]
            # Process each undirected edge once, from its lower-indexed triangle.
            if n < 0 or t >= n:
                continue
            jn = find_triangle_neighbor_edge_index(triangle_neighbors, n, t)
            if jn < 0:
                continue
            if not _DelaunayFlipper._edge_should_flip(tri, pos, ref, has_ref, area_eps, ref_eps, t, j, n, jn):
                continue

            # Preserving each triangle's cyclic slot layout, a flip only reads and
            # writes the rows of {t, n, n_bc, n_ad}, so it claims exactly those four.
            n_bc = triangle_neighbors[t, (j + 1) % 3]
            n_ad = triangle_neighbors[n, (jn + 1) % 3]
            prio = _EdgeFlipper._edge_priority(t, j)
            _EdgeFlipper._stake_claim(claim, t, prio)
            _EdgeFlipper._stake_claim(claim, n, prio)
            _EdgeFlipper._stake_claim(claim, n_bc, prio)
            _EdgeFlipper._stake_claim(claim, n_ad, prio)


def delaunay_edge_flip(
    positions: wp.array[wp.vec2],
    indices: wp.array2d[wp.int32],
    ref_positions: wp.array[wp.vec2] | None = None,
    max_passes: int = 1000,
    area_epsilon: float = 0.0,
    ref_area_epsilon: float = 1.0e-10,
) -> wp.array[wp.int32]:
    """Flip non-Delaunay interior edges in place until all are Delaunay or max
    passes is reached.

    The triangulation is modified in place: ``indices`` is updated with the
    flipped connectivity while ``positions`` is left unchanged. Each iteration
    flips an independent set of edges in parallel; CUDA graph capture is
    supported.

    Args:
        positions: A ``(vertex_count,)`` :class:`warp.array` of :class:`warp.vec2`
            vertex positions.
        indices: A ``(num_tris, 3)`` :class:`warp.array` of ``int32`` triangle
            vertex indices, assumed counterclockwise. Modified in place.
        ref_positions: Optional ``(vertex_count,)`` :class:`warp.array` of
            :class:`warp.vec2` reference positions. When provided, flips that
            would create a degenerate triangle in the reference configuration
            are rejected. Useful when the working mesh is a deformation of a
            reference mesh that must stay non-degenerate.
        max_passes: Maximum number of parallel flip passes before stopping. Must
            be at least 1.
        area_epsilon: Minimum signed area required for each triangle produced by
            a flip; guards against creating inverted or sliver triangles. Must be
            finite and non-negative.
        ref_area_epsilon: Degeneracy threshold applied to ``ref_positions``. Must be
            finite and non-negative.

    Returns:
        A (1,) int32 array on indices.device containing the total number of
        edges flipped. Under graph capture, read the value after replaying the
        graph.

    Note:
        Assumes a manifold mesh with consistent counterclockwise winding
        (positive signed triangle areas).
    """
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must be a (num_tris, 3) array of triangle vertex indices")

    device = indices.device

    if positions.device != device:
        raise ValueError("positions and indices must be on the same device")
    if ref_positions is not None and ref_positions.device != device:
        raise ValueError("ref_positions and indices must be on the same device")
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")
    if not math.isfinite(area_epsilon) or area_epsilon < 0.0:
        raise ValueError("area_epsilon must be a finite, non-negative number")
    if not math.isfinite(ref_area_epsilon) or ref_area_epsilon < 0.0:
        raise ValueError("ref_area_epsilon must be a finite, non-negative number")

    num_tris = indices.shape[0]
    if num_tris == 0:
        # Match the documented return type: always a device accumulator.
        return wp.zeros(shape=1, dtype=wp.int32, device=device)

    vertex_count = positions.shape[0]
    triangle_neighbors = tri_tri_adjacency(indices, vertex_count=vertex_count, return_neighbor_edge_indices=False)

    has_ref = wp.int32(1 if ref_positions is not None else 0)
    ref = ref_positions if ref_positions is not None else positions
    max_passes_i = wp.int32(max_passes)

    # 0 is the "unclaimed" sentinel; _edge_priority() never produces 0 (see its
    # docstring), so a real claim can never be confused with an empty row.
    claim = wp.empty(shape=num_tris, dtype=wp.uint32, device=device)
    num_flips = wp.empty(shape=1, dtype=wp.int32, device=device)
    total_flips = wp.zeros(shape=1, dtype=wp.int32, device=device)
    pass_count = wp.zeros(shape=1, dtype=wp.int32, device=device)
    # Seed the condition so the loop body runs at least once; the body clears it
    # once a pass makes no progress (or the pass budget is exhausted).
    condition = wp.ones(shape=1, dtype=wp.int32, device=device)

    def _flip_pass():
        claim.fill_(0)
        num_flips.zero_()
        wp.launch(
            _DelaunayFlipper._stake_flip_claims,
            dim=num_tris,
            inputs=[indices, triangle_neighbors, positions, ref, has_ref, area_epsilon, ref_area_epsilon, claim],
            device=device,
        )
        wp.launch(
            _EdgeFlipper._apply_won_flips,
            dim=num_tris,
            inputs=[indices, triangle_neighbors, claim, num_flips],
            device=device,
        )
        wp.launch(
            _EdgeFlipper._record_pass,
            dim=1,
            inputs=[num_flips, max_passes_i, total_flips, pass_count, condition],
            device=device,
        )

    # Device-driven convergence loop. Under CUDA graph capture this records
    # conditional graph nodes; otherwise wp.capture_while reads the condition
    # back to the host to decide when to stop.
    wp.capture_while(condition, _flip_pass)

    return total_flips


# ---------------------------------------------------------------------------
# Swept volume (motion envelope) of animated rigid meshes
# ---------------------------------------------------------------------------


class SweptVolumeSignMode(enum.IntEnum):
    """Method used to classify inside/outside when evaluating the per-mesh SDF.

    ``IntEnum`` members are integers, so a value can be passed straight into a
    kernel and compared against.
    """

    NORMAL = 0
    """Sign from :func:`warp.mesh_query_point_sign_normal`. """

    WINDING_NUMBER = 1
    """Sign from :func:`warp.mesh_query_point_sign_winding_number`. Requires
    every input mesh to be built with ``support_winding_number=True``."""

    PARITY = 2
    """Sign from :func:`warp.mesh_query_point_sign_parity`."""

    NO_SIGN = 3
    """Unsigned distance, from :func:`warp.mesh_query_point_no_sign`. The field
    is then positive everywhere, so it is only correct for extracting an
    ``threshold > 0`` isosurface of swept volume of a transforming thin shell. If the
    input is a solid then this may result in erroneous interior surfaces in the
    output."""


@wp.func
def swept_volume_sdf(
    point: wp.vec3,
    mesh_ids: wp.array[wp.uint64],
    transforms: wp.array2d[wp.transform],
    max_dist: wp.float32,
    sign_mode: wp.int32,
) -> wp.float32:
    """Evaluate the swept-volume pseudo signed distance at a single point.

    Returns :math:`\\min_m \\min_s \\mathrm{sdf}_m(X_{m,s}^{-1} p)`, the minimum
    over every mesh :math:`m` and every sampled pose :math:`s`. The motion is
    rigid, so instead of transforming the geometry the query point is pushed
    back into each mesh's rest frame, and one closest-point query per
    (mesh, sample) evaluates one pose. Callable from within your own kernels;
    not differentiable.

    Args:
        point: Query point, in world space.
        mesh_ids: Identifiers of the rest-pose meshes.
        transforms: Per-mesh, per-sample rigid poses, mapping rest space to
            world space.
        max_dist: Bound on the closest-point search. A query that finds nothing
            within it returns no sign either, so this returns ``+max_dist`` for
            a point deep inside the union just as it does for one far outside;
            pass a bound that spans the region of interest, as the array-level
            entry points do.
        sign_mode: Inside/outside classification method, as the integer value of
            a :class:`SweptVolumeSignMode` member.

    Returns:
        The pseudo signed distance at ``point``.
    """
    num_meshes = mesh_ids.shape[0]
    num_samples = transforms.shape[1]

    best = max_dist
    for m in range(num_meshes):
        mesh_id = mesh_ids[m]
        for s in range(num_samples):
            # Map the query point into the mesh's rest frame for this pose. The
            # map is rigid, so distances are preserved between the two frames.
            p_local = wp.transform_point(wp.transform_inverse(transforms[m, s]), point)

            sign = float(1.0)
            if sign_mode == wp.int32(SweptVolumeSignMode.WINDING_NUMBER.value):
                query = wp.mesh_query_point_sign_winding_number(mesh_id, p_local, max_dist)
                sign = query.sign
            elif sign_mode == wp.int32(SweptVolumeSignMode.PARITY.value):
                query = wp.mesh_query_point_sign_parity(mesh_id, p_local, max_dist)
                sign = query.sign
            elif sign_mode == wp.int32(SweptVolumeSignMode.NO_SIGN.value):
                # Unsigned distances cannot be negative, so anything farther than
                # the running minimum cannot improve it: shrink the search.
                query = wp.mesh_query_point_no_sign(mesh_id, p_local, best)
            else:
                query = wp.mesh_query_point_sign_normal(mesh_id, p_local, max_dist)
                sign = query.sign

            if query.result:
                closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                dist = sign * wp.length(p_local - closest)
                best = wp.min(best, dist)

    return best


@wp.kernel(enable_backward=False)
def swept_volume_field_kernel(
    mesh_ids: wp.array[wp.uint64],
    transforms: wp.array2d[wp.transform],
    origin: wp.vec3,
    spacing: wp.vec3,
    max_dist: wp.float32,
    sign_mode: wp.int32,
    out_field: wp.array3d[wp.float32],
):
    i, j, k = wp.tid()
    point = origin + wp.cw_mul(spacing, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))
    out_field[i, j, k] = swept_volume_sdf(point, mesh_ids, transforms, max_dist, sign_mode)


def _swept_volume_transforms(transforms, device):
    """Normalize the ``transforms`` argument into a :class:`warp.array2d`.

    The result has shape ``(num_meshes, num_samples)``, dtype
    :class:`warp.transform`, and lives on ``device``.
    """
    if isinstance(transforms, wp.array):
        if transforms.ndim != 2:
            raise ValueError(
                f"'transforms' must be a 2D array of shape (num_meshes, num_samples), got ndim {transforms.ndim}."
            )
        if not types_equal(transforms.dtype, wp.transform):
            raise TypeError(f"'transforms' must have dtype wp.transform, got {type_repr(transforms.dtype)}.")
        return transforms.to(device)

    # Accept nested Python sequences / NumPy arrays of shape (M, S, 7).
    arr = np.asarray(transforms, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 7:
        raise ValueError(
            "'transforms' must be a wp.array2d(dtype=wp.transform) or an array of "
            f"shape (num_meshes, num_samples, 7), got shape {arr.shape}."
        )
    return wp.array2d(arr, dtype=wp.transform, device=device)


@wp.kernel(enable_backward=False)
def swept_volume_mesh_aabb_kernel(
    points: wp.array[wp.vec3],
    slot: int,
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
):
    # Rest-pose axis-aligned bounds of one mesh, reduced into slot ``slot``.
    i = wp.tid()
    p = points[i]
    wp.atomic_min(out_lower, slot, p)
    wp.atomic_max(out_upper, slot, p)


@wp.kernel(enable_backward=False)
def swept_volume_bounds_kernel(
    mesh_lower: wp.array[wp.vec3],
    mesh_upper: wp.array[wp.vec3],
    transforms: wp.array2d[wp.transform],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
):
    # Union, over every (mesh, sample), of the pose-transformed rest-pose AABB.
    m, s = wp.tid()
    lo = mesh_lower[m]
    hi = mesh_upper[m]
    if lo[0] > hi[0]:  # empty mesh: nothing was reduced into this slot
        return
    xform = transforms[m, s]
    # A rigid map sends the box to one whose axis-aligned bounds are spanned by
    # its eight transformed corners, so it suffices to reduce those.
    for a in range(2):
        cx = wp.where(a == 0, lo[0], hi[0])
        for b in range(2):
            cy = wp.where(b == 0, lo[1], hi[1])
            for c in range(2):
                cz = wp.where(c == 0, lo[2], hi[2])
                p = wp.transform_point(xform, wp.vec3(cx, cy, cz))
                wp.atomic_min(out_lower, 0, p)
                wp.atomic_max(out_upper, 0, p)


def swept_volume_bounds(
    meshes: Sequence[wp.Mesh],
    transforms: wp.array2d[wp.transform] | npt.ArrayLike,
    padding: float = 0.0,
    device: DeviceLike | None = None,
) -> tuple[wp.vec3, wp.vec3]:
    """Return the world-space axis-aligned bounds of every mesh over every pose.

    Reduces each mesh's rest-pose vertices to an axis-aligned box, then transforms
    the eight box corners by every pose and reduces their union, entirely with
    Warp kernels.

    Use this to size a domain that several fields share: union these bounds and
    pass the result to :func:`swept_volume_field` as
    ``lower`` and ``upper``.

    Args:
        meshes: Sequence of rest-pose :class:`warp.Mesh` objects.
        transforms: Per-mesh, per-sample rigid poses; see
            :func:`swept_volume_field`.
        padding: Widening applied on every side, in world units.
        device: Device on which to run. Defaults to the device of the first mesh.

    Returns:
        A tuple ``(lower, upper)`` of :class:`warp.vec3` world coordinates.
    """
    if len(meshes) == 0:
        raise ValueError("'meshes' must contain at least one mesh.")
    device = wp.get_device(device) if device is not None else meshes[0].device
    transforms = _swept_volume_transforms(transforms, device)
    _check_transforms(meshes, transforms)
    return _swept_volume_bounds(meshes, transforms, padding, device)


def _check_transforms(meshes, transforms) -> None:
    """Reject pose arrays the bounds kernels would index out of bounds.

    ``_swept_volume_bounds`` sizes its per-mesh slots from ``transforms`` and
    fills them by mesh index, so a row count that disagrees with the mesh count
    has to be caught before any launch.
    """
    if transforms.shape[0] != len(meshes):
        raise ValueError(f"'transforms' has {transforms.shape[0]} rows but there are {len(meshes)} meshes.")
    if transforms.shape[1] == 0:
        raise ValueError("'transforms' must contain at least one pose sample.")


def _swept_volume_bounds(meshes, transforms, margin, device):
    """Bounds of the posed meshes, padded by ``margin``, with arguments pre-checked."""
    num_meshes, num_samples = transforms.shape[0], transforms.shape[1]

    pos_inf = wp.vec3(math.inf, math.inf, math.inf)
    neg_inf = wp.vec3(-math.inf, -math.inf, -math.inf)

    mesh_lower = wp.full(num_meshes, pos_inf, dtype=wp.vec3, device=device)
    mesh_upper = wp.full(num_meshes, neg_inf, dtype=wp.vec3, device=device)
    for m, mesh in enumerate(meshes):
        if mesh.points.shape[0] > 0:
            wp.launch(
                swept_volume_mesh_aabb_kernel,
                dim=mesh.points.shape[0],
                inputs=[mesh.points, m],
                outputs=[mesh_lower, mesh_upper],
                device=device,
            )

    out_lower = wp.full(1, pos_inf, dtype=wp.vec3, device=device)
    out_upper = wp.full(1, neg_inf, dtype=wp.vec3, device=device)
    wp.launch(
        swept_volume_bounds_kernel,
        dim=(num_meshes, num_samples),
        inputs=[mesh_lower, mesh_upper, transforms],
        outputs=[out_lower, out_upper],
        device=device,
    )

    # Read back the six numbers (as the OBB code does) and pad on the host.
    lo = out_lower.numpy()[0]
    hi = out_upper.numpy()[0]
    if not math.isfinite(float(lo[0])):
        raise ValueError("Could not compute swept-volume bounds: all input meshes are empty.")
    lower = wp.vec3(float(lo[0]) - margin, float(lo[1]) - margin, float(lo[2]) - margin)
    upper = wp.vec3(float(hi[0]) + margin, float(hi[1]) + margin, float(hi[2]) + margin)
    return lower, upper


def _swept_volume_grid(meshes, transforms, voxel_size, resolution, lower, upper, extra_padding, device):
    """Resolve the sampling domain and node counts.

    Returns ``(lower, upper, dims, tight_lower, tight_upper)``, where the last
    two are the unpadded bounds of the posed geometry. When the corners are supplied the domain is
    used verbatim and ``resolution`` sets the node counts. Otherwise the swept
    bounds are padded by ``extra_padding`` plus roughly two cells, so the level
    being extracted stays strictly inside the domain, and ``voxel_size`` snaps
    the extent up to whole cells so the spacing is exactly ``voxel_size``.
    """
    if (lower is None) != (upper is None):
        raise ValueError("Pass both 'lower' and 'upper', or neither.")
    if voxel_size is not None and resolution is not None:
        raise ValueError("Pass either 'voxel_size' or 'resolution', not both.")
    if voxel_size is None and resolution is None:
        raise ValueError("Provide either 'voxel_size' or 'resolution'.")
    if voxel_size is not None and voxel_size <= 0.0:
        raise ValueError(f"'voxel_size' must be positive, got {voxel_size}.")

    dims = None
    if resolution is not None:
        dims = tuple(int(n) for n in resolution)
        if len(dims) != 3:
            raise ValueError(f"'resolution' must have exactly 3 entries, got {dims}.")
        if any(n < 2 for n in dims):
            raise ValueError(f"'resolution' must be at least 2 along each axis, got {dims}.")

    tight_lower, tight_upper = _swept_volume_bounds(meshes, transforms, 0.0, device)
    tight = tuple(tight_upper[a] - tight_lower[a] for a in range(3))

    if lower is not None:
        # The caller owns the domain, so the extent cannot be snapped to whole
        # cells and only an explicit node count is meaningful.
        if voxel_size is not None:
            raise ValueError(
                "'voxel_size' cannot be combined with explicit domain bounds, since the extent would "
                "have to grow to fit whole cells. Pass 'resolution' instead."
            )
        lower = wp.vec3(*(float(c) for c in lower))
        upper = wp.vec3(*(float(c) for c in upper))
        if any(upper[a] <= lower[a] for a in range(3)):
            raise ValueError(f"'upper' {upper} must exceed 'lower' {lower}.")
        return lower, upper, dims, tight_lower, tight_upper

    if voxel_size is not None:
        # Pad, then grow the extent to a whole number of cells so the spacing is
        # exactly voxel_size on every axis.
        padding = (extra_padding + 2.0 * voxel_size,) * 3
        padded = tuple(tight[a] + 2.0 * padding[a] for a in range(3))
        dims = tuple(max(2, int(math.ceil(padded[a] / voxel_size)) + 1) for a in range(3))
        snapped = tuple((dims[a] - 1) * voxel_size for a in range(3))
        grow = tuple(0.5 * (snapped[a] - tight[a]) for a in range(3))
    else:
        # Measure the two cells of slack against the unpadded spacing, which
        # keeps the padding above extra_padding for any node count.
        padding = tuple(extra_padding + 2.0 * tight[a] / (dims[a] - 1) for a in range(3))
        grow = padding

    lower = wp.vec3(*(tight_lower[a] - grow[a] for a in range(3)))
    upper = wp.vec3(*(tight_upper[a] + grow[a] for a in range(3)))
    return lower, upper, dims, tight_lower, tight_upper


def swept_volume_field(
    meshes: Sequence[wp.Mesh],
    transforms: wp.array2d[wp.transform] | npt.ArrayLike,
    voxel_size: float | None = None,
    *,
    resolution: tuple[int, int, int] | None = None,
    lower: wp.vec3 | tuple[float, float, float] | None = None,
    upper: wp.vec3 | tuple[float, float, float] | None = None,
    sign_mode: SweptVolumeSignMode = SweptVolumeSignMode.WINDING_NUMBER,
    device: DeviceLike | None = None,
) -> tuple[wp.array3d[wp.float32], wp.vec3, wp.vec3]:
    """Compute a dense regular-grid discretization of the swept-volume pseudo signed-distance function.

    Every node is evaluated by :func:`swept_volume_sdf`, by brute force over
    every (mesh, sample) pair ("dense time stamping"). The result is negative
    inside the swept volume and positive outside, so extracting its zero
    isosurface (see :func:`swept_volume_mesh`) yields the motion envelope.
    With :attr:`SweptVolumeSignMode.NO_SIGN` the field is unsigned and therefore
    positive everywhere. Autodiff is not supported: the kernels that build the
    field run forward only.

    Because the poses are the *provided* samples, the field only accounts for the
    geometry at those instants; motion between consecutive samples is not
    conservatively bounded. Supply a sufficiently fine time sampling for the
    desired tolerance.

    Args:
        meshes: Sequence of rest-pose :class:`warp.Mesh` objects.
        transforms: Per-mesh, per-sample rigid poses, each mapping its mesh
            from rest space to world space, with a rotation quaternion of unit
            length. Either a :class:`warp.array2d` of :class:`warp.transform`
            with shape ``(num_meshes, num_samples)``, or an array of shape
            ``(num_meshes, num_samples, 7)`` (translation ``xyz`` followed by
            quaternion ``xyzw``).
        voxel_size: Edge length of a grid cell in world units. The domain is
            grown to a whole number of cells, so the spacing is exactly this on
            every axis. Cannot be combined with explicit domain bounds.
        resolution: Node counts ``(nx, ny, nz)``. Pass this or ``voxel_size``,
            not both. The spacing follows from the extent, so it is anisotropic
            unless the node counts match the domain's aspect ratio.
        lower: World coordinate that node ``(0, 0, 0)`` maps
            to, as in :meth:`warp.MarchingCubes.extract_surface_marching_cubes`.
            Defaults to the swept bounds padded so the surface is not clipped.
            Pass both corners or neither; see :func:`swept_volume_bounds` to size
            a domain that several fields share.
        upper: World coordinate that node
            ``(nx-1, ny-1, nz-1)`` maps to.
        sign_mode: Inside/outside classification method; see
            :class:`SweptVolumeSignMode`. The default
            (``SweptVolumeSignMode.WINDING_NUMBER``) requires every mesh to be built
            with ``support_winding_number=True``.
        device: Device on which to build the field. Defaults to the device of
            the first mesh.

    Returns:
        A tuple ``(field, lower, upper)`` where ``field`` is a
        ``wp.array3d(dtype=wp.float32)`` of signed distances, and ``lower`` and
        ``upper`` are the :class:`warp.vec3` world coordinates that grid nodes
        ``(0, 0, 0)`` and ``(nx-1, ny-1, nz-1)`` map to.
    """
    if len(meshes) == 0:
        raise ValueError("'meshes' must contain at least one mesh.")

    device = wp.get_device(device) if device is not None else meshes[0].device

    # The kernel launches on a single device, so every mesh must already live there.
    for i, mesh in enumerate(meshes):
        if mesh.device != device:
            raise ValueError(f"'meshes[{i}]' is on device '{mesh.device}' but the launch device is '{device}'.")
        # Querying the winding number of a mesh built without it returns garbage
        # signs rather than failing, so reject that combination up front.
        if sign_mode == SweptVolumeSignMode.WINDING_NUMBER and not mesh.support_winding_number:
            raise ValueError(
                f"'meshes[{i}]' was not built with support_winding_number=True, which the default "
                "SweptVolumeSignMode.WINDING_NUMBER requires. Rebuild it with "
                "wp.Mesh(..., support_winding_number=True), or pass sign_mode=SweptVolumeSignMode.NORMAL."
            )

    mesh_ids = wp.array([mesh.id for mesh in meshes], dtype=wp.uint64, device=device)
    transforms_wp = _swept_volume_transforms(transforms, device)
    _check_transforms(meshes, transforms_wp)

    lower, upper, dims, tight_lower, tight_upper = _swept_volume_grid(
        meshes,
        transforms_wp,
        voxel_size,
        resolution,
        lower,
        upper,
        0.0,
        device,
    )
    extent = tuple(upper[a] - lower[a] for a in range(3))
    spacing = tuple(extent[a] / (dims[a] - 1) for a in range(3))

    # A bounded query returns no sign at all when it finds nothing, so the bound
    # has to let every node reach the surface. The domain alone is not enough:
    # an explicit one can sit inside the solid, or away from it entirely, and
    # then its diagonal falls short. Span the domain together with the geometry.
    span = tuple(max(upper[a], tight_upper[a]) - min(lower[a], tight_lower[a]) for a in range(3))
    max_dist = math.sqrt(span[0] * span[0] + span[1] * span[1] + span[2] * span[2])

    origin = lower
    spacing_v = wp.vec3(*spacing)

    field = wp.empty(dims, dtype=wp.float32, device=device)
    wp.launch(
        swept_volume_field_kernel,
        dim=dims,
        inputs=[mesh_ids, transforms_wp, origin, spacing_v, float(max_dist), int(sign_mode)],
        outputs=[field],
        device=device,
    )

    lower_v = lower
    upper_v = upper
    return field, lower_v, upper_v


def swept_volume_mesh(
    meshes: Sequence[wp.Mesh],
    transforms: wp.array2d[wp.transform] | npt.ArrayLike,
    voxel_size: float | None = None,
    *,
    resolution: tuple[int, int, int] | None = None,
    lower: wp.vec3 | tuple[float, float, float] | None = None,
    upper: wp.vec3 | tuple[float, float, float] | None = None,
    threshold: float = 0.0,
    sign_mode: SweptVolumeSignMode = SweptVolumeSignMode.WINDING_NUMBER,
    device: DeviceLike | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Extract the swept volume (motion envelope) of animated rigid meshes.

    Samples the swept-volume signed-distance field with
    :func:`swept_volume_field` and extracts its ``threshold`` isosurface with marching
    cubes, returning a single triangle mesh in world coordinates that traces the
    union of every input mesh over every sampled pose.

    The field is evaluated by brute force at the provided pose samples. At low
    temporal sample rates relative to mesh speeds or thicknesses, the extracted
    swept volume will exhibit known "stroboscopic" defects.

    Args:
        meshes: Sequence of rest-pose :class:`warp.Mesh` objects.
        transforms: Per-mesh, per-sample rigid poses; see
            :func:`swept_volume_field`.
        voxel_size: Edge length of a grid cell in world units; see
            :func:`swept_volume_field`.
        resolution: Node counts ``(nx, ny, nz)``; see
            :func:`swept_volume_field`.
        lower: World coordinate that node ``(0, 0, 0)`` maps
            to. Defaults to the swept bounds padded by ``threshold`` plus roughly two
            cells, so the extracted level stays inside the domain. Supplying the
            corners makes that padding your responsibility.
        upper: World coordinate that node
            ``(nx-1, ny-1, nz-1)`` maps to.
        threshold: Field level to extract. ``threshold = 0.0`` traces the envelope through
            the sampled poses; a positive value dilates it outward. Due to grid
            sampling, the extracted marching cubes mesh is only guaranteed to
            enclose all sampled poses when ``threshold > 0.5 * hypot(hx, hy, hz)``.
        sign_mode: Inside/outside classification method; see
            :class:`SweptVolumeSignMode`. :attr:`SweptVolumeSignMode.NO_SIGN` requires a
            positive ``threshold`` to result in a non-empty surface, since the field
            is positive everywhere.
        device: Device on which to run. Defaults to the device of the first mesh.

    Returns:
        A tuple ``(vertices, indices)`` where ``vertices`` is a
        ``wp.array(dtype=wp.vec3)`` in world coordinates and ``indices`` is a
        flat ``wp.array(dtype=wp.int32)`` with three entries per triangle.

    Examples:
        Sweep a tetrahedron one unit along x. The swept solid spans
        ``(0, 0, 0)`` to ``(2, 1, 1)``, and extracting at the conservative
        ``threshold`` guarantees the envelope encloses it.

        >>> import numpy as np
        >>> import warp as wp
        >>> import warp.geometry as geo
        >>> points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> faces = [0, 2, 1, 0, 3, 2, 0, 1, 3, 1, 2, 3]
        >>> tet = wp.Mesh(wp.array(points, dtype=wp.vec3), wp.array(faces, dtype=wp.int32), support_winding_number=True)
        >>> xforms = np.zeros((1, 8, 7), dtype=np.float32)  # one mesh, eight poses
        >>> xforms[..., 6] = 1.0  # identity quaternions
        >>> xforms[0, :, 0] = np.linspace(0.0, 1.0, 8)  # translate along x
        >>> voxel_size = 0.05
        >>> vertices, indices = geo.swept_volume_mesh(
        ...     [tet], xforms, voxel_size=voxel_size, threshold=0.5 * np.sqrt(3.0) * voxel_size
        ... )
        >>> v = vertices.numpy()
        >>> bool(np.all(v.min(axis=0) <= 0.0) and np.all(v.max(axis=0) >= [2.0, 1.0, 1.0]))
        True
    """
    if sign_mode == SweptVolumeSignMode.NO_SIGN and threshold <= 0.0:
        raise ValueError(
            f"'threshold' must be positive when sign_mode is SweptVolumeSignMode.NO_SIGN, got {threshold}. "
            "An unsigned field is positive everywhere, so its zero isosurface is empty."
        )

    if len(meshes) == 0:
        raise ValueError("'meshes' must contain at least one mesh.")
    device = wp.get_device(device) if device is not None else meshes[0].device
    transforms = _swept_volume_transforms(transforms, device)
    _check_transforms(meshes, transforms)

    # Pad for the level being extracted, then sample that exact domain.
    lower, upper, dims, _, _ = _swept_volume_grid(
        meshes,
        transforms,
        voxel_size,
        resolution,
        lower,
        upper,
        max(threshold, 0.0),
        device,
    )

    field, lower, upper = swept_volume_field(
        meshes,
        transforms,
        resolution=dims,
        lower=lower,
        upper=upper,
        sign_mode=sign_mode,
        device=device,
    )

    return MarchingCubes.extract_surface_marching_cubes(
        field,
        threshold=threshold,
        domain_bounds_lower_corner=lower,
        domain_bounds_upper_corner=upper,
    )
