# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closest pairs on closed simplices, evaluated by finite feature tests."""

import enum
from typing import Any

import warp as wp
from warp._src.context import register_api_function


@wp.func(inline=True)
def _segment_pair(p0: Any, p1: Any, q0: Any, q1: Any, scale_free: bool):
    zero = type(p0[0])(0.0)
    one = type(p0[0])(1.0)
    u = p1 - p0
    v = q1 - q0
    r = p0 - q0
    a = wp.dot(u, u)
    e = wp.dot(v, v)
    f = wp.dot(v, r)
    s = zero
    t = zero
    if a == zero:
        if e > zero:
            t = wp.clamp(f / e, zero, one)
    elif e == zero:
        s = wp.clamp(-wp.dot(u, r) / a, zero, one)
    else:
        b = wp.dot(u, v)
        c = wp.dot(u, r)
        # A cross product avoids subtracting nearly equal squared dot products
        # for near-parallel segments. In particular, do not collapse short or
        # nearly parallel but nondegenerate inputs with an absolute epsilon.
        # s is invariant to scaling v. In float32, normalize its length to
        # reduce fourth-order products to second order. Dividing by e instead
        # can overflow when u is much longer than v. Float64 retains the
        # unscaled computation and its wider exponent range.
        w = v
        if scale_free:
            w = v * (one / wp.sqrt(e))
        n = wp.cross(u, w)
        denom = wp.dot(n, n)
        if denom > zero:
            s = wp.clamp(wp.dot(wp.cross(w, r), n) / denom, zero, one)
        t = (b * s + f) / e
        if t < zero:
            t = zero
            s = wp.clamp(-c / a, zero, one)
        elif t > one:
            t = one
            s = wp.clamp((b - c) / a, zero, one)
    delta = r + s * u - t * v
    return s, t, wp.dot(delta, delta)


@wp.func(inline=True)
def closest_point_seg_seg(p0: wp.vec3, p1: wp.vec3, q0: wp.vec3, q1: wp.vec3):
    """Return closest-point parameters and squared distance between two closed segments.

    Args:
        p0: First segment start.
        p1: First segment end.
        q0: Second segment start.
        q1: Second segment end.

    Returns:
        Tuple ``(s, t, distance_squared)``. The witnesses are
        ``p0 + s * (p1 - p0)`` and ``q0 + t * (q1 - q0)``, with both
        parameters in [0, 1]. A zero-length segment has parameter zero.
        Nonunique solutions select a deterministic pair for fixed input order.

    The selected branch is differentiable away from feature changes and ties.
    Derivatives at nonunique minima are branch derivatives, not unique geometric
    derivatives. No positive distance or length cutoff is applied. Inputs and
    intermediate products must be representable in the chosen precision. The
    float32 overload rescales its direction products to reduce sensitivity to
    input scale. Finite forward results do not guarantee finite intermediate
    products in reverse mode, particularly at extreme scales.
    """
    return _segment_pair(p0, p1, q0, q1, True)


@wp.func(inline=True)
def closest_point_seg_seg(p0: wp.vec3d, p1: wp.vec3d, q0: wp.vec3d, q1: wp.vec3d):
    return _segment_pair(p0, p1, q0, q1, False)


@wp.func(inline=True)
def _segment_triangle(p: Any, q: Any, a: Any, b: Any, c: Any, scale_free: bool):
    zero = type(a[0])(0.0)
    one = type(a[0])(1.0)
    ab = b - a
    ac = c - a
    direction = q - p
    # Normalize the float32 normal by a squared edge scale so that its squared
    # norm does not grow with the fourth power of the triangle's size.
    scale = one
    if scale_free:
        scale = wp.max(wp.dot(ab, ab), wp.dot(ac, ac))
    nn = zero
    n = type(a)(zero)
    if scale > zero:
        n = wp.cross(ab, ac)
        if scale_free:
            n = n * (one / scale)
        nn = wp.dot(n, n)
    if nn > zero:
        # The closest point on the segment to the supporting plane is globally
        # optimal if its projection lies in the triangle. Otherwise a closest
        # pair exists on a triangle edge (also when parallel to the plane).
        height = wp.dot(n, p - a)
        slope = wp.dot(n, direction)
        t = zero
        if slope != zero:
            t = wp.clamp(-height / slope, zero, one)
        offset = (p - a) + t * direction
        v = zero
        w = zero
        if scale_free:
            inv_nn = one / (nn * scale)
            v = wp.dot(wp.cross(offset, ac), n) * inv_nn
            w = wp.dot(wp.cross(ab, offset), n) * inv_nn
        else:
            v = wp.dot(wp.cross(offset, ac), n) / nn
            w = wp.dot(wp.cross(ab, offset), n) / nn
        # Test the returned weight itself: v + w <= 1 can hold after rounding
        # while one - v - w is still slightly negative.
        u = one - v - w
        if u >= zero and v >= zero and w >= zero:
            bary = type(a)(u, v, w)
            delta = v * ab + w * ac - offset
            return t, bary, wp.dot(delta, delta)

    s, t, best = closest_point_seg_seg(a, b, p, q)
    bary = type(a)(one - s, s, zero)
    edge_s, edge_t, d2 = closest_point_seg_seg(b, c, p, q)
    if d2 < best:
        bary = type(a)(zero, one - edge_s, edge_s)
        t = edge_t
        best = d2
    edge_s, edge_t, d2 = closest_point_seg_seg(c, a, p, q)
    if d2 < best:
        bary = type(a)(edge_s, zero, one - edge_s)
        t = edge_t
        best = d2
    return t, bary, best


@wp.func(inline=True)
def closest_point_seg_tri(p0: wp.vec3, p1: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3):
    """Return closest-point coordinates and squared distance for a segment and triangle.

    Args:
        p0: Segment start.
        p1: Segment end.
        v0: First triangle vertex.
        v1: Second triangle vertex.
        v2: Third triangle vertex.

    Returns:
        Tuple ``(t, bary, distance_squared)``. The witnesses are
        ``p0 + t * (p1 - p0)`` and ``bary[0] * v0 + bary[1] * v1 + bary[2] * v2``.
        ``t`` lies in [0, 1]; barycentric coordinates are nonnegative and sum to
        one. Degenerate triangles are treated as the union of their edges.

    This finite feature query includes piercing, coplanar, and boundary cases.
    Nonunique minima use deterministic input-order tie breaking. Derivatives
    follow the selected branch away from feature changes; they are not unique
    at ties. No geometric tolerance is applied. Inputs and intermediate products
    must be representable in the chosen precision. The float32 overload rescales
    its normal to reduce sensitivity to input scale. Finite forward results do
    not guarantee finite intermediate products in reverse mode, particularly at
    extreme scales.
    """
    return _segment_triangle(p0, p1, v0, v1, v2, True)


@wp.func(inline=True)
def closest_point_seg_tri(p0: wp.vec3d, p1: wp.vec3d, v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d):
    return _segment_triangle(p0, p1, v0, v1, v2, False)


register_api_function(closest_point_seg_seg, module="warp.geometry", python_callable=True, differentiable=True)
register_api_function(closest_point_seg_tri, module="warp.geometry", python_callable=True, differentiable=True)


class TriangleFeature(enum.IntEnum):
    """Triangle feature containing a closest point in its relative interior."""

    VERTEX_0 = 0
    """Vertex ``v0``."""
    VERTEX_1 = 1
    """Vertex ``v1``."""
    VERTEX_2 = 2
    """Vertex ``v2``."""
    EDGE_01 = 3
    """Open edge between ``v0`` and ``v1``."""
    EDGE_02 = 4
    """Open edge between ``v0`` and ``v2``."""
    EDGE_12 = 5
    """Open edge between ``v1`` and ``v2``."""
    FACE = 6
    """Open face interior."""


@wp.func(inline=True)
def _triangle_feature(bary: Any):
    zero = type(bary[0])(0.0)
    # Support masks 1, 2, 4 name vertices; 3, 5, 6 name edges; 7 names the face.
    mask = int(bary[0] > zero) + 2 * int(bary[1] > zero) + 4 * int(bary[2] > zero)
    return mask - 1 + int(mask == 3) - int(mask == 4)


@wp.func(inline=True)
def _point_triangle(p: Any, a: Any, b: Any, c: Any):
    zero = type(a[0])(0.0)
    one = type(a[0])(1.0)
    ab0 = b - a
    ac0 = c - a
    ap0 = p - a
    size = wp.max(wp.max(wp.abs(ab0)), wp.max(wp.abs(ac0)))
    # Exact powers of two change exponent range without rounding the shape.
    # These thresholds choose arithmetic, not a geometric degeneracy cutoff.
    factor = one
    if size < type(a[0])(1.0e-4):
        factor = type(a[0])(4294967296.0)
    elif size > type(a[0])(1.0e4):
        factor = type(a[0])(2.3283064365386963e-10)
    ab = ab0 * factor
    ac = ac0 * factor
    ap = ap0 * factor
    bp = (p - b) * factor
    cp = (p - c) * factor
    n = wp.cross(ab, ac)
    ns = wp.max(wp.abs(n))
    if ns > zero and ns < type(a[0])(1.0e-8):
        n = n * type(a[0])(18446744073709551616.0)
    elif ns > type(a[0])(1.0e8):
        n = n * type(a[0])(5.421010862427522e-20)
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    # Keep the established region determinants on well-shaped triangles.
    # Cross products avoid their cancellation on slivers; no feature is discarded.
    sliver = ns < type(a[0])(0.01) * (size * factor) * (size * factor)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    if sliver:
        vc = wp.dot(wp.cross(ab, ap), n)
        vb = wp.dot(wp.cross(ap, ac), n)
        va = wp.dot(wp.cross(bp, cp), n)
    weights = type(a)(va, vb, vc)
    if va < zero or vb < zero or vc < zero or va + vb + vc <= zero:
        if d1 <= zero and d2 <= zero:
            weights = type(a)(one, zero, zero)
        elif d3 >= zero and d4 <= d3:
            weights = type(a)(zero, one, zero)
        elif d6 >= zero and d5 <= d6:
            weights = type(a)(zero, zero, one)
        elif vc <= zero and d1 >= zero and d3 <= zero and d1 > d3:
            weights = type(a)(-d3, d1, zero)
        elif vb <= zero and d2 >= zero and d6 <= zero and d2 > d6:
            weights = type(a)(-d6, zero, d2)
        elif va <= zero and d4 >= d3 and d5 >= d6 and (d4 > d3 or d5 > d6):
            weights = type(a)(zero, d5 - d6, d4 - d3)
        else:
            weights = wp.max(weights, type(a)(zero))
    elif sliver:
        # On slivers, separately computed area weights lose their shared
        # long-edge coordinate. Recover that coordinate with a dot projection.
        denominator = wp.dot(wp.cross(ab, ac), n)
        ab2 = wp.dot(ab, ab)
        ac2 = wp.dot(ac, ac)
        abac = wp.dot(ab, ac)
        v = zero
        w = zero
        if ab2 >= ac2:
            w = vc / denominator
            v = (wp.dot(ab, ap) - w * abac) / ab2
        else:
            v = vb / denominator
            w = (wp.dot(ac, ap) - v * abac) / ac2
        u = one - v - w
        weights = wp.max(type(a)(u, v, w), type(a)(zero))
        if va == zero:
            weights[0] = zero
        if vb == zero:
            weights[1] = zero
        if vc == zero:
            weights[2] = zero
    # Share one reciprocal across all feature regions and all three weights.
    # Do not skip it just because total rounds to one: its derivative can be nonzero.
    total = weights[0] + weights[1] + weights[2]
    bary = type(a)(one, zero, zero)
    if total > zero:
        bary = weights * (one / total)
    # Assign the largest weight by partition of unity. Unlike reconstructing
    # an arbitrarily chosen weight, this cannot erase a small positive weight.
    if bary[0] >= bary[1] and bary[0] >= bary[2]:
        bary[0] = one - (bary[1] + bary[2])
    elif bary[1] >= bary[2]:
        bary[1] = one - (bary[0] + bary[2])
    else:
        bary[2] = one - (bary[0] + bary[1])
    delta = bary[1] * ab0 + bary[2] * ac0 - ap0
    return bary, wp.dot(delta, delta), _triangle_feature(bary)


@wp.func(inline=True)
def closest_point_point_tri(p: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3):
    """Return the closest point on a closed triangle as barycentric coordinates.

    Args:
        p: Query point.
        v0: First triangle vertex.
        v1: Second triangle vertex.
        v2: Third triangle vertex.

    Returns:
        Tuple ``(bary, distance_squared, feature)``. The closest point is
        ``bary[0] * v0 + bary[1] * v1 + bary[2] * v2``. The nonnegative weights
        sum to one up to rounding. ``feature`` is an integer :class:`TriangleFeature`
        identifying the relative interior of the returned barycentric support:
        a vertex, an open edge, or the open face.

    For reconstruction at large world coordinates, prefer
    ``v0 + bary[1] * (v1 - v0) + bary[2] * (v2 - v0)`` to the weighted sum.

    No geometric tolerance is applied. Degenerate triangles are treated as
    closed edges, with deterministic input-order tie breaking. At a degenerate
    triangle, different barycentric representations can name different features
    at the same physical point. Boundary points select the lower-dimensional
    feature. Floating-point rounding can change classifications near a boundary.

    The coordinates and squared distance are differentiable away from feature
    changes and ties; the integer feature is not differentiable. Inputs and
    intermediate differences and products must be representable in the chosen
    precision. Finite forward values do not guarantee finite reverse-mode
    intermediate values at extreme scales.
    """
    return _point_triangle(p, v0, v1, v2)


@wp.func(inline=True)
def closest_point_point_tri(p: wp.vec3d, v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d):
    return _point_triangle(p, v0, v1, v2)


register_api_function(closest_point_point_tri, module="warp.geometry", python_callable=True, differentiable=True)
