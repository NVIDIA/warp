# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closest pairs on closed simplices, evaluated by finite feature tests."""

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
def closest_points_seg_seg(p0: wp.vec3, p1: wp.vec3, q0: wp.vec3, q1: wp.vec3):
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
def closest_points_seg_seg(p0: wp.vec3d, p1: wp.vec3d, q0: wp.vec3d, q1: wp.vec3d):
    return _segment_pair(p0, p1, q0, q1, False)


@wp.func(inline=True)
def _triangle_segment(a: Any, b: Any, c: Any, p: Any, q: Any, scale_free: bool):
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
            return bary, t, wp.dot(delta, delta)

    s, t, best = closest_points_seg_seg(a, b, p, q)
    bary = type(a)(one - s, s, zero)
    edge_s, edge_t, d2 = closest_points_seg_seg(b, c, p, q)
    if d2 < best:
        bary = type(a)(zero, one - edge_s, edge_s)
        t = edge_t
        best = d2
    edge_s, edge_t, d2 = closest_points_seg_seg(c, a, p, q)
    if d2 < best:
        bary = type(a)(edge_s, zero, one - edge_s)
        t = edge_t
        best = d2
    return bary, t, best


@wp.func(inline=True)
def closest_points_tri_seg(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3, q: wp.vec3):
    """Return closest-point coordinates and squared distance for a triangle and segment.

    Args:
        a: First triangle vertex.
        b: Second triangle vertex.
        c: Third triangle vertex.
        p: Segment start.
        q: Segment end.

    Returns:
        Tuple ``(bary, t, distance_squared)``. The witnesses are
        ``bary[0]*a + bary[1]*b + bary[2]*c`` and ``p + t*(q-p)``.
        Barycentric coordinates are nonnegative and sum to one; ``t`` lies in
        [0, 1]. Degenerate triangles are treated as the union of their edges.

    This finite feature query includes piercing, coplanar, and boundary cases.
    Nonunique minima use deterministic input-order tie breaking. Derivatives
    follow the selected branch away from feature changes; they are not unique
    at ties. No geometric tolerance is applied. Inputs and intermediate products
    must be representable in the chosen precision. The float32 overload rescales
    its normal to reduce sensitivity to input scale. Finite forward results do
    not guarantee finite intermediate products in reverse mode, particularly at
    extreme scales.
    """
    return _triangle_segment(a, b, c, p, q, True)


@wp.func(inline=True)
def closest_points_tri_seg(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, p: wp.vec3d, q: wp.vec3d):
    return _triangle_segment(a, b, c, p, q, False)


register_api_function(closest_points_seg_seg, module="warp.geometry", python_callable=True, differentiable=True)
register_api_function(closest_points_tri_seg, module="warp.geometry", python_callable=True, differentiable=True)
