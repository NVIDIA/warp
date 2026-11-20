# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometric intersection predicates implemented as Warp functions."""

# The triangle-triangle test is adapted from Tomas Möller's division-free test,
# as used by ``warp/native/intersect_tri.h``. That implementation has this license:
#
# Copyright 2020 Tomas Akenine-Möller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Any

import warp as wp
from warp._src.context import register_api_function

_DEFAULT_INTERSECTION_TOL = wp.float64(1.0e-6)


def _register_predicate(function):
    """Document a predicate in ``warp.geometry`` and skip its adjoint at every call site.

    The predicates return ``bool``, so their adjoints are identically zero. Marking every
    overload non-differentiable stops backward-enabled kernels from calling those adjoints,
    which otherwise dominate the compile time of kernels that use the predicates.
    """
    for overload in (function, *function.user_overloads.values()):
        overload.is_differentiable = False
    register_api_function(function, module="warp.geometry", python_callable=True, differentiable=False)


@wp.func(inline=True)
def _component(v: Any, index: int):
    # Dynamic vector subscripts force CUDA to materialize local arrays. Select
    # from fixed components instead so projection coordinates stay in registers.
    if index == 0:
        return v[0]
    if index == 1:
        return v[1]
    return v[2]


@wp.func(inline=True)
def _snap_plane_distance(d: wp.float32, tol: wp.float32):
    if wp.abs(d) < tol:
        return wp.float32(0.0)
    return d


@wp.func(inline=True)
def _snap_plane_distance(d: wp.float32, tol: wp.float64):
    # Preserve the native double cutoff without float64 work for default calls.
    # float32(1e-6) is the largest float32 below the native double EPSILON.
    if tol == wp.float64(1.0e-6):
        if wp.abs(d) <= wp.float32(1.0e-6):
            return wp.float32(0.0)
    elif wp.float64(wp.abs(d)) < tol:
        return wp.float32(0.0)
    return d


@wp.func(inline=True)
def _snap_plane_distance(d: wp.float64, tol: wp.float64):
    if wp.abs(d) < tol:
        return wp.float64(0.0)
    return d


@wp.func(inline=True)
def _snap_plane_distance(d: wp.float64, tol: wp.float32):
    return _snap_plane_distance(d, wp.float64(tol))


@wp.func(inline=True)
def _plane_distance(n: Any, p: Any, offset: Any, tol: Any):
    d = (n[0] * p[0] + n[1] * p[1]) + n[2] * p[2] - offset
    return _snap_plane_distance(d, tol)


@wp.func(inline=True)
def _edge_edge(v0: Any, v1: Any, u0: Any, u1: Any, i: int, j: int):
    ax = _component(v1, i) - _component(v0, i)
    ay = _component(v1, j) - _component(v0, j)
    bx = _component(u0, i) - _component(u1, i)
    by = _component(u0, j) - _component(u1, j)
    cx = _component(v0, i) - _component(u0, i)
    cy = _component(v0, j) - _component(u0, j)
    f = ay * bx - ax * by
    d = by * cx - bx * cy
    zero = type(f)(0.0)
    if (f > zero and d >= zero and d <= f) or (f < zero and d <= zero and d >= f):
        e = ax * cy - ay * cx
        if f > zero:
            return e >= zero and e <= f
        return e <= zero and e >= f
    return False


@wp.func(inline=True)
def _point_in_triangle(p: Any, u0: Any, u1: Any, u2: Any, i: int, j: int):
    a = _component(u1, j) - _component(u0, j)
    b = -(_component(u1, i) - _component(u0, i))
    c = -a * _component(u0, i) - b * _component(u0, j)
    d0 = a * _component(p, i) + b * _component(p, j) + c
    a = _component(u2, j) - _component(u1, j)
    b = -(_component(u2, i) - _component(u1, i))
    c = -a * _component(u1, i) - b * _component(u1, j)
    d1 = a * _component(p, i) + b * _component(p, j) + c
    a = _component(u0, j) - _component(u2, j)
    b = -(_component(u0, i) - _component(u2, i))
    c = -a * _component(u2, i) - b * _component(u2, j)
    d2 = a * _component(p, i) + b * _component(p, j) + c
    zero = type(d0)(0.0)
    return d0 * d1 > zero and d0 * d2 > zero


@wp.func(inline=True)
def _coplanar_tri_tri(n: Any, v0: Any, v1: Any, v2: Any, u0: Any, u1: Any, u2: Any):
    # Project onto the coordinate plane with the largest triangle area.
    a = wp.abs(n)
    i = int(0)
    j = int(1)
    if a[0] > a[1]:
        if a[0] > a[2]:
            i = 1
            j = 2
    elif a[2] <= a[1]:
        j = 2

    if _edge_edge(v0, v1, u0, u1, i, j):
        return True
    if _edge_edge(v0, v1, u1, u2, i, j):
        return True
    if _edge_edge(v0, v1, u2, u0, i, j):
        return True
    if _edge_edge(v1, v2, u0, u1, i, j):
        return True
    if _edge_edge(v1, v2, u1, u2, i, j):
        return True
    if _edge_edge(v1, v2, u2, u0, i, j):
        return True
    if _edge_edge(v2, v0, u0, u1, i, j):
        return True
    if _edge_edge(v2, v0, u1, u2, i, j):
        return True
    if _edge_edge(v2, v0, u2, u0, i, j):
        return True
    if _point_in_triangle(v0, u0, u1, u2, i, j):
        return True
    if _point_in_triangle(u0, v0, v1, v2, i, j):
        return True
    return False


@wp.func(inline=True)
def _interval(v0: Any, v1: Any, v2: Any, d0: Any, d1: Any, d2: Any):
    # Choose the vertex opposite the two on the same side of the other plane.
    # The caller handles the all-zero (coplanar) distances before entering here.
    # Select the inputs first, then evaluate the coefficients once. Keeping
    # arithmetic outside divergent branches reduces double-precision work.
    zero = type(d0)(0.0)
    a = v0
    b = v1
    c = v2
    d = d0
    db = d1
    dc = d2
    if d0 * d1 > zero:
        a = v2
        b = v0
        c = v1
        d = d2
        db = d0
        dc = d1
    elif d0 * d2 > zero:
        a = v1
        b = v0
        c = v2
        d = d1
        db = d0
        dc = d2
    elif not (d1 * d2 > zero or d0 != zero):
        if d1 != zero:
            a = v1
            b = v0
            c = v2
            d = d1
            db = d0
            dc = d2
        else:
            a = v2
            b = v0
            c = v1
            d = d2
            db = d0
            dc = d1
    return a, (b - a) * d, (c - a) * d, d - db, d - dc


@wp.func(inline=True)
def _intersect_tri_tri(v0: Any, v1: Any, v2: Any, u0: Any, u1: Any, u2: Any, tol: Any):
    zero = type(v0[0])(0.0)
    n1 = wp.cross(v1 - v0, v2 - v0)
    offset1 = (n1[0] * v0[0] + n1[1] * v0[1]) + n1[2] * v0[2]
    du0 = _plane_distance(n1, u0, offset1, tol)
    du1 = _plane_distance(n1, u1, offset1, tol)
    du2 = _plane_distance(n1, u2, offset1, tol)
    if du0 * du1 > zero and du0 * du2 > zero:
        return False

    n2 = wp.cross(u1 - u0, u2 - u0)
    offset2 = (n2[0] * u0[0] + n2[1] * u0[1]) + n2[2] * u0[2]
    dv0 = _plane_distance(n2, v0, offset2, tol)
    dv1 = _plane_distance(n2, v1, offset2, tol)
    dv2 = _plane_distance(n2, v2, offset2, tol)
    if dv0 * dv1 > zero and dv0 * dv2 > zero:
        return False

    if dv0 == zero or du0 == zero:
        # Most noncoplanar pairs avoid the remaining four zero comparisons.
        if (dv0 == zero and dv1 == zero and dv2 == zero) or (du0 == zero and du1 == zero and du2 == zero):
            return _coplanar_tri_tri(n1, v0, v1, v2, u0, u1, u2)

    direction = wp.abs(wp.cross(n1, n2))
    index = int(0)
    largest = direction[0]
    if direction[1] > largest:
        index = 1
        largest = direction[1]
    if direction[2] > largest:
        index = 2

    a, b, c, x0, x1 = _interval(_component(v0, index), _component(v1, index), _component(v2, index), dv0, dv1, dv2)
    d, e, f, y0, y1 = _interval(_component(u0, index), _component(u1, index), _component(u2, index), du0, du1, du2)
    xx = x0 * x1
    yy = y0 * y1
    xxyy = xx * yy
    p0 = a * xxyy + b * x1 * yy
    p1 = a * xxyy + c * x0 * yy
    q0 = d * xxyy + e * xx * y1
    q1 = d * xxyy + f * xx * y0
    if wp.max(p0, p1) < wp.min(q0, q1) or wp.max(q0, q1) < wp.min(p0, p1):
        return False
    return True


@wp.func(inline=True)
def intersect_tri_tri(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    u0: wp.vec3,
    u1: wp.vec3,
    u2: wp.vec3,
    tol: wp.float64 = _DEFAULT_INTERSECTION_TOL,
) -> bool:
    """Test whether two triangles intersect.

    Overloads accept ``vec3`` or ``vec3d`` vertices, each with a float32 or
    float64 ``tol``. All vertices must have the same type and be in the same
    coordinate space. Coplanar overlap counts as an intersection.

    The test follows Möller's division-free method. A vertex's plane value is ``d = dot(n, p) - dot(n, a)``, where
    ``n = cross(b - a, c - a)`` is the other triangle's unnormalized normal.
    Values satisfying ``abs(d) < tol`` are snapped to zero. Consequently,
    ``tol`` is not a world-space distance: for a nondegenerate triangle its
    corresponding distance threshold is ``tol / length(n)``. Uniformly scaling
    all coordinates by ``s`` scales plane values by ``abs(s)**3``; scale ``tol``
    by the same factor to preserve the classification, apart from roundoff.
    Interval and coplanar edge tests receive no additional tolerance.

    The omitted tolerance matches :func:`warp.intersect_tri_tri` and is stored
    in float64 even for ``vec3`` inputs. Explicit float32 tolerances use their
    rounded float32 values. To spell the exact default explicitly, use
    ``tol=wp.float64(1e-6)``; an ordinary kernel literal ``tol=1e-6`` is float32
    and differs immediately at the cutoff.

    This is a floating-point predicate, not an exact orientation test. Results
    near degeneracy may be inaccurate; use ``vec3d`` for greater precision.
    Zero-area triangles are not supported. The discrete result is not
    differentiable.

    Args:
        v0: First vertex of the first triangle.
        v1: Second vertex of the first triangle.
        v2: Third vertex of the first triangle.
        u0: First vertex of the second triangle.
        u1: Second vertex of the second triangle.
        u2: Third vertex of the second triangle.
        tol: Nonnegative finite plane-equation tolerance, in float32 or float64.
            Zero disables snapping but retains exact floating-point coplanarity
            and boundary checks. Increasing it does not dilate triangle edges.

    Returns:
        ``True`` if the triangles intersect, or ``False`` otherwise.

    Example:
        Given triangle vertices inside a Warp kernel or function::

            hit = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2)
            hit = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol=1.0e-5)
            hit = wp.geometry.intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol=0.0)
    """
    return _intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol)


@wp.func(inline=True)
def intersect_tri_tri(
    v0: wp.vec3d,
    v1: wp.vec3d,
    v2: wp.vec3d,
    u0: wp.vec3d,
    u1: wp.vec3d,
    u2: wp.vec3d,
    tol: wp.float64 = _DEFAULT_INTERSECTION_TOL,
) -> bool:
    """Double-precision overload of :func:`warp.geometry.intersect_tri_tri`.

    Use the same coordinate space for all six vertices. Return ``True`` for an
    intersection, including coplanar overlap, or ``False`` otherwise. The plane
    tolerance and limitations are the same as for the single-precision overload.
    """
    return _intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol)


@wp.func(inline=True)
def intersect_tri_tri(
    v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, u0: wp.vec3, u1: wp.vec3, u2: wp.vec3, tol: wp.float32
) -> bool:
    """Overload accepting a float32 plane tolerance; see :func:`warp.geometry.intersect_tri_tri`."""
    return _intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol)


@wp.func(inline=True)
def intersect_tri_tri(
    v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d, u0: wp.vec3d, u1: wp.vec3d, u2: wp.vec3d, tol: wp.float32
) -> bool:
    """Overload accepting a float32 plane tolerance; see :func:`warp.geometry.intersect_tri_tri`."""
    return _intersect_tri_tri(v0, v1, v2, u0, u1, u2, tol)


intersect_tri_tri.__doc__ = intersect_tri_tri.doc
_register_predicate(intersect_tri_tri)


@wp.func(inline=True)
def _orient_projected(a: Any, b: Any, p: Any, i: int, j: int):
    return (_component(b, i) - _component(a, i)) * (_component(p, j) - _component(a, j)) - (
        _component(b, j) - _component(a, j)
    ) * (_component(p, i) - _component(a, i))


@wp.func(inline=True)
def _intersect_seg_tri_projected(p0: Any, p1: Any, v0: Any, v1: Any, v2: Any, n: Any, d0: Any, d1: Any):
    zero = type(p0[0])(0.0)
    one = type(p0[0])(1.0)
    a = wp.abs(n)
    # Project onto the coordinate plane with the largest triangle area.
    i = int(0)
    j = int(1)
    if a[0] > a[1]:
        if a[0] > a[2]:
            i = 1
            j = 2
    elif a[2] <= a[1]:
        j = 2
    area = _orient_projected(v0, v1, v2, i, j)
    if area == zero:
        return False

    # A single endpoint on the plane reduces the intersection to a point.
    if d0 != zero:
        p0 = p1
    elif d1 != zero:
        p1 = p0

    # Separating-axis tests need no intersection parameters or divisions. The
    # triangle edge normals and the segment normal form a complete set in 2D.
    orientation = one
    if area < zero:
        orientation = -one
    a0 = orientation * _orient_projected(v0, v1, p0, i, j)
    a1 = orientation * _orient_projected(v0, v1, p1, i, j)
    if a0 < zero and a1 < zero:
        return False
    b0 = orientation * _orient_projected(v1, v2, p0, i, j)
    b1 = orientation * _orient_projected(v1, v2, p1, i, j)
    if b0 < zero and b1 < zero:
        return False
    c0 = orientation * _orient_projected(v2, v0, p0, i, j)
    c1 = orientation * _orient_projected(v2, v0, p1, i, j)
    if c0 < zero and c1 < zero:
        return False

    # The segment's supporting line supplies the final separating axis.
    s0 = _orient_projected(p0, p1, v0, i, j)
    s1 = _orient_projected(p0, p1, v1, i, j)
    s2 = _orient_projected(p0, p1, v2, i, j)
    return not ((s0 > zero and s1 > zero and s2 > zero) or (s0 < zero and s1 < zero and s2 < zero))


@wp.func(inline=True)
def _intersect_seg_tri(p0: Any, p1: Any, v0: Any, v1: Any, v2: Any, tol: Any):
    zero = type(p0[0])(0.0)
    e1 = v1 - v0
    e2 = v2 - v0
    n = wp.cross(e1, e2)
    offset = (n[0] * v0[0] + n[1] * v0[1]) + n[2] * v0[2]
    d0 = _plane_distance(n, p0, offset, tol)
    d1 = _plane_distance(n, p1, offset, tol)
    if d0 == zero or d1 == zero:
        # Preserve projection-based snapping for endpoint and coplanar contact.
        return _intersect_seg_tri_projected(p0, p1, v0, v1, v2, n, d0, d1)
    if (d0 > zero) == (d1 > zero):
        return False

    # Moller-Trumbore barycentric numerators. Opposite plane sides already
    # establish 0 < t < 1, so neither t nor a reciprocal is needed. Compare
    # against the determinant directly and normalize its sign for both windings.
    direction = p0 - p1
    det = wp.dot(direction, n)
    e = wp.cross(direction, p0 - v0)
    v = wp.dot(e2, e)
    w = -wp.dot(e1, e)
    if det < zero:
        det = -det
        v = -v
        w = -w
    return det > zero and v >= zero and w >= zero and v + w <= det


@wp.func(inline=True)
def intersect_seg_tri(
    p0: wp.vec3, p1: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, tol: wp.float64 = _DEFAULT_INTERSECTION_TOL
) -> bool:
    """Test whether a closed segment intersects a triangle.

    Overloads accept ``vec3`` or ``vec3d`` points, each with a float32 or float64
    ``tol``. All points must have the same type and be in the same coordinate
    space. Endpoint, edge, vertex, and coplanar contact count as
    intersections. A zero-length segment is treated as a point. A zero-area
    triangle returns ``False``.

    With ``n = cross(v1 - v0, v2 - v0)``, endpoint plane values
    ``d = dot(n, p) - dot(n, v0)`` satisfying ``abs(d) < tol`` are snapped to
    zero. Such endpoints are treated as lying in the triangle's plane; the
    dominant-axis projection determines their position within that plane.
    Barycentric and projected edge tests receive no additional tolerance.

    The normal is unnormalized: the corresponding world-space distance
    threshold is ``tol / length(n)``, so ``tol`` depends on triangle area.
    Under uniform coordinate scaling by ``s``, scale ``tol`` by ``abs(s)**3``
    to preserve the classification, apart from roundoff. See
    :func:`warp.geometry.intersect_tri_tri` for the default tolerance's float64
    representation and the rounding of explicit float32 tolerances.

    This floating-point predicate is not exact and may be inaccurate near
    degeneracy. Use ``vec3d`` for greater precision. The result is not
    differentiable.

    Args:
        p0: First endpoint of the segment.
        p1: Second endpoint of the segment.
        v0: First vertex of the triangle.
        v1: Second vertex of the triangle.
        v2: Third vertex of the triangle.
        tol: Nonnegative finite plane-equation tolerance, in float32 or float64.
            Zero disables snapping but retains exact floating-point coplanarity
            and boundary checks. Increasing it does not dilate triangle edges.

    Returns:
        ``True`` if the segment intersects the triangle, or ``False`` otherwise.

    Example:
        Given segment and triangle vertices inside a Warp kernel or function::

            hit = wp.geometry.intersect_seg_tri(p0, p1, v0, v1, v2, tol=0.0)
            hit = wp.geometry.intersect_seg_tri(p0, p1, v0, v1, v2, tol=user_tolerance)

        ``user_tolerance`` may be a runtime float32 or float64 scalar.
    """
    return _intersect_seg_tri(p0, p1, v0, v1, v2, tol)


@wp.func(inline=True)
def intersect_seg_tri(
    p0: wp.vec3d, p1: wp.vec3d, v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d, tol: wp.float64 = _DEFAULT_INTERSECTION_TOL
) -> bool:
    """Double-precision overload of :func:`warp.geometry.intersect_seg_tri`.

    Return ``True`` for an intersection, including boundary and coplanar contact.
    The tolerance and limitations are the same as for the single-precision overload.
    """
    return _intersect_seg_tri(p0, p1, v0, v1, v2, tol)


@wp.func(inline=True)
def intersect_seg_tri(p0: wp.vec3, p1: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, tol: wp.float32) -> bool:
    """Overload accepting a float32 plane tolerance; see :func:`warp.geometry.intersect_seg_tri`."""
    return _intersect_seg_tri(p0, p1, v0, v1, v2, tol)


@wp.func(inline=True)
def intersect_seg_tri(p0: wp.vec3d, p1: wp.vec3d, v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d, tol: wp.float32) -> bool:
    """Overload accepting a float32 plane tolerance; see :func:`warp.geometry.intersect_seg_tri`."""
    return _intersect_seg_tri(p0, p1, v0, v1, v2, tol)


intersect_seg_tri.__doc__ = intersect_seg_tri.doc
_register_predicate(intersect_seg_tri)


@wp.func(inline=True)
def _intersect_aabb_aabb(a_lower: Any, a_upper: Any, b_lower: Any, b_upper: Any):
    # Evaluate all six comparisons without short-circuit control flow. Integer
    # masks compile to predicate operations and avoid divergent scalar loads.
    return bool(
        int(a_lower[0] <= b_upper[0])
        & int(a_lower[1] <= b_upper[1])
        & int(a_lower[2] <= b_upper[2])
        & int(a_upper[0] >= b_lower[0])
        & int(a_upper[1] >= b_lower[1])
        & int(a_upper[2] >= b_lower[2])
    )


@wp.func(inline=True)
def intersect_aabb_aabb(a_lower: wp.vec3, a_upper: wp.vec3, b_lower: wp.vec3, b_upper: wp.vec3) -> bool:
    """Test whether two closed axis-aligned bounding boxes overlap.

    Face, edge, and corner contact count as intersections. Zero-width boxes
    (planes, lines, or points) are supported. All coordinates must be finite,
    use the same type (``vec3`` or ``vec3d``), and share a coordinate space.
    Each lower bound must be componentwise less than or equal to its upper
    bound. No tolerance is added; expand the bounds to request a spatial margin.
    The boolean result is not differentiable.

    Args:
        a_lower: Lower corner of the first box.
        a_upper: Upper corner of the first box.
        b_lower: Lower corner of the second box.
        b_upper: Upper corner of the second box.

    Returns:
        ``True`` if the boxes overlap, or ``False`` otherwise.
    """
    return _intersect_aabb_aabb(a_lower, a_upper, b_lower, b_upper)


@wp.func(inline=True)
def intersect_aabb_aabb(a_lower: wp.vec3d, a_upper: wp.vec3d, b_lower: wp.vec3d, b_upper: wp.vec3d) -> bool:
    """Double-precision overload of :func:`warp.geometry.intersect_aabb_aabb`."""
    return _intersect_aabb_aabb(a_lower, a_upper, b_lower, b_upper)


intersect_aabb_aabb.__doc__ = intersect_aabb_aabb.doc
_register_predicate(intersect_aabb_aabb)


@wp.func(inline=True)
def _intersect_seg_aabb(p0: Any, p1: Any, lower: Any, upper: Any):
    # Combine separating axes without short-circuit branches. This avoids
    # divergent control flow for mixed hits/misses on the GPU.
    face = _intersect_aabb_aabb(wp.min(p0, p1), wp.max(p0, p1), lower, upper)
    # Doubled midpoint offset and box half-extents avoid division by two.
    # Translate endpoints before adding to reduce cancellation under translation.
    m = (p0 - lower) + (p1 - upper)
    d = p1 - p0
    h = upper - lower
    ad = wp.abs(d)
    return bool(
        int(face)
        & int(wp.abs(m[1] * d[2] - m[2] * d[1]) <= h[1] * ad[2] + h[2] * ad[1])
        & int(wp.abs(m[2] * d[0] - m[0] * d[2]) <= h[2] * ad[0] + h[0] * ad[2])
        & int(wp.abs(m[0] * d[1] - m[1] * d[0]) <= h[0] * ad[1] + h[1] * ad[0])
    )


@wp.func(inline=True)
def intersect_seg_aabb(p0: wp.vec3, p1: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> bool:
    """Test whether a closed segment intersects an axis-aligned bounding box.

    Endpoint, face, edge, and corner contact count as intersections. A zero-length
    segment is a point; zero-width boxes and axis-parallel segments are supported.

    Coordinates must be finite, use the same type (``vec3`` or ``vec3d``), and
    share a coordinate space. ``lower`` must be componentwise less than or equal
    to ``upper``. No tolerance is added; expand the bounds for a spatial margin.
    This is a floating-point predicate, not an exact predicate: roundoff and
    intermediate overflow or underflow can affect results. Use ``vec3d`` for
    greater precision. The boolean result is not differentiable.

    Args:
        p0: First endpoint of the segment.
        p1: Second endpoint of the segment.
        lower: Lower corner of the box.
        upper: Upper corner of the box.

    Returns:
        ``True`` if the segment intersects the box, or ``False`` otherwise.
    """
    return _intersect_seg_aabb(p0, p1, lower, upper)


@wp.func(inline=True)
def intersect_seg_aabb(p0: wp.vec3d, p1: wp.vec3d, lower: wp.vec3d, upper: wp.vec3d) -> bool:
    """Double-precision overload of :func:`warp.geometry.intersect_seg_aabb`."""
    return _intersect_seg_aabb(p0, p1, lower, upper)


intersect_seg_aabb.__doc__ = intersect_seg_aabb.doc
_register_predicate(intersect_seg_aabb)


@wp.func(inline=True)
def _tri_aabb_edge_axes(e: Any, a: Any, b: Any, h: Any):
    # For an axis perpendicular to this edge, its two endpoints have the same
    # projection. Only one endpoint (a) and the opposite vertex (b) are needed.
    p = e[2] * a[1] - e[1] * a[2]
    q = e[2] * b[1] - e[1] * b[2]
    r = wp.abs(e[2]) * h[1] + wp.abs(e[1]) * h[2]
    x = int(wp.min(p, q) <= r) & int(wp.max(p, q) >= -r)
    p = e[0] * a[2] - e[2] * a[0]
    q = e[0] * b[2] - e[2] * b[0]
    r = wp.abs(e[0]) * h[2] + wp.abs(e[2]) * h[0]
    y = int(wp.min(p, q) <= r) & int(wp.max(p, q) >= -r)
    p = e[1] * a[0] - e[0] * a[1]
    q = e[1] * b[0] - e[0] * b[1]
    r = wp.abs(e[1]) * h[0] + wp.abs(e[0]) * h[1]
    return bool(x & y & int(wp.min(p, q) <= r) & int(wp.max(p, q) >= -r))


@wp.func(inline=True)
def _intersect_tri_aabb(v0: Any, v1: Any, v2: Any, lower: Any, upper: Any):
    if not _intersect_aabb_aabb(wp.min(v0, wp.min(v1, v2)), wp.max(v0, wp.max(v1, v2)), lower, upper):
        return False
    h = (upper - lower) * type(v0[0])(0.5)
    a = (v0 - lower) - h
    b = (v1 - lower) - h
    c = (v2 - lower) - h
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    n = wp.cross(e0, e1)
    if wp.abs(wp.dot(n, a)) > wp.dot(wp.abs(n), h):
        return False
    return bool(
        int(_tri_aabb_edge_axes(e0, a, c, h))
        & int(_tri_aabb_edge_axes(e1, b, a, h))
        & int(_tri_aabb_edge_axes(e2, c, b, h))
    )


@wp.func(inline=True)
def intersect_tri_aabb(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> bool:
    """Test whether a triangle intersects a closed axis-aligned bounding box.

    Boundary contact counts as intersection. Degenerate triangles are treated as their segment or point
    convex hull; zero-width boxes are supported.

    Coordinates must be finite, use the same type (``vec3`` or ``vec3d``), and
    share a coordinate space. ``lower`` must be componentwise less than or equal
    to ``upper``. No tolerance is added; expand the bounds for a spatial margin.
    This is a floating-point predicate, not an exact predicate: roundoff and
    intermediate overflow or underflow can affect results. Use ``vec3d`` for
    greater precision. The boolean result is not differentiable.

    Args:
        v0: First vertex of the triangle.
        v1: Second vertex of the triangle.
        v2: Third vertex of the triangle.
        lower: Lower corner of the box.
        upper: Upper corner of the box.

    Returns:
        ``True`` if the triangle intersects the box, or ``False`` otherwise.
    """
    return _intersect_tri_aabb(v0, v1, v2, lower, upper)


@wp.func(inline=True)
def intersect_tri_aabb(v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d, lower: wp.vec3d, upper: wp.vec3d) -> bool:
    """Double-precision overload of :func:`warp.geometry.intersect_tri_aabb`."""
    return _intersect_tri_aabb(v0, v1, v2, lower, upper)


intersect_tri_aabb.__doc__ = intersect_tri_aabb.doc
_register_predicate(intersect_tri_aabb)


@wp.func(inline=True)
def _triangle_vertex(v0: Any, v1: Any, v2: Any, index: int):
    if index == 0:
        return v0
    if index == 1:
        return v1
    return v2


@wp.func(inline=True)
def _contains_vertex_id(ids: wp.vec3i, vertex_id: int):
    return ids[0] == vertex_id or ids[1] == vertex_id or ids[2] == vertex_id


@wp.func(inline=True)
def _self_intersect_tri_tri(
    a0: Any,
    a1: Any,
    a2: Any,
    b0: Any,
    b1: Any,
    b2: Any,
    vertex_ids_a: wp.vec3i,
    vertex_ids_b: wp.vec3i,
    tol: Any,
):
    m0 = _contains_vertex_id(vertex_ids_b, vertex_ids_a[0])
    m1 = _contains_vertex_id(vertex_ids_b, vertex_ids_a[1])
    m2 = _contains_vertex_id(vertex_ids_b, vertex_ids_a[2])
    shared = int(m0) + int(m1) + int(m2)
    if shared == 0:
        # Preserve ordinary intersection semantics, including tolerance and
        # arithmetic ordering, for topologically unrelated triangles.
        return _intersect_tri_tri(a0, a1, a2, b0, b1, b2, tol)
    if shared == 3:
        return True

    # Select the unique shared (one ID) or unshared (two IDs) corner of A.
    ia = int(2)
    if m0 == (shared == 1):
        ia = 0
    elif m1 == (shared == 1):
        ia = 1
    ja = (ia + 1) % 3
    ka = (ia + 2) % 3
    # Stable ordering under winding changes and vertex permutations.
    if _component(vertex_ids_a, ja) > _component(vertex_ids_a, ka):
        ja, ka = ka, ja
    a = _triangle_vertex(a0, a1, a2, ia)
    u = _triangle_vertex(a0, a1, a2, ja)
    v = _triangle_vertex(a0, a1, a2, ka)

    if shared == 1:
        shared_id = _component(vertex_ids_a, ia)
        ib = int(2)
        if vertex_ids_b[0] == shared_id:
            ib = 0
        elif vertex_ids_b[1] == shared_id:
            ib = 1
        jb = (ib + 1) % 3
        kb = (ib + 2) % 3
        if _component(vertex_ids_b, jb) > _component(vertex_ids_b, kb):
            jb, kb = kb, jb
        # Use the shared position as the origin. Its plane value is then zero
        # by construction, rather than a subtraction of large plane offsets.
        p = u - a
        q = v - a
        r = _triangle_vertex(b0, b1, b2, jb) - a
        s = _triangle_vertex(b0, b1, b2, kb) - a
        origin = a - a
        # By convexity, any intersection beyond the shared vertex reaches an
        # opposite edge of at least one triangle. Neither opposite edge contains
        # the shared vertex for nondegenerate inputs. No length cutoff is needed.
        return _intersect_seg_tri(p, q, origin, r, s, tol) or _intersect_seg_tri(r, s, origin, p, q, tol)

    # Two shared IDs: u-v is the common edge, a and b the opposite vertices.
    b = b2
    if not _contains_vertex_id(vertex_ids_a, vertex_ids_b[0]):
        b = b0
    elif not _contains_vertex_id(vertex_ids_a, vertex_ids_b[1]):
        b = b1
    e = v - u
    p = a - u
    q = b - u
    na = wp.cross(e, p)
    nb = wp.cross(e, q)
    zero = type(a0[0])(0.0)
    da = _plane_distance(na, q, zero, tol)
    db = _plane_distance(nb, p, zero, tol)
    # As in ordinary tri-tri, either plane classifying all opposite vertices
    # as coplanar selects the coplanar path. Shared endpoints are exact zeros.
    if da != zero and db != zero:
        return False
    # Use a common dominant projection, chosen symmetrically from both normals.
    # Sign comparisons avoid multiplying tiny projected areas together.
    n = wp.max(wp.abs(na), wp.abs(nb))
    index = int(0)
    if n[1] > n[0]:
        index = 1
    if n[2] > _component(n, index):
        index = 2
    sa = _component(na, index)
    sb = _component(nb, index)
    return (sa > zero and sb > zero) or (sa < zero and sb < zero)


@wp.func(inline=True)
def self_intersect_tri_tri(
    a0: wp.vec3,
    a1: wp.vec3,
    a2: wp.vec3,
    b0: wp.vec3,
    b1: wp.vec3,
    b2: wp.vec3,
    vertex_ids_a: wp.vec3i,
    vertex_ids_b: wp.vec3i,
    tol: wp.float64 = _DEFAULT_INTERSECTION_TOL,
) -> bool:
    """Test whether two triangles intersect beyond their topologically shared boundary.

    Adjacency is established only by equal vertex IDs, not by equal coordinates.
    With no shared IDs, this delegates to :func:`warp.geometry.intersect_tri_tri`
    with the same tolerance. One shared vertex permits contact only at that
    vertex; two shared vertices permit contact along their common edge. Three
    shared IDs report a duplicate face as an intersection, regardless of winding.
    Callers must exclude a face paired with itself before invoking this function.
    Contact elsewhere, including contact at geometrically coincident vertices
    with different IDs, counts as intersection. No intersection-length threshold
    is applied.

    Inputs must be finite, nondegenerate triangles in the same coordinate space,
    all using the same type. Overloads accept ``vec3`` or ``vec3d`` coordinates,
    each with a float32 or float64 ``tol``. Each face must have three
    distinct IDs from a common vertex-ID namespace, and coordinates associated
    with a shared ID must agree exactly. These preconditions are not checked.

    ``tol`` has the same default and unnormalized plane-value units as
    :func:`warp.geometry.intersect_tri_tri`. For a shared edge, the triangles
    are treated as coplanar if either opposite vertex's plane value relative to
    the other triangle is within ``tol``. Positive tolerance can consequently classify a slight fold as coplanar
    overlap. Zero disables snapping, but floating-point roundoff can still affect
    results; this is not an exact predicate. Use ``vec3d`` for greater precision.

    This convention applies to static intersection only, not positive-clearance
    or continuous collision queries. The boolean result is not differentiable.

    Args:
        a0: First vertex of triangle A.
        a1: Second vertex of triangle A.
        a2: Third vertex of triangle A.
        b0: First vertex of triangle B.
        b1: Second vertex of triangle B.
        b2: Third vertex of triangle B.
        vertex_ids_a: Vertex IDs of A, in the order ``a0``, ``a1``, ``a2``.
        vertex_ids_b: Vertex IDs of B, in the order ``b0``, ``b1``, ``b2``.
        tol: Nonnegative finite plane-equation tolerance, in float32 or float64.
            This does not dilate edges or suppress short intersections.

    Returns:
        ``True`` for intersection beyond the shared vertex or edge, or for a
        duplicate face; ``False`` for separation or ordinary adjacency.
    """
    return _self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, vertex_ids_a, vertex_ids_b, tol)


@wp.func(inline=True)
def self_intersect_tri_tri(
    a0: wp.vec3d,
    a1: wp.vec3d,
    a2: wp.vec3d,
    b0: wp.vec3d,
    b1: wp.vec3d,
    b2: wp.vec3d,
    vertex_ids_a: wp.vec3i,
    vertex_ids_b: wp.vec3i,
    tol: wp.float64 = _DEFAULT_INTERSECTION_TOL,
) -> bool:
    """Double-precision overload of :func:`warp.geometry.self_intersect_tri_tri`."""
    return _self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, vertex_ids_a, vertex_ids_b, tol)


@wp.func(inline=True)
def self_intersect_tri_tri(
    a0: wp.vec3,
    a1: wp.vec3,
    a2: wp.vec3,
    b0: wp.vec3,
    b1: wp.vec3,
    b2: wp.vec3,
    vertex_ids_a: wp.vec3i,
    vertex_ids_b: wp.vec3i,
    tol: wp.float32,
) -> bool:
    """Float32 tolerance overload; see :func:`warp.geometry.self_intersect_tri_tri`."""
    return _self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, vertex_ids_a, vertex_ids_b, tol)


@wp.func(inline=True)
def self_intersect_tri_tri(
    a0: wp.vec3d,
    a1: wp.vec3d,
    a2: wp.vec3d,
    b0: wp.vec3d,
    b1: wp.vec3d,
    b2: wp.vec3d,
    vertex_ids_a: wp.vec3i,
    vertex_ids_b: wp.vec3i,
    tol: wp.float32,
) -> bool:
    """Float32 tolerance overload; see :func:`warp.geometry.self_intersect_tri_tri`."""
    return _self_intersect_tri_tri(a0, a1, a2, b0, b1, b2, vertex_ids_a, vertex_ids_b, tol)


self_intersect_tri_tri.__doc__ = self_intersect_tri_tri.doc
_register_predicate(self_intersect_tri_tri)
