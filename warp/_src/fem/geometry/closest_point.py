# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp
from warp._src.fem.types import make_coords

_wp_module_name_ = "warp.fem.geometry.closest_point"


@wp.func
def project_on_seg_at_origin(q: Any, seg: Any, len_sq: Any):
    z = type(len_sq)(0.0)
    o = type(len_sq)(1.0)
    s = wp.clamp(wp.dot(q, seg) / len_sq, z, o)
    return wp.length_sq(q - s * seg), s


@wp.func
def _coplanarity_tolerance(_det: wp.float32):
    return 1.0e-6


@wp.func
def _coplanarity_tolerance(_det: wp.float64):
    return wp.float64(1.0e-12)


@wp.func
def project_on_tri_at_origin(q: Any, e1: Any, e2: Any):
    e1e1 = wp.dot(e1, e1)
    e1e2 = wp.dot(e1, e2)
    e2e2 = wp.dot(e2, e2)

    det = e1e1 * e2e2 - e1e2 * e1e2
    z = type(det)(0.0)
    o = type(det)(1.0)

    if det > e1e1 * e2e2 * _coplanarity_tolerance(det):
        e1p = wp.dot(e1, q)
        e2p = wp.dot(e2, q)

        s = (e2e2 * e1p - e1e2 * e2p) / det
        t = (e1e1 * e2p - e1e2 * e1p) / det

        if s >= z and t >= z and s + t <= o:
            # point inside triangle (distance can be non-zero in 3D case)
            return wp.length_sq(q - s * e1 - t * e2), make_coords(o - s - t, s, t)

    d1, s1 = project_on_seg_at_origin(q, e1, e1e1)
    d2, s2 = project_on_seg_at_origin(q, e2, e2e2)
    d12, s12 = project_on_seg_at_origin(q - e1, e2 - e1, wp.length_sq(e2 - e1))

    if d1 <= d2:
        if d1 <= d12:
            return d1, make_coords(o - s1, s1, z)
    elif d2 <= d12:
        return d2, make_coords(o - s2, z, s2)

    return d12, make_coords(z, o - s12, s12)


@wp.func
def project_on_tet_at_origin(q: Any, e1: Any, e2: Any, e3: Any):
    """Closest-point projection onto a tetrahedron with one vertex at the origin.

    Uses ``Any``-typed parameters so the same function handles both fp32 and fp64 inputs.
    """
    mat = wp.inverse(wp.matrix_from_cols(e1, e2, e3))
    coords = mat * q

    z = coords.dtype(0.0)  # zero of the right scalar type
    o = coords.dtype(1.0)  # one of the right scalar type

    if wp.min(coords) >= z and coords[0] + coords[1] + coords[2] <= o:
        return z, make_coords(coords[0], coords[1], coords[2])

    # Not inside tet, compare closest point on each tri

    d12, s12 = project_on_tri_at_origin(q, e1, e2)
    d23, s23 = project_on_tri_at_origin(q, e2, e3)
    d31, s31 = project_on_tri_at_origin(q, e3, e1)
    d123, s123 = project_on_tri_at_origin(q - e1, e2 - e1, e3 - e1)

    dmin = wp.min(wp.min(d12, d23), wp.min(d31, d123))

    if dmin == d12:
        return dmin, make_coords(s12[1], s12[2], z)
    elif dmin == d23:
        return dmin, make_coords(z, s23[1], s23[2])
    elif dmin == d31:
        return dmin, make_coords(s31[2], z, s31[1])
    else:
        return dmin, s123


@wp.func
def project_on_box_at_origin(coords: Any, sizes: Any):
    proj_coords = wp.min(wp.max(coords, type(coords)(coords.dtype(0.0))), sizes)
    return wp.length_sq(coords - proj_coords), wp.cw_div(proj_coords, sizes)


@wp.func
def project_on_box_at_origin_2d(coords: Any, sizes: Any):
    proj_coords = wp.min(wp.max(coords, type(coords)(coords.dtype(0.0))), sizes)
    norm_coords = wp.cw_div(proj_coords, sizes)
    return wp.length_sq(coords - proj_coords), make_coords(norm_coords[0], norm_coords[1])
