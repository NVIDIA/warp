# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collision handling functions and kernels.
"""

from typing import Optional

import numpy as np

import warp as wp
from warp.sim.model import Model, State

from .model import PARTICLE_FLAG_ACTIVE, ModelShapeGeometry

# types of triangle's closest point to a point
TRI_CONTACT_FEATURE_VERTEX_A = wp.constant(0)
TRI_CONTACT_FEATURE_VERTEX_B = wp.constant(1)
TRI_CONTACT_FEATURE_VERTEX_C = wp.constant(2)
TRI_CONTACT_FEATURE_EDGE_AB = wp.constant(3)
TRI_CONTACT_FEATURE_EDGE_AC = wp.constant(4)
TRI_CONTACT_FEATURE_EDGE_BC = wp.constant(5)
TRI_CONTACT_FEATURE_FACE_INTERIOR = wp.constant(6)

# constants used to access TriMeshCollisionDetector.resize_flags
VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(0)
TRI_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(1)
EDGE_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(2)
TRI_TRI_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(3)


@wp.func
def build_orthonormal_basis(n: wp.vec3):
    """
    Builds an orthonormal basis given a normal vector `n`. Return the two axes that are perpendicular to `n`.

    :param n: A 3D vector (list or array-like) representing the normal vector
    """
    b1 = wp.vec3()
    b2 = wp.vec3()
    if n[2] < 0.0:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]

        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]

        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]

    return b1, b2


@wp.func
def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return wp.vec3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)

    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return wp.vec3(0.0, 1.0 - w, w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    """
    feature_type type:
        TRI_CONTACT_FEATURE_VERTEX_A
        TRI_CONTACT_FEATURE_VERTEX_B
        TRI_CONTACT_FEATURE_VERTEX_C
        TRI_CONTACT_FEATURE_EDGE_AB      : at edge A-B
        TRI_CONTACT_FEATURE_EDGE_AC      : at edge A-C
        TRI_CONTACT_FEATURE_EDGE_BC      : at edge B-C
        TRI_CONTACT_FEATURE_FACE_INTERIOR
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_A
        bary = wp.vec3(1.0, 0.0, 0.0)
        return a, bary, feature_type

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_B
        bary = wp.vec3(0.0, 1.0, 0.0)
        return b, bary, feature_type

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_C
        bary = wp.vec3(0.0, 0.0, 1.0)
        return c, bary, feature_type

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AB
        bary = wp.vec3(1.0 - v, v, 0.0)
        return a + v * ab, bary, feature_type

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AC
        bary = wp.vec3(1.0 - v, 0.0, v)
        return a + v * ac, bary, feature_type

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        feature_type = TRI_CONTACT_FEATURE_EDGE_BC
        bary = wp.vec3(0.0, 1.0 - v, v)
        return b + v * (c - b), bary, feature_type

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    feature_type = TRI_CONTACT_FEATURE_FACE_INTERIOR
    bary = wp.vec3(1.0 - v - w, v, w)
    return a + v * ab + w * ac, bary, feature_type


@wp.func
def sphere_sdf(center: wp.vec3, radius: float, p: wp.vec3):
    return wp.length(p - center) - radius


@wp.func
def sphere_sdf_grad(center: wp.vec3, radius: float, p: wp.vec3):
    return wp.normalize(p - center)


@wp.func
def box_sdf(upper: wp.vec3, p: wp.vec3):
    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    qx = abs(p[0]) - upper[0]
    qy = abs(p[1]) - upper[1]
    qz = abs(p[2]) - upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))

    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def box_sdf_grad(upper: wp.vec3, p: wp.vec3):
    qx = abs(p[0]) - upper[0]
    qy = abs(p[1]) - upper[1]
    qz = abs(p[2]) - upper[2]

    # exterior case
    if qx > 0.0 or qy > 0.0 or qz > 0.0:
        x = wp.clamp(p[0], -upper[0], upper[0])
        y = wp.clamp(p[1], -upper[1], upper[1])
        z = wp.clamp(p[2], -upper[2], upper[2])

        return wp.normalize(p - wp.vec3(x, y, z))

    sx = wp.sign(p[0])
    sy = wp.sign(p[1])
    sz = wp.sign(p[2])

    # x projection
    if (qx > qy and qx > qz) or (qy == 0.0 and qz == 0.0):
        return wp.vec3(sx, 0.0, 0.0)

    # y projection
    if (qy > qx and qy > qz) or (qx == 0.0 and qz == 0.0):
        return wp.vec3(0.0, sy, 0.0)

    # z projection
    return wp.vec3(0.0, 0.0, sz)


@wp.func
def capsule_sdf(radius: float, half_height: float, p: wp.vec3):
    if p[1] > half_height:
        return wp.length(wp.vec3(p[0], p[1] - half_height, p[2])) - radius

    if p[1] < -half_height:
        return wp.length(wp.vec3(p[0], p[1] + half_height, p[2])) - radius

    return wp.length(wp.vec3(p[0], 0.0, p[2])) - radius


@wp.func
def capsule_sdf_grad(radius: float, half_height: float, p: wp.vec3):
    if p[1] > half_height:
        return wp.normalize(wp.vec3(p[0], p[1] - half_height, p[2]))

    if p[1] < -half_height:
        return wp.normalize(wp.vec3(p[0], p[1] + half_height, p[2]))

    return wp.normalize(wp.vec3(p[0], 0.0, p[2]))


@wp.func
def cylinder_sdf(radius: float, half_height: float, p: wp.vec3):
    dx = wp.length(wp.vec3(p[0], 0.0, p[2])) - radius
    dy = wp.abs(p[1]) - half_height
    return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))


@wp.func
def cylinder_sdf_grad(radius: float, half_height: float, p: wp.vec3):
    dx = wp.length(wp.vec3(p[0], 0.0, p[2])) - radius
    dy = wp.abs(p[1]) - half_height
    if dx > dy:
        return wp.normalize(wp.vec3(p[0], 0.0, p[2]))
    return wp.vec3(0.0, wp.sign(p[1]), 0.0)


@wp.func
def cone_sdf(radius: float, half_height: float, p: wp.vec3):
    dx = wp.length(wp.vec3(p[0], 0.0, p[2])) - radius * (p[1] + half_height) / (2.0 * half_height)
    dy = wp.abs(p[1]) - half_height
    return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))


@wp.func
def cone_sdf_grad(radius: float, half_height: float, p: wp.vec3):
    dx = wp.length(wp.vec3(p[0], 0.0, p[2])) - radius * (p[1] + half_height) / (2.0 * half_height)
    dy = wp.abs(p[1]) - half_height
    if dy < 0.0 or dx == 0.0:
        return wp.vec3(0.0, wp.sign(p[1]), 0.0)
    return wp.normalize(wp.vec3(p[0], 0.0, p[2])) + wp.vec3(0.0, radius / (2.0 * half_height), 0.0)


@wp.func
def plane_sdf(width: float, length: float, p: wp.vec3):
    # SDF for a quad in the xz plane
    if width > 0.0 and length > 0.0:
        d = wp.max(wp.abs(p[0]) - width, wp.abs(p[2]) - length)
        return wp.max(d, wp.abs(p[1]))
    return p[1]


@wp.func
def closest_point_plane(width: float, length: float, point: wp.vec3):
    # projects the point onto the quad in the xz plane (if width and length > 0.0, otherwise the plane is infinite)
    if width > 0.0:
        x = wp.clamp(point[0], -width, width)
    else:
        x = point[0]
    if length > 0.0:
        z = wp.clamp(point[2], -length, length)
    else:
        z = point[2]
    return wp.vec3(x, 0.0, z)


@wp.func
def closest_point_line_segment(a: wp.vec3, b: wp.vec3, point: wp.vec3):
    ab = b - a
    ap = point - a
    t = wp.dot(ap, ab) / wp.dot(ab, ab)
    t = wp.clamp(t, 0.0, 1.0)
    return a + t * ab


@wp.func
def closest_point_box(upper: wp.vec3, point: wp.vec3):
    # closest point to box surface
    x = wp.clamp(point[0], -upper[0], upper[0])
    y = wp.clamp(point[1], -upper[1], upper[1])
    z = wp.clamp(point[2], -upper[2], upper[2])
    if wp.abs(point[0]) <= upper[0] and wp.abs(point[1]) <= upper[1] and wp.abs(point[2]) <= upper[2]:
        # the point is inside, find closest face
        sx = wp.abs(wp.abs(point[0]) - upper[0])
        sy = wp.abs(wp.abs(point[1]) - upper[1])
        sz = wp.abs(wp.abs(point[2]) - upper[2])
        # return closest point on closest side, handle corner cases
        if (sx < sy and sx < sz) or (sy == 0.0 and sz == 0.0):
            x = wp.sign(point[0]) * upper[0]
        elif (sy < sx and sy < sz) or (sx == 0.0 and sz == 0.0):
            y = wp.sign(point[1]) * upper[1]
        else:
            z = wp.sign(point[2]) * upper[2]
    return wp.vec3(x, y, z)


@wp.func
def get_box_vertex(point_id: int, upper: wp.vec3):
    # box vertex numbering:
    #    6---7
    #    |\  |\       y
    #    | 2-+-3      |
    #    4-+-5 |   z \|
    #     \|  \|      o---x
    #      0---1
    # get the vertex of the box given its ID (0-7)
    sign_x = float(point_id % 2) * 2.0 - 1.0
    sign_y = float((point_id // 2) % 2) * 2.0 - 1.0
    sign_z = float((point_id // 4) % 2) * 2.0 - 1.0
    return wp.vec3(sign_x * upper[0], sign_y * upper[1], sign_z * upper[2])


@wp.func
def get_box_edge(edge_id: int, upper: wp.vec3):
    # get the edge of the box given its ID (0-11)
    if edge_id < 4:
        # edges along x: 0-1, 2-3, 4-5, 6-7
        i = edge_id * 2
        j = i + 1
        return wp.spatial_vector(get_box_vertex(i, upper), get_box_vertex(j, upper))
    elif edge_id < 8:
        # edges along y: 0-2, 1-3, 4-6, 5-7
        edge_id -= 4
        i = edge_id % 2 + edge_id // 2 * 4
        j = i + 2
        return wp.spatial_vector(get_box_vertex(i, upper), get_box_vertex(j, upper))
    # edges along z: 0-4, 1-5, 2-6, 3-7
    edge_id -= 8
    i = edge_id
    j = i + 4
    return wp.spatial_vector(get_box_vertex(i, upper), get_box_vertex(j, upper))


@wp.func
def get_plane_edge(edge_id: int, plane_width: float, plane_length: float):
    # get the edge of the plane given its ID (0-3)
    p0x = (2.0 * float(edge_id % 2) - 1.0) * plane_width
    p0z = (2.0 * float(edge_id // 2) - 1.0) * plane_length
    if edge_id == 0 or edge_id == 3:
        p1x = p0x
        p1z = -p0z
    else:
        p1x = -p0x
        p1z = p0z
    return wp.spatial_vector(wp.vec3(p0x, 0.0, p0z), wp.vec3(p1x, 0.0, p1z))


@wp.func
def closest_edge_coordinate_box(upper: wp.vec3, edge_a: wp.vec3, edge_b: wp.vec3, max_iter: int):
    # find point on edge closest to box, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = box_sdf(upper, query)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = box_sdf(upper, query)

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = box_sdf(upper, query)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = box_sdf(upper, query)

    if yc < yd:
        return 0.5 * (a + d)
    return 0.5 * (c + b)


@wp.func
def closest_edge_coordinate_plane(
    plane_width: float,
    plane_length: float,
    edge_a: wp.vec3,
    edge_b: wp.vec3,
    max_iter: int,
):
    # find point on edge closest to plane, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = plane_sdf(plane_width, plane_length, query)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = plane_sdf(plane_width, plane_length, query)

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = plane_sdf(plane_width, plane_length, query)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = plane_sdf(plane_width, plane_length, query)

    if yc < yd:
        return 0.5 * (a + d)
    return 0.5 * (c + b)


@wp.func
def closest_edge_coordinate_capsule(radius: float, half_height: float, edge_a: wp.vec3, edge_b: wp.vec3, max_iter: int):
    # find point on edge closest to capsule, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = capsule_sdf(radius, half_height, query)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = capsule_sdf(radius, half_height, query)

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = capsule_sdf(radius, half_height, query)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = capsule_sdf(radius, half_height, query)

    if yc < yd:
        return 0.5 * (a + d)

    return 0.5 * (c + b)


@wp.func
def mesh_sdf(mesh: wp.uint64, point: wp.vec3, max_dist: float):
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(mesh, point, max_dist, sign, face_index, face_u, face_v)

    if res:
        closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        return wp.length(point - closest) * sign
    return max_dist


@wp.func
def closest_point_mesh(mesh: wp.uint64, point: wp.vec3, max_dist: float):
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    res = wp.mesh_query_point_sign_normal(mesh, point, max_dist, sign, face_index, face_u, face_v)

    if res:
        return wp.mesh_eval_position(mesh, face_index, face_u, face_v)
    # return arbitrary point from mesh
    return wp.mesh_eval_position(mesh, 0, 0.0, 0.0)


@wp.func
def closest_edge_coordinate_mesh(mesh: wp.uint64, edge_a: wp.vec3, edge_b: wp.vec3, max_iter: int, max_dist: float):
    # find point on edge closest to mesh, return its barycentric edge coordinate
    # Golden-section search
    a = float(0.0)
    b = float(1.0)
    h = b - a
    invphi = 0.61803398875  # 1 / phi
    invphi2 = 0.38196601125  # 1 / phi^2
    c = a + invphi2 * h
    d = a + invphi * h
    query = (1.0 - c) * edge_a + c * edge_b
    yc = mesh_sdf(mesh, query, max_dist)
    query = (1.0 - d) * edge_a + d * edge_b
    yd = mesh_sdf(mesh, query, max_dist)

    for _k in range(max_iter):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            query = (1.0 - c) * edge_a + c * edge_b
            yc = mesh_sdf(mesh, query, max_dist)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            query = (1.0 - d) * edge_a + d * edge_b
            yd = mesh_sdf(mesh, query, max_dist)

    if yc < yd:
        return 0.5 * (a + d)
    return 0.5 * (c + b)


@wp.func
def volume_grad(volume: wp.uint64, p: wp.vec3):
    eps = 0.05  # TODO make this a parameter
    q = wp.volume_world_to_index(volume, p)

    # compute gradient of the SDF using finite differences
    dx = wp.volume_sample_f(volume, q + wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR
    )
    dy = wp.volume_sample_f(volume, q + wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR
    )
    dz = wp.volume_sample_f(volume, q + wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR
    )

    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def counter_increment(counter: wp.array(dtype=int), counter_index: int, tids: wp.array(dtype=int), tid: int):
    # increment counter, remember which thread received which counter value
    count = wp.atomic_add(counter, counter_index, 1)
    tids[tid] = count
    return count


@wp.func_replay(counter_increment)
def replay_counter_increment(counter: wp.array(dtype=int), counter_index: int, tids: wp.array(dtype=int), tid: int):
    return tids[tid]


@wp.func
def limited_counter_increment(
    counter: wp.array(dtype=int), counter_index: int, tids: wp.array(dtype=int), tid: int, index_limit: int
):
    # increment counter but only if it is smaller than index_limit, remember which thread received which counter value
    count = wp.atomic_add(counter, counter_index, 1)
    if count < index_limit or index_limit < 0:
        tids[tid] = count
        return count
    tids[tid] = -1
    return -1


@wp.func_replay(limited_counter_increment)
def replay_limited_counter_increment(
    counter: wp.array(dtype=int), counter_index: int, tids: wp.array(dtype=int), tid: int, index_limit: int
):
    return tids[tid]


@wp.kernel
def create_soft_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    body_X_wb: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geo: ModelShapeGeometry,
    margin: float,
    soft_contact_max: int,
    shape_count: int,
    # outputs
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    soft_contact_tids: wp.array(dtype=int),
):
    tid = wp.tid()
    particle_index, shape_index = tid // shape_count, tid % shape_count
    if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    rigid_index = shape_body[shape_index]

    px = particle_x[particle_index]
    radius = particle_radius[particle_index]

    X_wb = wp.transform_identity()
    if rigid_index >= 0:
        X_wb = body_X_wb[rigid_index]

    X_bs = shape_X_bs[shape_index]

    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)

    # transform particle position to shape local space
    x_local = wp.transform_point(X_sw, px)

    # geo description
    geo_type = geo.type[shape_index]
    geo_scale = geo.scale[shape_index]

    # evaluate shape sdf
    d = 1.0e6
    n = wp.vec3()
    v = wp.vec3()

    if geo_type == wp.sim.GEO_SPHERE:
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
        n = sphere_sdf_grad(wp.vec3(), geo_scale[0], x_local)

    if geo_type == wp.sim.GEO_BOX:
        d = box_sdf(geo_scale, x_local)
        n = box_sdf_grad(geo_scale, x_local)

    if geo_type == wp.sim.GEO_CAPSULE:
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
        n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    if geo_type == wp.sim.GEO_CYLINDER:
        d = cylinder_sdf(geo_scale[0], geo_scale[1], x_local)
        n = cylinder_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    if geo_type == wp.sim.GEO_CONE:
        d = cone_sdf(geo_scale[0], geo_scale[1], x_local)
        n = cone_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    if geo_type == wp.sim.GEO_MESH:
        mesh = geo.source[shape_index]

        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)

        min_scale = wp.min(geo_scale)
        if wp.mesh_query_point_sign_normal(
            mesh, wp.cw_div(x_local, geo_scale), margin + radius / min_scale, sign, face_index, face_u, face_v
        ):
            shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
            shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)

            shape_p = wp.cw_mul(shape_p, geo_scale)
            shape_v = wp.cw_mul(shape_v, geo_scale)

            delta = x_local - shape_p

            d = wp.length(delta) * sign
            n = wp.normalize(delta) * sign
            v = shape_v

    if geo_type == wp.sim.GEO_SDF:
        volume = geo.source[shape_index]
        xpred_local = wp.volume_world_to_index(volume, wp.cw_div(x_local, geo_scale))
        nn = wp.vec3(0.0, 0.0, 0.0)
        d = wp.volume_sample_grad_f(volume, xpred_local, wp.Volume.LINEAR, nn)
        n = wp.normalize(nn)

    if geo_type == wp.sim.GEO_PLANE:
        d = plane_sdf(geo_scale[0], geo_scale[1], x_local)
        n = wp.vec3(0.0, 1.0, 0.0)

    if d < margin + radius:
        index = counter_increment(soft_contact_count, 0, soft_contact_tids, tid)

        if index < soft_contact_max:
            # compute contact point in body local space
            body_pos = wp.transform_point(X_bs, x_local - n * d)
            body_vel = wp.transform_vector(X_bs, v)

            world_normal = wp.transform_vector(X_ws, n)

            soft_contact_shape[index] = shape_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = body_vel
            soft_contact_particle[index] = particle_index
            soft_contact_normal[index] = world_normal


@wp.kernel(enable_backward=False)
def count_contact_points(
    contact_pairs: wp.array(dtype=int, ndim=2),
    geo: ModelShapeGeometry,
    mesh_contact_max: int,
    # outputs
    contact_count: wp.array(dtype=int),
):
    tid = wp.tid()
    shape_a = contact_pairs[tid, 0]
    shape_b = contact_pairs[tid, 1]

    if shape_b == -1:
        actual_shape_a = shape_a
        actual_type_a = geo.type[shape_a]
        # ground plane
        actual_type_b = wp.sim.GEO_PLANE
        actual_shape_b = -1
    else:
        type_a = geo.type[shape_a]
        type_b = geo.type[shape_b]
        # unique ordering of shape pairs
        if type_a < type_b:
            actual_shape_a = shape_a
            actual_shape_b = shape_b
            actual_type_a = type_a
            actual_type_b = type_b
        else:
            actual_shape_a = shape_b
            actual_shape_b = shape_a
            actual_type_a = type_b
            actual_type_b = type_a

    # determine how many contact points need to be evaluated
    num_contacts = 0
    num_actual_contacts = 0
    if actual_type_a == wp.sim.GEO_SPHERE:
        num_contacts = 1
        num_actual_contacts = 1
    elif actual_type_a == wp.sim.GEO_CAPSULE:
        if actual_type_b == wp.sim.GEO_PLANE:
            if geo.scale[actual_shape_b][0] == 0.0 and geo.scale[actual_shape_b][1] == 0.0:
                num_contacts = 2  # vertex-based collision for infinite plane
                num_actual_contacts = 2
            else:
                num_contacts = 2 + 4  # vertex-based collision + plane edges
                num_actual_contacts = 2 + 4
        elif actual_type_b == wp.sim.GEO_MESH:
            num_contacts_a = 2
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            num_contacts = num_contacts_a + num_contacts_b
            if mesh_contact_max > 0:
                num_contacts_b = wp.min(mesh_contact_max, num_contacts_b)
            num_actual_contacts = num_contacts_a + num_contacts_b
        else:
            num_contacts = 2
            num_actual_contacts = 2
    elif actual_type_a == wp.sim.GEO_BOX:
        if actual_type_b == wp.sim.GEO_BOX:
            num_contacts = 24
            num_actual_contacts = 24
        elif actual_type_b == wp.sim.GEO_MESH:
            num_contacts_a = 8
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            num_contacts = num_contacts_a + num_contacts_b
            if mesh_contact_max > 0:
                num_contacts_b = wp.min(mesh_contact_max, num_contacts_b)
            num_actual_contacts = num_contacts_a + num_contacts_b
        elif actual_type_b == wp.sim.GEO_PLANE:
            if geo.scale[actual_shape_b][0] == 0.0 and geo.scale[actual_shape_b][1] == 0.0:
                num_contacts = 8  # vertex-based collision
                num_actual_contacts = 8
            else:
                num_contacts = 8 + 4  # vertex-based collision + plane edges
                num_actual_contacts = 8 + 4
        else:
            num_contacts = 8
            num_actual_contacts = 8
    elif actual_type_a == wp.sim.GEO_MESH:
        mesh_a = wp.mesh_get(geo.source[actual_shape_a])
        num_contacts_a = mesh_a.points.shape[0]
        if mesh_contact_max > 0:
            num_contacts_a = wp.min(mesh_contact_max, num_contacts_a)
        if actual_type_b == wp.sim.GEO_MESH:
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
            num_contacts = num_contacts_a + num_contacts_b
            if mesh_contact_max > 0:
                num_contacts_b = wp.min(mesh_contact_max, num_contacts_b)
        else:
            num_contacts_b = 0
        num_contacts = num_contacts_a + num_contacts_b
        num_actual_contacts = num_contacts_a + num_contacts_b
    elif actual_type_a == wp.sim.GEO_PLANE:
        return  # no plane-plane contacts
    else:
        wp.printf(
            "count_contact_points: unsupported geometry type combination %d and %d\n", actual_type_a, actual_type_b
        )

    wp.atomic_add(contact_count, 0, num_contacts)
    wp.atomic_add(contact_count, 1, num_actual_contacts)


@wp.kernel(enable_backward=False)
def broadphase_collision_pairs(
    contact_pairs: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    body_mass: wp.array(dtype=float),
    num_shapes: int,
    geo: ModelShapeGeometry,
    collision_radius: wp.array(dtype=float),
    rigid_contact_max: int,
    rigid_contact_margin: float,
    mesh_contact_max: int,
    iterate_mesh_vertices: bool,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
    contact_point_limit: wp.array(dtype=int),
):
    tid = wp.tid()
    shape_a = contact_pairs[tid, 0]
    shape_b = contact_pairs[tid, 1]

    mass_a = 0.0
    mass_b = 0.0
    rigid_a = shape_body[shape_a]
    if rigid_a == -1:
        X_ws_a = shape_X_bs[shape_a]
    else:
        X_ws_a = wp.transform_multiply(body_q[rigid_a], shape_X_bs[shape_a])
        mass_a = body_mass[rigid_a]
    rigid_b = shape_body[shape_b]
    if rigid_b == -1:
        X_ws_b = shape_X_bs[shape_b]
    else:
        X_ws_b = wp.transform_multiply(body_q[rigid_b], shape_X_bs[shape_b])
        mass_b = body_mass[rigid_b]
    if mass_a == 0.0 and mass_b == 0.0:
        # skip if both bodies are static
        return

    type_a = geo.type[shape_a]
    type_b = geo.type[shape_b]
    # unique ordering of shape pairs
    if type_a < type_b:
        actual_shape_a = shape_a
        actual_shape_b = shape_b
        actual_type_a = type_a
        actual_type_b = type_b
        actual_X_ws_a = X_ws_a
        actual_X_ws_b = X_ws_b
    else:
        actual_shape_a = shape_b
        actual_shape_b = shape_a
        actual_type_a = type_b
        actual_type_b = type_a
        actual_X_ws_a = X_ws_b
        actual_X_ws_b = X_ws_a

    p_a = wp.transform_get_translation(actual_X_ws_a)
    if actual_type_b == wp.sim.GEO_PLANE:
        if actual_type_a == wp.sim.GEO_PLANE:
            return
        query_b = wp.transform_point(wp.transform_inverse(actual_X_ws_b), p_a)
        scale = geo.scale[actual_shape_b]
        closest = closest_point_plane(scale[0], scale[1], query_b)
        d = wp.length(query_b - closest)
        r_a = collision_radius[actual_shape_a]
        if d > r_a + rigid_contact_margin:
            return
    else:
        p_b = wp.transform_get_translation(actual_X_ws_b)
        d = wp.length(p_a - p_b) * 0.5 - 0.1
        r_a = collision_radius[actual_shape_a]
        r_b = collision_radius[actual_shape_b]
        if d > r_a + r_b + rigid_contact_margin:
            return

    pair_index_ab = actual_shape_a * num_shapes + actual_shape_b
    pair_index_ba = actual_shape_b * num_shapes + actual_shape_a

    # determine how many contact points need to be evaluated
    num_contacts = 0
    if actual_type_a == wp.sim.GEO_SPHERE:
        num_contacts = 1
    elif actual_type_a == wp.sim.GEO_CAPSULE:
        if actual_type_b == wp.sim.GEO_PLANE:
            if geo.scale[actual_shape_b][0] == 0.0 and geo.scale[actual_shape_b][1] == 0.0:
                num_contacts = 2  # vertex-based collision for infinite plane
            else:
                num_contacts = 2 + 4  # vertex-based collision + plane edges
        elif actual_type_b == wp.sim.GEO_MESH:
            num_contacts_a = 2
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            if iterate_mesh_vertices:
                num_contacts_b = mesh_b.points.shape[0]
            else:
                num_contacts_b = 0
            num_contacts = num_contacts_a + num_contacts_b
            index = wp.atomic_add(contact_count, 0, num_contacts)
            if index + num_contacts - 1 >= rigid_contact_max:
                print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                return
            # allocate contact points from capsule A against mesh B
            for i in range(num_contacts_a):
                contact_shape0[index + i] = actual_shape_a
                contact_shape1[index + i] = actual_shape_b
                contact_point_id[index + i] = i
            # allocate contact points from mesh B against capsule A
            for i in range(num_contacts_b):
                contact_shape0[index + num_contacts_a + i] = actual_shape_b
                contact_shape1[index + num_contacts_a + i] = actual_shape_a
                contact_point_id[index + num_contacts_a + i] = i
            if mesh_contact_max > 0 and contact_point_limit and pair_index_ba < contact_point_limit.shape[0]:
                num_contacts_b = wp.min(mesh_contact_max, num_contacts_b)
                contact_point_limit[pair_index_ba] = num_contacts_b
            return
        else:
            num_contacts = 2
    elif actual_type_a == wp.sim.GEO_BOX:
        if actual_type_b == wp.sim.GEO_BOX:
            index = wp.atomic_add(contact_count, 0, 24)
            if index + 23 >= rigid_contact_max:
                print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                return
            # allocate contact points from box A against B
            for i in range(12):  # 12 edges
                contact_shape0[index + i] = shape_a
                contact_shape1[index + i] = shape_b
                contact_point_id[index + i] = i
            # allocate contact points from box B against A
            for i in range(12):
                contact_shape0[index + 12 + i] = shape_b
                contact_shape1[index + 12 + i] = shape_a
                contact_point_id[index + 12 + i] = i
            return
        elif actual_type_b == wp.sim.GEO_MESH:
            num_contacts_a = 8
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            if iterate_mesh_vertices:
                num_contacts_b = mesh_b.points.shape[0]
            else:
                num_contacts_b = 0
            num_contacts = num_contacts_a + num_contacts_b
            index = wp.atomic_add(contact_count, 0, num_contacts)
            if index + num_contacts - 1 >= rigid_contact_max:
                print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                return
            # allocate contact points from box A against mesh B
            for i in range(num_contacts_a):
                contact_shape0[index + i] = actual_shape_a
                contact_shape1[index + i] = actual_shape_b
                contact_point_id[index + i] = i
            # allocate contact points from mesh B against box A
            for i in range(num_contacts_b):
                contact_shape0[index + num_contacts_a + i] = actual_shape_b
                contact_shape1[index + num_contacts_a + i] = actual_shape_a
                contact_point_id[index + num_contacts_a + i] = i

            if mesh_contact_max > 0 and contact_point_limit and pair_index_ba < contact_point_limit.shape[0]:
                num_contacts_b = wp.min(mesh_contact_max, num_contacts_b)
                contact_point_limit[pair_index_ba] = num_contacts_b
            return
        elif actual_type_b == wp.sim.GEO_PLANE:
            if geo.scale[actual_shape_b][0] == 0.0 and geo.scale[actual_shape_b][1] == 0.0:
                num_contacts = 8  # vertex-based collision
            else:
                num_contacts = 8 + 4  # vertex-based collision + plane edges
        else:
            num_contacts = 8
    elif actual_type_a == wp.sim.GEO_MESH:
        mesh_a = wp.mesh_get(geo.source[actual_shape_a])
        num_contacts_a = mesh_a.points.shape[0]
        num_contacts_b = 0
        if actual_type_b == wp.sim.GEO_MESH:
            mesh_b = wp.mesh_get(geo.source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
        elif actual_type_b != wp.sim.GEO_PLANE:
            print("broadphase_collision_pairs: unsupported geometry type for mesh collision")
            return
        num_contacts = num_contacts_a + num_contacts_b
        if num_contacts > 0:
            index = wp.atomic_add(contact_count, 0, num_contacts)
            if index + num_contacts - 1 >= rigid_contact_max:
                print("Mesh contact: Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                return
            # allocate contact points from mesh A against B
            for i in range(num_contacts_a):
                contact_shape0[index + i] = actual_shape_a
                contact_shape1[index + i] = actual_shape_b
                contact_point_id[index + i] = i
            # allocate contact points from mesh B against A
            for i in range(num_contacts_b):
                contact_shape0[index + num_contacts_a + i] = actual_shape_b
                contact_shape1[index + num_contacts_a + i] = actual_shape_a
                contact_point_id[index + num_contacts_a + i] = i

            if mesh_contact_max > 0 and contact_point_limit:
                num_contacts_a = wp.min(mesh_contact_max, num_contacts_a)
                num_contacts_b = wp.min(mesh_contact_max, num_contacts_b)
                if pair_index_ab < contact_point_limit.shape[0]:
                    contact_point_limit[pair_index_ab] = num_contacts_a
                if pair_index_ba < contact_point_limit.shape[0]:
                    contact_point_limit[pair_index_ba] = num_contacts_b
        return
    elif actual_type_a == wp.sim.GEO_PLANE:
        return  # no plane-plane contacts
    else:
        print("broadphase_collision_pairs: unsupported geometry type")

    if num_contacts > 0:
        index = wp.atomic_add(contact_count, 0, num_contacts)
        if index + num_contacts - 1 >= rigid_contact_max:
            print("Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
            return
        # allocate contact points
        for i in range(num_contacts):
            cp_index = index + i
            contact_shape0[cp_index] = actual_shape_a
            contact_shape1[cp_index] = actual_shape_b
            contact_point_id[cp_index] = i
        if contact_point_limit:
            if pair_index_ab < contact_point_limit.shape[0]:
                contact_point_limit[pair_index_ab] = num_contacts
            if pair_index_ba < contact_point_limit.shape[0]:
                contact_point_limit[pair_index_ba] = 0


@wp.kernel
def handle_contact_pairs(
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geo: ModelShapeGeometry,
    rigid_contact_margin: float,
    contact_broad_shape0: wp.array(dtype=int),
    contact_broad_shape1: wp.array(dtype=int),
    num_shapes: int,
    contact_point_id: wp.array(dtype=int),
    contact_point_limit: wp.array(dtype=int),
    edge_sdf_iter: int,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_thickness: wp.array(dtype=float),
    contact_pairwise_counter: wp.array(dtype=int),
    contact_tids: wp.array(dtype=int),
):
    tid = wp.tid()
    shape_a = contact_broad_shape0[tid]
    shape_b = contact_broad_shape1[tid]
    if shape_a == shape_b:
        return

    if contact_point_limit:
        pair_index = shape_a * num_shapes + shape_b
        contact_limit = contact_point_limit[pair_index]
        if contact_pairwise_counter[pair_index] >= contact_limit:
            # reached limit of contact points per contact pair
            return

    point_id = contact_point_id[tid]

    rigid_a = shape_body[shape_a]
    X_wb_a = wp.transform_identity()
    if rigid_a >= 0:
        X_wb_a = body_q[rigid_a]
    X_bs_a = shape_X_bs[shape_a]
    X_ws_a = wp.transform_multiply(X_wb_a, X_bs_a)
    X_sw_a = wp.transform_inverse(X_ws_a)
    X_bw_a = wp.transform_inverse(X_wb_a)
    geo_type_a = geo.type[shape_a]
    geo_scale_a = geo.scale[shape_a]
    min_scale_a = min(geo_scale_a)
    thickness_a = geo.thickness[shape_a]
    # is_solid_a = geo.is_solid[shape_a]

    rigid_b = shape_body[shape_b]
    X_wb_b = wp.transform_identity()
    if rigid_b >= 0:
        X_wb_b = body_q[rigid_b]
    X_bs_b = shape_X_bs[shape_b]
    X_ws_b = wp.transform_multiply(X_wb_b, X_bs_b)
    X_sw_b = wp.transform_inverse(X_ws_b)
    X_bw_b = wp.transform_inverse(X_wb_b)
    geo_type_b = geo.type[shape_b]
    geo_scale_b = geo.scale[shape_b]
    min_scale_b = min(geo_scale_b)
    thickness_b = geo.thickness[shape_b]
    # is_solid_b = geo.is_solid[shape_b]

    distance = 1.0e6
    u = float(0.0)
    thickness = thickness_a + thickness_b

    if geo_type_a == wp.sim.GEO_SPHERE:
        p_a_world = wp.transform_get_translation(X_ws_a)
        if geo_type_b == wp.sim.GEO_SPHERE:
            p_b_world = wp.transform_get_translation(X_ws_b)
        elif geo_type_b == wp.sim.GEO_BOX:
            # contact point in frame of body B
            p_a_body = wp.transform_point(X_sw_b, p_a_world)
            p_b_body = closest_point_box(geo_scale_b, p_a_body)
            p_b_world = wp.transform_point(X_ws_b, p_b_body)
        elif geo_type_b == wp.sim.GEO_CAPSULE:
            half_height_b = geo_scale_b[1]
            # capsule B
            A_b = wp.transform_point(X_ws_b, wp.vec3(0.0, half_height_b, 0.0))
            B_b = wp.transform_point(X_ws_b, wp.vec3(0.0, -half_height_b, 0.0))
            p_b_world = closest_point_line_segment(A_b, B_b, p_a_world)
        elif geo_type_b == wp.sim.GEO_MESH:
            mesh_b = geo.source[shape_b]
            query_b_local = wp.transform_point(X_sw_b, p_a_world)
            face_index = int(0)
            face_u = float(0.0)
            face_v = float(0.0)
            sign = float(0.0)
            max_dist = (thickness + rigid_contact_margin) / min_scale_b
            res = wp.mesh_query_point_sign_normal(
                mesh_b, wp.cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index, face_u, face_v
            )
            if res:
                shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
                shape_p = wp.cw_mul(shape_p, geo_scale_b)
                p_b_world = wp.transform_point(X_ws_b, shape_p)
            else:
                return
        elif geo_type_b == wp.sim.GEO_PLANE:
            p_b_body = closest_point_plane(geo_scale_b[0], geo_scale_b[1], wp.transform_point(X_sw_b, p_a_world))
            p_b_world = wp.transform_point(X_ws_b, p_b_body)
        else:
            print("Unsupported geometry type in sphere collision handling")
            print(geo_type_b)
            return
        diff = p_a_world - p_b_world
        normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_BOX and geo_type_b == wp.sim.GEO_BOX:
        # edge-based box contact
        edge = get_box_edge(point_id, geo_scale_a)
        edge0_world = wp.transform_point(X_ws_a, wp.spatial_top(edge))
        edge1_world = wp.transform_point(X_ws_a, wp.spatial_bottom(edge))
        edge0_b = wp.transform_point(X_sw_b, edge0_world)
        edge1_b = wp.transform_point(X_sw_b, edge1_world)
        max_iter = edge_sdf_iter
        u = closest_edge_coordinate_box(geo_scale_b, edge0_b, edge1_b, max_iter)
        p_a_world = (1.0 - u) * edge0_world + u * edge1_world

        # find closest point + contact normal on box B
        query_b = wp.transform_point(X_sw_b, p_a_world)
        p_b_body = closest_point_box(geo_scale_b, query_b)
        p_b_world = wp.transform_point(X_ws_b, p_b_body)
        diff = p_a_world - p_b_world

        normal = wp.transform_vector(X_ws_b, box_sdf_grad(geo_scale_b, query_b))
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_BOX and geo_type_b == wp.sim.GEO_CAPSULE:
        half_height_b = geo_scale_b[1]
        # capsule B
        # depending on point id, we query an edge from 0 to 0.5 or 0.5 to 1
        e0 = wp.vec3(0.0, -half_height_b * float(point_id % 2), 0.0)
        e1 = wp.vec3(0.0, half_height_b * float((point_id + 1) % 2), 0.0)
        edge0_world = wp.transform_point(X_ws_b, e0)
        edge1_world = wp.transform_point(X_ws_b, e1)
        edge0_a = wp.transform_point(X_sw_a, edge0_world)
        edge1_a = wp.transform_point(X_sw_a, edge1_world)
        max_iter = edge_sdf_iter
        u = closest_edge_coordinate_box(geo_scale_a, edge0_a, edge1_a, max_iter)
        p_b_world = (1.0 - u) * edge0_world + u * edge1_world
        # find closest point + contact normal on box A
        query_a = wp.transform_point(X_sw_a, p_b_world)
        p_a_body = closest_point_box(geo_scale_a, query_a)
        p_a_world = wp.transform_point(X_ws_a, p_a_body)
        diff = p_a_world - p_b_world
        # the contact point inside the capsule should already be outside the box
        normal = -wp.transform_vector(X_ws_a, box_sdf_grad(geo_scale_a, query_a))
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_BOX and geo_type_b == wp.sim.GEO_PLANE:
        plane_width = geo_scale_b[0]
        plane_length = geo_scale_b[1]
        if point_id < 8:
            # vertex-based contact
            p_a_body = get_box_vertex(point_id, geo_scale_a)
            p_a_world = wp.transform_point(X_ws_a, p_a_body)
            query_b = wp.transform_point(X_sw_b, p_a_world)
            p_b_body = closest_point_plane(plane_width, plane_length, query_b)
            p_b_world = wp.transform_point(X_ws_b, p_b_body)
            diff = p_a_world - p_b_world
            normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
            if plane_width > 0.0 and plane_length > 0.0:
                if wp.abs(query_b[0]) > plane_width or wp.abs(query_b[2]) > plane_length:
                    # skip, we will evaluate the plane edge contact with the box later
                    return
                # check whether the COM is above the plane
                # sign = wp.sign(wp.dot(wp.transform_get_translation(X_ws_a) - p_b_world, normal))
                # if sign < 0.0:
                #     # the entire box is most likely below the plane
                #     return
            # the contact point is within plane boundaries
            distance = wp.dot(diff, normal)
        else:
            # contact between box A and edges of finite plane B
            edge = get_plane_edge(point_id - 8, plane_width, plane_length)
            edge0_world = wp.transform_point(X_ws_b, wp.spatial_top(edge))
            edge1_world = wp.transform_point(X_ws_b, wp.spatial_bottom(edge))
            edge0_a = wp.transform_point(X_sw_a, edge0_world)
            edge1_a = wp.transform_point(X_sw_a, edge1_world)
            max_iter = edge_sdf_iter
            u = closest_edge_coordinate_box(geo_scale_a, edge0_a, edge1_a, max_iter)
            p_b_world = (1.0 - u) * edge0_world + u * edge1_world

            # find closest point + contact normal on box A
            query_a = wp.transform_point(X_sw_a, p_b_world)
            p_a_body = closest_point_box(geo_scale_a, query_a)
            p_a_world = wp.transform_point(X_ws_a, p_a_body)
            query_b = wp.transform_point(X_sw_b, p_a_world)
            if wp.abs(query_b[0]) > plane_width or wp.abs(query_b[2]) > plane_length:
                # ensure that the closest point is actually inside the plane
                return
            diff = p_a_world - p_b_world
            com_a = wp.transform_get_translation(X_ws_a)
            query_b = wp.transform_point(X_sw_b, com_a)
            if wp.abs(query_b[0]) > plane_width or wp.abs(query_b[2]) > plane_length:
                # the COM is outside the plane
                normal = wp.normalize(com_a - p_b_world)
            else:
                normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
            distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_CAPSULE and geo_type_b == wp.sim.GEO_CAPSULE:
        # find closest edge coordinate to capsule SDF B
        half_height_a = geo_scale_a[1]
        half_height_b = geo_scale_b[1]
        # edge from capsule A
        # depending on point id, we query an edge from 0 to 0.5 or 0.5 to 1
        e0 = wp.vec3(0.0, half_height_a * float(point_id % 2), 0.0)
        e1 = wp.vec3(0.0, -half_height_a * float((point_id + 1) % 2), 0.0)
        edge0_world = wp.transform_point(X_ws_a, e0)
        edge1_world = wp.transform_point(X_ws_a, e1)
        edge0_b = wp.transform_point(X_sw_b, edge0_world)
        edge1_b = wp.transform_point(X_sw_b, edge1_world)
        max_iter = edge_sdf_iter
        u = closest_edge_coordinate_capsule(geo_scale_b[0], geo_scale_b[1], edge0_b, edge1_b, max_iter)
        p_a_world = (1.0 - u) * edge0_world + u * edge1_world
        p0_b_world = wp.transform_point(X_ws_b, wp.vec3(0.0, half_height_b, 0.0))
        p1_b_world = wp.transform_point(X_ws_b, wp.vec3(0.0, -half_height_b, 0.0))
        p_b_world = closest_point_line_segment(p0_b_world, p1_b_world, p_a_world)
        diff = p_a_world - p_b_world
        normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_CAPSULE and geo_type_b == wp.sim.GEO_MESH:
        # find closest edge coordinate to mesh SDF B
        half_height_a = geo_scale_a[1]
        # edge from capsule A
        # depending on point id, we query an edge from -h to 0 or 0 to h
        e0 = wp.vec3(0.0, -half_height_a * float(point_id % 2), 0.0)
        e1 = wp.vec3(0.0, half_height_a * float((point_id + 1) % 2), 0.0)
        edge0_world = wp.transform_point(X_ws_a, e0)
        edge1_world = wp.transform_point(X_ws_a, e1)
        edge0_b = wp.transform_point(X_sw_b, edge0_world)
        edge1_b = wp.transform_point(X_sw_b, edge1_world)
        max_iter = edge_sdf_iter
        max_dist = (rigid_contact_margin + thickness) / min_scale_b
        mesh_b = geo.source[shape_b]
        u = closest_edge_coordinate_mesh(
            mesh_b, wp.cw_div(edge0_b, geo_scale_b), wp.cw_div(edge1_b, geo_scale_b), max_iter, max_dist
        )
        p_a_world = (1.0 - u) * edge0_world + u * edge1_world
        query_b_local = wp.transform_point(X_sw_b, p_a_world)
        mesh_b = geo.source[shape_b]

        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        res = wp.mesh_query_point_sign_normal(
            mesh_b, wp.cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index, face_u, face_v
        )
        if res:
            shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
            shape_p = wp.cw_mul(shape_p, geo_scale_b)
            p_b_world = wp.transform_point(X_ws_b, shape_p)
            p_a_world = closest_point_line_segment(edge0_world, edge1_world, p_b_world)
            # contact direction vector in world frame
            diff = p_a_world - p_b_world
            normal = wp.normalize(diff)
            distance = wp.dot(diff, normal)
        else:
            return

    elif geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_CAPSULE:
        # vertex-based contact
        mesh = wp.mesh_get(geo.source[shape_a])
        body_a_pos = wp.cw_mul(mesh.points[point_id], geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, body_a_pos)
        # find closest point + contact normal on capsule B
        half_height_b = geo_scale_b[1]
        A_b = wp.transform_point(X_ws_b, wp.vec3(0.0, half_height_b, 0.0))
        B_b = wp.transform_point(X_ws_b, wp.vec3(0.0, -half_height_b, 0.0))
        p_b_world = closest_point_line_segment(A_b, B_b, p_a_world)
        diff = p_a_world - p_b_world
        # this is more reliable in practice than using the SDF gradient
        normal = wp.normalize(diff)
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_CAPSULE and geo_type_b == wp.sim.GEO_PLANE:
        plane_width = geo_scale_b[0]
        plane_length = geo_scale_b[1]
        if point_id < 2:
            # vertex-based collision
            half_height_a = geo_scale_a[1]
            side = float(point_id) * 2.0 - 1.0
            p_a_world = wp.transform_point(X_ws_a, wp.vec3(0.0, side * half_height_a, 0.0))
            query_b = wp.transform_point(X_sw_b, p_a_world)
            p_b_body = closest_point_plane(geo_scale_b[0], geo_scale_b[1], query_b)
            p_b_world = wp.transform_point(X_ws_b, p_b_body)
            diff = p_a_world - p_b_world
            if geo_scale_b[0] > 0.0 and geo_scale_b[1] > 0.0:
                normal = wp.normalize(diff)
            else:
                normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
            distance = wp.dot(diff, normal)
        else:
            # contact between capsule A and edges of finite plane B
            plane_width = geo_scale_b[0]
            plane_length = geo_scale_b[1]
            edge = get_plane_edge(point_id - 2, plane_width, plane_length)
            edge0_world = wp.transform_point(X_ws_b, wp.spatial_top(edge))
            edge1_world = wp.transform_point(X_ws_b, wp.spatial_bottom(edge))
            edge0_a = wp.transform_point(X_sw_a, edge0_world)
            edge1_a = wp.transform_point(X_sw_a, edge1_world)
            max_iter = edge_sdf_iter
            u = closest_edge_coordinate_capsule(geo_scale_a[0], geo_scale_a[1], edge0_a, edge1_a, max_iter)
            p_b_world = (1.0 - u) * edge0_world + u * edge1_world

            # find closest point + contact normal on capsule A
            half_height_a = geo_scale_a[1]
            p0_a_world = wp.transform_point(X_ws_a, wp.vec3(0.0, half_height_a, 0.0))
            p1_a_world = wp.transform_point(X_ws_a, wp.vec3(0.0, -half_height_a, 0.0))
            p_a_world = closest_point_line_segment(p0_a_world, p1_a_world, p_b_world)
            diff = p_a_world - p_b_world
            # normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
            normal = wp.normalize(diff)
            distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_BOX:
        # vertex-based contact
        mesh = wp.mesh_get(geo.source[shape_a])
        body_a_pos = wp.cw_mul(mesh.points[point_id], geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, body_a_pos)
        # find closest point + contact normal on box B
        query_b = wp.transform_point(X_sw_b, p_a_world)
        p_b_body = closest_point_box(geo_scale_b, query_b)
        p_b_world = wp.transform_point(X_ws_b, p_b_body)
        diff = p_a_world - p_b_world
        # this is more reliable in practice than using the SDF gradient
        normal = wp.normalize(diff)
        if box_sdf(geo_scale_b, query_b) < 0.0:
            normal = -normal
        distance = wp.dot(diff, normal)

    elif geo_type_a == wp.sim.GEO_BOX and geo_type_b == wp.sim.GEO_MESH:
        # vertex-based contact
        query_a = get_box_vertex(point_id, geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, query_a)
        query_b_local = wp.transform_point(X_sw_b, p_a_world)
        mesh_b = geo.source[shape_b]
        max_dist = (rigid_contact_margin + thickness) / min_scale_b
        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        res = wp.mesh_query_point_sign_normal(
            mesh_b, wp.cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index, face_u, face_v
        )

        if res:
            shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
            shape_p = wp.cw_mul(shape_p, geo_scale_b)
            p_b_world = wp.transform_point(X_ws_b, shape_p)
            # contact direction vector in world frame
            diff_b = p_a_world - p_b_world
            normal = wp.normalize(diff_b) * sign
            distance = wp.dot(diff_b, normal)
        else:
            return

    elif geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_MESH:
        # vertex-based contact
        mesh = wp.mesh_get(geo.source[shape_a])
        mesh_b = geo.source[shape_b]

        body_a_pos = wp.cw_mul(mesh.points[point_id], geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, body_a_pos)
        query_b_local = wp.transform_point(X_sw_b, p_a_world)

        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        min_scale = min(min_scale_a, min_scale_b)
        max_dist = (rigid_contact_margin + thickness) / min_scale

        res = wp.mesh_query_point_sign_normal(
            mesh_b, wp.cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index, face_u, face_v
        )

        if res:
            shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
            shape_p = wp.cw_mul(shape_p, geo_scale_b)
            p_b_world = wp.transform_point(X_ws_b, shape_p)
            # contact direction vector in world frame
            diff_b = p_a_world - p_b_world
            normal = wp.normalize(diff_b) * sign
            distance = wp.dot(diff_b, normal)
        else:
            return

    elif geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_PLANE:
        # vertex-based contact
        mesh = wp.mesh_get(geo.source[shape_a])
        body_a_pos = wp.cw_mul(mesh.points[point_id], geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, body_a_pos)
        query_b = wp.transform_point(X_sw_b, p_a_world)
        p_b_body = closest_point_plane(geo_scale_b[0], geo_scale_b[1], query_b)
        p_b_world = wp.transform_point(X_ws_b, p_b_body)
        diff = p_a_world - p_b_world

        # if the plane is infinite or the point is within the plane we fix the normal to prevent intersections
        if (geo_scale_b[0] == 0.0 and geo_scale_b[1] == 0.0) or (
            wp.abs(query_b[0]) < geo_scale_b[0] and wp.abs(query_b[2]) < geo_scale_b[1]
        ):
            normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
            distance = wp.dot(diff, normal)
        else:
            normal = wp.normalize(diff)
            distance = wp.dot(diff, normal)
            # ignore extreme penetrations (e.g. when mesh is below the plane)
            if distance < -rigid_contact_margin:
                return

    else:
        print("Unsupported geometry pair in collision handling")
        return

    d = distance - thickness
    if d < rigid_contact_margin:
        if contact_pairwise_counter:
            pair_contact_id = limited_counter_increment(
                contact_pairwise_counter, pair_index, contact_tids, tid, contact_limit
            )
            if pair_contact_id == -1:
                # wp.printf("Reached contact point limit %d >= %d for shape pair %d and %d (pair_index: %d)\n",
                #           contact_pairwise_counter[pair_index], contact_limit, shape_a, shape_b, pair_index)
                # reached contact point limit
                return
        index = counter_increment(contact_count, 0, contact_tids, tid)
        if index == -1:
            return
        contact_shape0[index] = shape_a
        contact_shape1[index] = shape_b
        # transform from world into body frame (so the contact point includes the shape transform)
        contact_point0[index] = wp.transform_point(X_bw_a, p_a_world)
        contact_point1[index] = wp.transform_point(X_bw_b, p_b_world)
        contact_offset0[index] = wp.transform_vector(X_bw_a, -thickness_a * normal)
        contact_offset1[index] = wp.transform_vector(X_bw_b, thickness_b * normal)
        contact_normal[index] = normal
        contact_thickness[index] = thickness


def collide(
    model: Model,
    state: State,
    edge_sdf_iter: int = 10,
    iterate_mesh_vertices: bool = True,
    requires_grad: Optional[bool] = None,
) -> None:
    """Generate contact points for the particles and rigid bodies in the model for use in contact-dynamics kernels.

    Args:
        model: The model to be simulated.
        state: The state of the model.
        edge_sdf_iter: Number of search iterations for finding closest contact points between edges and SDF.
        iterate_mesh_vertices: Whether to iterate over all vertices of a mesh for contact generation
            (used for capsule/box <> mesh collision).
        requires_grad: Whether to duplicate contact arrays for gradient computation
            (if ``None``, uses ``model.requires_grad``).
    """

    if requires_grad is None:
        requires_grad = model.requires_grad

    with wp.ScopedTimer("collide", False):
        # generate soft contacts for particles and shapes except ground plane (last shape)
        if model.particle_count and model.shape_count > 1:
            if requires_grad:
                model.soft_contact_body_pos = wp.empty_like(model.soft_contact_body_pos)
                model.soft_contact_body_vel = wp.empty_like(model.soft_contact_body_vel)
                model.soft_contact_normal = wp.empty_like(model.soft_contact_normal)
            # clear old count
            model.soft_contact_count.zero_()
            wp.launch(
                kernel=create_soft_contacts,
                dim=model.particle_count * (model.shape_count - 1),
                inputs=[
                    state.particle_q,
                    model.particle_radius,
                    model.particle_flags,
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_geo,
                    model.soft_contact_margin,
                    model.soft_contact_max,
                    model.shape_count - 1,
                ],
                outputs=[
                    model.soft_contact_count,
                    model.soft_contact_particle,
                    model.soft_contact_shape,
                    model.soft_contact_body_pos,
                    model.soft_contact_body_vel,
                    model.soft_contact_normal,
                    model.soft_contact_tids,
                ],
                device=model.device,
            )

        if model.shape_contact_pair_count or (model.ground and model.shape_ground_contact_pair_count):
            # clear old count
            model.rigid_contact_count.zero_()

            model.rigid_contact_broad_shape0.fill_(-1)
            model.rigid_contact_broad_shape1.fill_(-1)

        if model.shape_contact_pair_count:
            wp.launch(
                kernel=broadphase_collision_pairs,
                dim=model.shape_contact_pair_count,
                inputs=[
                    model.shape_contact_pairs,
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.body_mass,
                    model.shape_count,
                    model.shape_geo,
                    model.shape_collision_radius,
                    model.rigid_contact_max,
                    model.rigid_contact_margin,
                    model.rigid_mesh_contact_max,
                    iterate_mesh_vertices,
                ],
                outputs=[
                    model.rigid_contact_count,
                    model.rigid_contact_broad_shape0,
                    model.rigid_contact_broad_shape1,
                    model.rigid_contact_point_id,
                    model.rigid_contact_point_limit,
                ],
                device=model.device,
                record_tape=False,
            )

        if model.ground and model.shape_ground_contact_pair_count:
            wp.launch(
                kernel=broadphase_collision_pairs,
                dim=model.shape_ground_contact_pair_count,
                inputs=[
                    model.shape_ground_contact_pairs,
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.body_mass,
                    model.shape_count,
                    model.shape_geo,
                    model.shape_collision_radius,
                    model.rigid_contact_max,
                    model.rigid_contact_margin,
                    model.rigid_mesh_contact_max,
                    iterate_mesh_vertices,
                ],
                outputs=[
                    model.rigid_contact_count,
                    model.rigid_contact_broad_shape0,
                    model.rigid_contact_broad_shape1,
                    model.rigid_contact_point_id,
                    model.rigid_contact_point_limit,
                ],
                device=model.device,
                record_tape=False,
            )

        if model.shape_contact_pair_count or (model.ground and model.shape_ground_contact_pair_count):
            if requires_grad:
                model.rigid_contact_point0 = wp.empty_like(model.rigid_contact_point0)
                model.rigid_contact_point1 = wp.empty_like(model.rigid_contact_point1)
                model.rigid_contact_offset0 = wp.empty_like(model.rigid_contact_offset0)
                model.rigid_contact_offset1 = wp.empty_like(model.rigid_contact_offset1)
                model.rigid_contact_normal = wp.empty_like(model.rigid_contact_normal)
                model.rigid_contact_thickness = wp.empty_like(model.rigid_contact_thickness)
                model.rigid_contact_count = wp.zeros_like(model.rigid_contact_count)
                model.rigid_contact_tids = wp.full_like(model.rigid_contact_tids, -1)
                model.rigid_contact_shape0 = wp.empty_like(model.rigid_contact_shape0)
                model.rigid_contact_shape1 = wp.empty_like(model.rigid_contact_shape1)

                if model.rigid_contact_pairwise_counter is not None:
                    model.rigid_contact_pairwise_counter = wp.zeros_like(model.rigid_contact_pairwise_counter)
            else:
                model.rigid_contact_count.zero_()
                model.rigid_contact_tids.fill_(-1)

                if model.rigid_contact_pairwise_counter is not None:
                    model.rigid_contact_pairwise_counter.zero_()

            model.rigid_contact_shape0.fill_(-1)
            model.rigid_contact_shape1.fill_(-1)

            wp.launch(
                kernel=handle_contact_pairs,
                dim=model.rigid_contact_max,
                inputs=[
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_geo,
                    model.rigid_contact_margin,
                    model.rigid_contact_broad_shape0,
                    model.rigid_contact_broad_shape1,
                    model.shape_count,
                    model.rigid_contact_point_id,
                    model.rigid_contact_point_limit,
                    edge_sdf_iter,
                ],
                outputs=[
                    model.rigid_contact_count,
                    model.rigid_contact_shape0,
                    model.rigid_contact_shape1,
                    model.rigid_contact_point0,
                    model.rigid_contact_point1,
                    model.rigid_contact_offset0,
                    model.rigid_contact_offset1,
                    model.rigid_contact_normal,
                    model.rigid_contact_thickness,
                    model.rigid_contact_pairwise_counter,
                    model.rigid_contact_tids,
                ],
                device=model.device,
            )


@wp.func
def compute_tri_aabb(
    v1: wp.vec3,
    v2: wp.vec3,
    v3: wp.vec3,
):
    lower = wp.min(wp.min(v1, v2), v3)
    upper = wp.max(wp.max(v1, v2), v3)

    return lower, upper


@wp.kernel
def compute_tri_aabbs(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    v1 = pos[tri_indices[t_id, 0]]
    v2 = pos[tri_indices[t_id, 1]]
    v3 = pos[tri_indices[t_id, 2]]

    lower, upper = compute_tri_aabb(v1, v2, v3)

    lower_bounds[t_id] = lower
    upper_bounds[t_id] = upper


@wp.kernel
def compute_edge_aabbs(
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    e_id = wp.tid()

    v1 = pos[edge_indices[e_id, 2]]
    v2 = pos[edge_indices[e_id, 3]]

    lower_bounds[e_id] = wp.min(v1, v2)
    upper_bounds[e_id] = wp.max(v1, v2)


@wp.func
def tri_is_neighbor(a_1: wp.int32, a_2: wp.int32, a_3: wp.int32, b_1: wp.int32, b_2: wp.int32, b_3: wp.int32):
    tri_is_neighbor = (
        a_1 == b_1
        or a_1 == b_2
        or a_1 == b_3
        or a_2 == b_1
        or a_2 == b_2
        or a_2 == b_3
        or a_3 == b_1
        or a_3 == b_2
        or a_3 == b_3
    )

    return tri_is_neighbor


@wp.func
def vertex_adjacent_to_triangle(v: wp.int32, a: wp.int32, b: wp.int32, c: wp.int32):
    return v == a or v == b or v == c


@wp.kernel
def init_triangle_collision_data_kernel(
    query_radius: float,
    # outputs
    triangle_colliding_vertices_count: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    tri_index = wp.tid()

    triangle_colliding_vertices_count[tri_index] = 0
    triangle_colliding_vertices_min_dist[tri_index] = query_radius

    if tri_index == 0:
        for i in range(3):
            resize_flags[i] = 0


@wp.kernel
def vertex_triangle_collision_detection_kernel(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_offsets: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_buffer_sizes: wp.array(dtype=wp.int32),
    # outputs
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_min_dist: wp.array(dtype=float),
    triangle_colliding_vertices: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_count: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    """
    This function applies discrete collision detection between vertices and triangles. It uses pre-allocated spaces to
    record the collision data. This collision detector works both ways, i.e., it records vertices' colliding triangles to
    `vertex_colliding_triangles`, and records each triangles colliding vertices to `triangle_colliding_vertices`.

    This function assumes that all the vertices are on triangles, and can be indexed from the pos argument.

    Note:

        The collision date buffer is pre-allocated and cannot be changed during collision detection, therefore, the space
        may not be enough. If the space is not enough to record all the collision information, the function will set a
        certain element in resized_flag to be true. The user can reallocate the buffer based on vertex_colliding_triangles_count
        and vertex_colliding_triangles_count.

    Attributes:
        bvh_id (int): the bvh id you want to collide with
        query_radius (float): the contact radius. vertex-triangle pairs whose distance are less than this will get detected
        pos (array): positions of all the vertices that make up triangles
        vertex_colliding_triangles (array): flattened buffer of vertices' collision triangles
        vertex_colliding_triangles_count (array): number of triangles each vertex collides
        vertex_colliding_triangles_offsets (array): where each vertex' collision buffer starts
        vertex_colliding_triangles_buffer_sizes (array): size of each vertex' collision buffer, will be modified if resizing is needed
        vertex_colliding_triangles_min_dist (array): each vertex' min distance to all (non-neighbor) triangles
        triangle_colliding_vertices (array): positions of all the triangles' collision vertices, every two elements
            records the vertex index and a triangle index it collides to
        triangle_colliding_vertices_count (array): number of triangles each vertex collides
        triangle_colliding_vertices_offsets (array): where each triangle's collision buffer starts
        triangle_colliding_vertices_buffer_sizes (array): size of each triangle's collision buffer, will be modified if resizing is needed
        triangle_colliding_vertices_min_dist (array): each triangle's min distance to all (non-self) vertices
        resized_flag (array): size == 3, (vertex_buffer_resize_required, triangle_buffer_resize_required, edge_buffer_resize_required)
    """

    v_index = wp.tid()
    v = pos[v_index]
    vertex_buffer_offset = vertex_colliding_triangles_offsets[v_index]
    vertex_buffer_size = vertex_colliding_triangles_offsets[v_index + 1] - vertex_buffer_offset

    lower = wp.vec3(v[0] - query_radius, v[1] - query_radius, v[2] - query_radius)
    upper = wp.vec3(v[0] + query_radius, v[1] + query_radius, v[2] + query_radius)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    tri_index = wp.int32(0)
    vertex_num_collisions = wp.int32(0)
    min_dis_to_tris = query_radius
    while wp.bvh_query_next(query, tri_index):
        t1 = tri_indices[tri_index, 0]
        t2 = tri_indices[tri_index, 1]
        t3 = tri_indices[tri_index, 2]
        if vertex_adjacent_to_triangle(v_index, t1, t2, t3):
            continue

        u1 = pos[t1]
        u2 = pos[t2]
        u3 = pos[t3]

        closest_p, bary, feature_type = triangle_closest_point(u1, u2, u3, v)

        dist = wp.length(closest_p - v)

        if dist < query_radius:
            # record v-f collision to vertex
            min_dis_to_tris = wp.min(min_dis_to_tris, dist)
            if vertex_num_collisions < vertex_buffer_size:
                vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions)] = v_index
                vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions) + 1] = tri_index
            else:
                resize_flags[VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

            vertex_num_collisions = vertex_num_collisions + 1

            wp.atomic_min(triangle_colliding_vertices_min_dist, tri_index, dist)
            tri_buffer_size = triangle_colliding_vertices_buffer_sizes[tri_index]
            tri_num_collisions = wp.atomic_add(triangle_colliding_vertices_count, tri_index, 1)

            if tri_num_collisions < tri_buffer_size:
                tri_buffer_offset = triangle_colliding_vertices_offsets[tri_index]
                # record v-f collision to triangle
                triangle_colliding_vertices[tri_buffer_offset + tri_num_collisions] = v_index
            else:
                resize_flags[TRI_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

    vertex_colliding_triangles_count[v_index] = vertex_num_collisions
    vertex_colliding_triangles_min_dist[v_index] = min_dis_to_tris


@wp.kernel
def vertex_triangle_collision_detection_no_triangle_buffers_kernel(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32),
    # outputs
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_min_dist: wp.array(dtype=float),
    triangle_colliding_vertices_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    """
    This function applies discrete collision detection between vertices and triangles. It uses pre-allocated spaces to
    record the collision data. Unlike `vertex_triangle_collision_detection_kernel`, this collision detection kernel
    works only in one way, i.e., it only records vertices' colliding triangles to `vertex_colliding_triangles`.

    This function assumes that all the vertices are on triangles, and can be indexed from the pos argument.

    Note:

        The collision date buffer is pre-allocated and cannot be changed during collision detection, therefore, the space
        may not be enough. If the space is not enough to record all the collision information, the function will set a
        certain element in resized_flag to be true. The user can reallocate the buffer based on vertex_colliding_triangles_count
        and vertex_colliding_triangles_count.

    Attributes:
        bvh_id (int): the bvh id you want to collide with
        query_radius (float): the contact radius. vertex-triangle pairs whose distance are less than this will get detected
        pos (array): positions of all the vertices that make up triangles
        vertex_colliding_triangles (array): flattened buffer of vertices' collision triangles, every two elements records
            the vertex index and a triangle index it collides to
        vertex_colliding_triangles_count (array): number of triangles each vertex collides
        vertex_colliding_triangles_offsets (array): where each vertex' collision buffer starts
        vertex_colliding_triangles_buffer_sizes (array): size of each vertex' collision buffer, will be modified if resizing is needed
        vertex_colliding_triangles_min_dist (array): each vertex' min distance to all (non-neighbor) triangles
        triangle_colliding_vertices_min_dist (array): each triangle's min distance to all (non-self) vertices
        resized_flag (array): size == 3, (vertex_buffer_resize_required, triangle_buffer_resize_required, edge_buffer_resize_required)
    """

    v_index = wp.tid()
    v = pos[v_index]
    vertex_buffer_offset = vertex_colliding_triangles_offsets[v_index]
    vertex_buffer_size = vertex_colliding_triangles_offsets[v_index + 1] - vertex_buffer_offset

    lower = wp.vec3(v[0] - query_radius, v[1] - query_radius, v[2] - query_radius)
    upper = wp.vec3(v[0] + query_radius, v[1] + query_radius, v[2] + query_radius)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    tri_index = wp.int32(0)
    vertex_num_collisions = wp.int32(0)
    min_dis_to_tris = query_radius
    while wp.bvh_query_next(query, tri_index):
        t1 = tri_indices[tri_index, 0]
        t2 = tri_indices[tri_index, 1]
        t3 = tri_indices[tri_index, 2]
        if vertex_adjacent_to_triangle(v_index, t1, t2, t3):
            continue

        u1 = pos[t1]
        u2 = pos[t2]
        u3 = pos[t3]

        closest_p, bary, feature_type = triangle_closest_point(u1, u2, u3, v)

        dist = wp.length(closest_p - v)

        if dist < query_radius:
            # record v-f collision to vertex
            min_dis_to_tris = wp.min(min_dis_to_tris, dist)
            if vertex_num_collisions < vertex_buffer_size:
                vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions)] = v_index
                vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions) + 1] = tri_index
            else:
                resize_flags[VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

            vertex_num_collisions = vertex_num_collisions + 1

            wp.atomic_min(triangle_colliding_vertices_min_dist, tri_index, dist)

    vertex_colliding_triangles_count[v_index] = vertex_num_collisions
    vertex_colliding_triangles_min_dist[v_index] = min_dis_to_tris


@wp.kernel
def edge_colliding_edges_detection_kernel(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_colliding_edges_offsets: wp.array(dtype=wp.int32),
    edge_colliding_edges_buffer_sizes: wp.array(dtype=wp.int32),
    edge_edge_parallel_epsilon: float,
    # outputs
    edge_colliding_edges: wp.array(dtype=wp.int32),
    edge_colliding_edges_count: wp.array(dtype=wp.int32),
    edge_colliding_edges_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    """
    bvh_id (int): the bvh id you want to do collision detection on
    query_radius (float):
    pos (array): positions of all the vertices that make up edges
    edge_colliding_triangles (array): flattened buffer of edges' collision edges
    edge_colliding_edges_count (array): number of edges each edge collides
    edge_colliding_triangles_offsets (array): where each edge's collision buffer starts
    edge_colliding_triangles_buffer_size (array): size of each edge's collision buffer, will be modified if resizing is needed
    edge_min_dis_to_triangles (array): each vertex' min distance to all (non-neighbor) triangles
    resized_flag (array): size == 3, (vertex_buffer_resize_required, triangle_buffer_resize_required, edge_buffer_resize_required)
    """
    e_index = wp.tid()

    e0_v0 = edge_indices[e_index, 2]
    e0_v1 = edge_indices[e_index, 3]

    e0_v0_pos = pos[e0_v0]
    e0_v1_pos = pos[e0_v1]

    lower = wp.min(e0_v0_pos, e0_v1_pos)
    upper = wp.max(e0_v0_pos, e0_v1_pos)

    lower = wp.vec3(lower[0] - query_radius, lower[1] - query_radius, lower[2] - query_radius)
    upper = wp.vec3(upper[0] + query_radius, upper[1] + query_radius, upper[2] + query_radius)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)

    colliding_edge_index = wp.int32(0)
    edge_num_collisions = wp.int32(0)
    min_dis_to_edges = query_radius
    while wp.bvh_query_next(query, colliding_edge_index):
        e1_v0 = edge_indices[colliding_edge_index, 2]
        e1_v1 = edge_indices[colliding_edge_index, 3]

        if e0_v0 == e1_v0 or e0_v0 == e1_v1 or e0_v1 == e1_v0 or e0_v1 == e1_v1:
            continue

        e1_v0_pos = pos[e1_v0]
        e1_v1_pos = pos[e1_v1]

        st = wp.closest_point_edge_edge(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos, edge_edge_parallel_epsilon)
        s = st[0]
        t = st[1]
        c1 = e0_v0_pos + (e0_v1_pos - e0_v0_pos) * s
        c2 = e1_v0_pos + (e1_v1_pos - e1_v0_pos) * t

        dist = wp.length(c1 - c2)
        if dist < query_radius:
            edge_buffer_offset = edge_colliding_edges_offsets[e_index]
            edge_buffer_size = edge_colliding_edges_offsets[e_index + 1] - edge_buffer_offset

            # record e-e collision to e0, and leave e1; e1 will detect this collision from its own thread
            min_dis_to_edges = wp.min(min_dis_to_edges, dist)
            if edge_num_collisions < edge_buffer_size:
                edge_colliding_edges[2 * (edge_buffer_offset + edge_num_collisions)] = e_index
                edge_colliding_edges[2 * (edge_buffer_offset + edge_num_collisions) + 1] = colliding_edge_index
            else:
                resize_flags[EDGE_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

            edge_num_collisions = edge_num_collisions + 1

    edge_colliding_edges_count[e_index] = edge_num_collisions
    edge_colliding_edges_min_dist[e_index] = min_dis_to_edges


@wp.kernel
def triangle_triangle_collision_detection_kernel(
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    triangle_intersecting_triangles_offsets: wp.array(dtype=wp.int32),
    # outputs
    triangle_intersecting_triangles: wp.array(dtype=wp.int32),
    triangle_intersecting_triangles_count: wp.array(dtype=wp.int32),
    resize_flags: wp.array(dtype=wp.int32),
):
    tri_index = wp.tid()
    t1_v1 = tri_indices[tri_index, 0]
    t1_v2 = tri_indices[tri_index, 1]
    t1_v3 = tri_indices[tri_index, 2]

    v1 = pos[t1_v1]
    v2 = pos[t1_v2]
    v3 = pos[t1_v3]

    lower, upper = compute_tri_aabb(v1, v2, v3)

    buffer_offset = triangle_intersecting_triangles_offsets[tri_index]
    buffer_size = triangle_intersecting_triangles_offsets[tri_index + 1] - buffer_offset

    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    tri_index_2 = wp.int32(0)
    intersection_count = wp.int32(0)
    while wp.bvh_query_next(query, tri_index_2):
        t2_v1 = tri_indices[tri_index_2, 0]
        t2_v2 = tri_indices[tri_index_2, 1]
        t2_v3 = tri_indices[tri_index_2, 2]

        # filter out intersection test with neighbor triangles
        if (
            vertex_adjacent_to_triangle(t1_v1, t2_v1, t2_v2, t2_v3)
            or vertex_adjacent_to_triangle(t1_v2, t2_v1, t2_v2, t2_v3)
            or vertex_adjacent_to_triangle(t1_v3, t2_v1, t2_v2, t2_v3)
        ):
            continue

        u1 = pos[t2_v1]
        u2 = pos[t2_v2]
        u3 = pos[t2_v3]

        if wp.intersect_tri_tri(v1, v2, v3, u1, u2, u3):
            if intersection_count < buffer_size:
                triangle_intersecting_triangles[buffer_offset + intersection_count] = tri_index_2
            else:
                resize_flags[TRI_TRI_COLLISION_BUFFER_OVERFLOW_INDEX] = 1
            intersection_count = intersection_count + 1

    triangle_intersecting_triangles_count[tri_index] = intersection_count


@wp.struct
class TriMeshCollisionInfo:
    vertex_indices: wp.array(dtype=wp.int32)
    # size: 2 x sum(vertex_colliding_triangles_buffer_sizes)
    # every two elements records the vertex index and a triangle index it collides to
    vertex_colliding_triangles: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_min_dist: wp.array(dtype=float)

    triangle_colliding_vertices: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_offsets: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_buffer_sizes: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_count: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_min_dist: wp.array(dtype=float)

    # size: 2 x sum(edge_colliding_edges_buffer_sizes)
    # every two elements records the edge index and an edge index it collides to
    edge_colliding_edges: wp.array(dtype=wp.int32)
    edge_colliding_edges_offsets: wp.array(dtype=wp.int32)
    edge_colliding_edges_buffer_sizes: wp.array(dtype=wp.int32)
    edge_colliding_edges_count: wp.array(dtype=wp.int32)
    edge_colliding_edges_min_dist: wp.array(dtype=float)


@wp.func
def get_vertex_colliding_triangles_count(col_info: TriMeshCollisionInfo, v: int):
    return wp.min(col_info.vertex_colliding_triangles_count[v], col_info.vertex_colliding_triangles_buffer_sizes[v])


@wp.func
def get_vertex_colliding_triangles(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision) + 1]


@wp.func
def get_vertex_collision_buffer_vertex_index(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision)]


@wp.func
def get_triangle_colliding_vertices_count(col_info: TriMeshCollisionInfo, tri: int):
    return wp.min(
        col_info.triangle_colliding_vertices_count[tri], col_info.triangle_colliding_vertices_buffer_sizes[tri]
    )


@wp.func
def get_triangle_colliding_vertices(col_info: TriMeshCollisionInfo, tri: int, i_collision: int):
    offset = col_info.triangle_colliding_vertices_offsets[tri]
    return col_info.triangle_colliding_vertices[offset + i_collision]


@wp.func
def get_edge_colliding_edges_count(col_info: TriMeshCollisionInfo, e: int):
    return wp.min(col_info.edge_colliding_edges_count[e], col_info.edge_colliding_edges_buffer_sizes[e])


@wp.func
def get_edge_colliding_edges(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision) + 1]


@wp.func
def get_edge_collision_buffer_edge_index(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision)]


class TriMeshCollisionDetector:
    def __init__(
        self,
        model: Model,
        record_triangle_contacting_vertices=False,
        vertex_positions=None,
        vertex_collision_buffer_pre_alloc=8,
        vertex_collision_buffer_max_alloc=256,
        triangle_collision_buffer_pre_alloc=16,
        triangle_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=8,
        edge_collision_buffer_max_alloc=256,
        triangle_triangle_collision_buffer_pre_alloc=8,
        triangle_triangle_collision_buffer_max_alloc=256,
        edge_edge_parallel_epsilon=1e-5,
    ):
        self.model = model
        self.record_triangle_contacting_vertices = record_triangle_contacting_vertices
        self.vertex_positions = model.particle_q if vertex_positions is None else vertex_positions
        self.device = model.device
        self.vertex_collision_buffer_pre_alloc = vertex_collision_buffer_pre_alloc
        self.vertex_collision_buffer_max_alloc = vertex_collision_buffer_max_alloc
        self.triangle_collision_buffer_pre_alloc = triangle_collision_buffer_pre_alloc
        self.triangle_collision_buffer_max_alloc = triangle_collision_buffer_max_alloc
        self.edge_collision_buffer_pre_alloc = edge_collision_buffer_pre_alloc
        self.edge_collision_buffer_max_alloc = edge_collision_buffer_max_alloc
        self.triangle_triangle_collision_buffer_pre_alloc = triangle_triangle_collision_buffer_pre_alloc
        self.triangle_triangle_collision_buffer_max_alloc = triangle_triangle_collision_buffer_max_alloc

        self.edge_edge_parallel_epsilon = edge_edge_parallel_epsilon

        self.lower_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=model.tri_count,
            device=model.device,
        )

        self.bvh_tris = wp.Bvh(self.lower_bounds_tris, self.upper_bounds_tris)

        # collision detections results

        # vertex collision buffers
        self.vertex_colliding_triangles = wp.zeros(
            shape=(2 * model.particle_count * self.vertex_collision_buffer_pre_alloc,),
            dtype=wp.int32,
            device=self.device,
        )
        self.vertex_colliding_triangles_count = wp.array(
            shape=(model.particle_count,), dtype=wp.int32, device=self.device
        )
        self.vertex_colliding_triangles_min_dist = wp.array(
            shape=(model.particle_count,), dtype=float, device=self.device
        )
        self.vertex_colliding_triangles_buffer_sizes = wp.full(
            shape=(model.particle_count,),
            value=self.vertex_collision_buffer_pre_alloc,
            dtype=wp.int32,
            device=self.device,
        )
        self.vertex_colliding_triangles_offsets = wp.array(
            shape=(model.particle_count + 1,), dtype=wp.int32, device=self.device
        )
        self.compute_collision_buffer_offsets(
            self.vertex_colliding_triangles_buffer_sizes, self.vertex_colliding_triangles_offsets
        )

        if record_triangle_contacting_vertices:
            # triangle collision buffers
            self.triangle_colliding_vertices = wp.zeros(
                shape=(model.tri_count * self.triangle_collision_buffer_pre_alloc,), dtype=wp.int32, device=self.device
            )
            self.triangle_colliding_vertices_count = wp.zeros(
                shape=(model.tri_count,), dtype=wp.int32, device=self.device
            )
            self.triangle_colliding_vertices_buffer_sizes = wp.full(
                shape=(model.tri_count,),
                value=self.triangle_collision_buffer_pre_alloc,
                dtype=wp.int32,
                device=self.device,
            )

            self.triangle_colliding_vertices_offsets = wp.array(
                shape=(model.tri_count + 1,), dtype=wp.int32, device=self.device
            )
            self.compute_collision_buffer_offsets(
                self.triangle_colliding_vertices_buffer_sizes, self.triangle_colliding_vertices_offsets
            )
        else:
            self.triangle_colliding_vertices = None
            self.triangle_colliding_vertices_count = None
            self.triangle_colliding_vertices_buffer_sizes = None
            self.triangle_colliding_vertices_offsets = None

        # this is need regardless of whether we record triangle contacting vertices
        self.triangle_colliding_vertices_min_dist = wp.array(shape=(model.tri_count,), dtype=float, device=self.device)

        # edge collision buffers
        self.edge_colliding_edges = wp.zeros(
            shape=(2 * model.edge_count * self.edge_collision_buffer_pre_alloc,), dtype=wp.int32, device=self.device
        )
        self.edge_colliding_edges_count = wp.zeros(shape=(model.edge_count,), dtype=wp.int32, device=self.device)
        self.edge_colliding_edges_buffer_sizes = wp.full(
            shape=(model.edge_count,),
            value=self.edge_collision_buffer_pre_alloc,
            dtype=wp.int32,
            device=self.device,
        )
        self.edge_colliding_edges_offsets = wp.array(shape=(model.edge_count + 1,), dtype=wp.int32, device=self.device)
        self.compute_collision_buffer_offsets(self.edge_colliding_edges_buffer_sizes, self.edge_colliding_edges_offsets)
        self.edge_colliding_edges_min_dist = wp.array(shape=(model.edge_count,), dtype=float, device=self.device)

        self.lower_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=model.edge_count,
            device=model.device,
        )

        self.bvh_edges = wp.Bvh(self.lower_bounds_edges, self.upper_bounds_edges)

        self.resize_flags = wp.zeros(shape=(4,), dtype=wp.int32, device=self.device)

        self.collision_info = self.get_collision_data()

        # data for triangle-triangle intersection; they will only be initialized on demand, as triangle-triangle intersection is not needed for simulation
        self.triangle_intersecting_triangles = None
        self.triangle_intersecting_triangles_count = None
        self.triangle_intersecting_triangles_offsets = None

    def get_collision_data(self):
        collision_info = TriMeshCollisionInfo()

        collision_info.vertex_colliding_triangles = self.vertex_colliding_triangles
        collision_info.vertex_colliding_triangles_offsets = self.vertex_colliding_triangles_offsets
        collision_info.vertex_colliding_triangles_buffer_sizes = self.vertex_colliding_triangles_buffer_sizes
        collision_info.vertex_colliding_triangles_count = self.vertex_colliding_triangles_count
        collision_info.vertex_colliding_triangles_min_dist = self.vertex_colliding_triangles_min_dist

        if self.record_triangle_contacting_vertices:
            collision_info.triangle_colliding_vertices = self.triangle_colliding_vertices
            collision_info.triangle_colliding_vertices_offsets = self.triangle_colliding_vertices_offsets
            collision_info.triangle_colliding_vertices_buffer_sizes = self.triangle_colliding_vertices_buffer_sizes
            collision_info.triangle_colliding_vertices_count = self.triangle_colliding_vertices_count

        collision_info.triangle_colliding_vertices_min_dist = self.triangle_colliding_vertices_min_dist

        collision_info.edge_colliding_edges = self.edge_colliding_edges
        collision_info.edge_colliding_edges_offsets = self.edge_colliding_edges_offsets
        collision_info.edge_colliding_edges_buffer_sizes = self.edge_colliding_edges_buffer_sizes
        collision_info.edge_colliding_edges_count = self.edge_colliding_edges_count
        collision_info.edge_colliding_edges_min_dist = self.edge_colliding_edges_min_dist

        return collision_info

    def compute_collision_buffer_offsets(
        self, buffer_sizes: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32)
    ):
        assert offsets.size == buffer_sizes.size + 1
        offsets_np = np.empty(shape=(offsets.size,), dtype=np.int32)
        offsets_np[1:] = np.cumsum(buffer_sizes.numpy())[:]
        offsets_np[0] = 0

        offsets.assign(offsets_np)

    def rebuild(self, new_pos=None):
        if new_pos is not None:
            self.vertex_positions = new_pos

        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[
                self.vertex_positions,
                self.model.tri_indices,
            ],
            outputs=[self.lower_bounds_tris, self.upper_bounds_tris],
            dim=self.model.tri_count,
            device=self.model.device,
        )
        self.bvh_tris = wp.Bvh(self.lower_bounds_tris, self.upper_bounds_tris)

        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices],
            outputs=[self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges = wp.Bvh(self.lower_bounds_edges, self.upper_bounds_edges)

    def refit(self, new_pos=None):
        if new_pos is not None:
            self.vertex_positions = new_pos

        self.refit_triangles()
        self.refit_edges()

    def refit_triangles(self):
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, self.model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=self.model.tri_count,
            device=self.model.device,
        )
        self.bvh_tris.refit()

    def refit_edges(self):
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges.refit()

    def vertex_triangle_collision_detection(self, query_radius):
        self.vertex_colliding_triangles.fill_(-1)

        if self.record_triangle_contacting_vertices:
            wp.launch(
                kernel=init_triangle_collision_data_kernel,
                inputs=[
                    query_radius,
                ],
                outputs=[
                    self.triangle_colliding_vertices_count,
                    self.triangle_colliding_vertices_min_dist,
                    self.resize_flags,
                ],
                dim=self.model.tri_count,
                device=self.model.device,
            )

            wp.launch(
                kernel=vertex_triangle_collision_detection_kernel,
                inputs=[
                    query_radius,
                    self.bvh_tris.id,
                    self.vertex_positions,
                    self.model.tri_indices,
                    self.vertex_colliding_triangles_offsets,
                    self.vertex_colliding_triangles_buffer_sizes,
                    self.triangle_colliding_vertices_offsets,
                    self.triangle_colliding_vertices_buffer_sizes,
                ],
                outputs=[
                    self.vertex_colliding_triangles,
                    self.vertex_colliding_triangles_count,
                    self.vertex_colliding_triangles_min_dist,
                    self.triangle_colliding_vertices,
                    self.triangle_colliding_vertices_count,
                    self.triangle_colliding_vertices_min_dist,
                    self.resize_flags,
                ],
                dim=self.model.particle_count,
                device=self.model.device,
            )
        else:
            self.triangle_colliding_vertices_min_dist.fill_(query_radius)
            wp.launch(
                kernel=vertex_triangle_collision_detection_no_triangle_buffers_kernel,
                inputs=[
                    query_radius,
                    self.bvh_tris.id,
                    self.vertex_positions,
                    self.model.tri_indices,
                    self.vertex_colliding_triangles_offsets,
                    self.vertex_colliding_triangles_buffer_sizes,
                ],
                outputs=[
                    self.vertex_colliding_triangles,
                    self.vertex_colliding_triangles_count,
                    self.vertex_colliding_triangles_min_dist,
                    self.triangle_colliding_vertices_min_dist,
                    self.resize_flags,
                ],
                dim=self.model.particle_count,
                device=self.model.device,
            )

    def edge_edge_collision_detection(self, query_radius):
        self.edge_colliding_edges.fill_(-1)
        wp.launch(
            kernel=edge_colliding_edges_detection_kernel,
            inputs=[
                query_radius,
                self.bvh_edges.id,
                self.vertex_positions,
                self.model.edge_indices,
                self.edge_colliding_edges_offsets,
                self.edge_colliding_edges_buffer_sizes,
                self.edge_edge_parallel_epsilon,
            ],
            outputs=[
                self.edge_colliding_edges,
                self.edge_colliding_edges_count,
                self.edge_colliding_edges_min_dist,
                self.resize_flags,
            ],
            dim=self.model.edge_count,
            device=self.model.device,
        )

    def triangle_triangle_intersection_detection(self):
        if self.triangle_intersecting_triangles is None:
            self.triangle_intersecting_triangles = wp.zeros(
                shape=(self.model.tri_count * self.triangle_triangle_collision_buffer_pre_alloc,),
                dtype=wp.int32,
                device=self.device,
            )

        if self.triangle_intersecting_triangles_count is None:
            self.triangle_intersecting_triangles_count = wp.array(
                shape=(self.model.tri_count,), dtype=wp.int32, device=self.device
            )

        if self.triangle_intersecting_triangles_offsets is None:
            buffer_sizes = np.full((self.model.tri_count,), self.triangle_triangle_collision_buffer_pre_alloc)
            offsets = np.zeros((self.model.tri_count + 1,), dtype=np.int32)
            offsets[1:] = np.cumsum(buffer_sizes)

            self.triangle_intersecting_triangles_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)

        wp.launch(
            kernel=triangle_triangle_collision_detection_kernel,
            inputs=[
                self.bvh_tris.id,
                self.vertex_positions,
                self.model.tri_indices,
                self.triangle_intersecting_triangles_offsets,
            ],
            outputs=[
                self.triangle_intersecting_triangles,
                self.triangle_intersecting_triangles_count,
                self.resize_flags,
            ],
            dim=self.model.tri_count,
            device=self.model.device,
        )
