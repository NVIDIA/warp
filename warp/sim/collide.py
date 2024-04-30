# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Collision handling functions and kernels.
"""

import warp as wp

from .model import PARTICLE_FLAG_ACTIVE, ModelShapeGeometry


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
        return wp.vec3(0.0, w, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return wp.vec3(1.0 - v - w, v, w)


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
    if qx > qy and qx > qz or qy == 0.0 and qz == 0.0:
        return wp.vec3(sx, 0.0, 0.0)

    # y projection
    if qy > qx and qy > qz or qx == 0.0 and qz == 0.0:
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
        if sx < sy and sx < sz or sy == 0.0 and sz == 0.0:
            x = wp.sign(point[0]) * upper[0]
        elif sy < sx and sy < sz or sx == 0.0 and sz == 0.0:
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
    next_count = wp.atomic_add(counter, counter_index, 1)
    tids[tid] = next_count
    return next_count


@wp.func_replay(counter_increment)
def replay_counter_increment(counter: wp.array(dtype=int), counter_index: int, tids: wp.array(dtype=int), tid: int):
    return tids[tid]


@wp.func
def limited_counter_increment(
    counter: wp.array(dtype=int), counter_index: int, tids: wp.array(dtype=int), tid: int, index_limit: int
):
    # increment counter but only if it is smaller than index_limit, remember which thread received which counter value
    next_count = wp.atomic_add(counter, counter_index, 1)
    if next_count < index_limit or index_limit < 0:
        tids[tid] = next_count
        return next_count
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
            contact_point_limit[pair_index_ab] = 2
            if mesh_contact_max > 0:
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
            contact_point_limit[pair_index_ab] = 12
            # allocate contact points from box B against A
            for i in range(12):
                contact_shape0[index + 12 + i] = shape_b
                contact_shape1[index + 12 + i] = shape_a
                contact_point_id[index + 12 + i] = i
            contact_point_limit[pair_index_ba] = 12
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

            contact_point_limit[pair_index_ab] = num_contacts_a
            if mesh_contact_max > 0:
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

            if mesh_contact_max > 0:
                num_contacts_a = wp.min(mesh_contact_max, num_contacts_a)
                num_contacts_b = wp.min(mesh_contact_max, num_contacts_b)
            contact_point_limit[pair_index_ab] = num_contacts_a
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
        contact_point_limit[pair_index_ab] = num_contacts
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

    point_id = contact_point_id[tid]
    pair_index = shape_a * num_shapes + shape_b
    contact_limit = contact_point_limit[pair_index]
    if contact_pairwise_counter[pair_index] >= contact_limit:
        # reached limit of contact points per contact pair
        return

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
        # use center of box A to query normal to make sure we are not inside B
        query_b = wp.transform_point(X_sw_b, wp.transform_get_translation(X_ws_a))
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
        if (
            geo_scale_b[0] == 0.0
            and geo_scale_b[1] == 0.0
            or wp.abs(query_b[0]) < geo_scale_b[0]
            and wp.abs(query_b[2]) < geo_scale_b[1]
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
        pair_contact_id = limited_counter_increment(
            contact_pairwise_counter, pair_index, contact_tids, tid, contact_limit
        )
        if pair_contact_id == -1:
            # wp.printf("Reached contact point limit %d >= %d for shape pair %d and %d (pair_index: %d)\n",
            #           contact_pairwise_counter[pair_index], contact_limit, shape_a, shape_b, pair_index)
            # reached contact point limit
            return
        index = limited_counter_increment(contact_count, 0, contact_tids, tid, -1)
        contact_shape0[index] = shape_a
        contact_shape1[index] = shape_b
        # transform from world into body frame (so the contact point includes the shape transform)
        contact_point0[index] = wp.transform_point(X_bw_a, p_a_world)
        contact_point1[index] = wp.transform_point(X_bw_b, p_b_world)
        contact_offset0[index] = wp.transform_vector(X_bw_a, -thickness_a * normal)
        contact_offset1[index] = wp.transform_vector(X_bw_b, thickness_b * normal)
        contact_normal[index] = normal
        contact_thickness[index] = thickness


def collide(model, state, edge_sdf_iter: int = 10, iterate_mesh_vertices: bool = True, requires_grad: bool = None):
    """
    Generates contact points for the particles and rigid bodies in the model,
    to be used in the contact dynamics kernel of the integrator.

    Args:
        model: the model to be simulated
        state: the state of the model
        edge_sdf_iter: number of search iterations for finding closest contact points between edges and SDF
        iterate_mesh_vertices: whether to iterate over all vertices of a mesh for contact generation (used for capsule/box <> mesh collision)
        requires_grad: whether to duplicate contact arrays for gradient computation (if None uses model.requires_grad)
    """

    if requires_grad is None:
        requires_grad = model.requires_grad

    with wp.ScopedTimer("collide", False):
        # generate soft contacts for particles and shapes except ground plane (last shape)
        if model.particle_count and model.shape_count > 1:
            if requires_grad:
                model.soft_contact_body_pos = wp.clone(model.soft_contact_body_pos)
                model.soft_contact_body_vel = wp.clone(model.soft_contact_body_vel)
                model.soft_contact_normal = wp.clone(model.soft_contact_normal)
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

        if model.shape_contact_pair_count or model.ground and model.shape_ground_contact_pair_count:
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

        if model.shape_contact_pair_count or model.ground and model.shape_ground_contact_pair_count:
            if requires_grad:
                model.rigid_contact_point0 = wp.clone(model.rigid_contact_point0)
                model.rigid_contact_point1 = wp.clone(model.rigid_contact_point1)
                model.rigid_contact_offset0 = wp.clone(model.rigid_contact_offset0)
                model.rigid_contact_offset1 = wp.clone(model.rigid_contact_offset1)
                model.rigid_contact_normal = wp.clone(model.rigid_contact_normal)
                model.rigid_contact_thickness = wp.clone(model.rigid_contact_thickness)
                model.rigid_contact_count = wp.zeros_like(model.rigid_contact_count)
                model.rigid_contact_pairwise_counter = wp.zeros_like(model.rigid_contact_pairwise_counter)
                model.rigid_contact_tids = wp.zeros_like(model.rigid_contact_tids)
                model.rigid_contact_shape0 = wp.empty_like(model.rigid_contact_shape0)
                model.rigid_contact_shape1 = wp.empty_like(model.rigid_contact_shape1)
            else:
                model.rigid_contact_count.zero_()
                model.rigid_contact_pairwise_counter.zero_()
                model.rigid_contact_tids.zero_()
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
