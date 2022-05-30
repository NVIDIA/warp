# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""A module for building simulation models and state.
"""

import warp as wp

# todo: copied from integrators_euler.py, need to figure out how to share funcs across modules
@wp.func
def transform_inverse(t: wp.transform):

    p = wp.transform_get_translation(t)
    q = wp.transform_get_rotation(t)

    q_inv = wp.quat_inverse(q)
    return wp.transform(-wp.quat_rotate(q_inv, p), q_inv)


@wp.func
def sphere_sdf(center: wp.vec3, radius: float, p: wp.vec3):

    return wp.length(p-center) - radius

@wp.func
def sphere_sdf_grad(center: wp.vec3, radius: float, p: wp.vec3):

    return wp.normalize(p-center)

@wp.func
def box_sdf(upper: wp.vec3, p: wp.vec3):

    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    qx = abs(p[0])-upper[0]
    qy = abs(p[1])-upper[1]
    qz = abs(p[2])-upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))
    
    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def box_sdf_grad(upper: wp.vec3, p: wp.vec3):

    qx = abs(p[0])-upper[0]
    qy = abs(p[1])-upper[1]
    qz = abs(p[2])-upper[2]

    # exterior case
    if (qx > 0.0 or qy > 0.0 or qz > 0.0):
        
        x = wp.clamp(p[0], 0.0-upper[0], upper[0])
        y = wp.clamp(p[1], 0.0-upper[1], upper[1])
        z = wp.clamp(p[2], 0.0-upper[2], upper[2])

        return wp.normalize(p - wp.vec3(x, y, z))

    sx = wp.sign(p[0])
    sy = wp.sign(p[1])
    sz = wp.sign(p[2])

    # x projection
    if (qx > qy and qx > qz):
        return wp.vec3(sx, 0.0, 0.0)
    
    # y projection
    if (qy > qx and qy > qz):
        return wp.vec3(0.0, sy, 0.0)

    # z projection
    if (qz > qx and qz > qy):
        return wp.vec3(0.0, 0.0, sz)

@wp.func
def capsule_sdf(radius: float, half_width: float, p: wp.vec3):

    if (p[0] > half_width):
        return length(wp.vec3(p[0] - half_width, p[1], p[2])) - radius

    if (p[0] < 0.0 - half_width):
        return length(wp.vec3(p[0] + half_width, p[1], p[2])) - radius

    return wp.length(wp.vec3(0.0, p[1], p[2])) - radius

@wp.func
def capsule_sdf_grad(radius: float, half_width: float, p: wp.vec3):

    if (p[0] > half_width):
        return normalize(wp.vec3(p[0] - half_width, p[1], p[2]))

    if (p[0] < 0.0 - half_width):
        return normalize(wp.vec3(p[0] + half_width, p[1], p[2]))
        
    return normalize(wp.vec3(0.0, p[1], p[2]))




@wp.kernel
def create_soft_contacts(
    num_particles: int,
    particle_x: wp.array(dtype=wp.vec3), 
    body_X_sc: wp.array(dtype=wp.transform),
    shape_X_co: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_geo_type: wp.array(dtype=int), 
    shape_geo_id: wp.array(dtype=wp.uint64),
    shape_geo_scale: wp.array(dtype=wp.vec3),
    soft_contact_margin: float,
    #outputs,
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_body: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    soft_contact_max: int):
    
    tid = wp.tid()           

    shape_index = tid // num_particles     # which shape
    particle_index = tid % num_particles   # which particle
    rigid_index = shape_body[shape_index]

    px = particle_x[particle_index]

    X_sc = wp.transform_identity()
    if (rigid_index >= 0):
        X_sc = body_X_sc[rigid_index]
    
    X_co = shape_X_co[shape_index]

    X_so = wp.transform_multiply(X_sc, X_co)
    X_os = wp.transform_inverse(X_so)
    
    # transform particle position to shape local space
    x_local = wp.transform_point(X_os, px)

    # geo description
    geo_type = shape_geo_type[shape_index]
    geo_scale = shape_geo_scale[shape_index]

   # evaluate shape sdf
    d = 1.e+6 
    n = wp.vec3()
    v = wp.vec3()

    # GEO_SPHERE (0)
    if (geo_type == 0):
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
        n = sphere_sdf_grad(wp.vec3(), geo_scale[0], x_local)

    # GEO_BOX (1)
    if (geo_type == 1):
        d = box_sdf(geo_scale, x_local)
        n = box_sdf_grad(geo_scale, x_local)
        
    # GEO_CAPSULE (2)
    if (geo_type == 2):
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
        n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    # GEO_MESH (3)
    if (geo_type == 3):
        mesh = shape_geo_id[shape_index]

        face_index = int(0)
        face_u = float(0.0)  
        face_v = float(0.0)
        sign = float(0.0)

        if (wp.mesh_query_point(mesh, x_local/geo_scale[0], soft_contact_margin, sign, face_index, face_u, face_v)):

            shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
            shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)

            shape_p = shape_p*geo_scale[0]
            shape_v = shape_v*geo_scale[0]

            delta = x_local-shape_p
            d = wp.length(delta)*sign
            n = wp.normalize(delta)*sign
            v = shape_v


    if (d < soft_contact_margin):

        index = wp.atomic_add(soft_contact_count, 0, 1) 

        if (index < soft_contact_max):

            # compute contact point in body local space
            body_pos = wp.transform_point(X_co, x_local - n*d)
            body_vel = wp.transform_vector(X_co, v)

            world_normal = wp.transform_vector(X_so, n)

            soft_contact_body[index] = rigid_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = body_vel
            soft_contact_particle[index] = particle_index
            soft_contact_normal[index] = world_normal

def collide(model, state):

    # clear old count
    model.soft_contact_count.zero_()
    
    wp.launch(
        kernel=create_soft_contacts,
        dim=model.particle_count*model.shape_count,
        inputs=[
            model.particle_count,
            state.particle_q, 
            state.body_q,
            model.shape_transform,
            model.shape_body,
            model.shape_geo_type, 
            model.shape_geo_id,
            model.shape_geo_scale,
            model.soft_contact_margin,
            model.soft_contact_count,
            model.soft_contact_particle,
            model.soft_contact_body,
            model.soft_contact_body_pos,
            model.soft_contact_body_vel,
            model.soft_contact_normal,
            model.soft_contact_max],
            # outputs
        outputs=[],
        device=model.device)