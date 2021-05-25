"""A module for building simulation models and state.
"""

import oglang as og



# todo: copied from integratos_euler.py, need to figure out how to share funcs across modules
@og.func
def spatial_transform_inverse(t: og.spatial_transform):

    p = spatial_transform_get_translation(t)
    q = spatial_transform_get_rotation(t)

    q_inv = inverse(q)
    return spatial_transform(rotate(q_inv, p)*(0.0 - 1.0), q_inv)


@og.func
def sphere_sdf(center: og.vec3, radius: float, p: og.vec3):

    return og.length(p-center) - radius

@og.func
def sphere_sdf_grad(center: og.vec3, radius: float, p: og.vec3):

    return og.normalize(p-center)

@og.func
def box_sdf(upper: og.vec3, p: og.vec3):

    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    qx = abs(p[0])-upper[0]
    qy = abs(p[1])-upper[1]
    qz = abs(p[2])-upper[2]

    e = og.vec3(og.max(qx, 0.0), og.max(qy, 0.0), og.max(qz, 0.0))
    
    return og.length(e) + og.min(og.max(qx, og.max(qy, qz)), 0.0)


@og.func
def box_sdf_grad(upper: og.vec3, p: og.vec3):

    qx = abs(p[0])-upper[0]
    qy = abs(p[1])-upper[1]
    qz = abs(p[2])-upper[2]

    # exterior case
    if (qx > 0.0 or qy > 0.0 or qz > 0.0):
        
        x = og.clamp(p[0], 0.0-upper[0], upper[0])
        y = og.clamp(p[1], 0.0-upper[1], upper[1])
        z = og.clamp(p[2], 0.0-upper[2], upper[2])

        return og.normalize(p - og.vec3(x, y, z))

    sx = og.sign(p[0])
    sy = og.sign(p[1])
    sz = og.sign(p[2])

    # x projection
    if (qx > qy and qx > qz):
        return og.vec3(sx, 0.0, 0.0)
    
    # y projection
    if (qy > qx and qy > qz):
        return og.vec3(0.0, sy, 0.0)

    # z projection
    if (qz > qx and qz > qy):
        return og.vec3(0.0, 0.0, sz)

@og.func
def capsule_sdf(radius: float, half_width: float, p: og.vec3):

    if (p[0] > half_width):
        return length(og.vec3(p[0] - half_width, p[1], p[2])) - radius

    if (p[0] < 0.0 - half_width):
        return length(og.vec3(p[0] + half_width, p[1], p[2])) - radius

    return og.length(og.vec3(0.0, p[1], p[2])) - radius

@og.func
def capsule_sdf_grad(radius: float, half_width: float, p: og.vec3):

    if (p[0] > half_width):
        return normalize(og.vec3(p[0] - half_width, p[1], p[2]))

    if (p[0] < 0.0 - half_width):
        return normalize(og.vec3(p[0] + half_width, p[1], p[2]))
        
    return normalize(og.vec3(0.0, p[1], p[2]))




@og.kernel
def create_soft_contacts(
    num_particles: int,
    particle_x: og.array(dtype=og.vec3), 
    body_X_sc: og.array(dtype=og.spatial_transform),
    shape_X_co: og.array(dtype=og.spatial_transform),
    shape_body: og.array(dtype=int),
    shape_geo_type: og.array(dtype=int), 
    shape_geo_id: og.array(dtype=og.uint64),
    shape_geo_scale: og.array(dtype=og.vec3),
    soft_contact_margin: float,
    #outputs,
    soft_contact_count: og.array(dtype=int),
    soft_contact_particle: og.array(dtype=int),
    soft_contact_body: og.array(dtype=int),
    soft_contact_body_pos: og.array(dtype=og.vec3),
    soft_contact_body_vel: og.array(dtype=og.vec3),
    soft_contact_normal: og.array(dtype=og.vec3),
    soft_contact_max: int):
    
    tid = og.tid()           

    shape_index = tid // num_particles     # which shape
    particle_index = tid % num_particles   # which particle
    rigid_index = og.load(shape_body, shape_index)    

    px = og.load(particle_x, particle_index)

    X_sc = og.spatial_transform_identity()
    if (rigid_index >= 0):
        X_sc = og.load(body_X_sc, rigid_index)
    
    X_co = og.load(shape_X_co, shape_index)

    X_so = og.spatial_transform_multiply(X_sc, X_co)
    X_os = og.spatial_transform_inverse(X_so)
    
    # transform particle position to shape local space
    x_local = og.spatial_transform_point(X_os, px)

    # geo description
    geo_type = og.load(shape_geo_type, shape_index)
    geo_scale = og.load(shape_geo_scale, shape_index)

   # evaluate shape sdf
    d = 1.e+6 
    n = og.vec3()
    v = og.vec3()

    # GEO_SPHERE (0)
    if (geo_type == 0):
        d = sphere_sdf(og.vec3(0.0, 0.0, 0.0), geo_scale[0], x_local)
        n = sphere_sdf_grad(og.vec3(0.0, 0.0, 0.0), geo_scale[0], x_local)

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
        mesh = og.load(shape_geo_id, shape_index)

        face_index = int(0)
        face_v = float(0.0)  
        face_w = float(0.0)
        sign = float(0.0)

        if (og.mesh_query_point(mesh, x_local/geo_scale[0], soft_contact_margin, sign, face_index, face_v, face_w)):

            shape_p = og.mesh_eval_position(mesh, face_index, face_v, face_w)
            shape_v = og.mesh_eval_velocity(mesh, face_index, face_v, face_w)

            shape_p = shape_p*geo_scale[0]

            delta = x_local-shape_p
            d = og.length(delta)*sign
            n = og.normalize(delta)*sign
            v = shape_v


    if (d < soft_contact_margin):

        index = og.atomic_add(soft_contact_count, 0, 1) # index is zero, adding 1

        if (index < soft_contact_max):

            # compute contact point in body local space
            body_pos = og.spatial_transform_point(X_co, x_local - n*d)
            body_vel = og.spatial_transform_vector(X_co, v)

            world_normal = og.spatial_transform_vector(X_so, n)

            soft_contact_body[index] = rigid_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = body_vel
            soft_contact_particle[index] = particle_index
            soft_contact_normal[index] = world_normal


def collide(model, state):

    # clear old count
    model.soft_contact_count.zero_()
    
    og.launch(
        kernel=create_soft_contacts,
        dim=model.particle_count*model.shape_count,
        inputs=[
            model.particle_count,
            state.particle_q, 
            state.body_X_sc,
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