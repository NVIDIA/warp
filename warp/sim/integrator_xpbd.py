# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
from .model import ModelShapeMaterials, JOINT_MODE_TARGET_POSITION, JOINT_MODE_TARGET_VELOCITY, JOINT_MODE_LIMIT
from .utils import velocity_at_point, vec_min, vec_max, vec_abs
from .integrator_euler import integrate_bodies, integrate_particles


@wp.kernel
def solve_particle_ground_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    offset: float,
    ground: wp.array(dtype=float),
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    wi = invmass[tid]
    if wi == 0.0:
        return

    x = particle_x[tid]
    v = particle_v[tid]

    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.min(wp.dot(n, x) + ground[3] - offset, 0.0)

    if c > 0.0:
        return

    # normal
    lambda_n = c
    delta_n = n * lambda_n

    # friction
    vn = wp.dot(n, v)
    vt = v - n * vn

    lambda_f = wp.max(mu * lambda_n, 0.0 - wp.length(vt) * dt)
    delta_f = wp.normalize(vt) * lambda_f

    wp.atomic_add(delta, tid, (delta_f - delta_n) / wi * relaxation)


@wp.kernel
def apply_soft_restitution_ground(
    particle_x_new: wp.array(dtype=wp.vec3),
    particle_v_new: wp.array(dtype=wp.vec3),
    particle_x_old: wp.array(dtype=wp.vec3),
    particle_v_old: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    restitution: float,
    offset: float,
    ground: wp.array(dtype=float),
    dt: float,
    relaxation: float,
    particle_v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    wi = invmass[tid]
    if wi == 0.0:
        return

    x_new = particle_x_new[tid]
    v_new = particle_v_new[tid]
    x_old = particle_x_old[tid]
    v_old = particle_v_old[tid]

    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.dot(n, x_old) + ground[3] - offset

    if c > 0.0:
        return

    rel_vel_old = wp.dot(n, v_old)
    rel_vel_new = wp.dot(n, v_new)
    dv = n * wp.max(-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0), 0.0)

    wp.atomic_add(particle_v_out, tid, dv / wi * relaxation)


@wp.kernel
def solve_soft_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_distance: float,
    contact_max: int,
    dt: float,
    relaxation: float,
    # outputs
    delta: wp.array(dtype=wp.vec3),
    body_delta: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - contact_distance

    if c > particle_ka:
        return

    # take average material properties of shape and particle parameters
    mu = 0.5 * (particle_mu + shape_materials.mu[shape_index])

    # body velocity
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]

    body_w = wp.spatial_top(body_v_s)
    body_v = wp.spatial_bottom(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])

    # relative velocity
    v = pv - bv

    # normal
    lambda_n = c
    delta_n = n * lambda_n

    # friction
    vn = wp.dot(n, v)
    vt = v - n * vn

    # compute inverse masses
    w1 = particle_invmass[particle_index]
    w2 = 0.0
    if body_index >= 0:
        angular = wp.cross(r, n)
        q = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q, angular)
        I_inv = body_I_inv[body_index]
        w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
    denom = w1 + w2
    if denom == 0.0:
        return

    lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
    delta_f = wp.normalize(vt) * lambda_f
    delta_total = (delta_f - delta_n) / denom * relaxation

    wp.atomic_add(delta, particle_index, delta_total)

    if body_index >= 0:
        delta_t = wp.cross(r, delta_total)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_t, delta_total))


@wp.kernel
def solve_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    dt: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = wp.dot(dir, vij)

    # damping based on relative velocity.
    # fs = dir * (ke * c + kd * dcdt)

    wi = invmass[i]
    wj = invmass[j]

    denom = wi + wj
    alpha = 1.0 / (ke * dt * dt)

    multiplier = c / (denom)  # + alpha)

    xd = dir * multiplier

    wp.atomic_sub(delta, i, xd * wi)
    wp.atomic_add(delta, j, xd * wj)


@wp.kernel
def solve_tetrahedra(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    indices: wp.array(dtype=int, ndim=2),
    pose: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    v0 = v[i]
    v1 = v[j]
    v2 = v[k]
    v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = wp.mat33(x10, x20, x30)
    Dm = pose[tid]

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # C_sqrt
    # tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)
    # r_s = wp.sqrt(abs(tr - 3.0))
    # C = r_s

    # if (r_s == 0.0):
    #     return

    # if (tr < 3.0):
    #     r_s = 0.0 - r_s

    # dCdx = F*wp.transpose(Dm)*(1.0/r_s)
    # alpha = 1.0 + k_mu / k_lambda

    # C_Neo
    r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    if r_s == 0.0:
        return
    # tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)
    # if (tr < 3.0):
    #     r_s = -r_s
    r_s_inv = 1.0 / r_s
    C = r_s
    dCdx = F * wp.transpose(Dm) * r_s_inv
    alpha = 1.0 + k_mu / k_lambda

    # C_Spherical
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # r_s_inv = 1.0/r_s
    # C = r_s - wp.sqrt(3.0)
    # dCdx = F*wp.transpose(Dm)*r_s_inv
    # alpha = 1.0

    # C_D
    # r_s = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
    # C = r_s*r_s - 3.0
    # dCdx = F*wp.transpose(Dm)*2.0
    # alpha = 1.0

    grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    grad0 = (grad1 + grad2 + grad3) * (0.0 - 1.0)

    denom = (
        wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    )
    multiplier = C / (denom + 1.0 / (k_mu * dt * dt * rest_volume))

    delta0 = grad0 * multiplier
    delta1 = grad1 * multiplier
    delta2 = grad2 * multiplier
    delta3 = grad3 * multiplier

    # hydrostatic part
    J = wp.determinant(F)

    C_vol = J - alpha
    # dCdx = wp.mat33(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))*wp.transpose(Dm)

    # grad1 = wp.vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = wp.vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = wp.vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    s = inv_rest_volume / 6.0
    grad1 = wp.cross(x20, x30) * s
    grad2 = wp.cross(x30, x10) * s
    grad3 = wp.cross(x10, x20) * s
    grad0 = -(grad1 + grad2 + grad3)

    denom = (
        wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    )
    multiplier = C_vol / (denom + 1.0 / (k_lambda * dt * dt * rest_volume))

    delta0 += grad0 * multiplier
    delta1 += grad1 * multiplier
    delta2 += grad2 * multiplier
    delta3 += grad3 * multiplier

    # apply forces
    wp.atomic_sub(delta, i, delta0 * w0 * relaxation)
    wp.atomic_sub(delta, j, delta1 * w1 * relaxation)
    wp.atomic_sub(delta, k, delta2 * w2 * relaxation)
    wp.atomic_sub(delta, l, delta3 * w3 * relaxation)


@wp.kernel
def apply_deltas(
    x_orig: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    delta: wp.array(dtype=wp.vec3),
    dt: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x_orig[tid]
    xp = x_pred[tid]

    # constraint deltas
    d = delta[tid]

    x_new = xp + d
    v_new = (x_new - x0) / dt

    x_out[tid] = x_new
    v_out[tid] = v_new


@wp.kernel
def apply_body_deltas(
    q_in: wp.array(dtype=wp.transform),
    qd_in: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_I: wp.array(dtype=wp.mat33),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),
    deltas: wp.array(dtype=wp.spatial_vector),
    constraint_inv_weights: wp.array(dtype=float),
    dt: float,
    # outputs
    q_out: wp.array(dtype=wp.transform),
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    inv_m = body_inv_m[tid]
    if inv_m == 0.0:
        q_out[tid] = q_in[tid]
        qd_out[tid] = qd_in[tid]
        return
    inv_I = body_inv_I[tid]

    tf = q_in[tid]
    delta = deltas[tid]

    p0 = wp.transform_get_translation(tf)
    q0 = wp.transform_get_rotation(tf)

    weight = 1.0
    if constraint_inv_weights:
        if constraint_inv_weights[tid] > 0.0:
            weight = 1.0 / constraint_inv_weights[tid]

    dp = wp.spatial_bottom(delta) * (inv_m * weight)
    dq = wp.spatial_top(delta) * weight
    dq = wp.quat_rotate(q0, inv_I * wp.quat_rotate_inv(q0, dq))

    # update orientation
    q1 = q0 + 0.5 * wp.quat(dq * dt * dt, 0.0) * q0
    q1 = wp.normalize(q1)

    # update position
    com = body_com[tid]
    x_com = p0 + wp.quat_rotate(q0, com)
    p1 = x_com + dp * dt * dt
    p1 -= wp.quat_rotate(q1, com)

    q_out[tid] = wp.transform(p1, q1)

    v0 = wp.spatial_bottom(qd_in[tid])
    w0 = wp.spatial_top(qd_in[tid])

    # update linear and angular velocity
    v1 = v0 + dp * dt
    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(q0, w0 + dq * dt)
    tb = -wp.cross(wb, body_I[tid] * wb)  # coriolis forces
    w1 = wp.quat_rotate(q0, wb + inv_I * tb * dt)

    qd_out[tid] = wp.spatial_vector(w1, v1)


@wp.kernel
def apply_body_delta_velocities(
    qd_in: wp.array(dtype=wp.spatial_vector),
    deltas: wp.array(dtype=wp.spatial_vector),
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    qd_out[tid] = qd_in[tid] + deltas[tid]


@wp.kernel
def apply_joint_torques(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_act: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    type = joint_type[tid]
    if type == wp.sim.JOINT_FIXED:
        return
    if type == wp.sim.JOINT_FREE:
        return
    if type == wp.sim.JOINT_DISTANCE:
        return
    if type == wp.sim.JOINT_BALL:
        return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c
    com_c = body_com[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)

    # local joint rotations
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # joint properties (for 1D joints)
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    axis_start = joint_axis_start[tid]
    lin_axis_count = joint_axis_dim[tid, 0]
    ang_axis_count = joint_axis_dim[tid, 1]

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    # handle angular constraints
    if type == wp.sim.JOINT_REVOLUTE:
        axis = joint_axis[axis_start]
        act = joint_act[qd_start]
        a_p = wp.transform_vector(X_wp, axis)
        t_total += act * a_p
    elif type == wp.sim.JOINT_PRISMATIC:
        axis = joint_axis[axis_start]
        act = joint_act[qd_start]
        a_p = wp.transform_vector(X_wp, axis)
        f_total += act * a_p
    elif type == wp.sim.JOINT_COMPOUND:
        # q_off = wp.transform_get_rotation(X_cj)
        # q_pc = wp.quat_inverse(q_off)*wp.quat_inverse(q_p)*q_c*q_off
        # # decompose to a compound rotation each axis
        # angles = quat_decompose(q_pc)

        # # reconstruct rotation axes
        # axis_0 = wp.vec3(1.0, 0.0, 0.0)
        # q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        # axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        # q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        # axis_2 = wp.quat_rotate(q_1*q_0, wp.vec3(0.0, 0.0, 1.0))

        # q_w = q_p*q_off
        # t_total += joint_act[qd_start+0] * wp.quat_rotate(q_w, axis_0)
        # t_total += joint_act[qd_start+1] * wp.quat_rotate(q_w, axis_1)
        # t_total += joint_act[qd_start+2] * wp.quat_rotate(q_w, axis_2)

        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        axis_2 = joint_axis[axis_start + 2]
        t_total += joint_act[qd_start + 0] * wp.transform_vector(X_wp, axis_0)
        t_total += joint_act[qd_start + 1] * wp.transform_vector(X_wp, axis_1)
        t_total += joint_act[qd_start + 2] * wp.transform_vector(X_wp, axis_2)

    elif type == wp.sim.JOINT_UNIVERSAL:
        # q_off = wp.transform_get_rotation(X_cj)
        # q_pc = wp.quat_inverse(q_off)*wp.quat_inverse(q_p)*q_c*q_off

        # # decompose to a compound rotation each axis
        # angles = quat_decompose(q_pc)

        # # reconstruct rotation axes
        # axis_0 = wp.vec3(1.0, 0.0, 0.0)
        # q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        # axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        # q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        # axis_2 = wp.quat_rotate(q_1*q_0, wp.vec3(0.0, 0.0, 1.0))

        # q_w = q_p*q_off

        # free axes
        # t_total += joint_act[qd_start+0] * wp.quat_rotate(q_w, axis_0)
        # t_total += joint_act[qd_start+1] * wp.quat_rotate(q_w, axis_1)

        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        t_total += joint_act[qd_start + 0] * wp.transform_vector(X_wp, axis_0)
        t_total += joint_act[qd_start + 1] * wp.transform_vector(X_wp, axis_1)

    elif type == wp.sim.JOINT_D6:
        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a dynamic for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            act = joint_act[qd_start + 0]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += act * a_p
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            act = joint_act[qd_start + 1]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += act * a_p
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            act = joint_act[qd_start + 2]
            a_p = wp.transform_vector(X_wp, axis)
            f_total += act * a_p

        if ang_axis_count > 0:
            axis = joint_axis[axis_start + lin_axis_count + 0]
            act = joint_act[qd_start + lin_axis_count + 0]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += act * a_p
        if ang_axis_count > 1:
            axis = joint_axis[axis_start + lin_axis_count + 1]
            act = joint_act[qd_start + lin_axis_count + 1]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += act * a_p
        if ang_axis_count > 2:
            axis = joint_axis[axis_start + lin_axis_count + 2]
            act = joint_act[qd_start + lin_axis_count + 2]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += act * a_p

    else:
        print("joint type not handled in apply_joint_torques")

    # write forces
    if id_p >= 0:
        wp.atomic_add(body_f, id_p, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total))
    wp.atomic_sub(body_f, id_c, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))


@wp.func
def update_joint_axis_mode(mode: wp.uint8, axis: wp.vec3, input_axis_mode: wp.vec3ub):
    # update the 3D axis mode flags given the axis vector and mode of this axis
    mode_x = wp.max(wp.uint8(wp.nonzero(axis[0])) * mode, input_axis_mode[0])
    mode_y = wp.max(wp.uint8(wp.nonzero(axis[1])) * mode, input_axis_mode[1])
    mode_z = wp.max(wp.uint8(wp.nonzero(axis[2])) * mode, input_axis_mode[2])
    return wp.vec3ub(mode_x, mode_y, mode_z)


@wp.func
def update_joint_axis_limits(axis: wp.vec3, limit_lower: float, limit_upper: float, input_limits: wp.spatial_vector):
    # update the 3D linear/angular limits (spatial_vector [lower, upper]) given the axis vector and limits
    lo_temp = axis * limit_lower
    up_temp = axis * limit_upper
    lo = vec_min(lo_temp, up_temp)
    up = vec_max(lo_temp, up_temp)
    input_lower = wp.spatial_top(input_limits)
    input_upper = wp.spatial_bottom(input_limits)
    lower = vec_min(input_lower, lo)
    upper = vec_max(input_upper, up)
    return wp.spatial_vector(lower, upper)


@wp.func
def update_joint_axis_target_ke_kd(
    axis: wp.vec3, target: float, target_ke: float, target_kd: float, input_target_ke_kd: wp.mat33
):
    # update the 3D linear/angular target, target_ke, and target_kd (mat33 [target, ke, kd]) given the axis vector and target, target_ke, target_kd
    axis_target = input_target_ke_kd[0]
    axis_ke = input_target_ke_kd[1]
    axis_kd = input_target_ke_kd[2]
    stiffness = axis * target_ke
    axis_target += stiffness * target  # weighted target (to be normalized later by sum of target_ke)
    axis_ke += vec_abs(stiffness)
    axis_kd += vec_abs(axis * target_kd)
    return wp.mat33(
        axis_target[0],
        axis_target[1],
        axis_target[2],
        axis_ke[0],
        axis_ke[1],
        axis_ke[2],
        axis_kd[0],
        axis_kd[1],
        axis_kd[2],
    )


@wp.kernel
def solve_body_joints(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_axis_mode: wp.array(dtype=wp.uint8),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_target: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_linear_compliance: wp.array(dtype=float),
    joint_angular_compliance: wp.array(dtype=float),
    angular_relaxation: float,
    linear_relaxation: float,
    dt: float,
    deltas: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    type = joint_type[tid]

    if joint_enabled[tid] == 0 or type == wp.sim.JOINT_FREE:
        return

    # rigid body indices of the child and parent
    id_c = joint_child[tid]
    id_p = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    m_inv_p = 0.0
    I_inv_p = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pose_p = X_pj
    com_p = wp.vec3(0.0)
    vel_p = wp.vec3(0.0)
    omega_p = wp.vec3(0.0)
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
        m_inv_p = body_inv_m[id_p]
        I_inv_p = body_inv_I[id_p]
        vel_p = wp.spatial_bottom(body_qd[id_p])
        omega_p = wp.spatial_top(body_qd[id_p])

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    com_c = body_com[id_c]
    m_inv_c = body_inv_m[id_c]
    I_inv_c = body_inv_I[id_c]
    vel_c = wp.spatial_bottom(body_qd[id_c])
    omega_c = wp.spatial_top(body_qd[id_c])

    if m_inv_p == 0.0 and m_inv_c == 0.0:
        # connection between two immovable bodies
        return

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    rel_pose = wp.transform_inverse(X_wp) * X_wc
    rel_p = wp.transform_get_translation(rel_pose)

    # joint connection points
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    linear_compliance = joint_linear_compliance[tid]
    angular_compliance = joint_angular_compliance[tid]

    axis_start = joint_axis_start[tid]
    lin_axis_count = joint_axis_dim[tid, 0]
    ang_axis_count = joint_axis_dim[tid, 1]

    world_com_p = wp.transform_point(pose_p, com_p)
    world_com_c = wp.transform_point(pose_c, com_c)

    # handle positional constraints
    if type == wp.sim.JOINT_DISTANCE:
        r_p = wp.transform_get_translation(X_wp) - world_com_p
        r_c = wp.transform_get_translation(X_wc) - world_com_c
        lower = joint_limit_lower[axis_start]
        upper = joint_limit_upper[axis_start]
        if lower < 0.0 and upper < 0.0:
            # no limits
            return
        d = wp.length(rel_p)
        err = 0.0
        if lower >= 0.0 and d < lower:
            err = d - lower
            # use a more descriptive direction vector for the constraint
            # in case the joint parent and child anchors are very close
            rel_p = err * wp.normalize(world_com_c - world_com_p)
        elif upper >= 0.0 and d > upper:
            err = d - upper

        if wp.abs(err) > 1e-9:
            # compute gradients
            linear_c = rel_p
            linear_p = -linear_c
            r_c = x_c - world_com_c
            angular_p = -wp.cross(r_p, linear_c)
            angular_c = wp.cross(r_c, linear_c)
            # constraint time derivative
            derr = (
                wp.dot(linear_p, vel_p)
                + wp.dot(linear_c, vel_c)
                + wp.dot(angular_p, omega_p)
                + wp.dot(angular_c, omega_c)
            )
            lambda_in = 0.0
            compliance = linear_compliance
            ke = joint_target_ke[axis_start]
            if ke > 0.0:
                compliance = 1.0 / ke
            damping = joint_target_kd[axis_start]
            d_lambda = compute_positional_correction(
                err,
                derr,
                pose_p,
                pose_c,
                m_inv_p,
                m_inv_c,
                I_inv_p,
                I_inv_c,
                linear_p,
                linear_c,
                angular_p,
                angular_c,
                lambda_in,
                compliance,
                damping,
                dt,
            )

            lin_delta_p += linear_p * (d_lambda * linear_relaxation)
            ang_delta_p += angular_p * (d_lambda * angular_relaxation)
            lin_delta_c += linear_c * (d_lambda * linear_relaxation)
            ang_delta_c += angular_c * (d_lambda * angular_relaxation)

    else:
        # compute joint target, stiffness, damping
        ke_sum = float(0.0)
        axis_limits = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        axis_mode = wp.vec3ub(wp.uint8(0), wp.uint8(0), wp.uint8(0))
        axis_target_ke_kd = wp.mat33(0.0)
        # avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if lin_axis_count > 0:
            axis = joint_axis[axis_start]
            lo_temp = axis * joint_limit_lower[axis_start]
            up_temp = axis * joint_limit_upper[axis_start]
            axis_limits = wp.spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp))
            mode = joint_axis_mode[axis_start]
            if mode != JOINT_MODE_LIMIT:  # position or velocity target
                ke = joint_target_ke[axis_start]
                kd = joint_target_kd[axis_start]
                target = joint_target[axis_start]
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode)
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd)
                ke_sum += ke
        if lin_axis_count > 1:
            axis_idx = axis_start + 1
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            mode = joint_axis_mode[axis_idx]
            if mode != JOINT_MODE_LIMIT:  # position or velocity target
                ke = joint_target_ke[axis_idx]
                kd = joint_target_kd[axis_idx]
                target = joint_target[axis_idx]
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode)
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd)
                ke_sum += ke
        if lin_axis_count > 2:
            axis_idx = axis_start + 2
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            mode = joint_axis_mode[axis_idx]
            if mode != JOINT_MODE_LIMIT:  # position or velocity target
                ke = joint_target_ke[axis_idx]
                kd = joint_target_kd[axis_idx]
                target = joint_target[axis_idx]
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode)
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd)
                ke_sum += ke

        axis_target = axis_target_ke_kd[0]
        axis_stiffness = axis_target_ke_kd[1]
        axis_damping = axis_target_ke_kd[2]
        if ke_sum > 0.0:
            axis_target /= ke_sum
        axis_limits_lower = wp.spatial_top(axis_limits)
        axis_limits_upper = wp.spatial_bottom(axis_limits)

        frame_p = wp.quat_to_matrix(wp.transform_get_rotation(X_wp))
        # note that x_c appearing in both is correct
        r_p = x_c - world_com_p
        r_c = x_c - wp.transform_point(pose_c, com_c)

        # for loop will be unrolled, so we can modify local variables
        for dim in range(3):
            e = rel_p[dim]
            mode = axis_mode[dim]

            # compute gradients
            linear_c = wp.vec3(frame_p[0, dim], frame_p[1, dim], frame_p[2, dim])
            linear_p = -linear_c
            angular_p = -wp.cross(r_p, linear_c)
            angular_c = wp.cross(r_c, linear_c)
            # constraint time derivative
            derr = (
                wp.dot(linear_p, vel_p)
                + wp.dot(linear_c, vel_c)
                + wp.dot(angular_p, omega_p)
                + wp.dot(angular_c, omega_c)
            )

            err = 0.0
            compliance = linear_compliance
            damping = 0.0
            # consider joint limits irrespective of axis mode
            lower = axis_limits_lower[dim]
            upper = axis_limits_upper[dim]
            if e < lower:
                err = e - lower
                compliance = linear_compliance
                damping = 0.0
            elif e > upper:
                err = e - upper
                compliance = linear_compliance
                damping = 0.0
            else:
                target = wp.clamp(axis_target[dim], lower, upper)
                if mode == JOINT_MODE_TARGET_POSITION:
                    if axis_stiffness[dim] > 0.0:
                        err = e - target
                        compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                elif mode == JOINT_MODE_TARGET_VELOCITY:
                    if axis_stiffness[dim] > 0.0:
                        err = (derr - target) * dt
                        compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]

            if wp.abs(err) > 1e-9:
                lambda_in = 0.0
                d_lambda = compute_positional_correction(
                    err,
                    derr,
                    pose_p,
                    pose_c,
                    m_inv_p,
                    m_inv_c,
                    I_inv_p,
                    I_inv_c,
                    linear_p,
                    linear_c,
                    angular_p,
                    angular_c,
                    lambda_in,
                    compliance,
                    damping,
                    dt,
                )

                lin_delta_p += linear_p * (d_lambda * linear_relaxation)
                ang_delta_p += angular_p * (d_lambda * angular_relaxation)
                lin_delta_c += linear_c * (d_lambda * linear_relaxation)
                ang_delta_c += angular_c * (d_lambda * angular_relaxation)

    if (
        type == wp.sim.JOINT_FIXED
        or type == wp.sim.JOINT_PRISMATIC
        or type == wp.sim.JOINT_REVOLUTE
        or type == wp.sim.JOINT_UNIVERSAL
        or type == wp.sim.JOINT_COMPOUND
        or type == wp.sim.JOINT_D6
    ):
        # handle angular constraints

        # local joint rotations
        q_p = wp.transform_get_rotation(X_wp)
        q_c = wp.transform_get_rotation(X_wc)

        # make quats lie in same hemisphere
        if wp.dot(q_p, q_c) < 0.0:
            q_c *= -1.0

        rel_q = wp.quat_inverse(q_p) * q_c

        qtwist = wp.normalize(wp.quat(rel_q[0], 0.0, 0.0, rel_q[3]))
        qswing = rel_q * wp.quat_inverse(qtwist)

        # decompose to a compound rotation each axis
        s = wp.sqrt(rel_q[0] * rel_q[0] + rel_q[3] * rel_q[3])
        invs = 1.0 / s
        invscube = invs * invs * invs

        # handle axis-angle joints

        # rescale twist from quaternion space to angular
        err_0 = 2.0 * wp.asin(wp.clamp(qtwist[0], -1.0, 1.0))
        err_1 = qswing[1]
        err_2 = qswing[2]
        # analytic gradients of swing-twist decomposition
        grad_0 = wp.quat(invs - rel_q[0] * rel_q[0] * invscube, 0.0, 0.0, -(rel_q[3] * rel_q[0]) * invscube)
        grad_1 = wp.quat(
            -rel_q[3] * (rel_q[3] * rel_q[2] + rel_q[0] * rel_q[1]) * invscube,
            rel_q[3] * invs,
            -rel_q[0] * invs,
            rel_q[0] * (rel_q[3] * rel_q[2] + rel_q[0] * rel_q[1]) * invscube,
        )
        grad_2 = wp.quat(
            rel_q[3] * (rel_q[3] * rel_q[1] - rel_q[0] * rel_q[2]) * invscube,
            rel_q[0] * invs,
            rel_q[3] * invs,
            rel_q[0] * (rel_q[2] * rel_q[0] - rel_q[3] * rel_q[1]) * invscube,
        )
        grad_0 *= 2.0 / wp.abs(qtwist[3])
        # grad_0 *= 2.0 / wp.sqrt(1.0-qtwist[0]*qtwist[0])	# derivative of asin(x) = 1/sqrt(1-x^2)

        # rescale swing
        swing_sq = qswing[3] * qswing[3]
        # if swing axis magnitude close to zero vector, just treat in quaternion space
        angularEps = 1.0e-4
        if swing_sq + angularEps < 1.0:
            d = wp.sqrt(1.0 - qswing[3] * qswing[3])
            theta = 2.0 * wp.acos(wp.clamp(qswing[3], -1.0, 1.0))
            scale = theta / d

            err_1 *= scale
            err_2 *= scale

            grad_1 *= scale
            grad_2 *= scale

        errs = wp.vec3(err_0, err_1, err_2)
        grad_x = wp.vec3(grad_0[0], grad_1[0], grad_2[0])
        grad_y = wp.vec3(grad_0[1], grad_1[1], grad_2[1])
        grad_z = wp.vec3(grad_0[2], grad_1[2], grad_2[2])
        grad_w = wp.vec3(grad_0[3], grad_1[3], grad_2[3])

        # compute joint target, stiffness, damping
        ke_sum = float(0.0)
        axis_limits = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        axis_mode = wp.vec3ub(wp.uint8(0), wp.uint8(0), wp.uint8(0))
        axis_target_ke_kd = wp.mat33(0.0)
        # avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if ang_axis_count > 0:
            axis_idx = axis_start + lin_axis_count
            axis = joint_axis[axis_idx]
            lo_temp = axis * joint_limit_lower[axis_idx]
            up_temp = axis * joint_limit_upper[axis_idx]
            axis_limits = wp.spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp))
            mode = joint_axis_mode[axis_idx]
            if mode != JOINT_MODE_LIMIT:  # position or velocity target
                ke = joint_target_ke[axis_idx]
                kd = joint_target_kd[axis_idx]
                target = joint_target[axis_idx]
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode)
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd)
                ke_sum += ke
        if ang_axis_count > 1:
            axis_idx = axis_start + lin_axis_count + 1
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            mode = joint_axis_mode[axis_idx]
            if mode != JOINT_MODE_LIMIT:  # position or velocity target
                ke = joint_target_ke[axis_idx]
                kd = joint_target_kd[axis_idx]
                target = joint_target[axis_idx]
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode)
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd)
                ke_sum += ke
        if ang_axis_count > 2:
            axis_idx = axis_start + lin_axis_count + 2
            axis = joint_axis[axis_idx]
            lower = joint_limit_lower[axis_idx]
            upper = joint_limit_upper[axis_idx]
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits)
            mode = joint_axis_mode[axis_idx]
            if mode != JOINT_MODE_LIMIT:  # position or velocity target
                ke = joint_target_ke[axis_idx]
                kd = joint_target_kd[axis_idx]
                target = joint_target[axis_idx]
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode)
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd)
                ke_sum += ke

        axis_target = axis_target_ke_kd[0]
        axis_stiffness = axis_target_ke_kd[1]
        axis_damping = axis_target_ke_kd[2]
        if ke_sum > 0.0:
            axis_target /= ke_sum
        axis_limits_lower = wp.spatial_top(axis_limits)
        axis_limits_upper = wp.spatial_bottom(axis_limits)

        for dim in range(3):
            e = errs[dim]
            mode = axis_mode[dim]

            # analytic gradients of swing-twist decomposition
            grad = wp.quat(grad_x[dim], grad_y[dim], grad_z[dim], grad_w[dim])

            quat_c = 0.5 * q_p * grad * wp.quat_inverse(q_c)
            angular_c = wp.vec3(quat_c[0], quat_c[1], quat_c[2])
            angular_p = -angular_c
            # time derivative of the constraint
            derr = wp.dot(angular_p, omega_p) + wp.dot(angular_c, omega_c)

            err = 0.0
            compliance = angular_compliance
            damping = 0.0

            # consider joint limits irrespective of mode
            lower = axis_limits_lower[dim]
            upper = axis_limits_upper[dim]
            if e < lower:
                err = e - lower
                compliance = angular_compliance
                damping = 0.0
            elif e > upper:
                err = e - upper
                compliance = angular_compliance
                damping = 0.0
            else:
                target = wp.clamp(axis_target[dim], lower, upper)
                if mode == JOINT_MODE_TARGET_POSITION:
                    if axis_stiffness[dim] > 0.0:
                        err = e - target
                        compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                elif mode == JOINT_MODE_TARGET_VELOCITY:
                    if axis_stiffness[dim] > 0.0:
                        err = (derr - target) * dt
                        compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]

            d_lambda = (
                compute_angular_correction(
                    err, derr, pose_p, pose_c, I_inv_p, I_inv_c, angular_p, angular_c, 0.0, compliance, damping, dt
                )
                * angular_relaxation
            )
            # update deltas
            ang_delta_p += angular_p * d_lambda
            ang_delta_c += angular_c * d_lambda

    if id_p >= 0:
        wp.atomic_add(deltas, id_p, wp.spatial_vector(ang_delta_p, lin_delta_p))
    if id_c >= 0:
        wp.atomic_add(deltas, id_c, wp.spatial_vector(ang_delta_c, lin_delta_c))


@wp.func
def compute_contact_constraint_delta(
    err: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    m_inv_a: float,
    m_inv_b: float,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    linear_a: wp.vec3,
    linear_b: wp.vec3,
    angular_a: wp.vec3,
    angular_b: wp.vec3,
    relaxation: float,
    dt: float,
) -> float:
    denom = 0.0
    denom += wp.length_sq(linear_a) * m_inv_a
    denom += wp.length_sq(linear_b) * m_inv_b

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    deltaLambda = -err
    if denom > 0.0:
        deltaLambda /= dt * dt * denom

    return deltaLambda * relaxation


@wp.func
def compute_positional_correction(
    err: float,
    derr: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    m_inv_a: float,
    m_inv_b: float,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    linear_a: wp.vec3,
    linear_b: wp.vec3,
    angular_a: wp.vec3,
    angular_b: wp.vec3,
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float,
) -> float:
    denom = 0.0
    denom += wp.length_sq(linear_a) * m_inv_a
    denom += wp.length_sq(linear_b) * m_inv_b

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    alpha = compliance
    gamma = compliance * damping

    deltaLambda = -(err + alpha * lambda_in + gamma * derr)
    if denom + alpha > 0.0:
        deltaLambda /= dt * (dt + gamma) * denom + alpha

    return deltaLambda


@wp.func
def compute_angular_correction(
    err: float,
    derr: float,
    tf_a: wp.transform,
    tf_b: wp.transform,
    I_inv_a: wp.mat33,
    I_inv_b: wp.mat33,
    angular_a: wp.vec3,
    angular_b: wp.vec3,
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float,
) -> float:
    denom = 0.0

    q1 = wp.transform_get_rotation(tf_a)
    q2 = wp.transform_get_rotation(tf_b)

    # Eq. 2-3 (make sure to project into the frame of the body)
    rot_angular_a = wp.quat_rotate_inv(q1, angular_a)
    rot_angular_b = wp.quat_rotate_inv(q2, angular_b)

    denom += wp.dot(rot_angular_a, I_inv_a * rot_angular_a)
    denom += wp.dot(rot_angular_b, I_inv_b * rot_angular_b)

    alpha = compliance
    gamma = compliance * damping

    deltaLambda = -(err + alpha * lambda_in + gamma * derr)
    if denom + alpha > 0.0:
        deltaLambda /= dt * (dt + gamma) * denom + alpha

    return deltaLambda


@wp.kernel
def solve_body_contact_positions(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    contact_count: wp.array(dtype=int),
    contact_body0: wp.array(dtype=int),
    contact_body1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_thickness: wp.array(dtype=float),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    relaxation: float,
    dt: float,
    contact_torsional_friction: float,
    contact_rolling_friction: float,
    # outputs
    deltas: wp.array(dtype=wp.spatial_vector),
    active_contact_point0: wp.array(dtype=wp.vec3),
    active_contact_point1: wp.array(dtype=wp.vec3),
    active_contact_distance: wp.array(dtype=float),
    contact_inv_weight: wp.array(dtype=float),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    body_a = contact_body0[tid]
    body_b = contact_body1[tid]

    if body_a == body_b:
        return
    if contact_shape0[tid] == contact_shape1[tid]:
        return

    # find body to world transform
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]

    # compute body position in world space
    bx_a = wp.transform_point(X_wb_a, contact_point0[tid])
    bx_b = wp.transform_point(X_wb_b, contact_point1[tid])
    active_contact_point0[tid] = bx_a
    active_contact_point1[tid] = bx_b

    thickness = contact_thickness[tid]
    n = -contact_normal[tid]
    d = wp.dot(n, bx_b - bx_a) - thickness

    active_contact_distance[tid] = d

    if d >= 0.0:
        return

    m_inv_a = 0.0
    m_inv_b = 0.0
    I_inv_a = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_b = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # center of mass in body frame
    com_a = wp.vec3(0.0)
    com_b = wp.vec3(0.0)
    # body to world transform
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    # angular velocities
    omega_a = wp.vec3(0.0)
    omega_b = wp.vec3(0.0)
    # contact offset in body frame
    offset_a = contact_offset0[tid]
    offset_b = contact_offset1[tid]

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        com_a = body_com[body_a]
        m_inv_a = body_m_inv[body_a]
        I_inv_a = body_I_inv[body_a]
        omega_a = wp.spatial_top(body_qd[body_a])

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        com_b = body_com[body_b]
        m_inv_b = body_m_inv[body_b]
        I_inv_b = body_I_inv[body_b]
        omega_b = wp.spatial_top(body_qd[body_b])

    # use average contact material properties
    mat_nonzero = 0
    mu = 0.0
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a >= 0:
        mat_nonzero += 1
        mu += shape_materials.mu[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        mu += shape_materials.mu[shape_b]
    if mat_nonzero > 0:
        mu /= float(mat_nonzero)

    r_a = bx_a - wp.transform_point(X_wb_a, com_a)
    r_b = bx_b - wp.transform_point(X_wb_b, com_b)

    angular_a = -wp.cross(r_a, n)
    angular_b = wp.cross(r_b, n)

    if contact_inv_weight:
        if body_a >= 0:
            wp.atomic_add(contact_inv_weight, body_a, 1.0)
        if body_b >= 0:
            wp.atomic_add(contact_inv_weight, body_b, 1.0)

    lambda_n = compute_contact_constraint_delta(
        d, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, -n, n, angular_a, angular_b, relaxation, dt
    )

    lin_delta_a = -n * lambda_n
    lin_delta_b = n * lambda_n
    ang_delta_a = angular_a * lambda_n
    ang_delta_b = angular_b * lambda_n

    # linear friction
    if mu > 0.0:
        # add on displacement from surface offsets, this ensures we include any rotational effects due to thickness from feature
        # need to use the current rotation to account for friction due to angular effects (e.g.: slipping contact)
        bx_a += wp.transform_vector(X_wb_a, offset_a)
        bx_b += wp.transform_vector(X_wb_b, offset_b)

        # update delta
        delta = bx_b - bx_a
        friction_delta = delta - wp.dot(n, delta) * n

        perp = wp.normalize(friction_delta)

        r_a = bx_a - wp.transform_point(X_wb_a, com_a)
        r_b = bx_b - wp.transform_point(X_wb_b, com_b)

        angular_a = -wp.cross(r_a, perp)
        angular_b = wp.cross(r_b, perp)

        err = wp.length(friction_delta)

        if err > 0.0:
            lambda_fr = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, -perp, perp, angular_a, angular_b, 1.0, dt
            )

            # limit friction based on incremental normal force, good approximation to limiting on total force
            lambda_fr = wp.max(lambda_fr, -lambda_n * mu)

            lin_delta_a -= perp * lambda_fr
            lin_delta_b += perp * lambda_fr

            ang_delta_a += angular_a * lambda_fr
            ang_delta_b += angular_b * lambda_fr

    torsional_friction = mu * contact_torsional_friction

    delta_omega = omega_b - omega_a

    if torsional_friction > 0.0:
        err = wp.dot(delta_omega, n) * dt

        if wp.abs(err) > 0.0:
            lin = wp.vec3(0.0)
            lambda_torsion = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, lin, lin, -n, n, 1.0, dt
            )

            lambda_torsion = wp.clamp(lambda_torsion, -lambda_n * torsional_friction, lambda_n * torsional_friction)

            ang_delta_a -= n * lambda_torsion
            ang_delta_b += n * lambda_torsion

    rolling_friction = mu * contact_rolling_friction
    if rolling_friction > 0.0:
        delta_omega -= wp.dot(n, delta_omega) * n
        err = wp.length(delta_omega) * dt
        if err > 0.0:
            lin = wp.vec3(0.0)
            roll_n = wp.normalize(delta_omega)
            lambda_roll = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, lin, lin, -roll_n, roll_n, 1.0, dt
            )

            lambda_roll = wp.max(lambda_roll, -lambda_n * rolling_friction)

            ang_delta_a -= roll_n * lambda_roll
            ang_delta_b += roll_n * lambda_roll

    if body_a >= 0:
        wp.atomic_add(deltas, body_a, wp.spatial_vector(ang_delta_a, lin_delta_a))
    if body_b >= 0:
        wp.atomic_add(deltas, body_b, wp.spatial_vector(ang_delta_b, lin_delta_b))


@wp.kernel
def update_body_velocities(
    poses: wp.array(dtype=wp.transform),
    poses_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    dt: float,
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    pose = poses[tid]
    pose_prev = poses_prev[tid]

    x = wp.transform_get_translation(pose)
    x_prev = wp.transform_get_translation(pose_prev)

    q = wp.transform_get_rotation(pose)
    q_prev = wp.transform_get_rotation(pose_prev)

    # Update body velocities according to Alg. 2
    # XXX we consider the body COM as the origin of the body frame
    x_com = x + wp.quat_rotate(q, body_com[tid])
    x_com_prev = x_prev + wp.quat_rotate(q_prev, body_com[tid])

    # XXX consider the velocity of the COM
    v = (x_com - x_com_prev) / dt
    dq = q * wp.quat_inverse(q_prev)

    omega = 2.0 / dt * wp.vec3(dq[0], dq[1], dq[2])
    if dq[3] < 0.0:
        omega = -omega

    qd_out[tid] = wp.spatial_vector(omega, v)


@wp.kernel
def apply_rigid_restitution(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd_prev: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    contact_count: wp.array(dtype=int),
    contact_body0: wp.array(dtype=int),
    contact_body1: wp.array(dtype=int),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    active_contact_distance: wp.array(dtype=float),
    active_contact_point0: wp.array(dtype=wp.vec3),
    active_contact_point1: wp.array(dtype=wp.vec3),
    contact_inv_weight: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
    # outputs
    deltas: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return
    d = active_contact_distance[tid]
    if d >= 0.0:
        return

    # use average contact material properties
    mat_nonzero = 0
    restitution = 0.0
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a >= 0:
        mat_nonzero += 1
        restitution += shape_materials.restitution[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        restitution += shape_materials.restitution[shape_b]
    if mat_nonzero > 0:
        restitution /= float(mat_nonzero)

    body_a = contact_body0[tid]
    body_b = contact_body1[tid]

    m_inv_a = 0.0
    m_inv_b = 0.0
    I_inv_a = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    I_inv_b = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # body to world transform
    X_wb_a_prev = wp.transform_identity()
    X_wb_b_prev = wp.transform_identity()
    # center of mass in body frame
    com_a = wp.vec3(0.0)
    com_b = wp.vec3(0.0)
    # previous velocity at contact points
    v_a = wp.vec3(0.0)
    v_b = wp.vec3(0.0)
    # new velocity at contact points
    v_a_new = wp.vec3(0.0)
    v_b_new = wp.vec3(0.0)
    # inverse mass used to compute the impulse
    inv_mass = 0.0

    if body_a >= 0:
        X_wb_a_prev = body_q_prev[body_a]
        X_wb_a = body_q[body_a]
        m_inv_a = body_m_inv[body_a]
        I_inv_a = body_I_inv[body_a]
        com_a = body_com[body_a]

    if body_b >= 0:
        X_wb_b_prev = body_q_prev[body_b]
        X_wb_b = body_q[body_b]
        m_inv_b = body_m_inv[body_b]
        I_inv_b = body_I_inv[body_b]
        com_b = body_com[body_b]

    bx_a = active_contact_point0[tid]
    bx_b = active_contact_point1[tid]

    r_a = bx_a - wp.transform_point(X_wb_a, com_a)
    r_b = bx_b - wp.transform_point(X_wb_b, com_b)

    n = contact_normal[tid]
    if body_a >= 0:
        v_a = velocity_at_point(body_qd_prev[body_a], r_a) + gravity * dt
        v_a_new = velocity_at_point(body_qd[body_a], r_a)
        q_a = wp.transform_get_rotation(X_wb_a_prev)
        rxn = wp.quat_rotate_inv(q_a, wp.cross(r_a, n))
        # Eq. 2
        inv_mass_a = m_inv_a + wp.dot(rxn, I_inv_a * rxn)
        # if (contact_inv_weight):
        #    if (contact_inv_weight[body_a] > 0.0):
        #        inv_mass_a *= contact_inv_weight[body_a]
        inv_mass += inv_mass_a
        # inv_mass += m_inv_a + wp.dot(rxn, I_inv_a * rxn)
    if body_b >= 0:
        v_b = velocity_at_point(body_qd_prev[body_b], r_b) + gravity * dt
        v_b_new = velocity_at_point(body_qd[body_b], r_b)
        q_b = wp.transform_get_rotation(X_wb_b_prev)
        rxn = wp.quat_rotate_inv(q_b, wp.cross(r_b, n))
        # Eq. 3
        inv_mass_b = m_inv_b + wp.dot(rxn, I_inv_b * rxn)
        # if (contact_inv_weight):
        #    if (contact_inv_weight[body_b] > 0.0):
        #        inv_mass_b *= contact_inv_weight[body_b]
        inv_mass += inv_mass_b
        # inv_mass += m_inv_b + wp.dot(rxn, I_inv_b * rxn)

    if inv_mass == 0.0:
        return

    # Eq. 29
    rel_vel_old = wp.dot(n, v_a - v_b)
    rel_vel_new = wp.dot(n, v_a_new - v_b_new)

    # Eq. 34 (Eq. 33 from the ACM paper, note the max operation)
    dv = n * (-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0))

    # Eq. 33
    p = dv / inv_mass
    if body_a >= 0:
        p_a = p
        if contact_inv_weight:
            if contact_inv_weight[body_a] > 0.0:
                p_a /= contact_inv_weight[body_a]
        q_a = wp.transform_get_rotation(X_wb_a)
        rxp = wp.quat_rotate_inv(q_a, wp.cross(r_a, p_a))
        dq = wp.quat_rotate(q_a, I_inv_a * rxp)
        wp.atomic_add(deltas, body_a, wp.spatial_vector(dq, p_a * m_inv_a))

    if body_b >= 0:
        p_b = p
        if contact_inv_weight:
            if contact_inv_weight[body_b] > 0.0:
                p_b /= contact_inv_weight[body_b]
        q_b = wp.transform_get_rotation(X_wb_b)
        rxp = wp.quat_rotate_inv(q_b, wp.cross(r_b, p_b))
        dq = wp.quat_rotate(q_b, I_inv_b * rxp)
        wp.atomic_sub(deltas, body_b, wp.spatial_vector(dq, p_b * m_inv_b))


class XPBDIntegrator:
    """A implicit integrator using XPBD

    After constructing `Model` and `State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Example
    -------

    .. code-block:: python

        integrator = wp.SemiImplicitIntegrator()

        # simulation loop
        for i in range(100):
            state = integrator.simulate(model, state_in, state_out, dt)

    """

    def __init__(
        self,
        iterations=2,
        soft_body_relaxation=0.9,
        soft_contact_relaxation=0.9,
        joint_linear_relaxation=0.7,
        joint_angular_relaxation=0.4,
        rigid_contact_relaxation=0.8,
        rigid_contact_con_weighting=True,
        angular_damping=0.0,
        enable_restitution=False,
    ):
        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation
        self.soft_contact_relaxation = soft_contact_relaxation

        self.joint_linear_relaxation = joint_linear_relaxation
        self.joint_angular_relaxation = joint_angular_relaxation

        self.rigid_contact_relaxation = rigid_contact_relaxation
        self.rigid_contact_con_weighting = rigid_contact_con_weighting

        self.angular_damping = angular_damping

        self.enable_restitution = enable_restitution

    def simulate(self, model, state_in, state_out, dt, requires_grad=False):
        with wp.ScopedTimer("simulate", False):
            particle_q = None
            particle_qd = None

            if model.particle_count:
                if requires_grad:
                    particle_q = wp.zeros_like(state_in.particle_q)
                    particle_qd = wp.zeros_like(state_in.particle_qd)
                else:
                    particle_q = state_out.particle_q
                    particle_qd = state_out.particle_qd
                wp.launch(
                    kernel=integrate_particles,
                    dim=model.particle_count,
                    inputs=[
                        state_in.particle_q,
                        state_in.particle_qd,
                        state_out.particle_f,
                        model.particle_inv_mass,
                        model.gravity,
                        dt,
                    ],
                    outputs=[particle_q, particle_qd],
                    device=model.device,
                )

            if model.body_count:
                if model.joint_count:
                    wp.launch(
                        kernel=apply_joint_torques,
                        dim=model.joint_count,
                        inputs=[
                            state_in.body_q,
                            model.body_com,
                            model.joint_q_start,
                            model.joint_qd_start,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_X_p,
                            model.joint_X_c,
                            model.joint_axis_start,
                            model.joint_axis_dim,
                            model.joint_axis,
                            model.joint_act,
                        ],
                        outputs=[state_in.body_f],
                        device=model.device,
                    )

                wp.launch(
                    kernel=integrate_bodies,
                    dim=model.body_count,
                    inputs=[
                        state_in.body_q,
                        state_in.body_qd,
                        state_in.body_f,
                        model.body_com,
                        model.body_mass,
                        model.body_inertia,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.gravity,
                        self.angular_damping,
                        dt,
                    ],
                    outputs=[state_out.body_q, state_out.body_qd],
                    device=model.device,
                )

            for i in range(self.iterations):
                # print(f"### iteration {i} / {self.iterations-1}")

                if model.body_count:
                    if requires_grad:
                        out_body_q = wp.clone(state_out.body_q)
                        out_body_qd = wp.clone(state_out.body_qd)
                        state_out.body_deltas = wp.zeros_like(state_out.body_deltas)
                    else:
                        out_body_q = state_out.body_q
                        out_body_qd = state_out.body_qd
                        state_out.body_deltas.zero_()
                else:
                    out_body_q = None
                    out_body_qd = None

                # ----------------------------
                # handle particles
                if model.particle_count:
                    if requires_grad:
                        deltas = wp.zeros_like(state_out.particle_f)
                    else:
                        deltas = state_out.particle_f
                        deltas.zero_()

                    # particle ground contact
                    if model.ground:
                        wp.launch(
                            kernel=solve_particle_ground_contacts,
                            dim=model.particle_count,
                            inputs=[
                                particle_q,
                                particle_qd,
                                model.particle_inv_mass,
                                model.soft_contact_ke,
                                model.soft_contact_kd,
                                model.soft_contact_kf,
                                model.soft_contact_mu,
                                model.soft_contact_distance,
                                model.ground_plane,
                                dt,
                                self.soft_contact_relaxation,
                            ],
                            outputs=[deltas],
                            device=model.device,
                        )

                    # particle - rigid body contacts (besides ground plane)
                    if model.shape_count > 1:
                        wp.launch(
                            kernel=solve_soft_contacts,
                            dim=model.soft_contact_max,
                            inputs=[
                                particle_q,
                                particle_qd,
                                model.particle_inv_mass,
                                out_body_q,
                                out_body_qd,
                                model.body_com,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.shape_body,
                                model.shape_materials,
                                model.soft_contact_mu,
                                model.particle_adhesion,
                                model.soft_contact_count,
                                model.soft_contact_particle,
                                model.soft_contact_shape,
                                model.soft_contact_body_pos,
                                model.soft_contact_body_vel,
                                model.soft_contact_normal,
                                model.soft_contact_distance,
                                model.soft_contact_max,
                                dt,
                                self.soft_contact_relaxation,
                            ],
                            # outputs
                            outputs=[deltas, state_out.body_deltas],
                            device=model.device,
                        )

                    # damped springs
                    if model.spring_count:
                        wp.launch(
                            kernel=solve_springs,
                            dim=model.spring_count,
                            inputs=[
                                particle_q,
                                particle_qd,
                                model.particle_inv_mass,
                                model.spring_indices,
                                model.spring_rest_length,
                                model.spring_stiffness,
                                model.spring_damping,
                                dt,
                            ],
                            outputs=[deltas],
                            device=model.device,
                        )

                    # tetrahedral FEM
                    if model.tet_count:
                        wp.launch(
                            kernel=solve_tetrahedra,
                            dim=model.tet_count,
                            inputs=[
                                particle_q,
                                particle_qd,
                                model.particle_inv_mass,
                                model.tet_indices,
                                model.tet_poses,
                                model.tet_activations,
                                model.tet_materials,
                                dt,
                                self.soft_body_relaxation,
                            ],
                            outputs=[deltas],
                            device=model.device,
                        )

                    # apply particle deltas
                    if requires_grad:
                        new_particle_q = wp.clone(particle_q)
                        new_particle_qd = wp.clone(particle_qd)
                    else:
                        new_particle_q = particle_q
                        new_particle_qd = particle_qd

                    wp.launch(
                        kernel=apply_deltas,
                        dim=model.particle_count,
                        inputs=[state_in.particle_q, particle_q, deltas, dt],
                        outputs=[new_particle_q, new_particle_qd],
                        device=model.device,
                    )

                    if requires_grad:
                        particle_q.assign(new_particle_q)
                        particle_qd.assign(new_particle_qd)
                    else:
                        particle_q = new_particle_q
                        particle_qd = new_particle_qd

                # handle rigid bodies
                # ----------------------------

                if model.joint_count:
                    wp.launch(
                        kernel=solve_body_joints,
                        dim=model.joint_count,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.joint_type,
                            model.joint_enabled,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_X_p,
                            model.joint_X_c,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            model.joint_axis_start,
                            model.joint_axis_dim,
                            model.joint_axis_mode,
                            model.joint_axis,
                            model.joint_target,
                            model.joint_target_ke,
                            model.joint_target_kd,
                            model.joint_linear_compliance,
                            model.joint_angular_compliance,
                            self.joint_angular_relaxation,
                            self.joint_linear_relaxation,
                            dt,
                        ],
                        outputs=[state_out.body_deltas],
                        device=model.device,
                    )

                    # apply updates
                    wp.launch(
                        kernel=apply_body_deltas,
                        dim=model.body_count,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            model.body_com,
                            model.body_inertia,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            state_out.body_deltas,
                            None,
                            dt,
                        ],
                        outputs=[
                            out_body_q,
                            out_body_qd,
                        ],
                        device=model.device,
                    )

                if model.body_count and requires_grad:
                    # update state
                    state_out.body_q.assign(out_body_q)
                    state_out.body_qd.assign(out_body_qd)

                # Solve rigid contact constraints
                if model.rigid_contact_max and (
                    model.ground and model.shape_ground_contact_pair_count or model.shape_contact_pair_count
                ):
                    rigid_contact_inv_weight = None
                    if requires_grad:
                        body_deltas = wp.zeros_like(state_out.body_deltas)
                        rigid_active_contact_distance = wp.zeros_like(model.rigid_active_contact_distance)
                        rigid_active_contact_point0 = wp.empty_like(
                            model.rigid_active_contact_point0, requires_grad=True
                        )
                        rigid_active_contact_point1 = wp.empty_like(
                            model.rigid_active_contact_point1, requires_grad=True
                        )
                        if self.rigid_contact_con_weighting:
                            rigid_contact_inv_weight = wp.zeros_like(model.rigid_contact_inv_weight)
                    else:
                        body_deltas = state_out.body_deltas
                        body_deltas.zero_()
                        rigid_active_contact_distance = model.rigid_active_contact_distance
                        rigid_active_contact_point0 = model.rigid_active_contact_point0
                        rigid_active_contact_point1 = model.rigid_active_contact_point1
                        rigid_active_contact_distance.zero_()
                        if self.rigid_contact_con_weighting:
                            rigid_contact_inv_weight = model.rigid_contact_inv_weight
                            rigid_contact_inv_weight.zero_()

                    wp.launch(
                        kernel=solve_body_contact_positions,
                        dim=model.rigid_contact_max,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.rigid_contact_count,
                            model.rigid_contact_body0,
                            model.rigid_contact_body1,
                            model.rigid_contact_point0,
                            model.rigid_contact_point1,
                            model.rigid_contact_offset0,
                            model.rigid_contact_offset1,
                            model.rigid_contact_normal,
                            model.rigid_contact_thickness,
                            model.rigid_contact_shape0,
                            model.rigid_contact_shape1,
                            model.shape_materials,
                            self.rigid_contact_relaxation,
                            dt,
                            model.rigid_contact_torsional_friction,
                            model.rigid_contact_rolling_friction,
                        ],
                        outputs=[
                            body_deltas,
                            rigid_active_contact_point0,
                            rigid_active_contact_point1,
                            rigid_active_contact_distance,
                            rigid_contact_inv_weight,
                        ],
                        device=model.device,
                    )

                    if self.enable_restitution and i == 0:
                        # remember the contacts from the first iteration
                        if requires_grad:
                            model.rigid_active_contact_distance_prev = wp.clone(rigid_active_contact_distance)
                            model.rigid_active_contact_point0_prev = wp.clone(rigid_active_contact_point0)
                            model.rigid_active_contact_point1_prev = wp.clone(rigid_active_contact_point1)
                            if self.rigid_contact_con_weighting:
                                model.rigid_contact_inv_weight_prev = wp.clone(rigid_contact_inv_weight)
                            else:
                                model.rigid_contact_inv_weight_prev = None
                        else:
                            model.rigid_active_contact_distance_prev.assign(rigid_active_contact_distance)
                            model.rigid_active_contact_point0_prev.assign(rigid_active_contact_point0)
                            model.rigid_active_contact_point1_prev.assign(rigid_active_contact_point1)
                            if self.rigid_contact_con_weighting:
                                model.rigid_contact_inv_weight_prev.assign(rigid_contact_inv_weight)
                            else:
                                model.rigid_contact_inv_weight_prev = None

                    if requires_grad:
                        model.rigid_active_contact_distance = rigid_active_contact_distance
                        model.rigid_active_contact_point0 = rigid_active_contact_point0
                        model.rigid_active_contact_point1 = rigid_active_contact_point1
                        body_q = wp.clone(state_out.body_q)
                        body_qd = wp.clone(state_out.body_qd)
                    else:
                        body_q = state_out.body_q
                        body_qd = state_out.body_qd

                    # apply updates
                    wp.launch(
                        kernel=apply_body_deltas,
                        dim=model.body_count,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            model.body_com,
                            model.body_inertia,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            body_deltas,
                            rigid_contact_inv_weight,
                            dt,
                        ],
                        outputs=[
                            body_q,
                            body_qd,
                        ],
                        device=model.device,
                    )

                    if requires_grad:
                        state_out.body_q = body_q
                        state_out.body_qd = body_qd

            # update body velocities from position changes
            if model.body_count and not requires_grad:
                # causes gradient issues (probably due to numerical problems
                # when computing velocities from position changes)
                if requires_grad:
                    out_body_qd = wp.clone(state_out.body_qd)
                else:
                    out_body_qd = state_out.body_qd

                # update body velocities
                wp.launch(
                    kernel=update_body_velocities,
                    dim=model.body_count,
                    inputs=[state_out.body_q, state_in.body_q, model.body_com, dt],
                    outputs=[out_body_qd],
                    device=model.device,
                )

                if requires_grad:
                    state_out.body_qd.assign(out_body_qd)

            if self.enable_restitution:
                if model.particle_count:
                    if requires_grad:
                        new_particle_qd = wp.clone(particle_qd)
                    else:
                        new_particle_qd = particle_qd

                    wp.launch(
                        kernel=apply_soft_restitution_ground,
                        dim=model.particle_count,
                        inputs=[
                            particle_q,
                            particle_qd,
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.particle_inv_mass,
                            model.soft_contact_restitution,
                            model.soft_contact_distance,
                            model.ground_plane,
                            dt,
                            self.soft_contact_relaxation,
                        ],
                        outputs=[new_particle_qd],
                        device=model.device,
                    )

                    if requires_grad:
                        particle_qd.assign(new_particle_qd)
                    else:
                        particle_qd = new_particle_qd

                if model.body_count:
                    if requires_grad:
                        state_out.body_deltas = wp.zeros_like(state_out.body_deltas)
                    else:
                        state_out.body_deltas.zero_()
                    wp.launch(
                        kernel=apply_rigid_restitution,
                        dim=model.rigid_contact_max,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            state_in.body_q,
                            state_in.body_qd,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.rigid_contact_count,
                            model.rigid_contact_body0,
                            model.rigid_contact_body1,
                            model.rigid_contact_normal,
                            model.rigid_contact_shape0,
                            model.rigid_contact_shape1,
                            model.shape_materials,
                            model.rigid_active_contact_distance_prev,
                            model.rigid_active_contact_point0_prev,
                            model.rigid_active_contact_point1_prev,
                            model.rigid_contact_inv_weight_prev,
                            model.gravity,
                            dt,
                        ],
                        outputs=[
                            state_out.body_deltas,
                        ],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=apply_body_delta_velocities,
                        dim=model.body_count,
                        inputs=[
                            state_out.body_qd,
                            state_out.body_deltas,
                        ],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            if model.particle_count:
                # update particle state
                state_out.particle_q.assign(particle_q)
                state_out.particle_qd.assign(particle_qd)

            return state_out
