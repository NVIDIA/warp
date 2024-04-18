# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp

from .integrator import Integrator
from .model import (
    JOINT_MODE_FORCE,
    JOINT_MODE_TARGET_POSITION,
    JOINT_MODE_TARGET_VELOCITY,
    PARTICLE_FLAG_ACTIVE,
    Control,
    Model,
    ModelShapeMaterials,
    State,
)
from .utils import vec_abs, vec_leaky_max, vec_leaky_min, vec_max, vec_min, velocity_at_point


@wp.kernel
def solve_particle_ground_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    ground: wp.array(dtype=float),
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    wi = invmass[tid]
    if wi == 0.0:
        return

    x = particle_x[tid]
    v = particle_v[tid]

    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.min(wp.dot(n, x) + ground[3] - particle_radius[tid], 0.0)

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

    wp.atomic_add(delta, tid, (delta_f - delta_n) * relaxation)


@wp.kernel
def apply_particle_shape_restitution(
    particle_x_new: wp.array(dtype=wp.vec3),
    particle_v_new: wp.array(dtype=wp.vec3),
    particle_x_old: wp.array(dtype=wp.vec3),
    particle_v_old: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    particle_ka: float,
    restitution: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    dt: float,
    relaxation: float,
    particle_v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]

    if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    # x_new = particle_x_new[particle_index]
    v_new = particle_v_new[particle_index]
    px = particle_x_old[particle_index]
    v_old = particle_v_old[particle_index]

    X_wb = wp.transform_identity()
    # X_com = wp.vec3()

    if body_index >= 0:
        X_wb = body_q[body_index]
        # X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    # r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    rel_vel_old = wp.dot(n, v_old)
    rel_vel_new = wp.dot(n, v_new)

    if rel_vel_old < 0.0:
        # dv = -n * wp.max(-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0), 0.0)
        dv = n * (-rel_vel_new + wp.max(-restitution * rel_vel_old, 0.0))

        # compute inverse masses
        # w1 = particle_invmass[particle_index]
        # w2 = 0.0
        # if body_index >= 0:
        #     angular = wp.cross(r, n)
        #     q = wp.transform_get_rotation(X_wb)
        #     rot_angular = wp.quat_rotate_inv(q, angular)
        #     I_inv = body_I_inv[body_index]
        #     w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
        # denom = w1 + w2
        # if denom == 0.0:
        #     return

        wp.atomic_add(particle_v_out, tid, dv)


@wp.kernel
def apply_particle_ground_restitution(
    particle_x_new: wp.array(dtype=wp.vec3),
    particle_v_new: wp.array(dtype=wp.vec3),
    particle_x_old: wp.array(dtype=wp.vec3),
    particle_v_old: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    particle_ka: float,
    restitution: float,
    ground: wp.array(dtype=float),
    dt: float,
    relaxation: float,
    particle_v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    wi = particle_invmass[tid]
    if wi == 0.0:
        return

    x = particle_x_old[tid]
    v_old = particle_v_old[tid]
    v_new = particle_v_new[tid]

    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.dot(n, x) + ground[3] - particle_radius[tid]

    if c > particle_ka:
        return

    vn = wp.dot(n, v_old)
    vn_new = wp.dot(n, v_new)

    if vn < 0.0:
        dv = n * (-vn_new + wp.max(-restitution * vn, 0.0))
        wp.atomic_add(particle_v_out, tid, dv)


@wp.kernel
def solve_particle_shape_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
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

    if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:
        return

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
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

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

    wp.atomic_add(delta, particle_index, w1 * delta_total)

    if body_index >= 0:
        delta_t = wp.cross(r, delta_total)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_t, delta_total))


@wp.kernel
def solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    # outputs
    deltas: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # hash grid has not been built yet
        return
    if (particle_flags[i] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    w1 = particle_invmass[i]

    # particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    delta = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & PARTICLE_FLAG_ACTIVE) != 0 and index != i:
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]

            # compute inverse masses
            w2 = particle_invmass[index]
            denom = w1 + w2

            if err <= k_cohesion and denom > 0.0:
                n = n / d
                vrel = v - particle_v[index]

                # normal
                lambda_n = err
                delta_n = n * lambda_n

                # friction
                vn = wp.dot(n, vrel)
                vt = v - n * vn

                lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                delta_f = wp.normalize(vt) * lambda_f
                delta += (delta_f - delta_n) / denom

    wp.atomic_add(deltas, i, delta * w1 * relaxation)


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
    lambdas: wp.array(dtype=float),
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

    if l == 0.0:
        return

    n = xij / l

    c = l - rest
    grad_c_xi = n
    grad_c_xj = -1.0 * n

    wi = invmass[i]
    wj = invmass[j]

    denom = wi + wj

    # Note strict inequality for damping -- 0 damping is ok
    if denom <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_c_dot_v = dt * wp.dot(grad_c_xi, vij)  # Note: dt because from the paper we want x_i - x^n, not v...
    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_c_dot_v) / ((1.0 + gamma) * denom + alpha)

    dxi = wi * dlambda * grad_c_xi
    dxj = wj * dlambda * grad_c_xj

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, dxi)
    wp.atomic_add(delta, j, dxj)


@wp.kernel
def bending_constraint(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    indices: wp.array2d(dtype=int),
    rest: wp.array(dtype=float),
    bending_properties: wp.array2d(dtype=float),
    dt: float,
    lambdas: wp.array(dtype=float),
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    eps = 1.0e-6

    ke = bending_properties[tid, 0]
    kd = bending_properties[tid, 1]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    if i == -1 or j == -1 or k == -1 or l == -1:
        return

    rest_angle = rest[tid]

    x1 = x[i]
    x2 = x[j]
    x3 = x[k]
    x4 = x[l]

    v1 = v[i]
    v2 = v[j]
    v3 = v[k]
    v4 = v[l]

    w1 = invmass[i]
    w2 = invmass[j]
    w3 = invmass[k]
    w4 = invmass[l]

    n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1
    n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2

    n1_length = wp.length(n1)
    n2_length = wp.length(n2)

    if n1_length < eps or n2_length < eps:
        return

    n1 /= n1_length
    n2 /= n2_length

    cos_theta = wp.dot(n1, n2)

    e = x4 - x3
    e_hat = wp.normalize(e)
    e_length = wp.length(e)

    derivative_flip = wp.sign(wp.dot(wp.cross(n1, n2), e))
    derivative_flip *= -1.0
    angle = wp.acos(cos_theta)

    grad_x1 = n1 * e_length * derivative_flip
    grad_x2 = n2 * e_length * derivative_flip
    grad_x3 = (n1 * wp.dot(x1 - x4, e_hat) + n2 * wp.dot(x2 - x4, e_hat)) * derivative_flip
    grad_x4 = (n1 * wp.dot(x3 - x1, e_hat) + n2 * wp.dot(x3 - x2, e_hat)) * derivative_flip
    c = angle - rest_angle
    denominator = (
        w1 * wp.length_sq(grad_x1)
        + w2 * wp.length_sq(grad_x2)
        + w3 * wp.length_sq(grad_x3)
        + w4 * wp.length_sq(grad_x4)
    )

    # Note strict inequality for damping -- 0 damping is ok
    if denominator <= 0.0 or ke <= 0.0 or kd < 0.0:
        return

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)

    grad_dot_v = dt * (wp.dot(grad_x1, v1) + wp.dot(grad_x2, v2) + wp.dot(grad_x3, v3) + wp.dot(grad_x4, v4))

    dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_dot_v) / ((1.0 + gamma) * denominator + alpha)

    delta0 = w1 * dlambda * grad_x1
    delta1 = w2 * dlambda * grad_x2
    delta2 = w3 * dlambda * grad_x3
    delta3 = w4 * dlambda * grad_x4

    lambdas[tid] = lambdas[tid] + dlambda

    wp.atomic_add(delta, i, delta0)
    wp.atomic_add(delta, j, delta1)
    wp.atomic_add(delta, k, delta2)
    wp.atomic_add(delta, l, delta3)


@wp.kernel
def solve_tetrahedra(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    indices: wp.array(dtype=int, ndim=2),
    rest_matrix: wp.array(dtype=wp.mat33),
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

    # act = activation[tid]

    # k_mu = materials[tid, 0]
    # k_lambda = materials[tid, 1]
    # k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    # v0 = v[i]
    # v1 = v[j]
    # v2 = v[k]
    # v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.mat33(x10, x20, x30)
    Dm = rest_matrix[tid]
    inv_QT = wp.transpose(Dm)

    inv_rest_volume = wp.determinant(Dm) * 6.0

    # F = Xs*Xm^-1
    F = Ds * Dm

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    tr = wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3)

    C = float(0.0)
    dC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    compliance = float(0.0)

    stretching_compliance = relaxation
    volume_compliance = relaxation

    num_terms = 2
    for term in range(0, num_terms):
        if term == 0:
            # deviatoric, stable
            C = tr - 3.0
            dC = F * 2.0
            compliance = stretching_compliance
        elif term == 1:
            # volume conservation
            C = wp.determinant(F) - 1.0
            dC = wp.mat33(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))
            compliance = volume_compliance

        if C != 0.0:
            dP = dC * inv_QT
            grad1 = wp.vec3(dP[0][0], dP[1][0], dP[2][0])
            grad2 = wp.vec3(dP[0][1], dP[1][1], dP[2][1])
            grad3 = wp.vec3(dP[0][2], dP[1][2], dP[2][2])
            grad0 = -grad1 - grad2 - grad3

            w = (
                wp.dot(grad0, grad0) * w0
                + wp.dot(grad1, grad1) * w1
                + wp.dot(grad2, grad2) * w2
                + wp.dot(grad3, grad3) * w3
            )

            if w > 0.0:
                alpha = compliance / dt / dt
                if inv_rest_volume > 0.0:
                    alpha *= inv_rest_volume
                dlambda = -C / (w + alpha)

                wp.atomic_add(delta, i, w0 * dlambda * grad0)
                wp.atomic_add(delta, j, w1 * dlambda * grad1)
                wp.atomic_add(delta, k, w2 * dlambda * grad2)
                wp.atomic_add(delta, l, w3 * dlambda * grad3)
                # wp.atomic_add(particle.num_corr, id0, 1)
                # wp.atomic_add(particle.num_corr, id1, 1)
                # wp.atomic_add(particle.num_corr, id2, 1)
                # wp.atomic_add(particle.num_corr, id3, 1)

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

    # grad1 = wp.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
    # grad2 = wp.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
    # grad3 = wp.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
    # grad0 = (grad1 + grad2 + grad3) * (0.0 - 1.0)

    # denom = (
    #     wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    # )
    # multiplier = C / (denom + 1.0 / (k_mu * dt * dt * rest_volume))

    # delta0 = grad0 * multiplier
    # delta1 = grad1 * multiplier
    # delta2 = grad2 * multiplier
    # delta3 = grad3 * multiplier

    # # hydrostatic part
    # J = wp.determinant(F)

    # C_vol = J - alpha
    # # dCdx = wp.mat33(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))*wp.transpose(Dm)

    # # grad1 = wp.vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # # grad2 = wp.vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # # grad3 = wp.vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    # # grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0)

    # s = inv_rest_volume / 6.0
    # grad1 = wp.cross(x20, x30) * s
    # grad2 = wp.cross(x30, x10) * s
    # grad3 = wp.cross(x10, x20) * s
    # grad0 = -(grad1 + grad2 + grad3)

    # denom = (
    #     wp.dot(grad0, grad0) * w0 + wp.dot(grad1, grad1) * w1 + wp.dot(grad2, grad2) * w2 + wp.dot(grad3, grad3) * w3
    # )
    # multiplier = C_vol / (denom + 1.0 / (k_lambda * dt * dt * rest_volume))

    # delta0 += grad0 * multiplier
    # delta1 += grad1 * multiplier
    # delta2 += grad2 * multiplier
    # delta3 += grad3 * multiplier

    # # # apply forces
    # # wp.atomic_sub(delta, i, delta0 * w0 * relaxation)
    # # wp.atomic_sub(delta, j, delta1 * w1 * relaxation)
    # # wp.atomic_sub(delta, k, delta2 * w2 * relaxation)
    # # wp.atomic_sub(delta, l, delta3 * w3 * relaxation)


@wp.kernel
def solve_tetrahedra2(
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

    # act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    # k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    # v0 = v[i]
    # v1 = v[j]
    # v2 = v[k]
    # v3 = v[l]

    w0 = inv_mass[i]
    w1 = inv_mass[j]
    w2 = inv_mass[k]
    w3 = inv_mass[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

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
def apply_particle_deltas(
    x_orig: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    delta: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    x0 = x_orig[tid]
    xp = x_pred[tid]

    # constraint deltas
    d = delta[tid]

    x_new = xp + d
    v_new = (x_new - x0) / dt

    # enforce velocity limit to prevent instability
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

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
        inv_weight = constraint_inv_weights[tid]
        if inv_weight > 0.0:
            weight = 1.0 / inv_weight

    dp = wp.spatial_bottom(delta) * (inv_m * weight)
    dq = wp.spatial_top(delta) * weight
    dq = wp.quat_rotate(q0, inv_I * wp.quat_rotate_inv(q0, dq))

    # update orientation
    q1 = q0 + 0.5 * wp.quat(dq * dt, 0.0) * q0
    q1 = wp.normalize(q1)

    # update position
    com = body_com[tid]
    x_com = p0 + wp.quat_rotate(q0, com)
    p1 = x_com + dp * dt
    p1 -= wp.quat_rotate(q1, com)

    q_out[tid] = wp.transform(p1, q1)

    v0 = wp.spatial_bottom(qd_in[tid])
    w0 = wp.spatial_top(qd_in[tid])

    # update linear and angular velocity
    v1 = v0 + dp
    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(q0, w0 + dq)
    tb = -wp.cross(wb, body_I[tid] * wb)  # coriolis forces
    w1 = wp.quat_rotate(q0, wb + inv_I * tb * dt)

    # XXX this improves gradient stability
    if wp.length(v1) < 1e-4:
        v1 = wp.vec3(0.0)
    if wp.length(w1) < 1e-4:
        w1 = wp.vec3(0.0)

    qd_out[tid] = wp.spatial_vector(w1, v1)


@wp.kernel
def apply_body_delta_velocities(
    deltas: wp.array(dtype=wp.spatial_vector),
    qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    wp.atomic_add(qd_out, tid, deltas[tid])


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
    joint_axis_mode: wp.array(dtype=int),
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
    # X_cj = joint_X_c[tid]

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

    # # local joint rotations
    # q_p = wp.transform_get_rotation(X_wp)
    # q_c = wp.transform_get_rotation(X_wc)

    # joint properties (for 1D joints)
    # q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    axis_start = joint_axis_start[tid]
    lin_axis_count = joint_axis_dim[tid, 0]
    ang_axis_count = joint_axis_dim[tid, 1]

    # total force/torque on the parent
    t_total = wp.vec3()
    f_total = wp.vec3()

    # handle angular constraints
    if type == wp.sim.JOINT_REVOLUTE:
        mode = joint_axis_mode[axis_start]
        if mode == wp.sim.JOINT_MODE_FORCE:
            axis = joint_axis[axis_start]
            act = joint_act[qd_start]
            a_p = wp.transform_vector(X_wp, axis)
            t_total += act * a_p
    elif type == wp.sim.JOINT_PRISMATIC:
        mode = joint_axis_mode[axis_start]
        if mode == wp.sim.JOINT_MODE_FORCE:
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

        if joint_axis_mode[axis_start + 0] == wp.sim.JOINT_MODE_FORCE:
            axis_0 = joint_axis[axis_start + 0]
            t_total += joint_act[qd_start + 0] * wp.transform_vector(X_wp, axis_0)
        if joint_axis_mode[axis_start + 1] == wp.sim.JOINT_MODE_FORCE:
            axis_1 = joint_axis[axis_start + 1]
            t_total += joint_act[qd_start + 1] * wp.transform_vector(X_wp, axis_1)
        if joint_axis_mode[axis_start + 2] == wp.sim.JOINT_MODE_FORCE:
            axis_2 = joint_axis[axis_start + 2]
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

        if joint_axis_mode[axis_start + 0] == wp.sim.JOINT_MODE_FORCE:
            axis_0 = joint_axis[axis_start + 0]
            t_total += joint_act[qd_start + 0] * wp.transform_vector(X_wp, axis_0)
        if joint_axis_mode[axis_start + 1] == wp.sim.JOINT_MODE_FORCE:
            axis_1 = joint_axis[axis_start + 1]
            t_total += joint_act[qd_start + 1] * wp.transform_vector(X_wp, axis_1)

    elif type == wp.sim.JOINT_D6:
        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a dynamic for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            if joint_axis_mode[axis_start + 0] == wp.sim.JOINT_MODE_FORCE:
                axis = joint_axis[axis_start + 0]
                act = joint_act[qd_start + 0]
                a_p = wp.transform_vector(X_wp, axis)
                f_total += act * a_p
        if lin_axis_count > 1:
            if joint_axis_mode[axis_start + 1] == wp.sim.JOINT_MODE_FORCE:
                axis = joint_axis[axis_start + 1]
                act = joint_act[qd_start + 1]
                a_p = wp.transform_vector(X_wp, axis)
                f_total += act * a_p
        if lin_axis_count > 2:
            if joint_axis_mode[axis_start + 2] == wp.sim.JOINT_MODE_FORCE:
                axis = joint_axis[axis_start + 2]
                act = joint_act[qd_start + 2]
                a_p = wp.transform_vector(X_wp, axis)
                f_total += act * a_p

        if ang_axis_count > 0:
            if joint_axis_mode[axis_start + lin_axis_count + 0] == wp.sim.JOINT_MODE_FORCE:
                axis = joint_axis[axis_start + lin_axis_count + 0]
                act = joint_act[qd_start + lin_axis_count + 0]
                a_p = wp.transform_vector(X_wp, axis)
                t_total += act * a_p
        if ang_axis_count > 1:
            if joint_axis_mode[axis_start + lin_axis_count + 1] == wp.sim.JOINT_MODE_FORCE:
                axis = joint_axis[axis_start + lin_axis_count + 1]
                act = joint_act[qd_start + lin_axis_count + 1]
                a_p = wp.transform_vector(X_wp, axis)
                t_total += act * a_p
        if ang_axis_count > 2:
            if joint_axis_mode[axis_start + lin_axis_count + 2] == wp.sim.JOINT_MODE_FORCE:
                axis = joint_axis[axis_start + lin_axis_count + 2]
                act = joint_act[qd_start + lin_axis_count + 2]
                a_p = wp.transform_vector(X_wp, axis)
                t_total += act * a_p

    else:
        print("joint type not handled in apply_joint_torques")

    # write forces
    if id_p >= 0:
        wp.atomic_sub(body_f, id_p, wp.spatial_vector(t_total + wp.cross(r_p, f_total), f_total))
    wp.atomic_add(body_f, id_c, wp.spatial_vector(t_total + wp.cross(r_c, f_total), f_total))


@wp.func
def update_joint_axis_mode(mode: wp.int32, axis: wp.vec3, input_axis_mode: wp.vec3i):
    # update the 3D axis mode flags given the axis vector and mode of this axis
    mode_x = wp.max(wp.int32(wp.nonzero(axis[0])) * mode, input_axis_mode[0])
    mode_y = wp.max(wp.int32(wp.nonzero(axis[1])) * mode, input_axis_mode[1])
    mode_z = wp.max(wp.int32(wp.nonzero(axis[2])) * mode, input_axis_mode[2])
    return wp.vec3i(mode_x, mode_y, mode_z)


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


@wp.func
def compute_linear_correction_3d(
    dx: wp.vec3,
    r1: wp.vec3,
    r2: wp.vec3,
    tf1: wp.transform,
    tf2: wp.transform,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    lambda_in: float,
    compliance: float,
    damping: float,
    dt: float,
) -> float:
    c = wp.length(dx)
    if c == 0.0:
        # print("c == 0.0 in positional correction")
        return 0.0

    n = wp.normalize(dx)

    q1 = wp.transform_get_rotation(tf1)
    q2 = wp.transform_get_rotation(tf2)

    # Eq. 2-3 (make sure to project into the frame of the body)
    r1xn = wp.quat_rotate_inv(q1, wp.cross(r1, n))
    r2xn = wp.quat_rotate_inv(q2, wp.cross(r2, n))

    w1 = m_inv1 + wp.dot(r1xn, I_inv1 * r1xn)
    w2 = m_inv2 + wp.dot(r2xn, I_inv2 * r2xn)
    w = w1 + w2
    if w == 0.0:
        return 0.0
    alpha = compliance
    gamma = compliance * damping

    # Eq. 4-5
    d_lambda = -c - alpha * lambda_in
    # TODO consider damping for velocity correction?
    # delta_lambda = -(err + alpha * lambda_in + gamma * derr)
    if w + alpha > 0.0:
        d_lambda /= w * (dt + gamma) + alpha / dt

    return d_lambda


@wp.func
def compute_angular_correction_3d(
    corr: wp.vec3,
    q1: wp.quat,
    q2: wp.quat,
    m_inv1: float,
    m_inv2: float,
    I_inv1: wp.mat33,
    I_inv2: wp.mat33,
    alpha_tilde: float,
    # lambda_prev: float,
    relaxation: float,
    dt: float,
):
    # compute and apply the correction impulse for an angular constraint
    theta = wp.length(corr)
    if theta == 0.0:
        return 0.0

    n = wp.normalize(corr)

    # project variables to body rest frame as they are in local matrix
    n1 = wp.quat_rotate_inv(q1, n)
    n2 = wp.quat_rotate_inv(q2, n)

    # Eq. 11-12
    w1 = wp.dot(n1, I_inv1 * n1)
    w2 = wp.dot(n2, I_inv2 * n2)
    w = w1 + w2
    if w == 0.0:
        return 0.0

    # Eq. 13-14
    lambda_prev = 0.0
    d_lambda = (-theta - alpha_tilde * lambda_prev) / (w * dt + alpha_tilde / dt)
    # TODO consider lambda_prev?
    # p = d_lambda * n * relaxation

    # Eq. 15-16
    return d_lambda


@wp.kernel
def solve_simple_body_joints(
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
    joint_axis_mode: wp.array(dtype=int),
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

    if joint_enabled[tid] == 0:
        return
    if type == wp.sim.JOINT_FREE:
        return
    if type == wp.sim.JOINT_COMPOUND:
        return
    if type == wp.sim.JOINT_UNIVERSAL:
        return
    if type == wp.sim.JOINT_DISTANCE:
        return
    if type == wp.sim.JOINT_D6:
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
    # parent transform and moment arm
    if id_p >= 0:
        pose_p = body_q[id_p]
        X_wp = pose_p * X_wp
        com_p = body_com[id_p]
        m_inv_p = body_inv_m[id_p]
        I_inv_p = body_inv_I[id_p]
    r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, com_p)

    # child transform and moment arm
    pose_c = body_q[id_c]
    X_wc = pose_c * X_cj
    com_c = body_com[id_c]
    m_inv_c = body_inv_m[id_c]
    I_inv_c = body_inv_I[id_c]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, com_c)

    if m_inv_p == 0.0 and m_inv_c == 0.0:
        # connection between two immovable bodies
        return

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    # rel_pose = wp.transform_inverse(X_wp) * X_wc
    # rel_p = wp.transform_get_translation(rel_pose)

    # joint connection points
    # x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    # linear_compliance = joint_linear_compliance[tid]
    angular_compliance = joint_angular_compliance[tid]
    damping = 0.0

    axis_start = joint_axis_start[tid]
    # mode = joint_axis_mode[axis_start]

    # local joint rotations
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)
    inertial_q_p = wp.transform_get_rotation(pose_p)
    inertial_q_c = wp.transform_get_rotation(pose_c)

    # joint properties (for 1D joints)
    axis = joint_axis[axis_start]

    if type == wp.sim.JOINT_FIXED:
        limit_lower = 0.0
        limit_upper = 0.0
    else:
        limit_lower = joint_limit_lower[axis_start]
        limit_upper = joint_limit_upper[axis_start]

    # linear_alpha_tilde = linear_compliance / dt / dt
    angular_alpha_tilde = angular_compliance / dt / dt

    # prevent division by zero
    # linear_alpha_tilde = wp.max(linear_alpha_tilde, 1e-6)
    # angular_alpha_tilde = wp.max(angular_alpha_tilde, 1e-6)

    # accumulate constraint deltas
    lin_delta_p = wp.vec3(0.0)
    ang_delta_p = wp.vec3(0.0)
    lin_delta_c = wp.vec3(0.0)
    ang_delta_c = wp.vec3(0.0)

    # handle angular constraints
    if type == wp.sim.JOINT_REVOLUTE:
        # align joint axes
        a_p = wp.quat_rotate(q_p, axis)
        a_c = wp.quat_rotate(q_c, axis)
        # Eq. 20
        corr = wp.cross(a_p, a_c)
        ncorr = wp.normalize(corr)

        angular_relaxation = 0.2
        # angular_correction(
        #     corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
        #     angular_alpha_tilde, angular_relaxation, deltas, id_p, id_c)
        lambda_n = compute_angular_correction_3d(
            corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, angular_alpha_tilde, damping, dt
        )
        lambda_n *= angular_relaxation
        ang_delta_p -= lambda_n * ncorr
        ang_delta_c += lambda_n * ncorr

        # limit joint angles (Alg. 3)
        pi = 3.14159265359
        two_pi = 2.0 * pi
        if limit_lower > -two_pi or limit_upper < two_pi:
            # find a perpendicular vector to joint axis
            a = axis
            # https://math.stackexchange.com/a/3582461
            g = wp.sign(a[2])
            h = a[2] + g
            b = wp.vec3(g - a[0] * a[0] / h, -a[0] * a[1] / h, -a[0])
            c = wp.normalize(wp.cross(a, b))
            # b = c  # TODO verify

            # joint axis
            n = wp.quat_rotate(q_p, a)
            # the axes n1 and n2 are aligned with the two bodies
            n1 = wp.quat_rotate(q_p, b)
            n2 = wp.quat_rotate(q_c, b)

            phi = wp.asin(wp.dot(wp.cross(n1, n2), n))
            # print("phi")
            # print(phi)
            if wp.dot(n1, n2) < 0.0:
                phi = pi - phi
            if phi > pi:
                phi -= two_pi
            if phi < -pi:
                phi += two_pi
            if phi < limit_lower or phi > limit_upper:
                phi = wp.clamp(phi, limit_lower, limit_upper)
                # print("clamped phi")
                # print(phi)
                # rot = wp.quat(phi, n[0], n[1], n[2])
                # rot = wp.quat(n, phi)
                rot = wp.quat_from_axis_angle(n, phi)
                n1 = wp.quat_rotate(rot, n1)
                corr = wp.cross(n1, n2)
                # print("corr")
                # print(corr)
                # TODO expose
                # angular_alpha_tilde = 0.0001 / dt / dt
                # angular_relaxation = 0.5
                # TODO fix this constraint
                # angular_correction(
                #     corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
                #     angular_alpha_tilde, angular_relaxation, deltas, id_p, id_c)
                lambda_n = compute_angular_correction_3d(
                    corr,
                    inertial_q_p,
                    inertial_q_c,
                    m_inv_p,
                    m_inv_c,
                    I_inv_p,
                    I_inv_c,
                    angular_alpha_tilde,
                    damping,
                    dt,
                )
                lambda_n *= angular_relaxation
                ncorr = wp.normalize(corr)
                ang_delta_p -= lambda_n * ncorr
                ang_delta_c += lambda_n * ncorr

        # handle joint targets
        target_ke = joint_target_ke[axis_start]
        # target_kd = joint_target_kd[axis_start]
        target = joint_target[axis_start]
        if target_ke > 0.0:
            # find a perpendicular vector to joint axis
            a = axis
            # https://math.stackexchange.com/a/3582461
            g = wp.sign(a[2])
            h = a[2] + g
            b = wp.vec3(g - a[0] * a[0] / h, -a[0] * a[1] / h, -a[0])
            c = wp.normalize(wp.cross(a, b))
            b = c

            q = wp.quat_from_axis_angle(a_p, target)
            b_target = wp.quat_rotate(q, wp.quat_rotate(q_p, b))
            b2 = wp.quat_rotate(q_c, b)
            # Eq. 21
            d_target = wp.cross(b_target, b2)

            target_compliance = 1.0 / target_ke  # / dt / dt
            # angular_correction(
            #     d_target, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
            #     target_compliance, angular_relaxation, deltas, id_p, id_c)
            lambda_n = compute_angular_correction_3d(
                d_target, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, target_compliance, damping, dt
            )
            lambda_n *= angular_relaxation
            ncorr = wp.normalize(d_target)
            # TODO fix
            ang_delta_p -= lambda_n * ncorr
            ang_delta_c += lambda_n * ncorr

    if (type == wp.sim.JOINT_FIXED) or (type == wp.sim.JOINT_PRISMATIC):
        # align the mutual orientations of the two bodies
        # Eq. 18-19
        q = q_p * wp.quat_inverse(q_c)
        corr = -2.0 * wp.vec3(q[0], q[1], q[2])
        # angular_correction(
        #     -corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c,
        #     angular_alpha_tilde, angular_relaxation, deltas, id_p, id_c)
        lambda_n = compute_angular_correction_3d(
            corr, inertial_q_p, inertial_q_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, angular_alpha_tilde, damping, dt
        )
        lambda_n *= angular_relaxation
        ncorr = wp.normalize(corr)
        ang_delta_p -= lambda_n * ncorr
        ang_delta_c += lambda_n * ncorr

    # handle positional constraints

    # joint connection points
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    # compute error between the joint attachment points on both bodies
    # delta x is the difference of point r_2 minus point r_1 (Fig. 3)
    dx = x_c - x_p

    # rotate the error vector into the joint frame
    q_dx = q_p
    # q_dx = q_c
    # q_dx = wp.transform_get_rotation(pose_p)
    dx = wp.quat_rotate_inv(q_dx, dx)

    lower_pos_limits = wp.vec3(0.0)
    upper_pos_limits = wp.vec3(0.0)
    if type == wp.sim.JOINT_PRISMATIC:
        lower_pos_limits = axis * limit_lower
        upper_pos_limits = axis * limit_upper

    # compute linear constraint violations
    corr = wp.vec3(0.0)
    zero = wp.vec3(0.0)
    corr -= vec_leaky_min(zero, upper_pos_limits - dx)
    corr -= vec_leaky_max(zero, lower_pos_limits - dx)

    # if (type == wp.sim.JOINT_PRISMATIC):
    #     if mode == JOINT_MODE_TARGET_POSITION:
    #         target = wp.clamp(target, limit_lower, limit_upper)
    #         if target_ke > 0.0:
    #             err = dx - target * axis
    #             compliance = 1.0 / target_ke
    #         damping = axis_damping[dim]
    #     elif mode == JOINT_MODE_TARGET_VELOCITY:
    #         if target_ke > 0.0:
    #             err = (derr - target) * dt
    #             compliance = 1.0 / target_ke
    #         damping = axis_damping[dim]

    # rotate correction vector into world frame
    corr = wp.quat_rotate(q_dx, corr)

    lambda_in = 0.0
    linear_alpha = joint_linear_compliance[tid]
    lambda_n = compute_linear_correction_3d(
        corr, r_p, r_c, pose_p, pose_c, m_inv_p, m_inv_c, I_inv_p, I_inv_c, lambda_in, linear_alpha, damping, dt
    )
    lambda_n *= linear_relaxation
    n = wp.normalize(corr)

    lin_delta_p -= n * lambda_n
    lin_delta_c += n * lambda_n
    ang_delta_p -= wp.cross(r_p, n) * lambda_n
    ang_delta_c += wp.cross(r_c, n) * lambda_n

    if id_p >= 0:
        wp.atomic_add(deltas, id_p, wp.spatial_vector(ang_delta_p, lin_delta_p))
    if id_c >= 0:
        wp.atomic_add(deltas, id_c, wp.spatial_vector(ang_delta_c, lin_delta_c))


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
    joint_axis_mode: wp.array(dtype=int),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_act: wp.array(dtype=float),
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

    if joint_enabled[tid] == 0:
        return
    if type == wp.sim.JOINT_FREE:
        return
    # if type == wp.sim.JOINT_FIXED:
    #     return
    # if type == wp.sim.JOINT_REVOLUTE:
    #     return
    # if type == wp.sim.JOINT_PRISMATIC:
    #     return
    # if type == wp.sim.JOINT_BALL:
    #     return

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
    # x_p = wp.transform_get_translation(X_wp)
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
        axis_mode = wp.vec3i(0, 0, 0)
        axis_target_ke_kd = wp.mat33(0.0)
        # avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if lin_axis_count > 0:
            axis = joint_axis[axis_start]
            lo_temp = axis * joint_limit_lower[axis_start]
            up_temp = axis * joint_limit_upper[axis_start]
            axis_limits = wp.spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp))
            mode = joint_axis_mode[axis_start]
            if mode != JOINT_MODE_FORCE:  # position or velocity target
                ke = joint_target_ke[axis_start]
                kd = joint_target_kd[axis_start]
                target = joint_act[axis_start]
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
            if mode != JOINT_MODE_FORCE:  # position or velocity target
                ke = joint_target_ke[axis_idx]
                kd = joint_target_kd[axis_idx]
                target = joint_act[axis_idx]
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
            if mode != JOINT_MODE_FORCE:  # position or velocity target
                ke = joint_target_ke[axis_idx]
                kd = joint_target_kd[axis_idx]
                target = joint_act[axis_idx]
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
            elif e > upper:
                err = e - upper
            else:
                target = axis_target[dim]
                if mode == JOINT_MODE_TARGET_POSITION:
                    target = wp.clamp(target, lower, upper)
                    if axis_stiffness[dim] > 0.0:
                        err = e - target
                        compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                elif mode == JOINT_MODE_TARGET_VELOCITY:
                    if axis_stiffness[dim] > 0.0:
                        err = (derr - target) * dt
                        compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                    derr = 0.0

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
        axis_mode = wp.vec3i(0, 0, 0)
        axis_target_ke_kd = wp.mat33(0.0)
        # avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if ang_axis_count > 0:
            axis_idx = axis_start + lin_axis_count
            axis = joint_axis[axis_idx]
            lo_temp = axis * joint_limit_lower[axis_idx]
            up_temp = axis * joint_limit_upper[axis_idx]
            axis_limits = wp.spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp))
            mode = joint_axis_mode[axis_idx]
            if mode != JOINT_MODE_FORCE:  # position or velocity target
                ke = joint_target_ke[axis_idx]
                kd = joint_target_kd[axis_idx]
                target = joint_act[axis_idx]
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
            if mode != JOINT_MODE_FORCE:  # position or velocity target
                ke = joint_target_ke[axis_idx]
                kd = joint_target_kd[axis_idx]
                target = joint_act[axis_idx]
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
            if mode != JOINT_MODE_FORCE:  # position or velocity target
                ke = joint_target_ke[axis_idx]
                kd = joint_target_kd[axis_idx]
                target = joint_act[axis_idx]
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

        # if type == wp.sim.JOINT_D6:
        #     wp.printf("axis_target: %f %f %f\t axis_stiffness: %f %f %f\t axis_damping: %f %f %f\t axis_limits_lower: %f %f %f \t axis_limits_upper: %f %f %f\n",
        #               axis_target[0], axis_target[1], axis_target[2],
        #               axis_stiffness[0], axis_stiffness[1], axis_stiffness[2],
        #               axis_damping[0], axis_damping[1], axis_damping[2],
        #               axis_limits_lower[0], axis_limits_lower[1], axis_limits_lower[2],
        #               axis_limits_upper[0], axis_limits_upper[1], axis_limits_upper[2])
        #     # wp.printf("wp.sqrt(1.0-qtwist[0]*qtwist[0]) = %f\n", wp.sqrt(1.0-qtwist[0]*qtwist[0]))

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
            elif e > upper:
                err = e - upper
            else:
                target = axis_target[dim]
                if mode == JOINT_MODE_TARGET_POSITION:
                    target = wp.clamp(target, lower, upper)
                    if axis_stiffness[dim] > 0.0:
                        err = e - target
                        compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                elif mode == JOINT_MODE_TARGET_VELOCITY:
                    if axis_stiffness[dim] > 0.0:
                        err = (derr - target) * dt
                        compliance = 1.0 / axis_stiffness[dim]
                    damping = axis_damping[dim]
                    derr = 0.0

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

    delta_lambda = -err
    if denom > 0.0:
        delta_lambda /= dt * denom

    return delta_lambda * relaxation


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

    delta_lambda = -(err + alpha * lambda_in + gamma * derr)
    if denom + alpha > 0.0:
        delta_lambda /= (dt + gamma) * denom + alpha / dt

    return delta_lambda


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

    delta_lambda = -(err + alpha * lambda_in + gamma * derr)
    if denom + alpha > 0.0:
        delta_lambda /= (dt + gamma) * denom + alpha / dt

    return delta_lambda


@wp.kernel
def solve_body_contact_positions(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
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
    contact_inv_weight: wp.array(dtype=float),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    body_b = -1
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    if body_a == body_b:
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

    thickness = contact_thickness[tid]
    n = -contact_normal[tid]
    d = wp.dot(n, bx_b - bx_a) - thickness

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
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_thickness: wp.array(dtype=float),
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
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    body_b = -1

    # use average contact material properties
    mat_nonzero = 0
    restitution = 0.0
    if shape_a >= 0:
        mat_nonzero += 1
        restitution += shape_materials.restitution[shape_a]
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        restitution += shape_materials.restitution[shape_b]
        body_b = shape_body[shape_b]
    if mat_nonzero > 0:
        restitution /= float(mat_nonzero)
    if body_a == body_b:
        return

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
        # X_wb_a = body_q[body_a]
        m_inv_a = body_m_inv[body_a]
        I_inv_a = body_I_inv[body_a]
        com_a = body_com[body_a]

    if body_b >= 0:
        X_wb_b_prev = body_q_prev[body_b]
        # X_wb_b = body_q[body_b]
        m_inv_b = body_m_inv[body_b]
        I_inv_b = body_I_inv[body_b]
        com_b = body_com[body_b]

    # compute body position in world space
    bx_a = wp.transform_point(X_wb_a_prev, contact_point0[tid] + contact_offset0[tid])
    bx_b = wp.transform_point(X_wb_b_prev, contact_point1[tid] + contact_offset1[tid])

    thickness = contact_thickness[tid]
    n = contact_normal[tid]
    d = -wp.dot(n, bx_b - bx_a) - thickness
    if d >= 0.0:
        return

    r_a = bx_a - wp.transform_point(X_wb_a_prev, com_a)
    r_b = bx_b - wp.transform_point(X_wb_b_prev, com_b)

    rxn_a = wp.vec3(0.0)
    rxn_b = wp.vec3(0.0)
    if body_a >= 0:
        v_a = velocity_at_point(body_qd_prev[body_a], r_a) + gravity * dt
        v_a_new = velocity_at_point(body_qd[body_a], r_a)
        q_a = wp.transform_get_rotation(X_wb_a_prev)
        rxn_a = wp.quat_rotate_inv(q_a, wp.cross(r_a, n))
        # Eq. 2
        inv_mass_a = m_inv_a + wp.dot(rxn_a, I_inv_a * rxn_a)
        # if contact_inv_weight:
        #     if contact_inv_weight[body_a] > 0.0:
        #         inv_mass_a *= contact_inv_weight[body_a]
        inv_mass += inv_mass_a
    if body_b >= 0:
        v_b = velocity_at_point(body_qd_prev[body_b], r_b) + gravity * dt
        v_b_new = velocity_at_point(body_qd[body_b], r_b)
        q_b = wp.transform_get_rotation(X_wb_b_prev)
        rxn_b = wp.quat_rotate_inv(q_b, wp.cross(r_b, n))
        # Eq. 3
        inv_mass_b = m_inv_b + wp.dot(rxn_b, I_inv_b * rxn_b)
        # if contact_inv_weight:
        #     if contact_inv_weight[body_b] > 0.0:
        #         inv_mass_b *= contact_inv_weight[body_b]
        inv_mass += inv_mass_b

    if inv_mass == 0.0:
        return

    # Eq. 29
    rel_vel_old = wp.dot(n, v_a - v_b)
    rel_vel_new = wp.dot(n, v_a_new - v_b_new)

    if rel_vel_old >= 0.0:
        return

    # Eq. 34
    dv = (-rel_vel_new - restitution * rel_vel_old) / inv_mass

    # Eq. 33
    if body_a >= 0:
        dv_a = dv
        # if contact_inv_weight:
        #     if contact_inv_weight[body_a] > 0.0:
        #         dv_a *= contact_inv_weight[body_a]
        q_a = wp.transform_get_rotation(X_wb_a_prev)
        dq = wp.quat_rotate(q_a, I_inv_a * rxn_a * dv_a)
        wp.atomic_add(deltas, body_a, wp.spatial_vector(dq, n * m_inv_a * dv_a))

    if body_b >= 0:
        dv_b = -dv
        # if contact_inv_weight:
        #     if contact_inv_weight[body_b] > 0.0:
        #         dv_b *= contact_inv_weight[body_b]
        q_b = wp.transform_get_rotation(X_wb_b_prev)
        dq = wp.quat_rotate(q_b, I_inv_b * rxn_b * dv_b)
        wp.atomic_add(deltas, body_b, wp.spatial_vector(dq, n * m_inv_b * dv_b))


class XPBDIntegrator(Integrator):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16). Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. 2020. Detailed rigid body simulation with extended position based dynamics. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA '20). Eurographics Association, Goslar, DEU, Article 10, 1-12. https://doi.org/10.1111/cgf.14105

    After constructing :class:`Model`, :class:`State`, and :class:`Control` (optional) objects, this time-integrator
    may be used to advance the simulation state forward in time.

    Example
    -------

    .. code-block:: python

        integrator = wp.XPBDIntegrator()

        # simulation loop
        for i in range(100):
            state = integrator.simulate(model, state_in, state_out, dt, control)

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

        self.compute_body_velocity_from_position_delta = False

        # helper variables to track constraint resolution vars
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

    def apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
    ):
        if state_in.requires_grad:
            particle_q = state_out.particle_q
            # allocate new particle arrays so gradients can be tracked correctly without overwriting
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                model.particle_flags,
                particle_deltas,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[new_particle_q, new_particle_qd],
            device=model.device,
        )

        if state_in.requires_grad:
            state_out.particle_q = new_particle_q
            state_out.particle_qd = new_particle_qd

        return new_particle_q, new_particle_qd

    def apply_body_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        body_deltas: wp.array,
        dt: float,
        rigid_contact_inv_weight: wp.array = None,
    ):
        with wp.ScopedTimer("apply_body_deltas", False):
            if state_in.requires_grad:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                new_body_q = wp.clone(body_q)
                new_body_qd = wp.clone(body_qd)
                self._body_delta_counter += 1
            else:
                if self._body_delta_counter == 0:
                    body_q = state_out.body_q
                    body_qd = state_out.body_qd
                    new_body_q = state_in.body_q
                    new_body_qd = state_in.body_qd
                else:
                    body_q = state_in.body_q
                    body_qd = state_in.body_qd
                    new_body_q = state_out.body_q
                    new_body_qd = state_out.body_qd
                self._body_delta_counter = 1 - self._body_delta_counter

            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    model.body_com,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    body_deltas,
                    rigid_contact_inv_weight,
                    dt,
                ],
                outputs=[
                    new_body_q,
                    new_body_qd,
                ],
                device=model.device,
            )

            if state_in.requires_grad:
                state_out.body_q = new_body_q
                state_out.body_qd = new_body_qd

        return new_body_q, new_body_qd

    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        particle_q = None
        particle_qd = None
        particle_deltas = None

        body_q = None
        body_qd = None
        body_deltas = None

        rigid_contact_inv_weight = None

        if model.rigid_contact_max > 0:
            if self.rigid_contact_con_weighting:
                rigid_contact_inv_weight = wp.zeros_like(model.rigid_contact_thickness)
            rigid_contact_inv_weight_init = None

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            if model.particle_count:
                if requires_grad:
                    particle_q = state_out.particle_q
                    particle_qd = state_out.particle_qd
                else:
                    particle_q = state_out.particle_q
                    particle_qd = state_out.particle_qd

                self.particle_q_init = wp.clone(state_in.particle_q)
                if self.enable_restitution:
                    self.particle_qd_init = wp.clone(state_in.particle_qd)
                particle_deltas = wp.empty_like(state_out.particle_qd)

                self.integrate_particles(model, state_in, state_out, dt)

            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd

                if self.compute_body_velocity_from_position_delta or self.enable_restitution:
                    body_q_init = wp.clone(state_in.body_q)
                    body_qd_init = wp.clone(state_in.body_qd)

                body_deltas = wp.empty_like(state_out.body_qd)

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
                            model.joint_axis_mode,
                            control.joint_act,
                        ],
                        outputs=[state_in.body_f],
                        device=model.device,
                    )

                self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)

            spring_constraint_lambdas = None
            if model.spring_count:
                spring_constraint_lambdas = wp.empty_like(model.spring_rest_length)
            edge_constraint_lambdas = None
            if model.edge_count:
                edge_constraint_lambdas = wp.empty_like(model.edge_rest_angle)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if model.body_count:
                        if requires_grad and i > 0:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                    if model.particle_count:
                        if requires_grad and i > 0:
                            particle_deltas = wp.zeros_like(particle_deltas)
                        else:
                            particle_deltas.zero_()

                        # particle ground contact
                        if model.ground:
                            wp.launch(
                                kernel=solve_particle_ground_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.soft_contact_ke,
                                    model.soft_contact_kd,
                                    model.soft_contact_kf,
                                    model.soft_contact_mu,
                                    model.ground_plane,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # particle-rigid body contacts (besides ground plane)
                        if model.shape_count > 1:
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=model.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    body_q,
                                    body_qd,
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
                                    model.soft_contact_max,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                # outputs
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                            )

                        if model.particle_max_radius > 0.0 and model.particle_count > 1:
                            # assert model.particle_grid.reserved, "model.particle_grid must be built, see HashGrid.build()"
                            wp.launch(
                                kernel=solve_particle_particle_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.particle_mu,
                                    model.particle_cohesion,
                                    model.particle_max_radius,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # distance constraints
                        if model.spring_count:
                            spring_constraint_lambdas.zero_()
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
                                    spring_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # bending constraints
                        if model.edge_count:
                            edge_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=bending_constraint,
                                dim=model.edge_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.edge_indices,
                                    model.edge_rest_angle,
                                    model.edge_bending_properties,
                                    dt,
                                    edge_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
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
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        particle_q, particle_qd = self.apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                    # handle rigid bodies
                    # ----------------------------

                    if model.joint_count:
                        # wp.launch(
                        #     kernel=solve_simple_body_joints,
                        #     dim=model.joint_count,
                        #     inputs=[
                        #         body_q,
                        #         body_qd,
                        #         model.body_com,
                        #         model.body_inv_mass,
                        #         model.body_inv_inertia,
                        #         model.joint_type,
                        #         model.joint_enabled,
                        #         model.joint_parent,
                        #         model.joint_child,
                        #         model.joint_X_p,
                        #         model.joint_X_c,
                        #         model.joint_limit_lower,
                        #         model.joint_limit_upper,
                        #         model.joint_axis_start,
                        #         model.joint_axis_dim,
                        #         model.joint_axis_mode,
                        #         model.joint_axis,
                        #         control.joint_target,
                        #         model.joint_target_ke,
                        #         model.joint_target_kd,
                        #         model.joint_linear_compliance,
                        #         model.joint_angular_compliance,
                        #         self.joint_angular_relaxation,
                        #         self.joint_linear_relaxation,
                        #         dt,
                        #     ],
                        #     outputs=[body_deltas],
                        #     device=model.device,
                        # )

                        wp.launch(
                            kernel=solve_body_joints,
                            dim=model.joint_count,
                            inputs=[
                                body_q,
                                body_qd,
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
                                control.joint_act,
                                model.joint_target_ke,
                                model.joint_target_kd,
                                model.joint_linear_compliance,
                                model.joint_angular_compliance,
                                self.joint_angular_relaxation,
                                self.joint_linear_relaxation,
                                dt,
                            ],
                            outputs=[body_deltas],
                            device=model.device,
                        )

                        body_q, body_qd = self.apply_body_deltas(model, state_in, state_out, body_deltas, dt)

                    # Solve rigid contact constraints
                    if model.rigid_contact_max and (
                        model.ground and model.shape_ground_contact_pair_count or model.shape_contact_pair_count
                    ):
                        if self.rigid_contact_con_weighting:
                            rigid_contact_inv_weight.zero_()
                        body_deltas.zero_()

                        wp.launch(
                            kernel=solve_body_contact_positions,
                            dim=model.rigid_contact_max,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.shape_body,
                                model.rigid_contact_count,
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
                                rigid_contact_inv_weight,
                            ],
                            device=model.device,
                        )

                        # if model.rigid_contact_count.numpy()[0] > 0:
                        #     print("rigid_contact_count:", model.rigid_contact_count.numpy().flatten())
                        #     # print("rigid_active_contact_distance:", rigid_active_contact_distance.numpy().flatten())
                        #     # print("rigid_active_contact_point0:", rigid_active_contact_point0.numpy().flatten())
                        #     # print("rigid_active_contact_point1:", rigid_active_contact_point1.numpy().flatten())
                        #     print("body_deltas:", body_deltas.numpy().flatten())

                        # print(rigid_active_contact_distance.numpy().flatten())

                        if self.enable_restitution and i == 0:
                            # remember contact constraint weighting from the first iteration
                            if self.rigid_contact_con_weighting:
                                rigid_contact_inv_weight_init = wp.clone(rigid_contact_inv_weight)
                            else:
                                rigid_contact_inv_weight_init = None

                        body_q, body_qd = self.apply_body_deltas(
                            model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight
                        )

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            if model.body_count:
                if body_q.ptr != state_out.body_q.ptr:
                    state_out.body_q.assign(body_q)
                    state_out.body_qd.assign(body_qd)

            # update body velocities from position changes
            if self.compute_body_velocity_from_position_delta and model.body_count and not requires_grad:
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
                    inputs=[state_out.body_q, body_q_init, model.body_com, dt],
                    outputs=[out_body_qd],
                    device=model.device,
                )

            if self.enable_restitution:
                if model.particle_count:
                    wp.launch(
                        kernel=apply_particle_shape_restitution,
                        dim=model.particle_count,
                        inputs=[
                            particle_q,
                            particle_qd,
                            self.particle_q_init,
                            self.particle_qd_init,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            body_q,
                            body_qd,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.shape_body,
                            model.shape_materials,
                            model.particle_adhesion,
                            model.soft_contact_restitution,
                            model.soft_contact_count,
                            model.soft_contact_particle,
                            model.soft_contact_shape,
                            model.soft_contact_body_pos,
                            model.soft_contact_body_vel,
                            model.soft_contact_normal,
                            model.soft_contact_max,
                            dt,
                            self.soft_contact_relaxation,
                        ],
                        outputs=[state_out.particle_qd],
                        device=model.device,
                    )
                    if model.ground:
                        wp.launch(
                            kernel=apply_particle_ground_restitution,
                            dim=model.particle_count,
                            inputs=[
                                particle_q,
                                particle_qd,
                                self.particle_q_init,
                                self.particle_qd_init,
                                model.particle_inv_mass,
                                model.particle_radius,
                                model.particle_flags,
                                model.particle_adhesion,
                                model.soft_contact_restitution,
                                model.ground_plane,
                                dt,
                                self.soft_contact_relaxation,
                            ],
                            outputs=[state_out.particle_qd],
                            device=model.device,
                        )

                if model.body_count:
                    body_deltas.zero_()
                    wp.launch(
                        kernel=apply_rigid_restitution,
                        dim=model.rigid_contact_max,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            body_q_init,
                            body_qd_init,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.shape_body,
                            model.rigid_contact_count,
                            model.rigid_contact_normal,
                            model.rigid_contact_shape0,
                            model.rigid_contact_shape1,
                            model.shape_materials,
                            model.rigid_contact_point0,
                            model.rigid_contact_point1,
                            model.rigid_contact_offset0,
                            model.rigid_contact_offset1,
                            model.rigid_contact_thickness,
                            rigid_contact_inv_weight_init,
                            model.gravity,
                            dt,
                        ],
                        outputs=[
                            body_deltas,
                        ],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=apply_body_delta_velocities,
                        dim=model.body_count,
                        inputs=[
                            body_deltas,
                        ],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            return state_out
