"""This module contains time-integration objects for simulating
models + state forward in time.

"""

import math
from oglang.types import vec3
import numpy as np
import time

import oglang as og

from . optimizer import Optimizer

# Todo
#-----
#
# [x] Spring model
# [x] 2D FEM model
# [x] 3D FEM model
# [x] Cloth
#     [x] Wind/Drag model
#     [x] Bending model
#     [x] Triangle collision
# [x] Rigid body model
# [x] Rigid shape contact
#     [x] Sphere
#     [x] Capsule
#     [x] Box
#     [ ] Convex
#     [ ] Sdf
# [ ] Implicit solver
# [x] USD import
# [x] USD export
# -----

@og.func
def test(c: float):

    x = 1.0

    if (c < 3.0):
        x = 2.0

    return x*6.0


def kernel_init():
    global kernels
    kernels = og.compile()


@og.kernel
def integrate_particles(x: og.array(dtype=og.vec3),
                        v: og.array(dtype=og.vec3),
                        f: og.array(dtype=og.vec3),
                        w: og.array(dtype=float),
                        gravity: og.vec3,
                        dt: float,
                        x_new: og.array(dtype=og.vec3),
                        v_new: og.array(dtype=og.vec3)):

    tid = og.tid()

    x0 = og.load(x, tid)
    v0 = og.load(v, tid)
    f0 = og.load(f, tid)
    inv_mass = og.load(w, tid)

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + gravity * og.step(0.0 - inv_mass)) * dt
    x1 = x0 + v1 * dt

    og.store(x_new, tid, x1)
    og.store(v_new, tid, v1)


# semi-implicit Euler integration
@og.kernel
def integrate_rigids(rigid_x: og.array(dtype=og.vec3),
                     rigid_r: og.array(dtype=og.quat),
                     rigid_v: og.array(dtype=og.vec3),
                     rigid_w: og.array(dtype=og.vec3),
                     rigid_f: og.array(dtype=og.vec3),
                     rigid_t: og.array(dtype=og.vec3),
                     inv_m: og.array(dtype=float),
                     inv_I: og.array(dtype=og.mat33),
                     gravity: og.vec3,
                     dt: float,
                     rigid_x_new: og.array(dtype=og.vec3),
                     rigid_r_new: og.array(dtype=og.quat),
                     rigid_v_new: og.array(dtype=og.vec3),
                     rigid_w_new: og.array(dtype=og.vec3)):

    tid = og.tid()

    # positions
    x0 = og.load(rigid_x, tid)
    r0 = og.load(rigid_r, tid)

    # velocities
    v0 = og.load(rigid_v, tid)
    w0 = og.load(rigid_w, tid)         # angular velocity

    # forces
    f0 = og.load(rigid_f, tid)
    t0 = og.load(rigid_t, tid)

    # masses
    inv_mass = og.load(inv_m, tid)     # 1 / mass
    inv_inertia = og.load(inv_I, tid)  # inverse of 3x3 inertia matrix

    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * og.nonzero(inv_mass)) * dt           # linear integral (linear position/velocity)
    x1 = x0 + v1 * dt

    # angular part

    # so reverse multiplication by r0 takes you from global coordinates into local coordinates
    # because it's covector and thus gets pulled back rather than pushed forward
    wb = og.rotate_inv(r0, w0)         # angular integral (angular velocity and rotation), rotate into object reference frame
    tb = og.rotate_inv(r0, t0)         # also rotate torques into local coordinates

    # I^{-1} torque = angular acceleration and inv_inertia is always going to be in the object frame.
    # So we need to rotate into that frame, and then back into global.
    w1 = og.rotate(r0, wb + inv_inertia * tb * dt)                   # I^-1 * torque * dt., then go back into global coordinates
    r1 = og.normalize(r0 + og.quat(w1, 0.0) * r0 * 0.5 * dt)         # rotate around w1 by dt

    og.store(rigid_x_new, tid, x1)
    og.store(rigid_r_new, tid, r1)
    og.store(rigid_v_new, tid, v1)
    og.store(rigid_w_new, tid, w1)


@og.kernel
def eval_springs(x: og.array(dtype=og.vec3),
                 v: og.array(dtype=og.vec3),
                 spring_indices: og.array(dtype=int),
                 spring_rest_lengths: og.array(dtype=float),
                 spring_stiffness: og.array(dtype=float),
                 spring_damping: og.array(dtype=float),
                 f: og.array(dtype=og.vec3)):

    tid = og.tid()

    i = og.load(spring_indices, tid * 2 + 0)
    j = og.load(spring_indices, tid * 2 + 1)

    ke = og.load(spring_stiffness, tid)
    kd = og.load(spring_damping, tid)
    rest = og.load(spring_rest_lengths, tid)

    xi = og.load(x, i)
    xj = og.load(x, j)

    vi = og.load(v, i)
    vj = og.load(v, j)

    xij = xi - xj
    vij = vi - vj

    l = length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = dot(dir, vij)

    # damping based on relative velocity.
    fs = dir * (ke * c + kd * dcdt)

    og.atomic_sub(f, i, fs)
    og.atomic_add(f, j, fs)


@og.kernel
def eval_triangles(x: og.array(dtype=og.vec3),
                   v: og.array(dtype=og.vec3),
                   indices: og.array(dtype=int),
                   pose: og.array(dtype=og.mat22),
                   activation: og.array(dtype=float),
                   k_mu: float,
                   k_lambda: float,
                   k_damp: float,
                   k_drag: float,
                   k_lift: float,
                   f: og.array(dtype=og.vec3)):
    tid = og.tid()

    i = og.load(indices, tid * 3 + 0)
    j = og.load(indices, tid * 3 + 1)
    k = og.load(indices, tid * 3 + 2)

    x0 = og.load(x, i)        # point zero
    x1 = og.load(x, j)        # point one
    x2 = og.load(x, k)        # point two

    v0 = og.load(v, i)       # vel zero
    v1 = og.load(v, j)       # vel one
    v2 = og.load(v, k)       # vel two

    x10 = x1 - x0     # barycentric coordinates (centered at p)
    x20 = x2 - x0

    v10 = v1 - v0
    v20 = v2 - v0

    Dm = og.load(pose, tid)

    inv_rest_area = og.determinant(Dm) * 2.0     # 1 / det(A) = det(A^-1)
    rest_area = 1.0 / inv_rest_area

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    k_damp = k_damp * rest_area

    # F = Xs*Xm^-1
    F1 = x10 * Dm[0, 0] + x20 * Dm[1, 0]
    F2 = x10 * Dm[0, 1] + x20 * Dm[1, 1]

    # dFdt = Vs*Xm^-1
    dFdt1 = v10*Dm[0, 0] + v20*Dm[1, 0]
    dFdt2 = v10*Dm[0 ,1] + v20*Dm[1, 1]

    # deviatoric PK1 + damping term
    P1 = F1*k_mu + dFdt1*k_damp
    P2 = F2*k_mu + dFdt2*k_damp

    #-----------------------------
    # St. Venant-Kirchoff

    # # Green strain, F'*F-I
    # e00 = dot(f1, f1) - 1.0
    # e10 = dot(f2, f1)
    # e01 = dot(f1, f2)
    # e11 = dot(f2, f2) - 1.0

    # E = og.mat22(e00, e01,
    #              e10, e11)

    # # local forces (deviatoric part)
    # T = og.mul(E, og.transpose(Dm))

    # # spatial forces, F*T
    # fq = (f1*T[0,0] + f2*T[1,0])*k_mu*2.0
    # fr = (f1*T[0,1] + f2*T[1,1])*k_mu*2.0
    # alpha = 1.0

    #-----------------------------
    # Baraff & Witkin, note this model is not isotropic

    # c1 = length(f1) - 1.0
    # c2 = length(f2) - 1.0
    # f1 = normalize(f1)*c1*k1
    # f2 = normalize(f2)*c2*k1

    # fq = f1*Dm[0,0] + f2*Dm[0,1]
    # fr = f1*Dm[1,0] + f2*Dm[1,1]

    #-----------------------------
    # Neo-Hookean (with rest stability)

    # force = P*Dm'
    f1 = (P1 * Dm[0, 0] + P2 * Dm[0, 1])
    f2 = (P1 * Dm[1, 0] + P2 * Dm[1, 1])
    alpha = 1.0 + k_mu / k_lambda

    #-----------------------------
    # Area Preservation

    n = og.cross(x10, x20)
    area = og.length(n) * 0.5

    # actuation
    act = og.load(activation, tid)

    # J-alpha
    c = area * inv_rest_area - alpha + act

    # dJdx
    n = og.normalize(n)
    dcdq = og.cross(x20, n) * inv_rest_area * 0.5
    dcdr = og.cross(n, x10) * inv_rest_area * 0.5

    f_area = k_lambda * c

    #-----------------------------
    # Area Damping

    dcdt = dot(dcdq, v1) + dot(dcdr, v2) - dot(dcdq + dcdr, v0)
    f_damp = k_damp * dcdt

    f1 = f1 + dcdq * (f_area + f_damp)
    f2 = f2 + dcdr * (f_area + f_damp)
    f0 = f1 + f2

    #-----------------------------
    # Lift + Drag

    vmid = (v0 + v1 + v2) * 0.3333
    vdir = og.normalize(vmid)

    f_drag = vmid * (k_drag * area * og.abs(og.dot(n, vmid)))
    f_lift = n * (k_lift * area * (1.57079 - og.acos(og.dot(n, vdir)))) * dot(vmid, vmid)

    # note reversed sign due to atomic_add below.. need to write the unary op -
    f0 = f0 - f_drag - f_lift
    f1 = f1 + f_drag + f_lift
    f2 = f2 + f_drag + f_lift

    # apply forces
    og.atomic_add(f, i, f0)
    og.atomic_sub(f, j, f1)
    og.atomic_sub(f, k, f2)

@og.func
def triangle_closest_point_barycentric(a: og.vec3, b: og.vec3, c: og.vec3, p: og.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = og.dot(ab, ap)
    d2 = og.dot(ac, ap)

    if (d1 <= 0.0 and d2 <= 0.0):
        return vec3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = og.dot(ab, bp)
    d4 = og.dot(ac, bp)

    if (d3 >= 0.0 and d4 <= d3):
        return vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
        return vec3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    if (d6 >= 0.0 and d5 <= d6):
        return vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
        return vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
        return vec3(0.0, w, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return vec3(1.0 - v - w, v, w)

# @og.func
# def triangle_closest_point(a: og.vec3, b: og.vec3, c: og.vec3, p: og.vec3):
#     ab = b - a
#     ac = c - a
#     ap = p - a

#     d1 = og.dot(ab, ap)
#     d2 = og.dot(ac, ap)

#     if (d1 <= 0.0 and d2 <= 0.0):
#         return a

#     bp = p - b
#     d3 = og.dot(ab, bp)
#     d4 = og.dot(ac, bp)

#     if (d3 >= 0.0 and d4 <= d3):
#         return b

#     vc = d1 * d4 - d3 * d2
#     v = d1 / (d1 - d3)
#     if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
#         return a + ab * v

#     cp = p - c
#     d5 = dot(ab, cp)
#     d6 = dot(ac, cp)

#     if (d6 >= 0.0 and d5 <= d6):
#         return c

#     vb = d5 * d2 - d1 * d6
#     w = d2 / (d2 - d6)
#     if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
#         return a + ac * w

#     va = d3 * d6 - d5 * d4
#     w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
#     if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
#         return b + (c - b) * w

#     denom = 1.0 / (va + vb + vc)
#     v = vb * denom
#     w = vc * denom

#     return a + ab * v + ac * w


@og.kernel
def eval_triangles_contact(
                                       # idx : og.array(dtype=int), # list of indices for colliding particles
    num_particles: int,                # size of particles
    x: og.array(dtype=og.vec3),
    v: og.array(dtype=og.vec3),
    indices: og.array(dtype=int),
    pose: og.array(dtype=og.mat22),
    activation: og.array(dtype=float),
    k_mu: float,
    k_lambda: float,
    k_damp: float,
    k_drag: float,
    k_lift: float,
    f: og.array(dtype=og.vec3)):

    tid = og.tid()
    face_no = tid // num_particles     # which face
    particle_no = tid % num_particles  # which particle

    # index = og.load(idx, tid)
    pos = og.load(x, particle_no)      # at the moment, just one particle
                                       # vel0 = og.load(v, 0)

    i = og.load(indices, face_no * 3 + 0)
    j = og.load(indices, face_no * 3 + 1)
    k = og.load(indices, face_no * 3 + 2)

    if (i == particle_no or j == particle_no or k == particle_no):
        return

    p = og.load(x, i)        # point zero
    q = og.load(x, j)        # point one
    r = og.load(x, k)        # point two

    # vp = og.load(v, i) # vel zero
    # vq = og.load(v, j) # vel one
    # vr = og.load(v, k)  # vel two

    # qp = q-p # barycentric coordinates (centered at p)
    # rp = r-p

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest
    dist = og.dot(diff, diff)
    n = og.normalize(diff)
    c = og.min(dist - 0.01, 0.0)       # 0 unless within 0.01 of surface
                                       #c = og.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)
    fn = n * c * 1e5

    og.atomic_sub(f, particle_no, fn)

    # # apply forces (could do - f / 3 here)
    og.atomic_add(f, i, fn * bary[0])
    og.atomic_add(f, j, fn * bary[1])
    og.atomic_add(f, k, fn * bary[2])


@og.kernel
def eval_triangles_rigid_contacts(
    num_particles: int,                          # number of particles (size of contact_point)
    x: og.array(dtype=og.vec3),                     # position of particles
    v: og.array(dtype=og.vec3),
    indices: og.array(dtype=int),                     # triangle indices
    rigid_x: og.array(dtype=og.vec3),               # rigid body positions
    rigid_r: og.array(dtype=og.quat),
    rigid_v: og.array(dtype=og.vec3),
    rigid_w: og.array(dtype=og.vec3),
    contact_body: og.array(dtype=int),
    contact_point: og.array(dtype=og.vec3),         # position of contact points relative to body
    contact_dist: og.array(dtype=float),
    contact_mat: og.array(dtype=int),
    materials: og.array(dtype=float),
                                                 #   rigid_f : og.array(dtype=og.vec3),
                                                 #   rigid_t : og.array(dtype=og.vec3),
    tri_f: og.array(dtype=og.vec3)):

    tid = og.tid()

    face_no = tid // num_particles     # which face
    particle_no = tid % num_particles  # which particle

    # -----------------------
    # load rigid body point
    c_body = og.load(contact_body, particle_no)
    c_point = og.load(contact_point, particle_no)
    c_dist = og.load(contact_dist, particle_no)
    c_mat = og.load(contact_mat, particle_no)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = og.load(materials, c_mat * 4 + 0)       # restitution coefficient
    kd = og.load(materials, c_mat * 4 + 1)       # damping coefficient
    kf = og.load(materials, c_mat * 4 + 2)       # friction coefficient
    mu = og.load(materials, c_mat * 4 + 3)       # coulomb friction

    x0 = og.load(rigid_x, c_body)      # position of colliding body
    r0 = og.load(rigid_r, c_body)      # orientation of colliding body

    v0 = og.load(rigid_v, c_body)
    w0 = og.load(rigid_w, c_body)

    # transform point to world space
    pos = x0 + og.rotate(r0, c_point)
    # use x0 as center, everything is offset from center of mass

    # moment arm
    r = pos - x0                       # basically just c_point in the new coordinates
    rhat = og.normalize(r)
    pos = pos + rhat * c_dist          # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    # contact point velocity
    dpdt = v0 + og.cross(w0, r)        # this is rigid velocity cross offset, so it's the velocity of the contact point.

    # -----------------------
    # load triangle
    i = og.load(indices, face_no * 3 + 0)
    j = og.load(indices, face_no * 3 + 1)
    k = og.load(indices, face_no * 3 + 2)

    p = og.load(x, i)        # point zero
    q = og.load(x, j)        # point one
    r = og.load(x, k)        # point two

    vp = og.load(v, i)       # vel zero
    vq = og.load(v, j)       # vel one
    vr = og.load(v, k)       # vel two

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest               # vector from tri to point
    dist = og.dot(diff, diff)          # squared distance
    n = og.normalize(diff)             # points into the object
    c = og.min(dist - 0.05, 0.0)       # 0 unless within 0.05 of surface
                                       #c = og.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)
                                       # fn = n * c * 1e6    # points towards cloth (both n and c are negative)

    # og.atomic_sub(tri_f, particle_no, fn)

    fn = c * ke    # normal force (restitution coefficient * how far inside for ground) (negative)

    vtri = vp * bary[0] + vq * bary[1] + vr * bary[2]         # bad approximation for centroid velocity
    vrel = vtri - dpdt

    vn = dot(n, vrel)        # velocity component of rigid in negative normal direction
    vt = vrel - n * vn       # velocity component not in normal direction

    # contact damping
    fd = 0.0 - og.max(vn, 0.0) * kd * og.step(c)           # again, negative, into the ground

    # # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)
    upper = 0.0 - lower      # workaround because no unary ops yet

    nx = cross(n, vec3(0.0, 0.0, 1.0))         # basis vectors for tangent
    nz = cross(n, vec3(1.0, 0.0, 0.0))

    vx = og.clamp(dot(nx * kf, vt), lower, upper)
    vz = og.clamp(dot(nz * kf, vt), lower, upper)

    ft = (nx * vx + nz * vz) * (0.0 - og.step(c))          # og.vec3(vx, 0.0, vz)*og.step(c)

    # # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    # #ft = og.normalize(vt)*og.min(kf*og.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft

    og.atomic_add(tri_f, i, f_total * bary[0])
    og.atomic_add(tri_f, j, f_total * bary[1])
    og.atomic_add(tri_f, k, f_total * bary[2])


@og.kernel
def eval_bending(
    x: og.array(dtype=og.vec3), v: og.array(dtype=og.vec3), indices: og.array(dtype=int), rest: og.array(dtype=float), ke: float, kd: float, f: og.array(dtype=og.vec3)):

    tid = og.tid()

    i = og.load(indices, tid * 4 + 0)
    j = og.load(indices, tid * 4 + 1)
    k = og.load(indices, tid * 4 + 2)
    l = og.load(indices, tid * 4 + 3)

    rest_angle = og.load(rest, tid)

    x1 = og.load(x, i)
    x2 = og.load(x, j)
    x3 = og.load(x, k)
    x4 = og.load(x, l)

    v1 = og.load(v, i)
    v2 = og.load(v, j)
    v3 = og.load(v, k)
    v4 = og.load(v, l)

    n1 = og.cross(x3 - x1, x4 - x1)    # normal to face 1
    n2 = og.cross(x4 - x2, x3 - x2)    # normal to face 2

    n1_length = og.length(n1)
    n2_length = og.length(n2)

    if (n1_length < 1.e-3 or n2_length < 1.e-3):
        return

    rcp_n1 = 1.0 / n1_length
    rcp_n2 = 1.0 / n2_length

    cos_theta = og.dot(n1, n2) * rcp_n1 * rcp_n2

    n1 = n1 * rcp_n1 * rcp_n1
    n2 = n2 * rcp_n2 * rcp_n2

    e = x4 - x3
    e_hat = og.normalize(e)
    e_length = og.length(e)

    s = og.sign(og.dot(og.cross(n2, n1), e_hat))
    angle = og.acos(cos_theta) * s

    d1 = n1 * e_length
    d2 = n2 * e_length
    d3 = n1 * og.dot(x1 - x4, e_hat) + n2 * og.dot(x2 - x4, e_hat)
    d4 = n1 * og.dot(x3 - x1, e_hat) + n2 * og.dot(x3 - x2, e_hat)

    # elastic
    f_elastic = ke * (angle - rest_angle)

    # damping
    f_damp = kd * (og.dot(d1, v1) + og.dot(d2, v2) + og.dot(d3, v3) + og.dot(d4, v4))

    # total force, proportional to edge length
    f_total = 0.0 - e_length * (f_elastic + f_damp)

    og.atomic_add(f, i, d1 * f_total)
    og.atomic_add(f, j, d2 * f_total)
    og.atomic_add(f, k, d3 * f_total)
    og.atomic_add(f, l, d4 * f_total)


@og.kernel
def eval_tetrahedra(x: og.array(dtype=og.vec3),
                    v: og.array(dtype=og.vec3),
                    indices: og.array(dtype=int),
                    pose: og.array(dtype=og.mat33),
                    activation: og.array(dtype=float),
                    materials: og.array(dtype=float),
                    f: og.array(dtype=og.vec3)):

    tid = og.tid()

    i = og.load(indices, tid * 4 + 0)
    j = og.load(indices, tid * 4 + 1)
    k = og.load(indices, tid * 4 + 2)
    l = og.load(indices, tid * 4 + 3)

    act = og.load(activation, tid)

    k_mu = og.load(materials, tid * 3 + 0)
    k_lambda = og.load(materials, tid * 3 + 1)
    k_damp = og.load(materials, tid * 3 + 2)

    x0 = og.load(x, i)
    x1 = og.load(x, j)
    x2 = og.load(x, k)
    x3 = og.load(x, l)

    v0 = og.load(v, i)
    v1 = og.load(v, j)
    v2 = og.load(v, k)
    v3 = og.load(v, l)

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = og.mat33(x10, x20, x30)
    Dm = og.load(pose, tid)

    inv_rest_volume = og.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume
    k_damp = k_damp * rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm
    dFdt = og.mat33(v10, v20, v30) * Dm

    col1 = og.vec3(F[0, 0], F[1, 0], F[2, 0])
    col2 = og.vec3(F[0, 1], F[1, 1], F[2, 1])
    col3 = og.vec3(F[0, 2], F[1, 2], F[2, 2])

    #-----------------------------
    # Neo-Hookean (with rest stability [Smith et al 2018])
         
    Ic = dot(col1, col1) + dot(col2, col2) + dot(col3, col3)

    # deviatoric part
    P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp
    H = P * og.transpose(Dm)

    f1 = og.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = og.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = og.vec3(H[0, 2], H[1, 2], H[2, 2])

    #-----------------------------
    # C_sqrt

    # alpha = 1.0
       
    # r_s = og.sqrt(og.abs(dot(col1, col1) + dot(col2, col2) + dot(col3, col3) - 3.0))

    # f1 = og.vec3(0.0, 0.0, 0.0)
    # f2 = og.vec3(0.0, 0.0, 0.0)
    # f3 = og.vec3(0.0, 0.0, 0.0)

    # if (r_s > 0.0):
    #     r_s_inv = 1.0/r_s

    #     C = r_s 
    #     dCdx = F*og.transpose(Dm)*r_s_inv*og.sign(r_s)

    #     grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    #     grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    #     grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
    
    #     f1 = grad1*C*k_mu
    #     f2 = grad2*C*k_mu
    #     f3 = grad3*C*k_mu

    #-----------------------------
    # C_spherical
    
    # alpha = 1.0

    # r_s = og.sqrt(dot(col1, col1) + dot(col2, col2) + dot(col3, col3))
    # r_s_inv = 1.0/r_s

    # C = r_s - og.sqrt(3.0) 
    # dCdx = F*og.transpose(Dm)*r_s_inv

    # grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
 

    # f1 = grad1*C*k_mu
    # f2 = grad2*C*k_mu
    # f3 = grad3*C*k_mu

    #----------------------------
    # C_D

    # alpha = 1.0

    # r_s = og.sqrt(dot(col1, col1) + dot(col2, col2) + dot(col3, col3))

    # C = r_s*r_s - 3.0
    # dCdx = F*og.transpose(Dm)*2.0

    # grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0])
    # grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1])
    # grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2])
 
    # f1 = grad1*C*k_mu
    # f2 = grad2*C*k_mu
    # f3 = grad3*C*k_mu

    #----------------------------
    # Hookean
     
    # alpha = 1.0

    # I = og.mat33(og.vec3(1.0, 0.0, 0.0),
    #              og.vec3(0.0, 1.0, 0.0),
    #              og.vec3(0.0, 0.0, 1.0))

    # P = (F + og.transpose(F) + I*(0.0-2.0))*k_mu
    # H = P * og.transpose(Dm)

    # f1 = og.vec3(H[0, 0], H[1, 0], H[2, 0])
    # f2 = og.vec3(H[0, 1], H[1, 1], H[2, 1])
    # f3 = og.vec3(H[0, 2], H[1, 2], H[2, 2])



    # hydrostatic part
    J = og.determinant(F)

    #print(J)
    s = inv_rest_volume / 6.0
    dJdx1 = og.cross(x20, x30) * s
    dJdx2 = og.cross(x30, x10) * s
    dJdx3 = og.cross(x10, x20) * s

    f_volume = (J - alpha + act) * k_lambda
    f_damp = (og.dot(dJdx1, v1) + og.dot(dJdx2, v2) + og.dot(dJdx3, v3)) * k_damp

    f_total = f_volume + f_damp

    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    f3 = f3 + dJdx3 * f_total
    f0 = (f1 + f2 + f3) * (0.0 - 1.0)

    # apply forces
    og.atomic_sub(f, i, f0)
    og.atomic_sub(f, j, f1)
    og.atomic_sub(f, k, f2)
    og.atomic_sub(f, l, f3)


@og.kernel
def eval_contacts(x: og.array(dtype=og.vec3), v: og.array(dtype=og.vec3), ke: float, kd: float, kf: float, mu: float, offset: float, ground: og.vec4, f: og.array(dtype=og.vec3)):

    tid = og.tid()           # this just handles contact of particles with the ground plane, nothing else.

    x0 = og.load(x, tid)
    v0 = og.load(v, tid)

    n = og.vec3(ground[0], ground[1], ground[2])
    c = og.min(dot(n, x0) + ground[3] - offset, 0.0)

    vn = dot(n, v0)
    vt = v0 - n * vn

    fn = n * c * ke

    # contact damping
    fd = n * og.min(vn, 0.0) * kd

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * c * ke
    upper = 0.0 - lower

    vx = clamp(dot(vec3(kf, 0.0, 0.0), vt), lower, upper)
    vz = clamp(dot(vec3(0.0, 0.0, kf), vt), lower, upper)

    ft = og.vec3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = og.normalize(vt)*og.min(kf*og.length(vt), 0.0 - mu*c*ke)

    ftotal = fn + (fd + ft) * og.step(c)

    og.atomic_sub(f, tid, ftotal)


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
def eval_soft_contacts(
    num_particles: int,
    particle_x: og.array(dtype=og.vec3), 
    particle_v: og.array(dtype=og.vec3), 
    body_X_sc: og.array(dtype=og.spatial_transform),
    body_v_sc: og.array(dtype=og.spatial_vector),
    shape_X_co: og.array(dtype=og.spatial_transform),
    shape_body: og.array(dtype=int),
    shape_geo_type: og.array(dtype=int), 
    shape_geo_id: og.array(dtype=og.uint64),
    shape_geo_scale: og.array(dtype=og.vec3),
    shape_materials: og.array(dtype=float),
    ke: float,
    kd: float, 
    kf: float, 
    mu: float,
    contact_distance: float,
    contact_margin: float,
    # outputs
    particle_f: og.array(dtype=og.vec3),
    body_f: og.array(dtype=og.spatial_vector)):

    tid = og.tid()           

    shape_index = tid // num_particles     # which shape
    particle_index = tid % num_particles   # which particle
    rigid_index = og.load(shape_body, shape_index)

    px = og.load(particle_x, particle_index)
    pv = og.load(particle_v, particle_index)

    #center = vec3(0.0, 0.5, 0.0)
    #radius = 0.25
    #margin = 0.01

    # sphere collider
    # c = og.min(sphere_sdf(center, radius, x0)-margin, 0.0)
    # n = sphere_sdf_grad(center, radius, x0)

    # box collider
    #c = og.min(box_sdf(og.vec3(radius, radius, radius), x0-center)-margin, 0.0)
    #n = box_sdf_grad(og.vec3(radius, radius, radius), x0-center)

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

    margin = contact_distance

    # evaluate shape sdf
    c = 0.0
    n = og.vec3(0.0, 0.0, 0.0)

    # GEO_SPHERE (0)
    if (geo_type == 0):
        c = og.min(sphere_sdf(og.vec3(0.0, 0.0, 0.0), geo_scale[0], x_local)-margin, 0.0)
        n = og.spatial_transform_vector(X_so, sphere_sdf_grad(og.vec3(0.0, 0.0, 0.0), geo_scale[0], x_local))

    # GEO_BOX (1)
    if (geo_type == 1):
        c = og.min(box_sdf(geo_scale, x_local)-margin, 0.0)
        n = og.spatial_transform_vector(X_so, box_sdf_grad(geo_scale, x_local))
    
    # GEO_CAPSULE (2)
    if (geo_type == 2):
        c = og.min(capsule_sdf(geo_scale[0], geo_scale[1], x_local)-margin, 0.0)
        n = og.spatial_transform_vector(X_so, capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local))

    # GEO_MESH (3)
    if (geo_type == 3):
        mesh = og.load(shape_geo_id, shape_index)

        face_index = int(0)
        face_v = float(0.0)  
        face_w = float(0.0)
        sign = float(0.0)

        c = 0.0
        n = og.vec3()

        if (og.mesh_query_point(mesh, x_local/geo_scale[0], contact_margin, sign, face_index, face_v, face_w)):

            shape_p = og.mesh_eval_position(mesh, face_index, face_v, face_w)
            shape_v = og.mesh_eval_velocity(mesh, face_index, face_v, face_w)

            shape_p = shape_p*geo_scale[0]

            delta = x_local-shape_p
            c = og.min(og.length(delta)*sign - margin, 0.0)
            n = og.normalize(delta)*sign

            # subtract shape velocity off of particle velocity to make relative to any internal deformation
            pv = pv - shape_v


        
    # rigid velocity
    rigid_v_s = og.spatial_vector()
    if (rigid_index >= 0):
        rigid_v_s = og.load(body_v_sc, rigid_index)
    
    rigid_w = og.spatial_top(rigid_v_s)
    rigid_v = og.spatial_bottom(rigid_v_s)

    # compute the body velocity at the particle position
    bv = rigid_v + og.cross(rigid_w, px)

    # relative velocity
    v = pv - bv

    # decompose relative velocity
    vn = dot(n, v)
    vt = v - n * vn
    
    # contact elastic
    fn = n * c * ke

    # contact damping
    fd = n * og.min(vn, 0.0) * kd

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * c * ke
    upper = 0.0 - lower

    vx = clamp(dot(vec3(kf, 0.0, 0.0), vt), lower, upper)
    vz = clamp(dot(vec3(0.0, 0.0, kf), vt), lower, upper)

    ft = og.vec3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = og.normalize(vt)*og.min(kf*og.length(vt), 0.0 - mu*c*ke)

    f_total = fn + (fd + ft) * og.step(c)
    t_total = og.cross(px, f_total)

    og.atomic_sub(particle_f, particle_index, f_total)

    if (rigid_index >= 0):
        og.atomic_sub(body_f, rigid_index, og.spatial_vector(t_total, f_total))



@og.kernel
def eval_rigid_contacts(rigid_x: og.array(dtype=og.vec3),
                        rigid_r: og.array(dtype=og.quat),
                        rigid_v: og.array(dtype=og.vec3),
                        rigid_w: og.array(dtype=og.vec3),
                        contact_body: og.array(dtype=int),
                        contact_point: og.array(dtype=og.vec3),
                        contact_dist: og.array(dtype=float),
                        contact_mat: og.array(dtype=int),
                        materials: og.array(dtype=float),
                        rigid_f: og.array(dtype=og.vec3),
                        rigid_t: og.array(dtype=og.vec3)):

    tid = og.tid()

    c_body = og.load(contact_body, tid)
    c_point = og.load(contact_point, tid)
    c_dist = og.load(contact_dist, tid)
    c_mat = og.load(contact_mat, tid)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = og.load(materials, c_mat * 4 + 0)       # restitution coefficient
    kd = og.load(materials, c_mat * 4 + 1)       # damping coefficient
    kf = og.load(materials, c_mat * 4 + 2)       # friction coefficient
    mu = og.load(materials, c_mat * 4 + 3)       # coulomb friction

    x0 = og.load(rigid_x, c_body)      # position of colliding body
    r0 = og.load(rigid_r, c_body)      # orientation of colliding body

    v0 = og.load(rigid_v, c_body)
    w0 = og.load(rigid_w, c_body)

    n = vec3(0.0, 1.0, 0.0)

    # transform point to world space
    p = x0 + og.rotate(r0, c_point) - n * c_dist           # add on 'thickness' of shape, e.g.: radius of sphere/capsule
                                                           # use x0 as center, everything is offset from center of mass

    # moment arm
    r = p - x0     # basically just c_point in the new coordinates

    # contact point velocity
    dpdt = v0 + og.cross(w0, r)        # this is rigid velocity cross offset, so it's the velocity of the contact point.

    # check ground contact
    c = og.min(dot(n, p), 0.0)         # check if we're inside the ground

    vn = dot(n, dpdt)        # velocity component out of the ground
    vt = dpdt - n * vn       # velocity component not into the ground

    fn = c * ke    # normal force (restitution coefficient * how far inside for ground)

    # contact damping
    fd = og.min(vn, 0.0) * kd * og.step(c)       # again, velocity into the ground, negative

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)   # negative
    upper = 0.0 - lower      # positive, workaround for no unary ops

    vx = og.clamp(dot(vec3(kf, 0.0, 0.0), vt), lower, upper)
    vz = og.clamp(dot(vec3(0.0, 0.0, kf), vt), lower, upper)

    ft = og.vec3(vx, 0.0, vz) * og.step(c)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = og.normalize(vt)*og.min(kf*og.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft
    t_total = og.cross(r, f_total)

    og.atomic_sub(rigid_f, c_body, f_total)
    og.atomic_sub(rigid_t, c_body, t_total)

# # Frank & Park definition 3.20, pg 100
@og.func
def spatial_transform_twist(t: og.spatial_transform, x: og.spatial_vector):

    q = spatial_transform_get_rotation(t)
    p = spatial_transform_get_translation(t)

    w = spatial_top(x)
    v = spatial_bottom(x)

    w = rotate(q, w)
    v = rotate(q, v) + cross(p, w)

    return og.spatial_vector(w, v)


@og.func
def spatial_transform_wrench(t: og.spatial_transform, x: og.spatial_vector):

    q = spatial_transform_get_rotation(t)
    p = spatial_transform_get_translation(t)

    w = spatial_top(x)
    v = spatial_bottom(x)

    v = rotate(q, v)
    w = rotate(q, w) + cross(p, v)

    return og.spatial_vector(w, v)

@og.func
def spatial_transform_inverse(t: og.spatial_transform):

    p = spatial_transform_get_translation(t)
    q = spatial_transform_get_rotation(t)

    q_inv = inverse(q)
    return spatial_transform(rotate(q_inv, p)*(0.0 - 1.0), q_inv)



# computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
@og.func
def spatial_transform_inertia(t: og.spatial_transform, I: og.spatial_matrix):

    t_inv = spatial_transform_inverse(t)

    q = spatial_transform_get_rotation(t_inv)
    p = spatial_transform_get_translation(t_inv)

    r1 = rotate(q, vec3(1.0, 0.0, 0.0))
    r2 = rotate(q, vec3(0.0, 1.0, 0.0))
    r3 = rotate(q, vec3(0.0, 0.0, 1.0))

    R = mat33(r1, r2, r3)
    S = mul(skew(p), R)

    T = spatial_adjoint(R, S)
    
    return mul(mul(transpose(T), I), T)


@og.kernel
def eval_rigid_contacts_art(
    body_X_s: og.array(dtype=og.spatial_transform),
    body_v_s: og.array(dtype=og.spatial_vector),
    contact_body: og.array(dtype=int),
    contact_point: og.array(dtype=og.vec3),
    contact_dist: og.array(dtype=float),
    contact_mat: og.array(dtype=int),
    materials: og.array(dtype=float),
    body_f_s: og.array(dtype=og.spatial_vector)):

    tid = og.tid()

    c_body = og.load(contact_body, tid)
    c_point = og.load(contact_point, tid)
    c_dist = og.load(contact_dist, tid)
    c_mat = og.load(contact_mat, tid)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = og.load(materials, c_mat * 4 + 0)       # restitution coefficient
    kd = og.load(materials, c_mat * 4 + 1)       # damping coefficient
    kf = og.load(materials, c_mat * 4 + 2)       # friction coefficient
    mu = og.load(materials, c_mat * 4 + 3)       # coulomb friction

    X_s = og.load(body_X_s, c_body)              # position of colliding body
    v_s = og.load(body_v_s, c_body)              # orientation of colliding body

    n = vec3(0.0, 1.0, 0.0)

    # transform point to world space
    p = og.spatial_transform_point(X_s, c_point) - n * c_dist  # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    w = og.spatial_top(v_s)
    v = og.spatial_bottom(v_s)

    # contact point velocity
    dpdt = v + og.cross(w, p)

    # check ground contact
    c = og.min(dot(n, p), 0.0)         # check if we're inside the ground

    vn = dot(n, dpdt)        # velocity component out of the ground
    vt = dpdt - n * vn       # velocity component not into the ground

    fn = c * ke              # normal force (restitution coefficient * how far inside for ground)

    # contact damping
    fd = og.min(vn, 0.0) * kd * og.step(c)       # again, velocity into the ground, negative

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)   # negative
    upper = 0.0 - lower      # positive, workaround for no unary ops

    vx = og.clamp(dot(vec3(kf, 0.0, 0.0), vt), lower, upper)
    vz = og.clamp(dot(vec3(0.0, 0.0, kf), vt), lower, upper)

    ft = og.vec3(vx, 0.0, vz) * og.step(c)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = og.normalize(vt)*og.min(kf*og.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft
    t_total = og.cross(p, f_total)

    og.atomic_add(body_f_s, c_body, og.spatial_vector(t_total, f_total))


@og.func
def compute_muscle_force(
    i: int,
    body_X_s: og.array(dtype=og.spatial_transform),
    body_v_s: og.array(dtype=og.spatial_vector),    
    muscle_links: og.array(dtype=int),
    muscle_points: og.array(dtype=og.vec3),
    muscle_activation: float,
    body_f_s: og.array(dtype=og.spatial_vector)):

    link_0 = og.load(muscle_links, i)
    link_1 = og.load(muscle_links, i+1)

    if (link_0 == link_1):
        return 0

    r_0 = og.load(muscle_points, i)
    r_1 = og.load(muscle_points, i+1)

    xform_0 = og.load(body_X_s, link_0)
    xform_1 = og.load(body_X_s, link_1)

    pos_0 = og.spatial_transform_point(xform_0, r_0)
    pos_1 = og.spatial_transform_point(xform_1, r_1)

    n = og.normalize(pos_1 - pos_0)

    # todo: add passive elastic and viscosity terms
    f = n * muscle_activation

    og.atomic_sub(body_f_s, link_0, og.spatial_vector(og.cross(pos_0, f), f))
    og.atomic_add(body_f_s, link_1, og.spatial_vector(og.cross(pos_1, f), f))

    return 0


@og.kernel
def eval_muscles(
    body_X_s: og.array(dtype=og.spatial_transform),
    body_v_s: og.array(dtype=og.spatial_vector),
    muscle_start: og.array(dtype=int),
    muscle_params: og.array(dtype=float),
    muscle_links: og.array(dtype=int),
    muscle_points: og.array(dtype=og.vec3),
    muscle_activation: og.array(dtype=float),
    # output
    body_f_s: og.array(dtype=og.spatial_vector)):

    tid = og.tid()

    m_start = og.load(muscle_start, tid)
    m_end = og.load(muscle_start, tid+1) - 1

    activation = og.load(muscle_activation, tid)

    for i in range(m_start, m_end):
        compute_muscle_force(i, body_X_s, body_v_s, muscle_links, muscle_points, activation, body_f_s)
    

# compute transform across a joint
@og.func
def jcalc_transform(type: int, axis: og.vec3, joint_q: og.array(dtype=float), start: int):

    # prismatic
    if (type == 0):

        q = og.load(joint_q, start)
        X_jc = spatial_transform(axis * q, quat_identity())
        return X_jc

    # revolute
    if (type == 1):

        q = og.load(joint_q, start)
        X_jc = spatial_transform(vec3(0.0, 0.0, 0.0), quat_from_axis_angle(axis, q))
        return X_jc

    # ball
    if (type == 2):

        qx = og.load(joint_q, start + 0)
        qy = og.load(joint_q, start + 1)
        qz = og.load(joint_q, start + 2)
        qw = og.load(joint_q, start + 3)

        X_jc = spatial_transform(vec3(0.0, 0.0, 0.0), quat(qx, qy, qz, qw))
        return X_jc

    # fixed
    if (type == 3):

        X_jc = spatial_transform_identity()
        return X_jc

    # free
    if (type == 4):

        px = og.load(joint_q, start + 0)
        py = og.load(joint_q, start + 1)
        pz = og.load(joint_q, start + 2)

        qx = og.load(joint_q, start + 3)
        qy = og.load(joint_q, start + 4)
        qz = og.load(joint_q, start + 5)
        qw = og.load(joint_q, start + 6)

        X_jc = spatial_transform(vec3(px, py, pz), quat(qx, qy, qz, qw))
        return X_jc

    # default case
    return spatial_transform_identity()


# compute motion subspace and velocity for a joint
@og.func
def jcalc_motion(type: int, axis: og.vec3, X_sc: og.spatial_transform, joint_S_s: og.array(dtype=og.spatial_vector), joint_qd: og.array(dtype=float), joint_start: int):

    # prismatic
    if (type == 0):

        S_s = og.spatial_transform_twist(X_sc, og.spatial_vector(vec3(0.0, 0.0, 0.0), axis))
        v_j_s = S_s * og.load(joint_qd, joint_start)

        og.store(joint_S_s, joint_start, S_s)
        return v_j_s

    # revolute
    if (type == 1):

        S_s = og.spatial_transform_twist(X_sc, og.spatial_vector(axis, vec3(0.0, 0.0, 0.0)))
        v_j_s = S_s * og.load(joint_qd, joint_start)

        og.store(joint_S_s, joint_start, S_s)
        return v_j_s

    # ball
    if (type == 2):

        w = vec3(og.load(joint_qd, joint_start + 0),
                   og.load(joint_qd, joint_start + 1),
                   og.load(joint_qd, joint_start + 2))

        S_0 = og.spatial_transform_twist(X_sc, og.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        S_1 = og.spatial_transform_twist(X_sc, og.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        S_2 = og.spatial_transform_twist(X_sc, og.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

        # write motion subspace
        og.store(joint_S_s, joint_start + 0, S_0)
        og.store(joint_S_s, joint_start + 1, S_1)
        og.store(joint_S_s, joint_start + 2, S_2)

        return S_0*w[0] + S_1*w[1] + S_2*w[2]

    # fixed
    if (type == 3):
        return og.spatial_vector()

    # free
    if (type == 4):

        v_j_s = og.spatial_vector(og.load(joint_qd, joint_start + 0),
                               og.load(joint_qd, joint_start + 1),
                               og.load(joint_qd, joint_start + 2),
                               og.load(joint_qd, joint_start + 3),
                               og.load(joint_qd, joint_start + 4),
                               og.load(joint_qd, joint_start + 5))

        # write motion subspace
        og.store(joint_S_s, joint_start + 0, og.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        og.store(joint_S_s, joint_start + 1, og.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        og.store(joint_S_s, joint_start + 2, og.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        og.store(joint_S_s, joint_start + 3, og.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        og.store(joint_S_s, joint_start + 4, og.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        og.store(joint_S_s, joint_start + 5, og.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    # default case
    return og.spatial_vector()


# # compute the velocity across a joint
# #@og.func
# def jcalc_velocity(self, type, S_s, joint_qd, start):

#     # prismatic
#     if (type == 0):
#         v_j_s = og.load(S_s, start)*og.load(joint_qd, start)
#         return v_j_s

#     # revolute
#     if (type == 1):
#         v_j_s = og.load(S_s, start)*og.load(joint_qd, start)
#         return v_j_s

#     # fixed
#     if (type == 2):
#         v_j_s = og.spatial_vector()
#         return v_j_s

#     # free
#     if (type == 3):
#         v_j_s =  S_s[start+0]*joint_qd[start+0]
#         v_j_s += S_s[start+1]*joint_qd[start+1]
#         v_j_s += S_s[start+2]*joint_qd[start+2]
#         v_j_s += S_s[start+3]*joint_qd[start+3]
#         v_j_s += S_s[start+4]*joint_qd[start+4]
#         v_j_s += S_s[start+5]*joint_qd[start+5]
#         return v_j_s


# computes joint space forces/torques in tau
@og.func
def jcalc_tau(
    type: int, 
    target_k_e: float,
    target_k_d: float,
    limit_k_e: float,
    limit_k_d: float,
    joint_S_s: og.array(dtype=og.spatial_vector), 
    joint_q: og.array(dtype=float),
    joint_qd: og.array(dtype=float),
    joint_act: og.array(dtype=float),
    joint_target: og.array(dtype=float),
    joint_limit_lower: og.array(dtype=float),
    joint_limit_upper: og.array(dtype=float),
    coord_start: int,
    dof_start: int, 
    body_f_s: og.spatial_vector, 
    tau: og.array(dtype=float)):

    # prismatic / revolute
    if (type == 0 or type == 1):
        S_s = og.load(joint_S_s, dof_start)

        q = og.load(joint_q, coord_start)
        qd = og.load(joint_qd, dof_start)
        act = og.load(joint_act, dof_start)

        target = og.load(joint_target, coord_start)
        lower = og.load(joint_limit_lower, coord_start)
        upper = og.load(joint_limit_upper, coord_start)

        limit_f = 0.0

        # compute limit forces, damping only active when limit is violated
        if (q < lower):
            limit_f = limit_k_e*(lower-q) - limit_k_d*min(qd, 0.0)

        if (q > upper):
            limit_f = limit_k_e*(upper-q) - limit_k_d*max(qd, 0.0)

        # total torque / force on the joint
        t = 0.0 - spatial_dot(S_s, body_f_s) - target_k_e*(q - target) - target_k_d*qd + act + limit_f

        og.store(tau, dof_start, t)

    # ball
    if (type == 2):

        # elastic term.. this is proportional to the 
        # imaginary part of the relative quaternion
        r_j = vec3(og.load(joint_q, coord_start + 0),  
                     og.load(joint_q, coord_start + 1), 
                     og.load(joint_q, coord_start + 2))                     

        # angular velocity for damping
        w_j = vec3(og.load(joint_qd, dof_start + 0),  
                     og.load(joint_qd, dof_start + 1), 
                     og.load(joint_qd, dof_start + 2))

        for i in range(0, 3):
            S_s = og.load(joint_S_s, dof_start+i)

            w = w_j[i]
            r = r_j[i]

            og.store(tau, dof_start+i, 0.0 - spatial_dot(S_s, body_f_s) - w*target_k_d - r*target_k_e)

    # fixed
    # if (type == 3)
    #    pass

    # free
    if (type == 4):
            
        for i in range(0, 6):
            S_s = og.load(joint_S_s, dof_start+i)
            og.store(tau, dof_start+i, 0.0 - spatial_dot(S_s, body_f_s))

    return 0


@og.func
def jcalc_integrate(
    type: int,
    joint_q: og.array(dtype=float),
    joint_qd: og.array(dtype=float),
    joint_qdd: og.array(dtype=float),
    coord_start: int,
    dof_start: int,
    dt: float,
    joint_q_new: og.array(dtype=float),
    joint_qd_new: og.array(dtype=float)):

    # prismatic / revolute
    if (type == 0 or type == 1):

        qdd = og.load(joint_qdd, dof_start)
        qd = og.load(joint_qd, dof_start)
        q = og.load(joint_q, coord_start)

        qd_new = qd + qdd*dt
        q_new = q + qd_new*dt

        og.store(joint_qd_new, dof_start, qd_new)
        og.store(joint_q_new, coord_start, q_new)

    # ball
    if (type == 2):

        m_j = vec3(og.load(joint_qdd, dof_start + 0),
                     og.load(joint_qdd, dof_start + 1),
                     og.load(joint_qdd, dof_start + 2))

        w_j = vec3(og.load(joint_qd, dof_start + 0),  
                     og.load(joint_qd, dof_start + 1), 
                     og.load(joint_qd, dof_start + 2)) 

        r_j = quat(og.load(joint_q, coord_start + 0), 
                   og.load(joint_q, coord_start + 1), 
                   og.load(joint_q, coord_start + 2), 
                   og.load(joint_q, coord_start + 3))

        # symplectic Euler
        w_j_new = w_j + m_j*dt

        drdt_j = mul(quat(w_j_new, 0.0), r_j) * 0.5

        # new orientation (normalized)
        r_j_new = normalize(r_j + drdt_j * dt)

        # update joint coords
        og.store(joint_q_new, coord_start + 0, r_j_new[0])
        og.store(joint_q_new, coord_start + 1, r_j_new[1])
        og.store(joint_q_new, coord_start + 2, r_j_new[2])
        og.store(joint_q_new, coord_start + 3, r_j_new[3])

        # update joint vel
        og.store(joint_qd_new, dof_start + 0, w_j_new[0])
        og.store(joint_qd_new, dof_start + 1, w_j_new[1])
        og.store(joint_qd_new, dof_start + 2, w_j_new[2])

    # fixed joint
    #if (type == 3)
    #    pass

    # free joint
    if (type == 4):

        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear acceleration
        m_s = vec3(og.load(joint_qdd, dof_start + 0),
                     og.load(joint_qdd, dof_start + 1),
                     og.load(joint_qdd, dof_start + 2))

        a_s = vec3(og.load(joint_qdd, dof_start + 3), 
                     og.load(joint_qdd, dof_start + 4), 
                     og.load(joint_qdd, dof_start + 5))

        # angular and linear velocity
        w_s = vec3(og.load(joint_qd, dof_start + 0),  
                     og.load(joint_qd, dof_start + 1), 
                     og.load(joint_qd, dof_start + 2))
        
        v_s = vec3(og.load(joint_qd, dof_start + 3),
                     og.load(joint_qd, dof_start + 4),
                     og.load(joint_qd, dof_start + 5))

        # symplectic Euler
        w_s = w_s + m_s*dt
        v_s = v_s + a_s*dt
        
        # translation of origin
        p_s = vec3(og.load(joint_q, coord_start + 0),
                     og.load(joint_q, coord_start + 1), 
                     og.load(joint_q, coord_start + 2))

        # linear vel of origin (note q/qd switch order of linear angular elements) 
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velcity
        dpdt_s = v_s + cross(w_s, p_s)
        
        # quat and quat derivative
        r_s = quat(og.load(joint_q, coord_start + 3), 
                   og.load(joint_q, coord_start + 4), 
                   og.load(joint_q, coord_start + 5), 
                   og.load(joint_q, coord_start + 6))

        drdt_s = mul(quat(w_s, 0.0), r_s) * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = normalize(r_s + drdt_s * dt)

        # update transform
        og.store(joint_q_new, coord_start + 0, p_s_new[0])
        og.store(joint_q_new, coord_start + 1, p_s_new[1])
        og.store(joint_q_new, coord_start + 2, p_s_new[2])

        og.store(joint_q_new, coord_start + 3, r_s_new[0])
        og.store(joint_q_new, coord_start + 4, r_s_new[1])
        og.store(joint_q_new, coord_start + 5, r_s_new[2])
        og.store(joint_q_new, coord_start + 6, r_s_new[3])

        # update joint_twist
        og.store(joint_qd_new, dof_start + 0, w_s[0])
        og.store(joint_qd_new, dof_start + 1, w_s[1])
        og.store(joint_qd_new, dof_start + 2, w_s[2])
        og.store(joint_qd_new, dof_start + 3, v_s[0])
        og.store(joint_qd_new, dof_start + 4, v_s[1])
        og.store(joint_qd_new, dof_start + 5, v_s[2])

    return 0

@og.func
def compute_link_transform(i: int,
                           joint_type: og.array(dtype=int),
                           joint_parent: og.array(dtype=int),
                           joint_q_start: og.array(dtype=int),
                           joint_qd_start: og.array(dtype=int),
                           joint_q: og.array(dtype=float),
                           joint_X_pj: og.array(dtype=og.spatial_transform),
                           joint_X_cm: og.array(dtype=og.spatial_transform),
                           joint_axis: og.array(dtype=og.vec3),
                           body_X_sc: og.array(dtype=og.spatial_transform),
                           body_X_sm: og.array(dtype=og.spatial_transform)):

    # parent transform
    parent = load(joint_parent, i)

    # parent transform in spatial coordinates
    X_sp = spatial_transform_identity()
    if (parent >= 0):
        X_sp = load(body_X_sc, parent)

    type = load(joint_type, i)
    axis = load(joint_axis, i)
    coord_start = load(joint_q_start, i)
    dof_start = load(joint_qd_start, i)

    # compute transform across joint
    X_jc = jcalc_transform(type, axis, joint_q, coord_start)

    X_pj = load(joint_X_pj, i)
    X_sc = spatial_transform_multiply(X_sp, spatial_transform_multiply(X_pj, X_jc))

    # compute transform of center of mass
    X_cm = load(joint_X_cm, i)
    X_sm = spatial_transform_multiply(X_sc, X_cm)

    # store geometry transforms
    store(body_X_sc, i, X_sc)
    store(body_X_sm, i, X_sm)

    return 0


@og.kernel
def eval_rigid_fk(articulation_start: og.array(dtype=int),
                  joint_type: og.array(dtype=int),
                  joint_parent: og.array(dtype=int),
                  joint_q_start: og.array(dtype=int),
                  joint_qd_start: og.array(dtype=int),
                  joint_q: og.array(dtype=float),
                  joint_X_pj: og.array(dtype=og.spatial_transform),
                  joint_X_cm: og.array(dtype=og.spatial_transform),
                  joint_axis: og.array(dtype=og.vec3),
                  body_X_sc: og.array(dtype=og.spatial_transform),
                  body_X_sm: og.array(dtype=og.spatial_transform)):

    # one thread per-articulation
    index = tid()

    start = og.load(articulation_start, index)
    end = og.load(articulation_start, index+1)

    for i in range(start, end):
        compute_link_transform(i,
                               joint_type,
                               joint_parent,
                               joint_q_start,
                               joint_qd_start,
                               joint_q,
                               joint_X_pj,
                               joint_X_cm,
                               joint_axis,
                               body_X_sc,
                               body_X_sm)




@og.func
def compute_link_velocity(i: int,
                          joint_type: og.array(dtype=int),
                          joint_parent: og.array(dtype=int),
                          joint_qd_start: og.array(dtype=int),
                          joint_qd: og.array(dtype=float),
                          joint_axis: og.array(dtype=og.vec3),
                          body_I_m: og.array(dtype=og.spatial_matrix),
                          body_X_sc: og.array(dtype=og.spatial_transform),
                          body_X_sm: og.array(dtype=og.spatial_transform),
                          joint_X_pj: og.array(dtype=og.spatial_transform),
                          gravity: og.vec3,
                          # outputs
                          joint_S_s: og.array(dtype=og.spatial_vector),
                          body_I_s: og.array(dtype=og.spatial_matrix),
                          body_v_s: og.array(dtype=og.spatial_vector),
                          body_f_s: og.array(dtype=og.spatial_vector),
                          body_a_s: og.array(dtype=og.spatial_vector)):

    type = og.load(joint_type, i)
    axis = og.load(joint_axis, i)
    parent = og.load(joint_parent, i)
    dof_start = og.load(joint_qd_start, i)
    
    X_sc = og.load(body_X_sc, i)

    # parent transform in spatial coordinates
    X_sp = spatial_transform_identity()
    if (parent >= 0):
        X_sp = load(body_X_sc, parent)

    X_pj = load(joint_X_pj, i)
    X_sj = spatial_transform_multiply(X_sp, X_pj)

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    v_j_s = jcalc_motion(type, axis, X_sj, joint_S_s, joint_qd, dof_start)

    # parent velocity
    v_parent_s = og.spatial_vector()
    a_parent_s = og.spatial_vector()

    if (parent >= 0):
        v_parent_s = og.load(body_v_s, parent)
        a_parent_s = og.load(body_a_s, parent)

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s) # + self.joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = og.load(body_X_sm, i)
    I_m = og.load(body_I_m, i)

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    m = I_m[3, 3]

    f_g_m = og.spatial_vector(vec3(), gravity) * m
    f_g_s = spatial_transform_wrench(spatial_transform(spatial_transform_get_translation(X_sm), quat_identity()), f_g_m)

    #f_ext_s = og.load(body_f_s, i) + f_g_s

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = og.mul(I_s, a_s) + spatial_cross_dual(v_s, og.mul(I_s, v_s))

    og.store(body_v_s, i, v_s)
    og.store(body_a_s, i, a_s)
    og.store(body_f_s, i, f_b_s - f_g_s)
    og.store(body_I_s, i, I_s)

    return 0


@og.func
def compute_link_tau(offset: int,
                     joint_end: int,
                     joint_type: og.array(dtype=int),
                     joint_parent: og.array(dtype=int),
                     joint_q_start: og.array(dtype=int),
                     joint_qd_start: og.array(dtype=int),
                     joint_q: og.array(dtype=float),
                     joint_qd: og.array(dtype=float),
                     joint_act: og.array(dtype=float),
                     joint_target: og.array(dtype=float),
                     joint_target_ke: og.array(dtype=float),
                     joint_target_kd: og.array(dtype=float),
                     joint_limit_lower: og.array(dtype=float),
                     joint_limit_upper: og.array(dtype=float),
                     joint_limit_ke: og.array(dtype=float),
                     joint_limit_kd: og.array(dtype=float),
                     joint_S_s: og.array(dtype=og.spatial_vector),
                     body_fb_s: og.array(dtype=og.spatial_vector),
                     # outputs
                     body_ft_s: og.array(dtype=og.spatial_vector),
                     tau: og.array(dtype=float)):

    # for backwards traversal
    i = joint_end-offset-1

    type = og.load(joint_type, i)
    parent = og.load(joint_parent, i)
    dof_start = og.load(joint_qd_start, i)
    coord_start = og.load(joint_q_start, i)

    target_k_e = og.load(joint_target_ke, i)
    target_k_d = og.load(joint_target_kd, i)

    limit_k_e = og.load(joint_limit_ke, i)
    limit_k_d = og.load(joint_limit_kd, i)

    # total forces on body
    f_b_s = og.load(body_fb_s, i)
    f_t_s = og.load(body_ft_s, i)

    f_s = f_b_s + f_t_s

    # compute joint-space forces, writes out tau
    jcalc_tau(type, target_k_e, target_k_d, limit_k_e, limit_k_d, joint_S_s, joint_q, joint_qd, joint_act, joint_target, joint_limit_lower, joint_limit_upper, coord_start, dof_start, f_s, tau)

    # update parent forces, todo: check that this is valid for the backwards pass
    if (parent >= 0):
        og.atomic_add(body_ft_s, parent, f_s)

    return 0


@og.kernel
def eval_rigid_id(articulation_start: og.array(dtype=int),
                  joint_type: og.array(dtype=int),
                  joint_parent: og.array(dtype=int),
                  joint_q_start: og.array(dtype=int),
                  joint_qd_start: og.array(dtype=int),
                  joint_q: og.array(dtype=float),
                  joint_qd: og.array(dtype=float),
                  joint_axis: og.array(dtype=og.vec3),
                  joint_target_ke: og.array(dtype=float),
                  joint_target_kd: og.array(dtype=float),             
                  body_I_m: og.array(dtype=og.spatial_matrix),
                  body_X_sc: og.array(dtype=og.spatial_transform),
                  body_X_sm: og.array(dtype=og.spatial_transform),
                  joint_X_pj: og.array(dtype=og.spatial_transform),
                  gravity: og.vec3,
                  # outputs
                  joint_S_s: og.array(dtype=og.spatial_vector),
                  body_I_s: og.array(dtype=og.spatial_matrix),
                  body_v_s: og.array(dtype=og.spatial_vector),
                  body_f_s: og.array(dtype=og.spatial_vector),
                  body_a_s: og.array(dtype=og.spatial_vector)):

    # one thread per-articulation
    index = tid()

    start = og.load(articulation_start, index)
    end = og.load(articulation_start, index+1)
    count = end-start

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(
            i,
            joint_type,
            joint_parent,
            joint_qd_start,
            joint_qd,
            joint_axis,
            body_I_m,
            body_X_sc,
            body_X_sm,
            joint_X_pj,
            gravity,
            joint_S_s,
            body_I_s,
            body_v_s,
            body_f_s,
            body_a_s)


@og.kernel
def eval_rigid_tau(articulation_start: og.array(dtype=int),
                  joint_type: og.array(dtype=int),
                  joint_parent: og.array(dtype=int),
                  joint_q_start: og.array(dtype=int),
                  joint_qd_start: og.array(dtype=int),
                  joint_q: og.array(dtype=float),
                  joint_qd: og.array(dtype=float),
                  joint_act: og.array(dtype=float),
                  joint_target: og.array(dtype=float),
                  joint_target_ke: og.array(dtype=float),
                  joint_target_kd: og.array(dtype=float),
                  joint_limit_lower: og.array(dtype=float),
                  joint_limit_upper: og.array(dtype=float),
                  joint_limit_ke: og.array(dtype=float),
                  joint_limit_kd: og.array(dtype=float),
                  joint_axis: og.array(dtype=og.vec3),
                  joint_S_s: og.array(dtype=og.spatial_vector),
                  body_fb_s: og.array(dtype=og.spatial_vector),                  
                  # outputs
                  body_ft_s: og.array(dtype=og.spatial_vector),
                  tau: og.array(dtype=float)):

    # one thread per-articulation
    index = tid()

    start = og.load(articulation_start, index)
    end = og.load(articulation_start, index+1)
    count = end-start

    # compute joint forces
    for i in range(0, count):
        compute_link_tau(
            i,
            end,
            joint_type,
            joint_parent,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_qd,
            joint_act,
            joint_target,
            joint_target_ke,
            joint_target_kd,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_S_s,
            body_fb_s,
            body_ft_s,
            tau)

@og.kernel
def eval_rigid_jacobian(
    articulation_start: og.array(dtype=int),
    articulation_J_start: og.array(dtype=int),
    joint_parent: og.array(dtype=int),
    joint_qd_start: og.array(dtype=int),
    joint_S_s: og.array(dtype=og.spatial_vector),
    # outputs
    J: og.array(dtype=float)):

    # one thread per-articulation
    index = tid()

    joint_start = og.load(articulation_start, index)
    joint_end = og.load(articulation_start, index+1)
    joint_count = joint_end-joint_start

    J_offset = og.load(articulation_J_start, index)

    # in spatial.h
    spatial_jacobian(joint_S_s, joint_parent, joint_qd_start, joint_start, joint_count, J_offset, J)


# @og.kernel
# def eval_rigid_jacobian(
#     articulation_start: og.array(dtype=int),
#     articulation_J_start: og.array(dtype=int),    
#     joint_parent: og.array(dtype=int),
#     joint_qd_start: og.array(dtype=int),
#     joint_S_s: og.array(dtype=og.spatial_vector),
#     # outputs
#     J: og.array(dtype=float)):

#     # one thread per-articulation
#     index = tid()

#     joint_start = og.load(articulation_start, index)
#     joint_end = og.load(articulation_start, index+1)
#     joint_count = joint_end-joint_start

#     dof_start = og.load(joint_qd_start, joint_start)
#     dof_end = og.load(joint_qd_start, joint_end)
#     dof_count = dof_end-dof_start

#     #(const og.spatial_vector* S, const int* joint_parents, const int* joint_qd_start, int num_links, int num_dofs, float* J)
#     spatial_jacobian(joint_S_s, joint_parent, joint_qd_start, joint_count, dof_count, J)



@og.kernel
def eval_rigid_mass(
    articulation_start: og.array(dtype=int),
    articulation_M_start: og.array(dtype=int),    
    body_I_s: og.array(dtype=og.spatial_matrix),
    # outputs
    M: og.array(dtype=float)):

    # one thread per-articulation
    index = tid()

    joint_start = og.load(articulation_start, index)
    joint_end = og.load(articulation_start, index+1)
    joint_count = joint_end-joint_start

    M_offset = og.load(articulation_M_start, index)

    # in spatial.h
    spatial_mass(body_I_s, joint_start, joint_count, M_offset, M)

@og.kernel
def eval_dense_gemm(m: int, n: int, p: int, t1: int, t2: int, A: og.array(dtype=float), B: og.array(dtype=float), C: og.array(dtype=float)):
    dense_gemm(m, n, p, t1, t2, A, B, C)

@og.kernel
def eval_dense_gemm_batched(m: og.array(dtype=int), n: og.array(dtype=int), p: og.array(dtype=int), t1: int, t2: int, A_start: og.array(dtype=int), B_start: og.array(dtype=int), C_start: og.array(dtype=int), A: og.array(dtype=float), B: og.array(dtype=float), C: og.array(dtype=float)):
    dense_gemm_batched(m, n, p, t1, t2, A_start, B_start, C_start, A, B, C)

@og.kernel
def eval_dense_cholesky(n: int, A: og.array(dtype=float), regularization: float, L: og.array(dtype=float)):
    dense_chol(n, A, regularization, L)

@og.kernel
def eval_dense_cholesky_batched(A_start: og.array(dtype=int), A_dim: og.array(dtype=int), A: og.array(dtype=float), regularization: float, L: og.array(dtype=float)):
    dense_chol_batched(A_start, A_dim, A, regularization, L)

@og.kernel
def eval_dense_subs(n: int, L: og.array(dtype=float), b: og.array(dtype=float), x: og.array(dtype=float)):
    dense_subs(n, L, b, x)

# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@og.kernel
def eval_dense_solve(n: int, A: og.array(dtype=float), L: og.array(dtype=float), b: og.array(dtype=float), x: og.array(dtype=float)):
    dense_solve(n, A, L, b, x)

# helper that propagates gradients back to A, treating L as a constant / temporary variable
# allows us to reuse the Cholesky decomposition from the forward pass
@og.kernel
def eval_dense_solve_batched(b_start: og.array(dtype=int), A_start: og.array(dtype=int), A_dim: og.array(dtype=int), A: og.array(dtype=float), L: og.array(dtype=float), b: og.array(dtype=float), x: og.array(dtype=float)):
    dense_solve_batched(b_start, A_start, A_dim, A, L, b, x)


@og.kernel
def eval_rigid_integrate(
    joint_type: og.array(dtype=int),
    joint_q_start: og.array(dtype=int),
    joint_qd_start: og.array(dtype=int),
    joint_q: og.array(dtype=float),
    joint_qd: og.array(dtype=float),
    joint_qdd: og.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: og.array(dtype=float),
    joint_qd_new: og.array(dtype=float)):

    # one thread per-articulation
    index = tid()

    type = og.load(joint_type, index)
    coord_start = og.load(joint_q_start, index)
    dof_start = og.load(joint_qd_start, index)

    jcalc_integrate(
        type,
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        dt,
        joint_q_new,
        joint_qd_new)


def compute_forces(model, state, particle_f, rigid_f):

    # damped springs
    if (model.spring_count):

        og.launch(kernel=eval_springs,
                    dim=model.spring_count,
                    inputs=[
                        state.particle_q, 
                        state.particle_qd, 
                        model.spring_indices, 
                        model.spring_rest_length, 
                        model.spring_stiffness, 
                        model.spring_damping
                    ],
                    outputs=[particle_f],
                    device=model.device)

    # triangle elastic and lift/drag forces
    if (model.tri_count and model.tri_ke > 0.0):

        og.launch(kernel=eval_triangles,
                    dim=model.tri_count,
                    inputs=[
                        state.particle_q,
                        state.particle_qd,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_activations,
                        model.tri_ke,
                        model.tri_ka,
                        model.tri_kd,
                        model.tri_drag,
                        model.tri_lift
                    ],
                    outputs=[particle_f],
                    device=model.device)

    # triangle/triangle contacts
    if (model.enable_tri_collisions and model.tri_count and model.tri_ke > 0.0):

        og.launch(kernel=eval_triangles_contact,
                    dim=model.tri_count * model.particle_count,
                    inputs=[
                        model.particle_count,
                        state.particle_q,
                        state.particle_qd,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_activations,
                        model.tri_ke,
                        model.tri_ka,
                        model.tri_kd,
                        model.tri_drag,
                        model.tri_lift
                    ],
                    outputs=[particle_f],
                    device=model.device)

    # triangle bending
    if (model.edge_count):

        og.launch(kernel=eval_bending,
                    dim=model.edge_count,
                    inputs=[state.particle_q, state.particle_qd, model.edge_indices, model.edge_rest_angle, model.edge_ke, model.edge_kd],
                    outputs=[particle_f],
                    device=model.device)

    # particle ground contact
    if (model.ground and model.particle_count):

        og.launch(kernel=eval_contacts,
                    dim=model.particle_count,
                    inputs=[state.particle_q, state.particle_qd, model.contact_ke, model.contact_kd, model.contact_kf, model.contact_mu, model.contact_distance, model.ground_plane],
                    outputs=[particle_f],
                    device=model.device)

    # tetrahedral FEM
    if (model.tet_count):

        og.launch(kernel=eval_tetrahedra,
                    dim=model.tet_count,
                    inputs=[state.particle_q, state.particle_qd, model.tet_indices, model.tet_poses, model.tet_activations, model.tet_materials],
                    outputs=[particle_f],
                    device=model.device)


    if (model.articulation_count):
        
        if (model.ground and model.contact_count > 0):
            
            # evaluate contact forces
            og.launch(
                kernel=eval_rigid_contacts_art,
                dim=model.contact_count,
                inputs=[
                    state.body_X_sc,
                    state.body_v_s,
                    model.contact_body0,
                    model.contact_point0,
                    model.contact_dist,
                    model.contact_material,
                    model.shape_materials
                ],
                outputs=[
                    rigid_f
                ],
                device=model.device,
                preserve_output=True)


    # particle shape contact
    if (model.particle_count):
        
        if (model.link_count == 0):

            # if no links then just pass empty tensors for the body properties
            og.launch(kernel=eval_soft_contacts,
                        dim=model.particle_count*model.shape_count,
                        inputs=[
                            model.particle_count,
                            state.particle_q, 
                            state.particle_qd,
                            None,
                            None,
                            model.shape_transform,
                            model.shape_body,
                            model.shape_geo_type, 
                            model.shape_geo_id,
                            model.shape_geo_scale,
                            model.shape_materials,
                            model.contact_ke,
                            model.contact_kd, 
                            model.contact_kf, 
                            model.contact_mu,
                            model.contact_distance,
                            model.contact_margin],
                            # outputs
                        outputs=[
                            particle_f,
                            None],
                        device=model.device)
        else:

            og.launch(kernel=eval_soft_contacts,
                        dim=model.particle_count*model.shape_count,
                        inputs=[
                            model.particle_count,
                            state.particle_q, 
                            state.particle_qd,
                            state.body_X_sc,
                            state.body_v_s,
                            model.shape_transform,
                            model.shape_body,
                            model.shape_geo_type, 
                            model.shape_geo_id,
                            model.shape_geo_scale,
                            model.shape_materials,
                            model.contact_ke,
                            model.contact_kd, 
                            model.contact_kf, 
                            model.contact_mu,
                            model.contact_distance,
                            model.contact_margin],
                            # outputs
                        outputs=[
                            particle_f,
                            rigid_f],
                        device=model.device)

    # evaluate muscle actuation
    if (model.muscle_count):
        
        og.launch(
            kernel=eval_muscles,
            dim=model.muscle_count,
            inputs=[
                state.body_X_sc,
                state.body_v_s,
                model.muscle_start,
                model.muscle_params,
                model.muscle_links,
                model.muscle_points,
                model.muscle_activation
            ],
            outputs=[
                rigid_f
            ],
            device=model.device,
            preserve_output=True)


    if (model.articulation_count):
        
        # evaluate joint torques
        og.launch(
            kernel=eval_rigid_tau,
            dim=model.articulation_count,
            inputs=[
                model.articulation_joint_start,
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state.joint_q,
                state.joint_qd,
                state.joint_act,
                model.joint_target,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                model.joint_axis,
                state.joint_S_s,
                state.body_f_s
            ],
            outputs=[
                state.body_ft_s,
                state.joint_tau
            ],
            device=model.device,
            preserve_output=True)





class SemiImplicitIntegrator:
    """A semi-implicit integrator using symplectic Euler

    After constructing `Model` and `State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Semi-implicit time integration is a variational integrator that 
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example:

        >>> integrator = og.SemiImplicitIntegrator()
        >>>
        >>> # simulation loop
        >>> for i in range(100):
        >>>     state = integrator.forward(model, state, dt)

    """

    def __init__(self):
        pass


    def simulate(self, model, state, state_out, dt):

        with og.ScopedTimer("simulate", False):

            # alloc particle force buffer
            if (model.particle_count):
                state_out.particle_f.zero_()

            if (model.link_count):
                state_out.rigid_f = og.zeros((model.link_count, 6), dtype=og.spatial_vector, device=model.device, requires_grad=True)
                state_out.body_f_ext_s = og.zeros((model.link_count, 6), dtype=og.spatial_vector, device=model.device, requires_grad=True)


            compute_forces(model, state, state_out.particle_f, None)

            #-------------------------------------
            # integrate rigids

            if (model.articulation_count):

                og.launch(
                    kernel=eval_rigid_integrate,
                    dim=model.link_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state.joint_q,
                        state.joint_qd,
                        state_out.joint_qdd,
                        dt
                    ],
                    outputs=[
                        state_out.joint_q,
                        state_out.joint_qd
                    ],
                    device=model.device)

            #----------------------------
            # integrate particles

            if (model.particle_count):

                og.launch(
                    kernel=integrate_particles,
                    dim=model.particle_count,
                    inputs=[
                        state.particle_q, 
                        state.particle_qd, 
                        state_out.particle_f, 
                        model.particle_inv_mass, 
                        model.gravity, 
                        dt
                    ],
                    outputs=[
                        state_out.particle_q, 
                        state_out.particle_qd
                        ],
                    device=model.device)

            return state_out


@og.kernel
def compute_particle_residual(particle_qd_0: og.array(dtype=og.vec3),
                            particle_qd_1: og.array(dtype=og.vec3),
                            particle_f: og.array(dtype=og.vec3),
                            particle_m: og.array(dtype=float),
                            gravity: og.vec3,
                            dt: float,
                            residual: og.array(dtype=og.vec3)):

    tid = og.tid()

    m = og.load(particle_m, tid)
    v1 = og.load(particle_qd_1, tid)
    v0 = og.load(particle_qd_0, tid)
    f = og.load(particle_f, tid)

    err = og.vec3()

    if (m > 0.0):
        #err = (v1-v0)*m - f*dt - gravity*dt*m   
        #invm = 1.0/(m + 1.e+3*dt*dt*16.0)
        #err = (v1-v0)*m - f*dt - gravity*dt*m
        #err = err*invm
        err = (v1-v0)*m - f*dt - gravity*dt*m

    og.store(residual, tid, err)
 

@og.kernel
def update_particle_position(
    particle_q_0: og.array(dtype=og.vec3),
    particle_q_1: og.array(dtype=og.vec3),
    particle_qd_1: og.array(dtype=og.vec3),
    x: og.array(dtype=og.vec3),
    dt: float):

    tid = og.tid()

    qd_1 = og.load(x, tid)
    
    q_0 = og.load(particle_q_0, tid)
    q_1 = q_0 + qd_1*dt

    og.store(particle_q_1, tid, q_1)
    og.store(particle_qd_1, tid, qd_1)



def compute_residual(model, state_in, state_out, particle_f, residual, dt):

    og.launch(
        kernel=compute_particle_residual,
        dim=model.particle_count,
        inputs=[
            state_in.particle_qd,
            state_out.particle_qd,
            particle_f,
            model.particle_mass,
            model.gravity,
            dt,
            residual.astype(dtype=og.vec3)
        ], 
        device=model.device)

def init_state(model, state_in, state_out, dt):

    og.launch(
        kernel=integrate_particles,
        dim=model.particle_count,
        inputs=[
            state_in.particle_q, 
            state_in.particle_qd, 
            state_in.particle_f, 
            model.particle_inv_mass, 
            model.gravity, 
            dt
        ],
        outputs=[
            state_out.particle_q, 
            state_out.particle_qd
            ],
        device=model.device)


# compute the final positions given output velocity (x)
def update_state(model, state_in, state_out, x, dt):

    og.launch(
        kernel=update_particle_position,
        dim=model.particle_count,
        inputs=[
            state_in.particle_q,
            state_out.particle_q,
            state_out.particle_qd,
            x,
            dt
        ],
        device=model.device)



class VariationalImplicitIntegrator:

    def __init__(self, model, solver="gd", alpha=0.1, max_iters=32, report=False):

        self.solver = solver
        self.alpha = alpha
        self.max_iters = max_iters
        self.report = report

        self.opt = Optimizer(model.particle_count*3, mode=self.solver, device=model.device)

        # allocate temporary space for evaluating particle forces
        self.particle_f = og.zeros(model.particle_count, dtype=og.vec3, device=model.device)

    def simulate(self, model, state_in, state_out, dt): 

        if (state_in is state_out):
            raise RuntimeError("Implicit integrators require state objects to not alias each other")


        with og.ScopedTimer("simulate", False):

            # alloc particle force buffer
            if (model.particle_count):
                
                def residual_func(x, dfdx):

                    self.particle_f.zero_()

                    update_state(model, state_in, state_out, x.astype(og.vec3), dt)
                    compute_forces(model, state_out, self.particle_f, None)
                    compute_residual(model, state_in, state_out, self.particle_f, dfdx, dt)


                # initialize oututput state using the input velocity to create 'predicted state'
                init_state(model, state_in, state_out, dt)

                # our optimization variable
                x = state_out.particle_qd.astype(dtype=float)

                self.opt.solve(
                    x=x, 
                    grad_func=residual_func,
                    max_iters=self.max_iters,
                    alpha=self.alpha,
                    report=self.report)

                # final update to output state with solved velocity
                update_state(model, state_in, state_out, x.astype(og.vec3), dt)
  

            return state_out