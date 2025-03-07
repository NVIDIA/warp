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

import warp as wp

from .articulation import (
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
    eval_fk,
)
from .integrator import Integrator
from .integrator_euler import (
    eval_bending_forces,
    eval_joint_force,
    eval_muscle_forces,
    eval_particle_body_contact_forces,
    eval_particle_forces,
    eval_particle_ground_contact_forces,
    eval_rigid_contacts,
    eval_spring_forces,
    eval_tetrahedral_forces,
    eval_triangle_contact_forces,
    eval_triangle_forces,
)
from .model import Control, Model, State


# Frank & Park definition 3.20, pg 100
@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def transform_wrench(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    v = wp.quat_rotate(q, v)
    w = wp.quat_rotate(q, w) + wp.cross(p, v)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_adjoint(R: wp.mat33, S: wp.mat33):
    # T = [R  0]
    #     [S  R]

    # fmt: off
    return wp.spatial_matrix(
        R[0, 0], R[0, 1], R[0, 2],     0.0,     0.0,     0.0,
        R[1, 0], R[1, 1], R[1, 2],     0.0,     0.0,     0.0,
        R[2, 0], R[2, 1], R[2, 2],     0.0,     0.0,     0.0,
        S[0, 0], S[0, 1], S[0, 2], R[0, 0], R[0, 1], R[0, 2],
        S[1, 0], S[1, 1], S[1, 2], R[1, 0], R[1, 1], R[1, 2],
        S[2, 0], S[2, 1], S[2, 2], R[2, 0], R[2, 1], R[2, 2],
    )
    # fmt: on


@wp.kernel
def compute_spatial_inertia(
    body_inertia: wp.array(dtype=wp.mat33),
    body_mass: wp.array(dtype=float),
    # outputs
    body_I_m: wp.array(dtype=wp.spatial_matrix),
):
    tid = wp.tid()
    I = body_inertia[tid]
    m = body_mass[tid]
    # fmt: off
    body_I_m[tid] = wp.spatial_matrix(
        I[0, 0], I[0, 1], I[0, 2], 0.0, 0.0, 0.0,
        I[1, 0], I[1, 1], I[1, 2], 0.0, 0.0, 0.0,
        I[2, 0], I[2, 1], I[2, 2], 0.0, 0.0, 0.0,
        0.0,     0.0,     0.0,     m,   0.0, 0.0,
        0.0,     0.0,     0.0,     0.0, m,   0.0,
        0.0,     0.0,     0.0,     0.0, 0.0, m,
    )
    # fmt: on


@wp.kernel
def compute_com_transforms(
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_X_com: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())


# computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
@wp.func
def spatial_transform_inertia(t: wp.transform, I: wp.spatial_matrix):
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.skew(p) @ R

    T = spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


# compute transform across a joint
@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array(dtype=float),
    start: int,
):
    if type == wp.sim.JOINT_PRISMATIC:
        q = joint_q[start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    if type == wp.sim.JOINT_REVOLUTE:
        q = joint_q[start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
        return X_jc

    if type == wp.sim.JOINT_BALL:
        qx = joint_q[start + 0]
        qy = joint_q[start + 1]
        qz = joint_q[start + 2]
        qw = joint_q[start + 3]

        X_jc = wp.transform(wp.vec3(), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == wp.sim.JOINT_FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        px = joint_q[start + 0]
        py = joint_q[start + 1]
        pz = joint_q[start + 2]

        qx = joint_q[start + 3]
        qy = joint_q[start + 4]
        qz = joint_q[start + 5]
        qw = joint_q[start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == wp.sim.JOINT_COMPOUND:
        rot, _ = compute_3d_rotational_dofs(
            joint_axis[axis_start],
            joint_axis[axis_start + 1],
            joint_axis[axis_start + 2],
            joint_q[start + 0],
            joint_q[start + 1],
            joint_q[start + 2],
            0.0,
            0.0,
            0.0,
        )

        X_jc = wp.transform(wp.vec3(), rot)
        return X_jc

    if type == wp.sim.JOINT_UNIVERSAL:
        rot, _ = compute_2d_rotational_dofs(
            joint_axis[axis_start],
            joint_axis[axis_start + 1],
            joint_q[start + 0],
            joint_q[start + 1],
            0.0,
            0.0,
        )

        X_jc = wp.transform(wp.vec3(), rot)
        return X_jc

    if type == wp.sim.JOINT_D6:
        pos = wp.vec3(0.0)
        rot = wp.quat_identity()

        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            pos += axis * joint_q[start + 0]
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            pos += axis * joint_q[start + 1]
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            pos += axis * joint_q[start + 2]

        ia = axis_start + lin_axis_count
        iq = start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[ia]
            rot = wp.quat_from_axis_angle(axis, joint_q[iq])
        if ang_axis_count == 2:
            rot, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_q[iq + 0],
                joint_q[iq + 1],
                0.0,
                0.0,
            )
        if ang_axis_count == 3:
            rot, _ = compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
                joint_q[iq + 2],
                0.0,
                0.0,
                0.0,
            )

        X_jc = wp.transform(pos, rot)
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    q_start: int,
    qd_start: int,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
):
    if type == wp.sim.JOINT_PRISMATIC:
        axis = joint_axis[axis_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == wp.sim.JOINT_REVOLUTE:
        axis = joint_axis[axis_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == wp.sim.JOINT_UNIVERSAL:
        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))
        local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
        local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))

        axis_0 = local_0
        q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start + 0])

        axis_1 = wp.quat_rotate(q_0, local_1)

        S_0 = transform_twist(X_sc, wp.spatial_vector(axis_0, wp.vec3()))
        S_1 = transform_twist(X_sc, wp.spatial_vector(axis_1, wp.vec3()))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1]

    if type == wp.sim.JOINT_COMPOUND:
        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        axis_2 = joint_axis[axis_start + 2]
        q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2))
        local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
        local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
        local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

        axis_0 = local_0
        q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start + 0])

        axis_1 = wp.quat_rotate(q_0, local_1)
        q_1 = wp.quat_from_axis_angle(axis_1, joint_q[q_start + 1])

        axis_2 = wp.quat_rotate(q_1 * q_0, local_2)

        S_0 = transform_twist(X_sc, wp.spatial_vector(axis_0, wp.vec3()))
        S_1 = transform_twist(X_sc, wp.spatial_vector(axis_1, wp.vec3()))
        S_2 = transform_twist(X_sc, wp.spatial_vector(axis_2, wp.vec3()))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == wp.sim.JOINT_D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = joint_axis[axis_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0]
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = joint_axis[axis_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1]
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = joint_axis[axis_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2]
            joint_S_s[qd_start + lin_axis_count + 2] = S_s

        return v_j_s

    if type == wp.sim.JOINT_BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == wp.sim.JOINT_FIXED:
        return wp.spatial_vector()

    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        v_j_s = transform_twist(
            X_sc,
            wp.spatial_vector(
                joint_qd[qd_start + 0],
                joint_qd[qd_start + 1],
                joint_qd[qd_start + 2],
                joint_qd[qd_start + 3],
                joint_qd[qd_start + 4],
                joint_qd[qd_start + 5],
            ),
        )

        joint_S_s[qd_start + 0] = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 1] = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 2] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 3] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        joint_S_s[qd_start + 4] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        joint_S_s[qd_start + 5] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()


# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_axis_mode: wp.array(dtype=int),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vector,
    # outputs
    tau: wp.array(dtype=float),
):
    if type == wp.sim.JOINT_PRISMATIC or type == wp.sim.JOINT_REVOLUTE:
        S_s = joint_S_s[dof_start]

        q = joint_q[coord_start]
        qd = joint_qd[dof_start]
        act = joint_act[axis_start]

        lower = joint_limit_lower[axis_start]
        upper = joint_limit_upper[axis_start]

        limit_ke = joint_limit_ke[axis_start]
        limit_kd = joint_limit_kd[axis_start]
        target_ke = joint_target_ke[axis_start]
        target_kd = joint_target_kd[axis_start]
        mode = joint_axis_mode[axis_start]

        # total torque / force on the joint
        t = -wp.dot(S_s, body_f_s) + eval_joint_force(
            q, qd, act, target_ke, target_kd, lower, upper, limit_ke, limit_kd, mode
        )

        tau[dof_start] = t

        return

    if type == wp.sim.JOINT_BALL:
        # target_ke = joint_target_ke[axis_start]
        # target_kd = joint_target_kd[axis_start]

        for i in range(3):
            S_s = joint_S_s[dof_start + i]

            # w = joint_qd[dof_start + i]
            # r = joint_q[coord_start + i]

            tau[dof_start + i] = -wp.dot(S_s, body_f_s)  # - w * target_kd - r * target_ke

        return

    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = -wp.dot(S_s, body_f_s)

        return

    if type == wp.sim.JOINT_COMPOUND or type == wp.sim.JOINT_UNIVERSAL or type == wp.sim.JOINT_D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            S_s = joint_S_s[dof_start + i]

            q = joint_q[coord_start + i]
            qd = joint_qd[dof_start + i]
            act = joint_act[axis_start + i]

            lower = joint_limit_lower[axis_start + i]
            upper = joint_limit_upper[axis_start + i]
            limit_ke = joint_limit_ke[axis_start + i]
            limit_kd = joint_limit_kd[axis_start + i]
            target_ke = joint_target_ke[axis_start + i]
            target_kd = joint_target_kd[axis_start + i]
            mode = joint_axis_mode[axis_start + i]

            f = eval_joint_force(q, qd, act, target_ke, target_kd, lower, upper, limit_ke, limit_kd, mode)

            # total torque / force on the joint
            t = -wp.dot(S_s, body_f_s) + f

            tau[dof_start + i] = t

        return


@wp.func
def jcalc_integrate(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    if type == wp.sim.JOINT_FIXED:
        return

    # prismatic / revolute
    if type == wp.sim.JOINT_PRISMATIC or type == wp.sim.JOINT_REVOLUTE:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == wp.sim.JOINT_BALL:
        m_j = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        w_j_new = w_j + m_j * dt

        drdt_j = wp.quat(w_j_new, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        # update joint vel
        joint_qd_new[dof_start + 0] = w_j_new[0]
        joint_qd_new[dof_start + 1] = w_j_new[1]
        joint_qd_new[dof_start + 2] = w_j_new[2]

        return

    # free joint
    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear acceleration
        m_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        a_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        # angular and linear velocity
        w_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        v_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # symplectic Euler
        w_s = w_s + m_s * dt
        v_s = v_s + a_s * dt

        # translation of origin
        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # linear vel of origin (note q/qd switch order of linear angular elements)
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velocity
        dpdt_s = v_s + wp.cross(w_s, p_s)

        # quat and quat derivative
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.quat(w_s, 0.0) * r_s * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        # update transform
        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        # update joint_twist
        joint_qd_new[dof_start + 0] = w_s[0]
        joint_qd_new[dof_start + 1] = w_s[1]
        joint_qd_new[dof_start + 2] = w_s[2]
        joint_qd_new[dof_start + 3] = v_s[0]
        joint_qd_new[dof_start + 4] = v_s[1]
        joint_qd_new[dof_start + 5] = v_s[2]

        return

    # other joint types (compound, universal, D6)
    if type == wp.sim.JOINT_COMPOUND or type == wp.sim.JOINT_UNIVERSAL or type == wp.sim.JOINT_D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qdd = joint_qdd[dof_start + i]
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]

            qd_new = qd + qdd * dt
            q_new = q + qd_new * dt

            joint_qd_new[dof_start + i] = qd_new
            joint_q_new[coord_start + i] = q_new

        return


@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # parent transform
    parent = joint_parent[i]
    child = joint_child[i]

    # parent transform in spatial coordinates
    X_pj = joint_X_p[i]
    X_cj = joint_X_c[i]
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    type = joint_type[i]
    axis_start = joint_axis_start[i]
    lin_axis_count = joint_axis_dim[i, 0]
    ang_axis_count = joint_axis_dim[i, 1]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_j = jcalc_transform(type, joint_axis, axis_start, lin_axis_count, ang_axis_count, joint_q, coord_start)

    # transform from world to joint anchor frame at child body
    X_wcj = X_wpj * X_j
    # transform from world to child body frame
    X_wc = X_wcj * wp.transform_inverse(X_cj)

    # compute transform of center of mass
    X_cm = body_X_com[child]
    X_sm = X_wc * X_cm

    # store geometry transforms
    body_q[child] = X_wc
    body_q_com[child] = X_sm


@wp.kernel
def eval_rigid_fk(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_q,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            joint_axis_start,
            joint_axis_dim,
            body_q,
            body_q_com,
        )


@wp.func
def spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_top(a)
    v_a = wp.spatial_bottom(a)

    w_b = wp.spatial_top(b)
    v_b = wp.spatial_bottom(b)

    w = wp.cross(w_a, w_b)
    v = wp.cross(w_a, v_b) + wp.cross(v_a, w_b)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_top(a)
    v_a = wp.spatial_bottom(a)

    w_b = wp.spatial_top(b)
    v_b = wp.spatial_bottom(b)

    w = wp.cross(w_a, w_b) + wp.cross(v_a, v_b)
    v = wp.cross(w_a, v_b)

    return wp.spatial_vector(w, v)


@wp.func
def dense_index(stride: int, i: int, j: int):
    return i * stride + j


@wp.func
def compute_link_velocity(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    type = joint_type[i]
    child = joint_child[i]
    parent = joint_parent[i]
    q_start = joint_q_start[i]
    qd_start = joint_qd_start[i]

    X_pj = joint_X_p[i]
    # X_cj = joint_X_c[i]

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    axis_start = joint_axis_start[i]
    lin_axis_count = joint_axis_dim[i, 0]
    ang_axis_count = joint_axis_dim[i, 1]
    v_j_s = jcalc_motion(
        type,
        joint_axis,
        axis_start,
        lin_axis_count,
        ang_axis_count,
        X_wpj,
        joint_q,
        joint_qd,
        q_start,
        qd_start,
        joint_S_s,
    )

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if parent >= 0:
        v_parent_s = body_v_s[parent]
        a_parent_s = body_a_s[parent]

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s)  # + joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = body_q_com[child]
    I_m = body_I_m[child]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    m = I_m[3, 3]

    f_g = m * gravity
    r_com = wp.transform_get_translation(X_sm)
    f_g_s = wp.spatial_vector(wp.cross(r_com, f_g), f_g)

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = I_s * a_s + spatial_cross_dual(v_s, I_s * v_s)

    body_v_s[child] = v_s
    body_a_s[child] = a_s
    body_f_s[child] = f_b_s - f_g_s
    body_I_s[child] = I_s


# Inverse dynamics via Recursive Newton-Euler algorithm (Featherstone Table 5.1)
@wp.kernel
def eval_rigid_id(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_qd,
            joint_axis,
            joint_axis_start,
            joint_axis_dim,
            body_I_m,
            body_q,
            body_q_com,
            joint_X_p,
            joint_X_c,
            gravity,
            joint_S_s,
            body_I_s,
            body_v_s,
            body_f_s,
            body_a_s,
        )


@wp.kernel
def eval_rigid_tau(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_axis_mode: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    body_f_ext: wp.array(dtype=wp.spatial_vector),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]
    count = end - start

    # compute joint forces
    for offset in range(count):
        # for backwards traversal
        i = end - offset - 1

        type = joint_type[i]
        parent = joint_parent[i]
        child = joint_child[i]
        dof_start = joint_qd_start[i]
        coord_start = joint_q_start[i]
        axis_start = joint_axis_start[i]
        lin_axis_count = joint_axis_dim[i, 0]
        ang_axis_count = joint_axis_dim[i, 1]

        # total forces on body
        f_b_s = body_fb_s[child]
        f_t_s = body_ft_s[child]
        f_ext = body_f_ext[child]
        f_s = f_b_s + f_t_s + f_ext

        # compute joint-space forces, writes out tau
        jcalc_tau(
            type,
            joint_target_ke,
            joint_target_kd,
            joint_limit_ke,
            joint_limit_kd,
            joint_S_s,
            joint_q,
            joint_qd,
            joint_act,
            joint_axis_mode,
            joint_limit_lower,
            joint_limit_upper,
            coord_start,
            dof_start,
            axis_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            tau,
        )

        # update parent forces, todo: check that this is valid for the backwards pass
        if parent >= 0:
            wp.atomic_add(body_ft_s, parent, f_s)


# builds spatial Jacobian J which is an (joint_count*6)x(dof_count) matrix
@wp.kernel
def eval_rigid_jacobian(
    articulation_start: wp.array(dtype=int),
    articulation_J_start: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    J: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    joint_start = articulation_start[index]
    joint_end = articulation_start[index + 1]
    joint_count = joint_end - joint_start

    J_offset = articulation_J_start[index]

    articulation_dof_start = joint_qd_start[joint_start]
    articulation_dof_end = joint_qd_start[joint_end]
    articulation_dof_count = articulation_dof_end - articulation_dof_start

    for i in range(joint_count):
        row_start = i * 6

        j = joint_start + i
        while j != -1:
            joint_dof_start = joint_qd_start[j]
            joint_dof_end = joint_qd_start[j + 1]
            joint_dof_count = joint_dof_end - joint_dof_start

            # fill out each row of the Jacobian walking up the tree
            for dof in range(joint_dof_count):
                col = (joint_dof_start - articulation_dof_start) + dof
                S = joint_S_s[joint_dof_start + dof]

                for k in range(6):
                    J[J_offset + dense_index(articulation_dof_count, row_start + k, col)] = S[k]

            j = joint_ancestor[j]


@wp.func
def spatial_mass(
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    joint_start: int,
    joint_count: int,
    M_start: int,
    # outputs
    M: wp.array(dtype=float),
):
    stride = joint_count * 6
    for l in range(joint_count):
        I = body_I_s[joint_start + l]
        for i in range(6):
            for j in range(6):
                M[M_start + dense_index(stride, l * 6 + i, l * 6 + j)] = I[i, j]


@wp.kernel
def eval_rigid_mass(
    articulation_start: wp.array(dtype=int),
    articulation_M_start: wp.array(dtype=int),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    # outputs
    M: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    joint_start = articulation_start[index]
    joint_end = articulation_start[index + 1]
    joint_count = joint_end - joint_start

    M_offset = articulation_M_start[index]

    spatial_mass(body_I_s, joint_start, joint_count, M_offset, M)


@wp.func
def dense_gemm(
    m: int,
    n: int,
    p: int,
    transpose_A: bool,
    transpose_B: bool,
    add_to_C: bool,
    A_start: int,
    B_start: int,
    C_start: int,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    # outputs
    C: wp.array(dtype=float),
):
    # multiply a `m x p` matrix A by a `p x n` matrix B to produce a `m x n` matrix C
    for i in range(m):
        for j in range(n):
            sum = float(0.0)
            for k in range(p):
                if transpose_A:
                    a_i = k * m + i
                else:
                    a_i = i * p + k
                if transpose_B:
                    b_j = j * p + k
                else:
                    b_j = k * n + j
                sum += A[A_start + a_i] * B[B_start + b_j]

            if add_to_C:
                C[C_start + i * n + j] += sum
            else:
                C[C_start + i * n + j] = sum


# @wp.func_grad(dense_gemm)
# def adj_dense_gemm(
#     m: int,
#     n: int,
#     p: int,
#     transpose_A: bool,
#     transpose_B: bool,
#     add_to_C: bool,
#     A_start: int,
#     B_start: int,
#     C_start: int,
#     A: wp.array(dtype=float),
#     B: wp.array(dtype=float),
#     # outputs
#     C: wp.array(dtype=float),
# ):
#     add_to_C = True
#     if transpose_A:
#         dense_gemm(p, m, n, False, True, add_to_C, A_start, B_start, C_start, B, wp.adjoint[C], wp.adjoint[A])
#         dense_gemm(p, n, m, False, False, add_to_C, A_start, B_start, C_start, A, wp.adjoint[C], wp.adjoint[B])
#     else:
#         dense_gemm(
#             m, p, n, False, not transpose_B, add_to_C, A_start, B_start, C_start, wp.adjoint[C], B, wp.adjoint[A]
#         )
#         dense_gemm(p, n, m, True, False, add_to_C, A_start, B_start, C_start, A, wp.adjoint[C], wp.adjoint[B])


def create_inertia_matrix_kernel(num_joints, num_dofs):
    @wp.kernel
    def eval_dense_gemm_tile(
        J_arr: wp.array3d(dtype=float), M_arr: wp.array3d(dtype=float), H_arr: wp.array3d(dtype=float)
    ):
        articulation = wp.tid()

        J = wp.tile_load(J_arr[articulation], shape=(wp.static(6 * num_joints), num_dofs))
        P = wp.tile_zeros(shape=(wp.static(6 * num_joints), num_dofs), dtype=float)

        # compute P = M*J where M is a 6x6 block diagonal mass matrix
        for i in range(int(num_joints)):
            # 6x6 block matrices are on the diagonal
            M_body = wp.tile_load(M_arr[articulation], shape=(6, 6), offset=(i * 6, i * 6))

            # load a 6xN row from the Jacobian
            J_body = wp.tile_view(J, offset=(i * 6, 0), shape=(6, num_dofs))

            # compute weighted row
            P_body = wp.tile_matmul(M_body, J_body)

            # assign to the P slice
            wp.tile_assign(P, P_body, offset=(i * 6, 0))

        # compute H = J^T*P
        H = wp.tile_matmul(wp.tile_transpose(J), P)

        wp.tile_store(H_arr[articulation], H)

    return eval_dense_gemm_tile


def create_batched_cholesky_kernel(num_dofs):
    assert num_dofs == 18

    @wp.kernel
    def eval_tiled_dense_cholesky_batched(
        A: wp.array3d(dtype=float), R: wp.array2d(dtype=float), L: wp.array3d(dtype=float)
    ):
        articulation = wp.tid()

        a = wp.tile_load(A[articulation], shape=(num_dofs, num_dofs), storage="shared")
        r = wp.tile_load(R[articulation], shape=num_dofs, storage="shared")
        a_r = wp.tile_diag_add(a, r)
        l = wp.tile_cholesky(a_r)
        wp.tile_store(L[articulation], wp.tile_transpose(l))

    return eval_tiled_dense_cholesky_batched


def create_inertia_matrix_cholesky_kernel(num_joints, num_dofs):
    @wp.kernel
    def eval_dense_gemm_and_cholesky_tile(
        J_arr: wp.array3d(dtype=float),
        M_arr: wp.array3d(dtype=float),
        R_arr: wp.array2d(dtype=float),
        H_arr: wp.array3d(dtype=float),
        L_arr: wp.array3d(dtype=float),
    ):
        articulation = wp.tid()

        J = wp.tile_load(J_arr[articulation], shape=(wp.static(6 * num_joints), num_dofs))
        P = wp.tile_zeros(shape=(wp.static(6 * num_joints), num_dofs), dtype=float)

        # compute P = M*J where M is a 6x6 block diagonal mass matrix
        for i in range(int(num_joints)):
            # 6x6 block matrices are on the diagonal
            M_body = wp.tile_load(M_arr[articulation], shape=(6, 6), offset=(i * 6, i * 6))

            # load a 6xN row from the Jacobian
            J_body = wp.tile_view(J, offset=(i * 6, 0), shape=(6, num_dofs))

            # compute weighted row
            P_body = wp.tile_matmul(M_body, J_body)

            # assign to the P slice
            wp.tile_assign(P, P_body, offset=(i * 6, 0))

        # compute H = J^T*P
        H = wp.tile_matmul(wp.tile_transpose(J), P)
        wp.tile_store(H_arr[articulation], H)

        # cholesky L L^T = (H + diag(R))
        R = wp.tile_load(R_arr[articulation], shape=num_dofs, storage="shared")
        H_R = wp.tile_diag_add(H, R)
        L = wp.tile_cholesky(H_R)
        wp.tile_store(L_arr[articulation], L)

    return eval_dense_gemm_and_cholesky_tile


@wp.kernel
def eval_dense_gemm_batched(
    m: wp.array(dtype=int),
    n: wp.array(dtype=int),
    p: wp.array(dtype=int),
    transpose_A: bool,
    transpose_B: bool,
    A_start: wp.array(dtype=int),
    B_start: wp.array(dtype=int),
    C_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    C: wp.array(dtype=float),
):
    # on the CPU each thread computes the whole matrix multiply
    # on the GPU each block computes the multiply with one output per-thread
    batch = wp.tid()  # /kNumThreadsPerBlock;
    add_to_C = False

    dense_gemm(
        m[batch],
        n[batch],
        p[batch],
        transpose_A,
        transpose_B,
        add_to_C,
        A_start[batch],
        B_start[batch],
        C_start[batch],
        A,
        B,
        C,
    )


@wp.func
def dense_cholesky(
    n: int,
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array(dtype=float),
):
    # compute the Cholesky factorization of A = L L^T with diagonal regularization R
    for j in range(n):
        s = A[A_start + dense_index(n, j, j)] + R[R_start + j]

        for k in range(j):
            r = L[A_start + dense_index(n, j, k)]
            s -= r * r

        s = wp.sqrt(s)
        invS = 1.0 / s

        L[A_start + dense_index(n, j, j)] = s

        for i in range(j + 1, n):
            s = A[A_start + dense_index(n, i, j)]

            for k in range(j):
                s -= L[A_start + dense_index(n, i, k)] * L[A_start + dense_index(n, j, k)]

            L[A_start + dense_index(n, i, j)] = s * invS


@wp.func_grad(dense_cholesky)
def adj_dense_cholesky(
    n: int,
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array(dtype=float),
):
    # nop, use dense_solve to differentiate through (A^-1)b = x
    pass


@wp.kernel
def eval_dense_cholesky_batched(
    A_starts: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    L: wp.array(dtype=float),
):
    batch = wp.tid()

    n = A_dim[batch]
    A_start = A_starts[batch]
    R_start = n * batch

    dense_cholesky(n, A, R, A_start, R_start, L)


@wp.func
def dense_subs(
    n: int,
    L_start: int,
    b_start: int,
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
):
    # Solves (L L^T) x = b for x given the Cholesky factor L
    # forward substitution solves the lower triangular system L y = b for y
    for i in range(n):
        s = b[b_start + i]

        for j in range(i):
            s -= L[L_start + dense_index(n, i, j)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]

    # backward substitution solves the upper triangular system L^T x = y for x
    for i in range(n - 1, -1, -1):
        s = x[b_start + i]

        for j in range(i + 1, n):
            s -= L[L_start + dense_index(n, j, i)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]


@wp.func
def dense_solve(
    n: int,
    L_start: int,
    b_start: int,
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
):
    # helper function to include tmp argument for backward pass
    dense_subs(n, L_start, b_start, L, b, x)


@wp.func_grad(dense_solve)
def adj_dense_solve(
    n: int,
    L_start: int,
    b_start: int,
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
):
    if not tmp or not wp.adjoint[x] or not wp.adjoint[A] or not wp.adjoint[L]:
        return
    for i in range(n):
        tmp[b_start + i] = 0.0

    dense_subs(n, L_start, b_start, L, wp.adjoint[x], tmp)

    for i in range(n):
        wp.adjoint[b][b_start + i] += tmp[b_start + i]

    # A* = -adj_b*x^T
    for i in range(n):
        for j in range(n):
            wp.adjoint[L][L_start + dense_index(n, i, j)] += -tmp[b_start + i] * x[b_start + j]

    for i in range(n):
        for j in range(n):
            wp.adjoint[A][L_start + dense_index(n, i, j)] += -tmp[b_start + i] * x[b_start + j]


@wp.kernel
def eval_dense_solve_batched(
    L_start: wp.array(dtype=int),
    L_dim: wp.array(dtype=int),
    b_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
):
    batch = wp.tid()

    dense_solve(L_dim[batch], L_start[batch], b_start[batch], A, L, b, x, tmp)


@wp.kernel
def integrate_generalized_joints(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    type = joint_type[index]
    coord_start = joint_q_start[index]
    dof_start = joint_qd_start[index]
    lin_axis_count = joint_axis_dim[index, 0]
    ang_axis_count = joint_axis_dim[index, 1]

    jcalc_integrate(
        type,
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        lin_axis_count,
        ang_axis_count,
        dt,
        joint_q_new,
        joint_qd_new,
    )


class FeatherstoneIntegrator(Integrator):
    """A semi-implicit integrator using symplectic Euler that operates
    on reduced (also called generalized) coordinates to simulate articulated rigid body dynamics
    based on Featherstone's composite rigid body algorithm (CRBA).

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Instead of maximal coordinates :attr:`State.body_q` (rigid body positions) and :attr:`State.body_qd`
    (rigid body velocities) as is the case :class:`SemiImplicitIntegrator`, :class:`FeatherstoneIntegrator`
    uses :attr:`State.joint_q` and :attr:`State.joint_qd` to represent the positions and velocities of
    joints without allowing any redundant degrees of freedom.

    After constructing :class:`Model` and :class:`State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Note:
        Unlike :class:`SemiImplicitIntegrator` and :class:`XPBDIntegrator`, :class:`FeatherstoneIntegrator` does not simulate rigid bodies with nonzero mass as floating bodies if they are not connected through any joints. Floating-base systems require an explicit free joint with which the body is connected to the world, see :meth:`ModelBuilder.add_joint_free`.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example
    -------

    .. code-block:: python

        integrator = wp.FeatherstoneIntegrator(model)

        # simulation loop
        for i in range(100):
            state = integrator.simulate(model, state_in, state_out, dt)

    Note:
        The :class:`FeatherstoneIntegrator` requires the :class:`Model` to be passed in as a constructor argument.

    """

    def __init__(
        self,
        model,
        angular_damping=0.05,
        update_mass_matrix_every=1,
        friction_smoothing=1.0,
        use_tile_gemm=False,
        fuse_cholesky=True,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            angular_damping (float, optional): Angular damping factor. Defaults to 0.05.
            update_mass_matrix_every (int, optional): How often to update the mass matrix (every n-th time the :meth:`simulate` function gets called). Defaults to 1.
            friction_smoothing (float, optional): The delta value for the Huber norm (see :func:`warp.math.norm_huber`) used for the friction velocity normalization. Defaults to 1.0.
        """
        self.angular_damping = angular_damping
        self.update_mass_matrix_every = update_mass_matrix_every
        self.friction_smoothing = friction_smoothing
        self.use_tile_gemm = use_tile_gemm
        self.fuse_cholesky = fuse_cholesky

        self._step = 0

        self.compute_articulation_indices(model)
        self.allocate_model_aux_vars(model)

        if self.use_tile_gemm:
            # create a custom kernel to evaluate the system matrix for this type
            if self.fuse_cholesky:
                self.eval_inertia_matrix_cholesky_kernel = create_inertia_matrix_cholesky_kernel(
                    int(self.joint_count), int(self.dof_count)
                )
            else:
                self.eval_inertia_matrix_kernel = create_inertia_matrix_kernel(
                    int(self.joint_count), int(self.dof_count)
                )

            # ensure matrix is reloaded since otherwise an unload can happen during graph capture
            # todo: should not be necessary?
            wp.load_module(device=wp.get_device())

    def compute_articulation_indices(self, model):
        # calculate total size and offsets of Jacobian and mass matrices for entire system
        if model.joint_count:
            self.J_size = 0
            self.M_size = 0
            self.H_size = 0

            articulation_J_start = []
            articulation_M_start = []
            articulation_H_start = []

            articulation_M_rows = []
            articulation_H_rows = []
            articulation_J_rows = []
            articulation_J_cols = []

            articulation_dof_start = []
            articulation_coord_start = []

            articulation_start = model.articulation_start.numpy()
            joint_q_start = model.joint_q_start.numpy()
            joint_qd_start = model.joint_qd_start.numpy()

            for i in range(model.articulation_count):
                first_joint = articulation_start[i]
                last_joint = articulation_start[i + 1]

                first_coord = joint_q_start[first_joint]

                first_dof = joint_qd_start[first_joint]
                last_dof = joint_qd_start[last_joint]

                joint_count = last_joint - first_joint
                dof_count = last_dof - first_dof

                articulation_J_start.append(self.J_size)
                articulation_M_start.append(self.M_size)
                articulation_H_start.append(self.H_size)
                articulation_dof_start.append(first_dof)
                articulation_coord_start.append(first_coord)

                # bit of data duplication here, but will leave it as such for clarity
                articulation_M_rows.append(joint_count * 6)
                articulation_H_rows.append(dof_count)
                articulation_J_rows.append(joint_count * 6)
                articulation_J_cols.append(dof_count)

                if self.use_tile_gemm:
                    # store the joint and dof count assuming all
                    # articulations have the same structure
                    self.joint_count = joint_count
                    self.dof_count = dof_count

                self.J_size += 6 * joint_count * dof_count
                self.M_size += 6 * joint_count * 6 * joint_count
                self.H_size += dof_count * dof_count

            # matrix offsets for batched gemm
            self.articulation_J_start = wp.array(articulation_J_start, dtype=wp.int32, device=model.device)
            self.articulation_M_start = wp.array(articulation_M_start, dtype=wp.int32, device=model.device)
            self.articulation_H_start = wp.array(articulation_H_start, dtype=wp.int32, device=model.device)

            self.articulation_M_rows = wp.array(articulation_M_rows, dtype=wp.int32, device=model.device)
            self.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_rows = wp.array(articulation_J_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_cols = wp.array(articulation_J_cols, dtype=wp.int32, device=model.device)

            self.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32, device=model.device)
            self.articulation_coord_start = wp.array(articulation_coord_start, dtype=wp.int32, device=model.device)

    def allocate_model_aux_vars(self, model):
        # allocate mass, Jacobian matrices, and other auxiliary variables pertaining to the model
        if model.joint_count:
            # system matrices
            self.M = wp.zeros((self.M_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
            self.J = wp.zeros((self.J_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
            self.P = wp.empty_like(self.J, requires_grad=model.requires_grad)
            self.H = wp.empty((self.H_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)

            # zero since only upper triangle is set which can trigger NaN detection
            self.L = wp.zeros_like(self.H)

        if model.body_count:
            self.body_I_m = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=model.requires_grad
            )
            wp.launch(
                compute_spatial_inertia,
                model.body_count,
                inputs=[model.body_inertia, model.body_mass],
                outputs=[self.body_I_m],
                device=model.device,
            )
            self.body_X_com = wp.empty(
                (model.body_count,), dtype=wp.transform, device=model.device, requires_grad=model.requires_grad
            )
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=model.device,
            )

    def allocate_state_aux_vars(self, model, target, requires_grad):
        # allocate auxiliary variables that vary with state
        if model.body_count:
            # joints
            target.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_tau = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            if requires_grad:
                # used in the custom grad implementation of eval_dense_solve_batched
                target.joint_solve_tmp = wp.zeros_like(model.joint_qd, requires_grad=True)
            else:
                target.joint_solve_tmp = None
            target.joint_S_s = wp.empty(
                (model.joint_dof_count,),
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=requires_grad,
            )

            # derived rigid body data (maximal coordinates)
            target.body_q_com = wp.empty_like(model.body_q, requires_grad=requires_grad)
            target.body_I_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=requires_grad
            )
            target.body_v_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_a_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_f_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_ft_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )

            target._featherstone_augmented = True

    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
        requires_grad = state_in.requires_grad

        # optionally create dynamical auxiliary variables
        if requires_grad:
            state_aug = state_out
        else:
            state_aug = self

        if not getattr(state_aug, "_featherstone_augmented", False):
            self.allocate_state_aux_vars(model, state_aug, requires_grad)
        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            # damped springs
            eval_spring_forces(model, state_in, particle_f)

            # triangle elastic and lift/drag forces
            eval_triangle_forces(model, state_in, control, particle_f)

            # triangle/triangle contacts
            eval_triangle_contact_forces(model, state_in, particle_f)

            # triangle bending
            eval_bending_forces(model, state_in, particle_f)

            # tetrahedral FEM
            eval_tetrahedral_forces(model, state_in, control, particle_f)

            # particle-particle interactions
            eval_particle_forces(model, state_in, particle_f)

            # particle ground contacts
            eval_particle_ground_contact_forces(model, state_in, particle_f)

            # particle shape contact
            eval_particle_body_contact_forces(model, state_in, particle_f, body_f, body_f_in_world_frame=True)

            # muscles
            if False:
                eval_muscle_forces(model, state_in, control, body_f)

            # ----------------------------
            # articulations

            if model.joint_count:
                # evaluate body transforms
                wp.launch(
                    eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        state_in.joint_q,
                        model.joint_X_p,
                        model.joint_X_c,
                        self.body_X_com,
                        model.joint_axis,
                        model.joint_axis_start,
                        model.joint_axis_dim,
                    ],
                    outputs=[state_in.body_q, state_aug.body_q_com],
                    device=model.device,
                )

                # print("body_X_sc:")
                # print(state_in.body_q.numpy())

                # evaluate joint inertias, motion vectors, and forces
                state_aug.body_f_s.zero_()
                wp.launch(
                    eval_rigid_id,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        state_in.joint_qd,
                        model.joint_axis,
                        model.joint_axis_start,
                        model.joint_axis_dim,
                        self.body_I_m,
                        state_in.body_q,
                        state_aug.body_q_com,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.gravity,
                    ],
                    outputs=[
                        state_aug.joint_S_s,
                        state_aug.body_I_s,
                        state_aug.body_v_s,
                        state_aug.body_f_s,
                        state_aug.body_a_s,
                    ],
                    device=model.device,
                )

                if model.rigid_contact_max and (
                    model.ground and model.shape_ground_contact_pair_count or model.shape_contact_pair_count
                ):
                    wp.launch(
                        kernel=eval_rigid_contacts,
                        dim=model.rigid_contact_max,
                        inputs=[
                            state_in.body_q,
                            state_aug.body_v_s,
                            model.body_com,
                            model.shape_materials,
                            model.shape_geo,
                            model.shape_body,
                            model.rigid_contact_count,
                            model.rigid_contact_point0,
                            model.rigid_contact_point1,
                            model.rigid_contact_normal,
                            model.rigid_contact_shape0,
                            model.rigid_contact_shape1,
                            True,
                            self.friction_smoothing,
                        ],
                        outputs=[body_f],
                        device=model.device,
                    )

                    # if model.rigid_contact_count.numpy()[0] > 0:
                    #     print(body_f.numpy())

                if model.articulation_count:
                    # evaluate joint torques
                    state_aug.body_ft_s.zero_()
                    wp.launch(
                        eval_rigid_tau,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_q_start,
                            model.joint_qd_start,
                            model.joint_axis_start,
                            model.joint_axis_dim,
                            model.joint_axis_mode,
                            state_in.joint_q,
                            state_in.joint_qd,
                            control.joint_act,
                            model.joint_target_ke,
                            model.joint_target_kd,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            model.joint_limit_ke,
                            model.joint_limit_kd,
                            state_aug.joint_S_s,
                            state_aug.body_f_s,
                            body_f,
                        ],
                        outputs=[
                            state_aug.body_ft_s,
                            state_aug.joint_tau,
                        ],
                        device=model.device,
                    )

                    # print("joint_tau:")
                    # print(state_aug.joint_tau.numpy())
                    # print("body_q:")
                    # print(state_in.body_q.numpy())
                    # print("body_qd:")
                    # print(state_in.body_qd.numpy())

                    if self._step % self.update_mass_matrix_every == 0:
                        # build J
                        wp.launch(
                            eval_rigid_jacobian,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.articulation_J_start,
                                model.joint_ancestor,
                                model.joint_qd_start,
                                state_aug.joint_S_s,
                            ],
                            outputs=[self.J],
                            device=model.device,
                        )

                        # build M
                        wp.launch(
                            eval_rigid_mass,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.articulation_M_start,
                                state_aug.body_I_s,
                            ],
                            outputs=[self.M],
                            device=model.device,
                        )

                        if self.use_tile_gemm:
                            # reshape arrays
                            M_tiled = self.M.reshape((-1, 6 * self.joint_count, 6 * self.joint_count))
                            J_tiled = self.J.reshape((-1, 6 * self.joint_count, self.dof_count))
                            R_tiled = model.joint_armature.reshape((-1, self.dof_count))
                            H_tiled = self.H.reshape((-1, self.dof_count, self.dof_count))
                            L_tiled = self.L.reshape((-1, self.dof_count, self.dof_count))
                            assert H_tiled.shape == (model.articulation_count, 18, 18)
                            assert L_tiled.shape == (model.articulation_count, 18, 18)
                            assert R_tiled.shape == (model.articulation_count, 18)

                            if self.fuse_cholesky:
                                wp.launch_tiled(
                                    self.eval_inertia_matrix_cholesky_kernel,
                                    dim=model.articulation_count,
                                    inputs=[J_tiled, M_tiled, R_tiled],
                                    outputs=[H_tiled, L_tiled],
                                    device=model.device,
                                    block_dim=64,
                                )

                            else:
                                wp.launch_tiled(
                                    self.eval_inertia_matrix_kernel,
                                    dim=model.articulation_count,
                                    inputs=[J_tiled, M_tiled],
                                    outputs=[H_tiled],
                                    device=model.device,
                                    block_dim=256,
                                )

                                wp.launch(
                                    eval_dense_cholesky_batched,
                                    dim=model.articulation_count,
                                    inputs=[
                                        self.articulation_H_start,
                                        self.articulation_H_rows,
                                        self.H,
                                        model.joint_armature,
                                    ],
                                    outputs=[self.L],
                                    device=model.device,
                                )

                            # import numpy as np
                            # J = J_tiled.numpy()
                            # M = M_tiled.numpy()
                            # R = R_tiled.numpy()
                            # for i in range(model.articulation_count):
                            #     r = R[i,:,0]
                            #     H = J[i].T @ M[i] @ J[i]
                            #     L = np.linalg.cholesky(H + np.diag(r))
                            #     np.testing.assert_allclose(H, H_tiled.numpy()[i], rtol=1e-2, atol=1e-2)
                            #     np.testing.assert_allclose(L, L_tiled.numpy()[i], rtol=1e-1, atol=1e-1)

                        else:
                            # form P = M*J
                            wp.launch(
                                eval_dense_gemm_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    self.articulation_M_rows,
                                    self.articulation_J_cols,
                                    self.articulation_J_rows,
                                    False,
                                    False,
                                    self.articulation_M_start,
                                    self.articulation_J_start,
                                    # P start is the same as J start since it has the same dims as J
                                    self.articulation_J_start,
                                    self.M,
                                    self.J,
                                ],
                                outputs=[self.P],
                                device=model.device,
                            )

                            # form H = J^T*P
                            wp.launch(
                                eval_dense_gemm_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    self.articulation_J_cols,
                                    self.articulation_J_cols,
                                    # P rows is the same as J rows
                                    self.articulation_J_rows,
                                    True,
                                    False,
                                    self.articulation_J_start,
                                    # P start is the same as J start since it has the same dims as J
                                    self.articulation_J_start,
                                    self.articulation_H_start,
                                    self.J,
                                    self.P,
                                ],
                                outputs=[self.H],
                                device=model.device,
                            )

                            # compute decomposition
                            wp.launch(
                                eval_dense_cholesky_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    self.articulation_H_start,
                                    self.articulation_H_rows,
                                    self.H,
                                    model.joint_armature,
                                ],
                                outputs=[self.L],
                                device=model.device,
                            )

                        # print("joint_act:")
                        # print(control.joint_act.numpy())
                        # print("joint_tau:")
                        # print(state_aug.joint_tau.numpy())
                        # print("H:")
                        # print(self.H.numpy())
                        # print("L:")
                        # print(self.L.numpy())

                    # solve for qdd
                    state_aug.joint_qdd.zero_()
                    wp.launch(
                        eval_dense_solve_batched,
                        dim=model.articulation_count,
                        inputs=[
                            self.articulation_H_start,
                            self.articulation_H_rows,
                            self.articulation_dof_start,
                            self.H,
                            self.L,
                            state_aug.joint_tau,
                        ],
                        outputs=[
                            state_aug.joint_qdd,
                            state_aug.joint_solve_tmp,
                        ],
                        device=model.device,
                    )
                    # print("joint_qdd:")
                    # print(state_aug.joint_qdd.numpy())
                    # print("\n\n")

            # -------------------------------------
            # integrate bodies

            if model.joint_count:
                wp.launch(
                    kernel=integrate_generalized_joints,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_axis_dim,
                        state_in.joint_q,
                        state_in.joint_qd,
                        state_aug.joint_qdd,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    device=model.device,
                )

                # update maximal coordinates
                eval_fk(model, state_out.joint_q, state_out.joint_qd, None, state_out)

            self.integrate_particles(model, state_in, state_out, dt)

            self._step += 1

            return state_out
