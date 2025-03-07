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

from .utils import quat_decompose, quat_twist


@wp.func
def compute_2d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    q0: float,
    q1: float,
    qd0: float,
    qd1: float,
):
    """
    Computes the rotation quaternion and 3D angular velocity given the joint axes, coordinates and velocities.
    """
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))

    # body local axes
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))

    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, q0)

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, q1)

    rot = q_1 * q_0

    vel = axis_0 * qd0 + axis_1 * qd1

    return rot, vel


@wp.func
def invert_2d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    q_p: wp.quat,
    q_c: wp.quat,
    w_err: wp.vec3,
):
    """
    Computes generalized joint position and velocity coordinates for a 2D rotational joint given the joint axes, relative orientations and angular velocity differences between the two bodies the joint connects.
    """
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))
    q_pc = wp.quat_inverse(q_off) * wp.quat_inverse(q_p) * q_c * q_off

    # decompose to a compound rotation each axis
    angles = quat_decompose(q_pc)

    # find rotation axes
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
    local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

    axis_2 = wp.quat_rotate(q_1 * q_0, local_2)

    # convert angular velocity to local space
    w_err_p = wp.quat_rotate_inv(q_p, w_err)

    # given joint axes and angular velocity error, solve for joint velocities
    c12 = wp.cross(axis_1, axis_2)
    c02 = wp.cross(axis_0, axis_2)

    vel = wp.vec2(wp.dot(w_err_p, c12) / wp.dot(axis_0, c12), wp.dot(w_err_p, c02) / wp.dot(axis_1, c02))

    return wp.vec2(angles[0], angles[1]), vel


@wp.func
def compute_3d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    axis_2: wp.vec3,
    q0: float,
    q1: float,
    q2: float,
    qd0: float,
    qd1: float,
    qd2: float,
):
    """
    Computes the rotation quaternion and 3D angular velocity given the joint axes, coordinates and velocities.
    """
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2))

    # body local axes
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
    local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

    # reconstruct rotation axes
    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, q0)

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, q1)

    axis_2 = wp.quat_rotate(q_1 * q_0, local_2)
    q_2 = wp.quat_from_axis_angle(axis_2, q2)

    rot = q_2 * q_1 * q_0
    vel = axis_0 * qd0 + axis_1 * qd1 + axis_2 * qd2

    return rot, vel


@wp.func
def invert_3d_rotational_dofs(
    axis_0: wp.vec3, axis_1: wp.vec3, axis_2: wp.vec3, q_p: wp.quat, q_c: wp.quat, w_err: wp.vec3
):
    """
    Computes generalized joint position and velocity coordinates for a 3D rotational joint given the joint axes, relative orientations and angular velocity differences between the two bodies the joint connects.
    """
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2))
    q_pc = wp.quat_inverse(q_off) * wp.quat_inverse(q_p) * q_c * q_off

    # decompose to a compound rotation each axis
    angles = quat_decompose(q_pc)

    # find rotation axes
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
    local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

    axis_2 = wp.quat_rotate(q_1 * q_0, local_2)

    # convert angular velocity to local space
    w_err_p = wp.quat_rotate_inv(q_p, w_err)

    # given joint axes and angular velocity error, solve for joint velocities
    c12 = wp.cross(axis_1, axis_2)
    c02 = wp.cross(axis_0, axis_2)
    c01 = wp.cross(axis_0, axis_1)

    velocities = wp.vec3(
        wp.dot(w_err_p, c12) / wp.dot(axis_0, c12),
        wp.dot(w_err_p, c02) / wp.dot(axis_1, c02),
        wp.dot(w_err_p, c01) / wp.dot(axis_2, c01),
    )

    return angles, velocities


@wp.func
def eval_single_articulation_fk(
    joint_start: int,
    joint_end: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    for i in range(joint_start, joint_end):
        parent = joint_parent[i]
        child = joint_child[i]

        # compute transform across the joint
        type = joint_type[i]

        X_pj = joint_X_p[i]
        X_cj = joint_X_c[i]

        # parent anchor frame in world space
        X_wpj = X_pj
        # velocity of parent anchor point in world space
        v_wpj = wp.spatial_vector()
        if parent >= 0:
            X_wp = body_q[parent]
            X_wpj = X_wp * X_wpj
            r_p = wp.transform_get_translation(X_wpj) - wp.transform_point(X_wp, body_com[parent])

            v_wp = body_qd[parent]
            w_p = wp.spatial_top(v_wp)
            v_p = wp.spatial_bottom(v_wp) + wp.cross(w_p, r_p)
            v_wpj = wp.spatial_vector(w_p, v_p)

        q_start = joint_q_start[i]
        qd_start = joint_qd_start[i]
        axis_start = joint_axis_start[i]
        lin_axis_count = joint_axis_dim[i, 0]
        ang_axis_count = joint_axis_dim[i, 1]

        X_j = wp.transform_identity()
        v_j = wp.spatial_vector(wp.vec3(), wp.vec3())

        if type == wp.sim.JOINT_PRISMATIC:
            axis = joint_axis[axis_start]

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_j = wp.transform(axis * q, wp.quat_identity())
            v_j = wp.spatial_vector(wp.vec3(), axis * qd)

        if type == wp.sim.JOINT_REVOLUTE:
            axis = joint_axis[axis_start]

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_j = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
            v_j = wp.spatial_vector(axis * qd, wp.vec3())

        if type == wp.sim.JOINT_BALL:
            r = wp.quat(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2], joint_q[q_start + 3])

            w = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])

            X_j = wp.transform(wp.vec3(), r)
            v_j = wp.spatial_vector(w, wp.vec3())

        if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
            t = wp.transform(
                wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
                wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
            )

            v = wp.spatial_vector(
                wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
                wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),
            )

            X_j = t
            v_j = v

        if type == wp.sim.JOINT_COMPOUND:
            rot, vel_w = compute_3d_rotational_dofs(
                joint_axis[axis_start],
                joint_axis[axis_start + 1],
                joint_axis[axis_start + 2],
                joint_q[q_start + 0],
                joint_q[q_start + 1],
                joint_q[q_start + 2],
                joint_qd[qd_start + 0],
                joint_qd[qd_start + 1],
                joint_qd[qd_start + 2],
            )

            t = wp.transform(wp.vec3(0.0, 0.0, 0.0), rot)
            v = wp.spatial_vector(vel_w, wp.vec3(0.0, 0.0, 0.0))

            X_j = t
            v_j = v

        if type == wp.sim.JOINT_UNIVERSAL:
            rot, vel_w = compute_2d_rotational_dofs(
                joint_axis[axis_start],
                joint_axis[axis_start + 1],
                joint_q[q_start + 0],
                joint_q[q_start + 1],
                joint_qd[qd_start + 0],
                joint_qd[qd_start + 1],
            )

            t = wp.transform(wp.vec3(0.0, 0.0, 0.0), rot)
            v = wp.spatial_vector(vel_w, wp.vec3(0.0, 0.0, 0.0))

            X_j = t
            v_j = v

        if type == wp.sim.JOINT_D6:
            pos = wp.vec3(0.0)
            rot = wp.quat_identity()
            vel_v = wp.vec3(0.0)
            vel_w = wp.vec3(0.0)

            # unroll for loop to ensure joint actions remain differentiable
            # (since differentiating through a for loop that updates a local variable is not supported)

            if lin_axis_count > 0:
                axis = joint_axis[axis_start + 0]
                pos += axis * joint_q[q_start + 0]
                vel_v += axis * joint_qd[qd_start + 0]
            if lin_axis_count > 1:
                axis = joint_axis[axis_start + 1]
                pos += axis * joint_q[q_start + 1]
                vel_v += axis * joint_qd[qd_start + 1]
            if lin_axis_count > 2:
                axis = joint_axis[axis_start + 2]
                pos += axis * joint_q[q_start + 2]
                vel_v += axis * joint_qd[qd_start + 2]

            ia = axis_start + lin_axis_count
            iq = q_start + lin_axis_count
            iqd = qd_start + lin_axis_count
            if ang_axis_count == 1:
                axis = joint_axis[ia]
                rot = wp.quat_from_axis_angle(axis, joint_q[iq])
                vel_w = joint_qd[iqd] * axis
            if ang_axis_count == 2:
                rot, vel_w = compute_2d_rotational_dofs(
                    joint_axis[ia + 0],
                    joint_axis[ia + 1],
                    joint_q[iq + 0],
                    joint_q[iq + 1],
                    joint_qd[iqd + 0],
                    joint_qd[iqd + 1],
                )
            if ang_axis_count == 3:
                rot, vel_w = compute_3d_rotational_dofs(
                    joint_axis[ia + 0],
                    joint_axis[ia + 1],
                    joint_axis[ia + 2],
                    joint_q[iq + 0],
                    joint_q[iq + 1],
                    joint_q[iq + 2],
                    joint_qd[iqd + 0],
                    joint_qd[iqd + 1],
                    joint_qd[iqd + 2],
                )

            X_j = wp.transform(pos, rot)
            v_j = wp.spatial_vector(vel_w, vel_v)

        # transform from world to joint anchor frame at child body
        X_wcj = X_wpj * X_j
        # transform from world to child body frame
        X_wc = X_wcj * wp.transform_inverse(X_cj)

        # transform velocity across the joint to world space
        angular_vel = wp.transform_vector(X_wpj, wp.spatial_top(v_j))
        linear_vel = wp.transform_vector(X_wpj, wp.spatial_bottom(v_j))

        v_wc = v_wpj + wp.spatial_vector(angular_vel, linear_vel)

        body_q[child] = X_wc
        body_qd[child] = v_wc


# implementation where mask is an integer array
@wp.kernel
def eval_articulation_fk(
    articulation_start: wp.array(dtype=int),
    articulation_mask: wp.array(
        dtype=int
    ),  # used to enable / disable FK for an articulation, if None then treat all as enabled
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    # early out if disabling FK for this articulation
    if articulation_mask:
        if articulation_mask[tid] == 0:
            return

    joint_start = articulation_start[tid]
    joint_end = articulation_start[tid + 1]

    eval_single_articulation_fk(
        joint_start,
        joint_end,
        joint_q,
        joint_qd,
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_axis_start,
        joint_axis_dim,
        body_com,
        # outputs
        body_q,
        body_qd,
    )


# overload where mask is a bool array
@wp.kernel
def eval_articulation_fk(
    articulation_start: wp.array(dtype=int),
    articulation_mask: wp.array(
        dtype=bool
    ),  # used to enable / disable FK for an articulation, if None then treat all as enabled
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    # early out if disabling FK for this articulation
    if articulation_mask:
        if not articulation_mask[tid]:
            return

    joint_start = articulation_start[tid]
    joint_end = articulation_start[tid + 1]

    eval_single_articulation_fk(
        joint_start,
        joint_end,
        joint_q,
        joint_qd,
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_axis_start,
        joint_axis_dim,
        body_com,
        # outputs
        body_q,
        body_qd,
    )


# updates state body information based on joint coordinates
def eval_fk(model, joint_q, joint_qd, mask, state):
    """
    Evaluates the model's forward kinematics given the joint coordinates and updates the state's body information (:attr:`State.body_q` and :attr:`State.body_qd`).

    Args:
        model (Model): The model to evaluate.
        joint_q (array): Generalized joint position coordinates, shape [joint_coord_count], float
        joint_qd (array): Generalized joint velocity coordinates, shape [joint_dof_count], float
        mask (array): The mask to use to enable / disable FK for an articulation. If None then treat all as enabled, shape [articulation_count], int/bool
        state (State): The state to update.
    """
    wp.launch(
        kernel=eval_articulation_fk,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            mask,
            joint_q,
            joint_qd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_axis_start,
            model.joint_axis_dim,
            model.body_com,
        ],
        outputs=[
            state.body_q,
            state.body_qd,
        ],
        device=model.device,
    )


@wp.func
def reconstruct_angular_q_qd(q_pc: wp.quat, w_err: wp.vec3, X_wp: wp.transform, axis: wp.vec3):
    """
    Reconstructs the angular joint coordinates and velocities given the relative rotation and angular velocity
    between a parent and child body.

    Args:
        q_pc (quat): The relative rotation between the parent and child body.
        w_err (vec3): The angular velocity between the parent and child body.
        X_wp (transform): The parent body's transform in world space.
        axis (vec3): The joint axis in the frame of the parent body.

    Returns:
        q (float): The joint position coordinate.
        qd (float): The joint velocity coordinate.
    """
    axis_p = wp.transform_vector(X_wp, axis)
    twist = quat_twist(axis, q_pc)
    q = wp.acos(twist[3]) * 2.0 * wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
    qd = wp.dot(w_err, axis_p)
    return q, qd


@wp.kernel
def eval_articulation_ik(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
):
    tid = wp.tid()

    parent = joint_parent[tid]
    child = joint_child[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    w_p = wp.vec3()
    v_p = wp.vec3()
    v_wp = wp.spatial_vector()

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_pj
        r_p = wp.transform_get_translation(X_wpj) - wp.transform_point(X_wp, body_com[parent])

        v_wp = body_qd[parent]
        w_p = wp.spatial_top(v_wp)
        v_p = wp.spatial_bottom(v_wp) + wp.cross(w_p, r_p)

    # child transform and moment arm
    X_wc = body_q[child]
    X_wcj = X_wc * X_cj

    v_wc = body_qd[child]

    w_c = wp.spatial_top(v_wc)
    v_c = wp.spatial_bottom(v_wc)

    # joint properties
    type = joint_type[tid]

    # compute position and orientation differences between anchor frames
    x_p = wp.transform_get_translation(X_wpj)
    x_c = wp.transform_get_translation(X_wcj)

    q_p = wp.transform_get_rotation(X_wpj)
    q_c = wp.transform_get_rotation(X_wcj)

    x_err = x_c - x_p
    v_err = v_c - v_p
    w_err = w_c - w_p

    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]
    axis_start = joint_axis_start[tid]
    lin_axis_count = joint_axis_dim[tid, 0]
    ang_axis_count = joint_axis_dim[tid, 1]

    if type == wp.sim.JOINT_PRISMATIC:
        axis = joint_axis[axis_start]

        # world space joint axis
        axis_p = wp.quat_rotate(q_p, axis)

        # evaluate joint coordinates
        q = wp.dot(x_err, axis_p)
        qd = wp.dot(v_err, axis_p)

        joint_q[q_start] = q
        joint_qd[qd_start] = qd

        return

    if type == wp.sim.JOINT_REVOLUTE:
        axis = joint_axis[axis_start]
        q_pc = wp.quat_inverse(q_p) * q_c

        q, qd = reconstruct_angular_q_qd(q_pc, w_err, X_wpj, axis)

        joint_q[q_start] = q
        joint_qd[qd_start] = qd

        return

    if type == wp.sim.JOINT_BALL:
        q_pc = wp.quat_inverse(q_p) * q_c

        joint_q[q_start + 0] = q_pc[0]
        joint_q[q_start + 1] = q_pc[1]
        joint_q[q_start + 2] = q_pc[2]
        joint_q[q_start + 3] = q_pc[3]

        ang_vel = wp.transform_vector(wp.transform_inverse(X_wpj), w_err)
        joint_qd[qd_start + 0] = ang_vel[0]
        joint_qd[qd_start + 1] = ang_vel[1]
        joint_qd[qd_start + 2] = ang_vel[2]

        return

    if type == wp.sim.JOINT_FIXED:
        return

    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        q_pc = wp.quat_inverse(q_p) * q_c

        x_err_c = wp.quat_rotate_inv(q_p, x_err)
        v_err_c = wp.quat_rotate_inv(q_p, v_err)
        w_err_c = wp.quat_rotate_inv(q_p, w_err)

        joint_q[q_start + 0] = x_err_c[0]
        joint_q[q_start + 1] = x_err_c[1]
        joint_q[q_start + 2] = x_err_c[2]

        joint_q[q_start + 3] = q_pc[0]
        joint_q[q_start + 4] = q_pc[1]
        joint_q[q_start + 5] = q_pc[2]
        joint_q[q_start + 6] = q_pc[3]

        joint_qd[qd_start + 0] = w_err_c[0]
        joint_qd[qd_start + 1] = w_err_c[1]
        joint_qd[qd_start + 2] = w_err_c[2]

        joint_qd[qd_start + 3] = v_err_c[0]
        joint_qd[qd_start + 4] = v_err_c[1]
        joint_qd[qd_start + 5] = v_err_c[2]

        return

    if type == wp.sim.JOINT_COMPOUND:
        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        axis_2 = joint_axis[axis_start + 2]
        qs, qds = invert_3d_rotational_dofs(axis_0, axis_1, axis_2, q_p, q_c, w_err)
        joint_q[q_start + 0] = qs[0]
        joint_q[q_start + 1] = qs[1]
        joint_q[q_start + 2] = qs[2]
        joint_qd[qd_start + 0] = qds[0]
        joint_qd[qd_start + 1] = qds[1]
        joint_qd[qd_start + 2] = qds[2]

        return

    if type == wp.sim.JOINT_UNIVERSAL:
        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        qs2, qds2 = invert_2d_rotational_dofs(axis_0, axis_1, q_p, q_c, w_err)
        joint_q[q_start + 0] = qs2[0]
        joint_q[q_start + 1] = qs2[1]
        joint_qd[qd_start + 0] = qds2[0]
        joint_qd[qd_start + 1] = qds2[1]

        return

    if type == wp.sim.JOINT_D6:
        x_err_c = wp.quat_rotate_inv(q_p, x_err)
        v_err_c = wp.quat_rotate_inv(q_p, v_err)
        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            joint_q[q_start + 0] = wp.dot(x_err_c, axis)
            joint_qd[qd_start + 0] = wp.dot(v_err_c, axis)

        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            joint_q[q_start + 1] = wp.dot(x_err_c, axis)
            joint_qd[qd_start + 1] = wp.dot(v_err_c, axis)

        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            joint_q[q_start + 2] = wp.dot(x_err_c, axis)
            joint_qd[qd_start + 2] = wp.dot(v_err_c, axis)

        if ang_axis_count == 1:
            axis = joint_axis[axis_start]
            q_pc = wp.quat_inverse(q_p) * q_c
            q, qd = reconstruct_angular_q_qd(q_pc, w_err, X_wpj, joint_axis[axis_start + lin_axis_count])
            joint_q[q_start + lin_axis_count] = q
            joint_qd[qd_start + lin_axis_count] = qd

        if ang_axis_count == 2:
            axis_0 = joint_axis[axis_start + lin_axis_count + 0]
            axis_1 = joint_axis[axis_start + lin_axis_count + 1]
            qs2, qds2 = invert_2d_rotational_dofs(axis_0, axis_1, q_p, q_c, w_err)
            joint_q[q_start + lin_axis_count + 0] = qs2[0]
            joint_q[q_start + lin_axis_count + 1] = qs2[1]
            joint_qd[qd_start + lin_axis_count + 0] = qds2[0]
            joint_qd[qd_start + lin_axis_count + 1] = qds2[1]

        if ang_axis_count == 3:
            axis_0 = joint_axis[axis_start + lin_axis_count + 0]
            axis_1 = joint_axis[axis_start + lin_axis_count + 1]
            axis_2 = joint_axis[axis_start + lin_axis_count + 2]
            qs3, qds3 = invert_3d_rotational_dofs(axis_0, axis_1, axis_2, q_p, q_c, w_err)
            joint_q[q_start + lin_axis_count + 0] = qs3[0]
            joint_q[q_start + lin_axis_count + 1] = qs3[1]
            joint_q[q_start + lin_axis_count + 2] = qs3[2]
            joint_qd[qd_start + lin_axis_count + 0] = qds3[0]
            joint_qd[qd_start + lin_axis_count + 1] = qds3[1]
            joint_qd[qd_start + lin_axis_count + 2] = qds3[2]

        return


# given maximal coordinate model computes ik (closest point projection)
def eval_ik(model, state, joint_q, joint_qd):
    """
    Evaluates the model's inverse kinematics given the state's body information (:attr:`State.body_q` and :attr:`State.body_qd`) and updates the generalized joint coordinates `joint_q` and `joint_qd`.

    Args:
        model (Model): The model to evaluate.
        state (State): The state with the body's maximal coordinates (positions :attr:`State.body_q` and velocities :attr:`State.body_qd`) to use.
        joint_q (array): Generalized joint position coordinates, shape [joint_coord_count], float
        joint_qd (array): Generalized joint velocity coordinates, shape [joint_dof_count], float
    """
    wp.launch(
        kernel=eval_articulation_ik,
        dim=model.joint_count,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_axis_start,
            model.joint_axis_dim,
            model.joint_q_start,
            model.joint_qd_start,
        ],
        outputs=[joint_q, joint_qd],
        device=model.device,
    )
