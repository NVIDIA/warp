# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp


# decompose a quaternion into a sequence of 3 rotations around x,y',z' respectively, i.e.: q = q_z''q_y'q_x
@wp.func
def quat_decompose(q: wp.quat):

    R = wp.mat33(
            wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),
            wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),
            wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))

    # https://www.sedris.org/wg8home/Documents/WG80485.pdf
    phi = wp.atan2(R[1, 2], R[2, 2])
    theta = wp.asin(-R[0, 2])
    psi = wp.atan2(R[0, 1], R[0, 0])

    return -wp.vec3(phi, theta, psi)

@wp.kernel
def eval_articulation_fk(
    articulation_start: wp.array(dtype=int),
    articulation_mask: wp.array(dtype=int), # used to enable / disable FK for an articulation, if None then treat all as enabled
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    # early out if disabling FK for this articulation
    if (articulation_mask):
        if (articulation_mask[tid]==0):
            return

    joint_start = articulation_start[tid]
    joint_end = articulation_start[tid+1]

    for i in range(joint_start, joint_end):

        parent = joint_parent[i]
        X_wp = wp.transform_identity()
        v_wp = wp.spatial_vector()

        if (parent >= 0):
            X_wp = body_q[parent]
            v_wp = body_qd[parent]

        # compute transform across the joint
        type = joint_type[i]
        axis = joint_axis[i]

        X_pj = joint_X_p[i]
        X_cj = joint_X_c[i]  
        
        q_start = joint_q_start[i]
        qd_start = joint_qd_start[i]

        if type == wp.sim.JOINT_PRISMATIC:

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_jc = wp.transform(axis*q, wp.quat_identity())
            v_jc = wp.spatial_vector(wp.vec3(), axis*qd)

        if type == wp.sim.JOINT_REVOLUTE:

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
            v_jc = wp.spatial_vector(axis*qd, wp.vec3())

        if type == wp.sim.JOINT_BALL:

            r = wp.quat(joint_q[q_start+0],
                        joint_q[q_start+1],
                        joint_q[q_start+2],
                        joint_q[q_start+3])

            w = wp.vec3(joint_qd[qd_start+0],
                        joint_qd[qd_start+1],
                        joint_qd[qd_start+2])

            # print(r)
            
            X_jc = wp.transform(wp.vec3(), r)
            v_jc = wp.spatial_vector(w, wp.vec3())

        if type == wp.sim.JOINT_FIXED:
            
            X_jc = wp.transform_identity()
            v_jc = wp.spatial_vector(wp.vec3(), wp.vec3())

        if type == wp.sim.JOINT_FREE:

            t = wp.transform(
                    wp.vec3(joint_q[q_start+0], joint_q[q_start+1], joint_q[q_start+2]),
                    wp.quat(joint_q[q_start+3], joint_q[q_start+4], joint_q[q_start+5], joint_q[q_start+6]))

            v = wp.spatial_vector(
                    wp.vec3(joint_qd[qd_start+0], joint_qd[qd_start+1], joint_qd[qd_start+2]),
                    wp.vec3(joint_qd[qd_start+3], joint_qd[qd_start+4], joint_qd[qd_start+5]))

            X_jc = t
            v_jc = v

        if type == wp.sim.JOINT_COMPOUND:

            q_c = wp.transform_get_rotation(X_cj)

            # body local axes
            local_0 = wp.quat_rotate(q_c, wp.vec3(1.0, 0.0, 0.0))
            local_1 = wp.quat_rotate(q_c, wp.vec3(0.0, 1.0, 0.0))
            local_2 = wp.quat_rotate(q_c, wp.vec3(0.0, 0.0, 1.0))

            # reconstruct rotation axes, todo: can probably use fact that rz'*ry'*rx' == rx*ry*rz to avoid some work here
            axis_0 = local_0
            q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start+0])

            axis_1 = wp.quat_rotate(q_0, local_1)
            q_1 = wp.quat_from_axis_angle(axis_1, joint_q[q_start+1])

            axis_2 = wp.quat_rotate(q_1*q_0, local_2)
            q_2 = wp.quat_from_axis_angle(axis_2, joint_q[q_start+2])

            t = wp.transform(wp.vec3(), q_2*q_1*q_0)

            v = wp.spatial_vector(axis_0*joint_qd[qd_start+0] + 
                                  axis_1*joint_qd[qd_start+1] + 
                                  axis_2*joint_qd[qd_start+2], wp.vec3())

            X_jc = t
            v_jc = v

        if type == wp.sim.JOINT_UNIVERSAL:

            q_c = wp.transform_get_rotation(X_cj)

            # body local axes
            local_0 = wp.quat_rotate(q_c, wp.vec3(1.0, 0.0, 0.0))
            local_1 = wp.quat_rotate(q_c, wp.vec3(0.0, 1.0, 0.0))

            # reconstruct rotation axes
            axis_0 = local_0
            q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start+0])

            axis_1 = wp.quat_rotate(q_0, local_1)
            q_1 = wp.quat_from_axis_angle(axis_1, joint_q[q_start+1])

            t = wp.transform(wp.vec3(), q_1*q_0)

            v = wp.spatial_vector(axis_0*joint_qd[qd_start+0] + 
                                  axis_1*joint_qd[qd_start+1], wp.vec3())

            X_jc = t
            v_jc = v


        X_wj = X_wp*X_pj
        X_wc = X_wj*X_jc

        # transform velocity across the joint to world space
        angular_vel = wp.transform_vector(X_wj, wp.spatial_top(v_jc))
        linear_vel = wp.transform_vector(X_wj, wp.spatial_bottom(v_jc))

        v_wc = v_wp + wp.spatial_vector(angular_vel, linear_vel + wp.cross(angular_vel, body_com[i]))

        body_q[i] = X_wc
        body_qd[i] = v_wc
        

# updates state body information based on joint coordinates
def eval_fk(model, joint_q, joint_qd, mask, state):

    wp.launch(kernel=eval_articulation_fk,
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
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.body_com],
                outputs=[
                    state.body_q,
                    state.body_qd,
                ],
                device=model.device)


# returns the twist around an axis
@wp.func
def quat_twist(axis: wp.vec3, q: wp.quat):
    
    # project imaginary part onto axis
    a = wp.vec3(q[0], q[1], q[2])
    a = wp.dot(a, axis)*axis

    return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))


@wp.kernel
def eval_articulation_ik(body_q: wp.array(dtype=wp.transform),
                         body_qd: wp.array(dtype=wp.spatial_vector),
                         body_com: wp.array(dtype=wp.vec3),
                         joint_type: wp.array(dtype=int),
                         joint_parent: wp.array(dtype=int),
                         joint_X_p: wp.array(dtype=wp.transform),
                         joint_X_c: wp.array(dtype=wp.transform),
                         joint_axis: wp.array(dtype=wp.vec3),
                         joint_q_start: wp.array(dtype=int),
                         joint_qd_start: wp.array(dtype=int),
                         joint_q: wp.array(dtype=float),
                         joint_qd: wp.array(dtype=float)):

    tid = wp.tid()

    c_child = tid
    c_parent = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wp = X_pj
    r_p = wp.vec3()
    w_p = wp.vec3()
    v_p = wp.vec3()

    # parent transform and moment arm
    if (c_parent >= 0):
        X_wp = body_q[c_parent]*X_wp
        r_wp = wp.transform_get_translation(X_wp) - wp.transform_point(body_q[c_parent], body_com[c_parent])
        
        twist_p = body_qd[c_parent]

        w_p = wp.spatial_top(twist_p)
        v_p = wp.spatial_bottom(twist_p) + wp.cross(w_p, r_wp)

    # child transform and moment arm
    X_wc = body_q[c_child]#*joint_X_c[tid]
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(body_q[c_child], body_com[c_child])
    
    twist_c = body_qd[c_child]

    w_c = wp.spatial_top(twist_c)
    v_c = wp.spatial_bottom(twist_c) + wp.cross(w_c, r_c)

    # joint properties
    type = joint_type[tid]
    axis = joint_axis[tid]
  
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)

    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # translational error
    x_err = x_c - x_p
    v_err = v_c - v_p
    w_err = w_c - w_p

    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]

    if type == wp.sim.JOINT_PRISMATIC:
        
        # world space joint axis
        axis_p = wp.transform_vector(X_wp, axis)

        # evaluate joint coordinates
        q = wp.dot(x_err, axis_p)
        qd = wp.dot(v_err, axis_p)

        joint_q[q_start] = q
        joint_qd[qd_start] = qd
        
        return
   
    if type == wp.sim.JOINT_REVOLUTE:
        
        axis_p = wp.transform_vector(X_wp, axis)
        axis_c = wp.transform_vector(X_wc, axis)

        # swing twist decomposition
        q_pc = wp.quat_inverse(q_p)*q_c
        twist = quat_twist(axis, q_pc)

        q = wp.acos(twist[3])*2.0*wp.sign(wp.dot(axis, wp.vec3(twist[0], twist[1], twist[2])))
        qd = wp.dot(w_err, axis_p)

        joint_q[q_start] = q
        joint_qd[qd_start] = qd

        return
    
    if type == wp.sim.JOINT_BALL:
        
        q_pc = wp.quat_inverse(q_p)*q_c
    
        joint_q[q_start + 0] = q_pc[0]
        joint_q[q_start + 1] = q_pc[1]
        joint_q[q_start + 2] = q_pc[2]
        joint_q[q_start + 3] = q_pc[3]

        joint_qd[qd_start + 0] = w_err[0]
        joint_qd[qd_start + 1] = w_err[1]
        joint_qd[qd_start + 2] = w_err[2]

        return

    if type == wp.sim.JOINT_FIXED:
        return

    if type == wp.sim.JOINT_FREE:
        
        q_pc = wp.quat_inverse(q_p)*q_c

        joint_q[q_start + 0] = x_err[0]
        joint_q[q_start + 1] = x_err[1]
        joint_q[q_start + 2] = x_err[2]

        joint_q[q_start + 3] = q_pc[0]
        joint_q[q_start + 4] = q_pc[1]
        joint_q[q_start + 5] = q_pc[2]
        joint_q[q_start + 6] = q_pc[3]
        
        joint_qd[qd_start + 0] = w_err[0]
        joint_qd[qd_start + 1] = w_err[1]
        joint_qd[qd_start + 2] = w_err[2]

        joint_qd[qd_start + 3] = v_err[0]
        joint_qd[qd_start + 4] = v_err[1]
        joint_qd[qd_start + 5] = v_err[2]

    if type == wp.sim.JOINT_COMPOUND:

        q_off = wp.transform_get_rotation(X_cj)
        q_pc = wp.quat_inverse(q_off)*wp.quat_inverse(q_p)*q_c*q_off

        # decompose to a compound rotation each axis 
        angles = quat_decompose(q_pc)

        # reconstruct rotation axes
        axis_0 = wp.vec3(1.0, 0.0, 0.0)
        q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        axis_2 = wp.quat_rotate(q_1*q_0, wp.vec3(0.0, 0.0, 1.0))
        q_2 = wp.quat_from_axis_angle(axis_2, angles[2])

        q_w = q_p*q_off

        joint_q[q_start+0] = angles[0]
        joint_q[q_start+1] = angles[1]
        joint_q[q_start+2] = angles[2]

        joint_qd[qd_start+0] = wp.dot(wp.quat_rotate(q_w, axis_0), w_err)
        joint_qd[qd_start+1] = wp.dot(wp.quat_rotate(q_w, axis_1), w_err)
        joint_qd[qd_start+2] = wp.dot(wp.quat_rotate(q_w, axis_2), w_err)

        return

    if type == wp.sim.JOINT_UNIVERSAL:

        q_off = wp.transform_get_rotation(X_cj)
        q_pc = wp.quat_inverse(q_off)*wp.quat_inverse(q_p)*q_c*q_off

        # decompose to a compound rotation each axis 
        angles = quat_decompose(q_pc)

        # reconstruct rotation axes
        axis_0 = wp.vec3(1.0, 0.0, 0.0)
        q_0 = wp.quat_from_axis_angle(axis_0, angles[0])

        axis_1 = wp.quat_rotate(q_0, wp.vec3(0.0, 1.0, 0.0))
        q_1 = wp.quat_from_axis_angle(axis_1, angles[1])

        q_w = q_p*q_off

        joint_q[q_start+0] = angles[0]
        joint_q[q_start+1] = angles[1]

        joint_qd[qd_start+0] = wp.dot(wp.quat_rotate(q_w, axis_0), w_err)
        joint_qd[qd_start+1] = wp.dot(wp.quat_rotate(q_w, axis_1), w_err)

        return



# given maximal coordinate model computes ik (closest point projection)
def eval_ik(model, state, joint_q, joint_qd):

    wp.launch(kernel=eval_articulation_ik,
        dim=model.body_count,
        inputs=[    
            state.body_q,
            state.body_qd,
            model.body_com,
            model.joint_type,
            model.joint_parent,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_q_start,
            model.joint_qd_start],
        outputs=[
            joint_q,
            joint_qd
        ],
        device=model.device)

