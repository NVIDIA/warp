import warp as wp

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
    if (articulation_mask and articulation_mask[tid]==0):
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
        #X_cj = joint_X_c[i]  # todo: assuming child origin is aligned with child joint
        
        q_start = joint_q_start[i]
        qd_start = joint_qd_start[i]

        # prismatic
        if type == 0:

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_jc = wp.transform(axis*q, wp.quat_identity())
            v_jc = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), axis*qd)

        # revolute
        if type == 1:

            q = joint_q[q_start]
            qd = joint_qd[qd_start]

            X_jc = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(axis, q))
            v_jc = wp.spatial_vector(axis*qd, wp.vec3(0.0, 0.0, 0.0))

        # ball
        if type == 2:

            r = wp.quat(joint_q[q_start+0],
                        joint_q[q_start+1],
                        joint_q[q_start+2],
                        joint_q[q_start+3])

            w = wp.vec3(joint_qd[qd_start+0],
                         joint_qd[qd_start+1],
                         joint_qd[qd_start+2])

            X_jc = wp.transform(wp.vec3(0.0, 0.0, 0.0), r)
            v_jc = wp.spatial_vector(w, wp.vec3(0.0, 0.0, 0.0))

        # fixed
        if type == 3:
            
            X_jc = wp.transform_identity()
            v_jc = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0))

        # free
        if type == 4:

            t = wp.transform(
                    wp.vec3(joint_q[q_start+0], joint_q[q_start+1], joint_q[q_start+2]),
                    wp.quat(joint_q[q_start+3], joint_q[q_start+4], joint_q[q_start+5], joint_q[q_start+6]))

            v = wp.spatial_vector(
                    wp.vec3(joint_qd[qd_start+0], joint_qd[qd_start+1], joint_qd[qd_start+2]),
                    wp.vec3(joint_qd[qd_start+3], joint_qd[qd_start+4], joint_qd[qd_start+5]))

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

