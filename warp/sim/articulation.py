

@wp.kernel
def eval_body_contacts_art(
    body_X_s: wp.array(dtype=wp.spatial_transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),
    contact_dist: wp.array(dtype=float),
    contact_mat: wp.array(dtype=int),
    materials: wp.array(dtype=float),
    body_f_s: wp.array(dtype=wp.spatial_vector)):

    tid = wp.tid()

    c_body = wp.load(contact_body, tid)
    c_point = wp.load(contact_point, tid)
    c_dist = wp.load(contact_dist, tid)
    c_mat = wp.load(contact_mat, tid)

    # hard coded surface parameter tensor layout (ke, kd, kf, mu)
    ke = wp.load(materials, c_mat * 4 + 0)       # restitution coefficient
    kd = wp.load(materials, c_mat * 4 + 1)       # damping coefficient
    kf = wp.load(materials, c_mat * 4 + 2)       # friction coefficient
    mu = wp.load(materials, c_mat * 4 + 3)       # coulomb friction

    X_s = wp.load(body_X_s, c_body)              # position of colliding body
    v_s = wp.load(body_v_s, c_body)              # orientation of colliding body

    n = vec3(0.0, 1.0, 0.0)

    # transform point to world space
    p = wp.spatial_transform_point(X_s, c_point) - n * c_dist  # add on 'thickness' of shape, e.g.: radius of sphere/capsule

    w = wp.spatial_top(v_s)
    v = wp.spatial_bottom(v_s)

    # contact point velocity
    dpdt = v + wp.cross(w, p)

    # check ground contact
    c = wp.min(dot(n, p), 0.0)         # check if we're inside the ground

    vn = dot(n, dpdt)        # velocity component out of the ground
    vt = dpdt - n * vn       # velocity component not into the ground

    fn = c * ke              # normal force (restitution coefficient * how far inside for ground)

    # contact damping
    fd = wp.min(vn, 0.0) * kd * wp.step(c)       # again, velocity into the ground, negative

    # viscous friction
    #ft = vt*kf

    # Coulomb friction (box)
    lower = mu * (fn + fd)   # negative
    upper = 0.0 - lower      # positive, workaround for no unary ops

    vx = wp.clamp(dot(vec3(kf, 0.0, 0.0), vt), lower, upper)
    vz = wp.clamp(dot(vec3(0.0, 0.0, kf), vt), lower, upper)

    ft = wp.vec3(vx, 0.0, vz) * wp.step(c)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    #ft = wp.normalize(vt)*wp.min(kf*wp.length(vt), 0.0 - mu*c*ke)

    f_total = n * (fn + fd) + ft
    t_total = wp.cross(p, f_total)

    wp.atomic_add(body_f_s, c_body, wp.spatial_vector(t_total, f_total))




# compute transform across a joint
@wp.func
def jcalc_transform(type: int, axis: wp.vec3, joint_q: wp.array(dtype=float), start: int):

    # prismatic
    if (type == 0):

        q = wp.load(joint_q, start)
        X_jc = spatial_transform(axis * q, quat_identity())
        return X_jc

    # revolute
    if (type == 1):

        q = wp.load(joint_q, start)
        X_jc = spatial_transform(vec3(0.0, 0.0, 0.0), quat_from_axis_angle(axis, q))
        return X_jc

    # ball
    if (type == 2):

        qx = wp.load(joint_q, start + 0)
        qy = wp.load(joint_q, start + 1)
        qz = wp.load(joint_q, start + 2)
        qw = wp.load(joint_q, start + 3)

        X_jc = spatial_transform(vec3(0.0, 0.0, 0.0), quat(qx, qy, qz, qw))
        return X_jc

    # fixed
    if (type == 3):

        X_jc = spatial_transform_identity()
        return X_jc

    # free
    if (type == 4):

        px = wp.load(joint_q, start + 0)
        py = wp.load(joint_q, start + 1)
        pz = wp.load(joint_q, start + 2)

        qx = wp.load(joint_q, start + 3)
        qy = wp.load(joint_q, start + 4)
        qz = wp.load(joint_q, start + 5)
        qw = wp.load(joint_q, start + 6)

        X_jc = spatial_transform(vec3(px, py, pz), quat(qx, qy, qz, qw))
        return X_jc

    # default case
    return spatial_transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(type: int, axis: wp.vec3, X_sc: wp.spatial_transform, joint_S_s: wp.array(dtype=wp.spatial_vector), joint_qd: wp.array(dtype=float), joint_start: int):

    # prismatic
    if (type == 0):

        S_s = wp.spatial_transform_twist(X_sc, wp.spatial_vector(vec3(0.0, 0.0, 0.0), axis))
        v_j_s = S_s * wp.load(joint_qd, joint_start)

        wp.store(joint_S_s, joint_start, S_s)
        return v_j_s

    # revolute
    if (type == 1):

        S_s = wp.spatial_transform_twist(X_sc, wp.spatial_vector(axis, vec3(0.0, 0.0, 0.0)))
        v_j_s = S_s * wp.load(joint_qd, joint_start)

        wp.store(joint_S_s, joint_start, S_s)
        return v_j_s

    # ball
    if (type == 2):

        w = vec3(wp.load(joint_qd, joint_start + 0),
                   wp.load(joint_qd, joint_start + 1),
                   wp.load(joint_qd, joint_start + 2))

        S_0 = wp.spatial_transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        S_1 = wp.spatial_transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        S_2 = wp.spatial_transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

        # write motion subspace
        wp.store(joint_S_s, joint_start + 0, S_0)
        wp.store(joint_S_s, joint_start + 1, S_1)
        wp.store(joint_S_s, joint_start + 2, S_2)

        return S_0*w[0] + S_1*w[1] + S_2*w[2]

    # fixed
    if (type == 3):
        return wp.spatial_vector()

    # free
    if (type == 4):

        v_j_s = wp.spatial_vector(wp.load(joint_qd, joint_start + 0),
                               wp.load(joint_qd, joint_start + 1),
                               wp.load(joint_qd, joint_start + 2),
                               wp.load(joint_qd, joint_start + 3),
                               wp.load(joint_qd, joint_start + 4),
                               wp.load(joint_qd, joint_start + 5))

        # write motion subspace
        wp.store(joint_S_s, joint_start + 0, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        wp.store(joint_S_s, joint_start + 1, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        wp.store(joint_S_s, joint_start + 2, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        wp.store(joint_S_s, joint_start + 3, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        wp.store(joint_S_s, joint_start + 4, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        wp.store(joint_S_s, joint_start + 5, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    # default case
    return wp.spatial_vector()


# # compute the velocity across a joint
# #@wp.func
# def jcalc_velocity(self, type, S_s, joint_qd, start):

#     # prismatic
#     if (type == 0):
#         v_j_s = wp.load(S_s, start)*wp.load(joint_qd, start)
#         return v_j_s

#     # revolute
#     if (type == 1):
#         v_j_s = wp.load(S_s, start)*wp.load(joint_qd, start)
#         return v_j_s

#     # fixed
#     if (type == 2):
#         v_j_s = wp.spatial_vector()
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
@wp.func
def jcalc_tau(
    type: int, 
    target_k_e: float,
    target_k_d: float,
    limit_k_e: float,
    limit_k_d: float,
    joint_S_s: wp.array(dtype=wp.spatial_vector), 
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    coord_start: int,
    dof_start: int, 
    body_f_s: wp.spatial_vector, 
    tau: wp.array(dtype=float)):

    # prismatic / revolute
    if (type == 0 or type == 1):
        S_s = wp.load(joint_S_s, dof_start)

        q = wp.load(joint_q, coord_start)
        qd = wp.load(joint_qd, dof_start)
        act = wp.load(joint_act, dof_start)

        target = wp.load(joint_target, coord_start)
        lower = wp.load(joint_limit_lower, coord_start)
        upper = wp.load(joint_limit_upper, coord_start)

        limit_f = 0.0

        # compute limit forces, damping only active when limit is violated
        if (q < lower):
            limit_f = limit_k_e*(lower-q) - limit_k_d*min(qd, 0.0)

        if (q > upper):
            limit_f = limit_k_e*(upper-q) - limit_k_d*max(qd, 0.0)

        # total torque / force on the joint
        t = 0.0 - spatial_dot(S_s, body_f_s) - target_k_e*(q - target) - target_k_d*qd + act + limit_f

        wp.store(tau, dof_start, t)

    # ball
    if (type == 2):

        # elastic term.. this is proportional to the 
        # imaginary part of the relative quaternion
        r_j = vec3(wp.load(joint_q, coord_start + 0),  
                     wp.load(joint_q, coord_start + 1), 
                     wp.load(joint_q, coord_start + 2))                     

        # angular velocity for damping
        w_j = vec3(wp.load(joint_qd, dof_start + 0),  
                     wp.load(joint_qd, dof_start + 1), 
                     wp.load(joint_qd, dof_start + 2))

        for i in range(0, 3):
            S_s = wp.load(joint_S_s, dof_start+i)

            w = w_j[i]
            r = r_j[i]

            wp.store(tau, dof_start+i, 0.0 - spatial_dot(S_s, body_f_s) - w*target_k_d - r*target_k_e)

    # fixed
    # if (type == 3)
    #    pass

    # free
    if (type == 4):
            
        for i in range(0, 6):
            S_s = wp.load(joint_S_s, dof_start+i)
            wp.store(tau, dof_start+i, 0.0 - spatial_dot(S_s, body_f_s))

    return 0


@wp.func
def jcalc_integrate(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    dt: float,
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float)):

    # prismatic / revolute
    if (type == 0 or type == 1):

        qdd = wp.load(joint_qdd, dof_start)
        qd = wp.load(joint_qd, dof_start)
        q = wp.load(joint_q, coord_start)

        qd_new = qd + qdd*dt
        q_new = q + qd_new*dt

        wp.store(joint_qd_new, dof_start, qd_new)
        wp.store(joint_q_new, coord_start, q_new)

    # ball
    if (type == 2):

        m_j = vec3(wp.load(joint_qdd, dof_start + 0),
                     wp.load(joint_qdd, dof_start + 1),
                     wp.load(joint_qdd, dof_start + 2))

        w_j = vec3(wp.load(joint_qd, dof_start + 0),  
                     wp.load(joint_qd, dof_start + 1), 
                     wp.load(joint_qd, dof_start + 2)) 

        r_j = quat(wp.load(joint_q, coord_start + 0), 
                   wp.load(joint_q, coord_start + 1), 
                   wp.load(joint_q, coord_start + 2), 
                   wp.load(joint_q, coord_start + 3))

        # symplectic Euler
        w_j_new = w_j + m_j*dt

        drdt_j = mul(quat(w_j_new, 0.0), r_j) * 0.5

        # new orientation (normalized)
        r_j_new = normalize(r_j + drdt_j * dt)

        # update joint coords
        wp.store(joint_q_new, coord_start + 0, r_j_new[0])
        wp.store(joint_q_new, coord_start + 1, r_j_new[1])
        wp.store(joint_q_new, coord_start + 2, r_j_new[2])
        wp.store(joint_q_new, coord_start + 3, r_j_new[3])

        # update joint vel
        wp.store(joint_qd_new, dof_start + 0, w_j_new[0])
        wp.store(joint_qd_new, dof_start + 1, w_j_new[1])
        wp.store(joint_qd_new, dof_start + 2, w_j_new[2])

    # fixed joint
    #if (type == 3)
    #    pass

    # free joint
    if (type == 4):

        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear acceleration
        m_s = vec3(wp.load(joint_qdd, dof_start + 0),
                     wp.load(joint_qdd, dof_start + 1),
                     wp.load(joint_qdd, dof_start + 2))

        a_s = vec3(wp.load(joint_qdd, dof_start + 3), 
                     wp.load(joint_qdd, dof_start + 4), 
                     wp.load(joint_qdd, dof_start + 5))

        # angular and linear velocity
        w_s = vec3(wp.load(joint_qd, dof_start + 0),  
                     wp.load(joint_qd, dof_start + 1), 
                     wp.load(joint_qd, dof_start + 2))
        
        v_s = vec3(wp.load(joint_qd, dof_start + 3),
                     wp.load(joint_qd, dof_start + 4),
                     wp.load(joint_qd, dof_start + 5))

        # symplectic Euler
        w_s = w_s + m_s*dt
        v_s = v_s + a_s*dt
        
        # translation of origin
        p_s = vec3(wp.load(joint_q, coord_start + 0),
                     wp.load(joint_q, coord_start + 1), 
                     wp.load(joint_q, coord_start + 2))

        # linear vel of origin (note q/qd switch order of linear angular elements) 
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velcity
        dpdt_s = v_s + cross(w_s, p_s)
        
        # quat and quat derivative
        r_s = quat(wp.load(joint_q, coord_start + 3), 
                   wp.load(joint_q, coord_start + 4), 
                   wp.load(joint_q, coord_start + 5), 
                   wp.load(joint_q, coord_start + 6))

        drdt_s = mul(quat(w_s, 0.0), r_s) * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = normalize(r_s + drdt_s * dt)

        # update transform
        wp.store(joint_q_new, coord_start + 0, p_s_new[0])
        wp.store(joint_q_new, coord_start + 1, p_s_new[1])
        wp.store(joint_q_new, coord_start + 2, p_s_new[2])

        wp.store(joint_q_new, coord_start + 3, r_s_new[0])
        wp.store(joint_q_new, coord_start + 4, r_s_new[1])
        wp.store(joint_q_new, coord_start + 5, r_s_new[2])
        wp.store(joint_q_new, coord_start + 6, r_s_new[3])

        # update joint_twist
        wp.store(joint_qd_new, dof_start + 0, w_s[0])
        wp.store(joint_qd_new, dof_start + 1, w_s[1])
        wp.store(joint_qd_new, dof_start + 2, w_s[2])
        wp.store(joint_qd_new, dof_start + 3, v_s[0])
        wp.store(joint_qd_new, dof_start + 4, v_s[1])
        wp.store(joint_qd_new, dof_start + 5, v_s[2])

    return 0

@wp.func
def compute_link_transform(i: int,
                           joint_type: wp.array(dtype=int),
                           joint_parent: wp.array(dtype=int),
                           joint_q_start: wp.array(dtype=int),
                           joint_qd_start: wp.array(dtype=int),
                           joint_q: wp.array(dtype=float),
                           joint_X_pj: wp.array(dtype=wp.spatial_transform),
                           joint_X_cm: wp.array(dtype=wp.spatial_transform),
                           joint_axis: wp.array(dtype=wp.vec3),
                           body_X_sc: wp.array(dtype=wp.spatial_transform),
                           body_X_sm: wp.array(dtype=wp.spatial_transform)):

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


@wp.kernel
def eval_body_fk(articulation_start: wp.array(dtype=int),
                  joint_type: wp.array(dtype=int),
                  joint_parent: wp.array(dtype=int),
                  joint_q_start: wp.array(dtype=int),
                  joint_qd_start: wp.array(dtype=int),
                  joint_q: wp.array(dtype=float),
                  joint_X_pj: wp.array(dtype=wp.spatial_transform),
                  joint_X_cm: wp.array(dtype=wp.spatial_transform),
                  joint_axis: wp.array(dtype=wp.vec3),
                  body_X_sc: wp.array(dtype=wp.spatial_transform),
                  body_X_sm: wp.array(dtype=wp.spatial_transform)):

    # one thread per-articulation
    index = tid()

    start = wp.load(articulation_start, index)
    end = wp.load(articulation_start, index+1)

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




@wp.func
def compute_link_velocity(i: int,
                          joint_type: wp.array(dtype=int),
                          joint_parent: wp.array(dtype=int),
                          joint_qd_start: wp.array(dtype=int),
                          joint_qd: wp.array(dtype=float),
                          joint_axis: wp.array(dtype=wp.vec3),
                          body_I_m: wp.array(dtype=wp.spatial_matrix),
                          body_X_sc: wp.array(dtype=wp.spatial_transform),
                          body_X_sm: wp.array(dtype=wp.spatial_transform),
                          joint_X_pj: wp.array(dtype=wp.spatial_transform),
                          gravity: wp.vec3,
                          # outputs
                          joint_S_s: wp.array(dtype=wp.spatial_vector),
                          body_I_s: wp.array(dtype=wp.spatial_matrix),
                          body_v_s: wp.array(dtype=wp.spatial_vector),
                          body_f_s: wp.array(dtype=wp.spatial_vector),
                          body_a_s: wp.array(dtype=wp.spatial_vector)):

    type = wp.load(joint_type, i)
    axis = wp.load(joint_axis, i)
    parent = wp.load(joint_parent, i)
    dof_start = wp.load(joint_qd_start, i)
    
    X_sc = wp.load(body_X_sc, i)

    # parent transform in spatial coordinates
    X_sp = spatial_transform_identity()
    if (parent >= 0):
        X_sp = load(body_X_sc, parent)

    X_pj = load(joint_X_pj, i)
    X_sj = spatial_transform_multiply(X_sp, X_pj)

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    v_j_s = jcalc_motion(type, axis, X_sj, joint_S_s, joint_qd, dof_start)

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if (parent >= 0):
        v_parent_s = wp.load(body_v_s, parent)
        a_parent_s = wp.load(body_a_s, parent)

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s) # + self.joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = wp.load(body_X_sm, i)
    I_m = wp.load(body_I_m, i)

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    m = I_m[3, 3]

    f_g_m = wp.spatial_vector(vec3(), gravity) * m
    f_g_s = spatial_transform_wrench(spatial_transform(spatial_transform_get_translation(X_sm), quat_identity()), f_g_m)

    #f_ext_s = wp.load(body_f_s, i) + f_g_s

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = wp.mul(I_s, a_s) + spatial_cross_dual(v_s, wp.mul(I_s, v_s))

    wp.store(body_v_s, i, v_s)
    wp.store(body_a_s, i, a_s)
    wp.store(body_f_s, i, f_b_s - f_g_s)
    wp.store(body_I_s, i, I_s)

    return 0


@wp.func
def compute_link_tau(offset: int,
                     joint_end: int,
                     joint_type: wp.array(dtype=int),
                     joint_parent: wp.array(dtype=int),
                     joint_q_start: wp.array(dtype=int),
                     joint_qd_start: wp.array(dtype=int),
                     joint_q: wp.array(dtype=float),
                     joint_qd: wp.array(dtype=float),
                     joint_act: wp.array(dtype=float),
                     joint_target: wp.array(dtype=float),
                     joint_target_ke: wp.array(dtype=float),
                     joint_target_kd: wp.array(dtype=float),
                     joint_limit_lower: wp.array(dtype=float),
                     joint_limit_upper: wp.array(dtype=float),
                     joint_limit_ke: wp.array(dtype=float),
                     joint_limit_kd: wp.array(dtype=float),
                     joint_S_s: wp.array(dtype=wp.spatial_vector),
                     body_fb_s: wp.array(dtype=wp.spatial_vector),
                     # outputs
                     body_ft_s: wp.array(dtype=wp.spatial_vector),
                     tau: wp.array(dtype=float)):

    # for backwards traversal
    i = joint_end-offset-1

    type = wp.load(joint_type, i)
    parent = wp.load(joint_parent, i)
    dof_start = wp.load(joint_qd_start, i)
    coord_start = wp.load(joint_q_start, i)

    target_k_e = wp.load(joint_target_ke, i)
    target_k_d = wp.load(joint_target_kd, i)

    limit_k_e = wp.load(joint_limit_ke, i)
    limit_k_d = wp.load(joint_limit_kd, i)

    # total forces on body
    f_b_s = wp.load(body_fb_s, i)
    f_t_s = wp.load(body_ft_s, i)

    f_s = f_b_s + f_t_s

    # compute joint-space forces, writes out tau
    jcalc_tau(type, target_k_e, target_k_d, limit_k_e, limit_k_d, joint_S_s, joint_q, joint_qd, joint_act, joint_target, joint_limit_lower, joint_limit_upper, coord_start, dof_start, f_s, tau)

    # update parent forces, todo: check that this is valid for the backwards pass
    if (parent >= 0):
        wp.atomic_add(body_ft_s, parent, f_s)

    return 0


@wp.kernel
def eval_body_id(articulation_start: wp.array(dtype=int),
                  joint_type: wp.array(dtype=int),
                  joint_parent: wp.array(dtype=int),
                  joint_q_start: wp.array(dtype=int),
                  joint_qd_start: wp.array(dtype=int),
                  joint_q: wp.array(dtype=float),
                  joint_qd: wp.array(dtype=float),
                  joint_axis: wp.array(dtype=wp.vec3),
                  joint_target_ke: wp.array(dtype=float),
                  joint_target_kd: wp.array(dtype=float),             
                  body_I_m: wp.array(dtype=wp.spatial_matrix),
                  body_X_sc: wp.array(dtype=wp.spatial_transform),
                  body_X_sm: wp.array(dtype=wp.spatial_transform),
                  joint_X_pj: wp.array(dtype=wp.spatial_transform),
                  gravity: wp.vec3,
                  # outputs
                  joint_S_s: wp.array(dtype=wp.spatial_vector),
                  body_I_s: wp.array(dtype=wp.spatial_matrix),
                  body_v_s: wp.array(dtype=wp.spatial_vector),
                  body_f_s: wp.array(dtype=wp.spatial_vector),
                  body_a_s: wp.array(dtype=wp.spatial_vector)):

    # one thread per-articulation
    index = tid()

    start = wp.load(articulation_start, index)
    end = wp.load(articulation_start, index+1)
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


@wp.kernel
def eval_body_tau(articulation_start: wp.array(dtype=int),
                  joint_type: wp.array(dtype=int),
                  joint_parent: wp.array(dtype=int),
                  joint_q_start: wp.array(dtype=int),
                  joint_qd_start: wp.array(dtype=int),
                  joint_q: wp.array(dtype=float),
                  joint_qd: wp.array(dtype=float),
                  joint_act: wp.array(dtype=float),
                  joint_target: wp.array(dtype=float),
                  joint_target_ke: wp.array(dtype=float),
                  joint_target_kd: wp.array(dtype=float),
                  joint_limit_lower: wp.array(dtype=float),
                  joint_limit_upper: wp.array(dtype=float),
                  joint_limit_ke: wp.array(dtype=float),
                  joint_limit_kd: wp.array(dtype=float),
                  joint_axis: wp.array(dtype=wp.vec3),
                  joint_S_s: wp.array(dtype=wp.spatial_vector),
                  body_fb_s: wp.array(dtype=wp.spatial_vector),                  
                  # outputs
                  body_ft_s: wp.array(dtype=wp.spatial_vector),
                  tau: wp.array(dtype=float)):

    # one thread per-articulation
    index = tid()

    start = wp.load(articulation_start, index)
    end = wp.load(articulation_start, index+1)
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

@wp.kernel
def eval_body_jacobian(
    articulation_start: wp.array(dtype=int),
    articulation_J_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    J: wp.array(dtype=float)):

    # one thread per-articulation
    index = tid()

    joint_start = wp.load(articulation_start, index)
    joint_end = wp.load(articulation_start, index+1)
    joint_count = joint_end-joint_start

    J_offset = wp.load(articulation_J_start, index)

    # in spatial.h
    spatial_jacobian(joint_S_s, joint_parent, joint_qd_start, joint_start, joint_count, J_offset, J)


# @wp.kernel
# def eval_body_jacobian(
#     articulation_start: wp.array(dtype=int),
#     articulation_J_start: wp.array(dtype=int),    
#     joint_parent: wp.array(dtype=int),
#     joint_qd_start: wp.array(dtype=int),
#     joint_S_s: wp.array(dtype=wp.spatial_vector),
#     # outputs
#     J: wp.array(dtype=float)):

#     # one thread per-articulation
#     index = tid()

#     joint_start = wp.load(articulation_start, index)
#     joint_end = wp.load(articulation_start, index+1)
#     joint_count = joint_end-joint_start

#     dof_start = wp.load(joint_qd_start, joint_start)
#     dof_end = wp.load(joint_qd_start, joint_end)
#     dof_count = dof_end-dof_start

#     #(const wp.spatial_vector* S, const int* joint_parents, const int* joint_qd_start, int num_links, int num_dofs, float* J)
#     spatial_jacobian(joint_S_s, joint_parent, joint_qd_start, joint_count, dof_count, J)



@wp.kernel
def eval_body_mass(
    articulation_start: wp.array(dtype=int),
    articulation_M_start: wp.array(dtype=int),    
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    # outputs
    M: wp.array(dtype=float)):

    # one thread per-articulation
    index = tid()

    joint_start = wp.load(articulation_start, index)
    joint_end = wp.load(articulation_start, index+1)
    joint_count = joint_end-joint_start

    M_offset = wp.load(articulation_M_start, index)

    # in spatial.h
    spatial_mass(body_I_s, joint_start, joint_count, M_offset, M)




@wp.kernel
def eval_body_integrate(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float)):

    # one thread per-articulation
    index = tid()

    type = wp.load(joint_type, index)
    coord_start = wp.load(joint_q_start, index)
    dof_start = wp.load(joint_qd_start, index)

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
