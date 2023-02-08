import warp as wp

PI = wp.constant(3.14159265359)
PI_2 = wp.constant(1.57079632679)


@wp.func
def velocity_at_point(
    qd: wp.spatial_vector,
    r: wp.vec3
):
    """
    Returns the velocity of a point relative to the frame with the given spatial velocity.
    """
    return wp.cross(wp.spatial_top(qd), r) + wp.spatial_bottom(qd)


@wp.func
def quat_twist(axis: wp.vec3, q: wp.quat):
    """
    Returns the twist around an axis.
    """

    # project imaginary part onto axis
    a = wp.vec3(q[0], q[1], q[2])
    proj = wp.dot(a, axis)
    a = proj*axis
    # if proj < 0.0:
    #     # ensure twist points in same direction as axis
    #     a = -a
    return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))


@wp.func
def quat_twist_angle(axis: wp.vec3, q: wp.quat):
    """
    Returns the angle of the twist around an axis.
    """
    return 2.0 * wp.acos(quat_twist(axis, q)[3])


@wp.func
def quat_decompose(q: wp.quat):
    """
    Decompose a quaternion into a sequence of 3 rotations around x,y',z' respectively, i.e.: q = q_z''q_y'q_x.
    """

    R = wp.mat33(
        wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))

    # https://www.sedris.org/wg8home/Documents/WG80485.pdf
    phi = wp.atan2(R[1, 2], R[2, 2])
    sinp = -R[0, 2]
    if wp.abs(sinp) >= 1.0:
        theta = 1.57079632679 * wp.sign(sinp)
    else:
        theta = wp.asin(-R[0, 2])
    psi = wp.atan2(R[0, 1], R[0, 0])

    return -wp.vec3(phi, theta, psi)


@wp.func
def quat_to_rpy(q: wp.quat):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = wp.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = wp.clamp(t2, -1.0, 1.0)
    pitch_y = wp.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = wp.atan2(t3, t4)

    return wp.vec3(roll_x, pitch_y, yaw_z)


@wp.func
def quat_to_euler(q: wp.quat, i: int, j: int, k: int) -> wp.vec3:
    """
    Convert a quaternion into euler angles
    i, j, k are the indices in [1,2,3] of the axes to use
    (i != j, j != k)
    """
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276302
    not_proper = True
    if i == k:
        not_proper = False
        k = 6 - i - j  # because i + j + k = 1 + 2 + 3 = 6
    e = float((i-j)*(j-k)*(k-i)) / 2.0  # Levi-Civita symbol
    a = q[0]
    b = q[i]
    c = q[j]
    d = q[k]*e
    if not_proper:
        a -= q[j]
        b += q[k]*e
        c += q[0]
        d -= q[i]
    t2 = wp.acos(2.0*(a*a + b*b)/(a*a + b*b + c*c + d*d) - 1.0)
    tp = wp.atan2(b, a)
    tm = wp.atan2(d, c)
    t1 = 0.0
    t3 = 0.0
    if wp.abs(t2) < 1e-6:
        t3 = 2.0*tp - t1
    elif wp.abs(t2-PI_2) < 1e-6:
        t3 = 2.0*tm + t1
    else:
        t1 = tp - tm
        t3 = tp + tm
    if not_proper:
        t2 -= PI_2
        t3 *= e
    return wp.vec3(t1, t2, t3)


@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector):
    # Frank & Park definition 3.20, pg 100

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
def transform_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates).
    (Frank & Park, section 8.2.3, pg 290)
    """

    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.mat33(r1, r2, r3)
    S = wp.mul(wp.skew(p), R)

    T = wp.spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


@wp.func
def vec_min(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def vec_max(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.func
def vec_abs(a: wp.vec3):
    return wp.vec3(wp.abs(a[0]), wp.abs(a[1]), wp.abs(a[2]))
