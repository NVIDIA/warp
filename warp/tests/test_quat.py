import numpy as np
import math

import warp as wp
from warp.tests.test_base import *

wp.init()

############################################################

@wp.kernel
def rodriguez_kernel(rotators: wp.array(dtype=wp.vec3), u: wp.vec3, loss: wp.array(dtype=float), coord_idx: int):
    tid = wp.tid()
    v = rotators[tid]
    u_new = wp.rotate_rodriguez(v, u)
    wp.atomic_add(loss, 0, u_new[coord_idx])

@wp.kernel
def rodriguez_kernel_autodiff(rotators: wp.array(dtype=wp.vec3), u: wp.vec3, loss: wp.array(dtype=float), coord_idx: int):
    tid = wp.tid()
    v = rotators[tid]
    angle = wp.length(v)
    if angle != 0.0:
        axis = v / angle
        u_new = u * wp.cos(angle) + wp.cross(axis, u) * wp.sin(angle) + axis * wp.dot(axis, u) * (1.0 - wp.cos(angle))
    else:
        u_new = u
    wp.atomic_add(loss, 0, u_new[coord_idx])

@wp.kernel
def rodriguez_kernel_forward(rotators: wp.array(dtype=wp.vec3), u: wp.vec3, rotated: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    v = rotators[tid]
    angle = wp.length(v)
    if angle != 0.0:
        axis = v / angle
        rotated[tid] = u * wp.cos(angle) + wp.cross(axis, u) * wp.sin(angle) + axis * wp.dot(axis, u) * (1.0 - wp.cos(angle))
    else:
        rotated[tid] = u

def test_rotate_rodriguez_grad(test, device):
    np.random.seed(42)
    num_rotators = 50
    x = wp.vec3(0.5, 0.5, 0.0)
    rotators = np.random.randn(num_rotators, 3)
    rotators = wp.array(rotators, dtype=wp.vec3, device=device, requires_grad=True)

    def compute_gradients(kernel, index):
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel=kernel,
                dim=num_rotators,
                inputs=[rotators, x, loss, index],
                device=device)

            tape.backward(loss)

        gradients = tape.gradients[rotators].numpy()
        tape.zero()

        return gradients

    # gather gradients from builtin adjoints
    gradients_x = compute_gradients(rodriguez_kernel, 0)
    gradients_y = compute_gradients(rodriguez_kernel, 1)
    gradients_z = compute_gradients(rodriguez_kernel, 2)

    # gather gradients from autodiff
    gradients_x_auto = compute_gradients(rodriguez_kernel_autodiff, 0)
    gradients_y_auto = compute_gradients(rodriguez_kernel_autodiff, 1)
    gradients_z_auto = compute_gradients(rodriguez_kernel_autodiff, 2)

    # compare autodiff and adj_rotate_rodriguez
    eps = 2.0e-6

    test.assertTrue((np.abs(gradients_x - gradients_x_auto) < eps).all())
    test.assertTrue((np.abs(gradients_y - gradients_y_auto) < eps).all())
    test.assertTrue((np.abs(gradients_z - gradients_z_auto) < eps).all())

############################################################

@wp.kernel
def slerp_kernel(
    q0: wp.array(dtype=wp.quat),
    q1: wp.array(dtype=wp.quat),
    t: wp.array(dtype=float),
    loss: wp.array(dtype=float),
    index: int):

    tid = wp.tid()

    q = wp.quat_slerp(q0[tid], q1[tid], t[tid])
    wp.atomic_add(loss, 0, q[index])

@wp.kernel
def slerp_kernel_forward(
    q0: wp.array(dtype=wp.quat),
    q1: wp.array(dtype=wp.quat),
    t: wp.array(dtype=float),
    loss: wp.array(dtype=float),
    index: int):

    tid = wp.tid()

    axis = wp.vec3()
    angle = float(0.0)

    wp.quat_to_axis_angle(wp.mul(wp.quat_inverse(q0[tid]), q1[tid]), axis, angle)
    q = wp.mul(q0[tid], wp.quat_from_axis_angle(axis, t[tid] * angle))

    wp.atomic_add(loss, 0, q[index])

@wp.kernel
def quat_sampler(
    kernel_seed: int,
    quats: wp.array(dtype=wp.quat)):

    tid = wp.tid()

    state = wp.rand_init(kernel_seed, tid)

    angle = wp.randf(state, 0.0, 2.0 * 3.1415926535)
    dir = wp.sample_unit_sphere_surface(state) * wp.sin(angle*0.5)

    q = wp.quat(dir[0], dir[1], dir[2], wp.cos(angle*0.5))
    qn = wp.normalize(q)

    quats[tid] = qn

def test_slerp_grad(test, device):
    seed = 42
    np.random.seed(seed)
    N = 50

    q0 = wp.zeros(N, dtype=wp.quat, device=device, requires_grad=True)
    q1 = wp.zeros(N, dtype=wp.quat, device=device, requires_grad=True)

    wp.launch(kernel=quat_sampler, dim=N, inputs=[seed, q0], device=device)
    wp.launch(kernel=quat_sampler, dim=N, inputs=[seed, q1], device=device)

    t = np.random.uniform(0.0, 1.0, N)
    t = wp.array(t, dtype=float, device=device, requires_grad=True)

    def compute_gradients(kernel, wrt, index):
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel=kernel,
                dim=N,
                inputs=[q0, q1, t, loss, index],
                device=device)

            tape.backward(loss)

        gradients = tape.gradients[wrt].numpy()
        tape.zero()

        return gradients

    # wrt t

    # gather gradients from builtin adjoints
    gradients_x = compute_gradients(slerp_kernel, t, 0)
    gradients_y = compute_gradients(slerp_kernel, t, 1)
    gradients_z = compute_gradients(slerp_kernel, t, 2)
    gradients_w = compute_gradients(slerp_kernel, t, 3)

    # gather gradients from autodiff
    gradients_x_auto = compute_gradients(slerp_kernel_forward, t, 0)
    gradients_y_auto = compute_gradients(slerp_kernel_forward, t, 1)
    gradients_z_auto = compute_gradients(slerp_kernel_forward, t, 2)
    gradients_w_auto = compute_gradients(slerp_kernel_forward, t, 3)

    eps = 2.0e-6

    test.assertTrue((np.abs(gradients_x - gradients_x_auto) < eps).all())
    test.assertTrue((np.abs(gradients_y - gradients_y_auto) < eps).all())
    test.assertTrue((np.abs(gradients_z - gradients_z_auto) < eps).all())
    test.assertTrue((np.abs(gradients_w - gradients_w_auto) < eps).all())

    # wrt q0

    # gather gradients from builtin adjoints
    gradients_x = compute_gradients(slerp_kernel, q0, 0)
    gradients_y = compute_gradients(slerp_kernel, q0, 1)
    gradients_z = compute_gradients(slerp_kernel, q0, 2)
    gradients_w = compute_gradients(slerp_kernel, q0, 3)

    # gather gradients from autodiff
    gradients_x_auto = compute_gradients(slerp_kernel_forward, q0, 0)
    gradients_y_auto = compute_gradients(slerp_kernel_forward, q0, 1)
    gradients_z_auto = compute_gradients(slerp_kernel_forward, q0, 2)
    gradients_w_auto = compute_gradients(slerp_kernel_forward, q0, 3)

    test.assertTrue((np.abs(gradients_x - gradients_x_auto) < eps).all())
    test.assertTrue((np.abs(gradients_y - gradients_y_auto) < eps).all())
    test.assertTrue((np.abs(gradients_z - gradients_z_auto) < eps).all())
    test.assertTrue((np.abs(gradients_w - gradients_w_auto) < eps).all())

    # wrt q1

    # gather gradients from builtin adjoints
    gradients_x = compute_gradients(slerp_kernel, q1, 0)
    gradients_y = compute_gradients(slerp_kernel, q1, 1)
    gradients_z = compute_gradients(slerp_kernel, q1, 2)
    gradients_w = compute_gradients(slerp_kernel, q1, 3)

    # gather gradients from autodiff
    gradients_x_auto = compute_gradients(slerp_kernel_forward, q1, 0)
    gradients_y_auto = compute_gradients(slerp_kernel_forward, q1, 1)
    gradients_z_auto = compute_gradients(slerp_kernel_forward, q1, 2)
    gradients_w_auto = compute_gradients(slerp_kernel_forward, q1, 3)

    test.assertTrue((np.abs(gradients_x - gradients_x_auto) < eps).all())
    test.assertTrue((np.abs(gradients_y - gradients_y_auto) < eps).all())
    test.assertTrue((np.abs(gradients_z - gradients_z_auto) < eps).all())
    test.assertTrue((np.abs(gradients_w - gradients_w_auto) < eps).all())

############################################################

@wp.kernel
def quat_to_axis_angle_kernel(
    quats: wp.array(dtype=wp.quat),
    loss: wp.array(dtype=float),
    coord_idx: int):

    tid = wp.tid()
    axis = wp.vec3()
    angle = float(0.0)

    wp.quat_to_axis_angle(quats[tid], axis, angle)
    a = wp.vec4(axis[0], axis[1], axis[2], angle)

    wp.atomic_add(loss, 0, a[coord_idx])

@wp.kernel
def quat_to_axis_angle_kernel_forward(
    quats: wp.array(dtype=wp.quat),
    loss: wp.array(dtype=float),
    coord_idx: int):

    tid = wp.tid()
    q = quats[tid]
    axis = wp.vec3()
    angle = float(0.0)

    v = wp.vec3(q[0], q[1], q[2])
    if q[3] < 0.0:
        axis = -wp.normalize(v)
    else:
        axis = wp.normalize(v)

    angle = 2.0 * wp.atan2(wp.length(v), wp.abs(q[3]))
    a = wp.vec4(axis[0], axis[1], axis[2], angle)

    wp.atomic_add(loss, 0, a[coord_idx])

def test_quat_to_axis_angle_grad(test, device):
    
    np.random.seed(42)
    seed = 42
    num_rand = 50
    
    quats = wp.zeros(num_rand, dtype=wp.quat, device=device, requires_grad=True)
    wp.launch(kernel=quat_sampler, dim=num_rand, inputs=[seed, quats], device=device)
    
    edge_cases = np.array([(1.0, 0.0, 0.0, 0.0), (0.0, 1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)), (0.0, 0.0, 0.0, 0.0)])
    num_edge = len(edge_cases)
    edge_cases = wp.array(edge_cases, dtype=wp.quat, device=device, requires_grad=True)

    def compute_gradients(arr, kernel, dim, index):
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel=kernel,
                dim=dim,
                inputs=[arr, loss, index],
                device=device)

            tape.backward(loss)

        gradients = tape.gradients[arr].numpy()
        tape.zero()

        return gradients

    # gather gradients from builtin adjoints
    gradients_x = compute_gradients(quats, quat_to_axis_angle_kernel, num_rand, 0)
    gradients_y = compute_gradients(quats, quat_to_axis_angle_kernel, num_rand, 1)
    gradients_z = compute_gradients(quats, quat_to_axis_angle_kernel, num_rand, 2)
    gradients_w = compute_gradients(quats, quat_to_axis_angle_kernel, num_rand, 3)

    # gather gradients from autodiff
    gradients_x_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, num_rand, 0)
    gradients_y_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, num_rand, 1)
    gradients_z_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, num_rand, 2)
    gradients_w_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, num_rand, 3)

    # edge cases: gather gradients from builtin adjoints
    edge_gradients_x = compute_gradients(edge_cases, quat_to_axis_angle_kernel, num_edge, 0)
    edge_gradients_y = compute_gradients(edge_cases, quat_to_axis_angle_kernel, num_edge, 1)
    edge_gradients_z = compute_gradients(edge_cases, quat_to_axis_angle_kernel, num_edge, 2)
    edge_gradients_w = compute_gradients(edge_cases, quat_to_axis_angle_kernel, num_edge, 3)

    # edge cases: gather gradients from autodiff
    edge_gradients_x_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, num_edge, 0)
    edge_gradients_y_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, num_edge, 1)
    edge_gradients_z_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, num_edge, 2)
    edge_gradients_w_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, num_edge, 3)

    # compare
    eps = 1.0e6

    test.assertTrue((np.abs(gradients_x - gradients_x_auto) < eps).all())
    test.assertTrue((np.abs(gradients_y - gradients_y_auto) < eps).all())
    test.assertTrue((np.abs(gradients_z - gradients_z_auto) < eps).all())
    test.assertTrue((np.abs(gradients_w - gradients_w_auto) < eps).all())

    test.assertTrue((np.abs(edge_gradients_x - edge_gradients_x_auto) < eps).all())
    test.assertTrue((np.abs(edge_gradients_y - edge_gradients_y_auto) < eps).all())
    test.assertTrue((np.abs(edge_gradients_z - edge_gradients_z_auto) < eps).all())
    test.assertTrue((np.abs(edge_gradients_w - edge_gradients_w_auto) < eps).all())

############################################################

@wp.kernel
def rpy_to_quat_kernel(
    rpy_arr: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=float),
    coord_idx: int):

    tid = wp.tid()
    rpy = rpy_arr[tid]
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]

    q = wp.quat_rpy(roll, pitch, yaw)

    wp.atomic_add(loss, 0, q[coord_idx])

@wp.kernel
def rpy_to_quat_kernel_forward(
    rpy_arr: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=float),
    coord_idx: int):

    tid = wp.tid()
    rpy = rpy_arr[tid]
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]

    cy = wp.cos(yaw * 0.5)
    sy = wp.sin(yaw * 0.5)
    cr = wp.cos(roll * 0.5)
    sr = wp.sin(roll * 0.5)
    cp = wp.cos(pitch * 0.5)
    sp = wp.sin(pitch * 0.5)

    w = (cy * cr * cp + sy * sr * sp)
    x = (cy * sr * cp - sy * cr * sp)
    y = (cy * cr * sp + sy * sr * cp)
    z = (sy * cr * cp - cy * sr * sp)

    q = wp.quat(x, y, z, w)

    wp.atomic_add(loss, 0, q[coord_idx])

def test_quat_rpy_grad(test, device):
    seed = 42
    np.random.seed(seed)
    N = 3

    rpy_arr = np.random.uniform(-np.pi, np.pi, (N,3))
    rpy_arr = wp.array(rpy_arr, dtype=wp.vec3, device=device, requires_grad=True)

    def compute_gradients(kernel, wrt, index):
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel=kernel,
                dim=N,
                inputs=[wrt, loss, index],
                device=device)

            tape.backward(loss)

        gradients = tape.gradients[wrt].numpy()
        tape.zero()

        return gradients

    # wrt rpy

    # gather gradients from builtin adjoints
    gradients_r = compute_gradients(rpy_to_quat_kernel, rpy_arr, 0)
    gradients_p = compute_gradients(rpy_to_quat_kernel, rpy_arr, 1)
    gradients_y = compute_gradients(rpy_to_quat_kernel, rpy_arr, 2)

    # gather gradients from autodiff
    gradients_r_auto = compute_gradients(rpy_to_quat_kernel_forward, rpy_arr, 0)
    gradients_p_auto = compute_gradients(rpy_to_quat_kernel_forward, rpy_arr, 1)
    gradients_y_auto = compute_gradients(rpy_to_quat_kernel_forward, rpy_arr, 2)

    eps = 2.0e-6

    test.assertTrue((np.abs(gradients_r - gradients_r_auto) < eps).all())
    test.assertTrue((np.abs(gradients_p - gradients_p_auto) < eps).all())
    test.assertTrue((np.abs(gradients_y - gradients_y_auto) < eps).all())

############################################################

@wp.kernel
def quat_from_matrix(
    m: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
    idx: int):

    tid = wp.tid()

    matrix = wp.mat33(
        m[tid,0], m[tid,1], m[tid,2],
        m[tid,3], m[tid,4], m[tid,5],
        m[tid,6], m[tid,7], m[tid,8])

    q = wp.quat_from_matrix(matrix)

    wp.atomic_add(loss, 0, q[idx])

@wp.func
def quat_from_matrix_func(m: wp.mat33):

    tr = m[0][0] + m[1][1] + m[2][2]
    x = 0.0
    y = 0.0
    z = 0.0
    w = 0.0
    h = 0.0

    if (tr >= 0.0):
        h = wp.sqrt(tr + 1.0)
        w = 0.5 * h
        h = 0.5 / h

        x = (m[2][1] - m[1][2]) * h
        y = (m[0][2] - m[2][0]) * h
        z = (m[1][0] - m[0][1]) * h
    else:
        max_diag = 0
        if (m[1][1] > m[0][0]):
            max_diag = 1
        if (m[2][2] > m[max_diag][max_diag]):
            max_diag = 2

        if (max_diag == 0):
            h = wp.sqrt((m[0][0] - (m[1][1] + m[2][2])) + 1.0)
            x = 0.5 * h
            h = 0.5 / h

            y = (m[0][1] + m[1][0]) * h
            z = (m[2][0] + m[0][2]) * h
            w = (m[2][1] - m[1][2]) * h
        elif (max_diag == 1):
            h = wp.sqrt((m[1][1] - (m[2][2] + m[0][0])) + 1.0)
            y = 0.5 * h
            h = 0.5 / h

            z = (m[1][2] + m[2][1]) * h
            x = (m[0][1] + m[1][0]) * h
            w = (m[0][2] - m[2][0]) * h
        if (max_diag == 2):
            h = wp.sqrt((m[2][2] - (m[0][0] + m[1][1])) + 1.0)
            z = 0.5 * h
            h = 0.5 / h

            x = (m[2][0] + m[0][2]) * h
            y = (m[1][2] + m[2][1]) * h
            w = (m[1][0] - m[0][1]) * h

    q = wp.normalize(wp.quat(x, y, z, w))
    return q

@wp.kernel
def quat_from_matrix_forward(
    m: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
    idx: int):

    tid = wp.tid()

    matrix = wp.mat33(
        m[tid,0], m[tid,1], m[tid,2],
        m[tid,3], m[tid,4], m[tid,5],
        m[tid,6], m[tid,7], m[tid,8])

    q = quat_from_matrix_func(matrix)

    wp.atomic_add(loss, 0, q[idx])

def test_quat_from_matrix(test, device):
    m = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.5, 0.866, 0.0, -0.866, 0.5],
        [0.866, 0.0, 0.25, -.433, 0.5, 0.75, -0.25, -0.866, 0.433],
        [0.866, -0.433, 0.25, 0.0, 0.5, 0.866, -0.5, -0.75, 0.433],
        [-1.2, -1.6, -2.3, 0.25, -0.6, -0.33, 3.2, -1.0, -2.2]])
    m = wp.array2d(m, dtype=float, device=device, requires_grad=True)

    N = m.shape[0]

    def compute_gradients(kernel, wrt, index):
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        tape = wp.Tape()
        
        with tape:
            wp.launch(
                kernel=kernel,
                dim=N,
                inputs=[m, loss, index],
                device=device)

            tape.backward(loss)

        gradients = tape.gradients[wrt].numpy()
        tape.zero()

        return gradients

    # gather gradients from builtin adjoints
    gradients_x = compute_gradients(quat_from_matrix, m, 0)
    gradients_y = compute_gradients(quat_from_matrix, m, 1)
    gradients_z = compute_gradients(quat_from_matrix, m, 2)
    gradients_w = compute_gradients(quat_from_matrix, m, 3)

    # gather gradients from autodiff
    gradients_x_auto = compute_gradients(quat_from_matrix_forward, m, 0)
    gradients_y_auto = compute_gradients(quat_from_matrix_forward, m, 1)
    gradients_z_auto = compute_gradients(quat_from_matrix_forward, m, 2)
    gradients_w_auto = compute_gradients(quat_from_matrix_forward, m, 3)

    # compare
    eps = 1.0e6

    test.assertTrue((np.abs(gradients_x - gradients_x_auto) < eps).all())
    test.assertTrue((np.abs(gradients_y - gradients_y_auto) < eps).all())
    test.assertTrue((np.abs(gradients_z - gradients_z_auto) < eps).all())
    test.assertTrue((np.abs(gradients_w - gradients_w_auto) < eps).all())

def register(parent):

    devices = wp.get_devices()

    class TestQuat(parent):
        pass

    add_function_test(TestQuat, "test_rotate_rodriguez_grad", test_rotate_rodriguez_grad, devices=devices)
    add_function_test(TestQuat, "test_quat_to_axis_angle_grad", test_quat_to_axis_angle_grad, devices=devices)
    add_function_test(TestQuat, "test_slerp_grad", test_slerp_grad, devices=devices)
    add_function_test(TestQuat, "test_quat_rpy_grad", test_quat_rpy_grad, devices=devices)
    add_function_test(TestQuat, "test_quat_from_matrix", test_quat_from_matrix, devices=devices)

    return TestQuat

if __name__ == '__main__':
    wp.build.clear_kernel_cache()
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
