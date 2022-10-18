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
def slerp_trig_grad(
    q0: wp.array(dtype=wp.quat),
    q1: wp.array(dtype=wp.quat),
    t: wp.array(dtype=float),
    q: wp.array(dtype=wp.quat)):

    tid = wp.tid()

    theta = wp.acos(wp.dot(q0[tid], q1[tid]))

    if theta != 0.0:
        A = -theta * wp.cos(theta * (1.0 - t[tid])) / wp.sin(theta)
        B = theta * wp.cos(theta * t[tid]) / wp.sin(theta)
        q[tid] = wp.add(wp.mul(A, q0[tid]), wp.mul(B, q1[tid]))
    else:
        q[tid] = wp.quat(0.0, 0.0, 0.0, 0.0)

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
    np.random.seed(42)
    seed = 42
    N = 100

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
    gradients_w = compute_gradients(slerp_kernel, t, 3)

    # gather gradients from autodiff
    gradients_w_auto = compute_gradients(slerp_kernel_forward, t, 3)

    # trigonometric analytic gradient (for comparison)
    trig_grad = wp.zeros(N, dtype=wp.quat, device=device)
    wp.launch(
        kernel=slerp_trig_grad,
        dim=N,
        inputs=[q0, q1, t, trig_grad],
        device=device)
    # print(trig_grad.numpy()[0][3])

    test.assertTrue((np.abs(gradients_w - gradients_w_auto) < 1e4).all())

def test_slerp_smoothstep_grad(test, device):
    pass

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
    N = 5
    
    quats = wp.zeros(N, dtype=wp.quat, device=device, requires_grad=True)
    
    edge_cases = np.array([(1.0, 0.0, 0.0, 0.0), (0.0, 1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3))])
    edge_cases = wp.array(edge_cases, dtype=wp.quat, device=device, requires_grad=True)

    wp.launch(kernel=quat_sampler, dim=N, inputs=[seed, quats], device=device)

    def compute_gradients(arr, kernel, index):
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel=kernel,
                dim=N,
                inputs=[arr, loss, index],
                device=device)

            tape.backward(loss)

        gradients = tape.gradients[arr].numpy()
        tape.zero()

        return gradients

    # gather gradients from builtin adjoints
    gradients_x = compute_gradients(quats, quat_to_axis_angle_kernel, 0)
    gradients_y = compute_gradients(quats, quat_to_axis_angle_kernel, 1)
    gradients_z = compute_gradients(quats, quat_to_axis_angle_kernel, 2)
    gradients_w = compute_gradients(quats, quat_to_axis_angle_kernel, 3)

    # gather gradients from autodiff
    gradients_x_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, 0)
    gradients_y_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, 1)
    gradients_z_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, 2)
    gradients_w_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, 3)

    # edge cases: gather gradients from builtin adjoints
    edge_gradients_x = compute_gradients(edge_cases, quat_to_axis_angle_kernel, 0)
    edge_gradients_y = compute_gradients(edge_cases, quat_to_axis_angle_kernel, 1)
    edge_gradients_z = compute_gradients(edge_cases, quat_to_axis_angle_kernel, 2)
    edge_gradients_w = compute_gradients(edge_cases, quat_to_axis_angle_kernel, 3)

    # edge cases: gather gradients from autodiff
    edge_gradients_x_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, 0)
    edge_gradients_y_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, 1)
    edge_gradients_z_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, 2)
    edge_gradients_w_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, 3)

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


def test_quat_from_matrix_grad(test, device):
    pass

def test_quat_rpy_grad(test, device):
    pass

def test_normalize_grad(test, device):
    pass

def register(parent):

    devices = wp.get_devices()

    class TestQuat(parent):
        pass

    # add_function_test(TestQuat, "test_rotate_rodriguez_grad", test_rotate_rodriguez_grad, devices=devices)
    # add_function_test(TestQuat, "test_slerp_grad", test_slerp_grad, devices=devices)
    # add_function_test(TestQuat, "test_slerp_smoothstep_grad", test_slerp_smoothstep_grad, devices=devices)
    add_function_test(TestQuat, "test_quat_to_axis_angle_grad", test_quat_to_axis_angle_grad, devices=devices)
    # add_function_test(TestQuat, "test_quat_from_matrix_grad", test_quat_from_matrix_grad, devices=devices)
    # add_function_test(TestQuat, "test_quat_rpy_grad", test_quat_rpy_grad, devices=devices)
    # add_function_test(TestQuat, "test_normalize_grad", test_normalize_grad, devices=devices)

    return TestQuat

if __name__ == '__main__':
    wp.build.clear_kernel_cache()
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
