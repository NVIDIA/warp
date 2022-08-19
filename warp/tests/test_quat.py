import numpy as np
import math

import warp as wp
from warp.tests.test_base import *

wp.init()

@wp.kernel
def rodriguez_kernel(rotators: wp.array(dtype=wp.vec3), x: wp.vec3, loss: wp.array(dtype=float), coord_idx: int):
    tid = wp.tid()
    r = rotators[tid]
    r_new = wp.rotate_rodriguez(r, x)
    wp.atomic_add(loss, 0, r_new[coord_idx])

@wp.kernel
def rodriguez_kernel_autodiff(rotators: wp.array(dtype=wp.vec3), x: wp.vec3, loss: wp.array(dtype=float), coord_idx: int):
    tid = wp.tid()
    r = rotators[tid]
    angle = wp.length(r)
    if angle > 0.0 or angle < 0.0:
        axis = r / angle
        r_new = x * wp.cos(angle) + wp.cross(axis, x) * wp.sin(angle) + axis * wp.dot(axis, x) * (1.0 - wp.cos(angle))
    else:
        r_new = x
    wp.atomic_add(loss, 0, r_new[coord_idx])

# todo: add grad wrt to input vector 
def test_rotate_rodriguez_grad(test, device):
    num_rotators = 5
    x = wp.vec3(0.5, 0.5, 0.0)
    rotators = np.random.randn(num_rotators, 3)
    rotators = wp.array(rotators, dtype=wp.vec3, device=device, requires_grad=True)

    # gather gradients from builtin adjoints
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=rodriguez_kernel,
            dim=num_rotators,
            inputs=[rotators, x, loss, 0],
            device=device)

        tape.backward(loss)

    gradients_x = tape.gradients[rotators].numpy()
    tape.zero()

    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=rodriguez_kernel,
            dim=num_rotators,
            inputs=[rotators, x, loss, 1],
            device=device)

        tape.backward(loss)

    gradients_y = tape.gradients[rotators].numpy()
    tape.zero()

    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=rodriguez_kernel,
            dim=num_rotators,
            inputs=[rotators, x, loss, 2],
            device=device)

        tape.backward(loss)

    gradients_z = tape.gradients[rotators].numpy()
    tape.zero()

    # gather gradients from autodiff
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=rodriguez_kernel_autodiff,
            dim=num_rotators,
            inputs=[rotators, x, loss, 0],
            device=device)

        tape.backward(loss)

    gradients_x_auto = tape.gradients[rotators].numpy()
    tape.zero()

    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=rodriguez_kernel_autodiff,
            dim=num_rotators,
            inputs=[rotators, x, loss, 1],
            device=device)

        tape.backward(loss)

    gradients_y_auto = tape.gradients[rotators].numpy()
    tape.zero()

    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=rodriguez_kernel_autodiff,
            dim=num_rotators,
            inputs=[rotators, x, loss, 2],
            device=device)

        tape.backward(loss)

    gradients_z_auto = tape.gradients[rotators].numpy()
    tape.zero()

    # compare autodiff and adj_rotate_rodriguez
    print(gradients_x_auto)
    print(gradients_y_auto)
    print(gradients_z_auto)

def test_slerp_grad(test, device):
    pass

def test_quat_smoothstep_grad(test, device):
    pass

def test_quat_to_axis_angle_grad(test, device):
    pass

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

    add_function_test(TestQuat, "test_mesh_query_ray_edge", test_rotate_rodriguez_grad, devices=devices)
    add_function_test(TestQuat, "test_slerp_grad", test_slerp_grad, devices=devices)
    add_function_test(TestQuat, "test_quat_smoothstep_grad", test_quat_smoothstep_grad, devices=devices)
    add_function_test(TestQuat, "test_quat_to_axis_angle_grad", test_quat_to_axis_angle_grad, devices=devices)
    add_function_test(TestQuat, "test_quat_from_matrix_grad", test_quat_from_matrix_grad, devices=devices)
    add_function_test(TestQuat, "test_quat_rpy_grad", test_quat_rpy_grad, devices=devices)
    add_function_test(TestQuat, "test_normalize_grad", test_normalize_grad, devices=devices)

    return TestQuat

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
