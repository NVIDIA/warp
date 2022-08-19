import numpy as np
import math

import warp as wp
from warp.tests.test_base import *

wp.init()

def test_rotate_rodriguez_grad(test, device):
    pass
    # todo: add grad wrt to input vector 

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
