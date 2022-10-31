
# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np
import math

import warp as wp
from warp.tests.test_base import *
import unittest

import warp.optim
import warp.sim

wp.init()

@wp.kernel
def objective(params:wp.array(dtype=float), score:wp.array(dtype=float)):
    tid = wp.tid()
    U = params[tid] * params[tid]
    wp.atomic_add(score, 0, U)

# This test inspired by https://machinelearningmastery.com/adam-optimization-from-scratch/
def test_adam_solve_float(test, device):
    wp.set_device(device)
    params_start = np.array([0.1, 0.2], dtype=float)
    score = wp.zeros(1, dtype=float, requires_grad=True)
    params = wp.array(params_start, dtype=float, requires_grad=True)
    tape = wp.Tape()
    opt = warp.optim.Adam(params, lr=0.02, betas=(0.8, 0.999))

    def gradient_func():
        tape.reset()
        score.zero_()
        with tape:
            wp.launch(kernel=objective, dim=len(params), inputs=[params, score])
        tape.backward(score)
        return tape.gradients[params]

    niters = 100

    opt.reset_internal_state()
    for _ in range(niters):
        opt.step(gradient_func())

    result = params.numpy()
    # optimum is at the origin, so the result should be close to it in all N dimensions.
    tol = 1e-5
    for r in result:
        test.assertLessEqual(r, tol)  

@wp.kernel
def objective_vec3(params:wp.array(dtype=wp.vec3), score:wp.array(dtype=float)):
    tid = wp.tid()
    U = wp.dot(params[tid], params[tid])
    wp.atomic_add(score, 0, U)

# This test inspired by https://machinelearningmastery.com/adam-optimization-from-scratch/
def test_adam_solve_vec3(test, device):
    wp.set_device(device)
    params_start = np.array([[0.1, 0.2,-0.1]], dtype=float)
    score = wp.zeros(1, dtype=float, requires_grad=True)
    params = wp.array(params_start, dtype=wp.vec3, requires_grad=True)
    tape = wp.Tape()
    opt = warp.optim.Adam(params, lr=0.02, betas=(0.8, 0.999))

    def gradient_func():
        tape.reset()
        score.zero_()
        with tape:
            wp.launch(kernel=objective_vec3, dim=len(params), inputs=[params, score])
        tape.backward(score)
        return tape.gradients[params]

    niters = 100
    opt.reset_internal_state()
    for _ in range(niters):
        opt.step(gradient_func())

    result = params.numpy()
    tol = 1e-5
    # optimum is at the origin, so the result should be close to it in all N dimensions.
    for r in result:
        for v in r:
            test.assertLessEqual(v, tol)    

def register(parent):

    devices = wp.get_devices()

    class TestArray(parent):
        pass

    add_function_test(TestArray, "test_adam_solve_float", test_adam_solve_float, devices=devices)
    add_function_test(TestArray, "test_adam_solve_vec3", test_adam_solve_vec3, devices=devices)

    return TestArray

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)