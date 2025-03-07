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

import unittest

import numpy as np

import warp as wp
import warp.optim
import warp.sim
from warp.tests.unittest_utils import *


@wp.kernel
def objective(params: wp.array(dtype=float), score: wp.array(dtype=float)):
    tid = wp.tid()
    U = params[tid] * params[tid]
    wp.atomic_add(score, 0, U)


# This test inspired by https://machinelearningmastery.com/adam-optimization-from-scratch/
def test_adam_solve_float(test, device):
    with wp.ScopedDevice(device):
        params_start = np.array([0.1, 0.2], dtype=float)
        score = wp.zeros(1, dtype=float, requires_grad=True)
        params = wp.array(params_start, dtype=float, requires_grad=True)
        tape = wp.Tape()
        opt = warp.optim.Adam([params], lr=0.02, betas=(0.8, 0.999))

        def gradient_func():
            tape.reset()
            score.zero_()
            with tape:
                wp.launch(kernel=objective, dim=len(params), inputs=[params, score])
            tape.backward(score)
            return [tape.gradients[params]]

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
def objective_vec3(params: wp.array(dtype=wp.vec3), score: wp.array(dtype=float)):
    tid = wp.tid()
    U = wp.dot(params[tid], params[tid])
    wp.atomic_add(score, 0, U)


# This test inspired by https://machinelearningmastery.com/adam-optimization-from-scratch/
def test_adam_solve_vec3(test, device):
    with wp.ScopedDevice(device):
        params_start = np.array([[0.1, 0.2, -0.1]], dtype=float)
        score = wp.zeros(1, dtype=float, requires_grad=True)
        params = wp.array(params_start, dtype=wp.vec3, requires_grad=True)
        tape = wp.Tape()
        opt = warp.optim.Adam([params], lr=0.02, betas=(0.8, 0.999))

        def gradient_func():
            tape.reset()
            score.zero_()
            with tape:
                wp.launch(kernel=objective_vec3, dim=len(params), inputs=[params, score])
            tape.backward(score)
            return [tape.gradients[params]]

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


@wp.kernel
def objective_two_inputs_vec3(
    params1: wp.array(dtype=wp.vec3), params2: wp.array(dtype=wp.vec3), score: wp.array(dtype=float)
):
    tid = wp.tid()
    U = wp.dot(params1[tid], params1[tid])
    V = wp.dot(params2[tid], params2[tid])
    wp.atomic_add(score, 0, U + V)


# This test inspired by https://machinelearningmastery.com/adam-optimization-from-scratch/
def test_adam_solve_two_inputs(test, device):
    with wp.ScopedDevice(device):
        params_start1 = np.array([[0.1, 0.2, -0.1]], dtype=float)
        params_start2 = np.array([[0.2, 0.1, 0.1]], dtype=float)
        score = wp.zeros(1, dtype=float, requires_grad=True)
        params1 = wp.array(params_start1, dtype=wp.vec3, requires_grad=True)
        params2 = wp.array(params_start2, dtype=wp.vec3, requires_grad=True)
        tape = wp.Tape()
        opt = warp.optim.Adam([params1, params2], lr=0.02, betas=(0.8, 0.999))

        def gradient_func():
            tape.reset()
            score.zero_()
            with tape:
                wp.launch(kernel=objective_two_inputs_vec3, dim=len(params1), inputs=[params1, params2, score])
            tape.backward(score)
            return [tape.gradients[params1], tape.gradients[params2]]

        niters = 100
        opt.reset_internal_state()
        for _ in range(niters):
            opt.step(gradient_func())

        result = params1.numpy()
        tol = 1e-5
        # optimum is at the origin, so the result should be close to it in all N dimensions.
        for r in result:
            for v in r:
                test.assertLessEqual(v, tol)

        result = params2.numpy()
        tol = 1e-5
        # optimum is at the origin, so the result should be close to it in all N dimensions.
        for r in result:
            for v in r:
                test.assertLessEqual(v, tol)


devices = get_test_devices()


class TestAdam(unittest.TestCase):
    pass


add_function_test(TestAdam, "test_adam_solve_float", test_adam_solve_float, devices=devices)
add_function_test(TestAdam, "test_adam_solve_vec3", test_adam_solve_vec3, devices=devices)
add_function_test(TestAdam, "test_adam_solve_two_inputs", test_adam_solve_two_inputs, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
