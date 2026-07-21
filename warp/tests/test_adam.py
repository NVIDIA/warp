# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp.optim
from warp.tests.unittest_utils import *


@wp.kernel
def objective(params: wp.array[float], score: wp.array[float]):
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
def objective_vec3(params: wp.array[wp.vec3], score: wp.array[float]):
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
def objective_two_inputs_vec3(params1: wp.array[wp.vec3], params2: wp.array[wp.vec3], score: wp.array[float]):
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


def test_adam_set_params_preserves_fp16_state(test, device):
    """Verify repeated ``set_params()`` calls with unchanged params reuse the existing moment buffers.

    The buffers must not be re-allocated (and zeroed). Moments are always fp32, so for fp16 params
    the realloc guard must compare against the moment dtype, not the param dtype, otherwise fp16
    optimizer state is silently reset on every call.
    """
    with wp.ScopedDevice(device):
        for param_dtype in (wp.float32, wp.float16, wp.vec3):
            params = wp.zeros(4, dtype=param_dtype, requires_grad=True)
            opt = warp.optim.Adam([params], lr=0.02)

            m_buffer, v_buffer = opt.m[0], opt.v[0]
            # Dirty the moment state so a spurious realloc would be observable.
            m_buffer.fill_(1.0)
            v_buffer.fill_(1.0)

            for _ in range(2):
                opt.set_params([params])  # same params -> must be a no-op

                test.assertIs(opt.m[0], m_buffer, f"first moment re-allocated for {param_dtype}")
                test.assertIs(opt.v[0], v_buffer, f"second moment re-allocated for {param_dtype}")
                test.assertTrue((opt.m[0].numpy() == 1.0).all(), f"first moment reset for {param_dtype}")
                test.assertTrue((opt.v[0].numpy() == 1.0).all(), f"second moment reset for {param_dtype}")


def test_adam_set_params_migrates_state(test, device):
    """Verify compatible moment buffers follow parameters that move devices."""
    moved_param = wp.zeros(4, dtype=wp.float32, device="cpu")
    unmoved_param = wp.zeros(4, dtype=wp.float32, device="cpu")
    opt = warp.optim.Adam([moved_param, unmoved_param], lr=0.02)

    moved_m, moved_v = opt.m[0], opt.v[0]
    unmoved_m, unmoved_v = opt.m[1], opt.v[1]
    moved_m.fill_(1.0)
    moved_v.fill_(2.0)

    expected_m = moved_m.numpy().copy()
    expected_v = moved_v.numpy().copy()
    replacement = wp.zeros(4, dtype=wp.float32, device=device)
    opt.set_params([replacement, unmoved_param])

    test.assertEqual(opt.m[0].device, device)
    test.assertEqual(opt.v[0].device, device)
    np.testing.assert_array_equal(opt.m[0].numpy(), expected_m)
    np.testing.assert_array_equal(opt.v[0].numpy(), expected_v)
    test.assertIs(opt.m[1], unmoved_m)
    test.assertIs(opt.v[1], unmoved_v)

    opt.step([wp.ones_like(replacement), wp.ones_like(unmoved_param)])
    replacement.numpy()
    unmoved_param.numpy()

    expected_m = opt.m[0].numpy().copy()
    expected_v = opt.v[0].numpy().copy()
    replacement = wp.zeros(4, dtype=wp.float32, device="cpu")
    opt.set_params([replacement, unmoved_param])

    test.assertEqual(opt.m[0].device, wp.get_device("cpu"))
    test.assertEqual(opt.v[0].device, wp.get_device("cpu"))
    np.testing.assert_array_equal(opt.m[0].numpy(), expected_m)
    np.testing.assert_array_equal(opt.v[0].numpy(), expected_v)
    test.assertIs(opt.m[1], unmoved_m)
    test.assertIs(opt.v[1], unmoved_v)


devices = get_test_devices()


class TestAdam(unittest.TestCase):
    pass


add_function_test(TestAdam, "test_adam_solve_float", test_adam_solve_float, devices=devices)
add_function_test(TestAdam, "test_adam_solve_vec3", test_adam_solve_vec3, devices=devices)
add_function_test(TestAdam, "test_adam_solve_two_inputs", test_adam_solve_two_inputs, devices=devices)
add_function_test(
    TestAdam, "test_adam_set_params_preserves_fp16_state", test_adam_set_params_preserves_fp16_state, devices=devices
)
add_function_test(
    TestAdam,
    "test_adam_set_params_migrates_state",
    test_adam_set_params_migrates_state,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
