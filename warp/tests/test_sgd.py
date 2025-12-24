# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import warp as wp
import warp.optim
from warp.tests.unittest_utils import *


def test_sgd_momentum_accumulation(test, device):
    """Test that momentum accumulates correctly over multiple steps."""
    with wp.ScopedDevice(device):
        # Start with params = [1.0, 2.0], constant gradient = [0.1, 0.2]
        params = wp.array([1.0, 2.0], dtype=float, requires_grad=False)
        grad = wp.array([0.1, 0.2], dtype=float)

        lr = 0.1
        momentum = 0.9
        opt = warp.optim.SGD([params], lr=lr, momentum=momentum, dampening=0.0)

        # Step 1: momentum buffer should be initialized to gradient
        opt.step([grad])
        result1 = params.numpy()

        # Expected:
        # b[0] = g[0] = 0.1 (first step, b initialized to gradient)
        # params[0] = 1.0 - 0.1 * 0.1 = 0.99
        # b[1] = g[1] = 0.2
        # params[1] = 2.0 - 0.1 * 0.2 = 1.98
        test.assertAlmostEqual(result1[0], 0.99, places=5)
        test.assertAlmostEqual(result1[1], 1.98, places=5)

        # Step 2: momentum buffer should accumulate
        opt.step([grad])
        result2 = params.numpy()

        # Expected:
        # b[0] = 0.9 * 0.1 + (1 - 0.0) * 0.1 = 0.09 + 0.1 = 0.19
        # params[0] = 0.99 - 0.1 * 0.19 = 0.971
        # b[1] = 0.9 * 0.2 + (1 - 0.0) * 0.2 = 0.18 + 0.2 = 0.38
        # params[1] = 1.98 - 0.1 * 0.38 = 1.942
        test.assertAlmostEqual(result2[0], 0.971, places=5)
        test.assertAlmostEqual(result2[1], 1.942, places=5)


def test_sgd_weight_decay(test, device):
    """Test that weight decay applies L2 regularization correctly."""
    with wp.ScopedDevice(device):
        # Start with params = [1.0, 2.0], gradient = [0.0, 0.0]
        params = wp.array([1.0, 2.0], dtype=float, requires_grad=False)
        grad = wp.array([0.0, 0.0], dtype=float)

        lr = 0.1
        weight_decay = 0.1
        opt = warp.optim.SGD([params], lr=lr, momentum=0.0, weight_decay=weight_decay)

        # With zero gradient, only weight decay should apply
        opt.step([grad])
        result = params.numpy()

        # Expected:
        # gt[0] = g[0] + weight_decay * params[0] = 0.0 + 0.1 * 1.0 = 0.1
        # params[0] = 1.0 - 0.1 * 0.1 = 0.99
        # gt[1] = g[1] + weight_decay * params[1] = 0.0 + 0.1 * 2.0 = 0.2
        # params[1] = 2.0 - 0.1 * 0.2 = 1.98
        test.assertAlmostEqual(result[0], 0.99, places=5)
        test.assertAlmostEqual(result[1], 1.98, places=5)


def test_sgd_dampening(test, device):
    """Test that dampening correctly reduces momentum accumulation."""
    with wp.ScopedDevice(device):
        params = wp.array([1.0, 1.0], dtype=float, requires_grad=False)
        grad = wp.array([1.0, 1.0], dtype=float)

        lr = 0.1
        momentum = 0.9
        dampening = 0.5
        opt = warp.optim.SGD([params], lr=lr, momentum=momentum, dampening=dampening)

        # First step: buffer initialized to gradient (dampening not applied)
        opt.step([grad])
        result1 = params.numpy()

        # Expected: b = g = 1.0, params = 1.0 - 0.1 * 1.0 = 0.9
        test.assertAlmostEqual(result1[0], 0.9, places=5)

        # Second step: dampening should reduce the gradient contribution
        opt.step([grad])
        result2 = params.numpy()

        # Expected:
        # b = 0.9 * 1.0 + (1 - 0.5) * 1.0 = 0.9 + 0.5 = 1.4
        # params = 0.9 - 0.1 * 1.4 = 0.76
        test.assertAlmostEqual(result2[0], 0.76, places=5)


def test_sgd_nesterov_momentum(test, device):
    """Test that Nesterov momentum uses look-ahead gradient."""
    with wp.ScopedDevice(device):
        params = wp.array([1.0], dtype=float, requires_grad=False)
        grad = wp.array([1.0], dtype=float)

        lr = 0.1
        momentum = 0.9
        opt = warp.optim.SGD([params], lr=lr, momentum=momentum, nesterov=True)

        # First step
        opt.step([grad])
        result1 = params.numpy()

        # Expected with Nesterov:
        # b = g = 1.0 (first step)
        # gt = g + momentum * b = 1.0 + 0.9 * 1.0 = 1.9 (Nesterov look-ahead)
        # params = 1.0 - 0.1 * 1.9 = 0.81
        test.assertAlmostEqual(result1[0], 0.81, places=5)

        # Second step
        opt.step([grad])
        result2 = params.numpy()

        # Expected:
        # b = 0.9 * 1.0 + (1 - 0.0) * 1.0 = 1.9
        # gt = 1.0 + 0.9 * 1.9 = 2.71
        # params = 0.81 - 0.1 * 2.71 = 0.539
        test.assertAlmostEqual(result2[0], 0.539, places=5)


def test_sgd_combined_features(test, device):
    """Test SGD with momentum, weight decay, and dampening combined."""
    with wp.ScopedDevice(device):
        params = wp.array([10.0], dtype=float, requires_grad=False)
        grad = wp.array([1.0], dtype=float)

        lr = 0.01
        momentum = 0.9
        dampening = 0.1
        weight_decay = 0.01
        opt = warp.optim.SGD([params], lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)

        # First step
        opt.step([grad])
        result1 = params.numpy()

        # Expected:
        # gt = g + weight_decay * params = 1.0 + 0.01 * 10.0 = 1.1
        # b = gt = 1.1 (first step, no dampening)
        # params = 10.0 - 0.01 * 1.1 = 9.989
        test.assertAlmostEqual(result1[0], 9.989, places=5)

        # Second step
        opt.step([grad])
        result2 = params.numpy()

        # Expected:
        # gt = 1.0 + 0.01 * 9.989 = 1.09989
        # b = 0.9 * 1.1 + (1 - 0.1) * 1.09989 = 0.99 + 0.989901 = 1.979901
        # params = 9.989 - 0.01 * 1.979901 ≈ 9.969201
        test.assertAlmostEqual(result2[0], 9.969201, places=5)


def test_sgd_no_momentum(test, device):
    """Test vanilla SGD without momentum (should be simple gradient descent)."""
    with wp.ScopedDevice(device):
        params = wp.array([5.0, 10.0], dtype=float, requires_grad=False)
        grad = wp.array([2.0, 4.0], dtype=float)

        lr = 0.1
        opt = warp.optim.SGD([params], lr=lr, momentum=0.0)

        opt.step([grad])
        result = params.numpy()

        # Expected: simple gradient descent
        # params[0] = 5.0 - 0.1 * 2.0 = 4.8
        # params[1] = 10.0 - 0.1 * 4.0 = 9.6
        test.assertAlmostEqual(result[0], 4.8, places=5)
        test.assertAlmostEqual(result[1], 9.6, places=5)


def test_sgd_reset_internal_state(test, device):
    """Test that reset_internal_state properly clears momentum buffers."""
    with wp.ScopedDevice(device):
        params = wp.array([1.0], dtype=float, requires_grad=False)
        grad = wp.array([1.0], dtype=float)

        lr = 0.1
        momentum = 0.9
        opt = warp.optim.SGD([params], lr=lr, momentum=momentum)

        # Take a step to build up momentum
        opt.step([grad])
        result1 = params.numpy()
        test.assertAlmostEqual(result1[0], 0.9, places=5)

        # Reset and take another step - should behave like first step
        opt.reset_internal_state()
        params_reset = wp.array([1.0], dtype=float, requires_grad=False)
        opt.set_params([params_reset])
        opt.step([grad])
        result2 = params_reset.numpy()

        # Should match the first step again
        test.assertAlmostEqual(result2[0], 0.9, places=5)


devices = get_test_devices()


class TestSGD(unittest.TestCase):
    pass


add_function_test(TestSGD, "test_sgd_momentum_accumulation", test_sgd_momentum_accumulation, devices=devices)
add_function_test(TestSGD, "test_sgd_weight_decay", test_sgd_weight_decay, devices=devices)
add_function_test(TestSGD, "test_sgd_dampening", test_sgd_dampening, devices=devices)
add_function_test(TestSGD, "test_sgd_nesterov_momentum", test_sgd_nesterov_momentum, devices=devices)
add_function_test(TestSGD, "test_sgd_combined_features", test_sgd_combined_features, devices=devices)
add_function_test(TestSGD, "test_sgd_no_momentum", test_sgd_no_momentum, devices=devices)
add_function_test(TestSGD, "test_sgd_reset_internal_state", test_sgd_reset_internal_state, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
