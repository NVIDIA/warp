# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from warp.tests.unittest_utils import *

# Test for issue #1174: Gradients not propagating through array of structs


# Scalar struct
@wp.struct
class ScalarStruct:
    a: wp.float32


# Vec3 struct
@wp.struct
class Vec3Struct:
    v: wp.vec3


# Mat22 struct
@wp.struct
class Mat22Struct:
    m: wp.mat22


# ===== Scalar Tests =====
@wp.kernel
def pack_scalar_struct_kernel(x: wp.array(dtype=wp.float32), y: wp.array(dtype=ScalarStruct)):
    i = wp.tid()
    y[i].a = x[i]


@wp.kernel
def loss_from_scalar_struct_kernel(y: wp.array(dtype=ScalarStruct), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(loss, 0, y[i].a)


def test_struct_array_gradient_propagation_scalar(_test, device):
    """Test scalar field gradient propagation with multiple threads (issue #1174)"""
    with wp.ScopedDevice(device):
        n = 10
        x = wp.ones(n, dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=ScalarStruct, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(kernel=pack_scalar_struct_kernel, dim=n, inputs=[x], outputs=[y])
            wp.launch(kernel=loss_from_scalar_struct_kernel, dim=n, inputs=[y], outputs=[loss])

        tape.backward(loss=loss)

        # Check that gradients propagate correctly
        # Each element contributes 1.0 to the loss via atomic_add
        for i in range(n):
            assert_np_equal(y.grad.numpy()[i][0], 1.0, tol=1e-5)
            assert_np_equal(x.grad.numpy()[i], 1.0, tol=1e-5)


# ===== Vec3 Tests =====
@wp.kernel
def pack_vec3_struct_kernel(x: wp.array(dtype=wp.vec3), y: wp.array(dtype=Vec3Struct)):
    i = wp.tid()
    y[i].v = x[i]


@wp.kernel
def loss_from_vec3_struct_kernel(y: wp.array(dtype=Vec3Struct), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(loss, 0, wp.length(y[i].v))


def test_struct_array_gradient_propagation_vec3(_test, device):
    """Test vec3 field gradient propagation with multiple threads (issue #1174)"""
    with wp.ScopedDevice(device):
        n = 5
        x = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, device=device, requires_grad=True)
        y = wp.zeros(n, dtype=Vec3Struct, requires_grad=True, device=device)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(kernel=pack_vec3_struct_kernel, dim=n, inputs=[x], outputs=[y], device=device)
            wp.launch(kernel=loss_from_vec3_struct_kernel, dim=n, inputs=[y], outputs=[loss], device=device)

        tape.backward(loss=loss)

        # Gradient should flow back through struct to x
        assert x.grad is not None
        # Non-zero gradients confirm the chain is connected
        assert np.any(np.abs(x.grad.numpy()) > 1e-6)


# ===== Mat22 Tests =====
@wp.kernel
def pack_mat22_struct_kernel(x: wp.array(dtype=wp.mat22), y: wp.array(dtype=Mat22Struct)):
    i = wp.tid()
    y[i].m = x[i]


@wp.kernel
def loss_from_mat22_struct_kernel(y: wp.array(dtype=Mat22Struct), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(loss, 0, wp.determinant(y[i].m))


def test_struct_array_gradient_propagation_mat22(_test, device):
    """Test mat22 field gradient propagation with multiple threads (issue #1174)"""
    with wp.ScopedDevice(device):
        n = 3
        # Create identity matrices
        identity = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        matrices = np.tile(identity, (n, 1, 1))

        x = wp.array(matrices, dtype=wp.mat22, device=device, requires_grad=True)
        y = wp.zeros(n, dtype=Mat22Struct, requires_grad=True, device=device)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(kernel=pack_mat22_struct_kernel, dim=n, inputs=[x], outputs=[y], device=device)
            wp.launch(kernel=loss_from_mat22_struct_kernel, dim=n, inputs=[y], outputs=[loss], device=device)

        tape.backward(loss=loss)

        # Gradient should flow back through struct to x
        assert x.grad is not None
        # Non-zero gradients confirm the chain is connected
        assert np.any(np.abs(x.grad.numpy()) > 1e-6)


devices = get_test_devices()


class TestGradStructArray(unittest.TestCase):
    pass


add_function_test(
    TestGradStructArray,
    "test_struct_array_gradient_propagation_scalar",
    test_struct_array_gradient_propagation_scalar,
    devices=devices,
)
add_function_test(
    TestGradStructArray,
    "test_struct_array_gradient_propagation_vec3",
    test_struct_array_gradient_propagation_vec3,
    devices=devices,
)
add_function_test(
    TestGradStructArray,
    "test_struct_array_gradient_propagation_mat22",
    test_struct_array_gradient_propagation_mat22,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
