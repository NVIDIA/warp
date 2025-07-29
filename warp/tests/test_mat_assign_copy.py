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
from warp.tests.unittest_utils import *


def setUpModule():
    wp.config.enable_vector_component_overwrites = True


def tearDownModule():
    wp.config.enable_vector_component_overwrites = False


@wp.kernel
def mat_assign_element(x: wp.array(dtype=float), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()

    a = wp.mat22()
    a[0, 0] = 1.0 * x[i]
    a[0, 1] = 2.0 * x[i]
    a[1, 0] = 3.0 * x[i]
    a[1, 1] = 4.0 * x[i]

    y[i] = a


@wp.kernel
def mat_assign_row(x: wp.array(dtype=wp.vec2), y: wp.array(dtype=wp.mat22)):
    i = wp.tid()

    a = wp.mat22()
    a[0] = 1.0 * x[i]
    a[1] = 2.0 * x[i]

    y[i] = a


def test_mat_assign(test, device):
    # matrix element
    x = wp.ones(1, dtype=float, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_assign_element, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([10.0], dtype=float))

    # matrix row
    x = wp.ones(1, dtype=wp.vec2, requires_grad=True, device=device)
    y = wp.zeros(1, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_assign_row, 1, inputs=[x], outputs=[y], device=device)

    y.grad = wp.ones_like(y)
    tape.backward()

    assert_np_equal(y.numpy(), np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=float))
    assert_np_equal(x.grad.numpy(), np.array([[3.0, 3.0]], dtype=float))


def test_matrix_assign_copy(test, device):
    @wp.kernel(module="unique")
    def mat_in_register_overwrite(x: wp.array2d(dtype=wp.mat22), y: wp.array(dtype=wp.vec2)):
        i, j = wp.tid()

        a = wp.mat22()
        a[0] = y[i]
        a[0, 1] = 3.0
        x[i, j] = a

    x = wp.zeros((1, 1), dtype=wp.mat22, device=device, requires_grad=True)
    y = wp.ones(1, dtype=wp.vec2, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(mat_in_register_overwrite, dim=(1, 1), inputs=[x, y], device=device)

    tape.backward(grads={x: wp.ones_like(x, requires_grad=False)})

    assert_np_equal(x.numpy(), np.array([[[[1.0, 3.0], [0.0, 0.0]]]], dtype=float))
    assert_np_equal(y.grad.numpy(), np.array([[1.0, 0.0]], dtype=float))


def test_mat_slicing_assign_backward(test, device):
    mat23 = wp.mat((2, 3), float)

    @wp.kernel(module="unique")
    def kernel(
        arr_x: wp.array(dtype=wp.vec2),
        arr_y: wp.array(dtype=mat23),
        arr_z: wp.array(dtype=wp.mat44),
    ):
        i = wp.tid()

        z = arr_z[i]

        z[0, :2] = arr_x[i]
        z[:2, 1:] = arr_y[i]

        z[:2, 3] += arr_x[i][:2]
        z[1:-1, :2] += arr_y[i][::-1, :-1]

        z[2:, 3] -= arr_x[i][0:]
        z[3:, -1:] -= arr_y[i][:1, :1]

        arr_z[i] = z

    x = wp.ones(1, dtype=wp.vec2, requires_grad=True, device=device)
    y = wp.ones(1, dtype=mat23, requires_grad=True, device=device)
    z = wp.zeros(1, dtype=wp.mat44, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, 1, inputs=(x, y), outputs=(z,), device=device)

    z.grad = wp.ones_like(z)
    tape.backward()

    assert_np_equal(
        z.numpy(),
        np.array(
            (
                (
                    (1.0, 1.0, 1.0, 2.0),
                    (1.0, 2.0, 1.0, 2.0),
                    (1.0, 1.0, 0.0, -1.0),
                    (0.0, 0.0, 0.0, -2.0),
                ),
            ),
            dtype=float,
        ),
    )
    assert_np_equal(x.grad.numpy(), np.array(((1.0, 0.0),), dtype=float))
    assert_np_equal(y.grad.numpy(), np.array((((1.0, 2.0, 1.0), (2.0, 2.0, 1.0)),), dtype=float))


devices = get_test_devices()


class TestMatAssignCopy(unittest.TestCase):
    pass


add_function_test(TestMatAssignCopy, "test_mat_assign", test_mat_assign, devices=devices)
add_function_test(TestMatAssignCopy, "test_matrix_assign_copy", test_matrix_assign_copy, devices=devices)
add_function_test(
    TestMatAssignCopy, "test_mat_slicing_assign_backward", test_mat_slicing_assign_backward, devices=devices
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
