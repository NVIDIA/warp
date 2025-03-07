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

import contextlib
import io
import unittest

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def scale(
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
):
    y[0] = x[0] ** 2.0


@wp.kernel(enable_backward=True)
def scale_1(
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
):
    y[0] = x[0] ** 2.0


@wp.kernel(enable_backward=False)
def scale_2(
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
):
    y[0] = x[0] ** 2.0


def test_options_1(test, device):
    x = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    wp.set_module_options({"enable_backward": False})

    tape = wp.Tape()
    with tape:
        wp.launch(scale, dim=1, inputs=[x, y], device=device)

    with contextlib.redirect_stdout(io.StringIO()) as f:
        tape.backward(y)

    expected = f"Warp UserWarning: Running the tape backwards may produce incorrect gradients because recorded kernel {scale.key} is defined in a module with the option 'enable_backward=False' set.\n"

    assert f.getvalue() == expected
    assert_np_equal(tape.gradients[x].numpy(), np.array(0.0))


def test_options_2(test, device):
    x = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    wp.set_module_options({"enable_backward": True})

    tape = wp.Tape()
    with tape:
        wp.launch(scale, dim=1, inputs=[x, y], device=device)

    tape.backward(y)
    assert_np_equal(tape.gradients[x].numpy(), np.array(6.0))


def test_options_3(test, device):
    x = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    wp.set_module_options({"enable_backward": False})

    tape = wp.Tape()
    with tape:
        wp.launch(scale_1, dim=1, inputs=[x, y], device=device)

    tape.backward(y)
    assert_np_equal(tape.gradients[x].numpy(), np.array(6.0))


def test_options_4(test, device):
    x = wp.array([3.0], dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)

    wp.set_module_options({"enable_backward": True})

    tape = wp.Tape()
    with tape:
        wp.launch(scale_2, dim=1, inputs=[x, y], device=device)

    with contextlib.redirect_stdout(io.StringIO()) as f:
        tape.backward(y)

    expected = f"Warp UserWarning: Running the tape backwards may produce incorrect gradients because recorded kernel {scale_2.key} is configured with the option 'enable_backward=False'.\n"

    assert f.getvalue() == expected
    assert_np_equal(tape.gradients[x].numpy(), np.array(0.0))


devices = get_test_devices()


class TestOptions(unittest.TestCase):
    pass


add_function_test(TestOptions, "test_options_1", test_options_1, devices=devices)
add_function_test(TestOptions, "test_options_2", test_options_2, devices=devices)
add_function_test(TestOptions, "test_options_3", test_options_3, devices=devices)
add_function_test(TestOptions, "test_options_4", test_options_4, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
