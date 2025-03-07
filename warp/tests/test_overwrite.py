# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# kernels are defined in the global scope, to ensure wp.Kernel objects are not GC'ed in the MGPU case
# kernel args are assigned array modes during codegen, so wp.Kernel objects generated during codegen
# must be preserved for overwrite tracking to function


@wp.kernel
def square_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = x[tid] * x[tid]


@wp.kernel
def overwrite_kernel_a(z: wp.array(dtype=float), x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] = z[tid]


# (kernel READ) -> (kernel WRITE) failure case
def test_kernel_read_kernel_write(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        a = wp.array(np.array([1.0, 2.0, 3.0]), dtype=float, requires_grad=True, device=device)
        b = wp.zeros_like(a)
        c = wp.array(np.array([-1.0, -2.0, -3.0]), dtype=float, requires_grad=True, device=device)

        tape = wp.Tape()

        with contextlib.redirect_stdout(io.StringIO()) as f:
            with tape:
                wp.launch(square_kernel, a.shape, inputs=[a], outputs=[b], device=device)
                wp.launch(overwrite_kernel_a, c.shape, inputs=[c], outputs=[a], device=device)

        expected = "is being written to but has already been read from in a previous launch. This may corrupt gradient computation in the backward pass."
        test.assertIn(expected, f.getvalue())

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


@wp.kernel
def double_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = 2.0 * x[tid]


@wp.kernel
def triple_kernel(y: wp.array(dtype=float), z: wp.array(dtype=float)):
    tid = wp.tid()
    z[tid] = 3.0 * y[tid]


@wp.kernel
def overwrite_kernel_b(w: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = 1.0 * w[tid]


# (kernel WRITE) -> (kernel READ) -> (kernel WRITE) failure case
def test_kernel_write_kernel_read_kernel_write(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        tape = wp.Tape()

        a = wp.array(np.array([1.0, 2.0, 3.0]), dtype=float, requires_grad=True, device=device)
        b = wp.zeros_like(a)
        c = wp.zeros_like(a)
        d = wp.zeros_like(a)

        with contextlib.redirect_stdout(io.StringIO()) as f:
            with tape:
                wp.launch(double_kernel, a.shape, inputs=[a], outputs=[b], device=device)
                wp.launch(triple_kernel, b.shape, inputs=[b], outputs=[c], device=device)
                wp.launch(overwrite_kernel_b, d.shape, inputs=[d], outputs=[b], device=device)

        expected = "is being written to but has already been read from in a previous launch. This may corrupt gradient computation in the backward pass."
        test.assertIn(expected, f.getvalue())

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


@wp.kernel
def read_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid]


@wp.kernel
def writeread_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = c[tid] * c[tid]
    b[tid] = a[tid]


# (kernel READ) -> (kernel WRITE -> READ) failure case
def test_kernel_read_kernel_writeread(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        a = wp.array(np.arange(5), dtype=float, requires_grad=True, device=device)
        b = wp.zeros_like(a)
        c = wp.zeros_like(a)
        d = wp.zeros_like(a)

        tape = wp.Tape()

        with contextlib.redirect_stdout(io.StringIO()) as f:
            with tape:
                wp.launch(read_kernel, dim=5, inputs=[a, b], device=device)
                wp.launch(writeread_kernel, dim=5, inputs=[a, d, c], device=device)

        expected = "is being written to but has already been read from in a previous launch. This may corrupt gradient computation in the backward pass."
        test.assertIn(expected, f.getvalue())

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


@wp.kernel
def write_kernel(a: wp.array(dtype=float), d: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = d[tid]


# (kernel WRITE -> READ) -> (kernel WRITE) failure case
def test_kernel_writeread_kernel_write(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        c = wp.array(np.arange(5), dtype=float, requires_grad=True, device=device)
        b = wp.zeros_like(c)
        a = wp.zeros_like(c)
        d = wp.zeros_like(c)

        tape = wp.Tape()

        with contextlib.redirect_stdout(io.StringIO()) as f:
            with tape:
                wp.launch(writeread_kernel, dim=5, inputs=[a, b, c], device=device)
                wp.launch(write_kernel, dim=5, inputs=[a, d], device=device)

        expected = "is being written to but has already been read from in a previous launch. This may corrupt gradient computation in the backward pass."
        test.assertIn(expected, f.getvalue())

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


@wp.func
def read_func(a: wp.array(dtype=Any), idx: int):
    x = a[idx]
    return x


@wp.func
def read_return_func(b: wp.array(dtype=Any), idx: int):
    return 1.0, b[idx]


@wp.func
def write_func(c: wp.array(dtype=Any), idx: int):
    c[idx] = 1.0


@wp.func
def main_func(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float), idx: int):
    x = read_func(a, idx)
    y, z = read_return_func(b, idx)
    write_func(c, idx)
    return x + y + z


@wp.kernel
def func_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float), d: wp.array(dtype=float)):
    tid = wp.tid()
    d[tid] = main_func(a, b, c, tid)


# test various ways one might write to or read from an array inside warp functions
def test_nested_function_read_write(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        a = wp.zeros(5, dtype=float, requires_grad=True, device=device)
        b = wp.zeros_like(a)
        c = wp.zeros_like(a)
        d = wp.zeros_like(a)

        tape = wp.Tape()

        with tape:
            wp.launch(func_kernel, dim=5, inputs=[a, b, c, d], device=device)

        test.assertEqual(a._is_read, True)
        test.assertEqual(b._is_read, True)
        test.assertEqual(c._is_read, False)
        test.assertEqual(d._is_read, False)

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


@wp.kernel
def slice_kernel(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    x_slice = x[i, j]
    val = x_slice[k]

    y_slice = y[i, j]
    y_slice[k] = val


# test updating array r/w mode after indexing
def test_multidimensional_indexing(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        a = np.arange(3, dtype=float)
        b = np.tile(a, (3, 3, 1))
        x = wp.array3d(b, dtype=float, requires_grad=True, device=device)
        y = wp.zeros_like(x)

        tape = wp.Tape()

        with tape:
            wp.launch(slice_kernel, dim=(3, 3, 3), inputs=[x, y], device=device)

        test.assertEqual(x._is_read, True)
        test.assertEqual(y._is_read, False)

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


@wp.kernel
def inplace_a(x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] += 1.0


@wp.kernel
def inplace_b(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] += y[tid]


# in-place operators are treated as write
def test_in_place_operators(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        a = wp.zeros(3, dtype=float, requires_grad=True, device=device)
        b = wp.zeros_like(a)

        tape = wp.Tape()

        with tape:
            wp.launch(inplace_a, dim=3, inputs=[a], device=device)

        test.assertEqual(a._is_read, False)

        tape.reset()
        a.zero_()

        with tape:
            wp.launch(inplace_b, dim=3, inputs=[a, b], device=device)

        test.assertEqual(a._is_read, False)
        test.assertEqual(b._is_read, True)

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


def test_views(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        a = wp.zeros((3, 3), dtype=float, requires_grad=True, device=device)
        test.assertEqual(a._is_read, False)

        a.mark_write()

        b = a.view(dtype=int)
        test.assertEqual(b._is_read, False)

        c = b.flatten()
        test.assertEqual(c._is_read, False)

        c.mark_read()
        test.assertEqual(a._is_read, True)

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


def test_reset(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        a = wp.array(np.array([1.0, 2.0, 3.0]), dtype=float, requires_grad=True, device=device)
        b = wp.zeros_like(a)

        tape = wp.Tape()
        with tape:
            wp.launch(kernel=write_kernel, dim=3, inputs=[b, a], device=device)

        tape.backward(grads={b: wp.ones(3, dtype=float, device=device)})

        test.assertEqual(a._is_read, True)
        test.assertEqual(b._is_read, False)

        tape.reset()

        test.assertEqual(a._is_read, False)
        test.assertEqual(b._is_read, False)

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


# wp.copy uses wp.record_func. Ensure array modes are propagated correctly.
def test_copy(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        a = wp.array(np.array([1.0, 2.0, 3.0]), dtype=float, requires_grad=True, device=device)
        b = wp.zeros_like(a)

        tape = wp.Tape()

        with tape:
            wp.copy(b, a)

        test.assertEqual(a._is_read, True)
        test.assertEqual(b._is_read, False)

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


# write after read warning with in-place operators within a kernel
def test_in_place_operators_warning(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        with contextlib.redirect_stdout(io.StringIO()) as f:

            @wp.kernel
            def inplace_c(x: wp.array(dtype=float)):
                tid = wp.tid()
                x[tid] = 1.0
                a = x[tid]
                x[tid] += a

            a = wp.zeros(3, dtype=float, requires_grad=True, device=device)

            tape = wp.Tape()
            with tape:
                wp.launch(inplace_c, dim=3, inputs=[a], device=device)

        expected = "is being written to after it has been read from within the same kernel. This may corrupt gradient computation in the backward pass."
        test.assertIn(expected, f.getvalue())

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


# (kernel READ -> WRITE) failure case
def test_kernel_readwrite(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        with contextlib.redirect_stdout(io.StringIO()) as f:

            @wp.kernel
            def readwrite_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
                tid = wp.tid()
                b[tid] = a[tid] * a[tid]
                a[tid] = 1.0

            a = wp.array(np.arange(5), dtype=float, requires_grad=True, device=device)
            b = wp.zeros_like(a)

            tape = wp.Tape()
            with tape:
                wp.launch(readwrite_kernel, dim=5, inputs=[a, b], device=device)

        expected = "is being written to after it has been read from within the same kernel. This may corrupt gradient computation in the backward pass."
        test.assertIn(expected, f.getvalue())

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


# (kernel READ -> func WRITE) codegen failure case
def test_kernel_read_func_write(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        with contextlib.redirect_stdout(io.StringIO()) as f:

            @wp.func
            def write_func_2(x: wp.array(dtype=float), idx: int):
                x[idx] = 2.0

            @wp.kernel
            def read_kernel_func_write(x: wp.array(dtype=float), y: wp.array(dtype=float)):
                tid = wp.tid()
                a = x[tid]
                write_func_2(x, tid)
                y[tid] = a

            a = wp.array(np.array([1.0, 2.0, 3.0]), dtype=float, requires_grad=True, device=device)
            b = wp.zeros_like(a)

            tape = wp.Tape()
            with tape:
                wp.launch(kernel=read_kernel_func_write, dim=3, inputs=[a, b], device=device)

        expected = "written to after it has been read from within the same kernel. This may corrupt gradient computation in the backward pass."
        test.assertIn(expected, f.getvalue())

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


@wp.func
def atomic_func(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    c: wp.array(dtype=wp.int32),
    d: wp.array(dtype=wp.int32),
    i: int,
):
    wp.atomic_add(a, i, 1)
    wp.atomic_sub(b, i, 1)
    wp.atomic_min(c, i, 1)
    wp.atomic_max(d, i, 3)


@wp.kernel(enable_backward=False)
def atomic_kernel(
    a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32), c: wp.array(dtype=wp.int32), d: wp.array(dtype=wp.int32)
):
    i = wp.tid()
    atomic_func(a, b, c, d, i)


# atomic operations should mark arrays as WRITE
def test_atomic_operations(test, device):
    saved_verify_autograd_array_access_setting = wp.config.verify_autograd_array_access
    try:
        wp.config.verify_autograd_array_access = True

        a = wp.array((1, 2, 3), dtype=wp.int32, device=device)
        b = wp.array((1, 2, 3), dtype=wp.int32, device=device)
        c = wp.array((1, 2, 3), dtype=wp.int32, device=device)
        d = wp.array((1, 2, 3), dtype=wp.int32, device=device)

        wp.launch(atomic_kernel, dim=a.shape, inputs=(a, b, c, d), device=device)

        test.assertEqual(atomic_kernel.adj.args[0].is_write, True)
        test.assertEqual(atomic_kernel.adj.args[1].is_write, True)
        test.assertEqual(atomic_kernel.adj.args[2].is_write, True)
        test.assertEqual(atomic_kernel.adj.args[3].is_write, True)

    finally:
        wp.config.verify_autograd_array_access = saved_verify_autograd_array_access_setting


class TestOverwrite(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestOverwrite, "test_kernel_read_kernel_write", test_kernel_read_kernel_write, devices=devices)
add_function_test(
    TestOverwrite,
    "test_kernel_write_kernel_read_kernel_write",
    test_kernel_write_kernel_read_kernel_write,
    devices=devices,
)
add_function_test(
    TestOverwrite, "test_kernel_read_kernel_writeread", test_kernel_read_kernel_writeread, devices=devices
)
add_function_test(
    TestOverwrite, "test_kernel_writeread_kernel_write", test_kernel_writeread_kernel_write, devices=devices
)
add_function_test(TestOverwrite, "test_nested_function_read_write", test_nested_function_read_write, devices=devices)
add_function_test(TestOverwrite, "test_multidimensional_indexing", test_multidimensional_indexing, devices=devices)
add_function_test(TestOverwrite, "test_in_place_operators", test_in_place_operators, devices=devices)
add_function_test(TestOverwrite, "test_views", test_views, devices=devices)
add_function_test(TestOverwrite, "test_reset", test_reset, devices=devices)

add_function_test(TestOverwrite, "test_copy", test_copy, devices=devices)
add_function_test(TestOverwrite, "test_atomic_operations", test_atomic_operations, devices=devices)

# Some warning are only issued during codegen, and codegen only runs on cuda_0 in the MGPU case.
cuda_device = get_cuda_test_devices(mode="basic")

add_function_test(
    TestOverwrite, "test_in_place_operators_warning", test_in_place_operators_warning, devices=cuda_device
)
add_function_test(TestOverwrite, "test_kernel_readwrite", test_kernel_readwrite, devices=cuda_device)
add_function_test(TestOverwrite, "test_kernel_read_func_write", test_kernel_read_func_write, devices=cuda_device)

if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2)
