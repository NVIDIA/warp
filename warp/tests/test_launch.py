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

dim_x = wp.constant(2)
dim_y = wp.constant(2)
dim_z = wp.constant(2)
dim_w = wp.constant(2)


@wp.kernel
def kernel1d(a: wp.array(dtype=int, ndim=1)):
    i = wp.tid()

    wp.expect_eq(a[i], i)


@wp.kernel
def kernel2d(a: wp.array(dtype=int, ndim=2)):
    i, j = wp.tid()

    wp.expect_eq(a[i, j], i * dim_y + j)


@wp.kernel
def kernel3d(a: wp.array(dtype=int, ndim=3)):
    i, j, k = wp.tid()

    wp.expect_eq(a[i, j, k], i * dim_y * dim_z + j * dim_z + k)


@wp.kernel
def kernel4d(a: wp.array(dtype=int, ndim=4)):
    i, j, k, l = wp.tid()

    wp.expect_eq(a[i, j, k, l], i * dim_y * dim_z * dim_w + j * dim_z * dim_w + k * dim_w + l)


@wp.kernel
def square_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    i = wp.tid()
    output[i] = input[i] * input[i]


def test1d(test, device):
    a = np.arange(0, dim_x).reshape(dim_x)

    wp.launch(kernel1d, dim=a.shape, inputs=[wp.array(a, dtype=int, device=device)], device=device)


def test2d(test, device):
    a = np.arange(0, dim_x * dim_y).reshape(dim_x, dim_y)

    wp.launch(kernel2d, dim=a.shape, inputs=[wp.array(a, dtype=int, device=device)], device=device)


def test3d(test, device):
    a = np.arange(0, dim_x * dim_y * dim_z).reshape(dim_x, dim_y, dim_z)

    wp.launch(kernel3d, dim=a.shape, inputs=[wp.array(a, dtype=int, device=device)], device=device)


def test4d(test, device):
    a = np.arange(0, dim_x * dim_y * dim_z * dim_w).reshape(dim_x, dim_y, dim_z, dim_w)

    wp.launch(kernel4d, dim=a.shape, inputs=[wp.array(a, dtype=int, device=device)], device=device)


@wp.struct
class Params:
    a: wp.array(dtype=int)
    i: int
    f: float


@wp.kernel
def kernel_cmd(params: Params, i: int, f: float, v: wp.vec3, m: wp.mat33, out: wp.array(dtype=int)):
    tid = wp.tid()

    wp.expect_eq(params.i, i)
    wp.expect_eq(params.f, f)

    wp.expect_eq(i, int(f))

    wp.expect_eq(v[0], f)
    wp.expect_eq(v[1], f)
    wp.expect_eq(v[2], f)

    wp.expect_eq(m[0, 0], f)
    wp.expect_eq(m[1, 1], f)
    wp.expect_eq(m[2, 2], f)

    out[tid] = tid + i


def test_launch_cmd(test, device):
    """Tests recording and executing a kernel launch command.

    Verifies that:
    - A kernel can be recorded as a command without immediate execution
    - The recorded command can be launched later
    - Parameters are correctly passed to the kernel
    - Output matches expected results for both immediate and delayed launches

    Args:
        test: Test context
        device: Device to run the test on
    """
    n = 1
    ref = np.arange(0, n)
    out = wp.zeros(n, dtype=int, device=device)

    params = Params()
    params.i = 1
    params.f = 1.0

    v = wp.vec3(params.f, params.f, params.f)

    m = wp.mat33(params.f, 0.0, 0.0, 0.0, params.f, 0.0, 0.0, 0.0, params.f)

    # standard launch
    wp.launch(kernel_cmd, dim=n, inputs=[params, params.i, params.f, v, m, out], device=device)

    assert_np_equal(out.numpy(), ref + params.i)

    # cmd launch
    out.zero_()

    cmd = wp.launch(kernel_cmd, dim=n, inputs=[params, params.i, params.f, v, m, out], device=device, record_cmd=True)

    cmd.launch()

    assert_np_equal(out.numpy(), ref + params.i)


def test_launch_cmd_set_param(test, device):
    n = 1

    ref = np.arange(0, n)

    params = Params()
    v = wp.vec3()
    m = wp.mat33()

    cmd = wp.launch(kernel_cmd, dim=n, inputs=[params, 0, 0.0, v, m, None], device=device, record_cmd=True)

    # cmd param modification
    out = wp.zeros(n, dtype=int, device=device)

    params.i = 13
    params.f = 13.0

    v = wp.vec3(params.f, params.f, params.f)

    m = wp.mat33(params.f, 0.0, 0.0, 0.0, params.f, 0.0, 0.0, 0.0, params.f)

    cmd.set_param_at_index(0, params)
    cmd.set_param_at_index(1, params.i)
    cmd.set_param_at_index(2, params.f)
    cmd.set_param_at_index(3, v)
    cmd.set_param_at_index(4, m)
    cmd.set_param_by_name("out", out)

    cmd.launch()

    assert_np_equal(out.numpy(), ref + params.i)

    # test changing params after launch directly
    # because we now cache the ctypes object inside the wp.struct
    # instance  the command buffer will be automatically updated
    params.i = 14
    params.f = 14.0

    v = wp.vec3(params.f, params.f, params.f)

    m = wp.mat33(params.f, 0.0, 0.0, 0.0, params.f, 0.0, 0.0, 0.0, params.f)

    # this is the line we explicitly leave out to
    # ensure that param changes are reflected in the launch
    # launch.set_param_at_index(0, params)

    cmd.set_param_at_index(1, params.i)
    cmd.set_param_at_index(2, params.f)
    cmd.set_param_at_index(3, v)
    cmd.set_param_at_index(4, m)
    cmd.set_param_by_name("out", out)

    cmd.launch()

    assert_np_equal(out.numpy(), ref + params.i)


def test_launch_cmd_set_ctype(test, device):
    n = 1

    ref = np.arange(0, n)

    params = Params()
    v = wp.vec3()
    m = wp.mat33()

    cmd = wp.launch(kernel_cmd, dim=n, inputs=[params, 0, 0.0, v, m, None], device=device, record_cmd=True)

    # cmd param modification
    out = wp.zeros(n, dtype=int, device=device)

    # cmd param modification
    out.zero_()

    params.i = 13
    params.f = 13.0

    v = wp.vec3(params.f, params.f, params.f)

    m = wp.mat33(params.f, 0.0, 0.0, 0.0, params.f, 0.0, 0.0, 0.0, params.f)

    cmd.set_param_at_index_from_ctype(0, params.__ctype__())
    cmd.set_param_at_index_from_ctype(1, params.i)
    cmd.set_param_at_index_from_ctype(2, params.f)
    cmd.set_param_at_index_from_ctype(3, v)
    cmd.set_param_at_index_from_ctype(4, m)
    cmd.set_param_by_name_from_ctype("out", out.__ctype__())

    cmd.launch()

    assert_np_equal(out.numpy(), ref + params.i)


@wp.kernel
def arange(out: wp.array(dtype=int)):
    tid = wp.tid()
    out[tid] = tid


def test_launch_cmd_set_dim(test, device):
    n = 10

    ref = np.arange(0, n, dtype=int)
    out = wp.zeros(n, dtype=int, device=device)

    cmd = wp.launch(arange, dim=n, inputs=[out], device=device, record_cmd=True)

    cmd.set_dim(5)
    cmd.launch()

    # check first half the array is filled while rest is still zero
    assert_np_equal(out.numpy()[0:5], ref[0:5])
    assert_np_equal(out.numpy()[5:], np.zeros(5))

    out.zero_()

    cmd.set_dim(10)
    cmd.launch()

    # check the whole array was filled
    assert_np_equal(out.numpy(), ref)


def test_launch_cmd_empty(test, device):
    n = 10

    ref = np.arange(0, n, dtype=int)
    out = wp.zeros(n, dtype=int, device=device)

    cmd = wp.Launch(arange, device)
    cmd.set_dim(5)
    cmd.set_param_by_name("out", out)

    cmd.launch()

    # check first half the array is filled while rest is still zero
    assert_np_equal(out.numpy()[0:5], ref[0:5])
    assert_np_equal(out.numpy()[5:], np.zeros(5))

    out.zero_()

    cmd.set_dim(10)
    cmd.launch()

    # check the whole array was filled
    assert_np_equal(out.numpy(), ref)


def test_launch_cmd_adjoint(test, device):
    """Test recording an adjoint launch with record_cmd=True."""
    input_arr = wp.array([1.0, 2.0, 3.0], dtype=float, requires_grad=True, device=device)
    output_arr = wp.empty_like(input_arr)

    output_arr.grad.fill_(1.0)

    cmd = wp.launch(
        square_kernel,
        dim=input_arr.size,
        inputs=[input_arr, output_arr],
        adj_inputs=[None, None],
        adjoint=True,
        device=device,
        record_cmd=True,
    )

    cmd.launch()

    assert_np_equal(input_arr.grad.numpy(), np.array([2.0, 4.0, 6.0]))


def test_launch_cmd_adjoint_empty(test, device):
    """Test constructing a Launch object for an adjoint kernel."""
    input_arr = wp.array([1.0, 2.0, 3.0], dtype=float, requires_grad=True, device=device)
    output_arr = wp.empty_like(input_arr)
    output_arr.grad.fill_(1.0)

    cmd = wp.Launch(square_kernel, device, adjoint=True)
    cmd.set_param_by_name("input", input_arr)
    cmd.set_param_by_name("output", output_arr)
    cmd.set_dim(input_arr.size)
    cmd.launch()

    assert_np_equal(input_arr.grad.numpy(), np.array([2.0, 4.0, 6.0]))

    # Now update the launch object's parameters with arrays of different sizes and values
    # and check that the adjoints are correctly computed
    input_arr_updated = wp.array([4.0, 5.0, 6.0, 7.0], dtype=float, device=device)
    input_arr_updated_grad = wp.zeros_like(input_arr_updated)

    output_arr_updated = wp.empty_like(input_arr_updated)
    output_arr_updated_grad = wp.full_like(output_arr_updated, 1.0)

    cmd.set_param_by_name("input", input_arr_updated)
    cmd.set_param_by_name("output", output_arr_updated)
    cmd.set_param_by_name("input", input_arr_updated_grad, adjoint=True)
    cmd.set_param_by_name("output", output_arr_updated_grad, adjoint=True)
    cmd.set_dim(input_arr_updated.size)
    cmd.launch()

    assert_np_equal(input_arr_updated_grad.numpy(), np.array([8.0, 10.0, 12.0, 14.0]))


@wp.kernel
def kernel_mul(values: wp.array(dtype=int), coeff: int, out: wp.array(dtype=int)):
    tid = wp.tid()
    out[tid] = values[tid] * coeff


def test_launch_tuple_args(test, device):
    values = wp.array(np.arange(0, 4), dtype=int, device=device)
    coeff = 3
    out = wp.empty_like(values)

    wp.launch(
        kernel_mul,
        dim=len(values),
        inputs=(
            values,
            coeff,
        ),
        outputs=(out,),
        device=device,
    )
    assert_np_equal(out.numpy(), np.array((0, 3, 6, 9)))

    wp.launch(kernel_mul, dim=len(values), inputs=(values, coeff, out), device=device)
    assert_np_equal(out.numpy(), np.array((0, 3, 6, 9)))

    wp.launch(kernel_mul, dim=len(values), outputs=(values, coeff, out), device=device)
    assert_np_equal(out.numpy(), np.array((0, 3, 6, 9)))


devices = get_test_devices()


class TestLaunch(unittest.TestCase):
    pass


add_function_test(TestLaunch, "test_launch_1d", test1d, devices=devices)
add_function_test(TestLaunch, "test_launch_2d", test2d, devices=devices)
add_function_test(TestLaunch, "test_launch_3d", test3d, devices=devices)
add_function_test(TestLaunch, "test_launch_4d", test4d, devices=devices)

add_function_test(TestLaunch, "test_launch_cmd", test_launch_cmd, devices=devices)
add_function_test(TestLaunch, "test_launch_cmd_set_param", test_launch_cmd_set_param, devices=devices)
add_function_test(TestLaunch, "test_launch_cmd_set_ctype", test_launch_cmd_set_ctype, devices=devices)
add_function_test(TestLaunch, "test_launch_cmd_set_dim", test_launch_cmd_set_dim, devices=devices)
add_function_test(TestLaunch, "test_launch_cmd_empty", test_launch_cmd_empty, devices=devices)
add_function_test(TestLaunch, "test_launch_cmd_adjoint", test_launch_cmd_adjoint, devices=devices)
add_function_test(TestLaunch, "test_launch_cmd_adjoint_empty", test_launch_cmd_adjoint_empty, devices=devices)

add_function_test(TestLaunch, "test_launch_tuple_args", test_launch_tuple_args, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
