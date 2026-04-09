# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


# ==================================================================================
# Launch bounds tests
# ==================================================================================


@wp.kernel
def kernel_no_bounds(x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0


@wp.kernel(launch_bounds=256)
def kernel_single_bound(x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0


@wp.kernel(launch_bounds=(256, 1))
def kernel_tuple_bounds(x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0


@wp.kernel(launch_bounds=(512,))
def kernel_single_tuple_bound(x: wp.array(dtype=float)):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0


def test_launch_bounds_none(test, device):
    """Test kernel without launch_bounds"""
    n = 1024
    x = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device)
    wp.launch(kernel_no_bounds, dim=n, inputs=[x], device=device)
    wp.synchronize_device(device)
    assert_np_equal(x.numpy(), np.full(n, 2.0, dtype=np.float32))


def test_launch_bounds_single(test, device):
    """Test kernel with single int launch_bounds"""
    n = 1024
    x = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device)
    wp.launch(kernel_single_bound, dim=n, inputs=[x], device=device)
    wp.synchronize_device(device)
    assert_np_equal(x.numpy(), np.full(n, 2.0, dtype=np.float32))


def test_launch_bounds_tuple(test, device):
    """Test kernel with tuple launch_bounds (maxThreadsPerBlock, minBlocksPerMultiprocessor)"""
    n = 1024
    x = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device)
    wp.launch(kernel_tuple_bounds, dim=n, inputs=[x], device=device)
    wp.synchronize_device(device)
    assert_np_equal(x.numpy(), np.full(n, 2.0, dtype=np.float32))


def test_launch_bounds_single_tuple(test, device):
    """Test kernel with single-element tuple launch_bounds"""
    n = 1024
    x = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device)
    wp.launch(kernel_single_tuple_bound, dim=n, inputs=[x], device=device)
    wp.synchronize_device(device)
    assert_np_equal(x.numpy(), np.full(n, 2.0, dtype=np.float32))


# ==================================================================================
# Auto block_dim tests
# ==================================================================================


@wp.kernel
def saxpy(alpha: float, x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = alpha * x[i] + y[i]


def test_auto_block_dim_correctness(test, device):
    """Auto block_dim produces correct results for various launch sizes."""
    for n in [1, 31, 32, 33, 100, 1000, 100_000]:
        x = wp.ones(n, dtype=float, device=device)
        y = wp.zeros(n, dtype=float, device=device)
        wp.launch(saxpy, dim=n, inputs=[2.0, x, y], device=device)
        wp.synchronize_device(device)
        np.testing.assert_allclose(y.numpy(), np.full(n, 2.0))


def test_explicit_block_dim_passthrough(test, device):
    """Explicit block_dim is respected and not overridden."""
    n = 1000
    x = wp.ones(n, dtype=float, device=device)
    y = wp.zeros(n, dtype=float, device=device)
    wp.launch(saxpy, dim=n, inputs=[2.0, x, y], block_dim=64, device=device)
    wp.synchronize_device(device)
    np.testing.assert_allclose(y.numpy(), np.full(n, 2.0))


def test_auto_block_dim_values(test, device):
    """Auto block_dim is a warp-size multiple and within bounds."""
    if device == "cpu":
        return

    from warp._src.context import _select_block_dim  # noqa: PLC0415

    _suggested, min_blocks = wp.get_suggested_block_size(saxpy, device)
    dev = wp.get_device(device)

    for total in [1, 32, 1000, 100_000, 10_000_000]:
        bd = _select_block_dim(saxpy, total, dev)
        test.assertGreaterEqual(bd, 32, f"block_dim should be >= warp size for total={total}")
        test.assertEqual(bd % 32, 0, f"block_dim should be a warp-size multiple for total={total}")
        test.assertLessEqual(bd, 1024, f"block_dim should be <= 1024 for total={total}")

        default_blocks = (total + 256 - 1) // 256
        if default_blocks >= min_blocks:
            # Large launch — default fills the GPU, keep it.
            test.assertEqual(bd, 256, f"should use default for total={total}")
        else:
            # Small launch — should reduce below the default.
            test.assertLess(bd, 256, f"should reduce block_dim for total={total}")


def test_auto_block_dim_max_blocks(test, device):
    """When max_blocks caps the grid, block_dim should use the default."""
    if device == "cpu":
        return

    from warp._src.context import _select_block_dim  # noqa: PLC0415

    dev = wp.get_device(device)

    # Small total with max_blocks=4: grid is capped, should use default
    bd = _select_block_dim(saxpy, 100, dev, max_blocks=4)
    test.assertEqual(bd, 256, "should use default when max_blocks caps the grid")

    # Correctness check with max_blocks
    n = 1000
    x = wp.ones(n, dtype=float, device=device)
    y = wp.zeros(n, dtype=float, device=device)
    wp.launch(saxpy, dim=n, inputs=[2.0, x, y], max_blocks=4, device=device)
    wp.synchronize_device(device)
    np.testing.assert_allclose(y.numpy(), np.full(n, 2.0))


def test_auto_block_dim_graph_capture(test, device):
    """Auto block_dim works correctly under CUDA graph capture."""
    if device == "cpu":
        return

    n = 10_000
    x = wp.ones(n, dtype=float, device=device)
    y = wp.zeros(n, dtype=float, device=device)

    with wp.ScopedCapture(device=device) as capture:
        wp.launch(saxpy, dim=n, inputs=[2.0, x, y], device=device)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(y.numpy(), np.full(n, 2.0))


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

add_function_test(TestLaunch, "test_launch_bounds_none", test_launch_bounds_none, devices=devices)
add_function_test(TestLaunch, "test_launch_bounds_single", test_launch_bounds_single, devices=devices)
add_function_test(TestLaunch, "test_launch_bounds_tuple", test_launch_bounds_tuple, devices=devices)
add_function_test(TestLaunch, "test_launch_bounds_single_tuple", test_launch_bounds_single_tuple, devices=devices)

add_function_test(TestLaunch, "test_auto_block_dim_correctness", test_auto_block_dim_correctness, devices=devices)
add_function_test(
    TestLaunch, "test_explicit_block_dim_passthrough", test_explicit_block_dim_passthrough, devices=devices
)
add_function_test(TestLaunch, "test_auto_block_dim_values", test_auto_block_dim_values, devices=devices)
add_function_test(TestLaunch, "test_auto_block_dim_max_blocks", test_auto_block_dim_max_blocks, devices=devices)
add_function_test(TestLaunch, "test_auto_block_dim_graph_capture", test_auto_block_dim_graph_capture, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
