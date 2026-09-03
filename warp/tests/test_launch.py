# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import unittest
import warnings

import numpy as np

import warp as wp
from warp._src import logger as _logger
from warp.tests.unittest_utils import *

dim_x = wp.constant(2)
dim_y = wp.constant(2)
dim_z = wp.constant(2)
dim_w = wp.constant(2)


@wp.kernel
def kernel1d(a: wp.array[int]):
    i = wp.tid()

    wp.expect_eq(a[i], i)


@wp.kernel
def kernel2d(a: wp.array2d[int]):
    i, j = wp.tid()

    wp.expect_eq(a[i, j], i * dim_y + j)


@wp.kernel
def kernel3d(a: wp.array3d[int]):
    i, j, k = wp.tid()

    wp.expect_eq(a[i, j, k], i * dim_y * dim_z + j * dim_z + k)


@wp.kernel
def kernel4d(a: wp.array4d[int]):
    i, j, k, l = wp.tid()

    wp.expect_eq(a[i, j, k, l], i * dim_y * dim_z * dim_w + j * dim_z * dim_w + k * dim_w + l)


@wp.kernel
def square_kernel(input: wp.array[float], output: wp.array[float]):
    i = wp.tid()
    output[i] = input[i] * input[i]


@wp.kernel
def noop_kernel():
    tid = wp.tid()


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
    a: wp.array[int]
    i: int
    f: float


@wp.kernel
def kernel_cmd(params: Params, i: int, f: float, v: wp.vec3, m: wp.mat33, out: wp.array[int]):
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
def arange(out: wp.array[int]):
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
def kernel_mul(values: wp.array[int], coeff: int, out: wp.array[int]):
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
def kernel_no_bounds(x: wp.array[float]):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0


@wp.kernel(launch_bounds=256)
def kernel_single_bound(x: wp.array[float]):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0


@wp.kernel(launch_bounds=(256, 1))
def kernel_tuple_bounds(x: wp.array[float]):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0


@wp.kernel(launch_bounds=(512,))
def kernel_single_tuple_bound(x: wp.array[float]):
    tid = wp.tid()
    x[tid] = x[tid] * 2.0


@wp.kernel(launch_bounds=256)
def bounded_square_kernel(data: wp.array[float], output: wp.array[float]):
    i = wp.tid()
    output[i] = data[i] * data[i]


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


@wp.kernel
def kernel_vec3_param(v: wp.vec3):
    tid = wp.tid()


@wp.kernel
def kernel_mat22_param(m: wp.mat22):
    tid = wp.tid()


@wp.kernel
def kernel_quat_param(q: wp.quat):
    tid = wp.tid()


@wp.kernel
def kernel_transform_param(t: wp.transform):
    tid = wp.tid()


@wp.kernel
def kernel_composite_params(v: wp.vec3, m: wp.mat22, q: wp.quat, t: wp.transform, out: wp.array[float]):
    out[0] = v[0]
    out[1] = v[1]
    out[2] = v[2]
    out[3] = m[0, 0]
    out[4] = m[0, 1]
    out[5] = m[1, 0]
    out[6] = m[1, 1]
    out[7] = q[0]
    out[8] = q[1]
    out[9] = q[2]
    out[10] = q[3]
    out[11] = t[0]
    out[12] = t[1]
    out[13] = t[2]
    out[14] = t[3]
    out[15] = t[4]
    out[16] = t[5]
    out[17] = t[6]


def test_launch_cmd_composite_defaults(test, device):
    """Composite parameters left unset on a default-constructed Launch are zero-initialized."""
    out = wp.full(18, -1.0, dtype=float, device=device)

    cmd = wp.Launch(kernel_composite_params, device)
    cmd.set_param_by_name("out", out)
    cmd.set_dim(1)
    cmd.launch()

    wp.synchronize_device(device)

    assert_np_equal(out.numpy(), np.zeros(18, dtype=np.float32))

    out = wp.full(18, -1.0, dtype=float, device=device, requires_grad=True)
    out.grad.fill_(1.0)

    cmd = wp.Launch(kernel_composite_params, device, adjoint=True)
    adjoint_arg_offset = len(cmd.kernel.adj.args) + 1
    adjoint_composites = cmd.params[adjoint_arg_offset : adjoint_arg_offset + 4]
    adjoint_components = np.concatenate([np.array(value).reshape(-1) for value in adjoint_composites])
    assert_np_equal(adjoint_components, np.zeros(18, dtype=np.float32))

    cmd.set_param_by_name("out", out)
    cmd.set_param_by_name("out", out.grad, adjoint=True)
    cmd.set_dim(1)
    cmd.launch()

    wp.synchronize_device(device)


def test_launch_device_block_dim_failure(test, device):
    """Raise when CUDA rejects an oversized launch block.

    Protects users from continuing after native stderr with kernel outputs left unchanged.
    """
    with test.assertRaisesRegex(RuntimeError, r"Error launching kernel: .*noop_kernel.*Warp CUDA error"):
        wp.launch(noop_kernel, dim=1, block_dim=2048, device=device)


def test_launch_bounds_block_dim_failure(test, device):
    """Raise when CUDA rejects a launch-bounds violation.

    Protects users from silently skipping kernels whose outputs feed later simulation stages.
    """
    x = wp.ones(1, dtype=float, device=device)

    with test.assertRaisesRegex(RuntimeError, r"Error launching kernel: .*kernel_single_bound.*Warp CUDA error"):
        wp.launch(kernel_single_bound, dim=1, inputs=[x], block_dim=512, device=device)


def test_launch_cmd_block_dim_failure(test, device):
    """Raise when recorded launches hit CUDA launch errors.

    Protects recorded command replay from returning normally with stale outputs.
    """
    x = wp.ones(1, dtype=float, device=device)
    cmd = wp.launch(kernel_single_bound, dim=1, inputs=[x], block_dim=512, device=device, record_cmd=True)

    with test.assertRaisesRegex(RuntimeError, r"Error launching kernel: .*kernel_single_bound.*Warp CUDA error"):
        cmd.launch()


def test_launch_adjoint_block_dim_failure(test, device):
    """Raise when adjoint launches hit CUDA launch errors.

    Protects differentiable simulations from using missing or partial gradients.
    """
    input_arr = wp.array([1.0], dtype=float, requires_grad=True, device=device)
    output_arr = wp.empty_like(input_arr)
    output_arr.grad.fill_(1.0)

    with test.assertRaisesRegex(RuntimeError, r"Error launching kernel: .*bounded_square_kernel.*Warp CUDA error"):
        wp.launch(
            bounded_square_kernel,
            dim=input_arr.size,
            inputs=[input_arr, output_arr],
            adj_inputs=[None, None],
            adjoint=True,
            block_dim=512,
            device=device,
        )


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestLaunch(unittest.TestCase):
    def test_launch_scalar_to_composite_param_rejected(self):
        """A single value passed where a composite kernel parameter is expected must be rejected."""
        kernels = (
            (kernel_vec3_param, "v", "vec3f"),
            (kernel_mat22_param, "m", "mat22f"),
            (kernel_quat_param, "q", "quatf"),
            (kernel_transform_param, "t", "transformf"),
        )
        values = (123, 1.5, wp.float32(1.5), wp.int32(2))

        for kernel, param, type_name in kernels:
            for value in values:
                with self.subTest(param=param, value_type=type(value).__name__):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"argument '{param}' expects {type_name} but got a single value",
                    ):
                        wp.launch(kernel, dim=1, inputs=[value])

        # Conversions from containers must keep working.
        for value in ((1.0, 2.0, 3.0), [1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0]), wp.vec3(1.0, 2.0, 3.0)):
            with self.subTest(container=type(value).__name__):
                wp.launch(kernel_vec3_param, dim=1, inputs=[value])

        # `None` is not a single value being promoted, it keeps failing as before.
        with self.assertRaisesRegex(ValueError, r"Failed to convert argument for param v to vec3f"):
            wp.launch(kernel_vec3_param, dim=1, inputs=[None])

        wp.synchronize_device()

    def test_launch_numpy_and_boolean_scalar_to_composite_param_deprecated(self):
        """NumPy numeric scalars and Python, NumPy, and Warp Booleans must warn while promotion is supported."""
        values = (True, np.bool_(True), wp.bool(True), np.float32(1.5), np.int64(3))
        params = (("v", "vec3f"), ("m", "mat22f"), ("q", "quatf"), ("t", "transformf"))
        out = wp.empty(18, dtype=float)

        saved_warnings_seen = _logger._warnings_seen.copy()
        try:
            for value in values:
                with self.subTest(value_type=type(value).__name__):
                    _logger._warnings_seen.clear()
                    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()) as stderr:
                        warnings.simplefilter("always", DeprecationWarning)
                        wp.launch(kernel_composite_params, dim=1, inputs=[value, value, value, value, out])

                    warning_output = stderr.getvalue()
                    for param, type_name in params:
                        self.assertRegex(
                            warning_output,
                            rf"type `{type(value).__name__}`.*`{type_name}`.*kernel parameter '{param}'.*deprecated",
                        )

                    assert_np_equal(out.numpy(), np.full(18, float(value), dtype=np.float32))
        finally:
            _logger._warnings_seen.clear()
            _logger._warnings_seen.update(saved_warnings_seen)

    def test_launch_cmd_set_param_scalar_to_composite_rejected(self):
        """A single value set on a recorded launch's composite parameter must be rejected."""
        cmd = wp.Launch(kernel_vec3_param, wp.get_device())

        with self.assertRaisesRegex(RuntimeError, r"argument 'v' expects vec3f but got a single value"):
            cmd.set_param_at_index(0, 123)

        with self.assertRaisesRegex(RuntimeError, r"argument 'v' expects vec3f but got a single value"):
            cmd.set_param_by_name("v", 123)

        # Explicit construction must keep working.
        cmd.set_param_by_name("v", wp.vec3(1.0, 2.0, 3.0))


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
add_function_test(TestLaunch, "test_launch_cmd_composite_defaults", test_launch_cmd_composite_defaults, devices=devices)

add_function_test(TestLaunch, "test_launch_tuple_args", test_launch_tuple_args, devices=devices)

add_function_test(TestLaunch, "test_launch_bounds_none", test_launch_bounds_none, devices=devices)
add_function_test(TestLaunch, "test_launch_bounds_single", test_launch_bounds_single, devices=devices)
add_function_test(TestLaunch, "test_launch_bounds_tuple", test_launch_bounds_tuple, devices=devices)
add_function_test(TestLaunch, "test_launch_bounds_single_tuple", test_launch_bounds_single_tuple, devices=devices)
add_function_test(
    TestLaunch, "test_launch_device_block_dim_failure", test_launch_device_block_dim_failure, devices=cuda_devices
)
add_function_test(
    TestLaunch, "test_launch_bounds_block_dim_failure", test_launch_bounds_block_dim_failure, devices=cuda_devices
)
add_function_test(
    TestLaunch, "test_launch_cmd_block_dim_failure", test_launch_cmd_block_dim_failure, devices=cuda_devices
)
add_function_test(
    TestLaunch, "test_launch_adjoint_block_dim_failure", test_launch_adjoint_block_dim_failure, devices=cuda_devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
