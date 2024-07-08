# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

np_signed_int_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.byte,
]

np_unsigned_int_types = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ubyte,
]

np_int_types = np_signed_int_types + np_unsigned_int_types

np_float_types = [np.float16, np.float32, np.float64]

np_scalar_types = np_int_types + np_float_types


def randvals(rng, shape, dtype):
    if dtype in np_float_types:
        return rng.standard_normal(size=shape).astype(dtype)
    elif dtype in [np.int8, np.uint8, np.byte, np.ubyte]:
        return rng.integers(1, high=3, size=shape, dtype=dtype)
    return rng.integers(1, high=5, size=shape, dtype=dtype)


kernel_cache = {}


def getkernel(func, suffix=""):
    key = func.__name__ + "_" + suffix
    if key not in kernel_cache:
        kernel_cache[key] = wp.Kernel(func=func, key=key)
    return kernel_cache[key]


def get_select_kernel(dtype):
    def output_select_kernel_fn(
        input: wp.array(dtype=dtype),
        index: int,
        out: wp.array(dtype=dtype),
    ):
        out[0] = input[index]

    return getkernel(output_select_kernel_fn, suffix=dtype.__name__)


def get_select_kernel2(dtype):
    def output_select_kernel2_fn(
        input: wp.array(dtype=dtype, ndim=2),
        index0: int,
        index1: int,
        out: wp.array(dtype=dtype),
    ):
        out[0] = input[index0, index1]

    return getkernel(output_select_kernel2_fn, suffix=dtype.__name__)


def test_arrays(test, device, dtype):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    arr_np = randvals(rng, (10, 5), dtype)
    arr = wp.array(arr_np, dtype=wptype, requires_grad=True, device=device)

    assert_np_equal(arr.numpy(), arr_np, tol=tol)


def test_unary_ops(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_unary(
        inputs: wp.array(dtype=wptype, ndim=2),
        outputs: wp.array(dtype=wptype, ndim=2),
    ):
        for i in range(10):
            i0 = inputs[0, i]
            i1 = inputs[1, i]
            i2 = inputs[2, i]
            i3 = inputs[3, i]
            i4 = inputs[4, i]

            # multiply outputs by 2 so we've got something to backpropagate:
            outputs[0, i] = wptype(2.0) * (+i0)
            outputs[1, i] = wptype(2.0) * (-i1)
            outputs[2, i] = wptype(2.0) * wp.sign(i2)
            outputs[3, i] = wptype(2.0) * wp.abs(i3)
            outputs[4, i] = wptype(2.0) * wp.step(i4)

    kernel = getkernel(check_unary, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel2(wptype)

    if register_kernels:
        return

    if dtype in np_float_types:
        inputs = wp.array(
            rng.standard_normal(size=(5, 10)).astype(dtype), dtype=wptype, requires_grad=True, device=device
        )
    else:
        inputs = wp.array(
            rng.integers(-2, high=3, size=(5, 10), dtype=dtype), dtype=wptype, requires_grad=True, device=device
        )
    outputs = wp.zeros_like(inputs)

    wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
    assert_np_equal(outputs.numpy()[0], 2 * inputs.numpy()[0], tol=tol)
    assert_np_equal(outputs.numpy()[1], -2 * inputs.numpy()[1], tol=tol)
    expected = 2 * np.sign(inputs.numpy()[2])
    expected[expected == 0] = 2
    assert_np_equal(outputs.numpy()[2], expected, tol=tol)
    assert_np_equal(outputs.numpy()[3], 2 * np.abs(inputs.numpy()[3]), tol=tol)
    assert_np_equal(outputs.numpy()[4], 2 * (1 - np.heaviside(inputs.numpy()[4], 1)), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(10):
            # grad of 2x:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 0, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            expected_grads[0, i] = 2
            assert_np_equal(tape.gradients[inputs].numpy(), expected_grads, tol=tol)
            tape.zero()

            # grad of -2x:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 1, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            expected_grads[1, i] = -2
            assert_np_equal(tape.gradients[inputs].numpy(), expected_grads, tol=tol)
            tape.zero()

            # grad of 2 * sign(x):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 2, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            assert_np_equal(tape.gradients[inputs].numpy(), expected_grads, tol=tol)
            tape.zero()

            # grad of 2 * abs(x):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 3, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            expected_grads[3, i] = 2 * np.sign(inputs.numpy()[3, i])
            assert_np_equal(tape.gradients[inputs].numpy(), expected_grads, tol=tol)
            tape.zero()

            # grad of 2 * step(x):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 4, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            assert_np_equal(tape.gradients[inputs].numpy(), expected_grads, tol=tol)
            tape.zero()


def test_nonzero(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_nonzero(
        inputs: wp.array(dtype=wptype),
        outputs: wp.array(dtype=wptype),
    ):
        for i in range(10):
            i0 = inputs[i]
            outputs[i] = wp.nonzero(i0)

    kernel = getkernel(check_nonzero, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    inputs = wp.array(rng.integers(-2, high=3, size=10).astype(dtype), dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(inputs)

    wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
    assert_np_equal(outputs.numpy(), (inputs.numpy() != 0))

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(10):
            # grad should just be zero:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected_grads = np.zeros_like(inputs.numpy())
            assert_np_equal(tape.gradients[inputs].numpy(), expected_grads, tol=tol)
            tape.zero()


def test_binary_ops(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_binary_ops(
        in1: wp.array(dtype=wptype, ndim=2),
        in2: wp.array(dtype=wptype, ndim=2),
        outputs: wp.array(dtype=wptype, ndim=2),
    ):
        for i in range(10):
            i0 = in1[0, i]
            i1 = in1[1, i]
            i2 = in1[2, i]
            i3 = in1[3, i]
            i4 = in1[4, i]
            i5 = in1[5, i]
            i6 = in1[6, i]
            i7 = in1[7, i]

            j0 = in2[0, i]
            j1 = in2[1, i]
            j2 = in2[2, i]
            j3 = in2[3, i]
            j4 = in2[4, i]
            j5 = in2[5, i]
            j6 = in2[6, i]
            j7 = in2[7, i]

            outputs[0, i] = wptype(2) * wp.mul(i0, j0)
            outputs[1, i] = wptype(2) * wp.div(i1, j1)
            outputs[2, i] = wptype(2) * wp.add(i2, j2)
            outputs[3, i] = wptype(2) * wp.sub(i3, j3)
            outputs[4, i] = wptype(2) * wp.mod(i4, j4)
            outputs[5, i] = wptype(2) * wp.min(i5, j5)
            outputs[6, i] = wptype(2) * wp.max(i6, j6)
            outputs[7, i] = wptype(2) * wp.floordiv(i7, j7)

    kernel = getkernel(check_binary_ops, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel2(wptype)

    if register_kernels:
        return

    vals1 = randvals(rng, [8, 10], dtype)
    if dtype in [np_unsigned_int_types]:
        vals2 = vals1 + randvals(rng, [8, 10], dtype)
    else:
        vals2 = np.abs(randvals(rng, [8, 10], dtype))

    in1 = wp.array(vals1, dtype=wptype, requires_grad=True, device=device)
    in2 = wp.array(vals2, dtype=wptype, requires_grad=True, device=device)

    outputs = wp.zeros_like(in1)

    wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)

    assert_np_equal(outputs.numpy()[0], 2 * in1.numpy()[0] * in2.numpy()[0], tol=tol)
    if dtype in np_float_types:
        assert_np_equal(outputs.numpy()[1], 2 * in1.numpy()[1] / (in2.numpy()[1]), tol=tol)
    else:
        assert_np_equal(outputs.numpy()[1], 2 * (in1.numpy()[1] // (in2.numpy()[1])), tol=tol)
    assert_np_equal(outputs.numpy()[2], 2 * (in1.numpy()[2] + (in2.numpy()[2])), tol=tol)
    assert_np_equal(outputs.numpy()[3], 2 * (in1.numpy()[3] - (in2.numpy()[3])), tol=tol)

    # ...so this is actually the desired behaviour right? Looks like wp.mod doesn't behave like
    # python's % operator or np.mod()...
    assert_np_equal(
        outputs.numpy()[4],
        2
        * (
            (in1.numpy()[4])
            - (in2.numpy()[4]) * np.sign(in1.numpy()[4]) * np.floor(np.abs(in1.numpy()[4]) / (in2.numpy()[4]))
        ),
        tol=tol,
    )

    assert_np_equal(outputs.numpy()[5], 2 * np.minimum(in1.numpy()[5], in2.numpy()[5]), tol=tol)
    assert_np_equal(outputs.numpy()[6], 2 * np.maximum(in1.numpy()[6], in2.numpy()[6]), tol=tol)
    assert_np_equal(outputs.numpy()[7], 2 * np.floor_divide(in1.numpy()[7], in2.numpy()[7]), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(10):
            # multiplication:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 0, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[0, i] = 2.0 * in2.numpy()[0, i]
            assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
            expected[0, i] = 2.0 * in1.numpy()[0, i]
            assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            tape.zero()

            # division:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 1, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[1, i] = 2.0 / (in2.numpy()[1, i])
            assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
            # y = x1/x2
            # dy/dx2 = -x1/x2^2
            expected[1, i] = (-2.0) * (in1.numpy()[1, i] / (in2.numpy()[1, i] ** 2))
            assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            tape.zero()

            # addition:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 2, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[2, i] = 2.0
            assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
            expected[2, i] = 2.0
            assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            tape.zero()

            # subtraction:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 3, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[3, i] = 2.0
            assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
            expected[3, i] = -2.0
            assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            tape.zero()

            # modulus. unless at discontinuities,
            # d/dx1( x1 % x2 ) == 1
            # d/dx2( x1 % x2 ) == 0
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 4, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[4, i] = 2.0
            assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
            expected[4, i] = 0.0
            assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            tape.zero()

            # min
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 5, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[5, i] = 2.0 if (in1.numpy()[5, i] < in2.numpy()[5, i]) else 0.0
            assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
            expected[5, i] = 2.0 if (in2.numpy()[5, i] < in1.numpy()[5, i]) else 0.0
            assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            tape.zero()

            # max
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 6, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[6, i] = 2.0 if (in1.numpy()[6, i] > in2.numpy()[6, i]) else 0.0
            assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
            expected[6, i] = 2.0 if (in2.numpy()[6, i] > in1.numpy()[6, i]) else 0.0
            assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            tape.zero()

            # floor_divide. Returns integers so gradient is zero
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 7, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
            assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            tape.zero()


def test_special_funcs(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_special_funcs(
        inputs: wp.array(dtype=wptype, ndim=2),
        outputs: wp.array(dtype=wptype, ndim=2),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        for i in range(10):
            outputs[0, i] = wptype(2) * wp.log(inputs[0, i])
            outputs[1, i] = wptype(2) * wp.log2(inputs[1, i])
            outputs[2, i] = wptype(2) * wp.log10(inputs[2, i])
            outputs[3, i] = wptype(2) * wp.exp(inputs[3, i])
            outputs[4, i] = wptype(2) * wp.atan(inputs[4, i])
            outputs[5, i] = wptype(2) * wp.sin(inputs[5, i])
            outputs[6, i] = wptype(2) * wp.cos(inputs[6, i])
            outputs[7, i] = wptype(2) * wp.sqrt(inputs[7, i])
            outputs[8, i] = wptype(2) * wp.tan(inputs[8, i])
            outputs[9, i] = wptype(2) * wp.sinh(inputs[9, i])
            outputs[10, i] = wptype(2) * wp.cosh(inputs[10, i])
            outputs[11, i] = wptype(2) * wp.tanh(inputs[11, i])
            outputs[12, i] = wptype(2) * wp.acos(inputs[12, i])
            outputs[13, i] = wptype(2) * wp.asin(inputs[13, i])
            outputs[14, i] = wptype(2) * wp.cbrt(inputs[14, i])

    kernel = getkernel(check_special_funcs, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel2(wptype)

    if register_kernels:
        return

    invals = rng.normal(size=(15, 10)).astype(dtype)
    invals[[0, 1, 2, 7, 14]] = 0.1 + np.abs(invals[[0, 1, 2, 7, 14]])
    invals[12] = np.clip(invals[12], -0.9, 0.9)
    invals[13] = np.clip(invals[13], -0.9, 0.9)
    inputs = wp.array(invals, dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(inputs)

    wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)

    assert_np_equal(outputs.numpy()[0], 2 * np.log(inputs.numpy()[0]), tol=tol)
    assert_np_equal(outputs.numpy()[1], 2 * np.log2(inputs.numpy()[1]), tol=tol)
    assert_np_equal(outputs.numpy()[2], 2 * np.log10(inputs.numpy()[2]), tol=tol)
    assert_np_equal(outputs.numpy()[3], 2 * np.exp(inputs.numpy()[3]), tol=tol)
    assert_np_equal(outputs.numpy()[4], 2 * np.arctan(inputs.numpy()[4]), tol=tol)
    assert_np_equal(outputs.numpy()[5], 2 * np.sin(inputs.numpy()[5]), tol=tol)
    assert_np_equal(outputs.numpy()[6], 2 * np.cos(inputs.numpy()[6]), tol=tol)
    assert_np_equal(outputs.numpy()[7], 2 * np.sqrt(inputs.numpy()[7]), tol=tol)
    assert_np_equal(outputs.numpy()[8], 2 * np.tan(inputs.numpy()[8]), tol=tol)
    assert_np_equal(outputs.numpy()[9], 2 * np.sinh(inputs.numpy()[9]), tol=tol)
    assert_np_equal(outputs.numpy()[10], 2 * np.cosh(inputs.numpy()[10]), tol=tol)
    assert_np_equal(outputs.numpy()[11], 2 * np.tanh(inputs.numpy()[11]), tol=tol)
    assert_np_equal(outputs.numpy()[12], 2 * np.arccos(inputs.numpy()[12]), tol=tol)
    assert_np_equal(outputs.numpy()[13], 2 * np.arcsin(inputs.numpy()[13]), tol=tol)
    assert_np_equal(outputs.numpy()[14], 2 * np.cbrt(inputs.numpy()[14]), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(10):
            # log:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 0, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[0, i] = 2.0 / inputs.numpy()[0, i]
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # log2:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 1, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[1, i] = 2.0 / (inputs.numpy()[1, i] * np.log(2.0))
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # log10:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 2, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[2, i] = 2.0 / (inputs.numpy()[2, i] * np.log(10.0))
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # exp:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 3, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[3, i] = outputs.numpy()[3, i]
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # arctan:
            # looks like the autodiff formula in warp was wrong? Was (1 + x^2) rather than
            # 1/(1 + x^2)
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 4, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[4, i] = 2.0 / (inputs.numpy()[4, i] ** 2 + 1)
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # sin:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 5, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[5, i] = np.cos(inputs.numpy()[5, i]) * 2
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # cos:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 6, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[6, i] = -np.sin(inputs.numpy()[6, i]) * 2.0
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # sqrt:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 7, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[7, i] = 1.0 / (np.sqrt(inputs.numpy()[7, i]))
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # tan:
            # looks like there was a bug in autodiff formula here too - gradient was zero if cos(x) > 0
            # (should have been "if(cosx != 0)")
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 8, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[8, i] = 2.0 / (np.cos(inputs.numpy()[8, i]) ** 2)
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=200 * tol)
            tape.zero()

            # sinh:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 9, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[9, i] = 2.0 * np.cosh(inputs.numpy()[9, i])
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # cosh:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 10, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[10, i] = 2.0 * np.sinh(inputs.numpy()[10, i])
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # tanh:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 11, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[11, i] = 2.0 / (np.cosh(inputs.numpy()[11, i]) ** 2)
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # arccos:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 12, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[12, i] = -2.0 / np.sqrt(1 - inputs.numpy()[12, i] ** 2)
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()

            # arcsin:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 13, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            expected[13, i] = 2.0 / np.sqrt(1 - inputs.numpy()[13, i] ** 2)
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=6 * tol)
            tape.zero()

            # cbrt:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 14, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(inputs.numpy())
            cbrt = np.cbrt(inputs.numpy()[14, i], dtype=np.dtype(dtype))
            expected[14, i] = (2.0 / 3.0) * (1.0 / (cbrt * cbrt))
            assert_np_equal(tape.gradients[inputs].numpy(), expected, tol=tol)
            tape.zero()


def test_special_funcs_2arg(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_special_funcs_2arg(
        in1: wp.array(dtype=wptype, ndim=2),
        in2: wp.array(dtype=wptype, ndim=2),
        outputs: wp.array(dtype=wptype, ndim=2),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        for i in range(10):
            outputs[0, i] = wptype(2) * wp.pow(in1[0, i], in2[0, i])
            outputs[1, i] = wptype(2) * wp.atan2(in1[1, i], in2[1, i])

    kernel = getkernel(check_special_funcs_2arg, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel2(wptype)

    if register_kernels:
        return

    in1 = wp.array(np.abs(randvals(rng, [2, 10], dtype)), dtype=wptype, requires_grad=True, device=device)
    in2 = wp.array(randvals(rng, [2, 10], dtype), dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(in1)

    wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)

    assert_np_equal(outputs.numpy()[0], 2.0 * np.power(in1.numpy()[0], in2.numpy()[0]), tol=tol)
    assert_np_equal(outputs.numpy()[1], 2.0 * np.arctan2(in1.numpy()[1], in2.numpy()[1]), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(10):
            # pow:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 0, i], outputs=[out], device=device)
            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[0, i] = 2.0 * in2.numpy()[0, i] * np.power(in1.numpy()[0, i], in2.numpy()[0, i] - 1)
            assert_np_equal(tape.gradients[in1].numpy(), expected, tol=5 * tol)
            expected[0, i] = 2.0 * np.power(in1.numpy()[0, i], in2.numpy()[0, i]) * np.log(in1.numpy()[0, i])
            assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            tape.zero()

            # atan2:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 1, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(in1.numpy())
            expected[1, i] = 2.0 * in2.numpy()[1, i] / (in1.numpy()[1, i] ** 2 + in2.numpy()[1, i] ** 2)
            assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
            expected[1, i] = -2.0 * in1.numpy()[1, i] / (in1.numpy()[1, i] ** 2 + in2.numpy()[1, i] ** 2)
            assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            tape.zero()


def test_float_to_int(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_float_to_int(
        inputs: wp.array(dtype=wptype, ndim=2),
        outputs: wp.array(dtype=wptype, ndim=2),
    ):
        for i in range(10):
            outputs[0, i] = wp.round(inputs[0, i])
            outputs[1, i] = wp.rint(inputs[1, i])
            outputs[2, i] = wp.trunc(inputs[2, i])
            outputs[3, i] = wp.floor(inputs[3, i])
            outputs[4, i] = wp.ceil(inputs[4, i])
            outputs[5, i] = wp.frac(inputs[5, i])

    kernel = getkernel(check_float_to_int, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel2(wptype)

    if register_kernels:
        return

    inputs = wp.array(rng.standard_normal(size=(6, 10)).astype(dtype), dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(inputs)

    wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)

    assert_np_equal(outputs.numpy()[0], np.round(inputs.numpy()[0]))
    assert_np_equal(outputs.numpy()[1], np.rint(inputs.numpy()[1]))
    assert_np_equal(outputs.numpy()[2], np.trunc(inputs.numpy()[2]))
    assert_np_equal(outputs.numpy()[3], np.floor(inputs.numpy()[3]))
    assert_np_equal(outputs.numpy()[4], np.ceil(inputs.numpy()[4]))
    assert_np_equal(outputs.numpy()[5], np.modf(inputs.numpy()[5])[0])

    # all the gradients should be zero as these functions are piecewise constant:

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(10):
        for j in range(5):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[inputs], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, j, i], outputs=[out], device=device)

            tape.backward(loss=out)
            assert_np_equal(tape.gradients[inputs].numpy(), np.zeros_like(inputs.numpy()), tol=tol)
            tape.zero()


def test_interp(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 5.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_interp(
        in1: wp.array(dtype=wptype, ndim=2),
        in2: wp.array(dtype=wptype, ndim=2),
        in3: wp.array(dtype=wptype, ndim=2),
        outputs: wp.array(dtype=wptype, ndim=2),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        for i in range(10):
            outputs[0, i] = wptype(2) * wp.smoothstep(in1[0, i], in2[0, i], in3[0, i])
            outputs[1, i] = wptype(2) * wp.lerp(in1[1, i], in2[1, i], in3[1, i])

    kernel = getkernel(check_interp, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel2(wptype)

    if register_kernels:
        return

    e0 = randvals(rng, [2, 10], dtype)
    e1 = e0 + randvals(rng, [2, 10], dtype) + 0.1
    in1 = wp.array(e0, dtype=wptype, requires_grad=True, device=device)
    in2 = wp.array(e1, dtype=wptype, requires_grad=True, device=device)
    in3 = wp.array(randvals(rng, [2, 10], dtype), dtype=wptype, requires_grad=True, device=device)

    outputs = wp.zeros_like(in1)

    wp.launch(kernel, dim=1, inputs=[in1, in2, in3], outputs=[outputs], device=device)

    edge0 = in1.numpy()[0]
    edge1 = in2.numpy()[0]
    t_smoothstep = in3.numpy()[0]
    x = np.clip((t_smoothstep - edge0) / (edge1 - edge0), 0, 1)
    smoothstep_expected = 2.0 * x * x * (3 - 2 * x)

    assert_np_equal(outputs.numpy()[0], smoothstep_expected, tol=tol)

    a = in1.numpy()[1]
    b = in2.numpy()[1]
    t = in3.numpy()[1]
    assert_np_equal(outputs.numpy()[1], 2.0 * (a * (1 - t) + b * t), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(10):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2, in3], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 0, i], outputs=[out], device=device)
            tape.backward(loss=out)

            # e0 = in1
            # e1 = in2
            # t = in3

            # x = clamp((t - e0) / (e1 - e0), 0,1)
            # dx/dt = 1 / (e1 - e0) if e0 < t < e1 else 0

            # y = x * x * (3 - 2 * x)

            # y = 3 * x * x - 2 * x * x * x
            # dy/dx = 6 * ( x - x^2 )
            dydx = 6 * x * (1 - x)

            # dy/in1 = dy/dx dx/de0 de0/din1
            dxde0 = (t_smoothstep - edge1) / ((edge1 - edge0) ** 2)
            dxde0[x == 0] = 0
            dxde0[x == 1] = 0

            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[0, i] = 2.0 * dydx[i] * dxde0[i]
            assert_np_equal(tape.gradients[in1].numpy(), expected_grads, tol=tol)

            # dy/in2 = dy/dx dx/de1 de1/din2
            dxde1 = (edge0 - t_smoothstep) / ((edge1 - edge0) ** 2)
            dxde1[x == 0] = 0
            dxde1[x == 1] = 0

            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[0, i] = 2.0 * dydx[i] * dxde1[i]
            assert_np_equal(tape.gradients[in2].numpy(), expected_grads, tol=tol)

            # dy/in3 = dy/dx dx/dt dt/din3
            dxdt = 1.0 / (edge1 - edge0)
            dxdt[x == 0] = 0
            dxdt[x == 1] = 0

            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[0, i] = 2.0 * dydx[i] * dxdt[i]
            assert_np_equal(tape.gradients[in3].numpy(), expected_grads, tol=tol)
            tape.zero()

            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2, in3], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, 1, i], outputs=[out], device=device)
            tape.backward(loss=out)

            # y = a*(1-t) + b*t
            # a = in1
            # b = in2
            # t = in3

            # y = in1*( 1 - in3 ) + in2*in3

            # dy/din1 = (1-in3)
            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[1, i] = 2.0 * (1 - in3.numpy()[1, i])
            assert_np_equal(tape.gradients[in1].numpy(), expected_grads, tol=tol)

            # dy/din2 = in3
            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[1, i] = 2.0 * in3.numpy()[1, i]
            assert_np_equal(tape.gradients[in2].numpy(), expected_grads, tol=tol)

            # dy/din3 = 8*in2 - 1.5*4*in1
            expected_grads = np.zeros_like(in1.numpy())
            expected_grads[1, i] = 2.0 * (in2.numpy()[1, i] - in1.numpy()[1, i])
            assert_np_equal(tape.gradients[in3].numpy(), expected_grads, tol=tol)
            tape.zero()


def test_clamp(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-6,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_clamp(
        in1: wp.array(dtype=wptype),
        in2: wp.array(dtype=wptype),
        in3: wp.array(dtype=wptype),
        outputs: wp.array(dtype=wptype),
    ):
        for i in range(100):
            # multiply output by 2 so we've got something to backpropagate:
            outputs[i] = wptype(2) * wp.clamp(in1[i], in2[i], in3[i])

    kernel = getkernel(check_clamp, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    in1 = wp.array(randvals(rng, [100], dtype), dtype=wptype, requires_grad=True, device=device)
    starts = randvals(rng, [100], dtype)
    diffs = np.abs(randvals(rng, [100], dtype))
    in2 = wp.array(starts, dtype=wptype, requires_grad=True, device=device)
    in3 = wp.array(starts + diffs, dtype=wptype, requires_grad=True, device=device)
    outputs = wp.zeros_like(in1)

    wp.launch(kernel, dim=1, inputs=[in1, in2, in3], outputs=[outputs], device=device)

    assert_np_equal(2 * np.clip(in1.numpy(), in2.numpy(), in3.numpy()), outputs.numpy(), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(100):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[in1, in2, in3], outputs=[outputs], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, i], outputs=[out], device=device)

            tape.backward(loss=out)
            t = in1.numpy()[i]
            lower = in2.numpy()[i]
            upper = in3.numpy()[i]
            expected = np.zeros_like(in1.numpy())
            if t < lower:
                expected[i] = 2.0
                assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
                expected[i] = 0.0
                assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
                assert_np_equal(tape.gradients[in3].numpy(), expected, tol=tol)
            elif t > upper:
                expected[i] = 2.0
                assert_np_equal(tape.gradients[in3].numpy(), expected, tol=tol)
                expected[i] = 0.0
                assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
                assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
            else:
                expected[i] = 2.0
                assert_np_equal(tape.gradients[in1].numpy(), expected, tol=tol)
                expected[i] = 0.0
                assert_np_equal(tape.gradients[in2].numpy(), expected, tol=tol)
                assert_np_equal(tape.gradients[in3].numpy(), expected, tol=tol)

            tape.zero()


devices = get_test_devices()


class TestArithmetic(unittest.TestCase):
    pass


# these unary ops only make sense for signed values:
for dtype in np_signed_int_types + np_float_types:
    add_function_test_register_kernel(
        TestArithmetic, f"test_unary_ops_{dtype.__name__}", test_unary_ops, devices=devices, dtype=dtype
    )

for dtype in np_float_types:
    add_function_test_register_kernel(
        TestArithmetic, f"test_special_funcs_{dtype.__name__}", test_special_funcs, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestArithmetic,
        f"test_special_funcs_2arg_{dtype.__name__}",
        test_special_funcs_2arg,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestArithmetic, f"test_interp_{dtype.__name__}", test_interp, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestArithmetic, f"test_float_to_int_{dtype.__name__}", test_float_to_int, devices=devices, dtype=dtype
    )

for dtype in np_scalar_types:
    add_function_test_register_kernel(
        TestArithmetic, f"test_clamp_{dtype.__name__}", test_clamp, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestArithmetic, f"test_nonzero_{dtype.__name__}", test_nonzero, devices=devices, dtype=dtype
    )
    add_function_test(TestArithmetic, f"test_arrays_{dtype.__name__}", test_arrays, devices=devices, dtype=dtype)
    add_function_test_register_kernel(
        TestArithmetic, f"test_binary_ops_{dtype.__name__}", test_binary_ops, devices=devices, dtype=dtype
    )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
