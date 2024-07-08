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

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    v2_np = randvals(rng, (10, 2), dtype)
    v3_np = randvals(rng, (10, 3), dtype)
    v4_np = randvals(rng, (10, 4), dtype)
    v5_np = randvals(rng, (10, 5), dtype)

    v2 = wp.array(v2_np, dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(v3_np, dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(v4_np, dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(v5_np, dtype=vec5, requires_grad=True, device=device)

    assert_np_equal(v2.numpy(), v2_np, tol=1.0e-6)
    assert_np_equal(v3.numpy(), v3_np, tol=1.0e-6)
    assert_np_equal(v4.numpy(), v4_np, tol=1.0e-6)
    assert_np_equal(v5.numpy(), v5_np, tol=1.0e-6)

    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)

    v2 = wp.array(v2_np, dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(v3_np, dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(v4_np, dtype=vec4, requires_grad=True, device=device)

    assert_np_equal(v2.numpy(), v2_np, tol=1.0e-6)
    assert_np_equal(v3.numpy(), v3_np, tol=1.0e-6)
    assert_np_equal(v4.numpy(), v4_np, tol=1.0e-6)


def test_components(test, device, dtype):
    # test accessing vector components from Python - this is especially important
    # for float16, which requires special handling internally

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(length=3, dtype=wptype)

    v = vec3(1, 2, 3)

    # test __getitem__ for individual components
    test.assertEqual(v[0], 1)
    test.assertEqual(v[1], 2)
    test.assertEqual(v[2], 3)

    # test __getitem__ for slices
    s = v[:]
    test.assertEqual(s[0], 1)
    test.assertEqual(s[1], 2)
    test.assertEqual(s[2], 3)

    s = v[1:]
    test.assertEqual(s[0], 2)
    test.assertEqual(s[1], 3)

    s = v[:2]
    test.assertEqual(s[0], 1)
    test.assertEqual(s[1], 2)

    s = v[::2]
    test.assertEqual(s[0], 1)
    test.assertEqual(s[1], 3)

    # test __setitem__ for individual components
    v[0] = 4
    v[1] = 5
    v[2] = 6
    test.assertEqual(v[0], 4)
    test.assertEqual(v[1], 5)
    test.assertEqual(v[2], 6)

    # test __setitem__ for slices
    v[:] = [7, 8, 9]
    test.assertEqual(v[0], 7)
    test.assertEqual(v[1], 8)
    test.assertEqual(v[2], 9)

    v[1:] = [10, 11]
    test.assertEqual(v[0], 7)
    test.assertEqual(v[1], 10)
    test.assertEqual(v[2], 11)

    v[:2] = [12, 13]
    test.assertEqual(v[0], 12)
    test.assertEqual(v[1], 13)
    test.assertEqual(v[2], 11)

    v[::2] = [14, 15]
    test.assertEqual(v[0], 14)
    test.assertEqual(v[1], 13)
    test.assertEqual(v[2], 15)


def test_py_arithmetic_ops(test, device, dtype):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def make_vec(*args):
        if wptype in wp.types.int_types:
            # Cast to the correct integer type to simulate wrapping.
            return tuple(wptype._type_(x).value for x in args)

        return args

    vec_cls = wp.vec(3, wptype)

    v = vec_cls(1, -2, 3)
    test.assertSequenceEqual(+v, make_vec(1, -2, 3))
    test.assertSequenceEqual(-v, make_vec(-1, 2, -3))
    test.assertSequenceEqual(v + vec_cls(5, 5, 5), make_vec(6, 3, 8))
    test.assertSequenceEqual(v - vec_cls(5, 5, 5), make_vec(-4, -7, -2))

    v = vec_cls(2, 4, 6)
    test.assertSequenceEqual(v * wptype(2), make_vec(4, 8, 12))
    test.assertSequenceEqual(wptype(2) * v, make_vec(4, 8, 12))
    test.assertSequenceEqual(v / wptype(2), make_vec(1, 2, 3))
    test.assertSequenceEqual(wptype(24) / v, make_vec(12, 6, 4))


def test_constructors(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_scalar_constructor(
        input: wp.array(dtype=wptype),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2result = vec2(input[0])
        v3result = vec3(input[0])
        v4result = vec4(input[0])
        v5result = vec5(input[0])

        v2[0] = v2result
        v3[0] = v3result
        v4[0] = v4result
        v5[0] = v5result

        # multiply outputs by 2 so we've got something to backpropagate
        v20[0] = wptype(2) * v2result[0]
        v21[0] = wptype(2) * v2result[1]

        v30[0] = wptype(2) * v3result[0]
        v31[0] = wptype(2) * v3result[1]
        v32[0] = wptype(2) * v3result[2]

        v40[0] = wptype(2) * v4result[0]
        v41[0] = wptype(2) * v4result[1]
        v42[0] = wptype(2) * v4result[2]
        v43[0] = wptype(2) * v4result[3]

        v50[0] = wptype(2) * v5result[0]
        v51[0] = wptype(2) * v5result[1]
        v52[0] = wptype(2) * v5result[2]
        v53[0] = wptype(2) * v5result[3]
        v54[0] = wptype(2) * v5result[4]

    def check_vector_constructors(
        input: wp.array(dtype=wptype),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2result = vec2(input[0], input[1])
        v3result = vec3(input[2], input[3], input[4])
        v4result = vec4(input[5], input[6], input[7], input[8])
        v5result = vec5(input[9], input[10], input[11], input[12], input[13])

        v2[0] = v2result
        v3[0] = v3result
        v4[0] = v4result
        v5[0] = v5result

        # multiply the output by 2 so we've got something to backpropagate:
        v20[0] = wptype(2) * v2result[0]
        v21[0] = wptype(2) * v2result[1]

        v30[0] = wptype(2) * v3result[0]
        v31[0] = wptype(2) * v3result[1]
        v32[0] = wptype(2) * v3result[2]

        v40[0] = wptype(2) * v4result[0]
        v41[0] = wptype(2) * v4result[1]
        v42[0] = wptype(2) * v4result[2]
        v43[0] = wptype(2) * v4result[3]

        v50[0] = wptype(2) * v5result[0]
        v51[0] = wptype(2) * v5result[1]
        v52[0] = wptype(2) * v5result[2]
        v53[0] = wptype(2) * v5result[3]
        v54[0] = wptype(2) * v5result[4]

    vec_kernel = getkernel(check_vector_constructors, suffix=dtype.__name__)
    kernel = getkernel(check_scalar_constructor, suffix=dtype.__name__)

    if register_kernels:
        return

    input = wp.array(randvals(rng, [1], dtype), requires_grad=True, device=device)
    v2 = wp.zeros(1, dtype=vec2, device=device)
    v3 = wp.zeros(1, dtype=vec3, device=device)
    v4 = wp.zeros(1, dtype=vec4, device=device)
    v5 = wp.zeros(1, dtype=vec5, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[input],
            outputs=[v2, v3, v4, v5, v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54],
            device=device,
        )

    if dtype in np_float_types:
        for l in [v20, v21]:
            tape.backward(loss=l)
            test.assertEqual(tape.gradients[input].numpy()[0], 2.0)
            tape.zero()

        for l in [v30, v31, v32]:
            tape.backward(loss=l)
            test.assertEqual(tape.gradients[input].numpy()[0], 2.0)
            tape.zero()

        for l in [v40, v41, v42, v43]:
            tape.backward(loss=l)
            test.assertEqual(tape.gradients[input].numpy()[0], 2.0)
            tape.zero()

        for l in [v50, v51, v52, v53, v54]:
            tape.backward(loss=l)
            test.assertEqual(tape.gradients[input].numpy()[0], 2.0)
            tape.zero()

    val = input.numpy()[0]
    assert_np_equal(v2.numpy()[0], np.array([val, val]), tol=1.0e-6)
    assert_np_equal(v3.numpy()[0], np.array([val, val, val]), tol=1.0e-6)
    assert_np_equal(v4.numpy()[0], np.array([val, val, val, val]), tol=1.0e-6)
    assert_np_equal(v5.numpy()[0], np.array([val, val, val, val, val]), tol=1.0e-6)

    assert_np_equal(v20.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v21.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v30.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v31.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v32.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v40.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v41.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v42.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v43.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v50.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v51.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v52.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v53.numpy()[0], 2 * val, tol=1.0e-6)
    assert_np_equal(v54.numpy()[0], 2 * val, tol=1.0e-6)

    input = wp.array(randvals(rng, [14], dtype), requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            vec_kernel,
            dim=1,
            inputs=[input],
            outputs=[v2, v3, v4, v5, v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54],
            device=device,
        )

    if dtype in np_float_types:
        for i, l in enumerate([v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54]):
            tape.backward(loss=l)
            grad = tape.gradients[input].numpy()
            expected_grad = np.zeros_like(grad)
            expected_grad[i] = 2
            assert_np_equal(grad, expected_grad, tol=tol)
            tape.zero()

    assert_np_equal(v2.numpy()[0, 0], input.numpy()[0], tol=tol)
    assert_np_equal(v2.numpy()[0, 1], input.numpy()[1], tol=tol)
    assert_np_equal(v3.numpy()[0, 0], input.numpy()[2], tol=tol)
    assert_np_equal(v3.numpy()[0, 1], input.numpy()[3], tol=tol)
    assert_np_equal(v3.numpy()[0, 2], input.numpy()[4], tol=tol)
    assert_np_equal(v4.numpy()[0, 0], input.numpy()[5], tol=tol)
    assert_np_equal(v4.numpy()[0, 1], input.numpy()[6], tol=tol)
    assert_np_equal(v4.numpy()[0, 2], input.numpy()[7], tol=tol)
    assert_np_equal(v4.numpy()[0, 3], input.numpy()[8], tol=tol)
    assert_np_equal(v5.numpy()[0, 0], input.numpy()[9], tol=tol)
    assert_np_equal(v5.numpy()[0, 1], input.numpy()[10], tol=tol)
    assert_np_equal(v5.numpy()[0, 2], input.numpy()[11], tol=tol)
    assert_np_equal(v5.numpy()[0, 3], input.numpy()[12], tol=tol)
    assert_np_equal(v5.numpy()[0, 4], input.numpy()[13], tol=tol)

    assert_np_equal(v20.numpy()[0], 2 * input.numpy()[0], tol=tol)
    assert_np_equal(v21.numpy()[0], 2 * input.numpy()[1], tol=tol)
    assert_np_equal(v30.numpy()[0], 2 * input.numpy()[2], tol=tol)
    assert_np_equal(v31.numpy()[0], 2 * input.numpy()[3], tol=tol)
    assert_np_equal(v32.numpy()[0], 2 * input.numpy()[4], tol=tol)
    assert_np_equal(v40.numpy()[0], 2 * input.numpy()[5], tol=tol)
    assert_np_equal(v41.numpy()[0], 2 * input.numpy()[6], tol=tol)
    assert_np_equal(v42.numpy()[0], 2 * input.numpy()[7], tol=tol)
    assert_np_equal(v43.numpy()[0], 2 * input.numpy()[8], tol=tol)
    assert_np_equal(v50.numpy()[0], 2 * input.numpy()[9], tol=tol)
    assert_np_equal(v51.numpy()[0], 2 * input.numpy()[10], tol=tol)
    assert_np_equal(v52.numpy()[0], 2 * input.numpy()[11], tol=tol)
    assert_np_equal(v53.numpy()[0], 2 * input.numpy()[12], tol=tol)
    assert_np_equal(v54.numpy()[0], 2 * input.numpy()[13], tol=tol)


def test_anon_type_instance(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def check_scalar_init(
        input: wp.array(dtype=wptype),
        output: wp.array(dtype=wptype),
    ):
        v2result = wp.vector(input[0], length=2)
        v3result = wp.vector(input[1], length=3)
        v4result = wp.vector(input[2], length=4)
        v5result = wp.vector(input[3], length=5)

        idx = 0
        for i in range(2):
            output[idx] = wptype(2) * v2result[i]
            idx = idx + 1
        for i in range(3):
            output[idx] = wptype(2) * v3result[i]
            idx = idx + 1
        for i in range(4):
            output[idx] = wptype(2) * v4result[i]
            idx = idx + 1
        for i in range(5):
            output[idx] = wptype(2) * v5result[i]
            idx = idx + 1

    def check_component_init(
        input: wp.array(dtype=wptype),
        output: wp.array(dtype=wptype),
    ):
        v2result = wp.vector(input[0], input[1])
        v3result = wp.vector(input[2], input[3], input[4])
        v4result = wp.vector(input[5], input[6], input[7], input[8])
        v5result = wp.vector(input[9], input[10], input[11], input[12], input[13])

        idx = 0
        for i in range(2):
            output[idx] = wptype(2) * v2result[i]
            idx = idx + 1
        for i in range(3):
            output[idx] = wptype(2) * v3result[i]
            idx = idx + 1
        for i in range(4):
            output[idx] = wptype(2) * v4result[i]
            idx = idx + 1
        for i in range(5):
            output[idx] = wptype(2) * v5result[i]
            idx = idx + 1

    scalar_kernel = getkernel(check_scalar_init, suffix=dtype.__name__)
    component_kernel = getkernel(check_component_init, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    input = wp.array(randvals(rng, [4], dtype), requires_grad=True, device=device)
    output = wp.zeros(2 + 3 + 4 + 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(scalar_kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy()[:2], 2 * np.array([input.numpy()[0]] * 2), tol=1.0e-6)
    assert_np_equal(output.numpy()[2:5], 2 * np.array([input.numpy()[1]] * 3), tol=1.0e-6)
    assert_np_equal(output.numpy()[5:9], 2 * np.array([input.numpy()[2]] * 4), tol=1.0e-6)
    assert_np_equal(output.numpy()[9:], 2 * np.array([input.numpy()[3]] * 5), tol=1.0e-6)

    if dtype in np_float_types:
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for i in range(len(output)):
            tape = wp.Tape()
            with tape:
                wp.launch(scalar_kernel, dim=1, inputs=[input], outputs=[output], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[output, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(input.numpy())
            if i < 2:
                expected[0] = 2
            elif i < 5:
                expected[1] = 2
            elif i < 9:
                expected[2] = 2
            else:
                expected[3] = 2

            assert_np_equal(tape.gradients[input].numpy(), expected, tol=tol)

            tape.reset()
            tape.zero()

    input = wp.array(randvals(rng, [2 + 3 + 4 + 5], dtype), requires_grad=True, device=device)
    output = wp.zeros(2 + 3 + 4 + 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(component_kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy(), 2 * input.numpy(), tol=1.0e-6)

    if dtype in np_float_types:
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for i in range(len(output)):
            tape = wp.Tape()
            with tape:
                wp.launch(component_kernel, dim=1, inputs=[input], outputs=[output], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[output, i], outputs=[out], device=device)

            tape.backward(loss=out)
            expected = np.zeros_like(input.numpy())
            expected[i] = 2

            assert_np_equal(tape.gradients[input].numpy(), expected, tol=tol)

            tape.reset()
            tape.zero()


def test_indexing(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_indexing(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        v20[0] = wptype(2) * v2[0][0]
        v21[0] = wptype(2) * v2[0][1]

        v30[0] = wptype(2) * v3[0][0]
        v31[0] = wptype(2) * v3[0][1]
        v32[0] = wptype(2) * v3[0][2]

        v40[0] = wptype(2) * v4[0][0]
        v41[0] = wptype(2) * v4[0][1]
        v42[0] = wptype(2) * v4[0][2]
        v43[0] = wptype(2) * v4[0][3]

        v50[0] = wptype(2) * v5[0][0]
        v51[0] = wptype(2) * v5[0][1]
        v52[0] = wptype(2) * v5[0][2]
        v53[0] = wptype(2) * v5[0][3]
        v54[0] = wptype(2) * v5[0][4]

    kernel = getkernel(check_indexing, suffix=dtype.__name__)

    if register_kernels:
        return

    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[v2, v3, v4, v5],
            outputs=[v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54],
            device=device,
        )

    if dtype in np_float_types:
        for i, l in enumerate([v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54]):
            tape.backward(loss=l)
            allgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [v2, v3, v4, v5]])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = 2
            assert_np_equal(allgrads, expected_grads, tol=tol)
            tape.zero()

    assert_np_equal(v20.numpy()[0], 2.0 * v2.numpy()[0, 0], tol=tol)
    assert_np_equal(v21.numpy()[0], 2.0 * v2.numpy()[0, 1], tol=tol)
    assert_np_equal(v30.numpy()[0], 2.0 * v3.numpy()[0, 0], tol=tol)
    assert_np_equal(v31.numpy()[0], 2.0 * v3.numpy()[0, 1], tol=tol)
    assert_np_equal(v32.numpy()[0], 2.0 * v3.numpy()[0, 2], tol=tol)
    assert_np_equal(v40.numpy()[0], 2.0 * v4.numpy()[0, 0], tol=tol)
    assert_np_equal(v41.numpy()[0], 2.0 * v4.numpy()[0, 1], tol=tol)
    assert_np_equal(v42.numpy()[0], 2.0 * v4.numpy()[0, 2], tol=tol)
    assert_np_equal(v43.numpy()[0], 2.0 * v4.numpy()[0, 3], tol=tol)
    assert_np_equal(v50.numpy()[0], 2.0 * v5.numpy()[0, 0], tol=tol)
    assert_np_equal(v51.numpy()[0], 2.0 * v5.numpy()[0, 1], tol=tol)
    assert_np_equal(v52.numpy()[0], 2.0 * v5.numpy()[0, 2], tol=tol)
    assert_np_equal(v53.numpy()[0], 2.0 * v5.numpy()[0, 3], tol=tol)
    assert_np_equal(v54.numpy()[0], 2.0 * v5.numpy()[0, 4], tol=tol)


def test_equality(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_unsigned_equality(
        v20: wp.array(dtype=vec2),
        v21: wp.array(dtype=vec2),
        v22: wp.array(dtype=vec2),
        v30: wp.array(dtype=vec3),
        v40: wp.array(dtype=vec4),
        v50: wp.array(dtype=vec5),
    ):
        wp.expect_eq(v20[0], v20[0])
        wp.expect_neq(v21[0], v20[0])
        wp.expect_neq(v22[0], v20[0])
        wp.expect_eq(v30[0], v30[0])
        wp.expect_eq(v40[0], v40[0])
        wp.expect_eq(v50[0], v50[0])

    def check_signed_equality(
        v30: wp.array(dtype=vec3),
        v31: wp.array(dtype=vec3),
        v32: wp.array(dtype=vec3),
        v33: wp.array(dtype=vec3),
        v40: wp.array(dtype=vec4),
        v41: wp.array(dtype=vec4),
        v42: wp.array(dtype=vec4),
        v43: wp.array(dtype=vec4),
        v44: wp.array(dtype=vec4),
        v50: wp.array(dtype=vec5),
        v51: wp.array(dtype=vec5),
        v52: wp.array(dtype=vec5),
        v53: wp.array(dtype=vec5),
        v54: wp.array(dtype=vec5),
        v55: wp.array(dtype=vec5),
    ):
        wp.expect_neq(v31[0], v30[0])
        wp.expect_neq(v32[0], v30[0])
        wp.expect_neq(v33[0], v30[0])
        wp.expect_neq(v41[0], v40[0])
        wp.expect_neq(v42[0], v40[0])
        wp.expect_neq(v43[0], v40[0])
        wp.expect_neq(v44[0], v40[0])
        wp.expect_neq(v51[0], v50[0])
        wp.expect_neq(v52[0], v50[0])
        wp.expect_neq(v53[0], v50[0])
        wp.expect_neq(v54[0], v50[0])
        wp.expect_neq(v55[0], v50[0])

    unsigned_kernel = getkernel(check_unsigned_equality, suffix=dtype.__name__)
    signed_kernel = getkernel(check_signed_equality, suffix=dtype.__name__)

    if register_kernels:
        return

    v20 = wp.array([1.0, 2.0], dtype=vec2, requires_grad=True, device=device)
    v21 = wp.array([1.0, 3.0], dtype=vec2, requires_grad=True, device=device)
    v22 = wp.array([3.0, 2.0], dtype=vec2, requires_grad=True, device=device)

    v30 = wp.array([1.0, 2.0, 3.0], dtype=vec3, requires_grad=True, device=device)
    v40 = wp.array([1.0, 2.0, 3.0, 4.0], dtype=vec4, requires_grad=True, device=device)
    v50 = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=vec5, requires_grad=True, device=device)

    wp.launch(
        unsigned_kernel,
        dim=1,
        inputs=[
            v20,
            v21,
            v22,
            v30,
            v40,
            v50,
        ],
        outputs=[],
        device=device,
    )

    if dtype not in np_unsigned_int_types:
        v31 = wp.array([-1.0, 2.0, 3.0], dtype=vec3, requires_grad=True, device=device)
        v32 = wp.array([1.0, -2.0, 3.0], dtype=vec3, requires_grad=True, device=device)
        v33 = wp.array([1.0, 2.0, -3.0], dtype=vec3, requires_grad=True, device=device)

        v41 = wp.array([-1.0, 2.0, 3.0, 4.0], dtype=vec4, requires_grad=True, device=device)
        v42 = wp.array([1.0, -2.0, 3.0, 4.0], dtype=vec4, requires_grad=True, device=device)
        v43 = wp.array([1.0, 2.0, -3.0, 4.0], dtype=vec4, requires_grad=True, device=device)
        v44 = wp.array([1.0, 2.0, 3.0, -4.0], dtype=vec4, requires_grad=True, device=device)

        v51 = wp.array([-1.0, 2.0, 3.0, 4.0, 5.0], dtype=vec5, requires_grad=True, device=device)
        v52 = wp.array([1.0, -2.0, 3.0, 4.0, 5.0], dtype=vec5, requires_grad=True, device=device)
        v53 = wp.array([1.0, 2.0, -3.0, 4.0, 5.0], dtype=vec5, requires_grad=True, device=device)
        v54 = wp.array([1.0, 2.0, 3.0, -4.0, 5.0], dtype=vec5, requires_grad=True, device=device)
        v55 = wp.array([1.0, 2.0, 3.0, 4.0, -5.0], dtype=vec5, requires_grad=True, device=device)

        wp.launch(
            signed_kernel,
            dim=1,
            inputs=[
                v30,
                v31,
                v32,
                v33,
                v40,
                v41,
                v42,
                v43,
                v44,
                v50,
                v51,
                v52,
                v53,
                v54,
                v55,
            ],
            outputs=[],
            device=device,
        )


def test_scalar_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_mul(
        s: wp.array(dtype=wptype),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2result = s[0] * v2[0]
        v3result = s[0] * v3[0]
        v4result = s[0] * v4[0]
        v5result = s[0] * v5[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        v20[0] = wptype(2) * v2result[0]
        v21[0] = wptype(2) * v2result[1]

        v30[0] = wptype(2) * v3result[0]
        v31[0] = wptype(2) * v3result[1]
        v32[0] = wptype(2) * v3result[2]

        v40[0] = wptype(2) * v4result[0]
        v41[0] = wptype(2) * v4result[1]
        v42[0] = wptype(2) * v4result[2]
        v43[0] = wptype(2) * v4result[3]

        v50[0] = wptype(2) * v5result[0]
        v51[0] = wptype(2) * v5result[1]
        v52[0] = wptype(2) * v5result[2]
        v53[0] = wptype(2) * v5result[3]
        v54[0] = wptype(2) * v5result[4]

    kernel = getkernel(check_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(randvals(rng, [1], dtype), requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                s,
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=[v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54],
            device=device,
        )

    assert_np_equal(v20.numpy()[0], 2 * s.numpy()[0] * v2.numpy()[0, 0], tol=tol)
    assert_np_equal(v21.numpy()[0], 2 * s.numpy()[0] * v2.numpy()[0, 1], tol=tol)

    assert_np_equal(v30.numpy()[0], 2 * s.numpy()[0] * v3.numpy()[0, 0], tol=10 * tol)
    assert_np_equal(v31.numpy()[0], 2 * s.numpy()[0] * v3.numpy()[0, 1], tol=10 * tol)
    assert_np_equal(v32.numpy()[0], 2 * s.numpy()[0] * v3.numpy()[0, 2], tol=10 * tol)

    assert_np_equal(v40.numpy()[0], 2 * s.numpy()[0] * v4.numpy()[0, 0], tol=10 * tol)
    assert_np_equal(v41.numpy()[0], 2 * s.numpy()[0] * v4.numpy()[0, 1], tol=10 * tol)
    assert_np_equal(v42.numpy()[0], 2 * s.numpy()[0] * v4.numpy()[0, 2], tol=10 * tol)
    assert_np_equal(v43.numpy()[0], 2 * s.numpy()[0] * v4.numpy()[0, 3], tol=10 * tol)

    assert_np_equal(v50.numpy()[0], 2 * s.numpy()[0] * v5.numpy()[0, 0], tol=10 * tol)
    assert_np_equal(v51.numpy()[0], 2 * s.numpy()[0] * v5.numpy()[0, 1], tol=10 * tol)
    assert_np_equal(v52.numpy()[0], 2 * s.numpy()[0] * v5.numpy()[0, 2], tol=10 * tol)
    assert_np_equal(v53.numpy()[0], 2 * s.numpy()[0] * v5.numpy()[0, 3], tol=10 * tol)
    assert_np_equal(v54.numpy()[0], 2 * s.numpy()[0] * v5.numpy()[0, 4], tol=10 * tol)

    incmps = np.concatenate([v.numpy()[0] for v in [v2, v3, v4, v5]])

    if dtype in np_float_types:
        for i, l in enumerate([v20, v21, v30, v31, v32, v40, v41, v42, v43]):
            tape.backward(loss=l)
            sgrad = tape.gradients[s].numpy()[0]
            assert_np_equal(sgrad, 2 * incmps[i], tol=10 * tol)
            allgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [v2, v3, v4]])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = s.numpy()[0] * 2
            assert_np_equal(allgrads, expected_grads, tol=10 * tol)
            tape.zero()


def test_scalar_multiplication_rightmul(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_rightmul(
        s: wp.array(dtype=wptype),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2result = v2[0] * s[0]
        v3result = v3[0] * s[0]
        v4result = v4[0] * s[0]
        v5result = v5[0] * s[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        v20[0] = wptype(2) * v2result[0]
        v21[0] = wptype(2) * v2result[1]

        v30[0] = wptype(2) * v3result[0]
        v31[0] = wptype(2) * v3result[1]
        v32[0] = wptype(2) * v3result[2]

        v40[0] = wptype(2) * v4result[0]
        v41[0] = wptype(2) * v4result[1]
        v42[0] = wptype(2) * v4result[2]
        v43[0] = wptype(2) * v4result[3]

        v50[0] = wptype(2) * v5result[0]
        v51[0] = wptype(2) * v5result[1]
        v52[0] = wptype(2) * v5result[2]
        v53[0] = wptype(2) * v5result[3]
        v54[0] = wptype(2) * v5result[4]

    kernel = getkernel(check_rightmul, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(randvals(rng, [1], dtype), requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                s,
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=[v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54],
            device=device,
        )

    assert_np_equal(v20.numpy()[0], 2 * s.numpy()[0] * v2.numpy()[0, 0], tol=tol)
    assert_np_equal(v21.numpy()[0], 2 * s.numpy()[0] * v2.numpy()[0, 1], tol=tol)

    assert_np_equal(v30.numpy()[0], 2 * s.numpy()[0] * v3.numpy()[0, 0], tol=10 * tol)
    assert_np_equal(v31.numpy()[0], 2 * s.numpy()[0] * v3.numpy()[0, 1], tol=10 * tol)
    assert_np_equal(v32.numpy()[0], 2 * s.numpy()[0] * v3.numpy()[0, 2], tol=10 * tol)

    assert_np_equal(v40.numpy()[0], 2 * s.numpy()[0] * v4.numpy()[0, 0], tol=10 * tol)
    assert_np_equal(v41.numpy()[0], 2 * s.numpy()[0] * v4.numpy()[0, 1], tol=10 * tol)
    assert_np_equal(v42.numpy()[0], 2 * s.numpy()[0] * v4.numpy()[0, 2], tol=10 * tol)
    assert_np_equal(v43.numpy()[0], 2 * s.numpy()[0] * v4.numpy()[0, 3], tol=10 * tol)

    assert_np_equal(v50.numpy()[0], 2 * s.numpy()[0] * v5.numpy()[0, 0], tol=10 * tol)
    assert_np_equal(v51.numpy()[0], 2 * s.numpy()[0] * v5.numpy()[0, 1], tol=10 * tol)
    assert_np_equal(v52.numpy()[0], 2 * s.numpy()[0] * v5.numpy()[0, 2], tol=10 * tol)
    assert_np_equal(v53.numpy()[0], 2 * s.numpy()[0] * v5.numpy()[0, 3], tol=10 * tol)
    assert_np_equal(v54.numpy()[0], 2 * s.numpy()[0] * v5.numpy()[0, 4], tol=10 * tol)

    incmps = np.concatenate([v.numpy()[0] for v in [v2, v3, v4, v5]])

    if dtype in np_float_types:
        for i, l in enumerate([v20, v21, v30, v31, v32, v40, v41, v42, v43]):
            tape.backward(loss=l)
            sgrad = tape.gradients[s].numpy()[0]
            assert_np_equal(sgrad, 2 * incmps[i], tol=10 * tol)
            allgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [v2, v3, v4]])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = s.numpy()[0] * 2
            assert_np_equal(allgrads, expected_grads, tol=10 * tol)
            tape.zero()


def test_cw_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_cw_mul(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2result = wp.cw_mul(s2[0], v2[0])
        v3result = wp.cw_mul(s3[0], v3[0])
        v4result = wp.cw_mul(s4[0], v4[0])
        v5result = wp.cw_mul(s5[0], v5[0])

        v20[0] = wptype(2) * v2result[0]
        v21[0] = wptype(2) * v2result[1]

        v30[0] = wptype(2) * v3result[0]
        v31[0] = wptype(2) * v3result[1]
        v32[0] = wptype(2) * v3result[2]

        v40[0] = wptype(2) * v4result[0]
        v41[0] = wptype(2) * v4result[1]
        v42[0] = wptype(2) * v4result[2]
        v43[0] = wptype(2) * v4result[3]

        v50[0] = wptype(2) * v5result[0]
        v51[0] = wptype(2) * v5result[1]
        v52[0] = wptype(2) * v5result[2]
        v53[0] = wptype(2) * v5result[3]
        v54[0] = wptype(2) * v5result[4]

    kernel = getkernel(check_cw_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                s2,
                s3,
                s4,
                s5,
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=[v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54],
            device=device,
        )

    assert_np_equal(v20.numpy()[0], 2 * s2.numpy()[0, 0] * v2.numpy()[0, 0], tol=10 * tol)
    assert_np_equal(v21.numpy()[0], 2 * s2.numpy()[0, 1] * v2.numpy()[0, 1], tol=10 * tol)

    assert_np_equal(v30.numpy()[0], 2 * s3.numpy()[0, 0] * v3.numpy()[0, 0], tol=10 * tol)
    assert_np_equal(v31.numpy()[0], 2 * s3.numpy()[0, 1] * v3.numpy()[0, 1], tol=10 * tol)
    assert_np_equal(v32.numpy()[0], 2 * s3.numpy()[0, 2] * v3.numpy()[0, 2], tol=10 * tol)

    assert_np_equal(v40.numpy()[0], 2 * s4.numpy()[0, 0] * v4.numpy()[0, 0], tol=10 * tol)
    assert_np_equal(v41.numpy()[0], 2 * s4.numpy()[0, 1] * v4.numpy()[0, 1], tol=10 * tol)
    assert_np_equal(v42.numpy()[0], 2 * s4.numpy()[0, 2] * v4.numpy()[0, 2], tol=10 * tol)
    assert_np_equal(v43.numpy()[0], 2 * s4.numpy()[0, 3] * v4.numpy()[0, 3], tol=10 * tol)

    assert_np_equal(v50.numpy()[0], 2 * s5.numpy()[0, 0] * v5.numpy()[0, 0], tol=10 * tol)
    assert_np_equal(v51.numpy()[0], 2 * s5.numpy()[0, 1] * v5.numpy()[0, 1], tol=10 * tol)
    assert_np_equal(v52.numpy()[0], 2 * s5.numpy()[0, 2] * v5.numpy()[0, 2], tol=10 * tol)
    assert_np_equal(v53.numpy()[0], 2 * s5.numpy()[0, 3] * v5.numpy()[0, 3], tol=10 * tol)
    assert_np_equal(v54.numpy()[0], 2 * s5.numpy()[0, 4] * v5.numpy()[0, 4], tol=10 * tol)

    incmps = np.concatenate([v.numpy()[0] for v in [v2, v3, v4, v5]])
    scmps = np.concatenate([v.numpy()[0] for v in [s2, s3, s4, s5]])

    if dtype in np_float_types:
        for i, l in enumerate([v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54]):
            tape.backward(loss=l)
            sgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [s2, s3, s4, s5]])
            expected_grads = np.zeros_like(sgrads)
            expected_grads[i] = incmps[i] * 2
            assert_np_equal(sgrads, expected_grads, tol=10 * tol)

            allgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [v2, v3, v4, v5]])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = scmps[i] * 2
            assert_np_equal(allgrads, expected_grads, tol=10 * tol)

            tape.zero()


def test_scalar_division(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_div(
        s: wp.array(dtype=wptype),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2result = v2[0] / s[0]
        v3result = v3[0] / s[0]
        v4result = v4[0] / s[0]
        v5result = v5[0] / s[0]

        v20[0] = wptype(2) * v2result[0]
        v21[0] = wptype(2) * v2result[1]

        v30[0] = wptype(2) * v3result[0]
        v31[0] = wptype(2) * v3result[1]
        v32[0] = wptype(2) * v3result[2]

        v40[0] = wptype(2) * v4result[0]
        v41[0] = wptype(2) * v4result[1]
        v42[0] = wptype(2) * v4result[2]
        v43[0] = wptype(2) * v4result[3]

        v50[0] = wptype(2) * v5result[0]
        v51[0] = wptype(2) * v5result[1]
        v52[0] = wptype(2) * v5result[2]
        v53[0] = wptype(2) * v5result[3]
        v54[0] = wptype(2) * v5result[4]

    kernel = getkernel(check_div, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(randvals(rng, [1], dtype), requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                s,
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=[v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54],
            device=device,
        )

    if dtype in np_int_types:
        assert_np_equal(v20.numpy()[0], 2 * (v2.numpy()[0, 0] // (s.numpy()[0])), tol=tol)
        assert_np_equal(v21.numpy()[0], 2 * (v2.numpy()[0, 1] // (s.numpy()[0])), tol=tol)

        assert_np_equal(v30.numpy()[0], 2 * (v3.numpy()[0, 0] // (s.numpy()[0])), tol=10 * tol)
        assert_np_equal(v31.numpy()[0], 2 * (v3.numpy()[0, 1] // (s.numpy()[0])), tol=10 * tol)
        assert_np_equal(v32.numpy()[0], 2 * (v3.numpy()[0, 2] // (s.numpy()[0])), tol=10 * tol)

        assert_np_equal(v40.numpy()[0], 2 * (v4.numpy()[0, 0] // (s.numpy()[0])), tol=10 * tol)
        assert_np_equal(v41.numpy()[0], 2 * (v4.numpy()[0, 1] // (s.numpy()[0])), tol=10 * tol)
        assert_np_equal(v42.numpy()[0], 2 * (v4.numpy()[0, 2] // (s.numpy()[0])), tol=10 * tol)
        assert_np_equal(v43.numpy()[0], 2 * (v4.numpy()[0, 3] // (s.numpy()[0])), tol=10 * tol)

        assert_np_equal(v50.numpy()[0], 2 * (v5.numpy()[0, 0] // (s.numpy()[0])), tol=10 * tol)
        assert_np_equal(v51.numpy()[0], 2 * (v5.numpy()[0, 1] // (s.numpy()[0])), tol=10 * tol)
        assert_np_equal(v52.numpy()[0], 2 * (v5.numpy()[0, 2] // (s.numpy()[0])), tol=10 * tol)
        assert_np_equal(v53.numpy()[0], 2 * (v5.numpy()[0, 3] // (s.numpy()[0])), tol=10 * tol)
        assert_np_equal(v54.numpy()[0], 2 * (v5.numpy()[0, 4] // (s.numpy()[0])), tol=10 * tol)

    else:
        assert_np_equal(v20.numpy()[0], 2 * v2.numpy()[0, 0] / (s.numpy()[0]), tol=tol)
        assert_np_equal(v21.numpy()[0], 2 * v2.numpy()[0, 1] / (s.numpy()[0]), tol=tol)

        assert_np_equal(v30.numpy()[0], 2 * v3.numpy()[0, 0] / (s.numpy()[0]), tol=10 * tol)
        assert_np_equal(v31.numpy()[0], 2 * v3.numpy()[0, 1] / (s.numpy()[0]), tol=10 * tol)
        assert_np_equal(v32.numpy()[0], 2 * v3.numpy()[0, 2] / (s.numpy()[0]), tol=10 * tol)

        assert_np_equal(v40.numpy()[0], 2 * v4.numpy()[0, 0] / (s.numpy()[0]), tol=10 * tol)
        assert_np_equal(v41.numpy()[0], 2 * v4.numpy()[0, 1] / (s.numpy()[0]), tol=10 * tol)
        assert_np_equal(v42.numpy()[0], 2 * v4.numpy()[0, 2] / (s.numpy()[0]), tol=10 * tol)
        assert_np_equal(v43.numpy()[0], 2 * v4.numpy()[0, 3] / (s.numpy()[0]), tol=10 * tol)

        assert_np_equal(v50.numpy()[0], 2 * v5.numpy()[0, 0] / (s.numpy()[0]), tol=10 * tol)
        assert_np_equal(v51.numpy()[0], 2 * v5.numpy()[0, 1] / (s.numpy()[0]), tol=10 * tol)
        assert_np_equal(v52.numpy()[0], 2 * v5.numpy()[0, 2] / (s.numpy()[0]), tol=10 * tol)
        assert_np_equal(v53.numpy()[0], 2 * v5.numpy()[0, 3] / (s.numpy()[0]), tol=10 * tol)
        assert_np_equal(v54.numpy()[0], 2 * v5.numpy()[0, 4] / (s.numpy()[0]), tol=10 * tol)

    incmps = np.concatenate([v.numpy()[0] for v in [v2, v3, v4, v5]])

    if dtype in np_float_types:
        for i, l in enumerate([v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54]):
            tape.backward(loss=l)
            sgrad = tape.gradients[s].numpy()[0]

            # d/ds v/s = -v/s^2
            assert_np_equal(sgrad, -2 * incmps[i] / (s.numpy()[0] * s.numpy()[0]), tol=10 * tol)

            allgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [v2, v3, v4, v5]])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = 2 / s.numpy()[0]

            # d/dv v/s = 1/s
            assert_np_equal(allgrads, expected_grads, tol=tol)
            tape.zero()


def test_cw_division(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_cw_div(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2result = wp.cw_div(v2[0], s2[0])
        v3result = wp.cw_div(v3[0], s3[0])
        v4result = wp.cw_div(v4[0], s4[0])
        v5result = wp.cw_div(v5[0], s5[0])

        v20[0] = wptype(2) * v2result[0]
        v21[0] = wptype(2) * v2result[1]

        v30[0] = wptype(2) * v3result[0]
        v31[0] = wptype(2) * v3result[1]
        v32[0] = wptype(2) * v3result[2]

        v40[0] = wptype(2) * v4result[0]
        v41[0] = wptype(2) * v4result[1]
        v42[0] = wptype(2) * v4result[2]
        v43[0] = wptype(2) * v4result[3]

        v50[0] = wptype(2) * v5result[0]
        v51[0] = wptype(2) * v5result[1]
        v52[0] = wptype(2) * v5result[2]
        v53[0] = wptype(2) * v5result[3]
        v54[0] = wptype(2) * v5result[4]

    kernel = getkernel(check_cw_div, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                s2,
                s3,
                s4,
                s5,
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=[v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54],
            device=device,
        )

    if dtype in np_int_types:
        assert_np_equal(v20.numpy()[0], 2 * (v2.numpy()[0, 0] // s2.numpy()[0, 0]), tol=tol)
        assert_np_equal(v21.numpy()[0], 2 * (v2.numpy()[0, 1] // s2.numpy()[0, 1]), tol=tol)

        assert_np_equal(v30.numpy()[0], 2 * (v3.numpy()[0, 0] // s3.numpy()[0, 0]), tol=tol)
        assert_np_equal(v31.numpy()[0], 2 * (v3.numpy()[0, 1] // s3.numpy()[0, 1]), tol=tol)
        assert_np_equal(v32.numpy()[0], 2 * (v3.numpy()[0, 2] // s3.numpy()[0, 2]), tol=tol)

        assert_np_equal(v40.numpy()[0], 2 * (v4.numpy()[0, 0] // s4.numpy()[0, 0]), tol=tol)
        assert_np_equal(v41.numpy()[0], 2 * (v4.numpy()[0, 1] // s4.numpy()[0, 1]), tol=tol)
        assert_np_equal(v42.numpy()[0], 2 * (v4.numpy()[0, 2] // s4.numpy()[0, 2]), tol=tol)
        assert_np_equal(v43.numpy()[0], 2 * (v4.numpy()[0, 3] // s4.numpy()[0, 3]), tol=tol)

        assert_np_equal(v50.numpy()[0], 2 * (v5.numpy()[0, 0] // s5.numpy()[0, 0]), tol=tol)
        assert_np_equal(v51.numpy()[0], 2 * (v5.numpy()[0, 1] // s5.numpy()[0, 1]), tol=tol)
        assert_np_equal(v52.numpy()[0], 2 * (v5.numpy()[0, 2] // s5.numpy()[0, 2]), tol=tol)
        assert_np_equal(v53.numpy()[0], 2 * (v5.numpy()[0, 3] // s5.numpy()[0, 3]), tol=tol)
        assert_np_equal(v54.numpy()[0], 2 * (v5.numpy()[0, 4] // s5.numpy()[0, 4]), tol=tol)
    else:
        assert_np_equal(v20.numpy()[0], 2 * v2.numpy()[0, 0] / s2.numpy()[0, 0], tol=tol)
        assert_np_equal(v21.numpy()[0], 2 * v2.numpy()[0, 1] / s2.numpy()[0, 1], tol=tol)

        assert_np_equal(v30.numpy()[0], 2 * v3.numpy()[0, 0] / s3.numpy()[0, 0], tol=tol)
        assert_np_equal(v31.numpy()[0], 2 * v3.numpy()[0, 1] / s3.numpy()[0, 1], tol=tol)
        assert_np_equal(v32.numpy()[0], 2 * v3.numpy()[0, 2] / s3.numpy()[0, 2], tol=tol)

        assert_np_equal(v40.numpy()[0], 2 * v4.numpy()[0, 0] / s4.numpy()[0, 0], tol=tol)
        assert_np_equal(v41.numpy()[0], 2 * v4.numpy()[0, 1] / s4.numpy()[0, 1], tol=tol)
        assert_np_equal(v42.numpy()[0], 2 * v4.numpy()[0, 2] / s4.numpy()[0, 2], tol=tol)
        assert_np_equal(v43.numpy()[0], 2 * v4.numpy()[0, 3] / s4.numpy()[0, 3], tol=tol)

        assert_np_equal(v50.numpy()[0], 2 * v5.numpy()[0, 0] / s5.numpy()[0, 0], tol=tol)
        assert_np_equal(v51.numpy()[0], 2 * v5.numpy()[0, 1] / s5.numpy()[0, 1], tol=tol)
        assert_np_equal(v52.numpy()[0], 2 * v5.numpy()[0, 2] / s5.numpy()[0, 2], tol=tol)
        assert_np_equal(v53.numpy()[0], 2 * v5.numpy()[0, 3] / s5.numpy()[0, 3], tol=tol)
        assert_np_equal(v54.numpy()[0], 2 * v5.numpy()[0, 4] / s5.numpy()[0, 4], tol=tol)

    if dtype in np_float_types:
        incmps = np.concatenate([v.numpy()[0] for v in [v2, v3, v4, v5]])
        scmps = np.concatenate([v.numpy()[0] for v in [s2, s3, s4, s5]])

        for i, l in enumerate([v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54]):
            tape.backward(loss=l)
            sgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [s2, s3, s4, s5]])
            expected_grads = np.zeros_like(sgrads)

            # d/ds v/s = -v/s^2
            expected_grads[i] = -incmps[i] * 2 / (scmps[i] * scmps[i])
            assert_np_equal(sgrads, expected_grads, tol=20 * tol)

            allgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [v2, v3, v4, v5]])
            expected_grads = np.zeros_like(allgrads)

            # d/dv v/s = 1/s
            expected_grads[i] = 2 / scmps[i]
            assert_np_equal(allgrads, expected_grads, tol=tol)

            tape.zero()


def test_addition(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_add(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v20: wp.array(dtype=wptype),
        v21: wp.array(dtype=wptype),
        v30: wp.array(dtype=wptype),
        v31: wp.array(dtype=wptype),
        v32: wp.array(dtype=wptype),
        v40: wp.array(dtype=wptype),
        v41: wp.array(dtype=wptype),
        v42: wp.array(dtype=wptype),
        v43: wp.array(dtype=wptype),
        v50: wp.array(dtype=wptype),
        v51: wp.array(dtype=wptype),
        v52: wp.array(dtype=wptype),
        v53: wp.array(dtype=wptype),
        v54: wp.array(dtype=wptype),
    ):
        v2result = v2[0] + s2[0]
        v3result = v3[0] + s3[0]
        v4result = v4[0] + s4[0]
        v5result = v5[0] + s5[0]

        v20[0] = wptype(2) * v2result[0]
        v21[0] = wptype(2) * v2result[1]

        v30[0] = wptype(2) * v3result[0]
        v31[0] = wptype(2) * v3result[1]
        v32[0] = wptype(2) * v3result[2]

        v40[0] = wptype(2) * v4result[0]
        v41[0] = wptype(2) * v4result[1]
        v42[0] = wptype(2) * v4result[2]
        v43[0] = wptype(2) * v4result[3]

        v50[0] = wptype(2) * v5result[0]
        v51[0] = wptype(2) * v5result[1]
        v52[0] = wptype(2) * v5result[2]
        v53[0] = wptype(2) * v5result[3]
        v54[0] = wptype(2) * v5result[4]

    kernel = getkernel(check_add, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    v54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                s2,
                s3,
                s4,
                s5,
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=[v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54],
            device=device,
        )

    assert_np_equal(v20.numpy()[0], 2 * (v2.numpy()[0, 0] + s2.numpy()[0, 0]), tol=tol)
    assert_np_equal(v21.numpy()[0], 2 * (v2.numpy()[0, 1] + s2.numpy()[0, 1]), tol=tol)

    assert_np_equal(v30.numpy()[0], 2 * (v3.numpy()[0, 0] + s3.numpy()[0, 0]), tol=tol)
    assert_np_equal(v31.numpy()[0], 2 * (v3.numpy()[0, 1] + s3.numpy()[0, 1]), tol=tol)
    assert_np_equal(v32.numpy()[0], 2 * (v3.numpy()[0, 2] + s3.numpy()[0, 2]), tol=tol)

    assert_np_equal(v40.numpy()[0], 2 * (v4.numpy()[0, 0] + s4.numpy()[0, 0]), tol=tol)
    assert_np_equal(v41.numpy()[0], 2 * (v4.numpy()[0, 1] + s4.numpy()[0, 1]), tol=tol)
    assert_np_equal(v42.numpy()[0], 2 * (v4.numpy()[0, 2] + s4.numpy()[0, 2]), tol=tol)
    assert_np_equal(v43.numpy()[0], 2 * (v4.numpy()[0, 3] + s4.numpy()[0, 3]), tol=tol)

    assert_np_equal(v50.numpy()[0], 2 * (v5.numpy()[0, 0] + s5.numpy()[0, 0]), tol=tol)
    assert_np_equal(v51.numpy()[0], 2 * (v5.numpy()[0, 1] + s5.numpy()[0, 1]), tol=tol)
    assert_np_equal(v52.numpy()[0], 2 * (v5.numpy()[0, 2] + s5.numpy()[0, 2]), tol=tol)
    assert_np_equal(v53.numpy()[0], 2 * (v5.numpy()[0, 3] + s5.numpy()[0, 3]), tol=tol)
    assert_np_equal(v54.numpy()[0], 2 * (v5.numpy()[0, 4] + s5.numpy()[0, 4]), tol=2 * tol)

    if dtype in np_float_types:
        for i, l in enumerate([v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54]):
            tape.backward(loss=l)
            sgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [s2, s3, s4, s5]])
            expected_grads = np.zeros_like(sgrads)

            expected_grads[i] = 2
            assert_np_equal(sgrads, expected_grads, tol=10 * tol)

            allgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [v2, v3, v4, v5]])
            assert_np_equal(allgrads, expected_grads, tol=tol)

            tape.zero()


def test_dotproduct(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_dot(
        s2: wp.array(dtype=vec2),
        s3: wp.array(dtype=vec3),
        s4: wp.array(dtype=vec4),
        s5: wp.array(dtype=vec5),
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        dot2: wp.array(dtype=wptype),
        dot3: wp.array(dtype=wptype),
        dot4: wp.array(dtype=wptype),
        dot5: wp.array(dtype=wptype),
    ):
        dot2[0] = wptype(2) * wp.dot(v2[0], s2[0])
        dot3[0] = wptype(2) * wp.dot(v3[0], s3[0])
        dot4[0] = wptype(2) * wp.dot(v4[0], s4[0])
        dot5[0] = wptype(2) * wp.dot(v5[0], s5[0])

    kernel = getkernel(check_dot, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    dot2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    dot5 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                s2,
                s3,
                s4,
                s5,
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=[dot2, dot3, dot4, dot5],
            device=device,
        )

    assert_np_equal(dot2.numpy()[0], 2.0 * (v2.numpy() * s2.numpy()).sum(), tol=10 * tol)
    assert_np_equal(dot3.numpy()[0], 2.0 * (v3.numpy() * s3.numpy()).sum(), tol=10 * tol)
    assert_np_equal(dot4.numpy()[0], 2.0 * (v4.numpy() * s4.numpy()).sum(), tol=10 * tol)
    assert_np_equal(dot5.numpy()[0], 2.0 * (v5.numpy() * s5.numpy()).sum(), tol=10 * tol)

    if dtype in np_float_types:
        tape.backward(loss=dot2)
        sgrads = tape.gradients[s2].numpy()[0]
        expected_grads = 2.0 * v2.numpy()[0]
        assert_np_equal(sgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v2].numpy()[0]
        expected_grads = 2.0 * s2.numpy()[0]
        assert_np_equal(vgrads, expected_grads, tol=tol)

        tape.zero()

        tape.backward(loss=dot3)
        sgrads = tape.gradients[s3].numpy()[0]
        expected_grads = 2.0 * v3.numpy()[0]
        assert_np_equal(sgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v3].numpy()[0]
        expected_grads = 2.0 * s3.numpy()[0]
        assert_np_equal(vgrads, expected_grads, tol=tol)

        tape.zero()

        tape.backward(loss=dot4)
        sgrads = tape.gradients[s4].numpy()[0]
        expected_grads = 2.0 * v4.numpy()[0]
        assert_np_equal(sgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v4].numpy()[0]
        expected_grads = 2.0 * s4.numpy()[0]
        assert_np_equal(vgrads, expected_grads, tol=tol)

        tape.zero()

        tape.backward(loss=dot5)
        sgrads = tape.gradients[s5].numpy()[0]
        expected_grads = 2.0 * v5.numpy()[0]
        assert_np_equal(sgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v5].numpy()[0]
        expected_grads = 2.0 * s5.numpy()[0]
        assert_np_equal(vgrads, expected_grads, tol=10 * tol)

        tape.zero()


def test_equivalent_types(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    # vector types
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    # vector types equivalent to the above
    vec2_equiv = wp.types.vector(length=2, dtype=wptype)
    vec3_equiv = wp.types.vector(length=3, dtype=wptype)
    vec4_equiv = wp.types.vector(length=4, dtype=wptype)
    vec5_equiv = wp.types.vector(length=5, dtype=wptype)

    # declare kernel with original types
    def check_equivalence(
        v2: vec2,
        v3: vec3,
        v4: vec4,
        v5: vec5,
    ):
        wp.expect_eq(v2, vec2(wptype(1), wptype(2)))
        wp.expect_eq(v3, vec3(wptype(1), wptype(2), wptype(3)))
        wp.expect_eq(v4, vec4(wptype(1), wptype(2), wptype(3), wptype(4)))
        wp.expect_eq(v5, vec5(wptype(1), wptype(2), wptype(3), wptype(4), wptype(5)))

        wp.expect_eq(v2, vec2_equiv(wptype(1), wptype(2)))
        wp.expect_eq(v3, vec3_equiv(wptype(1), wptype(2), wptype(3)))
        wp.expect_eq(v4, vec4_equiv(wptype(1), wptype(2), wptype(3), wptype(4)))
        wp.expect_eq(v5, vec5_equiv(wptype(1), wptype(2), wptype(3), wptype(4), wptype(5)))

    kernel = getkernel(check_equivalence, suffix=dtype.__name__)

    if register_kernels:
        return

    # call kernel with equivalent types
    v2 = vec2_equiv(1, 2)
    v3 = vec3_equiv(1, 2, 3)
    v4 = vec4_equiv(1, 2, 3, 4)
    v5 = vec5_equiv(1, 2, 3, 4, 5)

    wp.launch(kernel, dim=1, inputs=[v2, v3, v4, v5], device=device)


def test_conversions(test, device, dtype, register_kernels=False):
    def check_vectors_equal(
        v0: wp.vec3,
        v1: wp.vec3,
        v2: wp.vec3,
        v3: wp.vec3,
    ):
        wp.expect_eq(v1, v0)
        wp.expect_eq(v2, v0)
        wp.expect_eq(v3, v0)

    kernel = getkernel(check_vectors_equal, suffix=dtype.__name__)

    if register_kernels:
        return

    v0 = wp.vec3(1, 2, 3)

    # test explicit conversions - constructing vectors from different containers
    v1 = wp.vec3((1, 2, 3))
    v2 = wp.vec3([1, 2, 3])
    v3 = wp.vec3(np.array([1, 2, 3], dtype=dtype))

    wp.launch(kernel, dim=1, inputs=[v0, v1, v2, v3], device=device)

    # test implicit conversions - passing different containers as vectors to wp.launch()
    v1 = (1, 2, 3)
    v2 = [1, 2, 3]
    v3 = np.array([1, 2, 3], dtype=dtype)

    wp.launch(kernel, dim=1, inputs=[v0, v1, v2, v3], device=device)


def test_constants(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    cv2 = wp.constant(vec2(1, 2))
    cv3 = wp.constant(vec3(1, 2, 3))
    cv4 = wp.constant(vec4(1, 2, 3, 4))
    cv5 = wp.constant(vec5(1, 2, 3, 4, 5))

    def check_vector_constants():
        wp.expect_eq(cv2, vec2(wptype(1), wptype(2)))
        wp.expect_eq(cv3, vec3(wptype(1), wptype(2), wptype(3)))
        wp.expect_eq(cv4, vec4(wptype(1), wptype(2), wptype(3), wptype(4)))
        wp.expect_eq(cv5, vec5(wptype(1), wptype(2), wptype(3), wptype(4), wptype(5)))

    kernel = getkernel(check_vector_constants, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=1, inputs=[], device=device)


def test_minmax(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    # \TODO: not quite sure why, but the numbers are off for 16 bit float
    # on the cpu (but not cuda). This is probably just the sketchy float16
    # arithmetic I implemented to get all this stuff working, so
    # hopefully that can be fixed when we do that correctly.
    tol = {
        np.float16: 1.0e-2,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    # \TODO: Also not quite sure why: this kernel compiles incredibly
    # slowly though...
    def check_vec_min_max(
        a: wp.array(dtype=wptype, ndim=2),
        b: wp.array(dtype=wptype, ndim=2),
        mins: wp.array(dtype=wptype, ndim=2),
        maxs: wp.array(dtype=wptype, ndim=2),
    ):
        for i in range(10):
            # multiplying by 2 so we've got something to backpropagate:
            a2read = vec2(a[i, 0], a[i, 1])
            b2read = vec2(b[i, 0], b[i, 1])
            c2 = wptype(2) * wp.min(a2read, b2read)
            d2 = wptype(2) * wp.max(a2read, b2read)

            a3read = vec3(a[i, 2], a[i, 3], a[i, 4])
            b3read = vec3(b[i, 2], b[i, 3], b[i, 4])
            c3 = wptype(2) * wp.min(a3read, b3read)
            d3 = wptype(2) * wp.max(a3read, b3read)

            a4read = vec4(a[i, 5], a[i, 6], a[i, 7], a[i, 8])
            b4read = vec4(b[i, 5], b[i, 6], b[i, 7], b[i, 8])
            c4 = wptype(2) * wp.min(a4read, b4read)
            d4 = wptype(2) * wp.max(a4read, b4read)

            a5read = vec5(a[i, 9], a[i, 10], a[i, 11], a[i, 12], a[i, 13])
            b5read = vec5(b[i, 9], b[i, 10], b[i, 11], b[i, 12], b[i, 13])
            c5 = wptype(2) * wp.min(a5read, b5read)
            d5 = wptype(2) * wp.max(a5read, b5read)

            mins[i, 0] = c2[0]
            mins[i, 1] = c2[1]

            mins[i, 2] = c3[0]
            mins[i, 3] = c3[1]
            mins[i, 4] = c3[2]

            mins[i, 5] = c4[0]
            mins[i, 6] = c4[1]
            mins[i, 7] = c4[2]
            mins[i, 8] = c4[3]

            mins[i, 9] = c5[0]
            mins[i, 10] = c5[1]
            mins[i, 11] = c5[2]
            mins[i, 12] = c5[3]
            mins[i, 13] = c5[4]

            maxs[i, 0] = d2[0]
            maxs[i, 1] = d2[1]

            maxs[i, 2] = d3[0]
            maxs[i, 3] = d3[1]
            maxs[i, 4] = d3[2]

            maxs[i, 5] = d4[0]
            maxs[i, 6] = d4[1]
            maxs[i, 7] = d4[2]
            maxs[i, 8] = d4[3]

            maxs[i, 9] = d5[0]
            maxs[i, 10] = d5[1]
            maxs[i, 11] = d5[2]
            maxs[i, 12] = d5[3]
            maxs[i, 13] = d5[4]

    kernel = getkernel(check_vec_min_max, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel2(wptype)

    if register_kernels:
        return

    a = wp.array(randvals(rng, (10, 14), dtype), dtype=wptype, requires_grad=True, device=device)
    b = wp.array(randvals(rng, (10, 14), dtype), dtype=wptype, requires_grad=True, device=device)

    mins = wp.zeros((10, 14), dtype=wptype, requires_grad=True, device=device)
    maxs = wp.zeros((10, 14), dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[a, b], outputs=[mins, maxs], device=device)

    assert_np_equal(mins.numpy(), 2 * np.minimum(a.numpy(), b.numpy()), tol=tol)
    assert_np_equal(maxs.numpy(), 2 * np.maximum(a.numpy(), b.numpy()), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    if dtype in np_float_types:
        for i in range(10):
            for j in range(14):
                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[a, b], outputs=[mins, maxs], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[mins, i, j], outputs=[out], device=device)

                tape.backward(loss=out)
                expected = np.zeros_like(a.numpy())
                expected[i, j] = 2 if (a.numpy()[i, j] < b.numpy()[i, j]) else 0
                assert_np_equal(tape.gradients[a].numpy(), expected, tol=tol)
                expected[i, j] = 2 if (b.numpy()[i, j] < a.numpy()[i, j]) else 0
                assert_np_equal(tape.gradients[b].numpy(), expected, tol=tol)
                tape.zero()

                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[a, b], outputs=[mins, maxs], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[maxs, i, j], outputs=[out], device=device)

                tape.backward(loss=out)
                expected = np.zeros_like(a.numpy())
                expected[i, j] = 2 if (a.numpy()[i, j] > b.numpy()[i, j]) else 0
                assert_np_equal(tape.gradients[a].numpy(), expected, tol=tol)
                expected[i, j] = 2 if (b.numpy()[i, j] > a.numpy()[i, j]) else 0
                assert_np_equal(tape.gradients[b].numpy(), expected, tol=tol)
                tape.zero()


devices = get_test_devices()


class TestVecScalarOps(unittest.TestCase):
    pass


for dtype in np_scalar_types:
    add_function_test(TestVecScalarOps, f"test_arrays_{dtype.__name__}", test_arrays, devices=devices, dtype=dtype)
    add_function_test(TestVecScalarOps, f"test_components_{dtype.__name__}", test_components, devices=None, dtype=dtype)
    add_function_test(
        TestVecScalarOps, f"test_py_arithmetic_ops_{dtype.__name__}", test_py_arithmetic_ops, devices=None, dtype=dtype
    )
    add_function_test_register_kernel(
        TestVecScalarOps, f"test_constructors_{dtype.__name__}", test_constructors, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestVecScalarOps,
        f"test_anon_type_instance_{dtype.__name__}",
        test_anon_type_instance,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestVecScalarOps, f"test_indexing_{dtype.__name__}", test_indexing, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestVecScalarOps, f"test_equality_{dtype.__name__}", test_equality, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestVecScalarOps,
        f"test_scalar_multiplication_{dtype.__name__}",
        test_scalar_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestVecScalarOps,
        f"test_scalar_multiplication_rightmul_{dtype.__name__}",
        test_scalar_multiplication_rightmul,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestVecScalarOps,
        f"test_cw_multiplication_{dtype.__name__}",
        test_cw_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestVecScalarOps, f"test_scalar_division_{dtype.__name__}", test_scalar_division, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestVecScalarOps, f"test_cw_division_{dtype.__name__}", test_cw_division, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestVecScalarOps, f"test_addition_{dtype.__name__}", test_addition, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestVecScalarOps, f"test_dotproduct_{dtype.__name__}", test_dotproduct, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestVecScalarOps, f"test_equivalent_types_{dtype.__name__}", test_equivalent_types, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestVecScalarOps, f"test_conversions_{dtype.__name__}", test_conversions, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestVecScalarOps, f"test_constants_{dtype.__name__}", test_constants, devices=devices, dtype=dtype
    )

    # the kernels in this test compile incredibly slowly...
    # add_function_test_register_kernel(TestVecScalarOps, f"test_minmax_{dtype.__name__}", test_minmax, devices=devices, dtype=dtype)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
