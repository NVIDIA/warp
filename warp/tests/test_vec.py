# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
from warp.tests.test_base import *

wp.init()

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


def randvals(shape, dtype):
    if dtype in np_float_types:
        return np.random.randn(*shape).astype(dtype)
    elif dtype in [np.int8, np.uint8, np.byte, np.ubyte]:
        return np.random.randint(1, 3, size=shape, dtype=dtype)
    return np.random.randint(1, 5, size=shape, dtype=dtype)


kernel_cache = dict()


def getkernel(func, suffix=""):
    module = wp.get_module(func.__module__)
    key = func.__name__ + "_" + suffix
    if key not in kernel_cache:
        kernel_cache[key] = wp.Kernel(func=func, key=key, module=module)
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
    np.random.seed(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    v2_np = randvals((10, 2), dtype)
    v3_np = randvals((10, 3), dtype)
    v4_np = randvals((10, 4), dtype)
    v5_np = randvals((10, 5), dtype)

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


def test_anon_type_instance(test, device, dtype, register_kernels=False):
    np.random.seed(123)

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

    input = wp.array(randvals([4], dtype), requires_grad=True, device=device)
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

    input = wp.array(randvals([2 + 3 + 4 + 5], dtype), requires_grad=True, device=device)
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

    wp.launch(kernel, dim=1, inputs=[])


def test_constructors(test, device, dtype, register_kernels=False):
    np.random.seed(123)

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

    input = wp.array(randvals([1], dtype), requires_grad=True, device=device)
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

    input = wp.array(randvals([14], dtype), requires_grad=True, device=device)
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


def test_indexing(test, device, dtype, register_kernels=False):
    np.random.seed(123)

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

    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
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
    np.random.seed(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_equality(
        v20: wp.array(dtype=vec2),
        v21: wp.array(dtype=vec2),
        v22: wp.array(dtype=vec2),
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
        wp.expect_eq(v20[0], v20[0])
        wp.expect_neq(v21[0], v20[0])
        wp.expect_neq(v22[0], v20[0])

        wp.expect_eq(v30[0], v30[0])
        wp.expect_neq(v31[0], v30[0])
        wp.expect_neq(v32[0], v30[0])
        wp.expect_neq(v33[0], v30[0])

        wp.expect_eq(v40[0], v40[0])
        wp.expect_neq(v41[0], v40[0])
        wp.expect_neq(v42[0], v40[0])
        wp.expect_neq(v43[0], v40[0])
        wp.expect_neq(v44[0], v40[0])

        wp.expect_eq(v50[0], v50[0])
        wp.expect_neq(v51[0], v50[0])
        wp.expect_neq(v52[0], v50[0])
        wp.expect_neq(v53[0], v50[0])
        wp.expect_neq(v54[0], v50[0])
        wp.expect_neq(v55[0], v50[0])

    kernel = getkernel(check_equality, suffix=dtype.__name__)

    if register_kernels:
        return

    v20 = wp.array([1.0, 2.0], dtype=vec2, requires_grad=True, device=device)
    v21 = wp.array([1.0, 3.0], dtype=vec2, requires_grad=True, device=device)
    v22 = wp.array([3.0, 2.0], dtype=vec2, requires_grad=True, device=device)

    v30 = wp.array([1.0, 2.0, 3.0], dtype=vec3, requires_grad=True, device=device)
    v31 = wp.array([-1.0, 2.0, 3.0], dtype=vec3, requires_grad=True, device=device)
    v32 = wp.array([1.0, -2.0, 3.0], dtype=vec3, requires_grad=True, device=device)
    v33 = wp.array([1.0, 2.0, -3.0], dtype=vec3, requires_grad=True, device=device)

    v40 = wp.array([1.0, 2.0, 3.0, 4.0], dtype=vec4, requires_grad=True, device=device)
    v41 = wp.array([-1.0, 2.0, 3.0, 4.0], dtype=vec4, requires_grad=True, device=device)
    v42 = wp.array([1.0, -2.0, 3.0, 4.0], dtype=vec4, requires_grad=True, device=device)
    v43 = wp.array([1.0, 2.0, -3.0, 4.0], dtype=vec4, requires_grad=True, device=device)
    v44 = wp.array([1.0, 2.0, 3.0, -4.0], dtype=vec4, requires_grad=True, device=device)

    v50 = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=vec5, requires_grad=True, device=device)
    v51 = wp.array([-1.0, 2.0, 3.0, 4.0, 5.0], dtype=vec5, requires_grad=True, device=device)
    v52 = wp.array([1.0, -2.0, 3.0, 4.0, 5.0], dtype=vec5, requires_grad=True, device=device)
    v53 = wp.array([1.0, 2.0, -3.0, 4.0, 5.0], dtype=vec5, requires_grad=True, device=device)
    v54 = wp.array([1.0, 2.0, 3.0, -4.0, 5.0], dtype=vec5, requires_grad=True, device=device)
    v55 = wp.array([1.0, 2.0, 3.0, 4.0, -5.0], dtype=vec5, requires_grad=True, device=device)
    wp.launch(
        kernel,
        dim=1,
        inputs=[
            v20,
            v21,
            v22,
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


def test_negation(test, device, dtype, register_kernels=False):
    np.random.seed(123)

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

    def check_negation(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        v2out: wp.array(dtype=vec2),
        v3out: wp.array(dtype=vec3),
        v4out: wp.array(dtype=vec4),
        v5out: wp.array(dtype=vec5),
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
        v2result = -v2[0]
        v3result = -v3[0]
        v4result = -v4[0]
        v5result = -v5[0]

        v2out[0] = v2result
        v3out[0] = v3result
        v4out[0] = v4result
        v5out[0] = v5result

        # multiply these outputs by 2 so we've got something to backpropagate:
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

    kernel = getkernel(check_negation, suffix=dtype.__name__)

    if register_kernels:
        return

    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5_np = randvals((1, 5), dtype)
    v5 = wp.array(v5_np, dtype=vec5, requires_grad=True, device=device)

    v2out = wp.zeros(1, dtype=vec2, device=device)
    v3out = wp.zeros(1, dtype=vec3, device=device)
    v4out = wp.zeros(1, dtype=vec4, device=device)
    v5out = wp.zeros(1, dtype=vec5, device=device)
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
            outputs=[v2out, v3out, v4out, v5out, v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54],
            device=device,
        )

    if dtype in np_float_types:
        for i, l in enumerate([v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54]):
            tape.backward(loss=l)
            allgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [v2, v3, v4, v5]])
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = -2
            assert_np_equal(allgrads, expected_grads, tol=tol)
            tape.zero()

    assert_np_equal(v2out.numpy()[0], -v2.numpy()[0], tol=tol)
    assert_np_equal(v3out.numpy()[0], -v3.numpy()[0], tol=tol)
    assert_np_equal(v4out.numpy()[0], -v4.numpy()[0], tol=tol)
    assert_np_equal(v5out.numpy()[0], -v5.numpy()[0], tol=tol)


def test_scalar_multiplication(test, device, dtype, register_kernels=False):
    np.random.seed(123)

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

    s = wp.array(randvals([1], dtype), requires_grad=True, device=device)
    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
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
    np.random.seed(123)

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

    s = wp.array(randvals([1], dtype), requires_grad=True, device=device)
    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
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
    np.random.seed(123)

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

    s2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
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
    np.random.seed(123)

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

    s = wp.array(randvals([1], dtype), requires_grad=True, device=device)
    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
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
    np.random.seed(123)

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

    s2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
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
    np.random.seed(123)

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

    s2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
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


def test_subtraction_unsigned(test, device, dtype, register_kernels=False):
    np.random.seed(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_subtraction_unsigned():
        wp.expect_eq(vec2(wptype(3), wptype(4)) - vec2(wptype(1), wptype(2)), vec2(wptype(2), wptype(2)))
        wp.expect_eq(
            vec3(
                wptype(3),
                wptype(4),
                wptype(4),
            )
            - vec3(wptype(1), wptype(2), wptype(3)),
            vec3(wptype(2), wptype(2), wptype(1)),
        )
        wp.expect_eq(
            vec4(
                wptype(3),
                wptype(4),
                wptype(4),
                wptype(5),
            )
            - vec4(wptype(1), wptype(2), wptype(3), wptype(4)),
            vec4(wptype(2), wptype(2), wptype(1), wptype(1)),
        )
        wp.expect_eq(
            vec5(
                wptype(3),
                wptype(4),
                wptype(4),
                wptype(5),
                wptype(4),
            )
            - vec5(wptype(1), wptype(2), wptype(3), wptype(4), wptype(4)),
            vec5(wptype(2), wptype(2), wptype(1), wptype(1), wptype(0)),
        )

    kernel = getkernel(check_subtraction_unsigned, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=1, inputs=[], outputs=[], device=device)


def test_subtraction(test, device, dtype, register_kernels=False):
    np.random.seed(123)

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

    def check_subtraction(
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
        v2result = v2[0] - s2[0]
        v3result = v3[0] - s3[0]
        v4result = v4[0] - s4[0]
        v5result = v5[0] - s5[0]

        # multiply outputs by 2 so there's something to backpropagate:
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

    kernel = getkernel(check_subtraction, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
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

    assert_np_equal(v20.numpy()[0], 2 * (v2.numpy()[0, 0] - s2.numpy()[0, 0]), tol=tol)
    assert_np_equal(v21.numpy()[0], 2 * (v2.numpy()[0, 1] - s2.numpy()[0, 1]), tol=tol)

    assert_np_equal(v30.numpy()[0], 2 * (v3.numpy()[0, 0] - s3.numpy()[0, 0]), tol=tol)
    assert_np_equal(v31.numpy()[0], 2 * (v3.numpy()[0, 1] - s3.numpy()[0, 1]), tol=tol)
    assert_np_equal(v32.numpy()[0], 2 * (v3.numpy()[0, 2] - s3.numpy()[0, 2]), tol=tol)

    assert_np_equal(v40.numpy()[0], 2 * (v4.numpy()[0, 0] - s4.numpy()[0, 0]), tol=2 * tol)
    assert_np_equal(v41.numpy()[0], 2 * (v4.numpy()[0, 1] - s4.numpy()[0, 1]), tol=2 * tol)
    assert_np_equal(v42.numpy()[0], 2 * (v4.numpy()[0, 2] - s4.numpy()[0, 2]), tol=2 * tol)
    assert_np_equal(v43.numpy()[0], 2 * (v4.numpy()[0, 3] - s4.numpy()[0, 3]), tol=2 * tol)

    assert_np_equal(v50.numpy()[0], 2 * (v5.numpy()[0, 0] - s5.numpy()[0, 0]), tol=tol)
    assert_np_equal(v51.numpy()[0], 2 * (v5.numpy()[0, 1] - s5.numpy()[0, 1]), tol=tol)
    assert_np_equal(v52.numpy()[0], 2 * (v5.numpy()[0, 2] - s5.numpy()[0, 2]), tol=tol)
    assert_np_equal(v53.numpy()[0], 2 * (v5.numpy()[0, 3] - s5.numpy()[0, 3]), tol=tol)
    assert_np_equal(v54.numpy()[0], 2 * (v5.numpy()[0, 4] - s5.numpy()[0, 4]), tol=tol)

    if dtype in np_float_types:
        for i, l in enumerate([v20, v21, v30, v31, v32, v40, v41, v42, v43, v50, v51, v52, v53, v54]):
            tape.backward(loss=l)
            sgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [s2, s3, s4, s5]])
            expected_grads = np.zeros_like(sgrads)

            expected_grads[i] = -2
            assert_np_equal(sgrads, expected_grads, tol=10 * tol)

            allgrads = np.concatenate([tape.gradients[v].numpy()[0] for v in [v2, v3, v4, v5]])
            expected_grads = np.zeros_like(allgrads)

            # d/dv v/s = 1/s
            expected_grads[i] = 2
            assert_np_equal(allgrads, expected_grads, tol=tol)

            tape.zero()


def test_dotproduct(test, device, dtype, register_kernels=False):
    np.random.seed(123)

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

    s2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    s3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    s4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    s5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)
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


def test_length(test, device, dtype, register_kernels=False):
    np.random.seed(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-7,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec2 = wp.types.vector(length=2, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec5 = wp.types.vector(length=5, dtype=wptype)

    def check_length(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        l2: wp.array(dtype=wptype),
        l3: wp.array(dtype=wptype),
        l4: wp.array(dtype=wptype),
        l5: wp.array(dtype=wptype),
        l22: wp.array(dtype=wptype),
        l23: wp.array(dtype=wptype),
        l24: wp.array(dtype=wptype),
        l25: wp.array(dtype=wptype),
    ):
        l2[0] = wptype(2) * wp.length(v2[0])
        l3[0] = wptype(2) * wp.length(v3[0])
        l4[0] = wptype(2) * wp.length(v4[0])
        l5[0] = wptype(2) * wp.length(v5[0])

        l22[0] = wptype(2) * wp.length_sq(v2[0])
        l23[0] = wptype(2) * wp.length_sq(v3[0])
        l24[0] = wptype(2) * wp.length_sq(v4[0])
        l25[0] = wptype(2) * wp.length_sq(v5[0])

    kernel = getkernel(check_length, suffix=dtype.__name__)

    if register_kernels:
        return

    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)

    l2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l5 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    l22 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l23 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l24 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l25 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=[l2, l3, l4, l5, l22, l23, l24, l25],
            device=device,
        )

    assert_np_equal(l2.numpy()[0], 2 * np.linalg.norm(v2.numpy()), tol=10 * tol)
    assert_np_equal(l3.numpy()[0], 2 * np.linalg.norm(v3.numpy()), tol=10 * tol)
    assert_np_equal(l4.numpy()[0], 2 * np.linalg.norm(v4.numpy()), tol=10 * tol)
    assert_np_equal(l5.numpy()[0], 2 * np.linalg.norm(v5.numpy()), tol=10 * tol)

    assert_np_equal(l22.numpy()[0], 2 * np.linalg.norm(v2.numpy()) ** 2, tol=10 * tol)
    assert_np_equal(l23.numpy()[0], 2 * np.linalg.norm(v3.numpy()) ** 2, tol=10 * tol)
    assert_np_equal(l24.numpy()[0], 2 * np.linalg.norm(v4.numpy()) ** 2, tol=10 * tol)
    assert_np_equal(l25.numpy()[0], 2 * np.linalg.norm(v5.numpy()) ** 2, tol=10 * tol)

    tape.backward(loss=l2)
    grad = tape.gradients[v2].numpy()[0]
    expected_grad = 2 * v2.numpy()[0] / np.linalg.norm(v2.numpy())
    assert_np_equal(grad, expected_grad, tol=10 * tol)
    tape.zero()

    tape.backward(loss=l3)
    grad = tape.gradients[v3].numpy()[0]
    expected_grad = 2 * v3.numpy()[0] / np.linalg.norm(v3.numpy())
    assert_np_equal(grad, expected_grad, tol=10 * tol)
    tape.zero()

    tape.backward(loss=l4)
    grad = tape.gradients[v4].numpy()[0]
    expected_grad = 2 * v4.numpy()[0] / np.linalg.norm(v4.numpy())
    assert_np_equal(grad, expected_grad, tol=10 * tol)
    tape.zero()

    tape.backward(loss=l5)
    grad = tape.gradients[v5].numpy()[0]
    expected_grad = 2 * v5.numpy()[0] / np.linalg.norm(v5.numpy())
    assert_np_equal(grad, expected_grad, tol=10 * tol)
    tape.zero()

    tape.backward(loss=l22)
    grad = tape.gradients[v2].numpy()[0]
    expected_grad = 4 * v2.numpy()[0]
    assert_np_equal(grad, expected_grad, tol=10 * tol)
    tape.zero()

    tape.backward(loss=l23)
    grad = tape.gradients[v3].numpy()[0]
    expected_grad = 4 * v3.numpy()[0]
    assert_np_equal(grad, expected_grad, tol=10 * tol)
    tape.zero()

    tape.backward(loss=l24)
    grad = tape.gradients[v4].numpy()[0]
    expected_grad = 4 * v4.numpy()[0]
    assert_np_equal(grad, expected_grad, tol=10 * tol)
    tape.zero()

    tape.backward(loss=l25)
    grad = tape.gradients[v5].numpy()[0]
    expected_grad = 4 * v5.numpy()[0]
    assert_np_equal(grad, expected_grad, tol=10 * tol)
    tape.zero()


def test_normalize(test, device, dtype, register_kernels=False):
    np.random.seed(123)

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

    def check_normalize(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        n20: wp.array(dtype=wptype),
        n21: wp.array(dtype=wptype),
        n30: wp.array(dtype=wptype),
        n31: wp.array(dtype=wptype),
        n32: wp.array(dtype=wptype),
        n40: wp.array(dtype=wptype),
        n41: wp.array(dtype=wptype),
        n42: wp.array(dtype=wptype),
        n43: wp.array(dtype=wptype),
        n50: wp.array(dtype=wptype),
        n51: wp.array(dtype=wptype),
        n52: wp.array(dtype=wptype),
        n53: wp.array(dtype=wptype),
        n54: wp.array(dtype=wptype),
    ):
        n2 = wptype(2) * wp.normalize(v2[0])
        n3 = wptype(2) * wp.normalize(v3[0])
        n4 = wptype(2) * wp.normalize(v4[0])
        n5 = wptype(2) * wp.normalize(v5[0])

        n20[0] = n2[0]
        n21[0] = n2[1]

        n30[0] = n3[0]
        n31[0] = n3[1]
        n32[0] = n3[2]

        n40[0] = n4[0]
        n41[0] = n4[1]
        n42[0] = n4[2]
        n43[0] = n4[3]

        n50[0] = n5[0]
        n51[0] = n5[1]
        n52[0] = n5[2]
        n53[0] = n5[3]
        n54[0] = n5[4]

    def check_normalize_alt(
        v2: wp.array(dtype=vec2),
        v3: wp.array(dtype=vec3),
        v4: wp.array(dtype=vec4),
        v5: wp.array(dtype=vec5),
        n20: wp.array(dtype=wptype),
        n21: wp.array(dtype=wptype),
        n30: wp.array(dtype=wptype),
        n31: wp.array(dtype=wptype),
        n32: wp.array(dtype=wptype),
        n40: wp.array(dtype=wptype),
        n41: wp.array(dtype=wptype),
        n42: wp.array(dtype=wptype),
        n43: wp.array(dtype=wptype),
        n50: wp.array(dtype=wptype),
        n51: wp.array(dtype=wptype),
        n52: wp.array(dtype=wptype),
        n53: wp.array(dtype=wptype),
        n54: wp.array(dtype=wptype),
    ):
        n2 = wptype(2) * v2[0] / wp.length(v2[0])
        n3 = wptype(2) * v3[0] / wp.length(v3[0])
        n4 = wptype(2) * v4[0] / wp.length(v4[0])
        n5 = wptype(2) * v5[0] / wp.length(v5[0])

        n20[0] = n2[0]
        n21[0] = n2[1]

        n30[0] = n3[0]
        n31[0] = n3[1]
        n32[0] = n3[2]

        n40[0] = n4[0]
        n41[0] = n4[1]
        n42[0] = n4[2]
        n43[0] = n4[3]

        n50[0] = n5[0]
        n51[0] = n5[1]
        n52[0] = n5[2]
        n53[0] = n5[3]
        n54[0] = n5[4]

    normalize_kernel = getkernel(check_normalize, suffix=dtype.__name__)
    normalize_alt_kernel = getkernel(check_normalize_alt, suffix=dtype.__name__)

    if register_kernels:
        return

    # I've already tested the things I'm using in check_normalize_alt, so I'll just
    # make sure the two are giving the same results/gradients
    v2 = wp.array(randvals((1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals((1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals((1, 5), dtype), dtype=vec5, requires_grad=True, device=device)

    n20 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n21 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n30 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n31 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n32 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n40 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n41 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n42 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n43 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n50 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n51 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n52 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n53 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n54 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    n20_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n21_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n30_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n31_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n32_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n40_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n41_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n42_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n43_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n50_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n51_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n52_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n53_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n54_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    outputs0 = [
        n20,
        n21,
        n30,
        n31,
        n32,
        n40,
        n41,
        n42,
        n43,
        n50,
        n51,
        n52,
        n53,
        n54,
    ]
    tape0 = wp.Tape()
    with tape0:
        wp.launch(
            normalize_kernel,
            dim=1,
            inputs=[
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=outputs0,
            device=device,
        )

    outputs1 = [
        n20_alt,
        n21_alt,
        n30_alt,
        n31_alt,
        n32_alt,
        n40_alt,
        n41_alt,
        n42_alt,
        n43_alt,
        n50_alt,
        n51_alt,
        n52_alt,
        n53_alt,
        n54_alt,
    ]
    tape1 = wp.Tape()
    with tape1:
        wp.launch(
            normalize_alt_kernel,
            dim=1,
            inputs=[
                v2,
                v3,
                v4,
                v5,
            ],
            outputs=outputs1,
            device=device,
        )

    for ncmp, ncmpalt in zip(outputs0, outputs1):
        assert_np_equal(ncmp.numpy()[0], ncmpalt.numpy()[0], tol=10 * tol)

    invecs = [
        v2,
        v2,
        v3,
        v3,
        v3,
        v4,
        v4,
        v4,
        v4,
        v5,
        v5,
        v5,
        v5,
        v5,
    ]
    for ncmp, ncmpalt, v in zip(outputs0, outputs1, invecs):
        tape0.backward(loss=ncmp)
        tape1.backward(loss=ncmpalt)
        assert_np_equal(tape0.gradients[v].numpy()[0], tape1.gradients[v].numpy()[0], tol=10 * tol)
        tape0.zero()
        tape1.zero()


def test_crossproduct(test, device, dtype, register_kernels=False):
    np.random.seed(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(length=3, dtype=wptype)

    def check_cross(
        s3: wp.array(dtype=vec3),
        v3: wp.array(dtype=vec3),
        c0: wp.array(dtype=wptype),
        c1: wp.array(dtype=wptype),
        c2: wp.array(dtype=wptype),
    ):
        c = wp.cross(s3[0], v3[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        c0[0] = wptype(2) * c[0]
        c1[0] = wptype(2) * c[1]
        c2[0] = wptype(2) * c[2]

    kernel = getkernel(check_cross, suffix=dtype.__name__)

    if register_kernels:
        return

    s3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v3 = wp.array(randvals((1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    c0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    c1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    c2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                s3,
                v3,
            ],
            outputs=[c0, c1, c2],
            device=device,
        )

    result = 2 * np.cross(s3.numpy(), v3.numpy())[0]
    assert_np_equal(c0.numpy()[0], result[0], tol=10 * tol)
    assert_np_equal(c1.numpy()[0], result[1], tol=10 * tol)
    assert_np_equal(c2.numpy()[0], result[2], tol=10 * tol)

    if dtype in np_float_types:
        # c.x = sy vz - sz vy
        # c.y = sz vx - sx vz
        # c.z = sx vy - sy vx

        # ( d/dsx d/dsy d/dsz )c.x = ( 0 vz -vy )
        # ( d/dsx d/dsy d/dsz )c.y = ( -vz 0 vx )
        # ( d/dsx d/dsy d/dsz )c.z = ( vy -vx 0 )

        # ( d/dvx d/dvy d/dvz )c.x = (0 -sz sy)
        # ( d/dvx d/dvy d/dvz )c.y = (sz 0 -sx)
        # ( d/dvx d/dvy d/dvz )c.z = (-sy sx 0)

        tape.backward(loss=c0)
        assert_np_equal(
            tape.gradients[s3].numpy(), 2.0 * np.array([0, v3.numpy()[0, 2], -v3.numpy()[0, 1]]), tol=10 * tol
        )
        assert_np_equal(
            tape.gradients[v3].numpy(), 2.0 * np.array([0, -s3.numpy()[0, 2], s3.numpy()[0, 1]]), tol=10 * tol
        )
        tape.zero()

        tape.backward(loss=c1)
        assert_np_equal(
            tape.gradients[s3].numpy(), 2.0 * np.array([-v3.numpy()[0, 2], 0, v3.numpy()[0, 0]]), tol=10 * tol
        )
        assert_np_equal(
            tape.gradients[v3].numpy(), 2.0 * np.array([s3.numpy()[0, 2], 0, -s3.numpy()[0, 0]]), tol=10 * tol
        )
        tape.zero()

        tape.backward(loss=c2)
        assert_np_equal(
            tape.gradients[s3].numpy(), 2.0 * np.array([v3.numpy()[0, 1], -v3.numpy()[0, 0], 0]), tol=10 * tol
        )
        assert_np_equal(
            tape.gradients[v3].numpy(), 2.0 * np.array([-s3.numpy()[0, 1], s3.numpy()[0, 0], 0]), tol=10 * tol
        )
        tape.zero()


def test_minmax(test, device, dtype, register_kernels=False):
    np.random.seed(123)

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

    a = wp.array(randvals((10, 14), dtype), dtype=wptype, requires_grad=True, device=device)
    b = wp.array(randvals((10, 14), dtype), dtype=wptype, requires_grad=True, device=device)

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


# Test matrix constructors using explicit type (float16)
# note that these tests are specifically not using generics / closure
# args to create kernels dynamically (like the rest of this file)
# as those use different code paths to resolve arg types which
# has lead to regressions.
@wp.kernel
def test_constructors_explicit_precision():
    # construction for custom matrix types
    ones = wp.vector(wp.float16(1.0), length=2)
    zeros = wp.vector(length=2, dtype=wp.float16)
    custom = wp.vector(wp.float16(0.0), wp.float16(1.0))

    for i in range(2):
        wp.expect_eq(ones[i], wp.float16(1.0))
        wp.expect_eq(zeros[i], wp.float16(0.0))
        wp.expect_eq(custom[i], wp.float16(i))


# Same as above but with a default (float/int) type
# which tests some different code paths that
# need to ensure types are correctly canonicalized
# during codegen
@wp.kernel
def test_constructors_default_precision():
    # construction for custom matrix types
    ones = wp.vector(1.0, length=2)
    zeros = wp.vector(length=2, dtype=float)
    custom = wp.vector(0.0, 1.0)

    for i in range(2):
        wp.expect_eq(ones[i], 1.0)
        wp.expect_eq(zeros[i], 0.0)
        wp.expect_eq(custom[i], float(i))


def register(parent):
    devices = get_test_devices()

    class TestVec(parent):
        pass

    add_kernel_test(TestVec, test_constructors_explicit_precision, dim=1, devices=devices)
    add_kernel_test(TestVec, test_constructors_default_precision, dim=1, devices=devices)

    for dtype in np_unsigned_int_types:
        add_function_test_register_kernel(
            TestVec,
            f"test_subtraction_unsigned_{dtype.__name__}",
            test_subtraction_unsigned,
            devices=devices,
            dtype=dtype,
        )

    for dtype in np_signed_int_types + np_float_types:
        add_function_test_register_kernel(
            TestVec, f"test_negation_{dtype.__name__}", test_negation, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_subtraction_{dtype.__name__}", test_subtraction, devices=devices, dtype=dtype
        )

    for dtype in np_float_types:
        add_function_test_register_kernel(
            TestVec, f"test_crossproduct_{dtype.__name__}", test_crossproduct, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_length_{dtype.__name__}", test_length, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_normalize_{dtype.__name__}", test_normalize, devices=devices, dtype=dtype
        )

    for dtype in np_scalar_types:
        add_function_test(TestVec, f"test_arrays_{dtype.__name__}", test_arrays, devices=devices, dtype=dtype)
        add_function_test_register_kernel(
            TestVec, f"test_constructors_{dtype.__name__}", test_constructors, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_anon_type_instance_{dtype.__name__}", test_anon_type_instance, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_indexing_{dtype.__name__}", test_indexing, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_equality_{dtype.__name__}", test_equality, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec,
            f"test_scalar_multiplication_{dtype.__name__}",
            test_scalar_multiplication,
            devices=devices,
            dtype=dtype,
        )
        add_function_test_register_kernel(
            TestVec,
            f"test_scalar_multiplication_rightmul_{dtype.__name__}",
            test_scalar_multiplication_rightmul,
            devices=devices,
            dtype=dtype,
        )
        add_function_test_register_kernel(
            TestVec, f"test_cw_multiplication_{dtype.__name__}", test_cw_multiplication, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_scalar_division_{dtype.__name__}", test_scalar_division, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_cw_division_{dtype.__name__}", test_cw_division, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_addition_{dtype.__name__}", test_addition, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_dotproduct_{dtype.__name__}", test_dotproduct, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_equivalent_types_{dtype.__name__}", test_equivalent_types, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_conversions_{dtype.__name__}", test_conversions, devices=devices, dtype=dtype
        )
        add_function_test_register_kernel(
            TestVec, f"test_constants_{dtype.__name__}", test_constants, devices=devices, dtype=dtype
        )

        # the kernels in this test compile incredibly slowly...
        # add_function_test_register_kernel(TestVec, f"test_minmax_{dtype.__name__}", test_minmax, devices=devices, dtype=dtype)

    return TestVec


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=True)
