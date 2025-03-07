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
from typing import Any

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

np_float_types = [np.float16, np.float32, np.float64]


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


def test_anon_constructor_error_length_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.vector(
            wp.vector(length=2, dtype=float),
            length=3,
            dtype=float,
        )

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible vector of length 3 given when copy constructing a vector of length 2$",
    ):
        wp.launch(
            kernel,
            dim=1,
            inputs=[],
            device=device,
        )


def test_anon_constructor_error_numeric_arg_missing(test, device):
    @wp.kernel
    def kernel():
        wp.vector(1.0, 2.0, length=12345)

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible number of values given \(2\) when constructing a vector of length 12345$",
    ):
        wp.launch(
            kernel,
            dim=1,
            inputs=[],
            device=device,
        )


def test_anon_constructor_error_length_arg_missing(test, device):
    @wp.kernel
    def kernel():
        wp.vector()

    with test.assertRaisesRegex(
        RuntimeError,
        r"the `length` argument must be specified when zero-initializing a vector$",
    ):
        wp.launch(
            kernel,
            dim=1,
            inputs=[],
            device=device,
        )


def test_anon_constructor_error_numeric_args_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.vector(1.0, 2)

    with test.assertRaisesRegex(
        RuntimeError,
        r"all values given when constructing a vector must have the same type$",
    ):
        wp.launch(
            kernel,
            dim=1,
            inputs=[],
            device=device,
        )


def test_tpl_constructor_error_incompatible_sizes(test, device):
    @wp.kernel
    def kernel():
        wp.vec3(wp.vec2(1.0, 2.0))

    with test.assertRaisesRegex(
        RuntimeError, "incompatible vector of length 3 given when copy constructing a vector of length 2"
    ):
        wp.launch(
            kernel,
            dim=1,
            inputs=[],
            device=device,
        )


def test_tpl_constructor_error_numeric_args_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.vec2(1.0, 2)

    with test.assertRaisesRegex(
        RuntimeError,
        r"all values given when constructing a vector must have the same type$",
    ):
        wp.launch(
            kernel,
            dim=1,
            inputs=[],
            device=device,
        )


def test_negation(test, device, dtype, register_kernels=False):
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

    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5_np = randvals(rng, (1, 5), dtype)
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


def test_subtraction_unsigned(test, device, dtype, register_kernels=False):
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


def test_length(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

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

    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)

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
    v2 = wp.array(randvals(rng, (1, 2), dtype), dtype=vec2, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, (1, 4), dtype), dtype=vec4, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, (1, 5), dtype), dtype=vec5, requires_grad=True, device=device)

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
    rng = np.random.default_rng(123)

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

    s3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, (1, 3), dtype), dtype=vec3, requires_grad=True, device=device)
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


def test_casting_constructors(test, device, dtype, register_kernels=False):
    np_type = np.dtype(dtype)
    wp_type = wp.types.np_dtype_to_warp_type[np_type]
    vec3 = wp.types.vector(length=3, dtype=wp_type)

    np16 = np.dtype(np.float16)
    wp16 = wp.types.np_dtype_to_warp_type[np16]

    np32 = np.dtype(np.float32)
    wp32 = wp.types.np_dtype_to_warp_type[np32]

    np64 = np.dtype(np.float64)
    wp64 = wp.types.np_dtype_to_warp_type[np64]

    def cast_float16(a: wp.array(dtype=wp_type, ndim=2), b: wp.array(dtype=wp16, ndim=2)):
        tid = wp.tid()

        v1 = vec3(a[tid, 0], a[tid, 1], a[tid, 2])
        v2 = wp.vector(v1, dtype=wp16)

        b[tid, 0] = v2[0]
        b[tid, 1] = v2[1]
        b[tid, 2] = v2[2]

    def cast_float32(a: wp.array(dtype=wp_type, ndim=2), b: wp.array(dtype=wp32, ndim=2)):
        tid = wp.tid()

        v1 = vec3(a[tid, 0], a[tid, 1], a[tid, 2])
        v2 = wp.vector(v1, dtype=wp32)

        b[tid, 0] = v2[0]
        b[tid, 1] = v2[1]
        b[tid, 2] = v2[2]

    def cast_float64(a: wp.array(dtype=wp_type, ndim=2), b: wp.array(dtype=wp64, ndim=2)):
        tid = wp.tid()

        v1 = vec3(a[tid, 0], a[tid, 1], a[tid, 2])
        v2 = wp.vector(v1, dtype=wp64)

        b[tid, 0] = v2[0]
        b[tid, 1] = v2[1]
        b[tid, 2] = v2[2]

    kernel_16 = getkernel(cast_float16, suffix=dtype.__name__)
    kernel_32 = getkernel(cast_float32, suffix=dtype.__name__)
    kernel_64 = getkernel(cast_float64, suffix=dtype.__name__)

    if register_kernels:
        return

    # check casting to float 16
    a = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, requires_grad=True, device=device)
    b = wp.array(np.zeros((1, 3), dtype=np16), dtype=wp16, requires_grad=True, device=device)
    b_result = np.ones((1, 3), dtype=np16)
    b_grad = wp.array(np.ones((1, 3), dtype=np16), dtype=wp16, device=device)
    a_grad = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=kernel_16, dim=1, inputs=[a, b], device=device)

    tape.backward(grads={b: b_grad})
    out = tape.gradients[a].numpy()

    assert_np_equal(b.numpy(), b_result)
    assert_np_equal(out, a_grad.numpy())

    # check casting to float 32
    a = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, requires_grad=True, device=device)
    b = wp.array(np.zeros((1, 3), dtype=np32), dtype=wp32, requires_grad=True, device=device)
    b_result = np.ones((1, 3), dtype=np32)
    b_grad = wp.array(np.ones((1, 3), dtype=np32), dtype=wp32, device=device)
    a_grad = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=kernel_32, dim=1, inputs=[a, b], device=device)

    tape.backward(grads={b: b_grad})
    out = tape.gradients[a].numpy()

    assert_np_equal(b.numpy(), b_result)
    assert_np_equal(out, a_grad.numpy())

    # check casting to float 64
    a = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, requires_grad=True, device=device)
    b = wp.array(np.zeros((1, 3), dtype=np64), dtype=wp64, requires_grad=True, device=device)
    b_result = np.ones((1, 3), dtype=np64)
    b_grad = wp.array(np.ones((1, 3), dtype=np64), dtype=wp64, device=device)
    a_grad = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=kernel_64, dim=1, inputs=[a, b], device=device)

    tape.backward(grads={b: b_grad})
    out = tape.gradients[a].numpy()

    assert_np_equal(b.numpy(), b_result)
    assert_np_equal(out, a_grad.numpy())


def test_vec_assign(test, device, dtype, register_kernels=False):
    np_type = np.dtype(dtype)
    wp_type = wp.types.np_dtype_to_warp_type[np_type]

    vec2 = wp.types.vector(length=2, dtype=wp_type)
    vec3 = wp.types.vector(length=3, dtype=wp_type)
    vec4 = wp.types.vector(length=4, dtype=wp_type)

    def vectest_read_write_store(
        x: wp.array(dtype=wp_type), a: wp.array(dtype=vec2), b: wp.array(dtype=vec3), c: wp.array(dtype=vec4)
    ):
        tid = wp.tid()

        t = a[tid]
        t[0] = x[tid]
        a[tid] = t

        u = b[tid]
        u[1] = x[tid]
        b[tid] = u

        v = c[tid]
        v[2] = x[tid]
        c[tid] = v

    def vectest_in_register(
        x: wp.array(dtype=wp_type), y: wp.array(dtype=vec3), a: wp.array(dtype=vec2), b: wp.array(dtype=vec3)
    ):
        tid = wp.tid()

        f = vec3(wp_type(0.0))
        b_vec = b[tid]
        f[0] = b_vec[1]
        f[2] = b_vec[0] * b_vec[1]
        y[tid] = f

        g = wp_type(0.0)
        a_vec = a[tid]
        g = a_vec[0] + a_vec[1]
        x[tid] = g

    def vectest_in_register_overwrite(x: wp.array(dtype=vec3), a: wp.array(dtype=vec3)):
        tid = wp.tid()

        f = vec3(wp_type(0.0))
        a_vec = a[tid]
        f = a_vec
        f[1] = wp_type(3.0)

        x[tid] = f

    def vectest_component(x: wp.array(dtype=vec3), y: wp.array(dtype=wp_type)):
        i = wp.tid()

        a = vec3(wp_type(0.0))
        a.x = wp_type(1.0) * y[i]
        a.y = wp_type(2.0) * y[i]
        a.z = wp_type(3.0) * y[i]
        x[i] = a

    kernel_read_write_store = getkernel(vectest_read_write_store, suffix=dtype.__name__)
    kernel_in_register = getkernel(vectest_in_register, suffix=dtype.__name__)
    kernel_in_register_overwrite = getkernel(vectest_in_register_overwrite, suffix=dtype.__name__)
    kernel_component = getkernel(vectest_component, suffix=dtype.__name__)

    if register_kernels:
        return

    a = wp.ones(1, dtype=vec2, device=device, requires_grad=True)
    b = wp.ones(1, dtype=vec3, device=device, requires_grad=True)
    c = wp.ones(1, dtype=vec4, device=device, requires_grad=True)
    x = wp.full(1, value=2.0, dtype=wp_type, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel_read_write_store, dim=1, inputs=[x, a, b, c], device=device)

    tape.backward(
        grads={
            a: wp.ones_like(a, requires_grad=False),
            b: wp.ones_like(b, requires_grad=False),
            c: wp.ones_like(c, requires_grad=False),
        }
    )

    assert_np_equal(a.numpy(), np.array([[2.0, 1.0]], dtype=np_type))
    assert_np_equal(b.numpy(), np.array([[1.0, 2.0, 1.0]], dtype=np_type))
    assert_np_equal(c.numpy(), np.array([[1.0, 1.0, 2.0, 1.0]], dtype=np_type))
    assert_np_equal(x.grad.numpy(), np.array([3.0], dtype=np_type))

    tape.reset()

    a = wp.ones(1, dtype=vec2, device=device, requires_grad=True)
    b = wp.ones(1, dtype=vec3, device=device, requires_grad=True)
    x = wp.zeros(1, dtype=wp_type, device=device, requires_grad=True)
    y = wp.zeros(1, dtype=vec3, device=device, requires_grad=True)

    with tape:
        wp.launch(kernel_in_register, dim=1, inputs=[x, y, a, b], device=device)

    tape.backward(grads={x: wp.ones_like(x, requires_grad=False), y: wp.ones_like(y, requires_grad=False)})

    assert_np_equal(x.numpy(), np.array([2.0], dtype=np_type))
    assert_np_equal(y.numpy(), np.array([[1.0, 0.0, 1.0]], dtype=np_type))
    assert_np_equal(a.grad.numpy(), np.array([[1.0, 1.0]], dtype=np_type))
    assert_np_equal(b.grad.numpy(), np.array([[1.0, 2.0, 0.0]], dtype=np_type))

    tape.reset()

    x = wp.zeros(1, dtype=vec3, device=device, requires_grad=True)
    y = wp.ones(1, dtype=wp_type, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel_component, dim=1, inputs=[x, y], device=device)

    tape.backward(grads={x: wp.ones_like(x, requires_grad=False)})

    assert_np_equal(x.numpy(), np.array([[1.0, 2.0, 3.0]], dtype=np_type))
    assert_np_equal(y.grad.numpy(), np.array([6.0], dtype=np_type))

    tape.reset()

    x = wp.zeros(1, dtype=vec3, device=device, requires_grad=True)
    a = wp.ones(1, dtype=vec3, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel_in_register_overwrite, dim=1, inputs=[x, a], device=device)

    tape.backward(grads={x: wp.ones_like(x, requires_grad=False)})

    assert_np_equal(x.numpy(), np.array([[1.0, 3.0, 1.0]], dtype=np_type))
    assert_np_equal(a.grad.numpy(), np.array([[1.0, 0.0, 1.0]], dtype=np_type))


@wp.kernel
def test_vector_constructor_value_func():
    a = wp.vec2()
    b = wp.vector(a, dtype=wp.float16)
    c = wp.vector(a)
    d = wp.vector(a, length=2)


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


@wp.kernel
def test_vector_mutation(expected: wp.types.vector(length=10, dtype=float)):
    v = wp.vector(length=10, dtype=float)

    # test element indexing
    v[0] = 1.0

    for i in range(1, 10):
        v[i] = float(i) + 1.0

    wp.expect_eq(v, expected)


CONSTANT_LENGTH = wp.constant(10)


# tests that we can use global constants in length keyword argument
# for vector constructor
@wp.kernel
def test_constructors_constant_length():
    v = wp.vector(length=(CONSTANT_LENGTH), dtype=float)

    for i in range(CONSTANT_LENGTH):
        v[i] = float(i)


Vec123 = wp.vec(123, dtype=wp.float16)


@wp.kernel
def vector_len_kernel(
    v1: wp.vec2,
    v2: wp.vec(3, float),
    v3: wp.vec(Any, float),
    v4: Vec123,
    out: wp.array(dtype=int),
):
    length = wp.static(len(v1))
    wp.expect_eq(len(v1), 2)
    out[0] = len(v1)

    length = len(v2)
    wp.expect_eq(wp.static(len(v2)), 3)
    out[1] = len(v2)

    length = len(v3)
    wp.expect_eq(len(v3), 4)
    out[2] = wp.static(len(v3))

    length = wp.static(len(v4))
    wp.expect_eq(wp.static(len(v4)), 123)
    out[3] = wp.static(len(v4))

    foo = wp.vec2()
    length = len(foo)
    wp.expect_eq(len(foo), 2)
    out[4] = len(foo)


def test_vector_len(test, device):
    v1 = wp.vec2()
    v2 = wp.vec3()
    v3 = wp.vec4()
    v4 = Vec123()
    out = wp.empty(5, dtype=int, device=device)
    wp.launch(vector_len_kernel, dim=(1,), inputs=(v1, v2, v3, v4), outputs=(out,), device=device)

    test.assertEqual(out.numpy()[0], 2)
    test.assertEqual(out.numpy()[1], 3)
    test.assertEqual(out.numpy()[2], 4)
    test.assertEqual(out.numpy()[3], 123)
    test.assertEqual(out.numpy()[4], 2)


@wp.kernel
def vector_augassign_kernel(
    a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), c: wp.array(dtype=wp.vec3), d: wp.array(dtype=wp.vec3)
):
    i = wp.tid()

    v1 = wp.vec3()
    v2 = b[i]

    v1[0] += v2[0]
    v1[1] += v2[1]
    v1[2] += v2[2]

    a[i] = v1

    v3 = wp.vec3()
    v4 = d[i]

    v3[0] -= v4[0]
    v3[1] -= v4[1]
    v3[2] -= v4[2]

    c[i] = v3


def test_vector_augassign(test, device):
    N = 3

    a = wp.zeros(N, dtype=wp.vec3, requires_grad=True)
    b = wp.ones(N, dtype=wp.vec3, requires_grad=True)

    c = wp.zeros(N, dtype=wp.vec3, requires_grad=True)
    d = wp.ones(N, dtype=wp.vec3, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(vector_augassign_kernel, N, inputs=[a, b, c, d])

    tape.backward(grads={a: wp.ones_like(a), c: wp.ones_like(c)})

    assert_np_equal(a.numpy(), wp.ones_like(a).numpy())
    assert_np_equal(a.grad.numpy(), wp.ones_like(a).numpy())
    assert_np_equal(b.grad.numpy(), wp.ones_like(a).numpy())

    assert_np_equal(c.numpy(), -wp.ones_like(c).numpy())
    assert_np_equal(c.grad.numpy(), wp.ones_like(c).numpy())
    assert_np_equal(d.grad.numpy(), -wp.ones_like(d).numpy())


devices = get_test_devices()


class TestVec(unittest.TestCase):
    def test_tpl_ops_with_anon(self):
        vec3i = wp.vec(3, dtype=int)

        v = wp.vec3i(1, 2, 3)
        v += vec3i(2, 3, 4)
        v -= vec3i(3, 4, 5)
        self.assertSequenceEqual(v, (0, 1, 2))

        v = vec3i(1, 2, 3)
        v += wp.vec3i(2, 3, 4)
        v -= wp.vec3i(3, 4, 5)
        self.assertSequenceEqual(v, (0, 1, 2))


add_kernel_test(TestVec, test_vector_constructor_value_func, dim=1, devices=devices)
add_kernel_test(TestVec, test_constructors_explicit_precision, dim=1, devices=devices)
add_kernel_test(TestVec, test_constructors_default_precision, dim=1, devices=devices)
add_kernel_test(TestVec, test_constructors_constant_length, dim=1, devices=devices)

vec10 = wp.types.vector(length=10, dtype=float)
add_kernel_test(
    TestVec,
    test_vector_mutation,
    dim=1,
    inputs=[vec10(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)],
    devices=devices,
)

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
    add_function_test_register_kernel(
        TestVec,
        f"test_casting_constructors_{dtype.__name__}",
        test_casting_constructors,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestVec,
        f"test_vec_assign_{dtype.__name__}",
        test_vec_assign,
        devices=devices,
        dtype=dtype,
    )

add_function_test(
    TestVec,
    "test_anon_constructor_error_length_mismatch",
    test_anon_constructor_error_length_mismatch,
    devices=devices,
)
add_function_test(
    TestVec,
    "test_anon_constructor_error_numeric_arg_missing",
    test_anon_constructor_error_numeric_arg_missing,
    devices=devices,
)
add_function_test(
    TestVec,
    "test_anon_constructor_error_length_arg_missing",
    test_anon_constructor_error_length_arg_missing,
    devices=devices,
)
add_function_test(
    TestVec,
    "test_anon_constructor_error_numeric_args_mismatch",
    test_anon_constructor_error_numeric_args_mismatch,
    devices=devices,
)
add_function_test(
    TestVec,
    "test_tpl_constructor_error_incompatible_sizes",
    test_tpl_constructor_error_incompatible_sizes,
    devices=devices,
)
add_function_test(
    TestVec,
    "test_tpl_constructor_error_numeric_args_mismatch",
    test_tpl_constructor_error_numeric_args_mismatch,
    devices=devices,
)
add_function_test(
    TestVec,
    "test_vector_len",
    test_vector_len,
    devices=devices,
)
add_function_test(
    TestVec,
    "test_vector_augassign",
    test_vector_augassign,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
