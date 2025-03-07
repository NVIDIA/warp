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
import warp.sim
from warp.tests.unittest_utils import *

np_float_types = [np.float32, np.float64, np.float16]

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


############################################################


def test_constructors(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(length=3, dtype=wptype)
    quat = wp.types.quaternion(dtype=wptype)

    def check_component_constructor(
        input: wp.array(dtype=wptype),
        q: wp.array(dtype=wptype),
    ):
        qresult = quat(input[0], input[1], input[2], input[3])

        # multiply the output by 2 so we've got something to backpropagate:
        q[0] = wptype(2) * qresult[0]
        q[1] = wptype(2) * qresult[1]
        q[2] = wptype(2) * qresult[2]
        q[3] = wptype(2) * qresult[3]

    def check_vector_constructor(
        input: wp.array(dtype=wptype),
        q: wp.array(dtype=wptype),
    ):
        qresult = quat(vec3(input[0], input[1], input[2]), input[3])

        # multiply the output by 2 so we've got something to backpropagate:
        q[0] = wptype(2) * qresult[0]
        q[1] = wptype(2) * qresult[1]
        q[2] = wptype(2) * qresult[2]
        q[3] = wptype(2) * qresult[3]

    kernel = getkernel(check_component_constructor, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)
    vec_kernel = getkernel(check_vector_constructor, suffix=dtype.__name__)

    if register_kernels:
        return

    input = wp.array(rng.standard_normal(size=4).astype(dtype), requires_grad=True, device=device)
    output = wp.zeros_like(input)
    wp.launch(kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy(), 2 * input.numpy(), tol=tol)

    for i in range(4):
        cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[input], outputs=[output], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[output, i], outputs=[cmp], device=device)
        tape.backward(loss=cmp)
        expectedgrads = np.zeros(len(input))
        expectedgrads[i] = 2
        assert_np_equal(tape.gradients[input].numpy(), expectedgrads)
        tape.zero()

    input = wp.array(rng.standard_normal(size=4).astype(dtype), requires_grad=True, device=device)
    output = wp.zeros_like(input)
    wp.launch(vec_kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy(), 2 * input.numpy(), tol=tol)

    for i in range(4):
        cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(vec_kernel, dim=1, inputs=[input], outputs=[output], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[output, i], outputs=[cmp], device=device)
        tape.backward(loss=cmp)
        expectedgrads = np.zeros(len(input))
        expectedgrads[i] = 2
        assert_np_equal(tape.gradients[input].numpy(), expectedgrads)
        tape.zero()


def test_casting_constructors(test, device, dtype, register_kernels=False):
    np_type = np.dtype(dtype)
    wp_type = wp.types.np_dtype_to_warp_type[np_type]
    quat = wp.types.quaternion(dtype=wp_type)

    np16 = np.dtype(np.float16)
    wp16 = wp.types.np_dtype_to_warp_type[np16]

    np32 = np.dtype(np.float32)
    wp32 = wp.types.np_dtype_to_warp_type[np32]

    np64 = np.dtype(np.float64)
    wp64 = wp.types.np_dtype_to_warp_type[np64]

    def cast_float16(a: wp.array(dtype=wp_type, ndim=2), b: wp.array(dtype=wp16, ndim=2)):
        tid = wp.tid()

        q1 = quat(a[tid, 0], a[tid, 1], a[tid, 2], a[tid, 3])
        q2 = wp.quaternion(q1, dtype=wp16)

        b[tid, 0] = q2[0]
        b[tid, 1] = q2[1]
        b[tid, 2] = q2[2]
        b[tid, 3] = q2[3]

    def cast_float32(a: wp.array(dtype=wp_type, ndim=2), b: wp.array(dtype=wp32, ndim=2)):
        tid = wp.tid()

        q1 = quat(a[tid, 0], a[tid, 1], a[tid, 2], a[tid, 3])
        q2 = wp.quaternion(q1, dtype=wp32)

        b[tid, 0] = q2[0]
        b[tid, 1] = q2[1]
        b[tid, 2] = q2[2]
        b[tid, 3] = q2[3]

    def cast_float64(a: wp.array(dtype=wp_type, ndim=2), b: wp.array(dtype=wp64, ndim=2)):
        tid = wp.tid()

        q1 = quat(a[tid, 0], a[tid, 1], a[tid, 2], a[tid, 3])
        q2 = wp.quaternion(q1, dtype=wp64)

        b[tid, 0] = q2[0]
        b[tid, 1] = q2[1]
        b[tid, 2] = q2[2]
        b[tid, 3] = q2[3]

    kernel_16 = getkernel(cast_float16, suffix=dtype.__name__)
    kernel_32 = getkernel(cast_float32, suffix=dtype.__name__)
    kernel_64 = getkernel(cast_float64, suffix=dtype.__name__)

    if register_kernels:
        return

    # check casting to float 16
    a = wp.array(np.ones((1, 4), dtype=np_type), dtype=wp_type, requires_grad=True, device=device)
    b = wp.array(np.zeros((1, 4), dtype=np16), dtype=wp16, requires_grad=True, device=device)
    b_result = np.ones((1, 4), dtype=np16)
    b_grad = wp.array(np.ones((1, 4), dtype=np16), dtype=wp16, device=device)
    a_grad = wp.array(np.ones((1, 4), dtype=np_type), dtype=wp_type, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=kernel_16, dim=1, inputs=[a, b], device=device)

    tape.backward(grads={b: b_grad})
    out = tape.gradients[a].numpy()

    assert_np_equal(b.numpy(), b_result)
    assert_np_equal(out, a_grad.numpy())

    # check casting to float 32
    a = wp.array(np.ones((1, 4), dtype=np_type), dtype=wp_type, requires_grad=True, device=device)
    b = wp.array(np.zeros((1, 4), dtype=np32), dtype=wp32, requires_grad=True, device=device)
    b_result = np.ones((1, 4), dtype=np32)
    b_grad = wp.array(np.ones((1, 4), dtype=np32), dtype=wp32, device=device)
    a_grad = wp.array(np.ones((1, 4), dtype=np_type), dtype=wp_type, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=kernel_32, dim=1, inputs=[a, b], device=device)

    tape.backward(grads={b: b_grad})
    out = tape.gradients[a].numpy()

    assert_np_equal(b.numpy(), b_result)
    assert_np_equal(out, a_grad.numpy())

    # check casting to float 64
    a = wp.array(np.ones((1, 4), dtype=np_type), dtype=wp_type, requires_grad=True, device=device)
    b = wp.array(np.zeros((1, 4), dtype=np64), dtype=wp64, requires_grad=True, device=device)
    b_result = np.ones((1, 4), dtype=np64)
    b_grad = wp.array(np.ones((1, 4), dtype=np64), dtype=wp64, device=device)
    a_grad = wp.array(np.ones((1, 4), dtype=np_type), dtype=wp_type, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=kernel_64, dim=1, inputs=[a, b], device=device)

    tape.backward(grads={b: b_grad})
    out = tape.gradients[a].numpy()

    assert_np_equal(b.numpy(), b_result)
    assert_np_equal(out, a_grad.numpy())


def test_inverse(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_quat_inverse(
        input: wp.array(dtype=wptype),
        shouldbeidentity: wp.array(dtype=quat),
        q: wp.array(dtype=wptype),
    ):
        qread = quat(input[0], input[1], input[2], input[3])
        qresult = wp.quat_inverse(qread)

        # this inverse should work for normalized quaternions:
        shouldbeidentity[0] = wp.normalize(qread) * wp.quat_inverse(wp.normalize(qread))

        # multiply the output by 2 so we've got something to backpropagate:
        q[0] = wptype(2) * qresult[0]
        q[1] = wptype(2) * qresult[1]
        q[2] = wptype(2) * qresult[2]
        q[3] = wptype(2) * qresult[3]

    kernel = getkernel(check_quat_inverse, suffix=dtype.__name__)

    if register_kernels:
        return

    input = wp.array(rng.standard_normal(size=4).astype(dtype), requires_grad=True, device=device)
    shouldbeidentity = wp.array(np.zeros((1, 4)), dtype=quat, requires_grad=True, device=device)
    output = wp.zeros_like(input)
    wp.launch(kernel, dim=1, inputs=[input], outputs=[shouldbeidentity, output], device=device)

    assert_np_equal(shouldbeidentity.numpy(), np.array([0, 0, 0, 1]), tol=tol)

    for i in range(4):
        cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[input], outputs=[shouldbeidentity, output], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[output, i], outputs=[cmp], device=device)
        tape.backward(loss=cmp)
        expectedgrads = np.zeros(len(input))
        expectedgrads[i] = -2 if i != 3 else 2
        assert_np_equal(tape.gradients[input].numpy(), expectedgrads)
        tape.zero()


def test_dotproduct(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)

    def check_quat_dot(
        s: wp.array(dtype=quat),
        v: wp.array(dtype=quat),
        dot: wp.array(dtype=wptype),
    ):
        dot[0] = wptype(2) * wp.dot(v[0], s[0])

    dotkernel = getkernel(check_quat_dot, suffix=dtype.__name__)
    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=4).astype(dtype), dtype=quat, requires_grad=True, device=device)
    v = wp.array(rng.standard_normal(size=4).astype(dtype), dtype=quat, requires_grad=True, device=device)
    dot = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            dotkernel,
            dim=1,
            inputs=[
                s,
                v,
            ],
            outputs=[dot],
            device=device,
        )

    assert_np_equal(dot.numpy()[0], 2.0 * (v.numpy() * s.numpy()).sum(), tol=tol)

    tape.backward(loss=dot)
    sgrads = tape.gradients[s].numpy()[0]
    expected_grads = 2.0 * v.numpy()[0]
    assert_np_equal(sgrads, expected_grads, tol=10 * tol)

    vgrads = tape.gradients[v].numpy()[0]
    expected_grads = 2.0 * s.numpy()[0]
    assert_np_equal(vgrads, expected_grads, tol=tol)


def test_length(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-7,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)

    def check_quat_length(
        q: wp.array(dtype=quat),
        l: wp.array(dtype=wptype),
        l2: wp.array(dtype=wptype),
    ):
        l[0] = wptype(2) * wp.length(q[0])
        l2[0] = wptype(2) * wp.length_sq(q[0])

    kernel = getkernel(check_quat_length, suffix=dtype.__name__)

    if register_kernels:
        return

    q = wp.array(rng.standard_normal(size=4).astype(dtype), dtype=quat, requires_grad=True, device=device)
    l = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                q,
            ],
            outputs=[l, l2],
            device=device,
        )

    assert_np_equal(l.numpy()[0], 2 * np.linalg.norm(q.numpy()), tol=10 * tol)
    assert_np_equal(l2.numpy()[0], 2 * np.linalg.norm(q.numpy()) ** 2, tol=10 * tol)

    tape.backward(loss=l)
    grad = tape.gradients[q].numpy()[0]
    expected_grad = 2 * q.numpy()[0] / np.linalg.norm(q.numpy())
    assert_np_equal(grad, expected_grad, tol=10 * tol)
    tape.zero()

    tape.backward(loss=l2)
    grad = tape.gradients[q].numpy()[0]
    expected_grad = 4 * q.numpy()[0]
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
    quat = wp.types.quaternion(dtype=wptype)

    def check_normalize(
        q: wp.array(dtype=quat),
        n0: wp.array(dtype=wptype),
        n1: wp.array(dtype=wptype),
        n2: wp.array(dtype=wptype),
        n3: wp.array(dtype=wptype),
    ):
        n = wptype(2) * (wp.normalize(q[0]))

        n0[0] = n[0]
        n1[0] = n[1]
        n2[0] = n[2]
        n3[0] = n[3]

    def check_normalize_alt(
        q: wp.array(dtype=quat),
        n0: wp.array(dtype=wptype),
        n1: wp.array(dtype=wptype),
        n2: wp.array(dtype=wptype),
        n3: wp.array(dtype=wptype),
    ):
        n = wptype(2) * (q[0] / wp.length(q[0]))

        n0[0] = n[0]
        n1[0] = n[1]
        n2[0] = n[2]
        n3[0] = n[3]

    normalize_kernel = getkernel(check_normalize, suffix=dtype.__name__)
    normalize_alt_kernel = getkernel(check_normalize_alt, suffix=dtype.__name__)

    if register_kernels:
        return

    # I've already tested the things I'm using in check_normalize_alt, so I'll just
    # make sure the two are giving the same results/gradients
    q = wp.array(rng.standard_normal(size=4).astype(dtype), dtype=quat, requires_grad=True, device=device)

    n0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    n0_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n1_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n2_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    n3_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    outputs0 = [
        n0,
        n1,
        n2,
        n3,
    ]
    tape0 = wp.Tape()
    with tape0:
        wp.launch(normalize_kernel, dim=1, inputs=[q], outputs=outputs0, device=device)

    outputs1 = [
        n0_alt,
        n1_alt,
        n2_alt,
        n3_alt,
    ]
    tape1 = wp.Tape()
    with tape1:
        wp.launch(
            normalize_alt_kernel,
            dim=1,
            inputs=[
                q,
            ],
            outputs=outputs1,
            device=device,
        )

    assert_np_equal(n0.numpy()[0], n0_alt.numpy()[0], tol=tol)
    assert_np_equal(n1.numpy()[0], n1_alt.numpy()[0], tol=tol)
    assert_np_equal(n2.numpy()[0], n2_alt.numpy()[0], tol=tol)
    assert_np_equal(n3.numpy()[0], n3_alt.numpy()[0], tol=tol)

    for ncmp, ncmpalt in zip(outputs0, outputs1):
        tape0.backward(loss=ncmp)
        tape1.backward(loss=ncmpalt)
        assert_np_equal(tape0.gradients[q].numpy()[0], tape1.gradients[q].numpy()[0], tol=tol)
        tape0.zero()
        tape1.zero()


def test_addition(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)

    def check_quat_add(
        q: wp.array(dtype=quat),
        v: wp.array(dtype=quat),
        r0: wp.array(dtype=wptype),
        r1: wp.array(dtype=wptype),
        r2: wp.array(dtype=wptype),
        r3: wp.array(dtype=wptype),
    ):
        result = q[0] + v[0]

        r0[0] = wptype(2) * result[0]
        r1[0] = wptype(2) * result[1]
        r2[0] = wptype(2) * result[2]
        r3[0] = wptype(2) * result[3]

    kernel = getkernel(check_quat_add, suffix=dtype.__name__)

    if register_kernels:
        return

    q = wp.array(rng.standard_normal(size=4).astype(dtype), dtype=quat, requires_grad=True, device=device)
    v = wp.array(rng.standard_normal(size=4).astype(dtype), dtype=quat, requires_grad=True, device=device)

    r0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                q,
                v,
            ],
            outputs=[r0, r1, r2, r3],
            device=device,
        )

    assert_np_equal(r0.numpy()[0], 2 * (v.numpy()[0, 0] + q.numpy()[0, 0]), tol=tol)
    assert_np_equal(r1.numpy()[0], 2 * (v.numpy()[0, 1] + q.numpy()[0, 1]), tol=tol)
    assert_np_equal(r2.numpy()[0], 2 * (v.numpy()[0, 2] + q.numpy()[0, 2]), tol=tol)
    assert_np_equal(r3.numpy()[0], 2 * (v.numpy()[0, 3] + q.numpy()[0, 3]), tol=tol)

    for i, l in enumerate([r0, r1, r2, r3]):
        tape.backward(loss=l)
        qgrads = tape.gradients[q].numpy()[0]
        expected_grads = np.zeros_like(qgrads)

        expected_grads[i] = 2
        assert_np_equal(qgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v].numpy()[0]
        assert_np_equal(vgrads, expected_grads, tol=tol)

        tape.zero()


def test_subtraction(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)

    def check_quat_sub(
        q: wp.array(dtype=quat),
        v: wp.array(dtype=quat),
        r0: wp.array(dtype=wptype),
        r1: wp.array(dtype=wptype),
        r2: wp.array(dtype=wptype),
        r3: wp.array(dtype=wptype),
    ):
        result = v[0] - q[0]

        r0[0] = wptype(2) * result[0]
        r1[0] = wptype(2) * result[1]
        r2[0] = wptype(2) * result[2]
        r3[0] = wptype(2) * result[3]

    kernel = getkernel(check_quat_sub, suffix=dtype.__name__)

    if register_kernels:
        return

    q = wp.array(rng.standard_normal(size=4).astype(dtype), dtype=quat, requires_grad=True, device=device)
    v = wp.array(rng.standard_normal(size=4).astype(dtype), dtype=quat, requires_grad=True, device=device)

    r0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                q,
                v,
            ],
            outputs=[r0, r1, r2, r3],
            device=device,
        )

    assert_np_equal(r0.numpy()[0], 2 * (v.numpy()[0, 0] - q.numpy()[0, 0]), tol=tol)
    assert_np_equal(r1.numpy()[0], 2 * (v.numpy()[0, 1] - q.numpy()[0, 1]), tol=tol)
    assert_np_equal(r2.numpy()[0], 2 * (v.numpy()[0, 2] - q.numpy()[0, 2]), tol=tol)
    assert_np_equal(r3.numpy()[0], 2 * (v.numpy()[0, 3] - q.numpy()[0, 3]), tol=tol)

    for i, l in enumerate([r0, r1, r2, r3]):
        tape.backward(loss=l)
        qgrads = tape.gradients[q].numpy()[0]
        expected_grads = np.zeros_like(qgrads)

        expected_grads[i] = -2
        assert_np_equal(qgrads, expected_grads, tol=10 * tol)

        vgrads = tape.gradients[v].numpy()[0]
        expected_grads[i] = 2
        assert_np_equal(vgrads, expected_grads, tol=tol)

        tape.zero()


def test_scalar_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)

    def check_quat_scalar_mul(
        s: wp.array(dtype=wptype),
        q: wp.array(dtype=quat),
        l0: wp.array(dtype=wptype),
        l1: wp.array(dtype=wptype),
        l2: wp.array(dtype=wptype),
        l3: wp.array(dtype=wptype),
        r0: wp.array(dtype=wptype),
        r1: wp.array(dtype=wptype),
        r2: wp.array(dtype=wptype),
        r3: wp.array(dtype=wptype),
    ):
        lresult = s[0] * q[0]
        rresult = q[0] * s[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        l0[0] = wptype(2) * lresult[0]
        l1[0] = wptype(2) * lresult[1]
        l2[0] = wptype(2) * lresult[2]
        l3[0] = wptype(2) * lresult[3]

        r0[0] = wptype(2) * rresult[0]
        r1[0] = wptype(2) * rresult[1]
        r2[0] = wptype(2) * rresult[2]
        r3[0] = wptype(2) * rresult[3]

    kernel = getkernel(check_quat_scalar_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=1).astype(dtype), requires_grad=True, device=device)
    q = wp.array(rng.standard_normal(size=(1, 4)).astype(dtype), dtype=quat, requires_grad=True, device=device)

    l0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    l3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    r0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[s, q],
            outputs=[
                l0,
                l1,
                l2,
                l3,
                r0,
                r1,
                r2,
                r3,
            ],
            device=device,
        )

    assert_np_equal(l0.numpy()[0], 2 * s.numpy()[0] * q.numpy()[0, 0], tol=tol)
    assert_np_equal(l1.numpy()[0], 2 * s.numpy()[0] * q.numpy()[0, 1], tol=tol)
    assert_np_equal(l2.numpy()[0], 2 * s.numpy()[0] * q.numpy()[0, 2], tol=tol)
    assert_np_equal(l3.numpy()[0], 2 * s.numpy()[0] * q.numpy()[0, 3], tol=tol)

    assert_np_equal(r0.numpy()[0], 2 * s.numpy()[0] * q.numpy()[0, 0], tol=tol)
    assert_np_equal(r1.numpy()[0], 2 * s.numpy()[0] * q.numpy()[0, 1], tol=tol)
    assert_np_equal(r2.numpy()[0], 2 * s.numpy()[0] * q.numpy()[0, 2], tol=tol)
    assert_np_equal(r3.numpy()[0], 2 * s.numpy()[0] * q.numpy()[0, 3], tol=tol)

    if dtype in np_float_types:
        for i, outputs in enumerate([(l0, r0), (l1, r1), (l2, r2), (l3, r3)]):
            for l in outputs:
                tape.backward(loss=l)
                sgrad = tape.gradients[s].numpy()[0]
                assert_np_equal(sgrad, 2 * q.numpy()[0, i], tol=tol)
                allgrads = tape.gradients[q].numpy()[0]
                expected_grads = np.zeros_like(allgrads)
                expected_grads[i] = s.numpy()[0] * 2
                assert_np_equal(allgrads, expected_grads, tol=10 * tol)
                tape.zero()


def test_scalar_division(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)

    def check_quat_scalar_div(
        s: wp.array(dtype=wptype),
        q: wp.array(dtype=quat),
        r0: wp.array(dtype=wptype),
        r1: wp.array(dtype=wptype),
        r2: wp.array(dtype=wptype),
        r3: wp.array(dtype=wptype),
    ):
        result = q[0] / s[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        r0[0] = wptype(2) * result[0]
        r1[0] = wptype(2) * result[1]
        r2[0] = wptype(2) * result[2]
        r3[0] = wptype(2) * result[3]

    kernel = getkernel(check_quat_scalar_div, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=1).astype(dtype), requires_grad=True, device=device)
    q = wp.array(rng.standard_normal(size=(1, 4)).astype(dtype), dtype=quat, requires_grad=True, device=device)

    r0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[s, q],
            outputs=[
                r0,
                r1,
                r2,
                r3,
            ],
            device=device,
        )
    assert_np_equal(r0.numpy()[0], 2 * q.numpy()[0, 0] / s.numpy()[0], tol=tol)
    assert_np_equal(r1.numpy()[0], 2 * q.numpy()[0, 1] / s.numpy()[0], tol=tol)
    assert_np_equal(r2.numpy()[0], 2 * q.numpy()[0, 2] / s.numpy()[0], tol=tol)
    assert_np_equal(r3.numpy()[0], 2 * q.numpy()[0, 3] / s.numpy()[0], tol=tol)

    if dtype in np_float_types:
        for i, r in enumerate([r0, r1, r2, r3]):
            tape.backward(loss=r)
            sgrad = tape.gradients[s].numpy()[0]
            assert_np_equal(sgrad, -2 * q.numpy()[0, i] / (s.numpy()[0] * s.numpy()[0]), tol=tol)

            allgrads = tape.gradients[q].numpy()[0]
            expected_grads = np.zeros_like(allgrads)
            expected_grads[i] = 2 / s.numpy()[0]
            assert_np_equal(allgrads, expected_grads, tol=10 * tol)
            tape.zero()


def test_quat_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)

    def check_quat_mul(
        s: wp.array(dtype=quat),
        q: wp.array(dtype=quat),
        r0: wp.array(dtype=wptype),
        r1: wp.array(dtype=wptype),
        r2: wp.array(dtype=wptype),
        r3: wp.array(dtype=wptype),
    ):
        result = s[0] * q[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        r0[0] = wptype(2) * result[0]
        r1[0] = wptype(2) * result[1]
        r2[0] = wptype(2) * result[2]
        r3[0] = wptype(2) * result[3]

    kernel = getkernel(check_quat_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=(1, 4)).astype(dtype), dtype=quat, requires_grad=True, device=device)
    q = wp.array(rng.standard_normal(size=(1, 4)).astype(dtype), dtype=quat, requires_grad=True, device=device)

    r0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[s, q],
            outputs=[
                r0,
                r1,
                r2,
                r3,
            ],
            device=device,
        )

    a = s.numpy()
    b = q.numpy()
    assert_np_equal(
        r0.numpy()[0], 2 * (a[0, 3] * b[0, 0] + b[0, 3] * a[0, 0] + a[0, 1] * b[0, 2] - b[0, 1] * a[0, 2]), tol=tol
    )
    assert_np_equal(
        r1.numpy()[0], 2 * (a[0, 3] * b[0, 1] + b[0, 3] * a[0, 1] + a[0, 2] * b[0, 0] - b[0, 2] * a[0, 0]), tol=tol
    )
    assert_np_equal(
        r2.numpy()[0], 2 * (a[0, 3] * b[0, 2] + b[0, 3] * a[0, 2] + a[0, 0] * b[0, 1] - b[0, 0] * a[0, 1]), tol=tol
    )
    assert_np_equal(
        r3.numpy()[0], 2 * (a[0, 3] * b[0, 3] - a[0, 0] * b[0, 0] - a[0, 1] * b[0, 1] - a[0, 2] * b[0, 2]), tol=tol
    )

    tape.backward(loss=r0)
    agrad = tape.gradients[s].numpy()[0]
    assert_np_equal(agrad, 2 * np.array([b[0, 3], b[0, 2], -b[0, 1], b[0, 0]]), tol=tol)

    bgrad = tape.gradients[q].numpy()[0]
    assert_np_equal(bgrad, 2 * np.array([a[0, 3], -a[0, 2], a[0, 1], a[0, 0]]), tol=tol)
    tape.zero()

    tape.backward(loss=r1)
    agrad = tape.gradients[s].numpy()[0]
    assert_np_equal(agrad, 2 * np.array([-b[0, 2], b[0, 3], b[0, 0], b[0, 1]]), tol=tol)

    bgrad = tape.gradients[q].numpy()[0]
    assert_np_equal(bgrad, 2 * np.array([a[0, 2], a[0, 3], -a[0, 0], a[0, 1]]), tol=tol)
    tape.zero()

    tape.backward(loss=r2)
    agrad = tape.gradients[s].numpy()[0]
    assert_np_equal(agrad, 2 * np.array([b[0, 1], -b[0, 0], b[0, 3], b[0, 2]]), tol=tol)

    bgrad = tape.gradients[q].numpy()[0]
    assert_np_equal(bgrad, 2 * np.array([-a[0, 1], a[0, 0], a[0, 3], a[0, 2]]), tol=tol)
    tape.zero()

    tape.backward(loss=r3)
    agrad = tape.gradients[s].numpy()[0]
    assert_np_equal(agrad, 2 * np.array([-b[0, 0], -b[0, 1], -b[0, 2], b[0, 3]]), tol=tol)

    bgrad = tape.gradients[q].numpy()[0]
    assert_np_equal(bgrad, 2 * np.array([-a[0, 0], -a[0, 1], -a[0, 2], a[0, 3]]), tol=tol)
    tape.zero()


def test_indexing(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)

    def check_quat_indexing(
        q: wp.array(dtype=quat),
        r0: wp.array(dtype=wptype),
        r1: wp.array(dtype=wptype),
        r2: wp.array(dtype=wptype),
        r3: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        r0[0] = wptype(2) * q[0][0]
        r1[0] = wptype(2) * q[0][1]
        r2[0] = wptype(2) * q[0][2]
        r3[0] = wptype(2) * q[0][3]

    kernel = getkernel(check_quat_indexing, suffix=dtype.__name__)

    if register_kernels:
        return

    q = wp.array(rng.standard_normal(size=(1, 4)).astype(dtype), dtype=quat, requires_grad=True, device=device)
    r0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[q], outputs=[r0, r1, r2, r3], device=device)

    for i, l in enumerate([r0, r1, r2, r3]):
        tape.backward(loss=l)
        allgrads = tape.gradients[q].numpy()[0]
        expected_grads = np.zeros_like(allgrads)
        expected_grads[i] = 2
        assert_np_equal(allgrads, expected_grads, tol=tol)
        tape.zero()

    assert_np_equal(r0.numpy()[0], 2.0 * q.numpy()[0, 0], tol=tol)
    assert_np_equal(r1.numpy()[0], 2.0 * q.numpy()[0, 1], tol=tol)
    assert_np_equal(r2.numpy()[0], 2.0 * q.numpy()[0, 2], tol=tol)
    assert_np_equal(r3.numpy()[0], 2.0 * q.numpy()[0, 3], tol=tol)


@wp.kernel
def test_assignment():
    q = wp.quat(1.0, 2.0, 3.0, 4.0)
    q[0] = 1.23
    q[1] = 2.34
    q[2] = 3.45
    q[3] = 4.56
    wp.expect_eq(q[0], 1.23)
    wp.expect_eq(q[1], 2.34)
    wp.expect_eq(q[2], 3.45)
    wp.expect_eq(q[3], 4.56)


def test_quat_lerp(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)

    def check_quat_lerp(
        s: wp.array(dtype=quat),
        q: wp.array(dtype=quat),
        t: wp.array(dtype=wptype),
        r0: wp.array(dtype=wptype),
        r1: wp.array(dtype=wptype),
        r2: wp.array(dtype=wptype),
        r3: wp.array(dtype=wptype),
    ):
        result = wp.lerp(s[0], q[0], t[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        r0[0] = wptype(2) * result[0]
        r1[0] = wptype(2) * result[1]
        r2[0] = wptype(2) * result[2]
        r3[0] = wptype(2) * result[3]

    kernel = getkernel(check_quat_lerp, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=(1, 4)).astype(dtype), dtype=quat, requires_grad=True, device=device)
    q = wp.array(rng.standard_normal(size=(1, 4)).astype(dtype), dtype=quat, requires_grad=True, device=device)
    t = wp.array(rng.uniform(size=1).astype(dtype), dtype=wptype, requires_grad=True, device=device)

    r0 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r1 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    r3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[s, q, t],
            outputs=[
                r0,
                r1,
                r2,
                r3,
            ],
            device=device,
        )

    a = s.numpy()
    b = q.numpy()
    tt = t.numpy()
    assert_np_equal(r0.numpy()[0], 2 * ((1 - tt) * a[0, 0] + tt * b[0, 0]), tol=tol)
    assert_np_equal(r1.numpy()[0], 2 * ((1 - tt) * a[0, 1] + tt * b[0, 1]), tol=tol)
    assert_np_equal(r2.numpy()[0], 2 * ((1 - tt) * a[0, 2] + tt * b[0, 2]), tol=tol)
    assert_np_equal(r3.numpy()[0], 2 * ((1 - tt) * a[0, 3] + tt * b[0, 3]), tol=tol)

    for i, l in enumerate([r0, r1, r2, r3]):
        tape.backward(loss=l)
        agrad = tape.gradients[s].numpy()[0]
        bgrad = tape.gradients[q].numpy()[0]
        tgrad = tape.gradients[t].numpy()[0]
        expected_grads = np.zeros_like(agrad)
        expected_grads[i] = 2 * (1 - tt)
        assert_np_equal(agrad, expected_grads, tol=tol)
        expected_grads[i] = 2 * tt
        assert_np_equal(bgrad, expected_grads, tol=tol)
        assert_np_equal(tgrad, 2 * (b[0, i] - a[0, i]), tol=tol)

        tape.zero()


def test_quat_rotate(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)

    def check_quat_rotate(
        q: wp.array(dtype=quat),
        v: wp.array(dtype=vec3),
        outputs: wp.array(dtype=wptype),
        outputs_inv: wp.array(dtype=wptype),
        outputs_manual: wp.array(dtype=wptype),
        outputs_inv_manual: wp.array(dtype=wptype),
    ):
        result = wp.quat_rotate(q[0], v[0])
        result_inv = wp.quat_rotate_inv(q[0], v[0])

        qv = vec3(q[0][0], q[0][1], q[0][2])
        qw = q[0][3]

        result_manual = v[0] * (wptype(2) * qw * qw - wptype(1))
        result_manual += wp.cross(qv, v[0]) * qw * wptype(2)
        result_manual += qv * wp.dot(qv, v[0]) * wptype(2)

        result_inv_manual = v[0] * (wptype(2) * qw * qw - wptype(1))
        result_inv_manual -= wp.cross(qv, v[0]) * qw * wptype(2)
        result_inv_manual += qv * wp.dot(qv, v[0]) * wptype(2)

        for i in range(3):
            # multiply outputs by 2 so we've got something to backpropagate:
            outputs[i] = wptype(2) * result[i]
            outputs_inv[i] = wptype(2) * result_inv[i]
            outputs_manual[i] = wptype(2) * result_manual[i]
            outputs_inv_manual[i] = wptype(2) * result_inv_manual[i]

    kernel = getkernel(check_quat_rotate, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    q = rng.standard_normal(size=(1, 4))
    q /= np.linalg.norm(q)
    q = wp.array(q.astype(dtype), dtype=quat, requires_grad=True, device=device)
    v = wp.array(0.5 * rng.standard_normal(size=(1, 3)).astype(dtype), dtype=vec3, requires_grad=True, device=device)

    # test values against the manually computed result:
    outputs = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    outputs_inv = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    outputs_manual = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    outputs_inv_manual = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[q, v],
        outputs=[
            outputs,
            outputs_inv,
            outputs_manual,
            outputs_inv_manual,
        ],
        device=device,
    )

    assert_np_equal(outputs.numpy(), outputs_manual.numpy(), tol=tol)
    assert_np_equal(outputs_inv.numpy(), outputs_inv_manual.numpy(), tol=tol)

    # test gradients against the manually computed result:
    for i in range(3):
        cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_inv = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_manual = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_inv_manual = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel,
                dim=1,
                inputs=[q, v],
                outputs=[
                    outputs,
                    outputs_inv,
                    outputs_manual,
                    outputs_inv_manual,
                ],
                device=device,
            )
            wp.launch(output_select_kernel, dim=1, inputs=[outputs, i], outputs=[cmp], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_inv, i], outputs=[cmp_inv], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_manual, i], outputs=[cmp_manual], device=device)
            wp.launch(
                output_select_kernel, dim=1, inputs=[outputs_inv_manual, i], outputs=[cmp_inv_manual], device=device
            )

        tape.backward(loss=cmp)
        qgrads = 1.0 * tape.gradients[q].numpy()
        vgrads = 1.0 * tape.gradients[v].numpy()
        tape.zero()
        tape.backward(loss=cmp_inv)
        qgrads_inv = 1.0 * tape.gradients[q].numpy()
        vgrads_inv = 1.0 * tape.gradients[v].numpy()
        tape.zero()
        tape.backward(loss=cmp_manual)
        qgrads_manual = 1.0 * tape.gradients[q].numpy()
        vgrads_manual = 1.0 * tape.gradients[v].numpy()
        tape.zero()
        tape.backward(loss=cmp_inv_manual)
        qgrads_inv_manual = 1.0 * tape.gradients[q].numpy()
        vgrads_inv_manual = 1.0 * tape.gradients[v].numpy()
        tape.zero()

        assert_np_equal(qgrads, qgrads_manual, tol=tol)
        assert_np_equal(vgrads, vgrads_manual, tol=tol)

        assert_np_equal(qgrads_inv, qgrads_inv_manual, tol=tol)
        assert_np_equal(vgrads_inv, vgrads_inv_manual, tol=tol)


def test_quat_to_matrix(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    quat = wp.types.quaternion(dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)

    def check_quat_to_matrix(
        q: wp.array(dtype=quat),
        outputs: wp.array(dtype=wptype),
        outputs_manual: wp.array(dtype=wptype),
    ):
        result = wp.quat_to_matrix(q[0])

        xaxis = wp.quat_rotate(
            q[0],
            vec3(
                wptype(1),
                wptype(0),
                wptype(0),
            ),
        )
        yaxis = wp.quat_rotate(
            q[0],
            vec3(
                wptype(0),
                wptype(1),
                wptype(0),
            ),
        )
        zaxis = wp.quat_rotate(
            q[0],
            vec3(
                wptype(0),
                wptype(0),
                wptype(1),
            ),
        )
        result_manual = wp.matrix_from_cols(xaxis, yaxis, zaxis)

        idx = 0
        for i in range(3):
            for j in range(3):
                # multiply outputs by 2 so we've got something to backpropagate:
                outputs[idx] = wptype(2) * result[i, j]
                outputs_manual[idx] = wptype(2) * result_manual[i, j]

                idx = idx + 1

    kernel = getkernel(check_quat_to_matrix, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    q = rng.standard_normal(size=(1, 4))
    q /= np.linalg.norm(q)
    q = wp.array(q.astype(dtype), dtype=quat, requires_grad=True, device=device)

    # test values against the manually computed result:
    outputs = wp.zeros(3 * 3, dtype=wptype, requires_grad=True, device=device)
    outputs_manual = wp.zeros(3 * 3, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[q],
        outputs=[
            outputs,
            outputs_manual,
        ],
        device=device,
    )

    assert_np_equal(outputs.numpy(), outputs_manual.numpy(), tol=tol)

    # sanity check: divide by 2 to remove that scale factor we put in there, and
    # it should be a rotation matrix
    R = 0.5 * outputs.numpy().reshape(3, 3)
    assert_np_equal(np.matmul(R, R.T), np.eye(3), tol=tol)

    # test gradients against the manually computed result:
    idx = 0
    for _i in range(3):
        for _j in range(3):
            cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
            cmp_manual = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
            tape = wp.Tape()
            with tape:
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[q],
                    outputs=[
                        outputs,
                        outputs_manual,
                    ],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outputs, idx], outputs=[cmp], device=device)
                wp.launch(
                    output_select_kernel, dim=1, inputs=[outputs_manual, idx], outputs=[cmp_manual], device=device
                )
            tape.backward(loss=cmp)
            qgrads = 1.0 * tape.gradients[q].numpy()
            tape.zero()
            tape.backward(loss=cmp_manual)
            qgrads_manual = 1.0 * tape.gradients[q].numpy()
            tape.zero()

            assert_np_equal(qgrads, qgrads_manual, tol=tol)
            idx = idx + 1


############################################################


def test_slerp_grad(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)
    seed = 42

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(3, wptype)
    quat = wp.types.quaternion(wptype)

    def slerp_kernel(
        q0: wp.array(dtype=quat),
        q1: wp.array(dtype=quat),
        t: wp.array(dtype=wptype),
        loss: wp.array(dtype=wptype),
        index: int,
    ):
        tid = wp.tid()

        q = wp.quat_slerp(q0[tid], q1[tid], t[tid])
        wp.atomic_add(loss, 0, q[index])

    slerp_kernel = getkernel(slerp_kernel, suffix=dtype.__name__)

    def slerp_kernel_forward(
        q0: wp.array(dtype=quat),
        q1: wp.array(dtype=quat),
        t: wp.array(dtype=wptype),
        loss: wp.array(dtype=wptype),
        index: int,
    ):
        tid = wp.tid()

        axis = vec3()
        angle = wptype(0.0)

        wp.quat_to_axis_angle(wp.mul(wp.quat_inverse(q0[tid]), q1[tid]), axis, angle)
        q = wp.mul(q0[tid], wp.quat_from_axis_angle(axis, t[tid] * angle))

        wp.atomic_add(loss, 0, q[index])

    slerp_kernel_forward = getkernel(slerp_kernel_forward, suffix=dtype.__name__)

    def quat_sampler_slerp(kernel_seed: int, quats: wp.array(dtype=quat)):
        tid = wp.tid()

        state = wp.rand_init(kernel_seed, tid)

        angle = wp.randf(state, 0.0, 2.0 * 3.1415926535)
        dir = wp.sample_unit_sphere_surface(state) * wp.sin(angle * 0.5)

        q = quat(wptype(dir[0]), wptype(dir[1]), wptype(dir[2]), wptype(wp.cos(angle * 0.5)))
        qn = wp.normalize(q)

        quats[tid] = qn

    quat_sampler = getkernel(quat_sampler_slerp, suffix=dtype.__name__)

    if register_kernels:
        return

    N = 50

    q0 = wp.zeros(N, dtype=quat, device=device, requires_grad=True)
    q1 = wp.zeros(N, dtype=quat, device=device, requires_grad=True)

    wp.launch(kernel=quat_sampler, dim=N, inputs=[seed, q0], device=device)
    wp.launch(kernel=quat_sampler, dim=N, inputs=[seed + 1, q1], device=device)

    t = rng.uniform(low=0.0, high=1.0, size=N)
    t = wp.array(t, dtype=wptype, device=device, requires_grad=True)

    def compute_gradients(kernel, wrt, index):
        loss = wp.zeros(1, dtype=wptype, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel=kernel, dim=N, inputs=[q0, q1, t, loss, index], device=device)

            tape.backward(loss)

        gradients = 1.0 * tape.gradients[wrt].numpy()
        tape.zero()

        return loss.numpy()[0], gradients

    eps = {
        np.float16: 2.0e-2,
        np.float32: 1.0e-5,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    # wrt t

    # gather gradients from builtin adjoints
    xcmp, gradients_x = compute_gradients(slerp_kernel, t, 0)
    ycmp, gradients_y = compute_gradients(slerp_kernel, t, 1)
    zcmp, gradients_z = compute_gradients(slerp_kernel, t, 2)
    wcmp, gradients_w = compute_gradients(slerp_kernel, t, 3)

    # gather gradients from autodiff
    xcmp_auto, gradients_x_auto = compute_gradients(slerp_kernel_forward, t, 0)
    ycmp_auto, gradients_y_auto = compute_gradients(slerp_kernel_forward, t, 1)
    zcmp_auto, gradients_z_auto = compute_gradients(slerp_kernel_forward, t, 2)
    wcmp_auto, gradients_w_auto = compute_gradients(slerp_kernel_forward, t, 3)

    assert_np_equal(gradients_x, gradients_x_auto, tol=eps)
    assert_np_equal(gradients_y, gradients_y_auto, tol=eps)
    assert_np_equal(gradients_z, gradients_z_auto, tol=eps)
    assert_np_equal(gradients_w, gradients_w_auto, tol=eps)
    assert_np_equal(xcmp, xcmp_auto, tol=eps)
    assert_np_equal(ycmp, ycmp_auto, tol=eps)
    assert_np_equal(zcmp, zcmp_auto, tol=eps)
    assert_np_equal(wcmp, wcmp_auto, tol=eps)

    # wrt q0

    # gather gradients from builtin adjoints
    xcmp, gradients_x = compute_gradients(slerp_kernel, q0, 0)
    ycmp, gradients_y = compute_gradients(slerp_kernel, q0, 1)
    zcmp, gradients_z = compute_gradients(slerp_kernel, q0, 2)
    wcmp, gradients_w = compute_gradients(slerp_kernel, q0, 3)

    # gather gradients from autodiff
    xcmp_auto, gradients_x_auto = compute_gradients(slerp_kernel_forward, q0, 0)
    ycmp_auto, gradients_y_auto = compute_gradients(slerp_kernel_forward, q0, 1)
    zcmp_auto, gradients_z_auto = compute_gradients(slerp_kernel_forward, q0, 2)
    wcmp_auto, gradients_w_auto = compute_gradients(slerp_kernel_forward, q0, 3)

    assert_np_equal(gradients_x, gradients_x_auto, tol=eps)
    assert_np_equal(gradients_y, gradients_y_auto, tol=eps)
    assert_np_equal(gradients_z, gradients_z_auto, tol=eps)
    assert_np_equal(gradients_w, gradients_w_auto, tol=eps)
    assert_np_equal(xcmp, xcmp_auto, tol=eps)
    assert_np_equal(ycmp, ycmp_auto, tol=eps)
    assert_np_equal(zcmp, zcmp_auto, tol=eps)
    assert_np_equal(wcmp, wcmp_auto, tol=eps)

    # wrt q1

    # gather gradients from builtin adjoints
    xcmp, gradients_x = compute_gradients(slerp_kernel, q1, 0)
    ycmp, gradients_y = compute_gradients(slerp_kernel, q1, 1)
    zcmp, gradients_z = compute_gradients(slerp_kernel, q1, 2)
    wcmp, gradients_w = compute_gradients(slerp_kernel, q1, 3)

    # gather gradients from autodiff
    xcmp_auto, gradients_x_auto = compute_gradients(slerp_kernel_forward, q1, 0)
    ycmp_auto, gradients_y_auto = compute_gradients(slerp_kernel_forward, q1, 1)
    zcmp_auto, gradients_z_auto = compute_gradients(slerp_kernel_forward, q1, 2)
    wcmp_auto, gradients_w_auto = compute_gradients(slerp_kernel_forward, q1, 3)

    assert_np_equal(gradients_x, gradients_x_auto, tol=eps)
    assert_np_equal(gradients_y, gradients_y_auto, tol=eps)
    assert_np_equal(gradients_z, gradients_z_auto, tol=eps)
    assert_np_equal(gradients_w, gradients_w_auto, tol=eps)
    assert_np_equal(xcmp, xcmp_auto, tol=eps)
    assert_np_equal(ycmp, ycmp_auto, tol=eps)
    assert_np_equal(zcmp, zcmp_auto, tol=eps)
    assert_np_equal(wcmp, wcmp_auto, tol=eps)


############################################################


def test_quat_to_axis_angle_grad(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)
    seed = 42
    num_rand = 50

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(3, wptype)
    vec4 = wp.types.vector(4, wptype)
    quat = wp.types.quaternion(wptype)

    def quat_to_axis_angle_kernel(quats: wp.array(dtype=quat), loss: wp.array(dtype=wptype), coord_idx: int):
        tid = wp.tid()
        axis = vec3()
        angle = wptype(0.0)

        wp.quat_to_axis_angle(quats[tid], axis, angle)
        a = vec4(axis[0], axis[1], axis[2], angle)

        wp.atomic_add(loss, 0, a[coord_idx])

    quat_to_axis_angle_kernel = getkernel(quat_to_axis_angle_kernel, suffix=dtype.__name__)

    def quat_to_axis_angle_kernel_forward(quats: wp.array(dtype=quat), loss: wp.array(dtype=wptype), coord_idx: int):
        tid = wp.tid()
        q = quats[tid]
        axis = vec3()
        angle = wptype(0.0)

        v = vec3(q[0], q[1], q[2])
        if q[3] < wptype(0):
            axis = -wp.normalize(v)
        else:
            axis = wp.normalize(v)

        angle = wptype(2) * wp.atan2(wp.length(v), wp.abs(q[3]))
        a = vec4(axis[0], axis[1], axis[2], angle)

        wp.atomic_add(loss, 0, a[coord_idx])

    quat_to_axis_angle_kernel_forward = getkernel(quat_to_axis_angle_kernel_forward, suffix=dtype.__name__)

    def quat_sampler(kernel_seed: int, angles: wp.array(dtype=float), quats: wp.array(dtype=quat)):
        tid = wp.tid()

        state = wp.rand_init(kernel_seed, tid)

        angle = angles[tid]
        dir = wp.sample_unit_sphere_surface(state) * wp.sin(angle * 0.5)

        q = quat(wptype(dir[0]), wptype(dir[1]), wptype(dir[2]), wptype(wp.cos(angle * 0.5)))
        qn = wp.normalize(q)

        quats[tid] = qn

    quat_sampler = getkernel(quat_sampler, suffix=dtype.__name__)

    if register_kernels:
        return

    quats = wp.zeros(num_rand, dtype=quat, device=device, requires_grad=True)
    angles = wp.array(
        np.linspace(0.0, 2.0 * np.pi, num_rand, endpoint=False, dtype=np.float32), dtype=float, device=device
    )
    wp.launch(kernel=quat_sampler, dim=num_rand, inputs=[seed, angles, quats], device=device)

    edge_cases = np.array(
        [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)), (0.0, 0.0, 0.0, 0.0)]
    )
    num_edge = len(edge_cases)
    edge_cases = wp.array(edge_cases, dtype=quat, device=device, requires_grad=True)

    def compute_gradients(arr, kernel, dim, index):
        loss = wp.zeros(1, dtype=wptype, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel=kernel, dim=dim, inputs=[arr, loss, index], device=device)

            tape.backward(loss)

        gradients = 1.0 * tape.gradients[arr].numpy()
        tape.zero()

        return loss.numpy()[0], gradients

    # gather gradients from builtin adjoints
    xcmp, gradients_x = compute_gradients(quats, quat_to_axis_angle_kernel, num_rand, 0)
    ycmp, gradients_y = compute_gradients(quats, quat_to_axis_angle_kernel, num_rand, 1)
    zcmp, gradients_z = compute_gradients(quats, quat_to_axis_angle_kernel, num_rand, 2)
    wcmp, gradients_w = compute_gradients(quats, quat_to_axis_angle_kernel, num_rand, 3)

    # gather gradients from autodiff
    xcmp_auto, gradients_x_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, num_rand, 0)
    ycmp_auto, gradients_y_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, num_rand, 1)
    zcmp_auto, gradients_z_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, num_rand, 2)
    wcmp_auto, gradients_w_auto = compute_gradients(quats, quat_to_axis_angle_kernel_forward, num_rand, 3)

    # edge cases: gather gradients from builtin adjoints
    _, edge_gradients_x = compute_gradients(edge_cases, quat_to_axis_angle_kernel, num_edge, 0)
    _, edge_gradients_y = compute_gradients(edge_cases, quat_to_axis_angle_kernel, num_edge, 1)
    _, edge_gradients_z = compute_gradients(edge_cases, quat_to_axis_angle_kernel, num_edge, 2)
    _, edge_gradients_w = compute_gradients(edge_cases, quat_to_axis_angle_kernel, num_edge, 3)

    # edge cases: gather gradients from autodiff
    _, edge_gradients_x_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, num_edge, 0)
    _, edge_gradients_y_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, num_edge, 1)
    _, edge_gradients_z_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, num_edge, 2)
    _, edge_gradients_w_auto = compute_gradients(edge_cases, quat_to_axis_angle_kernel_forward, num_edge, 3)

    eps = {
        np.float16: 2.0e-1,
        np.float32: 2.0e-4,
        np.float64: 2.0e-7,
    }.get(dtype, 0)

    assert_np_equal(xcmp, xcmp_auto, tol=eps)
    assert_np_equal(ycmp, ycmp_auto, tol=eps)
    assert_np_equal(zcmp, zcmp_auto, tol=eps)
    assert_np_equal(wcmp, wcmp_auto, tol=eps)

    assert_np_equal(gradients_x, gradients_x_auto, tol=eps)
    assert_np_equal(gradients_y, gradients_y_auto, tol=eps)
    assert_np_equal(gradients_z, gradients_z_auto, tol=eps)
    assert_np_equal(gradients_w, gradients_w_auto, tol=eps)

    assert_np_equal(edge_gradients_x, edge_gradients_x_auto, tol=eps)
    assert_np_equal(edge_gradients_y, edge_gradients_y_auto, tol=eps)
    assert_np_equal(edge_gradients_z, edge_gradients_z_auto, tol=eps)
    assert_np_equal(edge_gradients_w, edge_gradients_w_auto, tol=eps)


############################################################


def test_quat_rpy_grad(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)
    N = 3

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    vec3 = wp.types.vector(3, wptype)
    quat = wp.types.quaternion(wptype)

    def rpy_to_quat_kernel(rpy_arr: wp.array(dtype=vec3), loss: wp.array(dtype=wptype), coord_idx: int):
        tid = wp.tid()
        rpy = rpy_arr[tid]
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        q = wp.quat_rpy(roll, pitch, yaw)

        wp.atomic_add(loss, 0, q[coord_idx])

    rpy_to_quat_kernel = getkernel(rpy_to_quat_kernel, suffix=dtype.__name__)

    def rpy_to_quat_kernel_forward(rpy_arr: wp.array(dtype=vec3), loss: wp.array(dtype=wptype), coord_idx: int):
        tid = wp.tid()
        rpy = rpy_arr[tid]
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        cy = wp.cos(yaw * wptype(0.5))
        sy = wp.sin(yaw * wptype(0.5))
        cr = wp.cos(roll * wptype(0.5))
        sr = wp.sin(roll * wptype(0.5))
        cp = wp.cos(pitch * wptype(0.5))
        sp = wp.sin(pitch * wptype(0.5))

        w = cy * cr * cp + sy * sr * sp
        x = cy * sr * cp - sy * cr * sp
        y = cy * cr * sp + sy * sr * cp
        z = sy * cr * cp - cy * sr * sp

        q = quat(x, y, z, w)

        wp.atomic_add(loss, 0, q[coord_idx])

    rpy_to_quat_kernel_forward = getkernel(rpy_to_quat_kernel_forward, suffix=dtype.__name__)

    if register_kernels:
        return

    rpy_arr = rng.uniform(low=-np.pi, high=np.pi, size=(N, 3))
    rpy_arr = wp.array(rpy_arr, dtype=vec3, device=device, requires_grad=True)

    def compute_gradients(kernel, wrt, index):
        loss = wp.zeros(1, dtype=wptype, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel=kernel, dim=N, inputs=[wrt, loss, index], device=device)

            tape.backward(loss)

        gradients = 1.0 * tape.gradients[wrt].numpy()
        tape.zero()

        return loss.numpy()[0], gradients

    # wrt rpy
    # gather gradients from builtin adjoints
    rcmp, gradients_r = compute_gradients(rpy_to_quat_kernel, rpy_arr, 0)
    pcmp, gradients_p = compute_gradients(rpy_to_quat_kernel, rpy_arr, 1)
    ycmp, gradients_y = compute_gradients(rpy_to_quat_kernel, rpy_arr, 2)

    # gather gradients from autodiff
    rcmp_auto, gradients_r_auto = compute_gradients(rpy_to_quat_kernel_forward, rpy_arr, 0)
    pcmp_auto, gradients_p_auto = compute_gradients(rpy_to_quat_kernel_forward, rpy_arr, 1)
    ycmp_auto, gradients_y_auto = compute_gradients(rpy_to_quat_kernel_forward, rpy_arr, 2)

    eps = {
        np.float16: 2.0e-2,
        np.float32: 1.0e-5,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    assert_np_equal(rcmp, rcmp_auto, tol=eps)
    assert_np_equal(pcmp, pcmp_auto, tol=eps)
    assert_np_equal(ycmp, ycmp_auto, tol=eps)

    assert_np_equal(gradients_r, gradients_r_auto, tol=eps)
    assert_np_equal(gradients_p, gradients_p_auto, tol=eps)
    assert_np_equal(gradients_y, gradients_y_auto, tol=eps)


############################################################


def test_quat_from_matrix(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat33 = wp.types.matrix((3, 3), wptype)
    mat44 = wp.types.matrix((4, 4), wptype)
    quat = wp.types.quaternion(wptype)

    def quat_from_matrix(m: wp.array2d(dtype=wptype), loss: wp.array(dtype=wptype), idx: int):
        tid = wp.tid()

        # fmt: off
        m3 = mat33(
            m[tid, 0], m[tid, 1], m[tid, 2],
            m[tid, 3], m[tid, 4], m[tid, 5],
            m[tid, 6], m[tid, 7], m[tid, 8],
        )
        q1 = wp.quat_from_matrix(m3)

        m4 = mat44(
            m[tid, 0], m[tid, 1], m[tid, 2], wptype(0.0),
            m[tid, 3], m[tid, 4], m[tid, 5], wptype(0.0),
            m[tid, 6], m[tid, 7], m[tid, 8], wptype(0.0),
            wptype(0.0), wptype(0.0), wptype(0.0), wptype(1.0),
        )
        q2 = wp.quat_from_matrix(m4)
        # fmt: on

        wp.expect_eq(q1, q2)
        wp.atomic_add(loss, 0, q1[idx])

    def quat_from_matrix_forward(mats: wp.array2d(dtype=wptype), loss: wp.array(dtype=wptype), idx: int):
        tid = wp.tid()

        m = mat33(
            mats[tid, 0],
            mats[tid, 1],
            mats[tid, 2],
            mats[tid, 3],
            mats[tid, 4],
            mats[tid, 5],
            mats[tid, 6],
            mats[tid, 7],
            mats[tid, 8],
        )

        tr = m[0][0] + m[1][1] + m[2][2]
        x = wptype(0)
        y = wptype(0)
        z = wptype(0)
        w = wptype(0)
        h = wptype(0)

        if tr >= wptype(0):
            h = wp.sqrt(tr + wptype(1))
            w = wptype(0.5) * h
            h = wptype(0.5) / h

            x = (m[2][1] - m[1][2]) * h
            y = (m[0][2] - m[2][0]) * h
            z = (m[1][0] - m[0][1]) * h
        else:
            max_diag = 0
            if m[1][1] > m[0][0]:
                max_diag = 1
            if m[2][2] > m[max_diag][max_diag]:
                max_diag = 2

            if max_diag == 0:
                h = wp.sqrt((m[0][0] - (m[1][1] + m[2][2])) + wptype(1))
                x = wptype(0.5) * h
                h = wptype(0.5) / h

                y = (m[0][1] + m[1][0]) * h
                z = (m[2][0] + m[0][2]) * h
                w = (m[2][1] - m[1][2]) * h
            elif max_diag == 1:
                h = wp.sqrt((m[1][1] - (m[2][2] + m[0][0])) + wptype(1))
                y = wptype(0.5) * h
                h = wptype(0.5) / h

                z = (m[1][2] + m[2][1]) * h
                x = (m[0][1] + m[1][0]) * h
                w = (m[0][2] - m[2][0]) * h
            if max_diag == 2:
                h = wp.sqrt((m[2][2] - (m[0][0] + m[1][1])) + wptype(1))
                z = wptype(0.5) * h
                h = wptype(0.5) / h

                x = (m[2][0] + m[0][2]) * h
                y = (m[1][2] + m[2][1]) * h
                w = (m[1][0] - m[0][1]) * h

        q = wp.normalize(quat(x, y, z, w))

        wp.atomic_add(loss, 0, q[idx])

    quat_from_matrix = getkernel(quat_from_matrix, suffix=dtype.__name__)
    quat_from_matrix_forward = getkernel(quat_from_matrix_forward, suffix=dtype.__name__)

    if register_kernels:
        return

    m = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.5, 0.866, 0.0, -0.866, 0.5],
            [0.866, 0.0, 0.25, -0.433, 0.5, 0.75, -0.25, -0.866, 0.433],
            [0.866, -0.433, 0.25, 0.0, 0.5, 0.866, -0.5, -0.75, 0.433],
            [-1.2, -1.6, -2.3, 0.25, -0.6, -0.33, 3.2, -1.0, -2.2],
        ]
    )
    m = wp.array2d(m, dtype=wptype, device=device, requires_grad=True)

    N = m.shape[0]

    def compute_gradients(kernel, wrt, index):
        loss = wp.zeros(1, dtype=wptype, device=device, requires_grad=True)
        tape = wp.Tape()

        with tape:
            wp.launch(kernel=kernel, dim=N, inputs=[m, loss, index], device=device)

            tape.backward(loss)

        gradients = 1.0 * tape.gradients[wrt].numpy()
        tape.zero()

        return loss.numpy()[0], gradients

    # gather gradients from builtin adjoints
    cmpx, gradients_x = compute_gradients(quat_from_matrix, m, 0)
    cmpy, gradients_y = compute_gradients(quat_from_matrix, m, 1)
    cmpz, gradients_z = compute_gradients(quat_from_matrix, m, 2)
    cmpw, gradients_w = compute_gradients(quat_from_matrix, m, 3)

    # gather gradients from autodiff
    cmpx_auto, gradients_x_auto = compute_gradients(quat_from_matrix_forward, m, 0)
    cmpy_auto, gradients_y_auto = compute_gradients(quat_from_matrix_forward, m, 1)
    cmpz_auto, gradients_z_auto = compute_gradients(quat_from_matrix_forward, m, 2)
    cmpw_auto, gradients_w_auto = compute_gradients(quat_from_matrix_forward, m, 3)

    # compare
    eps = 1.0e6

    eps = {
        np.float16: 2.0e-2,
        np.float32: 1.0e-5,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    assert_np_equal(cmpx, cmpx_auto, tol=eps)
    assert_np_equal(cmpy, cmpy_auto, tol=eps)
    assert_np_equal(cmpz, cmpz_auto, tol=eps)
    assert_np_equal(cmpw, cmpw_auto, tol=eps)

    assert_np_equal(gradients_x, gradients_x_auto, tol=eps)
    assert_np_equal(gradients_y, gradients_y_auto, tol=eps)
    assert_np_equal(gradients_z, gradients_z_auto, tol=eps)
    assert_np_equal(gradients_w, gradients_w_auto, tol=eps)


def test_quat_identity(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def quat_identity_test(output: wp.array(dtype=wptype)):
        q = wp.quat_identity(dtype=wptype)
        output[0] = q[0]
        output[1] = q[1]
        output[2] = q[2]
        output[3] = q[3]

    def quat_identity_test_default(output: wp.array(dtype=wp.float32)):
        q = wp.quat_identity()
        output[0] = q[0]
        output[1] = q[1]
        output[2] = q[2]
        output[3] = q[3]

    quat_identity_kernel = getkernel(quat_identity_test, suffix=dtype.__name__)
    quat_identity_default_kernel = getkernel(quat_identity_test_default, suffix=np.float32.__name__)

    if register_kernels:
        return

    output = wp.zeros(4, dtype=wptype, device=device)
    wp.launch(quat_identity_kernel, dim=1, inputs=[], outputs=[output], device=device)
    expected = np.zeros_like(output.numpy())
    expected[3] = 1
    assert_np_equal(output.numpy(), expected)

    # let's just test that it defaults to float32:
    output = wp.zeros(4, dtype=wp.float32, device=device)
    wp.launch(quat_identity_default_kernel, dim=1, inputs=[], outputs=[output], device=device)
    expected = np.zeros_like(output.numpy())
    expected[3] = 1
    assert_np_equal(output.numpy(), expected)


############################################################


def test_quat_assign_inplace(test, device, dtype, register_kernels=False):
    np_type = np.dtype(dtype)
    wp_type = wp.types.np_dtype_to_warp_type[np_type]

    quat = wp.types.quaternion(dtype=wp_type)

    def quattest_read_write_store(x: wp.array(dtype=wp_type), a: wp.array(dtype=quat)):
        tid = wp.tid()

        t = a[tid]
        t[0] = x[tid]
        a[tid] = t

    def quattest_in_register(x: wp.array(dtype=wp_type), a: wp.array(dtype=quat)):
        tid = wp.tid()

        g = wp_type(0.0)
        q = a[tid]
        g = q[0] + wp_type(2.0) * q[1] + wp_type(3.0) * q[2] + wp_type(4.0) * q[3]
        x[tid] = g

    def quattest_component(x: wp.array(dtype=quat), y: wp.array(dtype=wp_type)):
        i = wp.tid()

        a = quat()
        a.x = wp_type(1.0) * y[i]
        a.y = wp_type(2.0) * y[i]
        a.z = wp_type(3.0) * y[i]
        a.w = wp_type(4.0) * y[i]
        x[i] = a

    kernel_read_write_store = getkernel(quattest_read_write_store, suffix=dtype.__name__)
    kernel_in_register = getkernel(quattest_in_register, suffix=dtype.__name__)
    kernel_component = getkernel(quattest_component, suffix=dtype.__name__)

    if register_kernels:
        return

    a = wp.ones(1, dtype=quat, device=device, requires_grad=True)
    x = wp.full(1, value=2.0, dtype=wp_type, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel_read_write_store, dim=1, inputs=[x, a], device=device)

    tape.backward(grads={a: wp.ones_like(a, requires_grad=False)})

    assert_np_equal(a.numpy(), np.array([[2.0, 1.0, 1.0, 1.0]], dtype=np_type))
    assert_np_equal(x.grad.numpy(), np.array([1.0], dtype=np_type))

    tape.reset()

    a = wp.ones(1, dtype=quat, device=device, requires_grad=True)
    x = wp.zeros(1, dtype=wp_type, device=device, requires_grad=True)

    with tape:
        wp.launch(kernel_in_register, dim=1, inputs=[x, a], device=device)

    tape.backward(grads={x: wp.ones_like(x, requires_grad=False)})

    assert_np_equal(x.numpy(), np.array([10.0], dtype=np_type))
    assert_np_equal(a.grad.numpy(), np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np_type))

    tape.reset()

    x = wp.zeros(1, dtype=quat, requires_grad=True)
    y = wp.ones(1, dtype=wp_type, requires_grad=True)

    with tape:
        wp.launch(kernel_component, dim=1, inputs=[x, y])

    tape.backward(grads={x: wp.ones_like(x, requires_grad=False)})

    assert_np_equal(x.numpy(), np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np_type))
    assert_np_equal(y.grad.numpy(), np.array([10.0], dtype=np_type))


############################################################


def test_quat_euler_conversion(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)
    N = 3

    rpy_arr = rng.uniform(low=-np.pi, high=np.pi, size=(N, 3))

    quats_from_euler = [list(wp.sim.quat_from_euler(wp.vec3(*rpy), 0, 1, 2)) for rpy in rpy_arr]
    quats_from_rpy = [list(wp.quat_rpy(rpy[0], rpy[1], rpy[2])) for rpy in rpy_arr]

    assert_np_equal(np.array(quats_from_euler), np.array(quats_from_rpy), tol=1e-4)


def test_anon_type_instance(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def quat_create_test(input: wp.array(dtype=wptype), output: wp.array(dtype=wptype)):
        # component constructor:
        q = wp.quaternion(input[0], input[1], input[2], input[3])
        output[0] = wptype(2) * q[0]
        output[1] = wptype(2) * q[1]
        output[2] = wptype(2) * q[2]
        output[3] = wptype(2) * q[3]

        # vector / scalar constructor:
        q2 = wp.quaternion(wp.vector(input[4], input[5], input[6]), input[7])
        output[4] = wptype(2) * q2[0]
        output[5] = wptype(2) * q2[1]
        output[6] = wptype(2) * q2[2]
        output[7] = wptype(2) * q2[3]

    quat_create_kernel = getkernel(quat_create_test, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    input = wp.array(rng.standard_normal(size=8).astype(dtype), requires_grad=True, device=device)
    output = wp.zeros(8, dtype=wptype, requires_grad=True, device=device)
    wp.launch(quat_create_kernel, dim=1, inputs=[input], outputs=[output], device=device)
    assert_np_equal(output.numpy(), 2 * input.numpy())

    for i in range(len(input)):
        cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(quat_create_kernel, dim=1, inputs=[input], outputs=[output], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[output, i], outputs=[cmp], device=device)
        tape.backward(loss=cmp)
        expectedgrads = np.zeros(len(input))
        expectedgrads[i] = 2
        assert_np_equal(tape.gradients[input].numpy(), expectedgrads)
        tape.zero()


# Same as above but with a default (float) type
# which tests some different code paths that
# need to ensure types are correctly canonicalized
# during codegen
@wp.kernel
def test_constructor_default():
    qzero = wp.quat()
    wp.expect_eq(qzero[0], 0.0)
    wp.expect_eq(qzero[1], 0.0)
    wp.expect_eq(qzero[2], 0.0)
    wp.expect_eq(qzero[3], 0.0)

    qval = wp.quat(1.0, 2.0, 3.0, 4.0)
    wp.expect_eq(qval[0], 1.0)
    wp.expect_eq(qval[1], 2.0)
    wp.expect_eq(qval[2], 3.0)
    wp.expect_eq(qval[3], 4.0)

    qeye = wp.quat_identity()
    wp.expect_eq(qeye[0], 0.0)
    wp.expect_eq(qeye[1], 0.0)
    wp.expect_eq(qeye[2], 0.0)
    wp.expect_eq(qeye[3], 1.0)


def test_py_arithmetic_ops(test, device, dtype):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def make_quat(*args):
        if wptype in wp.types.int_types:
            # Cast to the correct integer type to simulate wrapping.
            return tuple(wptype._type_(x).value for x in args)

        return args

    quat_cls = wp.types.quaternion(wptype)

    v = quat_cls(1, -2, 3, -4)
    test.assertSequenceEqual(+v, make_quat(1, -2, 3, -4))
    test.assertSequenceEqual(-v, make_quat(-1, 2, -3, 4))
    test.assertSequenceEqual(v + quat_cls(5, 5, 5, 5), make_quat(6, 3, 8, 1))
    test.assertSequenceEqual(v - quat_cls(5, 5, 5, 5), make_quat(-4, -7, -2, -9))

    v = quat_cls(2, 4, 6, 8)
    test.assertSequenceEqual(v * wptype(2), make_quat(4, 8, 12, 16))
    test.assertSequenceEqual(wptype(2) * v, make_quat(4, 8, 12, 16))
    test.assertSequenceEqual(v / wptype(2), make_quat(1, 2, 3, 4))
    test.assertSequenceEqual(wptype(24) / v, make_quat(12, 6, 4, 3))


@wp.kernel
def quat_len_kernel(
    q: wp.quat,
    out: wp.array(dtype=int),
):
    length = wp.static(len(q))
    wp.expect_eq(wp.static(len(q)), 4)
    out[0] = wp.static(len(q))

    foo = wp.quat()
    length = len(foo)
    wp.expect_eq(len(foo), 4)
    out[1] = len(foo)


def test_quat_len(test, device):
    q = wp.quat()
    out = wp.empty(2, dtype=int, device=device)
    wp.launch(quat_len_kernel, dim=(1,), inputs=(q,), outputs=(out,), device=device)

    test.assertEqual(out.numpy()[0], 4)
    test.assertEqual(out.numpy()[1], 4)


@wp.kernel
def quat_augassign_kernel(
    a: wp.array(dtype=wp.quat), b: wp.array(dtype=wp.quat), c: wp.array(dtype=wp.quat), d: wp.array(dtype=wp.quat)
):
    i = wp.tid()

    q1 = wp.quat()
    q2 = b[i]

    q1[0] += q2[0]
    q1[1] += q2[1]
    q1[2] += q2[2]
    q1[3] += q2[3]

    a[i] = q1

    q3 = wp.quat()
    q4 = d[i]

    q3[0] -= q4[0]
    q3[1] -= q4[1]
    q3[2] -= q4[2]
    q3[3] -= q4[3]

    c[i] = q3


def test_quat_augassign(test, device):
    N = 3

    a = wp.zeros(N, dtype=wp.quat, requires_grad=True, device=device)
    b = wp.ones(N, dtype=wp.quat, requires_grad=True, device=device)

    c = wp.zeros(N, dtype=wp.quat, requires_grad=True, device=device)
    d = wp.ones(N, dtype=wp.quat, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(quat_augassign_kernel, N, inputs=[a, b, c, d], device=device)

    tape.backward(grads={a: wp.ones_like(a), c: wp.ones_like(c)})

    assert_np_equal(a.numpy(), wp.ones_like(a).numpy())
    assert_np_equal(a.grad.numpy(), wp.ones_like(a).numpy())
    assert_np_equal(b.grad.numpy(), wp.ones_like(a).numpy())

    assert_np_equal(c.numpy(), -wp.ones_like(c).numpy())
    assert_np_equal(c.grad.numpy(), wp.ones_like(c).numpy())
    assert_np_equal(d.grad.numpy(), -wp.ones_like(d).numpy())


def test_quat_assign_copy(test, device):
    saved_enable_vector_component_overwrites_setting = wp.config.enable_vector_component_overwrites
    try:
        wp.config.enable_vector_component_overwrites = True

        @wp.kernel
        def quat_in_register_overwrite(x: wp.array(dtype=wp.quat), a: wp.array(dtype=wp.quat)):
            tid = wp.tid()

            f = wp.quat()
            a_quat = a[tid]
            f = a_quat
            f[1] = 3.0

            x[tid] = f

        x = wp.zeros(1, dtype=wp.quat, device=device, requires_grad=True)
        a = wp.ones(1, dtype=wp.quat, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(quat_in_register_overwrite, dim=1, inputs=[x, a], device=device)

        tape.backward(grads={x: wp.ones_like(x, requires_grad=False)})

        assert_np_equal(x.numpy(), np.array([[1.0, 3.0, 1.0, 1.0]], dtype=float))
        assert_np_equal(a.grad.numpy(), np.array([[1.0, 0.0, 1.0, 1.0]], dtype=float))

    finally:
        wp.config.enable_vector_component_overwrites = saved_enable_vector_component_overwrites_setting


devices = get_test_devices()


class TestQuat(unittest.TestCase):
    pass


add_kernel_test(TestQuat, test_constructor_default, dim=1, devices=devices)
add_kernel_test(TestQuat, test_assignment, dim=1, devices=devices)

for dtype in np_float_types:
    add_function_test_register_kernel(
        TestQuat, f"test_constructors_{dtype.__name__}", test_constructors, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat,
        f"test_casting_constructors_{dtype.__name__}",
        test_casting_constructors,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestQuat, f"test_anon_type_instance_{dtype.__name__}", test_anon_type_instance, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_inverse_{dtype.__name__}", test_inverse, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_quat_identity_{dtype.__name__}", test_quat_identity, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_dotproduct_{dtype.__name__}", test_dotproduct, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_length_{dtype.__name__}", test_length, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_normalize_{dtype.__name__}", test_normalize, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_addition_{dtype.__name__}", test_addition, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_subtraction_{dtype.__name__}", test_subtraction, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat,
        f"test_scalar_multiplication_{dtype.__name__}",
        test_scalar_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestQuat, f"test_scalar_division_{dtype.__name__}", test_scalar_division, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat,
        f"test_quat_multiplication_{dtype.__name__}",
        test_quat_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestQuat, f"test_indexing_{dtype.__name__}", test_indexing, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_quat_lerp_{dtype.__name__}", test_quat_lerp, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat,
        f"test_quat_to_axis_angle_grad_{dtype.__name__}",
        test_quat_to_axis_angle_grad,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestQuat, f"test_slerp_grad_{dtype.__name__}", test_slerp_grad, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_quat_rpy_grad_{dtype.__name__}", test_quat_rpy_grad, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_quat_from_matrix_{dtype.__name__}", test_quat_from_matrix, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_quat_rotate_{dtype.__name__}", test_quat_rotate, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat, f"test_quat_to_matrix_{dtype.__name__}", test_quat_to_matrix, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestQuat,
        f"test_quat_euler_conversion_{dtype.__name__}",
        test_quat_euler_conversion,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestQuat,
        f"test_quat_assign_inplace_{dtype.__name__}",
        test_quat_assign_inplace,
        devices=devices,
        dtype=dtype,
    )
    add_function_test(
        TestQuat, f"test_py_arithmetic_ops_{dtype.__name__}", test_py_arithmetic_ops, devices=None, dtype=dtype
    )

add_function_test(TestQuat, "test_quat_len", test_quat_len, devices=devices)
add_function_test(TestQuat, "test_quat_augassign", test_quat_augassign, devices=devices)
add_function_test(TestQuat, "test_quat_assign_copy", test_quat_assign_copy, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
