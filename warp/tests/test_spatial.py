# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def test_spatial_vector_constructors(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(length=3, dtype=wptype)
    spatial_vector = wp.types.vector(length=6, dtype=wptype)

    def check_spatial_vector_component_constructor(
        input: wp.array(dtype=wptype),
        out: wp.array(dtype=wptype),
    ):
        result = spatial_vector(input[0], input[1], input[2], input[3], input[4], input[5])

        # multiply the output by 2 so we've got something to backpropagate:
        out[0] = wptype(2) * result[0]
        out[1] = wptype(2) * result[1]
        out[2] = wptype(2) * result[2]
        out[3] = wptype(2) * result[3]
        out[4] = wptype(2) * result[4]
        out[5] = wptype(2) * result[5]

    def check_spatial_vector_vector_constructor(
        input: wp.array(dtype=wptype),
        out: wp.array(dtype=wptype),
    ):
        result = spatial_vector(vec3(input[0], input[1], input[2]), vec3(input[3], input[4], input[5]))

        # multiply the output by 2 so we've got something to backpropagate:
        out[0] = wptype(2) * result[0]
        out[1] = wptype(2) * result[1]
        out[2] = wptype(2) * result[2]
        out[3] = wptype(2) * result[3]
        out[4] = wptype(2) * result[4]
        out[5] = wptype(2) * result[5]

    kernel = getkernel(check_spatial_vector_component_constructor, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)
    vec_kernel = getkernel(check_spatial_vector_vector_constructor, suffix=dtype.__name__)

    if register_kernels:
        return

    input = wp.array(rng.standard_normal(size=6).astype(dtype), requires_grad=True, device=device)
    output = wp.zeros_like(input)
    wp.launch(kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy(), 2 * input.numpy(), tol=tol)

    for i in range(len(input)):
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

    input = wp.array(rng.standard_normal(size=6).astype(dtype), requires_grad=True, device=device)
    output = wp.zeros_like(input)
    wp.launch(vec_kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy(), 2 * input.numpy(), tol=tol)

    for i in range(len(input)):
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


def test_spatial_vector_indexing(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_vector = wp.types.vector(length=6, dtype=wptype)

    def check_spatial_vector_indexing(
        input: wp.array(dtype=spatial_vector),
        out: wp.array(dtype=wptype),
    ):
        inpt = input[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(6):
            out[idx] = wptype(2) * inpt[i]
            idx = idx + 1

    kernel = getkernel(check_spatial_vector_indexing, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    input = wp.array(
        rng.standard_normal(size=(1, 6)).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device
    )
    outcmps = wp.zeros(6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[input], outputs=[outcmps], device=device)

    assert_np_equal(outcmps.numpy(), 2 * input.numpy().ravel(), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(6):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[input], outputs=[outcmps], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outcmps, i], outputs=[out], device=device)
        tape.backward(loss=out)
        expectedresult = np.zeros(6, dtype=dtype)
        expectedresult[i] = 2
        assert_np_equal(tape.gradients[input].numpy()[0], expectedresult)
        tape.zero()


def test_spatial_vector_scalar_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_vector = wp.types.vector(length=6, dtype=wptype)

    def check_spatial_vector_scalar_mul(
        s: wp.array(dtype=wptype),
        q: wp.array(dtype=spatial_vector),
        outcmps_l: wp.array(dtype=wptype),
        outcmps_r: wp.array(dtype=wptype),
    ):
        lresult = s[0] * q[0]
        rresult = q[0] * s[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        for i in range(6):
            outcmps_l[i] = wptype(2) * lresult[i]
            outcmps_r[i] = wptype(2) * rresult[i]

    kernel = getkernel(check_spatial_vector_scalar_mul, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=1).astype(dtype), requires_grad=True, device=device)
    q = wp.array(
        rng.standard_normal(size=(1, 6)).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device
    )

    outcmps_l = wp.zeros(6, dtype=wptype, requires_grad=True, device=device)
    outcmps_r = wp.zeros(6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[s, q],
        outputs=[
            outcmps_l,
            outcmps_r,
        ],
        device=device,
    )

    assert_np_equal(outcmps_l.numpy(), 2 * s.numpy()[0] * q.numpy(), tol=tol)
    assert_np_equal(outcmps_r.numpy(), 2 * s.numpy()[0] * q.numpy(), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(6):
        # test left/right mul gradients:
        for wrt in [outcmps_l, outcmps_r]:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[s, q], outputs=[outcmps_l, outcmps_r], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[wrt, i], outputs=[out], device=device)
            tape.backward(loss=out)
            expectedresult = np.zeros(6, dtype=dtype)
            expectedresult[i] = 2 * s.numpy()[0]
            assert_np_equal(tape.gradients[q].numpy()[0], expectedresult, tol=tol)
            assert_np_equal(tape.gradients[s].numpy()[0], 2 * q.numpy()[0, i], tol=tol)
            tape.zero()


def test_spatial_vector_add_sub(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_vector = wp.types.vector(length=6, dtype=wptype)

    def check_spatial_vector_add_sub(
        q: wp.array(dtype=spatial_vector),
        v: wp.array(dtype=spatial_vector),
        outputs_add: wp.array(dtype=wptype),
        outputs_sub: wp.array(dtype=wptype),
    ):
        addresult = q[0] + v[0]
        subresult = q[0] - v[0]
        for i in range(6):
            outputs_add[i] = wptype(2) * addresult[i]
            outputs_sub[i] = wptype(2) * subresult[i]

    kernel = getkernel(check_spatial_vector_add_sub, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)
    if register_kernels:
        return

    q = wp.array(rng.standard_normal(size=6).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device)
    v = wp.array(rng.standard_normal(size=6).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device)

    outputs_add = wp.zeros(6, dtype=wptype, requires_grad=True, device=device)
    outputs_sub = wp.zeros(6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            q,
            v,
        ],
        outputs=[outputs_add, outputs_sub],
        device=device,
    )

    assert_np_equal(outputs_add.numpy(), 2 * (q.numpy() + v.numpy()), tol=tol)
    assert_np_equal(outputs_sub.numpy(), 2 * (q.numpy() - v.numpy()), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(6):
        # test add gradients:
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[q, v], outputs=[outputs_add, outputs_sub], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_add, i], outputs=[out], device=device)
        tape.backward(loss=out)
        expectedresult = np.zeros(6, dtype=dtype)
        expectedresult[i] = 2
        assert_np_equal(tape.gradients[q].numpy()[0], expectedresult, tol=tol)
        assert_np_equal(tape.gradients[v].numpy()[0], expectedresult, tol=tol)
        tape.zero()

        # test subtraction gradients:
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[q, v], outputs=[outputs_add, outputs_sub], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_sub, i], outputs=[out], device=device)
        tape.backward(loss=out)
        expectedresult = np.zeros(6, dtype=dtype)
        expectedresult[i] = 2
        assert_np_equal(tape.gradients[q].numpy()[0], expectedresult, tol=tol)
        assert_np_equal(tape.gradients[v].numpy()[0], -expectedresult, tol=tol)
        tape.zero()


def test_spatial_dot(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_vector = wp.types.vector(length=6, dtype=wptype)

    def check_spatial_dot(
        s: wp.array(dtype=spatial_vector),
        v: wp.array(dtype=spatial_vector),
        dot: wp.array(dtype=wptype),
    ):
        dot[0] = wptype(2) * wp.spatial_dot(v[0], s[0])

    kernel = getkernel(check_spatial_dot, suffix=dtype.__name__)
    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=6).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device)
    v = wp.array(rng.standard_normal(size=6).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device)
    dot = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
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


def test_spatial_cross(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_vector = wp.types.vector(length=6, dtype=wptype)

    def check_spatial_cross(
        s: wp.array(dtype=spatial_vector),
        v: wp.array(dtype=spatial_vector),
        outputs: wp.array(dtype=wptype),
        outputs_dual: wp.array(dtype=wptype),
        outputs_wcrossw: wp.array(dtype=wptype),
        outputs_vcrossw: wp.array(dtype=wptype),
        outputs_wcrossv: wp.array(dtype=wptype),
        outputs_vcrossv: wp.array(dtype=wptype),
    ):
        c = wp.spatial_cross(s[0], v[0])
        d = wp.spatial_cross_dual(s[0], v[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        for i in range(6):
            outputs[i] = wptype(2) * c[i]
            outputs_dual[i] = wptype(2) * d[i]

        sw = wp.spatial_top(s[0])
        sv = wp.spatial_bottom(s[0])
        vw = wp.spatial_top(v[0])
        vv = wp.spatial_bottom(v[0])

        wcrossw = wp.cross(sw, vw)
        vcrossw = wp.cross(sv, vw)
        wcrossv = wp.cross(sw, vv)
        vcrossv = wp.cross(sv, vv)

        for i in range(3):
            outputs_wcrossw[i] = wcrossw[i]
            outputs_vcrossw[i] = vcrossw[i]
            outputs_wcrossv[i] = wcrossv[i]
            outputs_vcrossv[i] = vcrossv[i]

    kernel = getkernel(check_spatial_cross, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=6).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device)
    v = wp.array(rng.standard_normal(size=6).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device)
    outputs = wp.zeros(6, dtype=wptype, requires_grad=True, device=device)
    outputs_dual = wp.zeros(6, dtype=wptype, requires_grad=True, device=device)
    outputs_wcrossw = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    outputs_vcrossw = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    outputs_wcrossv = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    outputs_vcrossv = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            s,
            v,
        ],
        outputs=[outputs, outputs_dual, outputs_wcrossw, outputs_vcrossw, outputs_wcrossv, outputs_vcrossv],
        device=device,
    )

    sw = s.numpy()[0, :3]
    sv = s.numpy()[0, 3:]
    vw = v.numpy()[0, :3]
    vv = v.numpy()[0, 3:]

    wcrossw = np.cross(sw, vw)
    vcrossw = np.cross(sv, vw)
    wcrossv = np.cross(sw, vv)
    vcrossv = np.cross(sv, vv)

    assert_np_equal(outputs.numpy()[:3], 2 * wcrossw, tol=tol)
    assert_np_equal(outputs.numpy()[3:], 2 * (vcrossw + wcrossv), tol=tol)

    assert_np_equal(outputs_dual.numpy()[:3], 2 * (wcrossw + vcrossv), tol=tol)
    assert_np_equal(outputs_dual.numpy()[3:], 2 * wcrossv, tol=tol)

    for i in range(3):
        cmp_w = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_v = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_w_dual = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_v_dual = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_wcrossw = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_vcrossw = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_wcrossv = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_vcrossv = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel,
                dim=1,
                inputs=[
                    s,
                    v,
                ],
                outputs=[outputs, outputs_dual, outputs_wcrossw, outputs_vcrossw, outputs_wcrossv, outputs_vcrossv],
                device=device,
            )

            # ith w and v vector components of spatial_cross:
            wp.launch(output_select_kernel, dim=1, inputs=[outputs, i], outputs=[cmp_w], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs, i + 3], outputs=[cmp_v], device=device)

            # ith w and v vector components of spatial_cross_dual:
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_dual, i], outputs=[cmp_w_dual], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_dual, i + 3], outputs=[cmp_v_dual], device=device)

            # ith vector components of some cross products:
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_wcrossw, i], outputs=[cmp_wcrossw], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_vcrossw, i], outputs=[cmp_vcrossw], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_wcrossv, i], outputs=[cmp_wcrossv], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_vcrossv, i], outputs=[cmp_vcrossv], device=device)

        def getgrads(cmp, tape=tape):
            tape.backward(loss=cmp)
            sgrads = 1.0 * tape.gradients[s].numpy()
            vgrads = 1.0 * tape.gradients[v].numpy()
            tape.zero()
            return sgrads, vgrads

        dcmp_w_ds, dcmp_w_dv = getgrads(cmp_w)
        dcmp_v_ds, dcmp_v_dv = getgrads(cmp_v)
        dcmp_w_dual_ds, dcmp_w_dual_dv = getgrads(cmp_w_dual)
        dcmp_v_dual_ds, dcmp_v_dual_dv = getgrads(cmp_v_dual)

        dcmp_wcrossw_ds, dcmp_wcrossw_dv = getgrads(cmp_wcrossw)
        dcmp_vcrossw_ds, dcmp_vcrossw_dv = getgrads(cmp_vcrossw)
        dcmp_wcrossv_ds, dcmp_wcrossv_dv = getgrads(cmp_wcrossv)
        dcmp_vcrossv_ds, dcmp_vcrossv_dv = getgrads(cmp_vcrossv)

        assert_np_equal(dcmp_w_ds, 2 * dcmp_wcrossw_ds, tol=tol)
        assert_np_equal(dcmp_w_dv, 2 * dcmp_wcrossw_dv, tol=tol)

        assert_np_equal(dcmp_v_ds, 2 * (dcmp_vcrossw_ds + dcmp_wcrossv_ds), tol=tol)
        assert_np_equal(dcmp_v_dv, 2 * (dcmp_vcrossw_dv + dcmp_wcrossv_dv), tol=tol)

        assert_np_equal(dcmp_w_dual_ds, 2 * (dcmp_wcrossw_ds + dcmp_vcrossv_ds), tol=tol)
        assert_np_equal(dcmp_w_dual_dv, 2 * (dcmp_wcrossw_dv + dcmp_vcrossv_dv), tol=tol)

        assert_np_equal(dcmp_v_dual_ds, 2 * dcmp_wcrossv_ds, tol=tol)
        assert_np_equal(dcmp_v_dual_dv, 2 * dcmp_wcrossv_dv, tol=tol)


def test_spatial_top_bottom(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_vector = wp.types.vector(length=6, dtype=wptype)

    def check_spatial_top_bottom(
        s: wp.array(dtype=spatial_vector),
        outputs: wp.array(dtype=wptype),
    ):
        top = wp.spatial_top(s[0])
        bottom = wp.spatial_bottom(s[0])

        outputs[0] = wptype(2) * top[0]
        outputs[1] = wptype(2) * top[1]
        outputs[2] = wptype(2) * top[2]

        outputs[3] = wptype(2) * bottom[0]
        outputs[4] = wptype(2) * bottom[1]
        outputs[5] = wptype(2) * bottom[2]

    kernel = getkernel(check_spatial_top_bottom, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=6).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device)
    outputs = wp.zeros(6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            s,
        ],
        outputs=[outputs],
        device=device,
    )

    assert_np_equal(outputs.numpy(), 2.0 * s.numpy(), tol=tol)

    for i in range(6):
        cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel,
                dim=1,
                inputs=[
                    s,
                ],
                outputs=[outputs],
                device=device,
            )
            wp.launch(output_select_kernel, dim=1, inputs=[outputs, i], outputs=[cmp], device=device)
        tape.backward(loss=cmp)
        expectedgrads = np.zeros(6)
        expectedgrads[i] = 2
        assert_np_equal(tape.gradients[s].numpy(), expectedgrads.reshape((1, 6)))
        tape.zero()


def test_transform_constructors(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(length=3, dtype=wptype)
    transform = wp.types.transformation(dtype=wptype)
    quat = wp.types.quaternion(dtype=wptype)

    def check_transform_constructor(
        input: wp.array(dtype=wptype),
        out: wp.array(dtype=wptype),
    ):
        result = transform(vec3(input[0], input[1], input[2]), quat(input[3], input[4], input[5], input[6]))

        # multiply the output by 2 so we've got something to backpropagate:
        out[0] = wptype(2) * result[0]
        out[1] = wptype(2) * result[1]
        out[2] = wptype(2) * result[2]
        out[3] = wptype(2) * result[3]
        out[4] = wptype(2) * result[4]
        out[5] = wptype(2) * result[5]
        out[6] = wptype(2) * result[6]

    kernel = getkernel(check_transform_constructor, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    p = rng.standard_normal(size=3).astype(dtype)
    q = rng.standard_normal(size=4).astype(dtype)
    q /= np.linalg.norm(q)

    input = wp.array(np.concatenate((p, q)), requires_grad=True, device=device)
    output = wp.zeros_like(input)

    wp.launch(kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy(), 2 * input.numpy(), tol=tol)

    for i in range(len(input)):
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


def test_transform_indexing(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    transform = wp.types.transformation(dtype=wptype)

    def check_transform_indexing(
        input: wp.array(dtype=transform),
        out: wp.array(dtype=wptype),
    ):
        inpt = input[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(7):
            out[idx] = wptype(2) * inpt[i]
            idx = idx + 1

    kernel = getkernel(check_transform_indexing, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    input = wp.array(rng.standard_normal(size=(1, 7)).astype(dtype), dtype=transform, requires_grad=True, device=device)
    outcmps = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[input], outputs=[outcmps], device=device)

    assert_np_equal(outcmps.numpy(), 2 * input.numpy().ravel(), tol=tol)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(7):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[input], outputs=[outcmps], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outcmps, i], outputs=[out], device=device)
        tape.backward(loss=out)
        expectedresult = np.zeros(7, dtype=dtype)
        expectedresult[i] = 2
        assert_np_equal(tape.gradients[input].numpy()[0], expectedresult)
        tape.zero()


def test_transform_scalar_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    transform = wp.types.transformation(dtype=wptype)

    def check_transform_scalar_mul(
        s: wp.array(dtype=wptype),
        q: wp.array(dtype=transform),
        outcmps_l: wp.array(dtype=wptype),
        outcmps_r: wp.array(dtype=wptype),
    ):
        lresult = s[0] * q[0]
        rresult = q[0] * s[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        for i in range(7):
            outcmps_l[i] = wptype(2) * lresult[i]
            outcmps_r[i] = wptype(2) * rresult[i]

    kernel = getkernel(check_transform_scalar_mul, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=1).astype(dtype), requires_grad=True, device=device)
    q = wp.array(rng.standard_normal(size=(1, 7)).astype(dtype), dtype=transform, requires_grad=True, device=device)

    outcmps_l = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)
    outcmps_r = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[s, q],
        outputs=[
            outcmps_l,
            outcmps_r,
        ],
        device=device,
    )

    assert_np_equal(outcmps_l.numpy(), 2 * s.numpy()[0] * q.numpy(), tol=tol)
    assert_np_equal(outcmps_r.numpy(), 2 * s.numpy()[0] * q.numpy(), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(7):
        # test left/right mul gradients:
        for wrt in [outcmps_l, outcmps_r]:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[s, q], outputs=[outcmps_l, outcmps_r], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[wrt, i], outputs=[out], device=device)
            tape.backward(loss=out)
            expectedresult = np.zeros(7, dtype=dtype)
            expectedresult[i] = 2 * s.numpy()[0]
            assert_np_equal(tape.gradients[q].numpy()[0], expectedresult, tol=tol)
            assert_np_equal(tape.gradients[s].numpy()[0], 2 * q.numpy()[0, i], tol=tol)
            tape.zero()


def test_transform_add_sub(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    transform = wp.types.transformation(dtype=wptype)

    def check_transform_add_sub(
        q: wp.array(dtype=transform),
        v: wp.array(dtype=transform),
        outputs_add: wp.array(dtype=wptype),
        outputs_sub: wp.array(dtype=wptype),
    ):
        addresult = q[0] + v[0]
        subresult = q[0] - v[0]
        for i in range(7):
            outputs_add[i] = wptype(2) * addresult[i]
            outputs_sub[i] = wptype(2) * subresult[i]

    kernel = getkernel(check_transform_add_sub, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    q = wp.array(rng.standard_normal(size=7).astype(dtype), dtype=transform, requires_grad=True, device=device)
    v = wp.array(rng.standard_normal(size=7).astype(dtype), dtype=transform, requires_grad=True, device=device)

    outputs_add = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)
    outputs_sub = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            q,
            v,
        ],
        outputs=[outputs_add, outputs_sub],
        device=device,
    )

    assert_np_equal(outputs_add.numpy(), 2 * (q.numpy() + v.numpy()), tol=tol)
    assert_np_equal(outputs_sub.numpy(), 2 * (q.numpy() - v.numpy()), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(7):
        # test add gradients:
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[q, v], outputs=[outputs_add, outputs_sub], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_add, i], outputs=[out], device=device)
        tape.backward(loss=out)
        expectedresult = np.zeros(7, dtype=dtype)
        expectedresult[i] = 2
        assert_np_equal(tape.gradients[q].numpy()[0], expectedresult, tol=tol)
        assert_np_equal(tape.gradients[v].numpy()[0], expectedresult, tol=tol)
        tape.zero()

        # test subtraction gradients:
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[q, v], outputs=[outputs_add, outputs_sub], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_sub, i], outputs=[out], device=device)
        tape.backward(loss=out)
        expectedresult = np.zeros(7, dtype=dtype)
        expectedresult[i] = 2
        assert_np_equal(tape.gradients[q].numpy()[0], expectedresult, tol=tol)
        assert_np_equal(tape.gradients[v].numpy()[0], -expectedresult, tol=tol)
        tape.zero()


def test_transform_get_trans_rot(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    transform = wp.types.transformation(dtype=wptype)

    def check_transform_get_trans_rot(
        s: wp.array(dtype=transform),
        outputs: wp.array(dtype=wptype),
    ):
        trans = wp.transform_get_translation(s[0])
        q = wp.transform_get_rotation(s[0])

        outputs[0] = wptype(2) * trans[0]
        outputs[1] = wptype(2) * trans[1]
        outputs[2] = wptype(2) * trans[2]

        outputs[3] = wptype(2) * q[0]
        outputs[4] = wptype(2) * q[1]
        outputs[5] = wptype(2) * q[2]
        outputs[6] = wptype(2) * q[3]

    kernel = getkernel(check_transform_get_trans_rot, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=7).astype(dtype), dtype=transform, requires_grad=True, device=device)
    outputs = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            s,
        ],
        outputs=[outputs],
        device=device,
    )

    assert_np_equal(outputs.numpy(), 2.0 * s.numpy(), tol=tol)

    for i in range(7):
        cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel,
                dim=1,
                inputs=[
                    s,
                ],
                outputs=[outputs],
                device=device,
            )
            wp.launch(output_select_kernel, dim=1, inputs=[outputs, i], outputs=[cmp], device=device)
        tape.backward(loss=cmp)
        expectedgrads = np.zeros(7)
        expectedgrads[i] = 2
        assert_np_equal(tape.gradients[s].numpy(), expectedgrads.reshape((1, 7)))
        tape.zero()


def test_transform_multiply(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    transform = wp.types.transformation(dtype=wptype)

    def check_transform_multiply(
        a: wp.array(dtype=transform),
        b: wp.array(dtype=transform),
        outputs: wp.array(dtype=wptype),
        outputs_fn: wp.array(dtype=wptype),
        outputs_manual: wp.array(dtype=wptype),
    ):
        result = a[0] * b[0]
        result_fn = wp.transform_multiply(a[0], b[0])

        # let's just work out the transform multiplication manually
        # and compare value/gradients with that:
        atrans = wp.transform_get_translation(a[0])
        arot = wp.transform_get_rotation(a[0])

        btrans = wp.transform_get_translation(b[0])
        brot = wp.transform_get_rotation(b[0])

        trans = wp.quat_rotate(arot, btrans) + atrans
        rot = arot * brot
        result_manual = transform(trans, rot)

        for i in range(7):
            outputs[i] = wptype(2) * result[i]
            outputs_fn[i] = wptype(2) * result_fn[i]
            outputs_manual[i] = wptype(2) * result_manual[i]

    kernel = getkernel(check_transform_multiply, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    q = rng.standard_normal(size=7)
    s = rng.standard_normal(size=7)
    q[3:] /= np.linalg.norm(q[3:])
    s[3:] /= np.linalg.norm(s[3:])

    q = wp.array(q.astype(dtype), dtype=transform, requires_grad=True, device=device)
    s = wp.array(s.astype(dtype), dtype=transform, requires_grad=True, device=device)
    outputs = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)
    outputs_fn = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)
    outputs_manual = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            q,
            s,
        ],
        outputs=[outputs, outputs_fn, outputs_manual],
        device=device,
    )

    assert_np_equal(outputs.numpy(), outputs_fn.numpy(), tol=tol)
    assert_np_equal(outputs.numpy(), outputs_manual.numpy(), tol=tol)

    for i in range(7):
        cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_fn = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_manual = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel,
                dim=1,
                inputs=[
                    q,
                    s,
                ],
                outputs=[outputs, outputs_fn, outputs_manual],
                device=device,
            )
            wp.launch(output_select_kernel, dim=1, inputs=[outputs, i], outputs=[cmp], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_fn, i], outputs=[cmp_fn], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_manual, i], outputs=[cmp_manual], device=device)
        tape.backward(loss=cmp)
        qgrads = 1.0 * tape.gradients[q].numpy()
        sgrads = 1.0 * tape.gradients[s].numpy()
        tape.zero()
        tape.backward(loss=cmp_fn)
        qgrads_fn = 1.0 * tape.gradients[q].numpy()
        sgrads_fn = 1.0 * tape.gradients[s].numpy()
        tape.zero()
        tape.backward(loss=cmp_manual)
        qgrads_manual = 1.0 * tape.gradients[q].numpy()
        sgrads_manual = 1.0 * tape.gradients[s].numpy()
        tape.zero()

        assert_np_equal(qgrads, qgrads_fn, tol=tol)
        assert_np_equal(sgrads, sgrads_fn, tol=tol)

        assert_np_equal(qgrads, qgrads_manual, tol=tol)
        assert_np_equal(sgrads, sgrads_manual, tol=tol)


def test_transform_inverse(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    transform = wp.types.transformation(dtype=wptype)

    def check_transform_inverse(
        a: wp.array(dtype=transform),
        outputs: wp.array(dtype=wptype),
        outputs_shouldbeidentity: wp.array(dtype=wptype),
        outputs_manual: wp.array(dtype=wptype),
    ):
        result = wp.transform_inverse(a[0])
        idt = result * a[0]

        # let's just work out the transform inverse manually
        # and compare value/gradients with that:
        atrans = wp.transform_get_translation(a[0])
        arot = wp.transform_get_rotation(a[0])

        rotinv = wp.quat_inverse(arot)
        result_manual = transform(-wp.quat_rotate(rotinv, atrans), rotinv)

        for i in range(7):
            outputs[i] = wptype(2) * result[i]
            outputs_shouldbeidentity[i] = wptype(2) * idt[i]
            outputs_manual[i] = wptype(2) * result_manual[i]

    kernel = getkernel(check_transform_inverse, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    q = rng.standard_normal(size=7)
    s = rng.standard_normal(size=7)
    q[3:] /= np.linalg.norm(q[3:])
    s[3:] /= np.linalg.norm(s[3:])

    q = wp.array(q.astype(dtype), dtype=transform, requires_grad=True, device=device)
    outputs = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)
    outputs_shouldbeidentity = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)
    outputs_manual = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            q,
        ],
        outputs=[outputs, outputs_shouldbeidentity, outputs_manual],
        device=device,
    )

    # check inverse:
    assert_np_equal(outputs_shouldbeidentity.numpy(), np.array([0, 0, 0, 0, 0, 0, 2]), tol=tol)

    # same as manual result:
    assert_np_equal(outputs.numpy(), outputs_manual.numpy(), tol=tol)

    for i in range(7):
        cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_manual = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel,
                dim=1,
                inputs=[
                    q,
                ],
                outputs=[outputs, outputs_shouldbeidentity, outputs_manual],
                device=device,
            )
            wp.launch(output_select_kernel, dim=1, inputs=[outputs, i], outputs=[cmp], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_manual, i], outputs=[cmp_manual], device=device)
        tape.backward(loss=cmp)
        qgrads = 1.0 * tape.gradients[q].numpy()
        tape.zero()
        tape.backward(loss=cmp_manual)
        qgrads_manual = 1.0 * tape.gradients[q].numpy()
        tape.zero()

        # check gradients against manual result:
        assert_np_equal(qgrads, qgrads_manual, tol=tol)


def test_transform_point_vector(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    transform = wp.types.transformation(dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)

    def check_transform_point_vector(
        t: wp.array(dtype=transform),
        v: wp.array(dtype=vec3),
        outputs_pt: wp.array(dtype=wptype),
        outputs_pt_manual: wp.array(dtype=wptype),
        outputs_vec: wp.array(dtype=wptype),
        outputs_vec_manual: wp.array(dtype=wptype),
    ):
        result_pt = wp.transform_point(t[0], v[0])
        result_pt_manual = wp.transform_get_translation(t[0]) + wp.quat_rotate(wp.transform_get_rotation(t[0]), v[0])

        result_vec = wp.transform_vector(t[0], v[0])
        result_vec_manual = wp.quat_rotate(wp.transform_get_rotation(t[0]), v[0])

        for i in range(3):
            outputs_pt[i] = wptype(2) * result_pt[i]
            outputs_pt_manual[i] = wptype(2) * result_pt_manual[i]
            outputs_vec[i] = wptype(2) * result_vec[i]
            outputs_vec_manual[i] = wptype(2) * result_vec_manual[i]

    kernel = getkernel(check_transform_point_vector, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    q = rng.standard_normal(size=7)
    q[3:] /= np.linalg.norm(q[3:])

    t = wp.array(q.astype(dtype), dtype=transform, requires_grad=True, device=device)
    v = wp.array(rng.standard_normal(size=3), dtype=vec3, requires_grad=True, device=device)
    outputs_pt = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    outputs_pt_manual = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    outputs_vec = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    outputs_vec_manual = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[t, v],
        outputs=[outputs_pt, outputs_pt_manual, outputs_vec, outputs_vec_manual],
        device=device,
    )

    # same as manual results:
    assert_np_equal(outputs_pt.numpy(), outputs_pt_manual.numpy(), tol=tol)
    assert_np_equal(outputs_vec.numpy(), outputs_vec_manual.numpy(), tol=tol)

    for i in range(3):
        cmp_pt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_pt_manual = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_vec = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        cmp_vec_manual = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel,
                dim=1,
                inputs=[t, v],
                outputs=[outputs_pt, outputs_pt_manual, outputs_vec, outputs_vec_manual],
                device=device,
            )
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_pt, i], outputs=[cmp_pt], device=device)
            wp.launch(
                output_select_kernel, dim=1, inputs=[outputs_pt_manual, i], outputs=[cmp_pt_manual], device=device
            )
            wp.launch(output_select_kernel, dim=1, inputs=[outputs_vec, i], outputs=[cmp_vec], device=device)
            wp.launch(
                output_select_kernel, dim=1, inputs=[outputs_vec_manual, i], outputs=[cmp_vec_manual], device=device
            )
        tape.backward(loss=cmp_pt)
        tgrads_pt = 1.0 * tape.gradients[t].numpy()
        vgrads_pt = 1.0 * tape.gradients[v].numpy()
        tape.zero()
        tape.backward(loss=cmp_pt_manual)
        tgrads_pt_manual = 1.0 * tape.gradients[t].numpy()
        vgrads_pt_manual = 1.0 * tape.gradients[v].numpy()
        tape.zero()
        tape.backward(loss=cmp_vec)
        tgrads_vec = 1.0 * tape.gradients[t].numpy()
        vgrads_vec = 1.0 * tape.gradients[v].numpy()
        tape.zero()
        tape.backward(loss=cmp_vec_manual)
        tgrads_vec_manual = 1.0 * tape.gradients[t].numpy()
        vgrads_vec_manual = 1.0 * tape.gradients[v].numpy()
        tape.zero()

        # check gradients against manual result:
        assert_np_equal(tgrads_pt, tgrads_pt_manual, tol=tol)
        assert_np_equal(vgrads_pt, vgrads_pt_manual, tol=tol)
        assert_np_equal(tgrads_vec, tgrads_vec_manual, tol=tol)
        assert_np_equal(vgrads_vec, vgrads_vec_manual, tol=tol)


def test_spatial_matrix_constructors(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_matrix = wp.types.matrix(shape=(6, 6), dtype=wptype)

    def check_spatial_matrix_constructor(
        input: wp.array(dtype=wptype),
        out: wp.array(dtype=wptype),
    ):
        # multiply the output by 2 so we've got something to backpropagate:
        result0 = spatial_matrix(
            input[0],
            input[1],
            input[2],
            input[3],
            input[4],
            input[5],
            input[6],
            input[7],
            input[8],
            input[9],
            input[10],
            input[11],
            input[12],
            input[13],
            input[14],
            input[15],
            input[16],
            input[17],
            input[18],
            input[19],
            input[20],
            input[21],
            input[22],
            input[23],
            input[24],
            input[25],
            input[26],
            input[27],
            input[28],
            input[29],
            input[30],
            input[31],
            input[32],
            input[33],
            input[34],
            input[35],
        )
        result1 = spatial_matrix()

        idx = 0
        for i in range(6):
            for j in range(6):
                out[idx] = wptype(2) * result0[i, j]
                idx = idx + 1

        for i in range(6):
            for j in range(6):
                out[idx] = result1[i, j]
                idx = idx + 1

    kernel = getkernel(check_spatial_matrix_constructor, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    input = wp.array(rng.standard_normal(size=6 * 6).astype(dtype), requires_grad=True, device=device)
    output = wp.zeros(2 * 6 * 6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[input], outputs=[output], device=device)

    assert_np_equal(output.numpy()[: 6 * 6], 2 * input.numpy(), tol=tol)
    assert_np_equal(output.numpy()[6 * 6 :], np.zeros_like(input.numpy()), tol=tol)

    for i in range(len(input)):
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
        break


def test_spatial_matrix_indexing(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_matrix = wp.types.matrix(shape=(6, 6), dtype=wptype)

    def check_spatial_matrix_indexing(
        input: wp.array(dtype=spatial_matrix),
        out: wp.array(dtype=wptype),
    ):
        inpt = input[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(6):
            for j in range(6):
                out[idx] = wptype(2) * inpt[i, j]
                idx = idx + 1

    kernel = getkernel(check_spatial_matrix_indexing, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    input = wp.array(
        rng.standard_normal(size=(1, 6, 6)).astype(dtype), dtype=spatial_matrix, requires_grad=True, device=device
    )
    outcmps = wp.zeros(6 * 6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[input], outputs=[outcmps], device=device)

    assert_np_equal(outcmps.numpy(), 2 * input.numpy().ravel(), tol=tol)
    idx = 0
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(6):
        for j in range(6):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[input], outputs=[outcmps], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcmps, idx], outputs=[out], device=device)
            tape.backward(loss=out)
            expectedresult = np.zeros((6, 6), dtype=dtype)
            expectedresult[i, j] = 2
            assert_np_equal(tape.gradients[input].numpy()[0], expectedresult)
            tape.zero()
            idx = idx + 1


def test_spatial_matrix_scalar_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_matrix = wp.types.matrix(shape=(6, 6), dtype=wptype)

    def check_spatial_matrix_scalar_mul(
        s: wp.array(dtype=wptype),
        q: wp.array(dtype=spatial_matrix),
        outcmps_l: wp.array(dtype=wptype),
        outcmps_r: wp.array(dtype=wptype),
    ):
        lresult = s[0] * q[0]
        rresult = q[0] * s[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(6):
            for j in range(6):
                outcmps_l[idx] = wptype(2) * lresult[i, j]
                outcmps_r[idx] = wptype(2) * rresult[i, j]
                idx = idx + 1

    kernel = getkernel(check_spatial_matrix_scalar_mul, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    s = wp.array(rng.standard_normal(size=1).astype(dtype), requires_grad=True, device=device)
    q = wp.array(
        rng.standard_normal(size=(1, 6, 6)).astype(dtype), dtype=spatial_matrix, requires_grad=True, device=device
    )

    outcmps_l = wp.zeros(6 * 6, dtype=wptype, requires_grad=True, device=device)
    outcmps_r = wp.zeros(6 * 6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[s, q],
        outputs=[
            outcmps_l,
            outcmps_r,
        ],
        device=device,
    )

    assert_np_equal(outcmps_l.numpy(), 2 * s.numpy()[0] * q.numpy(), tol=tol)
    assert_np_equal(outcmps_r.numpy(), 2 * s.numpy()[0] * q.numpy(), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for i in range(6):
        for j in range(6):
            # test left/right mul gradients:
            for wrt in [outcmps_l, outcmps_r]:
                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[s, q], outputs=[outcmps_l, outcmps_r], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[wrt, idx], outputs=[out], device=device)
                tape.backward(loss=out)
                expectedresult = np.zeros((6, 6), dtype=dtype)
                expectedresult[i, j] = 2 * s.numpy()[0]
                assert_np_equal(tape.gradients[q].numpy()[0], expectedresult, tol=tol)
                assert_np_equal(tape.gradients[s].numpy()[0], 2 * q.numpy()[0, i, j], tol=tol)
                tape.zero()
            idx = idx + 1


def test_spatial_matrix_add_sub(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_matrix = wp.types.matrix(shape=(6, 6), dtype=wptype)

    def check_spatial_matrix_add_sub(
        q: wp.array(dtype=spatial_matrix),
        v: wp.array(dtype=spatial_matrix),
        outputs_add: wp.array(dtype=wptype),
        outputs_sub: wp.array(dtype=wptype),
    ):
        addresult = q[0] + v[0]
        subresult = q[0] - v[0]
        idx = 0
        for i in range(6):
            for j in range(6):
                outputs_add[idx] = wptype(2) * addresult[i, j]
                outputs_sub[idx] = wptype(2) * subresult[i, j]
                idx = idx + 1

    kernel = getkernel(check_spatial_matrix_add_sub, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    q = wp.array(
        rng.standard_normal(size=(1, 6, 6)).astype(dtype), dtype=spatial_matrix, requires_grad=True, device=device
    )
    v = wp.array(
        rng.standard_normal(size=(1, 6, 6)).astype(dtype), dtype=spatial_matrix, requires_grad=True, device=device
    )

    outputs_add = wp.zeros(6 * 6, dtype=wptype, requires_grad=True, device=device)
    outputs_sub = wp.zeros(6 * 6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            q,
            v,
        ],
        outputs=[outputs_add, outputs_sub],
        device=device,
    )

    assert_np_equal(outputs_add.numpy(), 2 * (q.numpy() + v.numpy()), tol=tol)
    assert_np_equal(outputs_sub.numpy(), 2 * (q.numpy() - v.numpy()), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for i in range(6):
        for j in range(6):
            # test add gradients:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[q, v], outputs=[outputs_add, outputs_sub], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs_add, idx], outputs=[out], device=device)
            tape.backward(loss=out)
            expectedresult = np.zeros((6, 6), dtype=dtype)
            expectedresult[i, j] = 2
            assert_np_equal(tape.gradients[q].numpy()[0], expectedresult, tol=tol)
            assert_np_equal(tape.gradients[v].numpy()[0], expectedresult, tol=tol)
            tape.zero()

            # test subtraction gradients:
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[q, v], outputs=[outputs_add, outputs_sub], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outputs_sub, idx], outputs=[out], device=device)
            tape.backward(loss=out)
            expectedresult = np.zeros((6, 6), dtype=dtype)
            expectedresult[i, j] = 2
            assert_np_equal(tape.gradients[q].numpy()[0], expectedresult, tol=tol)
            assert_np_equal(tape.gradients[v].numpy()[0], -expectedresult, tol=tol)
            tape.zero()

            idx = idx + 1


def test_spatial_matvec_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.0e-2,
        np.float32: 5.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_matrix = wp.types.matrix(shape=(6, 6), dtype=wptype)
    spatial_vector = wp.types.vector(length=6, dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_spatial_mat_vec_mul(
        v: wp.array(dtype=spatial_vector),
        m: wp.array(dtype=spatial_matrix),
        outcomponents: wp.array(dtype=wptype),
    ):
        result = m[0] * v[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(6):
            outcomponents[idx] = wptype(2) * result[i]
            idx = idx + 1

    kernel = getkernel(check_spatial_mat_vec_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    v = wp.array(
        rng.standard_normal(size=(1, 6)).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device
    )
    m = wp.array(
        rng.standard_normal(size=(1, 6, 6)).astype(dtype), dtype=spatial_matrix, requires_grad=True, device=device
    )
    outcomponents = wp.zeros(6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v, m], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy(), 2 * np.matmul(m.numpy()[0], v.numpy()[0]), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(6):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[v, m], outputs=[outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, i], outputs=[out], device=device)
        tape.backward(loss=out)

        assert_np_equal(tape.gradients[v].numpy()[0], 2 * m.numpy()[0, i, :], tol=tol)
        expectedresult = np.zeros((6, 6), dtype=dtype)
        expectedresult[i, :] = 2 * v.numpy()[0]
        assert_np_equal(tape.gradients[m].numpy()[0], expectedresult, tol=tol)

        tape.zero()


def test_spatial_matmat_multiplication(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.0e-2,
        np.float32: 5.0e-6,
        np.float64: 5.0e-7,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_matrix = wp.types.matrix(shape=(6, 6), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_mat_mul(
        v: wp.array(dtype=spatial_matrix),
        m: wp.array(dtype=spatial_matrix),
        outcomponents: wp.array(dtype=wptype),
    ):
        result = m[0] * v[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(6):
            for j in range(6):
                outcomponents[idx] = wptype(2) * result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_mat_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    v = wp.array(
        rng.standard_normal(size=(1, 6, 6)).astype(dtype), dtype=spatial_matrix, requires_grad=True, device=device
    )
    m = wp.array(
        rng.standard_normal(size=(1, 6, 6)).astype(dtype), dtype=spatial_matrix, requires_grad=True, device=device
    )
    outcomponents = wp.zeros(6 * 6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v, m], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy(), 2 * np.matmul(m.numpy()[0], v.numpy()[0]), tol=tol)

    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for i in range(6):
        for j in range(6):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[v, m], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
            tape.backward(loss=out)

            expected = np.zeros((6, 6), dtype=dtype)
            expected[:, j] = 2 * m.numpy()[0, i, :]
            assert_np_equal(tape.gradients[v].numpy()[0], expected, tol=10 * tol)

            expected = np.zeros((6, 6), dtype=dtype)
            expected[i, :] = 2 * v.numpy()[0, :, j]
            assert_np_equal(tape.gradients[m].numpy()[0], expected, tol=10 * tol)

            tape.zero()
            idx = idx + 1


def test_spatial_mat_transpose(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_matrix = wp.types.matrix(shape=(6, 6), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_spatial_mat_transpose(
        m: wp.array(dtype=spatial_matrix),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        mat = wptype(2) * wp.transpose(m[0])

        idx = 0
        for i in range(6):
            for j in range(6):
                outcomponents[idx] = mat[i, j]
                idx = idx + 1

    kernel = getkernel(check_spatial_mat_transpose, suffix=dtype.__name__)

    if register_kernels:
        return

    m = wp.array(
        rng.standard_normal(size=(1, 6, 6)).astype(dtype), dtype=spatial_matrix, requires_grad=True, device=device
    )
    outcomponents = wp.zeros(6 * 6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy(), 2 * m.numpy()[0].T, tol=tol)

    idx = 0
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(6):
        for j in range(6):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[m], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
            tape.backward(loss=out)
            expectedresult = np.zeros((6, 6), dtype=dtype)
            expectedresult[j, i] = 2
            assert_np_equal(tape.gradients[m].numpy()[0], expectedresult)
            tape.zero()
            idx = idx + 1


def test_spatial_outer_product(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    spatial_vector = wp.types.vector(length=6, dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_spatial_outer_product(
        s: wp.array(dtype=spatial_vector),
        v: wp.array(dtype=spatial_vector),
        outcomponents: wp.array(dtype=wptype),
    ):
        mresult = wptype(2) * wp.outer(s[0], v[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(6):
            for j in range(6):
                outcomponents[idx] = mresult[i, j]
                idx = idx + 1

    kernel = getkernel(check_spatial_outer_product, suffix=dtype.__name__)

    if register_kernels:
        return

    s = wp.array(
        rng.standard_normal(size=(1, 6)).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device
    )
    v = wp.array(
        rng.standard_normal(size=(1, 6)).astype(dtype), dtype=spatial_vector, requires_grad=True, device=device
    )
    outcomponents = wp.zeros(6 * 6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[s, v], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy(), 2 * s.numpy()[0, :, None] * v.numpy()[0, None, :], tol=tol)

    idx = 0
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    for i in range(6):
        for j in range(6):
            tape = wp.Tape()
            with tape:
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[
                        s,
                        v,
                    ],
                    outputs=[outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
            tape.backward(loss=out)

            # this component's gonna be s_i * v_j, so its s gradient is gonna be nozero
            # at the ith component and its v gradient will be nonzero at the jth component:

            expectedresult = np.zeros((6), dtype=dtype)
            expectedresult[i] = 2 * v.numpy()[0, j]
            assert_np_equal(tape.gradients[s].numpy()[0], expectedresult, tol=10 * tol)

            expectedresult = np.zeros((6), dtype=dtype)
            expectedresult[j] = 2 * s.numpy()[0, i]
            assert_np_equal(tape.gradients[v].numpy()[0], expectedresult, tol=10 * tol)
            tape.zero()

            idx = idx + 1


def test_spatial_adjoint(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat3 = wp.types.matrix(shape=(3, 3), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_spatial_adjoint(
        R: wp.array(dtype=mat3),
        S: wp.array(dtype=mat3),
        outcomponents: wp.array(dtype=wptype),
    ):
        mresult = wptype(2) * wp.spatial_adjoint(R[0], S[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(6):
            for j in range(6):
                outcomponents[idx] = mresult[i, j]
                idx = idx + 1

    kernel = getkernel(check_spatial_adjoint, suffix=dtype.__name__)

    if register_kernels:
        return

    R = wp.array(rng.standard_normal(size=(1, 3, 3)).astype(dtype), dtype=mat3, requires_grad=True, device=device)
    S = wp.array(rng.standard_normal(size=(1, 3, 3)).astype(dtype), dtype=mat3, requires_grad=True, device=device)
    outcomponents = wp.zeros(6 * 6, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[R, S], outputs=[outcomponents], device=device)

    result = outcomponents.numpy().reshape(6, 6)
    expected = np.zeros_like(result)
    expected[:3, :3] = R.numpy()
    expected[3:, 3:] = R.numpy()
    expected[3:, :3] = S.numpy()

    assert_np_equal(result, 2 * expected, tol=tol)

    idx = 0
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for i in range(6):
        for j in range(6):
            tape = wp.Tape()
            with tape:
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[
                        R,
                        S,
                    ],
                    outputs=[outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
            tape.backward(loss=out)

            # this component's gonna be s_i * v_j, so its s gradient is gonna be nozero
            # at the ith component and its v gradient will be nonzero at the jth component:

            expectedresult = np.zeros((3, 3), dtype=dtype)
            if (i // 3 == 0 and j // 3 == 0) or (i // 3 == 1 and j // 3 == 1):
                expectedresult[i % 3, j % 3] = 2
            assert_np_equal(tape.gradients[R].numpy()[0], expectedresult, tol=10 * tol)

            expectedresult = np.zeros((3, 3), dtype=dtype)
            if i // 3 == 1 and j // 3 == 0:
                expectedresult[i % 3, j % 3] = 2
            assert_np_equal(tape.gradients[S].numpy()[0], expectedresult, tol=10 * tol)
            tape.zero()

            idx = idx + 1


def test_transform_identity(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def transform_identity_test(output: wp.array(dtype=wptype)):
        t = wp.transform_identity(dtype=wptype)
        for i in range(7):
            output[i] = t[i]

    def transform_identity_test_default(output: wp.array(dtype=wp.float32)):
        t = wp.transform_identity()
        for i in range(7):
            output[i] = t[i]

    quat_identity_kernel = getkernel(transform_identity_test, suffix=dtype.__name__)
    quat_identity_default_kernel = getkernel(transform_identity_test_default, suffix=np.float32.__name__)

    if register_kernels:
        return

    output = wp.zeros(7, dtype=wptype, device=device)
    wp.launch(quat_identity_kernel, dim=1, inputs=[], outputs=[output], device=device)
    expected = np.zeros_like(output.numpy())
    expected[-1] = 1
    assert_np_equal(output.numpy(), expected)

    # let's just test that it defaults to float32:
    output = wp.zeros(7, dtype=wp.float32, device=device)
    wp.launch(quat_identity_default_kernel, dim=1, inputs=[], outputs=[output], device=device)
    expected = np.zeros_like(output.numpy())
    expected[-1] = 1
    assert_np_equal(output.numpy(), expected)


def test_transform_anon_type_instance(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def transform_create_test(input: wp.array(dtype=wptype), output: wp.array(dtype=wptype)):
        t = wp.transformation(
            wp.vector(input[0], input[1], input[2]), wp.quaternion(input[3], input[4], input[5], input[6])
        )
        for i in range(7):
            output[i] = wptype(2) * t[i]

    transform_create_kernel = getkernel(transform_create_test, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    input = wp.array(rng.standard_normal(size=7).astype(dtype), requires_grad=True, device=device)
    output = wp.zeros(7, dtype=wptype, requires_grad=True, device=device)
    wp.launch(transform_create_kernel, dim=1, inputs=[input], outputs=[output], device=device)
    assert_np_equal(output.numpy(), 2 * input.numpy())

    for i in range(len(input)):
        cmp = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(transform_create_kernel, dim=1, inputs=[input], outputs=[output], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[output, i], outputs=[cmp], device=device)
        tape.backward(loss=cmp)
        expectedgrads = np.zeros(len(input))
        expectedgrads[i] = 2
        assert_np_equal(tape.gradients[input].numpy(), expectedgrads)
        tape.zero()


def test_transform_from_matrix(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat44 = wp.types.matrix((4, 4), wptype)
    vec3 = wp.types.vector(3, wptype)
    quat = wp.types.quaternion(wptype)

    def transform_from_matrix_kernel():
        # fmt: off
        m = mat44(
            wptype(0.6), wptype(0.48), wptype(0.64), wptype(1.0),
            wptype(-0.8), wptype(0.36), wptype(0.48), wptype(2.0),
            wptype(0.0), wptype(-0.8), wptype(0.6), wptype(3.0),
            wptype(0.0), wptype(0.0), wptype(0.0), wptype(1.0),
        )
        # fmt: on
        t = wp.transform_from_matrix(m)
        p = wp.transform_get_translation(t)
        q = wp.transform_get_rotation(t)
        wp.expect_near(p, vec3(wptype(1.0), wptype(2.0), wptype(3.0)), tolerance=wptype(1e-3))
        wp.expect_near(q, quat(wptype(-0.4), wptype(0.2), wptype(-0.4), wptype(0.8)), tolerance=wptype(1e-3))

    kernel = getkernel(transform_from_matrix_kernel, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=1, device=device)


def test_transform_to_matrix(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat44 = wp.types.matrix((4, 4), wptype)
    vec3 = wp.types.vector(3, wptype)
    quat = wp.types.quaternion(wptype)

    def transform_to_matrix_kernel():
        p = vec3(wptype(1.0), wptype(2.0), wptype(3.0))
        q = quat(wptype(-0.4), wptype(0.2), wptype(-0.4), wptype(0.8))
        t = wp.transformation(p, q)
        m = wp.transform_to_matrix(t)
        # fmt: off
        wp.expect_near(
            m,
            mat44(
                wptype(0.6), wptype(0.48), wptype(0.64), wptype(1.0),
                wptype(-0.8), wptype(0.36), wptype(0.48), wptype(2.0),
                wptype(0.0), wptype(-0.8), wptype(0.6), wptype(3.0),
                wptype(0.0), wptype(0.0), wptype(0.0), wptype(1.0),
            ),
            tolerance=wptype(1e-3),
        )
        # fmt: on

    kernel = getkernel(transform_to_matrix_kernel, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=1, device=device)


def test_transform_compose(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat44 = wp.types.matrix((4, 4), wptype)
    vec3 = wp.types.vector(3, wptype)
    quat = wp.types.quaternion(wptype)

    def transform_compose_kernel():
        p = vec3(wptype(1.0), wptype(2.0), wptype(3.0))
        q = quat(wptype(-0.4), wptype(0.2), wptype(-0.4), wptype(0.8))
        s = vec3(wptype(4.0), wptype(5.0), wptype(6.0))
        m = wp.transform_compose(p, q, s)
        # fmt: off
        wp.expect_near(
            m,
            mat44(
                wptype(0.6 * 4.0), wptype(0.48 * 5.0), wptype(0.64 * 6.0), wptype(1.0),
                wptype(-0.8 * 4.0), wptype(0.36 * 5.0), wptype(0.48 * 6.0), wptype(2.0),
                wptype(0.0 * 4.0), wptype(-0.8 * 5.0), wptype(0.6 * 6.0), wptype(3.0),
                wptype(0.0), wptype(0.0), wptype(0.0), wptype(1.0),
            ),
            tolerance=wptype(1e-2),
        )
        # fmt: on

    kernel = getkernel(transform_compose_kernel, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=1, device=device)


def test_transform_decompose(test, device, dtype, register_kernels=False):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat44 = wp.types.matrix((4, 4), wptype)
    vec3 = wp.types.vector(3, wptype)
    quat = wp.types.quaternion(wptype)

    def transform_decompose_kernel():
        # fmt: off
        m = mat44(
            wptype(0.6 * 4.0), wptype(0.48 * 5.0), wptype(0.64 * 6.0), wptype(1.0),
            wptype(-0.8 * 4.0), wptype(0.36 * 5.0), wptype(0.48 * 6.0), wptype(2.0),
            wptype(0.0 * 4.0), wptype(-0.8 * 5.0), wptype(0.6 * 6.0), wptype(3.0),
            wptype(0.0), wptype(0.0), wptype(0.0), wptype(1.0),
        )
        # fmt: on
        p, q, s = wp.transform_decompose(m)
        wp.expect_near(p, vec3(wptype(1.0), wptype(2.0), wptype(3.0)), tolerance=wptype(1e-2))
        wp.expect_near(q, quat(wptype(-0.4), wptype(0.2), wptype(-0.4), wptype(0.8)), tolerance=wptype(1e-2))
        wp.expect_near(s, vec3(wptype(4.0), wptype(5.0), wptype(6.0)), tolerance=wptype(1e-2))

    kernel = getkernel(transform_decompose_kernel, suffix=dtype.__name__)

    if register_kernels:
        return

    wp.launch(kernel, dim=1, device=device)


devices = get_test_devices()


class TestSpatial(unittest.TestCase):
    pass


for dtype in np_float_types:
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_vector_constructors_{dtype.__name__}",
        test_spatial_vector_constructors,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_vector_indexing_{dtype.__name__}",
        test_spatial_vector_indexing,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_vector_scalar_multiplication_{dtype.__name__}",
        test_spatial_vector_scalar_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_vector_add_sub_{dtype.__name__}",
        test_spatial_vector_add_sub,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial, f"test_spatial_dot_{dtype.__name__}", test_spatial_dot, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpatial, f"test_spatial_cross_{dtype.__name__}", test_spatial_cross, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_top_bottom_{dtype.__name__}",
        test_spatial_top_bottom,
        devices=devices,
        dtype=dtype,
    )

    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_constructors_{dtype.__name__}",
        test_transform_constructors,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_anon_type_instance_{dtype.__name__}",
        test_transform_anon_type_instance,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_identity_{dtype.__name__}",
        test_transform_identity,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_indexing_{dtype.__name__}",
        test_transform_indexing,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_get_trans_rot_{dtype.__name__}",
        test_transform_get_trans_rot,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_multiply_{dtype.__name__}",
        test_transform_multiply,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_inverse_{dtype.__name__}",
        test_transform_inverse,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_point_vector_{dtype.__name__}",
        test_transform_point_vector,
        devices=devices,
        dtype=dtype,
    )

    # are these two valid? They don't seem to be doing things you'd want to do,
    # maybe they should be removed
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_scalar_multiplication_{dtype.__name__}",
        test_transform_scalar_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_add_sub_{dtype.__name__}",
        test_transform_add_sub,
        devices=devices,
        dtype=dtype,
    )

    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_matrix_constructors_{dtype.__name__}",
        test_spatial_matrix_constructors,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_matrix_indexing_{dtype.__name__}",
        test_spatial_matrix_indexing,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_matrix_scalar_multiplication_{dtype.__name__}",
        test_spatial_matrix_scalar_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_matrix_add_sub_{dtype.__name__}",
        test_spatial_matrix_add_sub,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_matvec_multiplication_{dtype.__name__}",
        test_spatial_matvec_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_matmat_multiplication_{dtype.__name__}",
        test_spatial_matmat_multiplication,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_spatial_outer_product_{dtype.__name__}",
        test_spatial_outer_product,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial, f"test_spatial_adjoint_{dtype.__name__}", test_spatial_adjoint, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_from_matrix_{dtype.__name__}",
        test_transform_from_matrix,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_to_matrix_{dtype.__name__}",
        test_transform_to_matrix,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_compose_{dtype.__name__}",
        test_transform_compose,
        devices=devices,
        dtype=dtype,
    )
    add_function_test_register_kernel(
        TestSpatial,
        f"test_transform_decompose_{dtype.__name__}",
        test_transform_decompose,
        devices=devices,
        dtype=dtype,
    )

    # \TODO: test spatial_mass and spatial_jacobian


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
