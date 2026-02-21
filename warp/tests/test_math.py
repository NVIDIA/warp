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
from typing import Any, NamedTuple

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices

TAIT_BRYAN_SEQUENCES = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


class ScalarFloatValues(NamedTuple):
    degrees: wp.float32 = None
    radians: wp.float32 = None


@wp.kernel
def scalar_float_kernel(
    i: int,
    x: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    if i == 0:
        out[0] = wp.degrees(x[0])
    elif i == 1:
        out[0] = wp.radians(x[0])


def test_scalar_math(test, device):
    float_values = ScalarFloatValues(degrees=(0.123,), radians=(123.0,))
    float_results_expected = ScalarFloatValues(degrees=7.047381, radians=2.146755)
    adj_float_results_expected = ScalarFloatValues(degrees=57.29578, radians=0.017453)
    for i, values in enumerate(float_values):
        x = wp.array([values[0]], dtype=wp.float32, requires_grad=True, device=device)
        out = wp.array([0.0], dtype=wp.float32, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(scalar_float_kernel, dim=1, inputs=[i, x, out], device=device)

        assert_np_equal(out.numpy(), np.array([float_results_expected[i]]), tol=1e-6)

        tape.backward(out)

        assert_np_equal(tape.gradients[x].numpy(), np.array([adj_float_results_expected[i]]), tol=1e-6)


@wp.kernel
def erf_kernel(x: wp.array(dtype=Any), out: wp.array(dtype=Any)):
    i = wp.tid()

    if i == 0:
        out[i] = wp.erf(x[i])
    elif i == 1:
        out[i] = wp.erfc(x[i])
    elif i == 2:
        out[i] = wp.erfinv(x[i])
    elif i == 3:
        out[i] = wp.erfcinv(x[i])


def test_erf_math(test, device):
    for type, tol in ((wp.float16, 1e-3), (wp.float32, 1e-6), (wp.float64, 1e-6)):
        x = wp.full(4, value=0.123, dtype=type, requires_grad=True, device=device)
        out = wp.zeros(4, dtype=type, requires_grad=True, device=device)

        with wp.Tape() as tape:
            wp.launch(erf_kernel, dim=4, inputs=[x], outputs=[out], device=device)

        out.grad = wp.ones_like(out)

        tape.backward()

        out_true = np.array([0.13809388, 0.86190612, 0.10944129, 1.09057285])
        adj_x_true = np.array([1.11143641, -1.11143641, 0.89690544, -2.91120449])

        assert_np_equal(out.numpy(), out_true, tol=tol)
        assert_np_equal(adj_x_true, x.grad.numpy(), tol=tol)


@wp.kernel
def test_vec_norm_kernel(vs: wp.array(dtype=Any), out: wp.array(dtype=float, ndim=2)):
    tid = wp.tid()
    out[tid, 0] = wp.norm_l1(vs[tid])
    out[tid, 1] = wp.norm_l2(vs[tid])
    out[tid, 2] = wp.norm_huber(vs[tid])
    out[tid, 3] = wp.norm_pseudo_huber(vs[tid])


def test_vec_norm(test, device):
    # ground-truth implementations from SciPy
    def huber(delta, x):
        if x <= delta:
            return 0.5 * x**2
        else:
            return delta * (x - 0.5 * delta)

    def pseudo_huber(delta, x):
        return delta**2 * (np.sqrt(1 + (x / delta) ** 2) - 1)

    v0 = wp.vec3(-2.0, -1.0, -3.0)
    v1 = wp.vec3(2.0, 1.0, 3.0)
    v2 = wp.vec3(0.0, 0.0, 0.0)

    xs = wp.array([v0, v1, v2], dtype=wp.vec3, requires_grad=True, device=device)
    out = wp.empty((len(xs), 4), dtype=wp.float32, requires_grad=True, device=device)

    wp.launch(test_vec_norm_kernel, dim=len(xs), inputs=[xs], outputs=[out], device=device)

    for i, x in enumerate([v0, v1, v2]):
        assert_np_equal(
            out.numpy()[i],
            np.array(
                [
                    np.linalg.norm(x, ord=1),
                    np.linalg.norm(x, ord=2),
                    huber(1.0, wp.length(x)),
                    # note SciPy defines the Pseudo-Huber loss slightly differently
                    pseudo_huber(1.0, wp.length(x)) + 1.0,
                ]
            ),
            tol=1e-6,
        )


@wp.kernel
def smooth_normalize_kernel(out: wp.array(dtype=float)):
    zero = wp.vec3(0.0, 0.0, 0.0)
    zero_n = wp.smooth_normalize(zero)

    v = wp.vec3(1.2, -0.4, 0.8)
    v_n = wp.smooth_normalize(v)
    v_ref = v / wp.norm_pseudo_huber(v)

    out[0] = wp.length(zero_n)
    out[1] = wp.length(v_n - v_ref)


def test_smooth_normalize(test, device):
    out = wp.empty(2, dtype=wp.float32, device=device)
    wp.launch(smooth_normalize_kernel, dim=1, outputs=[out], device=device)
    assert_np_equal(out.numpy(), np.zeros(2, dtype=np.float32), tol=1e-6)


@wp.kernel
def quat_helpers_kernel(out: wp.array(dtype=float, ndim=2)):
    tid = wp.tid()

    rpy = wp.vec3(0.3, -0.25, 0.7)
    q_rpy = wp.quat_rpy(rpy[0], rpy[1], rpy[2])
    rpy_back = wp.quat_to_rpy(q_rpy)

    e_xyz = wp.vec3(0.2, -0.1, 0.3)
    q_xyz = wp.quat_from_euler(e_xyz, 0, 1, 2)
    e_xyz_back = wp.quat_to_euler(q_xyz, 0, 1, 2)
    q_xyz_back = wp.quat_from_euler(e_xyz_back, 0, 1, 2)
    e_yxz = wp.vec3(0.25, -0.4, 0.15)
    q_yxz = wp.quat_from_euler(e_yxz, 1, 0, 2)
    e_yxz_back = wp.quat_to_euler(q_yxz, 1, 0, 2)
    q_yxz_back = wp.quat_from_euler(e_yxz_back, 1, 0, 2)
    q_zyx = wp.quat_from_euler(wp.vec3(0.4, -0.2, 0.3), 2, 1, 0)
    q_zyx_expected = wp.quat(0.18083558, -0.12628517, 0.12611651, 0.96718415)

    q_axis = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), e_xyz[0])
    q_from_single_axis = wp.quat_from_euler(wp.vec3(e_xyz[0], 0.0, 0.0), 0, 1, 2)
    e_dec = wp.quat_to_euler(q_axis, 2, 1, 0)
    q_dec_axis = wp.quat_from_euler(e_dec, 0, 1, 2)
    e_compound = wp.vec3(0.3, -0.2, 0.4)
    q_compound_xyz = wp.quat_from_euler(e_compound, 0, 1, 2)
    q_compound_zyx = wp.quat_from_euler(e_compound, 2, 1, 0)
    e_dec_compound_zyx = wp.quat_to_euler(q_compound_zyx, 2, 1, 0)
    q_dec_compound_zyx = wp.quat_from_euler(e_dec_compound_zyx, 2, 1, 0)
    e_dec_compound_xyz = wp.quat_to_euler(q_compound_xyz, 2, 1, 0)
    _q_dec_compound_xyz_as_zyx = wp.quat_from_euler(e_dec_compound_xyz, 2, 1, 0)

    e_gimbal = wp.vec3(0.6, wp.HALF_PI, 0.2)
    q_gimbal = wp.quat_from_euler(e_gimbal, 2, 1, 0)
    e_dec_gimbal = wp.quat_to_euler(q_gimbal, 2, 1, 0)
    q_dec_gimbal = wp.quat_from_euler(e_dec_gimbal, 2, 1, 0)

    axis = wp.vec3(0.0, 0.0, 1.0)
    angle_in = 0.9
    q_tw = wp.quat_from_axis_angle(axis, angle_in)
    q_swing = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.4)
    q_mix = wp.mul(q_swing, q_tw)
    twist = wp.quat_twist(axis, q_mix)
    q_tw_neg = wp.quat_from_axis_angle(axis, -angle_in)

    out[tid, 0] = wp.length(rpy_back - rpy)
    out[tid, 1] = quat_sign_invariant_error(q_xyz_back, q_xyz)
    out[tid, 2] = wp.length(e_dec - wp.vec3(e_xyz[0], 0.0, 0.0))
    out[tid, 3] = wp.length(
        wp.vec4(
            q_from_single_axis[0] - q_axis[0],
            q_from_single_axis[1] - q_axis[1],
            q_from_single_axis[2] - q_axis[2],
            q_from_single_axis[3] - q_axis[3],
        )
    )
    out[tid, 4] = quat_sign_invariant_error(q_dec_axis, q_axis)
    out[tid, 5] = quat_sign_invariant_error(twist, q_tw)
    out[tid, 6] = wp.abs(wp.quat_twist_angle(axis, q_mix) - angle_in)
    out[tid, 7] = wp.abs(wp.quat_twist_angle(axis, q_tw_neg) - angle_in)
    out[tid, 8] = quat_sign_invariant_error(q_zyx, q_zyx_expected)
    out[tid, 9] = quat_sign_invariant_error(q_yxz_back, q_yxz)
    out[tid, 10] = quat_sign_invariant_error(q_dec_compound_zyx, q_compound_zyx)
    out[tid, 11] = wp.max(0.0, 1.0e-2 - wp.length(e_dec_compound_xyz - e_compound))
    out[tid, 12] = quat_sign_invariant_error(q_dec_gimbal, q_gimbal)


@wp.func
def quat_sign_invariant_error(a: wp.quat, b: wp.quat) -> float:
    q_d = wp.vec4(a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3])
    q_s = wp.vec4(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])
    return wp.min(wp.length(q_d), wp.length(q_s))


@wp.func
def euler_roundtrip_error(e: wp.vec3, i: int, j: int, k: int) -> float:
    q = wp.quat_from_euler(e, i, j, k)
    e_back = wp.quat_to_euler(q, i, j, k)
    q_back = wp.quat_from_euler(e_back, i, j, k)
    return quat_sign_invariant_error(q_back, q)


@wp.kernel
def quat_euler_roundtrip_kernel(out: wp.array(dtype=float, ndim=2)):
    tid = wp.tid()

    i = 0
    j = 1
    k = 2
    if tid == 1:
        i, j, k = 0, 2, 1
    elif tid == 2:
        i, j, k = 1, 0, 2
    elif tid == 3:
        i, j, k = 1, 2, 0
    elif tid == 4:
        i, j, k = 2, 0, 1
    elif tid == 5:
        i, j, k = 2, 1, 0

    e = wp.vec3(0.31, -0.27, 0.41)
    e_near_pos = wp.vec3(e[0], e[1], e[2])
    e_near_pos[j] = wp.HALF_PI - 1.0e-4
    e_near_neg = wp.vec3(e[0], e[1], e[2])
    e_near_neg[j] = -wp.HALF_PI + 1.0e-4

    out[tid, 0] = euler_roundtrip_error(e, i, j, k)
    out[tid, 1] = euler_roundtrip_error(e_near_pos, i, j, k)
    out[tid, 2] = euler_roundtrip_error(e_near_neg, i, j, k)


def quat_helpers(test, device):
    out = wp.empty((1, 13), dtype=wp.float32, device=device)
    wp.launch(quat_helpers_kernel, dim=1, outputs=[out], device=device)

    out_np = out.numpy()[0]
    assert_np_equal(out_np, np.zeros(13, dtype=np.float32), tol=1e-5)


def quat_euler_roundtrip(test, device):
    out = wp.empty((len(TAIT_BRYAN_SEQUENCES), 3), dtype=wp.float32, device=device)
    wp.launch(quat_euler_roundtrip_kernel, dim=len(TAIT_BRYAN_SEQUENCES), outputs=[out], device=device)

    out_np = out.numpy()
    assert_np_equal(out_np, np.zeros((len(TAIT_BRYAN_SEQUENCES), 3), dtype=np.float32), tol=1e-4)


@wp.kernel
def spatial_helpers_kernel(out: wp.array(dtype=float, ndim=2)):
    tid = wp.tid()

    q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
    p = wp.vec3(1.0, 2.0, 3.0)
    t = wp.transform(p, q)
    t_inv = wp.transform_inverse(t)

    # Spatial vectors follow the Warp convention: (angular, linear).
    twist_in = wp.spatial_vector(wp.vec3(0.0, 0.0, 2.0), wp.vec3(1.0, 0.0, 0.0))
    twist_out = wp.transform_twist(t, twist_in)
    twist_roundtrip = wp.transform_twist(t_inv, twist_out)
    w_expected = wp.vec3(0.0, 0.0, 2.0)
    v_expected = wp.vec3(4.0, -1.0, 0.0)

    wrench_in = wp.spatial_vector(wp.vec3(0.0, 1.0, 0.0), wp.vec3(1.0, 0.0, 0.0))
    wrench_out = wp.transform_wrench(t, wrench_in)
    wrench_roundtrip = wp.transform_wrench(t_inv, wrench_out)
    tau_expected = wp.vec3(-4.0, 0.0, 1.0)
    f_expected = wp.vec3(0.0, 1.0, 0.0)

    qd = wp.spatial_vector(wp.vec3(0.0, 0.0, 2.0), wp.vec3(1.0, 2.0, 3.0))
    r = wp.vec3(0.5, 0.0, 0.0)
    vel = wp.velocity_at_point(qd, r)
    vel_expected = wp.vec3(1.0, 3.0, 3.0)

    out[tid, 0] = wp.length(wp.spatial_top(twist_out) - w_expected)
    out[tid, 1] = wp.length(wp.spatial_bottom(twist_out) - v_expected)
    out[tid, 2] = wp.length(wp.spatial_top(twist_roundtrip) - wp.spatial_top(twist_in))
    out[tid, 3] = wp.length(wp.spatial_bottom(twist_roundtrip) - wp.spatial_bottom(twist_in))
    out[tid, 4] = wp.length(wp.spatial_top(wrench_out) - tau_expected)
    out[tid, 5] = wp.length(wp.spatial_bottom(wrench_out) - f_expected)
    out[tid, 6] = wp.length(wp.spatial_top(wrench_roundtrip) - wp.spatial_top(wrench_in))
    out[tid, 7] = wp.length(wp.spatial_bottom(wrench_roundtrip) - wp.spatial_bottom(wrench_in))
    out[tid, 8] = wp.length(vel - vel_expected)


def spatial_helpers(test, device):
    out = wp.empty((1, 9), dtype=wp.float32, device=device)
    wp.launch(spatial_helpers_kernel, dim=1, outputs=[out], device=device)

    out_np = out.numpy()[0]
    assert_np_equal(out_np, np.zeros(9, dtype=np.float32), tol=1e-5)


devices = get_test_devices()


class TestMath(unittest.TestCase):
    def test_vec_type(self):
        vec5 = wp.types.vector(length=5, dtype=float)
        v = vec5()
        w = vec5()
        a = vec5(1.0)
        b = vec5(0.0, 0.0, 0.0, 0.0, 0.0)
        c = vec5(0.0)

        v[0] = 1.0
        v.x = 0.0
        v[1:] = [1.0, 1.0, 1.0, 1.0]

        w[0] = 1.0
        w[1:] = [0.0, 0.0, 0.0, 0.0]

        self.assertEqual(v[0], w[1], "vec setter error")
        self.assertEqual(v.x, w.y, "vec setter error")

        for x in v[1:]:
            self.assertEqual(x, 1.0, "vec slicing error")

        self.assertEqual(b, c, "vec equality error")

        self.assertEqual(str(v), "[0.0, 1.0, 1.0, 1.0, 1.0]", "vec to string error")

    def test_mat_type(self):
        mat55 = wp.types.matrix(shape=(5, 5), dtype=float)
        m1 = mat55()
        m2 = mat55()

        for i in range(5):
            for j in range(5):
                if i == j:
                    m1[i, j] = 1.0
                else:
                    m1[i, j] = 0.0

        for i in range(5):
            m2[i] = [1.0, 1.0, 1.0, 1.0, 1.0]

        a = mat55(1.0)
        # fmt: off
        b = mat55(
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0,
        )
        # fmt: on

        self.assertEqual(m1, b, "mat element setting error")
        self.assertEqual(m2, a, "mat row setting error")
        self.assertEqual(m1[0, 0], 1.0, "mat element getting error")
        self.assertEqual(m2[0], [1.0, 1.0, 1.0, 1.0, 1.0], "mat row getting error")
        self.assertEqual(
            str(b),
            "[[1.0, 0.0, 0.0, 0.0, 0.0],\n [0.0, 1.0, 0.0, 0.0, 0.0],\n [0.0, 0.0, 1.0, 0.0, 0.0],\n [0.0, 0.0, 0.0, 1.0, 0.0],\n [0.0, 0.0, 0.0, 0.0, 1.0]]",
            "mat to string error",
        )


add_function_test(TestMath, "test_scalar_math", test_scalar_math, devices=devices)
add_function_test(TestMath, "test_erf_math", test_erf_math, devices=devices)
add_function_test(TestMath, "test_vec_norm", test_vec_norm, devices=devices)
add_function_test(TestMath, "test_smooth_normalize", test_smooth_normalize, devices=devices)
add_function_test(TestMath, "test_quat_helpers", quat_helpers, devices=devices)
add_function_test(TestMath, "test_quat_euler_roundtrip", quat_euler_roundtrip, devices=devices)
add_function_test(TestMath, "test_spatial_helpers", spatial_helpers, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
