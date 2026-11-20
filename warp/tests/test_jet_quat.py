# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quaternion forward-mode jets, checked against native reverse mode and finite differences."""

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# A quaternion and a point: width 7, seeded with the identity.
JQ = wp.JetSpace((wp.quatd, wp.vec3d))

# A rotation-vector tangent and a point: width 6, for the exp_map chart.
JT = wp.JetSpace((wp.vec3d, wp.vec3d))

# Second-order checks use few directions: a second-order jet carries width**2 Hessian
# entries through every operation, and the CUDA compile time grows quickly with that.
# Two scalar parameters drive a jet quaternion and a jet vector together.
J2S = wp.JetSpace2(2, dtype=wp.float64)
# A rotation-vector tangent alone, width 3, for the exp_map chart.
J2R = wp.JetSpace2((wp.vec3d,))
# Only used to check the types of a second-order quaternion slot.
J2Q = wp.JetSpace2((wp.quatd, wp.vec3d))

# The quaternion and point as functions of the two parameters (s, t), around a base point:
# q = q0 + (s, -t, t, 0) and v = v0 + (t, s, 0). Column j holds the derivative along parameter j.
PARAM_DIRECTIONS = np.array(
    [
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ]
)

# One quaternion with one explicit direction.
JQ_DIR = wp.JetSpace((wp.quatd,), width=1)

Q_POINT = np.array([0.3, -0.5, 0.4, 0.7])
V_POINT = np.array([0.6, -0.8, 1.3])


@wp.func
def quat_energy(q: Any, v: Any):
    # Exercises every quaternion operation on jets, with jet and plain operands.
    c = wp.vec3d(wp.float64(0.3), wp.float64(-0.2), wp.float64(0.5))
    q2 = wp.quatd(wp.float64(0.1), wp.float64(0.2), wp.float64(-0.3), wp.float64(0.9))
    half = wp.float64(0.5)
    r = wp.quat_rotate(q, v)
    s = wp.quat_rotate_inv(q, c)
    t = wp.quat_rotate(q2, v)
    u = wp.quat_rotate_inv(q2, v)
    a = wp.quat_from_axis_angle(v, q[0])
    b = wp.quat_from_axis_angle(c, q[1])
    d = wp.quat_from_axis_angle(v, half)
    e = wp.dot(r, s) + wp.dot(t, u) + wp.dot(q * q2, q2 * q) + wp.dot(q * q, q2) + wp.dot(q2, wp.normalize(q))
    e += wp.dot(wp.quat_inverse(q), a) + wp.dot(b, q) + wp.dot(d, q) + wp.length(q) + wp.length_sq(q)
    e += wp.dot((q + q2 - (-q) * half) * q[1], q2 - q) + wp.dot(q2 + q, half * q)
    return e


@wp.kernel
def native_energy_kernel(q: wp.array[wp.quatd], v: wp.array[wp.vec3d], out: wp.array[wp.float64]):
    out[0] = quat_energy(q[0], v[0])


@wp.kernel(enable_backward=False)
def jet_energy_kernel(
    q: wp.array[wp.quatd],
    v: wp.array[wp.vec3d],
    values: wp.array[wp.float64],
    grads: wp.array[JQ.grad_type_scalar],
    grads2: wp.array[J2S.grad_type_scalar],
    hessians2: wp.array[J2S.hess_type_scalar],
):
    jq, jv = JQ.seed(q[0], v[0])
    e = quat_energy(jq, jv)
    values[0] = e.value
    grads[0] = e.grad

    # Second order, along the two parameters of PARAM_DIRECTIONS.
    s = J2S.variable(wp.float64(0.0), 0)
    t = J2S.variable(wp.float64(0.0), 1)
    kq = J2S.make_quat(
        J2S.constant(q[0][0]) + s, J2S.constant(q[0][1]) - t, J2S.constant(q[0][2]) + t, J2S.constant(q[0][3])
    )
    kv = J2S.make_vec3(J2S.constant(v[0][0]) + t, J2S.constant(v[0][1]) + s, J2S.constant(v[0][2]))
    e2 = quat_energy(kq, kv)
    values[1] = e2.value
    grads2[0] = e2.grad
    hessians2[0] = e2.hess


def _native_energy_and_gradient(q_np, v_np, device):
    q = wp.array([q_np], dtype=wp.quatd, requires_grad=True, device=device)
    v = wp.array([v_np], dtype=wp.vec3d, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wp.float64, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(native_energy_kernel, dim=1, inputs=[q, v], outputs=[out], device=device)
    tape.backward(grads={out: wp.ones(1, dtype=wp.float64, device=device)})
    return float(out.numpy()[0]), np.concatenate([q.grad.numpy()[0], v.grad.numpy()[0]])


def test_quat_ops_match_native(test, device):
    """Match native reverse mode for every quaternion operation, and finite differences of it for the Hessian."""
    value, grad = _native_energy_and_gradient(Q_POINT, V_POINT, device)

    values = wp.zeros(2, dtype=wp.float64, device=device)
    grads = wp.zeros(1, dtype=JQ.grad_type_scalar, device=device)
    grads2 = wp.zeros(1, dtype=J2S.grad_type_scalar, device=device)
    hessians2 = wp.zeros(1, dtype=J2S.hess_type_scalar, device=device)
    wp.launch(
        jet_energy_kernel,
        dim=1,
        inputs=[
            wp.array([Q_POINT], dtype=wp.quatd, device=device),
            wp.array([V_POINT], dtype=wp.vec3d, device=device),
        ],
        outputs=[values, grads, grads2, hessians2],
        device=device,
    )
    np.testing.assert_allclose(values.numpy(), [value, value], rtol=1e-12)
    np.testing.assert_allclose(grads.numpy()[0], grad, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(grads2.numpy()[0], PARAM_DIRECTIONS.T @ grad, rtol=1e-10, atol=1e-12)

    # The Hessian along the parameters, from central differences of the native gradient.
    z = np.concatenate([Q_POINT, V_POINT])
    h = 1.0e-5
    fd = np.zeros((7, 2))
    for j in range(2):
        dz = h * PARAM_DIRECTIONS[:, j]
        plus = _native_energy_and_gradient((z + dz)[:4], (z + dz)[4:], device)[1]
        minus = _native_energy_and_gradient((z - dz)[:4], (z - dz)[4:], device)[1]
        fd[:, j] = (plus - minus) / (2.0 * h)
    expected = PARAM_DIRECTIONS.T @ fd
    hess = hessians2.numpy()[0]
    np.testing.assert_allclose(hess, expected, atol=1e-6 * np.abs(expected).max())
    np.testing.assert_allclose(hess, hess.T, atol=1e-12)


@wp.func
def native_exp_map(t: wp.vec3d):
    angle = wp.length(t)
    if angle > wp.float64(0.0):
        return wp.quat_from_axis_angle(t / angle, angle)
    return wp.quat_identity(wp.float64)


@wp.func
def chart_energy(q: Any, x: Any):
    c = wp.vec3d(wp.float64(0.3), wp.float64(-0.2), wp.float64(0.5))
    q2 = wp.quatd(wp.float64(0.1), wp.float64(0.2), wp.float64(-0.3), wp.float64(0.9))
    k = wp.dot(q, q2)
    return wp.dot(wp.quat_rotate(q, x), c) + k * k


@wp.kernel(enable_backward=False)
def native_chart_kernel(
    q0: wp.quatd, t: wp.array[wp.vec3d], x: wp.array[wp.vec3d], energy: wp.array[wp.float64], quats: wp.array[wp.quatd]
):
    i = wp.tid()
    q = q0 * native_exp_map(t[i])
    energy[i] = chart_energy(q, x[i])
    quats[i] = native_exp_map(t[i])


@wp.kernel(enable_backward=False)
def jet_chart_kernel(
    q0: wp.quatd,
    t: wp.array[wp.vec3d],
    x: wp.array[wp.vec3d],
    quats: wp.array2d[wp.quatd],
    grads: wp.array[JT.grad_type_scalar],
    grads2: wp.array[J2R.grad_type_scalar],
    hessians2: wp.array[J2R.hess_type_scalar],
):
    i = wp.tid()
    jt, jx = JT.seed(t[i], x[i])
    e = chart_energy(q0 * JT.exp_map(jt), jx)
    grads[i] = e.grad
    quats[i, 0] = JT.exp_map(jt).value
    # Second order along the tangent only.
    k = J2R.exp_map(J2R.seed(t[i]))
    e2 = chart_energy(q0 * k, x[i])
    grads2[i] = e2.grad
    hessians2[i] = e2.hess
    quats[i, 1] = wp.quatd(k.x.value, k.y.value, k.z.value, k.w.value)


def test_exp_map_chart(test, device):
    """Map rotation vectors to native quaternions, with derivatives matching finite differences at and away from 0."""
    q0 = wp.quatd(0.2, -0.3, 0.1, 0.9)
    # The origin and a small tangent use the series; the last one uses the closed form.
    t_np = np.array([[0.0, 0.0, 0.0], [2.0e-2, -1.0e-2, 1.0e-2], [0.4, -0.7, 0.3]])
    x_np = np.tile(V_POINT, (len(t_np), 1))
    n = len(t_np)

    def native(t, x):
        energy = wp.zeros(n, dtype=wp.float64, device=device)
        quats = wp.zeros(n, dtype=wp.quatd, device=device)
        wp.launch(
            native_chart_kernel,
            dim=n,
            inputs=[q0, wp.array(t, dtype=wp.vec3d, device=device), wp.array(x, dtype=wp.vec3d, device=device)],
            outputs=[energy, quats],
            device=device,
        )
        return energy.numpy(), quats.numpy()

    quats = wp.zeros((n, 2), dtype=wp.quatd, device=device)
    grads = wp.zeros(n, dtype=JT.grad_type_scalar, device=device)
    grads2 = wp.zeros(n, dtype=J2R.grad_type_scalar, device=device)
    hessians2 = wp.zeros(n, dtype=J2R.hess_type_scalar, device=device)
    wp.launch(
        jet_chart_kernel,
        dim=n,
        inputs=[q0, wp.array(t_np, dtype=wp.vec3d, device=device), wp.array(x_np, dtype=wp.vec3d, device=device)],
        outputs=[quats, grads, grads2, hessians2],
        device=device,
    )

    _, native_quats = native(t_np, x_np)
    np.testing.assert_allclose(quats.numpy()[:, 0], native_quats, atol=1e-14)
    np.testing.assert_allclose(quats.numpy()[:, 1], native_quats, atol=1e-14)

    # Gradient and Hessian by central differences of the native energy.
    z = np.concatenate([t_np, x_np], axis=1)

    def energy(zz):
        return native(np.ascontiguousarray(zz[:, :3]), np.ascontiguousarray(zz[:, 3:]))[0]

    h = 1.0e-4
    fd_grad = np.zeros((n, 6))
    fd_hess = np.zeros((n, 6, 6))
    for i in range(6):
        ei = np.zeros(6)
        ei[i] = h
        fd_grad[:, i] = (energy(z + ei) - energy(z - ei)) / (2.0 * h)
        for j in range(6):
            ej = np.zeros(6)
            ej[j] = h
            fd_hess[:, i, j] = (
                energy(z + ei + ej) - energy(z + ei - ej) - energy(z - ei + ej) + energy(z - ei - ej)
            ) / (4.0 * h * h)
    np.testing.assert_allclose(grads.numpy(), fd_grad, atol=1e-8)
    np.testing.assert_allclose(grads2.numpy(), fd_grad[:, :3], atol=1e-8)
    np.testing.assert_allclose(hessians2.numpy(), fd_hess[:, :3, :3], atol=1e-6)


@wp.kernel(enable_backward=False)
def native_degenerate_kernel(q: wp.array[wp.quatd], normalized: wp.array[wp.quatd], lengths: wp.array[wp.float64]):
    i = wp.tid()
    normalized[i] = wp.normalize(q[i])
    lengths[i] = wp.length(q[i])


@wp.kernel(enable_backward=False)
def jet_degenerate_kernel(
    q: wp.array[wp.quatd],
    normalized: wp.array2d[wp.quatd],
    lengths: wp.array2d[wp.float64],
    derivative_sums: wp.array2d[wp.float64],
):
    # Rows: first-order, second-order. The derivative sums add every gradient (and
    # Hessian) entry, so they are zero exactly when all derivatives are, and NaN if any is.
    i = wp.tid()
    jq = JQ_DIR.directional(q[i], wp.quatd(wp.float64(1.0), wp.float64(2.0), wp.float64(3.0), wp.float64(4.0)))
    n = wp.normalize(jq)
    length = wp.length(jq)
    normalized[0, i] = n.value
    lengths[0, i] = length.value
    total = length.grad[0]
    for k in range(4):
        total += n.grad[k, 0]
    derivative_sums[0, i] = total

    kq = J2S.make_quat(
        J2S.variable(q[i][0], 0), J2S.variable(q[i][1], 1), J2S.constant(q[i][2]), J2S.variable(q[i][3], 0)
    )
    n2 = wp.normalize(kq)
    length2 = wp.length(kq)
    normalized[1, i] = wp.quatd(n2.x.value, n2.y.value, n2.z.value, n2.w.value)
    lengths[1, i] = length2.value
    total2 = wp.float64(0.0)
    for k in range(2):
        total2 += length2.grad[k] + n2.x.grad[k] + n2.y.grad[k] + n2.z.grad[k] + n2.w.grad[k]
        for m in range(2):
            total2 += length2.hess[k, m] + n2.x.hess[k, m] + n2.w.hess[k, m]
    derivative_sums[1, i] = total2


def test_normalize_and_length_follow_native(test, device):
    """Normalize zero and NaN quaternions to the identity and keep zero lengths at zero derivatives, as native."""
    q_np = np.array([[0.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 1.0]])
    n = len(q_np)
    q = wp.array(q_np, dtype=wp.quatd, device=device)

    native_normalized = wp.zeros(n, dtype=wp.quatd, device=device)
    native_lengths = wp.zeros(n, dtype=wp.float64, device=device)
    wp.launch(native_degenerate_kernel, dim=n, inputs=[q], outputs=[native_normalized, native_lengths], device=device)

    normalized = wp.zeros((2, n), dtype=wp.quatd, device=device)
    lengths = wp.zeros((2, n), dtype=wp.float64, device=device)
    sums = wp.zeros((2, n), dtype=wp.float64, device=device)
    wp.launch(jet_degenerate_kernel, dim=n, inputs=[q], outputs=[normalized, lengths, sums], device=device)

    for order in range(2):
        np.testing.assert_array_equal(
            normalized.numpy()[order], native_normalized.numpy(), err_msg=f"order {order + 1}"
        )
        np.testing.assert_array_equal(lengths.numpy()[order], native_lengths.numpy(), err_msg=f"order {order + 1}")
    # Zero quaternion: every derivative is zero. NaN quaternion: normalize is constant, but the length is NaN.
    np.testing.assert_array_equal(sums.numpy()[:, 0], [0.0, 0.0])
    test.assertTrue(np.all(np.isnan(sums.numpy()[:, 1])))


@wp.kernel
def jet_origin_kernel(t: wp.array[wp.vec3d], out: wp.array[wp.float64]):
    jt, _jx = JT.seed(t[0], wp.vec3d())
    q = JT.exp_map(jt)
    zero = JQ_DIR.directional(wp.quatd(), wp.quatd(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)))
    k = J2S.exp_map(J2S.make_vec3(J2S.variable(t[0][0], 0), J2S.variable(t[0][1], 1), J2S.constant(t[0][2])))
    out[0] = q.value[0] + q.grad[0, 0] + q.grad[3, 1]
    out[1] = wp.normalize(zero).grad[0, 0] + wp.length(zero).grad[0]
    out[2] = k.x.value + k.x.grad[0] + k.w.hess[0, 0]


def test_reverse_mode_through_quat_jets(test, device):
    """Backpropagate through exp_map at the origin and through normalize and length of a zero jet without NaNs."""
    t = wp.zeros(1, dtype=wp.vec3d, requires_grad=True, device=device)
    out = wp.zeros(3, dtype=wp.float64, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(jet_origin_kernel, dim=1, inputs=[t], outputs=[out], device=device)
    tape.backward(grads={out: wp.ones(3, dtype=wp.float64, device=device)})
    test.assertTrue(np.all(np.isfinite(out.numpy())), out.numpy())
    test.assertTrue(np.all(np.isfinite(t.grad.numpy())), t.grad.numpy())


@wp.kernel(enable_backward=False)
def quat_slot_kernel(q: wp.quatd, dq: wp.quatd, seeded: wp.array[JQ.grad_type_quat], directional: wp.array[wp.quatd]):
    jq, _jv = JQ.seed(q, wp.vec3d())
    seeded[0] = jq.grad
    d = JQ_DIR.directional(q, dq)
    directional[0] = wp.quatd(d.grad[0, 0], d.grad[1, 0], d.grad[2, 0], d.grad[3, 0])


J1T = wp.JetSpace(1, dtype=wp.float64)
J2T = wp.JetSpace2(1, dtype=wp.float64)


@wp.kernel(enable_backward=False)
def blend_plain_endpoints_kernel(
    q0: wp.quatd, q1: wp.quatd, t0: wp.float64, values: wp.array[wp.quatd], derivatives: wp.array[wp.quatd]
):
    # Only the blend parameter is a jet; both operand orders scale a plain quaternion by it.
    t = J1T.variable(t0, 0)
    one = wp.float64(1.0)
    left = (one - t) * q0 + t * q1
    right = q0 * (one - t) + q1 * t
    values[0] = left.value
    values[1] = right.value
    derivatives[0] = wp.quatd(left.grad[0, 0], left.grad[1, 0], left.grad[2, 0], left.grad[3, 0])
    derivatives[1] = wp.quatd(right.grad[0, 0], right.grad[1, 0], right.grad[2, 0], right.grad[3, 0])
    t2 = J2T.variable(t0, 0)
    blend2 = (one - t2) * q0 + q1 * t2
    values[2] = wp.quatd(blend2.x.value, blend2.y.value, blend2.z.value, blend2.w.value)
    derivatives[2] = wp.quatd(blend2.x.grad[0], blend2.y.grad[0], blend2.z.grad[0], blend2.w.grad[0])
    derivatives[3] = wp.quatd(blend2.x.hess[0, 0], blend2.y.hess[0, 0], blend2.z.hess[0, 0], blend2.w.hess[0, 0])


def test_blend_plain_endpoints(test, device):
    """Check that a jet blend parameter scales plain quaternion endpoints in either operand order."""
    q0 = np.array([0.1, -0.2, 0.3, 0.9])
    q1 = np.array([-0.4, 0.5, 0.2, 0.7])
    t0 = 0.3
    values = wp.zeros(3, dtype=wp.quatd, device=device)
    derivatives = wp.zeros(4, dtype=wp.quatd, device=device)
    wp.launch(
        blend_plain_endpoints_kernel,
        1,
        inputs=[wp.quatd(*q0), wp.quatd(*q1), wp.float64(t0), values, derivatives],
        device=device,
    )
    expected = (1.0 - t0) * q0 + t0 * q1
    for k in range(3):
        np.testing.assert_allclose(values.numpy()[k], expected, rtol=1e-14)
        np.testing.assert_allclose(derivatives.numpy()[k], q1 - q0, rtol=1e-14)
    np.testing.assert_array_equal(derivatives.numpy()[3], np.zeros(4))


def test_quat_slots(test, device):
    """Seed quaternion slots with one direction per component, or with explicit directions."""
    test.assertIs(JQ.jet_type(wp.quatd), JQ.quat)
    test.assertIs(JQ.grad_type(wp.quatd), JQ.grad_type_quat)
    test.assertIs(J2Q.jet_type(wp.quatd), J2Q.quat)
    test.assertEqual(JQ.width, 7)
    with test.assertRaisesRegex(TypeError, "quat"):
        wp.JetSpace((wp.vec4d,))

    seeded = wp.zeros(1, dtype=JQ.grad_type_quat, device=device)
    directional = wp.zeros(1, dtype=wp.quatd, device=device)
    dq = wp.quatd(0.5, -1.0, 2.0, 0.25)
    wp.launch(quat_slot_kernel, dim=1, inputs=[wp.quatd(*Q_POINT), dq], outputs=[seeded, directional], device=device)
    np.testing.assert_array_equal(seeded.numpy()[0], np.eye(4, 7))
    np.testing.assert_array_equal(directional.numpy()[0], np.array(dq))


devices = get_test_devices()


class TestJetQuat(unittest.TestCase):
    pass


add_function_test(TestJetQuat, "test_quat_ops_match_native", test_quat_ops_match_native, devices=devices)
add_function_test(TestJetQuat, "test_exp_map_chart", test_exp_map_chart, devices=devices)
add_function_test(
    TestJetQuat, "test_normalize_and_length_follow_native", test_normalize_and_length_follow_native, devices=devices
)
add_function_test(
    TestJetQuat, "test_reverse_mode_through_quat_jets", test_reverse_mode_through_quat_jets, devices=devices
)
add_function_test(TestJetQuat, "test_quat_slots", test_quat_slots, devices=devices)
add_function_test(TestJetQuat, "test_blend_plain_endpoints", test_blend_plain_endpoints, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
