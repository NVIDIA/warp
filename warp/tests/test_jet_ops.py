# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Extended jet builtins (inverse-trig, pow variants, branching) for first- and
# second-order jets, checked against finite differences.
#
# CPU-only, by standard unittest.TestCase rather than add_function_test(). These
# kernels chain many of the new overloads into one large module, whose CUDA build
# takes far longer than a test should. CUDA coverage of the jets comes from
# test_jet.py, which runs on CPU and CUDA, and from the implicit-projection
# example test; the jet benchmarks only report their errors.

import unittest
from typing import Any

import numpy as np

import warp as wp

J2 = wp.JetSpace(2)
J2_2ND = wp.JetSpace2(2)

DEVICES = ["cpu"]

Z_NP = np.array([[0.7, -0.4], [-0.2, 0.9], [0.7, 0.9]], dtype=np.float32)
ZB_NP = np.array([[0.3, 0.7], [-0.2, 0.9], [0.4, -0.1]], dtype=np.float32)  # away from branch kinks


def _grad_fd(fn, z, h=1.0e-5):
    out = np.empty_like(z)
    for p in range(z.shape[1]):
        e = np.zeros_like(z)
        e[:, p] = h
        out[:, p] = (fn(z + e) - fn(z - e)) / (2.0 * h)
    return out


def _hess_fd(fn, z, h=1.0e-4):
    n, k = z.shape
    out = np.empty((n, k, k))
    for p in range(k):
        for q in range(k):
            ep = np.zeros_like(z)
            ep[:, p] = h
            eq = np.zeros_like(z)
            eq[:, q] = h
            out[:, p, q] = (fn(z + ep + eq) - fn(z + ep - eq) - fn(z - ep + eq) + fn(z - ep - eq)) / (4.0 * h * h)
    return out


# --------------------------------------------------------------------------
# First-order: inverse-trig, pow variants, and value-branching builtins.
# --------------------------------------------------------------------------


def smooth_np(z):
    a = z[:, 0]
    b = z[:, 1]
    return (
        np.arcsin(0.4 * a)
        + np.arccos(0.3 * b)
        + np.arctan(1.5 * a)
        + np.arctan2(a, b + 2.0)
        + np.power(a + 2.0, b)
        + np.power(3.0, b)
    )


@wp.kernel(enable_backward=False)
def jet_smooth(z: wp.array2d[float], val: wp.array[float], grad: wp.array[J2.grad_type_scalar]):
    i = wp.tid()
    a = J2.variable(z[i, 0], 0)
    b = J2.variable(z[i, 1], 1)
    e = (
        wp.asin(0.4 * a)
        + wp.acos(0.3 * b)
        + wp.atan(1.5 * a)
        + wp.atan2(a, b + 2.0)
        + wp.pow(a + 2.0, b)  # pow(jet, jet)
        + wp.pow(3.0, b)  # pow(dtype, jet)
    )
    val[i] = e.value
    grad[i] = e.grad


def branch_np(z):
    a = z[:, 0]
    b = z[:, 1]
    return np.minimum(a, b) + np.maximum(a, b) + np.clip(a, -0.5, 0.5) + np.where(a > 0.0, a, b) + np.abs(a)


@wp.kernel(enable_backward=False)
def jet_branch(z: wp.array2d[float], val: wp.array[float], grad: wp.array[J2.grad_type_scalar]):
    i = wp.tid()
    a = J2.variable(z[i, 0], 0)
    b = J2.variable(z[i, 1], 1)
    e = wp.min(a, b) + wp.max(a, b) + wp.clamp(a, -0.5, 0.5) + wp.where(a.value > 0.0, a, b) + wp.abs(a)
    val[i] = e.value
    grad[i] = e.grad


# --------------------------------------------------------------------------
# Second-order: same op families, checked for value, gradient, and Hessian.
# --------------------------------------------------------------------------


def g3_np(z):
    a = z[:, 0]
    b = z[:, 1]
    return (
        np.arctan2(a, b + 2.0)
        + np.power(a + 2.0, b)
        + np.power(3.0, b)
        + np.tan(0.3 * a)
        + np.arcsin(0.4 * b)
        + 1.0 / (a * a + 1.0)
    )


@wp.kernel(enable_backward=False)
def jet2_g3(z: wp.array2d[float], val: wp.array[float], grad: wp.array2d[float], hess: wp.array3d[float]):
    i = wp.tid()
    a = J2_2ND.variable(z[i, 0], 0)
    b = J2_2ND.variable(z[i, 1], 1)
    r = (
        wp.atan2(a, b + 2.0)
        + wp.pow(a + 2.0, b)  # pow(jet, jet)
        + wp.pow(3.0, b)  # pow(dtype, jet)
        + wp.tan(0.3 * a)
        + wp.asin(0.4 * b)
        + 1.0 / (a * a + 1.0)  # div(dtype, jet)
    )
    val[i] = r.value
    for p in range(2):
        grad[i, p] = r.grad[p]
        for q in range(2):
            hess[i, p, q] = r.hess[p, q]


def branch2_np(z):
    a = z[:, 0]
    b = z[:, 1]
    return np.minimum(a * a, b * b + 5.0) + np.maximum(a * a * a, b) + np.where(a > 0.0, a * a, b * b)


@wp.kernel(enable_backward=False)
def jet2_branch(z: wp.array2d[float], val: wp.array[float], grad: wp.array2d[float], hess: wp.array3d[float]):
    i = wp.tid()
    a = J2_2ND.variable(z[i, 0], 0)
    b = J2_2ND.variable(z[i, 1], 1)
    r = wp.min(a * a, b * b + 5.0) + wp.max(a * a * a, b) + wp.where(a.value > 0.0, a * a, b * b)
    val[i] = r.value
    for p in range(2):
        grad[i, p] = r.grad[p]
        for q in range(2):
            hess[i, p, q] = r.hess[p, q]


def _run_grad(kernel, z_np, device):
    m = z_np.shape[0]
    z = wp.array(z_np, dtype=float, device=device)
    val = wp.zeros(m, dtype=float, device=device)
    grad = wp.zeros(m, dtype=J2.grad_type_scalar, device=device)
    wp.launch(kernel, dim=m, inputs=[z], outputs=[val, grad], device=device)
    return val.numpy(), grad.numpy().reshape(m, 2)


def _run_hess(kernel, z_np, device):
    m = z_np.shape[0]
    z = wp.array(z_np, dtype=float, device=device)
    val = wp.zeros(m, dtype=float, device=device)
    grad = wp.zeros((m, 2), dtype=float, device=device)
    hess = wp.zeros((m, 2, 2), dtype=float, device=device)
    wp.launch(kernel, dim=m, inputs=[z], outputs=[val, grad, hess], device=device)
    return val.numpy(), grad.numpy(), hess.numpy()


# --------------------------------------------------------------------------
# pow(jet, jet) with a jet exponent, at and around a base of zero.
#
# a**b has finite partials at a = 0 for b > 1, but the textbook factorization
# a**b * (b/a) evaluates as 0 * inf there. Both orders are checked at a = 0
# exactly, where finite differences cannot be used as the oracle.
# --------------------------------------------------------------------------


@wp.kernel
def jet_pow_jet_exponent(z: wp.array2d[float], grad: wp.array[J2.grad_type_scalar]):
    i = wp.tid()
    a = J2.variable(z[i, 0], 0)
    b = J2.variable(z[i, 1], 1)
    grad[i] = wp.pow(a, b).grad


@wp.kernel
def jet_pow_const_exponent(z: wp.array[float], grad: wp.array2d[float]):
    # Degenerate constant exponents at a zero base: p * a**(p-1) and
    # p (p-1) a**(p-2) are 0 * inf there for p == 0 and p == 1.
    i = wp.tid()
    a = J2.variable(z[i], 0)
    grad[i, 0] = wp.pow(a, 0.0).grad[0]
    grad[i, 1] = wp.pow(a, 1.0).grad[0]
    grad[i, 2] = wp.pow(a, 2.0).grad[0]
    grad[i, 3] = wp.pow(a, 0).grad[0]
    grad[i, 4] = wp.pow(a, 1).grad[0]


@wp.kernel
def jet2_pow_const_exponent(z: wp.array[float], grad: wp.array2d[float], hess: wp.array2d[float]):
    i = wp.tid()
    a = J2_2ND.variable(z[i], 0)
    e0 = wp.pow(a, 0)
    grad[i, 0] = e0.grad[0]
    hess[i, 0] = e0.hess[0, 0]

    e1 = wp.pow(a, 1)
    grad[i, 1] = e1.grad[0]
    hess[i, 1] = e1.hess[0, 0]

    e1f = wp.pow(a, 1.0)
    grad[i, 2] = e1f.grad[0]
    hess[i, 2] = e1f.hess[0, 0]

    e2 = wp.pow(a, 2)
    grad[i, 3] = e2.grad[0]
    hess[i, 3] = e2.hess[0, 0]


@wp.kernel
def jet2_pow_jet_exponent(z: wp.array2d[float], grad: wp.array2d[float], hess: wp.array3d[float]):
    i = wp.tid()
    a = J2_2ND.variable(z[i, 0], 0)
    b = J2_2ND.variable(z[i, 1], 1)
    e = wp.pow(a, b)
    for p in range(2):
        grad[i, p] = e.grad[p]
        for q in range(2):
            hess[i, p, q] = e.hess[p, q]


# --------------------------------------------------------------------------
# Mixed jet/constant overloads that exist at first order must exist at second
# order too, or an energy written once cannot feed both strategies.
# --------------------------------------------------------------------------


@wp.func
def mixed_scalar_ops(a: J2.scalar, b: J2.scalar) -> J2.scalar:
    return wp.pow(a, 2.5) + wp.atan2(a, 0.75) + wp.atan2(1.25, b)


@wp.func
def mixed_scalar_ops2(a: J2_2ND.scalar, b: J2_2ND.scalar) -> J2_2ND.scalar:
    return wp.pow(a, 2.5) + wp.atan2(a, 0.75) + wp.atan2(1.25, b)


def mixed_scalar_np(z):
    return np.power(z[:, 0], 2.5) + np.arctan2(z[:, 0], 0.75) + np.arctan2(1.25, z[:, 1])


@wp.kernel
def jet_mixed(z: wp.array2d[float], val: wp.array[float], grad: wp.array[J2.grad_type_scalar]):
    i = wp.tid()
    e = mixed_scalar_ops(J2.variable(z[i, 0], 0), J2.variable(z[i, 1], 1))
    val[i] = e.value
    grad[i] = e.grad


@wp.kernel
def jet2_mixed(z: wp.array2d[float], val: wp.array[float], grad: wp.array2d[float], hess: wp.array3d[float]):
    i = wp.tid()
    e = mixed_scalar_ops2(J2_2ND.variable(z[i, 0], 0), J2_2ND.variable(z[i, 1], 1))
    val[i] = e.value
    for p in range(2):
        grad[i, p] = e.grad[p]
        for q in range(2):
            hess[i, p, q] = e.hess[p, q]


# --------------------------------------------------------------------------
# Square matrix jets: determinant, trace, transpose, matmul via a real energy.
# --------------------------------------------------------------------------

J4 = wp.JetSpace(4)
J9 = wp.JetSpace(9)
J9_nodes = wp.JetSpace((wp.vec3,) * 3)  # a triangle's three nodes; shares J9's types
J12 = wp.JetSpace((wp.vec3,) * 4)  # a tetrahedron's four nodes


# --------------------------------------------------------------------------
# mat22 multiplication: mat22 * mat22 and mat22 * vec2.
# --------------------------------------------------------------------------


@wp.kernel
def jet_mat22_matmul(z: wp.array2d[float], out: wp.array[J4.grad_type_scalar]):
    i = wp.tid()
    # A = [[z0, z2], [z1, z3]], built from its two seeded columns. Both
    # mat22 * mat22 and mat22 * vec2 appear in (A A) v.
    c0 = J4.make_vec2(J4.variable(z[i, 0], 0), J4.variable(z[i, 1], 1))
    c1 = J4.make_vec2(J4.variable(z[i, 2], 2), J4.variable(z[i, 3], 3))
    a = J4.make_mat22(c0, c1)
    v = J4.make_vec2(J4.constant(1.0), J4.constant(-2.0))
    w = (a * a) * v
    out[i] = (w[0] + w[1]).grad


def mat22_matmul_np(z):
    n = z.shape[0]
    a = np.empty((n, 2, 2))
    a[:, 0, 0] = z[:, 0]
    a[:, 1, 0] = z[:, 1]
    a[:, 0, 1] = z[:, 2]
    a[:, 1, 1] = z[:, 3]
    w = np.einsum("nij,njk,k->ni", a, a, np.array([1.0, -2.0]))
    return w[:, 0] + w[:, 1]


@wp.kernel(enable_backward=False)
def jet_mat22_detrace(a: wp.array2d[float], det_g: wp.array[J4.grad_type_scalar], tr_g: wp.array[J4.grad_type_scalar]):
    # Seed the four entries (row-major m00,m01,m10,m11) to directions 0..3.
    i = wp.tid()
    c0 = J4.make_vec2(J4.variable(a[i, 0], 0), J4.variable(a[i, 2], 2))  # column 0 = (m00, m10)
    c1 = J4.make_vec2(J4.variable(a[i, 1], 1), J4.variable(a[i, 3], 3))  # column 1 = (m01, m11)
    m = J4.make_mat22(c0, c1)
    det_g[i] = wp.determinant(m).grad
    tr_g[i] = wp.trace(m).grad


def tet_np(z):
    p0, p1, p2, p3 = z[:, 0:3], z[:, 3:6], z[:, 6:9], z[:, 9:12]
    f = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=2)  # columns
    i1 = (f * f).sum((1, 2))
    j = np.linalg.det(f)
    return 0.5 * (i1 - 3.0) - np.log(j) + 0.5 * np.log(j) ** 2


@wp.kernel(enable_backward=False)
def jet_tet(z: wp.array2d[float], grad: wp.array[J12.grad_type_scalar]):
    i = wp.tid()
    p0, p1, p2, p3 = J12.seed(
        wp.vec3(z[i, 0], z[i, 1], z[i, 2]),
        wp.vec3(z[i, 3], z[i, 4], z[i, 5]),
        wp.vec3(z[i, 6], z[i, 7], z[i, 8]),
        wp.vec3(z[i, 9], z[i, 10], z[i, 11]),
    )
    f = J12.make_mat33(p1 - p0, p2 - p0, p3 - p0)  # deformation gradient
    i1 = wp.trace(wp.transpose(f) * f)  # matmul -> trace = sum of squares
    logj = wp.log(wp.determinant(f))
    e = 0.5 * (i1 - 3.0) - logj + 0.5 * logj * logj
    grad[i] = e.grad


@wp.kernel(enable_backward=False)
def jet_mat33_inverse(a: wp.array2d[float], out: wp.array3d[float]):
    # Seed the 3x3 entries (row-major dirs 0..8); write d(inv[r,c])/d(entry k).
    i = wp.tid()
    c0 = J9.make_vec3(J9.variable(a[i, 0], 0), J9.variable(a[i, 3], 3), J9.variable(a[i, 6], 6))  # column 0
    c1 = J9.make_vec3(J9.variable(a[i, 1], 1), J9.variable(a[i, 4], 4), J9.variable(a[i, 7], 7))
    c2 = J9.make_vec3(J9.variable(a[i, 2], 2), J9.variable(a[i, 5], 5), J9.variable(a[i, 8], 8))
    inv = wp.inverse(J9.make_mat33(c0, c1, c2))
    for r in range(3):
        for c in range(3):
            for q in range(9):
                out[i, 3 * r + c, q] = inv.grad[3 * r + c, q]


def triangle_rect_np(z):
    p0, p1, p2 = z[:, 0:3], z[:, 3:6], z[:, 6:9]
    a, b = p1 - p0, p2 - p0
    s00 = (a * a).sum(1)
    s01 = (a * b).sum(1)
    s11 = (b * b).sum(1)
    tr = s00 + s11
    return tr + tr / (s00 * s11 - s01 * s01)


@wp.kernel(enable_backward=False)
def jet_triangle_rect(z: wp.array2d[float], grad: wp.array[J9.grad_type_scalar]):
    # 3x2 deformation gradient F, then S = F^T F (2x3 * 3x2 = 2x2), symmetric Dirichlet.
    i = wp.tid()
    p0, p1, p2 = J9_nodes.seed(
        wp.vec3(z[i, 0], z[i, 1], z[i, 2]), wp.vec3(z[i, 3], z[i, 4], z[i, 5]), wp.vec3(z[i, 6], z[i, 7], z[i, 8])
    )
    f = J9.make_mat32(p1 - p0, p2 - p0)
    s = wp.transpose(f) * f
    tr = wp.trace(s)
    e = tr + tr / wp.determinant(s)
    grad[i] = e.grad


class TestJetMatrix(unittest.TestCase):
    def test_mat22_det_trace(self):
        """Check mat22 determinant and trace derivatives against finite differences."""
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((5, 4)) + np.array([2.0, 0.1, 0.1, 2.0])).astype(np.float32)  # well-conditioned
        det_g = wp.zeros(5, dtype=J4.grad_type_scalar, device="cpu")
        tr_g = wp.zeros(5, dtype=J4.grad_type_scalar, device="cpu")
        wp.launch(
            jet_mat22_detrace,
            dim=5,
            inputs=[wp.array(a, dtype=float, device="cpu")],
            outputs=[det_g, tr_g],
            device="cpu",
        )
        m = a.astype(np.float64)
        # d det / d(m00,m01,m10,m11) = (m11, -m10, -m01, m00); d trace = (1,0,0,1)
        det_expected = np.stack([m[:, 3], -m[:, 2], -m[:, 1], m[:, 0]], axis=1)
        tr_expected = np.tile([1.0, 0.0, 0.0, 1.0], (5, 1))
        np.testing.assert_allclose(det_g.numpy().reshape(5, 4), det_expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(tr_g.numpy().reshape(5, 4), tr_expected, rtol=1e-5, atol=1e-6)

    def test_mat33_tet_energy(self):
        """Check a mat33 tetrahedron energy gradient against finite differences."""
        rng = np.random.default_rng(1)
        rest = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], np.float32)  # unit reference tet
        z = (rest + 0.05 * rng.standard_normal((6, 12))).astype(np.float32)
        grad = wp.zeros(6, dtype=J12.grad_type_scalar, device="cpu")
        wp.launch(jet_tet, dim=6, inputs=[wp.array(z, dtype=float, device="cpu")], outputs=[grad], device="cpu")
        np.testing.assert_allclose(
            grad.numpy().reshape(6, 12), _grad_fd(tet_np, z.astype(np.float64)), rtol=1e-3, atol=1e-4
        )

    def test_mat33_inverse(self):
        """Check the mat33 inverse derivative against finite differences."""
        rng = np.random.default_rng(2)
        a = (np.eye(3) + 0.3 * rng.standard_normal((4, 3, 3))).astype(np.float32).reshape(4, 9)
        out = wp.zeros((4, 9, 9), dtype=float, device="cpu")
        wp.launch(
            jet_mat33_inverse, dim=4, inputs=[wp.array(a, dtype=float, device="cpu")], outputs=[out], device="cpu"
        )
        got = out.numpy()
        # FD of inv(A) w.r.t. each entry, compared to d(inv[r,c])/d(entry k).
        h = 1e-4
        for n in range(4):
            A = a[n].reshape(3, 3).astype(np.float64)
            for k in range(9):
                r, c = k // 3, k % 3
                ap = A.copy()
                ap[r, c] += h
                am = A.copy()
                am[r, c] -= h
                dinv = (np.linalg.inv(ap) - np.linalg.inv(am)) / (2 * h)  # 3x3
                np.testing.assert_allclose(got[n, :, k], dinv.reshape(9), rtol=1e-3, atol=1e-4)

    def test_mat32_triangle(self):
        """Check a rectangular mat32 triangle energy gradient against finite differences."""
        rng = np.random.default_rng(3)
        rest = np.array([0, 0, 0, 1, 0, 0, 0.5, 0.8660254, 0], np.float32)
        z = (rest + 0.1 * rng.standard_normal((5, 9))).astype(np.float32)
        grad = wp.zeros(5, dtype=J9.grad_type_scalar, device="cpu")
        wp.launch(
            jet_triangle_rect, dim=5, inputs=[wp.array(z, dtype=float, device="cpu")], outputs=[grad], device="cpu"
        )
        np.testing.assert_allclose(
            grad.numpy().reshape(5, 9), _grad_fd(triangle_rect_np, z.astype(np.float64)), rtol=1e-3, atol=1e-4
        )

    def test_mat22_matmul(self):
        """Check mat22 * mat22 and mat22 * vec2 derivatives against finite differences.

        ``mat22`` is part of the public namespace and is what a rectangular
        ``mat23 * mat32`` product returns, so ordinary 2D matrix algebra has to
        work on the result.
        """
        rng = np.random.default_rng(0)
        z = (rng.standard_normal((5, 4)) + np.array([2.0, 0.1, 0.1, 2.0])).astype(np.float32)
        out = wp.zeros(5, dtype=J4.grad_type_scalar, device="cpu")
        wp.launch(jet_mat22_matmul, dim=5, inputs=[wp.array(z, dtype=float, device="cpu")], outputs=[out], device="cpu")
        np.testing.assert_allclose(
            out.numpy().reshape(5, 4), _grad_fd(mat22_matmul_np, z.astype(np.float64)), rtol=1e-3, atol=1e-4
        )


class TestJetOps(unittest.TestCase):
    def test_first_order_smooth(self):
        """Check first-order gradients of the smooth op surface against finite differences."""
        z64 = Z_NP.astype(np.float64)
        for device in DEVICES:
            val, grad = _run_grad(jet_smooth, Z_NP, device)
            np.testing.assert_allclose(val, smooth_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(smooth_np, z64), rtol=1.0e-3, atol=1.0e-4)

    def test_first_order_branch(self):
        """Check first-order gradients of the branching builtins against finite differences."""
        z64 = ZB_NP.astype(np.float64)
        for device in DEVICES:
            val, grad = _run_grad(jet_branch, ZB_NP, device)
            np.testing.assert_allclose(val, branch_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(branch_np, z64), rtol=1.0e-4, atol=1.0e-5)

    def test_second_order_smooth(self):
        """Check second-order gradients and Hessians of the smooth ops against finite differences."""
        z64 = Z_NP.astype(np.float64)
        for device in DEVICES:
            val, grad, hess = _run_hess(jet2_g3, Z_NP, device)
            np.testing.assert_allclose(val, g3_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(g3_np, z64), rtol=1.0e-3, atol=1.0e-4)
            np.testing.assert_allclose(hess, _hess_fd(g3_np, z64), rtol=1.0e-2, atol=1.0e-3)
            np.testing.assert_allclose(hess, np.transpose(hess, (0, 2, 1)), rtol=1.0e-6, atol=1.0e-7)

    def test_second_order_branch(self):
        """Check second-order gradients and Hessians of the branching builtins against finite differences."""
        z64 = ZB_NP.astype(np.float64)
        for device in DEVICES:
            val, grad, hess = _run_hess(jet2_branch, ZB_NP, device)
            np.testing.assert_allclose(val, branch2_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad, _grad_fd(branch2_np, z64), rtol=1.0e-3, atol=1.0e-4)
            np.testing.assert_allclose(hess, _hess_fd(branch2_np, z64), rtol=1.0e-2, atol=1.0e-3)

    def test_mixed_jet_and_constant_operands(self):
        """Check that mixed jet/constant pow and atan2 agree at first and second order.

        An energy written once against numeric literals has to compile and give
        the same derivatives under both spaces, so the second-order overload set
        must match the first-order one.
        """
        z = np.array([[0.7, 0.4], [1.3, -0.9], [0.2, 1.1]], dtype=np.float32)
        z64 = z.astype(np.float64)

        for device in DEVICES:
            val1, grad1 = _run_grad(jet_mixed, z, device)
            val2, grad2, hess2 = _run_hess(jet2_mixed, z, device)

            np.testing.assert_allclose(val1, mixed_scalar_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(val2, mixed_scalar_np(z64), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(grad1, _grad_fd(mixed_scalar_np, z64), rtol=1.0e-3, atol=1.0e-4)
            np.testing.assert_allclose(grad2, grad1, rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(hess2, _hess_fd(mixed_scalar_np, z64), rtol=1.0e-2, atol=1.0e-3)

    def test_pow_with_constant_exponent_at_zero_base(self):
        """Check that the degenerate constant exponents give finite derivatives at a zero base.

        ``x**0`` is constant and ``x**1`` is linear, so their missing
        derivatives are zero, but the general ``p a**(p-1)`` and
        ``p (p-1) a**(p-2)`` terms reach them through ``0 * inf``.
        """
        for device in DEVICES:
            z = wp.array(np.zeros(1, np.float32), dtype=float, device=device)

            g1 = wp.zeros((1, 5), dtype=float, device=device)
            wp.launch(jet_pow_const_exponent, dim=1, inputs=[z], outputs=[g1], device=device)
            # p = 0, 1, 2 as floats, then 0, 1 as ints.
            np.testing.assert_allclose(g1.numpy()[0], [0.0, 1.0, 0.0, 0.0, 1.0])

            g2 = wp.zeros((1, 4), dtype=float, device=device)
            h2 = wp.zeros((1, 4), dtype=float, device=device)
            wp.launch(jet2_pow_const_exponent, dim=1, inputs=[z], outputs=[g2, h2], device=device)
            # p = 0, 1 (int), 1.0 (float), 2.
            np.testing.assert_allclose(g2.numpy()[0], [0.0, 1.0, 1.0, 0.0])
            np.testing.assert_allclose(h2.numpy()[0], [0.0, 0.0, 0.0, 2.0])

    def test_pow_with_jet_exponent_at_zero_base(self):
        """Check that a jet exponent gives finite derivatives at a zero base.

        d(a**b)/da = b a**(b-1) is 0 at a = 0 for b > 1, and d/db is 0 in the
        limit, but the a**b * (b/a) factorization evaluates as 0 * inf there.
        Finite differences cannot be the oracle at the endpoint, so the closed
        form is used directly.
        """
        # Rows 2 and 3 straddle the zero base; b > 1 keeps both partials finite.
        z = np.array([[0.0, 2.0], [0.0, 3.0], [0.5, 2.0]], dtype=np.float32)

        for device in DEVICES:
            m = z.shape[0]
            zd = wp.array(z, dtype=float, device=device)

            grad = wp.zeros(m, dtype=J2.grad_type_scalar, device=device)
            wp.launch(jet_pow_jet_exponent, dim=m, inputs=[zd], outputs=[grad], device=device)
            g1 = grad.numpy().reshape(m, 2)

            g2 = wp.zeros((m, 2), dtype=float, device=device)
            h2 = wp.zeros((m, 2, 2), dtype=float, device=device)
            wp.launch(jet2_pow_jet_exponent, dim=m, inputs=[zd], outputs=[g2, h2], device=device)

            self.assertTrue(np.isfinite(g1).all(), f"first-order gradient not finite: {g1}")
            self.assertTrue(np.isfinite(g2.numpy()).all(), f"second-order gradient not finite: {g2.numpy()}")
            self.assertTrue(np.isfinite(h2.numpy()).all(), f"second-order Hessian not finite: {h2.numpy()}")

            a, b = z[:, 0].astype(np.float64), z[:, 1].astype(np.float64)
            da = b * np.power(a, b - 1.0)
            db = np.where(a > 0.0, np.power(a, b) * np.log(np.where(a > 0.0, a, 1.0)), 0.0)

            np.testing.assert_allclose(g1, np.stack([da, db], axis=1), rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(g2.numpy(), np.stack([da, db], axis=1), rtol=1.0e-5, atol=1.0e-6)


# --------------------------------------------------------------------------
# Mixed jet/plain operands. Each overload is checked against the jet/jet
# overload with the plain operand lifted to a constant jet, which carries the
# same value and a zero derivative, so the two must agree to rounding. The
# overloads do not depend on the width, so width-2 directional jets keep the
# kernels small; the directions are arbitrary functions of the inputs.
# --------------------------------------------------------------------------

mat23 = wp.types.matrix(shape=(2, 3), dtype=float)
mat32 = wp.types.matrix(shape=(3, 2), dtype=float)

JV = wp.JetSpace(1, width=2)
JM2 = wp.JetSpace((wp.mat22, wp.vec2), width=2)
JM3 = wp.JetSpace((wp.mat33, wp.vec3), width=2)
JR = wp.JetSpace((mat23, mat32), width=2)


@wp.func
def scalar_err(a: Any, b: Any) -> float:
    dc = a.grad - b.grad
    return (a.value - b.value) * (a.value - b.value) + wp.dot(dc, dc)


@wp.func
def vec_err(a: Any, b: Any) -> float:
    dv = a.value - b.value
    return wp.dot(dv, dv) + wp.ddot(a.grad - b.grad, a.grad - b.grad)


@wp.func
def mat_err(a: Any, b: Any) -> float:
    return wp.ddot(a.value - b.value, a.value - b.value) + wp.ddot(a.grad - b.grad, a.grad - b.grad)


@wp.kernel(enable_backward=False)
def mixed_vec_kernel(
    s_in: wp.array[float],
    p2: wp.array[wp.vec2],
    p3: wp.array[wp.vec3],
    c: wp.array[float],
    err: wp.array2d[float],
):
    i = wp.tid()
    s = JV.directional(s_in[i], 1.0, 0.5 * s_in[i] - 0.25)
    l2 = JV.constant_vec2(p2[i])
    l3 = JV.constant_vec3(p3[i])
    lc = JV.constant(c[i])

    err[i, 0] = vec_err(s * p2[i], s * l2)
    err[i, 1] = vec_err(p2[i] * s, l2 * s)
    err[i, 2] = vec_err(s * p3[i], s * l3)
    err[i, 3] = vec_err(p3[i] * s, l3 * s)
    err[i, 4] = vec_err(p2[i] / s, l2 / s)
    err[i, 5] = vec_err(p3[i] / s, l3 / s)
    err[i, 6] = scalar_err(wp.min(s, c[i]), wp.min(s, lc))
    err[i, 7] = scalar_err(wp.min(c[i], s), wp.min(lc, s))
    err[i, 8] = scalar_err(wp.max(s, c[i]), wp.max(s, lc))
    err[i, 9] = scalar_err(wp.max(c[i], s), wp.max(lc, s))


@wp.kernel(enable_backward=False)
def mixed_mat22_kernel(
    a_in: wp.array[wp.mat22],
    v_in: wp.array[wp.vec2],
    pm: wp.array[wp.mat22],
    pv: wp.array[wp.vec2],
    err: wp.array2d[float],
):
    i = wp.tid()
    a, v = JM2.directional(a_in[i], v_in[i], pm[i], pv[i], wp.transpose(a_in[i]), -v_in[i])
    lm = JM2.mat22()
    lm.value = pm[i]
    lv = JM2.constant_vec2(pv[i])

    err[i, 0] = mat_err(a * pm[i], a * lm)
    err[i, 1] = mat_err(pm[i] * a, lm * a)
    err[i, 2] = vec_err(a * pv[i], a * lv)
    err[i, 3] = vec_err(pm[i] * v, lm * v)
    err[i, 4] = mat_err(a + pm[i], a + lm)
    err[i, 5] = mat_err(pm[i] + a, lm + a)
    err[i, 6] = mat_err(a - pm[i], a - lm)
    err[i, 7] = mat_err(pm[i] - a, lm - a)


@wp.kernel(enable_backward=False)
def mixed_mat33_kernel(
    a_in: wp.array[wp.mat33],
    v_in: wp.array[wp.vec3],
    pm: wp.array[wp.mat33],
    pv: wp.array[wp.vec3],
    err: wp.array2d[float],
):
    i = wp.tid()
    a, v = JM3.directional(a_in[i], v_in[i], pm[i], pv[i], wp.transpose(a_in[i]), -v_in[i])
    lm = JM3.mat33()
    lm.value = pm[i]
    lv = JM3.constant_vec3(pv[i])

    err[i, 0] = mat_err(a * pm[i], a * lm)
    err[i, 1] = mat_err(pm[i] * a, lm * a)
    err[i, 2] = vec_err(a * pv[i], a * lv)
    err[i, 3] = vec_err(pm[i] * v, lm * v)
    err[i, 4] = mat_err(a + pm[i], a + lm)
    err[i, 5] = mat_err(pm[i] + a, lm + a)
    err[i, 6] = mat_err(a - pm[i], a - lm)
    err[i, 7] = mat_err(pm[i] - a, lm - a)


@wp.kernel(enable_backward=False)
def mixed_rect_kernel(
    a_in: wp.array[mat23],
    b_in: wp.array[mat32],
    p23: wp.array[mat23],
    p32: wp.array[mat32],
    err: wp.array2d[float],
):
    i = wp.tid()
    a, b = JR.directional(a_in[i], b_in[i], p23[i], p32[i], wp.transpose(b_in[i]), wp.transpose(a_in[i]))
    l23 = JR.mat23()
    l23.value = p23[i]
    l32 = JR.mat32()
    l32.value = p32[i]

    err[i, 0] = mat_err(a * p32[i], a * l32)
    err[i, 1] = mat_err(p23[i] * b, l23 * b)


@wp.kernel
def mixed_mat33_adjoint_kernel(
    a_in: wp.array[wp.mat33],
    v_in: wp.array[wp.vec3],
    pm: wp.array[wp.mat33],
    pv: wp.array[wp.vec3],
    da: wp.array2d[wp.mat33],
    dv: wp.array2d[wp.vec3],
    w: wp.array[wp.vec3],
    weights: wp.array[JM3.grad_type_vec3],
    out: wp.array[float],
):
    i = wp.tid()
    a, v = JM3.directional(a_in[i], v_in[i], da[i, 0], dv[i, 0], da[i, 1], dv[i, 1])
    y = a * pv[i] + pm[i] * v
    out[i] = wp.dot(y.value, w[i]) + wp.ddot(y.grad, weights[i])


def _random(rng, shape):
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


class TestJetMixedOperands(unittest.TestCase):
    def _assert_agree(self, err):
        # Squared differences of values and gradients against the lifted form.
        np.testing.assert_allclose(err, 0.0, atol=1.0e-10)

    def test_scalar_vector_and_selection(self):
        """Check jet-scalar times, and plain vector over, jet-scalar against lifted constants."""
        rng = np.random.default_rng(1)
        m = 6
        s = rng.uniform(0.5, 1.5, size=m).astype(np.float32) * np.where(np.arange(m) % 2, 1.0, -1.0).astype(np.float32)
        # Thresholds on both sides of s so min and max take each branch.
        c = (s + np.where(np.arange(m) % 3 == 0, 0.25, -0.25)).astype(np.float32)
        for device in DEVICES:
            err = wp.zeros((m, 10), dtype=float, device=device)
            wp.launch(
                mixed_vec_kernel,
                dim=m,
                inputs=[
                    wp.array(s, dtype=float, device=device),
                    wp.array(_random(rng, (m, 2)), dtype=wp.vec2, device=device),
                    wp.array(_random(rng, (m, 3)), dtype=wp.vec3, device=device),
                    wp.array(c, dtype=float, device=device),
                ],
                outputs=[err],
                device=device,
            )
            self._assert_agree(err.numpy())

    def test_square_matrix(self):
        """Check mat22 and mat33 products and sums with one plain operand against lifted constants."""
        rng = np.random.default_rng(2)
        m = 4
        for device in DEVICES:
            for kernel, mat, vec, n in (
                (mixed_mat22_kernel, wp.mat22, wp.vec2, 2),
                (mixed_mat33_kernel, wp.mat33, wp.vec3, 3),
            ):
                err = wp.zeros((m, 8), dtype=float, device=device)
                wp.launch(
                    kernel,
                    dim=m,
                    inputs=[
                        wp.array(_random(rng, (m, n, n)), dtype=mat, device=device),
                        wp.array(_random(rng, (m, n)), dtype=vec, device=device),
                        wp.array(_random(rng, (m, n, n)), dtype=mat, device=device),
                        wp.array(_random(rng, (m, n)), dtype=vec, device=device),
                    ],
                    outputs=[err],
                    device=device,
                )
                self._assert_agree(err.numpy())

    def test_rectangular_matrix(self):
        """Check mat23 times mat32 with either operand plain against lifted constants."""
        rng = np.random.default_rng(3)
        m = 4
        for device in DEVICES:
            err = wp.zeros((m, 2), dtype=float, device=device)
            wp.launch(
                mixed_rect_kernel,
                dim=m,
                inputs=[
                    wp.array(_random(rng, (m, 2, 3)), dtype=mat23, device=device),
                    wp.array(_random(rng, (m, 3, 2)), dtype=mat32, device=device),
                    wp.array(_random(rng, (m, 2, 3)), dtype=mat23, device=device),
                    wp.array(_random(rng, (m, 3, 2)), dtype=mat32, device=device),
                ],
                outputs=[err],
                device=device,
            )
            self._assert_agree(err.numpy())

    def test_plain_operand_adjoints(self):
        """Check that reverse mode through mixed products reaches the plain operands."""
        rng = np.random.default_rng(4)
        m = 3
        a_np = _random(rng, (m, 3, 3))
        v_np = _random(rng, (m, 3))
        pm_np = _random(rng, (m, 3, 3))
        pv_np = _random(rng, (m, 3))
        da_np = _random(rng, (m, 2, 3, 3))
        dv_np = _random(rng, (m, 2, 3))
        w_np = _random(rng, (m, 3))
        weights_np = _random(rng, (m, 3, 2))

        # y = A pv + Pm v carries y.grad[:, q] = dA_q pv + Pm dv_q along direction q.
        pv_grad = np.einsum("mrk,mr->mk", a_np, w_np) + np.einsum("mqrk,mrq->mk", da_np, weights_np)
        pm_grad = np.einsum("mr,mk->mrk", w_np, v_np) + np.einsum("mrq,mqk->mrk", weights_np, dv_np)

        for device in DEVICES:
            pm = wp.array(pm_np, dtype=wp.mat33, requires_grad=True, device=device)
            pv = wp.array(pv_np, dtype=wp.vec3, requires_grad=True, device=device)
            out = wp.zeros(m, dtype=float, requires_grad=True, device=device)
            tape = wp.Tape()
            with tape:
                wp.launch(
                    mixed_mat33_adjoint_kernel,
                    dim=m,
                    inputs=[
                        wp.array(a_np, dtype=wp.mat33, device=device),
                        wp.array(v_np, dtype=wp.vec3, device=device),
                        pm,
                        pv,
                        wp.array(da_np, dtype=wp.mat33, device=device),
                        wp.array(dv_np, dtype=wp.vec3, device=device),
                        wp.array(w_np, dtype=wp.vec3, device=device),
                        wp.array(weights_np, dtype=JM3.grad_type_vec3, device=device),
                    ],
                    outputs=[out],
                    device=device,
                )
            tape.backward(grads={out: wp.ones(m, dtype=float, device=device)})

            np.testing.assert_allclose(pv.grad.numpy(), pv_grad, rtol=1.0e-5, atol=1.0e-6)
            np.testing.assert_allclose(pm.grad.numpy(), pm_grad, rtol=1.0e-5, atol=1.0e-6)


# --------------------------------------------------------------------------
# Component-wise products, checked against the same product assembled from
# per-component scalar operations, for first- and second-order vector jets
# with either operand plain.
# --------------------------------------------------------------------------

JC2 = wp.JetSpace((wp.vec2, wp.vec2), width=2)
JC3 = wp.JetSpace((wp.vec3, wp.vec3), width=2)
J2C2 = wp.JetSpace2((wp.vec2, wp.vec2))
J2C3 = wp.JetSpace2((wp.vec3, wp.vec3))


@wp.func
def cw_mul_by_components(a: Any, b: Any, e0: Any, e1: Any):
    return a[0] * b[0] * e0 + a[1] * b[1] * e1


@wp.func
def cw_div_by_components(a: Any, b: Any, e0: Any, e1: Any):
    return a[0] / b[0] * e0 + a[1] / b[1] * e1


@wp.func
def cw_mul_by_components(a: Any, b: Any, e0: Any, e1: Any, e2: Any):
    return a[0] * b[0] * e0 + a[1] * b[1] * e1 + a[2] * b[2] * e2


@wp.func
def cw_div_by_components(a: Any, b: Any, e0: Any, e1: Any, e2: Any):
    return a[0] / b[0] * e0 + a[1] / b[1] * e1 + a[2] / b[2] * e2


@wp.func
def jet2_scalar_err(a: Any, b: Any) -> float:
    dg = a.grad - b.grad
    dh = a.hess - b.hess
    return (a.value - b.value) * (a.value - b.value) + wp.dot(dg, dg) + wp.ddot(dh, dh)


@wp.func
def jet2_vec2_err(a: Any, b: Any) -> float:
    return jet2_scalar_err(a.x, b.x) + jet2_scalar_err(a.y, b.y)


@wp.func
def jet2_vec3_err(a: Any, b: Any) -> float:
    return jet2_scalar_err(a.x, b.x) + jet2_scalar_err(a.y, b.y) + jet2_scalar_err(a.z, b.z)


@wp.kernel(enable_backward=False)
def cw_first_order_kernel(
    a2_in: wp.array[wp.vec2],
    b2_in: wp.array[wp.vec2],
    a3_in: wp.array[wp.vec3],
    b3_in: wp.array[wp.vec3],
    err: wp.array2d[float],
):
    i = wp.tid()
    pa2 = a2_in[i]
    pb2 = b2_in[i]
    pa3 = a3_in[i]
    pb3 = b3_in[i]
    a2, b2 = JC2.directional(pa2, pb2, pb2, -pa2, wp.vec2(1.0, -0.5), pa2)
    a3, b3 = JC3.directional(pa3, pb3, pb3, -pa3, wp.vec3(1.0, -0.5, 0.25), pa3)
    x2 = wp.vec2(1.0, 0.0)
    y2 = wp.vec2(0.0, 1.0)
    x3 = wp.vec3(1.0, 0.0, 0.0)
    y3 = wp.vec3(0.0, 1.0, 0.0)
    z3 = wp.vec3(0.0, 0.0, 1.0)

    err[i, 0] = vec_err(wp.cw_mul(a2, b2), cw_mul_by_components(a2, b2, x2, y2))
    err[i, 1] = vec_err(wp.cw_mul(a2, pb2), cw_mul_by_components(a2, pb2, x2, y2))
    err[i, 2] = vec_err(wp.cw_mul(pa2, b2), cw_mul_by_components(pa2, b2, x2, y2))
    err[i, 3] = vec_err(wp.cw_div(a2, b2), cw_div_by_components(a2, b2, x2, y2))
    err[i, 4] = vec_err(wp.cw_div(a2, pb2), cw_div_by_components(a2, pb2, x2, y2))
    err[i, 5] = vec_err(wp.cw_div(pa2, b2), cw_div_by_components(pa2, b2, x2, y2))
    err[i, 6] = vec_err(wp.cw_mul(a3, b3), cw_mul_by_components(a3, b3, x3, y3, z3))
    err[i, 7] = vec_err(wp.cw_mul(a3, pb3), cw_mul_by_components(a3, pb3, x3, y3, z3))
    err[i, 8] = vec_err(wp.cw_mul(pa3, b3), cw_mul_by_components(pa3, b3, x3, y3, z3))
    err[i, 9] = vec_err(wp.cw_div(a3, b3), cw_div_by_components(a3, b3, x3, y3, z3))
    err[i, 10] = vec_err(wp.cw_div(a3, pb3), cw_div_by_components(a3, pb3, x3, y3, z3))
    err[i, 11] = vec_err(wp.cw_div(pa3, b3), cw_div_by_components(pa3, b3, x3, y3, z3))


@wp.kernel(enable_backward=False)
def cw_second_order_kernel(
    a2_in: wp.array[wp.vec2],
    b2_in: wp.array[wp.vec2],
    a3_in: wp.array[wp.vec3],
    b3_in: wp.array[wp.vec3],
    err: wp.array2d[float],
):
    i = wp.tid()
    pa2 = a2_in[i]
    pb2 = b2_in[i]
    pa3 = a3_in[i]
    pb3 = b3_in[i]
    # Square the seeded jets so the operands carry nonzero Hessians.
    s2, t2 = J2C2.seed(pa2, pb2)
    s3, t3 = J2C3.seed(pa3, pb3)
    a2 = wp.cw_mul(s2, s2) + t2
    b2 = wp.cw_mul(t2, t2) - s2 + wp.vec2(2.0)
    a3 = wp.cw_mul(s3, s3) + t3
    b3 = wp.cw_mul(t3, t3) - s3 + wp.vec3(2.0)
    x2 = wp.vec2(1.0, 0.0)
    y2 = wp.vec2(0.0, 1.0)
    x3 = wp.vec3(1.0, 0.0, 0.0)
    y3 = wp.vec3(0.0, 1.0, 0.0)
    z3 = wp.vec3(0.0, 0.0, 1.0)

    err[i, 0] = jet2_vec2_err(wp.cw_mul(a2, b2), cw_mul_by_components(a2, b2, x2, y2))
    err[i, 1] = jet2_vec2_err(wp.cw_mul(a2, pb2), cw_mul_by_components(a2, pb2, x2, y2))
    err[i, 2] = jet2_vec2_err(wp.cw_mul(pa2, b2), cw_mul_by_components(pa2, b2, x2, y2))
    err[i, 3] = jet2_vec2_err(wp.cw_div(a2, b2), cw_div_by_components(a2, b2, x2, y2))
    err[i, 4] = jet2_vec2_err(wp.cw_div(a2, pb2), cw_div_by_components(a2, pb2, x2, y2))
    err[i, 5] = jet2_vec2_err(wp.cw_div(pa2, b2), cw_div_by_components(pa2, b2, x2, y2))
    err[i, 6] = jet2_vec3_err(wp.cw_mul(a3, b3), cw_mul_by_components(a3, b3, x3, y3, z3))
    err[i, 7] = jet2_vec3_err(wp.cw_mul(a3, pb3), cw_mul_by_components(a3, pb3, x3, y3, z3))
    err[i, 8] = jet2_vec3_err(wp.cw_mul(pa3, b3), cw_mul_by_components(pa3, b3, x3, y3, z3))
    err[i, 9] = jet2_vec3_err(wp.cw_div(a3, b3), cw_div_by_components(a3, b3, x3, y3, z3))
    err[i, 10] = jet2_vec3_err(wp.cw_div(a3, pb3), cw_div_by_components(a3, pb3, x3, y3, z3))
    err[i, 11] = jet2_vec3_err(wp.cw_div(pa3, b3), cw_div_by_components(pa3, b3, x3, y3, z3))


class TestJetComponentWise(unittest.TestCase):
    def _launch(self, kernel):
        rng = np.random.default_rng(5)
        m = 5

        def operand(n):
            # Magnitudes in [0.5, 1.5] with random signs keep the divisors away from zero.
            return (rng.uniform(0.5, 1.5, size=(m, n)) * rng.choice([-1.0, 1.0], size=(m, n))).astype(np.float32)

        for device in DEVICES:
            err = wp.zeros((m, 12), dtype=float, device=device)
            wp.launch(
                kernel,
                dim=m,
                inputs=[
                    wp.array(operand(2), dtype=wp.vec2, device=device),
                    wp.array(operand(2), dtype=wp.vec2, device=device),
                    wp.array(operand(3), dtype=wp.vec3, device=device),
                    wp.array(operand(3), dtype=wp.vec3, device=device),
                ],
                outputs=[err],
                device=device,
            )
            np.testing.assert_allclose(err.numpy(), 0.0, atol=1.0e-9)

    def test_first_order(self):
        """Check first-order cw_mul and cw_div against per-component products, with either operand plain."""
        self._launch(cw_first_order_kernel)

    def test_second_order(self):
        """Check second-order cw_mul and cw_div against per-component products, with either operand plain."""
        self._launch(cw_second_order_kernel)


# --------------------------------------------------------------------------
# wp.value: one generic function that branches on plain floats and on jets.
# --------------------------------------------------------------------------


@wp.func
def generic_relu(x: Any):
    if wp.value(x) > 0:
        return x
    return x - x


@wp.kernel(enable_backward=False)
def generic_relu_kernel(
    x: wp.array[float],
    out32: wp.array[float],
    out64: wp.array[wp.float64],
    grad_1st: wp.array[J2.grad_type_scalar],
    grad_2nd: wp.array[J2_2ND.grad_type_scalar],
    hess: wp.array[J2_2ND.hess_type_scalar],
):
    i = wp.tid()
    out32[i] = generic_relu(x[i])
    out64[i] = generic_relu(wp.float64(x[i]))
    a = J2.variable(x[i], 0) * J2.variable(1.0, 1)
    grad_1st[i] = generic_relu(a).grad
    b = J2_2ND.variable(x[i], 0) * J2_2ND.variable(1.0, 1)
    r = generic_relu(b)
    grad_2nd[i] = r.grad
    hess[i] = r.hess


class TestJetValue(unittest.TestCase):
    def test_generic_branch(self):
        """Check that wp.value lets one generic function branch on floats and on first- and second-order jets."""
        x_np = np.array([0.7, -0.3], dtype=np.float32)
        on = (x_np > 0).astype(np.float32)
        for device in DEVICES:
            out32 = wp.zeros(2, dtype=float, device=device)
            out64 = wp.zeros(2, dtype=wp.float64, device=device)
            grad_1st = wp.zeros(2, dtype=J2.grad_type_scalar, device=device)
            grad_2nd = wp.zeros(2, dtype=J2_2ND.grad_type_scalar, device=device)
            hess = wp.zeros(2, dtype=J2_2ND.hess_type_scalar, device=device)
            wp.launch(
                generic_relu_kernel,
                dim=2,
                inputs=[wp.array(x_np, dtype=float, device=device)],
                outputs=[out32, out64, grad_1st, grad_2nd, hess],
                device=device,
            )
            np.testing.assert_allclose(out32.numpy(), x_np * on)
            np.testing.assert_allclose(out64.numpy(), x_np * on, rtol=1.0e-6)
            # x0 * x1 at (x, 1) has value x, gradient (1, x) and Hessian [[0, 1], [1, 0]].
            np.testing.assert_allclose(grad_1st.numpy(), np.stack([on, x_np * on], axis=1), rtol=1.0e-6)
            np.testing.assert_allclose(grad_2nd.numpy(), np.stack([on, x_np * on], axis=1), rtol=1.0e-6)
            np.testing.assert_allclose(hess.numpy(), on[:, None, None] * np.array([[0.0, 1.0], [1.0, 0.0]]))


# --------------------------------------------------------------------------
# Builtins needed by FEM-style energies: matrix entries and rows, scalar-jet
# and matrix-jet products, ddot, outer, row-vector products, diagonals, skew,
# component assignment, and more scalar functions. Each probe is one generic
# function evaluated on jets and, for central differences, on plain values.
# --------------------------------------------------------------------------

JF = wp.JetSpace((float, wp.vec3, wp.mat33))  # width 13
JR2 = wp.JetSpace((mat23, wp.vec2, wp.vec3))  # width 11
J2F = wp.JetSpace2((float, wp.vec3))  # width 4
J1F = wp.JetSpace((float, wp.vec3))  # width 4, for finite differences of the gradient


@wp.func
def square_matrix_energy(s: Any, v: Any, F: Any):
    W = wp.mat33(0.3, -0.2, 0.5, 0.1, 0.7, -0.4, -0.6, 0.2, 0.9)
    G = F
    G[0, 1] = s * s
    w = v
    w[2] = s + v[0]
    e = F[0, 1] * F[2, 0] + wp.dot(F[1], v)
    e += wp.ddot(s * F, W) + wp.ddot(F * s, W) + wp.ddot(-F, W) + wp.ddot(F / 2.0, W) + wp.ddot(F / s, W)
    e += wp.ddot(F, F) + wp.ddot(wp.outer(v, w), W) + wp.dot(v * F, w)
    e += wp.ddot(wp.cw_mul(F, F), W) + wp.ddot(wp.cw_div(F, W + wp.identity(n=3, dtype=float) * 2.0), W)
    e += wp.dot(wp.get_diag(F), v) + wp.ddot(wp.diag(v), W) + wp.ddot(wp.skew(w), W)
    e += wp.ddot(G, W) + wp.dot(w, wp.vec3(1.0, 2.0, 3.0)) + wp.ddot(s * W, F) + wp.ddot(W / s, F)
    e += wp.min(v) + wp.max(w) + wp.dot(wp.abs(w), wp.vec3(0.5, -1.0, 2.0))
    return e


@wp.func
def rectangular_matrix_energy(A: Any, u: Any, w: Any):
    W23 = mat23(0.3, -0.2, 0.5, 0.1, 0.7, -0.4)
    W33 = wp.mat33(0.2, 0.1, -0.3, 0.4, -0.5, 0.6, 0.7, 0.8, -0.9)
    G = A
    G[1, 2] = u[0] * u[1]
    G[0] = w
    M = A * wp.transpose(A)  # mat22
    e = wp.ddot(wp.transpose(A) * A, W33) + wp.dot(A[1], w) + A[0, 2] * A[1, 0]
    e += wp.ddot(wp.outer(u, w), A) + wp.dot(u * A, w) + wp.ddot(wp.outer(w, u), wp.transpose(W23))
    e += wp.ddot(u[0] * A, W23) + wp.ddot(A * u[1], W23) + wp.ddot(-A, W23) + wp.ddot(A / u[0], W23)
    e += wp.ddot(A - wp.outer(u, w), A + wp.outer(u, w)) + wp.ddot(wp.cw_mul(A, A), W23) + wp.ddot(G, W23)
    e += wp.dot(wp.get_diag(M), u) + wp.ddot(wp.diag(u), M) + M[0, 1]
    return e


@wp.func
def scalar_functions_energy(s: Any, v: Any):
    w = v
    w[1] = s * v[2]
    e = wp.tanh(s) + wp.sinh(s) + wp.cosh(s) + wp.log2(s) + wp.log10(s) + wp.cbrt(s) + wp.erf(s) + wp.floor(s)
    e += wp.lerp(s, v[0], v[1]) + wp.lerp(s, v[0], 0.3) + wp.lerp(0.5, 2.0, v[2] * s)
    e += wp.smoothstep(0.0, 4.0, s) + wp.smoothstep(v[0], 4.0 + v[1], s)
    e += wp.min(w) * wp.max(w) + wp.dot(wp.abs(w), wp.vec3(0.5, -1.0, 2.0))
    return e


@wp.kernel(enable_backward=False)
def square_matrix_energy_kernel(
    x_s: float, x_v: wp.vec3, x_F: wp.mat33, jet_grad: wp.array[JF.grad_type_scalar], fd_grad: wp.array[float]
):
    s, v, F = JF.seed(x_s, x_v, x_F)
    jet_grad[0] = square_matrix_energy(s, v, F).grad
    h = 1.0e-3
    for k in range(13):
        ds = float(0.0)
        dv = wp.vec3(0.0)
        dF = wp.mat33(0.0)
        if k == 0:
            ds = 1.0
        elif k < 4:
            dv[k - 1] = 1.0
        else:
            dF[(k - 4) // 3, (k - 4) % 3] = 1.0
        fp = square_matrix_energy(x_s + h * ds, x_v + h * dv, x_F + h * dF)
        fm = square_matrix_energy(x_s - h * ds, x_v - h * dv, x_F - h * dF)
        fd_grad[k] = (fp - fm) / (2.0 * h)


@wp.kernel(enable_backward=False)
def rectangular_matrix_energy_kernel(
    x_A: mat23, x_u: wp.vec2, x_w: wp.vec3, jet_grad: wp.array[JR2.grad_type_scalar], fd_grad: wp.array[float]
):
    A, u, w = JR2.seed(x_A, x_u, x_w)
    jet_grad[0] = rectangular_matrix_energy(A, u, w).grad
    h = 1.0e-3
    for k in range(11):
        dA = mat23(0.0)
        du = wp.vec2(0.0)
        dw = wp.vec3(0.0)
        if k < 6:
            dA[k // 3, k % 3] = 1.0
        elif k < 8:
            du[k - 6] = 1.0
        else:
            dw[k - 8] = 1.0
        fp = rectangular_matrix_energy(x_A + h * dA, x_u + h * du, x_w + h * dw)
        fm = rectangular_matrix_energy(x_A - h * dA, x_u - h * du, x_w - h * dw)
        fd_grad[k] = (fp - fm) / (2.0 * h)


@wp.kernel(enable_backward=False)
def scalar_functions_energy_kernel(
    x_s: float,
    x_v: wp.vec3,
    grad: wp.array[J2F.grad_type_scalar],
    hess: wp.array[J2F.hess_type_scalar],
    fd_grad: wp.array[float],
    fd_hess: wp.array2d[float],
):
    s2, v2 = J2F.seed(x_s, x_v)
    e2 = scalar_functions_energy(s2, v2)
    grad[0] = e2.grad
    hess[0] = e2.hess
    # The gradient by central differences of the value, and the Hessian by
    # central differences of the first-order jet's gradient.
    h = 1.0e-3
    for k in range(4):
        ds = float(0.0)
        dv = wp.vec3(0.0)
        if k == 0:
            ds = 1.0
        else:
            dv[k - 1] = 1.0
        fp = scalar_functions_energy(x_s + h * ds, x_v + h * dv)
        fm = scalar_functions_energy(x_s - h * ds, x_v - h * dv)
        fd_grad[k] = (fp - fm) / (2.0 * h)
        sp, vp = J1F.seed(x_s + h * ds, x_v + h * dv)
        sm, vm = J1F.seed(x_s - h * ds, x_v - h * dv)
        dg = (scalar_functions_energy(sp, vp).grad - scalar_functions_energy(sm, vm).grad) / (2.0 * h)
        for j in range(4):
            fd_hess[k, j] = dg[j]


@wp.struct
class PlainPair:
    a: float
    b: float


@wp.kernel(enable_backward=False)
def jet_norm_huber_kernel(x: wp.array[wp.vec3], value: wp.array[float], grad: wp.array[J1F.grad_type_scalar]):
    i = wp.tid()
    s, v = J1F.seed(1.0, x[i])
    h = wp.norm_huber(s * v)
    value[i] = h.value
    grad[i] = h.grad


JA = wp.JetSpace((wp.vec3, wp.mat33), width=1)


@wp.func
def assigned_directional_derivative(x: wp.vec3, F: wp.mat33, dx: wp.vec3, dF: wp.mat33):
    v, A = JA.directional(x, F, dx, dF)
    w = v
    w[1] = v[0] * v[2]
    G = A
    G[2, 0] = w[1] * A[0, 0]
    G[1] = w
    # A is used after G = A; G[...] = ... so that writing G must leave A alone.
    return (wp.ddot(G, G) + wp.dot(w, w) + wp.ddot(A, A)).grad[0]


@wp.kernel
def assigned_directional_derivative_kernel(
    x: wp.array[wp.vec3], F: wp.array[wp.mat33], dx: wp.vec3, dF: wp.mat33, out: wp.array[float]
):
    out[0] = assigned_directional_derivative(x[0], F[0], dx, dF)


class TestJetFemBuiltins(unittest.TestCase):
    def test_square_matrix_ops(self):
        """Check mat33 entry, row, product, ddot, outer, diagonal and assignment overloads against central differences."""
        for device in DEVICES:
            jet_grad = wp.zeros(1, dtype=JF.grad_type_scalar, device=device)
            fd_grad = wp.zeros(13, dtype=float, device=device)
            x_F = wp.mat33(1.2, -0.7, 0.4, 0.3, 0.9, -1.1, 0.8, 0.5, 1.5)
            wp.launch(
                square_matrix_energy_kernel,
                dim=1,
                inputs=[1.7, wp.vec3(0.6, -0.8, 1.3), x_F],
                outputs=[jet_grad, fd_grad],
                device=device,
            )
            np.testing.assert_allclose(jet_grad.numpy()[0], fd_grad.numpy(), rtol=2.0e-3, atol=2.0e-2)

    def test_rectangular_matrix_ops(self):
        """Check mat23, mat32 and mat22 overloads, including mat32 times mat23, against central differences."""
        for device in DEVICES:
            jet_grad = wp.zeros(1, dtype=JR2.grad_type_scalar, device=device)
            fd_grad = wp.zeros(11, dtype=float, device=device)
            wp.launch(
                rectangular_matrix_energy_kernel,
                dim=1,
                inputs=[mat23(1.2, -0.7, 0.4, 0.3, 0.9, -1.1), wp.vec2(0.6, -0.8), wp.vec3(1.3, 0.2, -0.5)],
                outputs=[jet_grad, fd_grad],
                device=device,
            )
            np.testing.assert_allclose(jet_grad.numpy()[0], fd_grad.numpy(), rtol=2.0e-3, atol=2.0e-2)

    def test_scalar_functions_second_order(self):
        """Check the hyperbolic, log, cbrt, erf, floor, lerp and smoothstep overloads and vector reductions to second order."""
        for device in DEVICES:
            grad = wp.zeros(1, dtype=J2F.grad_type_scalar, device=device)
            hess = wp.zeros(1, dtype=J2F.hess_type_scalar, device=device)
            fd_grad = wp.zeros(4, dtype=float, device=device)
            fd_hess = wp.zeros((4, 4), dtype=float, device=device)
            wp.launch(
                scalar_functions_energy_kernel,
                dim=1,
                inputs=[1.7, wp.vec3(0.6, -0.8, 1.3)],
                outputs=[grad, hess, fd_grad, fd_hess],
                device=device,
            )
            np.testing.assert_allclose(grad.numpy()[0], fd_grad.numpy(), rtol=2.0e-3, atol=2.0e-2)
            np.testing.assert_allclose(hess.numpy()[0], fd_hess.numpy(), rtol=2.0e-3, atol=2.0e-2)

    def test_subscript_assignment_adjoint(self):
        """Check reverse mode through jet subscript assignment against central differences."""
        rng = np.random.default_rng(6)
        x_np = rng.uniform(-1.0, 1.0, (1, 3)).astype(np.float32)
        F_np = rng.uniform(-1.0, 1.0, (1, 3, 3)).astype(np.float32)
        dx = wp.vec3(0.3, -0.5, 0.2)
        dF = wp.mat33(*rng.uniform(-1.0, 1.0, 9).astype(np.float32))
        for device in DEVICES:

            def evaluate(z, device=device):
                out = wp.zeros(1, dtype=float, device=device)
                wp.launch(
                    assigned_directional_derivative_kernel,
                    dim=1,
                    inputs=[
                        wp.array(z[:3].reshape(1, 3), dtype=wp.vec3, device=device),
                        wp.array(z[3:].reshape(1, 3, 3), dtype=wp.mat33, device=device),
                        dx,
                        dF,
                    ],
                    outputs=[out],
                    device=device,
                )
                return float(out.numpy()[0])

            x = wp.array(x_np, dtype=wp.vec3, requires_grad=True, device=device)
            F = wp.array(F_np, dtype=wp.mat33, requires_grad=True, device=device)
            out = wp.zeros(1, dtype=float, requires_grad=True, device=device)
            tape = wp.Tape()
            with tape:
                wp.launch(
                    assigned_directional_derivative_kernel, dim=1, inputs=[x, F, dx, dF], outputs=[out], device=device
                )
            tape.backward(grads={out: wp.ones(1, dtype=float, device=device)})
            adjoint = np.concatenate([x.grad.numpy().ravel(), F.grad.numpy().ravel()])

            z = np.concatenate([x_np.ravel(), F_np.ravel()])
            h = 1.0e-3
            fd = np.array(
                [(evaluate(z + h * e) - evaluate(z - h * e)) / (2.0 * h) for e in np.eye(12, dtype=np.float32)]
            )
            np.testing.assert_allclose(adjoint, fd, rtol=1.0e-3, atol=1.0e-3)

    def test_subscript_assign_without_overload(self):
        """Check that subscript assignment on a struct without an assign_copy overload raises a codegen error."""

        # A module of its own, so that the failing build leaves the other kernels alone.
        @wp.kernel(enable_backward=False, module="unique")
        def struct_subscript_assign_kernel(out: wp.array[float]):
            p = PlainPair()
            p[0] = 1.0
            out[0] = p.a

        with self.assertRaisesRegex(wp._src.codegen.WarpCodegenError, "assign_copy"):
            wp.launch(
                struct_subscript_assign_kernel, dim=1, outputs=[wp.zeros(1, dtype=float, device="cpu")], device="cpu"
            )

    def test_norm_huber(self):
        """Check that wp.norm_huber differentiates on jets on both sides of its threshold."""
        x_np = np.array([[0.3, 0.4, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
        for device in DEVICES:
            value = wp.zeros(2, dtype=float, device=device)
            grad = wp.zeros(2, dtype=J1F.grad_type_scalar, device=device)
            wp.launch(
                jet_norm_huber_kernel,
                dim=2,
                inputs=[wp.array(x_np, dtype=wp.vec3, device=device)],
                outputs=[value, grad],
                device=device,
            )
            # Quadratic 0.5 |x|^2 inside delta = 1, linear |x| - 0.5 outside; s scales x.
            norms = np.linalg.norm(x_np, axis=1)
            np.testing.assert_allclose(value.numpy(), [0.5 * norms[0] ** 2, norms[1] - 0.5], rtol=1.0e-6)
            ds = [norms[0] ** 2, norms[1]]
            dx = [x_np[0], x_np[1] / norms[1]]
            expected = np.stack([np.concatenate([[ds[i]], dx[i]]) for i in range(2)])
            np.testing.assert_allclose(grad.numpy(), expected, rtol=1.0e-5, atol=1.0e-6)


# --------------------------------------------------------------------------
# Decompositions: the jet's directional derivative of a weighted sum of the
# factors must equal reverse mode's <A_bar, dA> (the same analytic rules), and
# central differences at a matrix with well-separated singular values.
# --------------------------------------------------------------------------

JD3 = wp.JetSpace((wp.mat33,), width=1)
JD2 = wp.JetSpace((wp.mat22,), width=1)


@wp.func
def svd3_objective(A: Any, Ubar: wp.mat33, sbar: wp.vec3, Vbar: wp.mat33):
    U, s, V = wp.svd3(A)
    return wp.ddot(Ubar, U) + wp.dot(sbar, s) + wp.ddot(Vbar, V)


@wp.func
def svd2_objective(A: Any, Ubar: wp.mat22, sbar: wp.vec2, Vbar: wp.mat22):
    U, s, V = wp.svd2(A)
    return wp.ddot(Ubar, U) + wp.dot(sbar, s) + wp.ddot(Vbar, V)


@wp.func
def eig3_objective(A: Any, Qbar: wp.mat33, dbar: wp.vec3, unused: wp.mat33):
    Q, d = wp.eig3(A)
    return wp.ddot(Qbar, Q) + wp.dot(dbar, d)


@wp.kernel(enable_backward=False)
def svd3_jet_kernel(A: wp.mat33, dA: wp.mat33, Ubar: wp.mat33, sbar: wp.vec3, Vbar: wp.mat33, out: wp.array[float]):
    out[0] = svd3_objective(JD3.directional(A, dA), Ubar, sbar, Vbar).grad[0]


@wp.kernel
def svd3_plain_kernel(A: wp.array[wp.mat33], Ubar: wp.mat33, sbar: wp.vec3, Vbar: wp.mat33, out: wp.array[float]):
    out[0] = svd3_objective(A[0], Ubar, sbar, Vbar)


@wp.kernel(enable_backward=False)
def svd2_jet_kernel(A: wp.mat22, dA: wp.mat22, Ubar: wp.mat22, sbar: wp.vec2, Vbar: wp.mat22, out: wp.array[float]):
    out[0] = svd2_objective(JD2.directional(A, dA), Ubar, sbar, Vbar).grad[0]


@wp.kernel
def svd2_plain_kernel(A: wp.array[wp.mat22], Ubar: wp.mat22, sbar: wp.vec2, Vbar: wp.mat22, out: wp.array[float]):
    out[0] = svd2_objective(A[0], Ubar, sbar, Vbar)


@wp.kernel(enable_backward=False)
def eig3_jet_kernel(A: wp.mat33, dA: wp.mat33, Qbar: wp.mat33, dbar: wp.vec3, unused: wp.mat33, out: wp.array[float]):
    out[0] = eig3_objective(JD3.directional(A, dA), Qbar, dbar, unused).grad[0]


@wp.kernel
def eig3_plain_kernel(A: wp.array[wp.mat33], Qbar: wp.mat33, dbar: wp.vec3, unused: wp.mat33, out: wp.array[float]):
    out[0] = eig3_objective(A[0], Qbar, dbar, unused)


class TestJetDecompositions(unittest.TestCase):
    def _check(self, jet_kernel, plain_kernel, mat, vec, A_np, dA_np, seed):
        rng = np.random.default_rng(seed)
        n = A_np.shape[0]
        weights = [
            mat(*rng.uniform(-1.0, 1.0, n * n).astype(np.float32)),
            vec(*rng.uniform(-1.0, 1.0, n).astype(np.float32)),
            mat(*rng.uniform(-1.0, 1.0, n * n).astype(np.float32)),
        ]
        for device in DEVICES:
            out = wp.zeros(1, dtype=float, device=device)
            wp.launch(
                jet_kernel,
                dim=1,
                inputs=[mat(*A_np.ravel()), mat(*dA_np.ravel()), *weights],
                outputs=[out],
                device=device,
            )
            jet = out.numpy()[0]

            A = wp.array(A_np[None], dtype=mat, requires_grad=True, device=device)
            value = wp.zeros(1, dtype=float, requires_grad=True, device=device)
            tape = wp.Tape()
            with tape:
                wp.launch(plain_kernel, dim=1, inputs=[A, *weights], outputs=[value], device=device)
            tape.backward(grads={value: wp.ones(1, dtype=float, device=device)})
            reverse = float((A.grad.numpy()[0] * dA_np).sum())

            def evaluate(M, device=device):
                result = wp.zeros(1, dtype=float, device=device)
                wp.launch(
                    plain_kernel,
                    dim=1,
                    inputs=[wp.array(M[None].astype(np.float32), dtype=mat, device=device), *weights],
                    outputs=[result],
                    device=device,
                )
                return float(result.numpy()[0])

            h = 1.0e-3
            fd = (evaluate(A_np + h * dA_np) - evaluate(A_np - h * dA_np)) / (2.0 * h)

            np.testing.assert_allclose(jet, reverse, rtol=1.0e-4, atol=1.0e-5)
            np.testing.assert_allclose(jet, fd, rtol=5.0e-3, atol=5.0e-3)

    def test_svd3(self):
        """Check the svd3 jet against reverse mode and central differences."""
        rng = np.random.default_rng(7)
        A = rng.uniform(-1.0, 1.0, (3, 3)).astype(np.float32)
        dA = rng.uniform(-1.0, 1.0, (3, 3)).astype(np.float32)
        self._check(svd3_jet_kernel, svd3_plain_kernel, wp.mat33, wp.vec3, A, dA, seed=8)

    def test_svd2(self):
        """Check the svd2 jet against reverse mode and central differences."""
        rng = np.random.default_rng(9)
        A = rng.uniform(-1.0, 1.0, (2, 2)).astype(np.float32)
        dA = rng.uniform(-1.0, 1.0, (2, 2)).astype(np.float32)
        self._check(svd2_jet_kernel, svd2_plain_kernel, wp.mat22, wp.vec2, A, dA, seed=10)

    def test_eig3(self):
        """Check the eig3 jet of a symmetric matrix against reverse mode and central differences."""
        rng = np.random.default_rng(11)
        B = rng.uniform(-1.0, 1.0, (3, 3)).astype(np.float32)
        dB = rng.uniform(-1.0, 1.0, (3, 3)).astype(np.float32)
        self._check(eig3_jet_kernel, eig3_plain_kernel, wp.mat33, wp.vec3, B + B.T, dB + dB.T, seed=12)

    def test_eig3_repeated_eigenvalues(self):
        """Check that the eig3 jet of the identity keeps finite derivatives with exact eigenvalue derivatives."""
        rng = np.random.default_rng(13)
        dB = rng.uniform(-1.0, 1.0, (3, 3)).astype(np.float32)
        dA = dB + dB.T
        zero = wp.mat33(0.0)
        for device in DEVICES:
            out = wp.zeros(1, dtype=float, device=device)
            # Eigenvalues only: the derivative of sum(d) = trace(A) is trace(dA).
            wp.launch(
                eig3_jet_kernel,
                dim=1,
                inputs=[
                    wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                    wp.mat33(*dA.ravel()),
                    zero,
                    wp.vec3(1.0),
                    zero,
                ],
                outputs=[out],
                device=device,
            )
            np.testing.assert_allclose(out.numpy()[0], np.trace(dA), rtol=1.0e-5, atol=1.0e-5)


@wp.func
def indexed_energy(s: Any, v: Any, F: Any, negative: bool):
    W = wp.mat33(0.3, -0.2, 0.5, 0.1, 0.7, -0.4, -0.6, 0.2, 0.9)
    w = v
    G = F
    H = F
    if negative:
        w[-1] = s
        G[0, -1] = s
        H[-1] = v
        e = v[-2] * F[0, -1] + wp.dot(F[-1], v)
    else:
        w[2] = s
        G[0, 2] = s
        H[2] = v
        e = v[1] * F[0, 2] + wp.dot(F[2], v)
    return e + wp.dot(w, wp.vec3(1.0, 2.0, 3.0)) + wp.ddot(G, W) + 2.0 * wp.ddot(H, W)


@wp.kernel
def negative_index_kernel(
    x_F: wp.mat33, values: wp.array[float], grads: wp.array[JF.grad_type_scalar], plain: wp.array[float]
):
    x_s = 1.7
    x_v = wp.vec3(0.6, -0.8, 1.3)
    for k in range(2):
        s, v, F = JF.seed(x_s, x_v, x_F)
        e = indexed_energy(s, v, F, k == 0)
        values[k] = e.value
        grads[k] = e.grad
        plain[k] = indexed_energy(x_s, x_v, x_F, k == 0)


@wp.func
def indexed_energy2(s: Any, v: Any, negative: bool):
    w = v
    if negative:
        w[-2] = s
    else:
        w[1] = s
    e = wp.dot(w, wp.vec3(1.0, 2.0, 3.0))
    return e * e


@wp.kernel
def negative_index2_kernel(
    values: wp.array[float], grads: wp.array[J2F.grad_type_scalar], hessians: wp.array[J2F.hess_type_scalar]
):
    for k in range(2):
        s, v = J2F.seed(1.7, wp.vec3(0.6, -0.8, 1.3))
        e = indexed_energy2(s, v, k == 0)
        values[k] = e.value
        grads[k] = e.grad
        hessians[k] = e.hess


@wp.kernel
def normalize_zero_kernel(
    values: wp.array[float],
    grads: wp.array[J1F.grad_type_scalar],
    grads2: wp.array[J2F.grad_type_scalar],
    hessians2: wp.array[J2F.hess_type_scalar],
):
    c = wp.vec3(1.0, 2.0, 3.0)
    _s, v = J1F.seed(1.7, wp.vec3(0.0, 0.0, 0.0))
    e = wp.dot(wp.normalize(v), c)
    values[0] = e.value
    grads[0] = e.grad
    _s2, v2 = J2F.seed(1.7, wp.vec3(0.0, 0.0, 0.0))
    e2 = wp.dot(wp.normalize(v2), c)
    values[1] = e2.value
    grads2[0] = e2.grad
    hessians2[0] = e2.hess
    values[2] = wp.dot(wp.normalize(wp.vec3(0.0, 0.0, 0.0)), c)


class TestJetIndexing(unittest.TestCase):
    def test_negative_indices(self):
        """Check that negative vector and matrix indices read and assign the same jet components as native types."""
        for device in DEVICES:
            values = wp.zeros(2, dtype=float, device=device)
            grads = wp.zeros(2, dtype=JF.grad_type_scalar, device=device)
            plain = wp.zeros(2, dtype=float, device=device)
            x_F = wp.mat33(1.2, -0.7, 0.4, 0.3, 0.9, -1.1, 0.8, 0.5, 1.5)
            wp.launch(negative_index_kernel, dim=1, inputs=[x_F], outputs=[values, grads, plain], device=device)
            np.testing.assert_array_equal(values.numpy()[0], values.numpy()[1])
            np.testing.assert_array_equal(grads.numpy()[0], grads.numpy()[1])
            np.testing.assert_allclose(values.numpy(), plain.numpy(), rtol=1.0e-6)

            values = wp.zeros(2, dtype=float, device=device)
            grads = wp.zeros(2, dtype=J2F.grad_type_scalar, device=device)
            hessians = wp.zeros(2, dtype=J2F.hess_type_scalar, device=device)
            wp.launch(negative_index2_kernel, dim=1, outputs=[values, grads, hessians], device=device)
            np.testing.assert_array_equal(values.numpy()[0], values.numpy()[1])
            np.testing.assert_array_equal(grads.numpy()[0], grads.numpy()[1])
            np.testing.assert_array_equal(hessians.numpy()[0], hessians.numpy()[1])

    def test_normalize_zero_vector(self):
        """Check that normalizing a zero-valued jet vector gives zero value and derivatives, like native normalize."""
        for device in DEVICES:
            values = wp.full(3, np.nan, dtype=float, device=device)
            grads = wp.full(1, np.nan, dtype=J1F.grad_type_scalar, device=device)
            grads2 = wp.full(1, np.nan, dtype=J2F.grad_type_scalar, device=device)
            hessians2 = wp.full(1, np.nan, dtype=J2F.hess_type_scalar, device=device)
            wp.launch(normalize_zero_kernel, dim=1, outputs=[values, grads, grads2, hessians2], device=device)
            np.testing.assert_array_equal(values.numpy(), np.zeros(3))
            np.testing.assert_array_equal(grads.numpy(), np.zeros((1, 4)))
            np.testing.assert_array_equal(grads2.numpy(), np.zeros((1, 4)))
            np.testing.assert_array_equal(hessians2.numpy(), np.zeros((1, 4, 4)))


@wp.func
def degenerate_derivatives(s: Any, v: Any, zero_exponent: Any, one_exponent: Any):
    # Each term sits where a naive derivative formula evaluates 0 * inf or 0 * log(0): the
    # length of a zero vector, a zero base raised to 0 or 1 given as plain floats and as
    # constant jets, and a plain zero base raised to a jet exponent.
    return (
        wp.length(v)
        + wp.pow(s, 0.0)
        + wp.pow(s, zero_exponent)
        + wp.pow(s, 1.0)
        + wp.pow(s, one_exponent)
        + wp.pow(0.0, s + 2.0)
    )


@wp.kernel
def degenerate_derivatives_kernel(
    values: wp.array[float],
    grads: wp.array[J1F.grad_type_scalar],
    grads2: wp.array[J2F.grad_type_scalar],
    hessians2: wp.array[J2F.hess_type_scalar],
):
    s, v = J1F.seed(0.0, wp.vec3(0.0, 0.0, 0.0))
    e = degenerate_derivatives(s, v, J1F.constant(0.0), J1F.constant(1.0))
    values[0] = e.value
    grads[0] = e.grad
    s2, v2 = J2F.seed(0.0, wp.vec3(0.0, 0.0, 0.0))
    e2 = degenerate_derivatives(s2, v2, J2F.constant(0.0), J2F.constant(1.0))
    values[1] = e2.value
    grads2[0] = e2.grad
    hessians2[0] = e2.hess


class TestJetDegenerateDerivatives(unittest.TestCase):
    def test_finite_derivatives_at_zero(self):
        """Return the finite derivatives native Warp gives where naive jet formulas produce 0 * inf."""
        for device in DEVICES:
            values = wp.zeros(2, dtype=float, device=device)
            grads = wp.zeros(1, dtype=J1F.grad_type_scalar, device=device)
            grads2 = wp.zeros(1, dtype=J2F.grad_type_scalar, device=device)
            hessians2 = wp.zeros(1, dtype=J2F.hess_type_scalar, device=device)
            wp.launch(degenerate_derivatives_kernel, dim=1, outputs=[values, grads, grads2, hessians2], device=device)

            # Values: |0| + 1 + 1 + 0 + 0 + 0**2. Only the two exponent-1 powers depend on s, each with slope 1;
            # every second derivative is 0.
            expected_grad = np.array([2.0, 0.0, 0.0, 0.0])
            np.testing.assert_array_equal(values.numpy(), [2.0, 2.0])
            np.testing.assert_array_equal(grads.numpy()[0], expected_grad)
            np.testing.assert_array_equal(grads2.numpy()[0], expected_grad)
            np.testing.assert_array_equal(hessians2.numpy()[0], np.zeros((4, 4)))


# Points where native Warp picks a derivative convention: the origin of atan2, zero and NaN
# for abs, ties and NaN operands for min/max, both clamp bounds, and asin/acos outside [-1, 1].
NATIVE_CONVENTION_POINTS = np.array(
    [
        [0.0, 0.0],
        [2.0, 2.0],
        [2.0, np.nan],
        [np.nan, 2.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [2.0, -2.0],
        [-2.0, 0.0],
    ],
    dtype=np.float32,
)
NATIVE_CONVENTION_OPS = 7  # atan2, abs, min, max, clamp, asin, acos
BINARY_OPS = (0, 2, 3)  # atan2, min, max also take mixed plain/jet operands


@wp.func
def native_convention_op(k: int, a: Any, b: Any):
    if k == 0:
        return wp.atan2(a, b)
    if k == 1:
        return wp.abs(a)
    if k == 2:
        return wp.min(a, b)
    if k == 3:
        return wp.max(a, b)
    if k == 4:
        return wp.clamp(a, -1.0, 1.0)
    if k == 5:
        return wp.asin(a)
    return wp.acos(a)


@wp.func
def native_convention_binary_op(k: int, a: Any, b: Any):
    # The ops of BINARY_OPS, for mixed plain/jet operands.
    if k == 0:
        return wp.atan2(a, b)
    if k == 2:
        return wp.min(a, b)
    return wp.max(a, b)


@wp.kernel
def native_convention_kernel(points: wp.array[wp.vec2], values: wp.array2d[float], grads: wp.array2d[wp.vec2]):
    p, k = wp.tid()
    out = native_convention_op(k, points[p][0], points[p][1])
    values[p, k] = out
    grads[p, k] = wp.vec2(0.0)


@wp.kernel
def jet_convention_kernel(
    points: wp.array[wp.vec2],
    values: wp.array2d[float],
    grads: wp.array2d[wp.vec2],
    mixed_grads: wp.array2d[wp.vec2],
    values2: wp.array2d[float],
    grads2: wp.array2d[wp.vec2],
    hessians2: wp.array2d[wp.mat22],
    mixed_grads2: wp.array2d[wp.vec2],
):
    p, k = wp.tid()
    x = points[p][0]
    y = points[p][1]

    a, b = J2.seed(x, y)
    out = native_convention_op(k, a, b)
    values[p, k] = out.value
    grads[p, k] = wp.vec2(out.grad[0], out.grad[1])
    # Mixed operands: the derivative with respect to the jet operand alone.
    mixed_grads[p, k] = wp.vec2(
        native_convention_binary_op(k, a, y).grad[0], native_convention_binary_op(k, x, b).grad[1]
    )

    a2, b2 = J2_2ND.seed(x, y)
    out2 = native_convention_op(k, a2, b2)
    values2[p, k] = out2.value
    grads2[p, k] = wp.vec2(out2.grad[0], out2.grad[1])
    hessians2[p, k] = wp.mat22(out2.hess[0, 0], out2.hess[0, 1], out2.hess[1, 0], out2.hess[1, 1])
    mixed_grads2[p, k] = wp.vec2(
        native_convention_binary_op(k, a2, y).grad[0], native_convention_binary_op(k, x, b2).grad[1]
    )


J1H = wp.JetSpace(1, dtype=wp.float16)
J2H = wp.JetSpace2(1, dtype=wp.float16)
J2VH = wp.JetSpace2(wp.vec3h)
J2VD = wp.JetSpace2(wp.vec3d)
JSVD = wp.JetSpace(wp.mat22, width=1)


@wp.kernel
def small_denominator_kernel(c: wp.float16, c32: float, values: wp.array[float]):
    # Each quotient of a value by itself is 1, even where 1 / c overflows.
    j = J1H.constant(c)
    j2 = J2H.constant(c)
    values[0] = float((j / j).value)
    values[1] = float((c / j).value)
    values[2] = float((j2 / j2).value)
    values[3] = float((c / j2).value)
    k = J2.constant(c32)
    values[4] = (k / k).value


@wp.kernel
def small_norm_hessian_kernel(v: wp.vec3d, hessians_h: wp.array[wp.mat33d], hessians_d: wp.array[wp.mat33d]):
    # The length and a component of the normalized vector, in float16 and in float64.
    vh = J2VH.seed(wp.vec3h(v))
    vd = J2VD.seed(v)
    hessians_h[0] = wp.mat33d(wp.length(vh).hess)
    hessians_h[1] = wp.mat33d(wp.normalize(vh)[1].hess)
    hessians_d[0] = wp.length(vd).hess
    hessians_d[1] = wp.normalize(vd)[1].hess


@wp.kernel
def svd2_polar_kernel(scale: float, direction: wp.mat22, out: wp.array[float]):
    # The polar factor U V^T of a scaled identity, differentiated along a rotation generator.
    A = JSVD.directional(wp.identity(n=2, dtype=float) * scale, direction)
    U, _s, V = wp.svd2(A)
    R = U * wp.transpose(V)
    out[0] = R.grad[2, 0]


class TestJetNativeConventions(unittest.TestCase):
    def test_match_native_derivative_conventions(self):
        """Match native reverse-mode derivatives where native Warp picks a convention, at both jet orders."""
        n = len(NATIVE_CONVENTION_POINTS)
        shape = (n, NATIVE_CONVENTION_OPS)
        for device in DEVICES:
            points = wp.array(NATIVE_CONVENTION_POINTS, dtype=wp.vec2, device=device)

            # Native values and reverse-mode gradients, one op and point at a time.
            native_values = np.zeros(shape, dtype=np.float32)
            native_grads = np.zeros((*shape, 2), dtype=np.float32)
            for p in range(n):
                for k in range(NATIVE_CONVENTION_OPS):
                    x = wp.array(NATIVE_CONVENTION_POINTS[p : p + 1], dtype=wp.vec2, requires_grad=True, device=device)
                    values = wp.zeros((1, NATIVE_CONVENTION_OPS), dtype=float, requires_grad=True, device=device)
                    grads = wp.zeros((1, NATIVE_CONVENTION_OPS), dtype=wp.vec2, device=device)
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            native_convention_kernel,
                            dim=(1, NATIVE_CONVENTION_OPS),
                            inputs=[x],
                            outputs=[values, grads],
                            device=device,
                        )
                    seed = np.zeros((1, NATIVE_CONVENTION_OPS), dtype=np.float32)
                    seed[0, k] = 1.0
                    tape.backward(grads={values: wp.array(seed, dtype=float, device=device)})
                    native_values[p, k] = values.numpy()[0, k]
                    native_grads[p, k] = x.grad.numpy()[0]

            outputs = [
                wp.zeros(shape, dtype=float, device=device),
                wp.zeros(shape, dtype=wp.vec2, device=device),
                wp.zeros(shape, dtype=wp.vec2, device=device),
                wp.zeros(shape, dtype=float, device=device),
                wp.zeros(shape, dtype=wp.vec2, device=device),
                wp.zeros(shape, dtype=wp.mat22, device=device),
                wp.zeros(shape, dtype=wp.vec2, device=device),
            ]
            wp.launch(jet_convention_kernel, dim=shape, inputs=[points], outputs=outputs, device=device)
            values, grads, mixed, values2, grads2, hessians2, mixed2 = (o.numpy() for o in outputs)

            for name, got in (("first-order value", values), ("second-order value", values2)):
                np.testing.assert_allclose(got, native_values, rtol=1e-6, equal_nan=True, err_msg=name)
            for name, got in (("first-order gradient", grads), ("second-order gradient", grads2)):
                np.testing.assert_allclose(got, native_grads, rtol=1e-6, atol=1e-7, err_msg=name)
            for name, got in (("first-order mixed", mixed), ("second-order mixed", mixed2)):
                np.testing.assert_allclose(
                    got[:, BINARY_OPS], native_grads[:, BINARY_OPS], rtol=1e-6, atol=1e-7, err_msg=name
                )
            self.assertTrue(np.all(np.isfinite(hessians2)), "second-order Hessians contain NaN or inf")

    def test_division_by_small_denominators(self):
        """Divide directly, as native division does, so that a quotient of a tiny value by itself is 1."""
        for device in DEVICES:
            values = wp.zeros(5, dtype=float, device=device)
            wp.launch(
                small_denominator_kernel,
                dim=1,
                inputs=[wp.float16(1.0e-5), 1.0e-39],
                outputs=[values],
                device=device,
            )
            np.testing.assert_array_equal(values.numpy(), np.ones(5))

    def test_small_norm_hessians_do_not_overflow(self):
        """Keep float16 Hessians of length and normalize finite and accurate for small nonzero vectors."""
        v = wp.vec3d(0.003, 0.006, 0.009)
        for device in DEVICES:
            hessians_h = wp.zeros(2, dtype=wp.mat33d, device=device)
            hessians_d = wp.zeros(2, dtype=wp.mat33d, device=device)
            wp.launch(small_norm_hessian_kernel, dim=1, inputs=[v], outputs=[hessians_h, hessians_d], device=device)
            got, want = hessians_h.numpy(), hessians_d.numpy()
            self.assertTrue(np.all(np.isfinite(got)))
            # float16 carries about three significant digits, and its input is rounded too.
            for name, g, w in zip(("length", "normalize"), got, want, strict=True):
                np.testing.assert_allclose(g, w, rtol=2.0e-2, atol=2.0e-2 * np.abs(w).max(), err_msg=name)

    def test_svd2_polar_factor_at_repeated_singular_values(self):
        """Keep the derivative of the svd2 polar factor U V^T at repeated singular values, as native svd2 does."""
        rotation = wp.mat22(0.0, -1.0, 1.0, 0.0)
        for device in DEVICES:
            for scale, expected in ((1.0, 1.0), (2.0, 0.5)):
                out = wp.zeros(1, dtype=float, device=device)
                wp.launch(svd2_polar_kernel, dim=1, inputs=[scale, rotation], outputs=[out], device=device)
                self.assertAlmostEqual(float(out.numpy()[0]), expected, places=5, msg=f"A = {scale} I")


J1S = wp.JetSpace(1)
J2S = wp.JetSpace2(1)
INVERSE_TRIG_POINTS = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)


@wp.kernel
def native_inverse_trig_kernel(x: wp.array[float], out: wp.array2d[float]):
    i = wp.tid()
    out[0, i] = wp.asin(x[i])
    out[1, i] = wp.acos(x[i])


@wp.kernel
def jet_inverse_trig_kernel(x: wp.array[float], values: wp.array2d[float], derivatives: wp.array2d[float]):
    # Rows: asin and acos over a first-order jet, then over a second-order jet.
    i = wp.tid()
    a = J1S.seed(x[i])
    a2 = J2S.seed(x[i])
    s = wp.asin(a)
    c = wp.acos(a)
    s2 = wp.asin(a2)
    c2 = wp.acos(a2)
    values[0, i] = s.value
    values[1, i] = c.value
    values[2, i] = s2.value
    values[3, i] = c2.value
    derivatives[0, i] = s.grad[0]
    derivatives[1, i] = c.grad[0]
    derivatives[2, i] = s2.grad[0]
    derivatives[3, i] = c2.grad[0]


@wp.kernel
def nan_length_kernel(values: wp.array[float]):
    v = wp.vec3(wp.nan, 0.0, 0.0)
    _s, jv = J1F.seed(0.0, v)
    _s2, jv2 = J2F.seed(0.0, v)
    values[0] = wp.length(jv).value
    values[1] = wp.length(jv2).value
    values[2] = wp.length(v)


def _row_input_gradient(kernel, x_np, outputs_shape, out_index, row, device):
    """Return the input gradient of one output row, by reverse mode through ``kernel``."""
    x = wp.array(x_np, dtype=float, requires_grad=True, device=device)
    outputs = [wp.zeros(shape, dtype=float, requires_grad=True, device=device) for shape in outputs_shape]
    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=len(x_np), inputs=[x], outputs=outputs, device=device)
    seed = np.zeros(outputs_shape[out_index], dtype=np.float32)
    seed[row] = 1.0
    tape.backward(grads={outputs[out_index]: wp.array(seed, dtype=float, device=device)})
    return x.grad.numpy()


class TestJetReverseThroughJets(unittest.TestCase):
    def test_inverse_trig_reverse_mode_through_jets(self):
        """Backpropagate through jet asin/acos to the native input gradients, with finite gradients through their derivatives."""
        n = len(INVERSE_TRIG_POINTS)
        for device in DEVICES:
            native = [
                _row_input_gradient(native_inverse_trig_kernel, INVERSE_TRIG_POINTS, [(2, n)], 0, row, device)
                for row in range(2)
            ]
            for row in range(4):
                through_value = _row_input_gradient(
                    jet_inverse_trig_kernel, INVERSE_TRIG_POINTS, [(4, n), (4, n)], 0, row, device
                )
                np.testing.assert_allclose(through_value, native[row % 2], rtol=1e-6, err_msg=f"value row {row}")
                through_derivative = _row_input_gradient(
                    jet_inverse_trig_kernel, INVERSE_TRIG_POINTS, [(4, n), (4, n)], 1, row, device
                )
                self.assertTrue(np.all(np.isfinite(through_derivative)), f"derivative row {row}: {through_derivative}")

    def test_nan_length_stays_nan(self):
        """Return NaN for the length of a NaN jet vector, as native length does."""
        for device in DEVICES:
            values = wp.zeros(3, dtype=float, device=device)
            wp.launch(nan_length_kernel, dim=1, outputs=[values], device=device)
            self.assertTrue(np.all(np.isnan(values.numpy())), values.numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
