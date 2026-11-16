# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Two local variables per term, so width 2 seeds the identity and one forward
# pass yields the whole local gradient.
J2 = wp.JetSpace(2)

# Six directions: two 3D endpoints.
J6 = wp.JetSpace((wp.vec3, wp.vec3))

# A second width at the same dtype, to check that spaces coexist.
J3 = wp.JetSpace(3)

J2D = wp.JetSpace(2, dtype=wp.float64)

# The same two variables with one direction, for Hessian-vector products.
J2_DIR = wp.JetSpace(2, width=1)


# ----------------------------------------------------------------------------
# g(a,b) = sin(a*b) + 0.1*a^3 + exp(b)
#
# Written as a plain @wp.func over jets: no manual differentiation, and no jet
# names bound into this module. The operators resolve through Warp's builtin
# overload table by argument type.
# ----------------------------------------------------------------------------


@wp.func
def local_energy(a: Any, b: Any):
    return wp.sin(a * b) + 0.1 * (a * a * a) + wp.exp(b)


@wp.kernel
def local_gradient(z: wp.array2d[float], grad_g: wp.array[J2.grad_type_scalar]):
    i = wp.tid()

    # Identity seed: dz0 = [1,0], dz1 = [0,1]
    z0 = J2.variable(z[i, 0], 0)
    z1 = J2.variable(z[i, 1], 1)

    grad_g[i] = local_energy(z0, z1).grad


@wp.kernel
def local_directional(z: wp.array2d[float], d: wp.array2d[float], out: wp.array[float]):
    i = wp.tid()
    a, b = J2_DIR.directional(z[i, 0], z[i, 1], d[i, 0], d[i, 1])
    out[i] = local_energy(a, b).grad[0]


@wp.kernel
def local_value(z: wp.array2d[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = local_energy(J2.variable(z[i, 0], 0), J2.variable(z[i, 1], 1)).value


Z_NP = np.array(
    [
        [0.7, -0.4],
        [-0.2, 0.9],
        [0.7, 0.9],
    ],
    dtype=np.float32,
)


def g_np(z):
    """G evaluated with NumPy, independent of any Warp code."""
    a = z[:, 0]
    b = z[:, 1]
    return np.sin(a * b) + 0.1 * a**3 + np.exp(b)


def grad_np(z):
    a = z[:, 0]
    b = z[:, 1]
    return np.stack(
        (
            b * np.cos(a * b) + 0.3 * a**2,
            a * np.cos(a * b) + np.exp(b),
        ),
        axis=1,
    )


def hessian_np(z):
    a = z[:, 0]
    b = z[:, 1]
    s = np.sin(a * b)

    h = np.empty((z.shape[0], 2, 2))
    h[:, 0, 0] = -(b**2) * s + 0.6 * a
    h[:, 0, 1] = np.cos(a * b) - a * b * s
    h[:, 1, 0] = h[:, 0, 1]
    h[:, 1, 1] = -(a**2) * s + np.exp(b)
    return h


def hessian_fd(z, h=1.0e-4):
    """Second differences of g in float64, deriving nothing by hand."""
    out = np.empty((z.shape[0], 2, 2))

    for p in range(2):
        for q in range(2):
            e_p = np.zeros_like(z)
            e_p[:, p] = h

            e_q = np.zeros_like(z)
            e_q[:, q] = h

            out[:, p, q] = (g_np(z + e_p + e_q) - g_np(z + e_p - e_q) - g_np(z - e_p + e_q) + g_np(z - e_p - e_q)) / (
                4.0 * h * h
            )

    return out


def hessian_from_tape(z_np, device):
    """Return the jet gradient and the reverse-over-forward Hessian of every local term.

    The gradient comes from one width-2 forward pass. Each Hessian column comes
    from a width-1 directional derivative along a basis direction and one
    backward sweep of the tape over it.
    """
    m = z_np.shape[0]

    grad_g = wp.zeros(m, dtype=J2.grad_type_scalar, device=device)
    wp.launch(
        local_gradient, dim=m, inputs=[wp.array(z_np, dtype=float, device=device)], outputs=[grad_g], device=device
    )

    hessian = np.empty((m, 2, 2), dtype=np.float32)

    for col in range(2):
        d_np = np.zeros((m, 2), dtype=np.float32)
        d_np[:, col] = 1.0

        z = wp.array(z_np, dtype=float, device=device, requires_grad=True)
        dv = wp.zeros(m, dtype=float, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(
                local_directional,
                dim=m,
                inputs=[z, wp.array(d_np, dtype=float, device=device)],
                outputs=[dv],
                device=device,
            )
        tape.backward(grads={dv: wp.ones(m, dtype=float, device=device)})

        # z.grad[i,b] = d (grad g_i . e_col) / d z[i,b] = H_i[b,col]
        hessian[:, :, col] = z.grad.numpy()

    return grad_g.numpy(), hessian


def test_jet_value(test, device):
    """Check that evaluating a jet energy reproduces the scalar value."""
    m = Z_NP.shape[0]
    out = wp.zeros(m, dtype=float, device=device)

    wp.launch(local_value, dim=m, inputs=[wp.array(Z_NP, dtype=float, device=device)], outputs=[out], device=device)

    np.testing.assert_allclose(out.numpy(), g_np(Z_NP.astype(np.float64)), rtol=1.0e-5, atol=1.0e-6)


def test_jet_gradient(test, device):
    """Check the jet gradient against the closed-form gradient."""
    grad, _ = hessian_from_tape(Z_NP, device)

    np.testing.assert_allclose(grad, grad_np(Z_NP.astype(np.float64)), rtol=1.0e-5, atol=1.0e-6)


def test_jet_hessian(test, device):
    """Check the reverse-over-jet Hessian against closed-form and finite differences.

    The two oracles are independent: an analytic NumPy Hessian and float64
    second differences.
    """
    _, hessian = hessian_from_tape(Z_NP, device)

    z64 = Z_NP.astype(np.float64)

    np.testing.assert_allclose(hessian, hessian_np(z64), rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(hessian, hessian_fd(z64), rtol=1.0e-3, atol=1.0e-4)


def test_jet_hessian_symmetric(test, device):
    """Check that the assembled Hessian is symmetric.

    Each off-diagonal comes from a separate backward pass, so agreement
    between the two is a real check rather than a structural guarantee.
    """
    _, hessian = hessian_from_tape(Z_NP, device)

    # The off-diagonals come from separate backward passes, so this is not
    # automatic.
    np.testing.assert_allclose(hessian, np.transpose(hessian, (0, 2, 1)), rtol=1.0e-5, atol=1.0e-6)


# ----------------------------------------------------------------------------
# 3D geometry through the ordinary Warp builtins.
# ----------------------------------------------------------------------------


@wp.func
def spring_energy(x0: J6.vec3, x1: J6.vec3) -> J6.scalar:
    d = x1 - x0
    r = wp.length(d) - 1.0
    return 0.5 * r * r


@wp.kernel
def spring_gradient(
    x0: wp.array[wp.vec3],
    x1: wp.array[wp.vec3],
    grad: wp.array[J6.grad_type_scalar],
):
    i = wp.tid()

    a, b = J6.seed(x0[i], x1[i])

    grad[i] = spring_energy(a, b).grad


def test_jet_spring_gradient(test, device):
    """Check a width-6 gradient over two vec3 endpoints against the closed form."""
    x0_np = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]], dtype=np.float32)
    x1_np = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32)

    x0 = wp.array(x0_np, dtype=wp.vec3, device=device)
    x1 = wp.array(x1_np, dtype=wp.vec3, device=device)
    grad = wp.zeros(2, dtype=J6.grad_type_scalar, device=device)

    wp.launch(spring_gradient, dim=2, inputs=[x0, x1], outputs=[grad], device=device)

    # E = 0.5*(|d|-1)^2, dE/dx1 = (|d|-1) * dhat, dE/dx0 = -dE/dx1
    d = x1_np - x0_np
    lengths = np.linalg.norm(d, axis=1, keepdims=True)
    dedx1 = (lengths - 1.0) * (d / lengths)

    expected = np.concatenate((-dedx1, dedx1), axis=1)

    np.testing.assert_allclose(grad.numpy(), expected, rtol=1.0e-5, atol=1.0e-6)


# ----------------------------------------------------------------------------
# Component access, dot, cross and normalize.
# ----------------------------------------------------------------------------


@wp.kernel
def extract_and_geometry(out: wp.array[float]):
    v, w = J6.seed(wp.vec3(3.0, 5.0, 0.0), wp.vec3(0.0, 2.0, 4.0))

    # v[i] resolves through the extract builtin
    p = v[0] * v[1]
    out[0] = p.value
    out[1] = p.grad[0]
    out[2] = p.grad[1]

    d = wp.dot(v, w)
    out[3] = d.value
    out[4] = d.grad[0]

    c = wp.cross(v, w)
    out[5] = c.value[0]

    n = wp.normalize(w)
    out[6] = n.value[2]

    out[7] = wp.length_sq(v).value


def test_jet_component_and_geometry(test, device):
    """Check component indexing and the vec3 geometry builtins on jets."""
    out = wp.zeros(8, dtype=float, device=device)
    wp.launch(extract_and_geometry, dim=1, outputs=[out], device=device)

    got = out.numpy()
    v = np.array([3.0, 5.0, 0.0])
    w = np.array([0.0, 2.0, 4.0])

    # v[0]*v[1] = 15, d/dv0 = v1 = 5, d/dv1 = v0 = 3
    np.testing.assert_allclose(got[0:3], [15.0, 5.0, 3.0], rtol=1.0e-5, atol=1.0e-6)

    # dot = 10, d(dot)/dv0 = w0 = 0
    np.testing.assert_allclose(got[3:5], [float(v @ w), w[0]], rtol=1.0e-5, atol=1.0e-6)

    np.testing.assert_allclose(got[5], np.cross(v, w)[0], rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(got[6], (w / np.linalg.norm(w))[2], rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(got[7], float(v @ v), rtol=1.0e-5, atol=1.0e-6)


# ----------------------------------------------------------------------------
# Several spaces coexisting, and a non-default dtype.
# ----------------------------------------------------------------------------


@wp.func
def width3_energy(a: J3.scalar, b: J3.scalar) -> J3.scalar:
    return a * b + a


@wp.kernel
def width3_kernel(out: wp.array[float]):
    g = width3_energy(J3.variable(2.0, 0), J3.variable(3.0, 1))
    out[0] = g.value
    out[1] = g.grad[0]
    out[2] = g.grad[1]
    out[3] = g.grad[2]


def test_jet_multiple_widths(test, device):
    """A second width must not disturb the first."""
    out = wp.zeros(4, dtype=float, device=device)
    wp.launch(width3_kernel, dim=1, outputs=[out], device=device)

    # g = a*b + a = 8 ; dg/da = b+1 = 4 ; dg/db = a = 2 ; dg/dc = 0
    np.testing.assert_allclose(out.numpy(), [8.0, 4.0, 2.0, 0.0], rtol=1.0e-6, atol=1.0e-6)

    # The width-2 space still resolves after width 3 registered its overloads.
    grad, _ = hessian_from_tape(Z_NP, device)
    np.testing.assert_allclose(grad, grad_np(Z_NP.astype(np.float64)), rtol=1.0e-5, atol=1.0e-6)


@wp.func
def energy_f64(a: J2D.scalar, b: J2D.scalar) -> J2D.scalar:
    return wp.sin(a * b) + wp.exp(b)


@wp.kernel
def kernel_f64(out: wp.array[wp.float64]):
    g = energy_f64(J2D.variable(wp.float64(0.7), 0), J2D.variable(wp.float64(-0.4), 1))
    out[0] = g.value
    out[1] = g.grad[0]
    out[2] = g.grad[1]


def test_jet_float64(test, device):
    """Check that a float64 jet space reaches float64 accuracy."""
    out = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(kernel_f64, dim=1, outputs=[out], device=device)

    a, b = 0.7, -0.4
    expected = [
        np.sin(a * b) + np.exp(b),
        b * np.cos(a * b),
        a * np.cos(a * b) + np.exp(b),
    ]

    # float64 throughout, so this should hold far tighter than float32 would.
    np.testing.assert_allclose(out.numpy(), expected, rtol=1.0e-12, atol=1.0e-14)


def float_huber_kernel(v: wp.array[wp.vec3], out: wp.array[float]):
    i = wp.tid()
    out[i] = wp.norm_huber(v[i]) + wp.value(out[i])


class TestJetSpace(unittest.TestCase):
    def test_new_jet_space_keeps_float_module_hash(self):
        """Keep the hash of a module that calls wp.value() on plain floats when a new jet space adds overloads."""
        kernel = wp.kernel(float_huber_kernel, module="unique")
        options = kernel.module.resolve_options(wp.config)

        def module_hash():
            return wp._src.context.ModuleHasher(kernel.module._get_live_kernels(), options).get_hash()

        before = module_hash()
        # A width and dtype no other test uses, so the space is new and adds its overloads.
        wp.JetSpace(29, dtype=wp.float16)
        wp.JetSpace2(29, dtype=wp.float16)
        self.assertEqual(module_hash(), before)

    def test_jet_space_is_cached(self):
        """Check that repeating a (width, dtype) request returns the cached space."""
        self.assertIs(wp.JetSpace(2), J2)
        self.assertIs(wp.JetSpace(2, dtype=wp.float32), J2)
        self.assertIsNot(wp.JetSpace(2, dtype=wp.float64), J2)
        self.assertIs(wp.JetSpace(2, dtype=wp.float64), J2D)

    def test_jet_space_rejects_bad_width(self):
        """Check that a non-positive width is rejected."""
        with self.assertRaises(ValueError):
            wp.JetSpace(0)

        with self.assertRaises(ValueError):
            wp.JetSpace(-1)

    def test_jet_space_rejects_non_float_dtype(self):
        """Check that an integer dtype is rejected before any overload is registered.

        Registering a space mutates Warp's global builtin overload table for the
        life of the process, so an integer space cannot simply be discarded once
        its truncating division shows up in a derivative.
        """
        for dtype in (wp.int32, wp.int64, wp.uint8):
            with self.assertRaises(TypeError):
                wp.JetSpace(2, dtype=dtype)

            with self.assertRaises(TypeError):
                wp.JetSpace2(2, dtype=dtype)

    def test_jet_space_reports_width(self):
        """Check that a space reports the width it was created with."""
        self.assertEqual(J2.width, 2)
        self.assertEqual(J6.width, 6)
        self.assertEqual(wp.types.type_size(J2.grad_type_scalar), 2)
        self.assertEqual(wp.types.type_size(J6.grad_type_scalar), 6)

    def test_jet_space_slots_set_width(self):
        """Check the width defaults to the number of scalar components in the slots."""
        self.assertEqual(wp.JetSpace((wp.vec3, wp.vec3)).width, 6)
        self.assertEqual(wp.JetSpace((wp.vec2, wp.float32, wp.mat22)).width, 7)
        self.assertEqual(wp.JetSpace((wp.vec3,) * 4).width, 12)
        self.assertEqual(wp.JetSpace(wp.mat33).width, 9)
        self.assertEqual(wp.JetSpace((wp.vec3, wp.vec3), width=1).width, 1)

    def test_jet_space_layouts_share_types(self):
        """Check layouts of one width and dtype share jet types and are cached."""
        nodes = wp.JetSpace((wp.vec3, wp.vec3))
        self.assertIs(nodes, J6)
        self.assertIs(wp.JetSpace((wp.vec2, wp.vec2, wp.float32, wp.float32)).scalar, wp.JetSpace(6).scalar)
        self.assertIs(nodes.scalar, wp.JetSpace(6).scalar)
        self.assertIsNot(nodes, wp.JetSpace(6))

    def test_jet_space_infers_dtype_from_slots(self):
        """Check the scalar type comes from the slot types when dtype is omitted."""
        self.assertIs(wp.JetSpace((wp.vec3d, wp.vec3d)).dtype, wp.float64)
        self.assertIs(wp.JetSpace((wp.vec3, wp.vec3)).dtype, wp.float32)
        self.assertIs(wp.JetSpace(float).dtype, wp.float32)
        self.assertIs(wp.JetSpace(2, dtype=wp.float64).dtype, wp.float64)

    def test_jet_space_rejects_mismatched_dtype(self):
        """Check slot types that disagree with each other or with dtype are rejected."""
        with self.assertRaises(TypeError):
            wp.JetSpace((wp.vec3, wp.vec3d))

        with self.assertRaises(TypeError):
            wp.JetSpace((wp.vec3, wp.vec3), dtype=wp.float64)

    def test_jet_space_rejects_unsupported_slots(self):
        """Check slot types without a jet type are rejected, as is an empty layout."""
        for slot in (wp.vec4, wp.vec3i, wp.quat, int, "vec3"):
            with self.subTest(slot=slot), self.assertRaises(TypeError):
                wp.JetSpace((slot, wp.vec3))

        with self.assertRaises(ValueError):
            wp.JetSpace(())

    def test_jet_space_seed_requires_full_width(self):
        """Check seed() is withheld, with an explanation, when width differs from the component count."""
        narrow = wp.JetSpace((wp.vec3, wp.vec3), width=1)
        self.assertFalse(hasattr(narrow, "seed"))
        with self.assertRaisesRegex(AttributeError, "directional"):
            getattr(narrow, "seed")  # noqa: B009

        self.assertTrue(hasattr(narrow, "directional"))

    def test_jet_space_type_lookup(self):
        """Check jet_type() and grad_type() return the types of each jet and its gradients."""
        kinds = (
            (wp.float32, "scalar"),
            (float, "scalar"),
            (wp.vec2, "vec2"),
            (wp.vec3, "vec3"),
            (wp.mat22, "mat22"),
            (wp.mat33, "mat33"),
            (wp.types.matrix((3, 2), wp.float32), "mat32"),
            (wp.types.matrix((2, 3), wp.float32), "mat23"),
        )
        for t, name in kinds:
            with self.subTest(type=name):
                self.assertIs(J6.jet_type(t), getattr(J6, name))
                self.assertIs(J6.grad_type(t), getattr(J6, f"grad_type_{name}"))
                self.assertIs(J6.jet_type(t).vars["grad"].type, J6.grad_type(t))

        self.assertEqual(J6.grad_type_vec2._shape_, (2, 6))
        self.assertEqual(J6.grad_type_mat22._shape_, (4, 6))

        with self.assertRaisesRegex(TypeError, "float64"):
            J6.grad_type(wp.vec2d)
        for t in (wp.vec4, wp.mat44, int):
            with self.subTest(type=t), self.assertRaises(TypeError):
                J6.jet_type(t)

        spring = wp.JetSpace2((wp.vec3, wp.vec3))
        self.assertIs(spring.jet_type(wp.vec3), spring.vec3)
        self.assertIs(spring.scalar.vars["grad"].type, spring.grad_type_scalar)
        self.assertIs(spring.scalar.vars["hess"].type, spring.hess_type_scalar)
        self.assertFalse(hasattr(spring, "grad_type"))
        with self.assertRaises(TypeError):
            spring.jet_type(wp.mat22)

    def test_jet_space2_slots(self):
        """Check second-order spaces take scalar, vec2 and vec3 slots but not matrices."""
        self.assertIs(wp.JetSpace2((wp.float32, wp.float32)), wp.JetSpace2(2))
        self.assertEqual(wp.JetSpace2(3).width, 3)

        # Vector slots set the width from their components and share the scalar types.
        spring = wp.JetSpace2((wp.vec3, wp.vec3))
        self.assertEqual(spring.width, 6)
        self.assertIs(spring.scalar, wp.JetSpace2(6).scalar)
        self.assertEqual(wp.JetSpace2((wp.vec3, wp.float32, wp.vec2)).width, 6)
        self.assertIs(wp.JetSpace2(wp.vec3d).dtype, wp.float64)

        with self.assertRaisesRegex(TypeError, "matrix"):
            wp.JetSpace2((wp.vec3, wp.mat33))


# ===========================================================================
# Second-order (forward-over-forward) jets: value, gradient, and Hessian from a
# single forward pass. Cross-checked against the analytic references above, the
# reverse-over-forward Hessian, and finite differences.
# ===========================================================================

J2_2ND = wp.JetSpace2(2)


@wp.kernel(enable_backward=False)
def jet2_value_grad_hess(
    z: wp.array2d[float],
    val: wp.array[float],
    grad: wp.array2d[float],
    hess: wp.array3d[float],
):
    # Same energy as local_energy: sin(a*b) + 0.1*a^3 + exp(b).
    i = wp.tid()
    a = J2_2ND.variable(z[i, 0], 0)
    b = J2_2ND.variable(z[i, 1], 1)
    r = wp.sin(a * b) + 0.1 * (a * a * a) + wp.exp(b)
    val[i] = r.value
    for p in range(2):
        grad[i, p] = r.grad[p]
        for q in range(2):
            hess[i, p, q] = r.hess[p, q]


def g2_np(z):
    # Exercises the ops local_energy does not: div(jet,jet), log, sqrt, cos.
    a = z[:, 0]
    b = z[:, 1]
    return np.log(a * a + b * b + 1.5) + np.sqrt(a * a + 1.0) + a / (b * b + 2.0) + np.cos(a * b)


@wp.kernel(enable_backward=False)
def jet2_g2(
    z: wp.array2d[float],
    val: wp.array[float],
    grad: wp.array2d[float],
    hess: wp.array3d[float],
):
    i = wp.tid()
    a = J2_2ND.variable(z[i, 0], 0)
    b = J2_2ND.variable(z[i, 1], 1)
    r = wp.log(a * a + b * b + 1.5) + wp.sqrt(a * a + 1.0) + a / (b * b + 2.0) + wp.cos(a * b)
    val[i] = r.value
    for p in range(2):
        grad[i, p] = r.grad[p]
        for q in range(2):
            hess[i, p, q] = r.hess[p, q]


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


def _run_jet2(kernel, z_np, device):
    m = z_np.shape[0]
    z = wp.array(z_np, dtype=float, device=device)
    val = wp.zeros(m, dtype=float, device=device)
    grad = wp.zeros((m, 2), dtype=float, device=device)
    hess = wp.zeros((m, 2, 2), dtype=float, device=device)
    wp.launch(kernel, dim=m, inputs=[z], outputs=[val, grad, hess], device=device)
    return val.numpy(), grad.numpy(), hess.numpy()


def test_jet2_value_grad_hessian(test, device):
    """Check the second-order jet value, gradient, and Hessian in one forward pass."""
    val, grad, hess = _run_jet2(jet2_value_grad_hess, Z_NP, device)
    z64 = Z_NP.astype(np.float64)

    # One forward pass reproduces the analytic value, gradient, and Hessian.
    np.testing.assert_allclose(val, g_np(z64), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(grad, grad_np(z64), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(hess, hessian_np(z64), rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(hess, hessian_fd(z64), rtol=1.0e-3, atol=1.0e-4)

    # It also agrees with the reverse-over-forward Hessian of the same energy.
    _, hess_tape = hessian_from_tape(Z_NP, device)
    np.testing.assert_allclose(hess, hess_tape, rtol=1.0e-4, atol=1.0e-5)


def test_jet2_hessian_symmetric(test, device):
    """Check that the pure-forward Hessian is symmetric."""
    # The forward-over-forward Hessian is symmetric by construction.
    _, _, hess = _run_jet2(jet2_value_grad_hess, Z_NP, device)
    np.testing.assert_allclose(hess, np.transpose(hess, (0, 2, 1)), rtol=1.0e-6, atol=1.0e-7)


def test_jet2_div_log_sqrt(test, device):
    """Check second-order division, log, and sqrt against finite differences."""
    # Covers div(jet, jet), log, sqrt, and cos, which local_energy does not.
    val, grad, hess = _run_jet2(jet2_g2, Z_NP, device)
    z64 = Z_NP.astype(np.float64)

    np.testing.assert_allclose(val, g2_np(z64), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(grad, _grad_fd(g2_np, z64), rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(hess, _hess_fd(g2_np, z64), rtol=1.0e-3, atol=1.0e-4)
    np.testing.assert_allclose(hess, np.transpose(hess, (0, 2, 1)), rtol=1.0e-6, atol=1.0e-7)


# ----------------------------------------------------------------------------
# Seeding by slot: seed() and directional() build jets from the slot layout.
# ----------------------------------------------------------------------------

# A vec2, a scalar and a 2x2 matrix: 2 + 1 + 4 = 7 directions, numbered in order
# with matrices row-major.
J_MIXED = wp.JetSpace((wp.vec2, wp.float32, wp.mat22))

# One slot, so seed() takes and returns the bare value.
J_SINGLE = wp.JetSpace(wp.vec3)

# The same two vec3 endpoints as J6, but one direction: a directional derivative.
J_DIR = wp.JetSpace((wp.vec3, wp.vec3), width=1)


@wp.kernel
def seed_mixed_slots(out_v: wp.array2d[float], out_s: wp.array[float], out_m: wp.array2d[float]):
    v, s, m = J_MIXED.seed(wp.vec2(1.0, 2.0), 3.0, wp.mat22(4.0, 5.0, 6.0, 7.0))

    for r in range(2):
        for q in range(7):
            out_v[r, q] = v.grad[r, q]
    for q in range(7):
        out_s[q] = s.grad[q]
    for r in range(4):
        for q in range(7):
            out_m[r, q] = m.grad[r, q]


def test_jet_seed_mixed_slots(test, device):
    """Check seed() numbers directions slot by slot, with matrices row-major."""
    out_v = wp.zeros((2, 7), dtype=float, device=device)
    out_s = wp.zeros(7, dtype=float, device=device)
    out_m = wp.zeros((4, 7), dtype=float, device=device)
    wp.launch(seed_mixed_slots, dim=1, inputs=[], outputs=[out_v, out_s, out_m], device=device)

    identity = np.eye(7, dtype=np.float32)
    np.testing.assert_array_equal(out_v.numpy(), identity[0:2])
    np.testing.assert_array_equal(out_s.numpy(), identity[2])
    np.testing.assert_array_equal(out_m.numpy(), identity[3:7])


@wp.kernel
def seed_single_slot(out: wp.array2d[float]):
    p = J_SINGLE.seed(wp.vec3(1.0, 2.0, 3.0))

    for r in range(3):
        for q in range(3):
            out[r, q] = p.grad[r, q]


def test_jet_seed_single_slot(test, device):
    """Check a single-slot space takes and returns the bare value."""
    out = wp.zeros((3, 3), dtype=float, device=device)
    wp.launch(seed_single_slot, dim=1, inputs=[], outputs=[out], device=device)

    np.testing.assert_array_equal(out.numpy(), np.eye(3, dtype=np.float32))


@wp.func
def spring_energy_dir(x0: J_DIR.vec3, x1: J_DIR.vec3) -> J_DIR.scalar:
    d = x1 - x0
    r = wp.length(d) - 1.0
    return 0.5 * r * r


@wp.kernel
def spring_directional(
    x0: wp.array[wp.vec3],
    x1: wp.array[wp.vec3],
    d0: wp.array[wp.vec3],
    d1: wp.array[wp.vec3],
    out: wp.array[float],
):
    i = wp.tid()

    # One combined direction (d0, d1) over both endpoints.
    a, b = J_DIR.directional(x0[i], x1[i], d0[i], d1[i])

    out[i] = spring_energy_dir(a, b).grad[0]


@wp.kernel
def spring_directional_basis(
    x0: wp.array[wp.vec3],
    x1: wp.array[wp.vec3],
    grad: wp.array[J6.grad_type_scalar],
):
    i = wp.tid()

    ex = wp.vec3(1.0, 0.0, 0.0)
    ey = wp.vec3(0.0, 1.0, 0.0)
    ez = wp.vec3(0.0, 0.0, 1.0)
    zero = wp.vec3(0.0)

    # Direction-major: for each direction, the vector given to slot 0 then slot 1.
    a, b = J6.directional(x0[i], x1[i], ex, zero, ey, zero, ez, zero, zero, ex, zero, ey, zero, ez)

    grad[i] = spring_energy(a, b).grad


def test_jet_directional(test, device):
    """Check directional() reproduces seed() for a basis and contracts the gradient for one direction."""
    x0_np = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]], dtype=np.float32)
    x1_np = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32)
    d0_np = np.array([[1.0, -2.0, 0.5], [0.0, 1.0, 1.0]], dtype=np.float32)
    d1_np = np.array([[0.5, 0.5, 0.0], [-1.0, 2.0, 3.0]], dtype=np.float32)

    x0 = wp.array(x0_np, dtype=wp.vec3, device=device)
    x1 = wp.array(x1_np, dtype=wp.vec3, device=device)
    d0 = wp.array(d0_np, dtype=wp.vec3, device=device)
    d1 = wp.array(d1_np, dtype=wp.vec3, device=device)

    # Six explicit coordinate directions match the six seeded ones.
    grad_basis = wp.zeros(2, dtype=J6.grad_type_scalar, device=device)
    grad_seed = wp.zeros(2, dtype=J6.grad_type_scalar, device=device)
    wp.launch(spring_directional_basis, dim=2, inputs=[x0, x1], outputs=[grad_basis], device=device)
    wp.launch(spring_gradient, dim=2, inputs=[x0, x1], outputs=[grad_seed], device=device)
    np.testing.assert_array_equal(grad_basis.numpy(), grad_seed.numpy())

    # One combined direction gives the gradient contracted with that direction.
    out = wp.zeros(2, dtype=float, device=device)
    wp.launch(spring_directional, dim=2, inputs=[x0, x1, d0, d1], outputs=[out], device=device)

    expected = np.sum(grad_seed.numpy() * np.concatenate((d0_np, d1_np), axis=1), axis=1)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1.0e-5, atol=1.0e-6)


@wp.kernel
def spring_gradient_dot_wrt_inputs(
    x0: wp.array[wp.vec3],
    x1: wp.array[wp.vec3],
    w: wp.array[J6.grad_type_scalar],
    out: wp.array[float],
):
    i = wp.tid()

    a, b = J6.seed(x0[i], x1[i])

    out[i] = wp.dot(spring_energy(a, b).grad, w[i])


@wp.kernel
def spring_directional_wrt_inputs(
    x0: wp.array[wp.vec3],
    x1: wp.array[wp.vec3],
    d0: wp.array[wp.vec3],
    d1: wp.array[wp.vec3],
    out: wp.array[float],
):
    i = wp.tid()

    a, b = J_DIR.directional(x0[i], x1[i], d0[i], d1[i])

    out[i] = spring_energy_dir(a, b).grad[0]


def _spring_hessian_np(x0, x1):
    """Return the 6x6 Hessian of the spring energy at one pair of endpoints, in closed form."""
    d = (x1 - x0).astype(np.float64)
    length = np.linalg.norm(d)
    dhat = d / length
    block = np.outer(dhat, dhat) + (1.0 - 1.0 / length) * (np.eye(3) - np.outer(dhat, dhat))
    return np.block([[block, -block], [-block, block]])


def test_jet_seed_is_differentiable(test, device):
    """Check reverse mode differentiates through seed() and directional() to give the Hessian-vector product."""
    x0_np = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]], dtype=np.float32)
    x1_np = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32)
    d0_np = np.array([[1.0, -2.0, 0.5], [0.0, 1.0, 1.0]], dtype=np.float32)
    d1_np = np.array([[0.5, 0.5, 0.0], [-1.0, 2.0, 3.0]], dtype=np.float32)
    m = x0_np.shape[0]

    hess_np = np.stack([_spring_hessian_np(x0_np[i], x1_np[i]) for i in range(m)])
    direction = np.concatenate((d0_np, d1_np), axis=1)
    expected = np.einsum("mij,mj->mi", hess_np, direction)

    # Width-6 jets: one backward sweep of (gradient . d) through seed() gives H d.
    x0 = wp.array(x0_np, dtype=wp.vec3, device=device, requires_grad=True)
    x1 = wp.array(x1_np, dtype=wp.vec3, device=device, requires_grad=True)
    w = wp.array(direction, dtype=J6.grad_type_scalar, device=device)
    out = wp.zeros(m, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(spring_gradient_dot_wrt_inputs, dim=m, inputs=[x0, x1, w], outputs=[out], device=device)
    tape.backward(grads={out: wp.ones(m, dtype=float, device=device)})

    np.testing.assert_allclose(
        np.concatenate((x0.grad.numpy(), x1.grad.numpy()), axis=1), expected, rtol=1.0e-4, atol=1.0e-5
    )

    # A width-1 jet over the same direction through directional(): one backward sweep gives H d.
    x0 = wp.array(x0_np, dtype=wp.vec3, device=device, requires_grad=True)
    x1 = wp.array(x1_np, dtype=wp.vec3, device=device, requires_grad=True)
    d0 = wp.array(d0_np, dtype=wp.vec3, device=device)
    d1 = wp.array(d1_np, dtype=wp.vec3, device=device)
    out = wp.zeros(m, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(spring_directional_wrt_inputs, dim=m, inputs=[x0, x1, d0, d1], outputs=[out], device=device)
    tape.backward(grads={out: wp.ones(m, dtype=float, device=device)})

    np.testing.assert_allclose(
        np.concatenate((x0.grad.numpy(), x1.grad.numpy()), axis=1), expected, rtol=1.0e-4, atol=1.0e-5
    )


@wp.func
def spring_directional_derivative(x0: wp.vec3, x1: wp.vec3, d0: wp.vec3, d1: wp.vec3):
    a, b = J_DIR.directional(x0, x1, d0, d1)
    return spring_energy_dir(a, b).grad[0]


@wp.func
def spring_seeded_gradient_dot(x0: wp.vec3, x1: wp.vec3, w: J6.grad_type_scalar):
    a, b = J6.seed(x0, x1)
    return wp.dot(spring_energy(a, b).grad, w)


@wp.kernel(enable_backward=False)
def spring_hessian_vector_in_kernel(
    x0: wp.array[wp.vec3],
    x1: wp.array[wp.vec3],
    d0: wp.array[wp.vec3],
    d1: wp.array[wp.vec3],
    hv_directional: wp.array2d[wp.vec3],
    hv_seeded: wp.array2d[wp.vec3],
):
    i = wp.tid()

    # Reverse mode taken inside the kernel over a first-order jet: no tape and no
    # global .grad array.
    g0, g1, _a, _b = wp.grad(spring_directional_derivative)(x0[i], x1[i], d0[i], d1[i])
    hv_directional[i, 0] = g0
    hv_directional[i, 1] = g1

    # The same product from the width-6 gradient contracted with the direction.
    w = J6.grad_type_scalar(d0[i][0], d0[i][1], d0[i][2], d1[i][0], d1[i][1], d1[i][2])
    h0, h1, _w = wp.grad(spring_seeded_gradient_dot)(x0[i], x1[i], w)
    hv_seeded[i, 0] = h0
    hv_seeded[i, 1] = h1


def test_jet_in_kernel_grad_hessian_vector(test, device):
    """Check wp.grad inside a kernel over first-order jets gives the Hessian-vector product."""
    x0_np = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]], dtype=np.float32)
    x1_np = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32)
    d0_np = np.array([[1.0, -2.0, 0.5], [0.0, 1.0, 1.0]], dtype=np.float32)
    d1_np = np.array([[0.5, 0.5, 0.0], [-1.0, 2.0, 3.0]], dtype=np.float32)
    m = x0_np.shape[0]

    hess_np = np.stack([_spring_hessian_np(x0_np[i], x1_np[i]) for i in range(m)])
    expected = np.einsum("mij,mj->mi", hess_np, np.concatenate((d0_np, d1_np), axis=1))

    arrays = [wp.array(a, dtype=wp.vec3, device=device) for a in (x0_np, x1_np, d0_np, d1_np)]
    hv_directional = wp.zeros((m, 2), dtype=wp.vec3, device=device)
    hv_seeded = wp.zeros((m, 2), dtype=wp.vec3, device=device)
    wp.launch(
        spring_hessian_vector_in_kernel,
        dim=m,
        inputs=arrays,
        outputs=[hv_directional, hv_seeded],
        device=device,
    )

    np.testing.assert_allclose(hv_directional.numpy().reshape(m, 6), expected, rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(hv_seeded.numpy().reshape(m, 6), expected, rtol=1.0e-4, atol=1.0e-5)


@wp.kernel
def jet2_seed(
    z: wp.array2d[float],
    val: wp.array[float],
    grad: wp.array2d[float],
    hess: wp.array3d[float],
):
    i = wp.tid()

    a, b = J2_2ND.seed(z[i, 0], z[i, 1])
    g = wp.sin(a * b) + 0.1 * (a * a * a) + wp.exp(b)

    val[i] = g.value
    for p in range(2):
        grad[i, p] = g.grad[p]
        for q in range(2):
            hess[i, p, q] = g.hess[p, q]


def test_jet2_seed(test, device):
    """Check second-order seed() over scalars matches the analytic Hessian."""
    z64 = Z_NP.astype(np.float64)
    m = Z_NP.shape[0]

    z = wp.array(Z_NP, dtype=float, device=device)
    val = wp.zeros(m, dtype=float, device=device)
    grad = wp.zeros((m, 2), dtype=float, device=device)
    hess = wp.zeros((m, 2, 2), dtype=float, device=device)
    wp.launch(jet2_seed, dim=m, inputs=[z], outputs=[val, grad, hess], device=device)

    np.testing.assert_allclose(val.numpy(), g_np(z64), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(grad.numpy(), grad_np(z64), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(hess.numpy(), hessian_np(z64), rtol=1.0e-4, atol=1.0e-5)


# ===========================================================================
# Second-order vector slots. A vector-typed function generic over its argument
# types runs unchanged on first- and second-order jets.
# ===========================================================================

J2_SPRING = wp.JetSpace2((wp.vec3, wp.vec3))

# The vec3 operations on a width-3 space, and the vec2 operations on a width-3
# space that also takes a scalar slot. Narrow widths keep these second-order
# kernels quick to compile.
J2_V3 = wp.JetSpace2(wp.vec3)
J2_V2 = wp.JetSpace2((wp.vec2, wp.float32))


@wp.func
def spring_energy_generic(x0: Any, x1: Any):
    r = wp.length(x1 - x0) - 1.0
    return 0.5 * r * r


@wp.kernel
def jet2_spring_hessian(x0: wp.array[wp.vec3], x1: wp.array[wp.vec3], hess: wp.array3d[float]):
    i = wp.tid()
    a, b = J2_SPRING.seed(x0[i], x1[i])
    h = spring_energy_generic(a, b).hess
    for p in range(6):
        for q in range(6):
            hess[i, p, q] = h[p, q]


@wp.kernel
def jet1_spring_gradient_generic(x0: wp.array[wp.vec3], x1: wp.array[wp.vec3], grad: wp.array[J6.grad_type_scalar]):
    i = wp.tid()
    a, b = J6.seed(x0[i], x1[i])
    grad[i] = spring_energy_generic(a, b).grad


def test_jet2_vector_spring(test, device):
    """Check one generic vec3 energy gives the gradient on first-order jets and the Hessian on second-order jets."""
    x0_np = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]], dtype=np.float32)
    x1_np = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32)
    m = x0_np.shape[0]
    x0 = wp.array(x0_np, dtype=wp.vec3, device=device)
    x1 = wp.array(x1_np, dtype=wp.vec3, device=device)

    hess = wp.zeros((m, 6, 6), dtype=float, device=device)
    wp.launch(jet2_spring_hessian, dim=m, inputs=[x0, x1], outputs=[hess], device=device)
    hess_np = np.stack([_spring_hessian_np(x0_np[i], x1_np[i]) for i in range(m)])
    np.testing.assert_allclose(hess.numpy(), hess_np, rtol=1.0e-5, atol=1.0e-6)

    grad = wp.zeros(m, dtype=J6.grad_type_scalar, device=device)
    wp.launch(jet1_spring_gradient_generic, dim=m, inputs=[x0, x1], outputs=[grad], device=device)
    d = (x1_np - x0_np).astype(np.float64)
    length = np.linalg.norm(d, axis=1, keepdims=True)
    g1 = (length - 1.0) * d / length
    np.testing.assert_allclose(grad.numpy(), np.concatenate((-g1, g1), axis=1), rtol=1.0e-5, atol=1.0e-6)


@wp.func
def vec3_ops_energy(p: Any):
    # Uses every second-order vec3 overload: each mix of jet and plain operands,
    # negative indexing, and the namespace constructors.
    c = wp.vec3(0.3, -0.2, 0.5)
    m = wp.mat33(0.9, -0.4, 0.2, 0.1, 1.1, -0.3, -0.5, 0.6, 0.8)
    s = 0.5 * p[1] + 1.2
    u = 2.0 * p - c
    v = c - p / 3.0 + p * s + 0.25 * (s * p)
    w = s * c + c * s + (+p) + (-u) + (c + p) * 0.5 + (p + c)
    return (
        wp.dot(wp.cross(p, c), wp.cross(c, u))
        + wp.dot(wp.cross(u, v), w)
        + wp.length(v)
        + wp.length(u - v)
        + wp.length_sq(w)
        + wp.dot(wp.normalize(u), c)
        + wp.dot(c, w)
        + p[-1] * u[0]
        + v[1]
        + wp.dot(c / s, p)
        + wp.length(u / wp.length(v))
        + wp.dot(J2_V3.make_vec3(s, p[0], p[2]), J2_V3.constant_vec3(c) + p)
        + wp.length_sq(m * u)
    )


@wp.func
def vec2_ops_energy(q: Any, s: Any):
    # The same for vec2, plus the 2D helpers that live on the namespace.
    n = wp.vec2(0.7, 0.1)
    m = wp.mat22(0.9, -0.4, 0.3, 1.2)
    a = q * 2.0 + n
    b = n - q / 2.0 + (-q) + (+q) + 0.5 * q
    d = s * n + n * s + q * s + 0.5 * (s * q) + (n + q) - (q - n)
    return (
        wp.dot(a, n)
        + wp.dot(n, b)
        + wp.dot(a, d)
        + wp.length(b)
        + wp.length(a + b)
        + wp.length_sq(d)
        + wp.length(a - d)
        + wp.dot(wp.normalize(a), n)
        + q[-1] * a[0]
        + wp.dot(n / s, q)
        + wp.length(a / wp.length(b))
        + J2_V2.cross2(a, n)
        + J2_V2.cross2(n, b)
        + J2_V2.cross2(a, J2_V2.perp(b))
        + wp.dot(J2_V2.make_vec2(q[1], s), J2_V2.constant_vec2(n) - q)
        + wp.length_sq(m * d)
    )


def _dot(x, y):
    return (x * y).sum(-1)


def _length(x):
    return np.linalg.norm(x, axis=-1)


def vec3_ops_energy_np(z):
    p = z[:, 0:3]
    c = np.array([0.3, -0.2, 0.5])
    m = np.array([[0.9, -0.4, 0.2], [0.1, 1.1, -0.3], [-0.5, 0.6, 0.8]])
    s = 0.5 * p[:, 1:2] + 1.2
    u = 2.0 * p - c
    v = c - p / 3.0 + p * s + 0.25 * (s * p)
    w = s * c + c * s + p - u + (c + p) * 0.5 + (p + c)
    made = np.stack((s[:, 0], p[:, 0], p[:, 2]), axis=1)
    return (
        _dot(np.cross(p, c), np.cross(c, u))
        + _dot(np.cross(u, v), w)
        + _length(v)
        + _length(u - v)
        + _dot(w, w)
        + _dot(u / _length(u)[:, None], c)
        + _dot(c, w)
        + p[:, 2] * u[:, 0]
        + v[:, 1]
        + _dot(c / s, p)
        + _length(u / _length(v)[:, None])
        + _dot(made, c + p)
        + _dot(u @ m.T, u @ m.T)
    )


def vec2_ops_energy_np(z):
    q, s = z[:, 0:2], z[:, 2:3]
    n = np.array([0.7, 0.1])
    m = np.array([[0.9, -0.4], [0.3, 1.2]])

    def cross2(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    a = q * 2.0 + n
    b = n - q / 2.0 - q + q + 0.5 * q
    d = s * n + n * s + q * s + 0.5 * (s * q) + (n + q) - (q - n)
    perp_b = np.stack((-b[:, 1], b[:, 0]), axis=1)
    made = np.stack((q[:, 1], s[:, 0]), axis=1)
    return (
        _dot(a, n)
        + _dot(n, b)
        + _dot(a, d)
        + _length(b)
        + _length(a + b)
        + _dot(d, d)
        + _length(a - d)
        + _dot(a / _length(a)[:, None], n)
        + q[:, 1] * a[:, 0]
        + _dot(n / s, q)
        + _length(a / _length(b)[:, None])
        + cross2(a, n)
        + cross2(n, b)
        + cross2(a, perp_b)
        + _dot(made, n - q)
        + _dot(d @ m.T, d @ m.T)
    )


@wp.kernel
def jet2_vec3_ops(
    p: wp.array[wp.vec3],
    val: wp.array[float],
    grad: wp.array2d[float],
    hess: wp.array3d[float],
):
    i = wp.tid()
    e = vec3_ops_energy(J2_V3.seed(p[i]))
    val[i] = e.value
    for a in range(3):
        grad[i, a] = e.grad[a]
        for b in range(3):
            hess[i, a, b] = e.hess[a, b]


@wp.kernel
def jet2_vec2_ops(
    q: wp.array[wp.vec2],
    s: wp.array[float],
    val: wp.array[float],
    grad: wp.array2d[float],
    hess: wp.array3d[float],
):
    i = wp.tid()
    qj, sj = J2_V2.seed(q[i], s[i])
    e = vec2_ops_energy(qj, sj)
    val[i] = e.value
    for a in range(3):
        grad[i, a] = e.grad[a]
        for b in range(3):
            hess[i, a, b] = e.hess[a, b]


def _check_against_fd(fn_np, z_np, val, grad, hess):
    z64 = z_np.astype(np.float64)
    np.testing.assert_allclose(val.numpy(), fn_np(z64), rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(grad.numpy(), _grad_fd(fn_np, z64), rtol=1.0e-4, atol=1.0e-4)
    np.testing.assert_allclose(hess.numpy(), _hess_fd(fn_np, z64), rtol=1.0e-3, atol=1.0e-3)


def test_jet2_vector_ops(test, device):
    """Check every second-order vector operation against finite differences of a NumPy mirror."""
    z_np = np.array([[0.4, -0.3, 0.8], [-0.7, 0.2, 0.3], [0.1, 0.6, -0.5]], dtype=np.float32)
    m = z_np.shape[0]

    def outputs():
        return (
            wp.zeros(m, dtype=float, device=device),
            wp.zeros((m, 3), dtype=float, device=device),
            wp.zeros((m, 3, 3), dtype=float, device=device),
        )

    # vec3 slot: p is the three components.
    val, grad, hess = outputs()
    p = wp.array(z_np, dtype=wp.vec3, device=device)
    wp.launch(jet2_vec3_ops, dim=m, inputs=[p], outputs=[val, grad, hess], device=device)
    _check_against_fd(vec3_ops_energy_np, z_np, val, grad, hess)

    # vec2 slot then a scalar slot: q is the first two components, s the third.
    z_np[:, 2] = np.abs(z_np[:, 2]) + 0.8  # keep the divisor s away from zero
    val, grad, hess = outputs()
    q = wp.array(z_np[:, 0:2], dtype=wp.vec2, device=device)
    s = wp.array(z_np[:, 2], dtype=float, device=device)
    wp.launch(jet2_vec2_ops, dim=m, inputs=[q, s], outputs=[val, grad, hess], device=device)
    _check_against_fd(vec2_ops_energy_np, z_np, val, grad, hess)


devices = get_test_devices()


class TestJet(unittest.TestCase):
    pass


add_function_test(TestJet, "test_jet_value", test_jet_value, devices=devices)
add_function_test(TestJet, "test_jet_gradient", test_jet_gradient, devices=devices)
add_function_test(TestJet, "test_jet_hessian", test_jet_hessian, devices=devices)
add_function_test(TestJet, "test_jet_hessian_symmetric", test_jet_hessian_symmetric, devices=devices)
add_function_test(TestJet, "test_jet_spring_gradient", test_jet_spring_gradient, devices=devices)
add_function_test(TestJet, "test_jet_component_and_geometry", test_jet_component_and_geometry, devices=devices)
add_function_test(TestJet, "test_jet_multiple_widths", test_jet_multiple_widths, devices=devices)
add_function_test(TestJet, "test_jet_float64", test_jet_float64, devices=devices)
add_function_test(TestJet, "test_jet2_value_grad_hessian", test_jet2_value_grad_hessian, devices=devices)
add_function_test(TestJet, "test_jet2_hessian_symmetric", test_jet2_hessian_symmetric, devices=devices)
add_function_test(TestJet, "test_jet2_div_log_sqrt", test_jet2_div_log_sqrt, devices=devices)
add_function_test(TestJet, "test_jet_seed_mixed_slots", test_jet_seed_mixed_slots, devices=devices)
add_function_test(TestJet, "test_jet_seed_single_slot", test_jet_seed_single_slot, devices=devices)
add_function_test(TestJet, "test_jet_directional", test_jet_directional, devices=devices)
add_function_test(TestJet, "test_jet_seed_is_differentiable", test_jet_seed_is_differentiable, devices=devices)
add_function_test(
    TestJet, "test_jet_in_kernel_grad_hessian_vector", test_jet_in_kernel_grad_hessian_vector, devices=devices
)
add_function_test(TestJet, "test_jet2_seed", test_jet2_seed, devices=devices)
add_function_test(TestJet, "test_jet2_vector_spring", test_jet2_vector_spring, devices=devices)
add_function_test(TestJet, "test_jet2_vector_ops", test_jet2_vector_ops, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
