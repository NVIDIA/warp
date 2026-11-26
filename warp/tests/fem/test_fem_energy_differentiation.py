# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite-difference and jet variations of FEM energy integrands."""

import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp.tests.unittest_utils import add_function_test, get_test_devices


@fem.integrand
def scalar_energy(s: fem.Sample, u: fem.Field):
    value = u(s)
    half = fem.scalar_type(u)(0.5)
    third = fem.scalar_type(u)(1.0) / fem.scalar_type(u)(3.0)
    return half * value * value + third * value * value * value


@fem.integrand
def quadratic_energy(s: fem.Sample, u: fem.Field):
    value = u(s)
    return 0.5 * value * value


@fem.integrand
def quadratic_gradient(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def scalar_gradient(s: fem.Sample, u: fem.Field, v: fem.Field):
    value = u(s)
    return (value + value * value) * v(s)


@fem.integrand
def scalar_hessian(s: fem.Sample, u: fem.Field, v: fem.Field, w: fem.Field):
    one = fem.scalar_type(u)(1.0)
    two = fem.scalar_type(u)(2.0)
    return (one + two * u(s)) * v(s) * w(s)


@fem.integrand
def vector_gradient_energy(s: fem.Sample, u: fem.Field):
    gradient = fem.grad(u, s)
    # Jets have no ddot overload, so write the squared norm with supported matrix operations
    squared = wp.trace(wp.transpose(gradient) * gradient)
    return 0.5 * squared + 0.25 * squared * squared


@fem.integrand
def vector_gradient_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    gradient = fem.grad(u, s)
    return (1.0 + wp.ddot(gradient, gradient)) * wp.ddot(gradient, fem.grad(v, s))


@fem.integrand
def vector_gradient_hessian(s: fem.Sample, u: fem.Field, v: fem.Field, w: fem.Field):
    gradient = fem.grad(u, s)
    return (1.0 + wp.ddot(gradient, gradient)) * wp.ddot(fem.grad(v, s), fem.grad(w, s)) + 2.0 * wp.ddot(
        gradient, fem.grad(v, s)
    ) * wp.ddot(gradient, fem.grad(w, s))


@fem.integrand
def mixed_energy(s: fem.Sample, u: fem.Field, p: fem.Field):
    value = u(s)
    return value * value * p(s)


@fem.integrand
def mixed_hessian(s: fem.Sample, u: fem.Field, v: fem.Field, q: fem.Field):
    return 2.0 * u(s) * v(s) * q(s)


@fem.integrand
def lookup_energy(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    value = u(s)
    # Values at looked-up samples have no test or trial direction
    other = u(fem.lookup(domain, 0.5 * domain(s)))
    return 0.5 * value * value + value * other


@fem.integrand
def lookup_gradient(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    return (u(s) + u(fem.lookup(domain, 0.5 * domain(s)))) * v(s)


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def reserved_step_energy(s: fem.Sample, u: fem.Field, _fd_test_step: float):
    return u(s) * _fd_test_step


def _dense(matrix):
    offsets = matrix.offsets.numpy()
    columns = matrix.columns.numpy()
    values = matrix.values.numpy()
    block_rows, block_cols = matrix.block_shape
    result = np.zeros((matrix.nrow * block_rows, matrix.ncol * block_cols), dtype=values.dtype)
    for row in range(matrix.nrow):
        for offset in range(offsets[row], offsets[row + 1]):
            result[
                row * block_rows : (row + 1) * block_rows,
                columns[offset] * block_cols : (columns[offset] + 1) * block_cols,
            ] = values[offset].reshape(block_rows, block_cols)
    return result


def _has_old_nvrtc():
    """Whether the current device is a CUDA device compiled with a toolkit older than CUDA 13.2."""
    toolkit_version = wp._src.context.runtime.toolkit_version
    return wp.get_device().is_cuda and toolkit_version is not None and toolkit_version[:2] < (13, 2)


class _EnergyDifferentiationCases:
    """Test bodies shared by all devices; run through ``TestFemEnergyDifferentiation``."""

    def setUp(self):
        self.geometry = fem.Grid2D(res=wp.vec2i(1))
        self.domain = fem.Cells(geometry=self.geometry)
        self.space = fem.make_polynomial_space(self.geometry, degree=1)
        self.current = self.space.make_field()
        self.current.dof_values = np.array([0.2, 0.4, 0.8, 1.1], dtype=np.float32)
        self.test = fem.make_test(self.space, domain=self.domain)
        self.trial = fem.make_trial(self.space, domain=self.domain)
        self.fd_options = {"test_step": 0.02, "trial_step": 0.03}

    def test_scalar_energy_gradient_hessian(self):
        binding = fem.bind(self.current, test=self.test, trial=self.trial)
        actual_gradient = fem.integrate(
            scalar_energy,
            fields={"u": fem.bind(self.current, test=self.test)},
            differentiation="finite_difference",
            differentiation_options={"test_step": self.fd_options["test_step"]},
        )
        expected_gradient = fem.integrate(
            scalar_gradient, fields={"u": self.current, "v": self.test}, assembly="generic"
        )
        np.testing.assert_allclose(actual_gradient.numpy(), expected_gradient.numpy(), rtol=2.0e-3, atol=2.0e-4)

        actual_hessian = fem.integrate(
            scalar_energy,
            fields={"u": binding},
            differentiation="finite_difference",
            differentiation_options=self.fd_options,
        )
        expected_hessian = fem.integrate(
            scalar_hessian,
            fields={"u": self.current, "v": self.test, "w": self.trial},
            assembly="generic",
        )
        np.testing.assert_allclose(_dense(actual_hessian), _dense(expected_hessian), rtol=2.0e-3, atol=2.0e-4)

        again = fem.integrate(
            scalar_energy,
            fields={"u": binding},
            differentiation="finite_difference",
            differentiation_options={"test_step": 0.04, "trial_step": 0.05},
        )
        np.testing.assert_allclose(_dense(again), _dense(expected_hessian), rtol=3.0e-3, atol=3.0e-4)

        self.current.dof_values.fill_(0.5)
        updated = fem.integrate(
            scalar_energy,
            fields={"u": binding},
            differentiation="finite_difference",
            differentiation_options=self.fd_options,
        )
        self.assertGreater(np.linalg.norm(_dense(updated) - _dense(actual_hessian)), 0.01)

    def test_default_quadrature_counts_both_field_degrees(self):
        binding = fem.bind(self.current, test=self.test)
        actual = fem.integrate(
            quadratic_energy,
            fields={"u": binding},
            differentiation="finite_difference",
            differentiation_options={"test_step": self.fd_options["test_step"]},
        )
        with_unused_trial_step = fem.integrate(
            quadratic_energy,
            fields={"u": binding},
            differentiation="finite_difference",
            differentiation_options=self.fd_options,
        )
        np.testing.assert_allclose(actual.numpy(), with_unused_trial_step.numpy())
        accurate = fem.integrate(
            quadratic_gradient,
            quadrature=fem.RegularQuadrature(domain=self.domain, order=4),
            fields={"u": self.current, "v": self.test},
            assembly="generic",
        )
        np.testing.assert_allclose(actual.numpy(), accurate.numpy(), rtol=1.0e-4, atol=1.0e-5)

    def test_vector_gradient(self):
        space = fem.make_polynomial_space(self.geometry, degree=1, dtype=wp.vec2)
        current = space.make_field()
        current.dof_values = np.array([[0.1, 0.2], [0.4, -0.1], [0.3, 0.6], [0.7, 0.1]], dtype=np.float32)
        test = fem.make_test(space, domain=self.domain)
        actual = fem.integrate(
            vector_gradient_energy,
            fields={"u": fem.bind(current, test=test)},
            differentiation="finite_difference",
            differentiation_options={"test_step": self.fd_options["test_step"]},
        )
        expected = fem.integrate(vector_gradient_form, fields={"u": current, "v": test}, assembly="generic")
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=3.0e-3, atol=3.0e-4)

    def test_mixed_block(self):
        pressure_space = fem.make_polynomial_space(self.geometry, degree=2)
        pressure = pressure_space.make_field()
        pressure.dof_values.fill_(0.7)
        test_bound = fem.bind(self.current, test=self.test)
        trial = fem.make_trial(pressure_space, domain=self.domain)
        trial_bound = fem.bind(pressure, trial=trial)
        actual = fem.integrate(
            mixed_energy,
            fields={"u": test_bound, "p": trial_bound},
            differentiation="finite_difference",
            differentiation_options=self.fd_options,
        )
        expected = fem.integrate(
            mixed_hessian,
            fields={"u": self.current, "v": self.test, "q": trial},
            assembly="generic",
        )
        self.assertEqual((actual.nrow, actual.ncol), (4, 9))
        np.testing.assert_allclose(_dense(actual), _dense(expected), rtol=3.0e-3, atol=3.0e-4)

    def test_float64(self):
        geometry = fem.Grid2D(res=wp.vec2i(1), scalar_type=wp.float64)
        domain = fem.Cells(geometry)
        space = fem.make_polynomial_space(geometry, degree=1)
        current = space.make_field()
        current.dof_values = np.array([0.2, 0.4, 0.8, 1.1], dtype=np.float64)
        test = fem.make_test(space, domain=domain)
        trial = fem.make_trial(space, domain=domain)
        actual = fem.integrate(
            scalar_energy,
            fields={"u": fem.bind(current, test=test, trial=trial)},
            differentiation="finite_difference",
            differentiation_options={"test_step": 0.001, "trial_step": 0.002},
        )
        expected = fem.integrate(scalar_hessian, fields={"u": current, "v": test, "w": trial}, assembly="generic")
        np.testing.assert_allclose(_dense(actual), _dense(expected), rtol=1.0e-8, atol=1.0e-9)

        expected_residual = fem.integrate(scalar_gradient, fields={"u": current, "v": test}, assembly="generic")
        for assembly in ("generic", "dispatch"):
            with self.subTest(assembly=assembly):
                residual = fem.integrate(
                    scalar_energy,
                    fields={"u": fem.bind(current, test=test)},
                    differentiation="finite_difference",
                    differentiation_options={"test_step": 0.001},
                    assembly=assembly,
                )
                self.assertEqual(residual.dtype, wp.float64)
                np.testing.assert_allclose(residual.numpy(), expected_residual.numpy(), rtol=1.0e-6, atol=1.0e-9)

    def test_validation(self):
        binding = fem.bind(self.current, test=self.test)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            fem.integrate(
                scalar_energy,
                fields={"u": binding},
                differentiation="finite_difference",
                differentiation_options={"test_step": 0.0},
            )
        with self.assertRaisesRegex(ValueError, "positive finite"):
            fem.integrate(
                scalar_energy,
                fields={"u": binding},
                differentiation="finite_difference",
                differentiation_options={"test_step": 0.02, "trial_step": float("inf")},
            )
        with self.assertRaisesRegex(ValueError, "missing=.*trial_step"):
            fem.integrate(
                scalar_energy,
                fields={"u": fem.bind(self.current, test=self.test, trial=self.trial)},
                differentiation="finite_difference",
                differentiation_options={"test_step": 0.02},
            )
        with self.assertRaisesRegex(ValueError, "extra=.*unknown"):
            fem.integrate(
                scalar_energy,
                fields={"u": binding},
                differentiation="finite_difference",
                differentiation_options={"test_step": 0.02, "unknown": 1.0},
            )
        with self.assertRaisesRegex(TypeError, "differentiation_options dictionary"):
            fem.integrate(scalar_energy, fields={"u": binding}, differentiation="finite_difference")
        with self.assertRaisesRegex(ValueError, "Unsupported FEM differentiation method"):
            fem.integrate(scalar_energy, fields={"u": binding}, differentiation="autodiff")
        with self.assertRaisesRegex(ValueError, "requires a differentiation method"):
            fem.integrate(scalar_energy, fields={"u": binding}, differentiation_options={"test_step": 0.02})
        with self.assertRaisesRegex(TypeError, "FEM test field"):
            fem.bind(self.current, test=self.current)
        with self.assertRaisesRegex(ValueError, "explicit differentiation"):
            fem.integrate(scalar_energy, fields={"u": fem.bind(self.current, test=self.test)})
        with self.assertRaisesRegex(ValueError, "without a test field"):
            fem.integrate(
                scalar_energy,
                fields={"u": fem.bind(self.current, trial=self.trial)},
                differentiation="finite_difference",
                differentiation_options=self.fd_options,
            )
        with self.assertRaisesRegex(NotImplementedError, "generic and dispatch assembly"):
            fem.integrate(
                scalar_energy,
                fields={"u": fem.bind(self.current, test=self.test)},
                differentiation="finite_difference",
                differentiation_options=self.fd_options,
                assembly="nodal",
            )
        with self.assertRaisesRegex(ValueError, "reserved finite-difference step names"):
            fem.integrate(
                reserved_step_energy,
                fields={"u": fem.bind(self.current, test=self.test)},
                differentiation="finite_difference",
                differentiation_options=self.fd_options,
            )

    def test_jets_scalar_energy_gradient_hessian(self):
        """Match jet residuals and tangents to analytic forms, including after DOF updates."""
        binding = fem.bind(self.current, test=self.test, trial=self.trial)
        actual_gradient = fem.integrate(
            scalar_energy, fields={"u": fem.bind(self.current, test=self.test)}, differentiation="jets"
        )
        expected_gradient = fem.integrate(
            scalar_gradient, fields={"u": self.current, "v": self.test}, assembly="generic"
        )
        np.testing.assert_allclose(actual_gradient.numpy(), expected_gradient.numpy(), rtol=1.0e-5, atol=1.0e-6)

        actual_hessian = fem.integrate(scalar_energy, fields={"u": binding}, differentiation="jets")
        expected_hessian = fem.integrate(
            scalar_hessian, fields={"u": self.current, "v": self.test, "w": self.trial}, assembly="generic"
        )
        np.testing.assert_allclose(_dense(actual_hessian), _dense(expected_hessian), rtol=1.0e-5, atol=1.0e-6)

        self.current.dof_values.fill_(0.5)
        updated = fem.integrate(
            scalar_energy, fields={"u": binding}, differentiation="jets", differentiation_options={}
        )
        expected_updated = fem.integrate(
            scalar_hessian, fields={"u": self.current, "v": self.test, "w": self.trial}, assembly="generic"
        )
        np.testing.assert_allclose(_dense(updated), _dense(expected_updated), rtol=1.0e-5, atol=1.0e-6)

    def test_jets_vector_gradient(self):
        """Differentiate a vector-gradient energy with matrix-valued first-order jets."""
        space = fem.make_polynomial_space(self.geometry, degree=1, dtype=wp.vec2)
        current = space.make_field()
        current.dof_values = np.array([[0.1, 0.2], [0.4, -0.1], [0.3, 0.6], [0.7, 0.1]], dtype=np.float32)
        test = fem.make_test(space, domain=self.domain)
        actual = fem.integrate(
            vector_gradient_energy, fields={"u": fem.bind(current, test=test)}, differentiation="jets"
        )
        expected = fem.integrate(vector_gradient_form, fields={"u": current, "v": test}, assembly="generic")
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1.0e-5, atol=1.0e-6)

        trial = fem.make_trial(space, domain=self.domain)
        with self.assertRaisesRegex(
            NotImplementedError, "operator `grad` on bound field 'u' cannot be evaluated as a jet"
        ):
            fem.integrate(
                vector_gradient_energy,
                fields={"u": fem.bind(current, test=test, trial=trial)},
                differentiation="jets",
            )

    def test_jets_mixed_block(self):
        """Assemble a mixed tangent block from test and trial directions on different fields."""
        pressure_space = fem.make_polynomial_space(self.geometry, degree=2)
        pressure = pressure_space.make_field()
        pressure.dof_values.fill_(0.7)
        trial = fem.make_trial(pressure_space, domain=self.domain)
        actual = fem.integrate(
            mixed_energy,
            fields={"u": fem.bind(self.current, test=self.test), "p": fem.bind(pressure, trial=trial)},
            differentiation="jets",
        )
        expected = fem.integrate(
            mixed_hessian, fields={"u": self.current, "v": self.test, "q": trial}, assembly="generic"
        )
        self.assertEqual((actual.nrow, actual.ncol), (4, 9))
        np.testing.assert_allclose(_dense(actual), _dense(expected), rtol=1.0e-5, atol=1.0e-6)

    def test_jets_float64(self):
        """Match a float64 jet tangent to its analytic form at double precision."""
        geometry = fem.Grid2D(res=wp.vec2i(1), scalar_type=wp.float64)
        domain = fem.Cells(geometry)
        space = fem.make_polynomial_space(geometry, degree=1)
        current = space.make_field()
        current.dof_values = np.array([0.2, 0.4, 0.8, 1.1], dtype=np.float64)
        test = fem.make_test(space, domain=domain)
        trial = fem.make_trial(space, domain=domain)
        actual = fem.integrate(
            scalar_energy, fields={"u": fem.bind(current, test=test, trial=trial)}, differentiation="jets"
        )
        expected = fem.integrate(scalar_hessian, fields={"u": current, "v": test, "w": trial}, assembly="generic")
        np.testing.assert_allclose(_dense(actual), _dense(expected), rtol=1.0e-12, atol=1.0e-13)

    def test_jets_validation(self):
        """Reject jet options and nodal assembly."""
        binding = fem.bind(self.current, test=self.test)
        with self.assertRaisesRegex(ValueError, "jets differentiation does not take options"):
            fem.integrate(
                scalar_energy, fields={"u": binding}, differentiation="jets", differentiation_options={"test_step": 0.1}
            )
        with self.assertRaisesRegex(NotImplementedError, "generic and dispatch assembly"):
            fem.integrate(scalar_energy, fields={"u": binding}, differentiation="jets", assembly="nodal")

    def test_dispatch_assembly(self):
        """Match dispatched and generic assembly of quadratic-element residuals and tangents."""
        geometry = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geometry)
        space = fem.make_polynomial_space(geometry, degree=2)
        current = space.make_field()
        current.dof_values = np.linspace(0.1, 1.2, current.dof_values.shape[0]).astype(np.float32)
        test = fem.make_test(space, domain=domain)
        trial = fem.make_trial(space, domain=domain)
        expected_gradient = fem.integrate(scalar_gradient, fields={"u": current, "v": test}, assembly="generic").numpy()
        expected_hessian = _dense(
            fem.integrate(scalar_hessian, fields={"u": current, "v": test, "w": trial}, assembly="generic")
        )

        for method, options, rtol, atol in (
            ("jets", None, 1.0e-5, 1.0e-6),
            ("finite_difference", {"test_step": 0.01, "trial_step": 0.01}, 2.0e-3, 2.0e-4),
        ):
            for assembly in ("dispatch", None):
                with self.subTest(method=method, assembly=assembly):
                    gradient = fem.integrate(
                        scalar_energy,
                        fields={"u": fem.bind(current, test=test)},
                        differentiation=method,
                        differentiation_options=options,
                        assembly=assembly,
                    )
                    np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=rtol, atol=atol)
                    hessian = fem.integrate(
                        scalar_energy,
                        fields={"u": fem.bind(current, test=test, trial=trial)},
                        differentiation=method,
                        differentiation_options=options,
                        assembly=assembly,
                    )
                    np.testing.assert_allclose(_dense(hessian), expected_hessian, rtol=rtol, atol=atol)

        vector_space = fem.make_polynomial_space(geometry, degree=2, dtype=wp.vec2)
        vector_current = vector_space.make_field()
        rng = np.random.default_rng(0)
        vector_current.dof_values = rng.uniform(-0.5, 0.5, size=(vector_current.dof_values.shape[0], 2)).astype(
            np.float32
        )
        vector_test = fem.make_test(vector_space, domain=domain)
        actual = fem.integrate(
            vector_gradient_energy,
            fields={"u": fem.bind(vector_current, test=vector_test)},
            differentiation="jets",
            assembly="dispatch",
        )
        expected = fem.integrate(
            vector_gradient_form, fields={"u": vector_current, "v": vector_test}, assembly="generic"
        )
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1.0e-5, atol=1.0e-5)

        pressure_space = fem.make_polynomial_space(geometry, degree=1)
        pressure = pressure_space.make_field()
        pressure.dof_values.fill_(0.7)
        pressure_trial = fem.make_trial(pressure_space, domain=domain)
        actual = fem.integrate(
            mixed_energy,
            fields={"u": fem.bind(current, test=test), "p": fem.bind(pressure, trial=pressure_trial)},
            differentiation="jets",
            assembly="dispatch",
        )
        expected = fem.integrate(
            mixed_hessian, fields={"u": current, "v": test, "q": pressure_trial}, assembly="generic"
        )
        np.testing.assert_allclose(_dense(actual), _dense(expected), rtol=1.0e-5, atol=1.0e-6)

    def test_lookup_samples(self):
        """Give values at looked-up samples no direction under both generic and dispatch assembly."""
        geometry = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geometry)
        space = fem.make_polynomial_space(geometry, degree=2)
        current = space.make_field()
        current.dof_values = np.linspace(0.1, 1.2, current.dof_values.shape[0]).astype(np.float32)
        test = fem.make_test(space, domain=domain)
        trial = fem.make_trial(space, domain=domain)

        expected_residual = fem.integrate(lookup_gradient, fields={"u": current, "v": test}, assembly="generic")
        expected_tangent = fem.integrate(mass_form, fields={"u": trial, "v": test}, assembly="generic")

        methods = ["jets"] if _has_old_nvrtc() else ["jets", "reverse_over_forward"]
        for differentiation in methods:
            for assembly in ("generic", "dispatch"):
                with self.subTest(differentiation=differentiation, assembly=assembly):
                    residual = fem.integrate(
                        lookup_energy,
                        fields={"u": fem.bind(current, test=test)},
                        differentiation=differentiation,
                        assembly=assembly,
                    )
                    np.testing.assert_allclose(residual.numpy(), expected_residual.numpy(), rtol=1.0e-5, atol=1.0e-6)
                    tangent = fem.integrate(
                        lookup_energy,
                        fields={"u": fem.bind(current, test=test, trial=trial)},
                        differentiation=differentiation,
                        assembly=assembly,
                    )
                    np.testing.assert_allclose(_dense(tangent), _dense(expected_tangent), rtol=1.0e-5, atol=1.0e-6)

    def test_reverse_over_forward(self):
        if _has_old_nvrtc():
            # Kernels that differentiate with wp.grad intermittently fault with an illegal memory access when
            # compiled by NVRTC 12.9 or 13.0, in different kernels depending on unrelated details of the compiled
            # module. They run correctly with NVRTC 13.2.
            self.skipTest("reverse_over_forward kernels fault when compiled by NVRTC older than 13.2")
        """Assemble tangents with width-1 jets and in-kernel reverse mode, including matrix-valued operators."""
        for assembly in (None, "dispatch", "generic"):
            with self.subTest(assembly=assembly):
                self._check_reverse_over_forward(assembly)

    def _check_reverse_over_forward(self, assembly):
        geometry = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geometry)
        space = fem.make_polynomial_space(geometry, degree=2)
        current = space.make_field()
        current.dof_values = np.linspace(0.1, 1.2, current.dof_values.shape[0]).astype(np.float32)
        test = fem.make_test(space, domain=domain)
        trial = fem.make_trial(space, domain=domain)

        expected_gradient = fem.integrate(scalar_gradient, fields={"u": current, "v": test}, assembly="generic")
        # Residuals are a single forward pass and match jets
        gradient = fem.integrate(
            scalar_energy,
            fields={"u": fem.bind(current, test=test)},
            differentiation="reverse_over_forward",
            assembly=assembly,
        )
        np.testing.assert_allclose(gradient.numpy(), expected_gradient.numpy(), rtol=1.0e-5, atol=1.0e-6)

        expected_hessian = fem.integrate(
            scalar_hessian, fields={"u": current, "v": test, "w": trial}, assembly="generic"
        )
        hessian = fem.integrate(
            scalar_energy,
            fields={"u": fem.bind(current, test=test, trial=trial)},
            differentiation="reverse_over_forward",
            assembly=assembly,
        )
        np.testing.assert_allclose(_dense(hessian), _dense(expected_hessian), rtol=1.0e-5, atol=1.0e-6)

        # Tangents of a vector field's gradient, which second-order jets cannot hold
        vector_space = fem.make_polynomial_space(geometry, degree=2, dtype=wp.vec2)
        vector_current = vector_space.make_field()
        rng = np.random.default_rng(0)
        vector_current.dof_values = rng.uniform(-0.5, 0.5, size=(vector_current.dof_values.shape[0], 2)).astype(
            np.float32
        )
        vector_test = fem.make_test(vector_space, domain=domain)
        vector_trial = fem.make_trial(vector_space, domain=domain)
        # The hand-written form counts the trial degree in its default order and the binding does not
        quadrature = fem.RegularQuadrature(domain=domain, order=6)
        expected = fem.integrate(
            vector_gradient_hessian,
            quadrature=quadrature,
            fields={"u": vector_current, "v": vector_test, "w": vector_trial},
            assembly="generic",
        )
        actual = fem.integrate(
            vector_gradient_energy,
            quadrature=quadrature,
            fields={"u": fem.bind(vector_current, test=vector_test, trial=vector_trial)},
            differentiation="reverse_over_forward",
            assembly=assembly,
        )
        np.testing.assert_allclose(_dense(actual), _dense(expected), rtol=1.0e-4, atol=1.0e-5)

        # Test and trial directions on different fields
        pressure_space = fem.make_polynomial_space(geometry, degree=1)
        pressure = pressure_space.make_field()
        pressure.dof_values.fill_(0.7)
        pressure_trial = fem.make_trial(pressure_space, domain=domain)
        actual = fem.integrate(
            mixed_energy,
            fields={"u": fem.bind(current, test=test), "p": fem.bind(pressure, trial=pressure_trial)},
            differentiation="reverse_over_forward",
            assembly=assembly,
        )
        expected = fem.integrate(
            mixed_hessian, fields={"u": current, "v": test, "q": pressure_trial}, assembly="generic"
        )
        np.testing.assert_allclose(_dense(actual), _dense(expected), rtol=1.0e-5, atol=1.0e-6)

        with self.assertRaisesRegex(NotImplementedError, "generic and dispatch assembly"):
            fem.integrate(
                scalar_energy,
                fields={"u": fem.bind(current, test=test, trial=trial)},
                differentiation="reverse_over_forward",
                assembly="nodal",
            )
        with self.assertRaisesRegex(ValueError, "does not take options"):
            fem.integrate(
                scalar_energy,
                fields={"u": fem.bind(current, test=test, trial=trial)},
                differentiation="reverse_over_forward",
                assembly=assembly,
                differentiation_options={"test_step": 1.0},
            )

    def test_reverse_over_forward_gradient_arrays(self):
        """Leave the gradients of arrays that tangents read untouched, and reject tapes."""
        geometry = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geometry)
        space = fem.make_polynomial_space(geometry, degree=1)
        values = np.linspace(0.1, 1.2, space.node_count())
        current = space.make_field()
        current.dof_values = wp.array(values, dtype=float, requires_grad=True)
        test = fem.make_test(space, domain=domain)
        trial = fem.make_trial(space, domain=domain)
        binding = fem.bind(current, test=test, trial=trial)

        # The in-kernel reverse sweep would otherwise accumulate array adjoints into these gradients
        if not _has_old_nvrtc():
            pressure = space.make_field()
            pressure.dof_values = wp.full(space.node_count(), 0.7, dtype=float, requires_grad=True)
            reference_current = space.make_field()
            reference_current.dof_values = wp.array(values, dtype=float)
            reference_pressure = space.make_field()
            reference_pressure.dof_values.fill_(0.7)

            for assembly in ("dispatch", "generic"):
                with self.subTest(assembly=assembly):
                    actual = fem.integrate(
                        mixed_energy,
                        fields={"u": binding, "p": pressure},
                        differentiation="reverse_over_forward",
                        assembly=assembly,
                    )
                    expected = fem.integrate(
                        mixed_energy,
                        fields={"u": fem.bind(reference_current, test=test, trial=trial), "p": reference_pressure},
                        differentiation="reverse_over_forward",
                        assembly=assembly,
                    )
                    np.testing.assert_array_equal(_dense(actual), _dense(expected))
                    self.assertTrue(np.any(_dense(actual)))
                    self.assertFalse(np.any(current.dof_values.grad.numpy()))
                    self.assertFalse(np.any(pressure.dof_values.grad.numpy()))

        current.dof_values.requires_grad = False
        with wp.Tape():
            with self.assertRaisesRegex(RuntimeError, "cannot be recorded on a wp.Tape"):
                fem.integrate(scalar_energy, fields={"u": binding}, differentiation="reverse_over_forward")

            # Residuals are a forward jet pass, which a tape may record
            fem.integrate(
                scalar_energy, fields={"u": fem.bind(current, test=test)}, differentiation="reverse_over_forward"
            )


class TestFemEnergyDifferentiation(unittest.TestCase):
    pass


def _register_on_devices(method_name):
    def run(test, device):
        # Build the cases on the target device, since fields are created in setUp()
        case = type("Case", (_EnergyDifferentiationCases, unittest.TestCase), {"runTest": lambda self: None})()
        with wp.ScopedDevice(device):
            case.setUp()
            getattr(case, method_name)()
            wp.synchronize_device(device)

    add_function_test(TestFemEnergyDifferentiation, method_name, run, devices=get_test_devices())


for _name in dir(_EnergyDifferentiationCases):
    if _name.startswith("test_"):
        _register_on_devices(_name)


if __name__ == "__main__":
    unittest.main()
