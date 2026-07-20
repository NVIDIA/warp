# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from typing import Any
from unittest.mock import patch

import warp as wp
from warp._src.autograd import FunctionMetadata
from warp.autograd import (
    gradcheck,
    gradcheck_tape,
    jacobian,
    jacobian_fd,
    jacobian_plot,
)
from warp.tests.unittest_utils import *


@wp.kernel
def kernel_3d(
    a: wp.array3d[Any],
    b: wp.array3d[Any],
    c: wp.array3d[Any],
    out1: wp.array3d[Any],
    out2: wp.array3d[Any],
):
    i, j, k = wp.tid()
    out1[i, j, k] = a[i, j, k] * b[i, j, k] + c[i, j, k]
    out2[i, j, k] = -a[i, j, k] * b[i, j, k] - c[i, j, k]


wp.overload(
    kernel_3d,
    [
        wp.array3d[wp.float32],
        wp.array3d[wp.float32],
        wp.array3d[wp.float32],
        wp.array3d[wp.float32],
        wp.array3d[wp.float32],
    ],
)

wp.overload(
    kernel_3d,
    [
        wp.array3d[wp.float64],
        wp.array3d[wp.float64],
        wp.array3d[wp.float64],
        wp.array3d[wp.float64],
        wp.array3d[wp.float64],
    ],
)


@wp.kernel
def kernel_mixed(
    a: wp.array[float],
    b: wp.array[wp.vec3],
    out1: wp.array[wp.vec2],
    out2: wp.array[wp.quat],
):
    tid = wp.tid()
    ai, bi = a[tid], b[tid]
    out1[tid] = wp.vec2(ai * wp.length(bi), -ai * wp.dot(bi, wp.vec3(0.1, 1.0, -0.1)))
    out2[tid] = wp.normalize(wp.quat(ai, bi[0], bi[1], bi[2]))


@wp.kernel
def vec_length_kernel(a: wp.array[wp.vec3], out: wp.array[float]):
    tid = wp.tid()
    v = a[tid]
    # instead of wp.length(v), we use a trivial implementation that
    # fails when a division by zero occurs in the backward pass of sqrt
    out[tid] = wp.sqrt(v[0] ** 2.0 + v[1] ** 2.0 + v[2] ** 2.0)


@wp.func
def wrong_grad_func(x: float):
    return x * x


@wp.func_grad(wrong_grad_func)
def adj_wrong_grad_func(x: float, adj: float):
    wp.adjoint[x] -= 2.0 * x * adj


@wp.kernel
def wrong_grad_kernel(a: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    out[tid] = wrong_grad_func(a[tid])


@wp.kernel
def transform_point_kernel(
    transforms: wp.array[wp.transform],
    points: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    tid = wp.tid()
    out[tid] = wp.transform_point(transforms[tid], points[tid])


@wp.kernel
def jacobian_plot_scale_kernel(a: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    out[tid] = 2.0 * a[tid]


def jacobian_plot_pipeline(a):
    out = wp.zeros_like(a, requires_grad=True)
    wp.launch(
        jacobian_plot_scale_kernel,
        dim=len(a),
        inputs=[a],
        outputs=[out],
        device=a.device,
    )
    return out


def _get_test_pyplot():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def test_gradcheck_3d(test, device, dtype):
    # Adjust tolerances based on dtype precision
    if dtype == wp.float64:
        eps = 1e-5
        atol, rtol = 1e-7, 1e-7
    else:
        eps = 1e-4
        atol, rtol = 1e-2, 1e-2

    a_3d = wp.array([((2.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    b_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    c_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    out1_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    out2_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    jacs_ad = jacobian(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_outputs_per_var=4,
        input_output_mask=[("a", "out1"), ("b", "out2")],
    )

    test.assertEqual(sorted(jacs_ad.keys()), [(0, 0), (1, 1)])
    test.assertEqual(jacs_ad[(0, 0)].shape, (6, 6))
    test.assertEqual(jacs_ad[(1, 1)].shape, (6, 6))
    # all entries beyond the max_outputs_per_var are NaN
    test.assertTrue(np.all(np.isnan(jacs_ad[(0, 0)].numpy()[4:])))
    test.assertTrue(np.all(np.isnan(jacs_ad[(1, 1)].numpy()[4:])))

    jacs_fd = jacobian_fd(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_inputs_per_var=4,
        # use integer indices instead of variable names
        input_output_mask=[(0, 0), (1, 1)],
        eps=eps,
    )

    test.assertEqual(sorted(jacs_fd.keys()), [(0, 0), (1, 1)])
    test.assertEqual(jacs_fd[(0, 0)].shape, (6, 6))
    test.assertEqual(jacs_fd[(1, 1)].shape, (6, 6))
    # all entries beyond the max_inputs_per_var are NaN
    test.assertTrue(np.all(np.isnan(jacs_fd[(0, 0)].numpy()[:, 4:])))
    test.assertTrue(np.all(np.isnan(jacs_fd[(1, 1)].numpy()[:, 4:])))

    # manual gradcheck
    np.testing.assert_allclose(jacs_ad[(0, 0)].numpy()[:4, :4], jacs_fd[(0, 0)].numpy()[:4, :4], atol=atol, rtol=rtol)
    np.testing.assert_allclose(jacs_ad[(1, 1)].numpy()[:4, :4], jacs_fd[(1, 1)].numpy()[:4, :4], atol=atol, rtol=rtol)

    passed = gradcheck(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_inputs_per_var=4,
        max_outputs_per_var=4,
        input_output_mask=[("a", "out1"), ("b", "out2")],
        show_summary=False,
    )
    test.assertTrue(
        passed,
        f"gradcheck failed for kernel_3d (dtype={dtype.__name__}, eps={eps}, atol={atol}, rtol={rtol})",
    )


def test_gradcheck_mixed(test, device):
    a = wp.array([2.0, -1.0], dtype=wp.float32, requires_grad=True, device=device)
    b = wp.array([wp.vec3(3.0, 1.0, 2.0), wp.vec3(-4.0, -1.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out1 = wp.zeros(2, dtype=wp.vec2, requires_grad=True, device=device)
    out2 = wp.zeros(2, dtype=wp.quat, requires_grad=True, device=device)

    jacs_ad = jacobian(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2])
    jacs_fd = jacobian_fd(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2], eps=1e-4)

    # manual gradcheck
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(jacs_ad[(i, j)].numpy(), jacs_fd[(i, j)].numpy(), atol=1e-2, rtol=1e-2)

    passed = gradcheck(
        kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2], raise_exception=False, show_summary=False
    )

    test.assertTrue(passed, "gradcheck failed for kernel_mixed")


def test_gradcheck_nan(test, device):
    a = wp.array([wp.vec3(1.0, 2.0, 3.0), wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out = wp.array([0.0, 0.0], dtype=float, requires_grad=True, device=device)

    with test.assertRaises(ValueError):
        gradcheck(vec_length_kernel, dim=a.shape, inputs=[a], outputs=[out], raise_exception=True, show_summary=False)


def test_gradcheck_incorrect(test, device):
    a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True, device=device)
    out = wp.zeros_like(a)

    with test.assertRaises(ValueError):
        gradcheck(wrong_grad_kernel, dim=a.shape, inputs=[a], outputs=[out], raise_exception=True, show_summary=False)


def test_gradcheck_tape_basic(test, device, dtype):
    a_3d = wp.array([((2.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    b_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    c_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    out1_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)
    out2_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=dtype, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(kernel_3d, dim=a_3d.shape, inputs=[a_3d, b_3d, c_3d], outputs=[out1_3d, out2_3d], device=device)

    passed = gradcheck_tape(tape, raise_exception=False, show_summary=False)

    test.assertTrue(passed, f"gradcheck_tape failed for kernel_3d (dtype={dtype.__name__})")


def test_gradcheck_tape_mixed(test, device):
    a = wp.array([2.0, -1.0], dtype=wp.float32, requires_grad=True, device=device)
    b = wp.array([wp.vec3(3.0, 1.0, 2.0), wp.vec3(-4.0, -1.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out1 = wp.zeros(2, dtype=wp.vec2, requires_grad=True, device=device)
    out2 = wp.zeros(2, dtype=wp.quat, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch(kernel_mixed, dim=len(a), inputs=[a, b], outputs=[out1, out2], device=device)

    passed = gradcheck_tape(tape, raise_exception=False, show_summary=False)

    test.assertTrue(passed, "gradcheck_tape failed for kernel_mixed")


devices = get_test_devices()


class TestGradDebug(unittest.TestCase):
    def test_jacobian_plot_function(self):
        """Verify Jacobian plotting supports Python function pipelines."""
        plt = _get_test_pyplot()
        self.addCleanup(plt.close, "all")

        for jacobian_function in (jacobian, jacobian_fd):
            with self.subTest(jacobian_function=jacobian_function.__name__):
                plt.close("all")
                a = wp.array([1.0, 2.0], dtype=wp.float32, requires_grad=True, device="cpu")

                with patch.object(plt, "show") as show:
                    jacobians = jacobian_function(
                        jacobian_plot_pipeline,
                        inputs=[a],
                        plot_jacobians=True,
                    )

                np.testing.assert_allclose(
                    jacobians[(0, 0)].numpy(),
                    np.eye(2, dtype=np.float32) * 2.0,
                    atol=1.0e-3,
                    rtol=1.0e-3,
                )
                figure = plt.gcf()
                self.assertEqual(figure.get_suptitle(), "jacobian_plot_pipeline kernel Jacobian")
                self.assertEqual(figure.axes[0].get_xlabel(), "a")
                self.assertEqual(figure.axes[0].get_ylabel(), "output_0")
                show.assert_called_once_with()

    def test_jacobian_plot_kernel(self):
        """Verify Jacobian plotting uses typed kernel metadata."""
        plt = _get_test_pyplot()
        self.addCleanup(plt.close, "all")

        for jacobian_function in (jacobian, jacobian_fd):
            with self.subTest(jacobian_function=jacobian_function.__name__):
                plt.close("all")
                a = wp.array([2.0, -1.0], dtype=wp.float32, requires_grad=True, device="cpu")
                b = wp.array(
                    [wp.vec3(3.0, 1.0, 2.0), wp.vec3(-4.0, -1.0, 0.0)],
                    dtype=wp.vec3,
                    requires_grad=True,
                    device="cpu",
                )
                out1 = wp.zeros(2, dtype=wp.vec2, requires_grad=True, device="cpu")
                out2 = wp.zeros(2, dtype=wp.quat, requires_grad=True, device="cpu")

                with patch.object(plt, "show") as show:
                    jacobians = jacobian_function(
                        kernel_mixed,
                        dim=len(a),
                        inputs=[a, b],
                        outputs=[out1, out2],
                        input_output_mask=[("b", "out1"), ("a", "out2")],
                        plot_jacobians=True,
                    )

                self.assertEqual(sorted(jacobians), [(0, 1), (1, 0)])
                figure = plt.gcf()
                self.assertEqual(figure.get_suptitle(), "kernel_mixed kernel Jacobian")
                self.assertEqual(len(figure.axes), 5)
                axes = figure.axes[:4]
                self.assertFalse(axes[0].axison)
                self.assertTrue(axes[1].axison)
                self.assertEqual(axes[1].get_xlabel(), "b")
                self.assertEqual(axes[1].get_ylabel(), "out1")
                self.assertTrue(axes[2].axison)
                self.assertEqual(axes[2].get_xlabel(), "a")
                self.assertEqual(axes[2].get_ylabel(), "out2")
                self.assertFalse(axes[3].axison)
                self.assertEqual(axes[0].get_gridspec().get_width_ratios(), [2, 6])
                self.assertEqual(axes[0].get_gridspec().get_height_ratios(), [4, 8])
                show.assert_called_once_with()

    def test_gradcheck_plotting(self):
        """Verify gradient checks render both error matrices without layout warnings."""
        plt = _get_test_pyplot()
        self.addCleanup(plt.close, "all")

        a = wp.array([1.0, 2.0], dtype=wp.float32, requires_grad=True, device="cpu")
        out = wp.zeros_like(a, requires_grad=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with patch.object(plt, "show") as show:
                passed = gradcheck(
                    jacobian_plot_scale_kernel,
                    dim=len(a),
                    inputs=[a],
                    outputs=[out],
                    plot_relative_error=True,
                    plot_absolute_error=True,
                    show_summary=False,
                )
            figures = [plt.figure(number) for number in plt.get_fignums()]
            for figure in figures:
                figure.canvas.draw()

        self.assertTrue(passed)
        self.assertEqual(
            [figure.get_suptitle() for figure in figures],
            [
                "jacobian_plot_scale_kernel kernel Jacobian relative error",
                "jacobian_plot_scale_kernel kernel Jacobian absolute error",
            ],
        )
        self.assertEqual(show.call_count, 2)

    def test_jacobian_plot_direct_kernel(self):
        """Verify direct kernel plotting preserves its public contract."""
        plt = _get_test_pyplot()
        self.addCleanup(plt.close, "all")

        inputs = [wp.zeros((1, 1, 1), dtype=wp.float32, requires_grad=True, device="cpu") for _ in range(3)]
        jacobian_matrix = wp.array([[1.0]], dtype=wp.float32, device="cpu")

        with patch.object(plt, "show") as show:
            figure = jacobian_plot(
                {(0, 0): jacobian_matrix},
                kernel_3d,
                inputs=inputs,
                show_plot=False,
                show_colorbar=False,
            )

        self.assertEqual(figure.get_suptitle(), "kernel_3d kernel Jacobian")
        self.assertEqual(figure.axes[0].get_xlabel(), "a")
        self.assertEqual(figure.axes[0].get_ylabel(), "out1")
        self.assertEqual(figure.axes[0].get_gridspec().get_width_ratios(), [1])
        show.assert_not_called()

    def test_jacobian_plot_validation(self):
        """Reject incomplete metadata before constructing Jacobian figures.

        A displayed Jacobian key must resolve to an input label, input array
        dtype, and output label in the normalized metadata.
        """
        plt = _get_test_pyplot()
        self.addCleanup(plt.close, "all")

        with self.assertRaisesRegex(ValueError, "inputs must be provided"):
            jacobian_plot({}, jacobian_plot_scale_kernel)

        empty_metadata = FunctionMetadata(
            key="empty",
            input_labels=[],
            output_labels=[],
            input_strides=[],
            output_strides=[],
            input_dtypes=[],
            output_dtypes=[],
        )
        self.assertIsNone(jacobian_plot({}, empty_metadata, show_plot=False))

        anonymous_metadata = FunctionMetadata(
            input_labels=["a"],
            output_labels=["output_0"],
            input_strides=[None],
            output_strides=[None],
            input_dtypes=[wp.float32],
            output_dtypes=[wp.float32],
        )
        jacobian_matrix = wp.array([[1.0]], dtype=wp.float32, device="cpu")
        figure = jacobian_plot(
            {(0, 0): jacobian_matrix},
            anonymous_metadata,
            show_plot=False,
            show_colorbar=False,
        )
        self.assertEqual(figure.get_suptitle(), "unknown kernel Jacobian")

        invalid_cases = [
            (
                "missing input label",
                FunctionMetadata(
                    key="malformed",
                    input_labels=[],
                    output_labels=["output_0"],
                    input_strides=[],
                    output_strides=[None],
                    input_dtypes=[],
                    output_dtypes=[wp.float32],
                ),
                "Jacobian input index 0",
            ),
            (
                "null input label",
                FunctionMetadata(
                    key="malformed",
                    input_labels=[None],
                    output_labels=["output_0"],
                    input_strides=[None],
                    output_strides=[None],
                    input_dtypes=[wp.float32],
                    output_dtypes=[wp.float32],
                ),
                "Jacobian input index 0: missing label",
            ),
            (
                "missing input dtype",
                FunctionMetadata(
                    key="malformed",
                    input_labels=["a"],
                    output_labels=["output_0"],
                    input_strides=[None],
                    output_strides=[None],
                    input_dtypes=[None],
                    output_dtypes=[wp.float32],
                ),
                "missing array dtype",
            ),
            (
                "generic input dtype",
                FunctionMetadata(
                    key="malformed",
                    input_labels=["a"],
                    output_labels=["output_0"],
                    input_strides=[None],
                    output_strides=[None],
                    input_dtypes=[Any],
                    output_dtypes=[wp.float32],
                ),
                "component count",
            ),
            (
                "missing output label",
                FunctionMetadata(
                    key="malformed",
                    input_labels=["a"],
                    output_labels=[],
                    input_strides=[None],
                    output_strides=[],
                    input_dtypes=[wp.float32],
                    output_dtypes=[],
                ),
                "Jacobian output index 0",
            ),
            (
                "null output label",
                FunctionMetadata(
                    key="malformed",
                    input_labels=["a"],
                    output_labels=[None],
                    input_strides=[None],
                    output_strides=[None],
                    input_dtypes=[wp.float32],
                    output_dtypes=[wp.float32],
                ),
                "Jacobian output index 0: missing label",
            ),
        ]
        for case, metadata, error in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, error):
                    jacobian_plot({(0, 0): jacobian_matrix}, metadata, show_plot=False)


for dtype in [wp.float32, wp.float64]:
    add_function_test(
        TestGradDebug, f"test_gradcheck_3d_{dtype.__name__}", test_gradcheck_3d, devices=devices, dtype=dtype
    )
    add_function_test(
        TestGradDebug,
        f"test_gradcheck_tape_basic_{dtype.__name__}",
        test_gradcheck_tape_basic,
        devices=devices,
        dtype=dtype,
    )
add_function_test(TestGradDebug, "test_gradcheck_mixed", test_gradcheck_mixed, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_nan", test_gradcheck_nan, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_incorrect", test_gradcheck_incorrect, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_tape_mixed", test_gradcheck_tape_mixed, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
