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

import os
import shutil
import tempfile
import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def test_pow(e: float, expected: float):
    y = wp.pow(-2.0, e)

    # Since equality comparisons with NaN's are false, we have to do something manually
    if wp.isnan(expected):
        if not wp.isnan(y):
            print("Error, comparison failed")
            wp.printf("    Expected: %f\n", expected)
            wp.printf("    Actual: %f\n", y)
    else:
        wp.expect_eq(y, expected)


def test_fast_math_disabled(test, device):
    # on all systems pow() should handle negative base correctly with fast math off
    wp.set_module_options({"fast_math": False})
    wp.launch(test_pow, dim=1, inputs=[2.0, 4.0], device=device)


def test_fast_math_cuda(test, device):
    # on CUDA with --fast-math enabled taking the pow()
    # of a negative number will result in a NaN

    wp.set_module_options({"fast_math": True})
    try:
        wp.launch(test_pow, dim=1, inputs=[2.0, wp.NAN], device=device)
    finally:
        wp.set_module_options({"fast_math": False})


@wp.kernel
def approx_div_kernel(a: wp.array(dtype=Any), b: wp.array(dtype=Any), out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = wp.div_approx(a[i], b[i])


# Pre-instantiate overloads to avoid module recompilation during tests
for scalar_type in (wp.float32, wp.float64):
    wp.overload(
        approx_div_kernel, [wp.array(dtype=scalar_type), wp.array(dtype=scalar_type), wp.array(dtype=scalar_type)]
    )


@wp.kernel
def approx_div_compound_kernel(a: wp.array(dtype=Any), b: wp.array(dtype=Any), out: wp.array(dtype=Any)):
    i = wp.tid()
    out[i] = wp.div_approx(a[i], b[i])


for _compound_type, _scalar_type in [
    (wp.vec3, wp.float32),
    (wp.mat22, wp.float32),
    (wp.quatf, wp.float32),
]:
    wp.overload(
        approx_div_compound_kernel,
        [wp.array(dtype=_compound_type), wp.array(dtype=_scalar_type), wp.array(dtype=_compound_type)],
    )


@wp.kernel
def approx_inverse_kernel(
    m: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    i = wp.tid()
    out[i] = wp.inverse_approx(m[i])


# Pre-instantiate overloads to avoid module recompilation during tests
for mat_type in (wp.mat22, wp.mat33, wp.mat44, wp.mat22d, wp.mat33d, wp.mat44d):
    wp.overload(approx_inverse_kernel, [wp.array(dtype=mat_type), wp.array(dtype=mat_type)])


@wp.kernel
def approx_inverse_backward_kernel_22(m: wp.array(dtype=wp.mat22), loss: wp.array(dtype=float)):
    i = wp.tid()
    inv = wp.inverse_approx(m[i])
    for r in range(2):
        for c in range(2):
            wp.atomic_add(loss, 0, inv[r, c])


@wp.kernel
def approx_inverse_backward_kernel_33(m: wp.array(dtype=wp.mat33), loss: wp.array(dtype=float)):
    i = wp.tid()
    inv = wp.inverse_approx(m[i])
    for r in range(3):
        for c in range(3):
            wp.atomic_add(loss, 0, inv[r, c])


@wp.kernel
def approx_inverse_backward_kernel_44(m: wp.array(dtype=wp.mat44), loss: wp.array(dtype=float)):
    i = wp.tid()
    inv = wp.inverse_approx(m[i])
    for r in range(4):
        for c in range(4):
            wp.atomic_add(loss, 0, inv[r, c])


@wp.kernel
def approx_div_backward_kernel(
    a: wp.array(dtype=Any), b: wp.array(dtype=Any), out: wp.array(dtype=Any), loss: wp.array(dtype=Any)
):
    i = wp.tid()
    out[i] = wp.div_approx(a[i], b[i])
    wp.atomic_add(loss, 0, out[i])


for _backward_scalar_type in (wp.float32, wp.float64):
    wp.overload(approx_div_backward_kernel, [wp.array(dtype=_backward_scalar_type)] * 4)


def test_approx_div_div(test, device):
    """Test that scalar division with approx=True produces approximately correct results."""
    n = 64
    rng = np.random.default_rng(42)
    for np_dtype, wp_dtype, rtol in [(np.float32, wp.float32, 1e-6), (np.float64, wp.float64, 1e-6)]:
        a_np = rng.uniform(0.1, 100.0, n).astype(np_dtype)
        b_np = rng.uniform(0.1, 100.0, n).astype(np_dtype)
        expected = a_np / b_np

        a_wp = wp.array(a_np, dtype=wp_dtype, device=device)
        b_wp = wp.array(b_np, dtype=wp_dtype, device=device)
        out_wp = wp.zeros(n, dtype=wp_dtype, device=device)

        wp.launch(approx_div_kernel, dim=n, inputs=[a_wp, b_wp, out_wp], device=device)
        np.testing.assert_allclose(out_wp.numpy(), expected, rtol=rtol)


def test_approx_div_inverse(test, device):
    """Test that matrix inverse with approx=True produces approximately correct results."""
    rng = np.random.default_rng(42)
    n = 4
    for np_dtype, mat_types in [
        (np.float32, [(wp.mat22, 2, 1e-4, 1e-6), (wp.mat33, 3, 1e-4, 1e-6), (wp.mat44, 4, 1e-3, 1e-5)]),
        (np.float64, [(wp.mat22d, 2, 1e-4, 1e-10), (wp.mat33d, 3, 1e-4, 1e-10), (wp.mat44d, 4, 1e-3, 1e-10)]),
    ]:
        for mat_type, dim, rtol, atol in mat_types:
            m_np = rng.uniform(1.0, 5.0, (n, dim, dim)).astype(np_dtype)
            expected = np.linalg.inv(m_np)
            m_wp = wp.array(m_np, dtype=mat_type, device=device)
            out_wp = wp.zeros(n, dtype=mat_type, device=device)
            wp.launch(approx_inverse_kernel, dim=n, inputs=[m_wp, out_wp], device=device)
            np.testing.assert_allclose(out_wp.numpy(), expected, rtol=rtol, atol=atol)


def test_approx_div_ptx_verification(test, device):
    """Verify that generated PTX contains approximate division/reciprocal instructions."""
    tmpdir = tempfile.mkdtemp()
    try:
        module = wp.get_module(__name__)
        module._compile(device=device, output_dir=tmpdir, use_ptx=True)

        ptx_files = [f for f in os.listdir(tmpdir) if f.endswith(".ptx")]
        test.assertTrue(len(ptx_files) > 0, "No PTX file generated")

        ptx_path = os.path.join(tmpdir, ptx_files[0])
        with open(ptx_path) as f:
            ptx_content = f.read()

        has_approx = "div.approx" in ptx_content or "rcp.approx" in ptx_content
        test.assertTrue(has_approx, "Expected div.approx or rcp.approx in PTX output")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_approx_div_compound(test, device):
    """Test that div_approx works with vector, matrix, and quaternion operands."""
    n = 16
    rng = np.random.default_rng(42)
    for compound_type, scalar_type, np_dtype, shape in [
        (wp.vec3, wp.float32, np.float32, (n, 3)),
        (wp.mat22, wp.float32, np.float32, (n, 2, 2)),
        (wp.quatf, wp.float32, np.float32, (n, 4)),
    ]:
        a_np = rng.uniform(0.1, 10.0, shape).astype(np_dtype)
        b_np = rng.uniform(0.1, 10.0, n).astype(np_dtype)
        expected = a_np / b_np.reshape(-1, *([1] * (len(shape) - 1)))

        a_wp = wp.array(a_np, dtype=compound_type, device=device)
        b_wp = wp.array(b_np, dtype=scalar_type, device=device)
        out_wp = wp.zeros(n, dtype=compound_type, device=device)

        wp.launch(approx_div_compound_kernel, dim=n, inputs=[a_wp, b_wp, out_wp], device=device)
        np.testing.assert_allclose(out_wp.numpy(), expected, rtol=1e-6)


def test_approx_div_backward(test, device):
    """Verify that backward pass through approx division works correctly."""
    n = 8
    rng = np.random.default_rng(42)
    # div_approx uses rcp.approx (float32 precision) even for float64, so use looser tolerance for f64
    for np_dtype, wp_dtype, rtol in [(np.float32, wp.float32, 1e-6), (np.float64, wp.float64, 1e-5)]:
        a_np = rng.uniform(1.0, 10.0, n).astype(np_dtype)
        b_np = rng.uniform(1.0, 10.0, n).astype(np_dtype)

        a_wp = wp.array(a_np, dtype=wp_dtype, device=device, requires_grad=True)
        b_wp = wp.array(b_np, dtype=wp_dtype, device=device, requires_grad=True)
        out_wp = wp.zeros(n, dtype=wp_dtype, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp_dtype, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(approx_div_backward_kernel, dim=n, inputs=[a_wp, b_wp, out_wp, loss], device=device)

        tape.backward(loss=loss)

        grad_a = a_wp.grad.numpy()
        grad_b = b_wp.grad.numpy()

        # d/da (a/b) = 1/b, d/db (a/b) = -a/b^2
        expected_grad_a = 1.0 / b_np
        expected_grad_b = -a_np / (b_np * b_np)

        np.testing.assert_allclose(grad_a, expected_grad_a, rtol=rtol)
        np.testing.assert_allclose(grad_b, expected_grad_b, rtol=rtol)


def test_approx_inverse_backward(test, device):
    """Verify that backward pass through inverse_approx works correctly."""
    n = 4
    rng = np.random.default_rng(42)
    backward_kernels = {
        wp.mat22: (2, approx_inverse_backward_kernel_22),
        wp.mat33: (3, approx_inverse_backward_kernel_33),
        wp.mat44: (4, approx_inverse_backward_kernel_44),
    }
    for mat_type, (dim, kernel) in backward_kernels.items():
        m_np = rng.uniform(1.0, 5.0, (n, dim, dim)).astype(np.float32)
        m_wp = wp.array(m_np, dtype=mat_type, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=n, inputs=[m_wp, loss], device=device)

        tape.backward(loss=loss)

        grad_m = m_wp.grad.numpy()

        # d(sum(A^{-1}))/dA = -(A^{-1} @ J @ A^{-1})^T where J is all-ones
        inv_np = np.linalg.inv(m_np)
        ones = np.ones((dim, dim), dtype=np.float32)
        expected_grad = np.zeros_like(m_np)
        for i in range(n):
            expected_grad[i] = -(inv_np[i] @ ones @ inv_np[i]).T

        np.testing.assert_allclose(grad_m, expected_grad, rtol=1e-4)


class TestFastMath(unittest.TestCase):
    def test_fast_math_cpu(self):
        # on all systems pow() should handle negative base correctly
        wp.set_module_options({"fast_math": True})
        try:
            wp.launch(test_pow, dim=1, inputs=[2.0, 4.0], device="cpu")
        finally:
            wp.set_module_options({"fast_math": False})

    def test_approx_div_cpu_fallback(self):
        """Verify that on CPU, div_approx produces exact results (CPU fallback)."""
        rng = np.random.default_rng(42)
        n = 64
        a_np = rng.uniform(0.1, 100.0, n).astype(np.float32)
        b_np = rng.uniform(0.1, 100.0, n).astype(np.float32)
        expected = a_np / b_np

        a_wp = wp.array(a_np, dtype=float, device="cpu")
        b_wp = wp.array(b_np, dtype=float, device="cpu")
        out_wp = wp.zeros(n, dtype=float, device="cpu")

        wp.launch(approx_div_kernel, dim=n, inputs=[a_wp, b_wp, out_wp], device="cpu")
        np.testing.assert_array_equal(out_wp.numpy(), expected)

    def test_approx_inverse_cpu_fallback(self):
        """Verify that on CPU, inverse_approx falls back to standard inverse."""
        rng = np.random.default_rng(42)
        n = 4
        m_np = rng.uniform(1.0, 5.0, (n, 3, 3)).astype(np.float32)
        expected = np.linalg.inv(m_np)

        m_wp = wp.array(m_np, dtype=wp.mat33, device="cpu")
        out_wp = wp.zeros(n, dtype=wp.mat33, device="cpu")

        wp.launch(approx_inverse_kernel, dim=n, inputs=[m_wp, out_wp], device="cpu")
        np.testing.assert_allclose(out_wp.numpy(), expected, rtol=1e-5)


devices = get_test_devices()

add_function_test(TestFastMath, "test_fast_math_cuda", test_fast_math_cuda, devices=get_cuda_test_devices())
add_function_test(TestFastMath, "test_fast_math_disabled", test_fast_math_disabled, devices=devices)
add_function_test(TestFastMath, "test_approx_div_div", test_approx_div_div, devices=get_cuda_test_devices())
add_function_test(TestFastMath, "test_approx_div_inverse", test_approx_div_inverse, devices=get_cuda_test_devices())
add_function_test(
    TestFastMath, "test_approx_div_ptx_verification", test_approx_div_ptx_verification, devices=get_cuda_test_devices()
)
add_function_test(TestFastMath, "test_approx_div_compound", test_approx_div_compound, devices=get_cuda_test_devices())
add_function_test(TestFastMath, "test_approx_div_backward", test_approx_div_backward, devices=get_cuda_test_devices())
add_function_test(
    TestFastMath, "test_approx_inverse_backward", test_approx_inverse_backward, devices=get_cuda_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
