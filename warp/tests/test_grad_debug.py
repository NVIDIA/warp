# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import warp as wp
from warp.autograd import gradcheck, gradcheck_tape, jacobian, jacobian_fd
from warp.tests.unittest_utils import *


@wp.kernel
def kernel_3d(
    a: wp.array(dtype=float, ndim=3),
    b: wp.array(dtype=float, ndim=3),
    c: wp.array(dtype=float, ndim=3),
    out1: wp.array(dtype=float, ndim=3),
    out2: wp.array(dtype=float, ndim=3),
):
    i, j, k = wp.tid()
    out1[i, j, k] = a[i, j, k] * b[i, j, k] + c[i, j, k]
    out2[i, j, k] = -a[i, j, k] * b[i, j, k] - c[i, j, k]


@wp.kernel
def kernel_mixed(
    a: wp.array(dtype=float),
    b: wp.array(dtype=wp.vec3),
    out1: wp.array(dtype=wp.vec2),
    out2: wp.array(dtype=wp.quat),
):
    tid = wp.tid()
    ai, bi = a[tid], b[tid]
    out1[tid] = wp.vec2(ai * wp.length(bi), -ai * wp.dot(bi, wp.vec3(0.1, 1.0, -0.1)))
    out2[tid] = wp.normalize(wp.quat(ai, bi[0], bi[1], bi[2]))


@wp.kernel
def vec_length_kernel(a: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    v = a[tid]
    # instead of wp.length(v), we use a trivial implementation that
    # fails when a division by zero is occurs in the backward pass of sqrt
    out[tid] = wp.sqrt(v[0] ** 2.0 + v[1] ** 2.0 + v[2] ** 2.0)


@wp.func
def wrong_grad_func(x: float):
    return x * x


@wp.func_grad(wrong_grad_func)
def adj_wrong_grad_func(x: float, adj: float):
    wp.adjoint[x] -= 2.0 * x * adj


@wp.kernel
def wrong_grad_kernel(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wrong_grad_func(a[tid])


def test_gradcheck_3d(test, device):
    a_3d = wp.array([((2.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=float, requires_grad=True, device=device)
    b_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=float, requires_grad=True, device=device)
    c_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=float, requires_grad=True, device=device)

    out1_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=float, requires_grad=True, device=device)
    out2_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=float, requires_grad=True, device=device)

    jacs_ad = jacobian(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_outputs_per_var=4,
        input_output_mask=[("a", "out1"), ("b", "out2")],
    )

    assert sorted(jacs_ad.keys()) == [(0, 0), (1, 1)]
    assert jacs_ad[(0, 0)].shape == (6, 6)
    assert jacs_ad[(1, 1)].shape == (6, 6)
    # all entries beyond the max_outputs_per_var are NaN
    assert np.all(np.isnan(jacs_ad[(0, 0)].numpy()[4:]))
    assert np.all(np.isnan(jacs_ad[(1, 1)].numpy()[4:]))

    jacs_fd = jacobian_fd(
        kernel_3d,
        dim=a_3d.shape,
        inputs=[a_3d, b_3d, c_3d],
        outputs=[out1_3d, out2_3d],
        max_inputs_per_var=4,
        input_output_mask=[("a", "out1"), ("b", "out2")],
        eps=1e-4,
    )

    assert sorted(jacs_fd.keys()) == [(0, 0), (1, 1)]
    assert jacs_fd[(0, 0)].shape == (6, 6)
    assert jacs_fd[(1, 1)].shape == (6, 6)
    # all entries beyond the max_inputs_per_var are NaN
    assert np.all(np.isnan(jacs_fd[(0, 0)].numpy()[:, 4:]))
    assert np.all(np.isnan(jacs_fd[(1, 1)].numpy()[:, 4:]))

    # manual gradcheck
    assert np.allclose(jacs_ad[(0, 0)].numpy()[:4, :4], jacs_fd[(0, 0)].numpy()[:4, :4], atol=1e-2, rtol=1e-2)
    assert np.allclose(jacs_ad[(1, 1)].numpy()[:4, :4], jacs_fd[(1, 1)].numpy()[:4, :4], atol=1e-2, rtol=1e-2)

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
    assert passed


def test_gradcheck_mixed(test, device):
    a = wp.array([2.0, -1.0], dtype=wp.float32, requires_grad=True, device=device)
    b = wp.array([wp.vec3(3.0, 1.0, 2.0), wp.vec3(-4.0, -1.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out1 = wp.zeros(2, dtype=wp.vec2, requires_grad=True, device=device)
    out2 = wp.zeros(2, dtype=wp.quat, requires_grad=True, device=device)

    jacs_ad = jacobian(
        kernel_mixed,
        dim=len(a),
        inputs=[a, b],
        outputs=[out1, out2],
    )
    jacs_fd = jacobian_fd(
        kernel_mixed,
        dim=len(a),
        inputs=[a, b],
        outputs=[out1, out2],
        eps=1e-4,
    )

    # manual gradcheck
    for i in range(2):
        for j in range(2):
            assert np.allclose(jacs_ad[(i, j)].numpy(), jacs_fd[(i, j)].numpy(), atol=1e-2, rtol=1e-2)

    passed = gradcheck(
        kernel_mixed,
        dim=len(a),
        inputs=[a, b],
        outputs=[out1, out2],
        raise_exception=False,
        show_summary=False,
    )

    assert passed


def test_gradcheck_nan(test, device):
    a = wp.array([wp.vec3(1.0, 2.0, 3.0), wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out = wp.array([0.0, 0.0], dtype=float, requires_grad=True, device=device)

    with test.assertRaises(ValueError):
        gradcheck(
            vec_length_kernel,
            dim=a.shape,
            inputs=[a],
            outputs=[out],
            raise_exception=True,
            show_summary=False,
        )


def test_gradcheck_incorrect(test, device):
    a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True, device=device)
    out = wp.zeros_like(a)

    with test.assertRaises(ValueError):
        gradcheck(
            wrong_grad_kernel,
            dim=a.shape,
            inputs=[a],
            outputs=[out],
            raise_exception=True,
            show_summary=False,
        )


def test_gradcheck_tape(test, device):
    a = wp.array([2.0, -1.0], dtype=wp.float32, requires_grad=True, device=device)
    b = wp.array([wp.vec3(3.0, 1.0, 2.0), wp.vec3(-4.0, -1.0, 0.0)], dtype=wp.vec3, requires_grad=True, device=device)
    out1 = wp.zeros(2, dtype=wp.vec2, requires_grad=True, device=device)
    out2 = wp.zeros(2, dtype=wp.quat, requires_grad=True, device=device)

    a_3d = wp.array([((2.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=float, requires_grad=True, device=device)
    b_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=float, requires_grad=True, device=device)
    c_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=float, requires_grad=True, device=device)

    out1_3d = wp.array([((3.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=float, requires_grad=True, device=device)
    out2_3d = wp.array([((4.0, 0.0), (1.0, 0.0), (2.0, 0.0))], dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel_mixed,
            dim=len(a),
            inputs=[a, b],
            outputs=[out1, out2],
            device=device,
        )

        wp.launch(
            kernel_3d,
            dim=a_3d.shape,
            inputs=[a_3d, b_3d, c_3d],
            outputs=[out1_3d, out2_3d],
            device=device,
        )

    passed = gradcheck_tape(
        tape,
        raise_exception=False,
        show_summary=False,
    )

    assert passed


devices = get_test_devices()


class TestGradDebug(unittest.TestCase):
    pass


add_function_test(TestGradDebug, "test_gradcheck_3d", test_gradcheck_3d, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_mixed", test_gradcheck_mixed, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_nan", test_gradcheck_nan, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_incorrect", test_gradcheck_incorrect, devices=devices)
add_function_test(TestGradDebug, "test_gradcheck_tape", test_gradcheck_tape, devices=devices)


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
