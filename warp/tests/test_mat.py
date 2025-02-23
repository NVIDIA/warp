# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

np_signed_int_types = [np.int8, np.int16, np.int32, np.int64, np.byte]
np_float_types = [np.float16, np.float32, np.float64]


def randvals(rng, shape, dtype):
    if dtype in np_float_types:
        return rng.standard_normal(size=shape).astype(dtype)
    elif dtype in [np.int8, np.uint8, np.byte, np.ubyte]:
        return rng.integers(1, high=3, size=shape, dtype=dtype)
    return rng.integers(1, high=5, size=shape, dtype=dtype)


kernel_cache = {}


def getkernel(func, suffix=""):
    key = func.__name__ + "_" + suffix
    if key not in kernel_cache:
        kernel_cache[key] = wp.Kernel(func=func, key=key)
    return kernel_cache[key]


def get_select_kernel(dtype):
    def output_select_kernel_fn(input: wp.array(dtype=dtype), index: int, out: wp.array(dtype=dtype)):
        out[0] = input[index]

    return getkernel(output_select_kernel_fn, suffix=dtype.__name__)


def test_anon_constructor_error_shape_arg_missing(test, device):
    @wp.kernel
    def kernel():
        wp.matrix(1.0, 2.0, 3.0)

    with test.assertRaisesRegex(
        RuntimeError,
        r"the `shape` argument must be specified when initializing a matrix by value$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_anon_constructor_error_shape_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.matrix(wp.matrix(shape=(1, 2), dtype=float), shape=(3, 4), dtype=float)

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible matrix of shape \(3, 4\) given when copy constructing a matrix of shape \(1, 2\)$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_anon_constructor_error_type_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.matrix(1.0, shape=(3, 2), dtype=wp.float16)

    with test.assertRaisesRegex(
        RuntimeError,
        r"the value used to fill this matrix is expected to be of the type `float16`$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_anon_constructor_error_invalid_arg_count(test, device):
    @wp.kernel
    def kernel():
        wp.matrix(1.0, 2.0, 3.0, shape=(2, 2), dtype=float)

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible number of values given \(3\) when constructing a matrix of shape \(2, 2\)$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_anon_xform_constructor_error_type_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.matrix(wp.vec3(1.0, 2.0, 3.0), wp.quat(0.0, 0.0, 0.0, 1.0), wp.vec3(2.0, 2.0, 2.0), wp.float64)

    with test.assertRaisesRegex(
        RuntimeError,
        r"all values used to initialize this transformation matrix are expected to be of the type `float64`$",
    ):
        wp.launch(
            kernel,
            dim=1,
            inputs=[],
            device=device,
        )


def test_tpl_constructor_error_incompatible_sizes(test, device):
    @wp.kernel
    def kernel():
        wp.mat33(wp.mat22(1.0, 2.0, 3.0, 4.0))

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible matrix of shape \(3, 3\) given when copy constructing a matrix of shape \(2, 2\)$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_tpl_constructor_error_invalid_vector_count(test, device):
    @wp.kernel
    def kernel():
        wp.mat33(wp.vec3(1.0, 2.0, 3.0), wp.vec3(1.0, 2.0, 3.0))

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible number of column vectors given \(2\) when constructing a matrix of shape \(3, 3\)$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_tpl_constructor_error_invalid_vector_shape(test, device):
    @wp.kernel
    def kernel():
        wp.mat22(wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0))

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible column vector lengths given when constructing a matrix of shape \(2, 2\)$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_tpl_constructor_error_invalid_arg_count(test, device):
    @wp.kernel
    def kernel():
        wp.mat22(1.0, 2.0, 3.0)

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible number of values given \(3\) when constructing a matrix of shape \(2, 2\)$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_py_arithmetic_ops(test, device, dtype):
    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]

    def make_mat(*args):
        if wptype in wp.types.int_types:
            # Cast to the correct integer type to simulate wrapping.
            return tuple(tuple(wptype._type_(x).value for x in row) for row in args)

        return args

    def make_vec(*args):
        if wptype in wp.types.int_types:
            # Cast to the correct integer type to simulate wrapping.
            return tuple(wptype._type_(x).value for x in args)

        return args

    mat_cls = wp.mat((3, 3), wptype)
    vec_cls = wp.vec(3, wptype)

    m = mat_cls(((-1, 2, 3), (4, -5, 6), (7, 8, -9)))
    test.assertSequenceEqual(+m, make_mat((-1, 2, 3), (4, -5, 6), (7, 8, -9)))
    test.assertSequenceEqual(-m, make_mat((1, -2, -3), (-4, 5, -6), (-7, -8, 9)))
    test.assertSequenceEqual(m + mat_cls((5, 5, 5) * 3), make_mat((4, 7, 8), (9, 0, 11), (12, 13, -4)))
    test.assertSequenceEqual(m - mat_cls((5, 5, 5) * 3), make_mat((-6, -3, -2), (-1, -10, 1), (2, 3, -14)))
    test.assertSequenceEqual(m * vec_cls(5, 5, 5), make_vec(20, 25, 30))
    test.assertSequenceEqual(m @ vec_cls(5, 5, 5), make_vec(20, 25, 30))
    test.assertSequenceEqual(vec_cls(5, 5, 5) * m, make_vec(50, 25, 0))
    test.assertSequenceEqual(vec_cls(5, 5, 5) @ m, make_vec(50, 25, 0))

    m = mat_cls(((2, 4, 6), (8, 10, 12), (14, 16, 18)))
    test.assertSequenceEqual(m * wptype(2), make_mat((4, 8, 12), (16, 20, 24), (28, 32, 36)))
    test.assertSequenceEqual(wptype(2) * m, make_mat((4, 8, 12), (16, 20, 24), (28, 32, 36)))
    test.assertSequenceEqual(m / wptype(2), make_mat((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    test.assertSequenceEqual(wptype(5040) / m, make_mat((2520, 1260, 840), (630, 504, 420), (360, 315, 280)))
    test.assertSequenceEqual(m * vec_cls(5, 5, 5), make_vec(60, 150, 240))
    test.assertSequenceEqual(m @ vec_cls(5, 5, 5), make_vec(60, 150, 240))
    test.assertSequenceEqual(vec_cls(5, 5, 5) * m, make_vec(120, 150, 180))
    test.assertSequenceEqual(vec_cls(5, 5, 5) @ m, make_vec(120, 150, 180))


def test_quat_constructor(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    vec4 = wp.types.vector(length=4, dtype=wptype)
    vec3 = wp.types.vector(length=3, dtype=wptype)
    quat = wp.types.quaternion(dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_quat_constructor(
        p: wp.array(dtype=vec3),
        r: wp.array(dtype=quat),
        s: wp.array(dtype=vec3),
        outcomponents: wp.array(dtype=wptype),
        outcomponents_alt: wp.array(dtype=wptype),
    ):
        m = mat44(p[0], r[0], s[0])

        R = wp.transpose(wp.quat_to_matrix(r[0]))
        c0 = s[0][0] * R[0]
        c1 = s[0][1] * R[1]
        c2 = s[0][2] * R[2]
        m_alt = mat44(
            vec4(c0[0], c0[1], c0[2], wptype(0.0)),
            vec4(c1[0], c1[1], c1[2], wptype(0.0)),
            vec4(c2[0], c2[1], c2[2], wptype(0.0)),
            vec4(p[0][0], p[0][1], p[0][2], wptype(1.0)),
        )

        idx = 0
        for i in range(4):
            for j in range(4):
                outcomponents[idx] = m[i, j]
                outcomponents_alt[idx] = m_alt[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_quat_constructor, suffix=dtype.__name__)

    if register_kernels:
        return

    # translation:
    p = wp.array(rng.standard_normal(size=(1, 3)).astype(dtype), dtype=vec3, requires_grad=True, device=device)

    # generate a normalized quaternion for the rotation:
    r = rng.standard_normal(size=(1, 4))
    r /= np.linalg.norm(r)
    r = wp.array(r.astype(dtype), dtype=quat, requires_grad=True, device=device)

    # scale:
    s = wp.array(rng.standard_normal(size=(1, 3)).astype(dtype), dtype=vec3, requires_grad=True, device=device)

    # just going to generate the matrix using the constructor, then
    # more manually, and make sure the values/gradients are the same:
    outcomponents = wp.zeros(4 * 4, dtype=wptype, requires_grad=True, device=device)
    outcomponents_alt = wp.zeros(4 * 4, dtype=wptype, requires_grad=True, device=device)
    wp.launch(kernel, dim=1, inputs=[p, r, s], outputs=[outcomponents, outcomponents_alt], device=device)
    assert_np_equal(outcomponents.numpy(), outcomponents_alt.numpy(), tol=1.0e-6)

    idx = 0
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    out_alt = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    for _i in range(4):
        for _j in range(4):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[p, r, s], outputs=[outcomponents, outcomponents_alt], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                wp.launch(
                    output_select_kernel, dim=1, inputs=[outcomponents_alt, idx], outputs=[out_alt], device=device
                )

            tape.backward(loss=out)
            p_grad = 1.0 * tape.gradients[p].numpy()[0]
            r_grad = 1.0 * tape.gradients[r].numpy()[0]
            s_grad = 1.0 * tape.gradients[s].numpy()[0]
            tape.zero()

            tape.backward(loss=out_alt)
            p_grad_alt = 1.0 * tape.gradients[p].numpy()[0]
            r_grad_alt = 1.0 * tape.gradients[r].numpy()[0]
            s_grad_alt = 1.0 * tape.gradients[s].numpy()[0]
            tape.zero()

            assert_np_equal(p_grad, p_grad_alt, tol=tol)
            assert_np_equal(r_grad, r_grad_alt, tol=tol)
            assert_np_equal(s_grad, s_grad_alt, tol=tol)

            idx = idx + 1


def test_negation(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-2,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_negation(
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        m5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        mat2 = -m2[0]
        mat3 = -m3[0]
        mat4 = -m4[0]
        mat5 = -m5[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * mat2[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * mat3[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * mat4[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * mat5[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_negation, suffix=dtype.__name__)

    if register_kernels:
        return

    m2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    m3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    m5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m2, m3, m4, m5], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], -2 * m2.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], -2 * m3.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], -2 * m4.numpy().reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], -2 * m5.numpy().reshape(-1), tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, input in [(2, m2), (3, m3), (4, m4), (5, m5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[m2, m3, m4, m5], outputs=[outcomponents], device=device)
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    expectedresult = np.zeros((dim, dim), dtype=dtype)
                    expectedresult[i, j] = -2
                    assert_np_equal(tape.gradients[input].numpy()[0], expectedresult)
                    tape.zero()
                    idx = idx + 1


def test_matmul(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-12,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat23 = wp.types.matrix(shape=(2, 3), dtype=wptype)
    mat32 = wp.types.matrix(shape=(3, 2), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_mul(
        i23: wp.array(dtype=mat23),
        i32: wp.array(dtype=mat32),
        i44: wp.array(dtype=mat44),
        o22: wp.array(dtype=mat22),
        o33: wp.array(dtype=mat33),
        o44: wp.array(dtype=mat44),
    ):
        i = wp.tid()
        o22[i] = i23[i] @ i32[i]
        o33[i] = i32[i] @ i23[i]
        o44[i] = i44[i] @ i44[i]

    kernel = getkernel(check_mat_mul, suffix=dtype.__name__)

    if register_kernels:
        return

    test_adj = dtype in np_float_types

    i23 = wp.array(randvals(rng, [1, 2, 3], dtype), dtype=mat23, requires_grad=test_adj, device=device)
    i32 = wp.array(randvals(rng, [1, 3, 2], dtype), dtype=mat32, requires_grad=test_adj, device=device)
    i44 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=test_adj, device=device)
    o22 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=test_adj, device=device)
    o33 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=test_adj, device=device)
    o44 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=test_adj, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel,
            dim=1,
            inputs=[i23, i32, i44],
            outputs=[o22, o33, o44],
            device=device,
        )

    assert_np_equal(o22.numpy(), i23.numpy() @ i32.numpy(), tol=tol)
    assert_np_equal(o33.numpy(), i32.numpy() @ i23.numpy(), tol=tol)
    assert_np_equal(o44.numpy(), i44.numpy() @ i44.numpy(), tol=tol)

    if test_adj:
        o22.grad.assign([np.eye(2)])
        o33.grad.assign([np.eye(3)])
        o44.grad.assign([np.eye(4)])

        tape.backward()

        assert_np_equal(i23.grad.numpy(), 2.0 * i32.numpy().T, tol=tol)
        assert_np_equal(i32.grad.numpy(), 2.0 * i23.numpy().T, tol=tol)
        assert_np_equal(i44.grad.numpy(), 2.0 * i44.numpy().T, tol=tol)


def test_subtraction(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)
    mat55 = wp.types.matrix(shape=(5, 5), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_sub(
        s2: wp.array(dtype=mat22),
        s3: wp.array(dtype=mat33),
        s4: wp.array(dtype=mat44),
        s5: wp.array(dtype=mat55),
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        v5: wp.array(dtype=mat55),
        outcomponents: wp.array(dtype=wptype),
    ):
        v2result = v2[0] - s2[0]
        v3result = v3[0] - s3[0]
        v4result = v4[0] - s4[0]
        v5result = v5[0] - s5[0]

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * v2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * v3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * v4result[i, j]
                idx = idx + 1

        for i in range(5):
            for j in range(5):
                outcomponents[idx] = wptype(2) * v5result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_sub, suffix=dtype.__name__)

    if register_kernels:
        return

    s2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    s3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    s4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    s5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    v5 = wp.array(randvals(rng, [1, 5, 5], dtype), dtype=mat55, requires_grad=True, device=device)
    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, dtype=wptype, requires_grad=True, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[
            s2,
            s3,
            s4,
            s5,
            v2,
            v3,
            v4,
            v5,
        ],
        outputs=[outcomponents],
        device=device,
    )

    assert_np_equal(outcomponents.numpy()[:4], 2 * (v2.numpy() - s2.numpy()).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 2 * (v3.numpy() - s3.numpy()).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[13:29], 2 * (v4.numpy() - s4.numpy()).reshape(-1), tol=tol)
    assert_np_equal(outcomponents.numpy()[29:54], 2 * (v5.numpy() - s5.numpy()).reshape(-1), tol=10 * tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, in1, in2 in [(2, s2, v2), (3, s3, v3), (4, s4, v4), (5, s5, v5)]:
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(
                            kernel,
                            dim=1,
                            inputs=[s2, s3, s4, s5, v2, v3, v4, v5],
                            outputs=[outcomponents],
                            device=device,
                        )
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    expected_result = np.zeros((dim, dim), dtype=dtype)
                    expected_result[i, j] = 2
                    assert_np_equal(tape.gradients[in2].numpy()[0], expected_result, tol=10 * tol)
                    expected_result[i, j] = -2
                    assert_np_equal(tape.gradients[in1].numpy()[0], expected_result, tol=10 * tol)
                    tape.zero()

                    idx = idx + 1


def test_determinant(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)

    def check_mat_det(
        v2: wp.array(dtype=mat22),
        v3: wp.array(dtype=mat33),
        v4: wp.array(dtype=mat44),
        det2: wp.array(dtype=wptype),
        det3: wp.array(dtype=wptype),
        det4: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        det2[0] = wptype(2) * wp.determinant(v2[0])
        det3[0] = wptype(2) * wp.determinant(v3[0])
        det4[0] = wptype(2) * wp.determinant(v4[0])

    kernel = getkernel(check_mat_det, suffix=dtype.__name__)
    if register_kernels:
        return

    v2 = wp.array(randvals(rng, [1, 2, 2], dtype), dtype=mat22, requires_grad=True, device=device)
    v3 = wp.array(randvals(rng, [1, 3, 3], dtype), dtype=mat33, requires_grad=True, device=device)
    v4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)
    det2 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    det3 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    det4 = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[v2, v3, v4], outputs=[det2, det3, det4], device=device)

    if dtype in np_float_types:
        assert_np_equal(det2.numpy()[0], 2 * np.linalg.det(v2.numpy()[0].astype(np.float64)), tol=100 * tol)
        assert_np_equal(det3.numpy()[0], 2 * np.linalg.det(v3.numpy()[0].astype(np.float64)), tol=100 * tol)
        assert_np_equal(det4.numpy()[0], 2 * np.linalg.det(v4.numpy()[0].astype(np.float64)), tol=420 * tol)
    else:
        assert_np_equal(det2.numpy()[0], 2 * np.around(np.linalg.det(v2.numpy()[0])).astype(int))
        assert_np_equal(det3.numpy()[0], 2 * np.around(np.linalg.det(v3.numpy()[0])).astype(int))
        assert_np_equal(det4.numpy()[0], 2 * np.around(np.linalg.det(v4.numpy()[0])).astype(int))

    if dtype in np_float_types:
        # determinant derivative formula is annoying so finite differences?
        tape.backward(loss=det2)
        v2grads = 1.0 * tape.gradients[v2].numpy()[0]
        tape.zero()

        tape.backward(loss=det3)
        v3grads = 1.0 * tape.gradients[v3].numpy()[0]
        tape.zero()

        tape.backward(loss=det4)
        v4grads = 1.0 * tape.gradients[v4].numpy()[0]
        tape.zero()

        # finite differences are also annoying hence the large tolerance...
        # absolute nightmare in float16 too innit...
        dx = 0.01 if dtype == np.float16 else 0.0001
        fdtol = 2.0e-1 if dtype == np.float16 else 2.0e-3
        for i in range(2):
            for j in range(2):
                v2test = v2.numpy()
                v2test[0, i, j] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(v2test, dtype=v2.dtype, requires_grad=True, device=device), v3, v4],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dplus = det2.numpy()[0]
                v2test[0, i, j] -= 2.0 * dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(v2test, dtype=v2.dtype, requires_grad=True, device=device), v3, v4],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dminus = det2.numpy()[0]
                assert_np_equal((dplus - dminus) / (2.0 * dx * dplus), v2grads[i, j] / dplus, tol=fdtol)

        for i in range(3):
            for j in range(3):
                v3test = v3.numpy()
                v3test[0, i, j] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[v2, wp.array(v3test, dtype=v3.dtype, requires_grad=True, device=device), v4],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dplus = det3.numpy()[0]
                v3test[0, i, j] -= 2.0 * dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[v2, wp.array(v3test, dtype=v3.dtype, requires_grad=True, device=device), v4],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dminus = det3.numpy()[0]
                assert_np_equal((dplus - dminus) / (2.0 * dx * dplus), v3grads[i, j] / dplus, tol=fdtol)

        for i in range(4):
            for j in range(4):
                v4test = v4.numpy()
                v4test[0, i, j] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[v2, v3, wp.array(v4test, dtype=v4.dtype, requires_grad=True, device=device)],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dplus = det4.numpy()[0]
                v4test[0, i, j] -= 2.0 * dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[v2, v3, wp.array(v4test, dtype=v4.dtype, requires_grad=True, device=device)],
                    outputs=[det2, det3, det4],
                    device=device,
                )
                dminus = det4.numpy()[0]
                assert_np_equal((dplus - dminus) / (2.0 * dx * dplus), v4grads[i, j] / dplus, tol=fdtol)


# Unused. Why?
# def test_get_diag(test, device, dtype, register_kernels=False):
#     tol = {
#         np.float16: 1.0e-3,
#         np.float32: 1.0e-6,
#         np.float64: 1.0e-8,
#     }.get(dtype, 0)
#
#     wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
#     mat55 = wp.types.vector(shape=(5, 5), dtype=wptype)
#
#     output_select_kernel = get_select_kernel(wptype)
#
#     def check_mat_diag(
#         m55: wp.array(dtype=mat55),
#         outcomponents: wp.array(dtype=wptype),
#     ):
#         # multiply outputs by 2 so we've got something to backpropagate:
#         vec5result = wptype(2) * wp.get_diag(m55[0])
#
#         idx = 0
#         for i in range(5):
#             outcomponents[idx] = vec5result[i]
#             idx = idx + 1
#
#     kernel = getkernel(check_mat_diag, suffix=dtype.__name__)
#
#     if register_kernels:
#         return
#
#     m55 = wp.array(randvals((1, 5, 5), dtype), dtype=mat55, requires_grad=True, device=device)
#     outcomponents = wp.zeros(5, dtype=wptype, requires_grad=True, device=device)
#     out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
#
#     wp.launch(kernel, dim=1, inputs=[m55], outputs=[outcomponents], device=device)
#
#     assert_np_equal(outcomponents.numpy(), 2 * np.diag(m55.numpy()[0]), tol=tol)
#
#     if dtype in np_float_types:
#         idx = 0
#         for i in range(5):
#             tape = wp.Tape()
#             with tape:
#                 wp.launch(kernel, dim=1, inputs=[m55], outputs=[outcomponents], device=device)
#                 wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
#             tape.backward(loss=out)
#             expectedresult = np.zeros((5, 5), dtype=dtype)
#             expectedresult[i, i] = 2
#             assert_np_equal(tape.gradients[m55].numpy()[0], expectedresult, tol=10 * tol)
#             tape.zero()
#
#             idx = idx + 1


def test_inverse(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-2,
        np.float32: 1.0e-5,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_inverse(
        m2: wp.array(dtype=mat22),
        m3: wp.array(dtype=mat33),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        m2result = wp.inverse(m2[0])
        m3result = wp.inverse(m3[0])
        m4result = wp.inverse(m4[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(2):
            for j in range(2):
                outcomponents[idx] = wptype(2) * m2result[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * m3result[i, j]
                idx = idx + 1

        for i in range(4):
            for j in range(4):
                outcomponents[idx] = wptype(2) * m4result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_inverse, suffix=dtype.__name__)

    if register_kernels:
        return

    m2 = wp.array(
        2 * (randvals(rng, [1, 2, 2], dtype) + 0.2 * np.eye(2)), dtype=mat22, requires_grad=True, device=device
    )
    m3 = wp.array(
        2 * (randvals(rng, [1, 3, 3], dtype) + 0.2 * np.eye(3)), dtype=mat33, requires_grad=True, device=device
    )
    m4 = wp.array(
        2 * (randvals(rng, [1, 4, 4], dtype) + 0.2 * np.eye(4)), dtype=mat44, requires_grad=True, device=device
    )

    outcomponents = wp.zeros(2 * 2 + 3 * 3 + 4 * 4, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)

    assert_np_equal(outcomponents.numpy()[:4], 2 * np.linalg.inv(m2.numpy()[0].astype(np.float64)), tol=tol)
    assert_np_equal(outcomponents.numpy()[4:13], 2 * np.linalg.inv(m3.numpy()[0].astype(np.float64)), tol=5 * tol)
    assert_np_equal(outcomponents.numpy()[13:], 2 * np.linalg.inv(m4.numpy()[0].astype(np.float64)), tol=5 * tol)

    if dtype in np_float_types:
        # check gradients:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
        for dim, input in [(2, m2), (3, m3), (4, m4)]:
            minv = np.linalg.inv(input.numpy()[0].astype(np.float64))
            for i in range(dim):
                for j in range(dim):
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)
                        wp.launch(
                            output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device
                        )
                    tape.backward(loss=out)
                    d = np.zeros((dim, dim))
                    d[j, i] = 2
                    assert_np_equal(
                        tape.gradients[input].numpy()[0], -np.matmul(minv, np.matmul(d, minv)).T, tol=10 * tol
                    )
                    tape.zero()

                    idx = idx + 1

    # let's check 2x2 using different formulae just for (in)sanity's sake:
    m = m2.numpy()[0]

    det = m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]
    expected = 2 * np.array([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]], dtype=dtype) / det
    assert_np_equal(expected, outcomponents.numpy()[:4], tol=tol)

    # 0,0 component is this:
    # 2 * m[1,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(2 * m[1, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]), outcomponents.numpy()[0], tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, 0], outputs=[out], device=device)

    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(-2 * m[1, 1] * m[1, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 0], tol=tol)
        assert_np_equal(2 * m[1, 1] * m[0, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 0], tol=tol)
        assert_np_equal(-2 * m[0, 1] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 1], tol=tol)
        assert_np_equal(2 * m[1, 1] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 1], tol=tol)
        tape.zero()

    # 0,1 component is this:
    # -2 * m[0,1] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(-2 * m[0, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]), outcomponents.numpy()[1], tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, 1], outputs=[out], device=device)
    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(2 * m[0, 1] * m[1, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 0], tol=tol)
        assert_np_equal(-2 * m[0, 1] * m[0, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 0], tol=tol)
        assert_np_equal(2 * m[0, 0] * m[0, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 1], tol=tol)
        assert_np_equal(-2 * m[1, 1] * m[0, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 1], tol=tol)
        tape.zero()

    # 1,0 component is this:
    # -2 * m[1,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(-2 * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]), outcomponents.numpy()[2], tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, 2], outputs=[out], device=device)

    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(2 * m[1, 1] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 0], tol=tol)
        assert_np_equal(-2 * m[0, 0] * m[1, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 0], tol=tol)
        assert_np_equal(2 * m[0, 0] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 1], tol=tol)
        assert_np_equal(-2 * m[1, 0] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 1], tol=tol)
        tape.zero()

    # 1,1 component is this:
    # 2 * m[0,0] / (m[0,0]*m[1,1] - m[1,0] * m[0,1])
    assert_np_equal(2 * m[0, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]), outcomponents.numpy()[3], tol=tol)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[m2, m3, m4], outputs=[outcomponents], device=device)
        wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, 3], outputs=[out], device=device)

    if dtype in np_float_types:
        tape.backward(loss=out)
        g = tape.gradients[m2].numpy()[0]
        assert_np_equal(-2 * m[0, 1] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 0], tol=tol)
        assert_np_equal(2 * m[0, 0] * m[0, 1] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 0], tol=tol)
        assert_np_equal(2 * m[0, 0] * m[1, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[0, 1], tol=tol)
        assert_np_equal(-2 * m[0, 0] * m[0, 0] / (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) ** 2, g[1, 1], tol=tol)
        tape.zero()


def test_svd(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-12,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(length=3, dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)

    def check_mat_svd(
        m3: wp.array(dtype=mat33),
        Uout: wp.array(dtype=mat33),
        sigmaout: wp.array(dtype=vec3),
        Vout: wp.array(dtype=mat33),
        outcomponents: wp.array(dtype=wptype),
    ):
        U = mat33()
        sigma = vec3()
        V = mat33()

        wp.svd3(m3[0], U, sigma, V)

        Uout[0] = U
        sigmaout[0] = sigma
        Vout[0] = V

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * U[i, j]
                idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * sigma[i]
            idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * V[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_svd, suffix=dtype.__name__)

    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    m3 = wp.array(randvals(rng, [1, 3, 3], dtype) + np.eye(3), dtype=mat33, requires_grad=True, device=device)

    outcomponents = wp.zeros(2 * 3 * 3 + 3, dtype=wptype, requires_grad=True, device=device)
    Uout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)
    sigmaout = wp.zeros(1, dtype=vec3, requires_grad=True, device=device)
    Vout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m3], outputs=[Uout, sigmaout, Vout, outcomponents], device=device)

    Uout_np = Uout.numpy()[0].astype(np.float64)
    sigmaout_np = np.diag(sigmaout.numpy()[0].astype(np.float64))
    Vout_np = Vout.numpy()[0].astype(np.float64)

    assert_np_equal(
        np.matmul(Uout_np, np.matmul(sigmaout_np, Vout_np.T)), m3.numpy()[0].astype(np.float64), tol=30 * tol
    )

    if dtype == np.float16:
        # I'm not even going to bother testing the gradients for float16
        # because the rounding errors are terrible...
        return

    # check gradients:
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for idx in range(3 * 3 + 3 + 3 * 3):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[m3], outputs=[Uout, sigmaout, Vout, outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
        tape.backward(out)
        m3grads = 1.0 * tape.gradients[m3].numpy()[0]

        tape.zero()

        dx = 0.0001
        fdtol = 5.0e-4 if dtype == np.float64 else 2.0e-2
        for ii in range(3):
            for jj in range(3):
                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Uout, sigmaout, Vout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                plusval = out.numpy()[0]

                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] -= dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Uout, sigmaout, Vout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                minusval = out.numpy()[0]

                assert_np_equal((plusval - minusval) / (2 * dx), m3grads[ii, jj], tol=fdtol)


def test_qr(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 2.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-6,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)

    def check_mat_qr(
        m3: wp.array(dtype=mat33),
        Qout: wp.array(dtype=mat33),
        Rout: wp.array(dtype=mat33),
        outcomponents: wp.array(dtype=wptype),
    ):
        Q = mat33()
        R = mat33()

        wp.qr3(m3[0], Q, R)

        Qout[0] = Q
        Rout[0] = R

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * Q[i, j]
                idx = idx + 1

        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * R[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_qr, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    m3 = wp.array(0.5 * (randvals(rng, [1, 3, 3], dtype) + np.eye(3)), dtype=mat33, requires_grad=True, device=device)

    outcomponents = wp.zeros(2 * 3 * 3, dtype=wptype, requires_grad=True, device=device)
    Qout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)
    Rout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m3], outputs=[Qout, Rout, outcomponents], device=device)

    Qout_np = Qout.numpy()[0].astype(np.float64)
    Rout_np = Rout.numpy()[0].astype(np.float64)

    # check it's actually a q and an r:
    assert_np_equal(np.matmul(Qout_np.T, Qout_np), np.eye(3, dtype=np.float64), tol=tol)
    assert_np_equal(Rout_np[1, [0]], np.zeros(1, dtype=np.float64), tol=tol)
    assert_np_equal(Rout_np[2, [0, 1]], np.zeros(2, dtype=np.float64), tol=tol)

    # check it's a factorization:
    assert_np_equal(np.matmul(Qout_np, Rout_np), m3.numpy()[0].astype(np.float64), tol=30 * tol)

    if dtype == np.float16:
        # I'm not even going to bother testing the gradients for float16
        # because the rounding errors are terrible...
        return

    # check gradients:
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for idx in range(len(outcomponents)):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[m3], outputs=[Qout, Rout, outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
        tape.backward(out)
        m3grads = 1.0 * tape.gradients[m3].numpy()[0]

        tape.zero()

        dx = 0.0001
        fdtol = 5.0e-4 if dtype == np.float64 else 2.0e-2
        for ii in range(3):
            for jj in range(3):
                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Qout, Rout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                plusval = out.numpy()[0]

                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] -= dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Qout, Rout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                minusval = out.numpy()[0]

                assert_np_equal((plusval - minusval) / (2 * dx), m3grads[ii, jj], tol=fdtol)


def test_eig(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 4.0e-2,
        np.float32: 1.0e-5,
        np.float64: 1.0e-5,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(length=3, dtype=wptype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wptype)

    def check_mat_eig(
        m3: wp.array(dtype=mat33),
        Qout: wp.array(dtype=mat33),
        dout: wp.array(dtype=vec3),
        outcomponents: wp.array(dtype=wptype),
    ):
        Q = mat33()
        d = vec3()

        wp.eig3(m3[0] + wp.transpose(m3[0]), Q, d)

        Qout[0] = Q
        dout[0] = d

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * Q[i, j]
                idx = idx + 1

        for i in range(3):
            outcomponents[idx] = wptype(2) * d[i]
            idx = idx + 1

    kernel = getkernel(check_mat_eig, suffix=dtype.__name__)
    output_select_kernel = get_select_kernel(wptype)

    if register_kernels:
        return

    m3_np = randvals(rng, [1, 3, 3], dtype) + np.eye(3, dtype=dtype)
    m3 = wp.array(m3_np, dtype=mat33, requires_grad=True, device=device)

    outcomponents = wp.zeros(3 * 3 + 3, dtype=wptype, requires_grad=True, device=device)
    Qout = wp.zeros(1, dtype=mat33, requires_grad=True, device=device)
    dout = wp.zeros(1, dtype=vec3, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[m3], outputs=[Qout, dout, outcomponents], device=device)

    Qout_np = Qout.numpy()[0].astype(np.float64)
    dout_np = dout.numpy()[0].astype(np.float64)
    Dout_np = np.diag(dout_np)

    # check Q is orthogonal:
    assert_np_equal(np.matmul(Qout_np.T, Qout_np), np.eye(3), tol=tol)

    # check Q contains eigenvectors:
    assert_np_equal(np.matmul(Qout_np, np.matmul(Dout_np, Qout_np.T)), (m3_np[0] + m3_np[0].transpose()), tol=tol)

    if dtype == np.float16:
        # I'm not even going to bother testing the gradients for float16
        # because the rounding errors are terrible...
        return

    # check gradients:
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)
    idx = 0
    for idx in range(len(outcomponents)):
        tape = wp.Tape()
        with tape:
            wp.launch(kernel, dim=1, inputs=[m3], outputs=[Qout, dout, outcomponents], device=device)
            wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
        tape.backward(out)
        m3grads = 1.0 * tape.gradients[m3].numpy()[0]

        tape.zero()

        dx = 0.0001
        fdtol = 5.0e-4 if dtype == np.float64 else 2.0e-2
        for ii in range(3):
            for jj in range(3):
                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] += dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Qout, dout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                plusval = out.numpy()[0]

                m3test = 1.0 * m3.numpy()
                m3test[0, ii, jj] -= dx
                wp.launch(
                    kernel,
                    dim=1,
                    inputs=[wp.array(m3test, dtype=mat33, device=device)],
                    outputs=[Qout, dout, outcomponents],
                    device=device,
                )
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                minusval = out.numpy()[0]

                assert_np_equal((plusval - minusval) / (2 * dx), m3grads[ii, jj], tol=fdtol)


def test_skew(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 1.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(length=3, dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_skew(
        v3: wp.array(dtype=vec3),
        outcomponents: wp.array(dtype=wptype),
    ):
        m3result = wp.skew(v3[0])

        # multiply outputs by 2 so we've got something to backpropagate:
        idx = 0
        for i in range(3):
            for j in range(3):
                outcomponents[idx] = wptype(2) * m3result[i, j]
                idx = idx + 1

    kernel = getkernel(check_mat_skew, suffix=dtype.__name__)

    if register_kernels:
        return

    v3 = wp.array(randvals(rng, [1, 3], dtype), dtype=vec3, requires_grad=True, device=device)

    outcomponents = wp.zeros(3 * 3, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v3], outputs=[outcomponents], device=device)

    # make sure it gives you a cross product matrix:
    crossprodmat = outcomponents.numpy().reshape(3, 3)
    v = np.array([1, 0, 0])
    assert_np_equal(
        np.matmul(crossprodmat, np.array([1, 0, 0])).reshape(-1),
        2 * np.cross(v3.numpy()[0], np.array([1, 0, 0])),
        tol=tol,
    )
    assert_np_equal(
        np.matmul(crossprodmat, np.array([0, 1, 0])).reshape(-1),
        2 * np.cross(v3.numpy()[0], np.array([0, 1, 0])),
        tol=tol,
    )
    assert_np_equal(
        np.matmul(crossprodmat, np.array([0, 0, 1])).reshape(-1),
        2 * np.cross(v3.numpy()[0], np.array([0, 0, 1])),
        tol=tol,
    )

    # check it another way:
    x0 = v3.numpy()[0, 0]
    x1 = v3.numpy()[0, 1]
    x2 = v3.numpy()[0, 2]
    crossprodmat_expected = np.array(
        [
            [0, -x2, x1],
            [x2, 0, -x0],
            [-x1, x0, 0],
        ],
        dtype=dtype,
    )
    assert_np_equal(crossprodmat, 2 * crossprodmat_expected, tol=tol)

    if dtype in np_float_types:
        idx = 0
        out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

        for i in range(3):
            for j in range(3):
                tape = wp.Tape()
                with tape:
                    wp.launch(kernel, dim=1, inputs=[v3], outputs=[outcomponents], device=device)
                    wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, idx], outputs=[out], device=device)
                tape.backward(loss=out)
                if i == j:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.zeros(3))
                elif [i, j] == [0, 1]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([0, 0, -2]))
                elif [i, j] == [1, 0]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([0, 0, 2]))
                elif [i, j] == [0, 2]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([0, 2, 0]))
                elif [i, j] == [2, 0]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([0, -2, 0]))
                elif [i, j] == [1, 2]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([-2, 0, 0]))
                elif [i, j] == [2, 1]:
                    assert_np_equal(tape.gradients[v3].numpy()[0], np.array([2, 0, 0]))
                tape.zero()

                idx = idx + 1


def test_transform_point(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(length=3, dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_transform_point(
        v3: wp.array(dtype=vec3),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        presult = wptype(2) * wp.transform_point(m4[0], v3[0])

        outcomponents[0] = presult[0]
        outcomponents[1] = presult[1]
        outcomponents[2] = presult[2]

    kernel = getkernel(check_mat_transform_point, suffix=dtype.__name__)

    if register_kernels:
        return

    v3 = wp.array(randvals(rng, [1, 3], dtype), dtype=vec3, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)

    outcomponents = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v3, m4], outputs=[outcomponents], device=device)

    v3homog = np.ones(4, dtype=dtype)
    v3homog[:3] = v3.numpy()[0]
    assert_np_equal(outcomponents.numpy(), 2 * np.matmul(m4.numpy()[0], v3homog)[:3], tol=10 * tol)

    if dtype in np_float_types:
        for j in range(3):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[v3, m4], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, j], outputs=[out], device=device)
            tape.backward(loss=out)

            assert_np_equal(2 * m4.numpy()[0, j, :3], tape.gradients[v3].numpy(), tol=tol)
            expected = np.zeros((4, 4), dtype=dtype)
            expected[j, :3] = 2 * v3.numpy()
            expected[j, 3] = 2
            assert_np_equal(tape.gradients[m4].numpy(), expected, tol=tol)

            tape.zero()


def test_transform_vector(test, device, dtype, register_kernels=False):
    rng = np.random.default_rng(123)

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    wptype = wp.types.np_dtype_to_warp_type[np.dtype(dtype)]
    vec3 = wp.types.vector(length=3, dtype=wptype)
    mat44 = wp.types.matrix(shape=(4, 4), dtype=wptype)

    output_select_kernel = get_select_kernel(wptype)

    def check_mat_transform_vector(
        v3: wp.array(dtype=vec3),
        m4: wp.array(dtype=mat44),
        outcomponents: wp.array(dtype=wptype),
    ):
        # multiply outputs by 2 so we've got something to backpropagate:
        presult = wptype(2) * wp.transform_vector(m4[0], v3[0])

        outcomponents[0] = presult[0]
        outcomponents[1] = presult[1]
        outcomponents[2] = presult[2]

    kernel = getkernel(check_mat_transform_vector, suffix=dtype.__name__)

    if register_kernels:
        return

    v3 = wp.array(randvals(rng, [1, 3], dtype), dtype=vec3, requires_grad=True, device=device)
    m4 = wp.array(randvals(rng, [1, 4, 4], dtype), dtype=mat44, requires_grad=True, device=device)

    outcomponents = wp.zeros(3, dtype=wptype, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=wptype, requires_grad=True, device=device)

    wp.launch(kernel, dim=1, inputs=[v3, m4], outputs=[outcomponents], device=device)

    v3homog = np.zeros(4, dtype=dtype)
    v3homog[:3] = v3.numpy()[0]
    assert_np_equal(outcomponents.numpy(), 2 * np.matmul(m4.numpy()[0], v3homog)[:3], tol=10 * tol)

    if dtype in np_float_types:
        for j in range(3):
            tape = wp.Tape()
            with tape:
                wp.launch(kernel, dim=1, inputs=[v3, m4], outputs=[outcomponents], device=device)
                wp.launch(output_select_kernel, dim=1, inputs=[outcomponents, j], outputs=[out], device=device)
            tape.backward(loss=out)

            assert_np_equal(2 * m4.numpy()[0, j, :3], tape.gradients[v3].numpy(), tol=tol)
            expected = np.zeros((4, 4), dtype=dtype)
            expected[j, :3] = 2 * v3.numpy()
            assert_np_equal(tape.gradients[m4].numpy(), expected, tol=tol)

            tape.zero()


def test_mat_array_type_indexing(test, device, dtype, register_kernels=False):
    np_type = np.dtype(dtype)
    wp_type = wp.types.np_dtype_to_warp_type[np_type]

    vec2 = wp.types.vector(length=2, dtype=wp_type)
    mat22 = wp.types.matrix(shape=(2, 2), dtype=wp_type)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wp_type)

    def mattest_read_write_store(x: wp.array(dtype=wp_type), a: wp.array(dtype=mat22)):
        tid = wp.tid()

        t = a[tid]
        t[0, 0] = x[tid]
        a[tid] = t

    def mattest_in_register(x: wp.array2d(dtype=mat22), y: wp.array(dtype=vec2)):
        i, j = wp.tid()

        a = mat22(wp_type(0.0))
        a[0] = y[i]
        a[1, 1] = wp_type(3.0)
        x[i, j] = a

    def mattest_in_register_overwrite(x: wp.array2d(dtype=mat22), y: wp.array(dtype=vec2)):
        i, j = wp.tid()

        a = mat22(wp_type(0.0))
        a[0] = y[i]
        a[0, 1] = wp_type(3.0)
        x[i, j] = a

    kernel_read_write_store = getkernel(mattest_read_write_store, suffix=dtype.__name__)
    kernel_in_register = getkernel(mattest_in_register, suffix=dtype.__name__)
    kernel_in_register_overwrite = getkernel(mattest_in_register_overwrite, suffix=dtype.__name__)

    if register_kernels:
        return

    a = wp.ones(1, dtype=mat22, device=device, requires_grad=True)
    x = wp.full(1, value=2.0, dtype=wp_type, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel_read_write_store, dim=1, inputs=[x, a], device=device)

    tape.backward(grads={a: wp.ones_like(a, requires_grad=False)})

    assert_np_equal(a.numpy(), np.array([[[2.0, 1.0], [1.0, 1.0]]], dtype=np_type))
    assert_np_equal(x.grad.numpy(), np.array([1.0], dtype=np_type))

    tape.reset()

    x = wp.zeros((1, 1), dtype=mat22, device=device, requires_grad=True)
    y = wp.ones(1, dtype=vec2, device=device, requires_grad=True)

    with tape:
        wp.launch(kernel_in_register, dim=(1, 1), inputs=[x, y], device=device)

    tape.backward(grads={x: wp.ones_like(x, requires_grad=False)})

    assert_np_equal(x.numpy(), np.array([[[[1.0, 1.0], [0.0, 3.0]]]], dtype=np_type))
    assert_np_equal(y.grad.numpy(), np.array([[1.0, 1.0]], dtype=np_type))

    tape.reset()

    x = wp.zeros((1, 1), dtype=mat22, device=device, requires_grad=True)
    y = wp.ones(1, dtype=vec2, device=device, requires_grad=True)

    with tape:
        wp.launch(kernel_in_register_overwrite, dim=(1, 1), inputs=[x, y], device=device)

    tape.backward(grads={x: wp.ones_like(x, requires_grad=False)})

    assert_np_equal(x.numpy(), np.array([[[[1.0, 3.0], [0.0, 0.0]]]], dtype=np_type))
    assert_np_equal(y.grad.numpy(), np.array([[1.0, 0.0]], dtype=np_type))


# Test matrix constructors using explicit type (float16)
# note that these tests are specifically not using generics / closure
# args to create kernels dynamically (like the rest of this file)
# as those use different code paths to resolve arg types which
# has lead to regressions.
@wp.kernel
def test_constructors_explicit_precision():
    # construction for custom matrix types
    eye = wp.identity(dtype=wp.float16, n=2)
    zeros = wp.matrix(shape=(2, 2), dtype=wp.float16)
    custom = wp.matrix(wp.float16(0.0), wp.float16(1.0), wp.float16(2.0), wp.float16(3.0), shape=(2, 2))

    for i in range(2):
        for j in range(2):
            if i == j:
                wp.expect_eq(eye[i, j], wp.float16(1.0))
            else:
                wp.expect_eq(eye[i, j], wp.float16(0.0))

            wp.expect_eq(zeros[i, j], wp.float16(0.0))
            wp.expect_eq(custom[i, j], wp.float16(i) * wp.float16(2.0) + wp.float16(j))


mat32d = wp.mat(shape=(3, 2), dtype=wp.float64)


@wp.kernel
def test_matrix_constructor_value_func():
    a = wp.mat22()
    b = wp.matrix(a, shape=(2, 2))
    c = mat32d()
    d = mat32d(c, shape=(3, 2))
    e = mat32d(wp.float64(1.0), wp.float64(2.0), wp.float64(1.0), wp.float64(2.0), wp.float64(1.0), wp.float64(2.0))
    f = mat32d(
        wp.vec3d(wp.float64(1.0), wp.float64(2.0), wp.float64(3.0)),
        wp.vec3d(wp.float64(1.0), wp.float64(2.0), wp.float64(3.0)),
    )


# Same as above but with a default (float/int) type
# which tests some different code paths that
# need to ensure types are correctly canonicalized
# during codegen
@wp.kernel
def test_constructors_default_precision():
    # construction for default (float) matrix types
    eye = wp.identity(dtype=float, n=2)
    zeros = wp.matrix(shape=(2, 2), dtype=float)
    custom = wp.matrix(0.0, 1.0, 2.0, 3.0, shape=(2, 2))

    for i in range(2):
        for j in range(2):
            if i == j:
                wp.expect_eq(eye[i, j], 1.0)
            else:
                wp.expect_eq(eye[i, j], 0.0)

            wp.expect_eq(zeros[i, j], 0.0)
            wp.expect_eq(custom[i, j], float(i) * 2.0 + float(j))


@wp.kernel
def test_matrix_mutation(expected: wp.types.matrix(shape=(10, 3), dtype=float)):
    m = wp.matrix(shape=(10, 3), dtype=float)

    # test direct element indexing
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[0, 2] = 3.0

    # The nested indexing (matrix->vector->scalar) below does not
    # currently modify m because m[0] returns row vector by
    # value rather than reference, this is different from NumPy
    # which always returns by ref. Not clear how we can support
    # this as well as auto-diff.

    # m[0][1] = 2.0
    # m[0][2] = 3.0

    # test setting rows
    for i in range(1, 10):
        m[i] = m[i - 1] + wp.vec3(1.0, 2.0, 3.0)

    wp.expect_eq(m, expected)


# NOTE: Compile tile is highly sensitive to shape so we use small values now
CONSTANT_SHAPE_ROWS = wp.constant(2)
CONSTANT_SHAPE_COLS = wp.constant(2)


# tests that we can use global constants in shape keyword argument
# for matrix constructor
@wp.kernel
def test_constructors_constant_shape():
    m = wp.matrix(shape=(CONSTANT_SHAPE_ROWS, CONSTANT_SHAPE_COLS), dtype=float)

    for i in range(CONSTANT_SHAPE_ROWS):
        for j in range(CONSTANT_SHAPE_COLS):
            m[i, j] = float(i * j)


Mat23 = wp.mat((2, 3), dtype=wp.float16)


@wp.kernel
def matrix_len_kernel(
    m1: wp.mat22, m2: wp.mat((3, 3), float), m3: wp.mat((Any, Any), float), m4: Mat23, out: wp.array(dtype=int)
):
    length = wp.static(len(m1))
    wp.expect_eq(len(m1), 2)
    out[0] = len(m1)

    length = len(m2)
    wp.expect_eq(wp.static(len(m2)), 3)
    out[1] = len(m2)

    length = len(m3)
    wp.expect_eq(len(m3), 4)
    out[2] = wp.static(len(m3))

    length = wp.static(len(m4))
    wp.expect_eq(wp.static(len(m4)), 2)
    out[3] = wp.static(len(m4))

    foo = wp.mat22()
    length = len(foo)
    wp.expect_eq(len(foo), 2)
    out[4] = len(foo)


def test_matrix_len(test, device):
    m1 = wp.mat22()
    m2 = wp.mat33()
    m3 = wp.mat44()
    m4 = Mat23()
    out = wp.empty(5, dtype=int, device=device)
    wp.launch(matrix_len_kernel, dim=(1,), inputs=(m1, m2, m3, m4), outputs=(out,), device=device)

    test.assertEqual(out.numpy()[0], 2)
    test.assertEqual(out.numpy()[1], 3)
    test.assertEqual(out.numpy()[2], 4)
    test.assertEqual(out.numpy()[3], 2)
    test.assertEqual(out.numpy()[4], 2)

    test.assertEqual(len(m1), 2)
    test.assertEqual(len(m2), 3)
    test.assertEqual(len(m3), 4)
    test.assertEqual(len(m4), 2)


@wp.kernel
def matrix_augassign_kernel(
    a: wp.array(dtype=wp.mat22), b: wp.array(dtype=wp.mat22), c: wp.array(dtype=wp.mat22), d: wp.array(dtype=wp.mat22)
):
    i = wp.tid()

    m1 = wp.mat22()
    m2 = b[i]

    m1[0, 0] += m2[0, 0]
    m1[0, 1] += m2[0, 1]
    m1[1, 0] += m2[1, 0]
    m1[1, 1] += m2[1, 1]

    a[i] = m1

    m3 = wp.mat22()
    m4 = d[i]

    m3[0, 0] -= m4[0, 0]
    m3[0, 1] -= m4[0, 1]
    m3[1, 0] -= m4[1, 0]
    m3[1, 1] -= m4[1, 1]

    c[i] = m3


def test_matrix_augassign(test, device):
    N = 3

    a = wp.zeros(N, dtype=wp.mat22, requires_grad=True)
    b = wp.ones(N, dtype=wp.mat22, requires_grad=True)

    c = wp.zeros(N, dtype=wp.mat22, requires_grad=True)
    d = wp.ones(N, dtype=wp.mat22, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(matrix_augassign_kernel, N, inputs=[a, b, c, d])

    tape.backward(grads={a: wp.ones_like(a), c: wp.ones_like(c)})

    assert_np_equal(a.numpy(), wp.ones_like(a).numpy())
    assert_np_equal(a.grad.numpy(), wp.ones_like(a).numpy())
    assert_np_equal(b.grad.numpy(), wp.ones_like(a).numpy())

    assert_np_equal(c.numpy(), -wp.ones_like(c).numpy())
    assert_np_equal(c.grad.numpy(), wp.ones_like(c).numpy())
    assert_np_equal(d.grad.numpy(), -wp.ones_like(d).numpy())


devices = get_test_devices()


class TestMat(unittest.TestCase):
    def test_tpl_ops_with_anon(self):
        mat22f = wp.mat((2, 2), dtype=float)

        m = wp.mat22f(1.0, 2.0, 3.0, 4.0)
        m += mat22f(2.0, 3.0, 4.0, 5.0)
        m -= mat22f(3.0, 4.0, 5.0, 6.0)
        self.assertSequenceEqual(m, ((0.0, 1.0), (2.0, 3.0)))

        m = mat22f(1.0, 2.0, 3.0, 4.0)
        m += wp.mat22f(2.0, 3.0, 4.0, 5.0)
        m -= wp.mat22f(3.0, 4.0, 5.0, 6.0)
        self.assertSequenceEqual(m, ((0.0, 1.0), (2.0, 3.0)))


add_kernel_test(TestMat, test_constructors_explicit_precision, dim=1, devices=devices)
add_kernel_test(TestMat, test_constructors_default_precision, dim=1, devices=devices)
add_kernel_test(TestMat, test_constructors_constant_shape, dim=1, devices=devices)
add_kernel_test(TestMat, test_matrix_constructor_value_func, dim=1, devices=devices)

mat103 = wp.types.matrix(shape=(10, 3), dtype=float)
add_kernel_test(
    TestMat,
    test_matrix_mutation,
    dim=1,
    inputs=[
        mat103(
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0,
            4.0, 8.0, 12.0,
            5.0, 10.0, 15.0,
            6.0, 12.0, 18.0,
            7.0, 14.0, 21.0,
            8.0, 16.0, 24.0,
            9.0, 18.0, 27.0,
            10.0, 20.0, 30.0,
        )
    ],
    devices=devices,
)  # fmt: skip

for dtype in np_signed_int_types + np_float_types:
    add_function_test_register_kernel(
        TestMat, f"test_negation_{dtype.__name__}", test_negation, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_subtraction_{dtype.__name__}", test_subtraction, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_matmul_{dtype.__name__}", test_matmul, devices=devices, dtype=dtype
    )

add_function_test(
    TestMat,
    "test_anon_constructor_error_shape_arg_missing",
    test_anon_constructor_error_shape_arg_missing,
    devices=devices,
)
add_function_test(
    TestMat, "test_anon_constructor_error_shape_mismatch", test_anon_constructor_error_shape_mismatch, devices=devices
)
add_function_test(
    TestMat, "test_anon_constructor_error_type_mismatch", test_anon_constructor_error_type_mismatch, devices=devices
)
add_function_test(
    TestMat,
    "test_anon_constructor_error_invalid_arg_count",
    test_anon_constructor_error_invalid_arg_count,
    devices=devices,
)
add_function_test(
    TestMat,
    "test_anon_xform_constructor_error_type_mismatch",
    test_anon_xform_constructor_error_type_mismatch,
    devices=devices,
)
add_function_test(
    TestMat,
    "test_tpl_constructor_error_incompatible_sizes",
    test_tpl_constructor_error_incompatible_sizes,
    devices=devices,
)
add_function_test(
    TestMat,
    "test_tpl_constructor_error_invalid_vector_count",
    test_tpl_constructor_error_invalid_vector_count,
    devices=devices,
)
add_function_test(
    TestMat,
    "test_tpl_constructor_error_invalid_vector_shape",
    test_tpl_constructor_error_invalid_vector_shape,
    devices=devices,
)
add_function_test(
    TestMat,
    "test_tpl_constructor_error_invalid_arg_count",
    test_tpl_constructor_error_invalid_arg_count,
    devices=devices,
)

for dtype in np_float_types:
    add_function_test(
        TestMat, f"test_py_arithmetic_ops_{dtype.__name__}", test_py_arithmetic_ops, devices=None, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_quat_constructor_{dtype.__name__}", test_quat_constructor, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_inverse_{dtype.__name__}", test_inverse, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(TestMat, f"test_svd_{dtype.__name__}", test_svd, devices=devices, dtype=dtype)
    add_function_test_register_kernel(TestMat, f"test_qr_{dtype.__name__}", test_qr, devices=devices, dtype=dtype)
    add_function_test_register_kernel(TestMat, f"test_eig_{dtype.__name__}", test_eig, devices=devices, dtype=dtype)
    add_function_test_register_kernel(
        TestMat, f"test_transform_point_{dtype.__name__}", test_transform_point, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_transform_vector_{dtype.__name__}", test_transform_vector, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(
        TestMat, f"test_determinant_{dtype.__name__}", test_determinant, devices=devices, dtype=dtype
    )
    add_function_test_register_kernel(TestMat, f"test_skew_{dtype.__name__}", test_skew, devices=devices, dtype=dtype)
    add_function_test_register_kernel(
        TestMat,
        f"test_mat_array_type_indexing_{dtype.__name__}",
        test_mat_array_type_indexing,
        devices=devices,
        dtype=dtype,
    )
add_function_test(TestMat, "test_matrix_len", test_matrix_len, devices=devices)
add_function_test(TestMat, "test_matrix_augassign", test_matrix_augassign, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
