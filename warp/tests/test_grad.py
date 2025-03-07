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

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def scalar_grad(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    y[0] = x[0] ** 2.0


def test_scalar_grad(test, device):
    x = wp.array([3.0], dtype=float, device=device, requires_grad=True)
    y = wp.zeros_like(x)

    tape = wp.Tape()
    with tape:
        wp.launch(scalar_grad, dim=1, inputs=[x, y], device=device)

    tape.backward(y)

    assert_np_equal(tape.gradients[x].numpy(), np.array(6.0))


@wp.kernel
def for_loop_grad(n: int, x: wp.array(dtype=float), s: wp.array(dtype=float)):
    sum = float(0.0)

    for i in range(n):
        sum = sum + x[i] * 2.0

    s[0] = sum


def test_for_loop_grad(test, device):
    n = 32
    val = np.ones(n, dtype=np.float32)

    x = wp.array(val, device=device, requires_grad=True)
    sum = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(for_loop_grad, dim=1, inputs=[n, x, sum], device=device)

    # ensure forward pass outputs correct
    assert_np_equal(sum.numpy(), 2.0 * np.sum(x.numpy()))

    tape.backward(loss=sum)

    # ensure forward pass outputs persist
    assert_np_equal(sum.numpy(), 2.0 * np.sum(x.numpy()))
    # ensure gradients correct
    assert_np_equal(tape.gradients[x].numpy(), 2.0 * val)


def test_for_loop_graph_grad(test, device):
    wp.load_module(device=device)

    n = 32
    val = np.ones(n, dtype=np.float32)

    x = wp.array(val, device=device, requires_grad=True)
    sum = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    wp.capture_begin(device, force_module_load=False)
    try:
        tape = wp.Tape()
        with tape:
            wp.launch(for_loop_grad, dim=1, inputs=[n, x, sum], device=device)

        tape.backward(loss=sum)
    finally:
        graph = wp.capture_end(device)

    wp.capture_launch(graph)
    wp.synchronize_device(device)

    # ensure forward pass outputs persist
    assert_np_equal(sum.numpy(), 2.0 * np.sum(x.numpy()))
    # ensure gradients correct
    assert_np_equal(x.grad.numpy(), 2.0 * val)

    wp.capture_launch(graph)
    wp.synchronize_device(device)


@wp.kernel
def for_loop_nested_if_grad(n: int, x: wp.array(dtype=float), s: wp.array(dtype=float)):
    sum = float(0.0)

    for i in range(n):
        if i < 16:
            if i < 8:
                sum = sum + x[i] * 2.0
            else:
                sum = sum + x[i] * 4.0
        else:
            if i < 24:
                sum = sum + x[i] * 6.0
            else:
                sum = sum + x[i] * 8.0

    s[0] = sum


def test_for_loop_nested_if_grad(test, device):
    n = 32
    val = np.ones(n, dtype=np.float32)
    # fmt: off
    expected_val = [
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
        6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
        8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
    ]
    expected_grad = [
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
        6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
        8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
    ]
    # fmt: on

    x = wp.array(val, device=device, requires_grad=True)
    sum = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(for_loop_nested_if_grad, dim=1, inputs=[n, x, sum], device=device)

    assert_np_equal(sum.numpy(), np.sum(expected_val))

    tape.backward(loss=sum)

    assert_np_equal(sum.numpy(), np.sum(expected_val))
    assert_np_equal(tape.gradients[x].numpy(), np.array(expected_grad))


@wp.kernel
def for_loop_grad_nested(n: int, x: wp.array(dtype=float), s: wp.array(dtype=float)):
    sum = float(0.0)

    for i in range(n):
        for j in range(n):
            sum = sum + x[i * n + j] * float(i * n + j) + 1.0

    s[0] = sum


def test_for_loop_nested_for_grad(test, device):
    x = wp.zeros(9, dtype=float, device=device, requires_grad=True)
    s = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(for_loop_grad_nested, dim=1, inputs=[3, x, s], device=device)

    tape.backward(s)

    assert_np_equal(s.numpy(), np.array([9.0]))
    assert_np_equal(tape.gradients[x].numpy(), np.arange(0.0, 9.0, 1.0))


# differentiating thought most while loops is not supported
# since doing things like i = i + 1 breaks adjointing

# @wp.kernel
# def while_loop_grad(n: int,
#                     x: wp.array(dtype=float),
#                     c: wp.array(dtype=int),
#                     s: wp.array(dtype=float)):

#     tid = wp.tid()

#     while i < n:
#         s[0] = s[0] + x[i]*2.0
#         i = i + 1


# def test_while_loop_grad(test, device):

#     n = 32
#     x = wp.array(np.ones(n, dtype=np.float32), device=device, requires_grad=True)
#     c = wp.zeros(1, dtype=int, device=device)
#     sum = wp.zeros(1, dtype=wp.float32, device=device)

#     tape = wp.Tape()
#     with tape:
#         wp.launch(while_loop_grad, dim=1, inputs=[n, x, c, sum], device=device)

#     tape.backward(loss=sum)

#     assert_np_equal(sum.numpy(), 2.0*np.sum(x.numpy()))
#     assert_np_equal(tape.gradients[x].numpy(), 2.0*np.ones_like(x.numpy()))


@wp.kernel
def preserve_outputs(
    n: int, x: wp.array(dtype=float), c: wp.array(dtype=float), s1: wp.array(dtype=float), s2: wp.array(dtype=float)
):
    tid = wp.tid()

    # plain store
    c[tid] = x[tid] * 2.0

    # atomic stores
    wp.atomic_add(s1, 0, x[tid] * 3.0)
    wp.atomic_sub(s2, 0, x[tid] * 2.0)


# tests that outputs from the forward pass are
# preserved by the backward pass, i.e.: stores
# are omitted during the forward reply
def test_preserve_outputs_grad(test, device):
    n = 32

    val = np.ones(n, dtype=np.float32)

    x = wp.array(val, device=device, requires_grad=True)
    c = wp.zeros_like(x)

    s1 = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    s2 = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(preserve_outputs, dim=n, inputs=[n, x, c, s1, s2], device=device)

    # ensure forward pass results are correct
    assert_np_equal(x.numpy(), val)
    assert_np_equal(c.numpy(), val * 2.0)
    assert_np_equal(s1.numpy(), np.array(3.0 * n))
    assert_np_equal(s2.numpy(), np.array(-2.0 * n))

    # run backward on first loss
    tape.backward(loss=s1)

    # ensure inputs, copy and sum are unchanged by backwards pass
    assert_np_equal(x.numpy(), val)
    assert_np_equal(c.numpy(), val * 2.0)
    assert_np_equal(s1.numpy(), np.array(3.0 * n))
    assert_np_equal(s2.numpy(), np.array(-2.0 * n))

    # ensure gradients are correct
    assert_np_equal(tape.gradients[x].numpy(), 3.0 * val)

    # run backward on second loss
    tape.zero()
    tape.backward(loss=s2)

    assert_np_equal(x.numpy(), val)
    assert_np_equal(c.numpy(), val * 2.0)
    assert_np_equal(s1.numpy(), np.array(3.0 * n))
    assert_np_equal(s2.numpy(), np.array(-2.0 * n))

    # ensure gradients are correct
    assert_np_equal(tape.gradients[x].numpy(), -2.0 * val)


def gradcheck(func, func_name, inputs, device, eps=1e-4, tol=1e-2):
    """
    Checks that the gradient of the Warp kernel is correct by comparing it to the
    numerical gradient computed using finite differences.
    """

    kernel = wp.Kernel(func=func, key=func_name)

    def f(xs):
        # call the kernel without taping for finite differences
        wp_xs = [wp.array(xs[i], ndim=1, dtype=inputs[i].dtype, device=device) for i in range(len(inputs))]
        output = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(kernel, dim=1, inputs=wp_xs, outputs=[output], device=device)
        return output.numpy()[0]

    # compute numerical gradient
    numerical_grad = []
    np_xs = []
    for i in range(len(inputs)):
        np_xs.append(inputs[i].numpy().flatten().copy())
        numerical_grad.append(np.zeros_like(np_xs[-1]))
        inputs[i].requires_grad = True

    for i in range(len(np_xs)):
        for j in range(len(np_xs[i])):
            np_xs[i][j] += eps
            y1 = f(np_xs)
            np_xs[i][j] -= 2 * eps
            y2 = f(np_xs)
            np_xs[i][j] += eps
            numerical_grad[i][j] = (y1 - y2) / (2 * eps)

    # compute analytical gradient
    tape = wp.Tape()
    output = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    with tape:
        wp.launch(kernel, dim=1, inputs=inputs, outputs=[output], device=device)

    tape.backward(loss=output)

    # compare gradients
    for i in range(len(inputs)):
        grad = tape.gradients[inputs[i]]
        assert_np_equal(grad.numpy(), numerical_grad[i], tol=tol)

    tape.zero()


def test_vector_math_grad(test, device):
    rng = np.random.default_rng(123)

    # test unary operations
    for dim, vec_type in [(2, wp.vec2), (3, wp.vec3), (4, wp.vec4), (4, wp.quat)]:

        def check_length(vs: wp.array(dtype=vec_type), out: wp.array(dtype=float)):
            out[0] = wp.length(vs[0])

        def check_length_sq(vs: wp.array(dtype=vec_type), out: wp.array(dtype=float)):
            out[0] = wp.length_sq(vs[0])

        def check_normalize(vs: wp.array(dtype=vec_type), out: wp.array(dtype=float)):
            out[0] = wp.length_sq(wp.normalize(vs[0]))  # compress to scalar output

        # run the tests with 5 different random inputs
        for _ in range(5):
            x = wp.array(rng.random(size=(1, dim), dtype=np.float32), dtype=vec_type, device=device)
            gradcheck(check_length, f"check_length_{vec_type.__name__}", [x], device)
            gradcheck(check_length_sq, f"check_length_sq_{vec_type.__name__}", [x], device)
            gradcheck(check_normalize, f"check_normalize_{vec_type.__name__}", [x], device)


def test_matrix_math_grad(test, device):
    rng = np.random.default_rng(123)

    # test unary operations
    for dim, mat_type in [(2, wp.mat22), (3, wp.mat33), (4, wp.mat44)]:

        def check_determinant(vs: wp.array(dtype=mat_type), out: wp.array(dtype=float)):
            out[0] = wp.determinant(vs[0])

        def check_trace(vs: wp.array(dtype=mat_type), out: wp.array(dtype=float)):
            out[0] = wp.trace(vs[0])

        # run the tests with 5 different random inputs
        for _ in range(5):
            x = wp.array(rng.random(size=(1, dim, dim), dtype=np.float32), ndim=1, dtype=mat_type, device=device)
            gradcheck(check_determinant, f"check_length_{mat_type.__name__}", [x], device)
            gradcheck(check_trace, f"check_length_sq_{mat_type.__name__}", [x], device)


def test_3d_math_grad(test, device):
    rng = np.random.default_rng(123)

    # test binary operations
    def check_cross(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        out[0] = wp.length(wp.cross(vs[0], vs[1]))

    def check_dot(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        out[0] = wp.dot(vs[0], vs[1])

    def check_mat33(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        a = vs[0]
        b = vs[1]
        c = wp.cross(a, b)
        m = wp.mat33(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2])
        out[0] = wp.determinant(m)

    def check_trace_diagonal(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        a = vs[0]
        b = vs[1]
        c = wp.cross(a, b)
        m = wp.mat33(
            1.0 / (a[0] + 10.0),
            0.0,
            0.0,
            0.0,
            1.0 / (b[1] + 10.0),
            0.0,
            0.0,
            0.0,
            1.0 / (c[2] + 10.0),
        )
        out[0] = wp.trace(m)

    def check_rot_rpy(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        v = vs[0]
        q = wp.quat_rpy(v[0], v[1], v[2])
        out[0] = wp.length(wp.quat_rotate(q, vs[1]))

    def check_rot_axis_angle(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        v = wp.normalize(vs[0])
        q = wp.quat_from_axis_angle(v, 0.5)
        out[0] = wp.length(wp.quat_rotate(q, vs[1]))

    def check_rot_quat_inv(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        v = vs[0]
        q = wp.normalize(wp.quat(v[0], v[1], v[2], 1.0))
        out[0] = wp.length(wp.quat_rotate_inv(q, vs[1]))

    # run the tests with 5 different random inputs
    for _ in range(5):
        x = wp.array(
            rng.standard_normal(size=(2, 3), dtype=np.float32), dtype=wp.vec3, device=device, requires_grad=True
        )
        gradcheck(check_cross, "check_cross_3d", [x], device)
        gradcheck(check_dot, "check_dot_3d", [x], device)
        gradcheck(check_mat33, "check_mat33_3d", [x], device, eps=2e-2)
        gradcheck(check_trace_diagonal, "check_trace_diagonal_3d", [x], device)
        gradcheck(check_rot_rpy, "check_rot_rpy_3d", [x], device)
        gradcheck(check_rot_axis_angle, "check_rot_axis_angle_3d", [x], device)
        gradcheck(check_rot_quat_inv, "check_rot_quat_inv_3d", [x], device)


def test_multi_valued_function_grad(test, device):
    rng = np.random.default_rng(123)

    @wp.func
    def multi_valued(x: float, y: float, z: float):
        return wp.sin(x), wp.cos(y) * z, wp.sqrt(wp.abs(z)) / wp.abs(x)

    # test multi-valued functions
    def check_multi_valued(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        tid = wp.tid()
        v = vs[tid]
        a, b, c = multi_valued(v[0], v[1], v[2])
        out[tid] = a + b + c

    # run the tests with 5 different random inputs
    for _ in range(5):
        x = wp.array(
            rng.standard_normal(size=(2, 3), dtype=np.float32), dtype=wp.vec3, device=device, requires_grad=True
        )
        gradcheck(check_multi_valued, "check_multi_valued_3d", [x], device)


def test_mesh_grad(test, device):
    pos = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=wp.vec3,
        device=device,
        requires_grad=True,
    )
    indices = wp.array(
        [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
        dtype=wp.int32,
        device=device,
    )

    mesh = wp.Mesh(points=pos, indices=indices)

    @wp.func
    def compute_triangle_area(mesh_id: wp.uint64, tri_id: int):
        mesh = wp.mesh_get(mesh_id)
        i, j, k = mesh.indices[tri_id * 3 + 0], mesh.indices[tri_id * 3 + 1], mesh.indices[tri_id * 3 + 2]
        a = mesh.points[i]
        b = mesh.points[j]
        c = mesh.points[k]
        return wp.length(wp.cross(b - a, c - a)) * 0.5

    @wp.kernel
    def compute_area(mesh_id: wp.uint64, out: wp.array(dtype=wp.float32)):
        wp.atomic_add(out, 0, compute_triangle_area(mesh_id, wp.tid()))

    num_tris = int(len(indices) / 3)

    # compute analytical gradient
    tape = wp.Tape()
    output = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    with tape:
        wp.launch(compute_area, dim=num_tris, inputs=[mesh.id], outputs=[output], device=device)

    tape.backward(loss=output)

    ad_grad = mesh.points.grad.numpy()

    # compute finite differences
    eps = 1e-3
    pos_np = pos.numpy()
    fd_grad = np.zeros_like(ad_grad)

    for i in range(len(pos)):
        for j in range(3):
            pos_np[i, j] += eps
            pos = wp.array(pos_np, dtype=wp.vec3, device=device)
            mesh = wp.Mesh(points=pos, indices=indices)
            output.zero_()
            wp.launch(compute_area, dim=num_tris, inputs=[mesh.id], outputs=[output], device=device)
            f1 = output.numpy()[0]
            pos_np[i, j] -= 2 * eps
            pos = wp.array(pos_np, dtype=wp.vec3, device=device)
            mesh = wp.Mesh(points=pos, indices=indices)
            output.zero_()
            wp.launch(compute_area, dim=num_tris, inputs=[mesh.id], outputs=[output], device=device)
            f2 = output.numpy()[0]
            pos_np[i, j] += eps
            fd_grad[i, j] = (f1 - f2) / (2 * eps)

    assert np.allclose(ad_grad, fd_grad, atol=1e-3)


@wp.func
def name_clash(a: float, b: float) -> float:
    return a + b


@wp.func_grad(name_clash)
def adj_name_clash(a: float, b: float, adj_ret: float):
    # names `adj_a` and `adj_b` must not clash with function args of generated function
    adj_a = 0.0
    adj_b = 0.0
    if a < 0.0:
        adj_a = adj_ret
    if b > 0.0:
        adj_b = adj_ret

    wp.adjoint[a] += adj_a
    wp.adjoint[b] += adj_b


@wp.kernel
def name_clash_kernel(
    input_a: wp.array(dtype=float),
    input_b: wp.array(dtype=float),
    output: wp.array(dtype=float),
):
    tid = wp.tid()
    output[tid] = name_clash(input_a[tid], input_b[tid])


def test_name_clash(test, device):
    # tests that no name clashes occur when variable names such as `adj_a` are used in custom gradient code
    with wp.ScopedDevice(device):
        input_a = wp.array([1.0, -2.0, 3.0], dtype=wp.float32, requires_grad=True)
        input_b = wp.array([4.0, 5.0, -6.0], dtype=wp.float32, requires_grad=True)
        output = wp.zeros(3, dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(name_clash_kernel, dim=len(input_a), inputs=[input_a, input_b], outputs=[output])

        tape.backward(grads={output: wp.array(np.ones(len(input_a), dtype=np.float32))})

        assert_np_equal(input_a.grad.numpy(), np.array([0.0, 1.0, 0.0]))
        assert_np_equal(input_b.grad.numpy(), np.array([1.0, 1.0, 0.0]))


@wp.struct
class NestedStruct:
    v: wp.vec2


@wp.struct
class ParentStruct:
    a: float
    n: NestedStruct


@wp.func
def noop(a: Any):
    pass


@wp.func
def sum2(v: wp.vec2):
    return v[0] + v[1]


@wp.kernel
def test_struct_attribute_gradient_kernel(src: wp.array(dtype=float), res: wp.array(dtype=float)):
    tid = wp.tid()

    p = ParentStruct(src[tid], NestedStruct(wp.vec2(2.0 * src[tid])))

    # test that we are not losing gradients when accessing attributes
    noop(p.a)
    noop(p.n)
    noop(p.n.v)

    res[tid] = p.a + sum2(p.n.v)


def test_struct_attribute_gradient(test, device):
    with wp.ScopedDevice(device):
        src = wp.array([1], dtype=float, requires_grad=True)
        res = wp.empty_like(src)

        tape = wp.Tape()
        with tape:
            wp.launch(test_struct_attribute_gradient_kernel, dim=1, inputs=[src, res])

        res.grad.fill_(1.0)
        tape.backward()

        test.assertEqual(src.grad.numpy()[0], 5.0)


@wp.kernel
def copy_kernel(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    ai = a[tid]
    bi = ai
    b[tid] = bi


def test_copy(test, device):
    with wp.ScopedDevice(device):
        a = wp.array([-1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True)
        b = wp.array([0.0, 0.0, 0.0], dtype=wp.float32, requires_grad=True)

        wp.launch(copy_kernel, 1, inputs=[a, b])

        b.grad = wp.array([1.0, 1.0, 1.0], dtype=wp.float32)
        wp.launch(copy_kernel, a.shape[0], inputs=[a, b], adjoint=True, adj_inputs=[None, None])

        assert_np_equal(a.grad.numpy(), np.array([1.0, 1.0, 1.0]))


@wp.kernel
def aliasing_kernel(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    x = a[tid]

    y = x
    if y > 0.0:
        y = x * x
    else:
        y = x * x * x

    b[tid] = y


def test_aliasing(test, device):
    with wp.ScopedDevice(device):
        a = wp.array([-1.0, 2.0, 3.0], dtype=wp.float32, requires_grad=True)
        b = wp.array([0.0, 0.0, 0.0], dtype=wp.float32, requires_grad=True)

        wp.launch(aliasing_kernel, 1, inputs=[a, b])

        b.grad = wp.array([1.0, 1.0, 1.0], dtype=wp.float32)
        wp.launch(aliasing_kernel, a.shape[0], inputs=[a, b], adjoint=True, adj_inputs=[None, None])

        assert_np_equal(a.grad.numpy(), np.array([3.0, 4.0, 6.0]))


@wp.kernel
def square_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = x[tid] ** 2.0


@wp.kernel
def square_slice_2d_kernel(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float), row_idx: int):
    tid = wp.tid()
    x_slice = x[row_idx]
    y_slice = y[row_idx]
    y_slice[tid] = x_slice[tid] ** 2.0


@wp.kernel
def square_slice_3d_1d_kernel(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float), slice_idx: int):
    i, j = wp.tid()
    x_slice = x[slice_idx]
    y_slice = y[slice_idx]
    y_slice[i, j] = x_slice[i, j] ** 2.0


@wp.kernel
def square_slice_3d_2d_kernel(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float), slice_i: int, slice_j: int):
    tid = wp.tid()
    x_slice = x[slice_i, slice_j]
    y_slice = y[slice_i, slice_j]
    y_slice[tid] = x_slice[tid] ** 2.0


def test_gradient_internal(test, device):
    with wp.ScopedDevice(device):
        a = wp.array([1.0, 2.0, 3.0], dtype=float, requires_grad=True)
        b = wp.array([0.0, 0.0, 0.0], dtype=float, requires_grad=True)

        wp.launch(square_kernel, dim=a.size, inputs=[a, b])

        # use internal gradients (.grad), adj_inputs are None
        b.grad = wp.array([1.0, 1.0, 1.0], dtype=float)
        wp.launch(square_kernel, dim=a.size, inputs=[a, b], adjoint=True, adj_inputs=[None, None])

        assert_np_equal(a.grad.numpy(), np.array([2.0, 4.0, 6.0]))


def test_gradient_external(test, device):
    with wp.ScopedDevice(device):
        a = wp.array([1.0, 2.0, 3.0], dtype=float, requires_grad=False)
        b = wp.array([0.0, 0.0, 0.0], dtype=float, requires_grad=False)

        wp.launch(square_kernel, dim=a.size, inputs=[a, b])

        # use external gradients passed in adj_inputs
        a_grad = wp.array([0.0, 0.0, 0.0], dtype=float)
        b_grad = wp.array([1.0, 1.0, 1.0], dtype=float)
        wp.launch(square_kernel, dim=a.size, inputs=[a, b], adjoint=True, adj_inputs=[a_grad, b_grad])

        assert_np_equal(a_grad.numpy(), np.array([2.0, 4.0, 6.0]))


def test_gradient_precedence(test, device):
    with wp.ScopedDevice(device):
        a = wp.array([1.0, 2.0, 3.0], dtype=float, requires_grad=True)
        b = wp.array([0.0, 0.0, 0.0], dtype=float, requires_grad=True)

        wp.launch(square_kernel, dim=a.size, inputs=[a, b])

        # if both internal and external gradients are present, the external one takes precedence,
        # because it's explicitly passed by the user in adj_inputs
        a_grad = wp.array([0.0, 0.0, 0.0], dtype=float)
        b_grad = wp.array([1.0, 1.0, 1.0], dtype=float)
        wp.launch(square_kernel, dim=a.size, inputs=[a, b], adjoint=True, adj_inputs=[a_grad, b_grad])

        assert_np_equal(a_grad.numpy(), np.array([2.0, 4.0, 6.0]))  # used
        assert_np_equal(a.grad.numpy(), np.array([0.0, 0.0, 0.0]))  # unused


def test_gradient_slice_2d(test, device):
    with wp.ScopedDevice(device):
        a = wp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float, requires_grad=True)
        b = wp.zeros_like(a, requires_grad=False)
        b.grad = wp.ones_like(a, requires_grad=False)

        wp.launch(square_slice_2d_kernel, dim=a.shape[1], inputs=[a, b, 1])

        # use internal gradients (.grad), adj_inputs are None
        wp.launch(square_slice_2d_kernel, dim=a.shape[1], inputs=[a, b, 1], adjoint=True, adj_inputs=[None, None, 1])

        assert_np_equal(a.grad.numpy(), np.array([[0.0, 0.0], [6.0, 8.0], [0.0, 0.0]]))


def test_gradient_slice_3d_1d(test, device):
    with wp.ScopedDevice(device):
        data = [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ],
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
            ],
        ]
        a = wp.array(data, dtype=float, requires_grad=True)
        b = wp.zeros_like(a, requires_grad=False)
        b.grad = wp.ones_like(a, requires_grad=False)

        wp.launch(square_slice_3d_1d_kernel, dim=a.shape[1:], inputs=[a, b, 1])

        # use internal gradients (.grad), adj_inputs are None
        wp.launch(
            square_slice_3d_1d_kernel, dim=a.shape[1:], inputs=[a, b, 1], adjoint=True, adj_inputs=[None, None, 1]
        )

        expected_grad = [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [11 * 2, 12 * 2, 13 * 2],
                [14 * 2, 15 * 2, 16 * 2],
                [17 * 2, 18 * 2, 19 * 2],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]
        assert_np_equal(a.grad.numpy(), np.array(expected_grad))


def test_gradient_slice_3d_2d(test, device):
    with wp.ScopedDevice(device):
        data = [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ],
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
            ],
        ]
        a = wp.array(data, dtype=float, requires_grad=True)
        b = wp.zeros_like(a, requires_grad=False)
        b.grad = wp.ones_like(a, requires_grad=False)

        wp.launch(square_slice_3d_2d_kernel, dim=a.shape[2], inputs=[a, b, 1, 1])

        # use internal gradients (.grad), adj_inputs are None
        wp.launch(
            square_slice_3d_2d_kernel, dim=a.shape[2], inputs=[a, b, 1, 1], adjoint=True, adj_inputs=[None, None, 1, 1]
        )

        expected_grad = [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [14 * 2, 15 * 2, 16 * 2],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]
        assert_np_equal(a.grad.numpy(), np.array(expected_grad))


devices = get_test_devices()


class TestGrad(unittest.TestCase):
    pass


# add_function_test(TestGrad, "test_while_loop_grad", test_while_loop_grad, devices=devices)
add_function_test(TestGrad, "test_for_loop_nested_for_grad", test_for_loop_nested_for_grad, devices=devices)
add_function_test(TestGrad, "test_scalar_grad", test_scalar_grad, devices=devices)
add_function_test(TestGrad, "test_for_loop_grad", test_for_loop_grad, devices=devices)
add_function_test(
    TestGrad, "test_for_loop_graph_grad", test_for_loop_graph_grad, devices=get_selected_cuda_test_devices()
)
add_function_test(TestGrad, "test_for_loop_nested_if_grad", test_for_loop_nested_if_grad, devices=devices)
add_function_test(TestGrad, "test_preserve_outputs_grad", test_preserve_outputs_grad, devices=devices)
add_function_test(TestGrad, "test_vector_math_grad", test_vector_math_grad, devices=devices)
add_function_test(TestGrad, "test_matrix_math_grad", test_matrix_math_grad, devices=devices)
add_function_test(TestGrad, "test_3d_math_grad", test_3d_math_grad, devices=devices)
add_function_test(TestGrad, "test_multi_valued_function_grad", test_multi_valued_function_grad, devices=devices)
add_function_test(TestGrad, "test_mesh_grad", test_mesh_grad, devices=devices)
add_function_test(TestGrad, "test_name_clash", test_name_clash, devices=devices)
add_function_test(TestGrad, "test_struct_attribute_gradient", test_struct_attribute_gradient, devices=devices)
add_function_test(TestGrad, "test_copy", test_copy, devices=devices)
add_function_test(TestGrad, "test_aliasing", test_aliasing, devices=devices)
add_function_test(TestGrad, "test_gradient_internal", test_gradient_internal, devices=devices)
add_function_test(TestGrad, "test_gradient_external", test_gradient_external, devices=devices)
add_function_test(TestGrad, "test_gradient_precedence", test_gradient_precedence, devices=devices)
add_function_test(TestGrad, "test_gradient_slice_2d", test_gradient_slice_2d, devices=devices)
add_function_test(TestGrad, "test_gradient_slice_3d_1d", test_gradient_slice_3d_1d, devices=devices)
add_function_test(TestGrad, "test_gradient_slice_3d_2d", test_gradient_slice_3d_2d, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
