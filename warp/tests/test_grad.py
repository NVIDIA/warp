# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
from warp.tests.test_base import *

wp.init()


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
    n = 32
    val = np.ones(n, dtype=np.float32)

    x = wp.array(val, device=device, requires_grad=True)
    sum = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    wp.force_load()

    wp.capture_begin()

    tape = wp.Tape()
    with tape:
        wp.launch(for_loop_grad, dim=1, inputs=[n, x, sum], device=device)

    tape.backward(loss=sum)

    graph = wp.capture_end()

    wp.capture_launch(graph)
    wp.synchronize()

    # ensure forward pass outputs persist
    assert_np_equal(sum.numpy(), 2.0 * np.sum(x.numpy()))
    # ensure gradients correct
    assert_np_equal(x.grad.numpy(), 2.0 * val)

    wp.capture_launch(graph)
    wp.synchronize()


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

    module = wp.get_module(func.__module__)
    kernel = wp.Kernel(func=func, key=func_name, module=module)

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
    np.random.seed(123)

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
            x = wp.array(np.random.randn(1, dim).astype(np.float32), dtype=vec_type, device=device)
            gradcheck(check_length, f"check_length_{vec_type.__name__}", [x], device)
            gradcheck(check_length_sq, f"check_length_sq_{vec_type.__name__}", [x], device)
            gradcheck(check_normalize, f"check_normalize_{vec_type.__name__}", [x], device)


def test_matrix_math_grad(test, device):
    np.random.seed(123)

    # test unary operations
    for dim, mat_type in [(2, wp.mat22), (3, wp.mat33), (4, wp.mat44)]:

        def check_determinant(vs: wp.array(dtype=mat_type), out: wp.array(dtype=float)):
            out[0] = wp.determinant(vs[0])

        def check_trace(vs: wp.array(dtype=mat_type), out: wp.array(dtype=float)):
            out[0] = wp.trace(vs[0])

        # run the tests with 5 different random inputs
        for _ in range(5):
            x = wp.array(np.random.randn(1, dim, dim).astype(np.float32), ndim=1, dtype=mat_type, device=device)
            gradcheck(check_determinant, f"check_length_{mat_type.__name__}", [x], device)
            gradcheck(check_trace, f"check_length_sq_{mat_type.__name__}", [x], device)


def test_3d_math_grad(test, device):
    np.random.seed(123)

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
        x = wp.array(np.random.randn(2, 3).astype(np.float32), dtype=wp.vec3, device=device, requires_grad=True)
        gradcheck(check_cross, "check_cross_3d", [x], device)
        gradcheck(check_dot, "check_dot_3d", [x], device)
        gradcheck(check_mat33, "check_mat33_3d", [x], device, eps=2e-2)
        gradcheck(check_trace_diagonal, "check_trace_diagonal_3d", [x], device)
        gradcheck(check_rot_rpy, "check_rot_rpy_3d", [x], device)
        gradcheck(check_rot_axis_angle, "check_rot_axis_angle_3d", [x], device)
        gradcheck(check_rot_quat_inv, "check_rot_quat_inv_3d", [x], device)


def test_multi_valued_function_grad(test, device):
    np.random.seed(123)

    @wp.func
    def multi_valued(x: float, y: float, z: float):
        return wp.sin(x), wp.cos(y) * z, wp.sqrt(z) / wp.abs(x)

    # test multi-valued functions
    def check_multi_valued(vs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        tid = wp.tid()
        v = vs[tid]
        a, b, c = multi_valued(v[0], v[1], v[2])
        out[tid] = a + b + c

    # run the tests with 5 different random inputs
    for _ in range(5):
        x = wp.array(np.random.randn(2, 3).astype(np.float32), dtype=wp.vec3, device=device, requires_grad=True)
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

    def compute_area(mesh_id: wp.uint64, out: wp.array(dtype=wp.float32)):
        wp.atomic_add(out, 0, compute_triangle_area(mesh_id, wp.tid()))

    module = wp.get_module(compute_area.__module__)
    kernel = wp.Kernel(func=compute_area, key="compute_area", module=module)

    num_tris = int(len(indices) / 3)

    # compute analytical gradient
    tape = wp.Tape()
    output = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    with tape:
        wp.launch(kernel, dim=num_tris, inputs=[mesh.id], outputs=[output], device=device)

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
            wp.launch(kernel, dim=num_tris, inputs=[mesh.id], outputs=[output], device=device)
            f1 = output.numpy()[0]
            pos_np[i, j] -= 2 * eps
            pos = wp.array(pos_np, dtype=wp.vec3, device=device)
            mesh = wp.Mesh(points=pos, indices=indices)
            output.zero_()
            wp.launch(kernel, dim=num_tris, inputs=[mesh.id], outputs=[output], device=device)
            f2 = output.numpy()[0]
            pos_np[i, j] += eps
            fd_grad[i, j] = (f1 - f2) / (2 * eps)

    assert np.allclose(ad_grad, fd_grad, atol=1e-3)


# atomic add function that memorizes which thread incremented the counter
# so that the correct counter value per thread can be used in the replay
# phase of the backward pass
@wp.func
def differentiable_atomic_add(
    counter: wp.array(dtype=int),
    counter_index: int,
    value: int,
    thread_values: wp.array(dtype=int),
    tid: int
):
    next_index = wp.atomic_add(counter, counter_index, value)
    thread_values[tid] = next_index
    return next_index


@differentiable_atomic_add.replay
def replay_differentiable_atomic_add(
    counter: wp.array(dtype=int),
    counter_index: int,
    value: int,
    thread_values: wp.array(dtype=int),
    tid: int
):
    return thread_values[tid]


def test_custom_replay_grad(test, device):
    num_threads = 128
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    thread_ids = wp.zeros(num_threads, dtype=wp.int32, device=device)
    inputs = wp.array(np.arange(num_threads, dtype=np.float32), device=device, requires_grad=True)
    outputs = wp.zeros_like(inputs)

    @wp.kernel
    def run_atomic_add(
        input: wp.array(dtype=float),
        counter: wp.array(dtype=int),
        thread_values: wp.array(dtype=int),
        output: wp.array(dtype=float)
    ):
        tid = wp.tid()
        idx = differentiable_atomic_add(counter, 0, 1, thread_values, tid)
        output[idx] = input[idx] ** 2.0

    tape = wp.Tape()
    with tape:
        wp.launch(run_atomic_add, dim=num_threads, inputs=[inputs, counter, thread_ids], outputs=[outputs], device=device)

    tape.backward(grads={outputs: wp.array(np.ones(num_threads, dtype=np.float32), device=device)})
    assert_np_equal(inputs.grad.numpy(), 2.0 * inputs.numpy(), tol=1e-4)


@wp.func
def overload_fn(x: float, y: float):
    return x * 3.0 + y / 3.0, y ** 2.5


@overload_fn.grad
def overload_fn_grad(x: float, y: float, adj_ret0: float, adj_ret1: float):
    wp.adjoint[x] += x * adj_ret0 * 42.0 + y * adj_ret1 * 10.0
    wp.adjoint[y] += y * adj_ret1 * 3.0


@wp.struct
class MyStruct:
    scalar: float
    vec: wp.vec3


@wp.func
def overload_fn(x: MyStruct):
    return x.vec[0] * x.vec[1] * x.vec[2] * 4.0, wp.length(x.vec), x.scalar ** 0.5


@overload_fn.grad
def overload_fn_grad(x: MyStruct, adj_ret0: float, adj_ret1: float, adj_ret2: float):
    wp.adjoint[x.scalar] += x.scalar * adj_ret0 * 10.0
    wp.adjoint[x.vec][0] += adj_ret0 * x.vec[1] * x.vec[2] * 20.0
    wp.adjoint[x.vec][1] += adj_ret1 * x.vec[0] * x.vec[2] * 30.0
    wp.adjoint[x.vec][2] += adj_ret2 * x.vec[0] * x.vec[1] * 40.0


@wp.kernel
def run_overload_float_fn(
    xs: wp.array(dtype=float),
    ys: wp.array(dtype=float),
    output0: wp.array(dtype=float),
    output1: wp.array(dtype=float)
):
    i = wp.tid()
    out0, out1 = overload_fn(xs[i], ys[i])
    output0[i] = out0
    output1[i] = out1


@wp.kernel
def run_overload_struct_fn(xs: wp.array(dtype=MyStruct), output: wp.array(dtype=float)):
    i = wp.tid()
    out0, out1, out2 = overload_fn(xs[i])
    output[i] = out0 + out1 + out2


def test_custom_overload_grad(test, device):
    dim = 3
    xs_float = wp.array(np.arange(1.0, dim + 1.0), dtype=wp.float32, requires_grad=True)
    ys_float = wp.array(np.arange(10.0, dim + 10.0), dtype=wp.float32, requires_grad=True)
    out0_float = wp.zeros(dim)
    out1_float = wp.zeros(dim)
    tape = wp.Tape()
    with tape:
        wp.launch(
            run_overload_float_fn,
            dim=dim,
            inputs=[xs_float, ys_float],
            outputs=[out0_float, out1_float])
    tape.backward(grads={
        out0_float: wp.array(np.ones(dim), dtype=wp.float32),
        out1_float: wp.array(np.ones(dim), dtype=wp.float32)})
    assert_np_equal(xs_float.grad.numpy(), xs_float.numpy() * 42.0 + ys_float.numpy() * 10.0)
    assert_np_equal(ys_float.grad.numpy(), ys_float.numpy() * 3.0)

    x0 = MyStruct()
    x0.vec = wp.vec3(1., 2., 3.)
    x0.scalar = 4.
    x1 = MyStruct()
    x1.vec = wp.vec3(5., 6., 7.)
    x1.scalar = -1.0
    x2 = MyStruct()
    x2.vec = wp.vec3(8., 9., 10.)
    x2.scalar = 19.0
    xs_struct = wp.array([x0, x1, x2], dtype=MyStruct, requires_grad=True)
    out_struct = wp.zeros(dim)
    tape = wp.Tape()
    with tape:
        wp.launch(
            run_overload_struct_fn,
            dim=dim,
            inputs=[xs_struct],
            outputs=[out_struct])
    tape.backward(grads={out_struct: wp.array(np.ones(dim), dtype=wp.float32)})
    xs_struct_np = xs_struct.numpy()
    struct_grads = xs_struct.grad.numpy()
    # fmt: off
    assert_np_equal(
        np.array([g[0] for g in struct_grads]),
        np.array([g[0] * 10.0 for g in xs_struct_np]))
    assert_np_equal(
        np.array([g[1][0] for g in struct_grads]),
        np.array([g[1][1] * g[1][2] * 20.0 for g in xs_struct_np]))
    assert_np_equal(
        np.array([g[1][1] for g in struct_grads]),
        np.array([g[1][0] * g[1][2] * 30.0 for g in xs_struct_np]))
    assert_np_equal(
        np.array([g[1][2] for g in struct_grads]),
        np.array([g[1][0] * g[1][1] * 40.0 for g in xs_struct_np]))
    # fmt: on


def register(parent):
    devices = get_test_devices()

    class TestGrad(parent):
        pass

    # add_function_test(TestGrad, "test_while_loop_grad", test_while_loop_grad, devices=devices)
    add_function_test(TestGrad, "test_for_loop_nested_for_grad", test_for_loop_nested_for_grad, devices=devices)
    add_function_test(TestGrad, "test_scalar_grad", test_scalar_grad, devices=devices)
    add_function_test(TestGrad, "test_for_loop_grad", test_for_loop_grad, devices=devices)
    add_function_test(TestGrad, "test_for_loop_graph_grad", test_for_loop_graph_grad, devices=wp.get_cuda_devices())
    add_function_test(TestGrad, "test_for_loop_nested_if_grad", test_for_loop_nested_if_grad, devices=devices)
    add_function_test(TestGrad, "test_preserve_outputs_grad", test_preserve_outputs_grad, devices=devices)
    add_function_test(TestGrad, "test_vector_math_grad", test_vector_math_grad, devices=devices)
    add_function_test(TestGrad, "test_matrix_math_grad", test_matrix_math_grad, devices=devices)
    add_function_test(TestGrad, "test_3d_math_grad", test_3d_math_grad, devices=devices)
    add_function_test(TestGrad, "test_multi_valued_function_grad", test_multi_valued_function_grad, devices=devices)
    add_function_test(TestGrad, "test_mesh_grad", test_mesh_grad, devices=devices)
    add_function_test(TestGrad, "test_custom_replay_grad", test_custom_replay_grad, devices=devices)
    add_function_test(TestGrad, "test_custom_overload_grad", test_custom_overload_grad, devices=devices)

    return TestGrad


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
