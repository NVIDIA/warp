# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


# atomic add function that memorizes which thread incremented the counter
# so that the correct counter value per thread can be used in the replay
# phase of the backward pass
@wp.func
def reversible_increment(
    counter: wp.array(dtype=int), counter_index: int, value: int, thread_values: wp.array(dtype=int), tid: int
):
    """This is a docstring"""
    next_index = wp.atomic_add(counter, counter_index, value)
    thread_values[tid] = next_index
    return next_index


@wp.func_replay(reversible_increment)
def replay_reversible_increment(
    counter: wp.array(dtype=int), counter_index: int, value: int, thread_values: wp.array(dtype=int), tid: int
):
    """This is a docstring"""
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
        output: wp.array(dtype=float),
    ):
        tid = wp.tid()
        idx = reversible_increment(counter, 0, 1, thread_values, tid)
        output[idx] = input[idx] ** 2.0

    tape = wp.Tape()
    with tape:
        wp.launch(
            run_atomic_add, dim=num_threads, inputs=[inputs, counter, thread_ids], outputs=[outputs], device=device
        )

    tape.backward(grads={outputs: wp.ones(num_threads, dtype=wp.float32, device=device)})
    assert_np_equal(inputs.grad.numpy(), 2.0 * inputs.numpy(), tol=1e-4)


@wp.func
def overload_fn(x: float, y: float):
    """This is a docstring"""
    return x * 3.0 + y / 3.0, y**2.5


@wp.func_grad(overload_fn)
def overload_fn_grad(x: float, y: float, adj_ret0: float, adj_ret1: float):
    """This is a docstring"""
    wp.adjoint[x] += x * adj_ret0 * 42.0 + y * adj_ret1 * 10.0
    wp.adjoint[y] += y * adj_ret1 * 3.0


@wp.struct
class MyStruct:
    """This is a docstring"""

    scalar: float
    vec: wp.vec3


@wp.func
def overload_fn(x: MyStruct):
    """This is a docstring"""
    return x.vec[0] * x.vec[1] * x.vec[2] * 4.0, wp.length(x.vec), x.scalar**0.5


@wp.func_grad(overload_fn)
def overload_fn_grad(x: MyStruct, adj_ret0: float, adj_ret1: float, adj_ret2: float):
    """This is a docstring"""
    wp.adjoint[x.scalar] += x.scalar * adj_ret0 * 10.0
    wp.adjoint[x.vec][0] += adj_ret0 * x.vec[1] * x.vec[2] * 20.0
    wp.adjoint[x.vec][1] += adj_ret1 * x.vec[0] * x.vec[2] * 30.0
    wp.adjoint[x.vec][2] += adj_ret2 * x.vec[0] * x.vec[1] * 40.0


@wp.kernel
def run_overload_float_fn(
    xs: wp.array(dtype=float), ys: wp.array(dtype=float), output0: wp.array(dtype=float), output1: wp.array(dtype=float)
):
    """This is a docstring"""
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
    xs_float = wp.array(np.arange(1.0, dim + 1.0), dtype=wp.float32, requires_grad=True, device=device)
    ys_float = wp.array(np.arange(10.0, dim + 10.0), dtype=wp.float32, requires_grad=True, device=device)
    out0_float = wp.zeros(dim, device=device)
    out1_float = wp.zeros(dim, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            run_overload_float_fn, dim=dim, inputs=[xs_float, ys_float], outputs=[out0_float, out1_float], device=device
        )
    tape.backward(
        grads={
            out0_float: wp.ones(dim, dtype=wp.float32, device=device),
            out1_float: wp.ones(dim, dtype=wp.float32, device=device),
        }
    )
    assert_np_equal(xs_float.grad.numpy(), xs_float.numpy() * 42.0 + ys_float.numpy() * 10.0)
    assert_np_equal(ys_float.grad.numpy(), ys_float.numpy() * 3.0)

    x0 = MyStruct()
    x0.vec = wp.vec3(1.0, 2.0, 3.0)
    x0.scalar = 4.0
    x1 = MyStruct()
    x1.vec = wp.vec3(5.0, 6.0, 7.0)
    x1.scalar = -1.0
    x2 = MyStruct()
    x2.vec = wp.vec3(8.0, 9.0, 10.0)
    x2.scalar = 19.0
    xs_struct = wp.array([x0, x1, x2], dtype=MyStruct, requires_grad=True, device=device)
    out_struct = wp.zeros(dim, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(run_overload_struct_fn, dim=dim, inputs=[xs_struct], outputs=[out_struct], device=device)
    tape.backward(grads={out_struct: wp.ones(dim, dtype=wp.float32, device=device)})
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


def test_custom_import_grad(test, device):
    from warp.tests.aux_test_grad_customs import aux_custom_fn

    @wp.kernel
    def run_defined_float_fn(
        xs: wp.array(dtype=float),
        ys: wp.array(dtype=float),
        output0: wp.array(dtype=float),
        output1: wp.array(dtype=float),
    ):
        i = wp.tid()
        out0, out1 = aux_custom_fn(xs[i], ys[i])
        output0[i] = out0
        output1[i] = out1

    dim = 3
    xs_float = wp.array(np.arange(1.0, dim + 1.0), dtype=wp.float32, requires_grad=True, device=device)
    ys_float = wp.array(np.arange(10.0, dim + 10.0), dtype=wp.float32, requires_grad=True, device=device)
    out0_float = wp.zeros(dim, device=device)
    out1_float = wp.zeros(dim, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(
            run_defined_float_fn, dim=dim, inputs=[xs_float, ys_float], outputs=[out0_float, out1_float], device=device
        )
    tape.backward(
        grads={
            out0_float: wp.ones(dim, dtype=wp.float32, device=device),
            out1_float: wp.ones(dim, dtype=wp.float32, device=device),
        }
    )
    assert_np_equal(xs_float.grad.numpy(), xs_float.numpy() * 42.0 + ys_float.numpy() * 10.0)
    assert_np_equal(ys_float.grad.numpy(), ys_float.numpy() * 3.0)


@wp.func
def sigmoid(x: float):
    return 1.0 / (1.0 + wp.exp(-x))


@wp.func_grad(sigmoid)
def adj_sigmoid(x: float, adj: float):
    # unused function to test that we don't run into infinite recursion when calling
    # the forward function from within the gradient function
    wp.adjoint[x] += adj * sigmoid(x) * (1.0 - sigmoid(x))


@wp.func
def sigmoid_no_return(i: int, xs: wp.array(dtype=float), ys: wp.array(dtype=float)):
    # test function that does not return anything
    ys[i] = sigmoid(xs[i])


@wp.func_grad(sigmoid_no_return)
def adj_sigmoid_no_return(i: int, xs: wp.array(dtype=float), ys: wp.array(dtype=float)):
    wp.adjoint[xs][i] += ys[i] * (1.0 - ys[i])


@wp.kernel
def eval_sigmoid(xs: wp.array(dtype=float), ys: wp.array(dtype=float)):
    i = wp.tid()
    sigmoid_no_return(i, xs, ys)


def test_custom_grad_no_return(test, device):
    xs = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32, requires_grad=True, device=device)
    ys = wp.zeros_like(xs, device=device)
    ys.grad.fill_(1.0)

    tape = wp.Tape()
    with tape:
        wp.launch(eval_sigmoid, dim=len(xs), inputs=[xs], outputs=[ys], device=device)
    tape.backward()

    sigmoids = ys.numpy()
    grad = xs.grad.numpy()
    assert_np_equal(grad, sigmoids * (1.0 - sigmoids))


@wp.func
def dense_gemm(
    m: int,
    n: int,
    p: int,
    transpose_A: bool,
    transpose_B: bool,
    add_to_C: bool,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    # outputs
    C: wp.array(dtype=float),
):
    # this function doesn't get called but it is an important test for code generation
    # multiply a `m x p` matrix A by a `p x n` matrix B to produce a `m x n` matrix C
    for i in range(m):
        for j in range(n):
            sum = float(0.0)
            for k in range(p):
                if transpose_A:
                    a_i = k * m + i
                else:
                    a_i = i * p + k
                if transpose_B:
                    b_j = j * p + k
                else:
                    b_j = k * n + j
                sum += A[a_i] * B[b_j]

            if add_to_C:
                C[i * n + j] += sum
            else:
                C[i * n + j] = sum


@wp.func_grad(dense_gemm)
def adj_dense_gemm(
    m: int,
    n: int,
    p: int,
    transpose_A: bool,
    transpose_B: bool,
    add_to_C: bool,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    # outputs
    C: wp.array(dtype=float),
):
    # code generation would break here if we didn't defer building the custom grad
    # function until after the forward functions + kernels of the module have been built
    add_to_C = True
    if transpose_A:
        dense_gemm(p, m, n, False, True, add_to_C, B, wp.adjoint[C], wp.adjoint[A])
        dense_gemm(p, n, m, False, False, add_to_C, A, wp.adjoint[C], wp.adjoint[B])
    else:
        dense_gemm(m, p, n, False, not transpose_B, add_to_C, wp.adjoint[C], B, wp.adjoint[A])
        dense_gemm(p, n, m, True, False, add_to_C, A, wp.adjoint[C], wp.adjoint[B])


devices = get_test_devices()


class TestGradCustoms(unittest.TestCase):
    def test_wrapped_docstring(self):
        self.assertTrue("This is a docstring" in reversible_increment.__doc__)
        self.assertTrue("This is a docstring" in replay_reversible_increment.__doc__)
        self.assertTrue("This is a docstring" in overload_fn.__doc__)
        self.assertTrue("This is a docstring" in overload_fn_grad.__doc__)
        self.assertTrue("This is a docstring" in run_overload_float_fn.__doc__)
        self.assertTrue("This is a docstring" in MyStruct.__doc__)


add_function_test(TestGradCustoms, "test_custom_replay_grad", test_custom_replay_grad, devices=devices)
add_function_test(TestGradCustoms, "test_custom_overload_grad", test_custom_overload_grad, devices=devices)
add_function_test(TestGradCustoms, "test_custom_import_grad", test_custom_import_grad, devices=devices)
add_function_test(TestGradCustoms, "test_custom_grad_no_return", test_custom_grad_no_return, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
