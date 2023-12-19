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

wp.init()


# atomic add function that memorizes which thread incremented the counter
# so that the correct counter value per thread can be used in the replay
# phase of the backward pass
@wp.func
def reversible_increment(
    counter: wp.array(dtype=int), counter_index: int, value: int, thread_values: wp.array(dtype=int), tid: int
):
    next_index = wp.atomic_add(counter, counter_index, value)
    thread_values[tid] = next_index
    return next_index


@wp.func_replay(reversible_increment)
def replay_reversible_increment(
    counter: wp.array(dtype=int), counter_index: int, value: int, thread_values: wp.array(dtype=int), tid: int
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

    tape.backward(grads={outputs: wp.array(np.ones(num_threads, dtype=np.float32), device=device)})
    assert_np_equal(inputs.grad.numpy(), 2.0 * inputs.numpy(), tol=1e-4)


@wp.func
def overload_fn(x: float, y: float):
    return x * 3.0 + y / 3.0, y**2.5


@wp.func_grad(overload_fn)
def overload_fn_grad(x: float, y: float, adj_ret0: float, adj_ret1: float):
    wp.adjoint[x] += x * adj_ret0 * 42.0 + y * adj_ret1 * 10.0
    wp.adjoint[y] += y * adj_ret1 * 3.0


@wp.struct
class MyStruct:
    scalar: float
    vec: wp.vec3


@wp.func
def overload_fn(x: MyStruct):
    return x.vec[0] * x.vec[1] * x.vec[2] * 4.0, wp.length(x.vec), x.scalar**0.5


@wp.func_grad(overload_fn)
def overload_fn_grad(x: MyStruct, adj_ret0: float, adj_ret1: float, adj_ret2: float):
    wp.adjoint[x.scalar] += x.scalar * adj_ret0 * 10.0
    wp.adjoint[x.vec][0] += adj_ret0 * x.vec[1] * x.vec[2] * 20.0
    wp.adjoint[x.vec][1] += adj_ret1 * x.vec[0] * x.vec[2] * 30.0
    wp.adjoint[x.vec][2] += adj_ret2 * x.vec[0] * x.vec[1] * 40.0


@wp.kernel
def run_overload_float_fn(
    xs: wp.array(dtype=float), ys: wp.array(dtype=float), output0: wp.array(dtype=float), output1: wp.array(dtype=float)
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
        wp.launch(run_overload_float_fn, dim=dim, inputs=[xs_float, ys_float], outputs=[out0_float, out1_float])
    tape.backward(
        grads={
            out0_float: wp.array(np.ones(dim), dtype=wp.float32),
            out1_float: wp.array(np.ones(dim), dtype=wp.float32),
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
    xs_struct = wp.array([x0, x1, x2], dtype=MyStruct, requires_grad=True)
    out_struct = wp.zeros(dim)
    tape = wp.Tape()
    with tape:
        wp.launch(run_overload_struct_fn, dim=dim, inputs=[xs_struct], outputs=[out_struct])
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


devices = get_test_devices()


class TestGradCustoms(unittest.TestCase):
    pass


add_function_test(TestGradCustoms, "test_custom_replay_grad", test_custom_replay_grad, devices=devices)
add_function_test(TestGradCustoms, "test_custom_overload_grad", test_custom_overload_grad, devices=devices)


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
