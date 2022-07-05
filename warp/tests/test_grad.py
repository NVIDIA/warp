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
def scalar_grad(x: wp.array(dtype=float),
                y: wp.array(dtype=float)):

    y[0] = x[0]**2.0



def test_scalar_grad(test, device):

    x = wp.array([3.0], dtype=float, device=device, requires_grad=True)
    y = wp.zeros_like(x)

    tape = wp.Tape()
    with tape:
        wp.launch(scalar_grad, dim=1, inputs=[x, y], device=device)

    tape.backward(y)

    assert_np_equal(tape.gradients[x].numpy(), np.array(6.0))
   



@wp.kernel
def for_loop_grad(n: int, 
                  x: wp.array(dtype=float),
                  s: wp.array(dtype=float)):

    sum = float(0.0)

    for i in range(n):
        sum = sum + x[i]*2.0

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
    assert_np_equal(sum.numpy(), 2.0*np.sum(x.numpy()))

    tape.backward(loss=sum)
    
    # ensure forward pass outputs persist
    assert_np_equal(sum.numpy(), 2.0*np.sum(x.numpy()))
    # ensure gradients correct
    assert_np_equal(tape.gradients[x].numpy(), 2.0*val)


def test_for_loop_graph_grad(test, device):

    n = 32
    val = np.ones(n, dtype=np.float32)

    x = wp.array(val, device=device, requires_grad=True)
    sum = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    wp.capture_begin()

    tape = wp.Tape()
    with tape:
        wp.launch(for_loop_grad, dim=1, inputs=[n, x, sum], device=device)
   
    tape.backward(loss=sum)

    graph = wp.capture_end()

    wp.capture_launch(graph)
    wp.synchronize()
    
    # ensure forward pass outputs persist
    assert_np_equal(sum.numpy(), 2.0*np.sum(x.numpy()))
    # ensure gradients correct
    assert_np_equal(x.grad.numpy(), 2.0*val)

    wp.capture_launch(graph)
    wp.synchronize()    

@wp.kernel
def for_loop_nested_if_grad(n: int, 
                            x: wp.array(dtype=float),
                            s: wp.array(dtype=float)):

    sum = float(0.0)

    for i in range(n):

        if i < 16:
            if i < 8:
                sum = sum + x[i]*2.0
            else:
                sum = sum + x[i]*4.0
        else:
            if i < 24:
                sum = sum + x[i]*6.0
            else:
                sum = sum + x[i]*8.0


    s[0] = sum


def test_for_loop_nested_if_grad(test, device):

    n = 32
    val = np.ones(n, dtype=np.float32)

    expected_val = [2., 2., 2., 2., 2., 2., 2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 6., 6., 6., 6., 6., 6., 6., 6., 8., 8., 8., 8., 8., 8., 8., 8.]
    expected_grad = [2., 2., 2., 2., 2., 2., 2., 2., 4., 4., 4., 4., 4., 4., 4., 4., 6., 6., 6., 6., 6., 6., 6., 6., 8., 8., 8., 8., 8., 8., 8., 8.]

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
def for_loop_grad_nested(n: int, 
                         x: wp.array(dtype=float),
                         s: wp.array(dtype=float)):

    sum = float(0.0)

    for i in range(n):
        for j in range(n):
            sum = sum + x[i*n + j]*float(i*n + j) + 1.0

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
def preserve_outputs(n: int, 
                     x: wp.array(dtype=float),
                     c: wp.array(dtype=float),
                     s1: wp.array(dtype=float),
                     s2: wp.array(dtype=float)):

    tid = wp.tid()

    # plain store
    c[tid] = x[tid]*2.0

    # atomic stores
    wp.atomic_add(s1, 0, x[tid]*3.0)
    wp.atomic_sub(s2, 0, x[tid]*2.0)


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
    assert_np_equal(c.numpy(), val*2.0)
    assert_np_equal(s1.numpy(), np.array(3.0*n))
    assert_np_equal(s2.numpy(), np.array(-2.0*n))
    
    # run backward on first loss
    tape.backward(loss=s1)

    # ensure inputs, copy and sum are unchanged by backwards pass
    assert_np_equal(x.numpy(), val)
    assert_np_equal(c.numpy(), val*2.0)
    assert_np_equal(s1.numpy(), np.array(3.0*n))
    assert_np_equal(s2.numpy(), np.array(-2.0*n))

    # ensure gradients are correct
    assert_np_equal(tape.gradients[x].numpy(), 3.0*val)

    # run backward on second loss
    tape.zero()
    tape.backward(loss=s2)

    assert_np_equal(x.numpy(), val)
    assert_np_equal(c.numpy(), val*2.0)
    assert_np_equal(s1.numpy(), np.array(3.0*n))
    assert_np_equal(s2.numpy(), np.array(-2.0*n))

    # ensure gradients are correct
    assert_np_equal(tape.gradients[x].numpy(), -2.0*val)


def register(parent):

    devices = wp.get_devices()

    class TestGrad(parent):
        pass

    #add_function_test(TestGrad, "test_while_loop_grad", test_while_loop_grad, devices=devices)
    add_function_test(TestGrad, "test_for_loop_nested_for_grad", test_for_loop_nested_for_grad, devices=devices)
    add_function_test(TestGrad, "test_scalar_grad", test_scalar_grad, devices=devices)
    add_function_test(TestGrad, "test_for_loop_grad", test_for_loop_grad, devices=devices)
    if wp.is_cuda_available():
        add_function_test(TestGrad, "test_for_loop_graph_grad", test_for_loop_graph_grad, devices=["cuda"])
    add_function_test(TestGrad, "test_for_loop_nested_if_grad", test_for_loop_nested_if_grad, devices=devices)
    add_function_test(TestGrad, "test_preserve_outputs_grad", test_preserve_outputs_grad, devices=devices)

    return TestGrad

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
