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


@wp.kernel
def mul_constant(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()

    y[tid] = x[tid] * 2.0


@wp.struct
class Multiplicands:
    x: wp.array(dtype=float)
    y: wp.array(dtype=float)


@wp.kernel
def mul_variable(mutiplicands: Multiplicands, z: wp.array(dtype=float)):
    tid = wp.tid()

    z[tid] = mutiplicands.x[tid] * mutiplicands.y[tid]


@wp.kernel
def dot_product(x: wp.array(dtype=float), y: wp.array(dtype=float), z: wp.array(dtype=float)):
    tid = wp.tid()

    wp.atomic_add(z, 0, x[tid] * y[tid])


def test_tape_mul_constant(test, device):
    dim = 8
    iters = 16
    tape = wp.Tape()

    # record onto tape
    with tape:
        # input data
        x0 = wp.array(np.zeros(dim), dtype=wp.float32, device=device, requires_grad=True)
        x = x0

        for _i in range(iters):
            y = wp.empty_like(x, requires_grad=True)
            wp.launch(kernel=mul_constant, dim=dim, inputs=[x], outputs=[y], device=device)
            x = y

    # loss = wp.sum(x)
    x.grad = wp.array(np.ones(dim), device=device, dtype=wp.float32)

    # run backward
    tape.backward()

    # grad = 2.0^iters
    assert_np_equal(tape.gradients[x0].numpy(), np.ones(dim) * (2**iters))


def test_tape_mul_variable(test, device):
    dim = 8
    tape = wp.Tape()

    # record onto tape
    with tape:
        # input data (Note: We're intentionally testing structs in tapes here)
        multiplicands = Multiplicands()
        multiplicands.x = wp.array(np.ones(dim) * 16.0, dtype=wp.float32, device=device, requires_grad=True)
        multiplicands.y = wp.array(np.ones(dim) * 32.0, dtype=wp.float32, device=device, requires_grad=True)
        z = wp.zeros_like(multiplicands.x)

        wp.launch(kernel=mul_variable, dim=dim, inputs=[multiplicands], outputs=[z], device=device)

    # loss = wp.sum(x)
    z.grad = wp.array(np.ones(dim), device=device, dtype=wp.float32)

    # run backward
    tape.backward()

    # grad_x=y, grad_y=x
    assert_np_equal(tape.gradients[multiplicands].x.numpy(), multiplicands.y.numpy())
    assert_np_equal(tape.gradients[multiplicands].y.numpy(), multiplicands.x.numpy())

    # run backward again with different incoming gradient
    # should accumulate the same gradients again onto output
    # so gradients = 2.0*prev
    tape.backward()

    assert_np_equal(tape.gradients[multiplicands].x.numpy(), multiplicands.y.numpy() * 2.0)
    assert_np_equal(tape.gradients[multiplicands].y.numpy(), multiplicands.x.numpy() * 2.0)

    # Clear launches and zero out the gradients
    tape.reset()
    assert_np_equal(tape.gradients[multiplicands].x.numpy(), np.zeros_like(tape.gradients[multiplicands].x.numpy()))
    test.assertFalse(tape.launches)


def test_tape_dot_product(test, device):
    dim = 8
    tape = wp.Tape()

    # record onto tape
    with tape:
        # input data
        x = wp.array(np.ones(dim) * 16.0, dtype=wp.float32, device=device, requires_grad=True)
        y = wp.array(np.ones(dim) * 32.0, dtype=wp.float32, device=device, requires_grad=True)
        z = wp.zeros(n=1, dtype=wp.float32, device=device, requires_grad=True)

        wp.launch(kernel=dot_product, dim=dim, inputs=[x, y], outputs=[z], device=device)

    # scalar loss
    tape.backward(loss=z)

    # grad_x=y, grad_y=x
    assert_np_equal(tape.gradients[x].numpy(), y.numpy())
    assert_np_equal(tape.gradients[y].numpy(), x.numpy())


def test_tape_visualize(test, device):
    dim = 8
    tape = wp.Tape()

    # record onto tape
    with tape:
        # input data
        x = wp.array(np.ones(dim) * 16.0, dtype=wp.float32, device=device, requires_grad=True)
        y = wp.array(np.ones(dim) * 32.0, dtype=wp.float32, device=device, requires_grad=True)
        z = wp.zeros(n=1, dtype=wp.float32, device=device, requires_grad=True)

        tape.record_scope_begin("my loop")
        for _ in range(16):
            wp.launch(kernel=dot_product, dim=dim, inputs=[x, y], outputs=[z], device=device)
        tape.record_scope_end()

    # generate GraphViz diagram code
    dot_code = tape.visualize(simplify_graph=True)

    assert "repeated 16x" in dot_code
    assert "my loop" in dot_code
    assert dot_code.count("dot_product") == 1


devices = get_test_devices()


class TestTape(unittest.TestCase):
    def test_tape_no_nested_tapes(self):
        with self.assertRaises(RuntimeError):
            with wp.Tape():
                with wp.Tape():
                    pass


add_function_test(TestTape, "test_tape_mul_constant", test_tape_mul_constant, devices=devices)
add_function_test(TestTape, "test_tape_mul_variable", test_tape_mul_variable, devices=devices)
add_function_test(TestTape, "test_tape_dot_product", test_tape_dot_product, devices=devices)
add_function_test(TestTape, "test_tape_visualize", test_tape_visualize, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
