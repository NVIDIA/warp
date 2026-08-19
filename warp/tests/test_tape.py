# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def mul_constant(x: wp.array[float], y: wp.array[float]):
    tid = wp.tid()

    y[tid] = x[tid] * 2.0


@wp.struct
class Multiplicands:
    x: wp.array[float]
    y: wp.array[float]


@wp.kernel
def mul_variable(mutiplicands: Multiplicands, z: wp.array[float]):
    tid = wp.tid()

    z[tid] = mutiplicands.x[tid] * mutiplicands.y[tid]


@wp.kernel
def dot_product(x: wp.array[float], y: wp.array[float], z: wp.array[float]):
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

    # run backward with loss = wp.sum(z)
    tape.backward(grads={z: wp.ones_like(z)})

    # grad_x=y, grad_y=x
    assert_np_equal(tape.gradients[multiplicands].x.numpy(), multiplicands.y.numpy())
    assert_np_equal(tape.gradients[multiplicands].y.numpy(), multiplicands.x.numpy())

    # run backward again with different incoming gradient
    # should accumulate the same gradients again onto output
    # so gradients = 2.0*prev
    tape.backward(grads={z: wp.ones_like(z)})

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


@wp.kernel
def assign_chain_kernel(x: wp.array[float], y: wp.array[float], z: wp.array[float]):
    tid = wp.tid()
    y[tid] = x[tid]
    z[tid] = y[tid]


def test_tape_zero_multiple_outputs(test, device):
    x = wp.array(np.arange(3), dtype=float, device=device, requires_grad=True)
    y = wp.zeros_like(x)
    z = wp.zeros_like(x)

    tape = wp.Tape()
    with tape:
        wp.launch(assign_chain_kernel, dim=3, inputs=[x, y, z], device=device)

    tape.backward(grads={y: wp.ones_like(x)})
    assert_np_equal(x.grad.numpy(), np.ones(3, dtype=float))
    tape.zero()

    tape.backward(grads={z: wp.ones_like(x)})
    assert_np_equal(x.grad.numpy(), np.ones(3, dtype=float))


@wp.struct
class NestedStruct:
    arr: wp.array[float]


@wp.struct
class WrapperStruct:
    nested: NestedStruct


@wp.kernel
def nested_loss_kernel(wrapper: WrapperStruct, loss: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(loss, 0, wrapper.nested.arr[i])


def test_tape_nested_struct(test, device):
    wrapper = WrapperStruct()
    wrapper.nested = NestedStruct()
    wrapper.nested.arr = wp.ones(shape=(1,), dtype=float, requires_grad=True, device=device)

    loss = wp.zeros(shape=(1,), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(nested_loss_kernel, dim=1, inputs=(wrapper, loss), device=device)

    assert_np_equal(loss.numpy(), [1.0])

    tape.backward(loss)
    assert_np_equal(wrapper.nested.arr.grad.numpy(), [1.0])

    tape.zero()

    assert_np_equal(wrapper.nested.arr.grad.numpy(), [0.0])


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


@wp.kernel
def dot_product_subscript(x: wp.array[float], y: wp.array[float], z: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(z, 0, x[tid] * y[tid])


# Subscript-style type hint variants (wp.array[dtype] syntax)
@wp.struct
class MultiplicandsSubscript:
    x: wp.array[float]
    y: wp.array[float]


@wp.kernel
def mul_variable_subscript(multiplicands: MultiplicandsSubscript, z: wp.array[float]):
    tid = wp.tid()
    z[tid] = multiplicands.x[tid] * multiplicands.y[tid]


@wp.struct
class NestedStructSubscript:
    arr: wp.array[float]


@wp.struct
class WrapperStructSubscript:
    nested: NestedStructSubscript


@wp.kernel
def nested_loss_kernel_subscript(wrapper: WrapperStructSubscript, loss: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(loss, 0, wrapper.nested.arr[i])


def test_tape_struct_subscript(test, device):
    """Test that struct fields using wp.array[float] subscript syntax work with Tape.backward() and Tape.zero()."""
    dim = 8
    tape = wp.Tape()

    with tape:
        multiplicands = MultiplicandsSubscript()
        multiplicands.x = wp.array(np.ones(dim) * 16.0, dtype=wp.float32, device=device, requires_grad=True)
        multiplicands.y = wp.array(np.ones(dim) * 32.0, dtype=wp.float32, device=device, requires_grad=True)
        z = wp.zeros_like(multiplicands.x)

        wp.launch(kernel=mul_variable_subscript, dim=dim, inputs=[multiplicands], outputs=[z], device=device)

    z.grad = wp.array(np.ones(dim), device=device, dtype=wp.float32)
    tape.backward()

    # grad_x=y, grad_y=x
    assert_np_equal(tape.gradients[multiplicands].x.numpy(), multiplicands.y.numpy())
    assert_np_equal(tape.gradients[multiplicands].y.numpy(), multiplicands.x.numpy())

    # zero should reset struct field gradients
    tape.zero()
    assert_np_equal(tape.gradients[multiplicands].x.numpy(), np.zeros(dim))
    assert_np_equal(tape.gradients[multiplicands].y.numpy(), np.zeros(dim))


def test_tape_nested_struct_subscript(test, device):
    """Test that nested struct fields using wp.array[float] subscript syntax work with Tape."""
    wrapper = WrapperStructSubscript()
    wrapper.nested = NestedStructSubscript()
    wrapper.nested.arr = wp.ones(shape=(1,), dtype=float, requires_grad=True, device=device)

    loss = wp.zeros(shape=(1,), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(nested_loss_kernel_subscript, dim=1, inputs=(wrapper, loss), device=device)

    assert_np_equal(loss.numpy(), np.ones(1))

    tape.backward(loss)
    assert_np_equal(wrapper.nested.arr.grad.numpy(), np.ones(1))

    tape.zero()
    assert_np_equal(wrapper.nested.arr.grad.numpy(), np.zeros(1))


def test_tape_visualize_subscript(test, device):
    """Test that tape visualization works with kernels using wp.array[float] subscript syntax."""
    dim = 8
    tape = wp.Tape()

    with tape:
        x = wp.array(np.ones(dim) * 16.0, dtype=wp.float32, device=device, requires_grad=True)
        y = wp.array(np.ones(dim) * 32.0, dtype=wp.float32, device=device, requires_grad=True)
        z = wp.zeros(n=1, dtype=wp.float32, device=device, requires_grad=True)

        wp.launch(kernel=dot_product_subscript, dim=dim, inputs=[x, y], outputs=[z], device=device)

    dot_code = tape.visualize()

    # Array args should get "array: dtype=..." tooltip, not fall through to the scalar branch
    test.assertIn("array: dtype=", dot_code)


@wp.kernel
def sum_kernel(x: wp.array[float], total: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(total, 0, x[tid])


@wp.kernel
def sum_kernel_2d(x: wp.array2d[float], total: wp.array[float]):
    i, j = wp.tid()
    wp.atomic_add(total, 0, x[i, j])


@wp.kernel
def sum_kernel_4d(x: wp.array4d[float], total: wp.array[float]):
    i, j, k, l = wp.tid()
    wp.atomic_add(total, 0, x[i, j, k, l])


@wp.kernel
def sum_kernel_vec(x: wp.array[wp.vec3], total: wp.array[float]):
    tid = wp.tid()
    v = x[tid]
    wp.atomic_add(total, 0, v[0] + v[1] + v[2])


@wp.struct
class CopyAdjointStruct:
    a: float
    v: wp.vec3


@wp.kernel
def sum_kernel_struct(xs: wp.array[CopyAdjointStruct], total: wp.array[float]):
    tid = wp.tid()
    s = xs[tid]
    wp.atomic_add(total, 0, s.a + s.v[0] + s.v[1] + s.v[2])


def test_tape_copy_adjoint_accumulation(test, device):
    """Verify the copy adjoint accumulates into the source gradient rather than overwriting it."""
    n = 4

    # the copy's source is also read by a later kernel
    x = wp.array(np.arange(1.0, n + 1), dtype=float, requires_grad=True, device=device)
    doubled = wp.zeros_like(x)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        cloned = wp.clone(x)
        wp.launch(mul_constant, dim=n, inputs=[x, doubled], device=device)
        wp.launch(sum_kernel, dim=n, inputs=[doubled], outputs=[loss], device=device)
        wp.launch(sum_kernel, dim=n, inputs=[cloned], outputs=[loss], device=device)
    tape.backward(loss)

    # dL/dx = 2 (through the kernel) + 1 (through the clone)
    assert_np_equal(x.grad.numpy(), np.full(n, 3.0))

    # iterated clone-then-overwrite: the clone path is dead (its result is fully
    # overwritten), so only the kernel path contributes
    x = wp.array(np.arange(1.0, n + 1), dtype=float, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        q = x
        for _ in range(3):
            out = wp.clone(q)
            wp.launch(mul_constant, dim=n, inputs=[q, out], device=device)
            q = out
        wp.launch(sum_kernel, dim=n, inputs=[q], outputs=[loss], device=device)
    tape.backward(loss)

    assert_np_equal(x.grad.numpy(), np.full(n, 8.0))

    # array.assign() records the same copy adjoint
    x = wp.array(np.arange(1.0, n + 1), dtype=float, requires_grad=True, device=device)
    doubled = wp.zeros_like(x)
    dst = wp.zeros_like(x)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        dst.assign(x)
        wp.launch(mul_constant, dim=n, inputs=[x, doubled], device=device)
        wp.launch(sum_kernel, dim=n, inputs=[doubled], outputs=[loss], device=device)
        wp.launch(sum_kernel, dim=n, inputs=[dst], outputs=[loss], device=device)
    tape.backward(loss)

    assert_np_equal(x.grad.numpy(), np.full(n, 3.0))

    # differently-spelled but equal dtypes take the accumulation path too
    xv = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, requires_grad=True, device=device)
    dstv = wp.zeros(n, dtype=wp.types.vector(length=3, dtype=wp.float32), requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(dstv, xv)
        wp.launch(sum_kernel_vec, dim=n, inputs=[xv], outputs=[loss], device=device)
        wp.launch(sum_kernel_vec, dim=n, inputs=[dstv], outputs=[loss], device=device)
    tape.backward(loss)

    # dL/dxv = 1 (direct read) + 1 (through the copy); an overwrite would yield 1
    assert_np_equal(xv.grad.numpy(), np.full((n, 3), 2.0))


def test_tape_copy_adjoint_consumption(test, device):
    """Verify the copy adjoint consumes (zeroes) the destination gradient, enforcing final-write-wins for dead writes."""
    n = 4

    # a copy that overwrites a kernel-written array makes the kernel's write
    # dead: its adjoint must be consumed (zeroed), matching kernel-adjoint
    # final-write-wins semantics
    x = wp.array(np.ones(n), dtype=float, requires_grad=True, device=device)
    s = wp.array(np.full(n, 5.0), dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mul_constant, dim=n, inputs=[x], outputs=[y], device=device)  # dead write
        wp.copy(y, s)  # final write
        wp.launch(sum_kernel, dim=n, inputs=[y], outputs=[loss], device=device)
    tape.backward(loss)

    assert_np_equal(s.grad.numpy(), np.ones(n))
    assert_np_equal(x.grad.numpy(), np.zeros(n))

    # two copies into the same destination: only the final one propagates
    a = wp.array(np.ones(n), dtype=float, requires_grad=True, device=device)
    b = wp.array(np.full(n, 2.0), dtype=float, requires_grad=True, device=device)
    dst = wp.zeros_like(a)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(dst, a)
        wp.copy(dst, b)
        wp.launch(sum_kernel, dim=n, inputs=[dst], outputs=[loss], device=device)
    tape.backward(loss)

    assert_np_equal(b.grad.numpy(), np.ones(n))
    assert_np_equal(a.grad.numpy(), np.zeros(n))

    # a partial-region copy consumes only the overwritten region's adjoint
    x = wp.array(np.ones(n), dtype=float, requires_grad=True, device=device)
    s2 = wp.array(np.full(2, 5.0), dtype=float, requires_grad=True, device=device)
    y = wp.zeros_like(x)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mul_constant, dim=n, inputs=[x], outputs=[y], device=device)
        wp.copy(y, s2, dest_offset=1, src_offset=0, count=2)  # overwrites y[1:3]
        wp.launch(sum_kernel, dim=n, inputs=[y], outputs=[loss], device=device)
    tape.backward(loss)

    assert_np_equal(s2.grad.numpy(), np.ones(2))
    assert_np_equal(x.grad.numpy(), np.array([2.0, 0.0, 0.0, 2.0]))

    # retain_grad on the destination skips the consumption zeroing, matching
    # the documented kernel-adjoint behavior for retained gradients
    x = wp.array(np.ones(n), dtype=float, requires_grad=True, device=device)
    s = wp.array(np.full(n, 5.0), dtype=float, requires_grad=True, device=device)
    y = wp.zeros(n, dtype=float, requires_grad=True, retain_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mul_constant, dim=n, inputs=[x], outputs=[y], device=device)
        wp.copy(y, s)
        wp.launch(sum_kernel, dim=n, inputs=[y], outputs=[loss], device=device)
    tape.backward(loss)

    # with retained gradients the dead write's adjoint is preserved (documented
    # double-counting hazard of retain_grad on multiply-written arrays)
    assert_np_equal(s.grad.numpy(), np.ones(n))
    assert_np_equal(x.grad.numpy(), np.full(n, 2.0))


def test_tape_copy_adjoint_views_and_offsets(test, device):
    """Verify accumulation and consumption land in exactly the copied window."""
    n = 4

    # partial copy with offsets: only the copied region receives the copy's
    # adjoint contribution, accumulated on top of the kernel's contribution
    x = wp.array(np.arange(1.0, n + 1), dtype=float, requires_grad=True, device=device)
    doubled = wp.zeros_like(x)
    window = wp.zeros(2, dtype=float, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(window, x, dest_offset=0, src_offset=1, count=2)
        wp.launch(mul_constant, dim=n, inputs=[x, doubled], device=device)
        wp.launch(sum_kernel, dim=n, inputs=[doubled], outputs=[loss], device=device)
        wp.launch(sum_kernel, dim=2, inputs=[window], outputs=[loss], device=device)
    tape.backward(loss)

    # dL/dx = 2 everywhere, plus 1 on the two elements routed through the window
    assert_np_equal(x.grad.numpy(), np.array([2.0, 3.0, 3.0, 2.0]))

    # windowed copy from a strided 1D source view (logical element offsets)
    base = wp.array(np.ones((3, 4), dtype=np.float32), dtype=float, requires_grad=True, device=device)
    col = base[:, 2]
    dst = wp.zeros(2, dtype=float, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(dst, col, src_offset=1, count=2)
        wp.launch(sum_kernel, dim=2, inputs=[dst], outputs=[loss], device=device)
    tape.backward(loss)

    expected = np.zeros((3, 4), dtype=np.float32)
    expected[1:, 2] = 1.0
    assert_np_equal(base.grad.numpy(), expected)

    # non-contiguous destination: the consumption zeroing routes through the
    # strided fill path and must clear exactly the copied column
    base = wp.array(np.ones((3, 4), dtype=np.float32), dtype=float, requires_grad=True, device=device)
    col = base[:, 2]
    src = wp.array(np.ones(3, dtype=np.float32), dtype=float, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(col, src)
        wp.launch(sum_kernel_2d, dim=(3, 4), inputs=[base], outputs=[loss], device=device)
    tape.backward(loss)

    assert_np_equal(src.grad.numpy(), np.ones(3, dtype=np.float32))
    expected = np.ones((3, 4), dtype=np.float32)
    expected[:, 2] = 0.0
    assert_np_equal(base.grad.numpy(), expected)

    # a higher-rank non-contiguous view (full-array copy) via clone
    x = wp.array(np.ones((2, 3, 4)), dtype=wp.float64, requires_grad=True, device=device)
    view = x[:, 0:2, :]

    tape = wp.Tape()
    with tape:
        cloned = wp.clone(view)
    tape.backward(grads={cloned: wp.ones_like(cloned)})

    expected = np.zeros((2, 3, 4))
    expected[:, 0:2, :] = 1.0
    assert_np_equal(x.grad.numpy(), expected)

    # 4D copy adjoints accumulate and consume the full copied region
    shape = (2, 2, 3, 4)
    x = wp.array(np.ones(shape, dtype=np.float32), dtype=float, requires_grad=True, device=device)
    dst = wp.zeros_like(x)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(dst, x)
        wp.launch(sum_kernel_4d, dim=shape, inputs=[x], outputs=[loss], device=device)
        wp.launch(sum_kernel_4d, dim=shape, inputs=[dst], outputs=[loss], device=device)
    tape.backward(loss)

    assert_np_equal(x.grad.numpy(), np.full(shape, 2.0, dtype=np.float32))
    assert_np_equal(dst.grad.numpy(), np.zeros(shape, dtype=np.float32))


def test_tape_copy_adjoint_out_of_scope(test, device):
    """Verify out-of-scope copies keep the previous byte-copy adjoint."""
    n = 4

    # overlapping self-copy: the incoming all-ones gradient is left in place
    # (the analytic gradient with accumulation would be [2, 1, 1, 0])
    y = wp.array(np.arange(1.0, n + 1), dtype=float, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(y, y, dest_offset=1, src_offset=0, count=3)
        wp.launch(sum_kernel, dim=n, inputs=[y], outputs=[loss], device=device)
    with patch("warp._src.context.log_warning"):
        tape.backward(loss)

    assert_np_equal(y.grad.numpy(), np.ones(n))

    # interleaved columns of one base: byte extents overlap, so the conservative
    # check routes to the byte-copy adjoint (accumulation would give [2, 0] per row)
    base = wp.array(np.ones((n, 2), dtype=np.float32), dtype=float, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(base[:, 1], base[:, 0])
        wp.launch(sum_kernel_2d, dim=(n, 2), inputs=[base], outputs=[loss], device=device)
    with patch("warp._src.context.log_warning"):
        tape.backward(loss)

    assert_np_equal(base.grad.numpy(), np.ones((n, 2), dtype=np.float32))

    # struct dtypes: the gradient propagates by overwrite and the destination
    # gradient is not consumed
    xs = wp.zeros(n, dtype=CopyAdjointStruct, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        cloned = wp.clone(xs)
        wp.launch(sum_kernel_struct, dim=n, inputs=[cloned], outputs=[loss], device=device)
    with patch("warp._src.context.log_warning"):
        tape.backward(loss)

    assert_np_equal(xs.grad.numpy()["a"], np.ones(n, dtype=np.float32))
    assert_np_equal(cloned.grad.numpy()["a"], np.ones(n, dtype=np.float32))

    # boolean scalars and composites have no addition overloads and must not
    # fail at backward; the overwrite adjoint leaves the destination gradient
    # unconsumed
    for bool_dtype in (wp.bool, wp.types.vector(length=3, dtype=wp.bool)):
        b = wp.zeros(n, dtype=bool_dtype, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            cloned_b = wp.clone(b)
        cloned_b.grad.fill_(True)
        with patch("warp._src.context.log_warning"):
            tape.backward()

        test.assertTrue(bool(np.all(b.grad.numpy())))

    # reinterpreting same-size dtypes: the source gradient receives the
    # destination gradient's bytes and the destination gradient is not consumed
    a = wp.array(np.ones(n, dtype=np.float32), dtype=float, requires_grad=True, device=device)
    b = wp.zeros(n, dtype=wp.int32, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(b, a)
    b.grad.assign(np.ones(n, dtype=np.int32))
    with patch("warp._src.context.log_warning"):
        tape.backward()

    assert_np_equal(a.grad.numpy(), np.ones(n, dtype=np.int32).view(np.float32))
    assert_np_equal(b.grad.numpy(), np.ones(n, dtype=np.int32))


def test_tape_copy_adjoint_fallback_warning(test, device):
    """Warn when a copy adjoint cannot be fully tracked."""
    n = 4
    y = wp.array(np.arange(1.0, n + 1), dtype=float, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(y, y, dest_offset=1, src_offset=0, count=3)
        wp.launch(sum_kernel, dim=n, inputs=[y], outputs=[loss], device=device)

    with patch("warp._src.context.log_warning") as mock_log_warning:
        tape.backward(loss)

    test.assertEqual(mock_log_warning.call_count, 1)
    message = mock_log_warning.call_args.args[0]
    test.assertIn("overlapping copy", message)
    test.assertIn("cannot yet be fully tracked", message)
    test.assertIs(mock_log_warning.call_args.kwargs.get("category"), UserWarning)
    test.assertEqual(mock_log_warning.call_args.kwargs.get("stacklevel"), 5)
    test.assertIs(mock_log_warning.call_args.kwargs.get("once"), True)

    if device.is_cuda:
        base_cpu = wp.zeros((n, 2), dtype=float, requires_grad=True, device="cpu")
        src_cuda = wp.ones(n, dtype=float, requires_grad=True, device=device)
        loss_cpu = wp.zeros(1, dtype=float, requires_grad=True, device="cpu")

        tape = wp.Tape()
        with tape:
            wp.copy(base_cpu[:, 1], src_cuda)
            wp.launch(sum_kernel_2d, dim=(n, 2), inputs=[base_cpu], outputs=[loss_cpu], device="cpu")

        with patch("warp._src.context.log_warning") as mock_log_warning:
            tape.backward(loss_cpu)

        test.assertTrue(
            any(call.args and "non-contiguous copy" in call.args[0] for call in mock_log_warning.call_args_list)
        )

        x_cpu = wp.array(np.ones(n, dtype=np.float32), dtype=float, requires_grad=True, device="cpu")
        dst_cuda = wp.zeros(n, dtype=wp.int32, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.copy(dst_cuda, x_cpu)
        dst_cuda.grad.assign(np.ones(n, dtype=np.int32))

        with patch("warp._src.context.log_warning") as mock_log_warning:
            tape.backward()

        test.assertTrue(
            any(
                call.args and "copy between unsupported gradient dtypes" in call.args[0]
                for call in mock_log_warning.call_args_list
            )
        )
        assert_np_equal(x_cpu.grad.numpy(), np.ones(n, dtype=np.int32).view(np.float32))


def test_tape_copy_adjoint_cpu_cuda(test, device):
    """Verify CPU/CUDA copy adjoints accumulate into the source and consume the destination."""
    n = 4

    # CPU -> CUDA: the CPU source receives both its direct kernel contribution and
    # the copy's destination adjoint.
    x_cpu = wp.array(np.ones(n, dtype=np.float32), dtype=float, requires_grad=True, device="cpu")
    dst_cuda = wp.zeros(n, dtype=float, requires_grad=True, device=device)
    doubled_cpu = wp.zeros_like(x_cpu)
    loss_cpu = wp.zeros(1, dtype=float, requires_grad=True, device="cpu")
    loss_cuda = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(dst_cuda, x_cpu)
        wp.launch(mul_constant, dim=n, inputs=[x_cpu], outputs=[doubled_cpu], device="cpu")
        wp.launch(sum_kernel, dim=n, inputs=[doubled_cpu], outputs=[loss_cpu], device="cpu")
        wp.launch(sum_kernel, dim=n, inputs=[dst_cuda], outputs=[loss_cuda], device=device)
    tape.backward(
        grads={
            loss_cpu: wp.ones(1, dtype=float, device="cpu"),
            loss_cuda: wp.ones(1, dtype=float, device=device),
        }
    )

    assert_np_equal(x_cpu.grad.numpy(), np.full(n, 3.0, dtype=np.float32))

    stream = wp.Stream(device)
    src_cpu = wp.array(np.ones(n, dtype=np.float32), dtype=float, requires_grad=True, device="cpu")
    dst_cuda = wp.zeros(2, dtype=float, requires_grad=True, retain_grad=True, device=device)
    loss_cuda = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(dst_cuda, src_cpu, src_offset=1, count=2, stream=stream)
        device.stream.wait_stream(stream)
        wp.launch(sum_kernel, dim=2, inputs=[dst_cuda], outputs=[loss_cuda], device=device)
    tape.backward(loss_cuda)

    assert_np_equal(src_cpu.grad.numpy(), np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32))
    assert_np_equal(dst_cuda.grad.numpy(), np.ones(2, dtype=np.float32))

    # CPU -> CUDA: an earlier CUDA write to the destination is dead and its
    # adjoint must be consumed by the copy.
    dead_src_cuda = wp.array(np.ones(n, dtype=np.float32), dtype=float, requires_grad=True, device=device)
    src_cpu = wp.array(np.full(n, 5.0, dtype=np.float32), dtype=float, requires_grad=True, device="cpu")
    dst_cuda = wp.zeros(n, dtype=float, requires_grad=True, device=device)
    loss_cuda = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(mul_constant, dim=n, inputs=[dead_src_cuda], outputs=[dst_cuda], device=device)
        wp.copy(dst_cuda, src_cpu)
        wp.launch(sum_kernel, dim=n, inputs=[dst_cuda], outputs=[loss_cuda], device=device)
    tape.backward(loss_cuda)

    assert_np_equal(src_cpu.grad.numpy(), np.ones(n, dtype=np.float32))
    assert_np_equal(dead_src_cuda.grad.numpy(), np.zeros(n, dtype=np.float32))

    # CUDA -> CPU: the CUDA source receives both its direct kernel contribution
    # and the copy's destination adjoint.
    x_cuda = wp.array(np.ones(n, dtype=np.float32), dtype=float, requires_grad=True, device=device)
    dst_cpu = wp.zeros(n, dtype=float, requires_grad=True, device="cpu")
    doubled_cuda = wp.zeros_like(x_cuda)
    loss_cuda = wp.zeros(1, dtype=float, requires_grad=True, device=device)
    loss_cpu = wp.zeros(1, dtype=float, requires_grad=True, device="cpu")

    tape = wp.Tape()
    with tape:
        wp.copy(dst_cpu, x_cuda)
        wp.launch(mul_constant, dim=n, inputs=[x_cuda], outputs=[doubled_cuda], device=device)
        wp.launch(sum_kernel, dim=n, inputs=[doubled_cuda], outputs=[loss_cuda], device=device)
        wp.launch(sum_kernel, dim=n, inputs=[dst_cpu], outputs=[loss_cpu], device="cpu")
    tape.backward(
        grads={
            loss_cuda: wp.ones(1, dtype=float, device=device),
            loss_cpu: wp.ones(1, dtype=float, device="cpu"),
        }
    )

    assert_np_equal(x_cuda.grad.numpy(), np.full(n, 3.0, dtype=np.float32))

    # CUDA -> CPU: an earlier CPU write to the destination is dead and its adjoint
    # must be consumed by the copy.
    dead_src_cpu = wp.array(np.ones(n, dtype=np.float32), dtype=float, requires_grad=True, device="cpu")
    src_cuda = wp.array(np.full(n, 5.0, dtype=np.float32), dtype=float, requires_grad=True, device=device)
    dst_cpu = wp.zeros(n, dtype=float, requires_grad=True, device="cpu")
    loss_cpu = wp.zeros(1, dtype=float, requires_grad=True, device="cpu")

    tape = wp.Tape()
    with tape:
        wp.launch(mul_constant, dim=n, inputs=[dead_src_cpu], outputs=[dst_cpu], device="cpu")
        wp.copy(dst_cpu, src_cuda)
        wp.launch(sum_kernel, dim=n, inputs=[dst_cpu], outputs=[loss_cpu], device="cpu")
    tape.backward(loss_cpu)

    assert_np_equal(src_cuda.grad.numpy(), np.ones(n, dtype=np.float32))
    assert_np_equal(dead_src_cpu.grad.numpy(), np.zeros(n, dtype=np.float32))


def test_tape_copy_adjoint_stream(test, device):
    """Verify copy adjoints are ordered with the backward pass on non-current streams."""
    # a forward copy issued on a non-current stream: the adjoint operations are
    # ordered against the rest of the backward pass via the recorded stream
    n = 1 << 20
    stream = wp.Stream(device)
    x = wp.array(np.ones(n, dtype=np.float32), dtype=float, requires_grad=True, device=device)
    doubled = wp.zeros_like(x)
    dst = wp.zeros_like(x)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(dst, x, stream=stream)
        wp.launch(mul_constant, dim=n, inputs=[x, doubled], device=device)
        wp.launch(sum_kernel, dim=n, inputs=[doubled], outputs=[loss], device=device)
        wp.launch(sum_kernel, dim=n, inputs=[dst], outputs=[loss], device=device)
    tape.backward(loss)

    assert_np_equal(x.grad.numpy(), np.full(n, 3.0))


def test_tape_copy_adjoint_graph_capture(test, device):
    """Verify copy-adjoint behavior during CUDA graph capture."""
    # the same-device copy adjoint performs no allocations (including for
    # non-contiguous views), so a backward pass containing it stays capturable
    # in a CUDA graph; gradients must accumulate identically across replays
    n = 4
    x = wp.array(np.ones((n, 2), dtype=np.float32), dtype=float, requires_grad=True, device=device)
    col = x[:, 0]  # strided view
    dst = wp.zeros(n, dtype=float, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(dst, col)
        wp.launch(sum_kernel, dim=n, inputs=[dst], outputs=[loss], device=device)

    # warm up module loads outside of the capture
    tape.backward(loss)
    tape.zero()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        tape.backward(loss)

    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)

    # two replays accumulate two copy-adjoint contributions into the viewed column
    expected = np.zeros((n, 2), dtype=np.float32)
    expected[:, 0] = 2.0
    assert_np_equal(x.grad.numpy(), expected)

    src_cpu = wp.ones(n, dtype=float, requires_grad=True, device="cpu")
    dst_cuda = wp.zeros(n, dtype=float, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.copy(dst_cuda, src_cpu)
    dst_cuda.grad.fill_(1.0)

    with test.assertRaisesRegex(RuntimeError, "Cannot run CPU/CUDA wp.copy\\(\\) adjoints during CUDA graph capture"):
        with wp.ScopedCapture(device, force_module_load=False):
            tape.backward()
    test.assertFalse(device.is_capturing)

    src_cuda = wp.ones(n, dtype=float, requires_grad=True, device=device)
    dst_cpu = wp.zeros(n, dtype=float, requires_grad=True, device="cpu")
    tape = wp.Tape()
    with tape:
        wp.copy(dst_cpu, src_cuda)
    dst_cpu.grad.fill_(1.0)

    with test.assertRaisesRegex(RuntimeError, "Cannot run CPU/CUDA wp.copy\\(\\) adjoints during CUDA graph capture"):
        with wp.ScopedCapture(device, force_module_load=False):
            tape.backward()
    test.assertFalse(device.is_capturing)


def test_tape_backward_cuda_launch_failure(test, device):
    """Raise when Tape backward hits CUDA launch errors.

    Corrupt the recorded backward launch block size to reproduce the stale-gradient failure mode.
    Protects ``Tape.backward()`` from returning after a failed replay with stale or missing gradients.
    """
    x = wp.array([1.0], dtype=wp.float32, device=device, requires_grad=True)
    y = wp.empty_like(x, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=mul_constant, dim=x.size, inputs=[x], outputs=[y], block_dim=256, device=device)

    launch = tape.launches[0]
    test.assertEqual(len(launch), 8)
    launch[6] = 2048  # block_dim

    with test.assertRaisesRegex(RuntimeError, r"Error launching kernel: .*mul_constant.*Warp CUDA error"):
        tape.backward(grads={y: wp.ones_like(y)})


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestTape(unittest.TestCase):
    def test_tape_no_nested_tapes(self):
        with self.assertRaises(RuntimeError):
            with wp.Tape():
                with wp.Tape():
                    pass

    def test_tape_scope_end_without_matching_begin(self):
        tape = wp.Tape()

        with self.assertRaisesRegex(RuntimeError, "ended tape scope, but scope not present"):
            tape.record_scope_end()

    def test_tape_scope_end_twice_raises(self):
        tape = wp.Tape()

        tape.record_scope_begin("scope")
        tape.record_scope_end()

        with self.assertRaisesRegex(RuntimeError, "ended tape scope, but scope not present"):
            tape.record_scope_end()

    def test_tape_nested_nonempty_scope_markers(self):
        tape = wp.Tape()

        tape.record_scope_begin("outer")
        tape.record_scope_begin("inner")
        tape.launches.append(object())
        tape.record_scope_end()
        tape.record_scope_end()

        self.assertEqual(
            tape.scopes,
            [
                (0, "outer", {}),
                (0, "inner", {}),
                (1, None, None),
                (1, None, None),
            ],
        )

    def test_tape_empty_nested_scope_markers_removed(self):
        tape = wp.Tape()

        tape.record_scope_begin("outer")
        tape.record_scope_begin("inner")
        tape.record_scope_end()
        tape.record_scope_end()

        self.assertEqual(tape.scopes, [])


add_function_test(TestTape, "test_tape_mul_constant", test_tape_mul_constant, devices=devices)
add_function_test(TestTape, "test_tape_mul_variable", test_tape_mul_variable, devices=devices)
add_function_test(TestTape, "test_tape_dot_product", test_tape_dot_product, devices=devices)
add_function_test(TestTape, "test_tape_zero_multiple_outputs", test_tape_zero_multiple_outputs, devices=devices)
add_function_test(TestTape, "test_tape_nested_struct", test_tape_nested_struct, devices=devices)
add_function_test(TestTape, "test_tape_visualize", test_tape_visualize, devices=devices)
add_function_test(TestTape, "test_tape_struct_subscript", test_tape_struct_subscript, devices=devices)
add_function_test(TestTape, "test_tape_nested_struct_subscript", test_tape_nested_struct_subscript, devices=devices)
add_function_test(TestTape, "test_tape_visualize_subscript", test_tape_visualize_subscript, devices=devices)
add_function_test(TestTape, "test_tape_copy_adjoint_accumulation", test_tape_copy_adjoint_accumulation, devices=devices)
add_function_test(TestTape, "test_tape_copy_adjoint_consumption", test_tape_copy_adjoint_consumption, devices=devices)
add_function_test(
    TestTape, "test_tape_copy_adjoint_views_and_offsets", test_tape_copy_adjoint_views_and_offsets, devices=devices
)
add_function_test(TestTape, "test_tape_copy_adjoint_out_of_scope", test_tape_copy_adjoint_out_of_scope, devices=devices)
add_function_test(
    TestTape, "test_tape_copy_adjoint_fallback_warning", test_tape_copy_adjoint_fallback_warning, devices=devices
)
add_function_test(TestTape, "test_tape_copy_adjoint_cpu_cuda", test_tape_copy_adjoint_cpu_cuda, devices=cuda_devices)
add_function_test(TestTape, "test_tape_copy_adjoint_stream", test_tape_copy_adjoint_stream, devices=cuda_devices)
add_function_test(
    TestTape, "test_tape_copy_adjoint_graph_capture", test_tape_copy_adjoint_graph_capture, devices=cuda_devices
)
add_function_test(
    TestTape, "test_tape_backward_cuda_launch_failure", test_tape_backward_cuda_launch_failure, devices=cuda_devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
