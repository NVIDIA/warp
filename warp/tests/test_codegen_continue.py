# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import unittest

import numpy as np

import warp as wp
from warp._src import context
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def continue_argmax(c: wp.array2d[float], x: wp.array[float], out: wp.array[float]):
    """Compute per-thread maxima while skipping zero coefficients."""
    tid = wp.tid()
    maximum = float(-1.0e20)
    for p in range(5):
        for q in range(5):
            coefficient = c[p, q]
            if coefficient == 0.0:
                continue
            value = x[tid] * coefficient * float(p + q + 2)
            if value > maximum:
                maximum = value
    out[tid] = maximum


def test_continue_argmax(test, device):
    """Check that each thread's maximum selects its own gradient."""
    coefficients = np.zeros((5, 5), dtype=np.float32)
    coefficients[0, 0] = 1.0
    coefficients[1, 1] = -1.0
    c = wp.array(coefficients, device=device, requires_grad=True)
    x = wp.array([1.0, -1.0, 2.0, -2.0], dtype=float, device=device, requires_grad=True)
    out = wp.zeros_like(x, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(continue_argmax, x.size, inputs=[c, x, out], device=device)
    tape.backward(grads={out: wp.ones_like(out)})
    np.testing.assert_allclose(out.numpy(), [2, 4, 4, 8])
    np.testing.assert_allclose(x.grad.numpy(), [2, -4, 2, -4])
    expected_c = np.zeros_like(coefficients)
    expected_c[0, 0] = 6.0
    expected_c[1, 1] = -12.0
    np.testing.assert_allclose(c.grad.numpy(), expected_c)


@wp.func
def continue_branches(x: float, gates: wp.array[int]):
    """Evaluate a nonlinear recurrence with nested and consecutive continues."""
    value = float(1.0)
    for i in range(6):
        value *= x
        if gates[i] == 1:
            value += float(i) * x
            if i % 2 == 0:
                continue
            value *= 0.5
        else:
            value += 0.25
            if gates[i] == 2:
                continue
        value *= x
        if value > 2.0:
            continue
        value += 0.125 * x
    return value


@wp.kernel
def continue_branches_kernel(x: wp.array[float], gates: wp.array[int], out: wp.array[float]):
    """Evaluate the branching recurrence through a Warp function."""
    tid = wp.tid()
    out[tid] = continue_branches(x[tid], gates)


def test_continue_branches(test, device):
    """Preserve updates before nested and consecutive continues in either branch."""
    xs = np.array([0.8, 1.2, -0.9], dtype=np.float32)
    gate_values = np.array([1, 1, 2, 0, 1, 0], dtype=np.int32)
    expected = []
    for x_value in xs.astype(np.float64):
        value, derivative = 1.0, 0.0
        for i in range(6):
            value, derivative = value * x_value, derivative * x_value + value
            if gate_values[i] == 1:
                value, derivative = value + i * x_value, derivative + i
                if i % 2 == 0:
                    continue
                value, derivative = value * 0.5, derivative * 0.5
            else:
                value += 0.25
                if gate_values[i] == 2:
                    continue
            value, derivative = value * x_value, derivative * x_value + value
            if value > 2.0:
                continue
            value, derivative = value + 0.125 * x_value, derivative + 0.125
        expected.append((value, derivative))

    x = wp.array(xs, device=device, requires_grad=True)
    gates = wp.array(gate_values, device=device)
    out = wp.zeros_like(x, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(continue_branches_kernel, x.size, inputs=[x, gates, out], device=device)
    tape.backward(grads={out: wp.ones_like(out)})
    expected = np.array(expected)
    np.testing.assert_allclose(out.numpy(), expected[:, 0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x.grad.numpy(), expected[:, 1], rtol=1e-5, atol=1e-5)


@wp.kernel
def continue_nested(x: wp.array[float], n: int, out: wp.array[float], writes: wp.array[int]):
    """Accumulate values and count writes across nested loops with continues."""
    tid = wp.tid()
    value = float(0.0)
    for p in range(n):
        if p == 1:
            continue
        for q in range(5):
            value += x[tid]
            if q == 2:
                continue
            for r in range(n):
                if r == 1:
                    continue
                value += x[tid]
            wp.atomic_add(writes, tid, 1)
    out[tid] = value


def test_continue_nested(test, device):
    """Keep inner-loop transfers scoped and skip side effects after a continue."""
    x = wp.full(2, 2.0, requires_grad=True, device=device)
    out = wp.zeros_like(x, requires_grad=True)
    writes = wp.zeros(2, dtype=int, device=device)
    with wp.Tape() as tape:
        wp.launch(continue_nested, 2, inputs=[x, 3, out, writes], device=device)
    tape.backward(grads={out: wp.ones_like(out)})
    np.testing.assert_allclose(out.numpy(), [52, 52])
    np.testing.assert_allclose(x.grad.numpy(), [26, 26])
    np.testing.assert_array_equal(writes.numpy(), [8, 8])


def test_continue_ranges(test, device):
    """Handle descending, strided, empty, and single-iteration ranges."""

    def make_kernel(start, stop, step):
        """Build a recurrence over the requested constant range."""

        @wp.kernel(module="unique")
        def recurrence(x: wp.array[float], gates: wp.array[int], out: wp.array[float]):
            """Evaluate the recurrence while skipping additions selected by the gates."""
            tid = wp.tid()
            value = float(1.0)
            for i in range(start, stop, step):
                value *= x[tid]
                if gates[i] == 1:
                    continue
                value += x[tid]
                continue
                value *= 10.0
            out[tid] = value

        return recurrence

    gate_values = np.array([1, 0, 1, 1, 0, 1], dtype=np.int32)
    for start, stop, step in ((5, -1, -1), (0, 6, 2), (4, 1, 1), (2, 3, 1)):
        with test.subTest(start=start, stop=stop, step=step):
            recurrence = make_kernel(start, stop, step)
            value, derivative = 1.0, 0.0
            for i in range(start, stop, step):
                value, derivative = value * 2.0, derivative * 2.0 + value
                if gate_values[i] == 1:
                    continue
                value, derivative = value + 2.0, derivative + 1.0
            x = wp.full(2, 2.0, requires_grad=True, device=device)
            gates = wp.array(gate_values, device=device)
            out = wp.zeros_like(x, requires_grad=True)
            with wp.Tape() as tape:
                wp.launch(recurrence, 2, inputs=[x, gates, out], device=device)
            tape.backward(grads={out: wp.ones_like(out)})
            np.testing.assert_allclose(out.numpy(), [value, value])
            np.testing.assert_allclose(x.grad.numpy(), [derivative, derivative])


def test_continue_static(test, device):
    """Honor forced unrolling and remove statically skipped iterations."""

    @wp.kernel(module="unique", module_options={"max_unroll": 2})
    def continue_static(x: wp.array[float], out: wp.array[float]):
        """Sum odd-indexed terms using a statically evaluated continue condition."""
        tid = wp.tid()
        value = float(0.0)
        for i in range(20):
            if wp.static(i % 2 == 0):
                continue
            value += float(i) * x[tid]
        out[tid] = value

    x = wp.full(2, 2.0, requires_grad=True, device=device)
    out = wp.zeros_like(x, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(continue_static, 2, inputs=[x, out], device=device)
    tape.backward(grads={out: wp.ones_like(out)})
    np.testing.assert_allclose(out.numpy(), [200, 200])
    np.testing.assert_allclose(x.grad.numpy(), [100, 100])


@wp.kernel(enable_backward=False)
def local_scope(gates: wp.array[int], n: int, out: wp.array[int]):
    """Exercise user locals and loop indices around loops containing continue."""
    _wp_continue_0 = int(7)
    _wp_continue_1 = int(11)
    total = int(0)
    for i in range(4):
        total += _wp_continue_0
        if gates[i] != 0:
            continue
        total += _wp_continue_1
    last = i
    while n > 0:
        n -= 1
        if n == 1:
            continue
        total += 1
    out[0] = total
    out[1] = last
    out[2] = _wp_continue_0
    out[3] = _wp_continue_1


def test_continue_local_scope(test, device):
    """Keep compiler predicates separate from user locals and later loops."""

    gates = wp.array([0, 1, 0, 1], dtype=int, device=device)
    out = wp.zeros(4, dtype=int, device=device)
    wp.launch(local_scope, 1, inputs=[gates, 3, out], device=device)
    np.testing.assert_array_equal(out.numpy(), [52, 3, 7, 11])


class TestCodeGenContinue(unittest.TestCase):
    def test_unrolling_policy(self):
        """Preserve unrolling eligibility and its existing limits."""

        def with_continue(gates: wp.array[int], out: wp.array[int]):
            """Sum selected indices in a constant range containing continue."""
            value = int(0)
            for i in range(4):
                if gates[i] != 0:
                    continue
                value += i
            out[0] = value

        def with_break(gates: wp.array[int], out: wp.array[int]):
            """Sum indices until a gate triggers a break in a constant range."""
            value = int(0)
            for i in range(4):
                if gates[i] != 0:
                    break
                value += i
            out[0] = value

        def runtime_range(n: int, gates: wp.array[int], out: wp.array[int]):
            """Sum selected indices in a runtime range containing continue."""
            value = int(0)
            for i in range(n):
                if gates[i] != 0:
                    continue
                value += i
            out[0] = value

        for func, limit, expected_loops in (
            (with_continue, 4, 0),
            (with_continue, 2, 1),
            (with_break, 4, 1),
            (runtime_range, 4, 1),
        ):
            with self.subTest(kernel=func.__name__, limit=limit):
                kernel = wp.kernel(func, module="unique", module_options={"max_unroll": limit})
                original_tree = ast.dump(kernel.adj.tree, include_attributes=True)
                options = kernel.module.resolve_options(wp.config)
                sources = []
                for _ in range(2):
                    builder = context.ModuleBuilder(kernel.module, options=options)
                    sources.append(builder.codegen("cpu"))
                    loops = [
                        line for line in kernel.adj.blocks[0].body_forward if line.lstrip().startswith("start_for_")
                    ]
                    self.assertEqual(len(loops), expected_loops)
                    self.assertEqual(ast.dump(kernel.adj.tree, include_attributes=True), original_tree)
                self.assertEqual(sources[0], sources[1])


devices = get_test_devices()
add_function_test(TestCodeGenContinue, "test_continue_argmax", test_continue_argmax, devices=devices)
add_function_test(TestCodeGenContinue, "test_continue_branches", test_continue_branches, devices=devices)
add_function_test(TestCodeGenContinue, "test_continue_nested", test_continue_nested, devices=devices)
add_function_test(TestCodeGenContinue, "test_continue_ranges", test_continue_ranges, devices=devices)
add_function_test(TestCodeGenContinue, "test_continue_static", test_continue_static, devices=devices)
add_function_test(TestCodeGenContinue, "test_continue_local_scope", test_continue_local_scope, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
