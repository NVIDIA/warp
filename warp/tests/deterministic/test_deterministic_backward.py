# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import sys
import unittest

import numpy as np

import warp as wp
import warp.tests.deterministic.test_deterministic_counter as counter_module
import warp.tests.deterministic.test_deterministic_scatter as scatter_module
from warp.tests.deterministic.common import (
    DeterministicTestBase,
    _reference_scatter_add_float32,
    all_devices,
    cuda_devices,
)
from warp.tests.deterministic.test_deterministic_counter import counter_kernel
from warp.tests.deterministic.test_deterministic_scatter import scatter_add_kernel
from warp.tests.unittest_utils import add_function_test

_THIS_MODULE = sys.modules[__name__]


def _set_test_module_options(options):
    wp.set_module_options(options, module=_THIS_MODULE)


def _get_test_module_options():
    return wp.get_module_options(module=_THIS_MODULE)


@wp.kernel
def gather_address_kernel(
    values: wp.array[wp.float32],
    source_indices: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    """Read many output lanes from the same input lanes.

    The forward pass is a plain gather, but the generated backward pass scatters
    output gradients back into ``values.grad`` with atomic additions.
    """
    tid = wp.tid()
    output[tid] = values[source_indices[tid]]


@wp.kernel
def vec3_gather_address_kernel(
    values: wp.array[wp.vec3],
    source_indices: wp.array[wp.int32],
    output: wp.array[wp.vec3],
):
    """Vector-valued gather covering cloth/particle-style position gradients."""
    tid = wp.tid()
    output[tid] = values[source_indices[tid]]


@wp.kernel
def gather_with_nongrad_input_kernel(
    values: wp.array[wp.float32],
    source_indices: wp.array[wp.int32],
    scale: wp.array[wp.float32],
    output: wp.array[wp.float32],
):
    """Read a non-differentiable input while accumulating into another gradient."""
    tid = wp.tid()
    output[tid] = values[source_indices[tid]] * scale[0]


@wp.func
def _det_custom_replay_counter(counter: wp.array[wp.int32], tids: wp.array[wp.int32], tid: int):
    slot = wp.atomic_add(counter, 0, 1)
    tids[tid] = slot
    return slot


@wp.func_replay(_det_custom_replay_counter)
def _replay_det_custom_replay_counter(counter: wp.array[wp.int32], tids: wp.array[wp.int32], tid: int):
    return tids[tid]


@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
def custom_replay_counter_kernel(
    data: wp.array[wp.float32],
    counter: wp.array[wp.int32],
    tids: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    tid = wp.tid()
    slot = _det_custom_replay_counter(counter, tids, tid)
    output[slot] = data[tid] * data[tid]


@wp.func
def _det_lookup_value(values: wp.array[wp.float32], index: int) -> wp.float32:
    return values[index]


@wp.func_grad(_det_lookup_value)
def _adj_det_lookup_value(values: wp.array[wp.float32], index: int, adj_ret: wp.float32):
    wp.adjoint[values][index] += adj_ret


@wp.kernel
def custom_adjoint_lookup_kernel(
    values: wp.array[wp.float32],
    indices: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    """Use a custom adjoint that accumulates into ``wp.adjoint[values]``."""
    tid = wp.tid()
    value = _det_lookup_value(values, indices[tid])
    wp.atomic_add(output, 0, value)


@wp.kernel
def custom_adjoint_gather_kernel(
    values: wp.array[wp.float32],
    indices: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    """Use a custom adjoint that scatters per-lane output gradients."""
    tid = wp.tid()
    output[tid] = _det_lookup_value(values, indices[tid])


@wp.func
def _det_custom_square_store(
    i: int,
    values: wp.array[wp.float32],
    output: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
):
    output[i] = values[i] * values[i]


@wp.func_grad(_det_custom_square_store)
def _adj_det_custom_square_store(
    i: int,
    values: wp.array[wp.float32],
    output: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
):
    scratch[i] = 0.0
    wp.adjoint[values][i] += 2.0 * values[i] * wp.adjoint[output][i]


@wp.kernel
def custom_adjoint_store_kernel(
    values: wp.array[wp.float32],
    output: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
):
    tid = wp.tid()
    _det_custom_square_store(tid, values, output, scratch)


@wp.func
def _not_guaranteed_lookup_value(values: wp.array[wp.float32], index: int) -> wp.float32:
    return values[index]


@wp.func_grad(_not_guaranteed_lookup_value)
def _adj_not_guaranteed_lookup_value(values: wp.array[wp.float32], index: int, adj_ret: wp.float32):
    wp.adjoint[values][index] += adj_ret


@wp.kernel
def custom_adjoint_not_guaranteed_kernel(
    values: wp.array[wp.float32],
    indices: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    """Use a custom adjoint while deterministic mode is disabled."""
    tid = wp.tid()
    output[tid] = _not_guaranteed_lookup_value(values, indices[tid])


@wp.func
def _det_counter_grad_value(
    values: wp.array[wp.float32],
    counter: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    tid: int,
) -> wp.float32:
    return values[tid]


@wp.func_grad(_det_counter_grad_value)
def _adj_det_counter_grad_value(
    values: wp.array[wp.float32],
    counter: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    tid: int,
    adj_ret: wp.float32,
):
    slot = wp.atomic_add(counter, 0, 1)
    slots[slot] = tid
    wp.adjoint[values][tid] += adj_ret


@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
def custom_adjoint_counter_kernel(
    values: wp.array[wp.float32],
    counter: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    tid = wp.tid()
    output[tid] = _det_counter_grad_value(values, counter, slots, tid)


@wp.func
def _det_square_via_scratch(scratch: wp.array[wp.float32], x: wp.float32):
    scratch[0] = x
    return scratch[0] * scratch[0]


@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
def scratch_overwrite_grad_kernel(
    x: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
    loss: wp.array[wp.float32],
):
    loss[0] = _det_square_via_scratch(scratch, x[0])


def test_deterministic_backward_scatter_add(test, device):
    """Verify deterministic scatter-add kernels launch backward and propagate value gradients."""
    n = 512
    out_size = 16
    rng = np.random.default_rng(300)
    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device, requires_grad=True)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(out_size, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)

    tape.backward(grads={output: wp.ones_like(output)})

    np.testing.assert_allclose(tape.gradients[data].numpy(), np.ones(n, dtype=np.float32), rtol=0, atol=0)


def test_deterministic_backward_address_scatter(test, device):
    """Verify generated array-read adjoints reduce contended gradients deterministically."""
    n = 4096
    value_count = 37
    rng = np.random.default_rng(303)
    values_np = rng.random(value_count, dtype=np.float32)
    indices_np = rng.integers(0, value_count, size=n, dtype=np.int32)
    grad_np = rng.random(n, dtype=np.float32)
    expected = _reference_scatter_add_float32(grad_np, indices_np, value_count)

    values = wp.array(values_np, dtype=wp.float32, device=device, requires_grad=True)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output_grad = wp.array(grad_np, dtype=wp.float32, device=device)

    old_det = _get_test_module_options()["deterministic"]
    try:
        _set_test_module_options({"deterministic": wp.DeterministicMode.GPU_TO_GPU})
        results = []
        for _ in range(3):
            values.grad.zero_()
            output = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(gather_address_kernel, dim=n, inputs=[values, indices], outputs=[output], device=device)
            tape.backward(grads={output: output_grad})
            results.append(tape.gradients[values].numpy().copy())
    finally:
        _set_test_module_options({"deterministic": old_det})

    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_deterministic_backward_strided_adjoint_address_scatter(test, device):
    """Verify manual backward reductions use explicit adjoint-buffer strides."""
    n = 1024
    value_count = 19
    rng = np.random.default_rng(307)
    values_np = rng.random(value_count, dtype=np.float32)
    indices_np = rng.integers(0, value_count, size=n, dtype=np.int32)
    grad_np = rng.random(n, dtype=np.float32)
    expected = _reference_scatter_add_float32(grad_np, indices_np, value_count)

    values = wp.array(values_np, dtype=wp.float32, device=device, requires_grad=True)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
    output_grad = wp.array(grad_np, dtype=wp.float32, device=device)
    base_grad = wp.zeros(value_count * 2, dtype=wp.float32, device=device)
    grad_view = base_grad[::2]
    test.assertFalse(grad_view.is_contiguous)

    old_det = _get_test_module_options()["deterministic"]
    try:
        _set_test_module_options({"deterministic": wp.DeterministicMode.GPU_TO_GPU})
        wp.launch(gather_address_kernel, dim=n, inputs=[values, indices], outputs=[output], device=device)
        wp.launch(
            gather_address_kernel,
            dim=n,
            inputs=[values, indices],
            outputs=[output],
            adj_inputs=[grad_view, None],
            adj_outputs=[output_grad],
            device=device,
            adjoint=True,
        )
    finally:
        _set_test_module_options({"deterministic": old_det})

    result = base_grad.numpy()
    np.testing.assert_array_equal(result[0::2], expected)
    np.testing.assert_array_equal(result[1::2], np.zeros(value_count, dtype=np.float32))


def test_deterministic_backward_vec3_address_scatter(test, device):
    """Verify vector-valued array-read adjoints are also reduced deterministically."""
    n = 2048
    value_count = 29
    rng = np.random.default_rng(304)
    values_np = rng.random((value_count, 3), dtype=np.float32)
    indices_np = rng.integers(0, value_count, size=n, dtype=np.int32)
    grad_np = rng.random((n, 3), dtype=np.float32)
    expected = np.zeros((value_count, 3), dtype=np.float32)
    for value, index in zip(grad_np, indices_np, strict=True):
        expected[index] = np.float32(expected[index] + value)

    values = wp.array(values_np, dtype=wp.vec3, device=device, requires_grad=True)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output_grad = wp.array(grad_np, dtype=wp.vec3, device=device)

    old_det = _get_test_module_options()["deterministic"]
    try:
        _set_test_module_options({"deterministic": wp.DeterministicMode.GPU_TO_GPU})
        results = []
        for _ in range(3):
            values.grad.zero_()
            output = wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(vec3_gather_address_kernel, dim=n, inputs=[values, indices], outputs=[output], device=device)
            tape.backward(grads={output: output_grad})
            results.append(tape.gradients[values].numpy().copy())
    finally:
        _set_test_module_options({"deterministic": old_det})

    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_deterministic_backward_missing_adjoint_target(test, device):
    """Verify backward deterministic reductions ignore array reads with no grad buffer."""
    n = 2048
    value_count = 23
    rng = np.random.default_rng(305)
    values_np = rng.random(value_count, dtype=np.float32)
    indices_np = rng.integers(0, value_count, size=n, dtype=np.int32)
    grad_np = rng.random(n, dtype=np.float32)
    expected = np.zeros(1, dtype=np.float32)
    for grad, index in zip(grad_np, indices_np, strict=True):
        expected[0] = np.float32(expected[0] + np.float32(grad * values_np[index]))

    values = wp.array(values_np, dtype=wp.float32, device=device, requires_grad=False)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    scale = wp.ones(1, dtype=wp.float32, device=device, requires_grad=True)
    output_grad = wp.array(grad_np, dtype=wp.float32, device=device)

    old_det = _get_test_module_options()["deterministic"]
    try:
        _set_test_module_options({"deterministic": wp.DeterministicMode.GPU_TO_GPU})
        results = []
        for _ in range(3):
            scale.grad.zero_()
            output = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(
                    gather_with_nongrad_input_kernel,
                    dim=n,
                    inputs=[values, indices, scale],
                    outputs=[output],
                    device=device,
                )
            tape.backward(grads={output: output_grad})
            results.append(tape.gradients[scale].numpy().copy())
    finally:
        _set_test_module_options({"deterministic": old_det})

    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_deterministic_backward_counter_store_rejected(test, device):
    """Verify generated backward replay of consumed-return counters fails closed."""
    n = 64
    data_np = np.arange(n, dtype=np.float32)

    data = wp.array(data_np, dtype=wp.float32, device=device, requires_grad=True)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(counter_kernel, dim=n, inputs=[data, counter], outputs=[output], device=device)

    with test.assertRaisesRegex(RuntimeError, "generated backward replay of consumed-return counter atomics"):
        tape.backward(grads={output: wp.ones_like(output)})


def test_deterministic_backward_counter_manual_launch_rejected(test, device):
    """Verify manual adjoint launches fail closed for generated counter replay."""
    n = 8
    data = wp.array(np.arange(n, dtype=np.float32), dtype=wp.float32, device=device, requires_grad=True)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
    output_grad = wp.ones(n, dtype=wp.float32, device=device)

    wp.launch(counter_kernel, dim=n, inputs=[data, counter], outputs=[output], device=device)

    with test.assertRaisesRegex(RuntimeError, "generated backward replay of consumed-return counter atomics"):
        wp.launch(
            counter_kernel,
            dim=n,
            inputs=[data, counter],
            outputs=[output],
            adj_inputs=[data.grad, None],
            adj_outputs=[output_grad],
            device=device,
            adjoint=True,
        )


def test_deterministic_custom_replay_counter(test, device):
    """Verify deterministic helper args do not leak into custom replay calls."""
    n = 64
    data_np = np.arange(n, dtype=np.float32)

    data = wp.array(data_np, dtype=wp.float32, device=device, requires_grad=True)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    tids = wp.zeros(n, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(custom_replay_counter_kernel, dim=n, inputs=[data, counter, tids], outputs=[output], device=device)

    tape.backward(grads={output: wp.ones_like(output)})

    np.testing.assert_allclose(tape.gradients[data].numpy(), 2.0 * data_np, rtol=0, atol=0)


def test_deterministic_custom_adjoint_array_atomic(test, device):
    """Verify custom adjoints that atomically update ``wp.adjoint[array]`` compile."""
    n = 48
    value_count = 7
    indices_np = (np.arange(n, dtype=np.int32) * 3) % value_count
    values_np = np.linspace(0.25, 1.75, value_count, dtype=np.float32)

    values = wp.array(values_np, dtype=wp.float32, device=device, requires_grad=True)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(custom_adjoint_lookup_kernel, dim=n, inputs=[values, indices], outputs=[output], device=device)

    tape.backward(grads={output: wp.ones_like(output)})

    expected = np.bincount(indices_np, minlength=value_count).astype(np.float32)
    np.testing.assert_allclose(tape.gradients[values].numpy(), expected, rtol=0, atol=0)


def test_deterministic_custom_adjoint_gather_atomic(test, device):
    """Verify custom ``wp.adjoint[array]`` atomics use deterministic backward reduction."""
    n = 2048
    value_count = 31
    rng = np.random.default_rng(305)
    values_np = rng.random(value_count, dtype=np.float32)
    indices_np = rng.integers(0, value_count, size=n, dtype=np.int32)
    grad_np = rng.random(n, dtype=np.float32)
    expected = _reference_scatter_add_float32(grad_np, indices_np, value_count)

    values = wp.array(values_np, dtype=wp.float32, device=device, requires_grad=True)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output_grad = wp.array(grad_np, dtype=wp.float32, device=device)

    old_det = _get_test_module_options()["deterministic"]
    try:
        _set_test_module_options({"deterministic": wp.DeterministicMode.GPU_TO_GPU})
        results = []
        for _ in range(3):
            values.grad.zero_()
            output = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(
                    custom_adjoint_gather_kernel, dim=n, inputs=[values, indices], outputs=[output], device=device
                )
            tape.backward(grads={output: output_grad})
            results.append(tape.gradients[values].numpy().copy())
    finally:
        _set_test_module_options({"deterministic": old_det})

    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_deterministic_custom_adjoint_store_function(test, device):
    """Verify custom adjoints on functions with deterministic store context compile."""
    n = 32
    values_np = np.linspace(0.5, 2.0, n, dtype=np.float32)

    values = wp.array(values_np, dtype=wp.float32, device=device, requires_grad=True)
    output = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
    scratch = wp.ones(n, dtype=wp.float32, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(custom_adjoint_store_kernel, dim=n, inputs=[values], outputs=[output, scratch], device=device)

    tape.backward(grads={output: wp.ones_like(output)})

    np.testing.assert_allclose(output.numpy(), values_np * values_np, rtol=0, atol=0)
    np.testing.assert_allclose(scratch.numpy(), np.zeros(n, dtype=np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(tape.gradients[values].numpy(), 2.0 * values_np, rtol=0, atol=0)


def test_deterministic_scratch_overwrite_replay_gradient(test, device):
    """Verify deterministic replay preserves gradients through overwritten scratch values."""
    x = wp.array(np.array([3.0], dtype=np.float32), dtype=wp.float32, device=device, requires_grad=True)
    scratch = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(scratch_overwrite_grad_kernel, dim=1, inputs=[x, scratch], outputs=[loss], device=device)

    tape.backward(grads={loss: wp.ones_like(loss)})

    np.testing.assert_allclose(loss.numpy(), np.array([9.0], dtype=np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(tape.gradients[x].numpy(), np.array([6.0], dtype=np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(tape.gradients[scratch].numpy(), np.zeros(1, dtype=np.float32), rtol=0, atol=0)


def test_custom_adjoint_not_guaranteed_mode(test, device):
    """Verify custom adjoints compile after disabling deterministic mode."""
    n = 48
    value_count = 7
    indices_np = (np.arange(n, dtype=np.int32) * 3) % value_count
    values_np = np.linspace(0.25, 1.75, value_count, dtype=np.float32)

    old_det = _get_test_module_options()["deterministic"]
    try:
        _set_test_module_options({"deterministic": wp.DeterministicMode.NOT_GUARANTEED})

        values = wp.array(values_np, dtype=wp.float32, device=device, requires_grad=True)
        indices = wp.array(indices_np, dtype=wp.int32, device=device)
        output = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(
                custom_adjoint_not_guaranteed_kernel, dim=n, inputs=[values, indices], outputs=[output], device=device
            )

        tape.backward(grads={output: wp.ones_like(output)})
        result = tape.gradients[values].numpy()

        if not device.is_cpu:
            _set_test_module_options({"deterministic": wp.DeterministicMode.RUN_TO_RUN})

            store_n = 32
            store_values_np = np.linspace(0.5, 2.0, store_n, dtype=np.float32)
            store_values = wp.array(store_values_np, dtype=wp.float32, device=device, requires_grad=True)
            store_output = wp.zeros(store_n, dtype=wp.float32, device=device, requires_grad=True)
            store_scratch = wp.ones(store_n, dtype=wp.float32, device=device)

            store_tape = wp.Tape()
            with store_tape:
                wp.launch(
                    custom_adjoint_store_kernel,
                    dim=store_n,
                    inputs=[store_values],
                    outputs=[store_output, store_scratch],
                    device=device,
                )

            store_tape.backward(grads={store_output: wp.ones_like(store_output)})
            store_output_result = store_output.numpy()
            store_scratch_result = store_scratch.numpy()
            store_gradient_result = store_tape.gradients[store_values].numpy()
    finally:
        _set_test_module_options({"deterministic": old_det})

    expected = np.bincount(indices_np, minlength=value_count).astype(np.float32)
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)
    if not device.is_cpu:
        np.testing.assert_allclose(store_output_result, store_values_np * store_values_np, rtol=0, atol=0)
        np.testing.assert_allclose(store_scratch_result, np.zeros(store_n, dtype=np.float32), rtol=0, atol=0)
        np.testing.assert_allclose(store_gradient_result, 2.0 * store_values_np, rtol=0, atol=0)


def test_deterministic_custom_adjoint_consumed_counter_rejected(test, device):
    """Verify consumed-return counters in custom adjoints fail closed."""
    n = 16
    values = wp.ones(n, dtype=wp.float32, device=device, requires_grad=True)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    slots = wp.zeros(n, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)

    with test.assertRaisesRegex(RuntimeError, "consumed-return counter atomics"):
        tape = wp.Tape()
        with tape:
            wp.launch(
                custom_adjoint_counter_kernel, dim=n, inputs=[values, counter, slots], outputs=[output], device=device
            )
        tape.backward(grads={output: wp.ones_like(output)})


class TestDeterministicBackward(DeterministicTestBase):
    """Test deterministic lowering through backward launches and custom adjoints."""

    deterministic_modules = (_THIS_MODULE, scatter_module, counter_module)


def _add(name, devices=cuda_devices):
    add_function_test(TestDeterministicBackward, name, globals()[name], devices=devices)


for _name in (
    "test_deterministic_backward_scatter_add",
    "test_deterministic_backward_address_scatter",
    "test_deterministic_backward_strided_adjoint_address_scatter",
    "test_deterministic_backward_vec3_address_scatter",
    "test_deterministic_backward_missing_adjoint_target",
    "test_deterministic_custom_replay_counter",
    "test_deterministic_custom_adjoint_array_atomic",
    "test_deterministic_custom_adjoint_gather_atomic",
    "test_deterministic_custom_adjoint_store_function",
    "test_deterministic_scratch_overwrite_replay_gradient",
):
    _add(_name)

_add("test_deterministic_backward_counter_store_rejected", devices=all_devices)
_add("test_deterministic_backward_counter_manual_launch_rejected", devices=cuda_devices)
_add("test_deterministic_custom_adjoint_consumed_counter_rejected", devices=all_devices)
_add("test_custom_adjoint_not_guaranteed_mode", devices=all_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
