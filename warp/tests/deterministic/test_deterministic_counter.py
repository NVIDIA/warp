# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import unittest

import numpy as np

import warp as wp
from warp.tests.deterministic.common import DeterministicTestBase, all_devices, assert_equal_repeated, cuda_devices
from warp.tests.unittest_utils import add_function_test


@wp.struct
class _DetStructCounterWriter:
    counter: wp.array[wp.int32]
    output: wp.array[wp.float32]


@wp.func
def _det_struct_counter_write(writer: _DetStructCounterWriter, value: wp.float32):
    slot = wp.atomic_add(writer.counter, 0, 1)
    writer.output[slot] = value


@wp.func
def _det_counter_write(counter: wp.array[wp.int32], output: wp.array[wp.float32], value: wp.float32):
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = value


@wp.func
def _det_set_int_scratch(scratch: wp.array[wp.int32], index: int, value: int):
    scratch[index] = value


@wp.func
def _det_increment_array(scratch: wp.array[wp.float32], index: int):
    scratch[index] = scratch[index] + 1.0


@wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
def struct_field_counter_kernel(
    data: wp.array[wp.float32],
    counts: wp.array[wp.int32],
    writer: _DetStructCounterWriter,
):
    tid = wp.tid()
    count = counts[tid]
    if count > 0:
        base = wp.atomic_add(writer.counter, 0, count)
        for i in range(count):
            writer.output[base + i] = data[tid] + wp.float32(i) * wp.float32(0.5)


@wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
def struct_field_helper_counter_kernel(
    data: wp.array[wp.float32],
    flags: wp.array[wp.int32],
    writer: _DetStructCounterWriter,
):
    tid = wp.tid()
    if flags[tid] != 0:
        _det_struct_counter_write(writer, data[tid])


@wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
def helper_counter_side_effect_kernel(
    counter: wp.array[wp.int32],
    output: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
):
    """Normal stores before helper counter calls must be suppressed in phase 0."""
    tid = wp.tid()
    scratch[tid] = scratch[tid] + 1.0
    _det_counter_write(counter, output, float(tid))


@wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
def counter_with_helper_store_kernel(
    counter: wp.array[wp.int32],
    output: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
):
    """Pure write helpers called from counter kernels must be skipped in phase 0."""
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    _det_increment_array(scratch, tid)
    output[slot] = float(tid)


@wp.kernel
def counter_kernel(
    data: wp.array[wp.float32],
    counter: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    """Allocate a slot and write data to it."""
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = data[tid]


@wp.kernel
def conditional_counter_kernel(
    data: wp.array[wp.float32],
    threshold: wp.float32,
    counter: wp.array[wp.int32],
    output: wp.array[wp.float32],
):
    """Stream compaction: only emit elements above threshold."""
    tid = wp.tid()
    val = data[tid]
    if val > threshold:
        slot = wp.atomic_add(counter, 0, 1)
        output[slot] = val


@wp.kernel
def variable_counter_kernel(
    counts: wp.array[wp.int32],
    counter: wp.array[wp.int32],
    output: wp.array[wp.int32],
):
    """Reserve a variable number of slots per thread."""
    tid = wp.tid()
    count = counts[tid]
    slot = wp.atomic_add(counter, 0, count)

    for i in range(count):
        output[slot + i] = tid * 10 + i


@wp.kernel
def static_index_counter_kernel(
    counter: wp.array[wp.int32],
    output: wp.array[wp.int32],
):
    """Reserve slots from a fixed nonzero counter index."""
    tid = wp.tid()
    slot = wp.atomic_add(counter, 1, 1)
    output[slot] = tid


@wp.kernel
def indexed_counter_kernel(
    values: wp.array[wp.int32],
    bins: wp.array[wp.int32],
    counters: wp.array[wp.int32],
    output: wp.array2d[wp.int32],
):
    """Reserve slots from a data-dependent counter index."""
    tid = wp.tid()
    bin = bins[tid]
    slot = wp.atomic_add(counters, bin, 1)
    output[bin, slot] = values[tid]


@wp.kernel
def sliced_counter_kernel(
    values: wp.array[wp.int32],
    bins: wp.array[wp.int32],
    counters: wp.array2d[wp.int32],
    output: wp.array2d[wp.int32],
):
    """Reserve slots through a sliced counter view such as ``counters[bin]``."""
    tid = wp.tid()
    bin = bins[tid]
    slot = wp.atomic_add(counters[bin], 0, 1)
    output[bin, slot] = values[tid]


@wp.kernel(module="unique", module_options={"deterministic": "run_to_run", "deterministic_max_records": 4})
def loop_indexed_counter_kernel(
    counts: wp.array[wp.int32],
    bins: wp.array[wp.int32],
    counters: wp.array[wp.int32],
    output: wp.array2d[wp.int32],
):
    """Emit a data-dependent number of counter records to dynamic destinations."""
    tid = wp.tid()
    count = counts[tid]

    for i in range(count):
        bin = (bins[tid] + i) % 4
        slot = wp.atomic_add(counters, bin, 1)
        output[bin, slot] = tid * 10 + i


@wp.kernel
def counter_side_effect_kernel(
    counter: wp.array[wp.int32],
    output: wp.array[wp.float32],
    scratch: wp.array[wp.float32],
):
    """Counter kernel with a normal array write that must not execute in phase 0."""
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    scratch[tid] = scratch[tid] + 1.0
    output[slot] = float(tid)


@wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
def local_scratch_counter_kernel(
    counter: wp.array[wp.int32],
    output: wp.array[wp.int32],
):
    """Local scratch stores must execute in phase 0 when they control counters."""
    tid = wp.tid()
    scratch = wp.zeros(shape=(2,), dtype=wp.int32)
    _det_set_int_scratch(scratch, 0, 1)
    _det_set_int_scratch(scratch, 1, 1)

    for i in range(2):
        if scratch[i] != 0:
            slot = wp.atomic_add(counter, 0, 1)
            output[slot] = tid * 2 + i


@wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
def counter_with_atomic_xor_kernel(
    counter: wp.array[wp.int32],
    flag: wp.array[wp.int32],
    output: wp.array[wp.int32],
):
    """Counter kernel with an unintercepted ``atomic_xor`` side effect."""
    tid = wp.tid()
    wp.atomic_xor(flag, 0, 1)
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = tid


@wp.func
def _det_set_flag(flag: wp.array[wp.int32], mask: wp.int32):
    wp.atomic_xor(flag, 0, mask)


@wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
def counter_with_helper_atomic_xor_kernel(
    counter: wp.array[wp.int32],
    flag: wp.array[wp.int32],
    output: wp.array[wp.int32],
):
    """Counter kernel that delegates the bitwise atomic to a ``@wp.func`` helper."""
    tid = wp.tid()
    _det_set_flag(flag, 1)
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = tid


@wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
def counter_with_consumed_atomic_xor_kernel(
    flag: wp.array[wp.int32],
    counter: wp.array[wp.int32],
    output: wp.array[wp.int32],
):
    """Counter kernel whose control flow consumes a wrapped bitwise atomic's return."""
    tid = wp.tid()
    f = wp.atomic_xor(flag, 0, 1)
    if f != 0:
        slot = wp.atomic_add(counter, 0, 1)
        output[slot] = tid


@wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
def counter_with_component_store_kernel(
    counter: wp.array[wp.int32],
    values: wp.array[wp.vec3],
    output: wp.array[wp.int32],
):
    """Counter kernel that writes through the array-slot fast path."""
    tid = wp.tid()
    values[0].x = values[0].x + 1.0
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = tid


@wp.kernel
def mixed_pattern_kernel(
    data: wp.array[wp.float32],
    counter: wp.array[wp.int32],
    output: wp.array[wp.float32],
    accum: wp.array[wp.float32],
):
    """Counter allocation + accumulation in the same kernel."""
    tid = wp.tid()
    val = data[tid]

    # Allocate a deterministic slot from a consumed-return atomic.
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = val

    # Accumulate through the scatter/reduce path.
    wp.atomic_add(accum, tid % 8, val)


@wp.kernel
def mixed_counter_int_atomic_kernel(
    counter: wp.array[wp.int32],
    output: wp.array[wp.int32],
    accum: wp.array[wp.int32],
):
    """Integer side-effect atomic must not run during counter phase 0."""
    tid = wp.tid()
    wp.atomic_add(accum, tid % 8, 1)
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = tid


@wp.kernel
def int_atomic_add_kernel(
    dest_indices: wp.array[wp.int32],
    output: wp.array[wp.int32],
):
    """Integer atomic add (should be deterministic without transformation)."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, 1)


def test_struct_field_counter_atomic(test, device):
    """Verify deterministic counters work when the target array lives in a struct field."""
    rng = np.random.default_rng(34)
    data_np = rng.random(64, dtype=np.float32)
    counts_np = rng.integers(0, 4, size=64, dtype=np.int32)
    expected = []
    for tid, count in enumerate(counts_np):
        for i in range(int(count)):
            expected.append(np.float32(data_np[tid] + np.float32(i) * np.float32(0.5)))
    expected = np.asarray(expected, dtype=np.float32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    counts = wp.array(counts_np, dtype=wp.int32, device=device)

    def launch_once():
        writer = _DetStructCounterWriter()
        writer.counter = wp.zeros(1, dtype=wp.int32, device=device)
        writer.output = wp.zeros(expected.shape[0] + 4, dtype=wp.float32, device=device)
        wp.launch(struct_field_counter_kernel, dim=64, inputs=[data, counts, writer], device=device)
        return int(writer.counter.numpy()[0]), writer.output.numpy()[: expected.shape[0]].copy()

    counter_value, result = assert_equal_repeated(launch_once)
    test.assertEqual(counter_value, expected.shape[0])
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


def test_struct_field_helper_counter_atomic(test, device):
    """Verify helper-function counters work when the target array lives in a struct field."""
    rng = np.random.default_rng(35)
    data_np = rng.random(64, dtype=np.float32)
    flags_np = (rng.random(64) > 0.4).astype(np.int32)
    expected = data_np[flags_np != 0]

    data = wp.array(data_np, dtype=wp.float32, device=device)
    flags = wp.array(flags_np, dtype=wp.int32, device=device)

    def launch_once():
        writer = _DetStructCounterWriter()
        writer.counter = wp.zeros(1, dtype=wp.int32, device=device)
        writer.output = wp.zeros(expected.shape[0] + 4, dtype=wp.float32, device=device)
        wp.launch(struct_field_helper_counter_kernel, dim=64, inputs=[data, flags, writer], device=device)
        return int(writer.counter.numpy()[0]), writer.output.numpy()[: expected.shape[0]].copy()

    counter_value, result = assert_equal_repeated(launch_once)
    test.assertEqual(counter_value, expected.shape[0])
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


def test_counter_reproducibility(test, device):
    """Verify consumed-return counters produce deterministic slot assignments."""
    n = 1024
    rng = np.random.default_rng(33)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    def launch_once():
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(
            counter_kernel,
            dim=n,
            inputs=[data, counter],
            outputs=[output],
            device=device,
        )
        return output.numpy().copy()

    assert_equal_repeated(launch_once, err_msg="counter runs differ")


def test_counter_phase0_suppresses_array_writes(test, device):
    """Verify non-counter array stores are skipped during the counting pass."""
    n = 128
    for kernel in (
        counter_side_effect_kernel,
        helper_counter_side_effect_kernel,
        counter_with_helper_store_kernel,
    ):
        with test.subTest(kernel=kernel.key):
            counter = wp.zeros(1, dtype=wp.int32, device=device)
            output = wp.zeros(n, dtype=wp.float32, device=device)
            scratch = wp.zeros(n, dtype=wp.float32, device=device)
            wp.launch(kernel, dim=n, inputs=[counter], outputs=[output, scratch], device=device)

            np.testing.assert_array_equal(scratch.numpy(), np.ones(n, dtype=np.float32))
            test.assertEqual(int(counter.numpy()[0]), n)


def test_counter_phase0_preserves_local_scratch_writes(test, device):
    """Verify phase 0 executes local scratch stores that affect counter calls."""
    n = 64
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.full(2 * n, value=-1, dtype=wp.int32, device=device)

    wp.launch(local_scratch_counter_kernel, dim=n, inputs=[counter], outputs=[output], device=device)

    test.assertEqual(int(counter.numpy()[0]), 2 * n)
    np.testing.assert_array_equal(output.numpy(), np.arange(2 * n, dtype=np.int32))


def test_counter_phase0_skips_unconsumed_bitwise_atomics(test, device):
    """Verify unconsumed bitwise atomics in counter kernels fire once, not twice."""
    for kernel in (counter_with_atomic_xor_kernel, counter_with_helper_atomic_xor_kernel):
        with test.subTest(kernel=kernel.key):
            counter = wp.zeros(1, dtype=wp.int32, device=device)
            flag = wp.zeros(1, dtype=wp.int32, device=device)
            output = wp.full(1, value=-1, dtype=wp.int32, device=device)
            wp.launch(kernel, dim=1, inputs=[counter, flag], outputs=[output], device=device)

            test.assertEqual(int(flag.numpy()[0]), 1)
            test.assertEqual(int(counter.numpy()[0]), 1)
            test.assertEqual(int(output.numpy()[0]), 0)


def test_counter_consumed_bitwise_atomic_rejected(test, device):
    """Consuming a bitwise atomic return inside a two-pass body must be rejected."""
    flag = wp.full(1, value=1, dtype=wp.int32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.full(1, value=-1, dtype=wp.int32, device=device)

    with test.assertRaisesRegex(Exception, r"wp\.atomic_xor"):
        wp.launch(
            counter_with_consumed_atomic_xor_kernel,
            dim=1,
            inputs=[flag, counter],
            outputs=[output],
            device=device,
        )


def test_counter_phase0_suppresses_component_stores(test, device):
    """Verify component-level array stores do not double-execute in counter mode."""
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    values = wp.zeros(1, dtype=wp.vec3, device=device)
    output = wp.full(1, value=-1, dtype=wp.int32, device=device)

    wp.launch(counter_with_component_store_kernel, dim=1, inputs=[counter, values], outputs=[output], device=device)

    np.testing.assert_array_equal(values.numpy(), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    test.assertEqual(int(counter.numpy()[0]), 1)
    test.assertEqual(int(output.numpy()[0]), 0)


def test_counter_correctness(test, device):
    """Verify consumed-return counters write all data (no lost elements)."""
    n = 512
    rng = np.random.default_rng(44)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(
        counter_kernel,
        dim=n,
        inputs=[data, counter],
        outputs=[output],
        device=device,
    )

    # Counter should equal n.
    count = int(counter.numpy()[0])
    test.assertEqual(count, n)

    # Output should contain a permutation of data.
    result = sorted(output.numpy().tolist())
    expected = sorted(data_np.tolist())
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_counter_nonzero_initial_value(test, device):
    """Verify consumed-return counters preserve a non-zero initial value.

    Before the fix to ``make_counter_prefix_states_kernel`` the deterministic
    prefix scan started from zero, ignoring the counter's existing value.  In
    real multi-launch workloads (e.g. particle emission across timesteps) this
    silently overwrote slots from earlier launches and produced a wrong final
    count.
    """
    n = 64
    k = 20  # non-zero initial counter value

    rng = np.random.default_rng(777)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    counter = wp.full(1, value=k, dtype=wp.int32, device=device)
    sentinel = np.float32(-7.0)
    output = wp.full(k + n, value=sentinel, dtype=wp.float32, device=device)

    wp.launch(counter_kernel, dim=n, inputs=[data, counter], outputs=[output], device=device)

    # Counter must reflect the seed plus the n new allocations.
    test.assertEqual(int(counter.numpy()[0]), k + n)

    # Slots [0, K) must be untouched.
    np.testing.assert_array_equal(
        output.numpy()[:k],
        np.full(k, sentinel, dtype=np.float32),
        err_msg="Initial slots were overwritten -- counter scan ignored its seed.",
    )

    # Slots [K, K+n) must contain a permutation of the input data.
    written = sorted(output.numpy()[k:].tolist())
    expected = sorted(data_np.tolist())
    np.testing.assert_allclose(written, expected, rtol=1e-6)


def test_counter_multi_launch_accumulates(test, device):
    """Verify consumed-return counters accumulate correctly across launches.

    Realistic multi-launch use case (e.g. emitting particles over several
    timesteps): each launch must append its allocations on top of the previous
    launches' slots, not overwrite them.
    """
    n = 32
    num_launches = 3
    capacity = n * num_launches

    rng = np.random.default_rng(778)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.full(capacity, value=np.float32(-1.0), dtype=wp.float32, device=device)

    all_data = []
    for _ in range(num_launches):
        data_np = rng.random(n, dtype=np.float32)
        all_data.append(data_np)
        data = wp.array(data_np, dtype=wp.float32, device=device)
        wp.launch(counter_kernel, dim=n, inputs=[data, counter], outputs=[output], device=device)

    test.assertEqual(int(counter.numpy()[0]), capacity)

    # All n*num_launches values must be present in the output, none overwritten.
    written = sorted(output.numpy().tolist())
    expected = sorted(np.concatenate(all_data).tolist())
    np.testing.assert_allclose(written, expected, rtol=1e-6)


def test_counter_variable_total_writeback(test, device):
    """Verify deterministic counters publish variable total counts on device."""
    counts_np = np.array([2, 0, 3, 1, 0, 4, 1, 0], dtype=np.int32)
    expected = []
    for tid, count in enumerate(counts_np):
        expected.extend([tid * 10 + i for i in range(count)])
    expected_np = np.array(expected, dtype=np.int32)

    counts = wp.array(counts_np, dtype=wp.int32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.full(expected_np.shape[0], value=-1, dtype=wp.int32, device=device)

    wp.launch(
        variable_counter_kernel, dim=counts_np.shape[0], inputs=[counts, counter], outputs=[output], device=device
    )

    np.testing.assert_array_equal(counter.numpy(), np.array([expected_np.shape[0]], dtype=np.int32))
    np.testing.assert_array_equal(output.numpy(), expected_np)


def test_counter_nonzero_index(test, device):
    """Verify consumed-return counters support fixed nonzero indices."""
    n = 32

    def launch_once():
        counter = wp.zeros(2, dtype=wp.int32, device=device)
        output = wp.full(n, value=-1, dtype=wp.int32, device=device)
        wp.launch(static_index_counter_kernel, dim=n, inputs=[counter], outputs=[output], device=device)
        return counter.numpy().copy(), output.numpy().copy()

    expected_counter = np.array([0, n], dtype=np.int32)
    expected_output = np.arange(n, dtype=np.int32)
    counter, result = assert_equal_repeated(launch_once)
    np.testing.assert_array_equal(counter, expected_counter)
    np.testing.assert_array_equal(result, expected_output)


def test_counter_indexed_destinations(test, device):
    """Verify consumed-return counters support data-dependent counter indices."""
    n = 128
    bin_count = 5
    rng = np.random.default_rng(92)
    bins_np = rng.integers(0, bin_count, size=n, dtype=np.int32)
    values_np = np.arange(n, dtype=np.int32) + 100

    expected_counts = np.bincount(bins_np, minlength=bin_count).astype(np.int32)
    expected_output = np.full((bin_count, n), -1, dtype=np.int32)
    offsets = np.zeros(bin_count, dtype=np.int32)
    for tid, bin in enumerate(bins_np):
        slot = offsets[bin]
        expected_output[bin, slot] = values_np[tid]
        offsets[bin] += 1

    values = wp.array(values_np, dtype=wp.int32, device=device)
    bins = wp.array(bins_np, dtype=wp.int32, device=device)

    def launch_once():
        counter = wp.zeros(bin_count, dtype=wp.int32, device=device)
        output = wp.full((bin_count, n), value=-1, dtype=wp.int32, device=device)
        wp.launch(indexed_counter_kernel, dim=n, inputs=[values, bins, counter], outputs=[output], device=device)
        return counter.numpy().copy(), output.numpy().copy()

    counter, result = assert_equal_repeated(launch_once)
    np.testing.assert_array_equal(counter, expected_counts)
    np.testing.assert_array_equal(result, expected_output)


def test_counter_sliced_destinations(test, device):
    """Verify consumed-return counters support sliced counter-array views."""
    n = 96
    bin_count = 4
    bins_np = (np.arange(n, dtype=np.int32) * 3) % bin_count
    values_np = np.arange(n, dtype=np.int32) + 1000

    expected_counts = np.bincount(bins_np, minlength=bin_count).astype(np.int32).reshape(bin_count, 1)
    expected_output = np.full((bin_count, n), -1, dtype=np.int32)
    offsets = np.zeros(bin_count, dtype=np.int32)
    for tid, bin in enumerate(bins_np):
        slot = offsets[bin]
        expected_output[bin, slot] = values_np[tid]
        offsets[bin] += 1

    values = wp.array(values_np, dtype=wp.int32, device=device)
    bins = wp.array(bins_np, dtype=wp.int32, device=device)

    counters = wp.zeros((bin_count, 1), dtype=wp.int32, device=device)
    output = wp.full((bin_count, n), value=-1, dtype=wp.int32, device=device)
    wp.launch(sliced_counter_kernel, dim=n, inputs=[values, bins, counters], outputs=[output], device=device)

    np.testing.assert_array_equal(counters.numpy(), expected_counts)
    np.testing.assert_array_equal(output.numpy(), expected_output)


def test_counter_indexed_dynamic_loop(test, device):
    """Verify dynamic loops can reserve multiple records for dynamic counter indices."""
    n = 64
    bin_count = 4
    counts_np = (np.arange(n, dtype=np.int32) % 4).astype(np.int32)
    bins_np = ((np.arange(n, dtype=np.int32) * 5) % bin_count).astype(np.int32)

    expected_counts = np.zeros(bin_count, dtype=np.int32)
    expected_output = np.full((bin_count, n * 4), -1, dtype=np.int32)
    for tid in range(n):
        for i in range(counts_np[tid]):
            bin = (bins_np[tid] + i) % bin_count
            slot = expected_counts[bin]
            expected_output[bin, slot] = tid * 10 + i
            expected_counts[bin] += 1

    counts = wp.array(counts_np, dtype=wp.int32, device=device)
    bins = wp.array(bins_np, dtype=wp.int32, device=device)
    counters = wp.zeros(bin_count, dtype=wp.int32, device=device)
    output = wp.full((bin_count, n * 4), value=-1, dtype=wp.int32, device=device)

    wp.launch(loop_indexed_counter_kernel, dim=n, inputs=[counts, bins, counters], outputs=[output], device=device)

    np.testing.assert_array_equal(counters.numpy(), expected_counts)
    np.testing.assert_array_equal(output.numpy(), expected_output)


def test_counter_int64_rejected(test, device):
    """Verify unsupported non-int32 consumed-return counters fail clearly."""
    with test.assertRaisesRegex(Exception, "int32 counter arrays"):

        @wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
        def counter_int64_kernel(
            counter: wp.array[wp.int64],
            output: wp.array[wp.float32],
        ):
            """Unsupported non-int32 consumed-return counter."""
            tid = wp.tid()
            slot = wp.atomic_add(counter, 0, wp.int64(1))
            output[slot] = float(tid)

        counter = wp.zeros(1, dtype=wp.int64, device=device)
        output = wp.zeros(8, dtype=wp.float32, device=device)
        wp.launch(counter_int64_kernel, dim=8, inputs=[counter], outputs=[output], device=device)


def test_counter_non_add_atomic_rejected(test, device):
    """Verify consumed-return deterministic counters only accept ``atomic_add``."""
    with test.assertRaisesRegex(Exception, "only for atomic_add"):

        @wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
        def counter_sub_kernel(
            counter: wp.array[wp.int32],
            output: wp.array[wp.float32],
        ):
            """Unsupported consumed-return counter using ``atomic_sub``."""
            tid = wp.tid()
            slot = wp.atomic_sub(counter, 0, 1)
            output[slot] = float(tid)

        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(8, dtype=wp.float32, device=device)
        wp.launch(counter_sub_kernel, dim=8, inputs=[counter], outputs=[output], device=device)

    with test.assertRaisesRegex(Exception, "only for atomic_add"):

        @wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
        def counter_max_kernel(
            counter: wp.array[wp.int32],
            output: wp.array[wp.float32],
        ):
            """Unsupported consumed-return counter using ``atomic_max``."""
            tid = wp.tid()
            old = wp.atomic_max(counter, 0, tid)
            output[tid] = float(old)

        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(8, dtype=wp.float32, device=device)
        wp.launch(counter_max_kernel, dim=8, inputs=[counter], outputs=[output], device=device)


def test_float_atomic_consumed_return_rejected(test, device):
    """Verify float atomics with consumed-not-assigned returns fail clearly."""
    pattern = "consumed-return counter atomics only for int32 counter arrays"

    with test.assertRaisesRegex(Exception, pattern):

        @wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
        def float_leader_if_kernel(
            total: wp.array[wp.float32],
            log: wp.array[wp.int32],
        ):
            """Return consumed by an ``if`` test."""
            tid = wp.tid()
            if wp.atomic_add(total, 0, 1.0) == 0.0:
                log[0] = tid

        total = wp.zeros(1, dtype=wp.float32, device=device)
        log = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(float_leader_if_kernel, dim=8, inputs=[total], outputs=[log], device=device)

    with test.assertRaisesRegex(Exception, pattern):

        @wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
        def float_nested_call_in_if_kernel(
            total: wp.array[wp.float32],
            out: wp.array[wp.float32],
        ):
            """Return consumed by a nested call inside an ``if`` test."""
            tid = wp.tid()
            if wp.abs(wp.atomic_add(total, 0, 1.0)) > 0.0:
                out[tid] = 1.0

        total = wp.zeros(1, dtype=wp.float32, device=device)
        out = wp.zeros(8, dtype=wp.float32, device=device)
        wp.launch(float_nested_call_in_if_kernel, dim=8, inputs=[total], outputs=[out], device=device)

    with test.assertRaisesRegex(Exception, pattern):

        @wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
        def float_nested_bare_atomic_kernel(
            total: wp.array[wp.float32],
            other: wp.array[wp.float32],
        ):
            """Return consumed as the value argument of a bare atomic statement."""
            wp.atomic_add(total, 0, wp.atomic_add(other, 0, 1.0))

        total = wp.zeros(1, dtype=wp.float32, device=device)
        other = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(float_nested_bare_atomic_kernel, dim=8, inputs=[total], outputs=[other], device=device)


def test_float_atomic_bare_statement_allowed(test, device):
    """Verify bare-statement float atomics still lower to the scatter path."""

    @wp.kernel(module="unique", module_options={"deterministic": "run_to_run"})
    def bare_float_accum_kernel(
        contribs: wp.array[wp.float32],
        accum: wp.array[wp.float32],
    ):
        tid = wp.tid()
        wp.atomic_add(accum, tid % 4, contribs[tid])

    n = 32
    rng = np.random.default_rng(0)
    contribs_np = rng.random(n, dtype=np.float32)
    contribs = wp.array(contribs_np, dtype=wp.float32, device=device)
    accum = wp.zeros(4, dtype=wp.float32, device=device)

    wp.launch(bare_float_accum_kernel, dim=n, inputs=[contribs], outputs=[accum], device=device)

    expected = np.zeros(4, dtype=np.float32)
    for i in range(n):
        expected[i % 4] += contribs_np[i]

    np.testing.assert_allclose(accum.numpy(), expected, rtol=0, atol=1e-5)


def test_conditional_counter(test, device):
    """Verify stream compaction with a conditional counter."""
    n = 2048
    threshold = 0.5
    rng = np.random.default_rng(55)
    data_np = rng.random(n, dtype=np.float32)
    expected_count = int(np.sum(data_np > threshold))

    data = wp.array(data_np, dtype=wp.float32, device=device)

    def launch_once():
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(
            conditional_counter_kernel,
            dim=n,
            inputs=[data, threshold, counter],
            outputs=[output],
            device=device,
        )
        return int(counter.numpy()[0]), output.numpy()[:expected_count].copy()

    count, _ = assert_equal_repeated(launch_once)
    test.assertEqual(count, expected_count)


def test_mixed_pattern(test, device):
    """Verify kernel with both counter and accumulation atomics."""
    n = 512
    rng = np.random.default_rng(77)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    def launch_once():
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(n, dtype=wp.float32, device=device)
        accum = wp.zeros(8, dtype=wp.float32, device=device)
        wp.launch(
            mixed_pattern_kernel,
            dim=n,
            inputs=[data, counter],
            outputs=[output, accum],
            device=device,
        )
        return output.numpy().copy(), accum.numpy().copy()

    assert_equal_repeated(launch_once)


def test_counter_with_integer_accumulation(test, device):
    """Verify integer atomics in counter kernels do not run during phase 0."""
    n = 512
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.int32, device=device)
    accum = wp.zeros(8, dtype=wp.int32, device=device)

    wp.launch(mixed_counter_int_atomic_kernel, dim=n, inputs=[counter], outputs=[output, accum], device=device)

    np.testing.assert_array_equal(counter.numpy(), [n])
    np.testing.assert_array_equal(accum.numpy(), np.full(8, n // 8, dtype=np.int32))


def test_int_atomic_passthrough(test, device):
    """Verify integer atomics (return unused) work without overhead."""
    n = 1024
    out_size = 16
    rng = np.random.default_rng(11)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    output = wp.zeros(out_size, dtype=wp.int32, device=device)
    wp.launch(
        int_atomic_add_kernel,
        dim=n,
        inputs=[indices],
        outputs=[output],
        device=device,
    )

    result = output.numpy()
    # Verify correctness: count of each index.
    expected = np.bincount(indices_np, minlength=out_size).astype(np.int32)
    np.testing.assert_array_equal(result, expected)


class TestDeterministicCounter(DeterministicTestBase):
    """Test deterministic consumed-return counter lowering."""


def _add(name, devices=cuda_devices):
    add_function_test(TestDeterministicCounter, name, globals()[name], devices=devices)


for _name in (
    "test_struct_field_counter_atomic",
    "test_struct_field_helper_counter_atomic",
    "test_counter_reproducibility",
    "test_counter_phase0_suppresses_array_writes",
    "test_counter_phase0_preserves_local_scratch_writes",
    "test_counter_phase0_skips_unconsumed_bitwise_atomics",
    "test_counter_consumed_bitwise_atomic_rejected",
    "test_counter_phase0_suppresses_component_stores",
    "test_counter_nonzero_initial_value",
    "test_counter_multi_launch_accumulates",
    "test_counter_variable_total_writeback",
    "test_counter_nonzero_index",
    "test_counter_indexed_destinations",
    "test_counter_sliced_destinations",
    "test_counter_indexed_dynamic_loop",
    "test_counter_int64_rejected",
    "test_counter_non_add_atomic_rejected",
    "test_float_atomic_consumed_return_rejected",
    "test_float_atomic_bare_statement_allowed",
    "test_conditional_counter",
    "test_mixed_pattern",
    "test_counter_with_integer_accumulation",
):
    _add(_name)

_add("test_counter_correctness", devices=all_devices)
_add("test_int_atomic_passthrough", devices=all_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
