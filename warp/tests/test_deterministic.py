# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic execution mode.

Validates that ``wp.config.deterministic = True`` produces bit-exact
reproducible results for atomic operations across multiple runs.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# ---------------------------------------------------------------------------
# Pattern A kernels: accumulation (return value unused)
# ---------------------------------------------------------------------------


@wp.kernel
def scatter_add_kernel(
    data: wp.array(dtype=wp.float32),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Each thread atomically adds to output[dest_indices[tid]]."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, data[tid])


@wp.kernel
def augassign_add_kernel(
    data: wp.array(dtype=wp.float32),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Same as scatter_add_kernel but using += syntax."""
    tid = wp.tid()
    idx = dest_indices[tid]
    output[idx] += data[tid]


@wp.kernel
def multi_array_atomic_kernel(
    data: wp.array(dtype=wp.float32),
    dest_indices: wp.array(dtype=wp.int32),
    out_a: wp.array(dtype=wp.float32),
    out_b: wp.array(dtype=wp.float32),
    out_c: wp.array(dtype=wp.float32),
):
    """Atomic add to three different output arrays from the same kernel."""
    tid = wp.tid()
    idx = dest_indices[tid]
    val = data[tid]
    wp.atomic_add(out_a, idx, val)
    wp.atomic_add(out_b, idx, val * 2.0)
    wp.atomic_add(out_c, idx, val * 3.0)


@wp.kernel
def atomic_sub_kernel(
    data: wp.array(dtype=wp.float32),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Atomic sub test."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_sub(output, idx, data[tid])


@wp.kernel
def atomic_add_2d_kernel(
    data: wp.array(dtype=wp.float32),
    row_indices: wp.array(dtype=wp.int32),
    col_indices: wp.array(dtype=wp.int32),
    output: wp.array2d(dtype=wp.float32),
):
    """Atomic add to a 2D array."""
    tid = wp.tid()
    r = row_indices[tid]
    c = col_indices[tid]
    wp.atomic_add(output, r, c, data[tid])


@wp.kernel
def atomic_double_kernel(
    data: wp.array(dtype=wp.float64),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float64),
):
    """Atomic add with float64."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, data[tid])


# ---------------------------------------------------------------------------
# Pattern B kernels: counter/allocator (return value used)
# ---------------------------------------------------------------------------


@wp.kernel
def counter_kernel(
    data: wp.array(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Allocate a slot and write data to it."""
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = data[tid]


@wp.kernel
def conditional_counter_kernel(
    data: wp.array(dtype=wp.float32),
    threshold: wp.float32,
    counter: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Stream compaction: only emit elements above threshold."""
    tid = wp.tid()
    val = data[tid]
    if val > threshold:
        slot = wp.atomic_add(counter, 0, 1)
        output[slot] = val


# ---------------------------------------------------------------------------
# Mixed kernels: both patterns in one kernel
# ---------------------------------------------------------------------------


@wp.kernel
def mixed_pattern_kernel(
    data: wp.array(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
    accum: wp.array(dtype=wp.float32),
):
    """Counter allocation + accumulation in the same kernel."""
    tid = wp.tid()
    val = data[tid]

    # Pattern B: allocate slot
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = val

    # Pattern A: accumulate
    wp.atomic_add(accum, tid % 8, val)


# ---------------------------------------------------------------------------
# Integer atomic kernels (should pass through unchanged when return unused)
# ---------------------------------------------------------------------------


@wp.kernel
def int_atomic_add_kernel(
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32),
):
    """Integer atomic add (should be deterministic without transformation)."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, 1)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_scatter_add_reproducibility(test, device):
    """Verify that float atomic_add produces bit-exact identical results across runs."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 4096
    out_size = 64
    rng = np.random.default_rng(42)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(10):
        output = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(
            scatter_add_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    # All runs must produce bit-exact identical results.
    for i in range(1, len(results)):
        np.testing.assert_array_equal(
            results[0],
            results[i],
            err_msg=f"Run 0 vs run {i} differ (deterministic mode should be bit-exact)",
        )


def test_augassign_add_reproducibility(test, device):
    """Verify += syntax (desugars to atomic_add) is also deterministic."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 2048
    out_size = 32
    rng = np.random.default_rng(123)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(10):
        output = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(
            augassign_add_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_scatter_add_correctness(test, device):
    """Compare deterministic GPU results against CPU sequential execution."""
    n = 2048
    out_size = 32
    rng = np.random.default_rng(99)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    # CPU sequential reference (guaranteed deterministic).
    expected = np.zeros(out_size, dtype=np.float32)
    for i in range(n):
        expected[indices_np[i]] += data_np[i]

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(out_size, dtype=wp.float32, device=device)

    wp.launch(
        scatter_add_kernel,
        dim=n,
        inputs=[data, indices],
        outputs=[output],
        device=device,
    )

    result = output.numpy()
    # Deterministic sum order may differ from Python loop order, so exact
    # match is not guaranteed.  Check within reasonable tolerance.
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


def test_multi_array_atomic(test, device):
    """Verify deterministic mode works with multiple target arrays."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 1024
    out_size = 16
    rng = np.random.default_rng(77)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results_a, results_b, results_c = [], [], []
    for _ in range(5):
        out_a = wp.zeros(out_size, dtype=wp.float32, device=device)
        out_b = wp.zeros(out_size, dtype=wp.float32, device=device)
        out_c = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(
            multi_array_atomic_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[out_a, out_b, out_c],
            device=device,
        )
        results_a.append(out_a.numpy().copy())
        results_b.append(out_b.numpy().copy())
        results_c.append(out_c.numpy().copy())

    for i in range(1, len(results_a)):
        np.testing.assert_array_equal(results_a[0], results_a[i])
        np.testing.assert_array_equal(results_b[0], results_b[i])
        np.testing.assert_array_equal(results_c[0], results_c[i])


def test_atomic_sub_deterministic(test, device):
    """Verify atomic_sub is deterministic."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 2048
    out_size = 32
    rng = np.random.default_rng(55)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(
            atomic_sub_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_atomic_add_2d(test, device):
    """Verify deterministic mode with 2D array indexing."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 1024
    rows, cols = 8, 8
    rng = np.random.default_rng(88)

    data_np = rng.random(n, dtype=np.float32)
    row_np = rng.integers(0, rows, size=n, dtype=np.int32)
    col_np = rng.integers(0, cols, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    row_idx = wp.array(row_np, dtype=wp.int32, device=device)
    col_idx = wp.array(col_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(shape=(rows, cols), dtype=wp.float32, device=device)
        wp.launch(
            atomic_add_2d_kernel,
            dim=n,
            inputs=[data, row_idx, col_idx],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_atomic_double_deterministic(test, device):
    """Verify deterministic mode with float64 atomics."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 2048
    out_size = 32
    rng = np.random.default_rng(66)

    data_np = rng.random(n).astype(np.float64)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float64, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(out_size, dtype=wp.float64, device=device)
        wp.launch(
            atomic_double_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_counter_reproducibility(test, device):
    """Verify counter/allocator pattern produces deterministic slot assignments."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 1024
    rng = np.random.default_rng(33)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    results = []
    for _ in range(10):
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(
            counter_kernel,
            dim=n,
            inputs=[data, counter],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(
            results[0],
            results[i],
            err_msg=f"Counter run 0 vs run {i} differ",
        )


def test_counter_correctness(test, device):
    """Verify counter pattern writes all data (no lost elements)."""
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


def test_conditional_counter(test, device):
    """Verify stream compaction pattern with conditional counter."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 2048
    threshold = 0.5
    rng = np.random.default_rng(55)
    data_np = rng.random(n, dtype=np.float32)
    expected_count = int(np.sum(data_np > threshold))

    data = wp.array(data_np, dtype=wp.float32, device=device)

    results = []
    counts = []
    for _ in range(5):
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(
            conditional_counter_kernel,
            dim=n,
            inputs=[data, threshold, counter],
            outputs=[output],
            device=device,
        )
        counts.append(int(counter.numpy()[0]))
        results.append(output.numpy()[:expected_count].copy())

    # Count should be correct.
    for c in counts:
        test.assertEqual(c, expected_count)

    # Results should be identical across runs.
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_mixed_pattern(test, device):
    """Verify kernel with both counter and accumulation atomics."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 512
    rng = np.random.default_rng(77)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    results_out, results_accum = [], []
    for _ in range(5):
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
        results_out.append(output.numpy().copy())
        results_accum.append(accum.numpy().copy())

    for i in range(1, len(results_out)):
        np.testing.assert_array_equal(results_out[0], results_out[i])
        np.testing.assert_array_equal(results_accum[0], results_accum[i])


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


def test_module_option_override(test, device):
    """Verify per-module deterministic option works."""

    # Create a kernel with per-module deterministic=True override.
    @wp.kernel(module_options={"deterministic": True}, module="unique")
    def per_kernel_det(
        data: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        wp.atomic_add(output, tid % 4, data[tid])

    n = 256
    rng = np.random.default_rng(22)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    # Ensure global config is False but per-kernel override still works.
    old_det = wp.config.deterministic
    try:
        wp.config.deterministic = False
        output = wp.zeros(4, dtype=wp.float32, device=device)
        wp.launch(per_kernel_det, dim=n, inputs=[data], outputs=[output], device=device)
        result = output.numpy()
        # Basic sanity: sum should be approximately correct.
        for bin_idx in range(4):
            mask = np.arange(n) % 4 == bin_idx
            expected_sum = data_np[mask].sum()
            np.testing.assert_allclose(result[bin_idx], expected_sum, rtol=1e-4)
    finally:
        wp.config.deterministic = old_det


# ---------------------------------------------------------------------------
# Test class registration
# ---------------------------------------------------------------------------

cuda_devices = get_selected_cuda_test_devices()
all_devices = get_test_devices()


class TestDeterministic(unittest.TestCase):
    """Test suite for deterministic execution mode."""

    @classmethod
    def setUpClass(cls):
        cls._old_deterministic = wp.config.deterministic
        wp.config.deterministic = True

    @classmethod
    def tearDownClass(cls):
        wp.config.deterministic = cls._old_deterministic


# Pattern A tests (accumulation).
add_function_test(
    TestDeterministic, "test_scatter_add_reproducibility", test_scatter_add_reproducibility, devices=cuda_devices
)
add_function_test(
    TestDeterministic, "test_augassign_add_reproducibility", test_augassign_add_reproducibility, devices=cuda_devices
)
add_function_test(TestDeterministic, "test_scatter_add_correctness", test_scatter_add_correctness, devices=all_devices)
add_function_test(TestDeterministic, "test_multi_array_atomic", test_multi_array_atomic, devices=cuda_devices)
add_function_test(
    TestDeterministic, "test_atomic_sub_deterministic", test_atomic_sub_deterministic, devices=cuda_devices
)
add_function_test(TestDeterministic, "test_atomic_add_2d", test_atomic_add_2d, devices=cuda_devices)
add_function_test(
    TestDeterministic, "test_atomic_double_deterministic", test_atomic_double_deterministic, devices=cuda_devices
)

# Pattern B tests (counter).
add_function_test(TestDeterministic, "test_counter_reproducibility", test_counter_reproducibility, devices=cuda_devices)
add_function_test(TestDeterministic, "test_counter_correctness", test_counter_correctness, devices=all_devices)
add_function_test(TestDeterministic, "test_conditional_counter", test_conditional_counter, devices=cuda_devices)

# Mixed pattern tests.
add_function_test(TestDeterministic, "test_mixed_pattern", test_mixed_pattern, devices=cuda_devices)

# Passthrough / override tests.
add_function_test(TestDeterministic, "test_int_atomic_passthrough", test_int_atomic_passthrough, devices=all_devices)
add_function_test(TestDeterministic, "test_module_option_override", test_module_option_override, devices=all_devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
