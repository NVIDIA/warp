# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic execution mode.

Validates that deterministic modes produce bit-exact reproducible results for
atomic operations across multiple runs.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


def _reference_scatter_add_float32(data_np, indices_np, out_size):
    """Compute the canonical left-to-right float32 scatter reduction on CPU."""
    expected = np.zeros(out_size, dtype=np.float32)
    for value, index in zip(data_np, indices_np, strict=True):
        expected[index] = np.float32(expected[index] + value)
    return expected


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


@wp.kernel
def vec3_scatter_add_kernel(
    data: wp.array(dtype=wp.vec3),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.vec3),
):
    """Atomic add with ``wp.vec3`` values."""
    tid = wp.tid()
    wp.atomic_add(output, dest_indices[tid], data[tid])


@wp.kernel
def vec3_atomic_minmax_kernel(
    points: wp.array(dtype=wp.vec3),
    out_min: wp.array(dtype=wp.vec3),
    out_max: wp.array(dtype=wp.vec3),
):
    """Component-wise deterministic min/max for bounding-box style reductions."""
    tid = wp.tid()
    p = points[tid]
    wp.atomic_min(out_min, 0, p)
    wp.atomic_max(out_max, 0, p)


@wp.kernel
def mat33_scatter_add_kernel(
    data: wp.array(dtype=wp.mat33),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.mat33),
):
    """Atomic add with ``wp.mat33`` values."""
    tid = wp.tid()
    wp.atomic_add(output, dest_indices[tid], data[tid])


@wp.kernel(deterministic="gpu_to_gpu", deterministic_max_records=4096)
def decorator_deterministic_kernel(
    data: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
):
    """Kernel-level GPU-to-GPU deterministic flag without module options."""
    tid = wp.tid()
    wp.atomic_add(output, tid % 8, data[tid])


@wp.func
def _det_closure_transform_a(x: wp.float32) -> wp.float32:
    return x + wp.float32(1.0)


@wp.func
def _det_closure_transform_b(x: wp.float32) -> wp.float32:
    return x + wp.float32(2.0)


def _make_deterministic_closure_kernel(transform_func):
    @wp.kernel(deterministic=True, module="unique")
    def _deterministic_closure_kernel(
        data: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        wp.atomic_add(output, tid % 8, transform_func(data[tid]))

    return _deterministic_closure_kernel


@wp.kernel
def triple_scatter_add_kernel(
    data: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
):
    """Emit three deterministic scatter records per thread to the same target."""
    tid = wp.tid()
    val = data[tid]
    wp.atomic_add(output, 0, val)
    wp.atomic_add(output, 0, val * 2.0)
    wp.atomic_add(output, 0, val * 3.0)


@wp.kernel(deterministic=True, deterministic_max_records=4)
def loop_scatter_add_kernel(
    data: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Emit a data-dependent number of scatter records to the same target."""
    tid = wp.tid()
    val = data[tid]
    count = counts[tid]
    for _ in range(count):
        wp.atomic_add(output, 0, val)


@wp.kernel
def mixed_reduce_op_same_array_kernel(
    data: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
):
    """Apply different atomic reductions to the same destination array."""
    tid = wp.tid()
    wp.atomic_add(output, 0, data[tid])
    wp.atomic_max(output, 0, 1.0)


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


@wp.kernel
def counter_side_effect_kernel(
    counter: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
    scratch: wp.array(dtype=wp.float32),
):
    """Counter kernel with a normal array write that must not execute in phase 0."""
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    scratch[tid] = scratch[tid] + 1.0
    output[slot] = float(tid)


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


def test_gpu_to_gpu_mode_reproducibility(test, device):
    """Verify the global ``gpu_to_gpu`` mode produces reproducible results."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 1024
    out_size = 16
    rng = np.random.default_rng(44)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    old_det = wp.config.deterministic
    try:
        wp.config.deterministic = "gpu_to_gpu"
        results = []
        for _ in range(3):
            output = wp.zeros(out_size, dtype=wp.float32, device=device)
            wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
            results.append(output.numpy().copy())
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
    finally:
        wp.config.deterministic = old_det


def test_gpu_to_gpu_matches_canonical_float32_reference(test, device):
    """Verify ``gpu_to_gpu`` matches the canonical float32 CPU reduction order."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    data_np = np.array(
        [
            1.0e20,
            1.0,
            -1.0e20,
            3.5,
            -2.25,
            2.0**-20,
            1.0e10,
            -1.0e10,
            7.0,
            -7.0,
            9.0,
            1.0e-7,
        ],
        dtype=np.float32,
    )
    indices_np = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 2, 2], dtype=np.int32)
    out_size = 3

    expected = _reference_scatter_add_float32(data_np, indices_np, out_size)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    old_det = wp.config.deterministic
    try:
        wp.config.deterministic = "gpu_to_gpu"
        output = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(scatter_add_kernel, dim=data_np.shape[0], inputs=[data, indices], outputs=[output], device=device)
        result = output.numpy()
    finally:
        wp.config.deterministic = old_det

    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


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


def test_vec3_atomic_add_deterministic(test, device):
    """Verify deterministic mode for composite ``wp.vec3`` atomic adds."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 1024
    out_size = 16
    rng = np.random.default_rng(67)

    data_np = rng.standard_normal((n, 3), dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(out_size, dtype=wp.vec3, device=device)
        wp.launch(vec3_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
        results.append(output.numpy().copy())

    expected = np.zeros((out_size, 3), dtype=np.float32)
    for i in range(n):
        expected[indices_np[i]] += data_np[i]

    for result in results:
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_vec3_atomic_minmax_deterministic(test, device):
    """Verify deterministic component-wise ``wp.vec3`` min/max reductions."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 2048
    rng = np.random.default_rng(68)
    points_np = rng.standard_normal((n, 3), dtype=np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)

    mins = []
    maxs = []
    for _ in range(5):
        out_min = wp.empty(1, dtype=wp.vec3, device=device)
        out_max = wp.empty(1, dtype=wp.vec3, device=device)
        out_min.fill_(wp.vec3(np.inf, np.inf, np.inf))
        out_max.fill_(wp.vec3(-np.inf, -np.inf, -np.inf))
        wp.launch(vec3_atomic_minmax_kernel, dim=n, inputs=[points], outputs=[out_min, out_max], device=device)
        mins.append(out_min.numpy().copy())
        maxs.append(out_max.numpy().copy())

    expected_min = np.min(points_np, axis=0, keepdims=True)
    expected_max = np.max(points_np, axis=0, keepdims=True)

    for result in mins:
        np.testing.assert_allclose(result, expected_min, rtol=0.0, atol=0.0)
    for result in maxs:
        np.testing.assert_allclose(result, expected_max, rtol=0.0, atol=0.0)
    for i in range(1, len(mins)):
        np.testing.assert_array_equal(mins[0], mins[i])
        np.testing.assert_array_equal(maxs[0], maxs[i])


def test_mat33_atomic_add_deterministic(test, device):
    """Verify deterministic mode for composite ``wp.mat33`` atomic adds."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 512
    out_size = 8
    rng = np.random.default_rng(69)

    data_np = rng.standard_normal((n, 3, 3), dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.mat33, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(out_size, dtype=wp.mat33, device=device)
        wp.launch(mat33_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
        results.append(output.numpy().copy())

    expected = np.zeros((out_size, 3, 3), dtype=np.float32)
    for i in range(n):
        expected[indices_np[i]] += data_np[i]

    for result in results:
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_triple_scatter_capacity_estimate(test, device):
    """Verify kernels with >2 scatters per thread do not overflow the buffer."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 512
    rng = np.random.default_rng(12)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    results = []
    for _ in range(3):
        output = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(triple_scatter_add_kernel, dim=n, inputs=[data], outputs=[output], device=device)
        results.append(output.numpy().copy())

    expected = np.array([6.0 * data_np.sum()], dtype=np.float32)
    for result in results:
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_loop_scatter_max_records_override(test, device):
    """Verify ``deterministic_max_records`` handles dynamic loop emission counts."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 256
    rng = np.random.default_rng(71)
    data_np = rng.random(n, dtype=np.float32)
    counts_np = np.full(n, 4, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    counts = wp.array(counts_np, dtype=wp.int32, device=device)

    expected = np.array([np.dot(data_np, counts_np).astype(np.float32)], dtype=np.float32)

    results = []
    for _ in range(3):
        output = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(loop_scatter_add_kernel, dim=n, inputs=[data, counts], outputs=[output], device=device)
        results.append(output.numpy().copy())

    for result in results:
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_mixed_reduce_ops_same_array(test, device):
    """Verify add/max atomics targeting one array are reduced independently."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    data_np = np.full(4, 0.05, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    output = wp.zeros(1, dtype=wp.float32, device=device)

    wp.launch(mixed_reduce_op_same_array_kernel, dim=data_np.shape[0], inputs=[data], outputs=[output], device=device)

    np.testing.assert_allclose(output.numpy(), np.array([1.0], dtype=np.float32), rtol=0.0, atol=0.0)


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


def test_counter_phase0_suppresses_array_writes(test, device):
    """Verify non-counter array stores are skipped during the counting pass."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 128
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device)
    scratch = wp.zeros(n, dtype=wp.float32, device=device)

    wp.launch(counter_side_effect_kernel, dim=n, inputs=[counter], outputs=[output, scratch], device=device)

    np.testing.assert_array_equal(scratch.numpy(), np.ones(n, dtype=np.float32))
    test.assertEqual(int(counter.numpy()[0]), n)


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

    # Create a kernel with a per-module deterministic override.
    @wp.kernel(module_options={"deterministic": "gpu_to_gpu"}, module="unique")
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

    # Ensure global config is disabled but per-kernel override still works.
    old_det = wp.config.deterministic
    try:
        wp.config.deterministic = "not_guaranteed"
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


def test_kernel_decorator_override(test, device):
    """Verify ``@wp.kernel(deterministic="gpu_to_gpu")`` works with global config off."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 512
    rng = np.random.default_rng(28)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    old_det = wp.config.deterministic
    try:
        wp.config.deterministic = "not_guaranteed"
        results = []
        for _ in range(3):
            output = wp.zeros(8, dtype=wp.float32, device=device)
            wp.launch(decorator_deterministic_kernel, dim=n, inputs=[data], outputs=[output], device=device)
            results.append(output.numpy().copy())
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
    finally:
        wp.config.deterministic = old_det


def test_deterministic_closure_kernel(test, device):
    """Verify deterministic closure kernels remain reproducible and distinct."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    kernel_a = _make_deterministic_closure_kernel(_det_closure_transform_a)
    kernel_b = _make_deterministic_closure_kernel(_det_closure_transform_b)

    test.assertIsNot(kernel_a, kernel_b)
    test.assertNotEqual(kernel_a.module.name, kernel_b.module.name)

    n = 512
    rng = np.random.default_rng(30)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    results_a = []
    results_b = []
    for _ in range(3):
        out_a = wp.zeros(8, dtype=wp.float32, device=device)
        out_b = wp.zeros(8, dtype=wp.float32, device=device)
        wp.launch(kernel_a, dim=n, inputs=[data], outputs=[out_a], device=device)
        wp.launch(kernel_b, dim=n, inputs=[data], outputs=[out_b], device=device)
        results_a.append(out_a.numpy().copy())
        results_b.append(out_b.numpy().copy())

    for i in range(1, len(results_a)):
        np.testing.assert_array_equal(results_a[0], results_a[i])
        np.testing.assert_array_equal(results_b[0], results_b[i])

    test.assertFalse(np.array_equal(results_a[0], results_b[0]))


def test_record_cmd_deterministic_launch(test, device):
    """Verify ``record_cmd=True`` works for deterministic CUDA launches."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 128
    out_size = 8
    rng = np.random.default_rng(19)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(out_size, dtype=wp.float32, device=device)

    cmd = wp.launch(
        scatter_add_kernel,
        dim=n,
        inputs=[data, indices],
        outputs=[output],
        device=device,
        record_cmd=True,
    )

    np.testing.assert_array_equal(output.numpy(), np.zeros(out_size, dtype=np.float32))

    cmd.launch()
    expected = output.numpy().copy()

    output_2 = wp.zeros(out_size, dtype=wp.float32, device=device)
    cmd.set_param_by_name("output", output_2)
    cmd.launch()

    np.testing.assert_array_equal(output_2.numpy(), expected)


def test_graph_capture_deterministic_launch(test, device):
    """Verify deterministic scatter launches can be captured and replayed."""
    if device.is_cpu:
        test.skipTest("Graph capture requires CUDA")

    n = 256
    rng = np.random.default_rng(29)
    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, 8, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(8, dtype=wp.float32, device=device)

    wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
    output.zero_()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)

    wp.capture_launch(capture.graph)
    first = output.numpy().copy()

    output.zero_()
    wp.capture_launch(capture.graph)
    second = output.numpy().copy()

    np.testing.assert_array_equal(first, second)


def test_graph_capture_deterministic_closure_kernel(test, device):
    """Verify deterministic closure kernels can be captured and replayed."""
    if device.is_cpu:
        test.skipTest("Graph capture requires CUDA")

    kernel = _make_deterministic_closure_kernel(_det_closure_transform_a)

    n = 256
    rng = np.random.default_rng(31)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    output = wp.zeros(8, dtype=wp.float32, device=device)

    wp.launch(kernel, dim=n, inputs=[data], outputs=[output], device=device)
    output.zero_()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(kernel, dim=n, inputs=[data], outputs=[output], device=device)

    wp.capture_launch(capture.graph)
    first = output.numpy().copy()

    output.zero_()
    wp.capture_launch(capture.graph)
    second = output.numpy().copy()

    np.testing.assert_array_equal(first, second)


def test_graph_capture_vec3_atomic_minmax(test, device):
    """Verify composite deterministic reductions remain capture-safe."""
    if device.is_cpu:
        test.skipTest("Graph capture requires CUDA")

    n = 512
    rng = np.random.default_rng(70)
    points_np = rng.standard_normal((n, 3), dtype=np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)

    out_min = wp.empty(1, dtype=wp.vec3, device=device)
    out_max = wp.empty(1, dtype=wp.vec3, device=device)
    min_init = wp.vec3(np.inf, np.inf, np.inf)
    max_init = wp.vec3(-np.inf, -np.inf, -np.inf)

    out_min.fill_(min_init)
    out_max.fill_(max_init)
    wp.launch(vec3_atomic_minmax_kernel, dim=n, inputs=[points], outputs=[out_min, out_max], device=device)
    out_min.fill_(min_init)
    out_max.fill_(max_init)

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(vec3_atomic_minmax_kernel, dim=n, inputs=[points], outputs=[out_min, out_max], device=device)

    wp.capture_launch(capture.graph)
    first_min = out_min.numpy().copy()
    first_max = out_max.numpy().copy()

    out_min.fill_(min_init)
    out_max.fill_(max_init)
    wp.capture_launch(capture.graph)
    second_min = out_min.numpy().copy()
    second_max = out_max.numpy().copy()

    np.testing.assert_array_equal(first_min, second_min)
    np.testing.assert_array_equal(first_max, second_max)
    np.testing.assert_allclose(first_min, np.min(points_np, axis=0, keepdims=True), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(first_max, np.max(points_np, axis=0, keepdims=True), rtol=0.0, atol=0.0)


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
        wp.config.deterministic = "run_to_run"

    @classmethod
    def tearDownClass(cls):
        wp.config.deterministic = cls._old_deterministic


# Pattern A tests (accumulation).
add_function_test(
    TestDeterministic, "test_scatter_add_reproducibility", test_scatter_add_reproducibility, devices=cuda_devices
)
add_function_test(
    TestDeterministic,
    "test_gpu_to_gpu_mode_reproducibility",
    test_gpu_to_gpu_mode_reproducibility,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_gpu_to_gpu_matches_canonical_float32_reference",
    test_gpu_to_gpu_matches_canonical_float32_reference,
    devices=cuda_devices,
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
add_function_test(
    TestDeterministic, "test_vec3_atomic_add_deterministic", test_vec3_atomic_add_deterministic, devices=cuda_devices
)
add_function_test(
    TestDeterministic,
    "test_vec3_atomic_minmax_deterministic",
    test_vec3_atomic_minmax_deterministic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic, "test_mat33_atomic_add_deterministic", test_mat33_atomic_add_deterministic, devices=cuda_devices
)
add_function_test(
    TestDeterministic,
    "test_triple_scatter_capacity_estimate",
    test_triple_scatter_capacity_estimate,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_loop_scatter_max_records_override",
    test_loop_scatter_max_records_override,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic, "test_mixed_reduce_ops_same_array", test_mixed_reduce_ops_same_array, devices=cuda_devices
)

# Pattern B tests (counter).
add_function_test(TestDeterministic, "test_counter_reproducibility", test_counter_reproducibility, devices=cuda_devices)
add_function_test(
    TestDeterministic,
    "test_counter_phase0_suppresses_array_writes",
    test_counter_phase0_suppresses_array_writes,
    devices=cuda_devices,
)
add_function_test(TestDeterministic, "test_counter_correctness", test_counter_correctness, devices=all_devices)
add_function_test(TestDeterministic, "test_conditional_counter", test_conditional_counter, devices=cuda_devices)

# Mixed pattern tests.
add_function_test(TestDeterministic, "test_mixed_pattern", test_mixed_pattern, devices=cuda_devices)

# Passthrough / override tests.
add_function_test(TestDeterministic, "test_int_atomic_passthrough", test_int_atomic_passthrough, devices=all_devices)
add_function_test(TestDeterministic, "test_module_option_override", test_module_option_override, devices=all_devices)
add_function_test(
    TestDeterministic, "test_kernel_decorator_override", test_kernel_decorator_override, devices=cuda_devices
)
add_function_test(
    TestDeterministic, "test_deterministic_closure_kernel", test_deterministic_closure_kernel, devices=cuda_devices
)
add_function_test(
    TestDeterministic,
    "test_record_cmd_deterministic_launch",
    test_record_cmd_deterministic_launch,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_graph_capture_deterministic_launch",
    test_graph_capture_deterministic_launch,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_graph_capture_deterministic_closure_kernel",
    test_graph_capture_deterministic_closure_kernel,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_graph_capture_vec3_atomic_minmax",
    test_graph_capture_vec3_atomic_minmax,
    devices=cuda_devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
