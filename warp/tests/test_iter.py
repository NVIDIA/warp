# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def reversed_kernel(
    start: wp.int32,
    end: wp.int32,
    step: wp.int32,
    out_count: wp.array[wp.int32],
    out_values: wp.array[wp.int32],
):
    count = wp.int32(0)
    for i in reversed(range(start, end, step)):
        out_values[count] = i
        count += 1

    out_count[0] = count


def test_reversed(test, device):
    count = wp.empty(1, dtype=wp.int32)
    values = wp.empty(32, dtype=wp.int32)

    start, end, step = (-2, 8, 3)
    wp.launch(
        reversed_kernel,
        dim=1,
        inputs=(start, end, step),
        outputs=(count, values),
    )
    expected = tuple(reversed(range(start, end, step)))
    assert count.numpy()[0] == len(expected)
    assert_np_equal(values.numpy()[: len(expected)], expected)

    start, end, step = (9, -3, -2)
    wp.launch(
        reversed_kernel,
        dim=1,
        inputs=(start, end, step),
        outputs=(count, values),
    )
    expected = tuple(reversed(range(start, end, step)))
    assert count.numpy()[0] == len(expected)
    assert_np_equal(values.numpy()[: len(expected)], expected)


@wp.kernel
def reversed_sweep_kernel(
    starts: wp.array[wp.int32],
    ends: wp.array[wp.int32],
    steps: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_values: wp.array2d[wp.int32],
):
    tid = wp.tid()
    count = wp.int32(0)
    for i in reversed(range(starts[tid], ends[tid], steps[tid])):
        if count < out_values.shape[1]:
            out_values[tid, count] = i
        count += 1

    out_counts[tid] = count


def test_reversed_sweep(test, device):
    # includes empty ranges such as range(10, 8, 4), which must not yield any iteration
    cases = [(s, e, k) for s in range(-10, 11) for e in range(-10, 11) for k in range(-5, 6) if k != 0]
    starts, ends, steps = (wp.array([c[j] for c in cases], dtype=wp.int32, device=device) for j in range(3))
    counts = wp.zeros(len(cases), dtype=wp.int32, device=device)
    values = wp.zeros((len(cases), 32), dtype=wp.int32, device=device)
    wp.launch(
        reversed_sweep_kernel,
        dim=len(cases),
        inputs=(starts, ends, steps),
        outputs=(counts, values),
        device=device,
    )

    counts_np = counts.numpy()
    values_np = values.numpy()
    for tid, (start, end, step) in enumerate(cases):
        expected = list(reversed(range(start, end, step)))
        actual = values_np[tid, : counts_np[tid]].tolist()
        test.assertEqual(actual, expected, msg=f"reversed(range({start}, {end}, {step}))")


devices = get_test_devices()


class TestIter(unittest.TestCase):
    pass


add_function_test(TestIter, "test_reversed", test_reversed, devices=devices)
add_function_test(TestIter, "test_reversed_sweep", test_reversed_sweep, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
