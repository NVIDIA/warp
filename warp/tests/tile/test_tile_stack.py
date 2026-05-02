# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_DIM = 64
CAPACITY = wp.constant(64)
SMALL_CAP = wp.constant(4)
ZERO_CAP = wp.constant(0)


# ----------------------------------------------------------------
# 1. push_all
# ----------------------------------------------------------------
@wp.kernel
def push_all_kernel(out: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    wp.tile_stack_push(s, j * 10, True)
    val, slot = wp.tile_stack_pop(s)
    if slot != -1:
        out[slot] = val


def test_push_all(test, device):
    n = TILE_DIM
    out = wp.full(n, -1, dtype=int, device=device)
    wp.launch_tiled(push_all_kernel, dim=[1], inputs=[out], block_dim=TILE_DIM, device=device)

    vals = out.numpy()
    if device == "cpu":
        written = vals[vals >= 0]
        expected = [0]
    else:
        written = vals[vals >= 0]
        expected = list(range(0, n * 10, 10))
    test.assertEqual(sorted(written.tolist()), expected)


# ----------------------------------------------------------------
# 2. push_partial
# ----------------------------------------------------------------
@wp.kernel
def push_partial_kernel(out_count: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    wp.tile_stack_push(s, j, j % 2 == 0)
    count = wp.tile_stack_count(s)
    out_count[j] = count


def test_push_partial(test, device):
    out_count = wp.zeros(TILE_DIM, dtype=int, device=device)
    wp.launch_tiled(push_partial_kernel, dim=[1], inputs=[out_count], block_dim=TILE_DIM, device=device)

    counts = out_count.numpy()
    if device == "cpu":
        # CPU: block_dim=1, only thread 0 runs, 0%2==0 so it pushes => count=1
        test.assertEqual(counts[0], 1)
    else:
        # All threads see the same count
        expected = TILE_DIM // 2
        test.assertEqual(counts[0], expected)


# ----------------------------------------------------------------
# 3. overflow
# ----------------------------------------------------------------
@wp.kernel
def overflow_kernel(out_idx: wp.array(dtype=int), out_count: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=SMALL_CAP, dtype=int)
    idx = wp.tile_stack_push(s, j * 10, True)
    out_idx[j] = idx
    count = wp.tile_stack_count(s)
    out_count[j] = count


def test_overflow(test, device):
    out_idx = wp.full(TILE_DIM, -2, dtype=int, device=device)
    out_count = wp.zeros(TILE_DIM, dtype=int, device=device)
    wp.launch_tiled(overflow_kernel, dim=[1], inputs=[out_idx, out_count], block_dim=TILE_DIM, device=device)

    idxs = out_idx.numpy()
    counts = out_count.numpy()
    if device == "cpu":
        # CPU: 1 thread, capacity 4 => 1 push succeeds
        test.assertEqual(idxs[0], 0)
        test.assertEqual(counts[0], 1)
    else:
        valid_count = np.sum(idxs >= 0)
        test.assertEqual(valid_count, 4)  # SMALL_CAP
        test.assertEqual(counts[0], 4)  # count clamped to capacity


# ----------------------------------------------------------------
# 4. pop_empty
# ----------------------------------------------------------------
@wp.kernel
def pop_empty_kernel(out_ok: wp.array(dtype=int), out_val: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    val, slot = wp.tile_stack_pop(s)
    out_ok[j] = wp.int32(slot >= 0)
    out_val[j] = val


def test_pop_empty(test, device):
    out_ok = wp.zeros(TILE_DIM, dtype=int, device=device)
    out_val = wp.zeros(TILE_DIM, dtype=int, device=device)
    wp.launch_tiled(pop_empty_kernel, dim=[1], inputs=[out_ok, out_val], block_dim=TILE_DIM, device=device)

    oks = out_ok.numpy()
    vals = out_val.numpy()
    if device == "cpu":
        test.assertEqual(oks[0], 0)
        test.assertEqual(vals[0], 0)
    else:
        assert_np_equal(oks, np.zeros(TILE_DIM, dtype=np.int32))
        assert_np_equal(vals, np.zeros(TILE_DIM, dtype=np.int32))


# ----------------------------------------------------------------
# 5. push_pop_clear_cycle
# ----------------------------------------------------------------
@wp.kernel
def push_pop_clear_cycle_kernel(out_count: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    # First push
    wp.tile_stack_push(s, j, True)
    # Pop all
    _val, _slot = wp.tile_stack_pop(s)
    # Clear
    wp.tile_stack_clear(s)
    # Second push
    wp.tile_stack_push(s, j * 2, True)
    count = wp.tile_stack_count(s)
    out_count[j] = count


def test_push_pop_clear_cycle(test, device):
    out_count = wp.zeros(TILE_DIM, dtype=int, device=device)
    wp.launch_tiled(push_pop_clear_cycle_kernel, dim=[1], inputs=[out_count], block_dim=TILE_DIM, device=device)

    counts = out_count.numpy()
    if device == "cpu":
        test.assertEqual(counts[0], 1)
    else:
        test.assertEqual(counts[0], TILE_DIM)


# ----------------------------------------------------------------
# 6. pop_more_than_pushed
# ----------------------------------------------------------------
@wp.kernel
def pop_more_than_pushed_kernel(
    out_ok1: wp.array(dtype=int),
    out_ok2: wp.array(dtype=int),
):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    wp.tile_stack_push(s, j, True)
    # First pop: should succeed
    _val1, slot1 = wp.tile_stack_pop(s)
    out_ok1[j] = wp.int32(slot1 >= 0)
    # Second pop: should fail
    _val2, slot2 = wp.tile_stack_pop(s)
    out_ok2[j] = wp.int32(slot2 >= 0)


def test_pop_more_than_pushed(test, device):
    out_ok1 = wp.zeros(TILE_DIM, dtype=int, device=device)
    out_ok2 = wp.zeros(TILE_DIM, dtype=int, device=device)
    wp.launch_tiled(
        pop_more_than_pushed_kernel,
        dim=[1],
        inputs=[out_ok1, out_ok2],
        block_dim=TILE_DIM,
        device=device,
    )

    oks1 = out_ok1.numpy()
    oks2 = out_ok2.numpy()
    if device == "cpu":
        test.assertEqual(oks1[0], 1)
        test.assertEqual(oks2[0], 0)
    else:
        assert_np_equal(oks1, np.ones(TILE_DIM, dtype=np.int32))
        assert_np_equal(oks2, np.zeros(TILE_DIM, dtype=np.int32))


# ----------------------------------------------------------------
# 7. multi_tile
# ----------------------------------------------------------------
@wp.kernel
def multi_tile_kernel(out_counts: wp.array(dtype=int)):
    i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    wp.tile_stack_push(s, j, j <= i)
    count = wp.tile_stack_count(s)
    # Only one thread per tile writes the count (no cooperative op in this branch)
    if j == 0:
        out_counts[i] = count


def test_multi_tile(test, device):
    num_tiles = 4
    out_counts = wp.zeros(num_tiles, dtype=int, device=device)
    wp.launch_tiled(
        multi_tile_kernel,
        dim=[num_tiles],
        inputs=[out_counts],
        block_dim=TILE_DIM,
        device=device,
    )

    counts = out_counts.numpy()
    if device == "cpu":
        # CPU: block_dim=1, each tile has j=0, j<=i always true => each pushes 1
        for t in range(num_tiles):
            test.assertEqual(counts[t], 1)
    else:
        for t in range(num_tiles):
            test.assertEqual(counts[t], t + 1)


# ----------------------------------------------------------------
# 8. has_value_false_returns_minus_one
# ----------------------------------------------------------------
@wp.kernel
def has_value_false_kernel(out_idx: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    idx = wp.tile_stack_push(s, j, False)
    out_idx[j] = idx


def test_has_value_false_returns_minus_one(test, device):
    out_idx = wp.zeros(TILE_DIM, dtype=int, device=device)
    wp.launch_tiled(
        has_value_false_kernel,
        dim=[1],
        inputs=[out_idx],
        block_dim=TILE_DIM,
        device=device,
    )

    idxs = out_idx.numpy()
    if device == "cpu":
        test.assertEqual(idxs[0], -1)
    else:
        assert_np_equal(idxs, np.full(TILE_DIM, -1, dtype=np.int32))


# ----------------------------------------------------------------
# 9. clear_resets_count
# ----------------------------------------------------------------
@wp.kernel
def clear_resets_count_kernel(out_count: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    wp.tile_stack_push(s, j, True)
    wp.tile_stack_clear(s)
    count = wp.tile_stack_count(s)
    out_count[j] = count


def test_clear_resets_count(test, device):
    out_count = wp.full(TILE_DIM, -1, dtype=int, device=device)
    wp.launch_tiled(clear_resets_count_kernel, dim=[1], inputs=[out_count], block_dim=TILE_DIM, device=device)

    counts = out_count.numpy()
    if device == "cpu":
        test.assertEqual(counts[0], 0)
    else:
        assert_np_equal(counts, np.zeros(TILE_DIM, dtype=np.int32))


# ----------------------------------------------------------------
# 10. float_dtype
# ----------------------------------------------------------------
@wp.kernel
def float_dtype_kernel(out: wp.array(dtype=float)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=float)
    wp.tile_stack_push(s, float(j) * 1.5, True)
    val, slot = wp.tile_stack_pop(s)
    if slot != -1:
        out[slot] = val


def test_float_dtype(test, device):
    n = TILE_DIM
    out = wp.full(n, -1.0, dtype=float, device=device)
    wp.launch_tiled(float_dtype_kernel, dim=[1], inputs=[out], block_dim=TILE_DIM, device=device)

    vals = out.numpy()
    if device == "cpu":
        expected = [0.0]
        written = vals[vals >= 0.0]
    else:
        expected = sorted([float(j) * 1.5 for j in range(n)])
        written = vals[vals >= 0.0]
    assert_np_equal(np.array(sorted(written.tolist())), np.array(expected), tol=1e-5)


# ----------------------------------------------------------------
# 11. overflow_data_integrity
# ----------------------------------------------------------------
@wp.kernel
def overflow_data_integrity_kernel(out_vals: wp.array(dtype=int), out_ok: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=SMALL_CAP, dtype=int)
    wp.tile_stack_push(s, j * 10, True)
    val, slot = wp.tile_stack_pop(s)
    out_ok[j] = wp.int32(slot >= 0)
    if slot != -1:
        out_vals[j] = val


def test_overflow_data_integrity(test, device):
    out_vals = wp.full(TILE_DIM, -1, dtype=int, device=device)
    out_ok = wp.zeros(TILE_DIM, dtype=int, device=device)
    wp.launch_tiled(
        overflow_data_integrity_kernel,
        dim=[1],
        inputs=[out_vals, out_ok],
        block_dim=TILE_DIM,
        device=device,
    )

    vals = out_vals.numpy()
    oks = out_ok.numpy()
    if device == "cpu":
        # 1 thread, capacity 4, push succeeds, pop succeeds
        test.assertEqual(oks[0], 1)
        test.assertEqual(vals[0], 0)  # j=0, 0*10=0
    else:
        # Exactly SMALL_CAP threads get ok=True on pop
        num_ok = np.sum(oks)
        test.assertEqual(num_ok, 4)
        # All popped values must be valid pushed values (multiples of 10)
        popped = vals[oks == 1]
        all_pushed = set(range(0, TILE_DIM * 10, 10))
        for v in popped:
            test.assertIn(v, all_pushed)


# ----------------------------------------------------------------
# 12. vec3_dtype
# ----------------------------------------------------------------
@wp.kernel
def vec3_dtype_kernel(out: wp.array(dtype=wp.vec3)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=wp.vec3)
    v = wp.vec3(float(j), float(j) * 2.0, float(j) * 3.0)
    wp.tile_stack_push(s, v, True)
    val, slot = wp.tile_stack_pop(s)
    if slot != -1:
        out[slot] = val


def test_vec3_dtype(test, device):
    n = TILE_DIM
    out = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch_tiled(vec3_dtype_kernel, dim=[1], inputs=[out], block_dim=TILE_DIM, device=device)

    vals = out.numpy()
    if device == "cpu":
        # Only thread 0: vec3(0, 0, 0)
        assert_np_equal(vals[0], np.array([0.0, 0.0, 0.0]))
    else:
        # Build expected set
        expected = sorted([(float(j), float(j) * 2.0, float(j) * 3.0) for j in range(n)])
        actual = sorted([tuple(v) for v in vals.tolist()])
        for e, a in zip(expected, actual, strict=True):
            assert_np_equal(np.array(a), np.array(e), tol=1e-5)


# ----------------------------------------------------------------
# 13. float16_dtype
# ----------------------------------------------------------------
@wp.kernel
def float16_dtype_kernel(out: wp.array(dtype=wp.float16), out_ok: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=wp.float16)
    wp.tile_stack_push(s, wp.float16(float(j)), True)
    val, slot = wp.tile_stack_pop(s)
    out_ok[j] = wp.int32(slot >= 0)
    if slot != -1:
        out[slot] = val


def test_float16_dtype(test, device):
    n = TILE_DIM
    out = wp.zeros(n, dtype=wp.float16, device=device)
    out_ok = wp.zeros(n, dtype=int, device=device)
    wp.launch_tiled(float16_dtype_kernel, dim=[1], inputs=[out, out_ok], block_dim=TILE_DIM, device=device)

    vals = out.numpy()
    oks = out_ok.numpy()
    if device == "cpu":
        test.assertEqual(oks[0], 1)
        assert_np_equal(np.array([float(vals[0])]), np.array([0.0]))
    else:
        expected = sorted([float(j) for j in range(n)])
        actual = sorted(vals[oks == 1].astype(float).tolist())
        assert_np_equal(np.array(actual), np.array(expected), tol=1e-2)


# ----------------------------------------------------------------
# 14. count_after_push
# ----------------------------------------------------------------
@wp.kernel
def count_after_push_kernel(out_count: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    wp.tile_stack_push(s, j, j < 10)
    count = wp.tile_stack_count(s)
    out_count[j] = count


def test_count_after_push(test, device):
    out_count = wp.zeros(TILE_DIM, dtype=int, device=device)
    wp.launch_tiled(count_after_push_kernel, dim=[1], inputs=[out_count], block_dim=TILE_DIM, device=device)

    counts = out_count.numpy()
    if device == "cpu":
        # CPU: 1 thread, j=0, 0<10 => pushes => count=1
        test.assertEqual(counts[0], 1)
    else:
        expected = 10
        test.assertEqual(counts[0], expected)
        # All threads should see the same count (cooperative semantics, R9)
        assert_np_equal(counts, np.full(TILE_DIM, expected))


# ----------------------------------------------------------------
# 15. two_stacks (LIFO deallocation of multiple stacks)
# ----------------------------------------------------------------
@wp.kernel
def two_stacks_kernel(out_ints: wp.array(dtype=int), out_floats: wp.array(dtype=float)):
    _i, j = wp.tid()
    s1 = wp.tile_stack(capacity=CAPACITY, dtype=int)
    s2 = wp.tile_stack(capacity=CAPACITY, dtype=float)
    wp.tile_stack_push(s1, j * 10, True)
    wp.tile_stack_push(s2, float(j) * 1.5, True)
    val1, slot1 = wp.tile_stack_pop(s1)
    val2, slot2 = wp.tile_stack_pop(s2)
    if slot1 != -1:
        out_ints[slot1] = val1
    if slot2 != -1:
        out_floats[slot2] = val2


def test_two_stacks(test, device):
    n = TILE_DIM
    out_ints = wp.full(n, -1, dtype=int, device=device)
    out_floats = wp.full(n, -1.0, dtype=float, device=device)
    wp.launch_tiled(two_stacks_kernel, dim=[1], inputs=[out_ints, out_floats], block_dim=TILE_DIM, device=device)

    ints = out_ints.numpy()
    floats = out_floats.numpy()
    if device == "cpu":
        test.assertEqual(ints[0], 0)
        assert_np_equal(np.array([floats[0]]), np.array([0.0]))
    else:
        # Both stacks should independently preserve their value sets
        int_written = ints[ints >= 0]
        test.assertEqual(len(int_written), n)
        test.assertEqual(sorted(int_written.tolist()), list(range(0, n * 10, 10)))

        float_written = floats[floats >= 0.0]
        test.assertEqual(len(float_written), n)
        expected_floats = sorted([j * 1.5 for j in range(n)])
        assert_np_equal(np.array(sorted(float_written.tolist())), np.array(expected_floats), tol=1e-5)


# ----------------------------------------------------------------
# 16. pop_slot_compact
# Push half the threads, pop all — verifies that successful pop slots
# form exactly the compact range [0, k-1] with no duplicates, and that
# failed pops return slot == -1.  This test cannot pass accidentally
# with the push index substituted for the pop slot.
# ----------------------------------------------------------------
HALF_DIM = wp.constant(TILE_DIM // 2)


@wp.kernel
def pop_slot_compact_kernel(out_slot: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    # Only the first half of threads push
    wp.tile_stack_push(s, j, j < HALF_DIM)
    _val, slot = wp.tile_stack_pop(s)
    out_slot[j] = slot


def test_pop_slot_compact(test, device):
    out_slot = wp.full(TILE_DIM, -2, dtype=int, device=device)
    wp.launch_tiled(pop_slot_compact_kernel, dim=[1], inputs=[out_slot], block_dim=TILE_DIM, device=device)

    slots = out_slot.numpy()
    if device == "cpu":
        # 1 thread: j=0 < HALF_DIM, pushes once, pops once successfully
        test.assertEqual(slots[0], 0)
    else:
        k = TILE_DIM // 2
        successful = slots[slots >= 0]
        failed = slots[slots < 0]
        # Exactly k pops succeed
        test.assertEqual(len(successful), k)
        # All failed slots are -1 (not some other negative value)
        assert_np_equal(failed, np.full(len(failed), -1, dtype=np.int32))
        # Successful slots form exactly the compact set {0, ..., k-1}
        test.assertEqual(sorted(successful.tolist()), list(range(k)))


# ----------------------------------------------------------------
# 14. pass tile_stack to @wp.func by reference
# ----------------------------------------------------------------
@wp.func
def helper_push(s: wp.tile_stack(capacity=CAPACITY, dtype=int), val: int):
    wp.tile_stack_push(s, val, True)


@wp.kernel
def func_tile_stack_kernel(out: wp.array(dtype=int)):
    _i, j = wp.tid()
    s = wp.tile_stack(capacity=CAPACITY, dtype=int)
    helper_push(s, j * 10)
    val, slot = wp.tile_stack_pop(s)
    if slot != -1:
        out[slot] = val


def test_func_tile_stack_arg(test, device):
    n = TILE_DIM
    out = wp.full(n, -1, dtype=int, device=device)
    wp.launch_tiled(func_tile_stack_kernel, dim=[1], inputs=[out], block_dim=n, device=device)
    vals = out.numpy()
    written = vals[vals >= 0]
    if device == "cpu":
        expected = [0]
    else:
        expected = list(range(0, n * 10, 10))
    test.assertEqual(sorted(written.tolist()), expected)


# ----------------------------------------------------------------
# Test class and registration
# ----------------------------------------------------------------
devices = get_test_devices()


class TestTileStack(unittest.TestCase):
    pass


add_function_test(TestTileStack, "test_push_all", test_push_all, devices=devices)
add_function_test(TestTileStack, "test_push_partial", test_push_partial, devices=devices)
add_function_test(TestTileStack, "test_overflow", test_overflow, devices=devices)
add_function_test(TestTileStack, "test_pop_empty", test_pop_empty, devices=devices)
add_function_test(TestTileStack, "test_push_pop_clear_cycle", test_push_pop_clear_cycle, devices=devices)
add_function_test(TestTileStack, "test_pop_more_than_pushed", test_pop_more_than_pushed, devices=devices)
add_function_test(TestTileStack, "test_multi_tile", test_multi_tile, devices=devices)
add_function_test(
    TestTileStack, "test_has_value_false_returns_minus_one", test_has_value_false_returns_minus_one, devices=devices
)
add_function_test(TestTileStack, "test_clear_resets_count", test_clear_resets_count, devices=devices)
add_function_test(TestTileStack, "test_float_dtype", test_float_dtype, devices=devices)
add_function_test(TestTileStack, "test_overflow_data_integrity", test_overflow_data_integrity, devices=devices)
add_function_test(TestTileStack, "test_vec3_dtype", test_vec3_dtype, devices=devices)
add_function_test(TestTileStack, "test_float16_dtype", test_float16_dtype, devices=devices)
add_function_test(TestTileStack, "test_count_after_push", test_count_after_push, devices=devices)
add_function_test(TestTileStack, "test_two_stacks", test_two_stacks, devices=devices)
add_function_test(TestTileStack, "test_pop_slot_compact", test_pop_slot_compact, devices=devices)
add_function_test(TestTileStack, "test_func_tile_stack_arg", test_func_tile_stack_arg, devices=devices)


def test_bool_capacity_rejected(test, device):
    def kernel_fn():
        s = wp.tile_stack(capacity=True, dtype=int)

    kernel = wp.Kernel(func=kernel_fn)
    with test.assertRaisesRegex(RuntimeError, r"tile_stack|overload"):
        wp.launch_tiled(kernel, dim=[1], inputs=[], block_dim=32, device=device)


def test_zero_capacity_rejected(test, device):
    def kernel_fn():
        s = wp.tile_stack(capacity=ZERO_CAP, dtype=int)

    kernel = wp.Kernel(func=kernel_fn)
    with test.assertRaisesRegex((RuntimeError, ValueError), "positive integer"):
        wp.launch_tiled(kernel, dim=[1], inputs=[], block_dim=32, device=device)


def test_push_type_mismatch_rejected(test, device):
    def kernel_fn():
        _i, _j = wp.tid()
        s = wp.tile_stack(capacity=SMALL_CAP, dtype=int)
        wp.tile_stack_push(s, 1.0, True)

    kernel = wp.Kernel(func=kernel_fn)
    with test.assertRaisesRegex((RuntimeError, TypeError), "does not match"):
        wp.launch_tiled(kernel, dim=[1], inputs=[], block_dim=32, device=device)


add_function_test(TestTileStack, "test_bool_capacity_rejected", test_bool_capacity_rejected, devices=devices)
add_function_test(TestTileStack, "test_zero_capacity_rejected", test_zero_capacity_rejected, devices=devices)
add_function_test(TestTileStack, "test_push_type_mismatch_rejected", test_push_type_mismatch_rejected, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
