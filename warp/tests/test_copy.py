# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def mul_1d(a: wp.array1d[float], s: float):
    i = wp.tid()
    a[i] = a[i] * s


@wp.kernel
def mul_2d(a: wp.array2d[float], s: float):
    i, j = wp.tid()
    a[i, j] = a[i, j] * s


@wp.kernel
def mul_3d(a: wp.array3d[float], s: float):
    i, j, k = wp.tid()
    a[i, j, k] = a[i, j, k] * s


@wp.kernel
def mul_4d(a: wp.array4d[float], s: float):
    i, j, k, l = wp.tid()
    a[i, j, k, l] = a[i, j, k, l] * s


@wp.kernel
def all_equal_kernel(a: wp.array[float], value: float, result: wp.array[int]):
    tid = wp.tid()
    wp.atomic_min(result, 0, int(a[tid] == value))


def assert_all_equal(a: wp.array[float], value: float):
    result = wp.ones(1, dtype=int, device=a.device)
    wp.launch(all_equal_kernel, dim=a.shape, inputs=[a, value, result], device=a.device)
    assert result.numpy()[0] == 1


def test_copy_strided(test, _, device1, device2):
    np_data1 = np.arange(10, dtype=np.float32)
    np_data2 = np.arange(100, dtype=np.float32).reshape((10, 10))
    np_data3 = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
    np_data4 = np.arange(10000, dtype=np.float32).reshape((10, 10, 10, 10))

    wp_data1 = wp.array(data=np_data1, copy=True, device=device1)
    wp_data2 = wp.array(data=np_data2, copy=True, device=device1)
    wp_data3 = wp.array(data=np_data3, copy=True, device=device1)
    wp_data4 = wp.array(data=np_data4, copy=True, device=device1)

    expected1 = np_data1[1::2]
    expected2 = np_data2[1::2, 1::2]
    expected3 = np_data3[1::2, 1::2, 1::2]
    expected4 = np_data4[1::2, 1::2, 1::2, 1::2]

    a1 = wp_data1[1::2]
    a2 = wp_data2[1::2, 1::2]
    a3 = wp_data3[1::2, 1::2, 1::2]
    a4 = wp_data4[1::2, 1::2, 1::2, 1::2]

    assert_np_equal(a1.numpy(), expected1)
    assert_np_equal(a2.numpy(), expected2)
    assert_np_equal(a3.numpy(), expected3)
    assert_np_equal(a4.numpy(), expected4)

    b1 = wp.zeros_like(a1, device=device2)
    b2 = wp.zeros_like(a2, device=device2)
    b3 = wp.zeros_like(a3, device=device2)
    b4 = wp.zeros_like(a4, device=device2)

    test.assertFalse(a1.is_contiguous)
    test.assertFalse(a2.is_contiguous)
    test.assertFalse(a3.is_contiguous)
    test.assertFalse(a4.is_contiguous)

    test.assertTrue(b1.is_contiguous)
    test.assertTrue(b2.is_contiguous)
    test.assertTrue(b3.is_contiguous)
    test.assertTrue(b4.is_contiguous)

    # copy non-contiguous to contiguous
    wp.synchronize_device(device1)
    wp.copy(b1, a1)
    wp.copy(b2, a2)
    wp.copy(b3, a3)
    wp.copy(b4, a4)

    assert_np_equal(a1.numpy(), b1.numpy())
    assert_np_equal(a2.numpy(), b2.numpy())
    assert_np_equal(a3.numpy(), b3.numpy())
    assert_np_equal(a4.numpy(), b4.numpy())

    s = 2.0

    wp.launch(mul_1d, dim=b1.shape, inputs=[b1, s], device=device2)
    wp.launch(mul_2d, dim=b2.shape, inputs=[b2, s], device=device2)
    wp.launch(mul_3d, dim=b3.shape, inputs=[b3, s], device=device2)
    wp.launch(mul_4d, dim=b4.shape, inputs=[b4, s], device=device2)

    # copy contiguous to non-contiguous
    wp.synchronize_device(device2)
    wp.copy(a1, b1)
    wp.copy(a2, b2)
    wp.copy(a3, b3)
    wp.copy(a4, b4)

    assert_np_equal(a1.numpy(), b1.numpy())
    assert_np_equal(a2.numpy(), b2.numpy())
    assert_np_equal(a3.numpy(), b3.numpy())
    assert_np_equal(a4.numpy(), b4.numpy())

    assert_np_equal(a1.numpy(), expected1 * s)
    assert_np_equal(a2.numpy(), expected2 * s)
    assert_np_equal(a3.numpy(), expected3 * s)
    assert_np_equal(a4.numpy(), expected4 * s)


def test_copy_indexed(test, _, device1, device2):
    np_data1 = np.arange(10, dtype=np.float32)
    np_data2 = np.arange(100, dtype=np.float32).reshape((10, 10))
    np_data3 = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
    np_data4 = np.arange(10000, dtype=np.float32).reshape((10, 10, 10, 10))

    wp_data1 = wp.array(data=np_data1, copy=True, device=device1)
    wp_data2 = wp.array(data=np_data2, copy=True, device=device1)
    wp_data3 = wp.array(data=np_data3, copy=True, device=device1)
    wp_data4 = wp.array(data=np_data4, copy=True, device=device1)

    np_indices = np.array([1, 5, 8, 9])
    wp_indices = wp.array(data=np_indices, dtype=wp.int32, device=device1)

    # Note: Indexing using multiple index arrays works differently
    #       in Numpy and Warp, so the syntax is different.

    expected1 = np_data1[np_indices]
    expected2 = np_data2[np_indices][:, np_indices]
    expected3 = np_data3[np_indices][:, np_indices][:, :, np_indices]
    expected4 = np_data4[np_indices][:, np_indices][:, :, np_indices][:, :, :, np_indices]

    a1 = wp_data1[wp_indices]
    a2 = wp_data2[wp_indices, wp_indices]
    a3 = wp_data3[wp_indices, wp_indices, wp_indices]
    a4 = wp_data4[wp_indices, wp_indices, wp_indices, wp_indices]

    assert_np_equal(a1.numpy(), expected1)
    assert_np_equal(a2.numpy(), expected2)
    assert_np_equal(a3.numpy(), expected3)
    assert_np_equal(a4.numpy(), expected4)

    b1 = wp.zeros_like(a1, device=device2)
    b2 = wp.zeros_like(a2, device=device2)
    b3 = wp.zeros_like(a3, device=device2)
    b4 = wp.zeros_like(a4, device=device2)

    test.assertFalse(a1.is_contiguous)
    test.assertFalse(a2.is_contiguous)
    test.assertFalse(a3.is_contiguous)
    test.assertFalse(a4.is_contiguous)

    test.assertTrue(b1.is_contiguous)
    test.assertTrue(b2.is_contiguous)
    test.assertTrue(b3.is_contiguous)
    test.assertTrue(b4.is_contiguous)

    # copy non-contiguous to contiguous
    wp.synchronize_device(device1)
    wp.copy(b1, a1)
    wp.copy(b2, a2)
    wp.copy(b3, a3)
    wp.copy(b4, a4)

    assert_np_equal(a1.numpy(), b1.numpy())
    assert_np_equal(a2.numpy(), b2.numpy())
    assert_np_equal(a3.numpy(), b3.numpy())
    assert_np_equal(a4.numpy(), b4.numpy())

    s = 2.0

    wp.launch(mul_1d, dim=b1.shape, inputs=[b1, s], device=device2)
    wp.launch(mul_2d, dim=b2.shape, inputs=[b2, s], device=device2)
    wp.launch(mul_3d, dim=b3.shape, inputs=[b3, s], device=device2)
    wp.launch(mul_4d, dim=b4.shape, inputs=[b4, s], device=device2)

    # copy contiguous to non-contiguous
    wp.synchronize_device(device2)
    wp.copy(a1, b1)
    wp.copy(a2, b2)
    wp.copy(a3, b3)
    wp.copy(a4, b4)

    assert_np_equal(a1.numpy(), b1.numpy())
    assert_np_equal(a2.numpy(), b2.numpy())
    assert_np_equal(a3.numpy(), b3.numpy())
    assert_np_equal(a4.numpy(), b4.numpy())

    assert_np_equal(a1.numpy(), expected1 * s)
    assert_np_equal(a2.numpy(), expected2 * s)
    assert_np_equal(a3.numpy(), expected3 * s)
    assert_np_equal(a4.numpy(), expected4 * s)


def test_copy_large_stride(test, _, device1, device2):
    # NOTE: This test can use ~8GB of memory. To prevent errors, we skip if there is insufficient memory.

    gc.collect()

    if device1.is_cpu or device2.is_cpu:
        if importlib.util.find_spec("psutil") is None:
            test.skipTest("The 'psutil' package is required to check available memory")

    if device1.free_memory < 12e9 or device2.free_memory < 12e9:
        test.skipTest("Insufficient free memory")

    N = 500_000_000

    a_data = wp.empty((N, 2), dtype=wp.float32, device=device1)
    a = a_data[:, 0]  # array with a large stride (offsets > 32 bits)

    b = wp.empty_like(a, device=device2)

    a.fill_(1)
    b.fill_(2)

    # NOTE: use a memory-efficient check to avoid running out of memory
    assert_all_equal(a, 1)
    assert_all_equal(b, 2)

    # copy to large-strided array
    wp.synchronize_device(device2)
    wp.copy(a, b)
    assert_all_equal(a, 2)

    a.fill_(1)

    # copy from large-strided array
    wp.synchronize_device(device1)
    wp.copy(b, a)
    assert_all_equal(b, 1)


def test_copy_adjoint(test, device):
    state_in = wp.from_numpy(
        np.array([1.0, 2.0, 3.0]).astype(np.float32), dtype=wp.float32, requires_grad=True, device=device
    )
    state_out = wp.zeros(state_in.shape, dtype=wp.float32, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(state_out, state_in)

    grads = {state_out: wp.from_numpy(np.array([1.0, 1.0, 1.0]).astype(np.float32), dtype=wp.float32, device=device)}
    tape.backward(grads=grads)

    assert_np_equal(state_in.grad.numpy(), np.array([1.0, 1.0, 1.0]).astype(np.float32))


def test_copy_offset_strided(test, _, device1, device2):
    """Honor src_offset/dest_offset/count for non-contiguous arrays.

    The non-contiguous code path is taken when either operand is non-contiguous, so all four
    contiguity combinations must agree.
    """

    def run(src_strided, dst_strided):
        if src_strided:
            # logical src == [1, 2, 3, 4]
            src = wp.array([1, 5, 2, 6, 3, 7, 4, 8], dtype=wp.int32, device=device1)[::2]
        else:
            src = wp.array([1, 2, 3, 4], dtype=wp.int32, device=device1)
        if dst_strided:
            # logical dst == [9, 9, 9, 9]
            dst = wp.array([9, 0, 9, 0, 9, 0, 9, 0], dtype=wp.int32, device=device2)[::2]
        else:
            dst = wp.array([9, 9, 9, 9], dtype=wp.int32, device=device2)

        test.assertEqual(src.is_contiguous, not src_strided)
        test.assertEqual(dst.is_contiguous, not dst_strided)

        wp.synchronize_device(device1)
        wp.copy(dst, src, src_offset=1, dest_offset=1, count=2)
        return dst.numpy().tolist()

    expected = [9, 2, 3, 9]
    for src_strided in (False, True):
        for dst_strided in (False, True):
            result = run(src_strided, dst_strided)
            test.assertEqual(result, expected, f"src_strided={src_strided}, dst_strided={dst_strided}")


def test_copy_offset_partial_ranges(test, _, device1, device2):
    """Check assorted (src_offset, dest_offset, count) combinations on 1-D strided views.

    Each case is verified against the NumPy equivalent.
    """
    np_src = np.arange(1, 21, dtype=np.float32)
    np_dst = np.full(20, -1.0, dtype=np.float32)

    cases = [
        (0, 0, 10),  # full range
        (2, 0, 5),
        (0, 3, 4),
        (1, 4, 3),
        (5, 5, 1),
    ]
    for src_offset, dest_offset, count in cases:
        src = wp.array(np_src, copy=True, device=device1)[::2]  # logical len 10
        dst = wp.array(np_dst, copy=True, device=device2)[::2]
        test.assertFalse(src.is_contiguous)
        test.assertFalse(dst.is_contiguous)

        wp.synchronize_device(device1)
        wp.copy(dst, src, src_offset=src_offset, dest_offset=dest_offset, count=count)

        ref = np_dst[::2].copy()
        ref[dest_offset : dest_offset + count] = np_src[::2][src_offset : src_offset + count]
        assert_np_equal(dst.numpy(), ref, tol=0)


def test_copy_offset_size_mismatch(test, _, device1, device2):
    """Copy a count that spans one array fully but not the other.

    Such a bounded copy must behave the same for contiguous and non-contiguous arrays. In
    particular, count == src.size while dst is larger (and the omitted-count variant) must not
    raise "Incompatible array shapes" for strided views.
    """

    def strided(values):
        # interleave with zeros so [::2] is a non-contiguous view of `values`
        data = np.empty(2 * len(values), dtype=np.float32)
        data[::2] = values
        return data

    # count == src.size (2), dst logical length 4
    src = wp.array(strided([1, 2]), device=device1)[::2]
    dst = wp.array(strided([9, 9, 9, 9]), device=device2)[::2]
    test.assertFalse(src.is_contiguous)
    test.assertFalse(dst.is_contiguous)
    wp.synchronize_device(device1)
    wp.copy(dst, src, count=2)
    assert_np_equal(dst.numpy(), np.array([1, 2, 9, 9], dtype=np.float32), tol=0)

    # omitted count: defaults to src.size (2), still smaller than dst
    src2 = wp.array(strided([7, 8]), device=device1)[::2]
    dst2 = wp.array(strided([9, 9, 9, 9]), device=device2)[::2]
    wp.synchronize_device(device1)
    wp.copy(dst2, src2)
    assert_np_equal(dst2.numpy(), np.array([7, 8, 9, 9], dtype=np.float32), tol=0)

    # reverse direction: src larger than count, dst exactly count
    src3 = wp.array(strided([1, 2, 3, 4]), device=device1)[::2]
    dst3 = wp.array(strided([9, 9]), device=device2)[::2]
    wp.synchronize_device(device1)
    wp.copy(dst3, src3, count=2)
    assert_np_equal(dst3.numpy(), np.array([1, 2], dtype=np.float32), tol=0)


def test_copy_offset_adjoint(test, device):
    """Flow gradients through a sub-range copy on a non-contiguous array.

    Uses symmetric offsets (src_offset == dest_offset).
    """
    src_data = wp.array(np.arange(8, dtype=np.float32), requires_grad=True, device=device)
    src = src_data[::2]  # non-contiguous, logical len 4
    dst = wp.zeros(4, dtype=wp.float32, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(dst, src, src_offset=1, dest_offset=1, count=2)  # dst[1:3] <- src[1:3]

    grads = {dst: wp.ones(4, dtype=wp.float32, device=device)}
    tape.backward(grads=grads)

    expected = np.zeros(4, dtype=np.float32)
    expected[1:3] = 1.0  # only the copied src sub-range receives gradient
    assert_np_equal(src.grad.numpy(), expected, tol=0)


def test_copy_offset_adjoint_asymmetric(test, device):
    """Map adjoint offsets to the correct side for an asymmetric sub-range copy.

    The forward copies dst[dest_offset:...] = src[src_offset:...], so the adjoint must map
    adj_src[src_offset:...] = adj_dst[dest_offset:...]. With distinct src_offset and dest_offset
    this exposes whether the offsets are applied to the correct side. Covers both the contiguous
    and non-contiguous copy paths, since they share adj_copy().
    """
    n = 6
    src_offset, dest_offset, count = 1, 3, 2

    for strided in (False, True):
        if strided:
            src = wp.array(np.arange(2 * n, dtype=np.float32), requires_grad=True, device=device)[::2]
            dst = wp.zeros(2 * n, dtype=wp.float32, requires_grad=True, device=device)[::2]
            test.assertFalse(src.is_contiguous)
            test.assertFalse(dst.is_contiguous)
        else:
            src = wp.array(np.arange(n, dtype=np.float32), requires_grad=True, device=device)
            dst = wp.zeros(n, dtype=wp.float32, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.copy(dst, src, src_offset=src_offset, dest_offset=dest_offset, count=count)

        # Non-uniform seed so the mapping (not just the magnitude) is checked.
        seed = np.arange(10, 10 + dst.shape[0], dtype=np.float32)
        tape.backward(grads={dst: wp.array(seed, device=device)})

        expected = np.zeros(src.shape[0], dtype=np.float32)
        expected[src_offset : src_offset + count] = seed[dest_offset : dest_offset + count]
        assert_np_equal(src.grad.numpy(), expected, tol=0)


def test_copy_offset_unsupported(test, device):
    """Raise a clear error for offsets that can't be expressed as a 1-D strided view.

    Offsets/count on non-contiguous arrays that cannot be reduced to a 1-D strided view must
    raise rather than silently copying everything.
    """

    # multi-dimensional non-contiguous
    src_nd = wp.array(np.arange(100, dtype=np.float32), device=device).reshape((10, 10))[1::2, 1::2]
    dst_nd = wp.zeros_like(src_nd)
    with test.assertRaisesRegex(RuntimeError, "only supported for 1-D"):
        wp.copy(dst_nd, src_nd, count=2)

    # indexed (non-contiguous) array
    base = wp.array([10, 11, 12, 13, 14, 15], dtype=wp.int32, device=device)
    indices = wp.array([0, 2, 4], dtype=wp.int32, device=device)
    src_idx = base[indices]
    dst_idx = wp.zeros(3, dtype=wp.int32, device=device)
    test.assertFalse(src_idx.is_contiguous)
    with test.assertRaisesRegex(RuntimeError, "indexed or Fabric"):
        wp.copy(dst_idx, src_idx, src_offset=1, count=1)

    # out-of-range count
    src_oob = wp.array([1, 2, 3, 4], dtype=wp.int32, device=device)[::2]
    dst_oob = wp.zeros(2, dtype=wp.int32, device=device)
    with test.assertRaisesRegex(RuntimeError, "exceeds the source size"):
        wp.copy(dst_oob, src_oob, count=5)


def test_copy_invalid_args(test, device):
    # wp.copy must reject malformed offsets/count and incompatible element sizes
    # before any pointer arithmetic or device work (contiguous path).
    src = wp.array([1, 2, 3, 4], dtype=wp.int32, device=device)
    dest = wp.zeros_like(src)

    with test.assertRaisesRegex(RuntimeError, "Source offset must be non-negative"):
        wp.copy(dest, src, src_offset=-1)

    with test.assertRaisesRegex(RuntimeError, "Destination offset must be non-negative"):
        wp.copy(dest, src, dest_offset=-1)

    with test.assertRaisesRegex(RuntimeError, "count must be non-negative"):
        wp.copy(dest, src, count=-1)

    # non-integer offsets/count
    with test.assertRaisesRegex(RuntimeError, "must be integers"):
        wp.copy(dest, src, src_offset=1.5)

    with test.assertRaisesRegex(RuntimeError, "must be integers"):
        wp.copy(dest, src, count=2.0)

    # mismatched element sizes on the contiguous path (float32 vs float64)
    src_f32 = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32, device=device)
    dest_f64 = wp.zeros(4, dtype=wp.float64, device=device)
    test.assertTrue(src_f32.is_contiguous)
    test.assertTrue(dest_f64.is_contiguous)
    with test.assertRaisesRegex(RuntimeError, "destination dtype=float64, source dtype=float32"):
        wp.copy(dest_f64, src_f32)

    # mismatched shapes on the non-contiguous path should report both shapes
    src_shape = wp.array(np.arange(18, dtype=np.float32).reshape((6, 3)), device=device)[::3, :]
    dest_shape = wp.zeros((6, 2), dtype=wp.float32, device=device)[::2, :]
    test.assertFalse(src_shape.is_contiguous)
    test.assertFalse(dest_shape.is_contiguous)
    with test.assertRaisesRegex(RuntimeError, r"destination shape=\(3, 2\), source shape=\(2, 3\)"):
        wp.copy(dest_shape, src_shape)


def test_copy_count_zero_copies_all(test, device):
    # count == 0 keeps its documented "copy all" meaning now that negative counts
    # are rejected (the historical `count <= 0` default narrowed to `count == 0`).
    expected = np.array([1, 2, 3, 4], dtype=np.int32)

    src = wp.array(expected, dtype=wp.int32, device=device)

    dest = wp.zeros_like(src)
    wp.copy(dest, src)  # omitted count
    assert_np_equal(dest.numpy(), expected, tol=0)

    dest2 = wp.zeros_like(src)
    wp.copy(dest2, src, count=0)  # explicit zero
    assert_np_equal(dest2.numpy(), expected, tol=0)


devices = get_test_devices()


class TestCopy(unittest.TestCase):
    pass


for src_device in devices:
    src_name = "cpu" if src_device.is_cpu else f"cuda{src_device.ordinal}"
    for dst_device in devices:
        dst_name = "cpu" if dst_device.is_cpu else f"cuda{dst_device.ordinal}"
        add_function_test(
            TestCopy,
            f"test_copy_strided_{src_name}_{dst_name}",
            test_copy_strided,
            devices=None,
            device1=src_device,
            device2=dst_device,
        )
        add_function_test(
            TestCopy,
            f"test_copy_indexed_{src_name}_{dst_name}",
            test_copy_indexed,
            devices=None,
            device1=src_device,
            device2=dst_device,
        )
        add_function_test(
            TestCopy,
            f"test_copy_large_stride_{src_name}_{dst_name}",
            test_copy_large_stride,
            devices=None,
            device1=src_device,
            device2=dst_device,
        )
        add_function_test(
            TestCopy,
            f"test_copy_offset_strided_{src_name}_{dst_name}",
            test_copy_offset_strided,
            devices=None,
            device1=src_device,
            device2=dst_device,
        )
        add_function_test(
            TestCopy,
            f"test_copy_offset_partial_ranges_{src_name}_{dst_name}",
            test_copy_offset_partial_ranges,
            devices=None,
            device1=src_device,
            device2=dst_device,
        )
        add_function_test(
            TestCopy,
            f"test_copy_offset_size_mismatch_{src_name}_{dst_name}",
            test_copy_offset_size_mismatch,
            devices=None,
            device1=src_device,
            device2=dst_device,
        )

add_function_test(TestCopy, "test_copy_adjoint", test_copy_adjoint, devices=devices)
add_function_test(TestCopy, "test_copy_offset_adjoint", test_copy_offset_adjoint, devices=devices)
add_function_test(TestCopy, "test_copy_offset_adjoint_asymmetric", test_copy_offset_adjoint_asymmetric, devices=devices)
add_function_test(TestCopy, "test_copy_offset_unsupported", test_copy_offset_unsupported, devices=devices)
add_function_test(TestCopy, "test_copy_invalid_args", test_copy_invalid_args, devices=devices)
add_function_test(TestCopy, "test_copy_count_zero_copies_all", test_copy_count_zero_copies_all, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
