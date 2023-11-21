from functools import partial
import unittest

import numpy as np
import warp as wp

from warp.utils import runlength_encode
from warp.tests.test_base import *

wp.init()


def test_runlength_encode_int(test, device, n):
    rng = np.random.default_rng(123)

    values_np = np.sort(rng.integers(-10, high=10, size=n, dtype=int))

    unique_values_np, unique_counts_np = np.unique(values_np, return_counts=True)

    values = wp.array(values_np, device=device, dtype=int)

    unique_values = wp.empty_like(values)
    unique_counts = wp.empty_like(values)

    run_count = runlength_encode(values, unique_values, unique_counts)

    test.assertEqual(run_count, len(unique_values_np))
    assert_np_equal(unique_values.numpy()[:run_count], unique_values_np[:run_count])
    assert_np_equal(unique_counts.numpy()[:run_count], unique_counts_np[:run_count])


def test_runlength_encode_error_devices_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device="cpu")
    run_values = wp.empty_like(values, device="cuda:0")
    run_lengths = wp.empty_like(values, device="cuda:0")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage devices do not match$",
    ):
        runlength_encode(values, run_values, run_lengths)

    values = wp.zeros(123, dtype=int, device="cpu")
    run_values = wp.empty_like(values, device="cpu")
    run_lengths = wp.empty_like(values, device="cuda:0")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage devices do not match$",
    ):
        runlength_encode(values, run_values, run_lengths)

    values = wp.zeros(123, dtype=int, device="cpu")
    run_values = wp.empty_like(values, device="cuda:0")
    run_lengths = wp.empty_like(values, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Array storage devices do not match$",
    ):
        runlength_encode(values, run_values, run_lengths)


def test_runlength_encode_error_insufficient_storage(test, device):
    values = wp.zeros(123, dtype=int, device="cpu")
    run_values = wp.empty(1, dtype=int, device="cpu")
    run_lengths = wp.empty(123, dtype=int, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Output array storage sizes must be at least equal to value_count$",
    ):
        runlength_encode(values, run_values, run_lengths)

    values = wp.zeros(123, dtype=int, device="cpu")
    run_values = wp.empty(123, dtype=int, device="cpu")
    run_lengths = wp.empty(1, dtype=int, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"Output array storage sizes must be at least equal to value_count$",
    ):
        runlength_encode(values, run_values, run_lengths)


def test_runlength_encode_error_dtypes_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device="cpu")
    run_values = wp.empty(123, dtype=float, device="cpu")
    run_lengths = wp.empty_like(values)
    with test.assertRaisesRegex(
        RuntimeError,
        r"values and run_values data types do not match$",
    ):
        runlength_encode(values, run_values, run_lengths)


def test_runlength_encode_error_run_length_unsupported_dtype(test, device):
    values = wp.zeros(123, dtype=int, device="cpu")
    run_values = wp.empty(123, dtype=int, device="cpu")
    run_lengths = wp.empty(123, dtype=float, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"run_lengths array must be of type int32$",
    ):
        runlength_encode(values, run_values, run_lengths)


def test_runlength_encode_error_run_count_device_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device="cpu")
    run_values = wp.empty_like(values, device="cpu")
    run_lengths = wp.empty_like(values, device="cpu")
    run_count = wp.empty(shape=(1,), dtype=int, device="cuda:0")
    with test.assertRaisesRegex(
        RuntimeError,
        r"run_count storage device does not match other arrays$",
    ):
        runlength_encode(values, run_values, run_lengths, run_count=run_count)


def test_runlength_encode_error_run_count_unsupported_dtype(test, device):
    values = wp.zeros(123, dtype=int, device="cpu")
    run_values = wp.empty_like(values, device="cpu")
    run_lengths = wp.empty_like(values, device="cpu")
    run_count = wp.empty(shape=(1,), dtype=float, device="cpu")
    with test.assertRaisesRegex(
        RuntimeError,
        r"run_count array must be of type int32$",
    ):
        runlength_encode(values, run_values, run_lengths, run_count=run_count)


def test_runlength_encode_error_unsupported_dtype(test, device):
    values = wp.zeros(123, dtype=float, device=device)
    run_values = wp.empty(123, dtype=float, device=device)
    run_lengths = wp.empty(123, dtype=int, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type$",
    ):
        runlength_encode(values, run_values, run_lengths)


def register(parent):
    devices = get_test_devices()

    class TestRunlengthEncode(parent):
        pass

    add_function_test(
        TestRunlengthEncode, "test_runlength_encode_int", partial(test_runlength_encode_int, n=100), devices=devices
    )
    add_function_test(
        TestRunlengthEncode, "test_runlength_encode_empty", partial(test_runlength_encode_int, n=0), devices=devices
    )
    add_function_test(
        TestRunlengthEncode,
        "test_runlength_encode_error_devices_mismatch",
        test_runlength_encode_error_devices_mismatch,
    )
    add_function_test(
        TestRunlengthEncode,
        "test_runlength_encode_error_insufficient_storage",
        test_runlength_encode_error_insufficient_storage,
    )
    add_function_test(
        TestRunlengthEncode, "test_runlength_encode_error_dtypes_mismatch", test_runlength_encode_error_dtypes_mismatch
    )
    add_function_test(
        TestRunlengthEncode,
        "test_runlength_encode_error_run_length_unsupported_dtype",
        test_runlength_encode_error_run_length_unsupported_dtype,
    )
    add_function_test(
        TestRunlengthEncode,
        "test_runlength_encode_error_run_count_device_mismatch",
        test_runlength_encode_error_run_count_device_mismatch,
    )
    add_function_test(
        TestRunlengthEncode,
        "test_runlength_encode_error_run_count_unsupported_dtype",
        test_runlength_encode_error_run_count_unsupported_dtype,
    )
    add_function_test(
        TestRunlengthEncode,
        "test_runlength_encode_error_unsupported_dtype",
        test_runlength_encode_error_unsupported_dtype,
        devices=devices,
    )

    return TestRunlengthEncode


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    _ = register(unittest.TestCase)
    unittest.main(verbosity=2)
