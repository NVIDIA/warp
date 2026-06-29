# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import sys
import unittest

import numpy as np

import warp as wp
from warp._src.types import _np_bfloat16_bits_to_float32
from warp.tests.unittest_utils import get_selected_cuda_test_devices, get_test_devices


def _reference_scatter_add_float32(data_np, idx_np, out_size):
    reference = np.zeros(out_size, dtype=np.float32)
    for value, idx in zip(data_np.astype(np.float32), idx_np, strict=True):
        reference[int(idx)] = np.float32(reference[int(idx)] + np.float32(value))
    return reference


def _bfloat16_numpy_bits(values):
    if values.dtype == np.uint16:
        return values.astype(np.uint16, copy=False)
    return (values.astype(np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _bfloat16_numpy_to_float32(values):
    bits = _bfloat16_numpy_bits(values)
    return _np_bfloat16_bits_to_float32(bits)


cuda_devices = get_selected_cuda_test_devices()
bfloat16_cuda_devices = [device for device in cuda_devices if device.arch >= 80]
all_devices = get_test_devices()
cpu_device = wp.get_device("cpu")
REPEAT_COUNT = 3


def assert_equal_repeated(make_result, *, runs=REPEAT_COUNT, err_msg=None):
    """Run ``make_result`` several times and require bit-exact identical outputs."""

    first = make_result()
    for i in range(1, runs):
        result = make_result()
        message = err_msg or f"Run 0 vs run {i} differ"
        if isinstance(first, tuple):
            for expected, actual in zip(first, result, strict=True):
                np.testing.assert_array_equal(expected, actual, err_msg=message)
        else:
            np.testing.assert_array_equal(first, result, err_msg=message)
    return first


class DeterministicTestBase(unittest.TestCase):
    """Base class that enables deterministic lowering for one or more test modules."""

    deterministic_modules = None

    @classmethod
    def setUpClass(cls):
        cls._old_deterministic = wp.config.deterministic
        cls._old_module_deterministic = []
        wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN

        modules = cls.deterministic_modules
        if modules is None:
            modules = (sys.modules[cls.__module__],)

        for module in modules:
            old_options = wp.get_module_options(module=module)
            cls._old_module_deterministic.append((module, old_options["deterministic"]))
            wp.set_module_options({"deterministic": wp.DeterministicMode.RUN_TO_RUN}, module=module)

    @classmethod
    def tearDownClass(cls):
        for module, old_deterministic in reversed(cls._old_module_deterministic):
            wp.set_module_options({"deterministic": old_deterministic}, module=module)
        wp.config.deterministic = cls._old_deterministic
