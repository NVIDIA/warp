# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


def test_atomic_bitwise(test, device):
    @wp.kernel
    def test_atomic_bitwise_kernel(a: wp.array(dtype=wp.uint32), b: wp.array(dtype=wp.uint32), op_type: int):
        i = wp.tid()
        word_idx = i // 32
        bit_idx = i % 32
        bit_mask = uint32(1) << uint32(bit_idx)
        if op_type == 0:
            a[word_idx] &= (b[word_idx] & bit_mask) | ~bit_mask
        elif op_type == 1:
            a[word_idx] |= b[word_idx] & bit_mask
        elif op_type == 2:
            a[word_idx] ^= b[word_idx] & bit_mask
        elif op_type == 3:
            wp.atomic_and(a, word_idx, (b[word_idx] & bit_mask) | ~bit_mask)
        elif op_type == 4:
            wp.atomic_or(a, word_idx, b[word_idx] & bit_mask)
        elif op_type == 5:
            wp.atomic_xor(a, word_idx, b[word_idx] & bit_mask)

    n = 1024
    rng = np.random.default_rng(42)

    a = rng.integers(0, np.iinfo(np.uint32).max, size=n, dtype=np.uint32)
    b = rng.integers(0, np.iinfo(np.uint32).max, size=n, dtype=np.uint32)

    expected_and = a & b
    expected_or = a | b
    expected_xor = a ^ b

    with wp.ScopedDevice(device):
        and_op_array = wp.array(a, dtype=wp.uint32, device=device)
        or_op_array = wp.array(a, dtype=wp.uint32, device=device)
        xor_op_array = wp.array(a, dtype=wp.uint32, device=device)
        atomic_and_array = wp.array(a, dtype=wp.uint32, device=device)
        atomic_or_array = wp.array(a, dtype=wp.uint32, device=device)
        atomic_xor_array = wp.array(a, dtype=wp.uint32, device=device)

        target_array = wp.array(b, dtype=wp.uint32, device=device)

        wp.launch(test_atomic_bitwise_kernel, dim=n * 32, inputs=[and_op_array, target_array, 0])
        wp.launch(test_atomic_bitwise_kernel, dim=n * 32, inputs=[or_op_array, target_array, 1])
        wp.launch(test_atomic_bitwise_kernel, dim=n * 32, inputs=[xor_op_array, target_array, 2])
        wp.launch(test_atomic_bitwise_kernel, dim=n * 32, inputs=[atomic_and_array, target_array, 3])
        wp.launch(test_atomic_bitwise_kernel, dim=n * 32, inputs=[atomic_or_array, target_array, 4])
        wp.launch(test_atomic_bitwise_kernel, dim=n * 32, inputs=[atomic_xor_array, target_array, 5])

        assert_np_equal(and_op_array.numpy(), expected_and)
        assert_np_equal(or_op_array.numpy(), expected_or)
        assert_np_equal(xor_op_array.numpy(), expected_xor)
        assert_np_equal(atomic_and_array.numpy(), expected_and)
        assert_np_equal(atomic_or_array.numpy(), expected_or)
        assert_np_equal(atomic_xor_array.numpy(), expected_xor)


devices = get_test_devices()


class TestAtomicBitwise(unittest.TestCase):
    pass


add_function_test(TestAtomicBitwise, "test_atomic_bitwise", test_atomic_bitwise, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
