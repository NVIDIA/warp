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


def test_tile_atomic_bitwise_scalar(test, device):
    @wp.kernel
    def test_tile_atomic_bitwise_scalar_kernel(
        a: wp.array(dtype=wp.uint32), b: wp.array(dtype=wp.uint32), c: wp.array(dtype=wp.uint32), op_type: int
    ):
        word_idx, bit_idx = wp.tid()
        block_dim = wp.block_dim()
        assert block_dim == 32
        s = wp.tile_zeros(shape=1, dtype=wp.uint32)
        # write to tile first, then write only once to the array
        s[0] = a[word_idx]
        if op_type < 3:
            bit_mask = wp.uint32(1) << wp.uint32(bit_idx)
            if op_type == 0:
                s[0] &= (b[word_idx] & bit_mask) | ~bit_mask
            elif op_type == 1:
                s[0] |= b[word_idx] & bit_mask
            elif op_type == 2:
                s[0] ^= b[word_idx] & bit_mask
        else:
            # inter-tile operations
            s_bit_mask = wp.tile_zeros(shape=32, dtype=wp.uint32)
            s_bit_mask[(bit_idx + 1) % 32] = wp.uint32(1) << wp.uint32((bit_idx + 1) % 32)
            if op_type == 3:
                s[0] &= (b[word_idx] & s_bit_mask[bit_idx]) | ~s_bit_mask[bit_idx]
            elif op_type == 4:
                s[0] |= b[word_idx] & s_bit_mask[bit_idx]
            elif op_type == 5:
                s[0] ^= b[word_idx] & s_bit_mask[bit_idx]
        c[word_idx] = s[0]

    @wp.kernel
    def test_tile_atomic_bitwise_scalar_tilewise_kernel(
        a: wp.array(dtype=wp.uint32), b: wp.array(dtype=wp.uint32), c: wp.array(dtype=wp.uint32), op_type: int
    ):
        batch_idx, _ = wp.tid()
        block_dim = wp.block_dim()
        assert block_dim == 32
        # Each tile is responsible for a batch of 32 elements
        s1 = wp.tile_load(a, shape=32, offset=batch_idx * 32)
        s2 = wp.tile_load(b, shape=32, offset=batch_idx * 32)
        # inter-tile operations (batch-wise)
        if op_type < 9:
            if op_type == 6:
                s1 &= s2
            elif op_type == 7:
                s1 |= s2
            elif op_type == 8:
                s1 ^= s2
            wp.tile_store(c, s1, offset=batch_idx * 32)
        else:
            if op_type == 9:
                s3 = s1 & s2
            elif op_type == 10:
                s3 = s1 | s2
            elif op_type == 11:
                s3 = s1 ^ s2
            wp.tile_store(c, s3, offset=batch_idx * 32)

    n = 1024
    rng = np.random.default_rng(42)

    a = rng.integers(0, np.iinfo(np.uint32).max, size=n, dtype=np.uint32)
    b = rng.integers(0, np.iinfo(np.uint32).max, size=n, dtype=np.uint32)

    expected_and = a & b
    expected_or = a | b
    expected_xor = a ^ b

    with wp.ScopedDevice(device):
        a_wp = wp.array(a, dtype=wp.uint32, device=device)
        b_wp = wp.array(b, dtype=wp.uint32, device=device)
        c_wp = wp.zeros(shape=n, dtype=wp.uint32, device=device)

        wp.launch_tiled(test_tile_atomic_bitwise_scalar_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 0], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(test_tile_atomic_bitwise_scalar_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 1], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(test_tile_atomic_bitwise_scalar_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 2], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_xor)
        wp.launch_tiled(test_tile_atomic_bitwise_scalar_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 3], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(test_tile_atomic_bitwise_scalar_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 4], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(test_tile_atomic_bitwise_scalar_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 5], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_xor)

        wp.launch_tiled(
            test_tile_atomic_bitwise_scalar_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 6], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(
            test_tile_atomic_bitwise_scalar_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 7], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(
            test_tile_atomic_bitwise_scalar_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 8], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_xor)
        wp.launch_tiled(
            test_tile_atomic_bitwise_scalar_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 9], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(
            test_tile_atomic_bitwise_scalar_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 10], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(
            test_tile_atomic_bitwise_scalar_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 11], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_xor)


def test_tile_atomic_bitwise_vector(test, device):
    @wp.kernel
    def test_tile_atomic_bitwise_vector_kernel(
        a: wp.array(dtype=wp.vec3ui), b: wp.array(dtype=wp.vec3ui), c: wp.array(dtype=wp.vec3ui), op_type: int
    ):
        word_idx, bit_idx = wp.tid()
        block_dim = wp.block_dim()
        assert block_dim == 32
        s = wp.tile_zeros(shape=1, dtype=wp.vec3ui)
        # write to tile first, then write only once to the array
        s[0] = a[word_idx]
        if op_type < 3:
            bit_mask = wp.vec3ui(wp.uint32(1)) << wp.uint32(bit_idx)
            if op_type == 0:
                s[0] &= (b[word_idx] & bit_mask) | ~bit_mask
            elif op_type == 1:
                s[0] |= b[word_idx] & bit_mask
            elif op_type == 2:
                s[0] ^= b[word_idx] & bit_mask
        else:
            # inter-tile operations
            s_bit_mask = wp.tile_zeros(shape=32, dtype=wp.vec3ui)
            s_bit_mask[(bit_idx + 1) % 32] = wp.vec3ui(wp.uint32(1)) << wp.uint32((bit_idx + 1) % 32)
            if op_type == 3:
                s[0] &= (b[word_idx] & s_bit_mask[bit_idx]) | ~s_bit_mask[bit_idx]
            elif op_type == 4:
                s[0] |= b[word_idx] & s_bit_mask[bit_idx]
            elif op_type == 5:
                s[0] ^= b[word_idx] & s_bit_mask[bit_idx]
        c[word_idx] = s[0]

    @wp.kernel
    def test_tile_atomic_bitwise_vector_tilewise_kernel(
        a: wp.array(dtype=wp.vec3ui), b: wp.array(dtype=wp.vec3ui), c: wp.array(dtype=wp.vec3ui), op_type: int
    ):
        batch_idx, _ = wp.tid()
        block_dim = wp.block_dim()
        assert block_dim == 32
        # Each tile is responsible for a batch of 32 elements
        s1 = wp.tile_load(a, shape=32, offset=batch_idx * 32)
        s2 = wp.tile_load(b, shape=32, offset=batch_idx * 32)
        # inter-tile operations (batch-wise)
        if op_type < 9:
            if op_type == 6:
                s1 &= s2
            elif op_type == 7:
                s1 |= s2
            elif op_type == 8:
                s1 ^= s2
            wp.tile_store(c, s1, offset=batch_idx * 32)
        else:
            if op_type == 9:
                s3 = s1 & s2
            elif op_type == 10:
                s3 = s1 | s2
            elif op_type == 11:
                s3 = s1 ^ s2
            wp.tile_store(c, s3, offset=batch_idx * 32)

    n = 1024
    rng = np.random.default_rng(42)

    a = rng.integers(0, np.iinfo(np.uint32).max, size=(n, 3), dtype=np.uint32)
    b = rng.integers(0, np.iinfo(np.uint32).max, size=(n, 3), dtype=np.uint32)

    expected_and = a & b
    expected_or = a | b
    expected_xor = a ^ b

    with wp.ScopedDevice(device):
        a_wp = wp.array(a, dtype=wp.vec3ui, device=device)
        b_wp = wp.array(b, dtype=wp.vec3ui, device=device)
        c_wp = wp.zeros(shape=n, dtype=wp.vec3ui, device=device)

        wp.launch_tiled(test_tile_atomic_bitwise_vector_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 0], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(test_tile_atomic_bitwise_vector_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 1], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(test_tile_atomic_bitwise_vector_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 2], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_xor)
        wp.launch_tiled(test_tile_atomic_bitwise_vector_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 3], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(test_tile_atomic_bitwise_vector_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 4], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(test_tile_atomic_bitwise_vector_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 5], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_xor)

        wp.launch_tiled(
            test_tile_atomic_bitwise_vector_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 6], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(
            test_tile_atomic_bitwise_vector_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 7], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(
            test_tile_atomic_bitwise_vector_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 8], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_xor)
        wp.launch_tiled(
            test_tile_atomic_bitwise_vector_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 9], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(
            test_tile_atomic_bitwise_vector_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 10], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(
            test_tile_atomic_bitwise_vector_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 11], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_xor)


mat33ui = wp.types.matrix(shape=(3, 3), dtype=wp.uint32)


def test_tile_atomic_bitwise_matrix(test, device):
    @wp.kernel
    def test_tile_atomic_bitwise_matrix_kernel(
        a: wp.array(dtype=mat33ui), b: wp.array(dtype=mat33ui), c: wp.array(dtype=mat33ui), op_type: int
    ):
        word_idx, bit_idx = wp.tid()
        block_dim = wp.block_dim()
        assert block_dim == 32
        s = wp.tile_zeros(shape=1, dtype=mat33ui)
        # write to tile first, then write only once to the array
        s[0] = a[word_idx]
        if op_type < 3:
            bit_mask = mat33ui(wp.uint32(1)) << wp.uint32(bit_idx)
            if op_type == 0:
                s[0] &= (b[word_idx] & bit_mask) | ~bit_mask
            elif op_type == 1:
                s[0] |= b[word_idx] & bit_mask
            elif op_type == 2:
                s[0] ^= b[word_idx] & bit_mask
        else:
            # inter-tile operations
            s_bit_mask = wp.tile_zeros(shape=32, dtype=mat33ui)
            s_bit_mask[(bit_idx + 1) % 32] = mat33ui(wp.uint32(1)) << wp.uint32((bit_idx + 1) % 32)
            if op_type == 3:
                s[0] &= (b[word_idx] & s_bit_mask[bit_idx]) | ~s_bit_mask[bit_idx]
            elif op_type == 4:
                s[0] |= b[word_idx] & s_bit_mask[bit_idx]
            elif op_type == 5:
                s[0] ^= b[word_idx] & s_bit_mask[bit_idx]
        c[word_idx] = s[0]

    @wp.kernel
    def test_tile_atomic_bitwise_matrix_tilewise_kernel(
        a: wp.array(dtype=mat33ui), b: wp.array(dtype=mat33ui), c: wp.array(dtype=mat33ui), op_type: int
    ):
        batch_idx, _ = wp.tid()
        block_dim = wp.block_dim()
        assert block_dim == 32
        # Each tile is responsible for a batch of 32 elements
        s1 = wp.tile_load(a, shape=32, offset=batch_idx * 32)
        s2 = wp.tile_load(b, shape=32, offset=batch_idx * 32)
        # inter-tile operations (batch-wise)
        if op_type < 9:
            if op_type == 6:
                s1 &= s2
            elif op_type == 7:
                s1 |= s2
            elif op_type == 8:
                s1 ^= s2
            wp.tile_store(c, s1, offset=batch_idx * 32)
        else:
            if op_type == 9:
                s3 = s1 & s2
            elif op_type == 10:
                s3 = s1 | s2
            elif op_type == 11:
                s3 = s1 ^ s2
            wp.tile_store(c, s3, offset=batch_idx * 32)

    n = 1024
    rng = np.random.default_rng(42)

    a = rng.integers(0, np.iinfo(np.uint32).max, size=(n, 3, 3), dtype=np.uint32)
    b = rng.integers(0, np.iinfo(np.uint32).max, size=(n, 3, 3), dtype=np.uint32)

    expected_and = a & b
    expected_or = a | b
    expected_xor = a ^ b

    with wp.ScopedDevice(device):
        a_wp = wp.array(a, dtype=mat33ui, device=device)
        b_wp = wp.array(b, dtype=mat33ui, device=device)
        c_wp = wp.zeros(shape=n, dtype=mat33ui, device=device)

        wp.launch_tiled(test_tile_atomic_bitwise_matrix_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 0], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(test_tile_atomic_bitwise_matrix_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 1], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(test_tile_atomic_bitwise_matrix_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 2], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_xor)
        wp.launch_tiled(test_tile_atomic_bitwise_matrix_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 3], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(test_tile_atomic_bitwise_matrix_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 4], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(test_tile_atomic_bitwise_matrix_kernel, dim=n, inputs=[a_wp, b_wp, c_wp, 5], block_dim=32)
        assert_np_equal(c_wp.numpy(), expected_xor)

        wp.launch_tiled(
            test_tile_atomic_bitwise_matrix_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 6], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(
            test_tile_atomic_bitwise_matrix_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 7], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(
            test_tile_atomic_bitwise_matrix_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 8], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_xor)
        wp.launch_tiled(
            test_tile_atomic_bitwise_matrix_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 9], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_and)
        wp.launch_tiled(
            test_tile_atomic_bitwise_matrix_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 10], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_or)
        wp.launch_tiled(
            test_tile_atomic_bitwise_matrix_tilewise_kernel, dim=n // 32, inputs=[a_wp, b_wp, c_wp, 11], block_dim=32
        )
        assert_np_equal(c_wp.numpy(), expected_xor)


devices = get_test_devices()


class TestTileAtomicBitwise(unittest.TestCase):
    pass


add_function_test(
    TestTileAtomicBitwise,
    "test_tile_atomic_bitwise_scalar",
    test_tile_atomic_bitwise_scalar,
    devices=get_cuda_test_devices(),
)

add_function_test(
    TestTileAtomicBitwise,
    "test_tile_atomic_bitwise_vector",
    test_tile_atomic_bitwise_vector,
    devices=get_cuda_test_devices(),
)

add_function_test(
    TestTileAtomicBitwise,
    "test_tile_atomic_bitwise_matrix",
    test_tile_atomic_bitwise_matrix,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
