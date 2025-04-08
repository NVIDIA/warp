# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def create_sort_kernel(KEY_TYPE, MAX_SORT_LENGTH):
    @wp.kernel
    def tile_sort_kernel(
        input_keys: wp.array(dtype=KEY_TYPE),
        input_values: wp.array(dtype=wp.int32),
        output_keys: wp.array(dtype=KEY_TYPE),
        output_values: wp.array(dtype=wp.int32),
    ):
        # Load input into shared memory
        keys = wp.tile_load(input_keys, shape=MAX_SORT_LENGTH, storage="shared")
        values = wp.tile_load(input_values, shape=MAX_SORT_LENGTH, storage="shared")

        # Perform in-place sorting
        wp.tile_sort(keys, values)

        # Store sorted shared memory into output arrays
        wp.tile_store(output_keys, keys)
        wp.tile_store(output_values, values)

    return tile_sort_kernel


def test_tile_sort(test, device):
    for dtype in [int, float]:  # Loop over int and float keys
        for j in range(5, 10):
            TILE_DIM = 2**j
            for i in range(0, 11):  # Start from 1 to avoid zero-length cases
                length = 2**i + 1

                rng = np.random.default_rng(42)  # Create a random generator instance

                if dtype == int:
                    np_keys = rng.choice(1000000000, size=length, replace=False)
                else:  # dtype == float
                    np_keys = rng.uniform(0, 1000000000, size=length)

                np_values = np.arange(length)

                # Generate random keys and iota indexer
                input_keys = wp.array(np_keys, dtype=dtype, device=device)
                input_values = wp.array(np_values, dtype=int, device=device)
                output_keys = wp.zeros_like(input_keys, device=device)
                output_values = wp.zeros_like(input_values, device=device)

                # Execute sorting kernel
                kernel = create_sort_kernel(dtype, length)
                wp.launch_tiled(
                    kernel,
                    dim=1,
                    inputs=[input_keys, input_values, output_keys, output_values],
                    block_dim=TILE_DIM,
                    device=device,
                )
                wp.synchronize()

                # Sort using NumPy for validation
                sorted_indices = np.argsort(np_keys)
                np_sorted_keys = np_keys[sorted_indices]
                np_sorted_values = np_values[sorted_indices]

                if dtype == int:
                    keys_match = np.array_equal(output_keys.numpy(), np_sorted_keys)
                else:  # dtype == float
                    keys_match = np.allclose(output_keys.numpy(), np_sorted_keys, atol=1e-6)  # Use tolerance for floats

                values_match = np.array_equal(output_values.numpy(), np_sorted_values)

                if not keys_match or not values_match:
                    print(f"Test failed for dtype={dtype}, TILE_DIM={TILE_DIM}, length={length}")
                    print("")
                    print(output_keys.numpy())
                    print(np_sorted_keys)
                    print("")
                    print(output_values.numpy())
                    print(np_sorted_values)
                    print("")

                # Validate results
                assert keys_match, f"Key sorting mismatch for dtype={dtype}!"
                assert values_match, f"Value sorting mismatch for dtype={dtype}!"


devices = get_test_devices()


class TestTileSort(unittest.TestCase):
    pass


add_function_test(TestTileSort, "test_tile_sort", test_tile_sort, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
