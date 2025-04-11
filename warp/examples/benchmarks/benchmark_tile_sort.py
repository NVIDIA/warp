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

import time

import numpy as np

import warp as wp

BLOCK_DIM = 128


def create_test_kernel(KEY_TYPE, MAX_SORT_LENGTH):
    @wp.kernel
    def tile_sort_kernel(
        input_keys: wp.array(dtype=KEY_TYPE, ndim=2),
        input_values: wp.array(dtype=wp.int32, ndim=2),
        output_keys: wp.array(dtype=KEY_TYPE, ndim=2),
        output_values: wp.array(dtype=wp.int32, ndim=2),
    ):
        batch_id, i = wp.tid()

        # Load input into shared memory
        keys = wp.tile_load(input_keys[batch_id], shape=MAX_SORT_LENGTH, storage="shared")
        values = wp.tile_load(input_values[batch_id], shape=MAX_SORT_LENGTH, storage="shared")

        # Perform in-place sorting
        wp.tile_sort(keys, values)

        # Store sorted shared memory into output arrays
        wp.tile_store(output_keys[batch_id], keys)
        wp.tile_store(output_values[batch_id], values)

    return tile_sort_kernel


if __name__ == "__main__":
    wp.config.quiet = True
    wp.init()
    wp.clear_kernel_cache()
    wp.set_module_options({"fast_math": True, "enable_backward": False})

    iterations = 100
    rng = np.random.default_rng(42)

    shared_benchmark_data = {}
    cub_segmented_sort_benchmark_data = {}

    array_length = list(range(16, 257, 16))

    print(
        f"{'Type':<12s} {'Batch Size':<12s} {'Length':<12s} {'Tile Sort (ms)':<16s} {'Cub Segmented Sort (ms)':<24s} {'CubTime/TileTime':<16s}"
    )
    print("-" * 100)

    for dtype in [int, float]:
        for batch_size_exponent in range(5, 11):
            batch_size = 2**batch_size_exponent
            for length in array_length:
                if dtype == int:
                    np_keys = rng.choice(1000000000, size=(batch_size, length), replace=False)
                else:  # dtype == float
                    np_keys = rng.choice(1000000, size=(batch_size, length), replace=False).astype(np.float32)

                np_values = np.tile(np.arange(length), (batch_size, 1))

                # Sort using NumPy for validation
                np_sorted_keys = np.zeros_like(np_keys)
                np_sorted_values = np.zeros_like(np_values)
                for b in range(batch_size):
                    sorted_indices = np.argsort(np_keys[b])
                    np_sorted_keys[b] = np_keys[b][sorted_indices]
                    np_sorted_values[b] = np_values[b][sorted_indices]

                # Generate random keys and iota indexer
                input_keys = wp.array(np_keys, dtype=dtype, ndim=2, device="cuda")
                input_values = wp.array(np_values, dtype=int, ndim=2, device="cuda")
                output_keys = wp.zeros_like(input_keys, device="cuda")
                output_values = wp.zeros_like(input_values, device="cuda")

                kernel = create_test_kernel(dtype, length)

                cmd = wp.launch_tiled(
                    kernel,
                    dim=batch_size,
                    inputs=[input_keys, input_values, output_keys, output_values],
                    block_dim=BLOCK_DIM,
                    record_cmd=True,
                )
                # Warmup
                for _ in range(5):
                    cmd.launch()

                with wp.ScopedTimer("benchmark", cuda_filter=wp.TIMING_KERNEL, print=False, synchronize=True) as timer:
                    for _ in range(iterations):
                        cmd.launch()
                    wp.synchronize()

                if dtype == int:
                    keys_match = np.array_equal(output_keys.numpy(), np_sorted_keys)
                else:  # dtype == float
                    keys_match = np.allclose(output_keys.numpy(), np_sorted_keys, atol=1e-6)  # Use tolerance for floats

                values_match = np.array_equal(output_values.numpy(), np_sorted_values)

                # Validate results
                assert keys_match, f"Key sorting mismatch for dtype={dtype}!"
                assert values_match, f"Value sorting mismatch for dtype={dtype}!"

                timing_results = [result.elapsed for result in timer.timing_results]
                mean_timing = np.mean(timing_results)

                shared_benchmark_data[length] = mean_timing

                # Allocate memory
                input_keys = wp.zeros(shape=(batch_size * 2, length), dtype=dtype, device="cuda")
                input_values = wp.zeros(shape=(batch_size * 2, length), dtype=int, device="cuda")

                # Copy data
                input_keys.assign(np_keys)
                input_values.assign(np_values)

                input_keys = input_keys.reshape(-1)
                input_values = input_values.reshape(-1)

                segments = wp.array(np.arange(0, batch_size + 1) * length, dtype=int, device="cuda")

                # Compare with cub segmented radix sort
                # Warmup
                for _ in range(5):
                    wp.utils.segmented_sort_pairs(input_keys, input_values, batch_size * length, segments)

                t1 = time.time_ns()
                for _ in range(iterations):
                    wp.utils.segmented_sort_pairs(input_keys, input_values, batch_size * length, segments)
                wp.synchronize()
                t2 = time.time_ns()
                cub_segmented_sort_benchmark_data[length] = (t2 - t1) / (1_000_000 * iterations)

                # Print results
                print(
                    f"{dtype!s:<12s} {batch_size:<12d} {length:<12d} {shared_benchmark_data[length]:<16.4g} {cub_segmented_sort_benchmark_data[length]:<24.4g} {cub_segmented_sort_benchmark_data[length] / shared_benchmark_data[length]:<16.4g}"
                )
