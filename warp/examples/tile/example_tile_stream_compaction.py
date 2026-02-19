# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

###########################################################################
# Example Tile Stream Compaction
#
# Shows how to implement parallel stream compaction using Warp tile
# primitives. Stream compaction filters an array to keep only elements
# that satisfy a condition, producing a dense output array.
#
# Key concepts demonstrated:
# - tile_scan_inclusive: Compute prefix sums for output positions
# - tile_from_thread: Broadcast a value from one thread to all threads in the block
# - tile_map: Apply a function to compute output indices
# - tile_store_indexed: Scatter write valid elements to output
#
###########################################################################

import argparse

import numpy as np

import warp as wp

wp.set_module_options({"enable_backward": False})

TILE_SIZE = 32
NUM_BLOCKS = 4


@wp.func
def compute_index(valid: int, scan_inclusive: int, offset: int) -> int:
    """Compute the output index for each element.

    Returns the write position for valid elements, or -1 for invalid
    elements (which will be ignored by tile_store_indexed).
    """
    if valid == 1:
        return scan_inclusive + offset - 1
    else:
        return -1


@wp.kernel
def stream_compaction_kernel(
    is_valid: wp.array(dtype=int),
    data: wp.array(dtype=float),
    num_output: wp.array(dtype=int),
    output: wp.array(dtype=float),
):
    i, j = wp.tid()  # i = block index, j = thread within block

    # Load data and validity flags for this block
    data_tile = wp.tile_load(data, TILE_SIZE, offset=i * TILE_SIZE)
    valid_tile = wp.tile_load(is_valid, TILE_SIZE, offset=i * TILE_SIZE)

    # Compute inclusive prefix sum of validity flags
    # The last element directly gives the total count of valid elements
    scan_inclusive = wp.tile_scan_inclusive(valid_tile)

    # Only the last thread in the block atomically reserves output space
    offset = 0
    if j == wp.block_dim() - 1:
        block_count = wp.tile_extract(scan_inclusive, wp.block_dim() - 1)
        offset = wp.atomic_add(num_output, 0, block_count)

    # Broadcast the offset to all threads using tile_from_thread
    # This uses only 1 element of shared memory for the broadcast
    offset_tile = wp.tile_from_thread(
        shape=TILE_SIZE,
        value=offset,
        thread_idx=wp.block_dim() - 1,
        storage="shared",
    )

    # Compute output indices: valid elements get (offset + scan - 1), invalid get -1
    index_tile = wp.tile_map(compute_index, valid_tile, scan_inclusive, offset_tile)

    # Scatter valid elements to output (indices of -1 are ignored)
    wp.tile_store_indexed(output, index_tile, data_tile, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    args = parser.parse_known_args()[0]

    if args.device == "cpu":
        print("This example only runs on CUDA devices.")
        exit(0)

    if args.device is None and not wp.is_cuda_available():
        print("This example requires a CUDA device.")
        exit(0)

    with wp.ScopedDevice(args.device):
        total_elements = NUM_BLOCKS * TILE_SIZE

        # Create test data: every other element is valid
        is_valid_np = np.zeros(total_elements, dtype=np.int32)
        is_valid_np[::2] = 1  # elements 0, 2, 4, 6, ... are valid

        data_np = np.arange(total_elements, dtype=np.float32) * 10.0

        is_valid = wp.array(is_valid_np, dtype=int)
        data = wp.array(data_np, dtype=float)
        num_output = wp.zeros(1, dtype=int)
        output = wp.zeros(total_elements, dtype=float)

        # Run stream compaction
        wp.launch_tiled(
            stream_compaction_kernel,
            dim=[NUM_BLOCKS],
            inputs=[is_valid, data, num_output, output],
            block_dim=TILE_SIZE,
        )

        # Get results
        num_valid = num_output.numpy()[0]
        result = output.numpy()[:num_valid]

        # Verify correctness
        expected = data_np[is_valid_np == 1]
        # Note: order within each block is preserved, but blocks may write
        # to output in any order depending on GPU scheduling
        np.testing.assert_allclose(np.sort(result), np.sort(expected))

        print(f"Input: {total_elements} elements, {np.sum(is_valid_np)} valid")
        print(f"Input data (first 16):  {data_np[:16]}")
        print(f"Validity   (first 16):  {is_valid_np[:16]}")
        print()
        print(f"Output: {num_valid} compacted elements")
        print(f"Output data: {result}")
