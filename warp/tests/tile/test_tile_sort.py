# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Compilation hygiene
#
# Tile sort generates substantial code for every length and block dim. These tests do not exercise autodiff, so
# compiling backward kernels would substantially increase the module size without adding coverage. The bfloat16
# kernel only runs at one block dim and uses a unique module so it is not included in all five regular sort module
# variants.


def create_sort_kernel(KEY_TYPE, MAX_SORT_LENGTH):
    @wp.kernel(enable_backward=False)
    def tile_sort_kernel(
        input_keys: wp.array[KEY_TYPE],
        input_values: wp.array[wp.int32],
        output_keys: wp.array[KEY_TYPE],
        output_values: wp.array[wp.int32],
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
    # Forward-declare kernels for more efficient compilation
    kernels = {}
    for dtype in [wp.int32, wp.int64, wp.uint64, wp.float32]:
        # Limit 64-bit types to 2^10 elements to avoid running out of shared memory
        max_power = 10 if dtype in [wp.int64, wp.uint64] else 11
        for i in range(0, max_power):
            length = 2**i + 1
            kernels[(dtype, length)] = create_sort_kernel(dtype, length)

    # Safe block_dim values are only powers of two >= 32 for the large-tile implementation.
    # For lengths <= 32, smaller block_dims may be accepted but are not reliable.
    # The test must only use safe block_dim values to avoid CUDA errors and mis-sorts.
    safe_block_dims = [32, 64, 128, 256, 512]

    for (dtype, length), kernel in kernels.items():
        for TILE_DIM in safe_block_dims:
            # Ensure the tile length fits within the block dim's shared memory allocation
            if length > TILE_DIM:
                continue

            rng = np.random.default_rng(42)  # Create a random generator instance

            if dtype == wp.int32:
                # Generate integers in range [-500000000, 500000000)
                np_keys = rng.integers(-500000000, 500000000, size=length, dtype=np.int32)
            elif dtype == wp.int64:
                # Generate integers in range [-500000000000, 500000000000)
                np_keys = rng.integers(-500000000000, 500000000000, size=length, dtype=np.int64)
            elif dtype == wp.uint64:
                np_keys = rng.integers(0, 1000000000000, size=length, dtype=np.uint64)
            else:  # dtype == wp.float32
                # Generate floats in range [-500000000, 500000000)
                np_keys = rng.uniform(-500000000, 500000000, size=length).astype(np.float32)

            np_values = np.arange(length)

            # Generate random keys and iota indexer
            input_keys = wp.array(np_keys, dtype=dtype, device=device)
            input_values = wp.array(np_values, dtype=int, device=device)
            output_keys = wp.zeros_like(input_keys, device=device)
            output_values = wp.zeros_like(input_values, device=device)

            # Execute sorting kernel
            wp.launch_tiled(
                kernel,
                dim=1,
                inputs=[input_keys, input_values, output_keys, output_values],
                block_dim=TILE_DIM,
                device=device,
            )

            # Sort using NumPy for validation
            sorted_indices = np.argsort(np_keys)
            np_sorted_keys = np_keys[sorted_indices]
            np_sorted_values = np_values[sorted_indices]

            context = f"dtype={dtype}, TILE_DIM={TILE_DIM}, length={length}"
            if dtype == wp.float32:
                np.testing.assert_allclose(
                    output_keys.numpy(),
                    np_sorted_keys,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Key sorting mismatch for {context}",
                )
            else:  # Integer types
                np.testing.assert_array_equal(
                    output_keys.numpy(),
                    np_sorted_keys,
                    err_msg=f"Key sorting mismatch for {context}",
                )

            np.testing.assert_array_equal(
                output_values.numpy(),
                np_sorted_values,
                err_msg=f"Value sorting mismatch for {context}",
            )


def create_bfloat16_payload_sort_kernel(length):
    @wp.kernel(enable_backward=False, module="unique")
    def tile_sort_bfloat16_kernel(
        input_keys: wp.array[wp.float32],
        input_values: wp.array[wp.bfloat16],
        output_keys: wp.array[wp.float32],
        output_values: wp.array[wp.bfloat16],
    ):
        keys = wp.tile_load(input_keys, shape=length, storage="shared")
        values = wp.tile_load(input_values, shape=length, storage="shared")
        wp.tile_sort(keys, values)
        wp.tile_store(output_keys, keys)
        wp.tile_store(output_values, values)

    return tile_sort_bfloat16_kernel


def test_tile_sort_bfloat16_payload(test, device):
    """Sort keys with a bfloat16 value payload.

    Exercises the bfloat16 warp-shuffle overload in the radix sort. Integer values are exact
    in bfloat16.
    """
    length = 8
    np_keys = np.arange(length - 1, -1, -1, dtype=np.float32)
    np_values = np.arange(1, length + 1, dtype=np.float32)

    input_keys = wp.array(np_keys, dtype=wp.float32, device=device)
    input_values = wp.array(np_values, dtype=wp.bfloat16, device=device)
    output_keys = wp.zeros_like(input_keys, device=device)
    output_values = wp.zeros_like(input_values, device=device)

    wp.launch_tiled(
        create_bfloat16_payload_sort_kernel(length),
        dim=1,
        inputs=[input_keys, input_values, output_keys, output_values],
        block_dim=32,
        device=device,
    )

    sorted_indices = np.argsort(np_keys)
    np.testing.assert_allclose(output_keys.numpy(), np_keys[sorted_indices], atol=1e-6)

    # Without ml_dtypes, .numpy() returns the raw uint16 bfloat16 bit patterns; decode to float32.
    np_out = output_values.numpy()
    if np_out.dtype == np.uint16:
        decoded = (np_out.astype(np.uint32) << 16).view(np.float32)
    else:
        decoded = np_out.astype(np.float32)
    assert_np_equal(decoded, np_values[sorted_indices])


devices = get_test_devices()


class TestTileSort(unittest.TestCase):
    pass


add_function_test(TestTileSort, "test_tile_sort", test_tile_sort, devices=devices)
add_function_test(TestTileSort, "test_tile_sort_bfloat16_payload", test_tile_sort_bfloat16_payload, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
