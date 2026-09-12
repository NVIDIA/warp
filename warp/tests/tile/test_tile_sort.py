# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
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

    for (dtype, length), kernel in kernels.items():
        for j in range(5, 10):
            TILE_DIM = 2**j

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


def _run_large_cpu_tile_sort():
    """Exercise the heap-backed radix path outside the parent runner."""
    wp.config.enable_cpu_blocks = True
    length = 2049
    np_keys = np.arange(length - 1, -1, -1, dtype=np.int32)
    np_values = np.arange(length, dtype=np.int32)

    input_keys = wp.array(np_keys, dtype=wp.int32, device="cpu")
    input_values = wp.array(np_values, dtype=wp.int32, device="cpu")
    output_keys = wp.zeros_like(input_keys)
    output_values = wp.zeros_like(input_values)

    wp.launch_tiled(
        create_sort_kernel(wp.int32, length),
        dim=1,
        inputs=[input_keys, input_values, output_keys, output_values],
        block_dim=32,
        device="cpu",
    )

    sorted_indices = np.argsort(np_keys)
    np.testing.assert_array_equal(output_keys.numpy(), np_keys[sorted_indices])
    np.testing.assert_array_equal(output_values.numpy(), np_values[sorted_indices])
    print("ok")


def test_tile_sort_large_cpu(test, device):
    result = subprocess.run(
        [sys.executable, "-u", "-c", "import warp.tests.tile.test_tile_sort as m; m._run_large_cpu_tile_sort()"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    test.assertEqual(
        result.returncode,
        0,
        f"large CPU tile sort failed (rc={result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    test.assertIn("ok", result.stdout)


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


SURVIVING_LANE_BLOCK_DIM = 32
SURVIVING_LANE_BLOCK_COUNT = 2


def create_surviving_lane_sort_kernel(length):
    @wp.kernel(enable_backward=False)
    def tile_sort_surviving_lane_kernel(
        input_keys: wp.array[wp.int32],
        input_values: wp.array[wp.int32],
        output_keys: wp.array[wp.int32],
        output_values: wp.array[wp.int32],
    ):
        tid = wp.tid()
        block = tid // SURVIVING_LANE_BLOCK_DIM
        lane = tid % SURVIVING_LANE_BLOCK_DIM
        offset = block * length

        keys = wp.tile_load(input_keys, shape=length, offset=offset, storage="shared")
        values = wp.tile_load(input_values, shape=length, offset=offset, storage="shared")

        # The sort must elect a surviving leader after lane zero leaves the block.
        if lane == 0:
            return

        wp.tile_sort(keys, values)

        # A surviving lane copies every shared result because lane zero can no
        # longer participate in a cooperative tile_store().
        if lane == 1:
            for i in range(length):
                output_keys[offset + i] = keys[i]
                output_values[offset + i] = values[i]

    return tile_sort_surviving_lane_kernel


def _run_surviving_lane_cpu_tile_sort(length):
    """Run two blocks so the second sort reuses fibers and fresh leader state."""
    wp.config.enable_cpu_blocks = True
    rng = np.random.default_rng(1638 + length)
    np_keys = np.concatenate([rng.permutation(length), rng.permutation(length)]).astype(np.int32)
    np_values = np.arange(SURVIVING_LANE_BLOCK_COUNT * length, dtype=np.int32)

    input_keys = wp.array(np_keys, dtype=wp.int32, device="cpu")
    input_values = wp.array(np_values, dtype=wp.int32, device="cpu")
    output_keys = wp.full_like(input_keys, -1)
    output_values = wp.full_like(input_values, -1)

    wp.launch(
        create_surviving_lane_sort_kernel(length),
        dim=SURVIVING_LANE_BLOCK_COUNT * SURVIVING_LANE_BLOCK_DIM,
        inputs=[input_keys, input_values],
        outputs=[output_keys, output_values],
        block_dim=SURVIVING_LANE_BLOCK_DIM,
        device="cpu",
    )

    for block in range(SURVIVING_LANE_BLOCK_COUNT):
        block_slice = slice(block * length, (block + 1) * length)
        sorted_indices = np.argsort(np_keys[block_slice])
        np.testing.assert_array_equal(output_keys.numpy()[block_slice], np_keys[block_slice][sorted_indices])
        np.testing.assert_array_equal(output_values.numpy()[block_slice], np_values[block_slice][sorted_indices])


def test_tile_sort_surviving_lane(test, device):
    _run_surviving_lane_cpu_tile_sort(17)


def test_tile_sort_surviving_lane_large_cpu(test, device):
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            "import warp.tests.tile.test_tile_sort as m; m._run_surviving_lane_cpu_tile_sort(2049); print('ok')",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    test.assertEqual(
        result.returncode,
        0,
        f"large surviving-lane CPU tile sort failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    test.assertIn("ok", result.stdout)


devices = get_test_devices()


class TestTileSort(unittest.TestCase):
    pass


add_function_test(TestTileSort, "test_tile_sort", test_tile_sort, devices=devices)
add_function_test(TestTileSort, "test_tile_sort_bfloat16_payload", test_tile_sort_bfloat16_payload, devices=devices)
add_function_test(TestTileSort, "test_tile_sort_large_cpu", test_tile_sort_large_cpu, devices=["cpu"])
add_function_test(
    TestTileSort,
    "test_tile_sort_surviving_lane",
    test_tile_sort_surviving_lane,
    devices=["cpu"] if wp.is_cpu_available() else [],
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileSort,
    "test_tile_sort_surviving_lane_large_cpu",
    test_tile_sort_surviving_lane_large_cpu,
    devices=["cpu"] if wp.is_cpu_available() else [],
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileSort,
    "test_tile_sort_cpu_blocks",
    test_tile_sort,
    devices=["cpu"] if wp.is_cpu_available() else [],
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileSort,
    "test_tile_sort_bfloat16_payload_cpu_blocks",
    test_tile_sort_bfloat16_payload,
    devices=["cpu"] if wp.is_cpu_available() else [],
    enable_cpu_blocks=True,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
