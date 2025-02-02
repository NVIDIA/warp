# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import warp as wp

BLOCK_DIM = 128

TILE = 32


def create_test_kernel(storage_type: str):
    @wp.kernel
    def load_store(a: wp.array2d(dtype=wp.float32), b: wp.array2d(dtype=wp.float32)):
        i, j = wp.tid()

        if wp.static(storage_type == "shared"):
            a_tile = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="shared")
        else:
            a_tile = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="register")

        wp.tile_store(b, a_tile, offset=(i * TILE, j * TILE))

    return load_store


if __name__ == "__main__":
    wp.config.quiet = True
    wp.init()
    wp.clear_kernel_cache()
    wp.set_module_options({"fast_math": True, "enable_backward": False})

    iterations = 100
    rng = np.random.default_rng(42)

    shared_benchmark_data = {}
    register_benchmark_data = {}
    memcpy_benchmark_data = {}

    sizes = list(range(128, 4097, 128))

    print(f"{'Transfer Size (Bytes)':<23s} {'Shared (GiB/s)':<16s} {'Register (GiB/s)':<18s} {'memcpy (GiB/s)':<16s}")
    print("-" * 79)

    for size in sizes:
        a = wp.array(rng.random((size, size), dtype=np.float32), dtype=wp.float32)
        b = wp.empty_like(a)

        for storage_type in ("shared", "register"):
            load_store = create_test_kernel(storage_type)

            cmd = wp.launch_tiled(
                load_store,
                dim=(a.shape[0] // TILE, a.shape[1] // TILE),
                inputs=[a],
                outputs=[b],
                block_dim=BLOCK_DIM,
                record_cmd=True,
            )
            # Warmup
            for _ in range(5):
                cmd.launch()

            with wp.ScopedTimer("benchmark", cuda_filter=wp.TIMING_KERNEL, print=False, synchronize=True) as timer:
                for _ in range(iterations):
                    cmd.launch()

            np.testing.assert_equal(a.numpy(), b.numpy())

            timing_results = [result.elapsed for result in timer.timing_results]
            avg_bw = 2.0 * (a.capacity / (1024 * 1024 * 1024)) / (1e-3 * np.mean(timing_results))

            if storage_type == "shared":
                shared_benchmark_data[a.capacity] = avg_bw
            else:
                register_benchmark_data[a.capacity] = avg_bw

        # Compare with memcpy
        with wp.ScopedTimer("benchmark", cuda_filter=wp.TIMING_MEMCPY, print=False, synchronize=True) as timer:
            for _ in range(iterations):
                wp.copy(b, a)

        timing_results = [result.elapsed for result in timer.timing_results]
        avg_bw = 2.0 * (a.capacity / (1024 * 1024 * 1024)) / (1e-3 * np.mean(timing_results))
        memcpy_benchmark_data[a.capacity] = avg_bw

        # Print results
        print(
            f"{a.capacity:<23d} {shared_benchmark_data[a.capacity]:<#16.4g} {register_benchmark_data[a.capacity]:<#18.4g} {memcpy_benchmark_data[a.capacity]:<#16.4g}"
        )
