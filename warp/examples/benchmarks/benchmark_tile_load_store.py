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

"""Benchmark tile load/store throughput against a device memcpy baseline.

Measures the effective memory bandwidth (GiB/s) of ``wp.tile_load`` /
``wp.tile_store`` round-trips for 1D, 2D, or 3D arrays, sweeping over a
range of array sizes. Results are compared against ``wp.copy`` (device memcpy)
to show how close tile operations get to peak device bandwidth.

Usage::

    python benchmark_tile_load_store.py
    python benchmark_tile_load_store.py --ndim 1 --dtype float16
    python benchmark_tile_load_store.py --storage shared --iterations 200
"""

import argparse

import numpy as np

import warp as wp

DTYPE_MAP = {
    "float16": wp.float16,
    "float32": wp.float32,
    "float64": wp.float64,
    "vec3": wp.vec3,
}


def _numpy_scalar_dtype(wp_dtype):
    """Return the numpy scalar dtype for a Warp type.

    For scalar types (float32, float64, etc.) returns the direct numpy equivalent.
    For structured types (vec3, mat33, etc.) returns the numpy dtype of the
    underlying scalar component (e.g. np.float32 for vec3f).
    """
    scalar_type = getattr(wp_dtype, "_wp_scalar_type_", wp_dtype)
    return wp.dtype_to_numpy(scalar_type)


def create_kernel_1d(tile_size, dtype, storage_type):
    TILE = tile_size

    @wp.kernel
    def load_store_1d(a: wp.array(dtype=dtype), b: wp.array(dtype=dtype)):
        i = wp.tid()
        if wp.static(storage_type == "shared"):
            t = wp.tile_load(a, shape=TILE, offset=i * TILE, storage="shared")
        else:
            t = wp.tile_load(a, shape=TILE, offset=i * TILE, storage="register")
        wp.tile_store(b, t, offset=i * TILE)

    return load_store_1d


def create_kernel_2d(tile_size, dtype, storage_type):
    TILE = tile_size

    @wp.kernel
    def load_store_2d(a: wp.array2d(dtype=dtype), b: wp.array2d(dtype=dtype)):
        i, j = wp.tid()
        if wp.static(storage_type == "shared"):
            t = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="shared")
        else:
            t = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="register")
        wp.tile_store(b, t, offset=(i * TILE, j * TILE))

    return load_store_2d


def create_kernel_3d(tile_size, dtype, storage_type):
    TILE = tile_size

    @wp.kernel
    def load_store_3d(a: wp.array3d(dtype=dtype), b: wp.array3d(dtype=dtype)):
        i, j, k = wp.tid()
        if wp.static(storage_type == "shared"):
            t = wp.tile_load(a, shape=(TILE, TILE, TILE), offset=(i * TILE, j * TILE, k * TILE), storage="shared")
        else:
            t = wp.tile_load(a, shape=(TILE, TILE, TILE), offset=(i * TILE, j * TILE, k * TILE), storage="register")
        wp.tile_store(b, t, offset=(i * TILE, j * TILE, k * TILE))

    return load_store_3d


KERNEL_CREATORS = {1: create_kernel_1d, 2: create_kernel_2d, 3: create_kernel_3d}


def _round_to_tile(values, tile_size):
    """Round values down to multiples of tile_size, dropping any that are too small."""
    return [n // tile_size * tile_size for n in values if n >= tile_size]


def generate_sizes(ndim, tile_size):
    """Generate array edge lengths for the size sweep, based on dimensionality."""
    if ndim == 1:
        raw = [2**p for p in range(16, 27)]
    elif ndim == 2:
        raw = list(range(128, 4097, 128))
    else:
        raw = list(range(16, 257, 16))
    return _round_to_tile(raw, tile_size)


def make_array(rng, ndim, size, wp_dtype):
    """Create a warp array filled with random data."""
    np_dtype = _numpy_scalar_dtype(wp_dtype)
    spatial = (size,) * ndim

    # Structured types (vec, mat) need trailing component dimensions in the numpy array
    component_shape = getattr(wp_dtype, "_shape_", ())
    np_shape = spatial + component_shape

    # Generator.random() only supports float32/float64; generate float32 and cast if needed
    gen_dtype = np_dtype if np_dtype in (np.float32, np.float64) else np.float32
    data = rng.random(np_shape, dtype=gen_dtype).astype(np_dtype)
    return wp.array(data, dtype=wp_dtype)


def compute_launch_dim(ndim, size, tile_size):
    """Compute the launch grid dimensions."""
    tiles_per_dim = size // tile_size
    # 1D launch requires a scalar, not a 1-tuple
    if ndim == 1:
        return tiles_per_dim
    return tuple(tiles_per_dim for _ in range(ndim))


def _bandwidth_gibs(capacity_bytes, timing_results_ms):
    """Compute bidirectional bandwidth in GiB/s from capacity and timing results."""
    return 2.0 * (capacity_bytes / (1024 * 1024 * 1024)) / (1e-3 * np.median(timing_results_ms))


def run_benchmark(ndim, tile_size, dtype_name, block_dim, storage_modes, iterations):
    wp_dtype = DTYPE_MAP[dtype_name]
    create_kernel = KERNEL_CREATORS[ndim]
    sizes = generate_sizes(ndim, tile_size)

    if not sizes:
        print("No valid sizes for the given tile size.")
        return

    columns = [("N", 10), ("Transfer Size (Bytes)", 23)]
    columns += [(f"{mode.capitalize()} (GiB/s)", 18) for mode in storage_modes]
    columns.append(("memcpy (GiB/s)", 16))

    header = "".join(f"{name:<{width}s}" for name, width in columns)
    print(header)
    print("-" * len(header))

    rng = np.random.default_rng(42)

    for size in sizes:
        a = make_array(rng, ndim, size, wp_dtype)
        b = wp.empty_like(a)

        bw_results = {}

        for storage_type in storage_modes:
            kernel = create_kernel(tile_size, wp_dtype, storage_type)
            dim = compute_launch_dim(ndim, size, tile_size)

            cmd = wp.launch_tiled(
                kernel,
                dim=dim,
                inputs=[a],
                outputs=[b],
                block_dim=block_dim,
                record_cmd=True,
            )
            for _ in range(5):
                cmd.launch()

            np.testing.assert_equal(a.numpy(), b.numpy())

            with wp.ScopedTimer("benchmark", cuda_filter=wp.TIMING_KERNEL, print=False, synchronize=True) as timer:
                for _ in range(iterations):
                    cmd.launch()

            timing_results = [result.elapsed for result in timer.timing_results]
            avg_bw = _bandwidth_gibs(a.capacity, timing_results)
            bw_results[storage_type] = avg_bw

        # memcpy baseline
        with wp.ScopedTimer("benchmark", cuda_filter=wp.TIMING_MEMCPY, print=False, synchronize=True) as timer:
            for _ in range(iterations):
                wp.copy(b, a)

        timing_results = [result.elapsed for result in timer.timing_results]
        memcpy_bw = _bandwidth_gibs(a.capacity, timing_results)

        row = f"{size:<10d}{a.capacity:<23d}"
        for mode in storage_modes:
            row += f" {bw_results[mode]:<#17.4g}"
        row += f" {memcpy_bw:<#15.4g}"
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tile load/store throughput")
    parser.add_argument("--ndim", type=int, choices=[1, 2, 3], default=2, help="Tile dimensionality (default: 2)")
    parser.add_argument("--tile-size", type=int, default=32, help="Tile edge size (default: 32)")
    parser.add_argument(
        "--dtype", choices=list(DTYPE_MAP.keys()), default="float32", help="Element type (default: float32)"
    )
    parser.add_argument("--block-dim", type=int, default=128, help="Block dimension (default: 128)")
    parser.add_argument(
        "--storage", choices=["shared", "register", "both"], default="both", help="Storage mode (default: both)"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of timed iterations per size (default: 100)"
    )
    args = parser.parse_args()

    wp.config.quiet = True
    wp.init()
    wp.set_module_options({"fast_math": True, "enable_backward": False})

    if not wp.is_cuda_available():
        print("Error: This benchmark requires a CUDA device.")
        raise SystemExit(1)

    storage_modes = ["shared", "register"] if args.storage == "both" else [args.storage]

    print(f"Device: {wp.get_cuda_device().name}")
    print(
        f"Config: ndim={args.ndim}, tile_size={args.tile_size}, dtype={args.dtype}, "
        f"block_dim={args.block_dim}, storage={args.storage}, iterations={args.iterations}"
    )
    print()

    run_benchmark(
        ndim=args.ndim,
        tile_size=args.tile_size,
        dtype_name=args.dtype,
        block_dim=args.block_dim,
        storage_modes=storage_modes,
        iterations=args.iterations,
    )
