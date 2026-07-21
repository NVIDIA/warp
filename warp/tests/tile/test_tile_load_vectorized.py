# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests shared-memory tile load paths with 128-thread blocks.

Covers alignment promises, partial outer dimensions, vectorized element
types, N-dimensional vectorization, and coalesced copies of large element
types. These tests are forward-only. Add tests here when they target a
vectorized or coalesced shared-memory path.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.set_module_options({"enable_backward": False})

ALIGNED_TILE = wp.constant(32)


@wp.kernel
def aligned_param_2d_kernel(input: wp.array2d[float], output: wp.array2d[float]):
    """Shared tile load/store with aligned=True — skips runtime alignment checks."""
    i, j = wp.tid()
    tile = wp.tile_load(
        input,
        shape=(ALIGNED_TILE, ALIGNED_TILE),
        offset=(i * ALIGNED_TILE, j * ALIGNED_TILE),
        storage="shared",
        aligned=True,
    )
    wp.tile_store(output, tile, offset=(i * ALIGNED_TILE, j * ALIGNED_TILE), aligned=True)


def test_tile_load_aligned_param_2d(test, device):
    """aligned=True with properly aligned 2D shared tile produces correct results."""
    rng = np.random.default_rng(42)
    size = 128
    arr_np = rng.random((size, size)).astype(np.float32)
    input = wp.array(arr_np, dtype=float, device=device)
    output = wp.zeros((size, size), dtype=float, device=device)

    wp.launch_tiled(
        aligned_param_2d_kernel,
        dim=[size // ALIGNED_TILE, size // ALIGNED_TILE],
        inputs=[input, output],
        block_dim=128,
        device=device,
    )

    np.testing.assert_allclose(output.numpy(), arr_np, rtol=1e-5)


PARTIAL_TILE = wp.constant(32)
PARTIAL_ROWS = wp.constant(50)  # not a multiple of PARTIAL_TILE


@wp.kernel
def partial_tile_outer_dim_kernel(
    input: wp.array2d[float],
    output: wp.array2d[float],
):
    """Load tiles where the last tile's outer dim extends past the array.

    With array rows=50 and tile rows=32, tile at offset (32,0) covers rows 32-63
    but only 50 exist. The vectorized path must NOT be taken for this tile —
    it should fall to scalar with zero-padding for the OOB rows.
    """
    i = wp.tid()
    t = wp.tile_load(input, shape=(PARTIAL_TILE, PARTIAL_TILE), offset=(i * PARTIAL_TILE, 0), storage="shared")
    wp.tile_store(output, t, offset=(i * PARTIAL_TILE, 0))


def test_tile_load_partial_outer_dim(test, device):
    """Regression: float4-aligned tile where array outer dim is not a multiple of tile dim.

    On main, the partial tile incorrectly hit the vectorized path because the
    2D-only vectorization check did not verify that the tile fits within bounds.
    """
    rng = np.random.default_rng(42)
    arr_np = rng.random((PARTIAL_ROWS, PARTIAL_TILE), dtype=np.float32)
    input = wp.array(arr_np, dtype=float, device=device)
    # Output is padded to fit 2 full tiles (64 rows)
    output = wp.zeros((PARTIAL_TILE * 2, PARTIAL_TILE), dtype=float, device=device)

    wp.launch_tiled(
        partial_tile_outer_dim_kernel,
        dim=[2],  # 2 tiles: rows 0-31 (full) and rows 32-63 (partial, only 50-32=18 valid)
        inputs=[input, output],
        block_dim=128,
        device=device,
    )

    out = output.numpy()

    # First tile (rows 0-31): should match input exactly
    assert_np_equal(out[:PARTIAL_TILE, :], arr_np[:PARTIAL_TILE, :])
    # Second tile (rows 32-49): valid rows should match input
    assert_np_equal(out[PARTIAL_TILE:PARTIAL_ROWS, :], arr_np[PARTIAL_TILE:PARTIAL_ROWS, :])
    # Second tile (rows 50-63): OOB rows should be zero-padded
    assert_np_equal(out[PARTIAL_ROWS : PARTIAL_TILE * 2, :], np.zeros((PARTIAL_TILE * 2 - PARTIAL_ROWS, PARTIAL_TILE)))


VEC3_TILE_M = wp.constant(4)
VEC3_TILE_N = wp.constant(8)  # 8 vec3s * 12 bytes = 96 bytes, 96 % 16 = 0 → float4-aligned


@wp.kernel
def vec3_vectorized_kernel(
    input: wp.array2d[wp.vec3],
    output: wp.array2d[wp.vec3],
):
    """2D shared tile load/store roundtrip for vec3 with float4-aligned last dim."""
    i, j = wp.tid()
    t = wp.tile_load(
        input, shape=(VEC3_TILE_M, VEC3_TILE_N), offset=(i * VEC3_TILE_M, j * VEC3_TILE_N), storage="shared"
    )
    wp.tile_store(output, t, offset=(i * VEC3_TILE_M, j * VEC3_TILE_N))


def test_tile_load_vec3_vectorized(test, device):
    """Correctness check for 2D vec3 shared tiles with float4-aligned last dim."""
    rng = np.random.default_rng(42)
    rows, cols = 16, 16
    arr_np = rng.random((rows, cols, 3), dtype=np.float32)
    input = wp.array(arr_np, dtype=wp.vec3, device=device)
    output = wp.zeros((rows, cols), dtype=wp.vec3, device=device)

    wp.launch_tiled(
        vec3_vectorized_kernel,
        dim=[rows // VEC3_TILE_M, cols // VEC3_TILE_N],
        inputs=[input, output],
        block_dim=128,
        device=device,
    )

    assert_np_equal(output.numpy(), arr_np)


# 3D shared tile — exercises the N-D float4 vectorized indexing loop (N=3).
# Last dim: 8 floats * 4 bytes = 32 bytes, float4-aligned → vectorized path.
TILE_3D_D0 = wp.constant(4)
TILE_3D_D1 = wp.constant(4)
TILE_3D_D2 = wp.constant(8)


@wp.kernel
def shared_3d_vectorized_kernel(
    input: wp.array3d[float],
    output: wp.array3d[float],
):
    """3D shared tile load/store roundtrip — hits vectorized float4 path."""
    i, j, k = wp.tid()
    t = wp.tile_load(
        input,
        shape=(TILE_3D_D0, TILE_3D_D1, TILE_3D_D2),
        offset=(i * TILE_3D_D0, j * TILE_3D_D1, k * TILE_3D_D2),
        storage="shared",
    )
    wp.tile_store(output, t, offset=(i * TILE_3D_D0, j * TILE_3D_D1, k * TILE_3D_D2))


def test_tile_load_3d_shared_vectorized(test, device):
    """Correctness check for 3D shared tiles taking the vectorized float4 path."""
    rng = np.random.default_rng(42)
    shape = (8, 8, 16)
    arr_np = rng.random(shape, dtype=np.float32)
    input = wp.array(arr_np, dtype=float, device=device)
    output = wp.zeros(shape, dtype=float, device=device)

    wp.launch_tiled(
        shared_3d_vectorized_kernel,
        dim=[shape[0] // TILE_3D_D0, shape[1] // TILE_3D_D1, shape[2] // TILE_3D_D2],
        inputs=[input, output],
        block_dim=128,
        device=device,
    )

    assert_np_equal(output.numpy(), arr_np)


# Coalesced byte-copy path — exercises can_coalesce() and the float* byte-copy loop.
# mat33 is 36 bytes (> 16 bytes = sizeof(float4)), so it takes the coalesced path.
COALESCED_TILE = wp.constant(16)


@wp.kernel
def coalesced_mat33_kernel(
    input: wp.array1d[wp.mat33],
    output: wp.array1d[wp.mat33],
):
    """1D shared tile load/store for mat33 — hits coalesced byte-copy path."""
    i = wp.tid()
    t = wp.tile_load(input, shape=(COALESCED_TILE,), offset=(i * COALESCED_TILE,), storage="shared")
    wp.tile_store(output, t, offset=(i * COALESCED_TILE,))


def test_tile_load_coalesced_mat33(test, device):
    """Correctness check for shared tiles of large types (mat33) via coalesced byte-copy."""
    rng = np.random.default_rng(42)
    n = 64
    arr_np = rng.random((n, 3, 3), dtype=np.float32)
    input = wp.array(arr_np, dtype=wp.mat33, device=device)
    output = wp.zeros(n, dtype=wp.mat33, device=device)

    wp.launch_tiled(
        coalesced_mat33_kernel,
        dim=[n // COALESCED_TILE],
        inputs=[input, output],
        block_dim=128,
        device=device,
    )

    np.testing.assert_allclose(output.numpy(), arr_np, rtol=1e-5)


devices = get_test_devices()


class TestTileLoadVectorized(unittest.TestCase):
    pass


add_function_test(
    TestTileLoadVectorized,
    "test_tile_load_aligned_param_2d",
    test_tile_load_aligned_param_2d,
    devices=devices,
)
add_function_test(
    TestTileLoadVectorized,
    "test_tile_load_partial_outer_dim",
    test_tile_load_partial_outer_dim,
    devices=devices,
)
add_function_test(
    TestTileLoadVectorized,
    "test_tile_load_vec3_vectorized",
    test_tile_load_vec3_vectorized,
    devices=devices,
)
add_function_test(
    TestTileLoadVectorized,
    "test_tile_load_3d_shared_vectorized",
    test_tile_load_3d_shared_vectorized,
    devices=devices,
)
add_function_test(
    TestTileLoadVectorized,
    "test_tile_load_coalesced_mat33",
    test_tile_load_coalesced_mat33,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
