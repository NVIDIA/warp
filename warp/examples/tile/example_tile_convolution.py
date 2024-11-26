# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Tile Convolution
#
# Shows how to write a simple convolution kernel using Warp FFT tile
# primitives.
#
###########################################################################

import numpy as np

import warp as wp

wp.set_module_options({"enable_backward": False})

BLOCK_DIM = 64
TILE_M = 1
TILE_N = 128

scale = wp.vec2d(wp.float64(1 / TILE_N), wp.float64(1 / TILE_N))


@wp.func
def filter(x: wp.vec2d):
    return wp.cw_mul(x, scale)


@wp.kernel
def conv_tiled(x: wp.array2d(dtype=wp.vec2d), y: wp.array2d(dtype=wp.vec2d)):
    i, j, _ = wp.tid()
    a = wp.tile_load(x, i, j, m=TILE_M, n=TILE_N)
    wp.tile_fft(a)
    b = wp.tile_map(filter, a)
    wp.tile_ifft(b)
    wp.tile_store(y, i, j, b)


if __name__ == "__main__":
    wp.set_device("cuda:0")

    rng = np.random.default_rng(42)

    x_h = rng.standard_normal((TILE_M, TILE_N, 2), dtype=np.float64)
    y_h = np.zeros_like(x_h)

    x_wp = wp.array2d(x_h, dtype=wp.vec2d)
    y_wp = wp.array2d(y_h, dtype=wp.vec2d)

    wp.launch_tiled(conv_tiled, dim=[1, 1], inputs=[x_wp], outputs=[y_wp], block_dim=BLOCK_DIM)

    # Since filter is 1/N, conv_tiled is a ~no-op
    assert np.allclose(x_h, y_wp.numpy())
