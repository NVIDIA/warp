# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Tile FFT
#
# Shows how to write a simple FFT kernel using Warp tile primitives.
#
###########################################################################

import numpy as np

import warp as wp

wp.set_module_options({"enable_backward": False})

BLOCK_DIM = 8
TILE_M = 1
TILE_N = 32


@wp.kernel
def fft_tiled(x: wp.array2d(dtype=wp.vec2d), y: wp.array2d(dtype=wp.vec2d)):
    i, j, _ = wp.tid()
    a = wp.tile_load(x, i, j, m=TILE_M, n=TILE_N)
    wp.tile_fft(a)
    wp.tile_ifft(a)
    wp.tile_store(y, i, j, a)


if __name__ == "__main__":
    wp.set_device("cuda:0")

    x_h = np.ones((TILE_M, TILE_N, 2), dtype=np.float64)
    x_h[:, :, 1] = 0
    y_h = 3 * np.ones((TILE_M, TILE_N, 2), dtype=np.float64)
    x_wp = wp.array2d(x_h, dtype=wp.vec2d)
    y_wp = wp.array2d(y_h, dtype=wp.vec2d)

    wp.launch_tiled(fft_tiled, dim=[1, 1], inputs=[x_wp], outputs=[y_wp], block_dim=BLOCK_DIM)

    print("Inputs:\n", x_wp)  # [1+0i, 1+0i, 1+0i, ...]
    print("Output:\n", y_wp)  # [32+0i, 0, 0, ...]
