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
    a = wp.tile_load(x, shape=(TILE_M, TILE_N))
    wp.tile_fft(a)
    b = wp.tile_map(filter, a)
    wp.tile_ifft(b)
    wp.tile_store(y, b)


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
