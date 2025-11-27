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
    a = wp.tile_load(x, shape=(TILE_M, TILE_N))
    wp.tile_fft(a)
    wp.tile_ifft(a)
    wp.tile_store(y, a)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing output.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice("cuda:0"):
        x_h = np.ones((TILE_M, TILE_N, 2), dtype=np.float64)
        x_h[:, :, 1] = 0
        y_h = 3 * np.ones((TILE_M, TILE_N, 2), dtype=np.float64)
        x_wp = wp.array2d(x_h, dtype=wp.vec2d)
        y_wp = wp.array2d(y_h, dtype=wp.vec2d)

        wp.launch_tiled(fft_tiled, dim=[1, 1], inputs=[x_wp], outputs=[y_wp], block_dim=BLOCK_DIM)

        expected = np.zeros((TILE_M, TILE_N, 2), dtype=np.float64)
        expected[0, :, 0] = 32.0

        np.testing.assert_allclose(y_wp.numpy(), expected, atol=1.0e-4)

        if not args.headless:
            print("Inputs:\n", x_wp)  # [1+0i, 1+0i, 1+0i, ...]
            print("Output:\n", y_wp)  # [32+0i, 0, 0, ...]
