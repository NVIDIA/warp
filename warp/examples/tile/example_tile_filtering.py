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
# Example Tile Filtering
#
# Shows how to write a simple filtering kernel using Warp FFT tile
# primitives.
#
###########################################################################

import numpy as np

import warp as wp

wp.set_module_options({"enable_backward": False})

BLOCK_DIM = 128
TILE_M = 1
TILE_N = 512

scale = wp.vec2d(wp.float64(1 / TILE_N), wp.float64(1 / TILE_N))


def cplx(array):
    return array[..., 0] + 1j * array[..., 1]


@wp.func
def cplx_prod(x: wp.vec2d, y: wp.vec2d):
    return wp.cw_mul(wp.vec2d(x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]), scale)


@wp.kernel
def conv_tiled(x: wp.array2d(dtype=wp.vec2d), y: wp.array2d(dtype=wp.vec2d), z: wp.array2d(dtype=wp.vec2d)):
    i, j, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N))
    b = wp.tile_load(y, shape=(TILE_M, TILE_N))
    wp.tile_fft(a)
    c = wp.tile_map(cplx_prod, a, b)
    wp.tile_ifft(c)
    wp.tile_store(z, c)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Create noisy input signal
    t = np.linspace(0, 2 * np.pi, TILE_N, dtype=np.float64)
    x = np.sin(t) + 0.5 * rng.random(TILE_N, dtype=np.float64)

    # Create filter. This filter keeps only ~10% of the frequencies at the center
    # of the spectrum.
    f = np.ones_like(x)
    freq = np.fft.fftfreq(TILE_N)
    f[np.abs(freq) > 0.05] = 0.0
    f[np.abs(freq) <= 0.05] = 1.0

    # Create Warp input data
    # We use vec2d to hold complex numbers
    x_h = np.zeros((TILE_M, TILE_N, 2), dtype=np.float64)
    f_h = np.zeros_like(x_h)
    y_h = np.zeros_like(f_h)

    x_h[:, :, 0] = x
    f_h[:, :, 0] = f

    x_wp = wp.array2d(x_h, dtype=wp.vec2d)
    f_wp = wp.array2d(f_h, dtype=wp.vec2d)
    y_wp = wp.array2d(y_h, dtype=wp.vec2d)

    wp.launch_tiled(conv_tiled, dim=[1, 1], inputs=[x_wp, f_wp], outputs=[y_wp], block_dim=BLOCK_DIM)

    # Extract output and compare with numpy
    x_np = cplx(x_h)
    f_np = cplx(f_h)
    y_test = cplx(y_wp.numpy())
    y_ref = np.fft.ifft(f_np * np.fft.fft(x_np))
    assert np.allclose(y_ref, y_test)

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x,
        color="#DDDDDD",
        linewidth=2,
        label="Original",
    )
    ax.plot(y_test[0, :].real, color="#76B900", linewidth=3, label="Smoothed")

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

except ModuleNotFoundError:
    print("Matplotlib not available; skipping figure")
