import numpy as np

import warp as wp

wp.init()
wp.set_module_options({"enable_backward": False})
wp.set_device("cuda:0")
wp.build.clear_kernel_cache()

BLOCK_DIM = 8
TILE_M = 1
TILE_N = 32


@wp.kernel
def fft_tiled(x: wp.array2d(dtype=wp.vec2d), y: wp.array2d(dtype=wp.vec2d)):
    i, j, _ = wp.tid()
    a = wp.tile_load(x, i, j, m=TILE_M, n=TILE_N)
    wp.tile_fft_dx(a)
    wp.tile_ifft_dx(a)
    wp.tile_store(y, i, j, a)


x_h = np.ones((TILE_M, TILE_N, 2), dtype=np.float64)
x_h[:, :, 1] = 0
y_h = 3 * np.ones((TILE_M, TILE_N, 2), dtype=np.float64)
x_wp = wp.array2d(x_h, dtype=wp.vec2d)
y_wp = wp.array2d(y_h, dtype=wp.vec2d)

wp.launch(fft_tiled, dim=[1, 1, BLOCK_DIM], inputs=[x_wp, y_wp], block_dim=BLOCK_DIM)

print("inputs:\n", x_wp)  # [1+0i, 1+0i, 1+0i, ...]
print("output:\n", y_wp)  # [32+0i, 0, 0, ...]
