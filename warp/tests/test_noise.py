# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
from warp.tests.test_base import *

import numpy as np
#import matplotlib.pyplot as plt

wp.init()

@wp.kernel
def pnoise(
    kernel_seed: int,
    W: int,
    px: int,
    py: int,
    noise_values: wp.array(dtype=float)):

    tid = wp.tid()

    state = wp.rand_init(kernel_seed)

    x = (float(tid % W) + 0.5) * 0.02
    y = (float(tid / W) + 0.5) * 0.02
    p = wp.vec2(x, y)

    n = wp.pnoise(state, p, px, py)
    n = n + 1.0
    n = n / 2.0

    g = n * 255.0

    noise_values[tid] = g

@wp.kernel
def curlnoise(
    kernel_seed: int,
    W: int,
    noise_coords: wp.array(dtype=wp.vec2),
    noise_vectors: wp.array(dtype=wp.vec2)):

    tid = wp.tid()

    state = wp.rand_init(kernel_seed)

    x = (float(tid % W) + 0.5) * 0.2
    y = (float(tid / W) + 0.5) * 0.2

    p = wp.vec2(x, y)
    v = wp.curlnoise(state, p)

    noise_coords[tid] = p
    noise_vectors[tid] = v

def test_pnoise(test, device):
    # image dim
    W = 64
    H = 64
    N = W * H
    seed = 42

    # periodic perlin noise test
    px = 8
    py = 8

    pixels_host = wp.zeros(N, dtype=float, device="cpu")
    pixels = wp.zeros(N, dtype=float, device=device)

    wp.launch(
        kernel=pnoise,
        dim=N,
        inputs=[seed, W, px, py, pixels],
        outputs=[],
        device=device
    )

    wp.copy(pixels_host, pixels)
    wp.synchronize()

    img = pixels_host.numpy()
    img = np.reshape(img, (W, H))

    ### Figure viewing ###
    # img = img.astype(np.uint8)
    # imgplot = plt.imshow(img, 'gray')
    # plt.savefig("pnoise_test.png")

    ### Generating pnoise_test_result_true.npy ###
    # np.save(os.path.join(os.path.dirname(__file__), "assets/pnoise_golden.npy"), img)

    ### Golden image comparison ###
    img_true = np.load(os.path.join(os.path.dirname(__file__), "assets/pnoise_golden.npy"))
    test.assertTrue(img.shape == img_true.shape)
    err = np.max(np.abs(img - img_true))
    tolerance = 1.5e-4
    test.assertTrue(err < tolerance, f"err is {err} which is >= {tolerance}")

def test_curlnoise(test, device):
    # image dim
    W = 128
    H = 128
    N = W * H
    seed = 42

    # curl noise test
    quiver_coords_host = wp.zeros(N, dtype=wp.vec2, device="cpu")
    quiver_coords = wp.zeros(N, dtype=wp.vec2, device=device)

    quiver_arrows_host = wp.zeros(N, dtype=wp.vec2, device="cpu")
    quiver_arrows = wp.zeros(N, dtype=wp.vec2, device=device)

    wp.launch(
        kernel=curlnoise,
        dim=N,
        inputs=[seed, W, quiver_coords, quiver_arrows],
        outputs=[],
        device=device
    )

    wp.copy(quiver_coords_host, quiver_coords)
    wp.copy(quiver_arrows_host, quiver_arrows)

    wp.synchronize()

    xy_coords = quiver_coords_host.numpy()
    uv_coords = quiver_arrows_host.numpy()

    # normalize
    norms = uv_coords[:,0] * uv_coords[:,0] + uv_coords[:,1] * uv_coords[:,1]
    uv_coords = uv_coords / np.sqrt(np.max(norms))

    X = xy_coords[:,0]
    Y = xy_coords[:,1]
    U = uv_coords[:,0]
    V = uv_coords[:,1]

    ### Figure viewing ###
    # fig, ax = plt.subplots(figsize=(25,25))
    # ax.quiver(X, Y, U, V)
    # ax.axis([0.0, 25.0, 0.0, 25.0])
    # ax.set_aspect('equal')
    # plt.savefig("curlnoise_test.png")

    ### Generating curlnoise_test_result_true.npy ###
    result = np.stack((xy_coords, uv_coords))
    # np.save(os.path.join(os.path.dirname(__file__), "assets/curlnoise_golden.npy"), result)

    ### Golden image comparison ###
    result_true = np.load(os.path.join(os.path.dirname(__file__), "assets/curlnoise_golden.npy"))
    test.assertTrue(result.shape, result_true.shape)
    err = np.max(np.abs(result - result_true))
    test.assertTrue(err < 1e-04)

def register(parent):

    devices = wp.get_devices()

    class TestNoise(parent):
        pass

    add_function_test(TestNoise, "test_pnoise", test_pnoise, devices=devices)
    add_function_test(TestNoise, "test_curlnoise", test_curlnoise, devices=devices)

    return TestNoise


if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)