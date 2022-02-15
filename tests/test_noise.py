import unittest
import sys
import os
import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import test_base


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

    wp.store(noise_values, tid, g)

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

    wp.store(noise_coords, tid, p)
    wp.store(noise_vectors, tid, v)

def test_pnoise(test, device):
    # image dim
    W = 1024
    H = 1024
    N = W * H
    seed = 42

    # periodic perlin noise test
    px = 128
    py = 512

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
    img = img.astype(np.uint8)

    ### Figure viewing ###
    # imgplot = plt.imshow(img, 'gray')
    # plt.savefig("pnoise_test.png")

    ### Generating pnoise_test_result_true.npy ###
    # np.save(os.path.join(os.path.dirname(__file__), "assets/pnoise_test_result_true_NEW.npy"), img)

    ### Golden image comparison ###
    img_true = np.load(os.path.join(os.path.dirname(__file__), "assets/pnoise_test_result_true.npy"))
    test.assertTrue(img.shape == img_true.shape)
    err = np.max(np.abs(img - img_true))
    test.assertTrue(err < 1e-04)

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
    # np.save(os.path.join(os.path.dirname(__file__), "assets/curlnoise_test_result_true_NEW.npy"), result)

    ### Golden image comparison ###
    result_true = np.load(os.path.join(os.path.dirname(__file__), "assets/curlnoise_test_result_true.npy"))
    test.assertTrue(result.shape, result_true.shape)
    err = np.max(np.abs(result - result_true))
    test.assertTrue(err < 1e-04)

devices = wp.get_devices()

class TestNoise(test_base.TestBase):
    pass

TestNoise.add_function_test("test_pnoise", test_pnoise, devices=devices)
TestNoise.add_function_test("test_curlnoise", test_curlnoise, devices=devices)

if __name__ == '__main__':
    unittest.main(verbosity=2)