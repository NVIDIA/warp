# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
from warp.tests.test_base import *

import numpy as np
# import matplotlib.pyplot as plt

wp.init()


@wp.kernel
def pnoise(
    kernel_seed: int,
    W: int,
    px: int,
    py: int,
    noise_values: wp.array(dtype=float),
    pixel_values: wp.array(dtype=float)):

    tid = wp.tid()

    state = wp.rand_init(kernel_seed)

    x = (float(tid % W) + 0.5) * 0.2
    y = (float(tid / W) + 0.5) * 0.2
    p = wp.vec2(x, y)

    n = wp.pnoise(state, p, px, py)
    noise_values[tid] = n

    g = ((n + 1.0) / 2.0) * 255.0
    pixel_values[tid] = g


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
    W = 256
    H = 256
    N = W * H
    seed = 42

    # periodic perlin noise test
    px = 16
    py = 16

    noise_values = wp.zeros(N, dtype=float, device=device)
    pixel_values = wp.zeros(N, dtype=float, device=device)

    wp.launch(
        kernel=pnoise,
        dim=N,
        inputs=[seed, W, px, py, noise_values, pixel_values],
        outputs=[],
        device=device
    )

    # Perlin theoretical range is [-0.5*sqrt(n), 0.5*sqrt(n)] for n dimensions
    n = noise_values.numpy()
    # max = np.max(n)
    # min = np.min(n)

    img = pixel_values.numpy()
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
    tolerance = 1.5e-3
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


@wp.kernel
def noise_loss_kernel(
    kernel_seed: int,
    query_positions: wp.array(dtype=wp.vec2),
    noise_values: wp.array(dtype=float),
    noise_loss: wp.array(dtype=float)):

    tid = wp.tid()
    state = wp.rand_init(kernel_seed)

    p = query_positions[tid]

    n = wp.noise(state, p)
    noise_values[tid] = n

    wp.atomic_add(noise_loss, 0, n)


@wp.kernel
def noise_cd(
    kernel_seed: int,
    query_positions: wp.array(dtype=wp.vec2),
    gradients: wp.array(dtype=wp.vec2)):
    
    tid = wp.tid()
    state = wp.rand_init(kernel_seed)
    p = query_positions[tid]

    eps = 1.e-3

    pl = wp.vec2(p[0] - eps, p[1])
    pr = wp.vec2(p[0] + eps, p[1])
    pd = wp.vec2(p[0], p[1] - eps)
    pu = wp.vec2(p[0], p[1] + eps)

    nl = wp.noise(state, pl)
    nr = wp.noise(state, pr)
    nd = wp.noise(state, pd)
    nu = wp.noise(state, pu)

    gx = (nr - nl) / (2.0*eps)
    gy = (nu - nd) / (2.0*eps)

    gradients[tid] = wp.vec2(gx, gy)


def test_adj_noise(test, device):
    # grid dim
    N = 9
    seed = 42

    tape = wp.Tape()

    positions = np.array(
        [
            [-0.1, -0.1], [0.0, -0.1], [0.1, -0.1],
            [-0.1,  0.0], [0.0,  0.0], [0.1,  0.0],
            [-0.1,  0.1], [0.0,  0.1], [0.1,  0.1]
        ]
    )

    with tape:

        query_positions = wp.array(positions, dtype=wp.vec2, device=device, requires_grad=True)
        noise_values = wp.zeros(N, dtype=float, device=device)
        noise_loss = wp.zeros(n=1, dtype=float, device=device, requires_grad=True)

        wp.launch(kernel=noise_loss_kernel, dim=N, inputs=[seed, query_positions, noise_values, noise_loss], device=device)

    # analytic
    tape.backward(loss=noise_loss)
    analytic = tape.gradients[query_positions].numpy().reshape((3,3,2))

    # central difference
    gradients = wp.zeros(N, dtype=wp.vec2, device=device)
    wp.launch(kernel=noise_cd, dim=N, inputs=[seed, query_positions, gradients], device=device)

    gradients_host = gradients.numpy().reshape((3,3,2))
    diff = analytic - gradients_host
    result = np.sum(diff * diff, axis=2)

    err = np.where(result > 1.e-3, result, 0).sum()
    test.assertTrue(err < 1.e-8)


def register(parent):

    devices = wp.get_devices()

    class TestNoise(parent):
        pass

    add_function_test(TestNoise, "test_pnoise", test_pnoise, devices=devices)
    add_function_test(TestNoise, "test_curlnoise", test_curlnoise, devices=devices)
    add_function_test(TestNoise, "test_adj_noise", test_adj_noise, devices=devices)

    return TestNoise


if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)