# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import warp as wp
from warp.tests.test_base import *

wp.init()

@wp.kernel
def test_kernel(
    kernel_seed: int,
    int_a: wp.array(dtype=int),
    int_ab: wp.array(dtype=int),
    float_01: wp.array(dtype=float),
    float_ab: wp.array(dtype=float)):

    tid = wp.tid()

    state = wp.rand_init(kernel_seed, tid)

    int_a[tid] = wp.randi(state)
    int_ab[tid] = wp.randi(state, 0, 100)
    float_01[tid] = wp.randf(state)
    float_ab[tid] = wp.randf(state, 0.0, 100.0)

def test_rand(test, device):

    N = 10

    int_a_device = wp.zeros(N, dtype=int, device=device)
    int_a_host = wp.zeros(N, dtype=int, device="cpu")
    int_ab_device = wp.zeros(N, dtype=int, device=device)
    int_ab_host = wp.zeros(N, dtype=int, device="cpu")

    float_01_device = wp.zeros(N, dtype=float, device=device)
    float_01_host = wp.zeros(N, dtype=float, device="cpu")
    float_ab_device = wp.zeros(N, dtype=float, device=device)
    float_ab_host = wp.zeros(N, dtype=float, device="cpu")

    seed = 42

    wp.launch(
        kernel=test_kernel,
        dim=N,
        inputs=[seed, int_a_device, int_ab_device, float_01_device, float_ab_device],
        outputs=[],
        device=device
    )

    wp.copy(int_a_host, int_a_device)
    wp.copy(int_ab_host, int_ab_device)
    wp.copy(float_01_host, float_01_device)
    wp.copy(float_ab_host, float_ab_device)
    wp.synchronize()

    int_a = int_a_host.numpy()
    int_ab = int_ab_host.numpy()
    float_01 = float_01_host.numpy()
    float_ab = float_ab_host.numpy()

    int_a_true = np.array([-575632308, 59537738, 1898992239, 442961864, -1069147335, -478445524, 1803659809, 2122909397, -1888556360, 334603718])
    int_ab_true = np.array([46, 58, 46, 83, 85, 39, 72, 99, 18, 41])
    float_01_true = np.array([0.72961855, 0.86200964, 0.28770837, 0.8187722, 0.186335, 0.6101239, 0.56432086, 0.70428324, 0.64812654, 0.27679986])
    float_ab_true = np.array([96.04259, 73.33809, 63.601555, 38.647305, 71.813896, 64.65809, 77.79791, 46.579605, 94.614456, 91.921814])

    test.assertTrue((int_a == int_a_true).all())
    test.assertTrue((int_ab == int_ab_true).all())

    err = np.max(np.abs(float_01 - float_01_true))
    test.assertTrue(err < 1e-04)

    err = np.max(np.abs(float_ab - float_ab_true))
    test.assertTrue(err < 1e-04)

def register(parent):

    devices = wp.get_devices()

    class TestNoise(parent):
        pass

    add_function_test(TestNoise, "test_rand", test_rand, devices=devices)

    return TestNoise

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)