# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def test_kernel(
    kernel_seed: int,
    int_a: wp.array(dtype=int),
    int_ab: wp.array(dtype=int),
    uint_a: wp.array(dtype=wp.uint32),
    uint_ab: wp.array(dtype=wp.uint32),
    float_01: wp.array(dtype=float),
    float_ab: wp.array(dtype=float),
):
    tid = wp.tid()

    state = wp.rand_init(kernel_seed, tid)

    int_a[tid] = wp.randi(state)
    int_ab[tid] = wp.randi(state, 0, 100)
    uint_a[tid] = wp.randu(state)
    uint_ab[tid] = wp.randu(state, wp.uint32(0), wp.uint32(100))
    float_01[tid] = wp.randf(state)
    float_ab[tid] = wp.randf(state, 0.0, 100.0)


def test_rand(test, device):
    N = 10

    int_a = wp.zeros(N, dtype=int, device=device)
    int_ab = wp.zeros(N, dtype=int, device=device)

    uint_a = wp.zeros(N, dtype=wp.uint32, device=device)
    uint_ab = wp.zeros(N, dtype=wp.uint32, device=device)

    float_01 = wp.zeros(N, dtype=float, device=device)
    float_ab = wp.zeros(N, dtype=float, device=device)

    seed = 42

    wp.launch(
        kernel=test_kernel,
        dim=N,
        inputs=[seed, int_a, int_ab, uint_a, uint_ab, float_01, float_ab],
        outputs=[],
        device=device,
    )

    int_a_true = np.array(
        [
            -575632308,
            59537738,
            1898992239,
            442961864,
            -1069147335,
            -478445524,
            1803659809,
            2122909397,
            -1888556360,
            334603718,
        ]
    )
    int_ab_true = np.array([46, 58, 46, 83, 85, 39, 72, 99, 18, 41])
    uint_a_true = np.array(
        [
            3133687854,
            3702303309,
            1235698096,
            3516599792,
            800302729,
            2620462179,
            2423739693,
            3024873594,
            2783682377,
            1188846332,
        ]
    )
    uint_ab_true = np.array([6, 55, 2, 92, 55, 93, 65, 23, 48, 0])
    float_01_true = np.array(
        [
            0.8265858,
            0.5874614,
            0.1508659,
            0.9498008,
            0.02531803,
            0.8520948,
            0.0001185536,
            0.4855958,
            0.06277305,
            0.2214079,
        ]
    )
    float_ab_true = np.array(
        [79.84678, 76.362206, 32.135242, 99.70866, 70.45863, 20.6523, 45.164482, 55.583008, 76.60291, 35.36277]
    )

    assert_np_equal(int_a.numpy(), int_a_true)
    assert_np_equal(int_ab.numpy(), int_ab_true)

    assert_np_equal(uint_a.numpy(), uint_a_true)
    assert_np_equal(uint_ab.numpy(), uint_ab_true)

    assert_np_equal(float_01.numpy(), float_01_true, 1e-04)
    assert_np_equal(float_ab.numpy(), float_ab_true, 1e-04)


@wp.kernel
def randn_kernel(
    x: wp.array(dtype=float),
):
    tid = wp.tid()
    r = wp.rand_init(tid)
    x[tid] = wp.randn(r)


def test_randn(test, device):
    N = 100000000

    samples = wp.zeros(N, dtype=float, device=device)

    wp.launch(randn_kernel, inputs=[samples], dim=N, device=device)

    randn_samples = samples.numpy()

    test.assertFalse(np.any(np.isinf(randn_samples)))
    test.assertFalse(np.any(np.isnan(randn_samples)))

    randn_true = np.array(
        [
            -1.8213255,
            0.27881497,
            -1.1122388,
            0.5936895,
            0.04976363,
            0.69087356,
            0.2935363,
            0.8405019,
            -0.8436684,
            0.53108305,
        ]
    )

    err = np.max(np.abs(randn_samples[0:10] - randn_true))
    test.assertTrue(err < 1e-04)


@wp.kernel
def sample_cdf_kernel(kernel_seed: int, cdf: wp.array(dtype=float), samples: wp.array(dtype=int)):
    tid = wp.tid()
    state = wp.rand_init(kernel_seed, tid)

    samples[tid] = wp.sample_cdf(state, cdf)


def test_sample_cdf(test, device):
    seed = 42
    cdf = np.arange(0.0, 1.0, 0.01, dtype=float)
    cdf = cdf * cdf
    cdf = wp.array(cdf, dtype=float, device=device)
    num_samples = 1000
    samples = wp.zeros(num_samples, dtype=int, device=device)

    wp.launch(kernel=sample_cdf_kernel, dim=num_samples, inputs=[seed, cdf, samples], device=device)

    # histogram should be linear
    # plt.hist(samples.numpy())
    # plt.show()


@wp.kernel
def sampling_kernel(
    kernel_seed: int,
    triangle_samples: wp.array(dtype=wp.vec2),
    square_samples: wp.array(dtype=wp.vec2),
    ring_samples: wp.array(dtype=wp.vec2),
    disk_samples: wp.array(dtype=wp.vec2),
    sphere_surface_samples: wp.array(dtype=wp.vec3),
    sphere_samples: wp.array(dtype=wp.vec3),
    hemisphere_surface_samples: wp.array(dtype=wp.vec3),
    hemisphere_samples: wp.array(dtype=wp.vec3),
    cube_samples: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    state = wp.rand_init(kernel_seed, tid)

    triangle_samples[tid] = wp.sample_triangle(state)
    ring_samples[tid] = wp.sample_unit_ring(state)
    disk_samples[tid] = wp.sample_unit_disk(state)
    sphere_surface_samples[tid] = wp.sample_unit_sphere_surface(state)
    sphere_samples[tid] = wp.sample_unit_sphere(state)
    hemisphere_surface_samples[tid] = wp.sample_unit_hemisphere_surface(state)
    hemisphere_samples[tid] = wp.sample_unit_hemisphere(state)
    square_samples[tid] = wp.sample_unit_square(state)
    cube_samples[tid] = wp.sample_unit_cube(state)


def test_sampling_methods(test, device):
    seed = 42
    num_samples = 100

    triangle_samples = wp.zeros(num_samples, dtype=wp.vec2, device=device)
    square_samples = wp.zeros(num_samples, dtype=wp.vec2, device=device)
    ring_samples = wp.zeros(num_samples, dtype=wp.vec2, device=device)
    disk_samples = wp.zeros(num_samples, dtype=wp.vec2, device=device)
    sphere_surface_samples = wp.zeros(num_samples, dtype=wp.vec3, device=device)
    sphere_samples = wp.zeros(num_samples, dtype=wp.vec3, device=device)
    hemisphere_surface_samples = wp.zeros(num_samples, dtype=wp.vec3, device=device)
    hemisphere_samples = wp.zeros(num_samples, dtype=wp.vec3, device=device)
    cube_samples = wp.zeros(num_samples, dtype=wp.vec3, device=device)

    wp.launch(
        kernel=sampling_kernel,
        dim=num_samples,
        inputs=[
            seed,
            triangle_samples,
            square_samples,
            ring_samples,
            disk_samples,
            sphere_surface_samples,
            sphere_samples,
            hemisphere_surface_samples,
            hemisphere_samples,
            cube_samples,
        ],
        device=device,
    )

    # bounds check
    test.assertTrue((triangle_samples.numpy()[:, 0] <= 1.0).all())
    test.assertTrue((triangle_samples.numpy()[:, 0] >= 0.0).all())
    test.assertTrue((triangle_samples.numpy()[:, 1] >= 0.0).all())
    test.assertTrue((triangle_samples.numpy()[:, 1] >= 0.0).all())
    test.assertTrue((square_samples.numpy()[:, 0] >= -0.5).all())
    test.assertTrue((square_samples.numpy()[:, 0] <= 1.5).all())
    test.assertTrue((square_samples.numpy()[:, 1] >= -0.5).all())
    test.assertTrue((square_samples.numpy()[:, 1] <= 0.5).all())
    test.assertTrue((cube_samples.numpy()[:, 0] >= -0.5).all())
    test.assertTrue((cube_samples.numpy()[:, 0] <= 0.5).all())
    test.assertTrue((cube_samples.numpy()[:, 1] >= -0.5).all())
    test.assertTrue((cube_samples.numpy()[:, 1] <= 0.5).all())
    test.assertTrue((cube_samples.numpy()[:, 2] >= -0.5).all())
    test.assertTrue((cube_samples.numpy()[:, 2] <= 0.5).all())
    test.assertTrue((hemisphere_surface_samples.numpy()[:, 2] >= 0.0).all())
    test.assertTrue((hemisphere_samples.numpy()[:, 2] >= 0.0).all())
    test.assertTrue((np.linalg.norm(ring_samples.numpy(), axis=1) <= 1.0 + 1e6).all())
    test.assertTrue((np.linalg.norm(disk_samples.numpy(), axis=1) <= 1.0 + 1e6).all())
    test.assertTrue((np.linalg.norm(sphere_surface_samples.numpy(), axis=1) <= 1.0 + 1e6).all())
    test.assertTrue((np.linalg.norm(sphere_samples.numpy(), axis=1) <= 1.0 + 1e6).all())
    test.assertTrue((np.linalg.norm(hemisphere_surface_samples.numpy(), axis=1) <= 1.0 + 1e6).all())
    test.assertTrue((np.linalg.norm(hemisphere_samples.numpy(), axis=1) <= 1.0 + 1e6).all())


@wp.kernel
def sample_poisson_kernel(
    kernel_seed: int, poisson_samples_low: wp.array(dtype=wp.uint32), poisson_samples_high: wp.array(dtype=wp.uint32)
):
    tid = wp.tid()
    state = wp.rand_init(kernel_seed, tid)

    x = wp.poisson(state, 3.0)
    y = wp.poisson(state, 42.0)

    poisson_samples_low[tid] = x
    poisson_samples_high[tid] = y


def test_poisson(test, device):
    seed = 13
    N = 20000
    poisson_low = wp.zeros(N, dtype=wp.uint32, device=device)
    poisson_high = wp.zeros(N, dtype=wp.uint32, device=device)

    wp.launch(kernel=sample_poisson_kernel, dim=N, inputs=[seed, poisson_low, poisson_high], device=device)

    # bins = np.arange(100)
    # _ = plt.hist(poisson_high.numpy(), bins)
    # plt.show()

    rng = np.random.default_rng(seed)

    np_poisson_low = rng.poisson(lam=3.0, size=N)
    np_poisson_high = rng.poisson(lam=42.0, size=N)

    poisson_low_mean = np.mean(poisson_low.numpy())
    np_poisson_low_mean = np.mean(np_poisson_low)

    poisson_high_mean = np.mean(poisson_high.numpy())
    np_poisson_high_mean = np.mean(np_poisson_high)

    poisson_low_std = np.std(poisson_low.numpy())
    np_poisson_low_std = np.std(np_poisson_low)

    poisson_high_std = np.std(poisson_high.numpy())
    np_poisson_high_std = np.std(np_poisson_high)

    # compare basic distribution characteristics
    test.assertTrue(np.abs(poisson_low_mean - np_poisson_low_mean) <= 5e-1)
    test.assertTrue(np.abs(poisson_high_mean - np_poisson_high_mean) <= 5e-1)
    test.assertTrue(np.abs(poisson_low_std - np_poisson_low_std) <= 2e-1)
    test.assertTrue(np.abs(poisson_high_std - np_poisson_high_std) <= 2e-1)


devices = get_test_devices()


class TestRand(unittest.TestCase):
    pass


add_function_test(TestRand, "test_rand", test_rand, devices=devices)
add_function_test(TestRand, "test_randn", test_randn, devices=devices)
add_function_test(TestRand, "test_sample_cdf", test_sample_cdf, devices=devices)
add_function_test(TestRand, "test_sampling_methods", test_sampling_methods, devices=devices)
add_function_test(TestRand, "test_poisson", test_poisson, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
