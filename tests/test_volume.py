import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import test_base

import warp as wp
import numpy as np

wp.init()


@wp.kernel
def test_volume_lookup(volume: wp.uint64,
                       points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]
    expected = p[0] * p[1] * p[2]
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        expected = 10.0

    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    expect_eq(wp.volume_lookup(volume, i, j, k), expected)


@wp.kernel
def test_volume_sample_closest(volume: wp.uint64,
                               points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]
    i = round(p[0])
    j = round(p[1])
    k = round(p[2])
    expected = i * j * k
    if abs(i) > 10.0 or abs(j) > 10.0 or abs(k) > 10.0:
        expected = 10.0

    expect_eq(wp.volume_sample_local(volume, p, wp.Volume.CLOSEST), expected)

    q = wp.volume_transform(volume, p)
    expect_eq(wp.volume_sample_world(volume, q, wp.Volume.CLOSEST), expected)

    q_inv = wp.volume_transform_inv(volume, q)
    expect_eq(p, q_inv)


@wp.kernel
def test_volume_sample_linear(volume: wp.uint64,
                              points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]

    expected = p[0] * p[1] * p[2]
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        return  # not testing against background values

    expect_near(wp.volume_sample_local(volume, p, wp.Volume.LINEAR), expected, 2.0e-4)

    q = wp.volume_transform(volume, p)
    expect_near(wp.volume_sample_world(volume, q, wp.Volume.LINEAR), expected, 2.0e-4)


class TestVolumes(test_base.TestBase):
    pass


devices = ["cpu", "cuda"]

volume_data = np.fromfile("tests/assets/test_grid.nvdbraw", dtype=np.byte)
volume_array = wp.array(volume_data, device="cpu")

volumes = {}
points = {}
points_jittered = {}
for device in devices:
    volumes[device] = wp.Volume(volume_array.to(device))

    data_np = volumes[device].array().numpy()
    magic = ''.join([chr(x) for x in data_np[0:8]])
    if magic != "NanoVDB0":
        print(f"FAILED: NanoVDB signature doesn't match!\nFound \"{magic}\"")
        sys.exit()

    axis = np.linspace(-11, 11, 23)
    points_np = np.array([[x, y, z] for x in axis for y in axis for z in axis])
    points_jittered_np = points_np + np.random.uniform(-0.5, 0.5, size=points_np.shape)
    points[device] = wp.array(points_np, dtype=wp.vec3, device=device)
    points_jittered[device] = wp.array(points_jittered_np, dtype=wp.vec3, device=device)

    TestVolumes.add_kernel_test(test_volume_lookup, dim=len(points_np), inputs=[volumes[device].id, points[device]], devices=[device])
    TestVolumes.add_kernel_test(test_volume_sample_closest, dim=len(points), inputs=[volumes[device].id, points_jittered[device]], devices=[device])
    TestVolumes.add_kernel_test(test_volume_sample_linear, dim=len(points), inputs=[volumes[device].id, points_jittered[device]], devices=[device])

if __name__ == '__main__':
    unittest.main(verbosity=2)
