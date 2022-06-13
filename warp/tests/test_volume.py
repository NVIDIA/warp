# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
from warp.tests.test_base import *

import numpy as np

wp.init()


# float volume tests
@wp.kernel
def test_volume_lookup_f(volume: wp.uint64,
                         points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]
    expected = p[0] * p[1] * p[2]
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        expected = 10.0

    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    expect_eq(wp.volume_lookup_f(volume, i, j, k), expected)


@wp.kernel
def test_volume_sample_closest_f(volume: wp.uint64,
                                 points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]
    i = round(p[0])
    j = round(p[1])
    k = round(p[2])
    expected = i * j * k
    if abs(i) > 10.0 or abs(j) > 10.0 or abs(k) > 10.0:
        expected = 10.0

    expect_eq(wp.volume_sample_f(volume, p, wp.Volume.CLOSEST), expected)

    q = wp.volume_index_to_world(volume, p)
    q_inv = wp.volume_world_to_index(volume, q)
    expect_eq(p, q_inv)


@wp.kernel
def test_volume_sample_linear_f(volume: wp.uint64,
                                points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]


    expected = p[0] * p[1] * p[2]
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        return  # not testing against background values

    expect_near(wp.volume_sample_f(volume, p, wp.Volume.LINEAR), expected, 2.0e-4)

@wp.kernel
def test_volume_sample_local_f_linear_values(volume: wp.uint64,
                              points: wp.array(dtype=wp.vec3),
                              values: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    p = points[tid]
    values[tid] = wp.volume_sample_f(volume, p, wp.Volume.LINEAR)

@wp.kernel
def test_volume_sample_world_f_linear_values(volume: wp.uint64,
                              points: wp.array(dtype=wp.vec3),
                              values: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    q = points[tid]
    p = wp.volume_world_to_index(volume, q)
    values[tid] = wp.volume_sample_f(volume, p, wp.Volume.LINEAR)

# vec3f volume tests
@wp.kernel
def test_volume_lookup_v(volume: wp.uint64,
                         points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]
    expected = wp.vec3(p[0] + 2.0*p[1] + 3.0*p[2], 4.0*p[0] + 5.0*p[1] + 6.0*p[2], 7.0*p[0] + 8.0*p[1] + 9.0*p[2])
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        expected = wp.vec3(10.8, -4.13, 10.26)

    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    expect_eq(wp.volume_lookup_v(volume, i, j, k), expected)


@wp.kernel
def test_volume_sample_closest_v(volume: wp.uint64,
                                 points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]
    i = round(p[0])
    j = round(p[1])
    k = round(p[2])
    expected = wp.vec3(i + 2.0*j + 3.0*k, 4.0*i + 5.0*j + 6.0*k, 7.0*i + 8.0*j + 9.0*k)
    if abs(i) > 10.0 or abs(j) > 10.0 or abs(k) > 10.0:
        expected = wp.vec3(10.8, -4.13, 10.26)

    expect_eq(wp.volume_sample_v(volume, p, wp.Volume.CLOSEST), expected)

    q = wp.volume_index_to_world(volume, p)
    q_inv = wp.volume_world_to_index(volume, q)
    expect_eq(p, q_inv)


@wp.kernel
def test_volume_sample_linear_v(volume: wp.uint64,
                                points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]

    expected = wp.vec3(p[0] + 2.0*p[1] + 3.0*p[2], 4.0*p[0] + 5.0*p[1] + 6.0*p[2], 7.0*p[0] + 8.0*p[1] + 9.0*p[2])
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        return  # not testing against background values

    expect_near(wp.volume_sample_v(volume, p, wp.Volume.LINEAR), expected, 2.0e-4)

@wp.kernel
def test_volume_sample_local_v_linear_values(volume: wp.uint64,
                              points: wp.array(dtype=wp.vec3),
                              values: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    p = points[tid]
    ones = wp.vec3(1.,1.,1.)
    values[tid] = wp.dot(wp.volume_sample_v(volume, p, wp.Volume.LINEAR), ones)

@wp.kernel
def test_volume_sample_world_v_linear_values(volume: wp.uint64,
                              points: wp.array(dtype=wp.vec3),
                              values: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    q = points[tid]
    p = wp.volume_world_to_index(volume, q)
    ones = wp.vec3(1.,1.,1.)
    values[tid] = wp.dot(wp.volume_sample_v(volume, p, wp.Volume.LINEAR), ones)

# int32 volume tests
@wp.kernel
def test_volume_lookup_i(volume: wp.uint64,
                         points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])
    expected = i * j * k
    if abs(i) > 10 or abs(j) > 10 or abs(k) > 10:
        expected = 10

    expect_eq(wp.volume_lookup_i(volume, i, j, k), expected)


@wp.kernel
def test_volume_sample_i(volume: wp.uint64,
                                 points: wp.array(dtype=wp.vec3)):

    tid = wp.tid()

    p = points[tid]
    i = round(p[0])
    j = round(p[1])
    k = round(p[2])
    expected = int(i * j * k)
    if abs(i) > 10.0 or abs(j) > 10.0 or abs(k) > 10.0:
        expected = 10

    expect_eq(wp.volume_sample_i(volume, p), expected)

    q = wp.volume_index_to_world(volume, p)
    q_inv = wp.volume_world_to_index(volume, q)
    expect_eq(p, q_inv)


# Index/world transformation tests
@wp.kernel
def test_volume_index_to_world(volume: wp.uint64,
                   points: wp.array(dtype=wp.vec3),
                   values: wp.array(dtype=wp.float32),
                   grad_values: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    p = points[tid]
    ones = wp.vec3(1.,1.,1.)
    values[tid] = wp.dot(wp.volume_index_to_world(volume, p), ones)
    grad_values[tid] = wp.volume_index_to_world_dir(volume, ones)

@wp.kernel
def test_volume_world_to_index(volume: wp.uint64,
                       points: wp.array(dtype=wp.vec3),
                       values: wp.array(dtype=wp.float32),
                       grad_values: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    p = points[tid]
    ones = wp.vec3(1.,1.,1.)
    values[tid] = wp.dot(wp.volume_world_to_index(volume, p), ones)
    grad_values[tid] = wp.volume_world_to_index_dir(volume, ones)




devices = wp.get_devices()
rng = np.random.default_rng(101215)

# Note about the test grids:
# test_grid and test_int32_grid
#   active region: [-10,10]^3
#   values: v[i,j,k] = i * j * k
#   voxel size: 0.25
#
# test_vec_grid
#   active region: [-10,10]^3
#   values: v[i,j,k] = (i + 2*j + 3*k, 4*i + 5*j + 6*k, 7*i + 8*j + 9*k)
#   voxel size: 0.25
#
# torus
#   index to world transformation:
#      [0.1, 0, 0, 0]
#      [0, 0, 0.1, 0]
#      [0, 0.1, 0, 0]
#      [1, 2, 3, 1]
#   (-90 degrees rotation along X)
#   voxel size: 0.1
volume_paths = {
    "float": os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/test_grid.nvdb")),
    "int32": os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/test_int32_grid.nvdb")),
    "vec3f": os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/test_vec_grid.nvdb")),
    "torus": os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/torus.nvdb"))
}

volumes = {}
points = {}
points_jittered = {}
for value_type, path in volume_paths.items():
    volumes[value_type] = {}
    volume_data = open(path, "rb").read()
    for device in devices:
        try:
            volume = wp.Volume.load_from_nvdb(volume_data, device)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load volume from \"{path}\" to {device} memory:\n{e}")

        volumes[value_type][device] = volume

        
def register(parent):

    class TestVolumes(parent):
        def test_volume_sample_linear_f_gradient(self):

            for device in devices:
                points = rng.uniform(-10., 10., size=(100, 3))
                values = wp.array(np.zeros(1), dtype=wp.float32, device=device, requires_grad=True)
                for case in points:
                    uvws = wp.array(case, dtype=wp.vec3, device=device, requires_grad=True)
                    xyzs = wp.array(case * 0.25, dtype=wp.vec3, device=device, requires_grad=True)

                    tape = wp.Tape()
                    with tape:
                        wp.launch(test_volume_sample_local_f_linear_values, dim=1, inputs=[volumes["float"][device].id, uvws, values], device=device)
                    tape.backward(values)

                    x, y, z = case
                    grad_expected = np.array([y*z, x*z, x*y])
                    grad_computed = tape.gradients[uvws].numpy()[0]
                    np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)

                    tape = wp.Tape()
                    with tape:
                        wp.launch(test_volume_sample_world_f_linear_values, dim=1, inputs=[volumes["float"][device].id, xyzs, values], device=device)
                    tape.backward(values)

                    x, y, z = case
                    grad_expected = np.array([y*z, x*z, x*y]) / 0.25
                    grad_computed = tape.gradients[xyzs].numpy()[0]
                    np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)

        def test_volume_sample_linear_v_gradient(self):

            for device in devices:
                points = rng.uniform(-10., 10., size=(100, 3))
                values = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
                for case in points:
                    uvws = wp.array(case, dtype=wp.vec3, device=device, requires_grad=True)
                    xyzs = wp.array(case * 0.25, dtype=wp.vec3, device=device, requires_grad=True)

                    tape = wp.Tape()
                    with tape:
                        wp.launch(test_volume_sample_local_v_linear_values, dim=1, inputs=[volumes["vec3f"][device].id, uvws, values], device=device)
                    tape.backward(values)

                    grad_expected = np.array([6.0, 15.0, 24.0])
                    grad_computed = tape.gradients[uvws].numpy()[0]
                    np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)

                    tape = wp.Tape()
                    with tape:
                        wp.launch(test_volume_sample_world_v_linear_values, dim=1, inputs=[volumes["vec3f"][device].id, xyzs, values], device=device)
                    tape.backward(values)

                    grad_expected = np.array([6.0, 15.0, 24.0]) / 0.25
                    grad_computed = tape.gradients[xyzs].numpy()[0]
                    np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)

        def test_volume_transform_gradient(self):

            for device in devices:
                values = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
                grad_values = wp.zeros(1, dtype=wp.vec3, device=device)
                points = rng.uniform(-10., 10., size=(10, 3))
                for case in points:
                    points = wp.array(case, dtype=wp.vec3, device=device, requires_grad=True)
                    tape = wp.Tape()
                    with tape:
                        wp.launch(test_volume_index_to_world, dim=1, inputs=[volumes["torus"][device].id, points, values, grad_values], device=device)
                    tape.backward(values)
               
                    grad_computed = tape.gradients[points].numpy()
                    grad_expected = grad_values.numpy()
                    np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)

                    tape.reset()

                    with tape:
                        wp.launch(test_volume_world_to_index, dim=1, inputs=[volumes["torus"][device].id, points, values, grad_values], device=device)
                    tape.backward(values)

                    grad_computed = tape.gradients[points].numpy()
                    grad_expected = grad_values.numpy()
                    np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)

    for device in devices:
        axis = np.linspace(-11, 11, 23)
        points_np = np.array([[x, y, z] for x in axis for y in axis for z in axis])
        points_jittered_np = points_np + rng.uniform(-0.5, 0.5, size=points_np.shape)
        points[device] = wp.array(points_np, dtype=wp.vec3, device=device)
        points_jittered[device] = wp.array(points_jittered_np, dtype=wp.vec3, device=device)

        add_kernel_test(TestVolumes, test_volume_lookup_f, dim=len(points_np), inputs=[volumes["float"][device].id, points[device]], devices=[device])
        add_kernel_test(TestVolumes, test_volume_sample_closest_f, dim=len(points_np), inputs=[volumes["float"][device].id, points_jittered[device]], devices=[device])
        add_kernel_test(TestVolumes, test_volume_sample_linear_f, dim=len(points_np), inputs=[volumes["float"][device].id, points_jittered[device]], devices=[device])

        add_kernel_test(TestVolumes, test_volume_lookup_v, dim=len(points_np), inputs=[volumes["vec3f"][device].id, points[device]], devices=[device])
        add_kernel_test(TestVolumes, test_volume_sample_closest_v, dim=len(points_np), inputs=[volumes["vec3f"][device].id, points_jittered[device]], devices=[device])
        add_kernel_test(TestVolumes, test_volume_sample_linear_v, dim=len(points_np), inputs=[volumes["vec3f"][device].id, points_jittered[device]], devices=[device])

        add_kernel_test(TestVolumes, test_volume_lookup_i, dim=len(points_np), inputs=[volumes["int32"][device].id, points[device]], devices=[device])
        add_kernel_test(TestVolumes, test_volume_sample_i, dim=len(points_np), inputs=[volumes["int32"][device].id, points_jittered[device]], devices=[device])

    return TestVolumes

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
