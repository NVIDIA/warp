import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np

import math

wp.config.mode = "release"
wp.config.verbose = True
wp.config.cache_kernels = False
wp.config.verify_cuda = True

wp.init()

num_points = 1000
num_runs = 16

radius = 1.0
voxel_size = 0.1

device = "cuda"

@wp.kernel
def analytical_distance(points: wp.array(dtype=wp.vec3),
                        values: wp.array(dtype=float),
                        radius: float):
    tid = wp.tid()

    p = points[tid]
    values[tid] = sqrt(dot(p, p)) - radius

@wp.kernel
def sample_volume(volume: wp.uint64,
                  points: wp.array(dtype=wp.vec3),
                  values: wp.array(dtype=float),
                  sampling_mode: int):

    tid = wp.tid()

    p = points[tid]

    values[tid] = wp.volume_sample(volume, p, sampling_mode)


sphere = None
with wp.ScopedTimer("creating volume"):
    sphere = wp.Volume.create_sphere(radius, 0, 0, 0, voxel_size, device)

data = sphere.array()
data_np = data.numpy()
magic = ''.join([chr(x) for x in data_np[0:8]])
if magic != "NanoVDB0":
    print("FAILED: NanoVDB signature doesn't match!")

volume = wp.Volume(data, device)

np.random.seed(1008)
for i in range(num_runs):

    print(f"Run: {i+1}")
    print("---------")

    # seeding points in the narrowband
    phi = np.random.uniform(0, 2*math.pi, num_points)
    theta = np.random.uniform(0, math.pi, num_points)
    d = np.random.uniform(-2*voxel_size, 2*voxel_size, num_points)

    points = ((radius + d) * np.stack((np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)))).transpose()
    points = np.round(points/voxel_size) * voxel_size

    points_arr = wp.array(points, dtype=wp.vec3, device=device)
    sdf_arr = wp.array(np.random.rand(len(points)), dtype=float, device=device)
    sdf_arr_ref = wp.zeros(len(points), dtype=float, device=device)

    wp.launch(kernel=analytical_distance, dim=len(points), inputs=[points_arr, sdf_arr_ref, radius], device=device)
    wp.synchronize()

    with wp.ScopedTimer("sample volume"):
        wp.launch(kernel=sample_volume, dim=len(points), inputs=[volume.id, points_arr, sdf_arr, wp.Volume.CLOSEST], device=device)
        wp.synchronize()


    sdf = sdf_arr.numpy()
    sdf_ref = sdf_arr_ref.numpy()
    error = np.max(np.abs(sdf-sdf_ref))

    if error < 1e-6:
        print(f"Pass! Max error in SDF values: {error}")
    else:
        print(f"FAIL! Max error in SDF values: {error}")