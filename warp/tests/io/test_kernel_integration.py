# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that loaded meshes work correctly in Warp kernels."""

import warp as wp

# Test that loaded mesh works in a kernel
mesh = wp.load_mesh("warp/tests/io/assets/cube.obj")


# Define a simple kernel that uses the mesh
@wp.kernel
def test_mesh_kernel(mesh: wp.uint64, points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    # Just access mesh points to ensure it works
    p = mesh.points[i]


# Launch kernel
test_points = wp.zeros(mesh.points.shape[0], dtype=wp.vec3)
wp.launch(kernel=test_mesh_kernel, dim=mesh.points.shape[0], inputs=[mesh.id, test_points])

print("Kernel execution successful!")
print(f"Mesh has {mesh.points.shape[0]} vertices")
