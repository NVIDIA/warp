# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that loaded meshes work correctly in Warp kernels."""

import warp as wp

# Test that loaded mesh works in a kernel
mesh = wp.load_mesh("warp/tests/io/assets/cube.obj")


# Define a simple kernel that uses the mesh
@wp.kernel
def test_mesh_kernel(mesh: wp.uint64, query_result: wp.array(dtype=int)):
    i = wp.tid()
    # Use mesh_query_point to verify mesh is queryable
    query = wp.mesh_query_point(mesh, wp.vec3(0.5, 0.5, 0.5), 10.0)
    if i == 0:
        query_result[0] = 1 if query.result else 0


# Launch kernel
query_result = wp.zeros(1, dtype=int)
wp.launch(kernel=test_mesh_kernel, dim=1, inputs=[mesh.id, query_result])

print("Kernel execution successful!")
print(f"Mesh has {mesh.points.shape[0]} vertices")
print(f"Mesh has {mesh.indices.shape[0] // 3} triangles")
print(f"Mesh query test: {'PASSED' if query_result.numpy()[0] else 'FAILED'}")
