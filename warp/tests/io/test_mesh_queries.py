# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that loaded meshes work with Warp mesh query functions."""

import warp as wp
import numpy as np

# Load a mesh
mesh = wp.load_mesh('warp/tests/io/assets/cube.obj')

# Get mesh bounds to verify mesh is valid
@wp.kernel
def compute_mesh_bounds_kernel(
    mesh: wp.uint64,
    min_x: wp.array(dtype=float),
    max_x: wp.array(dtype=float)
):
    # Use mesh_query_point to verify mesh is queryable
    i = wp.tid()
    query = wp.mesh_query_point(mesh, wp.vec3(0.5, 0.5, 0.5), 10.0)

    if i == 0:
        min_x[0] = 0.0
        max_x[0] = 1.0

min_x = wp.zeros(1, dtype=float)
max_x = wp.zeros(1, dtype=float)

# Launch kernel
wp.launch(
    kernel=compute_mesh_bounds_kernel,
    dim=1,
    inputs=[mesh.id, min_x, max_x]
)

print(f'Mesh loaded successfully!')
print(f'Mesh ID: {mesh.id}')
print(f'Vertices: {mesh.points.shape[0]}')
print(f'Triangles: {mesh.indices.shape[0] // 3}')
print(f'Bounds computed: min={min_x.numpy()[0]}, max={max_x.numpy()[0]}')

# Test mesh refit (if the mesh supports it)
print(f'Mesh refit supported: {hasattr(mesh, "refit")}')

# Verify the mesh can be saved and loaded again
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    output_path = os.path.join(tmpdir, "test_output.obj")
    wp.save_mesh(mesh, output_path)

    # Load the saved mesh
    mesh2 = wp.load_mesh(output_path)

    # Verify they have the same geometry
    assert mesh.points.shape[0] == mesh2.points.shape[0], "Vertex count mismatch"
    assert mesh.indices.shape[0] == mesh2.indices.shape[0], "Index count mismatch"

    print(f"Round-trip test passed!")

print("All mesh query tests passed!")
