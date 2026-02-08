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
def make_field_sphere_sdf(field: wp.array3d(dtype=float), center: wp.vec3, radius: float):
    """Make a sphere SDF for nodes on the integer domain with node coordinates 0,1,2,3,..."""

    i, j, k = wp.tid()

    p = wp.vec3(float(i), float(j), float(k))

    d = wp.length(p - center) - radius

    field[i, j, k] = d


@wp.kernel
def make_field_sphere_sdf_unit_domain(
    field: wp.array3d(dtype=float), center: wp.vec3, radius: wp.array(dtype=wp.float32)
):
    """Makes a sphere SDF for nodes on the unit domain [-1, 1]^3."""
    i, j, k = wp.tid()

    nx, ny, nz = field.shape[0], field.shape[1], field.shape[2]

    p = wp.vec3(
        2.0 * wp.float32(i) / (wp.float32(nx) - 1.0) - 1.0,
        2.0 * wp.float32(j) / (wp.float32(ny) - 1.0) - 1.0,
        2.0 * wp.float32(k) / (wp.float32(nz) - 1.0) - 1.0,
    )

    d = wp.length(p - center) - radius[0]

    field[i, j, k] = d


@wp.kernel
def compute_surface_area(
    verts: wp.array(dtype=wp.vec3), faces: wp.array(dtype=wp.int32), out_area: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    vi = faces[3 * tid + 0]
    vj = faces[3 * tid + 1]
    vk = faces[3 * tid + 2]

    p0 = verts[vi]
    p1 = verts[vj]
    p2 = verts[vk]

    # Heron's formula for triangle area
    a = wp.length(p1 - p0)
    b = wp.length(p2 - p0)
    c = wp.length(p2 - p1)
    s = (a + b + c) / 2.0
    area = wp.sqrt(s * (s - a) * (s - b) * (s - c))

    wp.atomic_add(out_area, 0, area)


def validate_marching_cubes_output(test, verts_np, faces_np, check_nonempty=True):
    # check that the face array seems valid
    if check_nonempty:
        test.assertGreater(faces_np.shape[0], 0)  # at least one face
    test.assertEqual(faces_np.shape[1], 3)  # all faces triangular
    test.assertTrue((faces_np >= 0).all())  # all face inds nonnegative
    test.assertTrue((faces_np < verts_np.shape[0]).all())  # all face inds are in-bounds on the vertex array
    test.assertTrue((faces_np[:, 0] != faces_np[:, 1]).all())  # all faces have unique vertices
    test.assertTrue((faces_np[:, 0] != faces_np[:, 2]).all())  # all faces have unique vertices
    test.assertTrue((faces_np[:, 1] != faces_np[:, 2]).all())  # all faces have unique vertices
    test.assertTrue(
        (np.unique(faces_np.flatten()) == np.arange(verts_np.shape[0])).all()
    )  # all vertices are used in at least one face

    # check that the vertex array seems valid
    if check_nonempty:
        test.assertGreater(verts_np.shape[0], 0)  # at least one vertex
    test.assertEqual(verts_np.shape[1], 3)  # all vertices are 3D
    test.assertTrue(np.isfinite(verts_np).all())  # all vertices are finite


def test_marching_cubes(test, device):
    """Basic test of typical usage."""
    node_dim = 64
    cell_dim = node_dim - 1
    field = wp.zeros(shape=(node_dim, node_dim, node_dim), dtype=float, device=device)
    bounds_low = (0.0, 0.0, 0.0)
    bounds_high = (float(cell_dim), float(cell_dim), float(cell_dim))

    iso = wp.MarchingCubes(
        nx=node_dim,
        ny=node_dim,
        nz=node_dim,
        device=device,
        domain_bounds_lower_corner=bounds_low,
        domain_bounds_upper_corner=bounds_high,
    )

    radius = node_dim / 4.0

    wp.launch(
        make_field_sphere_sdf,
        dim=field.shape,
        inputs=[field, wp.vec3(node_dim / 2, node_dim / 2, node_dim / 2), radius],
        device=device,
    )

    iso.surface(field=field, threshold=0.0)
    verts_np = iso.verts.numpy()
    faces_np = iso.indices.numpy().reshape(-1, 3)
    test.assertEqual(
        iso.indices.dtype, wp.int32
    )  # make sure we are following Warp convention of using a flat array of indices
    validate_marching_cubes_output(test, verts_np, faces_np)

    # check that all returned vertices lie on the surface of the sphere
    length = np.linalg.norm(verts_np - np.array([node_dim / 2, node_dim / 2, node_dim / 2]), axis=1)
    error = np.abs(length - radius)
    test.assertTrue(np.max(error) < 1.0)

    iso.resize(nx=node_dim * 2, ny=node_dim * 2, nz=node_dim * 2)  # smoke test for deprecated function


def test_marching_cubes_functional(test, device):
    """Ensure the single-function interface works as expected."""
    node_dim = 64
    cell_dim = node_dim - 1
    field = wp.zeros(shape=(node_dim, node_dim, node_dim), dtype=float, device=device)
    bounds_low = (0.0, 0.0, 0.0)
    bounds_high = (float(cell_dim), float(cell_dim), float(cell_dim))

    radius = node_dim / 4.0
    wp.launch(
        make_field_sphere_sdf,
        dim=field.shape,
        inputs=[field, wp.vec3(node_dim / 2, node_dim / 2, node_dim / 2), radius],
        device=device,
    )

    # call via the functional interface
    verts, faces = wp.MarchingCubes.extract_surface_marching_cubes(
        field, threshold=0.0, domain_bounds_lower_corner=bounds_low, domain_bounds_upper_corner=bounds_high
    )

    verts_np = verts.numpy()
    faces_np = faces.numpy().reshape(-1, 3)
    validate_marching_cubes_output(test, verts_np, faces_np)

    # check that all returned vertices lie on the surface of the sphere
    length = np.linalg.norm(verts_np - np.array([node_dim / 2, node_dim / 2, node_dim / 2]), axis=1)
    error = np.abs(length - radius)
    test.assertTrue(np.max(error) < 1.0)


def test_marching_cubes_nonuniform(test, device):
    """Test the logic for when the dimensions of the grid are not uniform."""

    dimX = 64
    dimY = 48
    dimZ = 72
    field = wp.zeros(shape=(dimX, dimY, dimZ), dtype=float, device=device)

    bounds_low = wp.vec3(0.0, 0.0, 0.0)
    bounds_high = wp.vec3(float(dimX), float(dimY), float(dimZ))

    iso = wp.MarchingCubes(
        nx=dimX,
        ny=dimY,
        nz=dimZ,
        device=device,
        domain_bounds_lower_corner=bounds_low,
        domain_bounds_upper_corner=bounds_high,
    )

    radius = dimX / 4.0
    wp.launch(
        make_field_sphere_sdf,
        dim=field.shape,
        inputs=[field, wp.vec3(dimX / 2, dimY / 2, dimZ / 2), radius],
        device=device,
    )

    iso.surface(field=field, threshold=0.0)
    verts_np = iso.verts.numpy()
    faces_np = iso.indices.numpy().reshape(-1, 3)
    validate_marching_cubes_output(test, verts_np, faces_np)


def test_marching_cubes_empty_output(test, device):
    """Make sure we handle the empty-output case correctly."""

    dim = 64
    field = wp.zeros(shape=(dim, dim, dim), dtype=float, device=device)

    iso = wp.MarchingCubes(nx=dim, ny=dim, nz=dim, device=device)

    wp.launch(make_field_sphere_sdf, dim=field.shape, inputs=[field, wp.vec3(0.5, 0.5, 0.5), 0.25], device=device)

    iso.surface(field=field, threshold=1000.0)  # set threshold to a large value so that no vertices are generated
    verts_np = iso.verts.numpy()
    faces_np = iso.indices.numpy().reshape(-1, 3)
    validate_marching_cubes_output(test, verts_np, faces_np, check_nonempty=False)

    test.assertEqual(faces_np.shape[0], 0)  # no faces
    test.assertEqual(verts_np.shape[0], 0)  # no vertices


def test_marching_cubes_differentiable(test, device):
    """Check that marching cubes has reasonable gradients.

    This test constructs an SDF of sphere, extracts a surface, computes its
    surface area, and then differentiates the surface area with respect to
    the sphere's radius.
    """
    node_dim = 64
    bounds_low = wp.vec3(-1.0, -1.0, -1.0)
    bounds_high = wp.vec3(1.0, 1.0, 1.0)

    radius = 0.5
    radius_wp = wp.full((1,), value=0.5, dtype=wp.float32, device=device, requires_grad=True)

    with wp.Tape() as tape:
        field = wp.zeros(shape=(node_dim, node_dim, node_dim), dtype=float, device=device, requires_grad=True)
        wp.launch(
            make_field_sphere_sdf_unit_domain,
            dim=field.shape,
            inputs=[field, wp.vec3(0.0, 0.0, 0.0), radius_wp],
            device=device,
        )

        # call via the functional interface
        verts, faces = wp.MarchingCubes.extract_surface_marching_cubes(
            field, threshold=0.0, domain_bounds_lower_corner=bounds_low, domain_bounds_upper_corner=bounds_high
        )

        # compute surface area
        area = wp.zeros(shape=(1,), dtype=float, device=device, requires_grad=True)
        wp.launch(compute_surface_area, dim=faces.shape[0] // 3, inputs=[verts, faces, area], device=device)

        # confirm surface area is correct vs. the analytical ground truth
        area_np = area.numpy()[0]
        test.assertTrue(np.abs(area_np - 4.0 * np.pi * radius * radius) < 1e-2)

    # compute the gradient of the surface area with respect to the radius
    tape.backward(area)

    # confirm the gradient is correct vs. the analytical ground truth
    grad_np = radius_wp.grad.numpy()[0]
    test.assertTrue(np.abs(grad_np - 8.0 * np.pi * radius) < 1e-2)


def test_mc_lookup_tables_structure(test, device):
    """Test that lookup tables have correct sizes and types."""
    # Access via class attributes
    CUBE_CORNER_OFFSETS = wp.MarchingCubes.CUBE_CORNER_OFFSETS
    EDGE_TO_CORNERS = wp.MarchingCubes.EDGE_TO_CORNERS
    CASE_TO_TRI_RANGE = wp.MarchingCubes.CASE_TO_TRI_RANGE
    TRI_LOCAL_INDICES = wp.MarchingCubes.TRI_LOCAL_INDICES

    # Verify types are tuples (immutable)
    test.assertIsInstance(CUBE_CORNER_OFFSETS, tuple)
    test.assertIsInstance(EDGE_TO_CORNERS, tuple)
    test.assertIsInstance(CASE_TO_TRI_RANGE, tuple)
    test.assertIsInstance(TRI_LOCAL_INDICES, tuple)

    # Verify sizes
    test.assertEqual(len(CUBE_CORNER_OFFSETS), 8)  # 8 corners of a cube
    test.assertEqual(len(EDGE_TO_CORNERS), 12)  # 12 edges of a cube
    test.assertEqual(len(CASE_TO_TRI_RANGE), 257)  # 256 cases + 1 for range end
    test.assertEqual(len(TRI_LOCAL_INDICES), 2460)  # 820 triangles x 3 vertices each

    # Verify nested structure
    for corner in CUBE_CORNER_OFFSETS:
        test.assertIsInstance(corner, tuple)
        test.assertEqual(len(corner), 3)  # (x, y, z)

    for edge in EDGE_TO_CORNERS:
        test.assertIsInstance(edge, tuple)
        test.assertEqual(len(edge), 2)  # (corner_from, corner_to)


def test_mc_lookup_tables_values(test, device):
    """Test that lookup table values are valid."""
    # Access via class attributes
    CUBE_CORNER_OFFSETS = wp.MarchingCubes.CUBE_CORNER_OFFSETS
    EDGE_TO_CORNERS = wp.MarchingCubes.EDGE_TO_CORNERS
    CASE_TO_TRI_RANGE = wp.MarchingCubes.CASE_TO_TRI_RANGE
    TRI_LOCAL_INDICES = wp.MarchingCubes.TRI_LOCAL_INDICES

    # Corner offsets should be 0 or 1
    for corner in CUBE_CORNER_OFFSETS:
        for coord in corner:
            test.assertIn(coord, (0, 1))

    # Edge corners should reference valid corner indices (0-7)
    for edge in EDGE_TO_CORNERS:
        for corner_idx in edge:
            test.assertGreaterEqual(corner_idx, 0)
            test.assertLessEqual(corner_idx, 7)

    # Case to tri range should be monotonically non-decreasing
    for i in range(len(CASE_TO_TRI_RANGE) - 1):
        test.assertLessEqual(CASE_TO_TRI_RANGE[i], CASE_TO_TRI_RANGE[i + 1])

    # First case (all corners outside) should have no triangles
    test.assertEqual(CASE_TO_TRI_RANGE[0], CASE_TO_TRI_RANGE[1])

    # Last case (all corners inside) should have no triangles
    test.assertEqual(CASE_TO_TRI_RANGE[255], CASE_TO_TRI_RANGE[256])

    # Triangle local indices should reference valid edge indices (0-11)
    for edge_idx in TRI_LOCAL_INDICES:
        test.assertGreaterEqual(edge_idx, 0)
        test.assertLessEqual(edge_idx, 11)


def test_mc_lookup_tables_to_warp_array(test, device):
    """Test that lookup tables can be converted to warp arrays."""
    # Access via class attributes
    CUBE_CORNER_OFFSETS = wp.MarchingCubes.CUBE_CORNER_OFFSETS
    EDGE_TO_CORNERS = wp.MarchingCubes.EDGE_TO_CORNERS
    CASE_TO_TRI_RANGE = wp.MarchingCubes.CASE_TO_TRI_RANGE
    TRI_LOCAL_INDICES = wp.MarchingCubes.TRI_LOCAL_INDICES

    # Convert to warp arrays with appropriate dtypes
    corner_offsets = wp.array(CUBE_CORNER_OFFSETS, dtype=wp.vec3ub, device=device)
    edge_to_corners = wp.array(EDGE_TO_CORNERS, dtype=wp.vec2ub, device=device)
    case_to_tri_range = wp.array(CASE_TO_TRI_RANGE, dtype=wp.int32, device=device)
    tri_local_indices = wp.array(TRI_LOCAL_INDICES, dtype=wp.int32, device=device)

    # Verify shapes
    test.assertEqual(corner_offsets.shape, (8,))
    test.assertEqual(edge_to_corners.shape, (12,))
    test.assertEqual(case_to_tri_range.shape, (257,))
    test.assertEqual(tri_local_indices.shape, (2460,))

    # Verify we can read values back
    corner_offsets_np = corner_offsets.numpy()
    test.assertEqual(tuple(corner_offsets_np[0]), (0, 0, 0))
    test.assertEqual(tuple(corner_offsets_np[1]), (1, 0, 0))

    edge_to_corners_np = edge_to_corners.numpy()
    test.assertEqual(tuple(edge_to_corners_np[0]), (0, 1))  # edge 0 connects corners 0 and 1


devices = get_test_devices()


class TestMarchingCubes(unittest.TestCase):
    def test_marching_cubes_new_del(self):
        # test the scenario in which a MarchingCubes instance is created but not initialized before gc
        instance = wp.MarchingCubes.__new__(wp.MarchingCubes)
        instance.__del__()


add_function_test(TestMarchingCubes, "test_marching_cubes", test_marching_cubes, devices=devices)
add_function_test(TestMarchingCubes, "test_marching_cubes_functional", test_marching_cubes_functional, devices=devices)
add_function_test(TestMarchingCubes, "test_marching_cubes_nonuniform", test_marching_cubes_nonuniform, devices=devices)
add_function_test(
    TestMarchingCubes, "test_marching_cubes_empty_output", test_marching_cubes_empty_output, devices=devices
)
add_function_test(
    TestMarchingCubes, "test_marching_cubes_differentiable", test_marching_cubes_differentiable, devices=devices
)
add_function_test(
    TestMarchingCubes, "test_mc_lookup_tables_structure", test_mc_lookup_tables_structure, devices=devices
)
add_function_test(TestMarchingCubes, "test_mc_lookup_tables_values", test_mc_lookup_tables_values, devices=devices)
add_function_test(
    TestMarchingCubes, "test_mc_lookup_tables_to_warp_array", test_mc_lookup_tables_to_warp_array, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
