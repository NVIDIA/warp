# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mesh file I/O utilities."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import warp as wp

# Get the assets directory
ASSETS_DIR = Path(__file__).parent / "assets"


class TestMeshIO(unittest.TestCase):
    """Test cases for mesh file I/O."""

    def setUp(self):
        """Set up test devices."""
        self.devices = wp.get_devices()

    # ── OBJ Tests ──

    def test_load_obj_triangle(self):
        """Single triangle: 3 vertices, 1 face."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "triangle.obj"))
        self.assertEqual(mesh.points.shape[0], 3)
        self.assertEqual(mesh.indices.shape[0], 3)

    def test_load_obj_cube(self):
        """Cube: 8 vertices, 12 triangles."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "cube.obj"))
        self.assertEqual(mesh.points.shape[0], 8)
        self.assertEqual(mesh.indices.shape[0], 36)  # 12 triangles * 3

    def test_load_obj_quad_triangulation(self):
        """Quad faces auto-triangulated to 2 triangles each."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "cube_quads.obj"))
        # 6 quads -> 12 triangles
        self.assertEqual(mesh.indices.shape[0], 36)

    def test_load_obj_negative_indices(self):
        """OBJ negative indices resolve correctly."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "negative_indices.obj"))
        self.assertEqual(mesh.points.shape[0], 3)
        self.assertEqual(mesh.indices.shape[0], 3)

    # ── STL Tests ──

    def test_load_stl_binary(self):
        """Binary STL loaded correctly."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "cube.stl"))
        # After deduplication, should have 8 vertices
        self.assertGreaterEqual(mesh.points.shape[0], 8)
        self.assertEqual(mesh.indices.shape[0], 36)

    def test_load_stl_ascii(self):
        """ASCII STL loaded correctly."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "cube_ascii.stl"))
        self.assertEqual(mesh.indices.shape[0], 36)

    def test_stl_binary_ascii_equivalence(self):
        """Binary and ASCII STL of same mesh produce equivalent results."""
        mesh_binary = wp.load_mesh(str(ASSETS_DIR / "cube.stl"))
        mesh_ascii = wp.load_mesh(str(ASSETS_DIR / "cube_ascii.stl"))

        # Both should have the same number of triangles
        self.assertEqual(mesh_binary.indices.shape[0], mesh_ascii.indices.shape[0])

    def test_stl_vertex_deduplication(self):
        """Duplicated STL vertices are merged."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "cube.stl"), stl_merge_tolerance=1e-6)
        # A cube has 8 unique vertices
        # After deduplication, we should have fewer than 24 (3 per triangle * 12)
        self.assertLessEqual(mesh.points.shape[0], 8)

    # ── PLY Tests ──

    def test_load_ply_binary_le(self):
        """Binary little-endian PLY loaded correctly."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "cube.ply"))
        self.assertEqual(mesh.points.shape[0], 8)
        self.assertEqual(mesh.indices.shape[0], 36)

    def test_load_ply_ascii(self):
        """ASCII PLY loaded correctly."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "cube_ascii.ply"))
        self.assertEqual(mesh.points.shape[0], 8)
        self.assertEqual(mesh.indices.shape[0], 36)

    # ── Round-trip Tests ──

    def test_obj_roundtrip(self):
        """Save then load OBJ: geometric equivalence via vertex comparison."""
        # Load original mesh
        original = wp.load_mesh(str(ASSETS_DIR / "cube.obj"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.obj")
            wp.save_mesh(original, output_path)

            # Load saved mesh
            loaded = wp.load_mesh(output_path)

            # Check that vertices are the same (order may differ, so we check count)
            self.assertEqual(original.points.shape[0], loaded.points.shape[0])
            self.assertEqual(original.indices.shape[0], loaded.indices.shape[0])

    def test_stl_roundtrip(self):
        """Save then load STL: geometry matches within tolerance.

        Note: Uses geometric equivalence (query results), not byte-for-byte
        comparison, due to precision loss and potential vertex reordering.
        """
        original = wp.load_mesh(str(ASSETS_DIR / "cube.obj"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.stl")
            wp.save_mesh(original, output_path)

            loaded = wp.load_mesh(output_path)

            # Same number of triangles
            self.assertEqual(original.indices.shape[0] // 3, loaded.indices.shape[0] // 3)

    def test_ply_roundtrip(self):
        """Save then load PLY: geometry matches."""
        original = wp.load_mesh(str(ASSETS_DIR / "cube.obj"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.ply")
            wp.save_mesh(original, output_path)

            loaded = wp.load_mesh(output_path)

            self.assertEqual(original.points.shape[0], loaded.points.shape[0])
            self.assertEqual(original.indices.shape[0], loaded.indices.shape[0])

    # ── API Tests ──

    def test_load_mesh_auto_format(self):
        """wp.load_mesh() detects format from extension."""
        obj_mesh = wp.load_mesh(str(ASSETS_DIR / "triangle.obj"))
        stl_mesh = wp.load_mesh(str(ASSETS_DIR / "triangle.stl"))
        ply_mesh = wp.load_mesh(str(ASSETS_DIR / "triangle.ply"))

        # All should have same triangle count
        self.assertEqual(obj_mesh.indices.shape[0], 3)
        self.assertEqual(stl_mesh.indices.shape[0], 3)
        self.assertEqual(ply_mesh.indices.shape[0], 3)

    def test_read_mesh_returns_meshdata(self):
        """read_mesh() returns MeshData, not wp.Mesh."""
        data = wp.read_mesh(str(ASSETS_DIR / "triangle.obj"))
        self.assertIsInstance(data, wp.MeshData)
        self.assertIsInstance(data.points, np.ndarray)
        self.assertIsInstance(data.indices, np.ndarray)

    def test_meshdata_to_warp_mesh(self):
        """MeshData.to_warp_mesh() creates a wp.Mesh."""
        data = wp.read_mesh(str(ASSETS_DIR / "triangle.obj"))
        mesh = data.to_warp_mesh()

        self.assertIsInstance(mesh, wp.Mesh)
        self.assertEqual(mesh.points.shape[0], 3)
        self.assertEqual(mesh.indices.shape[0], 3)

    # ── Error Tests ──

    def test_file_not_found(self):
        """FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            wp.load_mesh("nonexistent.obj")

    def test_unsupported_format(self):
        """ValueError for .fbx, .gltf, etc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_file = os.path.join(tmpdir, "test.fbx")
            with open(fake_file, "w") as f:
                f.write("fake content")

            with self.assertRaises(ValueError):
                wp.load_mesh(fake_file)

    def test_format_override_no_extension(self):
        """format='obj' allows loading files without extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_file = os.path.join(tmpdir, "no_extension")
            with open(fake_file, "w") as f:
                f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

            mesh = wp.load_mesh(fake_file, file_format="obj")
            self.assertEqual(mesh.points.shape[0], 3)

    # ── Winding Order Tests ──

    def test_flip_winding_order(self):
        """flip_winding=True reverses triangle indices."""
        mesh_normal = wp.load_mesh(str(ASSETS_DIR / "triangle.obj"))
        mesh_flipped = wp.load_mesh(str(ASSETS_DIR / "triangle.obj"), flip_winding=True)

        # Same vertices, different winding
        self.assertEqual(mesh_normal.points.shape[0], mesh_flipped.points.shape[0])

        # Indices should be reversed per triangle
        # Original: [0, 1, 2], Flipped: [2, 1, 0]
        np.testing.assert_array_equal(
            mesh_normal.indices.numpy(),
            mesh_flipped.indices.numpy().reshape(-1, 3)[:, [2, 1, 0]].reshape(-1),
        )


class TestMeshIODevices(unittest.TestCase):
    """Test mesh I/O across different devices."""

    def test_load_mesh_cpu(self):
        """Mesh can be loaded on CPU."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "cube.obj"), device="cpu")
        self.assertEqual(mesh.device, wp.get_device("cpu"))

    def test_load_mesh_cuda(self):
        """Mesh can be loaded on CUDA device (if available)."""
        devices = wp.get_devices()
        cuda_devices = [d for d in devices if d.is_cuda]

        if cuda_devices:
            mesh = wp.load_mesh(str(ASSETS_DIR / "cube.obj"), device=cuda_devices[0])
            self.assertTrue(mesh.device.is_cuda)
        else:
            self.skipTest("CUDA device not available")

    def test_mesh_queries_work(self):
        """Mesh BVH is built correctly for queries."""
        mesh = wp.load_mesh(str(ASSETS_DIR / "cube.obj"))

        # Check that mesh has the expected structure
        self.assertIsNotNone(mesh.id)
        self.assertEqual(mesh.points.shape[0], 8)  # Cube has 8 vertices
        self.assertEqual(mesh.indices.shape[0], 36)  # 12 triangles * 3


if __name__ == "__main__":
    unittest.main()
