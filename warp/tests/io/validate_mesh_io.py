# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive validation script for mesh I/O utilities.

This script performs additional validation beyond unit tests:
- Real-world mesh file compatibility
- Large mesh performance testing
- Edge case handling
- Integration with Warp mesh operations
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

import warp as wp


def test_real_world_meshes():
    """Test loading real-world mesh files."""
    print("Testing real-world meshes...")

    real_world_dir = Path(__file__).parent / "real_world_assets"

    if not real_world_dir.exists():
        print("  Skipping: real_world_assets directory not found")
        return

    files = list(real_world_dir.glob("*.obj"))
    if not files:
        print("  Skipping: no .obj files in real_world_assets")
        return

    for mesh_file in files:
        print(f"  {mesh_file.name}...", end=" ")
        mesh = wp.load_mesh(str(mesh_file))
        assert mesh.points.shape[0] > 0, "No vertices loaded"
        assert mesh.indices.shape[0] % 3 == 0, "Indices not divisible by 3"
        assert mesh.id is not None, "BVH not built"
        print(f"OK ({mesh.points.shape[0]} verts, {mesh.indices.shape[0] // 3} tris)")

    print("Real-world mesh tests passed!")


def test_large_mesh_performance():
    """Test performance with a large procedurally generated mesh."""
    print("\nTesting large mesh performance...")

    # Create a large sphere mesh (100k+ triangles)
    num_verts = 50000
    num_tris = 100000

    print(f"  Generating {num_verts} vertices, {num_tris} triangles...", end=" ")

    # Simple triangulated plane
    rng = np.random.default_rng()
    points = rng.random((num_verts, 3), dtype=np.float32) * 10.0
    indices = rng.integers(0, num_verts, (num_tris * 3,), dtype=np.int32)

    # Save and load via OBJ
    with tempfile.NamedTemporaryFile(suffix=".obj", mode="w", delete=False) as f:
        temp_path = f.name

    try:
        # Write simple OBJ
        with open(temp_path, "w") as f:
            for p in points:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")
            for i in range(0, len(indices), 3):
                f.write(f"f {indices[i] + 1} {indices[i + 1] + 1} {indices[i + 2] + 1}\n")

        # Time the load
        start = time.time()
        mesh = wp.load_mesh(temp_path)
        load_time = time.time() - start

        print(f"OK (load time: {load_time:.3f}s)")
        assert mesh.points.shape[0] == num_verts
        assert mesh.indices.shape[0] == num_tris * 3

        # Large file should load in reasonable time (< 5 seconds for 100k tris)
        assert load_time < 5.0, f"Loading too slow: {load_time:.3f}s"

    finally:
        Path(temp_path).unlink(missing_ok=True)

    print("Large mesh performance test passed!")


def test_edge_cases():
    """Test various edge cases."""
    print("\nTesting edge cases...")

    # Test 1: Empty lines in OBJ
    print("  Empty lines in OBJ...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".obj", mode="w", delete=False) as f:
        temp_path = f.name
        f.write("\n")
        f.write("# Comment\n")
        f.write("v 0 0 0\n")
        f.write("\n")
        f.write("v 1 0 0\n")
        f.write("v 0 1 0\n")
        f.write("\n")
        f.write("f 1 2 3\n")
        f.write("\n")

    mesh = wp.load_mesh(temp_path)
    assert mesh.points.shape[0] == 3
    Path(temp_path).unlink()
    print("OK")

    # Test 2: Very small values
    print("  Very small coordinate values...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".obj", mode="w", delete=False) as f:
        temp_path = f.name
        f.write("v 1e-10 1e-10 1e-10\n")
        f.write("v 1.0 0.0 0.0\n")
        f.write("v 0.0 1.0 0.0\n")
        f.write("f 1 2 3\n")

    mesh = wp.load_mesh(temp_path)
    Path(temp_path).unlink()
    print("OK")

    # Test 3: Very large values
    print("  Very large coordinate values...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".obj", mode="w", delete=False) as f:
        temp_path = f.name
        f.write("v 1e6 1e6 1e6\n")
        f.write("v 0.0 0.0 0.0\n")
        f.write("v 1.0 0.0 0.0\n")
        f.write("f 1 2 3\n")

    mesh = wp.load_mesh(temp_path)
    Path(temp_path).unlink()
    print("OK")

    # Test 4: Mixed face formats in OBJ
    print("  Mixed face formats (v/vt/vn, v//vn, v)...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".obj", mode="w", delete=False) as f:
        temp_path = f.name
        f.write("v 0 0 0\n")
        f.write("v 1 0 0\n")
        f.write("v 0 1 0\n")
        f.write("v 1 1 0\n")
        f.write("f 1 2 3\n")
        f.write("f 1/1 2/2 3/3\n")
        f.write("f 1//1 2//1 3//1\n")

    mesh = wp.load_mesh(temp_path)
    assert mesh.points.shape[0] == 4
    Path(temp_path).unlink()
    print("OK")

    print("Edge case tests passed!")


def test_mesh_operations_integration():
    """Test integration with Warp mesh operations."""
    print("\nTesting Warp mesh operations integration...")

    # Load a test mesh (path relative to this file)
    assets_dir = Path(__file__).parent / "assets"
    mesh = wp.load_mesh(str(assets_dir / "cube.obj"))

    # Test that mesh can be used in warp operations
    print("  Mesh properties...", end=" ")
    assert mesh.points is not None
    assert mesh.indices is not None
    assert mesh.device is not None
    assert mesh.id is not None
    print("OK")

    # Test that points array is accessible
    print("  Points array access...", end=" ")
    points_np = mesh.points.numpy()
    assert points_np.shape == (8, 3)
    assert points_np.dtype == np.float32
    print("OK")

    # Test that indices array is accessible
    print("  Indices array access...", end=" ")
    indices_np = mesh.indices.numpy()
    assert indices_np.shape == (36,)
    assert indices_np.dtype == np.int32
    print("OK")

    # Test mesh device
    print("  Mesh device...", end=" ")
    expected_device = wp.get_device()
    assert str(mesh.device) == str(expected_device), f"Expected {expected_device}, got {mesh.device}"
    print(f"OK ({mesh.device})")

    print("Mesh operations integration tests passed!")


def test_format_variations():
    """Test various format variations."""
    print("\nTesting format variations...")

    # Test PLY with different endianness
    # For now, we'll just test ASCII PLY with different property orders

    # Test OBJ with different vertex orders
    print("  OBJ with vertex normal before position...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".obj", mode="w", delete=False) as f:
        temp_path = f.name
        f.write("vn 0 0 1\n")
        f.write("v 0 0 0\n")
        f.write("v 1 0 0\n")
        f.write("v 0 1 0\n")
        f.write("f 1 2 3\n")

    mesh = wp.load_mesh(temp_path)
    assert mesh.points.shape[0] == 3
    Path(temp_path).unlink()
    print("OK")

    # Test OBJ with Windows line endings
    print("  OBJ with Windows line endings (\\r\\n)...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".obj", mode="wb", delete=False) as f:
        temp_path = f.name
        f.write(b"v 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nf 1 2 3\r\n")

    mesh = wp.load_mesh(temp_path)
    assert mesh.points.shape[0] == 3
    Path(temp_path).unlink()
    print("OK")

    print("Format variation tests passed!")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\nTesting error handling...")

    # Test non-existent file
    print("  Non-existent file...", end=" ")
    try:
        wp.load_mesh("nonexistent.obj")
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("OK")

    # Test unsupported format
    print("  Unsupported format...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".fbx", mode="w", delete=False) as f:
        temp_path = f.name
        f.write("fake content")

    try:
        wp.load_mesh(temp_path)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        Path(temp_path).unlink()
        print("OK")

    # Test file size limit
    print("  File size limit...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".obj", mode="w", delete=False) as f:
        temp_path = f.name
        # Write some valid OBJ content
        f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        # Set file size to simulate large file (by truncating)
        f.write("\n")

    try:
        # Try to load with very small limit
        wp.load_mesh(temp_path, max_file_size_mb=0.000001)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        Path(temp_path).unlink()
        print("OK")

    print("Error handling tests passed!")


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Mesh I/O Comprehensive Validation")
    print("=" * 60)

    test_real_world_meshes()
    test_large_mesh_performance()
    test_edge_cases()
    test_mesh_operations_integration()
    test_format_variations()
    test_error_handling()

    print("\n" + "=" * 60)
    print("All validation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
