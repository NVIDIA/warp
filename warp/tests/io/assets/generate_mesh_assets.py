# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate test mesh assets in multiple formats.

All formats should be geometrically equivalent to enable cross-format comparison tests.
Run this script to regenerate all test assets.
"""

from __future__ import annotations

import os
from pathlib import Path


def generate_triangle_obj(path: str) -> None:
    """Write a single triangle mesh in OBJ format."""
    with open(path, "w") as f:
        f.write("# Single triangle mesh for testing\n")
        f.write("v 0.0 0.0 0.0\n")
        f.write("v 1.0 0.0 0.0\n")
        f.write("v 0.5 1.0 0.0\n")
        f.write("f 1 2 3\n")


def generate_triangle_stl(path: str, binary: bool = True) -> None:
    """Write a single triangle mesh in STL format."""
    import struct

    if binary:
        with open(path, "wb") as f:
            f.write(b"\x00" * 80)  # Header
            f.write(struct.pack("<I", 1))  # 1 triangle

            # Triangle with normal (0, 0, 1)
            f.write(struct.pack("<3f", 0.0, 0.0, 1.0))  # Normal
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))  # v0
            f.write(struct.pack("<3f", 1.0, 0.0, 0.0))  # v1
            f.write(struct.pack("<3f", 0.5, 1.0, 0.0))  # v2
            f.write(struct.pack("<H", 0))  # Attribute byte count
    else:
        with open(path, "w") as f:
            f.write("solid triangle\n")
            f.write("  facet normal 0 0 1\n")
            f.write("    outer loop\n")
            f.write("      vertex 0 0 0\n")
            f.write("      vertex 1 0 0\n")
            f.write("      vertex 0.5 1 0\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
            f.write("endsolid triangle\n")


def generate_triangle_ply(path: str, binary: bool = True) -> None:
    """Write a single triangle mesh in PLY format."""
    import struct

    if binary:
        with open(path, "wb") as f:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(b"element vertex 3\n")
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"element face 1\n")
            f.write(b"property list uchar int vertex_indices\n")
            f.write(b"end_header\n")

            # Vertices
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            f.write(struct.pack("<3f", 1.0, 0.0, 0.0))
            f.write(struct.pack("<3f", 0.5, 1.0, 0.0))

            # Face
            f.write(struct.pack("<B", 3))  # 3 vertices
            f.write(struct.pack("<3i", 0, 1, 2))
    else:
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex 3\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("element face 1\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            f.write("0 0 0\n")
            f.write("1 0 0\n")
            f.write("0.5 1 0\n")
            f.write("3 0 1 2\n")


def generate_cube_obj(path: str, use_quads: bool = False) -> None:
    """Write a cube mesh in OBJ format.

    Args:
        path: Output file path.
        use_quads: If True, write 6 quad faces. If False, write 12 triangles.
    """
    with open(path, "w") as f:
        f.write("# Cube mesh for testing\n")

        # 8 vertices of a cube
        vertices = [
            (0.0, 0.0, 0.0),  # 0
            (1.0, 0.0, 0.0),  # 1
            (1.0, 1.0, 0.0),  # 2
            (0.0, 1.0, 0.0),  # 3
            (0.0, 0.0, 1.0),  # 4
            (1.0, 0.0, 1.0),  # 5
            (1.0, 1.0, 1.0),  # 6
            (0.0, 1.0, 1.0),  # 7
        ]

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        if use_quads:
            # 6 quad faces
            # Bottom (z=0): 0 1 2 3
            f.write("f 1 2 3 4\n")
            # Top (z=1): 4 5 6 7 (indices 5-8)
            f.write("f 5 6 7 8\n")
            # Front (y=0): 0 1 5 4
            f.write("f 1 2 6 5\n")
            # Back (y=1): 3 2 6 7
            f.write("f 4 3 7 8\n")
            # Left (x=0): 0 3 7 4
            f.write("f 1 4 8 5\n")
            # Right (x=1): 1 2 6 5
            f.write("f 2 3 7 6\n")
        else:
            # 12 triangular faces (2 per quad)
            # Bottom
            f.write("f 1 2 3\n")
            f.write("f 1 3 4\n")
            # Top
            f.write("f 5 6 7\n")
            f.write("f 5 7 8\n")
            # Front
            f.write("f 1 2 6\n")
            f.write("f 1 6 5\n")
            # Back
            f.write("f 4 3 7\n")
            f.write("f 4 7 8\n")
            # Left
            f.write("f 1 4 8\n")
            f.write("f 1 8 5\n")
            # Right
            f.write("f 2 3 7\n")
            f.write("f 2 7 6\n")


def generate_cube_stl(path: str, binary: bool = True) -> None:
    """Write a cube mesh in STL format (12 triangles)."""
    import struct

    # Cube vertices
    v = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]

    # 12 triangles (2 per face)
    faces = [
        # Bottom (z=0, normal: 0, 0, -1)
        ((0, 1, 2), (0.0, 0.0, -1.0)),
        ((0, 2, 3), (0.0, 0.0, -1.0)),
        # Top (z=1, normal: 0, 0, 1)
        ((4, 7, 6), (0.0, 0.0, 1.0)),
        ((4, 6, 5), (0.0, 0.0, 1.0)),
        # Front (y=0, normal: 0, -1, 0)
        ((0, 4, 5), (0.0, -1.0, 0.0)),
        ((0, 5, 1), (0.0, -1.0, 0.0)),
        # Back (y=1, normal: 0, 1, 0)
        ((3, 2, 6), (0.0, 1.0, 0.0)),
        ((3, 6, 7), (0.0, 1.0, 0.0)),
        # Left (x=0, normal: -1, 0, 0)
        ((0, 3, 7), (-1.0, 0.0, 0.0)),
        ((0, 7, 4), (-1.0, 0.0, 0.0)),
        # Right (x=1, normal: 1, 0, 0)
        ((1, 5, 6), (1.0, 0.0, 0.0)),
        ((1, 6, 2), (1.0, 0.0, 0.0)),
    ]

    if binary:
        with open(path, "wb") as f:
            f.write(b"\x00" * 80)  # Header
            f.write(struct.pack("<I", len(faces)))  # 12 triangles

            for face_indices, normal in faces:
                f.write(struct.pack("<3f", normal[0], normal[1], normal[2]))
                for idx in face_indices:
                    f.write(struct.pack("<3f", v[idx][0], v[idx][1], v[idx][2]))
                f.write(struct.pack("<H", 0))  # Attribute byte count
    else:
        with open(path, "w") as f:
            f.write("solid cube\n")
            for face_indices, normal in faces:
                f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                f.write("    outer loop\n")
                for idx in face_indices:
                    f.write(f"      vertex {v[idx][0]} {v[idx][1]} {v[idx][2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid cube\n")


def generate_cube_ply(path: str, binary: bool = True) -> None:
    """Write a cube mesh in PLY format (12 triangles)."""
    import struct

    # 8 vertices
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]

    # 12 triangles
    faces = [
        (0, 1, 2), (0, 2, 3),  # Bottom
        (4, 7, 6), (4, 6, 5),  # Top
        (0, 4, 5), (0, 5, 1),  # Front
        (3, 2, 6), (3, 6, 7),  # Back
        (0, 3, 7), (0, 7, 4),  # Left
        (1, 5, 6), (1, 6, 2),  # Right
    ]

    if binary:
        with open(path, "wb") as f:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {len(vertices)}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(f"element face {len(faces)}\n".encode())
            f.write(b"property list uchar int vertex_indices\n")
            f.write(b"end_header\n")

            # Write vertices
            for v in vertices:
                f.write(struct.pack("<3f", v[0], v[1], v[2]))

            # Write faces
            for face in faces:
                f.write(struct.pack("<B", 3))  # 3 vertices
                f.write(struct.pack("<3i", face[0], face[1], face[2]))
    else:
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")

            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def generate_negative_indices_obj(path: str) -> None:
    """Write a triangle mesh using negative indices."""
    with open(path, "w") as f:
        f.write("# Triangle with negative indices for testing\n")
        f.write("v 0.0 0.0 0.0\n")
        f.write("v 1.0 0.0 0.0\n")
        f.write("v 0.5 1.0 0.0\n")
        # Negative indices: -1 = last vertex, -2 = second to last, etc.
        f.write("f -3 -2 -1\n")


def generate_inverted_cube_obj(path: str) -> None:
    """Write a cube with clockwise winding order (inverted normals)."""
    with open(path, "w") as f:
        f.write("# Cube with inverted winding order for testing\n")

        # Same vertices as cube
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ]

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Same faces but reversed order (clockwise instead of counter-clockwise)
        # Bottom
        f.write("f 1 3 2\n")
        f.write("f 1 4 3\n")
        # Top
        f.write("f 5 7 6\n")
        f.write("f 5 8 7\n")
        # Front
        f.write("f 1 6 2\n")
        f.write("f 1 5 6\n")
        # Back
        f.write("f 4 7 3\n")
        f.write("f 4 8 7\n")
        # Left
        f.write("f 1 8 4\n")
        f.write("f 1 5 8\n")
        # Right
        f.write("f 1 7 2\n")
        f.write("f 2 7 6\n")


def main():
    """Generate all test mesh assets."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Create output directory
    assets_dir = script_dir
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating test mesh assets in: {assets_dir}")

    # Triangle meshes
    print("Generating triangle meshes...")
    generate_triangle_obj(str(assets_dir / "triangle.obj"))
    generate_triangle_stl(str(assets_dir / "triangle.stl"), binary=True)
    generate_triangle_stl(str(assets_dir / "triangle_ascii.stl"), binary=False)
    generate_triangle_ply(str(assets_dir / "triangle.ply"), binary=True)
    generate_triangle_ply(str(assets_dir / "triangle_ascii.ply"), binary=False)

    # Cube meshes
    print("Generating cube meshes...")
    generate_cube_obj(str(assets_dir / "cube.obj"), use_quads=False)
    generate_cube_obj(str(assets_dir / "cube_quads.obj"), use_quads=True)
    generate_cube_stl(str(assets_dir / "cube.stl"), binary=True)
    generate_cube_stl(str(assets_dir / "cube_ascii.stl"), binary=False)
    generate_cube_ply(str(assets_dir / "cube.ply"), binary=True)
    generate_cube_ply(str(assets_dir / "cube_ascii.ply"), binary=False)

    # Special test meshes
    print("Generating special test meshes...")
    generate_negative_indices_obj(str(assets_dir / "negative_indices.obj"))
    generate_inverted_cube_obj(str(assets_dir / "inverted_cube.obj"))

    print("Done!")


if __name__ == "__main__":
    main()
