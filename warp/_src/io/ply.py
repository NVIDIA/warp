# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PLY (Polygon File Format) parser."""

from __future__ import annotations

# Standard library
import os
import struct

# Third-party
import numpy as np

from warp._src.io.mesh import MeshData, _apply_flip_winding


def _read_ply_header(filename: str) -> dict:
    """Read PLY header and return metadata.

    Args:
        filename: Path to the PLY file.

    Returns:
        Dictionary with header information including format, vertex_count,
        face_count, vertex_properties, has_colors, and face_index_types.
    """
    with open(filename, "rb") as f:
        # Read magic number
        magic = f.readline().decode("ascii").strip()
        if magic != "ply":
            raise RuntimeError(f"Invalid PLY file: '{filename}' (missing magic number)")

        header_info = {
            "format": None,
            "vertex_count": 0,
            "face_count": 0,
            "vertex_properties": [],
            "has_colors": False,
            "has_normals": False,
            "face_index_count_type": None,  # Type for vertex count per face
            "face_index_value_type": None,  # Type for vertex indices
        }

        current_element = None
        vertex_prop_names = []

        while True:
            raw_line = f.readline()
            if raw_line == b"":
                raise RuntimeError(f"Unexpected EOF before PLY end_header: '{filename}'")
            line = raw_line.decode("ascii").strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            keyword = parts[0]

            if keyword == "end_header":
                break
            elif keyword == "format":
                header_info["format"] = parts[1]
            elif keyword == "element":
                current_element = parts[1]
                if parts[1] == "vertex":
                    header_info["vertex_count"] = int(parts[2])
                elif parts[1] == "face":
                    header_info["face_count"] = int(parts[2])
            elif keyword == "property":
                if current_element == "vertex" and len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    header_info["vertex_properties"].append((prop_type, prop_name))
                    vertex_prop_names.append(prop_name)
                elif (
                    current_element == "face"
                    and len(parts) >= 5
                    and parts[1] == "list"
                    and parts[4] in ("vertex_indices", "vertex_index")
                ):
                    # property list <count_type> <value_type> vertex_indices
                    header_info["face_index_count_type"] = parts[2]
                    header_info["face_index_value_type"] = parts[3]

        # Check if we have complete normal/color sets (all components required)
        header_info["has_normals"] = all(name in vertex_prop_names for name in ["nx", "ny", "nz"])
        header_info["has_colors"] = all(name in vertex_prop_names for name in ["red", "green", "blue"])

    return header_info


def _read_ply_ascii(filename: str, header: dict) -> MeshData:
    """Read ASCII PLY file.

    Args:
        filename: Path to the PLY file.
        header: Parsed header information.

    Returns:
        MeshData containing points, indices, and optional normals/colors.
    """
    points = []
    normals = []
    colors = []
    indices = []

    with open(filename, "rb") as f:
        # Skip to end of header
        while True:
            raw_line = f.readline()
            if raw_line == b"":
                raise RuntimeError(f"Unexpected EOF before PLY end_header: '{filename}'")
            line = raw_line.decode("ascii").strip()
            if line == "end_header":
                break

        # Use has_normals/has_colors from header (requires full component sets)
        has_normals = header.get("has_normals", False)
        has_colors = header.get("has_colors", False)

        # Pre-allocate arrays for normals/colors if needed
        if has_normals:
            normals = [[0.0, 0.0, 0.0] for _ in range(header["vertex_count"])]
        if has_colors:
            colors = [[0, 0, 0] for _ in range(header["vertex_count"])]

        # Validate required vertex position properties (first 3 should be x, y, z)
        if len(header["vertex_properties"]) < 3:
            raise RuntimeError(
                f"PLY file has insufficient vertex properties. Expected at least x, y, z. File: '{filename}'"
            )
        first_three_names = [p[1] for p in header["vertex_properties"][:3]]
        if first_three_names != ["x", "y", "z"]:
            raise RuntimeError(
                f"PLY file vertex properties must start with x, y, z. Got: {first_three_names}. File: '{filename}'"
            )

        # Read vertices
        for vertex_idx in range(header["vertex_count"]):
            line = f.readline().decode("ascii").strip()
            values = line.split()
            if len(values) < 3:
                raise RuntimeError(
                    f"Malformed vertex data at index {vertex_idx}: expected at least 3 values, got {len(values)}. Line: '{line}'. File: '{filename}'"
                )
            points.append([float(values[0]), float(values[1]), float(values[2])])

            # Process additional properties (normals, colors)
            idx = 3
            for _prop_type, prop_name in header["vertex_properties"][3:]:
                if idx < len(values):
                    if prop_name == "nx" and has_normals:
                        normals[vertex_idx][0] = float(values[idx])
                    elif prop_name == "ny" and has_normals:
                        normals[vertex_idx][1] = float(values[idx])
                    elif prop_name == "nz" and has_normals:
                        normals[vertex_idx][2] = float(values[idx])
                    elif prop_name == "red" and has_colors:
                        colors[vertex_idx][0] = int(values[idx])
                    elif prop_name == "green" and has_colors:
                        colors[vertex_idx][1] = int(values[idx])
                    elif prop_name == "blue" and has_colors:
                        colors[vertex_idx][2] = int(values[idx])
                    elif prop_name == "alpha" and has_colors:
                        # alpha is stored but we only keep RGB in the 3-element array
                        pass
                idx += 1

        # Read faces
        for _ in range(header["face_count"]):
            line = f.readline().decode("ascii").strip()
            values = list(map(int, line.split()))
            num_verts = values[0]
            face_verts = values[1:]

            # Fan triangulation for n-gons
            if num_verts >= 3:
                for i in range(1, num_verts - 1):
                    indices.extend([face_verts[0], face_verts[i], face_verts[i + 1]])

    if not points:
        raise RuntimeError(f"No vertices found in PLY file: '{filename}'")

    return MeshData(
        points=np.array(points, dtype=np.float32),
        indices=np.array(indices, dtype=np.int32),
        normals=np.array(normals, dtype=np.float32) if normals else None,
        colors=np.array(colors, dtype=np.uint8) if colors else None,
    )


def _read_ply_binary(filename: str, header: dict) -> MeshData:
    """Read binary PLY file (little-endian or big-endian).

    Args:
        filename: Path to the PLY file.
        header: Parsed header information.

    Returns:
        MeshData containing points, indices, and optional normals/colors.
    """
    is_little_endian = header["format"] == "binary_little_endian"
    endian = "<" if is_little_endian else ">"

    # Type mapping for struct
    type_map = {
        "char": "b",
        "uchar": "B",
        "short": "h",
        "ushort": "H",
        "int": "i",
        "int8": "b",
        "int16": "h",
        "int32": "i",
        "uint": "I",
        "uint8": "B",
        "uint16": "H",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "float64": "d",
        "double": "d",
    }

    # Build vertex format string
    vertex_format = ""
    vertex_size = 0
    prop_offsets = {}
    prop_index = 0  # Tuple index in unpacked data, not byte offset

    for prop_type, prop_name in header["vertex_properties"]:
        if prop_type in type_map:
            fmt_char = type_map[prop_type]
            vertex_format += fmt_char
            prop_offsets[prop_name] = prop_index
            prop_index += 1
            vertex_size += struct.calcsize(fmt_char)

    with open(filename, "rb") as f:
        # Skip to end of header
        while True:
            raw_line = f.readline()
            if raw_line == b"":
                raise RuntimeError(f"Unexpected EOF before PLY end_header: '{filename}'")
            line = raw_line.decode("ascii").strip()
            if line == "end_header":
                break

        # Pre-allocate arrays
        vertex_count = header["vertex_count"]
        points = np.empty((vertex_count, 3), dtype=np.float32)
        normals = None
        colors = None

        # Use has_normals/has_colors from header (requires full component sets)
        if header.get("has_normals"):
            if all(name in prop_offsets for name in ["nx", "ny", "nz"]):
                normals = np.empty((vertex_count, 3), dtype=np.float32)

        if header.get("has_colors"):
            if all(name in prop_offsets for name in ["red", "green", "blue"]):
                colors = np.empty((vertex_count, 3), dtype=np.uint8)

        # Validate required vertex position properties
        required_props = ["x", "y", "z"]
        missing_props = [p for p in required_props if p not in prop_offsets]
        if missing_props:
            raise RuntimeError(
                f"PLY file is missing required vertex position properties: {missing_props}. File: '{filename}'"
            )

        # Read vertices
        vertex_struct = struct.Struct(endian + vertex_format)
        for i in range(vertex_count):
            data = vertex_struct.unpack(f.read(vertex_size))
            # Only access properties that exist in the header
            points[i] = [data[prop_offsets["x"]], data[prop_offsets["y"]], data[prop_offsets["z"]]]

            if normals is not None:
                nx_idx = prop_offsets.get("nx")
                ny_idx = prop_offsets.get("ny")
                nz_idx = prop_offsets.get("nz")
                if nx_idx is not None and ny_idx is not None and nz_idx is not None:
                    normals[i] = [data[nx_idx], data[ny_idx], data[nz_idx]]

            if colors is not None:
                red_idx = prop_offsets.get("red")
                green_idx = prop_offsets.get("green")
                blue_idx = prop_offsets.get("blue")
                if red_idx is not None and green_idx is not None and blue_idx is not None:
                    colors[i] = [data[red_idx], data[green_idx], data[blue_idx]]

        # Read faces
        indices = []
        face_count = header["face_count"]

        # Get face index types from header, default to uchar count + int indices
        count_type = header.get("face_index_count_type") or "uchar"
        value_type = header.get("face_index_value_type") or "int"

        # Map PLY types to struct format characters
        ply_to_struct = {
            "char": "b",
            "uchar": "B",
            "short": "h",
            "ushort": "H",
            "int": "i",
            "int8": "b",
            "int16": "h",
            "int32": "i",
            "uint": "I",
            "uint8": "B",
            "uint16": "H",
            "uint32": "I",
        }

        count_fmt = ply_to_struct.get(count_type, "B")
        idx_fmt = ply_to_struct.get(value_type, "i")

        count_format = endian + count_fmt
        count_size = struct.calcsize(count_format)
        idx_format = endian + idx_fmt
        idx_size = struct.calcsize(idx_format)

        # For faces, format is: <count_type> count + <value_type> vertex_indices
        for face_idx in range(face_count):
            count_bytes = f.read(count_size)
            if not count_bytes or len(count_bytes) < count_size:
                raise RuntimeError(f"Unexpected EOF while reading face {face_idx} in PLY file: '{filename}'")
            num_verts = struct.unpack(count_format, count_bytes)[0]

            # Read vertex indices using the type from header
            face_verts = []
            for _ in range(num_verts):
                idx = struct.unpack(idx_format, f.read(idx_size))[0]
                face_verts.append(idx)

            # Fan triangulation
            if num_verts >= 3:
                for i in range(1, num_verts - 1):
                    indices.extend([face_verts[0], face_verts[i], face_verts[i + 1]])

    return MeshData(
        points=points,
        indices=np.array(indices, dtype=np.int32),
        normals=normals,
        colors=colors,
    )


def read_ply(filename: str, flip_winding: bool = False) -> MeshData:
    """Read a mesh from a PLY file (binary or ASCII).

    Supports ASCII, binary little-endian, and binary big-endian formats.
    Automatically detects vertex properties including normals and colors.

    Args:
        filename: Path to the PLY file.
        flip_winding: If True, reverse triangle winding order.

    Returns:
        MeshData containing points, indices, and optional normals/colors.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be parsed or contains no vertices.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"PLY file not found: '{filename}'")

    header = _read_ply_header(filename)

    if header["vertex_count"] == 0:
        raise RuntimeError(f"No vertices found in PLY file: '{filename}'")

    # Dispatch to format-specific reader
    if header["format"] == "ascii":
        data = _read_ply_ascii(filename, header)
    elif header["format"] in ("binary_little_endian", "binary_big_endian"):
        data = _read_ply_binary(filename, header)
    else:
        raise RuntimeError(f"Unsupported PLY format: '{header['format']}'")

    if flip_winding:
        data.indices, data.normals = _apply_flip_winding(data.indices, data.normals)

    return data


def write_ply(
    points: np.ndarray,
    indices: np.ndarray,
    filename: str,
    binary: bool = True,
    normals: np.ndarray | None = None,
    colors: np.ndarray | None = None,
) -> None:
    """Write a mesh to a PLY file.

    Args:
        points: Vertex positions, shape (N, 3).
        indices: Triangle indices, shape (M * 3,).
        filename: Output file path.
        binary: If True, write binary PLY (default). If False, write ASCII.
        normals: Optional vertex normals, shape (N, 3).
        colors: Optional vertex colors, shape (N, 3) or (N, 4).
    """
    indices_reshaped = indices.reshape(-1, 3)
    num_verts = len(points)
    num_faces = len(indices_reshaped)

    has_normals = normals is not None
    has_colors = colors is not None

    if binary:
        with open(filename, "wb") as f:
            # Write header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")

            # Vertex properties
            f.write(f"element vertex {num_verts}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            if has_normals:
                f.write(b"property float nx\n")
                f.write(b"property float ny\n")
                f.write(b"property float nz\n")
            if has_colors:
                f.write(b"property uchar red\n")
                f.write(b"property uchar green\n")
                f.write(b"property uchar blue\n")

            # Face properties
            f.write(f"element face {num_faces}\n".encode())
            f.write(b"property list uchar int vertex_indices\n")
            f.write(b"end_header\n")

            # Write vertices
            # Build format string with single byte-order prefix at start
            vertex_format = "<"
            vertex_format += "3f"  # x, y, z
            if has_normals:
                vertex_format += "3f"  # nx, ny, nz (no additional prefix)
            if has_colors:
                vertex_format += "3B"  # red, green, blue (no additional prefix)
            vertex_struct = struct.Struct(vertex_format)

            for i in range(num_verts):
                p = points[i]
                data = [p[0], p[1], p[2]]
                if has_normals:
                    n = normals[i]
                    data.extend([n[0], n[1], n[2]])
                if has_colors:
                    c = colors[i]
                    data.extend([int(c[0]), int(c[1]), int(c[2])])
                f.write(vertex_struct.pack(*data))

            # Write faces
            for face in indices_reshaped:
                f.write(struct.pack("<B", 3))  # 3 vertices per face
                f.write(struct.pack("<3i", face[0], face[1], face[2]))
    else:
        with open(filename, "w") as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")

            # Vertex properties
            f.write(f"element vertex {num_verts}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_normals:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
            if has_colors:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            # Face properties
            f.write(f"element face {num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Write vertices
            for i in range(num_verts):
                p = points[i]
                line = f"{p[0]} {p[1]} {p[2]}"
                if has_normals:
                    n = normals[i]
                    line += f" {n[0]} {n[1]} {n[2]}"
                if has_colors:
                    c = colors[i]
                    line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
                f.write(line + "\n")

            # Write faces
            for face in indices_reshaped:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
