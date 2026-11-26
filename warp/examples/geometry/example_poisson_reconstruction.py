# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct an oriented point cloud and extract its surface with sparse marching cubes.

Run headlessly with ``python -m warp.examples.geometry.example_poisson_reconstruction``.
Supply ``--stage-path surface.usd`` to save a mesh using the optional USD renderer.
"""

import numpy as np

import warp as wp
import warp.geometry as geometry


class Example:
    def __init__(self, count=2048, depth=5, resolution=None):
        index = np.arange(count, dtype=np.float64) + 0.5
        z = 1.0 - 2.0 * index / count
        angle = index * (np.pi * (3.0 - np.sqrt(5.0)))
        radius = np.sqrt(1.0 - z * z)
        sphere = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
        axes = np.array([1.0, 0.8, 0.6])
        points = wp.array(sphere * axes, dtype=wp.vec3)
        normals = wp.array(sphere / axes, dtype=wp.vec3)
        with wp.ScopedTimer("reconstruct"):
            self.result = geometry.poisson_reconstruct(points, normals, depth=depth)
        with wp.ScopedTimer("extract mesh"):
            self.vertices, self.indices = geometry.poisson_extract_mesh(
                self.result,
                resolution=resolution,
            )

    def save(self, path):
        """Save the reconstructed surface as a single USD frame."""
        import warp.render  # noqa: PLC0415

        renderer = warp.render.UsdRenderer(path)
        renderer.begin_frame(0.0)
        renderer.render_mesh("surface", self.vertices.numpy(), self.indices.numpy(), colors=(0.25, 0.6, 0.9))
        renderer.end_frame()
        renderer.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="cuda:0", help="CUDA device for reconstruction.")
    parser.add_argument("--count", type=int, default=2048, help="Number of oriented input points.")
    parser.add_argument("--depth", type=int, default=5, help="Maximum reconstruction depth.")
    parser.add_argument("--resolution", type=int, default=None, help="Optional native grid node count: 2**depth + 1.")
    parser.add_argument("--stage-path", default=None, help="Optional USD output path.")
    args = parser.parse_args()
    with wp.ScopedDevice(args.device):
        example = Example(args.count, args.depth, args.resolution)
        if args.stage_path:
            example.save(args.stage_path)
        print(f"Extracted {len(example.vertices)} vertices and {len(example.indices) // 3} triangles.")
