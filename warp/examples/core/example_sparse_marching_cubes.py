# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sparse Marching Cubes
#
# Extracts an isosurface directly from an implicit function using a
# Lipschitz octree, without ever building a dense grid. Only voxels near
# the level set are instantiated, so the cost scales with surface area
# rather than volume. This mirrors the libigl "Lipschitz octree" tutorial
# (tutorial 1001), but runs the whole pipeline on the GPU in pure Warp.
#
# The implicit function here is a signed distance field to the Stanford
# bunny mesh (evaluated on the GPU with a winding-number sign), re-meshed
# every frame as it spins. Compare with example_marching_cubes.py, which
# extracts a surface from a dense field.
#
# Rendering:
#   * By default the surface is written to a USD stage (headless).
#   * Pass --opengl to open an interactive window (non-headless).
#   * Pass --mode dense to extract the identical surface with the dense
#     wp.geometry.IsoSurfaceMarchingCubes instead, for a comparison.
#
# Note: requires a CUDA-capable device for interactive resolutions, and
# usd-core to load the bunny asset.
###########################################################################

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.geometry
import warp.render


@wp.kernel(enable_backward=False)
def bunny_sdf_batch_kernel(
    mesh_id: wp.uint64, points: wp.array(dtype=wp.vec3), angle: float, values: wp.array(dtype=wp.float32)
):
    """Signed distance to the (spinning) bunny, evaluated over a batch of points."""
    i = wp.tid()
    p = points[i]
    c = wp.cos(angle)
    s = wp.sin(angle)
    rp = wp.vec3(c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2])
    q = wp.mesh_query_point_sign_winding_number(mesh_id, rp, 1.0e6)
    values[i] = q.sign * wp.length(rp - wp.mesh_eval_position(mesh_id, q.face, q.u, q.v))


@wp.kernel(enable_backward=False)
def bunny_field_kernel(mesh_id: wp.uint64, field: wp.array3d(dtype=float), origin: wp.vec3, h: float, angle: float):
    """Dense field sampler used by the dense extractor for comparison."""
    i, j, k = wp.tid()
    p = origin + h * wp.vec3(float(i), float(j), float(k))
    c = wp.cos(angle)
    s = wp.sin(angle)
    rp = wp.vec3(c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2])
    q = wp.mesh_query_point_sign_winding_number(mesh_id, rp, 1.0e6)
    field[i, j, k] = q.sign * wp.length(rp - wp.mesh_eval_position(mesh_id, q.face, q.u, q.v))


class Example:
    def __init__(self, stage_path="example_sparse_marching_cubes.usd", mode="sparse", opengl=False, verbose=False):
        self.verbose = verbose
        self.mode = mode

        # Cubic domain around the normalized bunny, and the octree depth. A
        # depth of 8 matches a dense grid of 256^3 cells (257^3 corner
        # evaluations), which the sparse octree avoids materializing.
        self.origin = wp.vec3(-0.65, -0.65, -0.65)
        self.root_width = 1.3
        self.max_depth = 8

        # Load and normalize the bunny so its longest axis spans one unit.
        stage = Usd.Stage.Open(warp.examples.get_asset_directory() + "/bunny.usd")
        geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bunny"))
        points = np.array(geom.GetPointsAttr().Get(), dtype=np.float32)
        indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        points = (points - 0.5 * (points.min(0) + points.max(0))) / (points.max(0) - points.min(0)).max()
        self.mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3),
            indices=wp.array(indices, dtype=wp.int32),
        )

        # Distinct colors so sparse/dense screenshots are easy to tell apart.
        self.color = (0.86, 0.72, 0.45) if mode == "sparse" else (0.9, 0.55, 0.2)

        self.fps = 60
        self.frame = 0

        self.verts = None
        self.indices = None

        self.opengl_renderer = None
        self.usd_renderer = None
        try:
            if opengl:
                self.opengl_renderer = warp.render.OpenGLRenderer(
                    title=f"Sparse Marching Cubes ({mode})",
                    screen_width=1024,
                    screen_height=1024,
                )
        except Exception as err:
            import warnings  # noqa: PLC0415

            warnings.warn(f"Could not initialize OpenGL renderer: {err}.", stacklevel=2)
        try:
            if stage_path:
                self.usd_renderer = warp.render.UsdRenderer(stage_path)
        except Exception as err:
            print(f"Could not initialize USD renderer '{stage_path}': {err}.")

    def _make_evaluator(self, angle):
        def evaluate(points: wp.array) -> wp.array:
            values = wp.empty(points.shape[0], dtype=wp.float32, device=points.device)
            wp.launch(
                bunny_sdf_batch_kernel,
                dim=points.shape[0],
                inputs=[self.mesh.id, points, float(angle)],
                outputs=[values],
                device=points.device,
            )
            return values

        return evaluate

    def _extract_sparse(self, angle):
        return wp.geometry.sparse_marching_cubes(
            self._make_evaluator(angle), self.origin, self.root_width, self.max_depth, threshold=0.0, return_stats=True
        )

    def _extract_dense(self, angle):
        resolution = 1 << self.max_depth
        n_nodes = resolution + 1
        h = self.root_width / resolution
        field = wp.empty((n_nodes, n_nodes, n_nodes), dtype=float)
        wp.launch(
            bunny_field_kernel, dim=field.shape, inputs=[self.mesh.id, field, self.origin, float(h), float(angle)]
        )
        upper = wp.vec3(
            self.origin[0] + self.root_width, self.origin[1] + self.root_width, self.origin[2] + self.root_width
        )
        verts, indices = wp.geometry.IsoSurfaceMarchingCubes.extract(
            field, threshold=0.0, domain_bounds_lower_corner=self.origin, domain_bounds_upper_corner=upper
        )
        stats = {"resolution": resolution, "leaf_cells": resolution**3, "sdf_evaluations": n_nodes**3}
        return verts, indices, stats

    def step(self):
        with wp.ScopedTimer("step"):
            angle = self.frame / self.fps  # radians; ~1 rad/s spin
            with wp.ScopedTimer(f"{self.mode.title()} Surface Extraction", active=self.verbose):
                if self.mode == "dense":
                    verts, indices, stats = self._extract_dense(angle)
                else:
                    verts, indices, stats = self._extract_sparse(angle)
                wp.synchronize_device()

            self.verts = verts
            self.indices = indices

            if self.verbose:
                resolution = stats["resolution"]
                dense_evals = (resolution + 1) ** 3
                print(
                    f"  frame {self.frame} [{self.mode}]: resolution {resolution}^3, "
                    f"{stats['leaf_cells']:,} cells, "
                    f"{stats['sdf_evaluations']:,} implicit evaluations "
                    f"({dense_evals:,} for a dense grid, "
                    f"{dense_evals / max(stats['sdf_evaluations'], 1):.1f}x fewer)"
                )

    def _render_to(self, renderer):
        renderer.begin_frame(self.frame / self.fps)
        renderer.render_mesh(
            "surface",
            self.verts.numpy(),
            self.indices.numpy(),
            colors=self.color,
            update_topology=True,
        )
        renderer.end_frame()

    def render(self):
        if self.usd_renderer is None and self.opengl_renderer is None:
            return
        with wp.ScopedTimer("render"):
            if self.usd_renderer is not None:
                self._render_to(self.usd_renderer)
            if self.opengl_renderer is not None:
                self._render_to(self.opengl_renderer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_sparse_marching_cubes.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=240, help="Total number of frames.")
    parser.add_argument(
        "--mode",
        choices=("sparse", "dense"),
        default="sparse",
        help="Extraction method. Both produce the same surface; use 'dense' to capture a comparison.",
    )
    parser.add_argument("--opengl", action="store_true", help="Open an interactive OpenGL window (non-headless).")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, mode=args.mode, opengl=args.opengl, verbose=args.verbose)
        for _ in range(args.num_frames):
            example.step()
            example.render()
            example.frame += 1

        if example.usd_renderer is not None:
            example.usd_renderer.save()

        n_verts = len(example.verts) if example.verts is not None else 0
        n_tris = len(example.indices) // 3 if example.indices is not None else 0
        print(f"[{args.mode}] Extracted {n_verts} vertices and {n_tris} triangles.")
