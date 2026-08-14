# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Sparse Marching Cubes -- Polyscope comparison renders
#
# Companion to example_sparse_marching_cubes.py that renders figures with
# Polyscope, including a headless (EGL) path suitable for generating
# figures on a server. The implicit function is a signed distance field to
# the Stanford bunny mesh (evaluated on the GPU with a winding-number sign).
#
# It produces:
#   *_refine.gif        the surface and the sparse octree cells hugging it,
#                       as the octree depth increases (resolution refines)
#   *_surface.png       the extracted isosurface
#   *_dense_grid.png    a dense grid of cells at a coarse resolution
#   *_sparse_octree.png only the near-surface octree cells at that same
#                       resolution -- the thin shell the sparse method
#                       actually instantiates
#
# This is a visualization utility, not part of the Warp example test
# suite. It requires the third-party ``polyscope`` and ``pillow`` packages
# (and ``usd-core`` for the bunny asset):
#
#   uv run --with polyscope --with pillow --with usd-core \
#       -m warp.examples.core.example_sparse_marching_cubes_polyscope
#
# Uses wp.sparse_marching_cubes and the public wp.lipschitz_octree.
###########################################################################

import argparse
import os

import numpy as np

import warp as wp
import warp.examples

# Cubic domain covering the normalized mesh (fits in [-0.5, 0.5] plus padding).
ORIGIN = (-0.65, -0.65, -0.65)
ROOT_WIDTH = 1.3

# Unit-cube corner offsets and the 12 triangles (2 per face) used to turn a
# list of cell origins into a single voxel mesh.
_CUBE_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64
)
_CUBE_FACES = np.array(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7],
    ],
    dtype=np.int32,
)  # fmt: skip


@wp.kernel(enable_backward=False)
def _mesh_sdf_kernel(mesh_id: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32)):
    """Signed distance to a mesh, with the sign from the generalized winding number."""
    i = wp.tid()
    p = points[i]
    query = wp.mesh_query_point_sign_winding_number(mesh_id, p, 1.0e6)
    closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    values[i] = query.sign * wp.length(p - closest)


def load_bunny(device):
    """Load and normalize the bunny asset, returning a Warp mesh and a batched SDF."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415 -- optional dependency

    stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bunny"))
    points = np.array(geom.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    # Center and scale so the longest axis spans one unit, centered on the origin.
    center = 0.5 * (points.min(0) + points.max(0))
    points = (points - center) / (points.max(0) - points.min(0)).max()

    mesh = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=wp.int32, device=device),
    )

    def evaluate(query_points: wp.array) -> wp.array:
        values = wp.empty(query_points.shape[0], dtype=wp.float32, device=query_points.device)
        wp.launch(
            _mesh_sdf_kernel,
            dim=query_points.shape[0],
            inputs=[mesh.id, query_points],
            outputs=[values],
            device=query_points.device,
        )
        return values

    return mesh, evaluate


def voxel_mesh(origins, width):
    """Build one (V, F) triangle mesh containing a cube per cell origin."""
    n = origins.shape[0]
    verts = (origins[:, None, :] + width * _CUBE_CORNERS[None, :, :]).reshape(-1, 3)
    faces = (_CUBE_FACES[None, :, :] + 8 * np.arange(n)[:, None, None]).reshape(-1, 3)
    return verts, faces


SURFACE_COLOR = (0.86, 0.72, 0.45)
VOXEL_COLOR = (0.30, 0.55, 0.90)
CAMERA_EYE = (1.55, 1.15, 1.55)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to write the outputs into.")
    parser.add_argument("--prefix", type=str, default="sparse_marching_cubes", help="Output file name prefix.")
    parser.add_argument("--resolution", type=int, default=1024, help="Square render resolution in pixels.")
    parser.add_argument("--surface-depth", type=int, default=8, help="Octree depth for the surface still.")
    parser.add_argument("--grid-depth", type=int, default=5, help="Coarse depth for the grid-occupancy comparison.")
    parser.add_argument("--gif-min-depth", type=int, default=3, help="First depth in the refinement GIF.")
    parser.add_argument("--gif-max-depth", type=int, default=9, help="Last depth in the refinement GIF.")
    args = parser.parse_known_args()[0]

    import polyscope as ps  # noqa: PLC0415 -- optional third-party dependency
    from PIL import Image  # noqa: PLC0415 -- optional third-party dependency

    device = wp.get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    def path(name):
        return os.path.join(args.out_dir, f"{args.prefix}_{name}")

    _mesh, evaluate = load_bunny(device)  # evaluate keeps the mesh alive via closure

    ps.set_allow_headless_backends(True)
    ps.init()
    ps.set_window_size(args.resolution, args.resolution)
    ps.set_ground_plane_mode("shadow_only")
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.set_SSAA_factor(3)
    ps.set_view_projection_mode("orthographic")
    ps.set_transparency_mode("pretty")  # honor per-structure opacity

    # Fix the scene bounds to the coarsest octree cells and disable automatic
    # extent recomputation, so the orthographic framing is identical across
    # every frame as the resolution refines (the coarsest cells enclose all
    # finer ones, so nothing is ever clipped).
    ps.set_automatically_compute_scene_extents(False)
    coarse_origins, coarse_width = wp.lipschitz_octree(evaluate, ORIGIN, ROOT_WIDTH, args.gif_min_depth, device=device)
    coarse = coarse_origins.numpy()
    box_lo = coarse.min(0).astype(np.float32)
    box_hi = (coarse.max(0) + coarse_width).astype(np.float32)
    ps.set_bounding_box(box_lo, box_hi)
    camera_target = tuple(((box_lo + box_hi) * 0.5).tolist())

    def look():
        ps.look_at(CAMERA_EYE, camera_target)

    def to_image(buffer):
        arr = np.asarray(buffer)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(arr[:, :, :3], "RGB")

    def extract(depth):
        verts, indices = wp.sparse_marching_cubes(evaluate, ORIGIN, ROOT_WIDTH, depth, device=device)
        return verts.numpy().astype(np.float64), indices.numpy().reshape(-1, 3)

    # -- Refinement GIF: surface + sparse voxel shell as depth increases ---
    # Frames are captured straight to memory buffers (no intermediate files).
    frames = []
    for depth in range(args.gif_min_depth, args.gif_max_depth + 1):
        surf_v, surf_f = extract(depth)
        cell_origins, cell_width = wp.lipschitz_octree(evaluate, ORIGIN, ROOT_WIDTH, depth, device=device)
        vox_v, vox_f = voxel_mesh(cell_origins.numpy().astype(np.float64), cell_width)

        ps.remove_all_structures()
        sm = ps.register_surface_mesh("surface", surf_v, surf_f, smooth_shade=True)
        sm.set_color(SURFACE_COLOR)
        # Render the octree cells as a translucent blue cage over the opaque
        # surface, so the adaptive grid reads clearly and its refinement shows.
        vm = ps.register_surface_mesh("octree cells", vox_v, vox_f, edge_width=1.0)
        vm.set_color(VOXEL_COLOR)
        vm.set_transparency(0.3)
        look()
        frames.append(to_image(ps.screenshot_to_buffer(transparent_bg=False)))
        print(f"depth {depth}: {surf_f.shape[0]} tris, {cell_origins.shape[0]} leaf cells")

    # Ping-pong so the loop refines then coarsens smoothly, holding the ends.
    sequence = frames + frames[-1:] * 2 + frames[-2:0:-1] + frames[:1] * 2
    durations = [650] * len(frames) + [900] * 2 + [650] * (len(frames) - 2) + [900] * 2
    frames[0].save(
        path("refine.gif"),
        save_all=True,
        append_images=sequence[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    print(f"wrote {path('refine.gif')}")

    # -- Comparison stills -------------------------------------------------
    # Let each still frame itself to its own contents (the dense grid fills the
    # whole domain, the sparse shell only the near-surface band).
    ps.set_automatically_compute_scene_extents(True)
    camera_target = (0.0, 0.0, 0.0)

    surf_v, surf_f = extract(args.surface_depth)
    ps.remove_all_structures()
    sm = ps.register_surface_mesh("surface", surf_v, surf_f, smooth_shade=True)
    sm.set_color(SURFACE_COLOR)
    look()
    ps.screenshot(path("surface.png"), transparent_bg=False)

    res = 1 << args.grid_depth
    cell_origins, cell_width = wp.lipschitz_octree(evaluate, ORIGIN, ROOT_WIDTH, args.grid_depth, device=device)
    sparse_v, sparse_f = voxel_mesh(cell_origins.numpy().astype(np.float64), cell_width)

    axis = np.arange(res)
    gi, gj, gk = np.meshgrid(axis, axis, axis, indexing="ij")
    dense_origins = np.array(ORIGIN) + cell_width * np.stack([gi, gj, gk], axis=-1).reshape(-1, 3).astype(np.float64)
    dense_v, dense_f = voxel_mesh(dense_origins, cell_width)
    print(f"depth {args.grid_depth}: dense {dense_origins.shape[0]} cells vs sparse {cell_origins.shape[0]} cells")

    csurf_v, csurf_f = extract(args.grid_depth)

    ps.remove_all_structures()
    dg = ps.register_surface_mesh("dense grid", dense_v, dense_f)
    dg.set_color((0.85, 0.85, 0.88))
    dg.set_transparency(0.35)
    inner = ps.register_surface_mesh("surface", csurf_v, csurf_f, smooth_shade=True)
    inner.set_color(SURFACE_COLOR)
    look()
    ps.screenshot(path("dense_grid.png"), transparent_bg=False)

    ps.remove_all_structures()
    shell = ps.register_surface_mesh("sparse octree", sparse_v, sparse_f, edge_width=1.0)
    shell.set_color(VOXEL_COLOR)
    shell.set_transparency(0.55)
    inner = ps.register_surface_mesh("surface", csurf_v, csurf_f, smooth_shade=True)
    inner.set_color(SURFACE_COLOR)
    look()
    ps.screenshot(path("sparse_octree.png"), transparent_bg=False)
    print(f"wrote {path('surface.png')}, {path('dense_grid.png')}, {path('sparse_octree.png')}")


if __name__ == "__main__":
    main()
