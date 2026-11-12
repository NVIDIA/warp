# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Geometry Oriented Bounding Box
#
# Fits an oriented bounding box (OBB) to the Stanford bunny with
# warp.geometry.oriented_bounding_box, which searches 4096 candidate
# orientations drawn from a Super-Fibonacci spiral over SO(3), plus the identity
# rotation and the point set's principal (PCA) axes, and keeps whichever gives
# the smallest box.
#
# The example fits a box under both objectives -- minimum volume and minimum
# surface area -- compares each against the axis-aligned bounding box (AABB) of
# the same mesh, shows how much the two extra candidates contribute, and prints
# how the fit tightens as the sample count grows. The mesh and both boxes are
# written to a USD stage, and can optionally be shown in an interactive
# polyscope viewer.
#
#   uv run --with usd-core warp/examples/geometry/example_oriented_bounding_box.py
#   uv run --with usd-core --with polyscope warp/examples/geometry/example_oriented_bounding_box.py --polyscope
###########################################################################

import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.geometry
import warp.render

# The eight corners of a unit cube centered on the origin, and the twelve edges
# joining them, used to draw a box from its transform and extents.
BOX_CORNERS = np.array([(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
BOX_EDGES = np.array([(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)])

# An arbitrary rotation, used to pose the squashed mesh off-axis.
_AXIS = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])
_K = np.array([[0, -_AXIS[2], _AXIS[1]], [_AXIS[2], 0, -_AXIS[0]], [-_AXIS[1], _AXIS[0], 0]])
ROTATION = np.eye(3) + np.sin(0.7) * _K + (1.0 - np.cos(0.7)) * _K @ _K

# Search parameters that keep only the best sampled orientation: one start, no refinement or spin rounds.
SAMPLED_ONLY = {"num_refine_starts": 1, "num_refine_iterations": 0, "num_spin_samples": 0}


def box_corners(xform: wp.transform, extents: wp.vec3) -> np.ndarray:
    """World-space positions of a box's eight corners."""
    local = BOX_CORNERS * np.array([extents[0], extents[1], extents[2]])
    return np.array([wp.transform_point(xform, wp.vec3(*p)) for p in local])


def host_obb(result):
    """Read the length-1 device arrays from ``oriented_bounding_box`` into host scalars."""
    xform_arr, extents_arr, measure_arr = result
    row = xform_arr.numpy()[0]
    xform = wp.transform(wp.vec3(*row[:3]), wp.quat(*row[3:]))
    extents = wp.vec3(*extents_arr.numpy()[0])
    return xform, extents, float(measure_arr.numpy()[0])


def main(stage_path="example_geometry_oriented_bounding_box.usd", show_polyscope=False):
    usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))
    points_np = np.array(usd_geom.GetPointsAttr().Get())
    points = wp.array(points_np, dtype=wp.vec3)
    indices = wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int)

    print(f"bunny: {len(points_np)} vertices, {len(indices) // 3} triangles\n")

    # The axis-aligned bounding box, as a baseline. Because the identity rotation is
    # included as a candidate by default, an OBB can never come out worse than this.
    aabb_extents = points_np.max(axis=0) - points_np.min(axis=0)
    aabb_volume = float(np.prod(aabb_extents))
    aabb_area = 2.0 * float(
        aabb_extents[0] * aabb_extents[1] + aabb_extents[1] * aabb_extents[2] + aabb_extents[0] * aabb_extents[2]
    )

    # Fit under each objective. Minimizing volume and minimizing surface area are
    # different problems and generally pick different orientations.
    Measure = wp.geometry.OBBMeasureType

    # Warm up so the reported timings measure the search rather than kernel loading.
    wp.geometry.oriented_bounding_box(points, Measure.VOLUME)
    wp.synchronize_device()

    with wp.ScopedTimer("oriented_bounding_box (volume)", print=False) as timer_vol:
        vol_result = wp.geometry.oriented_bounding_box(points, Measure.VOLUME)
        wp.synchronize_device()
    with wp.ScopedTimer("oriented_bounding_box (area)", print=False) as timer_area:
        area_result = wp.geometry.oriented_bounding_box(points, Measure.SURFACE_AREA)
        wp.synchronize_device()

    vol_xform, vol_extents, vol_measure = host_obb(vol_result)
    area_xform, area_extents, area_measure = host_obb(area_result)

    # Surface area of the volume-minimizing box, and vice versa, to show that each
    # objective wins on its own measure.
    vol_box_area = 2.0 * (
        vol_extents[0] * vol_extents[1] + vol_extents[1] * vol_extents[2] + vol_extents[0] * vol_extents[2]
    )
    area_box_volume = area_extents[0] * area_extents[1] * area_extents[2]

    print(f"{'fit':<26}{'volume':>12}{'surface area':>15}   extents")
    print("-" * 78)
    print(f"{'AABB':<26}{aabb_volume:>12.5f}{aabb_area:>15.5f}   {np.round(aabb_extents, 4)}")
    print(f"{'OBB (min volume)':<26}{vol_measure:>12.5f}{vol_box_area:>15.5f}   {np.round(np.array(vol_extents), 4)}")
    print(
        f"{'OBB (min surface area)':<26}{area_box_volume:>12.5f}{area_measure:>15.5f}   "
        f"{np.round(np.array(area_extents), 4)}"
    )
    print(
        f"\nvolume reduced to {vol_measure / aabb_volume:.1%} of the AABB "
        f"({timer_vol.elapsed:.2f} ms), surface area to {area_measure / aabb_area:.1%} "
        f"({timer_area.elapsed:.2f} ms)"
    )

    # What the two extra candidates contribute. On a roundish shape like the bunny
    # the spiral already finds a good orientation on its own, so the interesting
    # case is a squashed and rotated copy of the same mesh: the spiral resolves
    # orientation only to its sample spacing, and on a strongly flattened shape a
    # few degrees of error costs a lot of volume, while the principal axes land
    # almost exactly on the right frame.
    squash = np.array([1.0, 0.06, 1.0])
    flat_np = (points_np * squash) @ ROTATION.T
    flat = wp.array(flat_np, dtype=wp.vec3)
    flat_aabb_volume = float(np.prod(flat_np.max(axis=0) - flat_np.min(axis=0)))
    # Undoing ROTATION is achievable by some orientation, so the axis-aligned volume
    # of the unrotated squashed mesh is an upper bound on the true optimum.
    best_known = float(np.prod((points_np * squash).max(axis=0) - (points_np * squash).min(axis=0)))

    # Refinement is disabled here (one start, no refinement or spin rounds) to isolate what each candidate set
    # contributes to the *initial* search; with it on (the default) the coarse-to-fine local
    # search and spin rounds tighten all of these further and largely closes the gap between them.
    print(f"\nsquashed ({squash[1]:g}x in y) and rotated bunny -- candidate sets, minimum volume:")
    for label, kwargs in (
        ("spiral only", {"include_axis_aligned": False, "include_pca": False}),
        ("spiral + axis-aligned", {"include_axis_aligned": True, "include_pca": False}),
        ("spiral + PCA", {"include_axis_aligned": False, "include_pca": True}),
        ("all three (default)", {}),
    ):
        _, _, measure = host_obb(wp.geometry.oriented_bounding_box(flat, Measure.VOLUME, **SAMPLED_ONLY, **kwargs))
        print(
            f"  {label:<24} volume {measure:.6f}   ({measure / flat_aabb_volume:6.1%} of AABB, "
            f"{measure / best_known:.2f}x the best achievable)"
        )

    # Raw spiral convergence, refinement disabled: more orientations give a tighter box, with
    # diminishing returns. Note the sequence is not nested -- the sample set for 8192 is not a
    # superset of the one for 4096 -- so the volume can tick up slightly between neighboring
    # sizes even though the trend is downward. With refinement on, even 16 samples reach this
    # plateau, which is the point of the coarse-to-fine local search.
    print("\nvolume vs. number of sampled orientations (refinement off):")
    for num_samples in (16, 64, 256, 1024, 4096, 16384):
        _, _, measure = host_obb(
            wp.geometry.oriented_bounding_box(points, Measure.VOLUME, num_samples=num_samples, **SAMPLED_ONLY)
        )
        marker = "   <- default" if num_samples == 4096 else ""
        print(f"  {num_samples:>6} samples   volume {measure:.5f}   ({measure / aabb_volume:.1%} of AABB){marker}")

    if stage_path:
        renderer = wp.render.UsdRenderer(stage_path)
        renderer.begin_frame(0.0)
        renderer.render_mesh(
            "bunny",
            points=points_np,
            indices=usd_geom.GetFaceVertexIndicesAttr().Get(),
            colors=(0.75, 0.75, 0.8),
        )
        # render_box takes half-extents, and a quaternion in (x, y, z, w) order.
        for name, xform, extents, color in (
            ("obb_volume", vol_xform, vol_extents, (0.2, 0.8, 0.3)),
            ("obb_surface_area", area_xform, area_extents, (0.9, 0.5, 0.1)),
        ):
            renderer.render_box(
                name,
                pos=tuple(wp.transform_get_translation(xform)),
                rot=tuple(wp.transform_get_rotation(xform)),
                extents=tuple(np.array(extents) * 0.5),
                color=color,
            )
        renderer.end_frame()
        renderer.save()
        print(f"\nwrote {stage_path}")

    if show_polyscope:
        import polyscope as ps  # noqa: PLC0415
        import polyscope.imgui as psim  # noqa: PLC0415

        ps.init()
        ps.set_up_dir("y_up")
        ps.register_surface_mesh("bunny", points_np, indices.numpy().reshape(-1, 3))
        for name, xform, extents, color in (
            ("OBB (min volume)", vol_xform, vol_extents, (0.2, 0.8, 0.3)),
            ("OBB (min surface area)", area_xform, area_extents, (0.9, 0.5, 0.1)),
        ):
            net = ps.register_curve_network(name, box_corners(xform, extents), BOX_EDGES, radius=0.002)
            net.set_color(color)

        def callback():
            if psim.IsKeyPressed(psim.ImGuiKey_Escape):
                ps.unshow()

        ps.set_user_callback(callback)
        ps.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        "--stage_path",
        dest="stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_geometry_oriented_bounding_box.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--polyscope", action="store_true", help="Launch an interactive polyscope viewer.")
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        main(stage_path=args.stage_path, show_polyscope=args.polyscope)
