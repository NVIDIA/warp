# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Geometry Swept Volume
#
# Computes the swept volume (motion envelope) of an animated rigid assembly
# with warp.geometry.swept_volume_mesh: the single closed mesh that encloses the
# union of every input mesh over every sampled pose.
#
# The method samples a dense signed-distance field
#
#     D(p) = min_mesh min_sample  sdf_mesh( X[mesh, sample]^-1 p )
#
# by pushing each grid point back into every mesh's rest frame (the motion is
# rigid, so one closest-point query per pose is enough), then extracts the zero
# isosurface with marching cubes. This is the dense-stamping baseline: no root
# finding or narrow band, so motion *between* the sampled poses is not
# conservatively bounded. Sample finely enough for the tolerance you need.
#
# By default the example animates a procedural two-link arm, so it runs with no
# external assets. Pass --usd-path to run on an animated USD hierarchy instead,
# such as a UR10 arm.
#
# Inside/outside is classified with the generalized winding number, which is
# what warp.geometry.swept_volume_mesh() defaults to. Pass --sign-mode normal for the
# faster closest-face-normal classifier, which suits watertight input like the
# procedural arm but is incoherent on the open, non-watertight visual shells
# that CAD parts like the UR10 are made of (spurious interior pockets, hundreds
# of disconnected junk shells).
#
# --threshold offsets the envelope outward, which is what makes --sign-mode no-sign
# usable: an unsigned field has no zero level to extract.
#
#   uv run --with usd-core warp/examples/geometry/example_swept_volume.py
#   uv run --with usd-core warp/examples/geometry/example_swept_volume.py --usd-path ur10_animated.usda
#   uv run --with usd-core warp/examples/geometry/example_swept_volume.py --sign-mode no-sign --threshold 0.1
###########################################################################

import math

import numpy as np
from pxr import Gf, Usd, UsdGeom

import warp as wp
import warp.geometry


def box_mesh(size, center=(0.0, 0.0, 0.0)):
    """Build an outward-oriented axis-aligned box, as (points, indices)."""
    sx, sy, sz = (0.5 * s for s in size)
    cx, cy, cz = center
    corners = np.array(
        [
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],  # bottom (-z)
            [4, 5, 6],
            [4, 6, 7],  # top (+z)
            [0, 1, 5],
            [0, 5, 4],  # -y
            [2, 3, 7],
            [2, 7, 6],  # +y
            [1, 2, 6],
            [1, 6, 5],  # +x
            [3, 0, 4],
            [3, 4, 7],  # -x
        ],
        dtype=np.int32,
    )
    return corners, faces.reshape(-1)


def quat_pitch(angle):
    """Build a rotation about +y (pitch) as a quaternion.

    The components are ordered ``(x, y, z, w)``, which is Warp's convention;
    USD stores the real and imaginary parts separately.
    """
    return np.array([0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)], dtype=np.float32)


def compose(a, b):
    """Compose two transforms given as (7,) arrays, applying b then a."""
    ta = wp.transform(wp.vec3(*a[:3]), wp.quat(*a[3:]))
    tb = wp.transform(wp.vec3(*b[:3]), wp.quat(*b[3:]))
    out = wp.transform_multiply(ta, tb)
    return np.array([*out.p, *out.q], dtype=np.float32)


def procedural_arm(num_samples=24, device=None):
    """Build a two-link arm that swings through a pick-and-place-like arc.

    Returns ``(meshes, transforms, times)`` where ``transforms`` has shape
    ``(num_meshes, num_samples, 7)`` (translation xyz + quaternion xyzw).
    """
    # Rest-pose geometry: three boxes forming base, upper arm, forearm. Each link
    # is modeled in its own local frame with its joint at the origin.
    base_pts, base_idx = box_mesh((0.6, 0.6, 0.3), center=(0.0, 0.0, 0.15))
    upper_pts, upper_idx = box_mesh((0.25, 0.25, 1.2), center=(0.0, 0.0, 0.6))
    fore_pts, fore_idx = box_mesh((0.2, 0.2, 1.0), center=(0.0, 0.0, 0.5))

    # Built with winding-number support, which the default classifier requires.
    meshes = [
        wp.Mesh(
            wp.array(pts, dtype=wp.vec3, device=device),
            wp.array(idx, dtype=wp.int32, device=device),
            support_winding_number=True,
        )
        for pts, idx in ((base_pts, base_idx), (upper_pts, upper_idx), (fore_pts, fore_idx))
    ]

    # Joint frames: the shoulder sits atop the base (z=0.3), the elbow atop the
    # upper arm (z=1.2), both pitching about y in their parent's frame.
    times = np.linspace(0.0, 1.0, num_samples).astype(np.float32)
    transforms = np.zeros((3, num_samples, 7), dtype=np.float32)
    transforms[:, :, 6] = 1.0  # identity quaternions by default

    for s, t in enumerate(times):
        # Two joints swing out of phase to sweep a broad, non-convex region.
        shoulder_angle = 1.2 * np.sin(2.0 * np.pi * t)
        elbow_angle = 1.4 * np.sin(2.0 * np.pi * t + 1.0)

        # Base is static.
        transforms[0, s] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Upper arm = shoulder joint rotation about the shoulder frame.
        upper_world = np.array([0.0, 0.0, 0.3, *quat_pitch(shoulder_angle)], dtype=np.float32)
        transforms[1, s] = upper_world

        # Forearm = upper-arm transform composed with the elbow rotation, so the
        # kinematic chain is respected.
        elbow_local = np.array([0.0, 0.0, 1.2, *quat_pitch(elbow_angle)], dtype=np.float32)
        transforms[2, s] = compose(upper_world, elbow_local)

    return meshes, transforms, times


def _triangulate(counts, idx):
    """Convert USD face counts and flattened indices to triangles using a fan."""
    counts = np.asarray(counts, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    if counts.size and (counts == 3).all():
        return idx.reshape(-1, 3)
    ntri = counts - 2
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    face = np.repeat(np.arange(len(counts)), ntri)
    k = np.arange(int(ntri.sum())) - np.repeat(np.cumsum(ntri) - ntri, ntri) + 1
    s = starts[face]
    return np.stack([idx[s], idx[s + k], idx[s + k + 1]], axis=1)


def load_usd_assembly(path, num_samples=24, device=None):
    """Extract rest-pose meshes and sampled world transforms from an animated USD.

    Every ``UsdGeomMesh`` (including through instance proxies) becomes one
    :class:`warp.Mesh`; its per-sample world transform is read from the stage's
    xform cache. Returns ``(meshes, transforms, times, up_axis)``, where
    ``up_axis`` is the stage's up axis, so the output can be written in the same
    coordinate system.

    The swept volume assumes rigid motion, so a prim's local-to-world transform
    must be a fixed scale followed by an animated rotation and translation. A
    static scale is baked into the rest-pose points; a scale that changes over
    time, or a transform carrying shear or a scale orientation, cannot be split
    into a fixed mesh plus rigid motion and raises :class:`ValueError`.

    The meshes are built with ``support_winding_number=True``, which the
    default classifier requires. CAD assemblies like the UR10 are made of open,
    non-watertight visual shells, for which closest-face-normal sign
    classification is unreliable and produces an incoherent field (spurious
    interior pockets, hundreds of junk shells), so the winding number matters
    here (see :class:`warp.geometry.SweptVolumeSignMode`).
    """
    stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    pred = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)

    prims, rest_points, rest_faces = [], [], []
    for prim in stage.Traverse(pred):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        geom = UsdGeom.Mesh(prim)
        points = geom.GetPointsAttr().Get()
        if not points:
            continue
        counts = geom.GetFaceVertexCountsAttr().Get()
        vertex_indices = geom.GetFaceVertexIndicesAttr().Get()
        if not counts or not vertex_indices:
            continue
        faces = _triangulate(counts, vertex_indices)
        # USD's leftHanded orientation is the opposite winding to the one Warp's
        # mesh queries expect, so flip those triangles.
        if geom.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            faces = faces[:, ::-1]
        prims.append(prim)
        rest_points.append(np.asarray(points, dtype=np.float64))
        rest_faces.append(faces)

    if not prims:
        raise SystemExit(f"No UsdGeomMesh found in {path}")

    t0 = stage.GetStartTimeCode()
    t1 = stage.GetEndTimeCode()
    if t1 <= t0:
        t1 = t0 + 1.0
    times = np.linspace(t0, t1, num_samples).astype(np.float32)

    cache = UsdGeom.XformCache()
    transforms = np.zeros((len(prims), num_samples, 7), dtype=np.float32)
    scales = np.zeros((len(prims), 3), dtype=np.float64)
    for s, t in enumerate(times):
        cache.SetTime(Usd.TimeCode(float(t)))
        for m, prim in enumerate(prims):
            mat = cache.GetLocalToWorldTransform(prim)
            xform = Gf.Transform(mat)

            # Reject anything that is not a scale followed by a rigid motion:
            # recomposing from just those three pieces has to reproduce the
            # matrix, which fails on shear, a scale orientation, or a pivot.
            rigid_with_scale = Gf.Transform()
            rigid_with_scale.SetScale(xform.GetScale())
            rigid_with_scale.SetRotation(xform.GetRotation())
            rigid_with_scale.SetTranslation(xform.GetTranslation())
            if not Gf.IsClose(rigid_with_scale.GetMatrix(), mat, 1e-5):
                raise ValueError(
                    f"'{prim.GetPath()}' has a local-to-world transform that is not a scale followed by a "
                    "rotation and a translation, so it cannot be expressed as a rest mesh under rigid motion."
                )

            scale = np.array(xform.GetScale(), dtype=np.float64)
            if s == 0:
                scales[m] = scale
            elif not np.allclose(scale, scales[m], rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f"'{prim.GetPath()}' is scaled by {scales[m].tolist()} at time {times[0]:g} and "
                    f"{scale.tolist()} at time {t:g}. The swept volume assumes rigid motion, so the "
                    "scale must not change over time."
                )

            trans = xform.GetTranslation()
            rot = xform.GetRotation().GetQuat()
            imag = rot.GetImaginary()
            transforms[m, s] = [trans[0], trans[1], trans[2], imag[0], imag[1], imag[2], rot.GetReal()]

    meshes = []
    for m in range(len(prims)):
        # Bake the static scale into the rest pose, leaving the motion rigid. A
        # negative determinant mirrors the mesh, which reverses its winding.
        points = rest_points[m] * scales[m]
        faces = rest_faces[m][:, ::-1] if np.prod(scales[m]) < 0.0 else rest_faces[m]
        meshes.append(
            wp.Mesh(
                wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
                wp.array(np.ascontiguousarray(faces).reshape(-1).astype(np.int32), dtype=wp.int32, device=device),
                support_winding_number=True,
            )
        )

    return meshes, transforms, times, UsdGeom.GetStageUpAxis(stage)


def write_usd(stage_path, verts, indices, up_axis):
    """Write the envelope to a USD stage with the given up axis."""
    stage = Usd.Stage.CreateNew(stage_path)
    # The geometry is never reoriented, so the output has to declare the same up
    # axis as the source or a viewer shows it lying on its side.
    UsdGeom.SetStageUpAxis(stage, up_axis)
    mesh = UsdGeom.Mesh.Define(stage, "/swept_volume_mesh")
    # Without this a viewer applies the default Catmull-Clark subdivision, which
    # smooths the marching-cubes triangles and pulls the surface inward.
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    v = verts.numpy()
    f = indices.numpy()
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in v.tolist()])
    mesh.CreateFaceVertexCountsAttr([3] * (len(f) // 3))
    mesh.CreateFaceVertexIndicesAttr(f.tolist())
    stage.Save()


def main(
    usd_path=None,
    num_samples=24,
    voxel_size=0.08,
    threshold=None,
    sign_mode=warp.geometry.SweptVolumeSignMode.WINDING_NUMBER,
    stage_path="example_swept_volume.usd",
):
    if usd_path is not None:
        meshes, transforms, times, up_axis = load_usd_assembly(usd_path, num_samples=num_samples)
        label = usd_path
    else:
        meshes, transforms, times = procedural_arm(num_samples=num_samples)
        up_axis = UsdGeom.Tokens.z
        label = "procedural two-link arm"

    total_tris = sum(len(m.indices.numpy()) // 3 for m in meshes)
    print(
        f"{label}: {len(meshes)} meshes, {total_tris} triangles, "
        f"{num_samples} pose samples over t in [{times[0]:g}, {times[-1]:g}], sign={sign_mode.name}"
    )

    # The grid's covering radius is the level warp.geometry.swept_volume_mesh
    # documents as enclosing every stamped pose; a larger one offsets the
    # envelope outward, e.g. for a clearance margin.
    if threshold is None:
        threshold = 0.5 * math.sqrt(3.0) * voxel_size

    with wp.ScopedTimer("swept_volume_mesh"):
        verts, indices = warp.geometry.swept_volume_mesh(
            meshes,
            transforms,
            voxel_size=voxel_size,
            threshold=threshold,
            sign_mode=sign_mode,
        )
        wp.synchronize_device()

    v = verts.numpy()
    print(f"envelope: {len(v)} vertices, {len(indices.numpy()) // 3} triangles")
    print(f"envelope AABB: min {np.round(v.min(axis=0), 3)}  max {np.round(v.max(axis=0), 3)}")

    if stage_path:
        write_usd(stage_path, verts, indices, up_axis)
        print(f"wrote {stage_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--usd-path",
        type=str,
        default=None,
        help="Path to an animated USD assembly (e.g. a UR10 arm). Uses a procedural arm if omitted.",
    )
    parser.add_argument("--num-samples", type=int, default=24, help="Number of pose samples to stamp.")
    parser.add_argument("--voxel-size", type=float, default=0.08, help="Grid cell size in world units.")
    parser.add_argument(
        "--sign-mode",
        type=str,
        default="winding-number",
        choices=["normal", "winding-number", "parity", "no-sign"],
        help=(
            "How to classify inside from outside. 'winding-number' is the "
            "default and most robust, 'normal' and 'parity' are faster but may "
            "fail causing spurious surfaces in the swept volume. 'no-sign' is the "
            "fastest but treats the input as a shell and is only meaningful "
            "combined with extracting a non-zero offset surface of the swept "
            "volume."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Level to extract, offsetting the envelope outward by this much. If omitted, uses the "
            "grid's covering radius, the smallest level that encloses every stamped pose. Must be "
            "positive for --sign-mode no-sign."
        ),
    )
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_swept_volume.usd",
        help="Path to the output USD file.",
    )
    args = parser.parse_known_args()[0]

    sign_mode = {
        "normal": warp.geometry.SweptVolumeSignMode.NORMAL,
        "winding-number": warp.geometry.SweptVolumeSignMode.WINDING_NUMBER,
        "parity": warp.geometry.SweptVolumeSignMode.PARITY,
        "no-sign": warp.geometry.SweptVolumeSignMode.NO_SIGN,
    }[args.sign_mode]
    with wp.ScopedDevice(args.device):
        main(
            usd_path=args.usd_path,
            num_samples=args.num_samples,
            voxel_size=args.voxel_size,
            threshold=args.threshold,
            sign_mode=sign_mode,
            stage_path=args.stage_path,
        )
