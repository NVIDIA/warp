# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense-stamping swept volume (motion envelope) of animated rigid meshes."""

import unittest

import numpy as np

import warp as wp
import warp.geometry as geo
from warp.tests.geometry import utils as U
from warp.tests.unittest_utils import *


def _sphere_mesh(device, radius=0.5, subdivisions=3, support_winding_number=True):
    points, indices = U.icosphere(subdivisions=subdivisions, radius=radius)
    return wp.Mesh(
        wp.array(points, dtype=wp.vec3, device=device),
        wp.array(indices, dtype=wp.int32, device=device),
        support_winding_number=support_winding_number,
    )


def _translation_transforms(offsets):
    """Build a ``(num_meshes, num_samples, 7)`` transform array of pure translations."""
    offsets = np.asarray(offsets, dtype=np.float32)
    tr = np.zeros((*offsets.shape[:-1], 7), dtype=np.float32)
    tr[..., 0:3] = offsets
    tr[..., 6] = 1.0  # identity quaternion (x, y, z, w)
    return tr


def _capsule_sdf(P, a, b, radius):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pa = P - a
    ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - np.outer(h, ba), axis=1) - radius


def _grid_points(lower, upper, dims):
    lower = np.array([lower[0], lower[1], lower[2]])
    upper = np.array([upper[0], upper[1], upper[2]])
    spacing = (upper - lower) / (np.array(dims) - 1)
    idx = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), axis=-1).reshape(-1, 3)
    return lower + idx * spacing, idx, spacing


@wp.kernel
def _signed_distance_to_mesh_kernel(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    out: wp.array[wp.float32],
):
    i = wp.tid()
    query = wp.mesh_query_point_sign_winding_number(mesh_id, points[i], max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        out[i] = query.sign * wp.length(points[i] - closest)
    else:
        out[i] = max_dist


def _signed_distance_to_mesh(verts, indices, points, device):
    """Signed distance from every point in ``points`` to the surface ``(verts, indices)``."""
    mesh = wp.Mesh(verts, indices, support_winding_number=True)
    pts = wp.array(points.astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.empty(len(points), dtype=wp.float32, device=device)
    wp.launch(
        _signed_distance_to_mesh_kernel,
        dim=len(points),
        inputs=[mesh.id, pts, 1.0e6],
        outputs=[out],
        device=device,
    )
    return out.numpy()


def test_swept_sphere_is_a_capsule(test, device):
    """Compare the field of a translating sphere against the analytic capsule SDF.

    A sphere of radius ``r`` translated along the x axis from -1 to 1 sweeps a
    capsule: the segment ``[-1, 1]`` on x, dilated by ``r``. The dense field
    should match the analytic capsule SDF to within the icosphere's polygonal
    error plus one voxel.
    """
    radius = 0.5
    mesh = _sphere_mesh(device, radius=radius, subdivisions=3)
    tr = _translation_transforms([[[x, 0.0, 0.0] for x in np.linspace(-1.0, 1.0, 21)]])

    field, lower, upper = geo.swept_volume_field([mesh], tr, voxel_size=0.05, device=device)

    fnp = field.numpy()
    P, idx, _ = _grid_points(lower, upper, fnp.shape)
    d_ref = _capsule_sdf(P, (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), radius)
    d_num = fnp[idx[:, 0], idx[:, 1], idx[:, 2]]

    # Restrict the comparison to the near-surface band, where a signed distance
    # is meaningful (the max_dist plateau far away is expected to differ).
    band = np.abs(d_ref) < 0.3
    np.testing.assert_allclose(d_num[band], d_ref[band], atol=0.03)


def test_swept_sphere_mesh_bounds(test, device):
    """Check that the extracted envelope of a swept sphere spans the capsule's AABB."""
    radius = 0.5
    mesh = _sphere_mesh(device, radius=radius)
    tr = _translation_transforms([[[x, 0.0, 0.0] for x in np.linspace(-1.0, 1.0, 21)]])

    verts, indices = geo.swept_volume_mesh([mesh], tr, voxel_size=0.05, device=device)

    v = verts.numpy()
    test.assertGreater(len(v), 0)
    test.assertEqual(len(indices.numpy()) % 3, 0)
    np.testing.assert_allclose(v.min(axis=0), [-1.0 - radius, -radius, -radius], atol=0.06)
    np.testing.assert_allclose(v.max(axis=0), [1.0 + radius, radius, radius], atol=0.06)


def test_static_single_pose_matches_mesh(test, device):
    """Check that a single identity pose recovers the input mesh.

    With one pose the field reduces to the mesh's own SDF, so the zero
    isosurface should recover a sphere of the input radius.
    """
    radius = 0.7
    mesh = _sphere_mesh(device, radius=radius, subdivisions=3)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])

    verts, _ = geo.swept_volume_mesh([mesh], tr, voxel_size=0.05, device=device)

    v = verts.numpy()
    r = np.linalg.norm(v - v.mean(axis=0), axis=1)
    np.testing.assert_allclose(r.mean(), radius, atol=0.05)
    test.assertLess(r.std(), 0.05)


def test_union_of_two_static_spheres(test, device):
    """Check that two separated spheres give the per-sphere minimum field.

    Two separated spheres yield two connected components; the midpoint region is
    sampled to confirm the union.
    """
    radius = 0.4
    left = _sphere_mesh(device, radius=radius)
    right = _sphere_mesh(device, radius=radius)
    tr = _translation_transforms([[[-1.5, 0.0, 0.0]], [[1.5, 0.0, 0.0]]])

    field, lower, upper = geo.swept_volume_field([left, right], tr, voxel_size=0.05, device=device)

    fnp = field.numpy()
    P, idx, _ = _grid_points(lower, upper, fnp.shape)
    d_num = fnp[idx[:, 0], idx[:, 1], idx[:, 2]]
    d_left = np.linalg.norm(P - np.array([-1.5, 0.0, 0.0]), axis=1) - radius
    d_right = np.linalg.norm(P - np.array([1.5, 0.0, 0.0]), axis=1) - radius
    d_ref = np.minimum(d_left, d_right)

    band = np.abs(d_ref) < 0.3
    np.testing.assert_allclose(d_num[band], d_ref[band], atol=0.03)


def test_conservative_encloses_all_poses(test, device):
    """Check that every stamped pose stays inside the envelope.

    Every input vertex, at every sampled pose, must lie inside the field's zero
    level (within the nearest-node rounding error), and inside the mesh
    extracted at the documented conservative ``threshold``.
    """
    radius = 0.5
    mesh = _sphere_mesh(device, radius=radius)
    points = mesh.points.numpy().astype(np.float64)
    offsets = np.array([[x, 0.3 * x, 0.0] for x in np.linspace(-1.0, 1.0, 11)], dtype=np.float32)
    tr = _translation_transforms([offsets])

    voxel = 0.05
    field, lower, upper = geo.swept_volume_field([mesh], tr, voxel_size=voxel, device=device)
    fnp = field.numpy()
    lower = np.array([lower[0], lower[1], lower[2]])
    upper = np.array([upper[0], upper[1], upper[2]])
    dims = np.array(fnp.shape)
    spacing = (upper - lower) / (dims - 1)

    # Trilinearly-cheap nearest-node lookup for each posed vertex.
    posed = (points[None, :, :] + offsets[:, None, :3]).reshape(-1, 3)
    node = np.rint((posed - lower) / spacing).astype(int)
    node = np.clip(node, 0, dims - 1)
    d = fnp[node[:, 0], node[:, 1], node[:, 2]]
    # Interior/boundary vertices should be at or inside the surface, allowing the
    # nearest-node rounding of up to half a voxel diagonal.
    covering_radius = 0.5 * float(np.linalg.norm(spacing))
    test.assertLessEqual(float(d.max()), covering_radius + 1e-4)

    # The documented conservative level must enclose the poses in the extracted
    # mesh too, not just in the sampled field.
    verts, indices = geo.swept_volume_mesh([mesh], tr, voxel_size=voxel, threshold=covering_radius, device=device)
    test.assertGreater(len(verts.numpy()), 0)
    signed = _signed_distance_to_mesh(verts, indices, posed, device)
    test.assertLessEqual(float(signed.max()), 0.0)


def test_rotation_pose(test, device):
    """Exercise the quaternion inverse-transform path with a rotated pose.

    An off-center sphere rotated 90 degrees about z sweeps a quarter-annulus, so
    the envelope's radial extent must reach the rotated positions.
    """
    radius = 0.3
    # Rest-pose sphere centered at +x, so a rotation about z sweeps an arc.
    points, indices = U.icosphere(subdivisions=2, radius=radius, center=(1.0, 0.0, 0.0))
    mesh = wp.Mesh(
        wp.array(points, dtype=wp.vec3, device=device),
        wp.array(indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )

    angles = np.linspace(0.0, np.pi / 2.0, 13)
    tr = np.zeros((1, len(angles), 7), dtype=np.float32)
    for s, a in enumerate(angles):
        # Rotate the rest-pose sphere (centered at +x) about z by angle a.
        tr[0, s, 6] = np.cos(a / 2.0)
        tr[0, s, 5] = np.sin(a / 2.0)  # quaternion z component

    verts, _ = geo.swept_volume_mesh([mesh], tr, voxel_size=0.05, device=device)
    v = verts.numpy()
    # The sphere center traces an arc of radius 1 from +x to +y; the envelope must
    # span both extremes in x and y.
    test.assertGreater(v[:, 0].max(), 1.0 + radius - 0.06)
    test.assertGreater(v[:, 1].max(), 1.0 + radius - 0.06)


def test_sign_modes_agree_on_watertight_mesh(test, device):
    """Check that every signed classifier gives the same envelope on watertight input."""
    radius = 0.5
    mesh = _sphere_mesh(device, radius=radius)
    tr = _translation_transforms([[[x, 0.0, 0.0] for x in np.linspace(-1.0, 1.0, 21)]])

    for sign_mode in (
        geo.SweptVolumeSignMode.WINDING_NUMBER,
        geo.SweptVolumeSignMode.NORMAL,
        geo.SweptVolumeSignMode.PARITY,
    ):
        verts, _ = geo.swept_volume_mesh([mesh], tr, voxel_size=0.05, sign_mode=sign_mode, device=device)
        v = verts.numpy()
        np.testing.assert_allclose(v.min(axis=0), [-1.0 - radius, -radius, -radius], atol=0.06)
        np.testing.assert_allclose(v.max(axis=0), [1.0 + radius, radius, radius], atol=0.06)


def test_unsigned_dilates_the_envelope(test, device):
    """Check that NO_SIGN gives a positive field whose threshold surface is an outward offset.

    The unsigned field cannot be negative anywhere, and extracting it at ``threshold``
    should push the surface out by that much. A sphere thicker than ``2 * threshold``
    also picks up the documented inward shell, so the extracted mesh spans the
    dilated bounds while the field stays non-negative.
    """
    radius = 0.5
    mesh = _sphere_mesh(device, radius=radius)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])

    field, _, _ = geo.swept_volume_field(
        [mesh], tr, voxel_size=0.05, sign_mode=geo.SweptVolumeSignMode.NO_SIGN, device=device
    )
    test.assertGreaterEqual(float(field.numpy().min()), 0.0)

    threshold = 0.2
    verts, _ = geo.swept_volume_mesh(
        [mesh], tr, voxel_size=0.05, threshold=threshold, sign_mode=geo.SweptVolumeSignMode.NO_SIGN, device=device
    )
    v = verts.numpy()
    np.testing.assert_allclose(v.max(axis=0), [radius + threshold] * 3, atol=0.06)
    np.testing.assert_allclose(v.min(axis=0), [-radius - threshold] * 3, atol=0.06)

    # The documented inward shell: the sphere is thicker than 2 * threshold, so the
    # isosurface also runs at radius - threshold.
    r = np.linalg.norm(v, axis=1)
    test.assertGreater(int(np.count_nonzero(np.abs(r - (radius - threshold)) < 0.05)), 0)
    test.assertGreater(int(np.count_nonzero(np.abs(r - (radius + threshold)) < 0.05)), 0)


def test_unsigned_requires_positive_iso(test, device):
    """Check that NO_SIGN rejects the degenerate zero isosurface."""
    mesh = _sphere_mesh(device, radius=0.5)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])
    with test.assertRaises(ValueError):
        geo.swept_volume_mesh([mesh], tr, voxel_size=0.05, sign_mode=geo.SweptVolumeSignMode.NO_SIGN, device=device)


def _open_box_mesh(device, half=0.5, support_winding_number=True):
    """Build an axis-aligned box with its +z face removed.

    The result is a non-watertight shell whose interior the closest-face-normal
    test cannot classify reliably.
    """
    h = half
    pts = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float32,
    )
    # Five of the six faces (the +z face, tris [4,5,6]/[4,6,7], is omitted).
    faces = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],  # -z
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
    ).reshape(-1)
    return wp.Mesh(
        wp.array(pts, dtype=wp.vec3, device=device),
        wp.array(faces, dtype=wp.int32, device=device),
        support_winding_number=support_winding_number,
    )


def test_winding_number_handles_non_watertight(test, device):
    """Check that the winding number classifies the interior of an open shell.

    A robot's visual meshes are open shells like this box with a missing face.
    The winding number stays robust on them and reports the shell interior as
    inside (negative field). The closest-face-normal classifier cannot be relied
    on here, since it even disagrees between CPU and CUDA on the same point,
    which is why the USD example defaults to the winding number.
    """
    mesh = _open_box_mesh(device, half=0.5)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])

    field, lower, upper = geo.swept_volume_field([mesh], tr, voxel_size=0.05, device=device)
    fnp = field.numpy()

    # The box center is half a metre inside the shell, so winding must report it
    # inside with a distance close to the wall half-thickness (0.5).
    lower = np.array([lower[0], lower[1], lower[2]])
    upper = np.array([upper[0], upper[1], upper[2]])
    spacing = (upper - lower) / (np.array(fnp.shape) - 1)
    center_node = np.rint((np.zeros(3) - lower) / spacing).astype(int)
    center_value = float(fnp[center_node[0], center_node[1], center_node[2]])
    test.assertLess(center_value, 0.0)
    np.testing.assert_allclose(center_value, -0.5, atol=spacing[0])


def test_resolution_argument(test, device):
    """Check that an explicit resolution produces a field of that shape."""
    mesh = _sphere_mesh(device, radius=0.5)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])
    field, _, _ = geo.swept_volume_field([mesh], tr, resolution=(16, 20, 24), device=device)
    test.assertEqual(field.shape, (16, 20, 24))
    wp.synchronize_device(device)


def test_voxel_size_gives_cubic_cells(test, device):
    """Check that voxel_size is the spacing, not an upper bound on it."""
    mesh = _sphere_mesh(device, radius=0.5)
    tr = _translation_transforms([[[x, 0.0, 0.0] for x in np.linspace(0.0, 2.0, 8)]])

    voxel_size = 0.05
    field, lower, upper = geo.swept_volume_field([mesh], tr, voxel_size=voxel_size, device=device)
    lower = np.array([lower[0], lower[1], lower[2]])
    upper = np.array([upper[0], upper[1], upper[2]])
    spacing = (upper - lower) / (np.array(field.shape) - 1)
    np.testing.assert_allclose(spacing, voxel_size, rtol=1e-5)
    wp.synchronize_device(device)


def test_positive_iso_is_not_clipped(test, device):
    """Check that the domain grows with threshold instead of clipping the dilated envelope."""
    radius = 0.5
    mesh = _sphere_mesh(device, radius=radius)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])

    for threshold in (0.25, 0.7):
        verts, _ = geo.swept_volume_mesh([mesh], tr, voxel_size=0.05, threshold=threshold, device=device)
        v = verts.numpy()
        np.testing.assert_allclose(v.max(axis=0), [radius + threshold] * 3, atol=0.06)
        np.testing.assert_allclose(v.min(axis=0), [-radius - threshold] * 3, atol=0.06)


def test_explicit_domain_is_used_verbatim(test, device):
    """Check that supplied domain corners are honored, so fields can share a grid."""
    mesh = _sphere_mesh(device, radius=0.5)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])

    lower, upper = geo.swept_volume_bounds([mesh], tr, padding=0.3, device=device)
    field, out_lower, out_upper = geo.swept_volume_field(
        [mesh],
        tr,
        resolution=(24, 24, 24),
        lower=lower,
        upper=upper,
        device=device,
    )
    test.assertEqual(field.shape, (24, 24, 24))
    for a in range(3):
        test.assertAlmostEqual(out_lower[a], lower[a], places=5)
        test.assertAlmostEqual(out_upper[a], upper[a], places=5)
    # The padding requested is what separates the sphere from the boundary.
    np.testing.assert_allclose([lower[0], lower[1], lower[2]], [-0.8] * 3, atol=0.02)
    wp.synchronize_device(device)


def test_domain_inside_the_solid_keeps_its_sign(test, device):
    """Check that an explicit domain enclosed by the geometry still reads as inside.

    The closest-point search has to reach the surface from every node. A domain
    the caller supplies need not contain the geometry, so its own diagonal is
    not a sufficient bound.
    """
    radius = 10.0
    points, indices = U.icosphere(subdivisions=3, radius=radius)
    mesh = wp.Mesh(
        wp.array(points, dtype=wp.vec3, device=device),
        wp.array(indices, dtype=wp.int32, device=device),
        support_winding_number=True,
    )
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])

    # Every node of this domain lies about `radius` deep inside the sphere.
    field, _, _ = geo.swept_volume_field(
        [mesh], tr, resolution=(9, 9, 9), lower=(-1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0), device=device
    )
    f = field.numpy()
    test.assertEqual(int((f >= 0.0).sum()), 0)
    np.testing.assert_allclose(f[4, 4, 4], -radius, atol=0.1)


def test_invalid_arguments(test, device):
    """Check that malformed arguments raise ``ValueError``."""
    mesh = _sphere_mesh(device, radius=0.5)
    tr = _translation_transforms([[[0.0, 0.0, 0.0]]])
    with test.assertRaises(ValueError):
        geo.swept_volume_field([], tr, voxel_size=0.05, device=device)
    with test.assertRaises(ValueError):
        geo.swept_volume_field([mesh], tr, device=device)  # no voxel_size or resolution
    mismatched = _translation_transforms([[[0.0] * 3], [[0.0] * 3]])
    with test.assertRaises(ValueError):
        # Two rows of transforms but only one mesh.
        geo.swept_volume_field([mesh], mismatched, voxel_size=0.1, device=device)
    with test.assertRaises(ValueError):
        # Same check before swept_volume_mesh() reaches the bounds kernels, which
        # would otherwise write past their per-mesh slots.
        geo.swept_volume_mesh([mesh], mismatched, voxel_size=0.1, device=device)
    with test.assertRaises(ValueError):
        geo.swept_volume_bounds([mesh], mismatched, device=device)
    with test.assertRaises(ValueError):
        geo.swept_volume_field([mesh], tr, voxel_size=0.0, device=device)
    with test.assertRaises(ValueError):
        geo.swept_volume_field([mesh], tr, voxel_size=-0.05, device=device)
    with test.assertRaises(ValueError):
        geo.swept_volume_field([mesh], np.zeros((1, 0, 7), dtype=np.float32), voxel_size=0.05, device=device)
    with test.assertRaises(ValueError):
        geo.swept_volume_field([mesh], tr, resolution=(16, 20), device=device)
    with test.assertRaises(ValueError):
        geo.swept_volume_field([mesh], tr, resolution=(16, 20, 1), device=device)
    with test.assertRaises(ValueError):
        # Sizing the grid two ways at once is ambiguous.
        geo.swept_volume_field([mesh], tr, voxel_size=0.05, resolution=(16, 16, 16), device=device)
    with test.assertRaises(ValueError):
        # An explicit domain cannot be snapped to whole cells.
        geo.swept_volume_field(
            [mesh],
            tr,
            voxel_size=0.05,
            lower=(-1.0, -1.0, -1.0),
            upper=(1.0, 1.0, 1.0),
            device=device,
        )
    with test.assertRaises(ValueError):
        geo.swept_volume_field([mesh], tr, resolution=(16, 16, 16), lower=(-1.0, -1.0, -1.0), device=device)
    with test.assertRaises(ValueError):
        # The default classifier needs a mesh built with support_winding_number=True.
        plain = _sphere_mesh(device, radius=0.5, support_winding_number=False)
        geo.swept_volume_field([plain], tr, voxel_size=0.05, device=device)


devices = get_test_devices()


class TestSweptVolume(unittest.TestCase):
    pass


add_function_test(TestSweptVolume, "test_swept_sphere_is_a_capsule", test_swept_sphere_is_a_capsule, devices=devices)
add_function_test(TestSweptVolume, "test_swept_sphere_mesh_bounds", test_swept_sphere_mesh_bounds, devices=devices)
add_function_test(
    TestSweptVolume, "test_static_single_pose_matches_mesh", test_static_single_pose_matches_mesh, devices=devices
)
add_function_test(
    TestSweptVolume, "test_union_of_two_static_spheres", test_union_of_two_static_spheres, devices=devices
)
add_function_test(
    TestSweptVolume, "test_conservative_encloses_all_poses", test_conservative_encloses_all_poses, devices=devices
)
add_function_test(TestSweptVolume, "test_rotation_pose", test_rotation_pose, devices=devices)
add_function_test(
    TestSweptVolume,
    "test_sign_modes_agree_on_watertight_mesh",
    test_sign_modes_agree_on_watertight_mesh,
    devices=devices,
)
add_function_test(
    TestSweptVolume, "test_unsigned_dilates_the_envelope", test_unsigned_dilates_the_envelope, devices=devices
)
add_function_test(
    TestSweptVolume, "test_unsigned_requires_positive_iso", test_unsigned_requires_positive_iso, devices=devices
)
add_function_test(
    TestSweptVolume,
    "test_winding_number_handles_non_watertight",
    test_winding_number_handles_non_watertight,
    devices=devices,
)
add_function_test(
    TestSweptVolume, "test_voxel_size_gives_cubic_cells", test_voxel_size_gives_cubic_cells, devices=devices
)
add_function_test(
    TestSweptVolume, "test_positive_iso_is_not_clipped", test_positive_iso_is_not_clipped, devices=devices
)
add_function_test(
    TestSweptVolume, "test_explicit_domain_is_used_verbatim", test_explicit_domain_is_used_verbatim, devices=devices
)
add_function_test(TestSweptVolume, "test_resolution_argument", test_resolution_argument, devices=devices)
add_function_test(
    TestSweptVolume,
    "test_domain_inside_the_solid_keeps_its_sign",
    test_domain_inside_the_solid_keeps_its_sign,
    devices=devices,
)
add_function_test(TestSweptVolume, "test_invalid_arguments", test_invalid_arguments, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
