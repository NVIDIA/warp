# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from collections import Counter

import numpy as np

import warp as wp
import warp.geometry
from warp._src import surface_nets as _surface_nets_module
from warp.tests.unittest_utils import *


@wp.kernel
def make_field_sphere_sdf(field: wp.array3d[float], center: wp.vec3, radius: float):
    """Make a sphere SDF for nodes on the integer domain with node coordinates 0,1,2,3,..."""

    i, j, k = wp.tid()

    p = wp.vec3(float(i), float(j), float(k))

    d = wp.length(p - center) - radius

    field[i, j, k] = d


@wp.kernel
def make_field_sphere_sdf_unit_domain(field: wp.array3d[float], center: wp.vec3, radius: wp.array[wp.float32]):
    """Makes a sphere SDF for nodes on the unit domain [-1, 1]^3."""
    i, j, k = wp.tid()

    nx, ny, nz = field.shape[0], field.shape[1], field.shape[2]

    p = wp.vec3(
        2.0 * wp.float32(i) / (wp.float32(nx) - 1.0) - 1.0,
        2.0 * wp.float32(j) / (wp.float32(ny) - 1.0) - 1.0,
        2.0 * wp.float32(k) / (wp.float32(nz) - 1.0) - 1.0,
    )

    d = wp.length(p - center) - radius[0]

    field[i, j, k] = d


@wp.kernel
def compute_surface_area(verts: wp.array[wp.vec3], faces: wp.array[wp.int32], out_area: wp.array[wp.float32]):
    tid = wp.tid()
    vi = faces[3 * tid + 0]
    vj = faces[3 * tid + 1]
    vk = faces[3 * tid + 2]

    p0 = verts[vi]
    p1 = verts[vj]
    p2 = verts[vk]

    # Heron's formula for triangle area
    a = wp.length(p1 - p0)
    b = wp.length(p2 - p0)
    c = wp.length(p2 - p1)
    s = (a + b + c) / 2.0
    area = wp.sqrt(s * (s - a) * (s - b) * (s - c))

    wp.atomic_add(out_area, 0, area)


def make_sphere_field(device, node_dim, radius=None):
    """Build an integer-domain sphere SDF centered in a cubic grid."""
    if radius is None:
        radius = node_dim / 4.0
    field = wp.zeros(shape=(node_dim, node_dim, node_dim), dtype=float, device=device)
    center = wp.vec3(node_dim / 2, node_dim / 2, node_dim / 2)
    wp.launch(make_field_sphere_sdf, dim=field.shape, inputs=[field, center, radius], device=device)
    return field, np.array([node_dim / 2, node_dim / 2, node_dim / 2]), radius


def quad_edge_counts(quads_np):
    """Count the number of quads incident to each undirected quad edge."""
    counts = Counter()
    for quad in quads_np:
        for i in range(4):
            a, b = int(quad[i]), int(quad[(i + 1) % 4])
            counts[(min(a, b), max(a, b))] += 1
    return counts


def signed_volume(verts_np, indices_np):
    """Signed volume enclosed by the triangles (positive for outward normals)."""
    tris = indices_np.reshape(-1, 3)
    p0, p1, p2 = verts_np[tris[:, 0]], verts_np[tris[:, 1]], verts_np[tris[:, 2]]
    return np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0


def validate_surface_nets_output(test, verts_np, indices_np, topology="triangle", check_nonempty=True):
    """Structural checks shared by the tests below."""
    if topology == "quad":
        # check that the quad array seems valid
        test.assertEqual(indices_np.shape[0] % 4, 0)  # four indices per quad
        quads_np = indices_np.reshape(-1, 4)
        if check_nonempty:
            test.assertGreater(quads_np.shape[0], 0)  # at least one quad
        test.assertTrue((quads_np >= 0).all())  # all quad inds nonnegative
        test.assertTrue((quads_np < max(verts_np.shape[0], 1)).all())  # all quad inds in-bounds on the vertex array
        for i in range(4):  # all quads have unique vertices
            for j in range(i + 1, 4):
                test.assertTrue((quads_np[:, i] != quads_np[:, j]).all())
    else:
        # check that the triangle index array seems valid
        if check_nonempty:
            test.assertGreater(indices_np.shape[0], 0)  # at least one triangle
        test.assertEqual(indices_np.shape[0] % 6, 0)  # two triangles per quad
        test.assertTrue((indices_np >= 0).all())  # all tri inds nonnegative
        test.assertTrue((indices_np < max(verts_np.shape[0], 1)).all())  # all tri inds in-bounds on the vertex array

    # check that the vertex array seems valid
    if check_nonempty:
        test.assertGreater(verts_np.shape[0], 0)  # at least one vertex
    test.assertEqual(verts_np.shape[1], 3)  # all vertices are 3D
    test.assertTrue(np.isfinite(verts_np).all())  # all vertices are finite


def extract_quads(test, field, threshold=0.0, **kwargs):
    """Extract and validate the native quads, as ``(verts_np, quads_np)``.

    The quads are reshaped from the flat index array into one row of four
    vertex indices per quad, which the connectivity checks below work on.
    """
    verts, indices = wp.geometry.IsoSurfaceNets.extract(field, threshold=threshold, topology="quad", **kwargs)
    verts_np, indices_np = verts.numpy(), indices.numpy()
    validate_surface_nets_output(test, verts_np, indices_np, topology="quad")
    return verts_np, indices_np.reshape(-1, 4)


def test_surface_nets(test, device):
    """Test typical usage of the stateful interface."""
    node_dim = 64
    cell_dim = node_dim - 1
    bounds_low = (0.0, 0.0, 0.0)
    bounds_high = (float(cell_dim), float(cell_dim), float(cell_dim))

    iso = wp.geometry.IsoSurfaceNets(
        nx=node_dim,
        ny=node_dim,
        nz=node_dim,
        domain_bounds_lower_corner=bounds_low,
        domain_bounds_upper_corner=bounds_high,
    )

    field, center, radius = make_sphere_field(device, node_dim)

    iso.surface(field=field, threshold=0.0)
    verts_np = iso.verts.numpy()
    indices_np = iso.indices.numpy()
    test.assertEqual(iso.indices.dtype, wp.int32)  # flat array of indices, following Warp convention
    validate_surface_nets_output(test, verts_np, indices_np)

    # the isosurface is interior, so every vertex must be referenced by the triangles
    test.assertTrue((np.unique(indices_np) == np.arange(verts_np.shape[0])).all())

    # check that all returned vertices lie on the surface of the sphere
    length = np.linalg.norm(verts_np - center, axis=1)
    error = np.abs(length - radius)
    test.assertTrue(np.max(error) < 1.0)

    # smoke test reuse with new dimensions
    iso.resize(nx=node_dim // 2, ny=node_dim // 2, nz=node_dim // 2)
    small_field, _, _ = make_sphere_field(device, node_dim // 2)
    iso.surface(field=small_field, threshold=0.0)
    validate_surface_nets_output(test, iso.verts.numpy(), iso.indices.numpy())


def test_surface_nets_functional(test, device):
    """Ensure the single-function interface works as expected, along with the error contract."""
    node_dim = 32
    field, _center, _radius = make_sphere_field(device, node_dim)

    verts, indices = wp.geometry.IsoSurfaceNets.extract(field, threshold=0.0)
    validate_surface_nets_output(test, verts.numpy(), indices.numpy())

    # the functional and stateful interfaces must agree exactly
    iso = wp.geometry.IsoSurfaceNets(nx=node_dim, ny=node_dim, nz=node_dim)
    iso.surface(field=field, threshold=0.0)
    np.testing.assert_array_equal(verts.numpy(), iso.verts.numpy())
    np.testing.assert_array_equal(indices.numpy(), iso.indices.numpy())

    # error contract
    with test.assertRaises(ValueError):
        wp.geometry.IsoSurfaceNets.extract(wp.zeros(shape=8, dtype=wp.float32, device=device))
    with test.assertRaises(TypeError):
        wp.geometry.IsoSurfaceNets.extract(wp.zeros(shape=(8, 8, 8), dtype=wp.float64, device=device))
    with test.assertRaises(ValueError):
        iso.surface(field=wp.zeros(shape=(8, 8, 8), dtype=wp.float32, device=device), threshold=0.0)
    with test.assertRaises(ValueError):
        wp.geometry.IsoSurfaceNets.extract(field, threshold=0.0, topology="triangles")
    with test.assertRaises(ValueError):
        wp.geometry.IsoSurfaceNets(nx=node_dim, ny=node_dim, nz=node_dim, topology="quads")


def test_surface_nets_topology(test, device):
    """Check that the topology parameter selects the face type written to the same indices array."""
    node_dim = 32
    field, _center, _radius = make_sphere_field(device, node_dim)

    tri_verts, tri_indices = wp.geometry.IsoSurfaceNets.extract(field, threshold=0.0, topology="triangle")
    quad_verts, quad_indices = wp.geometry.IsoSurfaceNets.extract(field, threshold=0.0, topology="quad")

    # both topologies expose the same flat wp.int32 index array
    test.assertEqual(tri_indices.dtype, wp.int32)
    test.assertEqual(quad_indices.dtype, wp.int32)
    test.assertEqual(len(tri_indices.shape), 1)
    test.assertEqual(len(quad_indices.shape), 1)
    validate_surface_nets_output(test, tri_verts.numpy(), tri_indices.numpy())
    validate_surface_nets_output(test, quad_verts.numpy(), quad_indices.numpy(), topology="quad")

    # both topologies describe the same mesh: same vertices, and the triangles
    # are exactly the (0, 1, 2)/(0, 2, 3) split of the quads, in order
    np.testing.assert_array_equal(tri_verts.numpy(), quad_verts.numpy())
    quads_np = quad_indices.numpy().reshape(-1, 4)
    tris_np = tri_indices.numpy().reshape(-1, 6)
    test.assertEqual(tris_np.shape[0], quads_np.shape[0])
    np.testing.assert_array_equal(tris_np[:, [0, 1, 2]], quads_np[:, [0, 1, 2]])
    np.testing.assert_array_equal(tris_np[:, [3, 4, 5]], quads_np[:, [0, 2, 3]])

    # the stateful interface honors the constructor's topology selection
    iso = wp.geometry.IsoSurfaceNets(nx=node_dim, ny=node_dim, nz=node_dim, topology="quad")
    iso.surface(field=field, threshold=0.0)
    np.testing.assert_array_equal(iso.indices.numpy(), quad_indices.numpy())

    iso = wp.geometry.IsoSurfaceNets(nx=node_dim, ny=node_dim, nz=node_dim)
    iso.surface(field=field, threshold=0.0)
    np.testing.assert_array_equal(iso.indices.numpy(), tri_indices.numpy())


def test_surface_nets_closed_manifold(test, device):
    """Check that a fully interior isosurface is a closed 2-manifold mesh."""
    node_dim = 48
    field, _, _ = make_sphere_field(device, node_dim, radius=14.0)

    verts_np, quads_np = extract_quads(test, field)

    # every undirected quad edge is shared by exactly two quads
    edge_counts = quad_edge_counts(quads_np)
    test.assertTrue(all(count == 2 for count in edge_counts.values()))

    # Euler characteristic of a sphere-like closed surface
    euler = verts_np.shape[0] - len(edge_counts) + quads_np.shape[0]
    test.assertEqual(euler, 2)


def test_surface_nets_orientation_matches_marching_cubes(test, device):
    """Check that both backends produce outward-oriented (CCW from outside) triangles for an SDF."""
    node_dim = 48
    radius = 14.0
    field, _, _ = make_sphere_field(device, node_dim, radius=radius)

    sn_verts, sn_indices = wp.geometry.IsoSurfaceNets.extract(field, threshold=0.0)
    sn_volume = signed_volume(sn_verts.numpy(), sn_indices.numpy())

    mc_verts, mc_indices = wp.geometry.IsoSurfaceMarchingCubes.extract(field, threshold=0.0)
    mc_volume = signed_volume(mc_verts.numpy(), mc_indices.numpy())

    test.assertGreater(sn_volume, 0.0)
    test.assertGreater(mc_volume, 0.0)

    # the enclosed volume must also be close to the analytic sphere volume
    analytic = 4.0 / 3.0 * np.pi * radius**3
    test.assertLess(abs(sn_volume - analytic) / analytic, 0.05)


def test_surface_nets_multi_vertex_cells(test, device):
    """Check that cells crossed by two surface sheets produce one vertex per edge group.

    Two diagonally opposite inside nodes give the cell between them sign
    configuration 65 (corners 0 and 6 inside), an unambiguous two-sheet cell
    that must produce two vertices; naive one-vertex-per-cell surface nets
    would produce one.
    """
    node_dim = 4
    field_np = np.full((node_dim, node_dim, node_dim), 1.0, dtype=np.float32)
    field_np[1, 1, 1] = -1.0
    field_np[2, 2, 2] = -1.0
    field = wp.array(field_np, device=device)

    # expected vertex count from the ported table, summed over the mixed cells
    table = _surface_nets_module._SN_EDGE_GROUP_TABLE
    corners = _surface_nets_module._SN_CORNER_OFFSETS
    expected = 0
    for ci in range(node_dim - 1):
        for cj in range(node_dim - 1):
            for ck in range(node_dim - 1):
                signs = 0
                for bit, (dx, dy, dz) in enumerate(corners):
                    if field_np[ci + dx, cj + dy, ck + dz] < 0.0:
                        signs |= 1 << bit
                expected += table[signs * 13]

    verts_np, quads_np = extract_quads(test, field)
    test.assertEqual(verts_np.shape[0], expected)
    test.assertEqual(verts_np.shape[0], 16)  # 15 mixed cells, one of which is two-sheet

    # the two isolated surface components must still be closed and manifold,
    # and every generated vertex must be stitched into the quads
    test.assertTrue((np.unique(quads_np.flatten()) == np.arange(verts_np.shape[0])).all())
    edge_counts = quad_edge_counts(quads_np)
    test.assertTrue(all(count == 2 for count in edge_counts.values()))

    # the two sheets must stay disjoint: exactly two connected components,
    # which requires the stitching to address each vertex of the two-sheet
    # cell through its edge-group offset
    parent = list(range(verts_np.shape[0]))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for quad in quads_np:
        root = find(int(quad[0]))
        for i in range(1, 4):
            parent[find(int(quad[i]))] = root
    components = {find(i) for i in range(verts_np.shape[0])}
    test.assertEqual(len(components), 2)


def test_surface_nets_ambiguous_configs(test, device):
    """Exercise the ambiguous-face sign correction with random quantized fields."""
    ambiguous_face = _surface_nets_module._SN_AMBIGUOUS_FACE
    corners = _surface_nets_module._SN_CORNER_OFFSETS
    rng = np.random.default_rng(7)
    node_dim = 16

    ambiguous_cells = 0
    for _ in range(4):
        field_np = np.sign(rng.uniform(-1.0, 1.0, size=(node_dim, node_dim, node_dim))).astype(np.float32)

        # count ambiguous cells to prove this test exercises the correction
        for ci, cj, ck in np.ndindex(node_dim - 1, node_dim - 1, node_dim - 1):
            signs = 0
            for bit, (dx, dy, dz) in enumerate(corners):
                if field_np[ci + dx, cj + dy, ck + dz] < 0.0:
                    signs |= 1 << bit
            if ambiguous_face[signs] != 0:
                ambiguous_cells += 1

        _verts_np, quads_np = extract_quads(test, wp.array(field_np, device=device))

        # the surface is open at the domain boundary, but must stay manifold:
        # no undirected edge may be shared by more than two quads
        edge_counts = quad_edge_counts(quads_np)
        test.assertTrue(all(count <= 2 for count in edge_counts.values()))

    test.assertGreater(ambiguous_cells, 0)


def _reference_replica(field_np, threshold):
    """Independent NumPy reimplementation of OpenVDB's uniform meshing path.

    Derived from VolumeToMesh.h (v13.0.0) and the ported tables only (it
    shares no code with warp._src.surface_nets, and computes edge crossings
    from the corner-pair table rather than unrolled per-edge expressions), so
    it pins the intended semantics of every stage: classification, ambiguity
    correction, per-group vertex placement, stitching, and winding.
    """
    table = _surface_nets_module._SN_EDGE_GROUP_TABLE
    ambiguous_face = _surface_nets_module._SN_AMBIGUOUS_FACE
    corners = np.array(_surface_nets_module._SN_CORNER_OFFSETS)
    edges = _surface_nets_module._SN_EDGE_TO_CORNERS
    face_neighbor = {
        1: ((0, 0, -1), 3),
        2: ((1, 0, 0), 4),
        3: ((0, 0, 1), 1),
        4: ((-1, 0, 0), 2),
        5: ((0, -1, 0), 6),
        6: ((0, 1, 0), 5),
    }

    ncell = tuple(n - 1 for n in field_np.shape)

    def cell_signs(i, j, k):
        signs = 0
        for c in range(8):
            if field_np[i + corners[c][0], j + corners[c][1], k + corners[c][2]] < threshold:
                signs |= 1 << c
        return signs

    corrected, inside_flags, edge_flags, base = {}, {}, {}, {}
    verts = []
    for cell in np.ndindex(*ncell):
        signs = cell_signs(*cell)
        if signs in (0, 255):
            continue
        inside = signs & 1
        edge_flags[cell] = (
            inside != ((signs >> 1) & 1),
            inside != ((signs >> 4) & 1),
            inside != ((signs >> 3) & 1),
        )
        face = ambiguous_face[signs]
        if face:
            offset, complementary = face_neighbor[face]
            neighbor = tuple(np.array(cell) + offset)
            if all(0 <= neighbor[axis] < ncell[axis] for axis in range(3)):
                if ambiguous_face[cell_signs(*neighbor)] == complementary:
                    signs = (~signs) & 0xFF
        corrected[cell] = signs
        inside_flags[cell] = bool(inside)
        base[cell] = len(verts)
        for group in range(1, table[signs * 13] + 1):
            accum, count = np.zeros(3), 0
            for e in range(12):
                if table[signs * 13 + 1 + e] == group:
                    c0, c1 = edges[e]
                    v0 = float(field_np[tuple(np.array(cell) + corners[c0])])
                    v1 = float(field_np[tuple(np.array(cell) + corners[c1])])
                    t = (threshold - v0) / (v1 - v0)
                    accum += corners[c0] + t * (corners[c1] - corners[c0])
                    count += 1
            verts.append(np.array(cell, dtype=float) + accum / count)

    def point(cell, column):
        signs = corrected[cell]
        offset = table[signs * 13 + column] - 1 if table[signs * 13] > 1 else 0
        return base[cell] + offset

    quads = []
    for cell in sorted(corrected):
        ci, cj, ck = cell
        xedge, yedge, zedge = edge_flags[cell]
        inside = inside_flags[cell]
        if xedge and cj > 0 and ck > 0:
            quad = [
                point(cell, 1),
                point((ci, cj - 1, ck), 5),
                point((ci, cj - 1, ck - 1), 7),
                point((ci, cj, ck - 1), 3),
            ]
            quads.append(quad if inside else quad[::-1])
        if yedge and ci > 0 and ck > 0:
            quad = [
                point(cell, 9),
                point((ci, cj, ck - 1), 12),
                point((ci - 1, cj, ck - 1), 11),
                point((ci - 1, cj, ck), 10),
            ]
            quads.append(quad if inside else quad[::-1])
        if zedge and ci > 0 and cj > 0:
            quad = [
                point(cell, 4),
                point((ci, cj - 1, ck), 8),
                point((ci - 1, cj - 1, ck), 6),
                point((ci - 1, cj, ck), 2),
            ]
            quads.append(quad[::-1] if inside else quad)
    return np.array(verts).reshape(-1, 3), np.array(quads, dtype=np.int64).reshape(-1, 4)


def test_surface_nets_matches_reference_replica(test, device):
    """Check that the full pipeline agrees with an independent replica of the OpenVDB semantics."""
    ambiguous_face = _surface_nets_module._SN_AMBIGUOUS_FACE
    corners = _surface_nets_module._SN_CORNER_OFFSETS
    rng = np.random.default_rng(7)
    node_dim = 8

    ambiguous_cells = 0
    for trial in range(6):
        field_np = rng.uniform(-1.0, 1.0, size=(node_dim, node_dim, node_dim)).astype(np.float32)
        if trial >= 3:
            # quantized fields hammer the multi-sheet and ambiguous configurations
            field_np = np.sign(field_np).astype(np.float32) + np.float32(0.5) * (trial - 4)

        for ci, cj, ck in np.ndindex(node_dim - 1, node_dim - 1, node_dim - 1):
            signs = 0
            for bit, (dx, dy, dz) in enumerate(corners):
                if field_np[ci + dx, cj + dy, ck + dz] < 0.0:
                    signs |= 1 << bit
            if ambiguous_face[signs] != 0:
                ambiguous_cells += 1

        ref_verts, ref_quads = _reference_replica(field_np, 0.0)
        verts_np, quads_np = extract_quads(test, wp.array(field_np, device=device))

        test.assertEqual(verts_np.shape[0], ref_verts.shape[0])
        test.assertEqual(quads_np.shape[0], ref_quads.shape[0])
        if ref_verts.shape[0] > 0:
            np.testing.assert_allclose(verts_np, ref_verts, atol=1e-5)
            np.testing.assert_array_equal(quads_np, ref_quads)

    # prove that the trials exercised the ambiguity correction
    test.assertGreater(ambiguous_cells, 0)


def test_surface_nets_nonuniform(test, device):
    """Test the logic for when the dimensions of the grid are not uniform."""

    dim_x = 64
    dim_y = 48
    dim_z = 72
    field = wp.zeros(shape=(dim_x, dim_y, dim_z), dtype=float, device=device)

    # anisotropic domain bounds: each axis gets a different cell size
    bounds_low = wp.vec3(-1.0, 0.0, 2.0)
    bounds_high = wp.vec3(float(dim_x) * 0.5 - 1.0, float(dim_y) * 2.0, float(dim_z) + 2.0)

    radius = dim_x / 4.0
    center = wp.vec3(dim_x / 2, dim_y / 2, dim_z / 2)
    wp.launch(make_field_sphere_sdf, dim=field.shape, inputs=[field, center, radius], device=device)

    iso = wp.geometry.IsoSurfaceNets(
        nx=dim_x,
        ny=dim_y,
        nz=dim_z,
        domain_bounds_lower_corner=bounds_low,
        domain_bounds_upper_corner=bounds_high,
    )
    iso.surface(field=field, threshold=0.0)
    verts_np = iso.verts.numpy()
    validate_surface_nets_output(test, verts_np, iso.indices.numpy())

    # map the vertices back to index space and check they lie on the sphere
    lower = np.array(bounds_low)
    delta = (np.array(bounds_high) - lower) / np.array([dim_x - 1, dim_y - 1, dim_z - 1])
    index_space_verts = (verts_np - lower) / delta
    error = np.abs(np.linalg.norm(index_space_verts - np.array(center), axis=1) - radius)
    test.assertTrue(np.max(error) < 1.0)


def test_surface_nets_empty_output(test, device):
    """Make sure we handle the empty-output case correctly."""

    dim = 64
    field, _, _ = make_sphere_field(device, dim)

    iso = wp.geometry.IsoSurfaceNets(nx=dim, ny=dim, nz=dim)

    iso.surface(field=field, threshold=1000.0)  # set threshold to a large value so that no quads are generated
    test.assertEqual(iso.verts.shape, (0,))
    test.assertEqual(iso.indices.shape, (0,))
    validate_surface_nets_output(test, iso.verts.numpy(), iso.indices.numpy(), check_nonempty=False)

    # empty outputs keep their expected shapes in the other topology too
    verts, indices = wp.geometry.IsoSurfaceNets.extract(field, threshold=1000.0, topology="quad")
    test.assertEqual(verts.shape, (0,))
    test.assertEqual(indices.shape, (0,))


def test_surface_nets_open_boundary(test, device):
    """Check that the mesh is left open where the isosurface exits the grid domain."""
    node_dim = 32
    field = wp.zeros(shape=(node_dim, node_dim, node_dim), dtype=float, device=device)

    # sphere centered on a domain corner: only one octant is inside the grid
    wp.launch(
        make_field_sphere_sdf,
        dim=field.shape,
        inputs=[field, wp.vec3(0.0, 0.0, 0.0), node_dim / 2.0],
        device=device,
    )

    _verts_np, quads_np = extract_quads(test, field)

    # boundary edges (exactly one incident quad) must exist on the open rim
    edge_counts = quad_edge_counts(quads_np)
    test.assertTrue(all(count <= 2 for count in edge_counts.values()))
    test.assertGreater(sum(1 for count in edge_counts.values() if count == 1), 0)


def test_surface_nets_differentiable(test, device):
    """Check that surface nets extraction has reasonable gradients.

    This test constructs an SDF of a sphere, extracts a surface, computes its
    surface area, and then differentiates the surface area with respect to
    the sphere's radius.
    """
    node_dim = 64
    bounds_low = wp.vec3(-1.0, -1.0, -1.0)
    bounds_high = wp.vec3(1.0, 1.0, 1.0)

    radius = 0.5
    radius_wp = wp.full((1,), value=0.5, dtype=wp.float32, device=device, requires_grad=True)

    with wp.Tape() as tape:
        field = wp.zeros(shape=(node_dim, node_dim, node_dim), dtype=float, device=device, requires_grad=True)
        wp.launch(
            make_field_sphere_sdf_unit_domain,
            dim=field.shape,
            inputs=[field, wp.vec3(0.0, 0.0, 0.0), radius_wp],
            device=device,
        )

        verts, indices = wp.geometry.IsoSurfaceNets.extract(
            field, threshold=0.0, domain_bounds_lower_corner=bounds_low, domain_bounds_upper_corner=bounds_high
        )
        test.assertTrue(verts.requires_grad)

        # compute surface area
        area = wp.zeros(shape=(1,), dtype=float, device=device, requires_grad=True)
        wp.launch(compute_surface_area, dim=indices.shape[0] // 3, inputs=[verts, indices, area], device=device)

        # confirm surface area is correct vs. the analytical ground truth
        area_np = area.numpy()[0]
        analytic_area = 4.0 * np.pi * radius * radius
        test.assertLess(abs(area_np - analytic_area) / analytic_area, 0.05)

    # compute the gradient of the surface area with respect to the radius
    tape.backward(area)

    # confirm the gradient is correct vs. the analytical ground truth
    grad_np = radius_wp.grad.numpy()[0]
    analytic_grad = 8.0 * np.pi * radius
    test.assertLess(abs(grad_np - analytic_grad) / analytic_grad, 0.05)


def test_iso_surface_base_contract(test, device, extractor_class):
    """Check that backends are interchangeable behind the wp.geometry.IsoSurfaceBase interface."""
    node_dim = 32
    field, _, _ = make_sphere_field(device, node_dim)

    test.assertTrue(issubclass(extractor_class, wp.geometry.IsoSurfaceBase))

    # stateful interface
    iso = extractor_class(node_dim, node_dim, node_dim)
    test.assertIsInstance(iso, wp.geometry.IsoSurfaceBase)
    test.assertIsNone(iso.verts)
    test.assertIsNone(iso.indices)
    iso.surface(field, 0.0)
    test.assertEqual(iso.verts.dtype, wp.vec3f)
    test.assertEqual(iso.indices.dtype, wp.int32)
    test.assertEqual(iso.indices.shape[0] % 3, 0)

    # stateless interface: the outputs are always (verts, tris)
    verts, tris = extractor_class.extract(field, threshold=0.0)
    test.assertEqual(verts.shape[0], iso.verts.shape[0])
    test.assertEqual(tris.shape[0], iso.indices.shape[0])


devices = get_test_devices()


class TestSurfaceNets(unittest.TestCase):
    def test_sn_edge_group_table_invariants(self):
        """Validate the ported OpenVDB tables against derivable invariants."""
        table = _surface_nets_module._SN_EDGE_GROUP_TABLE
        ambiguous_face = _surface_nets_module._SN_AMBIGUOUS_FACE
        corners = _surface_nets_module._SN_CORNER_OFFSETS
        edges = _surface_nets_module._SN_EDGE_TO_CORNERS

        self.assertEqual(len(table), 256 * 13)
        self.assertEqual(len(ambiguous_face), 256)
        self.assertEqual(len(corners), 8)
        self.assertEqual(len(edges), 12)

        # each edge connects two corners differing along exactly one axis
        for c0, c1 in edges:
            offset0, offset1 = corners[c0], corners[c1]
            self.assertEqual(sum(abs(offset0[axis] - offset1[axis]) for axis in range(3)), 1)

        for signs in range(256):
            row = table[signs * 13 : signs * 13 + 13]
            num_points = row[0]
            self.assertIn(num_points, range(5))
            self.assertIn(ambiguous_face[signs], range(7))

            if signs in (0, 255):
                # cells without a sign change produce nothing
                self.assertTrue(all(value == 0 for value in row))
                self.assertEqual(ambiguous_face[signs], 0)
                continue

            self.assertGreaterEqual(num_points, 1)
            groups = set()
            for e, (c0, c1) in enumerate(edges):
                crossed = ((signs >> c0) & 1) != ((signs >> c1) & 1)
                # an edge belongs to a group iff its corner signs differ
                self.assertEqual(row[1 + e] != 0, crossed, msg=f"signs={signs}, edge={e}")
                if row[1 + e] != 0:
                    groups.add(row[1 + e])
            # the nonzero groups are exactly 1..num_points
            self.assertEqual(groups, set(range(1, num_points + 1)), msg=f"signs={signs}")


add_function_test(TestSurfaceNets, "test_surface_nets", test_surface_nets, devices=devices)
add_function_test(TestSurfaceNets, "test_surface_nets_functional", test_surface_nets_functional, devices=devices)
add_function_test(TestSurfaceNets, "test_surface_nets_topology", test_surface_nets_topology, devices=devices)
add_function_test(
    TestSurfaceNets, "test_surface_nets_closed_manifold", test_surface_nets_closed_manifold, devices=devices
)
add_function_test(
    TestSurfaceNets,
    "test_surface_nets_orientation_matches_marching_cubes",
    test_surface_nets_orientation_matches_marching_cubes,
    devices=devices,
)
add_function_test(
    TestSurfaceNets, "test_surface_nets_multi_vertex_cells", test_surface_nets_multi_vertex_cells, devices=devices
)
add_function_test(
    TestSurfaceNets, "test_surface_nets_ambiguous_configs", test_surface_nets_ambiguous_configs, devices=devices
)
add_function_test(
    TestSurfaceNets,
    "test_surface_nets_matches_reference_replica",
    test_surface_nets_matches_reference_replica,
    devices=devices,
)
add_function_test(TestSurfaceNets, "test_surface_nets_nonuniform", test_surface_nets_nonuniform, devices=devices)
add_function_test(TestSurfaceNets, "test_surface_nets_empty_output", test_surface_nets_empty_output, devices=devices)
add_function_test(TestSurfaceNets, "test_surface_nets_open_boundary", test_surface_nets_open_boundary, devices=devices)
add_function_test(
    TestSurfaceNets, "test_surface_nets_differentiable", test_surface_nets_differentiable, devices=devices
)
for _extractor_class in (wp.geometry.IsoSurfaceMarchingCubes, wp.geometry.IsoSurfaceNets):
    add_function_test(
        TestSurfaceNets,
        f"test_iso_surface_base_contract_{_extractor_class.__name__}",
        test_iso_surface_base_contract,
        devices=devices,
        extractor_class=_extractor_class,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
