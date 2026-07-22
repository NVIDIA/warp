# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import platform
import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp.tests.unittest_utils import *


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_environment_cells(s: fem.Sample, domain: fem.Domain, base_cell_count: int):
    env_index = fem.environment_index(domain, s)
    wp.expect_eq(env_index, s.element_index // base_cell_count)

    pos = domain(s)
    lookup_guess = fem.lookup(domain, pos, s)
    wp.expect_eq(lookup_guess.element_index, s.element_index)
    wp.expect_near(domain(lookup_guess), pos, 0.001)

    lookup_s = fem.lookup(domain, pos, env_index)
    wp.expect_eq(lookup_s.element_index, s.element_index)
    wp.expect_near(domain(lookup_s), pos, 0.001)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_deformed_environment_lookup(s: fem.Sample, domain: fem.Domain):
    env_index = fem.environment_index(domain, s)
    pos = domain(s)
    lookup_s = fem.lookup(domain, pos, env_index)
    wp.expect_eq(lookup_s.element_index, s.element_index)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_environment_sides(
    s: fem.Sample,
    domain: fem.Domain,
    base_cell_count: int,
    base_side_count: int,
):
    env_index = fem.environment_index(domain, s)
    wp.expect_eq(env_index, s.element_index // base_side_count)

    inner_s = fem.to_inner_cell(domain, s)
    outer_s = fem.to_outer_cell(domain, s)

    wp.expect_eq(inner_s.element_index // base_cell_count, env_index)
    wp.expect_eq(outer_s.element_index // base_cell_count, env_index)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_sparse_environment_cells(s: fem.Sample, domain: fem.Domain, cell_env: wp.array[int]):
    env_index = fem.environment_index(domain, s)
    wp.expect_eq(env_index, cell_env[s.element_index])

    pos = domain(s)
    lookup_guess = fem.lookup(domain, pos, s)
    wp.expect_eq(lookup_guess.element_index, s.element_index)
    wp.expect_near(domain(lookup_guess), pos, 0.001)

    lookup_s = fem.lookup(domain, pos, env_index)
    wp.expect_eq(lookup_s.element_index, s.element_index)
    wp.expect_near(domain(lookup_s), pos, 0.001)

    far_pos = pos + type(pos)(pos.dtype(4.0), pos.dtype(0.0), pos.dtype(0.0))
    filtered_s = fem.lookup(domain, far_pos, 10.0, cell_env, env_index, env_index)
    wp.expect_neq(filtered_s.element_index, fem.NULL_ELEMENT_INDEX)
    if filtered_s.element_index != fem.NULL_ELEMENT_INDEX:
        wp.expect_eq(cell_env[filtered_s.element_index], env_index)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_sparse_lookup_excludes_neighbor_env(s: fem.Sample, domain: fem.Domain, cell_env: wp.array[int]):
    env_index = fem.environment_index(domain, s)
    if env_index == 0:
        pos = domain(s)
        neighbor_pos = pos + type(pos)(pos.dtype(1.0), pos.dtype(0.0), pos.dtype(0.0))
        lookup_s = fem.lookup(domain, neighbor_pos, 2.0, 0)
        wp.expect_eq(lookup_s.element_index, s.element_index)
        if lookup_s.element_index != fem.NULL_ELEMENT_INDEX:
            wp.expect_eq(cell_env[lookup_s.element_index], 0)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_sparse_environment_sides(s: fem.Sample, domain: fem.Domain, cell_env: wp.array[int]):
    env_index = fem.environment_index(domain, s)

    cells = fem.cells(domain)
    inner_s = fem.to_inner_cell(domain, s)
    outer_s = fem.to_outer_cell(domain, s)

    wp.expect_eq(cell_env[inner_s.element_index], env_index)
    wp.expect_eq(cell_env[outer_s.element_index], env_index)
    wp.expect_near(cells(inner_s), domain(s), 0.001)
    wp.expect_near(cells(outer_s), domain(s), 0.001)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_mesh_environment_cells(s: fem.Sample, domain: fem.Domain, cell_env: wp.array[int]):
    env_index = fem.environment_index(domain, s)
    wp.expect_eq(env_index, cell_env[s.element_index])

    pos = domain(s)
    lookup_guess = fem.lookup(domain, pos, s)
    wp.expect_eq(lookup_guess.element_index, s.element_index)
    wp.expect_near(domain(lookup_guess), pos, 0.001)

    lookup_s = fem.lookup(domain, pos, env_index)
    wp.expect_eq(lookup_s.element_index, s.element_index)
    wp.expect_near(domain(lookup_s), pos, 0.001)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_mesh_environment_sides(s: fem.Sample, domain: fem.Domain, cell_env: wp.array[int]):
    env_index = fem.environment_index(domain, s)

    inner_s = fem.to_inner_cell(domain, s)
    outer_s = fem.to_outer_cell(domain, s)

    wp.expect_eq(cell_env[inner_s.element_index], env_index)
    wp.expect_eq(cell_env[outer_s.element_index], env_index)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_empty_environment_lookup(s: fem.Sample, domain: fem.Domain, empty_env: int):
    pos = domain(s)
    lookup_s = fem.lookup(domain, pos, empty_env)
    wp.expect_eq(lookup_s.element_index, fem.NULL_ELEMENT_INDEX)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_no_env_lookup_rejected(s: fem.Sample, domain: fem.Domain):
    pos = domain(s)
    lookup_s = fem.lookup(domain, pos)
    wp.expect_eq(lookup_s.element_index, s.element_index)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_pic_environment_cells(s: fem.Sample, domain: fem.Domain, particle_env: wp.array[int]):
    env_index = fem.environment_index(domain, s)
    wp.expect_eq(env_index, particle_env[s.qp_index])


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _environment_value(s: fem.Sample, domain: fem.Domain):
    return 3.0 + 4.0 * float(fem.environment_index(domain, s))


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_nonconforming_environment_field(s: fem.Sample, domain: fem.Domain, source: fem.Field):
    expected = _environment_value(s, domain)
    wp.expect_near(source(s), expected, 0.001)


def _assert_multi_env_node_isolation(test, geo: fem.Geometry, space: fem.FunctionSpace, base_cell_count: int, device):
    base_node_count = space.node_count() // geo.environment_count()

    with wp.ScopedDevice(device):
        element_nodes = space.topology.element_node_indices().numpy()

    for env_index in range(geo.environment_count()):
        nodes = element_nodes[env_index * base_cell_count : (env_index + 1) * base_cell_count]
        nodes = nodes[nodes != fem.NULL_NODE_INDEX]
        test.assertTrue(np.all(nodes >= env_index * base_node_count))
        test.assertTrue(np.all(nodes < (env_index + 1) * base_node_count))


def _assert_sparse_multi_env_node_isolation(test, geo: fem.Geometry, space: fem.FunctionSpace, cell_env, device):
    with wp.ScopedDevice(device):
        cell_env_np = cell_env.numpy()
        element_nodes = space.topology.element_node_indices().numpy()

    env_nodes = []
    for env_index in range(geo.environment_count()):
        nodes = element_nodes[cell_env_np == env_index]
        nodes = nodes[nodes != fem.NULL_NODE_INDEX]
        test.assertGreater(nodes.size, 0)
        env_nodes.append(set(nodes.tolist()))

    for i in range(len(env_nodes)):
        for j in range(i + 1, len(env_nodes)):
            test.assertFalse(env_nodes[i] & env_nodes[j])


def _assert_environment_first_pressure_partition(test, geo: fem.Geometry, cell_env, device):
    pressure_space = fem.make_polynomial_space(geo, degree=0, discontinuous=True)
    pressure_partition = fem.make_space_partition(
        space_topology=pressure_space.topology,
        environment_first=True,
        device=device,
    )

    test.assertEqual(pressure_partition.node_count(), pressure_space.node_count())
    env_offsets = pressure_partition.env_offsets
    test.assertIsNotNone(env_offsets)

    with wp.ScopedDevice(device):
        cell_env_np = cell_env.numpy()
        offsets_np = env_offsets.numpy()
        node_indices = pressure_partition.space_node_indices().numpy()

    ref_offsets = [0]
    for env_index in range(geo.environment_count()):
        ref_offsets.append(ref_offsets[-1] + int(np.count_nonzero(cell_env_np == env_index)))
    ref_offsets = np.array(ref_offsets, dtype=np.int32)

    np.testing.assert_array_equal(offsets_np, ref_offsets)

    max_node_count = pressure_space.node_count() - 1
    capped_partition = fem.make_space_partition(
        space_topology=pressure_space.topology,
        environment_first=True,
        max_node_count=max_node_count,
        device=device,
    )

    test.assertEqual(capped_partition.node_count(), max_node_count)
    np.testing.assert_array_equal(capped_partition.space_node_indices().numpy(), node_indices[:max_node_count])
    np.testing.assert_array_equal(capped_partition.env_offsets.numpy(), np.minimum(ref_offsets, max_node_count))

    for env_index in range(geo.environment_count()):
        env_node_indices = node_indices[offsets_np[env_index] : offsets_np[env_index + 1]]
        np.testing.assert_array_equal(cell_env_np[env_node_indices], np.full(env_node_indices.shape, env_index))


def _make_env_offsets(offsets, device):
    return wp.array(np.array(offsets, dtype=np.int32), dtype=wp.vec3i, device=device)


def _pack_env_voxels(env_cells, device):
    cell_np = np.concatenate([env_cell.numpy() for env_cell in env_cells], axis=0)
    env_np = np.concatenate(
        [np.full(env_cell.shape[0], env_index, dtype=np.int32) for env_index, env_cell in enumerate(env_cells)]
    )
    return (
        wp.array(cell_np, dtype=env_cells[0].dtype, device=device),
        wp.array(env_np, dtype=wp.int32, device=device),
        len(env_cells),
    )


def _pack_env_voxels_with_levels(env_cells, env_levels, device):
    points, point_envs, env_count = _pack_env_voxels(env_cells, device)
    level_np = np.concatenate([env_level.numpy() for env_level in env_levels], axis=0)
    return points, wp.array(level_np, dtype=wp.uint8, device=device), point_envs, env_count


def _make_multi_env_nanogrid(device, env_offsets=None, point_dtype=wp.vec3i):
    env_cells = (
        wp.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=wp.vec3i, device=device),
        wp.array([[0, 0, 0], [1, 0, 0]], dtype=wp.vec3i, device=device),
    )
    points, point_envs, env_count = _pack_env_voxels(env_cells, device)
    if point_dtype == wp.vec3f:
        points = wp.array(points.numpy().astype(np.float32), dtype=wp.vec3f, device=device)

    geo = fem.Nanogrid.from_environment_voxels(
        points,
        point_envs,
        env_count,
        env_offsets=env_offsets,
        voxel_size=1.0,
        device=device,
    )
    return geo, geo.cell_env


def _make_multi_env_adaptive_nanogrid(device, env_offsets=None):
    fine_cells = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                fine_cells.append([i, j, k])

    env_cells = (
        wp.array([[0, 0, 0]], dtype=wp.vec3i, device=device),
        wp.array(fine_cells, dtype=wp.vec3i, device=device),
    )
    env_levels = (
        wp.array([1], dtype=wp.uint8, device=device),
        wp.array([0] * len(fine_cells), dtype=wp.uint8, device=device),
    )
    points, cell_levels, point_envs, env_count = _pack_env_voxels_with_levels(env_cells, env_levels, device)

    geo = fem.AdaptiveNanogrid.from_environment_voxels(
        points,
        cell_levels,
        point_envs,
        env_count,
        level_count=2,
        env_offsets=env_offsets,
        voxel_size=0.5,
        device=device,
    )
    return geo, geo.cell_env


def _make_multi_env_trimesh(device):
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    tri_vertex_indices = wp.array([[0, 1, 2], [3, 4, 5]], dtype=int, device=device)
    cell_env = wp.array([0, 1], dtype=wp.int32, device=device)
    return fem.Trimesh3D(
        tri_vertex_indices,
        positions,
        build_bvh=True,
        cell_env=cell_env,
        env_count=2,
    )


def _make_multi_env_quadmesh(device):
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    quad_vertex_indices = wp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=int, device=device)
    cell_env = wp.array([0, 1], dtype=wp.int32, device=device)
    return fem.Quadmesh3D(
        quad_vertex_indices,
        positions,
        build_bvh=True,
        cell_env=cell_env,
        env_count=2,
    )


def _make_multi_env_tetmesh(device):
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    tet_vertex_indices = wp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=int, device=device)
    cell_env = wp.array([0, 1], dtype=wp.int32, device=device)
    return fem.Tetmesh(
        tet_vertex_indices,
        positions,
        build_bvh=True,
        cell_env=cell_env,
        env_count=2,
    )


def _make_multi_env_hexmesh(device):
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    hex_vertex_indices = wp.array(
        [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        dtype=int,
        device=device,
    )
    cell_env = wp.array([0, 1], dtype=wp.int32, device=device)
    return fem.Hexmesh(
        hex_vertex_indices,
        positions,
        assume_parallelepiped_cells=True,
        build_bvh=True,
        cell_env=cell_env,
        env_count=2,
    )


def _test_pic_multi_env(test, device, geo, particle_positions, domain=None, use_domain_element_indices=False):
    if domain is None:
        domain = fem.Cells(geo)

    particle_env = wp.array([0, 1], dtype=int, device=device)
    pic = fem.PicQuadrature(
        domain,
        positions=particle_positions,
        env_indices=particle_env,
        use_domain_element_indices=use_domain_element_indices,
    )

    test.assertEqual(pic.total_point_count(), 2)
    test.assertEqual(pic.evaluation_point_count(), 2)
    test.assertEqual(pic.active_cell_count(), 2)
    test.assertEqual(pic.max_points_per_element(), 1)

    fem.interpolate(_test_pic_environment_cells, at=pic, values={"particle_env": particle_env})


def _assert_multi_env_lookup_requires_environment(test, geo, device):
    if not wp.get_device(device).is_cpu:
        return

    domain = fem.Cells(geo)
    with test.assertRaisesRegex(wp.WarpCodegenError, "Couldn't find function overload.*cell_lookup"):
        fem.interpolate(_test_no_env_lookup_rejected, at=fem.RegularQuadrature(domain, order=1))


def _test_nonconforming_multi_env(test, device, geo):
    source_space = fem.make_polynomial_space(geo, degree=0, discontinuous=True)
    source_field = source_space.make_field()
    fem.interpolate(_environment_value, dest=source_field)

    cell_domain = fem.Cells(geo)
    nonconforming = fem.NonconformingField(cell_domain, source_field, background=-1.0)
    fem.interpolate(
        _test_nonconforming_environment_field,
        at=fem.RegularQuadrature(cell_domain, order=1),
        fields={"source": nonconforming},
    )
    fem.interpolate(
        _test_nonconforming_environment_field,
        at=fem.RegularQuadrature(fem.Sides(geo), order=1),
        fields={"source": nonconforming.trace()},
    )


def _test_mesh_multi_env(test, device, geo, particle_positions):
    cell_env = geo.cell_env

    test.assertEqual(geo.environment_count(), 2)
    test.assertIn("_env", geo.name)
    np.testing.assert_array_equal(cell_env.numpy(), np.array([0, 1], dtype=np.int32))

    with wp.ScopedDevice(device):
        domain = fem.Cells(geo)
        cell_quadrature = fem.RegularQuadrature(domain, order=2)
        side_quadrature = fem.RegularQuadrature(fem.Sides(geo), order=2)

        fem.interpolate(_test_mesh_environment_cells, at=cell_quadrature, values={"cell_env": cell_env})
        fem.interpolate(_test_mesh_environment_sides, at=side_quadrature, values={"cell_env": cell_env})
        _assert_multi_env_lookup_requires_environment(test, geo, device)

        _test_pic_multi_env(test, device, geo, particle_positions)
        _test_nonconforming_multi_env(test, device, geo)
        _assert_environment_first_pressure_partition(test, geo, cell_env, device)


def _test_mesh_multi_env_constructor_errors(test, device):
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    tri_vertex_indices = wp.array([[0, 1, 2]], dtype=int, device=device)
    cell_env = wp.array([0], dtype=wp.int32, device=device)

    with test.assertRaisesRegex(ValueError, "Environment count must be provided"):
        fem.Trimesh3D(tri_vertex_indices, positions, cell_env=cell_env)

    with test.assertRaisesRegex(ValueError, "dtype wp.int32"):
        fem.Trimesh3D(
            tri_vertex_indices,
            positions,
            cell_env=wp.array([0], dtype=wp.int64, device=device),
            env_count=1,
        )

    with test.assertRaisesRegex(ValueError, "one entry per cell"):
        fem.Trimesh3D(
            tri_vertex_indices,
            positions,
            cell_env=wp.array([0, 1], dtype=wp.int32, device=device),
            env_count=2,
        )


def _test_mesh_empty_environment_lookup(test, device):
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    tri_vertex_indices = wp.array([[0, 1, 2]], dtype=int, device=device)
    cell_env = wp.array([0], dtype=wp.int32, device=device)
    geo = fem.Trimesh3D(
        tri_vertex_indices,
        positions,
        build_bvh=True,
        cell_env=cell_env,
        env_count=2,
    )

    with wp.ScopedDevice(device):
        domain = fem.Cells(geo)
        fem.interpolate(
            _test_empty_environment_lookup,
            at=fem.RegularQuadrature(domain, order=1),
            values={"empty_env": 1},
        )

        pic = fem.PicQuadrature(
            domain,
            positions=wp.array([[0.25, 0.25, 0.0]], dtype=wp.vec3, device=device),
            env_indices=wp.array([1], dtype=int, device=device),
        )
        test.assertEqual(pic.active_cell_count(), 0)
        test.assertEqual(pic.max_points_per_element(), 0)


def _test_nonconforming_multi_env_constructor_errors(test):
    source_geo = fem.Grid2D(res=wp.vec2i(1), env_count=2)
    source_space = fem.make_polynomial_space(source_geo, degree=0, discontinuous=True)
    source_field = source_space.make_field()
    target_geo = fem.Grid2D(res=wp.vec2i(1), env_count=3)

    with test.assertRaisesRegex(ValueError, "matching environment counts"):
        fem.NonconformingField(fem.Cells(target_geo), source_field)


def _test_pic_partition_multi_env(test, device):
    geo = fem.Grid2D(res=wp.vec2i(2), env_count=2)
    cell_mask = np.zeros(geo.cell_count(), dtype=np.int32)
    cell_mask[0] = 1
    cell_mask[4] = 1

    partition = fem.ExplicitGeometryPartition(
        geo,
        wp.array(cell_mask, dtype=int, device=device),
        max_cell_count=2,
    )
    domain = fem.Cells(partition)
    particle_positions = wp.array([[0.25, 0.25], [0.25, 0.25]], dtype=wp.vec2, device=device)

    _test_pic_multi_env(test, device, geo, particle_positions, domain=domain)
    _test_pic_multi_env(test, device, geo, particle_positions, domain=domain, use_domain_element_indices=True)

    pressure_space = fem.make_polynomial_space(geo, degree=0, discontinuous=True)
    with test.assertRaisesRegex(ValueError, "with_halo=True"):
        fem.make_space_partition(
            space_topology=pressure_space.topology,
            geometry_partition=partition,
            environment_first=True,
            device=device,
        )

    pressure_partition = fem.make_space_partition(
        space_topology=pressure_space.topology,
        geometry_partition=partition,
        environment_first=True,
        with_halo=False,
        device=device,
    )

    test.assertEqual(pressure_partition.node_count(), 2)
    np.testing.assert_array_equal(pressure_partition.env_offsets.numpy(), np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(pressure_partition.space_node_indices().numpy(), np.array([0, 4], dtype=np.int32))

    no_halo_partition = fem.make_space_partition(
        space_topology=pressure_space.topology,
        geometry_partition=partition,
        with_halo=False,
        device=device,
    )
    halo_partition = fem.make_space_partition(
        space_topology=pressure_space.topology,
        geometry_partition=partition,
        with_halo=True,
        device=device,
    )

    test.assertEqual(pressure_partition.node_count(), no_halo_partition.node_count())
    np.testing.assert_array_equal(
        pressure_partition.space_node_indices().numpy(), no_halo_partition.space_node_indices().numpy()
    )
    test.assertGreater(halo_partition.node_count(), no_halo_partition.node_count())

    capped_pressure_partition = fem.make_space_partition(
        space_topology=pressure_space.topology,
        geometry_partition=partition,
        environment_first=True,
        with_halo=False,
        max_node_count=1,
        device=device,
    )

    test.assertEqual(capped_pressure_partition.node_count(), 1)
    np.testing.assert_array_equal(capped_pressure_partition.env_offsets.numpy(), np.array([0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(capped_pressure_partition.space_node_indices().numpy(), np.array([0], dtype=np.int32))

    capacity = 4
    underfilled_pressure_partition = fem.make_space_partition(
        space_topology=pressure_space.topology,
        geometry_partition=partition,
        environment_first=True,
        with_halo=False,
        max_node_count=capacity,
        device=device,
    )

    test.assertEqual(underfilled_pressure_partition.node_count(), capacity)
    np.testing.assert_array_equal(
        underfilled_pressure_partition.space_node_indices().numpy()[:2], np.array([0, 4], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        underfilled_pressure_partition.env_offsets.numpy(), np.array([0, 1, 2], dtype=np.int32)
    )
    expected_space_to_partition = np.full(pressure_space.node_count(), fem.NULL_NODE_INDEX, dtype=np.int32)
    expected_space_to_partition[[0, 4]] = [0, 1]
    np.testing.assert_array_equal(
        underfilled_pressure_partition._space_to_partition.numpy(), expected_space_to_partition
    )


def _test_pic_env_indices_api(test, device):
    geo = fem.Grid2D(res=wp.vec2i(2), env_count=2)
    domain = fem.Cells(geo)
    particle_positions = wp.array([[0.25, 0.25], [0.25, 0.25]], dtype=wp.vec2, device=device)

    pic = fem.PicQuadrature(
        domain,
        positions=particle_positions,
        env_indices=wp.array([0, 1], dtype=wp.int64, device=device),
    )
    test.assertEqual(pic.total_point_count(), 2)
    test.assertEqual(pic.evaluation_point_count(), 2)
    test.assertEqual(pic.active_cell_count(), 2)

    with test.assertRaisesRegex(ValueError, "one entry per particle position"):
        fem.PicQuadrature(
            domain,
            positions=particle_positions,
            env_indices=wp.array([0], dtype=int, device=device),
        )

    with test.assertRaisesRegex(ValueError, "integer dtype"):
        fem.PicQuadrature(
            domain,
            positions=particle_positions,
            env_indices=wp.array([0.0, 1.0], dtype=float, device=device),
        )

    with test.assertRaisesRegex(ValueError, "environment indices"):
        fem.PicQuadrature(domain, positions=particle_positions)


def _test_sparse_grid_multi_env(test, device, geo, cell_env):
    test.assertEqual(geo.environment_count(), 2)
    test.assertIn("_env", geo.name)

    with wp.ScopedDevice(device):
        cell_quadrature = fem.RegularQuadrature(fem.Cells(geo), order=2)
        side_quadrature = fem.RegularQuadrature(fem.Sides(geo), order=2)

        fem.interpolate(_test_sparse_environment_cells, at=cell_quadrature, values={"cell_env": cell_env})
        fem.interpolate(_test_sparse_environment_sides, at=side_quadrature, values={"cell_env": cell_env})
        _assert_multi_env_lookup_requires_environment(test, geo, device)
        _test_pic_multi_env(
            test, device, geo, wp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        )
        _test_nonconforming_multi_env(test, device, geo)

        for space_kwargs in (
            {"degree": 1},
            {"degree": 2},
            {"degree": 2, "element_basis": fem.ElementBasis.SERENDIPITY},
        ):
            space = fem.make_polynomial_space(geo, **space_kwargs)
            _assert_sparse_multi_env_node_isolation(test, geo, space, cell_env, device)


def _test_sparse_grid_lookup_with_adjacent_env_offsets(test, device):
    with wp.ScopedDevice(device):
        env_cells = (
            wp.array([[0, 0, 0]], dtype=wp.vec3i, device=device),
            wp.array([[0, 0, 0]], dtype=wp.vec3i, device=device),
        )
        points, point_envs, env_count = _pack_env_voxels(env_cells, device)
        geo = fem.Nanogrid.from_environment_voxels(
            points,
            point_envs,
            env_count,
            env_offsets=_make_env_offsets([[0, 0, 0], [1, 0, 0]], device),
            voxel_size=1.0,
            device=device,
        )

        fem.interpolate(
            _test_sparse_lookup_excludes_neighbor_env,
            at=fem.RegularQuadrature(fem.Cells(geo), order=1),
            values={"cell_env": geo.cell_env},
        )


def test_grid_2d_multi_env(test, device):
    N = 2
    env_count = 3

    geo = fem.Grid2D(res=wp.vec2i(N), env_count=env_count)
    base_cell_count = N**2
    base_side_count = 2 * (N + 1) * N

    test.assertEqual(geo.environment_count(), env_count)
    test.assertIn("_env", geo.name)
    test.assertEqual(geo.cell_count(), env_count * base_cell_count)
    test.assertEqual(geo.vertex_count(), env_count * (N + 1) ** 2)
    test.assertEqual(geo.side_count(), env_count * base_side_count)
    test.assertEqual(geo.boundary_side_count(), env_count * 4 * N)

    with wp.ScopedDevice(device):
        cell_quadrature = fem.RegularQuadrature(fem.Cells(geo), order=2)
        side_quadrature = fem.RegularQuadrature(fem.Sides(geo), order=2)

        fem.interpolate(_test_environment_cells, at=cell_quadrature, values={"base_cell_count": base_cell_count})
        fem.interpolate(
            _test_environment_sides,
            at=side_quadrature,
            values={"base_cell_count": base_cell_count, "base_side_count": base_side_count},
        )
        _assert_multi_env_lookup_requires_environment(test, geo, device)
        _test_pic_multi_env(test, device, geo, wp.array([[0.25, 0.25], [0.25, 0.25]], dtype=wp.vec2, device=device))
        _test_nonconforming_multi_env(test, device, geo)
        _test_nonconforming_multi_env_constructor_errors(test)
        _test_pic_partition_multi_env(test, device)
        _test_pic_env_indices_api(test, device)

        for space_kwargs in (
            {"degree": 1},
            {"degree": 2},
            {"degree": 2, "element_basis": fem.ElementBasis.SERENDIPITY},
            {"degree": 2, "family": fem.Polynomial.EQUISPACED_OPEN},
        ):
            space = fem.make_polynomial_space(geo, **space_kwargs)
            if space_kwargs["degree"] == 1:
                test.assertEqual(space.node_count(), env_count * (N + 1) ** 2)
            _assert_multi_env_node_isolation(test, geo, space, base_cell_count, device)


def test_grid_3d_multi_env(test, device):
    N = 2
    env_count = 2

    geo = fem.Grid3D(res=wp.vec3i(N), env_count=env_count)
    base_cell_count = N**3
    base_side_count = 3 * (N + 1) * N**2

    test.assertEqual(geo.environment_count(), env_count)
    test.assertIn("_env", geo.name)
    test.assertEqual(geo.cell_count(), env_count * base_cell_count)
    test.assertEqual(geo.vertex_count(), env_count * (N + 1) ** 3)
    test.assertEqual(geo.side_count(), env_count * base_side_count)
    test.assertEqual(geo.boundary_side_count(), env_count * 6 * N * N)
    test.assertEqual(geo.edge_count(), env_count * 3 * N * (N + 1) ** 2)

    with wp.ScopedDevice(device):
        cell_quadrature = fem.RegularQuadrature(fem.Cells(geo), order=2)
        side_quadrature = fem.RegularQuadrature(fem.Sides(geo), order=2)

        fem.interpolate(_test_environment_cells, at=cell_quadrature, values={"base_cell_count": base_cell_count})
        fem.interpolate(
            _test_environment_sides,
            at=side_quadrature,
            values={"base_cell_count": base_cell_count, "base_side_count": base_side_count},
        )
        _assert_multi_env_lookup_requires_environment(test, geo, device)
        _test_pic_multi_env(
            test,
            device,
            geo,
            wp.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]], dtype=wp.vec3, device=device),
        )
        _test_nonconforming_multi_env(test, device, geo)

        for space_kwargs in (
            {"degree": 1},
            {"degree": 2},
            {"degree": 2, "element_basis": fem.ElementBasis.SERENDIPITY},
            {"degree": 2, "family": fem.Polynomial.EQUISPACED_OPEN},
        ):
            space = fem.make_polynomial_space(geo, **space_kwargs)
            if space_kwargs["degree"] == 1:
                test.assertEqual(space.node_count(), env_count * (N + 1) ** 3)
            _assert_multi_env_node_isolation(test, geo, space, base_cell_count, device)


def test_deformed_grid_3d_multi_env(test, device):
    geo = fem.Grid3D(res=wp.vec3i(1), env_count=2)

    with wp.ScopedDevice(device):
        deformation_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
        deformation_field = deformation_space.make_field()
        deformed_geo = deformation_field.make_deformed_geometry(relative=True)
        deformed_geo.build_bvh(device)

        np.testing.assert_array_equal(deformed_geo.cell_bvh_groups(device).numpy(), np.array([0, 1], dtype=np.int32))

        deformed_domain = fem.Cells(deformed_geo)
        fem.interpolate(_test_deformed_environment_lookup, at=fem.RegularQuadrature(deformed_domain, order=1))

        particle_positions = wp.array(
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            dtype=wp.vec3,
            device=device,
        )
        particle_env = wp.array([0, 1], dtype=int, device=device)
        pic = fem.PicQuadrature(deformed_domain, positions=particle_positions, env_indices=particle_env)

        test.assertEqual(pic.active_cell_count(), 2)
        np.testing.assert_array_equal(pic.cell_particle_offsets.numpy(), np.array([0, 1, 2], dtype=np.int32))


def test_mesh_multi_env(test, device):
    _test_mesh_multi_env(
        test,
        device,
        _make_multi_env_trimesh(device),
        wp.array([[0.25, 0.25, 0.0], [0.25, 0.25, 0.0]], dtype=wp.vec3, device=device),
    )
    _test_mesh_multi_env(
        test,
        device,
        _make_multi_env_quadmesh(device),
        wp.array([[0.25, 0.25, 0.0], [0.25, 0.25, 0.0]], dtype=wp.vec3, device=device),
    )
    _test_mesh_multi_env(
        test,
        device,
        _make_multi_env_tetmesh(device),
        wp.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]], dtype=wp.vec3, device=device),
    )
    _test_mesh_multi_env(
        test,
        device,
        _make_multi_env_hexmesh(device),
        wp.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]], dtype=wp.vec3, device=device),
    )
    _test_mesh_multi_env_constructor_errors(test, device)
    _test_mesh_empty_environment_lookup(test, device)


def test_nanogrid_multi_env(test, device):
    geo, cell_env = _make_multi_env_nanogrid(device)

    np.testing.assert_array_equal(geo.env_offsets.numpy(), np.array([[0, 0, 0], [5, 0, 0]], dtype=np.int32))

    test.assertEqual(geo.cell_count(), 10)
    test.assertEqual(geo.vertex_count(), 39)
    test.assertEqual(geo.side_count(), 47)
    test.assertEqual(geo.boundary_side_count(), 34)
    test.assertEqual(geo.edge_count(), 74)

    _test_sparse_grid_multi_env(test, device, geo, cell_env)
    _test_sparse_grid_lookup_with_adjacent_env_offsets(test, device)
    _assert_environment_first_pressure_partition(test, geo, cell_env, device)

    bspline_space = fem.make_polynomial_space(geo, degree=2, element_basis=fem.ElementBasis.BSPLINE)
    _assert_sparse_multi_env_node_isolation(test, geo, bspline_space, cell_env, device)

    world_geo, _ = _make_multi_env_nanogrid(device, point_dtype=wp.vec3f)
    np.testing.assert_array_equal(world_geo.env_offsets.numpy(), np.array([[0, 0, 0], [5, 0, 0]], dtype=np.int32))
    test.assertEqual(world_geo.cell_count(), 10)

    masked_points = wp.array([[0, 0, 0], [9, 0, 0], [0, 0, 0]], dtype=wp.vec3i, device=device)
    masked_envs = wp.array([0, 0, 1], dtype=wp.int32, device=device)
    point_mask = wp.array([1, 0, 1], dtype=wp.int32, device=device)
    masked_geo = fem.Nanogrid.from_environment_voxels(
        masked_points,
        masked_envs,
        2,
        point_mask=point_mask,
        voxel_size=1.0,
        device=device,
    )
    np.testing.assert_array_equal(masked_geo.env_offsets.numpy(), np.array([[0, 0, 0], [4, 0, 0]], dtype=np.int32))
    test.assertEqual(masked_geo.cell_count(), 2)
    test.assertEqual(masked_geo.cell_grid.get_active_stats().voxel_count, 2)


def test_nanogrid_multi_env_rebuildable(test, device):
    env_cells = (
        wp.array([[0, 0, 0]], dtype=wp.vec3i, device=device),
        wp.array([[0, 0, 0]], dtype=wp.vec3i, device=device),
    )
    points, point_envs, env_count = _pack_env_voxels(env_cells, device)
    status = wp.zeros(1, dtype=wp.uint32, device=device)
    geo = fem.Nanogrid.from_environment_voxels(
        points,
        point_envs,
        env_count,
        voxel_size=1.0,
        device=device,
        rebuildable=True,
        max_active_voxels=4,
        max_leaf_nodes=4,
        max_lower_nodes=4,
        max_upper_nodes=4,
        status=status,
    )
    wp.synchronize_device(device)

    test.assertEqual(int(status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
    test.assertTrue(geo.cell_grid.is_rebuildable)
    test.assertEqual(geo.cell_count(), 4)
    test.assertEqual(geo.cell_grid.get_active_stats().voxel_count, 2)
    test.assertEqual(geo.environment_count(), 2)

    rebuilt_points = wp.array([[4, 0, 0], [0, 0, 0]], dtype=wp.vec3i, device=device)
    geo.rebuild(rebuilt_points, point_envs=point_envs, status=status)
    wp.synchronize_device(device)

    test.assertEqual(int(status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
    test.assertEqual(geo.cell_grid.get_active_stats().voxel_count, 2)
    np.testing.assert_array_equal(geo.env_offsets.numpy(), np.array([[-4, 0, 0], [4, 0, 0]], dtype=np.int32))

    rebuilt_voxels = wp.full((geo.cell_count(),), wp.vec3i(-999), dtype=wp.vec3i, device=device)
    geo.cell_grid.get_voxels(out=rebuilt_voxels)
    rebuilt_voxels_np = rebuilt_voxels.numpy()[:2]
    rebuilt_voxels_np = rebuilt_voxels_np[np.argsort(rebuilt_voxels_np[:, 0])]
    np.testing.assert_array_equal(rebuilt_voxels_np, np.array([[0, 0, 0], [4, 0, 0]], dtype=np.int32))


def test_adaptive_nanogrid_multi_env(test, device):
    if platform.system() == "Windows":
        test.skipTest("Skipping test due to NVRTC bug on Windows")

    geo, cell_env = _make_multi_env_adaptive_nanogrid(device)

    np.testing.assert_array_equal(geo.env_offsets.numpy(), np.array([[0, 0, 0], [4, 0, 0]], dtype=np.int32))

    test.assertEqual(geo.cell_count(), 9)
    test.assertEqual(geo.vertex_count(), 35)
    test.assertEqual(geo.side_count(), 42)
    test.assertEqual(geo.boundary_side_count(), 30)
    test.assertEqual(geo.stacked_face_count(), 42)
    test.assertEqual(geo.stacked_edge_count(), 66)

    _test_sparse_grid_multi_env(test, device, geo, cell_env)
    _assert_environment_first_pressure_partition(test, geo, cell_env, device)

    legacy_env_cells = (
        wp.array([[0, 0, 0]], dtype=wp.vec3i, device=device),
        wp.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=wp.vec3i, device=device),
    )
    legacy_env_levels = (
        wp.array([1], dtype=wp.uint8, device=device),
        wp.array([0] * legacy_env_cells[1].shape[0], dtype=wp.uint8, device=device),
    )
    legacy_geo = fem.AdaptiveNanogrid.from_environment_voxels(
        legacy_env_cells,
        legacy_env_levels,
        level_count=2,
        voxel_size=0.5,
        device=device,
    )
    np.testing.assert_array_equal(legacy_geo.env_offsets.numpy(), np.array([[0, 0, 0], [4, 0, 0]], dtype=np.int32))
    test.assertEqual(legacy_geo.cell_count(), 9)

    aligned_points = wp.array([[-1, 0, 0], [0, 0, 0]], dtype=wp.vec3i, device=device)
    aligned_levels = wp.array([2, 0], dtype=wp.uint8, device=device)
    aligned_envs = wp.array([0, 1], dtype=wp.int32, device=device)
    aligned_geo = fem.AdaptiveNanogrid.from_environment_voxels(
        aligned_points,
        aligned_levels,
        aligned_envs,
        2,
        level_count=3,
        voxel_size=0.25,
        device=device,
    )
    aligned_offsets = aligned_geo.env_offsets.numpy()
    env0_packed_max = -1 + aligned_offsets[0, 0] + (1 << 2) - 1
    env1_packed_min = aligned_offsets[1, 0]
    test.assertGreaterEqual(env1_packed_min - env0_packed_max - 1, 4)

    masked_points = wp.array([[0, 0, 0], [16, 0, 0], [0, 0, 0]], dtype=wp.vec3i, device=device)
    masked_levels = wp.array([1, 0, 0], dtype=wp.uint8, device=device)
    masked_envs = wp.array([0, 0, 1], dtype=wp.int32, device=device)
    point_mask = wp.array([1, 0, 1], dtype=wp.int32, device=device)
    masked_geo = fem.AdaptiveNanogrid.from_environment_voxels(
        masked_points,
        masked_levels,
        masked_envs,
        2,
        level_count=2,
        point_mask=point_mask,
        voxel_size=0.5,
        device=device,
    )
    np.testing.assert_array_equal(masked_geo.env_offsets.numpy(), np.array([[0, 0, 0], [4, 0, 0]], dtype=np.int32))
    test.assertEqual(masked_geo.cell_count(), 2)


def test_environment_space_partition_capture(test, device):
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(2, 1), env_count=3)
        cell_mask = wp.array([1, 0, 0, 1, 0, 0], dtype=int, device=device)
        geo_partition = fem.ExplicitGeometryPartition(
            geo,
            cell_mask,
            max_cell_count=4,
            max_side_count=0,
        )
        space = fem.make_polynomial_space(geo, degree=0, discontinuous=True)

        def make_partition():
            return fem.make_space_partition(
                space_topology=space.topology,
                geometry_partition=geo_partition,
                environment_first=True,
                with_halo=False,
                max_node_count=5,
                device=device,
            )

        make_partition()
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            captured_partition = make_partition()
        wp.capture_launch(capture.graph)

        test.assertEqual(captured_partition.node_count(), 5)
        np.testing.assert_array_equal(
            captured_partition.space_node_indices().numpy()[:2], np.array([0, 3], dtype=np.int32)
        )
        np.testing.assert_array_equal(captured_partition.env_offsets.numpy(), np.array([0, 1, 2, 2], dtype=np.int32))
        expected_space_to_partition = np.full(space.node_count(), fem.NULL_NODE_INDEX, dtype=np.int32)
        expected_space_to_partition[[0, 3]] = [0, 1]
        np.testing.assert_array_equal(captured_partition._space_to_partition.numpy(), expected_space_to_partition)


def test_single_environment_space_partition_environmentless_tail(test, device):
    with wp.ScopedDevice(device):
        positions = wp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=wp.vec3,
            device=device,
        )
        geo = fem.Trimesh3D(
            wp.array([[0, 1, 2]], dtype=int, device=device),
            positions,
            build_bvh=True,
        )
        space = fem.make_polynomial_space(geo, degree=1)
        partition = fem.make_space_partition(
            space_topology=space.topology,
            environment_first=True,
            max_node_count=space.node_count(),
            device=device,
        )

        test.assertEqual(partition.node_count(), 4)
        np.testing.assert_array_equal(partition.space_node_indices().numpy(), np.array([0, 1, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(partition.env_offsets.numpy(), np.array([0, 3], dtype=np.int32))


# -- Device setup and test registration --

devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()
cuda_devices_with_mempool = get_selected_cuda_test_devices_with_mempool()


class TestFemMultiEnv(unittest.TestCase):
    pass


add_function_test(TestFemMultiEnv, "test_grid_2d_multi_env", test_grid_2d_multi_env, devices=devices)
add_function_test(TestFemMultiEnv, "test_grid_3d_multi_env", test_grid_3d_multi_env, devices=devices)
add_function_test(TestFemMultiEnv, "test_deformed_grid_3d_multi_env", test_deformed_grid_3d_multi_env, devices=devices)
add_function_test(TestFemMultiEnv, "test_mesh_multi_env", test_mesh_multi_env, devices=devices)
add_function_test(TestFemMultiEnv, "test_nanogrid_multi_env", test_nanogrid_multi_env, devices=cuda_devices)
add_function_test(
    TestFemMultiEnv, "test_nanogrid_multi_env_rebuildable", test_nanogrid_multi_env_rebuildable, devices=cuda_devices
)
add_function_test(
    TestFemMultiEnv, "test_adaptive_nanogrid_multi_env", test_adaptive_nanogrid_multi_env, devices=cuda_devices
)
add_function_test(
    TestFemMultiEnv,
    "test_environment_space_partition_capture",
    test_environment_space_partition_capture,
    devices=cuda_devices_with_mempool,
)
add_function_test(
    TestFemMultiEnv,
    "test_single_environment_space_partition_environmentless_tail",
    test_single_environment_space_partition_environmentless_tail,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
