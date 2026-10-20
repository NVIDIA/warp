# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU benchmark for inlining struct-taking edge-intersection helpers."""

import numpy as np

import warp as wp

from .benchmarks_utils import setup_once

NUM_CELLS = 1 << 18


@wp.struct
class MeshData:
    offsets: wp.array[wp.int32]
    connectivity: wp.array[wp.int32]
    scalar: wp.array[wp.float32]
    active: wp.array[wp.int32]
    node_map: wp.array[wp.int32]
    positions: wp.array[wp.vec3]


@wp.struct
class EdgeIntersection:
    key: wp.int64
    position: wp.vec3


@wp.func
def intersect_edge(mesh: MeshData, cell: wp.int32, edge: wp.vec2i, iso: wp.float32) -> EdgeIntersection:
    # Interpolation keeps the helper above LLVM's default inlining budget on both x86-64 ABIs.
    first = mesh.connectivity[mesh.offsets[cell] + edge[0]]
    second = mesh.connectivity[mesh.offsets[cell] + edge[1]]
    first = mesh.node_map[first]
    second = mesh.node_map[second]
    if second < first:
        temp = first
        first = second
        second = temp
    first_value = mesh.scalar[first]
    second_value = mesh.scalar[second]
    weight = float(0.0)
    if first_value == iso:
        second = first
    elif second_value == iso:
        first = second
    else:
        weight = wp.clamp((iso - first_value) / (second_value - first_value), 0.0, 1.0)
    result = EdgeIntersection()
    result.key = (wp.int64(first) << wp.int64(32)) + wp.int64(second)
    result.position = wp.lerp(mesh.positions[first], mesh.positions[second], weight)
    return result


@wp.kernel(enable_backward=False)
def classify_edges(mesh: MeshData, output: wp.array[wp.float32]):
    cell = wp.tid()
    if mesh.active[cell] != 0:
        edge0 = intersect_edge(mesh, cell, wp.vec2i(0, 1), 0.5)
        edge1 = intersect_edge(mesh, cell, wp.vec2i(1, 2), 0.5)
        edge2 = intersect_edge(mesh, cell, wp.vec2i(2, 3), 0.5)
        if edge0.key != edge1.key and edge1.key != edge2.key and edge2.key != edge0.key:
            # Consume the positions as well as the keys so interpolation remains part of the workload.
            normal = wp.cross(edge1.position - edge0.position, edge2.position - edge0.position)
            output[cell] = wp.length_sq(normal)


class StructHelperInliningCPU:
    """Measure CPU edge classification and geometry through struct-taking helpers."""

    repeat = 10
    number = 1

    @setup_once
    def setup(self):
        wp.init()
        nodes = np.arange(NUM_CELLS * 4, dtype=np.int32)
        scalar = (nodes % 7).astype(np.float32) / 6.0
        positions = np.tile(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=np.float32), (NUM_CELLS, 1))

        self.mesh = MeshData()
        self.mesh.offsets = wp.array(np.arange(NUM_CELLS, dtype=np.int32) * 4, device="cpu")
        self.mesh.connectivity = wp.array(nodes, device="cpu")
        self.mesh.scalar = wp.array(scalar, device="cpu")
        self.mesh.active = wp.array(np.ones(NUM_CELLS, dtype=np.int32), device="cpu")
        self.mesh.node_map = wp.array(nodes, device="cpu")
        self.mesh.positions = wp.array(positions, dtype=wp.vec3, device="cpu")
        self.output = wp.zeros(NUM_CELLS, dtype=wp.float32, device="cpu")

        self.command = wp.launch(
            classify_edges,
            dim=NUM_CELLS,
            inputs=[self.mesh],
            outputs=[self.output],
            device="cpu",
            record_cmd=True,
        )
        for _ in range(3):
            self.command.launch()

        first = nodes.reshape(-1, 4)[:, :3]
        second = nodes.reshape(-1, 4)[:, 1:]
        first_value = scalar[first]
        second_value = scalar[second]
        weight = np.clip((0.5 - first_value) / (second_value - first_value), 0.0, 1.0)
        intersections = positions[first] + weight[..., None] * (positions[second] - positions[first])
        keys = (np.where(second_value == 0.5, second, first).astype(np.int64) << 32) + np.where(
            first_value == 0.5, first, second
        )
        distinct = (keys[:, 0] != keys[:, 1]) & (keys[:, 1] != keys[:, 2]) & (keys[:, 2] != keys[:, 0])
        normal = np.cross(intersections[:, 1] - intersections[:, 0], intersections[:, 2] - intersections[:, 0])
        expected = np.where(distinct, np.sum(normal * normal, axis=1), 0.0)
        np.testing.assert_allclose(self.output.numpy(), expected, rtol=1e-6, atol=1e-6)

    def time_classify_edges(self):
        self.command.launch()
