# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

import numpy as np

import warp as wp
import warp.utils


class ColoringAlgorithm(Enum):
    MCS = 0
    GREEDY = 1


@wp.kernel
def construct_trimesh_graph_edges_kernel(
    trimesh_edge_indices: wp.array(dtype=int, ndim=2),
    add_bending: bool,
    graph_edge_indices: wp.array(dtype=int, ndim=2),
    graph_num_edges: wp.array(dtype=int),
):
    num_diagonal_edges = wp.int32(0)
    num_non_diagonal_edges = trimesh_edge_indices.shape[0]
    for e_idx in range(trimesh_edge_indices.shape[0]):
        v1 = trimesh_edge_indices[e_idx, 2]
        v2 = trimesh_edge_indices[e_idx, 3]

        graph_edge_indices[e_idx, 0] = v1
        graph_edge_indices[e_idx, 1] = v2

        o1 = trimesh_edge_indices[e_idx, 0]
        o2 = trimesh_edge_indices[e_idx, 1]

        if o1 != -1 and o2 != -1 and add_bending:
            graph_edge_indices[num_non_diagonal_edges + num_diagonal_edges, 0] = o1
            graph_edge_indices[num_non_diagonal_edges + num_diagonal_edges, 1] = o2

            num_diagonal_edges = num_diagonal_edges + 1

    graph_num_edges[0] = num_diagonal_edges + num_non_diagonal_edges


@wp.kernel
def validate_graph_coloring(edge_indices: wp.array(dtype=int, ndim=2), colors: wp.array(dtype=int)):
    edge_idx = wp.tid()
    e_v_1 = edge_indices[edge_idx, 0]
    e_v_2 = edge_indices[edge_idx, 1]

    wp.expect_neq(colors[e_v_1], colors[e_v_2])


@wp.kernel
def count_color_group_size(
    colors: wp.array(dtype=int),
    group_sizes: wp.array(dtype=int),
):
    for particle_idx in range(colors.shape[0]):
        particle_color = colors[particle_idx]
        group_sizes[particle_color] = group_sizes[particle_color] + 1


@wp.kernel
def fill_color_groups(
    colors: wp.array(dtype=int),
    group_fill_count: wp.array(dtype=int),
    group_offsets: wp.array(dtype=int),
    # flattened color groups
    color_groups_flatten: wp.array(dtype=int),
):
    for particle_idx in range(colors.shape[0]):
        particle_color = colors[particle_idx]
        group_offset = group_offsets[particle_color]
        group_idx = group_fill_count[particle_color]
        color_groups_flatten[group_idx + group_offset] = wp.int32(particle_idx)

        group_fill_count[particle_color] = group_idx + 1


def convert_to_color_groups(num_colors, particle_colors, return_wp_array=False, device="cpu"):
    group_sizes = wp.zeros(shape=(num_colors,), dtype=int, device="cpu")
    wp.launch(kernel=count_color_group_size, inputs=[particle_colors, group_sizes], device="cpu", dim=1)

    group_sizes_np = group_sizes.numpy()
    group_offsets_np = np.concatenate([np.array([0]), np.cumsum(group_sizes_np)])
    group_offsets = wp.array(group_offsets_np, dtype=int, device="cpu")

    group_fill_count = wp.zeros(shape=(num_colors,), dtype=int, device="cpu")
    color_groups_flatten = wp.empty(shape=(group_sizes_np.sum(),), dtype=int, device="cpu")
    wp.launch(
        kernel=fill_color_groups,
        inputs=[particle_colors, group_fill_count, group_offsets, color_groups_flatten],
        device="cpu",
        dim=1,
    )

    color_groups_flatten_np = color_groups_flatten.numpy()

    color_groups = []
    if return_wp_array:
        for color_idx in range(num_colors):
            color_groups.append(
                wp.array(
                    color_groups_flatten_np[group_offsets_np[color_idx] : group_offsets_np[color_idx + 1]],
                    dtype=int,
                    device=device,
                )
            )
    else:
        for color_idx in range(num_colors):
            color_groups.append(color_groups_flatten_np[group_offsets_np[color_idx] : group_offsets_np[color_idx + 1]])

    return color_groups


def construct_trimesh_graph_edges(trimesh_edge_indices, return_wp_array=False):
    if isinstance(trimesh_edge_indices, np.ndarray):
        trimesh_edge_indices = wp.array(trimesh_edge_indices, dtype=int, device="cpu")

    # preallocate maximum amount of memory, which is model.edge_count * 2
    graph_edge_indices = wp.empty(shape=(trimesh_edge_indices.shape[0] * 2, 2), dtype=int, device="cpu")
    graph_num_edges = wp.zeros(shape=(1,), dtype=int, device="cpu")

    wp.launch(
        kernel=construct_trimesh_graph_edges_kernel,
        inputs=[
            trimesh_edge_indices.to("cpu"),
            True,
        ],
        outputs=[graph_edge_indices, graph_num_edges],
        dim=1,
        device="cpu",
    )

    num_edges = graph_num_edges.numpy()[0]
    graph_edge_indices_true_size = graph_edge_indices.numpy()[:num_edges, :]

    if return_wp_array:
        graph_edge_indices_true_size = wp.array(graph_edge_indices_true_size, dtype=int, device="cpu")

    return graph_edge_indices_true_size


def color_trimesh(
    num_nodes,
    trimesh_edge_indices,
    include_bending_energy,
    balance_colors=True,
    target_max_min_color_ratio=1.1,
    algorithm: ColoringAlgorithm = ColoringAlgorithm.MCS,
):
    """
    A function that generates vertex coloring for a trimesh, which is represented by the number of vertices and edges of the mesh.
    It will convert the trimesh to a graph and then apply coloring.
    It returns a list of `np.array` with `dtype`=`int`. The length of the list is the number of colors
    and each `np.array` contains the indices of vertices with this color.

    Args:
        num_nodes: The number of the nodes in the graph
        trimesh_edge_indices: A `wp.array` with of shape (number_edges, 4), each row is (o1, o2, v1, v2), see `sim.Model`'s definition of `edge_indices`.
        include_bending_energy: whether to consider bending energy in the coloring process. If set to `True`, the generated
            graph will contain all the edges connecting o1 and o2; otherwise, the graph will be equivalent to the trimesh.
        balance_colors: the parameter passed to `color_graph`, see `color_graph`'s document
        target_max_min_color_ratio: the parameter passed to `color_graph`, see `color_graph`'s document
        algorithm: the parameter passed to `color_graph`, see `color_graph`'s document

    """
    if num_nodes == 0:
        return []

    if trimesh_edge_indices.shape[0] == 0:
        # no edge, all the particle can have same color
        return [np.arange(0, num_nodes, dtype=int)]

    if include_bending_energy:
        graph_edge_indices = construct_trimesh_graph_edges(trimesh_edge_indices, return_wp_array=True)
    else:
        graph_edge_indices = wp.array(trimesh_edge_indices[:, 2:], dtype=int, device="cpu")

    color_groups = color_graph(num_nodes, graph_edge_indices, balance_colors, target_max_min_color_ratio, algorithm)
    return color_groups


def color_graph(
    num_nodes,
    graph_edge_indices,
    balance_colors=True,
    target_max_min_color_ratio=1.1,
    algorithm: ColoringAlgorithm = ColoringAlgorithm.MCS,
):
    """
    A function that generates coloring for a graph, which is represented by the number of nodes and an array of edges.
    It returns a list of `np.array` with `dtype`=`int`. The length of the list is the number of colors
    and each `np.array` contains the indices of vertices with this color.

    Args:
        num_nodes: The number of the nodes in the graph
        graph_edge_indices: A `wp.array` with of shape (number_edges, 2)
        balance_colors: Whether to apply the color balancing algorithm to balance the size of each color
        target_max_min_color_ratio: the color balancing algorithm will stop when the ratio between the largest color and
            the smallest color reaches this value
        algorithm: Value should an enum type of ColoringAlgorithm, otherwise it will raise an error. ColoringAlgorithm.mcs means using the MCS coloring algorithm,
            while ColoringAlgorithm.ordered_greedy means using the degree-ordered greedy algorithm. The MCS algorithm typically generates 30% to 50% fewer colors
            compared to the ordered greedy algorithm, while maintaining the same linear complexity. Although MCS has a constant overhead that makes it about twice
            as slow as the greedy algorithm, it produces significantly better coloring results. We recommend using MCS, especially if coloring is only part of the
            preprocessing stage.e.

    Note:

        References to the coloring algorithm:
        MCS: Pereira, F. M. Q., & Palsberg, J. (2005, November). Register allocation via coloring of chordal graphs. In Asian Symposium on Programming Languages and Systems (pp. 315-329). Berlin, Heidelberg: Springer Berlin Heidelberg.
        Ordered Greedy: Ton-That, Q. M., Kry, P. G., & Andrews, S. (2023). Parallel block Neo-Hookean XPBD using graph clustering. Computers & Graphics, 110, 1-10.
    """
    if num_nodes == 0:
        return []

    particle_colors = wp.empty(shape=(num_nodes), dtype=wp.int32, device="cpu")

    if graph_edge_indices.ndim != 2:
        raise ValueError(
            f"graph_edge_indices must be a 2 dimensional array! The provided one is {graph_edge_indices.ndim} dimensional."
        )

    num_colors = wp.context.runtime.core.wp_graph_coloring(
        num_nodes,
        graph_edge_indices.__ctype__(),
        algorithm.value,
        particle_colors.__ctype__(),
    )

    if balance_colors:
        max_min_ratio = wp.context.runtime.core.wp_balance_coloring(
            num_nodes,
            graph_edge_indices.__ctype__(),
            num_colors,
            target_max_min_color_ratio,
            particle_colors.__ctype__(),
        )

        if max_min_ratio > target_max_min_color_ratio:
            wp.utils.warn(
                f"The graph is not optimizable anymore, terminated with a max/min ratio: {max_min_ratio} without reaching the target ratio: {target_max_min_color_ratio}"
            )

    color_groups = convert_to_color_groups(num_colors, particle_colors, return_wp_array=False)

    return color_groups


def combine_independent_particle_coloring(color_groups_1, color_groups_2):
    """
    A function that combines 2 independent coloring groups. Note that color_groups_1 and color_groups_2 must be from 2 independent
    graphs so that there is no connection between them. This algorithm will sort color_groups_1 in ascending order and
    sort color_groups_2 in descending order, and combine each group with the same index, this way we are always combining
    the smaller group with the larger group.

    Args:
        color_groups_1: A list of `np.array` with `dtype`=`int`. The length of the list is the number of colors
            and each `np.array` contains the indices of vertices with this color.
        color_groups_2: A list of `np.array` with `dtype`=`int`. The length of the list is the number of colors
            and each `np.array` contains the indices of vertices with this color.

    """
    if len(color_groups_1) == 0:
        return color_groups_2
    if len(color_groups_2) == 0:
        return color_groups_1

    num_colors_after_combining = max(len(color_groups_1), len(color_groups_2))
    color_groups_combined = []

    # this made sure that the leftover groups are always the largest
    if len(color_groups_1) < len(color_groups_2):
        color_groups_1, color_groups_2 = color_groups_2, color_groups_1

    # sort group 1 in ascending order
    color_groups_1_sorted = sorted(color_groups_1, key=lambda group: len(group))
    # sort group 1 in descending order
    color_groups_2_sorted = sorted(color_groups_2, key=lambda group: -len(group))
    # so that we are combining the smaller group with the larger group
    # which will balance the load of each group

    for i in range(num_colors_after_combining):
        group_1 = color_groups_1_sorted[i] if i < len(color_groups_1) else None
        group_2 = color_groups_2_sorted[i] if i < len(color_groups_2) else None

        if group_1 is not None and group_2 is not None:
            color_groups_combined.append(np.concatenate([group_1, group_2]))
        elif group_1 is not None:
            color_groups_combined.append(group_1)
        else:
            color_groups_combined.append(group_2)

    return color_groups_combined
