# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Graph coloring utilities for parallel constraint solving."""

from __future__ import annotations

import enum

import numpy as np

import warp as wp
from warp._src.types import type_repr


class GraphColoringAlgorithm(enum.IntEnum):
    """Graph coloring algorithm selection."""

    MCS = 0
    """Maximum Cardinality Search based coloring algorithm."""

    GREEDY = 1
    """Degree-ordered greedy coloring algorithm."""


def graph_coloring_assign(
    edges: wp.array,
    node_colors: wp.array,
    algorithm: GraphColoringAlgorithm = GraphColoringAlgorithm.MCS,
) -> int:
    """Assign colors to graph nodes such that no two adjacent nodes share the same color.

    This function assigns colors to nodes of an undirected graph represented by its
    edge list. The coloring ensures that no two nodes connected by an edge have the
    same color, which is useful for parallel constraint solving where nodes of the
    same color can be processed simultaneously.

    Args:
        edges: A 2D array of shape ``(edge_count, 2)`` containing edge pairs, where each
            row ``[i, j]`` represents an undirected edge between nodes ``i`` and ``j``.
            Must be a CPU array with ``int32`` dtype.
        node_colors: A 1D array of shape ``(node_count,)`` that will be filled with the
            computed color assignments. Must be a CPU array with ``int32`` dtype.
            The array size determines the number of nodes in the graph.
        algorithm: The coloring algorithm to use.

    Returns:
        The number of colors used in the coloring.

    See Also:
        :func:`graph_coloring_balance`.
    """
    from warp._src.context import runtime  # noqa: PLC0415

    # Validate device
    if not edges.device.is_cpu:
        raise RuntimeError("edges array must be on the CPU")
    if not node_colors.device.is_cpu:
        raise RuntimeError("node_colors array must be on the CPU")

    # Validate dtype
    if edges.dtype != wp.int32:
        raise RuntimeError(f"edges array must have dtype int32, got {type_repr(edges.dtype)}")
    if node_colors.dtype != wp.int32:
        raise RuntimeError(f"node_colors array must have dtype int32, got {type_repr(node_colors.dtype)}")

    # Validate shape
    if edges.ndim != 2:
        raise RuntimeError(f"edges array must be 2-dimensional, got {edges.ndim} dimensions")
    if edges.shape[1] != 2:
        raise RuntimeError(f"edges array must have shape (edge_count, 2), got shape {edges.shape}")
    if node_colors.ndim != 1:
        raise RuntimeError(f"node_colors array must be 1-dimensional, got {node_colors.ndim} dimensions")

    node_count = node_colors.shape[0]

    if node_count == 0:
        raise RuntimeError("Cannot color an empty graph")

    color_count = runtime.core.wp_graph_coloring(
        node_count,
        edges.__ctype__(),
        int(algorithm),
        node_colors.__ctype__(),
    )

    if color_count < 0:
        raise RuntimeError("Graph coloring failed")

    return color_count


def graph_coloring_balance(
    edges: wp.array,
    node_colors: wp.array,
    color_count: int,
    target_max_min_ratio: float,
) -> float:
    """Balance the sizes of color groups in a graph coloring.

    This function adjusts the color assignments to make the color groups more
    balanced in size, which improves load balancing when processing nodes in
    parallel by color. It attempts to move nodes between color groups while
    maintaining the validity of the coloring (no adjacent nodes share a color).

    Args:
        edges: A 2D array of shape ``(edge_count, 2)`` containing edge pairs, where each
            row ``[i, j]`` represents an undirected edge between nodes ``i`` and ``j``.
            Must be a CPU array with ``int32`` dtype.
        node_colors: A 1D array of shape ``(node_count,)`` containing the current color
            assignments. Will be modified in-place with the balanced coloring.
            Must be a CPU array with ``int32`` dtype.
        color_count: The number of colors in the current coloring (as returned by
            :func:`graph_coloring_assign`).
        target_max_min_ratio: The target ratio between the largest and smallest color
            group sizes. The algorithm will stop when this ratio is achieved or when
            no further improvements are possible.

    Returns:
        The actual max/min ratio achieved after balancing. This may be higher than
        the target if the graph structure prevents further balancing.

    Example:

        .. code-block:: python

            import warp as wp

            edges = wp.array([[0, 1], [1, 2], [2, 3]], dtype=wp.int32, device="cpu")
            colors = wp.empty(4, dtype=wp.int32, device="cpu")

            color_count = wp.utils.graph_coloring_assign(edges, colors, wp.utils.GraphColoringAlgorithm.MCS)
            ratio = wp.utils.graph_coloring_balance(edges, colors, color_count, 1.1)
    """
    from warp._src.context import runtime  # noqa: PLC0415

    if not edges.device.is_cpu:
        raise RuntimeError("edges array must be on the CPU")
    if not node_colors.device.is_cpu:
        raise RuntimeError("node_colors array must be on the CPU")

    if edges.dtype != wp.int32:
        raise RuntimeError(f"edges array must have dtype int32, got {type_repr(edges.dtype)}")
    if node_colors.dtype != wp.int32:
        raise RuntimeError(f"node_colors array must have dtype int32, got {type_repr(node_colors.dtype)}")

    if edges.ndim != 2:
        raise RuntimeError(f"edges array must be 2-dimensional, got {edges.ndim} dimensions")
    if edges.shape[1] != 2:
        raise RuntimeError(f"edges array must have shape (edge_count, 2), got shape {edges.shape}")
    if node_colors.ndim != 1:
        raise RuntimeError(f"node_colors array must be 1-dimensional, got {node_colors.ndim} dimensions")

    node_count = node_colors.shape[0]

    return runtime.core.wp_balance_coloring(
        node_count,
        edges.__ctype__(),
        color_count,
        target_max_min_ratio,
        node_colors.__ctype__(),
    )


@wp.kernel
def count_color_group_sizes(
    node_colors: wp.array(dtype=int),
    group_sizes: wp.array(dtype=int),
):
    """Count the size of each color group.

    Note:
        This kernel is not parallel-safe due to race conditions on ``group_sizes``.
        It must be launched with ``dim=(1,)`` to run single-threaded.
    """
    for node_idx in range(node_colors.shape[0]):
        node_color = node_colors[node_idx]
        group_sizes[node_color] = group_sizes[node_color] + 1


@wp.kernel
def fill_color_groups(
    node_colors: wp.array(dtype=int),
    group_offsets: wp.array(dtype=int),
    group_fill_count: wp.array(dtype=int),
    color_groups_flatten: wp.array(dtype=int),
):
    """Fill the flattened color groups array with node indices.

    Note:
        This kernel is not parallel-safe due to race conditions on ``group_fill_count`` and
        ``color_groups_flatten``. It must be launched with ``dim=(1,)`` to run single-threaded.
    """
    for node_idx in range(node_colors.shape[0]):
        node_color = node_colors[node_idx]
        group_offset = group_offsets[node_color]
        group_idx = group_fill_count[node_color]
        color_groups_flatten[group_idx + group_offset] = wp.int32(node_idx)

        group_fill_count[node_color] = group_idx + 1


def graph_coloring_get_groups(
    node_colors: wp.array,
    color_count: int,
    return_wp_array: bool = True,
    device: wp.DeviceLike = "cpu",
) -> tuple[np.ndarray, ...] | tuple[wp.array, ...]:
    """Convert node colors into per-color groups.

    This converts a ``node_colors`` array to a tuple of arrays, where each array contains
    the IDs of nodes with a certain color. The output can be used to process nodes
    in color order after :func:`graph_coloring_assign` and :func:`graph_coloring_balance`.

    Args:
        node_colors: A 1D array of shape ``(node_count,)`` containing color assignments.
            Must be a CPU array with ``int32`` dtype.
        color_count: The number of colors in the coloring.
        return_wp_array: Whether to return Warp arrays instead of numpy arrays.
        device: Warp device for returned arrays when ``return_wp_array`` is ``True``.

    Returns:
        A tuple of arrays. Each array contains the node IDs assigned to a color.
        Numpy arrays are returned by default, or Warp arrays when ``return_wp_array`` is ``True``.

    See Also:
        :func:`graph_coloring_assign`, :func:`graph_coloring_balance`.
    """
    if color_count < 0:
        raise RuntimeError("color_count must be non-negative")
    if color_count == 0:
        return ()

    if not node_colors.device.is_cpu:
        raise RuntimeError("node_colors array must be on the CPU")
    if node_colors.dtype != wp.int32:
        raise RuntimeError(f"node_colors array must have dtype int32, got {type_repr(node_colors.dtype)}")
    if node_colors.ndim != 1:
        raise RuntimeError(f"node_colors array must be 1-dimensional, got {node_colors.ndim} dimensions")

    group_sizes = wp.zeros(shape=(color_count,), dtype=wp.int32, device="cpu")
    wp.launch(count_color_group_sizes, dim=(1,), inputs=[node_colors, group_sizes], device="cpu")

    group_sizes_np = group_sizes.numpy()
    group_offsets_np = np.concatenate(
        (
            np.array([0], dtype=group_sizes_np.dtype),
            np.cumsum(group_sizes_np, dtype=group_sizes_np.dtype),
        )
    )
    group_offsets = wp.array(group_offsets_np, dtype=wp.int32, device="cpu")

    group_fill_count = wp.zeros(shape=(color_count,), dtype=wp.int32, device="cpu")
    color_groups_flatten = wp.empty(shape=(int(group_sizes_np.sum()),), dtype=wp.int32, device="cpu")
    wp.launch(
        fill_color_groups,
        dim=(1,),
        inputs=(node_colors, group_offsets),
        outputs=(group_fill_count, color_groups_flatten),
        device="cpu",
    )

    color_groups_flatten_np = color_groups_flatten.numpy()

    if return_wp_array:
        return tuple(
            wp.array(
                color_groups_flatten_np[group_offsets_np[color_idx] : group_offsets_np[color_idx + 1]],
                dtype=wp.int32,
                device=device,
            )
            for color_idx in range(color_count)
        )

    return tuple(
        color_groups_flatten_np[group_offsets_np[color_idx] : group_offsets_np[color_idx + 1]]
        for color_idx in range(color_count)
    )
