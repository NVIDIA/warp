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

import enum
import itertools

import numpy as np

import warp as wp
import warp.examples
from warp.tests.unittest_utils import *

# ============================================================================
# Graph Coloring Algorithm Selection
# ============================================================================


class ColoringAlgorithm(enum.IntEnum):
    """Graph coloring algorithm selection."""

    MCS = 0
    """Maximum Cardinality Search based coloring algorithm"""

    GREEDY = 1
    """Degree-ordered greedy coloring algorithm"""


# ============================================================================
# Graph Coloring Utilities
# ============================================================================


def validate_graph_coloring(edge_indices_np, colors_np):
    """Validate that graph coloring is correct - no adjacent nodes have the same color.

    Args:
        edge_indices_np: numpy array of shape (num_edges, 2) with edge pairs
        colors_np: numpy array of node colors

    Returns:
        Number of invalid edges (0 means valid coloring)
    """
    v1 = edge_indices_np[:, 0]
    v2 = edge_indices_np[:, 1]
    invalid_edges = np.sum(colors_np[v1] == colors_np[v2])
    return invalid_edges


@wp.kernel
def count_color_group_size(
    colors: wp.array(dtype=int),
    group_sizes: wp.array(dtype=int),
):
    """Count the size of each color group.

    Note: This kernel is NOT parallel-safe due to race conditions on group_sizes.
    Must be launched with dim=(1,) to run single-threaded.
    """
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
    """Fill the flattened color groups array with particle indices.

    Note: This kernel is NOT parallel-safe due to race conditions on group_fill_count
    and color_groups_flatten. Must be launched with dim=(1,) to run single-threaded.
    """
    for particle_idx in range(colors.shape[0]):
        particle_color = colors[particle_idx]
        group_offset = group_offsets[particle_color]
        group_idx = group_fill_count[particle_color]
        color_groups_flatten[group_idx + group_offset] = wp.int32(particle_idx)

        group_fill_count[particle_color] = group_idx + 1


def convert_to_color_groups(num_colors, particle_colors, return_wp_array=False, device="cpu"):
    """Convert a color array to a list of color groups."""
    group_sizes = wp.zeros((num_colors,), dtype=int, device="cpu")
    wp.launch(count_color_group_size, dim=(1,), inputs=[particle_colors, group_sizes], device="cpu")

    group_sizes_np = group_sizes.numpy()
    group_offsets_np = np.concatenate([np.array([0]), np.cumsum(group_sizes_np)])
    group_offsets = wp.array(group_offsets_np, dtype=int, device="cpu")

    group_fill_count = wp.zeros((num_colors,), dtype=int, device="cpu")
    color_groups_flatten = wp.empty((group_sizes_np.sum(),), dtype=int, device="cpu")
    wp.launch(
        fill_color_groups,
        dim=(1,),
        inputs=[particle_colors, group_fill_count, group_offsets, color_groups_flatten],
        device="cpu",
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


@wp.kernel
def construct_trimesh_graph_edges_kernel(
    trimesh_edge_indices: wp.array(dtype=int, ndim=2),
    add_bending: bool,
    graph_edge_indices: wp.array(dtype=int, ndim=2),
    graph_num_edges: wp.array(dtype=int),
):
    """Build graph edges from trimesh edges, optionally including bending edges."""
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


def construct_trimesh_graph_edges(trimesh_edge_indices, return_wp_array=False):
    """Construct graph edges from trimesh edges with bending."""
    if isinstance(trimesh_edge_indices, np.ndarray):
        trimesh_edge_indices = wp.array(trimesh_edge_indices, dtype=int, device="cpu")

    # preallocate maximum amount of memory, which is model.edge_count * 2
    graph_edge_indices = wp.empty((trimesh_edge_indices.shape[0] * 2, 2), dtype=int, device="cpu")
    graph_num_edges = wp.zeros((1,), dtype=int, device="cpu")

    wp.launch(
        construct_trimesh_graph_edges_kernel,
        dim=(1,),
        inputs=[trimesh_edge_indices.to("cpu"), True],
        outputs=[graph_edge_indices, graph_num_edges],
        device="cpu",
    )

    num_edges = graph_num_edges.numpy()[0]
    graph_edge_indices_true_size = graph_edge_indices.numpy()[:num_edges, :]

    if return_wp_array:
        graph_edge_indices_true_size = wp.array(graph_edge_indices_true_size, dtype=int, device="cpu")

    return graph_edge_indices_true_size


# ============================================================================
# Mesh-to-Edge Helper Functions
# ============================================================================


def build_trimesh_edges_from_faces(faces):
    """Build trimesh edge list from triangle faces.

    Args:
        faces: Flat array of face vertex indices [v0, v1, v2, v0, v1, v2, ...]

    Returns:
        np.array of shape (num_edges, 4) with format [o1, o2, v1, v2] where:
        - v1, v2: the two vertices forming the edge
        - o1, o2: opposite vertices in adjacent triangles (-1 if boundary edge)
    """
    faces = np.array(faces, dtype=int)
    num_tris = len(faces) // 3

    # Dictionary to track edges: (v1, v2) -> [tri_idx, opposite_vertex]
    edge_dict = {}

    for tri_idx in range(num_tris):
        base = tri_idx * 3
        v0, v1, v2 = faces[base], faces[base + 1], faces[base + 2]

        # Three edges of the triangle
        edges = [
            (min(v0, v1), max(v0, v1), v2),  # edge (v0,v1), opposite v2
            (min(v1, v2), max(v1, v2), v0),  # edge (v1,v2), opposite v0
            (min(v2, v0), max(v2, v0), v1),  # edge (v2,v0), opposite v1
        ]

        for v_min, v_max, opposite in edges:
            edge_key = (v_min, v_max)
            if edge_key not in edge_dict:
                edge_dict[edge_key] = []
            edge_dict[edge_key].append((tri_idx, opposite))

    # Build edge list in format [o1, o2, v1, v2]
    edge_list = []
    for (v1, v2), tri_info in edge_dict.items():
        o1 = tri_info[0][1]  # opposite vertex in first triangle
        o2 = tri_info[1][1] if len(tri_info) > 1 else -1  # opposite in second triangle, or -1 if boundary
        edge_list.append([o1, o2, v1, v2])

    return np.array(edge_list, dtype=int)


# ============================================================================
# Test Helper Functions
# ============================================================================


def create_lattice_grid(N):
    """Create a lattice grid mesh for testing."""
    size = 10
    position = (0, 0)

    X = np.linspace(-0.5 * size + position[0], 0.5 * size + position[0], N)
    Y = np.linspace(-0.5 * size + position[1], 0.5 * size + position[1], N)

    X, Y = np.meshgrid(X, Y)

    Z = []
    for _i in range(N):
        Z.append(np.linspace(0, size, N))

    Z = np.array(Z)

    vs = []
    for i, j in itertools.product(range(N), range(N)):
        vs.append(wp.vec3((X[i, j], Y[i, j], Z[i, j])))

    fs = []
    for i, j in itertools.product(range(0, N - 1), range(0, N - 1)):
        vId = j + i * N

        if (j + i) % 2:
            fs.extend([vId, vId + N + 1, vId + 1])
            fs.extend([vId, vId + N, vId + N + 1])
        else:
            fs.extend([vId, vId + N, vId + 1])
            fs.extend([vId + N, vId + N + 1, vId + 1])

    return vs, fs


class TestColoring(unittest.TestCase):
    def test_coloring_corner_case(self):
        """Test corner cases: empty graph and simple 2-node graph."""
        # Test 1: Simple 2-node graph with one edge connecting the nodes
        num_nodes = 2

        # Single edge connecting nodes 0 and 1
        edge_indices = wp.array([[0, 1]], dtype=int, device="cpu")
        particle_colors = wp.empty((num_nodes,), dtype=wp.int32, device="cpu")

        # Call C++ function
        num_colors = wp._src.context.runtime.core.wp_graph_coloring(
            num_nodes,
            edge_indices.__ctype__(),
            ColoringAlgorithm.GREEDY,
            particle_colors.__ctype__(),
        )

        # Two connected nodes should have different colors
        self.assertEqual(num_colors, 2)
        colors_np = particle_colors.numpy()
        self.assertNotEqual(colors_np[0], colors_np[1])

        # SANITY CHECK: Verify that our validation actually works by testing an invalid coloring
        # Intentionally create an INVALID coloring where adjacent nodes have the same color
        invalid_colors_np = np.array([0, 0], dtype=int)
        edge_indices_np = edge_indices.numpy()

        # Should detect exactly 1 invalid edge (the edge connecting nodes 0 and 1)
        invalid_count = validate_graph_coloring(edge_indices_np, invalid_colors_np)
        self.assertEqual(invalid_count, 1, "Sanity check failed: validation should detect the invalid coloring")

        # Now verify the actual valid coloring has no invalid edges
        colors_np = particle_colors.numpy()
        invalid_count = validate_graph_coloring(edge_indices_np, colors_np)
        self.assertEqual(invalid_count, 0, "Valid coloring should have no invalid edges")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_coloring_trimesh(self):
        """Test graph coloring on a trimesh (bunny mesh)."""
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        # Load bunny mesh from USD
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        vertices = np.array(usd_geom.GetPointsAttr().Get())
        faces = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        num_vertices = len(vertices)

        # SANITY CHECK: Verify mesh has reasonable size
        self.assertGreater(num_vertices, 100, "Bunny mesh should have at least 100 vertices")
        self.assertGreater(len(faces), 300, "Bunny mesh should have at least 100 triangles (300 face indices)")

        # Build trimesh edges from faces
        trimesh_edge_indices = build_trimesh_edges_from_faces(faces)

        # SANITY CHECK: Verify we got a non-empty edge list
        self.assertGreater(len(trimesh_edge_indices), 0, "Edge list should not be empty")
        self.assertEqual(trimesh_edge_indices.shape[1], 4, "Edge list should have 4 columns [o1, o2, v1, v2]")

        # Test 1: Coloring without bending - use simple edge list (v1, v2 pairs)
        edge_indices_cpu = wp.array(trimesh_edge_indices[:, 2:], dtype=int, device="cpu")
        particle_colors = wp.empty(shape=(num_vertices), dtype=int, device="cpu")

        # Test GREEDY algorithm
        num_colors_greedy = wp._src.context.runtime.core.wp_graph_coloring(
            num_vertices,
            edge_indices_cpu.__ctype__(),
            ColoringAlgorithm.GREEDY,
            particle_colors.__ctype__(),
        )

        # SANITY CHECK: Number of colors should be reasonable
        self.assertGreater(num_colors_greedy, 0, "Should produce at least 1 color")
        self.assertLess(num_colors_greedy, num_vertices, "Should not need more colors than vertices")
        self.assertLessEqual(num_colors_greedy, 10, "Bunny mesh should need at most ~10 colors")

        # Validate coloring
        edge_indices_np = edge_indices_cpu.numpy()
        colors_np = particle_colors.numpy()
        invalid_edges = validate_graph_coloring(edge_indices_np, colors_np)
        self.assertEqual(invalid_edges, 0, f"GREEDY coloring has {invalid_edges} invalid edges")

        # Test MCS algorithm
        num_colors_mcs = wp._src.context.runtime.core.wp_graph_coloring(
            num_vertices,
            edge_indices_cpu.__ctype__(),
            ColoringAlgorithm.MCS,
            particle_colors.__ctype__(),
        )

        # SANITY CHECK: MCS should produce fewer or equal colors than GREEDY
        self.assertLessEqual(
            num_colors_mcs, num_colors_greedy, "MCS algorithm should use fewer or equal colors than GREEDY"
        )

        # Validate coloring
        colors_np = particle_colors.numpy()
        invalid_edges = validate_graph_coloring(edge_indices_np, colors_np)
        self.assertEqual(invalid_edges, 0, f"MCS coloring has {invalid_edges} invalid edges")

        # Test 2: Coloring with bending - add diagonal edges
        edge_indices_cpu_with_bending = construct_trimesh_graph_edges(trimesh_edge_indices, return_wp_array=True)

        # SANITY CHECK: Bending edges should add more edges than without bending
        self.assertGreater(
            edge_indices_cpu_with_bending.shape[0],
            edge_indices_cpu.shape[0],
            "Graph with bending should have more edges than without",
        )

        # Call C++ function
        num_colors_greedy_bending = wp._src.context.runtime.core.wp_graph_coloring(
            num_vertices,
            edge_indices_cpu_with_bending.__ctype__(),
            ColoringAlgorithm.GREEDY,
            particle_colors.__ctype__(),
        )
        wp._src.context.runtime.core.wp_balance_coloring(
            num_vertices,
            edge_indices_cpu_with_bending.__ctype__(),
            num_colors_greedy_bending,
            1.1,
            particle_colors.__ctype__(),
        )
        # Validate coloring with bending
        edge_indices_bending_np = edge_indices_cpu_with_bending.numpy()
        colors_np = particle_colors.numpy()
        invalid_edges = validate_graph_coloring(edge_indices_bending_np, colors_np)
        self.assertEqual(invalid_edges, 0, f"GREEDY+bending coloring has {invalid_edges} invalid edges")

        # Call C++ function
        num_colors_mcs_bending = wp._src.context.runtime.core.wp_graph_coloring(
            num_vertices,
            edge_indices_cpu_with_bending.__ctype__(),
            ColoringAlgorithm.MCS,
            particle_colors.__ctype__(),
        )

        # Get color distribution before balancing
        color_groups_before = convert_to_color_groups(num_colors_mcs_bending, particle_colors)
        sizes_before = np.array([len(g) for g in color_groups_before], dtype=np.float32)
        ratio_before = np.max(sizes_before) / np.min(sizes_before) if len(sizes_before) > 0 else 1.0

        max_min_ratio = wp._src.context.runtime.core.wp_balance_coloring(
            num_vertices,
            edge_indices_cpu_with_bending.__ctype__(),
            num_colors_mcs_bending,
            1.1,
            particle_colors.__ctype__(),
        )
        # Validate coloring with bending and balancing
        colors_np = particle_colors.numpy()
        invalid_edges = validate_graph_coloring(edge_indices_bending_np, colors_np)
        self.assertEqual(invalid_edges, 0, f"MCS+bending+balancing coloring has {invalid_edges} invalid edges")

        # Verify color balance
        color_categories_balanced = convert_to_color_groups(num_colors_mcs_bending, particle_colors)

        color_sizes = np.array([c.shape[0] for c in color_categories_balanced], dtype=np.float32)
        ratio_after = np.max(color_sizes) / np.min(color_sizes)

        # SANITY CHECK: Color balancing should improve or maintain the ratio
        self.assertLessEqual(
            ratio_after,
            ratio_before + 0.01,  # Allow tiny numerical error
            "Color balancing should improve or maintain the balance",
        )
        self.assertLessEqual(ratio_after, max_min_ratio, "Balanced ratio should be within the returned max_min_ratio")

        # Test on a lattice grid with bending
        vs, fs = create_lattice_grid(100)
        trimesh_edge_indices_grid = build_trimesh_edges_from_faces(fs)
        edge_indices_grid_with_bending = construct_trimesh_graph_edges(trimesh_edge_indices_grid, return_wp_array=True)

        num_grid_vertices = len(vs)
        particle_colors_grid = wp.empty(shape=(num_grid_vertices), dtype=int, device="cpu")

        # Call C++ function
        num_colors_grid = wp._src.context.runtime.core.wp_graph_coloring(
            num_grid_vertices,
            edge_indices_grid_with_bending.__ctype__(),
            ColoringAlgorithm.MCS,
            particle_colors_grid.__ctype__(),
        )

        # Validate the coloring
        edge_indices_grid_np = edge_indices_grid_with_bending.numpy()
        colors_grid_np = particle_colors_grid.numpy()
        invalid_edges = validate_graph_coloring(edge_indices_grid_np, colors_grid_np)
        self.assertEqual(invalid_edges, 0, f"Lattice grid coloring has {invalid_edges} invalid edges")

    def test_combine_coloring(self):
        """Test combining colorings from two independent graphs."""
        # Create two simple independent graphs

        # Graph 1: A triangle (3 nodes, 3 edges)
        num_nodes_1 = 3
        edge_indices_1 = wp.array([[0, 1], [1, 2], [2, 0]], dtype=int, device="cpu")
        particle_colors_1 = wp.empty(shape=(num_nodes_1), dtype=wp.int32, device="cpu")

        # Call C++ function
        num_colors_1 = wp._src.context.runtime.core.wp_graph_coloring(
            num_nodes_1,
            edge_indices_1.__ctype__(),
            ColoringAlgorithm.MCS,
            particle_colors_1.__ctype__(),
        )

        color_groups_1 = convert_to_color_groups(num_colors_1, particle_colors_1)

        # Graph 2: A square (4 nodes, 4 edges)
        num_nodes_2 = 4
        edge_indices_2 = wp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int, device="cpu")
        particle_colors_2 = wp.empty(shape=(num_nodes_2), dtype=wp.int32, device="cpu")

        # Call C++ function
        num_colors_2 = wp._src.context.runtime.core.wp_graph_coloring(
            num_nodes_2,
            edge_indices_2.__ctype__(),
            ColoringAlgorithm.MCS,
            particle_colors_2.__ctype__(),
        )

        color_groups_2 = convert_to_color_groups(num_colors_2, particle_colors_2)

        # Verify each graph is colored correctly
        edge_indices_1_np = edge_indices_1.numpy()
        colors_1_np = particle_colors_1.numpy()
        invalid_edges_1 = validate_graph_coloring(edge_indices_1_np, colors_1_np)
        self.assertEqual(invalid_edges_1, 0, f"Triangle coloring has {invalid_edges_1} invalid edges")

        edge_indices_2_np = edge_indices_2.numpy()
        colors_2_np = particle_colors_2.numpy()
        invalid_edges_2 = validate_graph_coloring(edge_indices_2_np, colors_2_np)
        self.assertEqual(invalid_edges_2, 0, f"Square coloring has {invalid_edges_2} invalid edges")

        # Verify that all nodes are colored
        self.assertEqual(len(color_groups_1), num_colors_1)
        self.assertEqual(len(color_groups_2), num_colors_2)

        total_nodes_1 = sum(len(group) for group in color_groups_1)
        total_nodes_2 = sum(len(group) for group in color_groups_2)

        self.assertEqual(total_nodes_1, num_nodes_1)
        self.assertEqual(total_nodes_2, num_nodes_2)

        # SANITY CHECK: Triangle (cycle of 3) needs exactly 3 colors
        self.assertEqual(num_colors_1, 3, "Triangle graph should require exactly 3 colors")

        # SANITY CHECK: Square (cycle of 4) needs exactly 2 colors (it's bipartite)
        self.assertEqual(num_colors_2, 2, "Square graph should require exactly 2 colors (bipartite)")

        # SANITY CHECK: Verify each color group has at least one node
        for i, group in enumerate(color_groups_1):
            self.assertGreater(len(group), 0, f"Color group {i} in graph 1 should not be empty")
        for i, group in enumerate(color_groups_2):
            self.assertGreater(len(group), 0, f"Color group {i} in graph 2 should not be empty")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
