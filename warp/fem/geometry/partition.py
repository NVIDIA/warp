# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from functools import cached_property
from typing import Any

import warp as wp
from warp.fem.cache import TemporaryStore, borrow_temporary, cached_arg_value, dynamic_struct
from warp.fem.types import NULL_ELEMENT_INDEX, ElementIndex
from warp.fem.utils import masked_indices

from .geometry import Geometry

wp.set_module_options({"enable_backward": False})


class GeometryPartition:
    """Base class for geometry partitions, i.e. subset of cells and sides"""

    class CellArg:
        pass

    class SideArg:
        pass

    def __init__(self, geometry: Geometry):
        self.geometry = geometry

    def cell_count(self) -> int:
        """Number of cells that are 'owned' by this partition"""
        raise NotImplementedError()

    def side_count(self) -> int:
        """Number of sides that are 'owned' by this partition"""
        raise NotImplementedError()

    def boundary_side_count(self) -> int:
        """Number of geo-boundary sides that are 'owned' by this partition"""
        raise NotImplementedError()

    def frontier_side_count(self) -> int:
        """Number of sides with neighbors owned by this and another partition"""
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return f"{self.geometry.name}_{self.__class__.__name__}"

    def __str__(self) -> str:
        return self.name

    def cell_arg_value(self, device):
        raise NotImplementedError()

    def side_arg_value(self, device):
        raise NotImplementedError()

    @staticmethod
    def cell_index(args: CellArg, partition_cell_index: int):
        """Index in the geometry of a partition cell"""
        raise NotImplementedError()

    @staticmethod
    def partition_cell_index(args: CellArg, cell_index: int):
        """Index of a geometry cell in the partition (or ``NULL_ELEMENT_INDEX``)"""
        raise NotImplementedError()

    @staticmethod
    def side_index(args: SideArg, partition_side_index: int):
        """Partition side to side index"""
        raise NotImplementedError()

    @staticmethod
    def boundary_side_index(args: SideArg, boundary_side_index: int):
        """Boundary side to side index"""
        raise NotImplementedError()

    @staticmethod
    def frontier_side_index(args: SideArg, frontier_side_index: int):
        """Frontier side to side index"""
        raise NotImplementedError()


class WholeGeometryPartition(GeometryPartition):
    """Trivial (NOP) partition"""

    def __init__(
        self,
        geometry: Geometry,
    ):
        super().__init__(geometry)

        self.SideArg = geometry.SideIndexArg
        self.side_arg_value = geometry.side_index_arg_value

        self.cell_index = WholeGeometryPartition._identity_element_index
        self.partition_cell_index = WholeGeometryPartition._identity_element_index

        self.side_index = WholeGeometryPartition._identity_element_index
        self.boundary_side_index = geometry.boundary_side_index
        self.frontier_side_index = WholeGeometryPartition._identity_element_index

    def __eq__(self, other: GeometryPartition) -> bool:
        # Ensures that two whole partition instances of the same geometry are considered equal
        return isinstance(other, WholeGeometryPartition) and self.geometry == other.geometry

    def cell_count(self) -> int:
        return self.geometry.cell_count()

    def side_count(self) -> int:
        return self.geometry.side_count()

    def boundary_side_count(self) -> int:
        return self.geometry.boundary_side_count()

    def frontier_side_count(self) -> int:
        return 0

    @wp.struct
    class CellArg:
        pass

    def cell_arg_value(self, device):
        arg = WholeGeometryPartition.CellArg()
        return arg

    @wp.func
    def _identity_element_index(args: Any, idx: ElementIndex):
        return idx

    @property
    def name(self) -> str:
        return self.geometry.name

    @wp.func
    def side_to_cell_arg(side_arg: Any):
        return WholeGeometryPartition.CellArg()


class CellBasedGeometryPartition(GeometryPartition):
    """Geometry partition based on a subset of cells. Interior, boundary and frontier sides are automatically categorized."""

    def __init__(
        self,
        geometry: Geometry,
        device=None,
    ):
        super().__init__(geometry)

    @cached_property
    def SideArg(self):
        return self._make_side_arg()

    def _make_side_arg(self):
        @dynamic_struct(suffix=self.name)
        class SideArg:
            cell_arg: self.CellArg
            partition_side_indices: wp.array(dtype=int)
            boundary_side_indices: wp.array(dtype=int)
            frontier_side_indices: wp.array(dtype=int)

        return SideArg

    def side_count(self) -> int:
        return self._partition_side_indices.array.shape[0]

    def boundary_side_count(self) -> int:
        return self._boundary_side_indices.array.shape[0]

    def frontier_side_count(self) -> int:
        return self._frontier_side_indices.array.shape[0]

    @cached_arg_value
    def side_arg_value(self, device):
        arg = self.SideArg()
        arg.cell_arg = self.cell_arg_value(device)
        arg.partition_side_indices = self._partition_side_indices.array.to(device)
        arg.boundary_side_indices = self._boundary_side_indices.array.to(device)
        arg.frontier_side_indices = self._frontier_side_indices.array.to(device)
        return arg

    @wp.func
    def side_index(args: Any, partition_side_index: int):
        """partition side to side index"""
        return args.partition_side_indices[partition_side_index]

    @wp.func
    def boundary_side_index(args: Any, boundary_side_index: int):
        """Boundary side to side index"""
        return args.boundary_side_indices[boundary_side_index]

    @wp.func
    def frontier_side_index(args: Any, frontier_side_index: int):
        """Frontier side to side index"""
        return args.frontier_side_indices[frontier_side_index]

    def compute_side_indices_from_cells(
        self, cell_arg_value: Any, cell_inclusion_test_func: wp.Function, device, temporary_store: TemporaryStore = None
    ):
        from warp.fem import cache

        cell_arg_type = next(iter(cell_inclusion_test_func.input_types.values()))

        @cache.dynamic_kernel(suffix=f"{self.geometry.name}_{cell_inclusion_test_func.key}")
        def count_sides(
            geo_arg: self.geometry.SideArg,
            cell_arg_value: cell_arg_type,
            partition_side_mask: wp.array(dtype=int),
            boundary_side_mask: wp.array(dtype=int),
            frontier_side_mask: wp.array(dtype=int),
        ):
            side_index = wp.tid()
            inner_cell_index = self.geometry.side_inner_cell_index(geo_arg, side_index)
            outer_cell_index = self.geometry.side_outer_cell_index(geo_arg, side_index)

            inner_in = cell_inclusion_test_func(cell_arg_value, inner_cell_index)
            outer_in = cell_inclusion_test_func(cell_arg_value, outer_cell_index)

            if inner_in:
                # Inner neighbor in partition; count as partition side
                partition_side_mask[side_index] = 1

                # Inner and outer element as the same -- this is a boundary side
                if inner_cell_index == outer_cell_index:
                    boundary_side_mask[side_index] = 1

            if inner_in != outer_in:
                # Exactly one neighbor in partition; count as frontier side
                frontier_side_mask[side_index] = 1

        partition_side_mask = borrow_temporary(
            temporary_store,
            shape=(self.geometry.side_count(),),
            dtype=int,
            device=device,
        )
        boundary_side_mask = borrow_temporary(
            temporary_store,
            shape=(self.geometry.side_count(),),
            dtype=int,
            device=device,
        )
        frontier_side_mask = borrow_temporary(
            temporary_store,
            shape=(self.geometry.side_count(),),
            dtype=int,
            device=device,
        )

        partition_side_mask.array.zero_()
        boundary_side_mask.array.zero_()
        frontier_side_mask.array.zero_()

        wp.launch(
            dim=partition_side_mask.array.shape[0],
            kernel=count_sides,
            inputs=[
                self.geometry.side_arg_value(device),
                cell_arg_value,
                partition_side_mask.array,
                boundary_side_mask.array,
                frontier_side_mask.array,
            ],
            device=device,
        )

        # Convert counts to indices
        self._partition_side_indices, _ = masked_indices(partition_side_mask.array, temporary_store=temporary_store)
        self._boundary_side_indices, _ = masked_indices(boundary_side_mask.array, temporary_store=temporary_store)
        self._frontier_side_indices, _ = masked_indices(frontier_side_mask.array, temporary_store=temporary_store)

        partition_side_mask.release()
        boundary_side_mask.release()
        frontier_side_mask.release()

    @wp.func
    def side_to_cell_arg(side_arg: Any):
        return side_arg.cell_arg


class LinearGeometryPartition(CellBasedGeometryPartition):
    def __init__(
        self,
        geometry: Geometry,
        partition_rank: int,
        partition_count: int,
        device=None,
        temporary_store: TemporaryStore = None,
    ):
        """Creates a geometry partition by uniformly partionning cell indices

        Args:
            geometry: the geometry to partition
            partition_rank: the index of the partition being created
            partition_count: the number of partitions that will be created over the geometry
            device: Warp device on which to perform and store computations
        """
        super().__init__(geometry)

        total_cell_count = geometry.cell_count()

        cells_per_partition = (total_cell_count + partition_count - 1) // partition_count
        self.cell_begin = cells_per_partition * partition_rank
        self.cell_end = min(self.cell_begin + cells_per_partition, total_cell_count)

        super().compute_side_indices_from_cells(
            self.cell_arg_value(device),
            LinearGeometryPartition._cell_inclusion_test,
            device,
            temporary_store=temporary_store,
        )

    def cell_count(self) -> int:
        return self.cell_end - self.cell_begin

    @wp.struct
    class CellArg:
        cell_begin: int
        cell_end: int

    def cell_arg_value(self, device):
        arg = LinearGeometryPartition.CellArg()
        arg.cell_begin = self.cell_begin
        arg.cell_end = self.cell_end
        return arg

    @wp.func
    def cell_index(args: CellArg, partition_cell_index: int):
        """Partition cell to cell index"""
        return args.cell_begin + partition_cell_index

    @wp.func
    def partition_cell_index(args: CellArg, cell_index: int):
        """Partition cell to cell index"""
        return wp.where(
            cell_index >= args.cell_begin and cell_index < args.cell_end,
            cell_index - args.cell_begin,
            NULL_ELEMENT_INDEX,
        )

    @wp.func
    def _cell_inclusion_test(arg: CellArg, cell_index: int):
        return cell_index >= arg.cell_begin and cell_index < arg.cell_end


class ExplicitGeometryPartition(CellBasedGeometryPartition):
    def __init__(self, geometry: Geometry, cell_mask: "wp.array(dtype=int)", temporary_store: TemporaryStore = None):
        """Creates a geometry partition by uniformly partionning cell indices

        Args:
            geometry: the geometry to partition
            cell_mask: warp array of length ``geometry.cell_count()`` indicating which cells are selected. Array values must be either ``1`` (selected) or ``0`` (not selected).
        """

        super().__init__(geometry)

        self._cell_mask = cell_mask
        self._cells, self._partition_cells = masked_indices(self._cell_mask, temporary_store=temporary_store)

        super().compute_side_indices_from_cells(
            self._cell_mask,
            ExplicitGeometryPartition._cell_inclusion_test,
            self._cell_mask.device,
            temporary_store=temporary_store,
        )

    def cell_count(self) -> int:
        return self._cells.array.shape[0]

    @wp.struct
    class CellArg:
        cell_index: wp.array(dtype=int)
        partition_cell_index: wp.array(dtype=int)

    @cached_arg_value
    def cell_arg_value(self, device):
        arg = ExplicitGeometryPartition.CellArg()
        arg.cell_index = self._cells.array.to(device)
        arg.partition_cell_index = self._partition_cells.array.to(device)
        return arg

    @wp.func
    def cell_index(args: CellArg, partition_cell_index: int):
        return args.cell_index[partition_cell_index]

    @wp.func
    def partition_cell_index(args: CellArg, cell_index: int):
        return args.partition_cell_index[cell_index]

    @wp.func
    def _cell_inclusion_test(mask: wp.array(dtype=int), cell_index: int):
        return mask[cell_index] > 0
