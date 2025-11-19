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
from typing import Any, Optional

import warp as wp
from warp._src.fem import cache
from warp._src.fem.types import NULL_ELEMENT_INDEX, ElementIndex
from warp._src.fem.utils import masked_indices

from .geometry import Geometry

_wp_module_name_ = "warp.fem.geometry.partition"

wp.set_module_options({"enable_backward": False})


class GeometryPartition:
    """Base class for geometry partitions, i.e. subset of cells and sides"""

    class CellArg:
        pass

    class SideArg:
        pass

    geometry: Geometry
    """Underlying geometry"""

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

    @cache.cached_arg_value
    def cell_arg_value(self, device):
        args = self.CellArg()
        self.fill_cell_arg(args, device)
        return args

    def fill_cell_arg(self, args: CellArg, device):
        if self.cell_arg_value is __class__.cell_arg_value:
            raise NotImplementedError()
        args.assign(self.cell_arg_value(device))

    @cache.cached_arg_value
    def side_arg_value(self, device):
        args = self.SideArg()
        self.fill_side_arg(args, device)
        return args

    def fill_side_arg(self, args: SideArg, device):
        if self.side_arg_value is __class__.side_arg_value:
            raise NotImplementedError()
        args.assign(self.side_arg_value(device))

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

        self.cell_index = WholeGeometryPartition._identity_element_index
        self.partition_cell_index = WholeGeometryPartition._identity_element_index

        self.side_index = WholeGeometryPartition._identity_element_index
        self.boundary_side_index = geometry.boundary_side_index
        self.frontier_side_index = WholeGeometryPartition._identity_element_index

    def __eq__(self, other: GeometryPartition) -> bool:
        # Ensures that two whole partition instances of the same geometry are considered equal
        return isinstance(other, WholeGeometryPartition) and self.geometry == other.geometry

    @property
    def side_arg_value(self):
        return self.geometry.side_index_arg_value

    @property
    def fill_side_arg(self):
        return self.geometry.fill_side_index_arg

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

    def fill_cell_arg(self, args: CellArg, device):
        pass

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

        self._partition_side_indices: wp.array = None
        self._boundary_side_indices: wp.array = None
        self._frontier_side_indices: wp.array = None

    @cached_property
    def SideArg(self):
        return self._make_side_arg()

    def _make_side_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class SideArg:
            cell_arg: self.CellArg
            partition_side_indices: wp.array(dtype=int)
            boundary_side_indices: wp.array(dtype=int)
            frontier_side_indices: wp.array(dtype=int)

        return SideArg

    def side_count(self) -> int:
        return self._partition_side_indices.shape[0]

    def boundary_side_count(self) -> int:
        return self._boundary_side_indices.shape[0]

    def frontier_side_count(self) -> int:
        return self._frontier_side_indices.shape[0]

    def fill_side_arg(self, args: SideArg, device):
        self.fill_cell_arg(args.cell_arg, device)
        args.partition_side_indices = self._partition_side_indices.to(device)
        args.boundary_side_indices = self._boundary_side_indices.to(device)
        args.frontier_side_indices = self._frontier_side_indices.to(device)

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
        self,
        cell_arg_value: Any,
        cell_inclusion_test_func: wp.Function,
        device,
        max_side_count: int = -1,
        temporary_store: cache.TemporaryStore = None,
    ):
        self.side_arg_value.invalidate(self)

        if max_side_count == 0:
            self._partition_side_indices = cache.borrow_temporary(temporary_store, dtype=int, shape=(0,), device=device)
            self._boundary_side_indices = self._partition_side_indices
            self._frontier_side_indices = self._partition_side_indices
            return

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

        partition_side_mask = cache.borrow_temporary(
            temporary_store,
            shape=(self.geometry.side_count(),),
            dtype=int,
            device=device,
        )
        boundary_side_mask = cache.borrow_temporary(
            temporary_store,
            shape=(self.geometry.side_count(),),
            dtype=int,
            device=device,
        )
        frontier_side_mask = cache.borrow_temporary(
            temporary_store,
            shape=(self.geometry.side_count(),),
            dtype=int,
            device=device,
        )

        partition_side_mask.zero_()
        boundary_side_mask.zero_()
        frontier_side_mask.zero_()

        wp.launch(
            dim=partition_side_mask.shape[0],
            kernel=count_sides,
            inputs=[
                self.geometry.side_arg_value(device),
                cell_arg_value,
                partition_side_mask,
                boundary_side_mask,
                frontier_side_mask,
            ],
            device=device,
        )

        # Convert counts to indices
        self._partition_side_indices, _ = masked_indices(
            partition_side_mask,
            max_index_count=max_side_count,
            local_to_global=self._partition_side_indices,
            temporary_store=temporary_store,
        )
        self._boundary_side_indices, _ = masked_indices(
            boundary_side_mask,
            max_index_count=max_side_count,
            local_to_global=self._boundary_side_indices,
            temporary_store=temporary_store,
        )
        self._frontier_side_indices, _ = masked_indices(
            frontier_side_mask,
            max_index_count=max_side_count,
            local_to_global=self._frontier_side_indices,
            temporary_store=temporary_store,
        )

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
        temporary_store: cache.TemporaryStore = None,
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

    def fill_cell_arg(self, args: CellArg, device):
        args.cell_begin = self.cell_begin
        args.cell_end = self.cell_end

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
    def __init__(
        self,
        geometry: Geometry,
        cell_mask: "wp.array(dtype=int)",
        max_cell_count: int = -1,
        max_side_count: int = -1,
        temporary_store: Optional[cache.TemporaryStore] = None,
    ):
        """Creates a geometry partition from an active cell mask

        Args:
            geometry: the geometry to partition
            cell_mask: warp array of length ``geometry.cell_count()`` indicating which cells are selected. Array values must be either ``1`` (selected) or ``0`` (not selected).
            max_cell_count: if positive, will be used to limit the number of cells to avoid device/host synchronization
            max_side_count: if positive, will be used to limit the number of sides to avoid device/host synchronization
        """

        super().__init__(geometry)

        self._cells: wp.array = None
        self._partition_cells: wp.array = None

        self._max_cell_count = max_cell_count
        self._max_side_count = max_side_count

        self.rebuild(cell_mask, temporary_store)

    def rebuild(
        self,
        cell_mask: "wp.array(dtype=int)",
        temporary_store: Optional[cache.TemporaryStore] = None,
    ):
        """
        Rebuilds the geometry partition from a new active cell mask

        Args:
            geometry: the geometry to partition
            cell_mask: warp array of length ``geometry.cell_count()`` indicating which cells are selected. Array values must be either ``1`` (selected) or ``0`` (not selected).
            max_cell_count: if positive, will be used to limit the number of cells to avoid device/host synchronization
            max_side_count: if positive, will be used to limit the number of sides to avoid device/host synchronization
        """
        self.cell_arg_value.invalidate(self)

        self._cells, self._partition_cells = masked_indices(
            cell_mask,
            local_to_global=self._cells,
            global_to_local=self._partition_cells,
            max_index_count=self._max_cell_count,
            temporary_store=temporary_store,
        )

        super().compute_side_indices_from_cells(
            self.cell_arg_value(cell_mask.device),
            ExplicitGeometryPartition._cell_inclusion_test,
            max_side_count=self._max_side_count,
            device=cell_mask.device,
            temporary_store=temporary_store,
        )

    def cell_count(self) -> int:
        return self._cells.shape[0]

    @wp.struct
    class CellArg:
        cell_index: wp.array(dtype=int)
        partition_cell_index: wp.array(dtype=int)

    def fill_cell_arg(self, args: CellArg, device):
        args.cell_index = self._cells.to(device)
        args.partition_cell_index = self._partition_cells.to(device)

    @wp.func
    def cell_index(args: CellArg, partition_cell_index: int):
        return args.cell_index[partition_cell_index]

    @wp.func
    def partition_cell_index(args: CellArg, cell_index: int):
        return args.partition_cell_index[cell_index]

    @wp.func
    def _cell_inclusion_test(arg: CellArg, cell_index: int):
        return arg.partition_cell_index[cell_index] != NULL_ELEMENT_INDEX
