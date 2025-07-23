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
from typing import Any, Optional, Set, Union

import warp as wp
import warp.codegen
import warp.context
import warp.fem.cache as cache
import warp.fem.utils as utils
from warp.fem.geometry import (
    Element,
    Geometry,
    GeometryPartition,
    WholeGeometryPartition,
)
from warp.fem.operator import Operator
from warp.fem.types import NULL_ELEMENT_INDEX, ElementKind

GeometryOrPartition = Union[Geometry, GeometryPartition]


class GeometryDomain:
    """Interface class for domains, i.e. (partial) views of elements in a Geometry"""

    def __init__(self, geometry: GeometryOrPartition):
        if isinstance(geometry, GeometryPartition):
            self.geometry_partition = geometry
        else:
            self.geometry_partition = WholeGeometryPartition(geometry)

        self.geometry = self.geometry_partition.geometry

    @property
    def name(self) -> str:
        return f"{self.geometry_partition.name}_{self.__class__.__name__}"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__ and self.geometry_partition == other.geometry_partition

    @property
    def element_kind(self) -> ElementKind:
        """Kind of elements that this domain contains (cells or sides)"""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Dimension of the elements of the domain"""
        raise NotImplementedError

    def element_count(self) -> int:
        """Number of elements in the domain"""
        raise NotImplementedError

    def geometry_element_count(self) -> int:
        """Number of elements in the underlying geometry"""
        return self.geometry.cell_count()

    def reference_element(self) -> Element:
        """Protypical element"""
        raise NotImplementedError

    def element_index_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        """Value of the argument to be passed to device functions"""
        raise NotImplementedError

    def element_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        """Value of the argument to be passed to device functions"""
        raise NotImplementedError

    ElementIndexArg: warp.codegen.Struct
    """Structure containing arguments to be passed to device functions computing element indices"""

    element_index: wp.Function
    """Device function for retrieving an ElementIndex from a linearized index"""

    element_partition_index: wp.Function
    """Device function for retrieving linearized index in the domain's partition from an ElementIndex"""

    ElementArg: warp.codegen.Struct
    """Structure containing arguments to be passed to device functions computing element geometry"""

    element_measure: wp.Function
    """Device function returning the measure determinant (e.g. volume, area) at a given point"""

    element_measure_ratio: wp.Function
    """Device function returning the ratio of the measure of a side to that of its neighbour cells"""

    element_position: wp.Function
    """Device function returning the element position at a sample point"""

    element_deformation_gradient: wp.Function
    """Device function returning the gradient of the position with respect to the element's reference space"""

    element_normal: wp.Function
    """Device function returning the element normal at a sample point"""

    element_closest_point: wp.Function
    """Device function returning the coordinates of the closest point in a given element to a world position"""

    element_coordinates: wp.Function
    """Device function returning the coordinates corresponding to a world position in a given element reference system"""

    element_lookup: wp.Function
    """Device function returning the sample point in the domain's geometry corresponding to a world position"""

    element_partition_lookup: wp.Function
    """Device function returning the sample point in the domain's geometry partition corresponding to a world position"""

    def notify_operator_usage(self, ops: Set[Operator]):
        """Makes the Domain aware that the operators `ops` will be applied"""
        pass

    @cached_property
    def DomainArg(self):
        return self._make_domain_arg()

    def _make_domain_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class DomainArg:
            geo: self.ElementArg
            index: self.ElementIndexArg

        return DomainArg


class Cells(GeometryDomain):
    """A Domain containing all cells of the geometry or geometry partition"""

    def __init__(self, geometry: GeometryOrPartition):
        super().__init__(geometry)

    @property
    def element_kind(self) -> ElementKind:
        return ElementKind.CELL

    @property
    def dimension(self) -> int:
        return self.geometry.dimension

    def reference_element(self) -> Element:
        return self.geometry.reference_cell()

    def element_count(self) -> int:
        return self.geometry_partition.cell_count()

    def geometry_element_count(self) -> int:
        return self.geometry.cell_count()

    @property
    def ElementIndexArg(self) -> warp.codegen.Struct:
        return self.geometry_partition.CellArg

    def element_index_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        return self.geometry_partition.cell_arg_value(device)

    def fill_element_index_arg(self, arg: ElementIndexArg, device: warp.context.Devicelike):
        self.geometry_partition.fill_cell_arg(arg, device)

    @property
    def element_index(self) -> wp.Function:
        return self.geometry_partition.cell_index

    @property
    def element_partition_index(self) -> wp.Function:
        return self.geometry_partition.partition_cell_index

    def element_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        return self.geometry.cell_arg_value(device)

    def fill_element_arg(self, arg: "ElementArg", device: warp.context.Devicelike):
        self.geometry.fill_cell_arg(arg, device)

    @property
    def ElementArg(self) -> warp.codegen.Struct:
        return self.geometry.CellArg

    @property
    def element_position(self) -> wp.Function:
        return self.geometry.cell_position

    @property
    def element_deformation_gradient(self) -> wp.Function:
        return self.geometry.cell_deformation_gradient

    @property
    def element_measure(self) -> wp.Function:
        return self.geometry.cell_measure

    @property
    def element_measure_ratio(self) -> wp.Function:
        return self.geometry.cell_measure_ratio

    @property
    def element_normal(self) -> wp.Function:
        return self.geometry.cell_normal

    @property
    def element_closest_point(self) -> wp.Function:
        return self.geometry.cell_closest_point

    @property
    def element_coordinates(self) -> wp.Function:
        return self.geometry.cell_coordinates

    @property
    def element_lookup(self) -> wp.Function:
        return self.geometry.cell_lookup

    @property
    def element_partition_lookup(self) -> wp.Function:
        pos_type = cache.cached_vec_type(self.geometry.dimension, dtype=float)

        @cache.dynamic_func(suffix=self.geometry_partition.name)
        def is_in_partition(args: self.ElementIndexArg, cell_index: int):
            return self.geometry_partition.partition_cell_index(args, cell_index) != NULL_ELEMENT_INDEX

        filtered_cell_lookup = self.geometry.make_filtered_cell_lookup(filter_func=is_in_partition)

        # overloads
        filter_target = True
        pos_type = cache.cached_vec_type(self.geometry.dimension, dtype=float)

        @cache.dynamic_func(suffix=self.name, allow_overloads=True)
        def cell_partition_lookup(args: self.DomainArg, pos: pos_type, max_dist: float):
            return filtered_cell_lookup(args.geo, pos, max_dist, args.index, filter_target)

        @cache.dynamic_func(suffix=self.name, allow_overloads=True)
        def cell_partition_lookup(args: self.DomainArg, pos: pos_type):
            max_dist = 0.0
            return filtered_cell_lookup(args.geo, pos, max_dist, args.index, filter_target)

        return cell_partition_lookup

    def supports_lookup(self, device):
        return self.geometry.supports_cell_lookup(device)

    @property
    def domain_cell_arg(self) -> wp.Function:
        return Cells._identity_fn

    def cell_domain(self):
        return self

    @wp.func
    def _identity_fn(x: Any):
        return x


class Sides(GeometryDomain):
    """A Domain containing all (interior and boundary) sides of the geometry or geometry partition"""

    def __init__(self, geometry: GeometryOrPartition):
        self.geometry = geometry
        super().__init__(geometry)

        self.element_lookup = None
        self.element_partition_lookup = None
        self.element_filtered_lookup = None

    def supports_lookup(self, device):
        return False

    @property
    def element_kind(self) -> ElementKind:
        return ElementKind.SIDE

    @property
    def dimension(self) -> int:
        return self.geometry.dimension - 1

    def reference_element(self) -> Element:
        return self.geometry.reference_side()

    def element_count(self) -> int:
        return self.geometry_partition.side_count()

    def geometry_element_count(self) -> int:
        return self.geometry.side_count()

    @property
    def ElementIndexArg(self) -> warp.codegen.Struct:
        return self.geometry_partition.SideArg

    def element_index_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        return self.geometry_partition.side_arg_value(device)

    def fill_element_index_arg(self, arg: "ElementIndexArg", device: warp.context.Devicelike):
        self.geometry_partition.fill_side_arg(arg, device)

    @property
    def element_index(self) -> wp.Function:
        return self.geometry_partition.side_index

    @property
    def ElementArg(self) -> warp.codegen.Struct:
        return self.geometry.SideArg

    def element_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        return self.geometry.side_arg_value(device)

    def fill_element_arg(self, arg: "ElementArg", device: warp.context.Devicelike):
        self.geometry.fill_side_arg(arg, device)

    @property
    def element_position(self) -> wp.Function:
        return self.geometry.side_position

    @property
    def element_deformation_gradient(self) -> wp.Function:
        return self.geometry.side_deformation_gradient

    @property
    def element_measure(self) -> wp.Function:
        return self.geometry.side_measure

    @property
    def element_measure_ratio(self) -> wp.Function:
        return self.geometry.side_measure_ratio

    @property
    def element_normal(self) -> wp.Function:
        return self.geometry.side_normal

    @property
    def element_closest_point(self) -> wp.Function:
        return self.geometry.side_closest_point

    @property
    def element_coordinates(self) -> wp.Function:
        return self.geometry.side_coordinates

    @property
    def element_inner_cell_index(self) -> wp.Function:
        return self.geometry.side_inner_cell_index

    @property
    def element_outer_cell_index(self) -> wp.Function:
        return self.geometry.side_outer_cell_index

    @property
    def element_inner_cell_coords(self) -> wp.Function:
        return self.geometry.side_inner_cell_coords

    @property
    def element_outer_cell_coords(self) -> wp.Function:
        return self.geometry.side_outer_cell_coords

    @property
    def cell_to_element_coords(self) -> wp.Function:
        return self.geometry.side_from_cell_coords

    @cached_property
    def domain_cell_arg(self) -> wp.Function:
        CellDomainArg = self.cell_domain().DomainArg

        @cache.dynamic_func(suffix=self.name)
        def domain_cell_arg(x: self.DomainArg):
            return CellDomainArg(
                self.geometry.side_to_cell_arg(x.geo),
                self.geometry_partition.side_to_cell_arg(x.index),
            )

        return domain_cell_arg

    def cell_domain(self):
        return Cells(self.geometry_partition)


class BoundarySides(Sides):
    """A Domain containing boundary sides of the geometry or geometry partition"""

    def __init__(self, geometry: GeometryOrPartition):
        super().__init__(geometry)

    def element_count(self) -> int:
        return self.geometry_partition.boundary_side_count()

    def geometry_element_count(self) -> int:
        return self.geometry.boundary_side_count()

    @property
    def element_index(self) -> wp.Function:
        return self.geometry_partition.boundary_side_index


class FrontierSides(Sides):
    """A Domain containing frontier sides of the geometry partition (sides shared with at least another partition)"""

    def __init__(self, geometry: GeometryOrPartition):
        super().__init__(geometry)

    def element_count(self) -> int:
        return self.geometry_partition.frontier_side_count()

    def geometry_element_count(self) -> int:
        raise RuntimeError("Frontier sides not defined at the geometry level")

    @property
    def element_index(self) -> wp.Function:
        return self.geometry_partition.frontier_side_index


class Subdomain(GeometryDomain):
    """Subdomain -- restriction of domain to a subset of its elements"""

    def __init__(
        self,
        domain: GeometryDomain,
        element_mask: Optional[wp.array] = None,
        element_indices: Optional[wp.array] = None,
        temporary_store: Optional[cache.TemporaryStore] = None,
    ):
        """
        Create a subdomain from a subset of elements.

        Exactly one of `element_mask` and `element_indices` should be provided.

        Args:
            domain: the containing domain
            element_mask: Array of length ``domain.element_count()`` indicating which elements should be included. Array values must be either ``1`` (selected) or ``0`` (not selected).
            element_indices: Explicit array of element indices to include
        """

        super().__init__(domain.geometry_partition)

        if element_indices is None:
            if element_mask is None:
                raise ValueError("Either 'element_mask' or 'element_indices' should be provided")
            element_indices, _ = utils.masked_indices(mask=element_mask, temporary_store=temporary_store)
            element_indices = element_indices.detach()
        elif element_mask is not None:
            raise ValueError("Only one of 'element_mask' and 'element_indices' should be provided")

        self._domain = domain
        self._element_indices = element_indices
        self.ElementIndexArg = self._make_element_index_arg()
        self.element_index = self._make_element_index()

        # forward
        self.ElementArg = self._domain.ElementArg
        self.geometry_element_count = self._domain.geometry_element_count
        self.reference_element = self._domain.reference_element
        self.element_arg_value = self._domain.element_arg_value
        self.fill_element_arg = self._domain.fill_element_arg
        self.element_measure = self._domain.element_measure
        self.element_measure_ratio = self._domain.element_measure_ratio
        self.element_position = self._domain.element_position
        self.element_deformation_gradient = self._domain.element_deformation_gradient
        self.element_lookup = self._domain.element_lookup
        self.element_partition_lookup = self._domain.element_partition_lookup
        self.element_normal = self._domain.element_normal

    @property
    def name(self) -> str:
        return f"{self._domain.name}_Subdomain"

    def __eq__(self, other) -> bool:
        return (
            self.__class__ == other.__class__
            and self.geometry_partition == other.geometry_partition
            and self._element_indices == other._element_indices
        )

    @property
    def element_kind(self) -> ElementKind:
        return self._domain.element_kind

    @property
    def dimension(self) -> int:
        return self._domain.dimension

    def element_count(self) -> int:
        return self._element_indices.shape[0]

    def _make_element_index_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class ElementIndexArg:
            domain_arg: self._domain.ElementIndexArg
            element_indices: wp.array(dtype=int)

        return ElementIndexArg

    @cache.cached_arg_value
    def element_index_arg_value(self, device: warp.context.Devicelike):
        arg = self.ElementIndexArg()
        self.fill_element_index_arg(arg, device)
        return arg

    def fill_element_index_arg(self, arg: "GeometryDomain.ElementIndexArg", device: warp.context.Devicelike):
        self._domain.fill_element_index_arg(arg.domain_arg, device)
        arg.element_indices = self._element_indices.to(device)

    def _make_element_index(self) -> wp.Function:
        @cache.dynamic_func(suffix=self.name)
        def element_index(arg: self.ElementIndexArg, index: int):
            return self._domain.element_index(arg.domain_arg, arg.element_indices[index])

        return element_index

    def _make_element_partition_index(self) -> wp.Function:
        @cache.dynamic_func(suffix=self.name)
        def element_partition_index(arg: self.ElementIndexArg, element_index: int):
            return self._domain.element_partition_index(arg.domain_arg, element_index)

        return element_partition_index

    def supports_lookup(self, device):
        return self._domain.supports_lokup(device)

    def cell_domain(self):
        return self._domain.cell_domain()

    @cached_property
    def domain_cell_arg(self) -> wp.Function:
        CellDomainArg = self.cell_domain().DomainArg

        @cache.dynamic_func(suffix=self.name)
        def domain_cell_arg(x: self.DomainArg):
            return CellDomainArg(
                self.geometry.side_to_cell_arg(x.geo),
                self.geometry_partition.side_to_cell_arg(x.index.domain_arg),
            )

        return domain_cell_arg
