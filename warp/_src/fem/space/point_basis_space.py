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

from typing import Any, ClassVar, Optional

import warp as wp
from warp._src.fem import cache
from warp._src.fem.operator import integrand
from warp._src.fem.quadrature import Quadrature
from warp._src.fem.types import (
    NULL_ELEMENT_INDEX,
    Coords,
    ElementIndex,
    ElementKind,
    QuadraturePointIndex,
    make_free_sample,
)
from warp._src.types import type_to_warp

from .basis_space import BasisSpace
from .shape import ShapeFunction
from .topology import SpaceTopology


class UnstructuredPointTopology(SpaceTopology):
    """Topology for unstructured points defined from quadrature formula. See :class:`PointBasisSpace`"""

    _dynamic_attribute_constructors: ClassVar = {
        "TopologyArg": lambda obj: obj._make_topology_arg(),
        "domain_element_index": lambda obj: obj._make_domain_element_index(),
        "element_node_index": lambda obj: obj._make_element_node_index(),
        "element_node_count": lambda obj: obj._make_element_node_count(),
        "side_neighbor_node_counts": lambda obj: obj.make_generic_side_neighbor_node_counts(),
    }

    def __init__(self, quadrature: Quadrature, max_nodes_per_element: int = -1):
        if max_nodes_per_element < 0:
            max_nodes_per_element = quadrature.max_points_per_element()
            if max_nodes_per_element is None:
                raise ValueError("Quadrature must define a maximum number of points per element")

        geo_partition = quadrature.domain.geometry_partition
        if (
            quadrature.domain.element_count() != geo_partition.cell_count()
            or quadrature.domain.element_kind != ElementKind.CELL
        ):
            raise ValueError(
                "Point topology may only be defined on quadrature domains that span the whole geometry partition"
            )

        self._quadrature = quadrature
        self._geo_partition = geo_partition

        super().__init__(quadrature.domain.geometry, max_nodes_per_element=max_nodes_per_element)

        cache.setup_dynamic_attributes(self, cls=__class__)

    def node_count(self):
        return self._quadrature.total_point_count()

    def _make_topology_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class TopologyArg:
            quadrature_arg: self._quadrature.Arg
            element_index_arg: self._geo_partition.CellArg

        return TopologyArg

    def fill_topo_arg(self, arg: "UnstructuredPointTopology.TopologyArg", device):
        self._quadrature.fill_arg(arg.quadrature_arg, device)
        self._quadrature.domain.fill_element_index_arg(arg.element_index_arg, device)

    @property
    def name(self):
        return f"PointTopology_{self._quadrature.name}"

    def _make_domain_element_index(self):
        @cache.dynamic_func(suffix=self.name)
        def domain_element_index(element_index_arg: self._geo_partition.CellArg, element_index: int):
            return self._geo_partition.partition_cell_index(element_index_arg, element_index)

        return domain_element_index

    def _make_element_node_index(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            domain_element_index = self.domain_element_index(topo_arg.element_index_arg, element_index)
            return self._quadrature.point_index(
                elt_arg, topo_arg.quadrature_arg, domain_element_index, element_index, node_index_in_elt
            )

        return element_node_index

    def _make_element_node_count(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_count(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
        ):
            domain_element_index = self.domain_element_index(topo_arg.element_index_arg, element_index)
            if domain_element_index == NULL_ELEMENT_INDEX:
                return 0
            return self._quadrature.point_count(elt_arg, topo_arg.quadrature_arg, domain_element_index, element_index)

        return element_node_count


class PointBasisSpace(BasisSpace):
    _dynamic_attribute_constructors: ClassVar = {
        "ValueStruct": lambda obj: obj._make_value_struct(),
        "squared_distance": lambda obj: obj._make_squared_distance(),
        "squared_distance_gradient": lambda obj: obj._make_squared_distance_gradient(),
    }

    def __init__(
        self,
        quadrature: Quadrature,
        kernel_func: Optional[wp.Function] = None,
        kernel_grad_func: Optional[wp.Function] = None,
        kernel_values: Optional[dict[str, Any]] = None,
        distance_space: str = "reference",
        max_nodes_per_element: int = -1,
    ):
        """
        An unstructured :class:`BasisSpace` with radial basis kernels (by default, the Dirac delta function)

        Args:
            quadrature: Quadrature formula defining the node locations and quadrature weights
            kernel_func: Kernel function to be used for the basis space. First two arguments must be squared distance to the kernel center and the quadrature point index,
              then optionally additional kernel values. Default to Dirac delta function.
            kernel_grad_func: Gradient of the kernel function. Must take same arguments as `kernel_func`. Defaults to zero gradient.
            kernel_values: Dictionary of additional values to be passed to the kernel function
            distance_space: Space in which to compute the distance between the sample and the kernel center point. Can be "reference" or "world". Defaults to "reference".
            max_nodes_per_element: Maximum number of point nodes per element to consider. If not provided, get from the quadrature.
        """

        self._quadrature = quadrature
        self._distance_space = distance_space

        self.ORDER = 0
        self._geo_partition = quadrature.domain.geometry_partition

        if kernel_func is None:
            self.kernel_func = self._dirac_radial_kernel
            self.kernel_grad_func = None
            kernel_values = {"squared_radius": 1.0e-6}
        else:
            self.kernel_func = kernel_func
            self.kernel_grad_func = kernel_grad_func

        self._topology = UnstructuredPointTopology(quadrature, max_nodes_per_element=max_nodes_per_element)

        cache.setup_dynamic_attributes(self)
        self._kernel_arg = self.ValueStruct()
        self.kernel_values = kernel_values or {}

        super().__init__(self._topology)

    @property
    def kernel_values(self) -> dict[str, Any]:
        """Dictionary of additional values to be passed to the kernel function"""
        return self._kernel_values

    @kernel_values.setter
    def kernel_values(self, v: dict[str, Any]):
        self._kernel_values = v
        cache.populate_argument_struct(self._kernel_arg, v, self.kernel_func.func.__name__)

    @property
    def name(self):
        return f"{self._topology.name}_{self.kernel_func.key}_{self._distance_space}"

    @property
    def value(self) -> ShapeFunction.Value:
        return ShapeFunction.Value.Scalar

    def _make_value_struct(self):
        argspec = integrand(self.kernel_func.func).argspec
        arg_types = argspec.annotations.copy()

        try:
            first_arg_type = type_to_warp(arg_types.pop(argspec.args[0]))
            second_arg_type = type_to_warp(arg_types.pop(argspec.args[1]))

            assert first_arg_type == wp.float32 and second_arg_type == wp.int32
        except Exception as err:
            raise ValueError(
                f"First argument of radial kernel '{self.kernel_func.func.__name__}' must be a float (squared distance to kernel center), and second argument must be a int (quadrature point index)"
            ) from err

        return cache.get_argument_struct(arg_types)

    @property
    def BasisArg(self) -> "PointBasisSpace.BasisArg":
        return self.ValueStruct

    def fill_basis_arg(self, arg: "PointBasisSpace.BasisArg", device):
        arg.assign(self._kernel_arg)

    def basis_arg_value(self, device) -> "PointBasisSpace.BasisArg":
        return self._kernel_arg

    def make_node_coords_in_element(self):
        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            elt_arg: self._quadrature.domain.ElementArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            domain_element_index = self.topology.domain_element_index(topo_arg.element_index_arg, element_index)
            return self._quadrature.point_coords(
                elt_arg, topo_arg.quadrature_arg, domain_element_index, element_index, node_index_in_elt
            )

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        @cache.dynamic_func(
            suffix=self.name,
        )
        def node_quadrature_weight(
            elt_arg: self._quadrature.domain.ElementArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            domain_element_index = self.topology.domain_element_index(topo_arg.element_index_arg, element_index)
            return self._quadrature.point_weight(
                elt_arg, topo_arg.quadrature_arg, domain_element_index, element_index, node_index_in_elt
            )

        return node_quadrature_weight

    def _make_squared_distance(self):
        if self._distance_space == "reference":
            ref_delta = self._quadrature.domain.reference_element().prototype.ref_delta

            @cache.dynamic_func(suffix=self.name)
            def squared_distance_reference(
                elt_arg: self._quadrature.domain.ElementArg,
                element_index: ElementIndex,
                coords: Coords,
                point_coords: Coords,
            ):
                return wp.length_sq(ref_delta(coords - point_coords))

            return squared_distance_reference

        @cache.dynamic_func(suffix=self.name)
        def squared_distance_world(
            elt_arg: self._quadrature.domain.ElementArg,
            element_index: ElementIndex,
            coords: Coords,
            point_coords: Coords,
        ):
            sample_x = self._quadrature.domain.element_position(elt_arg, make_free_sample(element_index, coords))
            point_x = self._quadrature.domain.element_position(elt_arg, make_free_sample(element_index, point_coords))
            return wp.length_sq(sample_x - point_x)

        return squared_distance_world

    def _make_squared_distance_gradient(self):
        if self._distance_space == "reference":
            ref_delta = self._quadrature.domain.reference_element().prototype.ref_delta

            @cache.dynamic_func(suffix=self.name)
            def squared_distance_gradient_reference(
                elt_arg: self._quadrature.domain.ElementArg,
                element_index: ElementIndex,
                coords: Coords,
                point_coords: Coords,
            ):
                return 2.0 * ref_delta(coords - point_coords)

            return squared_distance_gradient_reference

        @cache.dynamic_func(suffix=self.name)
        def squared_distance_gradient_world(
            elt_arg: self._quadrature.domain.ElementArg,
            element_index: ElementIndex,
            coords: Coords,
            point_coords: Coords,
        ):
            sample_x = self._quadrature.domain.element_position(elt_arg, make_free_sample(element_index, coords))
            sample_F = self._quadrature.domain.element_deformation_gradient(
                elt_arg, make_free_sample(element_index, coords)
            )
            point_x = self._quadrature.domain.element_position(elt_arg, make_free_sample(element_index, point_coords))
            return 2.0 * (sample_x - point_x) @ sample_F

        return squared_distance_gradient_world

    def make_element_inner_weight(self):
        @cache.dynamic_func(
            suffix=self.name,
            code_transformers=[cache.ExpandStarredArgumentStruct({"basis_arg": self.ValueStruct})],
        )
        def element_inner_weight(
            elt_arg: self._quadrature.domain.ElementArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            domain_element_index = self.topology.domain_element_index(topo_arg.element_index_arg, element_index)
            point_index = self._quadrature.point_index(
                elt_arg, topo_arg.quadrature_arg, domain_element_index, element_index, node_index_in_elt
            )
            point_coord = self._quadrature.point_coords(
                elt_arg, topo_arg.quadrature_arg, domain_element_index, element_index, node_index_in_elt
            )
            squared_dist = self.squared_distance(elt_arg, element_index, coords, point_coord)
            return self.kernel_func(squared_dist, point_index, *basis_arg)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        if wp.static(self.kernel_grad_func is None):
            gradient_vec = cache.cached_vec_type(length=self.geometry.cell_dimension, dtype=float)

            @cache.dynamic_func(suffix=self.name)
            def element_inner_weight_null_gradient(
                elt_arg: self._quadrature.domain.ElementArg,
                topo_arg: self.topology.TopologyArg,
                basis_arg: self.BasisArg,
                element_index: ElementIndex,
                coords: Coords,
                node_index_in_elt: int,
                qp_index: QuadraturePointIndex,
            ):
                return gradient_vec(0.0)

            return element_inner_weight_null_gradient

        @cache.dynamic_func(
            suffix=self.name,
            code_transformers=[cache.ExpandStarredArgumentStruct({"basis_arg": self.ValueStruct})],
        )
        def element_inner_weight_gradient(
            elt_arg: self._quadrature.domain.ElementArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            domain_element_index = self.topology.domain_element_index(topo_arg.element_index_arg, element_index)
            point_index = self._quadrature.point_index(
                elt_arg, topo_arg.quadrature_arg, domain_element_index, element_index, node_index_in_elt
            )
            point_coord = self._quadrature.point_coords(
                elt_arg, topo_arg.quadrature_arg, domain_element_index, element_index, node_index_in_elt
            )

            squared_dist = self.squared_distance(elt_arg, element_index, coords, point_coord)
            kernel_grad = self.kernel_grad_func(squared_dist, point_index, *basis_arg)
            squared_dist_gradient = self.squared_distance_gradient(elt_arg, element_index, coords, point_coord)

            return squared_dist_gradient * kernel_grad

        return element_inner_weight_gradient

    def make_element_outer_weight(self):
        return self.make_element_inner_weight()

    def make_element_outer_weight_gradient(self):
        return self.make_element_inner_weight_gradient()

    def make_trace_node_quadrature_weight(self, trace_basis):
        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(
            elt_arg: trace_basis.geometry.SideArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: trace_basis.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 0.0

        return trace_node_quadrature_weight

    @wp.func
    def _dirac_radial_kernel(
        squared_distance: float,
        qp_index: int,
        squared_radius: float,
    ):
        return wp.where(squared_distance < squared_radius, 1.0, 0.0)
