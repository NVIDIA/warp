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

from typing import Any, Optional

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Geometry
from warp.fem.linalg import generalized_inner, generalized_outer
from warp.fem.types import NULL_QP_INDEX, Coords, ElementIndex, make_free_sample
from warp.fem.utils import type_basis_element

from .basis_space import BasisSpace
from .dof_mapper import DofMapper, IdentityMapper
from .function_space import FunctionSpace
from .partition import SpacePartition, make_space_partition


class CollocatedFunctionSpace(FunctionSpace):
    """Function space where values are collocated at nodes"""

    @wp.struct
    class LocalValueMap:
        pass

    def __init__(self, basis: BasisSpace, dtype: type = float, dof_mapper: DofMapper = None):
        self.dof_mapper = IdentityMapper(dtype) if dof_mapper is None else dof_mapper
        self._basis = basis

        super().__init__(topology=basis.topology)

        self.dtype = self.dof_mapper.value_dtype
        self.dof_dtype = self.dof_mapper.dof_dtype
        self.VALUE_DOF_COUNT = self.dof_mapper.DOF_SIZE
        self.NODE_DOF_COUNT = self.dof_mapper.DOF_SIZE

        self.SpaceArg = self._basis.BasisArg
        self.space_arg_value = self._basis.basis_arg_value

        self.ORDER = self._basis.ORDER

        self.node_basis_element = self._make_node_basis_element()
        self.value_basis_element = self._make_value_basis_element()

        self.node_coords_in_element = self._basis.make_node_coords_in_element()
        self.node_quadrature_weight = self._basis.make_node_quadrature_weight()
        self.element_inner_weight = self._basis.make_element_inner_weight()
        self.element_inner_weight_gradient = self._basis.make_element_inner_weight_gradient()
        self.element_outer_weight = self._basis.make_element_outer_weight()
        self.element_outer_weight_gradient = self._basis.make_element_outer_weight_gradient()

        self.space_value = self._make_space_value()
        self.space_gradient = self._make_space_gradient()
        self.space_divergence = self._make_space_divergence()

        self.node_dof_value = self._make_node_dof_value()

        # For backward compatibility
        if hasattr(basis, "node_grid"):
            self.node_grid = basis.node_grid
        if hasattr(basis, "node_triangulation"):
            self.node_triangulation = basis.node_triangulation
        if hasattr(basis, "node_tets"):
            self.node_tets = basis.node_tets
        if hasattr(basis, "node_hexes"):
            self.node_hexes = basis.node_hexes
        if hasattr(basis, "vtk_cells"):
            self.vtk_cells = basis.vtk_cells

    @property
    def name(self):
        return f"{self._basis.name}_{self.dof_mapper}".replace(".", "_")

    def node_positions(self, out: Optional[wp.array] = None) -> wp.array:
        return self._basis.node_positions(out=out)

    def make_field(
        self,
        space_partition: Optional[SpacePartition] = None,
    ) -> "wp.fem.field.NodalField":
        from warp.fem.field import NodalField

        if space_partition is None:
            space_partition = make_space_partition(space_topology=self.topology)

        return NodalField(space=self, space_partition=space_partition)

    def trace(self) -> "CollocatedFunctionSpace":
        return CollocatedFunctionSpaceTrace(self)

    def _make_node_basis_element(self):
        basis_element = type_basis_element(self.dof_dtype)
        return basis_element

    def _make_value_basis_element(self):
        @cache.dynamic_func(suffix=self.name)
        def value_basis_element(dof_coord: int, value_map: CollocatedFunctionSpace.LocalValueMap):
            return self.dof_mapper.dof_to_value(self.node_basis_element(dof_coord))

        return value_basis_element

    @wp.func
    def local_value_map_inner(
        elt_arg: Any,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        return CollocatedFunctionSpace.LocalValueMap()

    @wp.func
    def local_value_map_outer(
        elt_arg: Any,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        return CollocatedFunctionSpace.LocalValueMap()

    def _make_space_value(self):
        @cache.dynamic_func(suffix=self.name)
        def value_func(
            dof_value: self.dof_dtype,
            node_weight: self._basis.weight_type,
            local_value_map: self.LocalValueMap,
        ):
            return node_weight * self.dof_mapper.dof_to_value(dof_value)

        return value_func

    def _make_space_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def gradient_func(
            dof_value: self.dof_dtype,
            node_weight_gradient: self._basis.weight_gradient_type,
            local_value_map: self.LocalValueMap,
            grad_transform: Any,
        ):
            return generalized_outer(self.dof_mapper.dof_to_value(dof_value), node_weight_gradient * grad_transform)

        return gradient_func

    def _make_space_divergence(self):
        @cache.dynamic_func(suffix=self.name)
        def divergence_func(
            dof_value: self.dof_dtype,
            node_weight_gradient: self._basis.weight_gradient_type,
            local_value_map: self.LocalValueMap,
            grad_transform: Any,
        ):
            return generalized_inner(self.dof_mapper.dof_to_value(dof_value), node_weight_gradient * grad_transform)

        return divergence_func

    def _make_node_dof_value(self):
        @cache.dynamic_func(suffix=self.name)
        def node_dof_value(
            elt_arg: self.ElementArg,
            space_arg: self.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
            space_value: self.dtype,
        ):
            return self.dof_mapper.value_to_dof(space_value)

        return node_dof_value


class CollocatedFunctionSpaceTrace(CollocatedFunctionSpace):
    """Trace of a :class:`CollocatedFunctionSpace`"""

    def __init__(self, space: CollocatedFunctionSpace):
        self._space = space
        super().__init__(space._basis.trace(), space.dtype, space.dof_mapper)

    @property
    def name(self):
        return f"{self._space.name}_Trace"

    def __eq__(self, other: "CollocatedFunctionSpaceTrace") -> bool:
        return self._space == other._space


class VectorValuedFunctionSpace(FunctionSpace):
    """Function space whose values are vectors"""

    def __init__(self, basis: BasisSpace):
        self._basis = basis

        super().__init__(topology=basis.topology)

        self.dtype = cache.cached_vec_type(self.geometry.dimension, dtype=float)
        self.dof_dtype = wp.float32

        self.VALUE_DOF_COUNT = self.geometry.dimension
        self.NODE_DOF_COUNT = 1

        self.SpaceArg = self._basis.BasisArg
        self.space_arg_value = self._basis.basis_arg_value

        self.ORDER = self._basis.ORDER

        self.LocalValueMap = cache.cached_mat_type(
            shape=(self.geometry.dimension, self.geometry.cell_dimension), dtype=float
        )

        self.value_basis_element = self._make_value_basis_element()

        self.node_coords_in_element = self._basis.make_node_coords_in_element()
        self.node_quadrature_weight = self._basis.make_node_quadrature_weight()
        self.element_inner_weight = self._basis.make_element_inner_weight()
        self.element_inner_weight_gradient = self._basis.make_element_inner_weight_gradient()
        self.element_outer_weight = self._basis.make_element_outer_weight()
        self.element_outer_weight_gradient = self._basis.make_element_outer_weight_gradient()

        self.space_value = self._make_space_value()
        self.space_gradient = self._make_space_gradient()
        self.space_divergence = self._make_space_divergence()

        self.node_dof_value = self._make_node_dof_value()

    @property
    def name(self):
        return self._basis.name

    def node_positions(self, out: Optional[wp.array] = None) -> wp.array:
        return self._basis.node_positions(out=out)

    def make_field(
        self,
        space_partition: Optional[SpacePartition] = None,
    ) -> "wp.fem.field.NodalField":
        from warp.fem.field import NodalField

        if space_partition is None:
            space_partition = make_space_partition(space_topology=self.topology)

        return NodalField(space=self, space_partition=space_partition)

    @wp.func
    def node_basis_element(dof_coord: int):
        return 1.0

    def _make_value_basis_element(self):
        basis_element = type_basis_element(self.dtype)

        @cache.dynamic_func(suffix=self.name)
        def value_basis_element(dof_coord: int, value_map: Any):
            return value_map * basis_element(dof_coord)

        return value_basis_element

    def _make_space_value(self):
        @cache.dynamic_func(suffix=self.name)
        def value_func(
            dof_value: self.dof_dtype,
            node_weight: self._basis.weight_type,
            local_value_map: self.LocalValueMap,
        ):
            return local_value_map * (node_weight * dof_value)

        return value_func

    def _make_space_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def gradient_func(
            dof_value: self.dof_dtype,
            node_weight_gradient: self._basis.weight_gradient_type,
            local_value_map: self.LocalValueMap,
            grad_transform: Any,
        ):
            return dof_value * local_value_map * node_weight_gradient * grad_transform

        return gradient_func

    def _make_space_divergence(self):
        @cache.dynamic_func(suffix=self.name)
        def divergence_func(
            dof_value: self.dof_dtype,
            node_weight_gradient: self._basis.weight_gradient_type,
            local_value_map: self.LocalValueMap,
            grad_transform: Any,
        ):
            return dof_value * wp.trace(local_value_map * node_weight_gradient * grad_transform)

        return divergence_func

    def _make_node_dof_value(self):
        @cache.dynamic_func(suffix=self.name)
        def node_dof_value(
            elt_arg: self.ElementArg,
            space_arg: self.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
            space_value: self.dtype,
        ):
            coords = self.node_coords_in_element(elt_arg, space_arg, element_index, node_index_in_elt)
            weight = self.element_inner_weight(
                elt_arg, space_arg, element_index, coords, node_index_in_elt, NULL_QP_INDEX
            )
            local_value_map = self.local_value_map_inner(elt_arg, element_index, coords)

            unit_value = local_value_map * weight
            return wp.dot(space_value, unit_value) / wp.length_sq(unit_value)

        return node_dof_value


class CovariantFunctionSpace(VectorValuedFunctionSpace):
    """Function space whose values are covariant vectors"""

    def __init__(self, basis: BasisSpace):
        super().__init__(basis)

        self.local_value_map_inner = self._make_local_value_map()
        self.local_value_map_outer = self.local_value_map_inner

    def trace(self) -> "CovariantFunctionSpaceTrace":
        return CovariantFunctionSpaceTrace(self)

    def _make_local_value_map(self):
        @cache.dynamic_func(suffix=self.name)
        def local_value_map(
            elt_arg: self.ElementArg,
            element_index: ElementIndex,
            element_coords: Coords,
        ):
            J = wp.transpose(
                self.geometry.cell_inverse_deformation_gradient(
                    elt_arg, make_free_sample(element_index, element_coords)
                )
            )
            return J

        return local_value_map


class CovariantFunctionSpaceTrace(VectorValuedFunctionSpace):
    """Trace of a :class:`CovariantFunctionSpace`"""

    def __init__(self, space: VectorValuedFunctionSpace):
        self._space = space
        super().__init__(space._basis.trace())

        self.local_value_map_inner = self._make_local_value_map_inner()
        self.local_value_map_outer = self._make_local_value_map_outer()

    @property
    def name(self):
        return f"{self._space.name}_Trace"

    def __eq__(self, other: "CovariantFunctionSpaceTrace") -> bool:
        return self._space == other._space

    def _make_local_value_map_inner(self):
        @cache.dynamic_func(suffix=self.name)
        def local_value_map_inner(
            elt_arg: self.ElementArg,
            element_index: ElementIndex,
            element_coords: Coords,
        ):
            return wp.transpose(
                self.geometry.side_inner_inverse_deformation_gradient(
                    elt_arg, make_free_sample(element_index, element_coords)
                )
            )

        return local_value_map_inner

    def _make_local_value_map_outer(self):
        @cache.dynamic_func(suffix=self.name)
        def local_value_map_outer(
            elt_arg: self.ElementArg,
            element_index: ElementIndex,
            element_coords: Coords,
        ):
            return wp.transpose(
                self.geometry.side_outer_inverse_deformation_gradient(
                    elt_arg, make_free_sample(element_index, element_coords)
                )
            )

        return local_value_map_outer


class ContravariantFunctionSpace(VectorValuedFunctionSpace):
    """Function space whose values are contravariant vectors"""

    def __init__(self, basis: BasisSpace):
        super().__init__(basis)

        self.local_value_map_inner = self._make_local_value_map()
        self.local_value_map_outer = self.local_value_map_inner

    def trace(self) -> "ContravariantFunctionSpaceTrace":
        return ContravariantFunctionSpaceTrace(self)

    def _make_local_value_map(self):
        @cache.dynamic_func(suffix=self.name)
        def local_value_map(
            elt_arg: self.ElementArg,
            element_index: ElementIndex,
            element_coords: Coords,
        ):
            F = self.geometry.cell_deformation_gradient(elt_arg, make_free_sample(element_index, element_coords))
            return F / Geometry._element_measure(F)

        return local_value_map


class ContravariantFunctionSpaceTrace(VectorValuedFunctionSpace):
    """Trace of a :class:`ContravariantFunctionSpace`"""

    def __init__(self, space: ContravariantFunctionSpace):
        self._space = space
        super().__init__(space._basis.trace())

        self.local_value_map_inner = self._make_local_value_map_inner()
        self.local_value_map_outer = self._make_local_value_map_outer()

    @property
    def name(self):
        return f"{self._space.name}_Trace"

    def __eq__(self, other: "ContravariantFunctionSpaceTrace") -> bool:
        return self._space == other._space

    def _make_local_value_map_inner(self):
        @cache.dynamic_func(suffix=self.name)
        def local_value_map_inner(
            elt_arg: self.ElementArg,
            element_index: ElementIndex,
            element_coords: Coords,
        ):
            cell_index = self.geometry.side_inner_cell_index(elt_arg, element_index)
            cell_coords = self.geometry.side_inner_cell_coords(elt_arg, element_index, element_coords)
            cell_arg = self.geometry.side_to_cell_arg(elt_arg)

            F = self.geometry.cell_deformation_gradient(cell_arg, make_free_sample(cell_index, cell_coords))
            return F / Geometry._element_measure(F)

        return local_value_map_inner

    def _make_local_value_map_outer(self):
        @cache.dynamic_func(suffix=self.name)
        def local_value_map_outer(
            elt_arg: self.ElementArg,
            element_index: ElementIndex,
            element_coords: Coords,
        ):
            cell_index = self.geometry.side_outer_cell_index(elt_arg, element_index)
            cell_coords = self.geometry.side_outer_cell_coords(elt_arg, element_index, element_coords)
            cell_arg = self.geometry.side_to_cell_arg(elt_arg)

            F = self.geometry.cell_deformation_gradient(cell_arg, make_free_sample(cell_index, cell_coords))
            return F / Geometry._element_measure(F)

        return local_value_map_outer
