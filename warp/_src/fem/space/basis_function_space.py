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

from functools import cached_property
from typing import Any, ClassVar, Optional

import warp as wp
from warp._src.fem import cache
from warp._src.fem.geometry import Geometry
from warp._src.fem.linalg import generalized_inner, generalized_outer
from warp._src.fem.types import Coords, ElementIndex, make_free_sample
from warp._src.fem.utils import type_basis_element

from .basis_space import BasisSpace
from .dof_mapper import DofMapper, IdentityMapper
from .function_space import FunctionSpace
from .partition import SpacePartition, make_space_partition

_wp_module_name_ = "warp.fem.space.basis_function_space"


class CollocatedFunctionSpace(FunctionSpace):
    """Function space where values are collocated at nodes"""

    _dynamic_attribute_constructors: ClassVar = {
        "node_basis_element": lambda obj: obj._make_node_basis_element(),
        "value_basis_element": lambda obj: obj._make_value_basis_element(),
        "space_value": lambda obj: obj._make_space_value(),
        "space_gradient": lambda obj: obj._make_space_gradient(),
        "space_divergence": lambda obj: obj._make_space_divergence(),
        "dof_value": lambda obj: obj._make_dof_value(),
    }

    LocalValueMap = float

    def __init__(self, basis: BasisSpace, dtype: type = float, dof_mapper: DofMapper = None):
        self.dof_mapper = IdentityMapper(dtype) if dof_mapper is None else dof_mapper

        super().__init__(basis=basis)

        self.dtype = self.dof_mapper.value_dtype
        self.dof_dtype = self.dof_mapper.dof_dtype
        self.weight_dtype = self.basis.weight_type
        self.VALUE_DOF_COUNT = self.dof_mapper.DOF_SIZE
        self.NODE_DOF_COUNT = self.dof_mapper.DOF_SIZE

        self.ORDER = self.basis.ORDER

        cache.setup_dynamic_attributes(self)

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

    @cached_property
    def name(self):
        return f"{self._basis.name}_{self.dof_mapper}".replace(".", "_")

    def node_positions(self, out: Optional[wp.array] = None) -> wp.array:
        return self.basis.node_positions(out=out)

    def make_field(
        self,
        space_partition: Optional[SpacePartition] = None,
    ) -> "wp._src.fem.NodalField":
        from warp._src.fem.field import NodalField  # noqa: PLC0415 (circular import)

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
        return CollocatedFunctionSpace.LocalValueMap(1.0)

    @wp.func
    def local_value_map_outer(
        elt_arg: Any,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        return CollocatedFunctionSpace.LocalValueMap(1.0)

    def _make_space_value(self):
        @cache.dynamic_func(suffix=self.name)
        def value_func(
            dof_value: self.dof_dtype,
            node_weight: self.basis.weight_type,
            local_value_map: self.LocalValueMap,
        ):
            return node_weight * self.dof_mapper.dof_to_value(dof_value)

        return value_func

    def _make_space_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def gradient_func(
            dof_value: self.dof_dtype,
            node_weight_gradient: Any,
            local_value_map: self.LocalValueMap,
        ):
            return generalized_outer(self.dof_mapper.dof_to_value(dof_value), node_weight_gradient)

        return gradient_func

    def _make_space_divergence(self):
        @cache.dynamic_func(suffix=self.name)
        def divergence_func(
            dof_value: self.dof_dtype,
            node_weight_gradient: Any,
            local_value_map: self.LocalValueMap,
        ):
            return generalized_inner(self.dof_mapper.dof_to_value(dof_value), node_weight_gradient)

        return divergence_func

    def _make_dof_value(self):
        @cache.dynamic_func(suffix=self.name)
        def dof_value(
            space_value: self.dtype,
            node_weight: self.weight_dtype,
            local_value_map: self.LocalValueMap,
        ):
            return self.dof_mapper.value_to_dof(space_value) / node_weight

        return dof_value


class CollocatedFunctionSpaceTrace(CollocatedFunctionSpace):
    """Trace of a :class:`CollocatedFunctionSpace`"""

    def __init__(self, space: CollocatedFunctionSpace):
        self._space = space
        super().__init__(space.basis.trace(), space.dtype, space.dof_mapper)

    @cached_property
    def name(self):
        return f"{self._space.name}_Trace"

    def __eq__(self, other: "CollocatedFunctionSpaceTrace") -> bool:
        return self._space == other._space


class VectorValuedFunctionSpace(FunctionSpace):
    """Function space whose values are vectors"""

    _dynamic_attribute_constructors: ClassVar = {
        "value_basis_element": lambda obj: obj._make_value_basis_element(),
        "space_value": lambda obj: obj._make_space_value(),
        "space_gradient": lambda obj: obj._make_space_gradient(),
        "space_divergence": lambda obj: obj._make_space_divergence(),
        "dof_value": lambda obj: obj._make_dof_value(),
    }

    def __init__(self, basis: BasisSpace):
        super().__init__(basis=basis)

        self.dtype = cache.cached_vec_type(self.geometry.dimension, dtype=float)
        self.weight_dtype = self.basis.weight_type
        self.dof_dtype = wp.float32

        self.VALUE_DOF_COUNT = self.geometry.dimension
        self.NODE_DOF_COUNT = 1
        self.ORDER = self.basis.ORDER

        self.LocalValueMap = cache.cached_mat_type(
            shape=(self.geometry.dimension, self.geometry.cell_dimension), dtype=float
        )

        cache.setup_dynamic_attributes(self, cls=__class__)

    @property
    def name(self):
        return self.basis.name

    def node_positions(self, out: Optional[wp.array] = None) -> wp.array:
        return self.basis.node_positions(out=out)

    def make_field(
        self,
        space_partition: Optional[SpacePartition] = None,
    ) -> "wp._src.fem.NodalField":
        from warp._src.fem.field import NodalField  # noqa: PLC0415 (circular import)

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
            node_weight: self.basis.weight_type,
            local_value_map: self.LocalValueMap,
        ):
            return local_value_map * (node_weight * dof_value)

        return value_func

    def _make_space_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def gradient_func(
            dof_value: self.dof_dtype,
            node_weight_gradient: Any,
            local_value_map: self.LocalValueMap,
        ):
            return dof_value * local_value_map * node_weight_gradient

        return gradient_func

    def _make_space_divergence(self):
        @cache.dynamic_func(suffix=self.name)
        def divergence_func(
            dof_value: self.dof_dtype,
            node_weight_gradient: Any,
            local_value_map: self.LocalValueMap,
        ):
            return dof_value * wp.ddot(wp.transpose(local_value_map), node_weight_gradient)

        return divergence_func

    def _make_dof_value(self):
        @cache.dynamic_func(suffix=self.name)
        def dof_value(
            space_value: self.dtype,
            node_weight: self.weight_dtype,
            local_value_map: self.LocalValueMap,
        ):
            dof_axis = local_value_map * node_weight
            return wp.dot(space_value, dof_axis) / wp.length_sq(dof_axis)

        return dof_value


class CovariantFunctionSpace(VectorValuedFunctionSpace):
    """Function space whose values are covariant vectors"""

    _dynamic_attribute_constructors: ClassVar = {
        "local_value_map_inner": lambda obj: obj._make_local_value_map(),
        "local_value_map_outer": lambda obj: obj.local_value_map_inner,
    }

    def __init__(self, basis: BasisSpace):
        super().__init__(basis)

        cache.setup_dynamic_attributes(self, cls=__class__)

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

    _dynamic_attribute_constructors: ClassVar = {
        "local_value_map_inner": lambda obj: obj._make_local_value_map_inner(),
        "local_value_map_outer": lambda obj: obj._make_local_value_map_outer(),
    }

    def __init__(self, space: VectorValuedFunctionSpace):
        self._space = space
        super().__init__(space.basis.trace())

        cache.setup_dynamic_attributes(self, cls=__class__)

    @cached_property
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

    _dynamic_attribute_constructors: ClassVar = {
        "local_value_map_inner": lambda obj: obj._make_local_value_map(),
        "local_value_map_outer": lambda obj: obj.local_value_map_inner,
    }

    def __init__(self, basis: BasisSpace):
        super().__init__(basis)

        cache.setup_dynamic_attributes(self, cls=__class__)

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

    _dynamic_attribute_constructors: ClassVar = {
        "local_value_map_inner": lambda obj: obj._make_local_value_map_inner(),
        "local_value_map_outer": lambda obj: obj._make_local_value_map_outer(),
    }

    def __init__(self, space: ContravariantFunctionSpace):
        self._space = space
        super().__init__(space.basis.trace())

        cache.setup_dynamic_attributes(self, cls=__class__)

    @cached_property
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
