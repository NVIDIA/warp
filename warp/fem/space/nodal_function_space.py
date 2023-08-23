from typing import Any

import warp as wp

from warp.fem.types import DofIndex, ElementIndex, Coords, get_node_coord
from warp.fem.geometry import GeometryPartition
from warp.fem import utils


from .function_space import FunctionSpace
from .dof_mapper import DofMapper, IdentityMapper
from .partition import make_space_partition, SpacePartition


class NodalFunctionSpace(FunctionSpace):
    """Function space where values are collocated at nodes"""

    def __init__(self, dtype: type = float, dof_mapper: DofMapper = None):
        self.dof_mapper = IdentityMapper(dtype) if dof_mapper is None else dof_mapper
        self.dtype = self.dof_mapper.value_dtype
        self.dof_dtype = self.dof_mapper.dof_dtype

        if self.dtype == wp.float32:
            self.gradient_dtype = wp.vec2
        elif self.dtype == wp.vec2:
            self.gradient_dtype = wp.mat22
        elif self.dtype == wp.vec3:
            self.gradient_dtype = wp.mat33
        else:
            self.gradient_dtype = None

        self.VALUE_DOF_COUNT = self.dof_mapper.DOF_SIZE
        self.unit_dof_value = self._make_unit_dof_value(self.dof_mapper)

    @property
    def name(self):
        return f"{self.__class__.__qualname__}_{self.ORDER}_{self.dof_mapper}".replace(".", "_")

    def make_field(
        self,
        space_partition: SpacePartition = None,
        geometry_partition: GeometryPartition = None,
    ) -> "wp.fem.field.NodalField":
        from warp.fem.field import NodalField

        if space_partition is None:
            space_partition = make_space_partition(self, geometry_partition)

        return NodalField(space=self, space_partition=space_partition)

    @staticmethod
    def _make_unit_dof_value(dof_mapper: DofMapper):
        from warp.fem import cache

        def unit_dof_value(args: Any, dof: DofIndex):
            return dof_mapper.dof_to_value(utils.unit_element(dof_mapper.dof_dtype(0.0), get_node_coord(dof)))

        return cache.get_func(unit_dof_value, str(dof_mapper))

    # Interface for generating Trace space

    def _inner_cell_index(args: Any, side_index: ElementIndex):
        """Given a side, returns the index of the inner cell"""
        raise NotImplementedError

    def _outer_cell_index(args: Any, side_index: ElementIndex):
        """Given a side, returns the index of the outer cell"""
        raise NotImplementedError

    def _inner_cell_coords(args: Any, side_index: ElementIndex, side_coords: Coords):
        """Given coordinates within a side, returns coordinates within the inner cell"""
        raise NotImplementedError

    def _outer_cell_coords(args: Any, side_index: ElementIndex, side_coords: Coords):
        """Given coordinates within a side, returns coordinates within the outer cell"""
        raise NotImplementedError

    def _cell_to_side_coords(
        args: Any,
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        """Given coordinates within a cell, returns coordinates within a side, or OUTSIDE"""
        raise NotImplementedError


class NodalFunctionSpaceTrace(NodalFunctionSpace):
    """Trace of a NodalFunctionSpace"""

    def __init__(self, space: NodalFunctionSpace):
        self._space = space

        super().__init__(space.dtype, space.dof_mapper)
        self.geometry = space.geometry

        self.NODES_PER_ELEMENT = wp.constant(2 * space.NODES_PER_ELEMENT)
        self.DIMENSION = space.DIMENSION - 1

        self.SpaceArg = space.SpaceArg
        self.space_arg_value = space.space_arg_value

    def node_count(self) -> int:
        return self._space.node_count()

    @property
    def name(self):
        return f"{self._space.name}_Trace"

    @staticmethod
    def _make_element_node_index(space: NodalFunctionSpace):
        from warp.fem import cache

        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        def trace_element_node_index(args: space.SpaceArg, element_index: ElementIndex, node_index_in_elt: int):
            if node_index_in_elt < NODES_PER_ELEMENT:
                inner_element = space._inner_cell_index(args, element_index)
                return space.element_node_index(args, inner_element, node_index_in_elt)

            outer_element = space._outer_cell_index(args, element_index)
            return space.element_node_index(args, outer_element, node_index_in_elt - NODES_PER_ELEMENT)

        return cache.get_func(trace_element_node_index, space.name)

    @staticmethod
    def _make_node_coords_in_element(space: NodalFunctionSpace):
        from warp.fem import cache

        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        def trace_node_coords_in_element(
            args: space.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < NODES_PER_ELEMENT:
                neighbour_elem = space._inner_cell_index(args, element_index)
                neighbour_coords = space.node_coords_in_element(args, neighbour_elem, node_index_in_elt)
            else:
                neighbour_elem = space._outer_cell_index(args, element_index)
                neighbour_coords = space.node_coords_in_element(
                    args,
                    neighbour_elem,
                    node_index_in_elt - NODES_PER_ELEMENT,
                )

            return space._cell_to_side_coords(args, element_index, neighbour_elem, neighbour_coords)

        return cache.get_func(trace_node_coords_in_element, space.name)

    @staticmethod
    def _make_element_inner_weight(space: NodalFunctionSpace):
        from warp.fem import cache

        def trace_element_inner_weight(
            args: space.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            return space.element_inner_weight(
                args,
                space._inner_cell_index(args, element_index),
                space._inner_cell_coords(args, element_index, coords),
                node_index_in_elt,
            )

        return cache.get_func(trace_element_inner_weight, space.name)

    @staticmethod
    def _make_element_outer_weight(space: NodalFunctionSpace):
        from warp.fem import cache

        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        def trace_element_outer_weight(
            args: space.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            return space.element_outer_weight(
                args,
                space._outer_cell_index(args, element_index),
                space._outer_cell_coords(args, element_index, coords),
                node_index_in_elt - NODES_PER_ELEMENT,
            )

        return cache.get_func(trace_element_outer_weight, space.name)

    @staticmethod
    def _make_element_inner_weight_gradient(space: NodalFunctionSpace):
        from warp.fem import cache

        def trace_element_inner_weight_gradient(
            args: space.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            return space.element_inner_weight_gradient(
                args,
                space._inner_cell_index(args, element_index),
                space._inner_cell_coords(args, element_index, coords),
                node_index_in_elt,
            )

        return cache.get_func(trace_element_inner_weight_gradient, space.name)

    @staticmethod
    def _make_element_outer_weight_gradient(space: NodalFunctionSpace):
        from warp.fem import cache

        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        def trace_element_outer_weight_gradient(
            args: space.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            return space.element_outer_weight_gradient(
                args,
                space._outer_cell_index(args, element_index),
                space._outer_cell_coords(args, element_index, coords),
                node_index_in_elt - NODES_PER_ELEMENT,
            )

        return cache.get_func(trace_element_outer_weight_gradient, space.name)

    def __eq__(self, other: "NodalFunctionSpaceTrace") -> bool:
        return self._space == other._space
