from typing import Optional

import warp as wp
from warp.fem import cache, utils
from warp.fem.types import DofIndex, get_node_coord

from .basis_space import BasisSpace
from .dof_mapper import DofMapper, IdentityMapper
from .function_space import FunctionSpace
from .partition import SpacePartition, make_space_partition


class CollocatedFunctionSpace(FunctionSpace):
    """Function space where values are collocated at nodes"""

    def __init__(self, basis: BasisSpace, dtype: type = float, dof_mapper: DofMapper = None):
        super().__init__(topology=basis.topology)

        self.dof_mapper = IdentityMapper(dtype) if dof_mapper is None else dof_mapper
        self.dtype = self.dof_mapper.value_dtype
        self.dof_dtype = self.dof_mapper.dof_dtype
        self.VALUE_DOF_COUNT = self.dof_mapper.DOF_SIZE

        self._basis = basis
        self.SpaceArg = self._basis.BasisArg

        self.ORDER = self._basis.ORDER

        self.unit_dof_value = self._make_unit_dof_value(self.dof_mapper)

        self.node_coords_in_element = self._basis.make_node_coords_in_element()
        self.node_quadrature_weight = self._basis.make_node_quadrature_weight()
        self.element_inner_weight = self._basis.make_element_inner_weight()
        self.element_inner_weight_gradient = self._basis.make_element_inner_weight_gradient()
        self.element_outer_weight = self._basis.make_element_outer_weight()
        self.element_outer_weight_gradient = self._basis.make_element_outer_weight_gradient()

        # For backward compatibility
        if hasattr(basis, "node_grid"):
            self.node_grid = basis.node_grid
        if hasattr(basis, "node_triangulation"):
            self.node_triangulation = basis.node_triangulation
        if hasattr(basis, "node_tets"):
            self.node_tets = basis.node_tets
        if hasattr(basis, "node_hexes"):
            self.node_hexes = basis.node_hexes

    def space_arg_value(self, device):
        return self._basis.basis_arg_value(device)

    @property
    def name(self):
        return f"{self._basis.name}_{self.dof_mapper}".replace(".", "_")

    @property
    def degree(self):
        """Maximum polynomial degree of the underlying basis"""
        return self.ORDER

    def make_field(
        self,
        space_partition: Optional[SpacePartition] = None,
    ) -> "wp.fem.field.NodalField":
        from warp.fem.field import NodalField

        if space_partition is None:
            space_partition = make_space_partition(space_topology=self.topology)

        return NodalField(space=self, space_partition=space_partition)

    def _make_unit_dof_value(self, dof_mapper: DofMapper):
        @cache.dynamic_func(suffix=self.name)
        def unit_dof_value(geo_arg: self.topology.ElementArg, space_arg: self.SpaceArg, dof: DofIndex):
            return dof_mapper.dof_to_value(utils.unit_element(dof_mapper.dof_dtype(0.0), get_node_coord(dof)))

        return unit_dof_value

    def node_count(self):
        return self.topology.node_count()

    def node_positions(self, out: Optional[wp.array] = None) -> wp.array:
        return self._basis.node_positions(out=out)

    def trace(self) -> "CollocatedFunctionSpace":
        return CollocatedFunctionSpaceTrace(self)


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
