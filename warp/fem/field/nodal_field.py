import warp as wp
from warp.fem import cache, utils
from warp.fem.space import CollocatedFunctionSpace, SpacePartition
from warp.fem.types import NULL_NODE_INDEX, ElementIndex, Sample

from .field import DiscreteField


class NodalFieldBase(DiscreteField):
    """Base class for nodal field and nodal field traces. Does not hold values"""

    def __init__(self, space: CollocatedFunctionSpace, space_partition: SpacePartition):
        super().__init__(space, space_partition)

        self.EvalArg = self._make_eval_arg()
        self.ElementEvalArg = self._make_element_eval_arg()
        self.eval_degree = DiscreteField._make_eval_degree(self)

        self._read_node_value = self._make_read_node_value()

        self.eval_inner = self._make_eval_inner()
        self.eval_outer = self._make_eval_outer()
        self.eval_grad_inner = self._make_eval_grad_inner(world_space=True)
        self.eval_grad_outer = self._make_eval_grad_outer(world_space=True)
        self.eval_reference_grad_inner = self._make_eval_grad_inner(world_space=False)
        self.eval_reference_grad_outer = self._make_eval_grad_outer(world_space=False)
        self.eval_div_inner = self._make_eval_div_inner()
        self.eval_div_outer = self._make_eval_div_outer()

        self.set_node_value = self._make_set_node_value()
        self.node_partition_index = self._make_node_partition_index()

    def _make_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class EvalArg:
            dof_values: wp.array(dtype=self.space.dof_dtype)
            space_arg: self.space.SpaceArg
            topology_arg: self.space.topology.TopologyArg
            partition_arg: self.space_partition.PartitionArg

        return EvalArg

    def _make_element_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: self.space.topology.ElementArg
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_read_node_value(self):
        @cache.dynamic_func(suffix=self.name)
        def read_node_value(args: self.ElementEvalArg, geo_element_index: ElementIndex, node_index_in_elt: int):
            nidx = self.space.topology.element_node_index(
                args.elt_arg, args.eval_arg.topology_arg, geo_element_index, node_index_in_elt
            )
            pidx = self.space_partition.partition_node_index(args.eval_arg.partition_arg, nidx)
            if pidx == NULL_NODE_INDEX:
                return self.space.dtype(0.0)

            return self.space.dof_mapper.dof_to_value(args.eval_arg.dof_values[pidx])

        return read_node_value

    def _make_eval_inner(self):
        @cache.dynamic_func(suffix=self.name)
        def eval_inner(args: self.ElementEvalArg, s: Sample):
            res = self.space.element_inner_weight(
                args.elt_arg, args.eval_arg.space_arg, s.element_index, s.element_coords, 0
            ) * self._read_node_value(args, s.element_index, 0)
            node_count = self.space.topology.element_node_count(
                args.elt_arg, args.eval_arg.topology_arg, s.element_index
            )
            for k in range(1, node_count):
                res += self.space.element_inner_weight(
                    args.elt_arg, args.eval_arg.space_arg, s.element_index, s.element_coords, k
                ) * self._read_node_value(args, s.element_index, k)
            return res

        return eval_inner

    def _make_eval_grad_inner(self, world_space: bool):
        if not self.gradient_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_grad_inner_ref_space(args: self.ElementEvalArg, s: Sample):
            res = utils.generalized_outer(
                self._read_node_value(args, s.element_index, 0),
                self.space.element_inner_weight_gradient(
                    args.elt_arg, args.eval_arg.space_arg, s.element_index, s.element_coords, 0
                ),
            )
            node_count = self.space.topology.element_node_count(
                args.elt_arg, args.eval_arg.topology_arg, s.element_index
            )
            for k in range(1, node_count):
                res += utils.generalized_outer(
                    self._read_node_value(args, s.element_index, k),
                    self.space.element_inner_weight_gradient(
                        args.elt_arg, args.eval_arg.space_arg, s.element_index, s.element_coords, k
                    ),
                )
            return res

        @cache.dynamic_func(suffix=self.name)
        def eval_grad_inner_world_space(args: self.ElementEvalArg, s: Sample):
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            res = eval_grad_inner_ref_space(args, s)
            return utils.apply_right(res, grad_transform)

        return eval_grad_inner_world_space if world_space else eval_grad_inner_ref_space

    def _make_eval_div_inner(self):
        if not self.divergence_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_div_inner(args: self.ElementEvalArg, s: Sample):
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)

            res = utils.generalized_inner(
                self._read_node_value(args, s.element_index, 0),
                utils.apply_right(
                    self.space.element_inner_weight_gradient(
                        args.elt_arg, args.eval_arg.space_arg, s.element_index, s.element_coords, 0
                    ),
                    grad_transform,
                ),
            )

            node_count = self.space.topology.element_node_count(
                args.elt_arg, args.eval_arg.topology_arg, s.element_index
            )
            for k in range(1, node_count):
                res += utils.generalized_inner(
                    self._read_node_value(args, s.element_index, k),
                    utils.apply_right(
                        self.space.element_inner_weight_gradient(
                            args.elt_arg, args.eval_arg.space_arg, s.element_index, s.element_coords, k
                        ),
                        grad_transform,
                    ),
                )
            return res

        return eval_div_inner

    def _make_eval_outer(self):
        @cache.dynamic_func(suffix=self.name)
        def eval_outer(args: self.ElementEvalArg, s: Sample):
            res = self.space.element_outer_weight(
                args.elt_arg,
                args.eval_arg.space_arg,
                s.element_index,
                s.element_coords,
                0,
            ) * self._read_node_value(args, s.element_index, 0)
            node_count = self.space.topology.element_node_count(
                args.elt_arg, args.eval_arg.topology_arg, s.element_index
            )
            for k in range(1, node_count):
                res += self.space.element_outer_weight(
                    args.elt_arg,
                    args.eval_arg.space_arg,
                    s.element_index,
                    s.element_coords,
                    k,
                ) * self._read_node_value(args, s.element_index, k)
            return res

        return eval_outer

    def _make_eval_grad_outer(self, world_space: bool):
        if not self.gradient_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_grad_outer_ref_space(args: self.ElementEvalArg, s: Sample):
            res = utils.generalized_outer(
                self._read_node_value(args, s.element_index, 0),
                self.space.element_outer_weight_gradient(
                    args.elt_arg, args.eval_arg.space_arg, s.element_index, s.element_coords, 0
                ),
            )
            node_count = self.space.topology.element_node_count(
                args.elt_arg, args.eval_arg.topology_arg, s.element_index
            )
            for k in range(1, node_count):
                res += utils.generalized_outer(
                    self._read_node_value(args, s.element_index, k),
                    self.space.element_outer_weight_gradient(
                        args.elt_arg, args.eval_arg.space_arg, s.element_index, s.element_coords, k
                    ),
                )
            return res

        @cache.dynamic_func(suffix=self.name)
        def eval_grad_outer_world_space(args: self.ElementEvalArg, s: Sample):
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            res = eval_grad_outer_ref_space(args, s)
            return utils.apply_right(res, grad_transform)

        return eval_grad_outer_world_space if world_space else eval_grad_outer_ref_space

    def _make_eval_div_outer(self):
        if not self.divergence_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_div_outer(args: self.ElementEvalArg, s: Sample):
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)

            res = utils.generalized_inner(
                self._read_node_value(args, s.element_index, 0),
                utils.apply_right(
                    self.space.element_outer_weight_gradient(
                        args.elt_arg, args.eval_arg.space_arg, s.element_index, s.element_coords, 0
                    ),
                    grad_transform,
                ),
            )

            node_count = self.space.topology.element_node_count(
                args.elt_arg, args.eval_arg.topology_arg, s.element_index
            )
            for k in range(1, node_count):
                res += utils.generalized_inner(
                    self._read_node_value(args, s.element_index, k),
                    utils.apply_right(
                        self.space.element_outer_weight_gradient(
                            args.elt_arg, args.eval_arg.space_arg, s.element_index, s.element_coords, k
                        ),
                        grad_transform,
                    ),
                )
            return res

        return eval_div_outer

    def _make_set_node_value(self):
        @cache.dynamic_func(suffix=self.name)
        def set_node_value(args: self.EvalArg, partition_node_index: int, value: self.space.dtype):
            args.dof_values[partition_node_index] = self.space.dof_mapper.value_to_dof(value)

        return set_node_value

    def _make_node_partition_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_partition_index(args: self.EvalArg, node_index: int):
            return self.space_partition.partition_node_index(args.eval_arg.partition_arg, node_index)

        return node_partition_index


class NodalField(NodalFieldBase):
    """A field holding values for all degrees of freedom at each node of the underlying function space partition

    See also: warp.fem.space.CollocatedFunctionSpace.make_field
    """

    def __init__(self, space: CollocatedFunctionSpace, space_partition: SpacePartition):
        if space.topology != space_partition.space_topology:
            raise ValueError("Incompatible space and space partition topologies")

        super().__init__(space, space_partition)

        self._dof_values = wp.zeros(n=self.space_partition.node_count(), dtype=self.dof_dtype)

    def eval_arg_value(self, device):
        arg = self.EvalArg()
        arg.dof_values = self._dof_values.to(device)
        arg.space_arg = self.space.space_arg_value(device)
        arg.partition_arg = self.space_partition.partition_arg_value(device)
        arg.topology_arg = self.space.topology.topo_arg_value(device)

        return arg

    @property
    def dof_values(self) -> wp.array:
        """Returns a warp array containing the values at all degrees of freedom of the underlying space partition"""
        return self._dof_values

    @dof_values.setter
    def dof_values(self, values):
        """Sets the degrees-of-freedom values

        Args:
            values: Array that is convertible to a warp array of length ``self.space_partition.node_count()`` and data type ``self.space.dof_dtype``
        """

        if isinstance(values, wp.array):
            self._dof_values = values
        else:
            self._dof_values = wp.array(values, dtype=self.dof_dtype)

    class Trace(NodalFieldBase):
        def __init__(self, field):
            self._field = field
            super().__init__(field.space.trace(), field.space_partition)

        def eval_arg_value(self, device):
            arg = self.EvalArg()
            arg.dof_values = self._field.dof_values.to(device)
            arg.space_arg = self.space.space_arg_value(device)
            arg.partition_arg = self.space_partition.partition_arg_value(device)
            arg.topology_arg = self.space.topology.topo_arg_value(device)

            return arg

    def trace(self) -> Trace:
        trace_field = NodalField.Trace(self)
        return trace_field
