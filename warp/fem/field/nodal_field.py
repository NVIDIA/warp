import warp as wp

from warp.fem.space import NodalFunctionSpace, SpacePartition
from warp.fem import cache, utils
from warp.fem.types import Sample, ElementIndex, NULL_NODE_INDEX

from .discrete_field import DiscreteField


class NodalField(DiscreteField):
    def __init__(self, space_partition: SpacePartition, space: NodalFunctionSpace = None):
        if space is None:
            space = space_partition.space

        super().__init__(space, space_partition)

        self._dof_values = wp.zeros(n=self.space_partition.node_count(), dtype=self.dof_dtype)

        self.EvalArg = NodalField._make_eval_arg(self.space, self.space_partition)
        self.eval_degree = DiscreteField._make_eval_degree(self.EvalArg, self.space)
        self.set_node_value = NodalField._make_set_node_value(self.EvalArg, self.space)

        read_node_value = NodalField._make_read_node_value(self.EvalArg, self.space, self.space_partition)

        self.eval_inner = NodalField._make_eval_inner(self.EvalArg, self.space, read_node_value)
        self.eval_outer = NodalField._make_eval_outer(self.EvalArg, self.space, read_node_value)
        self.eval_grad_inner = NodalField._make_eval_grad_inner(self.EvalArg, self.space, read_node_value)
        self.eval_grad_outer = NodalField._make_eval_grad_outer(self.EvalArg, self.space, read_node_value)

    def eval_arg_value(self, device):
        arg = self.EvalArg()
        arg.space_arg = self.space.space_arg_value(device)
        arg.partition_arg = self.space_partition.partition_arg_value(device)
        arg.dof_values = self._dof_values.to(device)

        return arg

    @property
    def dof_values(self):
        return self._dof_values

    @dof_values.setter
    def dof_values(self, values):
        if isinstance(values, wp.array):
            self._dof_values = values
        else:
            self._dof_values = wp.array(values, dtype=self.dof_dtype)

    class Trace(DiscreteField):
        def __init__(self, field):
            self._field = field
            super().__init__(field.space.trace(), field.space_partition)

            self.EvalArg = field.EvalArg
            self.eval_degree = DiscreteField._make_eval_degree(self.EvalArg, self.space)
            self.eval_arg_value = field.eval_arg_value

            self.set_node_value = field.set_node_value

            read_node_value = NodalField._make_read_node_value(self.EvalArg, self.space, self.space_partition)

            self.eval_inner = NodalField._make_eval_inner(self.EvalArg, self.space, read_node_value)
            self.eval_outer = NodalField._make_eval_outer(self.EvalArg, self.space, read_node_value)
            self.eval_grad_inner = NodalField._make_eval_grad_inner(self.EvalArg, self.space, read_node_value)
            self.eval_grad_outer = NodalField._make_eval_grad_outer(self.EvalArg, self.space, read_node_value)

    def trace(self) -> Trace:
        trace_field = NodalField.Trace(self)
        return trace_field

    @staticmethod
    def _make_eval_arg(space: NodalFunctionSpace, space_partition: SpacePartition):
        from warp.fem import cache

        class EvalArg:
            space_arg: space.SpaceArg
            partition_arg: space_partition.PartitionArg
            dof_values: wp.array(dtype=space.dof_dtype)

        EvalArg.__qualname__ = f"{space.name}_{space_partition.name}_FieldEvalArg"
        return cache.get_struct(EvalArg)

    @staticmethod
    def _make_set_node_value(EvalArg, space: NodalFunctionSpace):
        def set_node_value(args: EvalArg, partition_node_index: int, value: space.dtype):
            args.dof_values[partition_node_index] = space.dof_mapper.value_to_dof(value)

        return cache.get_func(set_node_value, space)

    @staticmethod
    def _make_read_node_value(EvalArg, space: NodalFunctionSpace, space_partition: SpacePartition):
        def read_node_value(args: EvalArg, geo_element_index: ElementIndex, node_index_in_elt: int):
            nidx = space.element_node_index(args.space_arg, geo_element_index, node_index_in_elt)
            pidx = space_partition.partition_node_index(args.partition_arg, nidx)
            if pidx == NULL_NODE_INDEX:
                return space.dtype(0.0)

            return space.dof_mapper.dof_to_value(args.dof_values[pidx])

        return cache.get_func(read_node_value, f"{space}_{space_partition}")

    @staticmethod
    def _make_eval_inner(
        EvalArg,
        space: NodalFunctionSpace,
        read_node_value: wp.Function,
    ):
        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        def eval_inner(args: EvalArg, s: Sample):
            res = space.element_inner_weight(args.space_arg, s.element_index, s.element_coords, 0) * read_node_value(
                args, s.element_index, 0
            )
            for k in range(1, NODES_PER_ELEMENT):
                res += space.element_inner_weight(
                    args.space_arg, s.element_index, s.element_coords, k
                ) * read_node_value(args, s.element_index, k)
            return res

        return cache.get_func(eval_inner, read_node_value.key)

    @staticmethod
    def _make_eval_grad_inner(
        EvalArg,
        space: NodalFunctionSpace,
        read_node_value: wp.Function,
    ):
        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        if wp.types.type_is_matrix(space.dtype):
            # There is no Warp high-order tensor type to represent matrix gradients
            return None

        def eval_grad_inner(args: EvalArg, s: Sample):
            res = utils.generalized_outer(
                space.element_inner_weight_gradient(args.space_arg, s.element_index, s.element_coords, 0),
                read_node_value(args, s.element_index, 0),
            )

            for k in range(1, NODES_PER_ELEMENT):
                res += utils.generalized_outer(
                    space.element_inner_weight_gradient(args.space_arg, s.element_index, s.element_coords, k),
                    read_node_value(args, s.element_index, k),
                )
            return res

        return cache.get_func(eval_grad_inner, read_node_value.key)

    @staticmethod
    def _make_eval_outer(
        EvalArg,
        space: NodalFunctionSpace,
        read_node_value: wp.Function,
    ):
        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        def eval_outer(args: EvalArg, s: Sample):
            res = space.element_outer_weight(args.space_arg, s.element_index, s.element_coords, 0) * read_node_value(
                args, s.element_index, 0
            )
            for k in range(1, NODES_PER_ELEMENT):
                res += space.element_outer_weight(
                    args.space_arg, s.element_index, s.element_coords, k
                ) * read_node_value(args, s.element_index, k)
            return res

        return cache.get_func(eval_outer, read_node_value.key)

    @staticmethod
    def _make_eval_grad_outer(
        EvalArg,
        space: NodalFunctionSpace,
        read_node_value: wp.Function,
    ):
        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        if wp.types.type_is_matrix(space.dtype):
            # There is no Warp high-order tensor type to represent matrix gradients
            return None

        def eval_grad_outer(args: EvalArg, s: Sample):
            res = utils.generalized_outer(
                space.element_outer_weight_gradient(args.space_arg, s.element_index, s.element_coords, 0),
                read_node_value(args, s.element_index, 0),
            )
            for k in range(1, NODES_PER_ELEMENT):
                res += utils.generalized_outer(
                    space.element_outer_weight_gradient(args.space_arg, s.element_index, s.element_coords, k),
                    read_node_value(args, s.element_index, k),
                )
            return res

        return cache.get_func(eval_grad_outer, read_node_value.key)
