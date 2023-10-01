import warp as wp
from warp.fem.domain import GeometryDomain
from warp.fem.space import FunctionSpace, SpacePartition
from warp.fem.types import Sample, get_node_index_in_element
from warp.fem import utils, cache


class TrialField:
    """Field defined over a domain that can be used as a trial function"""

    def __init__(
        self,
        space: FunctionSpace,
        space_partition: SpacePartition,
        domain: GeometryDomain,
    ):
        if domain.dimension() == space.DIMENSION - 1:
            space = space.trace()

        if domain.dimension() != space.DIMENSION:
            raise ValueError("Incompatible space and domain dimensions")

        self.space = space
        self.domain = domain
        self.space_partition = space_partition
        self.name = self.space.name + "Trial"

        self.eval_degree = TrialField._make_eval_degree(self.space)
        self.eval_inner = TrialField._make_eval_inner(self.space)
        self.eval_grad_inner = TrialField._make_eval_grad_inner(self.space)
        self.eval_div_inner = TrialField._make_eval_div_inner(self.space)
        self.eval_outer = TrialField._make_eval_outer(self.space)
        self.eval_grad_outer = TrialField._make_eval_grad_outer(self.space)
        self.eval_div_outer = TrialField._make_eval_div_outer(self.space)
        self.at_node = TrialField._make_at_node(self.space)

    def partition_node_count(self) -> int:
        return self.space_partition.node_count()

    @property
    def EvalArg(self) -> wp.codegen.Struct:
        return self.space.SpaceArg

    def eval_arg_value(self, device) -> wp.codegen.StructInstance:
        return self.space.space_arg_value(device)

    @staticmethod
    def _make_eval_degree(space: FunctionSpace):
        ORDER = space.ORDER

        def degree(args: space.SpaceArg):
            return ORDER

        return cache.get_func(degree, space)

    @staticmethod
    def _make_eval_inner(space: FunctionSpace):
        def eval_trial_inner(args: space.SpaceArg, s: Sample):
            weight = space.element_inner_weight(
                args,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            return weight * space.unit_dof_value(args, s.trial_dof)

        return cache.get_func(eval_trial_inner, space.name)

    @staticmethod
    def _make_eval_grad_inner(space: FunctionSpace):
        if wp.types.type_is_matrix(space.dtype):
            # There is no Warp high-order tensor type to represent matrix gradients
            return None

        def eval_nabla_trial_inner(args: space.SpaceArg, s: Sample):
            nabla_weight = space.element_inner_weight_gradient(
                args,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            return utils.generalized_outer(
                space.unit_dof_value(args, s.trial_dof),
                nabla_weight,
            )

        return cache.get_func(eval_nabla_trial_inner, space.name)

    @staticmethod
    def _make_eval_div_inner(space: FunctionSpace):

        def eval_div_trial_inner(args: space.SpaceArg, s: Sample):
            nabla_weight = space.element_inner_weight_gradient(
                args,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            return utils.generalized_inner(
                space.unit_dof_value(args, s.trial_dof),
                nabla_weight,
            )

        return cache.get_func(eval_div_trial_inner, space.name)

    @staticmethod
    def _make_eval_outer(space: FunctionSpace):
        def eval_trial_outer(args: space.SpaceArg, s: Sample):
            weight = space.element_outer_weight(
                args,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            return weight * space.unit_dof_value(args, s.trial_dof)

        return cache.get_func(eval_trial_outer, space.name)

    @staticmethod
    def _make_eval_grad_outer(space: FunctionSpace):
        if wp.types.type_is_matrix(space.dtype):
            # There is no Warp high-order tensor type to represent matrix gradients
            return None

        def eval_nabla_trial_outer(args: space.SpaceArg, s: Sample):
            nabla_weight = space.element_outer_weight_gradient(
                args,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            return utils.generalized_outer(
                space.unit_dof_value(args, s.trial_dof),
                nabla_weight,
            )

        return cache.get_func(eval_nabla_trial_outer, space.name)

    @staticmethod
    def _make_eval_div_outer(space: FunctionSpace):
        if wp.types.type_is_matrix(space.dtype):
            # There is no Warp high-order tensor type to represent matrix gradients
            return None

        def eval_div_trial_outer(args: space.SpaceArg, s: Sample):
            nabla_weight = space.element_outer_weight_gradient(
                args,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            return utils.generalized_inner(
                space.unit_dof_value(args, s.trial_dof),
                nabla_weight,
            )

        return cache.get_func(eval_div_trial_outer, space.name)

    @staticmethod
    def _make_at_node(space: FunctionSpace):
        def at_node(args: space.SpaceArg, s: Sample):
            node_coords = space.node_coords_in_element(args, s.element_index, get_node_index_in_element(s.trial_dof))
            return Sample(s.element_index, node_coords, s.qp_index, s.qp_weight, s.test_dof, s.trial_dof)

        return cache.get_func(at_node, space.name)
