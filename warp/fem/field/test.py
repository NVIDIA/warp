import warp as wp

from warp.fem.space import SpaceRestriction, FunctionSpace
from warp.fem.types import Sample, get_node_index_in_element
from warp.fem import utils, cache


class TestField:
    """Field defined over a space restriction that can be used as a test function"""

    def __init__(self, space_restriction: SpaceRestriction):
        self.space_restriction = space_restriction
        self.space_partition = self.space_restriction.space_partition
        self.space = self.space_restriction.space
        self.domain = self.space_restriction.domain
        self.name = self.space.name + "Test"

        self.eval_degree = TestField._make_eval_degree(self.space)
        self.eval_inner = TestField._make_eval_inner(self.space)
        self.eval_grad_inner = TestField._make_eval_grad_inner(self.space)
        self.eval_outer = TestField._make_eval_outer(self.space)
        self.eval_grad_outer = TestField._make_eval_grad_outer(self.space)

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
        def eval_test(args: space.SpaceArg, s: Sample):
            weight = space.element_inner_weight(
                args,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.test_dof),
            )
            return weight * space.unit_dof_value(args, s.test_dof)

        return cache.get_func(eval_test, space.name)

    @staticmethod
    def _make_eval_grad_inner(space: FunctionSpace):
        def eval_nabla_test_inner(args: space.SpaceArg, s: Sample):

            nabla_weight = space.element_inner_weight_gradient(
                args,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.test_dof),
            )
            return utils.generalized_outer(
                nabla_weight,
                space.unit_dof_value(args, s.test_dof),
            )

        return cache.get_func(eval_nabla_test_inner, space.name)

    @staticmethod
    def _make_eval_outer(space: FunctionSpace):
        def eval_test_outer(args: space.SpaceArg, s: Sample):

            weight = space.element_outer_weight(
                args,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.test_dof),
            )
            return weight * space.unit_dof_value(args, s.test_dof)

        return cache.get_func(eval_test_outer, space.name)

    @staticmethod
    def _make_eval_grad_outer(space: FunctionSpace):
        def eval_nabla_test(args: space.SpaceArg, s: Sample):

            nabla_weight = space.element_outer_weight_gradient(
                args,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.test_dof),
            )
            return utils.generalized_outer(
                nabla_weight,
                space.unit_dof_value(args, s.test_dof),
            )

        return cache.get_func(eval_nabla_test, space.name)
