import warp as wp

from warp.fem.space import SpaceRestriction, FunctionSpace
from warp.fem.types import Sample, get_node_index_in_element
from warp.fem import utils, cache

from .field import SpaceField


class TestField(SpaceField):
    """Field defined over a space restriction that can be used as a test function.

    In order to reuse computations, it is possible to define the test field using a SpaceRestriction
    defined for a different value type than the test function value type, as long as the node topology is similar.
    """

    def __init__(self, space_restriction: SpaceRestriction, space: FunctionSpace):
        if space_restriction.domain.dimension == space.dimension - 1:
            space = space.trace()

        if space_restriction.domain.dimension != space.dimension:
            raise ValueError("Incompatible space and domain dimensions")

        if space.topology != space_restriction.space_topology:
            raise ValueError("Incompatible space and space partition topologies")

        super().__init__(space, space_restriction.space_partition)

        self.space_restriction = space_restriction
        self.domain = self.space_restriction.domain

        self.EvalArg = self.space.SpaceArg
        self.ElementEvalArg = self._make_element_eval_arg()

        self.eval_degree = self._make_eval_degree()
        self.eval_inner = self._make_eval_inner()
        self.eval_grad_inner = self._make_eval_grad_inner()
        self.eval_div_inner = self._make_eval_div_inner()
        self.eval_outer = self._make_eval_outer()
        self.eval_grad_outer = self._make_eval_grad_outer()
        self.eval_div_outer = self._make_eval_div_outer()
        self.at_node = self._make_at_node()

    @property
    def name(self) -> str:
        return self.space.name + "Test"

    def eval_arg_value(self, device) -> wp.codegen.StructInstance:
        return self.space.space_arg_value(device)

    def _make_element_eval_arg(self):
        from warp.fem import cache

        @cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: self.domain.ElementArg
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_eval_inner(self):
        @cache.dynamic_func(suffix=self.name)
        def eval_test_inner(args: self.ElementEvalArg, s: Sample):
            weight = self.space.element_inner_weight(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.test_dof),
            )
            return weight * self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.test_dof)

        return eval_test_inner

    def _make_eval_grad_inner(self):
        if not self.gradient_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_nabla_test_inner(args: self.ElementEvalArg, s: Sample):
            nabla_weight = self.space.element_inner_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.test_dof),
            )
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            return utils.generalized_outer(
                self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.test_dof),
                utils.apply_right(nabla_weight, grad_transform),
            )

        return eval_nabla_test_inner

    def _make_eval_div_inner(self):
        if not self.divergence_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_div_test_inner(args: self.ElementEvalArg, s: Sample):
            nabla_weight = self.space.element_inner_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.test_dof),
            )
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            return utils.generalized_inner(
                self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.test_dof),
                utils.apply_right(nabla_weight, grad_transform),
            )

        return eval_div_test_inner

    def _make_eval_outer(self):
        @cache.dynamic_func(suffix=self.name)
        def eval_test_outer(args: self.ElementEvalArg, s: Sample):
            weight = self.space.element_outer_weight(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.test_dof),
            )
            return weight * self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.test_dof)

        return eval_test_outer

    def _make_eval_grad_outer(self):
        if not self.gradient_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_nabla_test_outer(args: self.ElementEvalArg, s: Sample):
            nabla_weight = self.space.element_outer_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.test_dof),
            )
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            return utils.generalized_outer(
                self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.test_dof),
                utils.apply_right(nabla_weight, grad_transform),
            )

        return eval_nabla_test_outer

    def _make_eval_div_outer(self):
        if not self.divergence_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_div_test_outer(args: self.ElementEvalArg, s: Sample):
            nabla_weight = self.space.element_outer_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.test_dof),
            )
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            return utils.generalized_inner(
                self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.test_dof),
                utils.apply_right(nabla_weight, grad_transform),
            )

        return eval_div_test_outer

    def _make_at_node(self):
        @cache.dynamic_func(suffix=self.name)
        def at_node(args: self.ElementEvalArg, s: Sample):
            node_coords = self.space.node_coords_in_element(
                args.elt_arg, args.eval_arg, s.element_index, get_node_index_in_element(s.test_dof)
            )
            return Sample(s.element_index, node_coords, s.qp_index, s.qp_weight, s.test_dof, s.trial_dof)

        return at_node
