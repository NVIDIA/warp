import warp as wp
from warp.fem.domain import GeometryDomain
from warp.fem.space import FunctionSpace, SpacePartition
from warp.fem.types import Sample, get_node_index_in_element
from warp.fem import utils, cache

from .field import SpaceField


class TrialField(SpaceField):
    """Field defined over a domain that can be used as a trial function"""

    def __init__(
        self,
        space: FunctionSpace,
        space_partition: SpacePartition,
        domain: GeometryDomain,
    ):
        if domain.dimension == space.dimension - 1:
            space = space.trace()

        if domain.dimension != space.dimension:
            raise ValueError("Incompatible space and domain dimensions")

        if not space.topology.is_derived_from(space_partition.space_topology):
            raise ValueError("Incompatible space and space partition topologies")

        super().__init__(space, space_partition)

        self.domain = domain

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

    def partition_node_count(self) -> int:
        """Returns the number of nodes in the associated space topology partition"""
        return self.space_partition.node_count()

    @property
    def name(self) -> str:
        return self.space.name + "Trial"

    def eval_arg_value(self, device) -> wp.codegen.StructInstance:
        return self.space.space_arg_value(device)

    def _make_element_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: self.domain.ElementArg
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_eval_inner(self):
        @cache.dynamic_func(suffix=self.name)
        def eval_trial_inner(args: self.ElementEvalArg, s: Sample):
            weight = self.space.element_inner_weight(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            return weight * self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.trial_dof)

        return eval_trial_inner

    def _make_eval_grad_inner(self):
        if not self.gradient_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_nabla_trial_inner(args: self.ElementEvalArg, s: Sample):
            nabla_weight = self.space.element_inner_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            return utils.generalized_outer(
                self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.trial_dof),
                utils.apply_right(nabla_weight, grad_transform),
            )

        return eval_nabla_trial_inner

    def _make_eval_div_inner(self):
        if not self.divergence_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_div_trial_inner(args: self.ElementEvalArg, s: Sample):
            nabla_weight = self.space.element_inner_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            return utils.generalized_inner(
                self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.trial_dof),
                utils.apply_right(nabla_weight, grad_transform),
            )

        return eval_div_trial_inner

    def _make_eval_outer(self):
        @cache.dynamic_func(suffix=self.name)
        def eval_trial_outer(args: self.ElementEvalArg, s: Sample):
            weight = self.space.element_outer_weight(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            return weight * self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.trial_dof)

        return eval_trial_outer

    def _make_eval_grad_outer(self):
        if not self.gradient_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_nabla_trial_outer(args: self.ElementEvalArg, s: Sample):
            nabla_weight = self.space.element_outer_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            return utils.generalized_outer(
                self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.trial_dof),
                utils.apply_right(nabla_weight, grad_transform),
            )

        return eval_nabla_trial_outer

    def _make_eval_div_outer(self):
        if not self.divergence_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_div_trial_outer(args: self.ElementEvalArg, s: Sample):
            nabla_weight = self.space.element_outer_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(s.trial_dof),
            )
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            return utils.generalized_inner(
                self.space.unit_dof_value(args.elt_arg, args.eval_arg, s.trial_dof),
                utils.apply_right(nabla_weight, grad_transform),
            )

        return eval_div_trial_outer

    def _make_at_node(self):
        @cache.dynamic_func(suffix=self.name)
        def at_node(args: self.ElementEvalArg, s: Sample):
            node_coords = self.space.node_coords_in_element(
                args.elt_arg, args.eval_arg, s.element_index, get_node_index_in_element(s.trial_dof)
            )
            return Sample(s.element_index, node_coords, s.qp_index, s.qp_weight, s.test_dof, s.trial_dof)

        return at_node
