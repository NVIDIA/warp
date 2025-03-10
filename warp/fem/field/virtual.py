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

from typing import Any, Set

import warp as wp
import warp.fem.operator as operator
from warp.fem import cache
from warp.fem.domain import GeometryDomain
from warp.fem.linalg import basis_coefficient, generalized_inner, generalized_outer
from warp.fem.quadrature import Quadrature
from warp.fem.space import FunctionSpace, SpacePartition, SpaceRestriction
from warp.fem.types import NULL_NODE_INDEX, DofIndex, Sample, get_node_coord, get_node_index_in_element

from .field import SpaceField


class AdjointField(SpaceField):
    """Adjoint of a discrete field with respect to its degrees of freedom"""

    def __init__(self, space: FunctionSpace, space_partition: SpaceRestriction):
        super().__init__(space, space_partition=space_partition)

        self.node_dof_count = self.space.NODE_DOF_COUNT
        self.value_dof_count = self.space.VALUE_DOF_COUNT

        self.EvalArg = self.space.SpaceArg
        self.ElementEvalArg = self._make_element_eval_arg()

        self.eval_arg_value = self.space.space_arg_value

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
        return f"{self.__class__.__name__}{self.space.name}{self._space_partition.name}"

    def _make_element_eval_arg(self):
        from warp.fem import cache

        @cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: self.space.topology.ElementArg
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_eval_inner(self):
        @cache.dynamic_func(suffix=self.name)
        def eval_test_inner(args: self.ElementEvalArg, s: Sample):
            dof = self._get_dof(s)
            node_weight = self.space.element_inner_weight(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(dof),
                s.qp_index,
            )
            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.node_basis_element(get_node_coord(dof))
            return self.space.space_value(dof_value, node_weight, local_value_map)

        return eval_test_inner

    def _make_eval_grad_inner(self):
        if not self.space.gradient_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_grad_inner(args: self.ElementEvalArg, s: Sample):
            dof = self._get_dof(s)
            nabla_weight = self.space.element_inner_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(dof),
                s.qp_index,
            )
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.node_basis_element(get_node_coord(dof))
            return self.space.space_gradient(dof_value, nabla_weight, local_value_map, grad_transform)

        return eval_grad_inner

    def _make_eval_div_inner(self):
        if not self.space.divergence_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_div_inner(args: self.ElementEvalArg, s: Sample):
            dof = self._get_dof(s)
            nabla_weight = self.space.element_inner_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(dof),
                s.qp_index,
            )
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.node_basis_element(get_node_coord(dof))
            return self.space.space_divergence(dof_value, nabla_weight, local_value_map, grad_transform)

        return eval_div_inner

    def _make_eval_outer(self):
        @cache.dynamic_func(suffix=self.name)
        def eval_test_outer(args: self.ElementEvalArg, s: Sample):
            dof = self._get_dof(s)
            node_weight = self.space.element_outer_weight(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(dof),
                s.qp_index,
            )
            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.node_basis_element(get_node_coord(dof))
            return self.space.space_value(dof_value, node_weight, local_value_map)

        return eval_test_outer

    def _make_eval_grad_outer(self):
        if not self.space.gradient_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_grad_outer(args: self.ElementEvalArg, s: Sample):
            dof = self._get_dof(s)
            nabla_weight = self.space.element_outer_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(dof),
                s.qp_index,
            )
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.node_basis_element(get_node_coord(dof))
            return self.space.space_gradient(dof_value, nabla_weight, local_value_map, grad_transform)

        return eval_grad_outer

    def _make_eval_div_outer(self):
        if not self.space.divergence_valid():
            return None

        @cache.dynamic_func(suffix=self.name)
        def eval_div_outer(args: self.ElementEvalArg, s: Sample):
            dof = self._get_dof(s)
            nabla_weight = self.space.element_outer_weight_gradient(
                args.elt_arg,
                args.eval_arg,
                s.element_index,
                s.element_coords,
                get_node_index_in_element(dof),
                s.qp_index,
            )
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.node_basis_element(get_node_coord(dof))
            return self.space.space_divergence(dof_value, nabla_weight, local_value_map, grad_transform)

        return eval_div_outer

    def _make_at_node(self):
        @cache.dynamic_func(suffix=self.name)
        def at_node(args: self.ElementEvalArg, s: Sample):
            dof = self._get_dof(s)
            node_coords = self.space.node_coords_in_element(
                args.elt_arg, args.eval_arg, s.element_index, get_node_index_in_element(dof)
            )
            return Sample(s.element_index, node_coords, s.qp_index, s.qp_weight, s.test_dof, s.trial_dof)

        return at_node


class TestField(AdjointField):
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
        self.domain = space_restriction.domain

    @wp.func
    def _get_dof(s: Sample):
        return s.test_dof


class TrialField(AdjointField):
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

    def partition_node_count(self) -> int:
        """Returns the number of nodes in the associated space topology partition"""
        return self.space_partition.node_count()

    @wp.func
    def _get_dof(s: Sample):
        return s.trial_dof


class LocalAdjointField(SpaceField):
    """
    A custom field specially for dispatched assembly.
    Stores adjoint and gradient adjoint at quadrature point locations.
    """

    INNER_DOF = wp.constant(0)
    OUTER_DOF = wp.constant(1)
    INNER_GRAD_DOF = wp.constant(2)
    OUTER_GRAD_DOF = wp.constant(3)
    DOF_TYPE_COUNT = wp.constant(4)

    _OP_DOF_MAP_CONTINUOUS = {
        operator.inner: INNER_DOF,
        operator.outer: INNER_DOF,
        operator.grad: INNER_GRAD_DOF,
        operator.grad_outer: INNER_GRAD_DOF,
        operator.div: INNER_GRAD_DOF,
        operator.div_outer: INNER_GRAD_DOF,
    }

    _OP_DOF_MAP_DISCONTINUOUS = {
        operator.inner: INNER_DOF,
        operator.outer: OUTER_DOF,
        operator.grad: INNER_GRAD_DOF,
        operator.grad_outer: OUTER_GRAD_DOF,
        operator.div: INNER_GRAD_DOF,
        operator.div_outer: OUTER_GRAD_DOF,
    }

    DofOffsets = wp.vec(length=DOF_TYPE_COUNT, dtype=int)

    @wp.struct
    class EvalArg:
        pass

    def __init__(self, field: AdjointField):
        # if not isinstance(field.space, CollocatedFunctionSpace):
        #     raise NotImplementedError("Local assembly only implemented for collocated function spaces")

        super().__init__(field.space, space_partition=field.space_partition)
        self.global_field = field

        self.domain = self.global_field.domain
        self.node_dof_count = self.space.NODE_DOF_COUNT
        self.value_dof_count = self.space.VALUE_DOF_COUNT

        self._dof_suffix = ""

        self.ElementEvalArg = self._make_element_eval_arg()
        self.eval_degree = self._make_eval_degree()
        self.at_node = None

        self._is_discontinuous = (self.space.element_inner_weight != self.space.element_outer_weight) or (
            self.space.element_inner_weight_gradient != self.space.element_outer_weight_gradient
        )

        self._TAYLOR_DOF_OFFSETS = LocalAdjointField.DofOffsets(0)
        self._TAYLOR_DOF_COUNTS = LocalAdjointField.DofOffsets(0)
        self.TAYLOR_DOF_COUNT = 0

    def notify_operator_usage(self, ops: Set[operator.Operator]):
        # Rebuild degrees-of-freedom offsets based on used operators

        operators_dof_map = (
            LocalAdjointField._OP_DOF_MAP_DISCONTINUOUS
            if self._is_discontinuous
            else LocalAdjointField._OP_DOF_MAP_CONTINUOUS
        )

        dof_counts = LocalAdjointField.DofOffsets(0)
        for op in ops:
            if op in operators_dof_map:
                dof_counts[operators_dof_map[op]] = 1

        grad_dim = self.geometry.cell_dimension
        dof_counts[LocalAdjointField.INNER_GRAD_DOF] *= grad_dim
        dof_counts[LocalAdjointField.OUTER_GRAD_DOF] *= grad_dim

        dof_offsets = LocalAdjointField.DofOffsets(0)
        for k in range(1, LocalAdjointField.DOF_TYPE_COUNT):
            dof_offsets[k] = dof_offsets[k - 1] + dof_counts[k - 1]

        self.TAYLOR_DOF_COUNT = wp.constant(dof_offsets[k] + dof_counts[k])

        self._TAYLOR_DOF_OFFSETS = dof_offsets
        self._TAYLOR_DOF_COUNTS = dof_counts

        self._dof_suffix = "".join(str(c) for c in dof_counts)

        self._split_dof = self._make_split_dof()

        self.eval_inner = self._make_eval_inner()
        self.eval_grad_inner = self._make_eval_grad_inner()
        self.eval_div_inner = self._make_eval_div_inner()

        if self._is_discontinuous:
            self.eval_outer = self._make_eval_outer()
            self.eval_grad_outer = self._make_eval_grad_outer()
            self.eval_div_outer = self._make_eval_div_outer()
        else:
            self.eval_outer = self.eval_inner
            self.eval_grad_outer = self.eval_grad_inner
            self.eval_div_outer = self.eval_div_inner

    @property
    def name(self) -> str:
        return f"{self.global_field.name}_Taylor{self._dof_suffix}"

    def eval_arg_value(self, device):
        return LocalAdjointField.EvalArg()

    def _make_element_eval_arg(self):
        from warp.fem import cache

        @cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: self.space.topology.ElementArg
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_split_dof(self):
        TAYLOR_DOF_COUNT = self.TAYLOR_DOF_COUNT

        @cache.dynamic_func(suffix=str(TAYLOR_DOF_COUNT))
        def split_dof(dof_index: DofIndex, dof_begin: int):
            taylor_dof = get_node_index_in_element(dof_index) - dof_begin
            value_dof = get_node_coord(dof_index)
            return value_dof, taylor_dof

        return split_dof

    def _make_eval_inner(self):
        DOF_BEGIN = wp.constant(self._TAYLOR_DOF_OFFSETS[LocalAdjointField.INNER_DOF])

        @cache.dynamic_func(suffix=self.name)
        def eval_test_inner(args: self.ElementEvalArg, s: Sample):
            value_dof, taylor_dof = self._split_dof(self._get_dof(s), DOF_BEGIN)

            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.value_basis_element(value_dof, local_value_map)
            return wp.where(taylor_dof == 0, dof_value, self.dtype(0.0))

        return eval_test_inner

    def _make_eval_grad_inner(self):
        if not self.gradient_valid():
            return None

        DOF_BEGIN = wp.constant(self._TAYLOR_DOF_OFFSETS[LocalAdjointField.INNER_GRAD_DOF])
        DOF_COUNT = wp.constant(self._TAYLOR_DOF_COUNTS[LocalAdjointField.INNER_GRAD_DOF])

        @cache.dynamic_func(suffix=self.name)
        def eval_nabla_test_inner(args: self.ElementEvalArg, s: Sample):
            value_dof, taylor_dof = self._split_dof(self._get_dof(s), DOF_BEGIN)

            if taylor_dof < 0 or taylor_dof >= DOF_COUNT:
                return self.gradient_dtype(0.0)

            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.value_basis_element(value_dof, local_value_map)
            return generalized_outer(dof_value, grad_transform[taylor_dof])

        return eval_nabla_test_inner

    def _make_eval_div_inner(self):
        if not self.divergence_valid():
            return None

        DOF_BEGIN = wp.constant(self._TAYLOR_DOF_OFFSETS[LocalAdjointField.INNER_GRAD_DOF])
        DOF_COUNT = wp.constant(self._TAYLOR_DOF_COUNTS[LocalAdjointField.INNER_GRAD_DOF])

        @cache.dynamic_func(suffix=self.name)
        def eval_div_test_inner(args: self.ElementEvalArg, s: Sample):
            value_dof, taylor_dof = self._split_dof(self._get_dof(s), DOF_BEGIN)

            if taylor_dof < 0 or taylor_dof >= DOF_COUNT:
                return self.divergence_dtype(0.0)

            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.value_basis_element(value_dof, local_value_map)
            return generalized_inner(dof_value, grad_transform[taylor_dof])

        return eval_div_test_inner

    def _make_eval_outer(self):
        DOF_BEGIN = wp.constant(self._TAYLOR_DOF_OFFSETS[LocalAdjointField.OUTER_DOF])

        @cache.dynamic_func(suffix=self.name)
        def eval_test_outer(args: self.ElementEvalArg, s: Sample):
            value_dof, taylor_dof = self._split_dof(self._get_dof(s), DOF_BEGIN)

            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.value_basis_element(value_dof, local_value_map)
            return wp.where(taylor_dof == 0, dof_value, self.dtype(0.0))

        return eval_test_outer

    def _make_eval_grad_outer(self):
        if not self.gradient_valid():
            return None

        DOF_BEGIN = wp.constant(self._TAYLOR_DOF_OFFSETS[LocalAdjointField.OUTER_GRAD_DOF])
        DOF_COUNT = wp.constant(self._TAYLOR_DOF_COUNTS[LocalAdjointField.OUTER_GRAD_DOF])

        @cache.dynamic_func(suffix=self.name)
        def eval_nabla_test_outer(args: self.ElementEvalArg, s: Sample):
            value_dof, taylor_dof = self._split_dof(self._get_dof(s), DOF_BEGIN)

            if taylor_dof < 0 or taylor_dof >= DOF_COUNT:
                return self.gradient_dtype(0.0)

            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.value_basis_element(value_dof, local_value_map)
            return generalized_outer(dof_value, grad_transform[taylor_dof])

        return eval_nabla_test_outer

    def _make_eval_div_outer(self):
        if not self.divergence_valid():
            return None

        DOF_BEGIN = wp.constant(self._TAYLOR_DOF_OFFSETS[LocalAdjointField.OUTER_GRAD_DOF])
        DOF_COUNT = wp.constant(self._TAYLOR_DOF_COUNTS[LocalAdjointField.OUTER_GRAD_DOF])

        @cache.dynamic_func(suffix=self.name)
        def eval_div_test_outer(args: self.ElementEvalArg, s: Sample):
            value_dof, taylor_dof = self._split_dof(self._get_dof(s), DOF_BEGIN)

            if taylor_dof < 0 or taylor_dof >= DOF_COUNT:
                return self.divergence_dtype(0.0)

            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            dof_value = self.space.value_basis_element(value_dof, local_value_map)
            return generalized_inner(dof_value, grad_transform[taylor_dof])

        return eval_div_test_outer


class LocalTestField(LocalAdjointField):
    def __init__(self, test_field: TestField):
        super().__init__(test_field)
        self.space_restriction = test_field.space_restriction

    @wp.func
    def _get_dof(s: Sample):
        return s.test_dof


class LocalTrialField(LocalAdjointField):
    def __init__(self, trial_field: TrialField):
        super().__init__(trial_field)

    @wp.func
    def _get_dof(s: Sample):
        return s.trial_dof


def make_linear_dispatch_kernel(test: LocalTestField, quadrature: Quadrature, accumulate_dtype: type):
    global_test: TestField = test.global_field
    space_restriction = global_test.space_restriction
    domain = global_test.domain

    TEST_INNER_COUNT = test._TAYLOR_DOF_COUNTS[LocalAdjointField.INNER_DOF]
    TEST_OUTER_COUNT = test._TAYLOR_DOF_COUNTS[LocalAdjointField.OUTER_DOF]
    TEST_INNER_GRAD_COUNT = test._TAYLOR_DOF_COUNTS[LocalAdjointField.INNER_GRAD_DOF]
    TEST_OUTER_GRAD_COUNT = test._TAYLOR_DOF_COUNTS[LocalAdjointField.OUTER_GRAD_DOF]

    TEST_INNER_BEGIN = test._TAYLOR_DOF_OFFSETS[LocalAdjointField.INNER_DOF]
    TEST_OUTER_BEGIN = test._TAYLOR_DOF_OFFSETS[LocalAdjointField.OUTER_DOF]
    TEST_INNER_GRAD_BEGIN = test._TAYLOR_DOF_OFFSETS[LocalAdjointField.INNER_GRAD_DOF]
    TEST_OUTER_GRAD_BEGIN = test._TAYLOR_DOF_OFFSETS[LocalAdjointField.OUTER_GRAD_DOF]

    TEST_NODE_DOF_DIM = test.value_dof_count // test.node_dof_count

    @cache.dynamic_kernel(f"{test.name}_{quadrature.name}_{wp.types.get_type_code(accumulate_dtype)}")
    def dispatch_linear_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        test_arg: space_restriction.NodeArg,
        test_space_arg: test.space.SpaceArg,
        local_result: wp.array3d(dtype=Any),
        result: wp.array2d(dtype=Any),
    ):
        local_node_index, test_node_dof = wp.tid()
        node_index = space_restriction.node_partition_index(test_arg, local_node_index)
        element_beg, element_end = space_restriction.node_element_range(test_arg, node_index)

        val_sum = accumulate_dtype(0.0)

        for n in range(element_beg, element_end):
            test_element_index = space_restriction.node_element_index(test_arg, n)
            element_index = domain.element_index(domain_index_arg, test_element_index.domain_element_index)

            qp_point_count = quadrature.point_count(
                domain_arg, qp_arg, test_element_index.domain_element_index, element_index
            )
            for k in range(qp_point_count):
                qp_index = quadrature.point_index(
                    domain_arg, qp_arg, test_element_index.domain_element_index, element_index, k
                )
                qp_eval_index = quadrature.point_evaluation_index(
                    domain_arg, qp_arg, test_element_index.domain_element_index, element_index, k
                )
                coords = quadrature.point_coords(
                    domain_arg, qp_arg, test_element_index.domain_element_index, element_index, k
                )

                qp_result = local_result[qp_eval_index]

                qp_sum = float(0.0)

                if wp.static(0 != TEST_INNER_COUNT):
                    w = test.space.element_inner_weight(
                        domain_arg,
                        test_space_arg,
                        element_index,
                        coords,
                        test_element_index.node_index_in_element,
                        qp_index,
                    )
                    for val_dof in range(TEST_NODE_DOF_DIM):
                        test_dof = test_node_dof * TEST_NODE_DOF_DIM + val_dof
                        qp_sum += basis_coefficient(w, val_dof) * qp_result[TEST_INNER_BEGIN, test_dof]

                if wp.static(0 != TEST_OUTER_COUNT):
                    w = test.space.element_outer_weight(
                        domain_arg,
                        test_space_arg,
                        element_index,
                        coords,
                        test_element_index.node_index_in_element,
                        qp_index,
                    )
                    for val_dof in range(TEST_NODE_DOF_DIM):
                        test_dof = test_node_dof * TEST_NODE_DOF_DIM + val_dof
                        qp_sum += basis_coefficient(w, val_dof) * qp_result[TEST_OUTER_BEGIN, test_dof]

                if wp.static(0 != TEST_INNER_GRAD_COUNT):
                    w_grad = test.space.element_inner_weight_gradient(
                        domain_arg,
                        test_space_arg,
                        element_index,
                        coords,
                        test_element_index.node_index_in_element,
                        qp_index,
                    )
                    for val_dof in range(TEST_NODE_DOF_DIM):
                        test_dof = test_node_dof * TEST_NODE_DOF_DIM + val_dof
                        for grad_dof in range(TEST_INNER_GRAD_COUNT):
                            qp_sum += (
                                basis_coefficient(w_grad, val_dof, grad_dof)
                                * qp_result[grad_dof + TEST_INNER_GRAD_BEGIN, test_dof]
                            )

                if wp.static(0 != TEST_OUTER_GRAD_COUNT):
                    w_grad = test.space.element_outer_weight_gradient(
                        domain_arg,
                        test_space_arg,
                        element_index,
                        coords,
                        test_element_index.node_index_in_element,
                        qp_index,
                    )
                    for val_dof in range(TEST_NODE_DOF_DIM):
                        test_dof = test_node_dof * TEST_NODE_DOF_DIM + val_dof
                        for grad_dof in range(TEST_OUTER_GRAD_COUNT):
                            qp_sum += (
                                basis_coefficient(w_grad, val_dof, grad_dof)
                                * qp_result[grad_dof + TEST_OUTER_GRAD_BEGIN, test_dof]
                            )

                val_sum += accumulate_dtype(qp_sum)

        result[node_index, test_node_dof] += result.dtype(val_sum)

    return dispatch_linear_kernel_fn


def make_bilinear_dispatch_kernel(
    test: LocalTestField, trial: LocalTrialField, quadrature: Quadrature, accumulate_dtype: type
):
    global_test: TestField = test.global_field
    space_restriction = global_test.space_restriction
    domain = global_test.domain

    TEST_INNER_COUNT = test._TAYLOR_DOF_COUNTS[LocalAdjointField.INNER_DOF]
    TEST_OUTER_COUNT = test._TAYLOR_DOF_COUNTS[LocalAdjointField.OUTER_DOF]
    TEST_INNER_GRAD_COUNT = test._TAYLOR_DOF_COUNTS[LocalAdjointField.INNER_GRAD_DOF]
    TEST_OUTER_GRAD_COUNT = test._TAYLOR_DOF_COUNTS[LocalAdjointField.OUTER_GRAD_DOF]

    TEST_INNER_BEGIN = test._TAYLOR_DOF_OFFSETS[LocalAdjointField.INNER_DOF]
    TEST_OUTER_BEGIN = test._TAYLOR_DOF_OFFSETS[LocalAdjointField.OUTER_DOF]
    TEST_INNER_GRAD_BEGIN = test._TAYLOR_DOF_OFFSETS[LocalAdjointField.INNER_GRAD_DOF]
    TEST_OUTER_GRAD_BEGIN = test._TAYLOR_DOF_OFFSETS[LocalAdjointField.OUTER_GRAD_DOF]

    TRIAL_INNER_COUNT = trial._TAYLOR_DOF_COUNTS[LocalAdjointField.INNER_DOF]
    TRIAL_OUTER_COUNT = trial._TAYLOR_DOF_COUNTS[LocalAdjointField.OUTER_DOF]
    TRIAL_INNER_GRAD_COUNT = trial._TAYLOR_DOF_COUNTS[LocalAdjointField.INNER_GRAD_DOF]
    TRIAL_OUTER_GRAD_COUNT = trial._TAYLOR_DOF_COUNTS[LocalAdjointField.OUTER_GRAD_DOF]

    TRIAL_INNER_BEGIN = trial._TAYLOR_DOF_OFFSETS[LocalAdjointField.INNER_DOF]
    TRIAL_OUTER_BEGIN = trial._TAYLOR_DOF_OFFSETS[LocalAdjointField.OUTER_DOF]
    TRIAL_INNER_GRAD_BEGIN = trial._TAYLOR_DOF_OFFSETS[LocalAdjointField.INNER_GRAD_DOF]
    TRIAL_OUTER_GRAD_BEGIN = trial._TAYLOR_DOF_OFFSETS[LocalAdjointField.OUTER_GRAD_DOF]

    TEST_NODE_DOF_DIM = test.value_dof_count // test.node_dof_count
    TRIAL_NODE_DOF_DIM = trial.value_dof_count // trial.node_dof_count

    MAX_NODES_PER_ELEMENT = trial.space.topology.MAX_NODES_PER_ELEMENT

    trial_dof_vec = cache.cached_vec_type(length=trial.TAYLOR_DOF_COUNT, dtype=float)

    @cache.dynamic_kernel(f"{trial.name}_{test.name}_{quadrature.name}{wp.types.get_type_code(accumulate_dtype)}")
    def dispatch_bilinear_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        test_arg: test.space_restriction.NodeArg,
        test_space_arg: test.space.SpaceArg,
        trial_partition_arg: trial.space_partition.PartitionArg,
        trial_topology_arg: trial.space_partition.space_topology.TopologyArg,
        trial_space_arg: trial.space.SpaceArg,
        local_result: wp.array4d(dtype=trial_dof_vec),
        triplet_rows: wp.array(dtype=int),
        triplet_cols: wp.array(dtype=int),
        triplet_values: wp.array3d(dtype=Any),
    ):
        test_local_node_index, test_node_dof, trial_node_dof, trial_node = wp.tid()

        test_node_index = space_restriction.node_partition_index(test_arg, test_local_node_index)
        element_beg, element_end = space_restriction.node_element_range(test_arg, test_node_index)

        for element in range(element_beg, element_end):
            test_element_index = space_restriction.node_element_index(test_arg, element)
            element_index = domain.element_index(domain_index_arg, test_element_index.domain_element_index)
            test_node = test_element_index.node_index_in_element

            element_trial_node_count = trial.space.topology.element_node_count(
                domain_arg, trial_topology_arg, element_index
            )

            qp_point_count = wp.where(
                trial_node < element_trial_node_count,
                quadrature.point_count(domain_arg, qp_arg, test_element_index.domain_element_index, element_index),
                0,
            )

            val_sum = accumulate_dtype(0.0)

            for k in range(qp_point_count):
                qp_index = quadrature.point_index(
                    domain_arg, qp_arg, test_element_index.domain_element_index, element_index, k
                )
                qp_eval_index = quadrature.point_evaluation_index(
                    domain_arg, qp_arg, test_element_index.domain_element_index, element_index, k
                )
                coords = quadrature.point_coords(
                    domain_arg, qp_arg, test_element_index.domain_element_index, element_index, k
                )

                qp_result = local_result[qp_eval_index]
                trial_result = float(0.0)

                if wp.static(0 != TEST_INNER_COUNT):
                    w_test_inner = test.space.element_inner_weight(
                        domain_arg, test_space_arg, element_index, coords, test_node, qp_index
                    )

                if wp.static(0 != TEST_OUTER_COUNT):
                    w_test_outer = test.space.element_outer_weight(
                        domain_arg, test_space_arg, element_index, coords, test_node, qp_index
                    )

                if wp.static(0 != TEST_INNER_GRAD_COUNT):
                    w_test_grad_inner = test.space.element_inner_weight_gradient(
                        domain_arg, test_space_arg, element_index, coords, test_node, qp_index
                    )

                if wp.static(0 != TEST_OUTER_GRAD_COUNT):
                    w_test_grad_outer = test.space.element_outer_weight_gradient(
                        domain_arg, test_space_arg, element_index, coords, test_node, qp_index
                    )

                if wp.static(0 != TRIAL_INNER_COUNT):
                    w_trial_inner = trial.space.element_inner_weight(
                        domain_arg, trial_space_arg, element_index, coords, trial_node, qp_index
                    )

                if wp.static(0 != TRIAL_OUTER_COUNT):
                    w_trial_outer = trial.space.element_outer_weight(
                        domain_arg, trial_space_arg, element_index, coords, trial_node, qp_index
                    )

                if wp.static(0 != TRIAL_INNER_GRAD_COUNT):
                    w_trial_grad_inner = trial.space.element_inner_weight_gradient(
                        domain_arg, trial_space_arg, element_index, coords, trial_node, qp_index
                    )

                if wp.static(0 != TRIAL_OUTER_GRAD_COUNT):
                    w_trial_grad_outer = trial.space.element_outer_weight_gradient(
                        domain_arg, trial_space_arg, element_index, coords, trial_node, qp_index
                    )

                for trial_val_dof in range(TRIAL_NODE_DOF_DIM):
                    trial_dof = trial_node_dof * TRIAL_NODE_DOF_DIM + trial_val_dof
                    test_result = trial_dof_vec(0.0)

                    if wp.static(0 != TEST_INNER_COUNT):
                        for test_val_dof in range(TEST_NODE_DOF_DIM):
                            test_dof = test_node_dof * TEST_NODE_DOF_DIM + test_val_dof
                            test_result += (
                                basis_coefficient(w_test_inner, test_val_dof)
                                * qp_result[test_dof, trial_dof, TEST_INNER_BEGIN]
                            )

                    if wp.static(0 != TEST_OUTER_COUNT):
                        for test_val_dof in range(TEST_NODE_DOF_DIM):
                            test_dof = test_node_dof * TEST_NODE_DOF_DIM + test_val_dof
                            test_result += (
                                basis_coefficient(w_test_outer, test_val_dof)
                                * qp_result[test_dof, trial_dof, TEST_OUTER_BEGIN]
                            )

                    if wp.static(0 != TEST_INNER_GRAD_COUNT):
                        for test_val_dof in range(TEST_NODE_DOF_DIM):
                            test_dof = test_node_dof * TEST_NODE_DOF_DIM + test_val_dof
                            for grad_dof in range(TEST_INNER_GRAD_COUNT):
                                test_result += (
                                    basis_coefficient(w_test_grad_inner, test_val_dof, grad_dof)
                                    * qp_result[test_dof, trial_dof, grad_dof + TEST_INNER_GRAD_BEGIN]
                                )

                    if wp.static(0 != TEST_OUTER_GRAD_COUNT):
                        for test_val_dof in range(TEST_NODE_DOF_DIM):
                            test_dof = test_node_dof * TEST_NODE_DOF_DIM + test_val_dof
                            for grad_dof in range(TEST_OUTER_GRAD_COUNT):
                                test_result += (
                                    basis_coefficient(w_test_grad_outer, test_val_dof, grad_dof)
                                    * qp_result[test_dof, trial_dof, grad_dof + TEST_OUTER_GRAD_BEGIN]
                                )

                    if wp.static(0 != TRIAL_INNER_COUNT):
                        trial_result += basis_coefficient(w_trial_inner, trial_val_dof) * test_result[TRIAL_INNER_BEGIN]

                    if wp.static(0 != TRIAL_OUTER_COUNT):
                        trial_result += basis_coefficient(w_trial_outer, trial_val_dof) * test_result[TRIAL_OUTER_BEGIN]

                    if wp.static(0 != TRIAL_INNER_GRAD_COUNT):
                        for grad_dof in range(TRIAL_INNER_GRAD_COUNT):
                            trial_result += (
                                basis_coefficient(w_trial_grad_inner, trial_val_dof, grad_dof)
                                * test_result[grad_dof + TRIAL_INNER_GRAD_BEGIN]
                            )

                    if wp.static(0 != TRIAL_OUTER_GRAD_COUNT):
                        for grad_dof in range(TRIAL_OUTER_GRAD_COUNT):
                            trial_result += (
                                basis_coefficient(w_trial_grad_outer, trial_val_dof, grad_dof)
                                * test_result[grad_dof + TRIAL_OUTER_GRAD_BEGIN]
                            )

                val_sum += accumulate_dtype(trial_result)

            block_offset = element * MAX_NODES_PER_ELEMENT + trial_node
            triplet_values[block_offset, test_node_dof, trial_node_dof] = triplet_values.dtype(val_sum)

            # Set row and column indices
            if test_node_dof == 0 and trial_node_dof == 0:
                if trial_node < element_trial_node_count:
                    trial_node_index = trial.space_partition.partition_node_index(
                        trial_partition_arg,
                        trial.space.topology.element_node_index(
                            domain_arg, trial_topology_arg, element_index, trial_node
                        ),
                    )
                else:
                    trial_node_index = NULL_NODE_INDEX  # will get ignored when converting to bsr

                triplet_rows[block_offset] = test_node_index
                triplet_cols[block_offset] = trial_node_index

    return dispatch_bilinear_kernel_fn
