# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, ClassVar

import warp as wp
from warp._src.fem import cache
from warp._src.fem.space import FunctionSpace, SpacePartition
from warp._src.fem.types import NULL_NODE_INDEX, ElementIndex, Sample
from warp._src.fem.utils import type_zero_element

from .field import DiscreteField

_wp_module_name_ = "warp.fem.field.nodal_field"


class NodalFieldBase(DiscreteField):
    """Base class for nodal field and nodal field traces. Does not hold values"""

    _dynamic_attribute_constructors: ClassVar = {
        "EvalArg": lambda obj: obj._make_eval_arg(),
        "ElementEvalArg": lambda obj: obj._make_element_eval_arg(),
        "eval_degree": lambda obj: DiscreteField._make_eval_degree(obj),
        "_read_node_value": lambda obj: obj._make_read_node_value(),
        "eval_inner": lambda obj: obj._make_eval_inner(),
        "eval_outer": lambda obj: obj._make_eval_outer(),
        "eval_grad_inner": lambda obj: obj._make_eval_grad_inner(world_space=True),
        "eval_grad_outer": lambda obj: obj._make_eval_grad_outer(world_space=True),
        "eval_reference_grad_inner": lambda obj: obj._make_eval_grad_inner(world_space=False),
        "eval_reference_grad_outer": lambda obj: obj._make_eval_grad_outer(world_space=False),
        "eval_div_inner": lambda obj: obj._make_eval_div_inner(),
        "eval_div_outer": lambda obj: obj._make_eval_div_outer(),
        "set_node_value": lambda obj: obj._make_set_node_value(),
        "node_partition_index": lambda obj: obj._make_node_partition_index(),
        "node_count": lambda obj: obj._make_node_count(),
        "node_index": lambda obj: obj._make_node_index(),
        "at_node": lambda obj: obj._make_at_node(),
        "node_inner_weight": lambda obj: obj._make_node_inner_weight(),
        "node_outer_weight": lambda obj: obj._make_node_outer_weight(),
        "node_inner_weight_gradient": lambda obj: obj._make_node_inner_weight_gradient(),
        "node_outer_weight_gradient": lambda obj: obj._make_node_outer_weight_gradient(),
    }

    def __init__(self, space: FunctionSpace, space_partition: SpacePartition):
        super().__init__(space, space_partition)
        cache.setup_dynamic_attributes(self)

    def _make_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class EvalArg:
            dof_values: wp.array(dtype=self.space.dof_dtype)
            basis_arg: self.space.basis.BasisArg
            partition_arg: self.space_partition.PartitionArg
            topo_arg: self.space.topology.TopologyArg

        return EvalArg

    def _make_element_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: self.space.topology.ElementArg
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_read_node_value(self):
        zero_element = type_zero_element(self.dof_dtype)

        @cache.dynamic_func(suffix=self.name)
        def read_node_value(args: self.ElementEvalArg, geo_element_index: ElementIndex, node_index_in_elt: int):
            topo_arg = args.eval_arg.topo_arg
            nidx = self.space.topology.element_node_index(args.elt_arg, topo_arg, geo_element_index, node_index_in_elt)
            pidx = self.space_partition.partition_node_index(args.eval_arg.partition_arg, nidx)
            if pidx == NULL_NODE_INDEX:
                return zero_element()

            return args.eval_arg.dof_values[pidx]

        return read_node_value

    def _make_eval_inner(self):
        zero_element = type_zero_element(self.dtype)

        @cache.dynamic_func(suffix=self.name, allow_overloads=True)
        def eval_inner(args: self.ElementEvalArg, s: Sample):
            topo_arg = args.eval_arg.topo_arg
            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            node_count = self.space.topology.element_node_count(args.elt_arg, topo_arg, s.element_index)
            res = zero_element()
            for k in range(node_count):
                w = self.space.basis.element_inner_weight(
                    args.elt_arg, topo_arg, args.eval_arg.basis_arg, s.element_index, s.element_coords, k, s.qp_index
                )
                res += self.space.space_value(
                    self._read_node_value(args, s.element_index, k),
                    w,
                    local_value_map,
                )
            return res

        @cache.dynamic_func(suffix=self.name, allow_overloads=True)
        def eval_inner(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            w = self.space.basis.element_inner_weight(
                args.elt_arg,
                args.eval_arg.topo_arg,
                args.eval_arg.basis_arg,
                s.element_index,
                s.element_coords,
                node_index_in_elt,
                s.qp_index,
            )
            return self.space.space_value(
                self._read_node_value(args, s.element_index, node_index_in_elt),
                w,
                local_value_map,
            )

        return eval_inner

    def _make_eval_grad_inner(self, world_space: bool):
        if not self.space.gradient_valid():
            return None

        gradient_dtype = self.gradient_dtype if world_space else self.reference_gradient_dtype
        zero_element = type_zero_element(gradient_dtype)

        @cache.dynamic_func(suffix=f"{self.name}{world_space}")
        def eval_grad_inner(args: self.ElementEvalArg, s: Sample, grad_transform: Any):
            topo_arg = args.eval_arg.topo_arg
            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            node_count = self.space.topology.element_node_count(args.elt_arg, topo_arg, s.element_index)

            res = zero_element()
            for k in range(node_count):
                res += self.space.space_gradient(
                    self._read_node_value(args, s.element_index, k),
                    self.space.basis.element_inner_weight_gradient(
                        args.elt_arg,
                        topo_arg,
                        args.eval_arg.basis_arg,
                        s.element_index,
                        s.element_coords,
                        k,
                        s.qp_index,
                    )
                    * grad_transform,
                    local_value_map,
                )
            return res

        if world_space:

            @cache.dynamic_func(suffix=self.name, allow_overloads=True)
            def eval_grad_inner_world_space(args: self.ElementEvalArg, s: Sample):
                grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
                return eval_grad_inner(args, s, grad_transform)

            @cache.dynamic_func(suffix=self.name, allow_overloads=True)
            def eval_grad_inner_world_space(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
                local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
                grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
                weight_gradient = self.space.basis.element_inner_weight_gradient(
                    args.elt_arg,
                    args.eval_arg.topo_arg,
                    args.eval_arg.basis_arg,
                    s.element_index,
                    s.element_coords,
                    node_index_in_elt,
                    s.qp_index,
                )
                return self.space.space_gradient(
                    self._read_node_value(args, s.element_index, node_index_in_elt),
                    weight_gradient * grad_transform,
                    local_value_map,
                )

            return eval_grad_inner_world_space
        else:

            @cache.dynamic_func(suffix=self.name, allow_overloads=True)
            def eval_grad_inner_ref_space(args: self.ElementEvalArg, s: Sample):
                grad_transform = 1.0
                return eval_grad_inner(args, s, grad_transform)

            return eval_grad_inner_ref_space

    def _make_eval_div_inner(self):
        if not self.divergence_valid():
            return None
        zero_element = type_zero_element(self.divergence_dtype)

        @cache.dynamic_func(suffix=self.name, allow_overloads=True)
        def eval_div_inner(args: self.ElementEvalArg, s: Sample):
            topo_arg = args.eval_arg.topo_arg
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            node_count = self.space.topology.element_node_count(args.elt_arg, topo_arg, s.element_index)

            res = zero_element()
            for k in range(node_count):
                res += self.space.space_divergence(
                    self._read_node_value(args, s.element_index, k),
                    self.space.basis.element_inner_weight_gradient(
                        args.elt_arg,
                        topo_arg,
                        args.eval_arg.basis_arg,
                        s.element_index,
                        s.element_coords,
                        k,
                        s.qp_index,
                    )
                    * grad_transform,
                    local_value_map,
                )
            return res

        @cache.dynamic_func(suffix=self.name, allow_overloads=True)
        def eval_div_inner(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_inner(args.elt_arg, s.element_index, s.element_coords)
            weight_gradient = self.space.basis.element_inner_weight_gradient(
                args.elt_arg,
                args.eval_arg.topo_arg,
                args.eval_arg.basis_arg,
                s.element_index,
                s.element_coords,
                node_index_in_elt,
                s.qp_index,
            )
            return self.space.space_divergence(
                self._read_node_value(args, s.element_index, node_index_in_elt),
                weight_gradient * grad_transform,
                local_value_map,
            )

        return eval_div_inner

    def _make_eval_outer(self):
        zero_element = type_zero_element(self.dtype)

        @cache.dynamic_func(suffix=self.name, allow_overloads=True)
        def eval_outer(args: self.ElementEvalArg, s: Sample):
            topo_arg = args.eval_arg.topo_arg
            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            node_count = self.space.topology.element_node_count(args.elt_arg, topo_arg, s.element_index)

            res = zero_element()
            for k in range(node_count):
                res += self.space.space_value(
                    self._read_node_value(args, s.element_index, k),
                    self.space.basis.element_outer_weight(
                        args.elt_arg,
                        topo_arg,
                        args.eval_arg.basis_arg,
                        s.element_index,
                        s.element_coords,
                        k,
                        s.qp_index,
                    ),
                    local_value_map,
                )
            return res

        @cache.dynamic_func(suffix=self.name, allow_overloads=True)
        def eval_outer(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            w = self.space.basis.element_outer_weight(
                args.elt_arg,
                args.eval_arg.topo_arg,
                args.eval_arg.basis_arg,
                s.element_index,
                s.element_coords,
                node_index_in_elt,
                s.qp_index,
            )
            return self.space.space_value(
                self._read_node_value(args, s.element_index, node_index_in_elt),
                w,
                local_value_map,
            )

        return eval_outer

    def _make_eval_grad_outer(self, world_space: bool):
        if not self.space.gradient_valid():
            return None

        gradient_dtype = self.gradient_dtype if world_space else self.reference_gradient_dtype
        zero_element = type_zero_element(gradient_dtype)

        @cache.dynamic_func(suffix=f"{self.name}{world_space}")
        def eval_grad_outer(args: self.ElementEvalArg, s: Sample, grad_transform: Any):
            topo_arg = args.eval_arg.topo_arg
            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            node_count = self.space.topology.element_node_count(args.elt_arg, topo_arg, s.element_index)

            res = zero_element()
            for k in range(node_count):
                res += self.space.space_gradient(
                    self._read_node_value(args, s.element_index, k),
                    self.space.basis.element_outer_weight_gradient(
                        args.elt_arg,
                        topo_arg,
                        args.eval_arg.basis_arg,
                        s.element_index,
                        s.element_coords,
                        k,
                        s.qp_index,
                    )
                    * grad_transform,
                    local_value_map,
                )
            return res

        if world_space:

            @cache.dynamic_func(suffix=self.name, allow_overloads=True)
            def eval_grad_outer_world_space(args: self.ElementEvalArg, s: Sample):
                grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
                return eval_grad_outer(args, s, grad_transform)

            @cache.dynamic_func(suffix=self.name, allow_overloads=True)
            def eval_grad_outer_world_space(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
                local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
                grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
                weight_gradient = self.space.basis.element_outer_weight_gradient(
                    args.elt_arg,
                    args.eval_arg.topo_arg,
                    args.eval_arg.basis_arg,
                    s.element_index,
                    s.element_coords,
                    node_index_in_elt,
                    s.qp_index,
                )
                return self.space.space_gradient(
                    self._read_node_value(args, s.element_index, node_index_in_elt),
                    weight_gradient * grad_transform,
                    local_value_map,
                )

            return eval_grad_outer_world_space
        else:

            @cache.dynamic_func(suffix=self.name, allow_overloads=True)
            def eval_grad_outer_ref_space(args: self.ElementEvalArg, s: Sample):
                grad_transform = 1.0
                return eval_grad_outer(args, s, grad_transform)

            return eval_grad_outer_ref_space

    def _make_eval_div_outer(self):
        if not self.divergence_valid():
            return None

        zero_element = type_zero_element(self.divergence_dtype)

        @cache.dynamic_func(suffix=self.name, allow_overloads=True)
        def eval_div_outer(args: self.ElementEvalArg, s: Sample):
            topo_arg = args.eval_arg.topo_arg
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            node_count = self.space.topology.element_node_count(args.elt_arg, topo_arg, s.element_index)

            res = zero_element()
            for k in range(node_count):
                res += self.space.space_divergence(
                    self._read_node_value(args, s.element_index, k),
                    self.space.basis.element_outer_weight_gradient(
                        args.elt_arg,
                        topo_arg,
                        args.eval_arg.basis_arg,
                        s.element_index,
                        s.element_coords,
                        k,
                        s.qp_index,
                    )
                    * grad_transform,
                    local_value_map,
                )
            return res

        @cache.dynamic_func(suffix=self.name, allow_overloads=True)
        def eval_div_outer(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            local_value_map = self.space.local_value_map_outer(args.elt_arg, s.element_index, s.element_coords)
            weight_gradient = self.space.basis.element_outer_weight_gradient(
                args.elt_arg,
                args.eval_arg.topo_arg,
                args.eval_arg.basis_arg,
                s.element_index,
                s.element_coords,
                node_index_in_elt,
                s.qp_index,
            )
            return self.space.space_divergence(
                self._read_node_value(args, s.element_index, node_index_in_elt),
                weight_gradient * grad_transform,
                local_value_map,
            )

        return eval_div_outer

    def _make_set_node_value(self):
        @cache.dynamic_func(suffix=self.name)
        def set_node_value(
            eval_arg: self.EvalArg,
            partition_node_index: ElementIndex,
            dof_value: self.space.dof_dtype,
        ):
            eval_arg.dof_values[partition_node_index] = dof_value

        return set_node_value

    def _make_node_partition_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_partition_index(args: self.ElementEvalArg, node_index: int):
            return self.space_partition.partition_node_index(args.eval_arg.partition_arg, node_index)

        return node_partition_index

    def _make_node_count(self):
        @cache.dynamic_func(suffix=self.name)
        def node_count(args: self.ElementEvalArg, s: Sample):
            topo_arg = args.eval_arg.topo_arg
            return self.space.topology.element_node_count(args.elt_arg, topo_arg, s.element_index)

        return node_count

    def _make_at_node(self):
        @cache.dynamic_func(suffix=self.name)
        def at_node(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
            node_coords = self.space.basis.node_coords_in_element(
                args.elt_arg, args.eval_arg.topo_arg, args.eval_arg.basis_arg, s.element_index, node_index_in_elt
            )
            return Sample(s.element_index, node_coords, s.qp_index, s.qp_weight, s.test_dof, s.trial_dof)

        return at_node

    def _make_node_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_index(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
            topo_arg = args.eval_arg.topo_arg
            return self.space.topology.element_node_index(args.elt_arg, topo_arg, s.element_index, node_index_in_elt)

        return node_index

    def _make_node_inner_weight(self):
        @cache.dynamic_func(suffix=self.name)
        def node_inner_weight(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
            return self.space.basis.element_inner_weight(
                args.elt_arg,
                args.eval_arg.topo_arg,
                args.eval_arg.basis_arg,
                s.element_index,
                s.element_coords,
                node_index_in_elt,
                s.qp_index,
            )

        return node_inner_weight

    def _make_node_outer_weight(self):
        @cache.dynamic_func(suffix=self.name)
        def node_outer_weight(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
            return self.space.basis.element_outer_weight(
                args.elt_arg,
                args.eval_arg.topo_arg,
                args.eval_arg.basis_arg,
                s.element_index,
                s.element_coords,
                node_index_in_elt,
                s.qp_index,
            )

        return node_outer_weight

    def _make_node_inner_weight_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def node_inner_weight_gradient(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
            node_weight_gradient = self.space.basis.element_inner_weight_gradient(
                args.elt_arg,
                args.eval_arg.topo_arg,
                args.eval_arg.basis_arg,
                s.element_index,
                s.element_coords,
                node_index_in_elt,
                s.qp_index,
            )
            grad_transform = self.space.element_inner_reference_gradient_transform(args.elt_arg, s)
            return node_weight_gradient * grad_transform

        return node_inner_weight_gradient

    def _make_node_outer_weight_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def node_outer_weight_gradient(args: self.ElementEvalArg, s: Sample, node_index_in_elt: int):
            node_weight_gradient = self.space.basis.element_outer_weight_gradient(
                args.elt_arg,
                args.eval_arg.topo_arg,
                args.eval_arg.basis_arg,
                s.element_index,
                s.element_coords,
                node_index_in_elt,
                s.qp_index,
            )
            grad_transform = self.space.element_outer_reference_gradient_transform(args.elt_arg, s)
            return node_weight_gradient * grad_transform

        return node_outer_weight_gradient


class NodalField(NodalFieldBase):
    """A field holding values for all degrees of freedom at each node of the underlying function space partition

    See also: warp.fem.space.FunctionSpace.make_field
    """

    def __init__(self, space: FunctionSpace, space_partition: SpacePartition):
        if space.topology != space_partition.space_topology:
            raise ValueError("Incompatible space and space partition topologies")

        super().__init__(space, space_partition)

        self._dof_values = wp.zeros(n=self.space_partition.node_count(), dtype=self.dof_dtype)

    @cache.cached_arg_value
    def eval_arg_value(self, device):
        return super().eval_arg_value(device)

    def fill_eval_arg(self, arg, device):
        arg.dof_values = self._dof_values.to(device)
        self.space.basis.fill_basis_arg(arg.basis_arg, device)
        self.space_partition.fill_partition_arg(arg.partition_arg, device)
        self.space.topology.fill_topo_arg(arg.topo_arg, device)

    def rebind(self, space: FunctionSpace, space_partition: SpacePartition):
        """Rebind the field to a new space partition and space.
        The new space partition and space must be of similar types as the current ones
        """

        if space_partition.name != self.space_partition.name or space.name != self.space.name:
            raise ValueError("Incompatible space and/or space partition")

        self._space = space
        self._space_partition = space_partition

        node_count = space_partition.node_count()
        if node_count < self._dof_values.shape[0]:
            self._dof_values = self._dof_values[:node_count]
        elif node_count > self._dof_values.shape[0]:
            self._dof_values = wp.empty(n=node_count, dtype=self.dof_dtype)

        self.eval_arg_value.invalidate(self)

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

        self.eval_arg_value.invalidate(self)

    class Trace(NodalFieldBase):
        def __init__(self, field):
            self._field = field
            super().__init__(field.space.trace(), field.space_partition)

        def fill_eval_arg(self, arg, device):
            arg.dof_values = self._field.dof_values.to(device)
            self.space.basis.fill_basis_arg(arg.basis_arg, device)
            self.space.topology.fill_topo_arg(arg.topo_arg, device)
            self.space_partition.fill_partition_arg(arg.partition_arg, device)

    def trace(self) -> Trace:
        trace_field = NodalField.Trace(self)
        return trace_field
