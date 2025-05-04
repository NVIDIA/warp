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

import ast
import inspect
import textwrap
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union

import warp as wp
import warp.fem.operator as operator
from warp.codegen import get_annotations
from warp.fem import cache
from warp.fem.domain import GeometryDomain
from warp.fem.field import (
    DiscreteField,
    FieldLike,
    FieldRestriction,
    GeometryField,
    LocalTestField,
    LocalTrialField,
    TestField,
    TrialField,
    make_restriction,
)
from warp.fem.field.virtual import make_bilinear_dispatch_kernel, make_linear_dispatch_kernel
from warp.fem.linalg import array_axpy, basis_coefficient
from warp.fem.operator import (
    Integrand,
    Operator,
    integrand,
)
from warp.fem.quadrature import Quadrature, RegularQuadrature
from warp.fem.types import (
    NULL_DOF_INDEX,
    NULL_ELEMENT_INDEX,
    NULL_NODE_INDEX,
    OUTSIDE,
    Coords,
    DofIndex,
    Domain,
    Field,
    Sample,
    make_free_sample,
)
from warp.fem.utils import type_zero_element
from warp.sparse import BsrMatrix, bsr_set_from_triplets, bsr_zeros
from warp.types import type_length
from warp.utils import array_cast


def _resolve_path(func, node):
    """
    Resolves variable and path from ast node/attribute (adapted from warp.codegen)
    """

    modules = []

    while isinstance(node, ast.Attribute):
        modules.append(node.attr)
        node = node.value

    if isinstance(node, ast.Name):
        modules.append(node.id)

    # reverse list since ast presents it backward order
    path = [*reversed(modules)]

    if len(path) == 0:
        return None, path

    # try and evaluate object path
    try:
        # Look up the closure info and append it to adj.func.__globals__
        # in case you want to define a kernel inside a function and refer
        # to variables you've declared inside that function:
        capturedvars = dict(zip(func.__code__.co_freevars, [c.cell_contents for c in (func.__closure__ or [])]))

        vars_dict = {**func.__globals__, **capturedvars}
        func = eval(".".join(path), vars_dict)
        return func, path
    except (NameError, AttributeError):
        pass

    return None, path


class IntegrandVisitor(ast.NodeTransformer):
    class FieldInfo(NamedTuple):
        field: FieldLike
        abstract_type: type
        concrete_type: type
        root_arg_name: type

    def __init__(
        self,
        integrand: Integrand,
        field_info: Dict[str, FieldInfo],
    ):
        self._integrand = integrand
        self._field_symbols = field_info.copy()
        self._field_nodes = {}

    @staticmethod
    def _build_field_info(integrand: Integrand, field_args: Dict[str, FieldLike]):
        def get_concrete_type(field: Union[FieldLike, Domain]):
            if isinstance(field, FieldLike):
                return field.ElementEvalArg
            elif isinstance(field, GeometryDomain):
                return field.DomainArg
            return field.ElementArg

        return {
            name: IntegrandVisitor.FieldInfo(
                field=field,
                abstract_type=integrand.argspec.annotations[name],
                concrete_type=get_concrete_type(field),
                root_arg_name=name,
            )
            for name, field in field_args.items()
        }

    def _get_field_info(self, node: ast.expr):
        field_info = self._field_nodes.get(node)
        if field_info is None and isinstance(node, ast.Name):
            field_info = self._field_symbols.get(node.id)

        return field_info

    def visit_Call(self, call: ast.Call):
        call = self.generic_visit(call)

        callee = getattr(call.func, "id", None)
        if callee in self._field_symbols:
            # Shortcut for evaluating fields as f(x...)
            field_info = self._field_symbols[callee]

            # Replace with default call operator
            default_operator = field_info.abstract_type.call_operator

            self._process_operator_call(call, callee, default_operator, field_info)

            return call

        func, _ = _resolve_path(self._integrand.func, call.func)

        if isinstance(func, Operator) and len(call.args) > 0:
            # Evaluating operators as op(field, x, ...)
            field_info = self._get_field_info(call.args[0])
            if field_info is not None:
                self._process_operator_call(call, func, func, field_info)

                if func.field_result:
                    res = func.field_result(field_info.field)
                    self._field_nodes[call] = IntegrandVisitor.FieldInfo(
                        field=res[0],
                        abstract_type=res[1],
                        concrete_type=res[2],
                        root_arg_name=f"{field_info.root_arg_name}.{func.name}",
                    )

        if isinstance(func, Integrand):
            callee_field_args = self._get_callee_field_args(func, call.args)
            self._process_integrand_call(call, func, callee_field_args)

        # print(ast.dump(call, indent=4))

        return call

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)

        # Check if we're assigning a field
        src_field_info = self._get_field_info(node.value)
        if src_field_info is not None:
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise NotImplementedError("warp.fem Fields and Domains may only be assigned to simple variables")

            self._field_symbols[node.targets[0].id] = src_field_info

        return node

    def _get_callee_field_args(self, callee: Integrand, args: List[ast.AST]):
        # Get field types for call site arguments
        call_site_field_args: List[IntegrandVisitor.FieldInfo] = []
        for arg in args:
            field_info = self._get_field_info(arg)
            if field_info is not None:
                call_site_field_args.append(field_info)

        call_site_field_args.reverse()

        # Pass to callee in same order
        callee_field_args = {}
        for arg in callee.argspec.args:
            arg_type = callee.argspec.annotations[arg]
            if arg_type in (Field, Domain):
                passed_field_info = call_site_field_args.pop()
                if passed_field_info.abstract_type != arg_type:
                    raise TypeError(
                        f"Attempting to pass a {passed_field_info.abstract_type.__name__} to argument '{arg}' of '{callee.name}' expecting a {arg_type.__name__}"
                    )
                callee_field_args[arg] = passed_field_info

        return callee_field_args


class IntegrandOperatorParser(IntegrandVisitor):
    def __init__(self, integrand: Integrand, field_info: Dict[str, IntegrandVisitor.FieldInfo], callback: Callable):
        super().__init__(integrand, field_info)
        self._operator_callback = callback

    def _process_operator_call(
        self, call: ast.Call, callee: Union[str, Operator], operator: Operator, field_info: IntegrandVisitor.FieldInfo
    ):
        self._operator_callback(field_info, operator)

    def _process_integrand_call(
        self, call: ast.Call, callee: Integrand, callee_field_args: Dict[str, IntegrandVisitor.FieldInfo]
    ):
        callee_field_args = self._get_callee_field_args(callee, call.args)
        callee_parser = IntegrandOperatorParser(callee, callee_field_args, callback=self._operator_callback)
        callee_parser._apply()

    def _apply(self):
        source = textwrap.dedent(inspect.getsource(self._integrand.func))
        tree = ast.parse(source)
        self.visit(tree)

    @staticmethod
    def apply(
        integrand: Integrand, field_args: Dict[str, FieldLike], operator_callback: Optional[Callable] = None
    ) -> wp.Function:
        field_info = IntegrandVisitor._build_field_info(integrand, field_args)
        IntegrandOperatorParser(integrand, field_info, callback=operator_callback)._apply()


class IntegrandTransformer(IntegrandVisitor):
    def _process_operator_call(
        self, call: ast.Call, callee: Union[str, Operator], operator: Operator, field_info: IntegrandVisitor.FieldInfo
    ):
        field = field_info.field

        try:
            # Retrieve the function pointer corresponding to the operator implementation for the field type
            pointer = operator.resolver(field)
            if not isinstance(pointer, wp.context.Function):
                raise NotImplementedError(operator.resolver.__name__)

        except (AttributeError, NotImplementedError) as e:
            raise TypeError(
                f"Operator {operator.func.__name__} is not defined for {field_info.abstract_type.__name__} {field.name}"
            ) from e

        # Update the ast Call node to use the new function pointer
        call.func = ast.Attribute(value=call.func, attr=pointer.key, ctx=ast.Load())

        # Save the pointer as an attribute than can be accessed from the calling scope
        # For usual operator call syntax, we can use the operator itself, but for the
        # shortcut default operator syntax, we store it on the callee's concrete type
        if isinstance(callee, Operator):
            setattr(callee, pointer.key, pointer)
        else:
            setattr(field_info.concrete_type, pointer.key, pointer)

            # also insert callee as first argument
            call.args = [ast.Name(id=callee, ctx=ast.Load()), *call.args]

        # replace first argument with selected attribute
        if operator.attr:
            call.args[0] = ast.Attribute(value=call.args[0], attr=operator.attr)

    def _process_integrand_call(
        self, call: ast.Call, callee: Integrand, callee_field_args: Dict[str, IntegrandVisitor.FieldInfo]
    ):
        callee_field_args = self._get_callee_field_args(callee, call.args)
        transformer = IntegrandTransformer(callee, callee_field_args)
        key = transformer._apply().key
        call.func = ast.Attribute(
            value=call.func,
            attr=key,
            ctx=ast.Load(),
        )

    def _apply(self) -> wp.Function:
        # Transform field evaluation calls
        field_info = self._field_symbols

        # Specialize field argument types
        argspec = self._integrand.argspec
        annotations = argspec.annotations.copy()
        annotations.update({name: f.concrete_type for name, f in field_info.items()})

        suffix = "_".join([f.field.name for f in field_info.values()])
        func = cache.get_integrand_function(
            integrand=self._integrand,
            suffix=suffix,
            annotations=annotations,
            code_transformers=[self],
        )

        # func = self._integrand.module.functions[func.key] #no longer needed?
        setattr(self._integrand, func.key, func)

        return func

    @staticmethod
    def apply(integrand: Integrand, field_args: Dict[str, FieldLike]) -> wp.Function:
        field_info = IntegrandVisitor._build_field_info(integrand, field_args)
        return IntegrandTransformer(integrand, field_info)._apply()


class IntegrandArguments(NamedTuple):
    field_args: Dict[str, Union[FieldLike, GeometryDomain]]
    value_args: Dict[str, Any]
    domain_name: str
    sample_name: str
    test_name: str
    trial_name: str


def _parse_integrand_arguments(
    integrand: Integrand,
    fields: Dict[str, FieldLike],
):
    # parse argument types
    field_args = {}
    value_args = {}

    domain_name = None
    sample_name = None
    test_name = None
    trial_name = None

    argspec = integrand.argspec
    for arg in argspec.args:
        arg_type = argspec.annotations[arg]
        if arg_type == Field:
            try:
                field = fields[arg]
            except KeyError as err:
                raise ValueError(f"Missing field for argument '{arg}' of integrand '{integrand.name}'") from err
            if not isinstance(field, FieldLike):
                raise ValueError(f"Passed field argument '{arg}' is not a proper Field")
            if isinstance(field, TestField):
                if test_name is not None:
                    raise ValueError(f"More than one test field argument: '{test_name}' and '{arg}'")
                test_name = arg
            elif isinstance(field, TrialField):
                if trial_name is not None:
                    raise ValueError(f"More than one trial field argument: '{trial_name}' and '{arg}'")
                trial_name = arg
            field_args[arg] = field
        elif arg_type == Domain:
            if domain_name is not None:
                raise SyntaxError(f"Integrand '{integrand.name}' must have at most one argument of type Domain")
            if arg in fields:
                raise ValueError(
                    f"Domain argument '{arg}' of '{integrand.name}' will be automatically populated and must not be passed as a field argument."
                )
            domain_name = arg
        elif arg_type == Sample:
            if sample_name is not None:
                raise SyntaxError(f"Integrand '{integrand.name}' must have at most one argument of type Sample")
            if arg in fields:
                raise ValueError(
                    f"Sample argument '{arg}' of '{integrand.name}' will be automatically populated and must not be passed as a field argument."
                )
            sample_name = arg
        else:
            if arg in fields:
                raise ValueError(
                    f"Cannot pass a field argument to '{arg}'  of '{integrand.name}' with is not of type 'Field'"
                )
            value_args[arg] = arg_type

    return IntegrandArguments(field_args, value_args, domain_name, sample_name, test_name, trial_name)


def _check_field_compat(integrand: Integrand, arguments: IntegrandArguments, domain: GeometryDomain):
    # Check field compatibility
    for name, field in arguments.field_args.items():
        if isinstance(field, GeometryField) and domain is not None:
            if field.geometry != domain.geometry:
                raise ValueError(f"Field '{name}' must be defined on the same geometry as the integration domain")
            if field.element_kind != domain.element_kind:
                raise ValueError(
                    f"Field '{name}' is not defined on the same kind of elements (cells or sides) as the integration domain. Maybe a forgotten `.trace()`?"
                )


def _find_integrand_operators(integrand: Integrand, field_args: Dict[str, FieldLike]):
    if integrand.operators is None:
        # Integrands operator dictionary does not depend on concrete field type,
        # so only needs to be built once per integrand

        operators = {}

        def operator_callback(field: IntegrandVisitor.FieldInfo, op: Operator):
            if field.root_arg_name in operators:
                operators[field.root_arg_name].add(op)
            else:
                operators[field.root_arg_name] = {op}

        IntegrandOperatorParser.apply(integrand, field_args, operator_callback=operator_callback)

        integrand.operators = operators


def _notify_operator_usage(
    integrand: Integrand,
    field_args: Dict[str, FieldLike],
):
    for arg, field_ops in integrand.operators.items():
        if arg in field_args:
            # print(f"{arg} {field_args[arg].name} : {', '.join(op.name for op in field_ops)}")
            field_args[arg].notify_operator_usage(field_ops)


def _gen_field_struct(field_args: Dict[str, FieldLike]):
    class Fields:
        pass

    annotations = get_annotations(Fields)

    for name, arg in field_args.items():
        if isinstance(arg, GeometryDomain):
            continue
        setattr(Fields, name, arg.EvalArg())
        annotations[name] = arg.EvalArg

    try:
        Fields.__annotations__ = annotations
    except AttributeError:
        Fields.__dict__.__annotations__ = annotations

    suffix = "_".join([f"{name}_{arg_struct.cls.__qualname__}" for name, arg_struct in annotations.items()])

    return cache.get_struct(Fields, suffix=suffix)


def _get_trial_arg():
    pass


def _get_test_arg():
    pass


class PassFieldArgsToIntegrand(ast.NodeTransformer):
    def __init__(
        self,
        arg_names: List[str],
        parsed_args: IntegrandArguments,
        integrand_func: wp.Function,
        func_name: str = "integrand_func",
        fields_var_name: str = "fields",
        values_var_name: str = "values",
        domain_var_name: str = "domain_arg",
        domain_index_var_name: str = "domain_index_arg",
        sample_var_name: str = "sample",
        field_wrappers_attr: str = "_field_wrappers",
    ):
        self._arg_names = arg_names
        self._field_args = parsed_args.field_args
        self._value_args = parsed_args.value_args
        self._domain_name = parsed_args.domain_name
        self._sample_name = parsed_args.sample_name
        self._test_name = parsed_args.test_name
        self._trial_name = parsed_args.trial_name
        self._func_name = func_name
        self._fields_var_name = fields_var_name
        self._values_var_name = values_var_name
        self._domain_var_name = domain_var_name
        self._domain_index_var_name = domain_index_var_name
        self._sample_var_name = sample_var_name

        self._field_wrappers_attr = field_wrappers_attr
        self._register_integrand_field_wrappers(integrand_func, parsed_args.field_args)

    class _FieldWrappers:
        pass

    def _register_integrand_field_wrappers(self, integrand_func: wp.Function, fields: Dict[str, FieldLike]):
        # Mechanism to pass the geometry argument only once to the root kernel
        # Field wrappers are used to forward it to all fields in nested integrand calls
        field_wrappers = PassFieldArgsToIntegrand._FieldWrappers()
        for name, field in fields.items():
            if isinstance(field, FieldLike):
                setattr(field_wrappers, name, field.ElementEvalArg)
            elif isinstance(field, GeometryDomain):
                setattr(field_wrappers, name, field.DomainArg)
        setattr(integrand_func, self._field_wrappers_attr, field_wrappers)

    def _emit_field_wrapper_call(self, field_name, *data_arguments):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id=self._func_name, ctx=ast.Load()),
                    attr=self._field_wrappers_attr,
                    ctx=ast.Load(),
                ),
                attr=field_name,
                ctx=ast.Load(),
            ),
            args=[
                ast.Name(id=self._domain_var_name, ctx=ast.Load()),
                *data_arguments,
            ],
            keywords=[],
        )

    def visit_Call(self, call: ast.Call):
        call = self.generic_visit(call)

        callee = getattr(call.func, "id", None)

        if callee == self._func_name:
            # Replace function arguments with our generated structs
            call.args.clear()
            for arg in self._arg_names:
                if arg == self._domain_name:
                    call.args.append(
                        self._emit_field_wrapper_call(
                            arg,
                            ast.Name(id=self._domain_index_var_name, ctx=ast.Load()),
                        )
                    )

                elif arg == self._sample_name:
                    call.args.append(
                        ast.Name(id=self._sample_var_name, ctx=ast.Load()),
                    )
                elif arg in self._field_args:
                    call.args.append(
                        self._emit_field_wrapper_call(
                            arg,
                            ast.Attribute(
                                value=ast.Name(id=self._fields_var_name, ctx=ast.Load()),
                                attr=arg,
                                ctx=ast.Load(),
                            ),
                        )
                    )
                elif arg in self._value_args:
                    call.args.append(
                        ast.Attribute(
                            value=ast.Name(id=self._values_var_name, ctx=ast.Load()),
                            attr=arg,
                            ctx=ast.Load(),
                        )
                    )
                else:
                    raise RuntimeError(f"Unhandled argument {arg}")
            # print(ast.dump(call, indent=4))
        elif callee == _get_test_arg.__name__:
            # print(ast.dump(call, indent=4))
            call = ast.Attribute(
                value=ast.Name(id=self._fields_var_name, ctx=ast.Load()),
                attr=self._test_name,
                ctx=ast.Load(),
            )
        elif callee == _get_trial_arg.__name__:
            # print(ast.dump(call, indent=4))
            call = ast.Attribute(
                value=ast.Name(id=self._fields_var_name, ctx=ast.Load()),
                attr=self._trial_name,
                ctx=ast.Load(),
            )

        return call


def _combined_kernel_options(integrand_options: Optional[Dict[str, Any]], call_site_options: Optional[Dict[str, Any]]):
    if integrand_options is None:
        return {} if call_site_options is None else call_site_options

    options = integrand_options.copy()
    if call_site_options is not None:
        options.update(call_site_options)
    return options


def get_integrate_constant_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    accumulate_dtype,
):
    def integrate_kernel_fn(
        qp_arg: quadrature.Arg,
        qp_element_index_arg: quadrature.ElementIndexArg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array(dtype=accumulate_dtype),
    ):
        qp_eval_index = wp.tid()
        domain_element_index, qp = quadrature.evaluation_point_element_index(qp_element_index_arg, qp_eval_index)
        if domain_element_index == NULL_ELEMENT_INDEX:
            return

        element_index = domain.element_index(domain_index_arg, domain_element_index)

        qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)

        test_dof_index = NULL_DOF_INDEX
        trial_dof_index = NULL_DOF_INDEX

        sample = Sample(element_index, qp_coords, qp_index, qp_weight, test_dof_index, trial_dof_index)
        vol = domain.element_measure(domain_arg, sample)

        val = integrand_func(sample, fields, values)

        wp.atomic_add(result, 0, accumulate_dtype(qp_weight * vol * val))

    return integrate_kernel_fn


def get_integrate_linear_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    test: TestField,
    output_dtype,
    accumulate_dtype,
):
    def integrate_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        test_arg: test.space_restriction.NodeArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array2d(dtype=output_dtype),
    ):
        local_node_index, test_dof = wp.tid()
        node_index = test.space_restriction.node_partition_index(test_arg, local_node_index)
        element_beg, element_end = test.space_restriction.node_element_range(test_arg, node_index)

        trial_dof_index = NULL_DOF_INDEX

        val_sum = accumulate_dtype(0.0)

        for n in range(element_beg, element_end):
            node_element_index = test.space_restriction.node_element_index(test_arg, n)
            element_index = domain.element_index(domain_index_arg, node_element_index.domain_element_index)

            test_dof_index = DofIndex(node_element_index.node_index_in_element, test_dof)

            qp_point_count = quadrature.point_count(
                domain_arg, qp_arg, node_element_index.domain_element_index, element_index
            )
            for k in range(qp_point_count):
                qp_index = quadrature.point_index(
                    domain_arg, qp_arg, node_element_index.domain_element_index, element_index, k
                )
                qp_coords = quadrature.point_coords(
                    domain_arg, qp_arg, node_element_index.domain_element_index, element_index, k
                )
                qp_weight = quadrature.point_weight(
                    domain_arg, qp_arg, node_element_index.domain_element_index, element_index, k
                )

                vol = domain.element_measure(domain_arg, make_free_sample(element_index, qp_coords))

                sample = Sample(element_index, qp_coords, qp_index, qp_weight, test_dof_index, trial_dof_index)
                val = integrand_func(sample, fields, values)

                val_sum += accumulate_dtype(qp_weight * vol * val)

        result[node_index, test_dof] += output_dtype(val_sum)

    return integrate_kernel_fn


def get_integrate_linear_nodal_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    test: TestField,
    output_dtype,
    accumulate_dtype,
):
    def integrate_kernel_fn(
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        test_restriction_arg: test.space_restriction.NodeArg,
        test_topo_arg: test.space.topology.TopologyArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array2d(dtype=output_dtype),
    ):
        local_node_index, dof = wp.tid()

        partition_node_index = test.space_restriction.node_partition_index(test_restriction_arg, local_node_index)
        element_beg, element_end = test.space_restriction.node_element_range(test_restriction_arg, partition_node_index)

        trial_dof_index = NULL_DOF_INDEX

        val_sum = accumulate_dtype(0.0)

        for n in range(element_beg, element_end):
            node_element_index = test.space_restriction.node_element_index(test_restriction_arg, n)
            element_index = domain.element_index(domain_index_arg, node_element_index.domain_element_index)

            if n == element_beg:
                node_index = test.space.topology.element_node_index(
                    domain_arg, test_topo_arg, element_index, node_element_index.node_index_in_element
                )

            coords = test.space.node_coords_in_element(
                domain_arg,
                _get_test_arg().space_arg,
                element_index,
                node_element_index.node_index_in_element,
            )

            if coords[0] != OUTSIDE:
                node_weight = test.space.node_quadrature_weight(
                    domain_arg,
                    _get_test_arg().space_arg,
                    element_index,
                    node_element_index.node_index_in_element,
                )

                test_dof_index = DofIndex(node_element_index.node_index_in_element, dof)

                sample = Sample(
                    element_index,
                    coords,
                    node_index,
                    node_weight,
                    test_dof_index,
                    trial_dof_index,
                )
                vol = domain.element_measure(domain_arg, sample)
                val = integrand_func(sample, fields, values)

                val_sum += accumulate_dtype(node_weight * vol * val)

        result[partition_node_index, dof] += output_dtype(val_sum)

    return integrate_kernel_fn


def get_integrate_linear_local_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    test: LocalTestField,
):
    def integrate_kernel_fn(
        qp_arg: quadrature.Arg,
        qp_element_index_arg: quadrature.ElementIndexArg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array3d(dtype=float),
    ):
        qp_eval_index, taylor_dof, test_dof = wp.tid()
        domain_element_index, qp = quadrature.evaluation_point_element_index(qp_element_index_arg, qp_eval_index)

        if domain_element_index == NULL_ELEMENT_INDEX:
            return

        element_index = domain.element_index(domain_index_arg, domain_element_index)

        qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)

        vol = domain.element_measure(domain_arg, make_free_sample(element_index, qp_coords))

        trial_dof_index = NULL_DOF_INDEX
        test_dof_index = DofIndex(taylor_dof, test_dof)

        sample = Sample(element_index, qp_coords, qp_index, qp_weight, test_dof_index, trial_dof_index)
        val = integrand_func(sample, fields, values)
        result[qp_eval_index, taylor_dof, test_dof] = qp_weight * vol * val

    return integrate_kernel_fn


def get_integrate_bilinear_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    test: TestField,
    trial: TrialField,
    output_dtype,
    accumulate_dtype,
):
    MAX_NODES_PER_ELEMENT = trial.space.topology.MAX_NODES_PER_ELEMENT

    def integrate_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        test_arg: test.space_restriction.NodeArg,
        trial_partition_arg: trial.space_partition.PartitionArg,
        trial_topology_arg: trial.space_partition.space_topology.TopologyArg,
        fields: FieldStruct,
        values: ValueStruct,
        triplet_rows: wp.array(dtype=int),
        triplet_cols: wp.array(dtype=int),
        triplet_values: wp.array3d(dtype=output_dtype),
    ):
        test_local_node_index, trial_node, test_dof, trial_dof = wp.tid()

        test_node_index = test.space_restriction.node_partition_index(test_arg, test_local_node_index)
        element_beg, element_end = test.space_restriction.node_element_range(test_arg, test_node_index)

        trial_dof_index = DofIndex(trial_node, trial_dof)

        for element in range(element_beg, element_end):
            test_element_index = test.space_restriction.node_element_index(test_arg, element)
            element_index = domain.element_index(domain_index_arg, test_element_index.domain_element_index)

            element_trial_node_count = trial.space.topology.element_node_count(
                domain_arg, trial_topology_arg, element_index
            )
            qp_point_count = wp.where(
                trial_node < element_trial_node_count,
                quadrature.point_count(domain_arg, qp_arg, test_element_index.domain_element_index, element_index),
                0,
            )

            test_dof_index = DofIndex(
                test_element_index.node_index_in_element,
                test_dof,
            )

            val_sum = accumulate_dtype(0.0)

            for k in range(qp_point_count):
                qp_index = quadrature.point_index(
                    domain_arg, qp_arg, test_element_index.domain_element_index, element_index, k
                )
                coords = quadrature.point_coords(
                    domain_arg, qp_arg, test_element_index.domain_element_index, element_index, k
                )

                qp_weight = quadrature.point_weight(
                    domain_arg, qp_arg, test_element_index.domain_element_index, element_index, k
                )
                vol = domain.element_measure(domain_arg, make_free_sample(element_index, coords))

                sample = Sample(
                    element_index,
                    coords,
                    qp_index,
                    qp_weight,
                    test_dof_index,
                    trial_dof_index,
                )
                val = integrand_func(sample, fields, values)
                val_sum += accumulate_dtype(qp_weight * vol * val)

            block_offset = element * MAX_NODES_PER_ELEMENT + trial_node
            triplet_values[block_offset, test_dof, trial_dof] = output_dtype(val_sum)

            # Set row and column indices
            if test_dof == 0 and trial_dof == 0:
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

    return integrate_kernel_fn


def get_integrate_bilinear_nodal_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    test: TestField,
    output_dtype,
    accumulate_dtype,
):
    def integrate_kernel_fn(
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        test_restriction_arg: test.space_restriction.NodeArg,
        test_topo_arg: test.space.topology.TopologyArg,
        fields: FieldStruct,
        values: ValueStruct,
        triplet_rows: wp.array(dtype=int),
        triplet_cols: wp.array(dtype=int),
        triplet_values: wp.array3d(dtype=output_dtype),
    ):
        local_node_index, test_dof, trial_dof = wp.tid()

        partition_node_index = test.space_restriction.node_partition_index(test_restriction_arg, local_node_index)
        element_beg, element_end = test.space_restriction.node_element_range(test_restriction_arg, partition_node_index)

        val_sum = accumulate_dtype(0.0)

        for n in range(element_beg, element_end):
            node_element_index = test.space_restriction.node_element_index(test_restriction_arg, n)
            element_index = domain.element_index(domain_index_arg, node_element_index.domain_element_index)

            if n == element_beg:
                node_index = test.space.topology.element_node_index(
                    domain_arg, test_topo_arg, element_index, node_element_index.node_index_in_element
                )

            coords = test.space.node_coords_in_element(
                domain_arg,
                _get_test_arg().space_arg,
                element_index,
                node_element_index.node_index_in_element,
            )

            if coords[0] != OUTSIDE:
                node_weight = test.space.node_quadrature_weight(
                    domain_arg,
                    _get_test_arg().space_arg,
                    element_index,
                    node_element_index.node_index_in_element,
                )

                test_dof_index = DofIndex(node_element_index.node_index_in_element, test_dof)
                trial_dof_index = DofIndex(node_element_index.node_index_in_element, trial_dof)

                sample = Sample(
                    element_index,
                    coords,
                    node_index,
                    node_weight,
                    test_dof_index,
                    trial_dof_index,
                )
                vol = domain.element_measure(domain_arg, sample)
                val = integrand_func(sample, fields, values)

                val_sum += accumulate_dtype(node_weight * vol * val)

        triplet_values[local_node_index, test_dof, trial_dof] = output_dtype(val_sum)
        triplet_rows[local_node_index] = partition_node_index
        triplet_cols[local_node_index] = partition_node_index

    return integrate_kernel_fn


def get_integrate_bilinear_local_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    test: LocalTestField,
    trial: LocalTrialField,
):
    TEST_TAYLOR_DOF_COUNT = test.TAYLOR_DOF_COUNT
    TRIAL_TAYLOR_DOF_COUNT = trial.TAYLOR_DOF_COUNT

    def integrate_kernel_fn(
        qp_arg: quadrature.Arg,
        qp_element_index_arg: quadrature.ElementIndexArg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array4d(dtype=float),
    ):
        qp_eval_index, test_dof, trial_dof, trial_taylor_dof = wp.tid()

        domain_element_index, qp = quadrature.evaluation_point_element_index(qp_element_index_arg, qp_eval_index)
        if domain_element_index == NULL_ELEMENT_INDEX:
            return

        element_index = domain.element_index(domain_index_arg, domain_element_index)

        qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)

        vol = domain.element_measure(domain_arg, make_free_sample(element_index, qp_coords))
        qp_vol = vol * qp_weight

        trial_dof_index = DofIndex(trial_taylor_dof, trial_dof)

        for test_taylor_dof in range(TEST_TAYLOR_DOF_COUNT):
            taylor_dof = test_taylor_dof * TRIAL_TAYLOR_DOF_COUNT + trial_taylor_dof

            test_dof_index = DofIndex(test_taylor_dof, test_dof)

            sample = Sample(element_index, qp_coords, qp_index, qp_weight, test_dof_index, trial_dof_index)
            val = integrand_func(sample, fields, values)
            result[qp_eval_index, test_dof, trial_dof, taylor_dof] = qp_vol * val

    return integrate_kernel_fn


def _generate_integrate_kernel(
    integrand: Integrand,
    domain: GeometryDomain,
    quadrature: Quadrature,
    arguments: IntegrandArguments,
    test: Optional[TestField],
    trial: Optional[TrialField],
    output_dtype: type,
    accumulate_dtype: type,
    kernel_options: Optional[Dict[str, Any]] = None,
) -> wp.Kernel:
    output_dtype = wp.types.type_scalar_type(output_dtype)

    FieldStruct = _gen_field_struct(arguments.field_args)
    ValueStruct = cache.get_argument_struct(arguments.value_args)

    _notify_operator_usage(integrand, arguments.field_args)

    # Check if kernel exist in cache
    field_names = "_".join(f"{k}{f.name}" for k, f in arguments.field_args.items())
    kernel_suffix = f"_itg_{wp.types.type_typestr(output_dtype)}{wp.types.type_typestr(accumulate_dtype)}_{field_names}"

    if quadrature is not None:
        kernel_suffix += quadrature.name

    kernel = cache.get_integrand_kernel(integrand=integrand, suffix=kernel_suffix, kernel_options=kernel_options)
    if kernel is not None:
        return kernel, FieldStruct, ValueStruct

    # Not found in cache, transform integrand and generate kernel
    _check_field_compat(integrand, arguments, domain)

    integrand_func = IntegrandTransformer.apply(integrand, arguments.field_args)

    nodal = quadrature is None

    if test is None and trial is None:
        integrate_kernel_fn = get_integrate_constant_kernel(
            integrand_func,
            domain,
            quadrature,
            FieldStruct,
            ValueStruct,
            accumulate_dtype=accumulate_dtype,
        )
    elif trial is None:
        if nodal:
            integrate_kernel_fn = get_integrate_linear_nodal_kernel(
                integrand_func,
                domain,
                FieldStruct,
                ValueStruct,
                test=test,
                output_dtype=output_dtype,
                accumulate_dtype=accumulate_dtype,
            )
        elif isinstance(test, LocalTestField):
            integrate_kernel_fn = get_integrate_linear_local_kernel(
                integrand_func,
                domain,
                quadrature,
                FieldStruct,
                ValueStruct,
                test=test,
            )
        else:
            integrate_kernel_fn = get_integrate_linear_kernel(
                integrand_func,
                domain,
                quadrature,
                FieldStruct,
                ValueStruct,
                test=test,
                output_dtype=output_dtype,
                accumulate_dtype=accumulate_dtype,
            )
    else:
        if nodal:
            integrate_kernel_fn = get_integrate_bilinear_nodal_kernel(
                integrand_func,
                domain,
                FieldStruct,
                ValueStruct,
                test=test,
                output_dtype=output_dtype,
                accumulate_dtype=accumulate_dtype,
            )
        elif isinstance(test, LocalTestField):
            integrate_kernel_fn = get_integrate_bilinear_local_kernel(
                integrand_func,
                domain,
                quadrature,
                FieldStruct,
                ValueStruct,
                test=test,
                trial=trial,
            )
        else:
            integrate_kernel_fn = get_integrate_bilinear_kernel(
                integrand_func,
                domain,
                quadrature,
                FieldStruct,
                ValueStruct,
                test=test,
                trial=trial,
                output_dtype=output_dtype,
                accumulate_dtype=accumulate_dtype,
            )

    kernel = cache.get_integrand_kernel(
        integrand=integrand,
        kernel_fn=integrate_kernel_fn,
        suffix=kernel_suffix,
        kernel_options=kernel_options,
        code_transformers=[
            PassFieldArgsToIntegrand(
                arg_names=integrand.argspec.args, parsed_args=arguments, integrand_func=integrand_func
            )
        ],
    )

    return kernel, FieldStruct, ValueStruct


def _launch_integrate_kernel(
    integrand: Integrand,
    kernel: wp.Kernel,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    domain: GeometryDomain,
    quadrature: Quadrature,
    test: Optional[TestField],
    trial: Optional[TrialField],
    fields: Dict[str, FieldLike],
    values: Dict[str, Any],
    accumulate_dtype: type,
    temporary_store: Optional[cache.TemporaryStore],
    output_dtype: type,
    output: Optional[Union[wp.array, BsrMatrix]],
    add_to_output: bool,
    bsr_options: Optional[Dict[str, Any]],
    device,
):
    # Set-up launch arguments
    domain_elt_arg = domain.element_arg_value(device=device)
    domain_elt_index_arg = domain.element_index_arg_value(device=device)

    if quadrature is not None:
        qp_arg = quadrature.arg_value(device=device)

    field_arg_values = FieldStruct()
    for k, v in fields.items():
        if not isinstance(v, GeometryDomain):
            setattr(field_arg_values, k, v.eval_arg_value(device=device))

    value_struct_values = cache.populate_argument_struct(ValueStruct, values, func_name=integrand.name)

    # Constant form
    if test is None and trial is None:
        if output is not None and output.dtype == accumulate_dtype:
            if output.size < 1:
                raise RuntimeError("Output array must be of size at least 1")
            accumulate_array = output
        else:
            accumulate_temporary = cache.borrow_temporary(
                shape=(1),
                device=device,
                dtype=accumulate_dtype,
                temporary_store=temporary_store,
                requires_grad=output is not None and output.requires_grad,
            )
            accumulate_array = accumulate_temporary.array

        if output != accumulate_array or not add_to_output:
            accumulate_array.zero_()

        wp.launch(
            kernel=kernel,
            dim=quadrature.evaluation_point_count(),
            inputs=[
                qp_arg,
                quadrature.element_index_arg_value(device),
                domain_elt_arg,
                domain_elt_index_arg,
                field_arg_values,
                value_struct_values,
                accumulate_array,
            ],
            device=device,
        )

        if output == accumulate_array:
            return output
        if output is None:
            return accumulate_array.numpy()[0]

        if add_to_output:
            # accumulate dtype is distinct from output dtype
            array_axpy(x=accumulate_array, y=output)
        else:
            array_cast(in_array=accumulate_array, out_array=output)
        return output

    test_arg = test.space_restriction.node_arg(device=device)
    nodal = quadrature is None

    # Linear form
    if trial is None:
        # If an output array is provided with the correct type, accumulate directly into it
        # Otherwise, grab a temporary array
        if output is None:
            if type_length(output_dtype) == test.node_dof_count:
                output_shape = (test.space_partition.node_count(),)
            elif type_length(output_dtype) == 1:
                output_shape = (test.space_partition.node_count(), test.node_dof_count)
            else:
                raise RuntimeError(
                    f"Incompatible output type {wp.types.type_repr(output_dtype)}, must be scalar or vector of length {test.node_dof_count}"
                )

            output_temporary = cache.borrow_temporary(
                temporary_store=temporary_store,
                shape=output_shape,
                dtype=output_dtype,
                device=device,
            )

            output = output_temporary.array

        else:
            output_temporary = None

            if output.shape[0] < test.space_partition.node_count():
                raise RuntimeError(f"Output array must have at least {test.space_partition.node_count()} rows")

            output_dtype = output.dtype
            if type_length(output_dtype) != test.node_dof_count:
                if type_length(output_dtype) != 1:
                    raise RuntimeError(
                        f"Incompatible output type {wp.types.type_repr(output_dtype)}, must be scalar or vector of length {test.node_dof_count}"
                    )
                if output.ndim != 2 and output.shape[1] != test.node_dof_count:
                    raise RuntimeError(
                        f"Incompatible output array shape, last dimension must be of size {test.node_dof_count}"
                    )

        # Launch the integration on the kernel on a 2d scalar view of the actual array
        if not add_to_output:
            output.zero_()

        def as_2d_array(array):
            return wp.array(
                data=None,
                ptr=array.ptr,
                capacity=array.capacity,
                device=array.device,
                shape=(test.space_partition.node_count(), test.node_dof_count),
                dtype=wp.types.type_scalar_type(output_dtype),
                grad=None if array.grad is None else as_2d_array(array.grad),
            )

        output_view = output if output.ndim == 2 else as_2d_array(output)

        if nodal:
            wp.launch(
                kernel=kernel,
                dim=(test.space_restriction.node_count(), test.node_dof_count),
                inputs=[
                    domain_elt_arg,
                    domain_elt_index_arg,
                    test_arg,
                    test.space.topology.topo_arg_value(device),
                    field_arg_values,
                    value_struct_values,
                    output_view,
                ],
                device=device,
            )
        elif isinstance(test, LocalTestField):
            local_result = cache.borrow_temporary(
                temporary_store=temporary_store,
                device=device,
                requires_grad=output.requires_grad,
                shape=(quadrature.evaluation_point_count(), test.TAYLOR_DOF_COUNT, test.value_dof_count),
                dtype=float,
            )

            wp.launch(
                kernel=kernel,
                dim=local_result.array.shape,
                inputs=[
                    qp_arg,
                    quadrature.element_index_arg_value(device),
                    domain_elt_arg,
                    domain_elt_index_arg,
                    field_arg_values,
                    value_struct_values,
                    local_result.array,
                ],
                device=device,
            )

            dispatch_kernel = make_linear_dispatch_kernel(test, quadrature, accumulate_dtype)
            wp.launch(
                kernel=dispatch_kernel,
                dim=(test.space_restriction.node_count(), test.node_dof_count),
                inputs=[
                    qp_arg,
                    domain_elt_arg,
                    domain_elt_index_arg,
                    test_arg,
                    test.space.space_arg_value(device),
                    local_result.array,
                    output_view,
                ],
                device=device,
            )

            local_result.release()

        else:
            wp.launch(
                kernel=kernel,
                dim=(test.space_restriction.node_count(), test.node_dof_count),
                inputs=[
                    qp_arg,
                    domain_elt_arg,
                    domain_elt_index_arg,
                    test_arg,
                    field_arg_values,
                    value_struct_values,
                    output_view,
                ],
                device=device,
            )

        if output_temporary is not None:
            return output_temporary.detach()

        return output

    # Bilinear form

    if test.node_dof_count == 1 and trial.node_dof_count == 1:
        block_type = output_dtype
    else:
        block_type = cache.cached_mat_type(shape=(test.node_dof_count, trial.node_dof_count), dtype=output_dtype)

    if nodal:
        nnz = test.space_restriction.node_count()
    else:
        nnz = test.space_restriction.total_node_element_count() * trial.space.topology.MAX_NODES_PER_ELEMENT

    triplet_rows_temp = cache.borrow_temporary(temporary_store, shape=(nnz,), dtype=int, device=device)
    triplet_cols_temp = cache.borrow_temporary(temporary_store, shape=(nnz,), dtype=int, device=device)
    triplet_values_temp = cache.borrow_temporary(
        temporary_store,
        shape=(
            nnz,
            test.node_dof_count,
            trial.node_dof_count,
        ),
        dtype=output_dtype,
        device=device,
    )
    triplet_cols = triplet_cols_temp.array
    triplet_rows = triplet_rows_temp.array
    triplet_values = triplet_values_temp.array

    triplet_values.zero_()

    if nodal:
        wp.launch(
            kernel=kernel,
            dim=triplet_values.shape,
            inputs=[
                domain_elt_arg,
                domain_elt_index_arg,
                test_arg,
                test.space.topology.topo_arg_value(device),
                field_arg_values,
                value_struct_values,
                triplet_rows,
                triplet_cols,
                triplet_values,
            ],
            device=device,
        )
    elif isinstance(test, LocalTestField):
        local_result = cache.borrow_temporary(
            temporary_store=temporary_store,
            device=device,
            requires_grad=False,
            shape=(
                quadrature.evaluation_point_count(),
                test.value_dof_count,
                trial.value_dof_count,
                test.TAYLOR_DOF_COUNT * trial.TAYLOR_DOF_COUNT,
            ),
            dtype=float,
        )

        wp.launch(
            kernel=kernel,
            dim=(
                quadrature.evaluation_point_count(),
                test.value_dof_count,
                trial.value_dof_count,
                trial.TAYLOR_DOF_COUNT,
            ),
            inputs=[
                qp_arg,
                quadrature.element_index_arg_value(device),
                domain_elt_arg,
                domain_elt_index_arg,
                field_arg_values,
                value_struct_values,
                local_result.array,
            ],
            device=device,
        )

        vec_array_shape = (*local_result.array.shape[:-1], test.TAYLOR_DOF_COUNT)
        vec_array_dtype = cache.cached_vec_type(length=trial.TAYLOR_DOF_COUNT, dtype=float)
        local_result_as_vec = wp.array(
            data=None,
            ptr=local_result.array.ptr,
            capacity=local_result.array.capacity,
            device=local_result.array.device,
            shape=vec_array_shape,
            dtype=vec_array_dtype,
        )

        dispatch_kernel = make_bilinear_dispatch_kernel(test, trial, quadrature, accumulate_dtype)

        trial_partition_arg = trial.space_partition.partition_arg_value(device)
        trial_topology_arg = trial.space_partition.space_topology.topo_arg_value(device)
        wp.launch(
            kernel=dispatch_kernel,
            dim=(
                test.space_restriction.node_count(),
                test.node_dof_count,
                trial.node_dof_count,
                trial.space.topology.MAX_NODES_PER_ELEMENT,
            ),
            inputs=[
                qp_arg,
                domain_elt_arg,
                domain_elt_index_arg,
                test_arg,
                test.space.space_arg_value(device),
                trial_partition_arg,
                trial_topology_arg,
                trial.space.space_arg_value(device),
                local_result_as_vec,
                triplet_rows,
                triplet_cols,
                triplet_values,
            ],
            device=device,
        )

        local_result.release()

    else:
        trial_partition_arg = trial.space_partition.partition_arg_value(device)
        trial_topology_arg = trial.space_partition.space_topology.topo_arg_value(device)
        wp.launch(
            kernel=kernel,
            dim=(
                test.space_restriction.node_count(),
                trial.space.topology.MAX_NODES_PER_ELEMENT,
                test.node_dof_count,
                trial.node_dof_count,
            ),
            inputs=[
                qp_arg,
                domain_elt_arg,
                domain_elt_index_arg,
                test_arg,
                trial_partition_arg,
                trial_topology_arg,
                field_arg_values,
                value_struct_values,
                triplet_rows,
                triplet_cols,
                triplet_values,
            ],
            device=device,
        )

    if output is not None:
        if output.nrow != test.space_partition.node_count() or output.ncol != trial.space_partition.node_count():
            raise RuntimeError(
                f"Output matrix must have {test.space_partition.node_count()} rows and {trial.space_partition.node_count()} columns of blocks"
            )

    if output is None or add_to_output:
        bsr_result = bsr_zeros(
            rows_of_blocks=test.space_partition.node_count(),
            cols_of_blocks=trial.space_partition.node_count(),
            block_type=block_type,
            device=device,
        )
    else:
        bsr_result = output

    bsr_set_from_triplets(bsr_result, triplet_rows, triplet_cols, triplet_values, **(bsr_options or {}))

    # Do not wait for garbage collection
    triplet_values_temp.release()
    triplet_rows_temp.release()
    triplet_cols_temp.release()

    if add_to_output:
        output += bsr_result
    else:
        output = bsr_result

    return output


def _pick_assembly_strategy(
    assembly: Optional[str], nodal: bool, operators: Dict[str, Set[Operator]], arguments: IntegrandArguments
):
    if assembly is not None:
        if assembly not in ("generic", "nodal", "dispatch"):
            raise ValueError(f"Invalid assembly strategy'{assembly}'")
        return assembly
    elif nodal is not None:
        wp.utils.warn(
            "'nodal' argument of `warp.fem.integrate` is deprecated and will be removed in a future version. Please use `assembly='nodal'` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        if nodal:
            return "nodal"

    test_operators = operators.get(arguments.test_name, set())
    trial_operators = operators.get(arguments.trial_name, set())

    uses_virtual_node_operator = {operator.at_node, operator.node_count, operator.node_index} & (
        test_operators | trial_operators
    )

    return "generic" if uses_virtual_node_operator else "dispatch"


def integrate(
    integrand: Integrand,
    domain: Optional[GeometryDomain] = None,
    quadrature: Optional[Quadrature] = None,
    nodal: Optional[bool] = None,
    fields: Optional[Dict[str, FieldLike]] = None,
    values: Optional[Dict[str, Any]] = None,
    accumulate_dtype: type = wp.float64,
    output_dtype: Optional[type] = None,
    output: Optional[Union[BsrMatrix, wp.array]] = None,
    device=None,
    temporary_store: Optional[cache.TemporaryStore] = None,
    kernel_options: Optional[Dict[str, Any]] = None,
    assembly: Optional[str] = None,
    add: bool = False,
    bsr_options: Optional[Dict[str, Any]] = None,
):
    """
    Integrates a constant, linear or bilinear form, and returns a scalar, array, or sparse matrix, respectively.

    Args:
        integrand: Form to be integrated, must have :func:`integrand` decorator
        domain: Integration domain. If None, deduced from fields
        quadrature: Quadrature formula. If None, deduced from domain and fields degree.
        nodal: Deprecated. Use the equivalent assembly="nodal" instead.
        fields: Discrete, test, and trial fields to be passed to the integrand. Keys in the dictionary must match integrand parameter names.
        values: Additional variable values to be passed to the integrand, can be of any type accepted by warp kernel launches. Keys in the dictionary must match integrand parameter names.
        temporary_store: shared pool from which to allocate temporary arrays
        accumulate_dtype: Scalar type to be used for accumulating integration samples
        output: Sparse matrix or warp array into which to store the result of the integration
        output_dtype: Scalar type for returned results in `output` is not provided. If None, defaults to `accumulate_dtype`
        device: Device on which to perform the integration
        kernel_options: Overloaded options to be passed to the kernel builder (e.g, ``{"enable_backward": True}``)
        assembly: Specifies the strategy for assembling the integrated vector or matrix:
            - "nodal": For linear or bilinear forms, use the test function nodes as the quadrature points. Assumes Lagrange interpolation functions are used, and no differential or DG operator is evaluated on the test or trial functions.
            - "generic": Single-pass integration and shape-function evaluation. Makes no assumption about the integrand's content, but may lead to many redundant computations.
            - "dispatch": For linear or bilinear forms, first evaluate the form at quadrature points then dispatch to nodes in a second pass. More efficient for integrands that are expensive to evaluate. Incompatible with `at_node` and `node_index` operators on test or trial functions.
            - `None` (default): Automatically picks a suitable assembly strategy (either "generic" or "dispatch")
        add: If True and `output` is provided, add the integration result to `output` instead of replacing its content
        bsr_options: Additional options to be passed to the sparse matrix construction algorithm. See :func:`warp.sparse.bsr_set_from_triplets()`
    """
    if fields is None:
        fields = {}

    if values is None:
        values = {}

    if not isinstance(integrand, Integrand):
        raise ValueError("integrand must be tagged with @warp.fem.integrand decorator")

    # test, test_name, trial, trial_name = _get_test_and_trial_fields(fields)
    arguments = _parse_integrand_arguments(integrand, fields)

    test = None
    if arguments.test_name:
        test = arguments.field_args[arguments.test_name]
    trial = None
    if arguments.trial_name:
        if test is None:
            raise ValueError("A trial field cannot be provided without a test field")
        trial = arguments.field_args[arguments.trial_name]
        if test.domain != trial.domain:
            raise ValueError("Incompatible test and trial domains")

    if domain is None:
        if quadrature is not None:
            domain = quadrature.domain
        elif test is not None:
            domain = test.domain

    if domain is None:
        raise ValueError("Must provide at least one of domain, quadrature, or test field")
    if test is not None and domain != test.domain:
        raise NotImplementedError("Mixing integration and test domain is not supported yet")

    if add and output is None:
        raise ValueError("An 'output' array or matrix needs to be provided for add=True")

    if arguments.domain_name is not None:
        arguments.field_args[arguments.domain_name] = domain

    _find_integrand_operators(integrand, arguments.field_args)

    if operator.lookup in integrand.operators.get(arguments.domain_name, []) and not domain.supports_lookup(device):
        wp.utils.warn(f"{integrand.name}: using lookup() operator on a domain that does not support it")

    assembly = _pick_assembly_strategy(assembly, nodal, arguments=arguments, operators=integrand.operators)
    # print("assembly for ", integrand.name, ":", strategy)

    if assembly == "dispatch":
        if test is not None:
            test = LocalTestField(test)
            arguments.field_args[arguments.test_name] = test
        if trial is not None:
            trial = LocalTrialField(trial)
            arguments.field_args[arguments.trial_name] = trial

    if assembly == "nodal":
        if quadrature is not None:
            raise ValueError("Cannot specify quadrature for nodal integration")

        if test is None:
            raise ValueError("Nodal integration requires specifying a test function")

        if trial is not None and test.space_partition != trial.space_partition:
            raise ValueError(
                "Bilinear nodal integration requires test and trial to be defined on the same function space"
            )
    else:
        if quadrature is None:
            order = sum(field.degree for field in fields.values())
            quadrature = RegularQuadrature(domain=domain, order=order)
        elif domain != quadrature.domain:
            raise ValueError("Incompatible integration and quadrature domain")

    # Canonicalize types
    accumulate_dtype = wp.types.type_to_warp(accumulate_dtype)
    if output is not None:
        if isinstance(output, BsrMatrix):
            output_dtype = output.scalar_type
        else:
            output_dtype = output.dtype
    elif output_dtype is None:
        output_dtype = accumulate_dtype
    else:
        output_dtype = wp.types.type_to_warp(output_dtype)

    kernel, FieldStruct, ValueStruct = _generate_integrate_kernel(
        integrand=integrand,
        domain=domain,
        quadrature=quadrature,
        arguments=arguments,
        test=test,
        trial=trial,
        accumulate_dtype=accumulate_dtype,
        output_dtype=output_dtype,
        kernel_options=kernel_options,
    )

    return _launch_integrate_kernel(
        integrand=integrand,
        kernel=kernel,
        FieldStruct=FieldStruct,
        ValueStruct=ValueStruct,
        domain=domain,
        quadrature=quadrature,
        test=test,
        trial=trial,
        fields=arguments.field_args,
        values=values,
        accumulate_dtype=accumulate_dtype,
        temporary_store=temporary_store,
        output_dtype=output_dtype,
        output=output,
        add_to_output=add,
        bsr_options=bsr_options,
        device=device,
    )


def get_interpolate_to_field_function(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    dest: FieldRestriction,
):
    zero_value = type_zero_element(dest.space.dtype)

    def interpolate_to_field_fn(
        local_node_index: int,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        dest_node_arg: dest.space_restriction.NodeArg,
        dest_eval_arg: dest.field.EvalArg,
        fields: FieldStruct,
        values: ValueStruct,
    ):
        partition_node_index = dest.space_restriction.node_partition_index(dest_node_arg, local_node_index)
        element_beg, element_end = dest.space_restriction.node_element_range(dest_node_arg, partition_node_index)

        test_dof_index = NULL_DOF_INDEX
        trial_dof_index = NULL_DOF_INDEX
        node_weight = 1.0

        # Volume-weighted average across elements
        # Superfluous if the interpolated function is continuous, but helpful for visualizing discontinuous spaces

        val_sum = zero_value()
        vol_sum = float(0.0)

        for n in range(element_beg, element_end):
            node_element_index = dest.space_restriction.node_element_index(dest_node_arg, n)
            element_index = domain.element_index(domain_index_arg, node_element_index.domain_element_index)

            if n == element_beg:
                node_index = dest.space.topology.element_node_index(
                    domain_arg, dest_eval_arg.topology_arg, element_index, node_element_index.node_index_in_element
                )

            coords = dest.space.node_coords_in_element(
                domain_arg,
                dest_eval_arg.space_arg,
                element_index,
                node_element_index.node_index_in_element,
            )

            if coords[0] != OUTSIDE:
                sample = Sample(
                    element_index,
                    coords,
                    node_index,
                    node_weight,
                    test_dof_index,
                    trial_dof_index,
                )
                vol = domain.element_measure(domain_arg, sample)
                val = integrand_func(sample, fields, values)

                vol_sum += vol
                val_sum += vol * val

        return val_sum, vol_sum

    return interpolate_to_field_fn


def get_interpolate_to_field_kernel(
    interpolate_to_field_fn: wp.Function,
    domain: GeometryDomain,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    dest: FieldRestriction,
):
    @wp.func
    def _find_node_in_element(
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        dest_node_arg: dest.space_restriction.NodeArg,
        dest_eval_arg: dest.field.EvalArg,
        partition_node_index: int,
    ):
        element_beg, element_end = dest.space_restriction.node_element_range(dest_node_arg, partition_node_index)

        for n in range(element_beg, element_end):
            node_element_index = dest.space_restriction.node_element_index(dest_node_arg, n)
            element_index = domain.element_index(domain_index_arg, node_element_index.domain_element_index)
            coords = dest.space.node_coords_in_element(
                domain_arg,
                dest_eval_arg.space_arg,
                element_index,
                node_element_index.node_index_in_element,
            )
            if coords[0] != OUTSIDE:
                return element_index, node_element_index.node_index_in_element

        return NULL_ELEMENT_INDEX, NULL_NODE_INDEX

    def interpolate_to_field_kernel_fn(
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        dest_node_arg: dest.space_restriction.NodeArg,
        dest_eval_arg: dest.field.EvalArg,
        fields: FieldStruct,
        values: ValueStruct,
    ):
        local_node_index = wp.tid()

        val_sum, vol_sum = interpolate_to_field_fn(
            local_node_index, domain_arg, domain_index_arg, dest_node_arg, dest_eval_arg, fields, values
        )

        if vol_sum > 0.0:
            partition_node_index = dest.space_restriction.node_partition_index(dest_node_arg, local_node_index)

            # Grab first element containing node; there must be at least one since vol_sum != 0
            element_index, node_index_in_element = _find_node_in_element(
                domain_arg, domain_index_arg, dest_node_arg, dest_eval_arg, partition_node_index
            )
            dest.field.set_node_value(
                domain_arg,
                dest_eval_arg,
                element_index,
                node_index_in_element,
                partition_node_index,
                val_sum / vol_sum,
            )

    return interpolate_to_field_kernel_fn


def get_interpolate_at_quadrature_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    value_type: type,
):
    def interpolate_at_quadrature_nonvalued_kernel_fn(
        qp_arg: quadrature.Arg,
        qp_element_index_arg: quadrature.ElementIndexArg,
        domain_arg: quadrature.domain.ElementArg,
        domain_index_arg: quadrature.domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array(dtype=float),
    ):
        qp_eval_index = wp.tid()
        domain_element_index, qp = quadrature.evaluation_point_element_index(qp_element_index_arg, qp_eval_index)
        if domain_element_index == NULL_ELEMENT_INDEX:
            return

        element_index = domain.element_index(domain_index_arg, domain_element_index)

        test_dof_index = NULL_DOF_INDEX
        trial_dof_index = NULL_DOF_INDEX

        coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)

        sample = Sample(element_index, coords, qp_index, qp_weight, test_dof_index, trial_dof_index)
        integrand_func(sample, fields, values)

    def interpolate_at_quadrature_kernel_fn(
        qp_arg: quadrature.Arg,
        qp_element_index_arg: quadrature.ElementIndexArg,
        domain_arg: quadrature.domain.ElementArg,
        domain_index_arg: quadrature.domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array(dtype=value_type),
    ):
        qp_eval_index = wp.tid()
        domain_element_index, qp = quadrature.evaluation_point_element_index(qp_element_index_arg, qp_eval_index)
        if domain_element_index == NULL_ELEMENT_INDEX:
            return

        element_index = domain.element_index(domain_index_arg, domain_element_index)

        test_dof_index = NULL_DOF_INDEX
        trial_dof_index = NULL_DOF_INDEX

        coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)

        sample = Sample(element_index, coords, qp_index, qp_weight, test_dof_index, trial_dof_index)
        result[qp_index] = integrand_func(sample, fields, values)

    return interpolate_at_quadrature_nonvalued_kernel_fn if value_type is None else interpolate_at_quadrature_kernel_fn


def get_interpolate_jacobian_at_quadrature_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    trial: TrialField,
    value_size: int,
    value_type: type,
):
    MAX_NODES_PER_ELEMENT = trial.space.topology.MAX_NODES_PER_ELEMENT
    VALUE_SIZE = wp.constant(value_size)

    def interpolate_jacobian_kernel_fn(
        qp_arg: quadrature.Arg,
        qp_element_index_arg: quadrature.ElementIndexArg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        trial_partition_arg: trial.space_partition.PartitionArg,
        trial_topology_arg: trial.space_partition.space_topology.TopologyArg,
        fields: FieldStruct,
        values: ValueStruct,
        triplet_rows: wp.array(dtype=int),
        triplet_cols: wp.array(dtype=int),
        triplet_values: wp.array3d(dtype=value_type),
    ):
        qp_eval_index, trial_node, trial_dof = wp.tid()
        domain_element_index, qp = quadrature.evaluation_point_element_index(qp_element_index_arg, qp_eval_index)

        if domain_element_index == NULL_ELEMENT_INDEX:
            return

        element_index = domain.element_index(domain_index_arg, domain_element_index)
        if qp >= quadrature.point_count(domain_arg, qp_arg, domain_element_index, element_index):
            return

        element_trial_node_count = trial.space.topology.element_node_count(
            domain_arg, trial_topology_arg, element_index
        )

        qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)
        qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)

        block_offset = qp_index * MAX_NODES_PER_ELEMENT + trial_node

        test_dof_index = NULL_DOF_INDEX
        trial_dof_index = DofIndex(trial_node, trial_dof)

        sample = Sample(element_index, qp_coords, qp_index, qp_weight, test_dof_index, trial_dof_index)
        val = integrand_func(sample, fields, values)

        for k in range(VALUE_SIZE):
            triplet_values[block_offset, k, trial_dof] = basis_coefficient(val, k)

        if trial_dof == 0:
            if trial_node < element_trial_node_count:
                trial_node_index = trial.space_partition.partition_node_index(
                    trial_partition_arg,
                    trial.space.topology.element_node_index(domain_arg, trial_topology_arg, element_index, trial_node),
                )
            else:
                trial_node_index = NULL_NODE_INDEX  # will get ignored when converting to bsr
            triplet_rows[block_offset] = qp_index
            triplet_cols[block_offset] = trial_node_index

    return interpolate_jacobian_kernel_fn


def get_interpolate_free_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    value_type: type,
):
    def interpolate_free_nonvalued_kernel_fn(
        dim: int,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array(dtype=float),
    ):
        qp_index = wp.tid()
        qp_weight = 1.0 / float(dim)
        element_index = NULL_ELEMENT_INDEX
        coords = Coords(OUTSIDE)

        test_dof_index = NULL_DOF_INDEX
        trial_dof_index = NULL_DOF_INDEX

        sample = Sample(element_index, coords, qp_index, qp_weight, test_dof_index, trial_dof_index)
        integrand_func(sample, fields, values)

    def interpolate_free_kernel_fn(
        dim: int,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array(dtype=value_type),
    ):
        qp_index = wp.tid()
        qp_weight = 1.0 / float(dim)
        element_index = NULL_ELEMENT_INDEX
        coords = Coords(OUTSIDE)

        test_dof_index = NULL_DOF_INDEX
        trial_dof_index = NULL_DOF_INDEX

        sample = Sample(element_index, coords, qp_index, qp_weight, test_dof_index, trial_dof_index)

        result[qp_index] = integrand_func(sample, fields, values)

    return interpolate_free_nonvalued_kernel_fn if value_type is None else interpolate_free_kernel_fn


def _generate_interpolate_kernel(
    integrand: Integrand,
    domain: GeometryDomain,
    dest: Optional[Union[FieldLike, wp.array]],
    quadrature: Optional[Quadrature],
    arguments: IntegrandArguments,
    kernel_options: Optional[Dict[str, Any]] = None,
) -> wp.Kernel:
    # Generate field struct
    FieldStruct = _gen_field_struct(arguments.field_args)
    ValueStruct = cache.get_argument_struct(arguments.value_args)

    _notify_operator_usage(integrand, arguments.field_args)

    # Check if kernel exist in cache
    field_names = "_".join(f"{k}{f.name}" for k, f in arguments.field_args.items())
    if isinstance(dest, FieldRestriction):
        kernel_suffix = f"_itp_{field_names}_{dest.domain.name}_{dest.space_restriction.space_partition.name}"
    else:
        dest_dtype = dest.dtype if dest else None
        type_str = wp.types.get_type_code(dest_dtype) if dest_dtype else ""
        if quadrature is None:
            kernel_suffix = f"_itp_{field_names}_{domain.name}_{type_str}"
        else:
            kernel_suffix = f"_itp_{field_names}_{domain.name}_{quadrature.name}_{type_str}"

    kernel = cache.get_integrand_kernel(
        integrand=integrand,
        suffix=kernel_suffix,
        kernel_options=kernel_options,
    )
    if kernel is not None:
        return kernel, FieldStruct, ValueStruct

    # Not found in cache, transform integrand and generate kernel
    _check_field_compat(integrand, arguments, domain)

    integrand_func = IntegrandTransformer.apply(integrand, arguments.field_args)

    # Generate interpolation kernel
    if isinstance(dest, FieldRestriction):
        # need to split into kernel + function for differentiability
        interpolate_fn = get_interpolate_to_field_function(
            integrand_func,
            domain,
            dest=dest,
            FieldStruct=FieldStruct,
            ValueStruct=ValueStruct,
        )

        interpolate_fn = cache.get_integrand_function(
            integrand=integrand,
            func=interpolate_fn,
            suffix=kernel_suffix,
            code_transformers=[
                PassFieldArgsToIntegrand(
                    arg_names=integrand.argspec.args, parsed_args=arguments, integrand_func=integrand_func
                )
            ],
        )

        interpolate_kernel_fn = get_interpolate_to_field_kernel(
            interpolate_fn,
            domain,
            dest=dest,
            FieldStruct=FieldStruct,
            ValueStruct=ValueStruct,
        )
    elif quadrature is not None:
        if arguments.trial_name:
            trial = arguments.field_args[arguments.trial_name]
            interpolate_kernel_fn = get_interpolate_jacobian_at_quadrature_kernel(
                integrand_func,
                domain=domain,
                quadrature=quadrature,
                FieldStruct=FieldStruct,
                ValueStruct=ValueStruct,
                trial=trial,
                value_size=dest.block_shape[0],
                value_type=dest.scalar_type,
            )
        else:
            interpolate_kernel_fn = get_interpolate_at_quadrature_kernel(
                integrand_func,
                domain=domain,
                quadrature=quadrature,
                value_type=dest_dtype,
                FieldStruct=FieldStruct,
                ValueStruct=ValueStruct,
            )
    else:
        interpolate_kernel_fn = get_interpolate_free_kernel(
            integrand_func,
            domain=domain,
            value_type=dest_dtype,
            FieldStruct=FieldStruct,
            ValueStruct=ValueStruct,
        )

    kernel = cache.get_integrand_kernel(
        integrand=integrand,
        kernel_fn=interpolate_kernel_fn,
        suffix=kernel_suffix,
        kernel_options=kernel_options,
        code_transformers=[
            PassFieldArgsToIntegrand(
                arg_names=integrand.argspec.args, parsed_args=arguments, integrand_func=integrand_func
            )
        ],
    )

    return kernel, FieldStruct, ValueStruct


def _launch_interpolate_kernel(
    integrand: Integrand,
    kernel: wp.kernel,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    domain: GeometryDomain,
    dest: Optional[Union[FieldRestriction, wp.array]],
    quadrature: Optional[Quadrature],
    dim: int,
    trial: Optional[TrialField],
    fields: Dict[str, FieldLike],
    values: Dict[str, Any],
    temporary_store: Optional[cache.TemporaryStore],
    bsr_options: Optional[Dict[str, Any]],
    device,
) -> wp.Kernel:
    # Set-up launch arguments
    elt_arg = domain.element_arg_value(device=device)
    elt_index_arg = domain.element_index_arg_value(device=device)

    field_arg_values = FieldStruct()
    for k, v in fields.items():
        if not isinstance(v, GeometryDomain):
            setattr(field_arg_values, k, v.eval_arg_value(device=device))

    value_struct_values = cache.populate_argument_struct(ValueStruct, values, func_name=integrand.name)

    if isinstance(dest, FieldRestriction):
        dest_node_arg = dest.space_restriction.node_arg(device=device)
        dest_eval_arg = dest.field.eval_arg_value(device=device)

        wp.launch(
            kernel=kernel,
            dim=dest.space_restriction.node_count(),
            inputs=[
                elt_arg,
                elt_index_arg,
                dest_node_arg,
                dest_eval_arg,
                field_arg_values,
                value_struct_values,
            ],
            device=device,
        )
        return

    if quadrature is None:
        wp.launch(
            kernel=kernel,
            dim=dim,
            inputs=[dim, elt_arg, elt_index_arg, field_arg_values, value_struct_values, dest],
            device=device,
        )
        return

    qp_arg = quadrature.arg_value(device)
    qp_element_index_arg = quadrature.element_index_arg_value(device)
    if trial is None:
        wp.launch(
            kernel=kernel,
            dim=quadrature.evaluation_point_count(),
            inputs=[qp_arg, qp_element_index_arg, elt_arg, elt_index_arg, field_arg_values, value_struct_values, dest],
            device=device,
        )
        return

    nnz = quadrature.total_point_count() * trial.space.topology.MAX_NODES_PER_ELEMENT

    if dest.nrow != quadrature.total_point_count() or dest.ncol != trial.space_partition.node_count():
        raise RuntimeError(
            f"'dest' matrix must have {quadrature.total_point_count()} rows and {trial.space_partition.node_count()} columns of blocks"
        )
    if dest.block_shape[1] != trial.node_dof_count:
        raise f"'dest' matrix blocks must have {trial.node_dof_count} columns"

    triplet_rows_temp = cache.borrow_temporary(temporary_store, shape=(nnz,), dtype=int, device=device)
    triplet_cols_temp = cache.borrow_temporary(temporary_store, shape=(nnz,), dtype=int, device=device)
    triplet_values_temp = cache.borrow_temporary(
        temporary_store,
        dtype=dest.scalar_type,
        shape=(nnz, *dest.block_shape),
        device=device,
    )
    triplet_cols = triplet_cols_temp.array
    triplet_rows = triplet_rows_temp.array
    triplet_values = triplet_values_temp.array
    triplet_rows.fill_(-1)
    triplet_values.zero_()

    trial_partition_arg = trial.space_partition.partition_arg_value(device)
    trial_topology_arg = trial.space_partition.space_topology.topo_arg_value(device)

    wp.launch(
        kernel=kernel,
        dim=(quadrature.evaluation_point_count(), trial.space.topology.MAX_NODES_PER_ELEMENT, trial.node_dof_count),
        inputs=[
            qp_arg,
            qp_element_index_arg,
            elt_arg,
            elt_index_arg,
            trial_partition_arg,
            trial_topology_arg,
            field_arg_values,
            value_struct_values,
            triplet_rows,
            triplet_cols,
            triplet_values,
        ],
        device=device,
    )

    bsr_set_from_triplets(dest, triplet_rows, triplet_cols, triplet_values, **(bsr_options or {}))


@integrand
def _identity_field(field: Field, s: Sample):
    return field(s)


def interpolate(
    integrand: Union[Integrand, FieldLike],
    dest: Optional[Union[DiscreteField, FieldRestriction, wp.array]] = None,
    quadrature: Optional[Quadrature] = None,
    dim: Optional[int] = None,
    domain: Optional[Domain] = None,
    fields: Optional[Dict[str, FieldLike]] = None,
    values: Optional[Dict[str, Any]] = None,
    device=None,
    kernel_options: Optional[Dict[str, Any]] = None,
    temporary_store: Optional[cache.TemporaryStore] = None,
    bsr_options: Optional[Dict[str, Any]] = None,
):
    """
    Interpolates a function at a finite set of sample points and optionally assigns the result to a discrete field or a raw warp array.

    Args:
        integrand: Function to be interpolated: either a function with :func:`warp.fem.integrand` decorator or a field
        dest: Where to store the interpolation result. Can be either

         - a :class:`DiscreteField`, or restriction of a discrete field to a domain (from :func:`make_restriction`). In this case, interpolation will be performed at each node.
         - a normal warp ``array``, or ``None``. In this case, the interpolation samples will determined by the `quadrature` or `dim` arguments, in that order.
        quadrature: Quadrature formula defining the interpolation samples if `dest` is not a discrete field or field restriction.
        dim: Number of interpolation samples if `dest` is not a discrete field or restriction and `quadrature` is ``None``.
          In this case, the ``Sample`` passed to the `integrand` will be invalid, but the sample point index ``s.qp_index`` can be used to define custom interpolation logic.
        domain: Interpolation domain, only used if `dest` is not a field restriction and `quadrature` is ``None``
        fields: Discrete fields to be passed to the integrand. Keys in the dictionary must match integrand parameters names.
        values: Additional variable values to be passed to the integrand, can be of any type accepted by warp kernel launches. Keys in the dictionary must match integrand parameter names.
        device: Device on which to perform the interpolation
        kernel_options: Overloaded options to be passed to the kernel builder (e.g, ``{"enable_backward": True}``)
        temporary_store: shared pool from which to allocate temporary arrays
        bsr_options: Additional options to be passed to the sparse matrix construction algorithm. See :func:`warp.sparse.bsr_set_from_triplets()`
    """

    if isinstance(integrand, FieldLike):
        fields = {"field": integrand}
        values = {}
        integrand = _identity_field

    if fields is None:
        fields = {}

    if values is None:
        values = {}

    if not isinstance(integrand, Integrand):
        raise ValueError("integrand must be tagged with @integrand decorator")

    arguments = _parse_integrand_arguments(integrand, fields)
    if arguments.test_name:
        raise ValueError(f"Test field '{arguments.test_name}' maybe not be used for interpolation")
    if arguments.trial_name and not isinstance(dest, BsrMatrix):
        raise ValueError(
            f"Interpolation using trial field '{arguments.trial_name}' requires 'dest' to be a `warp.sparse.BsrMatrix`"
        )

    trial = arguments.field_args.get(arguments.trial_name, None)

    if isinstance(dest, DiscreteField):
        dest = make_restriction(dest, domain=domain)

    if isinstance(dest, FieldRestriction):
        domain = dest.domain
    elif quadrature is not None:
        domain = quadrature.domain
    elif dim is None:
        if trial is not None:
            domain = trial.domain
        elif domain is None:
            raise ValueError(
                "Unable to determine interpolation domain, provide an explicit field restriction or quadrature"
            )

        # Default to one sample per domain element
        quadrature = RegularQuadrature(domain, order=0)

    if arguments.domain_name:
        arguments.field_args[arguments.domain_name] = domain

    _find_integrand_operators(integrand, arguments.field_args)

    if operator.lookup in integrand.operators.get(arguments.domain_name, []) and not domain.supports_lookup(device):
        wp.utils.warn(f"{integrand.name}: using lookup() operator on a domain that does not support it")

    kernel, FieldStruct, ValueStruct = _generate_interpolate_kernel(
        integrand=integrand,
        domain=domain,
        dest=dest,
        quadrature=quadrature,
        arguments=arguments,
        kernel_options=kernel_options,
    )

    return _launch_interpolate_kernel(
        integrand=integrand,
        kernel=kernel,
        FieldStruct=FieldStruct,
        ValueStruct=ValueStruct,
        domain=domain,
        dest=dest,
        quadrature=quadrature,
        dim=dim,
        trial=trial,
        fields=arguments.field_args,
        values=values,
        temporary_store=temporary_store,
        bsr_options=bsr_options,
        device=device,
    )
