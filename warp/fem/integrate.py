import ast
from typing import Any, Dict, List, Optional, Set, Union

import warp as wp
from warp.codegen import get_annotations
from warp.fem import cache
from warp.fem.domain import GeometryDomain
from warp.fem.field import (
    DiscreteField,
    FieldLike,
    FieldRestriction,
    SpaceField,
    TestField,
    TrialField,
    make_restriction,
)
from warp.fem.operator import Integrand, Operator
from warp.fem.quadrature import Quadrature, RegularQuadrature
from warp.fem.types import NULL_DOF_INDEX, NULL_NODE_INDEX, OUTSIDE, DofIndex, Domain, Field, Sample, make_free_sample
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


class IntegrandTransformer(ast.NodeTransformer):
    def __init__(self, integrand: Integrand, field_args: Dict[str, FieldLike], annotations: Dict[str, Any]):
        self._integrand = integrand
        self._field_args = field_args
        self._annotations = annotations

    def visit_Call(self, call: ast.Call):
        call = self.generic_visit(call)

        callee = getattr(call.func, "id", None)
        if callee in self._field_args:
            # Shortcut for evaluating fields as f(x...)
            field = self._field_args[callee]

            # Replace with default call operator
            abstract_arg_type = self._integrand.argspec.annotations[callee]
            default_operator = abstract_arg_type.call_operator
            concrete_arg_type = self._annotations[callee]
            self._replace_call_func(call, concrete_arg_type, default_operator, field)

            # insert callee as first argument
            call.args = [ast.Name(id=callee, ctx=ast.Load())] + call.args

            return call

        func, _ = _resolve_path(self._integrand.func, call.func)

        if isinstance(func, Operator) and len(call.args) > 0:
            # Evaluating operators as op(field, x, ...)
            callee = getattr(call.args[0], "id", None)
            if callee in self._field_args:
                field = self._field_args[callee]
                self._replace_call_func(call, func, func, field)

        if isinstance(func, Integrand):
            key = self._translate_callee(func, call.args)
            call.func = ast.Attribute(
                value=call.func,
                attr=key,
                ctx=ast.Load(),
            )

        # print(ast.dump(call, indent=4))

        return call

    def _replace_call_func(self, call: ast.Call, callee: Union[type, Operator], operator: Operator, field: FieldLike):
        try:
            # Retrieve the function pointer corresponding to the operator implementation for the field type
            pointer = operator.resolver(field)
        except AttributeError as e:
            raise ValueError(f"Operator {operator.func.__name__} is not defined for field {field.name}") from e
        # Save the pointer as an attribute than can be accessed from the callee scope
        setattr(callee, pointer.key, pointer)
        # Update the ast Call node to use the new function pointer
        call.func = ast.Attribute(value=call.func, attr=pointer.key, ctx=ast.Load())

    def _translate_callee(self, callee: Integrand, args: List[ast.AST]):
        # Get field types for call site arguments
        call_site_field_args = []
        for arg in args:
            name = getattr(arg, "id", None)
            if name in self._field_args:
                call_site_field_args.append(self._field_args[name])

        call_site_field_args.reverse()

        # Pass to callee in same order
        callee_field_args = {}
        for arg in callee.argspec.args:
            arg_type = callee.argspec.annotations[arg]
            if arg_type in (Field, Domain):
                callee_field_args[arg] = call_site_field_args.pop()

        return _translate_integrand(callee, callee_field_args).key


def _translate_integrand(integrand: Integrand, field_args: Dict[str, FieldLike]) -> wp.Function:
    # Specialize field argument types
    argspec = integrand.argspec
    annotations = {}
    for arg in argspec.args:
        arg_type = argspec.annotations[arg]
        if arg_type == Field:
            annotations[arg] = field_args[arg].ElementEvalArg
        elif arg_type == Domain:
            annotations[arg] = field_args[arg].ElementArg
        else:
            annotations[arg] = arg_type

    # Transform field evaluation calls
    transformer = IntegrandTransformer(integrand, field_args, annotations)

    suffix = "_".join([f.name for f in field_args.values()])

    func = cache.get_integrand_function(
        integrand=integrand,
        suffix=suffix,
        annotations=annotations,
        code_transformers=[transformer],
    )

    key = func.key
    setattr(integrand, key, integrand.module.functions[key])

    return getattr(integrand, key)


def _get_integrand_field_arguments(
    integrand: Integrand,
    fields: Dict[str, FieldLike],
    domain: GeometryDomain = None,
):
    # parse argument types
    field_args = {}
    value_args = {}

    domain_name = None
    sample_name = None

    argspec = integrand.argspec
    for arg in argspec.args:
        arg_type = argspec.annotations[arg]
        if arg_type == Field:
            if arg not in fields:
                raise ValueError(f"Missing field for argument '{arg}' of integrand '{integrand.name}'")
            field_args[arg] = fields[arg]
        elif arg_type == Domain:
            domain_name = arg
            field_args[arg] = domain
        elif arg_type == Sample:
            sample_name = arg
        else:
            value_args[arg] = arg_type

    return field_args, value_args, domain_name, sample_name


def _check_field_compat(
    integrand: Integrand,
    fields: Dict[str, FieldLike],
    field_args: Dict[str, FieldLike],
    domain: GeometryDomain = None,
):
    # Check field compatibility
    for name, field in fields.items():
        if name not in field_args:
            raise ValueError(
                f"Passed field argument '{name}' does not match any parameter of integrand '{integrand.name}'"
            )

        if isinstance(field, SpaceField) and domain is not None:
            space = field.space
            if space.geometry != domain.geometry:
                raise ValueError(f"Field '{name}' must be defined on the same geometry as the integration domain")
            if space.dimension != domain.dimension:
                raise ValueError(
                    f"Field '{name}' dimension ({space.dimension}) does not match that of the integration domain ({domain.dimension}). Maybe a forgotten `.trace()`?"
                )


def _populate_value_struct(ValueStruct: wp.codegen.Struct, values: Dict[str, Any], integrand_name: str):
    value_struct_values = ValueStruct()
    for k, v in values.items():
        try:
            setattr(value_struct_values, k, v)
        except Exception as err:
            if k not in ValueStruct.vars:
                raise ValueError(
                    f"Passed value argument '{k}' does not match any of the integrand '{integrand_name}' parameters"
                ) from err
            raise ValueError(
                f"Passed value argument '{k}' of type '{wp.types.type_repr(v)}' is incompatible with the integrand '{integrand_name}' parameter of type '{wp.types.type_repr(ValueStruct.vars[k].type)}'"
            ) from err

    missing_values = ValueStruct.vars.keys() - values.keys()
    if missing_values:
        wp.utils.warn(
            f"Missing values for parameter(s) '{', '.join(missing_values)}' of the integrand '{integrand_name}', will be zero-initialized"
        )

    return value_struct_values


def _get_test_and_trial_fields(
    fields: Dict[str, FieldLike],
):
    test = None
    trial = None
    test_name = None
    trial_name = None

    for name, field in fields.items():
        if not isinstance(field, FieldLike):
            raise ValueError(f"Passed field argument '{name}' is not a proper Field")

        if isinstance(field, TestField):
            if test is not None:
                raise ValueError(f"More than one test field argument: '{test_name}' and '{name}'")
            test = field
            test_name = name
        elif isinstance(field, TrialField):
            if trial is not None:
                raise ValueError(f"More than one trial field argument: '{trial_name}' and '{name}'")
            trial = field
            trial_name = name

    if trial is not None:
        if test is None:
            raise ValueError("A trial field cannot be provided without a test field")

        if test.domain != trial.domain:
            raise ValueError("Incompatible test and trial domains")

    return test, test_name, trial, trial_name


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


def _gen_value_struct(value_args: Dict[str, type]):
    class Values:
        pass

    annotations = get_annotations(Values)

    for name, arg_type in value_args.items():
        setattr(Values, name, None)
        annotations[name] = arg_type

    def arg_type_name(arg_type):
        if isinstance(arg_type, wp.codegen.Struct):
            return arg_type_name(arg_type.cls)
        return getattr(arg_type, "__name__", str(arg_type))

    def arg_type_name(arg_type):
        if isinstance(arg_type, wp.codegen.Struct):
            return arg_type_name(arg_type.cls)
        return getattr(arg_type, "__name__", str(arg_type))

    try:
        Values.__annotations__ = annotations
    except AttributeError:
        Values.__dict__.__annotations__ = annotations

    suffix = "_".join([f"{name}_{arg_type_name(arg_type)}" for name, arg_type in annotations.items()])

    return cache.get_struct(Values, suffix=suffix)


def _get_trial_arg():
    pass


def _get_test_arg():
    pass


class _FieldWrappers:
    pass


def _register_integrand_field_wrappers(integrand_func: wp.Function, fields: Dict[str, FieldLike]):
    integrand_func._field_wrappers = _FieldWrappers()
    for name, field in fields.items():
        setattr(integrand_func._field_wrappers, name, field.ElementEvalArg)


class PassFieldArgsToIntegrand(ast.NodeTransformer):
    def __init__(
        self,
        arg_names: List[str],
        field_args: Set[str],
        value_args: Set[str],
        sample_name: str,
        domain_name: str,
        test_name: str = None,
        trial_name: str = None,
        func_name: str = "integrand_func",
        fields_var_name: str = "fields",
        values_var_name: str = "values",
        domain_var_name: str = "domain_arg",
        sample_var_name: str = "sample",
        field_wrappers_attr: str = "_field_wrappers",
    ):
        self._arg_names = arg_names
        self._field_args = field_args
        self._value_args = value_args
        self._domain_name = domain_name
        self._sample_name = sample_name
        self._func_name = func_name
        self._test_name = test_name
        self._trial_name = trial_name
        self._fields_var_name = fields_var_name
        self._values_var_name = values_var_name
        self._domain_var_name = domain_var_name
        self._sample_var_name = sample_var_name
        self._field_wrappers_attr = field_wrappers_attr

    def visit_Call(self, call: ast.Call):
        call = self.generic_visit(call)

        callee = getattr(call.func, "id", None)

        if callee == self._func_name:
            # Replace function arguments with ours generated structs
            call.args.clear()
            for arg in self._arg_names:
                if arg == self._domain_name:
                    call.args.append(
                        ast.Name(id=self._domain_var_name, ctx=ast.Load()),
                    )
                elif arg == self._sample_name:
                    call.args.append(
                        ast.Name(id=self._sample_var_name, ctx=ast.Load()),
                    )
                elif arg in self._field_args:
                    call.args.append(
                        ast.Call(
                            func=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id=self._func_name, ctx=ast.Load()),
                                    attr=self._field_wrappers_attr,
                                    ctx=ast.Load(),
                                ),
                                attr=arg,
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Name(id=self._domain_var_name, ctx=ast.Load()),
                                ast.Attribute(
                                    value=ast.Name(id=self._fields_var_name, ctx=ast.Load()),
                                    attr=arg,
                                    ctx=ast.Load(),
                                ),
                            ],
                            keywords=[],
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
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array(dtype=accumulate_dtype),
    ):
        domain_element_index = wp.tid()
        element_index = domain.element_index(domain_index_arg, domain_element_index)
        elem_sum = accumulate_dtype(0.0)

        test_dof_index = NULL_DOF_INDEX
        trial_dof_index = NULL_DOF_INDEX

        qp_point_count = quadrature.point_count(domain_arg, qp_arg, domain_element_index, element_index)
        for k in range(qp_point_count):
            qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, k)
            coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, k)
            qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, k)

            sample = Sample(element_index, coords, qp_index, qp_weight, test_dof_index, trial_dof_index)
            vol = domain.element_measure(domain_arg, sample)

            val = integrand_func(sample, fields, values)

            elem_sum += accumulate_dtype(qp_weight * vol * val)

        wp.atomic_add(result, 0, elem_sum)

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

        result[node_index, test_dof] = output_dtype(val_sum)

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
                _get_test_arg(),
                element_index,
                node_element_index.node_index_in_element,
            )

            if coords[0] != OUTSIDE:
                node_weight = test.space.node_quadrature_weight(
                    domain_arg,
                    _get_test_arg(),
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

        result[partition_node_index, dof] = output_dtype(val_sum)

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
            qp_point_count = wp.select(
                trial_node < element_trial_node_count,
                0,
                quadrature.point_count(domain_arg, qp_arg, test_element_index.domain_element_index, element_index),
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
                _get_test_arg(),
                element_index,
                node_element_index.node_index_in_element,
            )

            if coords[0] != OUTSIDE:
                node_weight = test.space.node_quadrature_weight(
                    domain_arg,
                    _get_test_arg(),
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


def _generate_integrate_kernel(
    integrand: Integrand,
    domain: GeometryDomain,
    nodal: bool,
    quadrature: Quadrature,
    test: Optional[TestField],
    test_name: str,
    trial: Optional[TrialField],
    trial_name: str,
    fields: Dict[str, FieldLike],
    output_dtype: type,
    accumulate_dtype: type,
    kernel_options: Optional[Dict[str, Any]] = None,
) -> wp.Kernel:
    if kernel_options is None:
        kernel_options = {}

    output_dtype = wp.types.type_scalar_type(output_dtype)

    # Extract field arguments from integrand
    field_args, value_args, domain_name, sample_name = _get_integrand_field_arguments(
        integrand, fields=fields, domain=domain
    )

    FieldStruct = _gen_field_struct(field_args)
    ValueStruct = _gen_value_struct(value_args)

    # Check if kernel exist in cache
    kernel_suffix = f"_itg_{wp.types.type_typestr(output_dtype)}{wp.types.type_typestr(accumulate_dtype)}_{domain.name}_{FieldStruct.key}"
    if nodal:
        kernel_suffix += "_nodal"
    else:
        kernel_suffix += quadrature.name

    if test:
        kernel_suffix += f"_test_{test.space_partition.name}_{test.space.name}"
    if trial:
        kernel_suffix += f"_trial_{trial.space_partition.name}_{trial.space.name}"

    kernel = cache.get_integrand_kernel(
        integrand=integrand,
        suffix=kernel_suffix,
    )
    if kernel is not None:
        return kernel, FieldStruct, ValueStruct

    # Not found in cache, transform integrand and generate  kernel

    _check_field_compat(integrand, fields, field_args, domain)

    integrand_func = _translate_integrand(
        integrand,
        field_args,
    )

    _register_integrand_field_wrappers(integrand_func, fields)

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
                arg_names=integrand.argspec.args,
                field_args=field_args.keys(),
                value_args=value_args.keys(),
                sample_name=sample_name,
                domain_name=domain_name,
                test_name=test_name,
                trial_name=trial_name,
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
    nodal: bool,
    quadrature: Quadrature,
    test: Optional[TestField],
    trial: Optional[TrialField],
    fields: Dict[str, FieldLike],
    values: Dict[str, Any],
    accumulate_dtype: type,
    temporary_store: Optional[cache.TemporaryStore],
    output_dtype: type,
    output: Optional[Union[wp.array, BsrMatrix]],
    device,
):
    # Set-up launch arguments
    domain_elt_arg = domain.element_arg_value(device=device)
    domain_elt_index_arg = domain.element_index_arg_value(device=device)

    if quadrature is not None:
        qp_arg = quadrature.arg_value(device=device)

    field_arg_values = FieldStruct()
    for k, v in fields.items():
        setattr(field_arg_values, k, v.eval_arg_value(device=device))

    value_struct_values = _populate_value_struct(ValueStruct, values, integrand_name=integrand.name)

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

        accumulate_array.zero_()
        wp.launch(
            kernel=kernel,
            dim=domain.element_count(),
            inputs=[
                qp_arg,
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
        elif output is None:
            return accumulate_array.numpy()[0]
        else:
            array_cast(in_array=accumulate_array, out_array=output)
            return output

    test_arg = test.space_restriction.node_arg(device=device)

    # Linear form
    if trial is None:
        # If an output array is provided with the correct type, accumulate directly into it
        # Otherwise, grab a temporary array
        if output is None:
            if type_length(output_dtype) == test.space.VALUE_DOF_COUNT:
                output_shape = (test.space_partition.node_count(),)
            elif type_length(output_dtype) == 1:
                output_shape = (test.space_partition.node_count(), test.space.VALUE_DOF_COUNT)
            else:
                raise RuntimeError(
                    f"Incompatible output type {wp.types.type_repr(output_dtype)}, must be scalar or vector of length {test.space.VALUE_DOF_COUNT}"
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
            if type_length(output_dtype) != test.space.VALUE_DOF_COUNT:
                if type_length(output_dtype) != 1:
                    raise RuntimeError(
                        f"Incompatible output type {wp.types.type_repr(output_dtype)}, must be scalar or vector of length {test.space.VALUE_DOF_COUNT}"
                    )
                if output.ndim != 2 and output.shape[1] != test.space.VALUE_DOF_COUNT:
                    raise RuntimeError(
                        f"Incompatible output array shape, last dimension must be of size {test.space.VALUE_DOF_COUNT}"
                    )

        # Launch the integration on the kernel on a 2d scalar view of the actual array
        output.zero_()

        def as_2d_array(array):
            return wp.array(
                data=None,
                ptr=array.ptr,
                capacity=array.capacity,
                device=array.device,
                shape=(test.space_partition.node_count(), test.space.VALUE_DOF_COUNT),
                dtype=wp.types.type_scalar_type(output_dtype),
                grad=None if array.grad is None else as_2d_array(array.grad),
            )

        output_view = output if output.ndim == 2 else as_2d_array(output)

        if nodal:
            wp.launch(
                kernel=kernel,
                dim=(test.space_restriction.node_count(), test.space.VALUE_DOF_COUNT),
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
        else:
            wp.launch(
                kernel=kernel,
                dim=(test.space_restriction.node_count(), test.space.VALUE_DOF_COUNT),
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

    if test.space.VALUE_DOF_COUNT == 1 and trial.space.VALUE_DOF_COUNT == 1:
        block_type = output_dtype
    else:
        block_type = cache.cached_mat_type(
            shape=(test.space.VALUE_DOF_COUNT, trial.space.VALUE_DOF_COUNT), dtype=output_dtype
        )

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
            test.space.VALUE_DOF_COUNT,
            trial.space.VALUE_DOF_COUNT,
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

    else:
        trial_partition_arg = trial.space_partition.partition_arg_value(device)
        trial_topology_arg = trial.space_partition.space_topology.topo_arg_value(device)
        wp.launch(
            kernel=kernel,
            dim=(
                test.space_restriction.node_count(),
                trial.space.topology.MAX_NODES_PER_ELEMENT,
                test.space.VALUE_DOF_COUNT,
                trial.space.VALUE_DOF_COUNT,
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

    else:
        output = bsr_zeros(
            rows_of_blocks=test.space_partition.node_count(),
            cols_of_blocks=trial.space_partition.node_count(),
            block_type=block_type,
            device=device,
        )

    bsr_set_from_triplets(output, triplet_rows, triplet_cols, triplet_values)

    # Do not wait for garbage collection
    triplet_values_temp.release()
    triplet_rows_temp.release()
    triplet_cols_temp.release()

    return output


def integrate(
    integrand: Integrand,
    domain: Optional[GeometryDomain] = None,
    quadrature: Optional[Quadrature] = None,
    nodal: bool = False,
    fields: Optional[Dict[str, FieldLike]] = None,
    values: Optional[Dict[str, Any]] = None,
    accumulate_dtype: type = wp.float64,
    output_dtype: Optional[type] = None,
    output: Optional[Union[BsrMatrix, wp.array]] = None,
    device=None,
    temporary_store: Optional[cache.TemporaryStore] = None,
    kernel_options: Optional[Dict[str, Any]] = None,
):
    """
    Integrates a constant, linear or bilinear form, and returns a scalar, array, or sparse matrix, respectively.

    Args:
        integrand: Form to be integrated, must have :func:`integrand` decorator
        domain: Integration domain. If None, deduced from fields
        quadrature: Quadrature formula. If None, deduced from domain and fields degree.
        nodal: For linear or bilinear form only, use the test function nodes as the quadrature points. Assumes Lagrange interpolation functions are used, and no differential or DG operator is evaluated on the test or trial functions.
        fields: Discrete, test, and trial fields to be passed to the integrand. Keys in the dictionary must match integrand parameter names.
        values: Additional variable values to be passed to the integrand, can be of any type accepted by warp kernel launches. Keys in the dictionary must match integrand parameter names.
        temporary_store: shared pool from which to allocate temporary arrays
        accumulate_dtype: Scalar type to be used for accumulating integration samples
        output: Sparse matrix or warp array into which to store the result of the integration
        output_dtype: Scalar type for returned results in `output` is not provided. If None, defaults to `accumulate_dtype`
        device: Device on which to perform the integration
        kernel_options: Overloaded options to be passed to the kernel builder (e.g, ``{"enable_backward": True}``)
    """
    if fields is None:
        fields = {}

    if values is None:
        values = {}

    if kernel_options is None:
        kernel_options = {}

    if not isinstance(integrand, Integrand):
        raise ValueError("integrand must be tagged with @warp.fem.integrand decorator")

    test, test_name, trial, trial_name = _get_test_and_trial_fields(fields)

    if domain is None:
        if quadrature is not None:
            domain = quadrature.domain
        elif test is not None:
            domain = test.domain

    if domain is None:
        raise ValueError("Must provide at least one of domain, quadrature, or test field")
    if test is not None and domain != test.domain:
        raise NotImplementedError("Mixing integration and test domain is not supported yet")

    if nodal:
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
        nodal=nodal,
        quadrature=quadrature,
        test=test,
        test_name=test_name,
        trial=trial,
        trial_name=trial_name,
        fields=fields,
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
        nodal=nodal,
        quadrature=quadrature,
        test=test,
        trial=trial,
        fields=fields,
        values=values,
        accumulate_dtype=accumulate_dtype,
        temporary_store=temporary_store,
        output_dtype=output_dtype,
        output=output,
        device=device,
    )


def get_interpolate_to_field_function(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    dest: FieldRestriction,
):
    value_type = dest.space.dtype

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

        val_sum = value_type(0.0)
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
            node_index = dest.space_restriction.node_partition_index(dest_node_arg, local_node_index)
            dest.field.set_node_value(dest_eval_arg, node_index, val_sum / vol_sum)

    return interpolate_to_field_kernel_fn


def get_interpolate_to_array_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
    value_type: type,
):
    def interpolate_to_array_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: quadrature.domain.ElementArg,
        domain_index_arg: quadrature.domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        result: wp.array(dtype=value_type),
    ):
        domain_element_index = wp.tid()
        element_index = domain.element_index(domain_index_arg, domain_element_index)

        test_dof_index = NULL_DOF_INDEX
        trial_dof_index = NULL_DOF_INDEX

        qp_point_count = quadrature.point_count(domain_arg, qp_arg, domain_element_index, element_index)
        for k in range(qp_point_count):
            qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, k)
            coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, k)
            qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, k)

            sample = Sample(element_index, coords, qp_index, qp_weight, test_dof_index, trial_dof_index)

            result[qp_index] = integrand_func(sample, fields, values)

    return interpolate_to_array_kernel_fn


def get_interpolate_nonvalued_kernel(
    integrand_func: wp.Function,
    domain: GeometryDomain,
    quadrature: Quadrature,
    FieldStruct: wp.codegen.Struct,
    ValueStruct: wp.codegen.Struct,
):
    def interpolate_nonvalued_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: quadrature.domain.ElementArg,
        domain_index_arg: quadrature.domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
    ):
        domain_element_index = wp.tid()
        element_index = domain.element_index(domain_index_arg, domain_element_index)

        test_dof_index = NULL_DOF_INDEX
        trial_dof_index = NULL_DOF_INDEX

        qp_point_count = quadrature.point_count(domain_arg, qp_arg, domain_element_index, element_index)
        for k in range(qp_point_count):
            qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, k)
            coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, k)
            qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, k)

            sample = Sample(element_index, coords, qp_index, qp_weight, test_dof_index, trial_dof_index)
            integrand_func(sample, fields, values)

    return interpolate_nonvalued_kernel_fn


def _generate_interpolate_kernel(
    integrand: Integrand,
    domain: GeometryDomain,
    dest: Optional[Union[FieldLike, wp.array]],
    quadrature: Optional[Quadrature],
    fields: Dict[str, FieldLike],
    kernel_options: Optional[Dict[str, Any]] = None,
) -> wp.Kernel:
    if kernel_options is None:
        kernel_options = {}

    # Extract field arguments from integrand
    field_args, value_args, domain_name, sample_name = _get_integrand_field_arguments(
        integrand, fields=fields, domain=domain
    )

    # Generate field struct
    integrand_func = _translate_integrand(
        integrand,
        field_args,
    )

    _register_integrand_field_wrappers(integrand_func, fields)

    FieldStruct = _gen_field_struct(field_args)
    ValueStruct = _gen_value_struct(value_args)

    # Check if kernel exist in cache
    if isinstance(dest, FieldRestriction):
        kernel_suffix = (
            f"_itp_{FieldStruct.key}_{dest.domain.name}_{dest.space_restriction.space_partition.name}_{dest.space.name}"
        )
    elif wp.types.is_array(dest):
        kernel_suffix = f"_itp_{FieldStruct.key}_{quadrature.name}_{wp.types.type_repr(dest.dtype)}"
    else:
        kernel_suffix = f"_itp_{FieldStruct.key}_{quadrature.name}"

    kernel = cache.get_integrand_kernel(
        integrand=integrand,
        suffix=kernel_suffix,
    )
    if kernel is not None:
        return kernel, FieldStruct, ValueStruct

    _check_field_compat(integrand, fields, field_args, domain)

    # Generate interpolation kernel
    if isinstance(dest, FieldRestriction):
        # need to split into kernel + function for diffferentiability
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
                    arg_names=integrand.argspec.args,
                    field_args=field_args.keys(),
                    value_args=value_args.keys(),
                    sample_name=sample_name,
                    domain_name=domain_name,
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
    elif wp.types.is_array(dest):
        interpolate_kernel_fn = get_interpolate_to_array_kernel(
            integrand_func,
            domain=domain,
            quadrature=quadrature,
            value_type=dest.dtype,
            FieldStruct=FieldStruct,
            ValueStruct=ValueStruct,
        )
    else:
        interpolate_kernel_fn = get_interpolate_nonvalued_kernel(
            integrand_func,
            domain=domain,
            quadrature=quadrature,
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
                arg_names=integrand.argspec.args,
                field_args=field_args.keys(),
                value_args=value_args.keys(),
                sample_name=sample_name,
                domain_name=domain_name,
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
    fields: Dict[str, FieldLike],
    values: Dict[str, Any],
    device,
) -> wp.Kernel:
    # Set-up launch arguments
    elt_arg = domain.element_arg_value(device=device)
    elt_index_arg = domain.element_index_arg_value(device=device)

    field_arg_values = FieldStruct()
    for k, v in fields.items():
        setattr(field_arg_values, k, v.eval_arg_value(device=device))

    value_struct_values = _populate_value_struct(ValueStruct, values, integrand_name=integrand.name)

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
    elif wp.types.is_array(dest):
        qp_arg = quadrature.arg_value(device)
        wp.launch(
            kernel=kernel,
            dim=domain.element_count(),
            inputs=[qp_arg, elt_arg, elt_index_arg, field_arg_values, value_struct_values, dest],
            device=device,
        )
    else:
        qp_arg = quadrature.arg_value(device)
        wp.launch(
            kernel=kernel,
            dim=domain.element_count(),
            inputs=[qp_arg, elt_arg, elt_index_arg, field_arg_values, value_struct_values],
            device=device,
        )


def interpolate(
    integrand: Integrand,
    dest: Optional[Union[DiscreteField, FieldRestriction, wp.array]] = None,
    quadrature: Optional[Quadrature] = None,
    fields: Optional[Dict[str, FieldLike]] = None,
    values: Optional[Dict[str, Any]] = None,
    device=None,
    kernel_options: Optional[Dict[str, Any]] = None,
):
    """
    Interpolates a function at a finite set of sample points and optionally assigns the result to a discrete field or a raw warp array.

    Args:
        integrand: Function to be interpolated, must have :func:`integrand` decorator
        dest: Where to store the interpolation result. Can be either

         - a :class:`DiscreteField`, or restriction of a discrete field to a domain (from :func:`make_restriction`). In this case, interpolation will be performed at each node.
         - a normal warp array. In this case, the `quadrature` argument defining the interpolation locations must be provided and the result of the `integrand` at each quadrature point will be assigned to the array.
         - ``None``. In this case, the `quadrature` argument must also be provided and the `integrand` function is responsible for dealing with the interpolation result.
        quadrature: Quadrature formula defining the interpolation samples if `dest` is not a discrete field or field restriction.
        fields: Discrete fields to be passed to the integrand. Keys in the dictionary must match integrand parameters names.
        values: Additional variable values to be passed to the integrand, can be of any type accepted by warp kernel launches. Keys in the dictionary must match integrand parameter names.
        device: Device on which to perform the interpolation
        kernel_options: Overloaded options to be passed to the kernel builder (e.g, ``{"enable_backward": True}``)
    """
    if fields is None:
        fields = {}

    if values is None:
        values = {}

    if kernel_options is None:
        kernel_options = {}

    if not isinstance(integrand, Integrand):
        raise ValueError("integrand must be tagged with @integrand decorator")

    test, _, trial, __ = _get_test_and_trial_fields(fields)
    if test is not None or trial is not None:
        raise ValueError("Test or Trial fields should not be used for interpolation")

    if isinstance(dest, DiscreteField):
        dest = make_restriction(dest)

    if isinstance(dest, FieldRestriction):
        domain = dest.domain
    else:
        if quadrature is None:
            raise ValueError("When not interpolating to a field, a quadrature formula must be provided")

        domain = quadrature.domain

    kernel, FieldStruct, ValueStruct = _generate_interpolate_kernel(
        integrand=integrand,
        domain=domain,
        dest=dest,
        quadrature=quadrature,
        fields=fields,
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
        fields=fields,
        values=values,
        device=device,
    )
