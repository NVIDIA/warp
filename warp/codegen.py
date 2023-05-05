# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import os
import re
import sys
import importlib
import ast
import math
import inspect
import typing
import weakref
import ctypes
import copy
import textwrap

import numpy as np

from typing import Tuple
from typing import List
from typing import Dict
from typing import Any
from typing import Callable
from typing import Mapping
from typing import NamedTuple
from typing import Union

from warp.types import *
import warp.config

# map operator to function name
builtin_operators = {}

# see https://www.ics.uci.edu/~pattis/ICS-31/lectures/opexp.pdf for a
# nice overview of python operators

builtin_operators[ast.Add] = "add"
builtin_operators[ast.Sub] = "sub"
builtin_operators[ast.Mult] = "mul"
builtin_operators[ast.Div] = "div"
builtin_operators[ast.FloorDiv] = "floordiv"
builtin_operators[ast.Pow] = "pow"
builtin_operators[ast.Mod] = "mod"
builtin_operators[ast.UAdd] = "pos"
builtin_operators[ast.USub] = "neg"
builtin_operators[ast.Not] = "unot"

builtin_operators[ast.Gt] = ">"
builtin_operators[ast.Lt] = "<"
builtin_operators[ast.GtE] = ">="
builtin_operators[ast.LtE] = "<="
builtin_operators[ast.Eq] = "=="
builtin_operators[ast.NotEq] = "!="


def get_annotations(obj: Any) -> Mapping[str, Any]:
    """Alternative to `inspect.get_annotations()` for Python 3.9 and older."""
    # See https://docs.python.org/3/howto/annotations.html#accessing-the-annotations-dict-of-an-object-in-python-3-9-and-older
    if isinstance(obj, type):
        return obj.__dict__.get("__annotations__", {})

    return getattr(obj, "__annotations__", {})


def _get_struct_instance_ctype(
    inst: StructInstance,
    parent_ctype: Union[StructInstance, None],
    parent_field: Union[str, None],
) -> ctypes.Structure:
    if inst._struct_.ctype._fields_ == [("_dummy_", ctypes.c_int)]:
        return inst._struct_.ctype()

    if parent_ctype is None:
        inst_ctype = inst._struct_.ctype()
    else:
        inst_ctype = getattr(parent_ctype, parent_field)

    for field_name, _ in inst_ctype._fields_:
        value = getattr(inst, field_name, None)

        var_type = inst._struct_.vars[field_name].type
        if isinstance(var_type, array):
            if value is None:
                # create array with null pointer
                setattr(inst_ctype, field_name, array_t())
            else:
                # wp.array
                assert isinstance(value, array)
                assert (
                    value.dtype == var_type.dtype
                ), "assign to struct member variable {} failed, expected type {}, got type {}".format(
                    field_name, var_type.dtype, value.dtype
                )
                setattr(inst_ctype, field_name, value.__ctype__())
        elif isinstance(var_type, Struct):
            if value is None:
                _get_struct_instance_ctype(StructInstance(var_type), inst_ctype, field_name)
            else:
                _get_struct_instance_ctype(value, inst_ctype, field_name)
        elif issubclass(var_type, ctypes.Array):
            # vector/matrix type, e.g. vec3
            if value is None:
                setattr(inst_ctype, field_name, var_type())
            elif types_equal(type(value), var_type):
                setattr(inst_ctype, field_name, value)
            else:
                # conversion from list/tuple, ndarray, etc.
                setattr(inst_ctype, field_name, var_type(value))
        else:
            # primitive type
            if value is None:
                setattr(inst_ctype, field_name, var_type._type_())
            else:
                setattr(inst_ctype, field_name, var_type._type_(value))

    return inst_ctype


def _fmt_struct_instance_repr(inst: StructInstance, depth: int) -> str:
    indent = "\t"

    if inst._struct_.ctype._fields_ == [("_dummy_", ctypes.c_int)]:
        return f"{inst._struct_.key}()"

    lines = []
    lines.append(f"{inst._struct_.key}(")

    for field_name, _ in inst._struct_.ctype._fields_:
        if field_name == "_dummy_":
            continue

        field_value = getattr(inst, field_name, None)

        if isinstance(field_value, StructInstance):
            field_value = _fmt_struct_instance_repr(field_value, depth + 1)

        lines.append(f"{indent * (depth + 1)}{field_name}={field_value},")

    lines.append(f"{indent * depth})")
    return "\n".join(lines)


class StructInstance:
    def __init__(self, struct: Struct):
        self.__dict__["_struct_"] = struct

    def __setattr__(self, name, value):
        assert name in self._struct_.vars, "invalid struct member variable {}".format(name)
        super().__setattr__(name, value)

    def __ctype__(self):
        return _get_struct_instance_ctype(self, None, None)

    def __repr__(self):
        return _fmt_struct_instance_repr(self, 0)


class Struct:
    def __init__(self, cls, key, module):
        self.cls = cls
        self.module = module
        self.key = key

        self.vars = {}
        annotations = get_annotations(self.cls)
        for label, type in annotations.items():
            self.vars[label] = Var(label, type)

        fields = []
        for label, var in self.vars.items():
            if isinstance(var.type, array):
                fields.append((label, array_t))
            elif isinstance(var.type, Struct):
                fields.append((label, var.type.ctype))
            elif issubclass(var.type, ctypes.Array):
                fields.append((label, var.type))
            else:
                fields.append((label, var.type._type_))

        class StructType(ctypes.Structure):
            # if struct is empty, add a dummy field to avoid launch errors on CPU device ("ffi_prep_cif failed")
            _fields_ = fields or [("_dummy_", ctypes.c_int)]

        self.ctype = StructType

        if module:
            module.register_struct(self)

    def __call__(self):
        """
        This function returns s = StructInstance(self)
        s uses self.cls as template.
        To enable autocomplete on s, we inherit from self.cls.
        For example,

        @wp.struct
        class A:
            # annotations
            ...

        The type annotations are inherited in A(), allowing autocomplete in kernels
        """
        # return StructInstance(self)

        class NewStructInstance(self.cls, StructInstance):
            def __init__(inst):
                StructInstance.__init__(inst, self)

        return NewStructInstance()

    def initializer(self):
        input_types = {label: var.type for label, var in self.vars.items()}

        return warp.context.Function(
            func=None,
            key=self.key,
            namespace="",
            value_func=lambda *_: self,
            input_types=input_types,
            initializer_list_func=lambda *_: False,
            native_func=make_full_qualified_name(self.cls),
        )


def compute_type_str(base_name, template_params):
    if template_params is None or len(template_params) == 0:
        return base_name
    else:

        def param2str(p):
            if isinstance(p, int):
                return str(p)
            return p.__name__

        return f"{base_name}<{','.join(map(param2str, template_params))}>"


class Var:
    def __init__(self, label, type, requires_grad=False, constant=None):

        # convert built-in types to wp types
        if type == float:
            type = float32
        elif type == int:
            type = int32

        self.label = label
        self.type = type
        self.requires_grad = requires_grad
        self.constant = constant

    def __str__(self):
        return self.label

    def ctype(self):
        if is_array(self.type):
            if hasattr(self.type.dtype, "_wp_generic_type_str_"):
                dtypestr = compute_type_str(self.type.dtype._wp_generic_type_str_, self.type.dtype._wp_type_params_)
            else:
                dtypestr = str(self.type.dtype.__name__)
            classstr = type(self.type).__name__
            return f"{classstr}_t<{dtypestr}>"
        elif isinstance(self.type, Struct):
            return make_full_qualified_name(self.type.cls)
        elif hasattr(self.type, "_wp_generic_type_str_"):
            return compute_type_str(self.type._wp_generic_type_str_, self.type._wp_type_params_)
        else:
            return str(self.type.__name__)


class Block:
    # Represents a basic block of instructions, e.g.: list
    # of straight line instructions inside a for-loop or conditional

    def __init__(self):
        # list of statements inside this block
        self.body_forward = []
        self.body_replay = []
        self.body_reverse = []

        # list of vars declared in this block
        self.vars = []


class Adjoint:
    # Source code transformer, this class takes a Python function and
    # generates forward and backward SSA forms of the function instructions

    def __init__(adj, func, overload_annotations=None):
        adj.func = func

        # build AST from function object
        adj.source = inspect.getsource(func)

        # get source code lines and line number where function starts
        adj.raw_source, adj.fun_lineno = inspect.getsourcelines(func)

        # keep track of line number in function code
        adj.lineno = None

        # ensures that indented class methods can be parsed as kernels
        adj.source = textwrap.dedent(adj.source)

        # extract name of source file
        adj.filename = inspect.getsourcefile(func) or "unknown source file"

        # build AST
        adj.tree = ast.parse(adj.source)

        adj.fun_name = adj.tree.body[0].name

        # parse argument types
        argspec = inspect.getfullargspec(func)

        # ensure all arguments are annotated
        if overload_annotations is None:
            # use source-level argument annotations
            if len(argspec.annotations) < len(argspec.args):
                raise RuntimeError(f"Incomplete argument annotations on function {adj.fun_name}")
            adj.arg_types = argspec.annotations
        else:
            # use overload argument annotations
            for arg_name in argspec.args:
                if arg_name not in overload_annotations:
                    raise RuntimeError(f"Incomplete overload annotations for function {adj.fun_name}")
            adj.arg_types = overload_annotations.copy()

        adj.args = []

        for name, type in adj.arg_types.items():
            # skip return hint
            if name == "return":
                continue

            # add variable for argument
            arg = Var(name, type, False)
            adj.args.append(arg)

    # generate function ssa form and adjoint
    def build(adj, builder):
        adj.builder = builder

        adj.symbols = {}  # map from symbols to adjoint variables
        adj.variables = []  # list of local variables (in order)

        adj.cond = None  # condition variable if in branch
        adj.return_var = None  # return type for function or kernel

        # blocks
        adj.blocks = [Block()]
        adj.loop_blocks = []

        # holds current indent level
        adj.prefix = ""

        # used to generate new label indices
        adj.label_count = 0

        # update symbol map for each argument
        for a in adj.args:
            adj.symbols[a.label] = a

        # recursively evaluate function body
        try:
            adj.eval(adj.tree.body[0])
        except Exception as e:
            try:
                lineno = adj.lineno + adj.fun_lineno
                line = adj.source.splitlines()[adj.lineno]
                msg = f'Error while parsing function "{adj.fun_name}" at {adj.filename}:{lineno}:\n{line}\n'
                ex, data, traceback = sys.exc_info()
                e = ex("".join([msg] + list(data.args))).with_traceback(traceback)
            finally:
                raise e

        for a in adj.args:
            if isinstance(a.type, Struct):
                builder.build_struct_recursive(a.type)

    # code generation methods
    def format_template(adj, template, input_vars, output_var):
        # output var is always the 0th index
        args = [output_var] + input_vars
        s = template.format(*args)

        return s

    # generates a list of formatted args
    def format_args(adj, prefix, args):

        arg_strs = []

        for a in args:
            if type(a) == warp.context.Function:
                # functions don't have a var_ prefix so strip it off here
                if (prefix == "var_"):
                    arg_strs.append(a.key)
                else:
                    arg_strs.append(prefix + a.key)

            else:

                arg_strs.append(prefix + str(a))

        return arg_strs

    # generates argument string for a forward function call
    def format_forward_call_args(adj, args, use_initializer_list):
        arg_str = ", ".join(adj.format_args("var_", args))
        if (use_initializer_list):
            return "{{{}}}".format(arg_str)
        return arg_str

    # generates argument string for a reverse function call
    def format_reverse_call_args(adj, args, args_out, non_adjoint_args, non_adjoint_outputs, use_initializer_list):
        formatted_var = adj.format_args("var_", args)
        formatted_var_adj = adj.format_args(
            "&adj_" if use_initializer_list else "adj_",
            [a for i, a in enumerate(args) if i not in non_adjoint_args])
        formatted_out_adj = adj.format_args(
            "adj_", [a for i, a in enumerate(args_out) if i not in non_adjoint_outputs])

        if len(formatted_var_adj) == 0 and len(formatted_out_adj) == 0:
            # there are no adjoint arguments, so we don't need to call the reverse function
            return None

        if use_initializer_list:
            var_str = "{{{}}}".format(", ".join(formatted_var))
            adj_str = "{{{}}}".format(", ".join(formatted_var_adj))
            out_str = ", ".join(formatted_out_adj)
            arg_str = ", ".join([var_str, adj_str, out_str])
        else:
            arg_str = ", ".join(formatted_var + formatted_var_adj + formatted_out_adj)
        return arg_str

    def indent(adj):
        adj.prefix = adj.prefix + "\t"

    def dedent(adj):
        adj.prefix = adj.prefix[0:-1]

    def begin_block(adj):
        b = Block()

        # give block a unique id
        b.label = adj.label_count
        adj.label_count += 1

        adj.blocks.append(b)
        return b

    def end_block(adj):
        return adj.blocks.pop()

    def add_var(adj, type=None, constant=None, name=None):
        if name is None:
            index = len(adj.variables)
            name = str(index)

        # allocate new variable
        v = Var(name, type=type, constant=constant)

        adj.variables.append(v)

        adj.blocks[-1].vars.append(v)

        return v

    # append a statement to the forward pass
    def add_forward(adj, statement, replay=None, skip_replay=False):
        adj.blocks[-1].body_forward.append(adj.prefix + statement)

        if not skip_replay:
            if replay:
                # if custom replay specified then output it
                adj.blocks[-1].body_replay.append(adj.prefix + replay)
            else:
                # by default just replay the original statement
                adj.blocks[-1].body_replay.append(adj.prefix + statement)

    # append a statement to the reverse pass
    def add_reverse(adj, statement):
        adj.blocks[-1].body_reverse.append(adj.prefix + statement)

    def add_constant(adj, n):
        output = adj.add_var(type=type(n), constant=n)
        return output

    def add_comp(adj, op_strings, left, comps):
        output = adj.add_var(bool)

        s = "var_" + str(output) + " = " + ("(" * len(comps)) + "var_" + str(left) + " "
        for op, comp in zip(op_strings, comps):
            s += op + " var_" + str(comp) + ") "

        s = s.rstrip() + ";"

        adj.add_forward(s)

        return output

    def add_bool_op(adj, op_string, exprs):
        output = adj.add_var(bool)
        command = (
            "var_" + str(output) + " = " + (" " + op_string + " ").join(["var_" + str(expr) for expr in exprs]) + ";"
        )
        adj.add_forward(command)

        return output

    def add_call(adj, func, args, min_outputs=None, templates=[], kwds=None):
        # if func is overloaded then perform overload resolution here
        # we validate argument types before they go to generated native code
        resolved_func = None

        if func.is_builtin():
            for f in func.overloads:
                match = True

                # skip type checking for variadic functions
                if not f.variadic:
                    # check argument counts match (todo: default arguments?)
                    if len(f.input_types) != len(args):
                        match = False
                        continue

                    # check argument types equal
                    for i, a in enumerate(f.input_types.values()):
                        # if arg type registered as Any, treat as
                        # template allowing any type to match
                        if a == Any:
                            continue

                        # handle function refs as a special case
                        if a == Callable and type(args[i]) is warp.context.Function:
                            continue

                        # otherwise check arg type matches input variable type
                        if not types_equal(a, args[i].type, match_generic=True):
                            match = False
                            break

                # check output dimensions match expectations
                if min_outputs:
                    try:
                        value_type = f.value_func(args, kwds, templates)
                        if len(value_type) != min_outputs:
                            match = False
                            continue
                    except Exception:
                        # value func may fail if the user has given
                        # incorrect args, so we need to catch this
                        match = False
                        continue

                # found a match, use it
                if match:
                    resolved_func = f
                    break
        else:
            # user-defined function
            arg_types = [a.type for a in args]
            resolved_func = func.get_overload(arg_types)

        if resolved_func is None:
            arg_types = []

            for x in args:
                if isinstance(x, Var):
                    # shorten Warp primitive type names
                    if x.type.__module__ == "warp.types":
                        arg_types.append(x.type.__name__)
                    else:
                        arg_types.append(x.type.__module__ + "." + x.type.__name__)

                if isinstance(x, warp.context.Function):
                    arg_types.append("function")

            raise Exception(
                f"Couldn't find function overload for '{func.key}' that matched inputs with types: [{', '.join(arg_types)}]"
            )

        else:
            func = resolved_func

        # if it is a user-function then build it recursively
        if not func.is_builtin():
            adj.builder.build_function(func)

        # evaluate the function type based on inputs
        value_type = func.value_func(args, kwds, templates)

        func_name = compute_type_str(func.native_func, templates)

        use_initializer_list = func.initializer_list_func(args, templates)

        if value_type is None:
            # handles expression (zero output) functions, e.g.: void do_something();

            forward_call = "{}{}({});".format(func.namespace, func_name, adj.format_forward_call_args(args, use_initializer_list))
            if func.skip_replay:
                adj.add_forward(forward_call, replay="//" + forward_call)
            else:
                adj.add_forward(forward_call)

            if (not func.missing_grad and len(args)):
                arg_str = adj.format_reverse_call_args(args, [], {}, {}, use_initializer_list)
                if (arg_str is not None):
                    reverse_call = "{}adj_{}({});".format(func.namespace, func.native_func, arg_str)
                    adj.add_reverse(reverse_call)

            return None

        elif isinstance(value_type, list):
            # handle multiple value functions

            output = [adj.add_var(v) for v in value_type]
            forward_call = "{}{}({});".format(func.namespace, func_name, adj.format_forward_call_args(args + output, use_initializer_list))
            adj.add_forward(forward_call)

            if (not func.missing_grad and len(args)):
                arg_str = adj.format_reverse_call_args(args, output, {}, {}, use_initializer_list)
                if (arg_str is not None):
                    reverse_call = "{}adj_{}({});".format(func.namespace, func.native_func, arg_str)
                    adj.add_reverse(reverse_call)

            if len(output) == 1:
                return output[0]

            return output

        # handle simple function (one output)
        else:
            output = adj.add_var(func.value_func(args, kwds, templates))
            forward_call = "var_{} = {}{}({});".format(output, func.namespace, func_name, adj.format_forward_call_args(args, use_initializer_list))

            if func.skip_replay:
                adj.add_forward(forward_call, replay="//" + forward_call)
            else:
                adj.add_forward(forward_call)
            
            if (not func.missing_grad and len(args)):
                arg_str = adj.format_reverse_call_args(args, [output], {}, {}, use_initializer_list)
                if (arg_str is not None):
                    reverse_call = "{}adj_{}({});".format(func.namespace, func.native_func, arg_str)
                    adj.add_reverse(reverse_call)

            return output

    def add_return(adj, var):
        if var.ctype() == "void":
            adj.add_forward("return;".format(var), "goto label{};".format(adj.label_count))
        else:
            adj.add_forward("return var_{};".format(var), "goto label{};".format(adj.label_count))
            adj.add_reverse("adj_" + str(var) + " += adj_ret;")

        adj.add_reverse("label{}:;".format(adj.label_count))

        adj.label_count += 1

    # define an if statement
    def begin_if(adj, cond):
        adj.add_forward("if (var_{}) {{".format(cond))
        adj.add_reverse("}")

        adj.indent()

    def end_if(adj, cond):
        adj.dedent()

        adj.add_forward("}")
        adj.add_reverse(f"if (var_{cond}) {{")

    def begin_else(adj, cond):
        adj.add_forward(f"if (!var_{cond}) {{")
        adj.add_reverse("}")

        adj.indent()

    def end_else(adj, cond):
        adj.dedent()

        adj.add_forward("}")
        adj.add_reverse(f"if (!var_{cond}) {{")

    # define a for-loop
    def begin_for(adj, iter):
        cond_block = adj.begin_block()
        adj.loop_blocks.append(cond_block)
        adj.add_forward(f"for_start_{cond_block.label}:;")
        adj.indent()

        # evaluate cond
        adj.add_forward(f"if (iter_cmp(var_{iter}) == 0) goto for_end_{cond_block.label};")

        # evaluate iter
        val = adj.add_call(warp.context.builtin_functions["iter_next"], [iter])

        adj.begin_block()

        return val

    def end_for(adj, iter):
        body_block = adj.end_block()
        cond_block = adj.end_block()
        adj.loop_blocks.pop()

        ####################
        # forward pass

        for i in cond_block.body_forward:
            adj.blocks[-1].body_forward.append(i)

        for i in body_block.body_forward:
            adj.blocks[-1].body_forward.append(i)

        adj.add_forward(f"goto for_start_{cond_block.label};", skip_replay=True)

        adj.dedent()
        adj.add_forward(f"for_end_{cond_block.label}:;", skip_replay=True)

        ####################
        # reverse pass

        reverse = []

        # reverse iterator
        reverse.append(adj.prefix + f"var_{iter} = wp::iter_reverse(var_{iter});")

        for i in cond_block.body_forward:
            reverse.append(i)

        # zero adjoints
        for i in body_block.vars:
            if isinstance(i.type, Struct):
                reverse.append(adj.prefix + f"\tadj_{i} = {i.ctype()}{{}};")
            else:
                reverse.append(adj.prefix + f"\tadj_{i} = {i.ctype()}(0);")

        # replay
        for i in body_block.body_replay:
            reverse.append(i)

        # reverse
        for i in reversed(body_block.body_reverse):
            reverse.append(i)

        reverse.append(adj.prefix + f"\tgoto for_start_{cond_block.label};")
        reverse.append(adj.prefix + f"for_end_{cond_block.label}:;")

        adj.blocks[-1].body_reverse.extend(reversed(reverse))

    # define a while loop
    def begin_while(adj, cond):
        # evaulate condition in its own block
        # so we can control replay
        cond_block = adj.begin_block()
        adj.loop_blocks.append(cond_block)
        cond_block.body_forward.append(f"while_start_{cond_block.label}:;")

        c = adj.eval(cond)

        cond_block.body_forward.append(f"if ((var_{c}) == false) goto while_end_{cond_block.label};")

        # being block around loop
        adj.begin_block()
        adj.indent()

    def end_while(adj):
        adj.dedent()
        body_block = adj.end_block()
        cond_block = adj.end_block()
        adj.loop_blocks.pop()

        ####################
        # forward pass

        for i in cond_block.body_forward:
            adj.blocks[-1].body_forward.append(i)

        for i in body_block.body_forward:
            adj.blocks[-1].body_forward.append(i)

        adj.blocks[-1].body_forward.append(f"goto while_start_{cond_block.label};")
        adj.blocks[-1].body_forward.append(f"while_end_{cond_block.label}:;")

        ####################
        # reverse pass
        reverse = []

        # cond
        for i in cond_block.body_forward:
            reverse.append(i)

        # zero adjoints of local vars
        for i in body_block.vars:
            if isinstance(i.type, Struct):
                reverse.append(f"adj_{i} = {i.ctype()}{{}};")
            else:
                reverse.append(f"adj_{i} = {i.ctype()}(0);")

        # replay
        for i in body_block.body_replay:
            reverse.append(i)

        # reverse
        for i in reversed(body_block.body_reverse):
            reverse.append(i)

        reverse.append(f"goto while_start_{cond_block.label};")
        reverse.append(f"while_end_{cond_block.label}:;")

        # output
        adj.blocks[-1].body_reverse.extend(reversed(reverse))

    def emit_FunctionDef(adj, node):
        for f in node.body:
            adj.eval(f)

    def emit_If(adj, node):
        if len(node.body) == 0:
            return None

        # eval condition
        cond = adj.eval(node.test)

        # save symbol map
        symbols_prev = adj.symbols.copy()

        # eval body
        adj.begin_if(cond)

        for stmt in node.body:
            adj.eval(stmt)

        adj.end_if(cond)

        # detect existing symbols with conflicting definitions (variables assigned inside the branch)
        # and resolve with a phi (select) function
        for items in symbols_prev.items():
            sym = items[0]
            var1 = items[1]
            var2 = adj.symbols[sym]

            if var1 != var2:
                # insert a phi function that selects var1, var2 based on cond
                out = adj.add_call(warp.context.builtin_functions["select"], [cond, var1, var2])
                adj.symbols[sym] = out

        symbols_prev = adj.symbols.copy()

        # evaluate 'else' statement as if (!cond)
        if len(node.orelse) > 0:
            adj.begin_else(cond)

            for stmt in node.orelse:
                adj.eval(stmt)

            adj.end_else(cond)

        # detect existing symbols with conflicting definitions (variables assigned inside the else)
        # and resolve with a phi (select) function
        for items in symbols_prev.items():
            sym = items[0]
            var1 = items[1]
            var2 = adj.symbols[sym]

            if var1 != var2:
                # insert a phi function that selects var1, var2 based on cond
                # note the reversed order of vars since we want to use !cond as our select
                out = adj.add_call(warp.context.builtin_functions["select"], [cond, var2, var1])
                adj.symbols[sym] = out

    def emit_Compare(adj, node):
        # node.left, node.ops (list of ops), node.comparators (things to compare to)
        # e.g. (left ops[0] node.comparators[0]) ops[1] node.comparators[1]

        left = adj.eval(node.left)
        comps = [adj.eval(comp) for comp in node.comparators]
        op_strings = [builtin_operators[type(op)] for op in node.ops]

        return adj.add_comp(op_strings, left, comps)

    def emit_BoolOp(adj, node):
        # op, expr list values

        op = node.op
        if isinstance(op, ast.And):
            func = "&&"
        elif isinstance(op, ast.Or):
            func = "||"
        else:
            raise KeyError("Op {} is not supported".format(op))

        return adj.add_bool_op(func, [adj.eval(expr) for expr in node.values])

    def emit_Name(adj, node):
        # lookup symbol, if it has already been assigned to a variable then return the existing mapping
        if node.id in adj.symbols:
            return adj.symbols[node.id]

        # try and resolve the name using the function's globals context (used to lookup constants + functions)
        elif node.id in adj.func.__globals__:
            obj = adj.func.__globals__[node.id]

            if warp.types.is_value(obj):
                # evaluate constant
                out = adj.add_constant(obj)
                adj.symbols[node.id] = out
                return out

            elif isinstance(obj, warp.context.Function):
                # pass back ref. to function (will be converted to name during function call)
                return obj

            else:
                raise TypeError(f"'{node.id}' is not a local variable, function, or warp.constant")

        else:
            # Lookup constant in captured contents
            capturedvars = dict(
                zip(adj.func.__code__.co_freevars, [c.cell_contents for c in (adj.func.__closure__ or [])])
            )
            obj = capturedvars.get(str(node.id), None)

            if warp.types.is_value(obj):
                # evaluate constant
                out = adj.add_constant(obj)
                adj.symbols[node.id] = out
                return out

            raise KeyError("Referencing undefined symbol: " + str(node.id))

    def emit_Attribute(adj, node):
        def attribute_to_str(node):
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return attribute_to_str(node.value) + "." + node.attr
            else:
                raise RuntimeError(f"Failed to parse attribute")

        def attribute_to_val(node, context):
            if isinstance(node, ast.Name):
                if node.id in context:
                    return context[node.id]
                return None
            elif isinstance(node, ast.Attribute):
                val = attribute_to_val(node.value, context)
                if val is None:
                    return None
                return getattr(val, node.attr)
            else:
                raise RuntimeError(f"Failed to parse attribute")

        key = attribute_to_str(node)

        if key in adj.symbols:
            return adj.symbols[key]
        elif isinstance(node.value, ast.Name) and node.value.id in adj.symbols:
            struct = adj.symbols[node.value.id]

            try:
                attr_name = struct.label + "." + node.attr
                attr_type = struct.type.vars[node.attr].type
            except:
                raise RuntimeError(f"Error, `{node.attr}` is not an attribute of '{node.value.id}' ({struct.type})")

            # create a Var that points to the struct attribute, i.e.: directly generates `struct.attr` when used
            return Var(attr_name, attr_type)
        else:
            # try and resolve to either a wp.constant
            # or a wp.func object
            obj = attribute_to_val(node, adj.func.__globals__)

            if warp.types.is_value(obj):
                out = adj.add_constant(obj)
                adj.symbols[key] = out
                return out

            elif isinstance(node.value, ast.Attribute):
                # resolve nested attribute
                val = adj.eval(node.value)

                try:
                    attr_name = val.label + "." + node.attr
                    attr_type = val.type.vars[node.attr].type
                except:
                    raise RuntimeError(f"Error, `{node.attr}` is not an attribute of '{val.label}' ({val.type})")

                # create a Var that points to the struct attribute, i.e.: directly generates `struct.attr` when used
                return Var(attr_name, attr_type)
            else:
                raise TypeError(f"'{key}' is not a local variable, warp function, nested attribute, or warp constant")

    def emit_String(adj, node):
        # string constant
        return adj.add_constant(node.s)

    def emit_Num(adj, node):
        # lookup constant, if it has already been assigned then return existing var
        key = (node.n, type(node.n))

        if key in adj.symbols:
            return adj.symbols[key]
        else:
            out = adj.add_constant(node.n)
            adj.symbols[key] = out
            return out

    def emit_NameConstant(adj, node):
        if node.value == True:
            return adj.add_constant(True)
        elif node.value == False:
            return adj.add_constant(False)
        elif node.value is None:
            raise TypeError("None type unsupported")

    def emit_Constant(adj, node):
        if isinstance(node, ast.Str):
            return adj.emit_String(node)
        elif isinstance(node, ast.Num):
            return adj.emit_Num(node)
        else:
            assert isinstance(node, ast.NameConstant)
            return adj.emit_NameConstant(node)

    def emit_BinOp(adj, node):
        # evaluate binary operator arguments
        left = adj.eval(node.left)
        right = adj.eval(node.right)

        name = builtin_operators[type(node.op)]
        func = warp.context.builtin_functions[name]

        return adj.add_call(func, [left, right])

    def emit_UnaryOp(adj, node):
        # evaluate unary op arguments
        arg = adj.eval(node.operand)

        name = builtin_operators[type(node.op)]
        func = warp.context.builtin_functions[name]

        return adj.add_call(func, [arg])

    def emit_While(adj, node):
        adj.begin_while(node.test)

        symbols_prev = adj.symbols.copy()

        # eval body
        for s in node.body:
            adj.eval(s)

        # detect symbols with conflicting definitions (assigned inside the for loop)
        for items in symbols_prev.items():
            sym = items[0]
            var1 = items[1]
            var2 = adj.symbols[sym]

            if var1 != var2:
                if warp.config.verbose:
                    print(
                        "Warning: detected mutated variable {} during a dynamic for-loop, this is a non-differentiable operation".format(
                            sym
                        )
                    )

                if var1.constant is not None:
                    raise Exception(
                        "Error mutating a constant {} inside a dynamic loop, use the following syntax: pi = float(3.141) to declare a dynamic variable".format(
                            sym
                        )
                    )

                # overwrite the old variable value (violates SSA)
                adj.add_call(warp.context.builtin_functions["copy"], [var1, var2])

                # reset the symbol to point to the original variable
                adj.symbols[sym] = var1

        adj.end_while()

    def emit_For(adj, node):
        def is_num(a):
            # simple constant
            if isinstance(a, ast.Num):
                return True
            # expression of form -constant
            elif isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Num):
                return True
            else:
                # try and resolve the expression to an object
                # e.g.: wp.constant in the globals scope
                obj, path = adj.resolve_path(a)
                if warp.types.is_int(obj):
                    return True
                else:
                    return False

        def eval_num(a):
            if isinstance(a, ast.Num):
                return a.n
            elif isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Num):
                return -a.operand.n
            else:
                # try and resolve the expression to an object
                # e.g.: wp.constant in the globals scope
                obj, path = adj.resolve_path(a)
                if warp.types.is_int(obj):
                    return obj
                else:
                    return False

        # try and unroll simple range() statements that use constant args
        unrolled = False

        if isinstance(node.iter, ast.Call) and node.iter.func.id == "range":
            is_constant = True
            for a in node.iter.args:
                # if all range() arguments are numeric constants we will unroll
                # note that this only handles trivial constants, it will not unroll
                # constant compile-time expressions e.g.: range(0, 3*2)
                if not is_num(a):
                    is_constant = False
                    break

            if is_constant:
                # range(end)
                if len(node.iter.args) == 1:
                    start = 0
                    end = eval_num(node.iter.args[0])
                    step = 1

                # range(start, end)
                elif len(node.iter.args) == 2:
                    start = eval_num(node.iter.args[0])
                    end = eval_num(node.iter.args[1])
                    step = 1

                # range(start, end, step)
                elif len(node.iter.args) == 3:
                    start = eval_num(node.iter.args[0])
                    end = eval_num(node.iter.args[1])
                    step = eval_num(node.iter.args[2])

                # test if we're above max unroll count
                max_iters = abs(end - start) // abs(step)
                max_unroll = adj.builder.options["max_unroll"]

                if max_iters > max_unroll:
                    if warp.config.verbose:
                        print(
                            f"Warning: fixed-size loop count of {max_iters} is larger than the module 'max_unroll' limit of {max_unroll}, will generate dynamic loop."
                        )
                else:
                    # unroll
                    for i in range(start, end, step):
                        var_iter = adj.add_constant(i)
                        adj.symbols[node.target.id] = var_iter

                        # eval body
                        for s in node.body:
                            adj.eval(s)

                    unrolled = True

        # couldn't unroll so generate a dynamic loop
        if not unrolled:
            # evaluate the Iterable
            iter = adj.eval(node.iter)

            adj.symbols[node.target.id] = adj.begin_for(iter)

            # for loops should be side-effect free, here we store a copy
            symbols_prev = adj.symbols.copy()

            # eval body
            for s in node.body:
                adj.eval(s)

            # detect symbols with conflicting definitions (assigned inside the for loop)
            for items in symbols_prev.items():
                sym = items[0]
                var1 = items[1]
                var2 = adj.symbols[sym]

                if var1 != var2:
                    if warp.config.verbose:
                        lineno = adj.lineno + adj.fun_lineno
                        line = adj.source.splitlines()[adj.lineno]
                        msg = f'Warning: detected mutated variable {sym} during a dynamic for-loop in function "{adj.fun_name}" at {adj.filename}:{lineno}: this is a non-differentiable operation.\n{line}\n'
                        print(msg)

                    if var1.constant is not None:
                        raise Exception(
                            "Error mutating a constant {} inside a dynamic loop, use the following syntax: pi = float(3.141) to declare a dynamic variable".format(
                                sym
                            )
                        )

                    # overwrite the old variable value (violates SSA)
                    adj.add_call(warp.context.builtin_functions["copy"], [var1, var2])

                    # reset the symbol to point to the original variable
                    adj.symbols[sym] = var1

            adj.end_for(iter)

    def emit_Break(adj, node):
        adj.add_forward(f"goto for_end_{adj.loop_blocks[-1].label};")

    def emit_Expr(adj, node):
        return adj.eval(node.value)

    def emit_Call(adj, node):
        # try and lookup function in globals by
        # resolving path (e.g.: module.submodule.attr)
        func, path = adj.resolve_path(node.func)
        templates = []

        if isinstance(func, warp.context.Function) == False:
            if len(path) == 0:
                raise RuntimeError(f"Unrecognized syntax for function call, path not valid: '{node.func}'")

            attr = path[-1]
            caller = func
            func = None

            # try and lookup function name in builtins (e.g.: using `dot` directly without wp prefix)
            if attr in warp.context.builtin_functions:
                func = warp.context.builtin_functions[attr]

            # vector class type e.g.: wp.vec3f constructor
            if func is None and hasattr(caller, "_wp_generic_type_str_"):
                templates = caller._wp_type_params_
                func = warp.context.builtin_functions.get(caller._wp_constructor_)

            # scalar class type e.g.: wp.int8 constructor
            if func is None and hasattr(caller, "__name__") and caller.__name__ in warp.context.builtin_functions:
                func = warp.context.builtin_functions.get(caller.__name__)

            # struct constructor
            if func is None and isinstance(caller, Struct):
                adj.builder.build_struct_recursive(caller)
                func = caller.initializer()

            if func is None:
                raise RuntimeError(
                    f"Could not find function {'.'.join(path)} as a built-in or user-defined function. Note that user functions must be annotated with a @wp.func decorator to be called from a kernel."
                )

        args = []

        # eval all arguments
        for arg in node.args:
            var = adj.eval(arg)
            args.append(var)

        # eval all keyword ags
        def kwval(kw):
            if isinstance(kw.value, ast.Num):
                return kw.value.n
            elif isinstance(kw.value, ast.Tuple):
                return tuple(e.n for e in kw.value.elts)
            return adj.resolve_path(kw.value)[0]

        kwds = {kw.arg: kwval(kw) for kw in node.keywords}

        # get expected return count, e.g.: for multi-assignment
        min_outputs = None
        if hasattr(node, "expects"):
            min_outputs = node.expects

        # add var with value type from the function
        out = adj.add_call(func=func, args=args, kwds=kwds, templates=templates, min_outputs=min_outputs)
        return out

    def emit_Index(adj, node):
        # the ast.Index node appears in 3.7 versions
        # when performing array slices, e.g.: x = arr[i]
        # but in version 3.8 and higher it does not appear
        return adj.eval(node.value)

    def emit_Subscript(adj, node):
        target = adj.eval(node.value)

        indices = []

        if isinstance(node.slice, ast.Tuple):
            # handles the x[i,j] case (Python 3.8.x upward)
            for arg in node.slice.elts:
                var = adj.eval(arg)
                indices.append(var)

        elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Tuple):
            # handles the x[i,j] case (Python 3.7.x)
            for arg in node.slice.value.elts:
                var = adj.eval(arg)
                indices.append(var)
        else:
            # simple expression, e.g.: x[i]
            var = adj.eval(node.slice)
            indices.append(var)

        if is_array(target.type):
            if len(indices) == target.type.ndim:
                # handles array loads (where each dimension has an index specified)
                out = adj.add_call(warp.context.builtin_functions["load"], [target, *indices])
            else:
                # handles array views (fewer indices than dimensions)
                out = adj.add_call(warp.context.builtin_functions["view"], [target, *indices])

        else:
            # handles non-array type indexing, e.g: vec3, mat33, etc
            out = adj.add_call(warp.context.builtin_functions["index"], [target, *indices])

        return out

    def emit_Assign(adj, node):
        # handle the case where we are assigning multiple output variables
        if isinstance(node.targets[0], ast.Tuple):
            # record the expected number of outputs on the node
            # we do this so we can decide which function to
            # call based on the number of expected outputs
            if isinstance(node.value, ast.Call):
                node.value.expects = len(node.targets[0].elts)

            # evaluate values
            if (isinstance(node.value, ast.Tuple)):
                out = [adj.eval(v) for v in node.value.elts]
            else:
                out = adj.eval(node.value)

            names = []
            for v in node.targets[0].elts:
                if isinstance(v, ast.Name):
                    names.append(v.id)
                else:
                    raise RuntimeError(
                        "Multiple return functions can only assign to simple variables, e.g.: x, y = func()"
                    )

            if len(names) != len(out):
                raise RuntimeError(
                    "Multiple return functions need to receive all their output values, incorrect number of values to unpack (expected {}, got {})".format(
                        len(out), len(names)
                    )
                )

            for name, rhs in zip(names, out):
                if name in adj.symbols:
                    if not types_equal(rhs.type, adj.symbols[name].type):
                        raise TypeError(
                            "Error, assigning to existing symbol {} ({}) with different type ({})".format(
                                name, adj.symbols[name].type, rhs.type
                            )
                        )

                adj.symbols[name] = rhs

            return out

        # handles the case where we are assigning to an array index (e.g.: arr[i] = 2.0)
        elif isinstance(node.targets[0], ast.Subscript):
            target = adj.eval(node.targets[0].value)
            value = adj.eval(node.value)

            slice = node.targets[0].slice
            indices = []

            if isinstance(slice, ast.Tuple):
                # handles the x[i, j] case (Python 3.8.x upward)
                for arg in slice.elts:
                    var = adj.eval(arg)
                    indices.append(var)

            elif isinstance(slice, ast.Index) and isinstance(slice.value, ast.Tuple):
                # handles the x[i, j] case (Python 3.7.x)
                for arg in slice.value.elts:
                    var = adj.eval(arg)
                    indices.append(var)
            else:
                # simple expression, e.g.: x[i]
                var = adj.eval(slice)
                indices.append(var)

            if is_array(target.type):
                adj.add_call(warp.context.builtin_functions["store"], [target, *indices, value])
            else:
                raise RuntimeError("Can only subscript assign array types")

            return var

        elif isinstance(node.targets[0], ast.Name):
            # symbol name
            name = node.targets[0].id

            # evaluate rhs
            rhs = adj.eval(node.value)

            # check type matches if symbol already defined
            if name in adj.symbols:
                if not types_equal(rhs.type, adj.symbols[name].type):
                    raise TypeError(
                        "Error, assigning to existing symbol {} ({}) with different type ({})".format(
                            name, adj.symbols[name].type, rhs.type
                        )
                    )

            # handle simple assignment case (a = b), where we generate a value copy rather than reference
            if isinstance(node.value, ast.Name):
                out = adj.add_var(rhs.type)
                adj.add_call(warp.context.builtin_functions["copy"], [out, rhs])
            else:
                out = rhs

            # update symbol map (assumes lhs is a Name node)
            adj.symbols[name] = out
            return out

        elif isinstance(node.targets[0], ast.Attribute):
            raise RuntimeError("Error, assignment to member variables is not currently support (structs are immutable)")

        else:
            raise RuntimeError("Error, unsupported assignment statement.")

    def emit_Return(adj, node):
        cond = adj.cond

        if node.value is not None:
            var = adj.eval(node.value)
        else:
            var = Var("void", void)

        # set return type of function
        if adj.return_var is not None:
            if adj.return_var.ctype() != var.ctype():
                raise TypeError(
                    f"Error, function returned different types, previous: {adj.return_var.ctype()}, new {var.ctype()}"
                )
        adj.return_var = var

        adj.add_return(var)

    def emit_AugAssign(adj, node):
        # convert inplace operations (+=, -=, etc) to ssa form, e.g.: c = a + b
        left = adj.eval(node.target)
        right = adj.eval(node.value)

        # lookup
        name = builtin_operators[type(node.op)]
        func = warp.context.builtin_functions[name]

        out = adj.add_call(func, [left, right])

        # update symbol map
        adj.symbols[node.target.id] = out

    def emit_Tuple(adj, node):
        # LHS for expressions, such as i, j, k = 1, 2, 3
        for elem in node.elts:
            adj.eval(elem)

    def eval(adj, node):
        if hasattr(node, "lineno"):
            adj.set_lineno(node.lineno - 1)

        node_visitors = {
            ast.FunctionDef: Adjoint.emit_FunctionDef,
            ast.If: Adjoint.emit_If,
            ast.Compare: Adjoint.emit_Compare,
            ast.BoolOp: Adjoint.emit_BoolOp,
            ast.Name: Adjoint.emit_Name,
            ast.Attribute: Adjoint.emit_Attribute,
            ast.Str: Adjoint.emit_String,  # Deprecated in 3.8; use Constant
            ast.Num: Adjoint.emit_Num,  # Deprecated in 3.8; use Constant
            ast.NameConstant: Adjoint.emit_NameConstant,  # Deprecated in 3.8; use Constant
            ast.Constant: Adjoint.emit_Constant,
            ast.BinOp: Adjoint.emit_BinOp,
            ast.UnaryOp: Adjoint.emit_UnaryOp,
            ast.While: Adjoint.emit_While,
            ast.For: Adjoint.emit_For,
            ast.Break: Adjoint.emit_Break,
            ast.Expr: Adjoint.emit_Expr,
            ast.Call: Adjoint.emit_Call,
            ast.Index: Adjoint.emit_Index,  # Deprecated in 3.8; Use the index value directly instead.
            ast.Subscript: Adjoint.emit_Subscript,
            ast.Assign: Adjoint.emit_Assign,
            ast.Return: Adjoint.emit_Return,
            ast.AugAssign: Adjoint.emit_AugAssign,
            ast.Tuple: Adjoint.emit_Tuple,
        }

        emit_node = node_visitors.get(type(node))

        if emit_node is not None:
            return emit_node(adj, node)
        else:
            raise Exception("Error, ast node of type {} not supported".format(type(node)))

    # helper to evaluate expressions of the form
    # obj1.obj2.obj3.attr in the function's global scope
    def resolve_path(adj, node):
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
            extract_contents = (
                lambda contents: contents
                if isinstance(contents, warp.context.Function) or not callable(contents)
                else contents
            )
            capturedvars = dict(
                zip(
                    adj.func.__code__.co_freevars,
                    [extract_contents(c.cell_contents) for c in (adj.func.__closure__ or [])],
                )
            )

            vars_dict = {**adj.func.__globals__, **capturedvars}
            func = eval(".".join(path), vars_dict)
            return func, path
        except:
            pass

        # I added this so people can eg do this kind of thing
        # in a kernel:

        # v = vec3(0.0,0.2,0.4)

        # vec3 is now an alias and is not in warp.context.builtin_functions.
        # This means it can't be directly looked up in Adjoint.add_call, and
        # needs to be looked up by digging some information out of the
        # python object it actually came from.

        # Before this fix, resolve_path was returning None, as the
        # "vec3" symbol is not available. In this situation I'm assuming
        # it's a member of the warp module and trying to look it up:
        try:
            evalstr = ".".join(["warp"] + path)
            func = eval(evalstr, {"warp": warp})
            return func, path
        except:
            return None, path

    # annotate generated code with the original source code line
    def set_lineno(adj, lineno):
        if adj.lineno is None or adj.lineno != lineno:
            line = lineno + adj.fun_lineno
            source = adj.raw_source[lineno].strip().ljust(70)
            adj.add_forward(f"// {source}       <L {line}>")
            adj.add_reverse(f"// adj: {source}  <L {line}>")
        adj.lineno = lineno


# ----------------
# code generation

cpu_module_header = """
#define WP_NO_CRT
#include "../native/builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

using namespace wp;

"""

cuda_module_header = """
#define WP_NO_CRT
#include "../native/builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)


using namespace wp;

"""

struct_template = """
struct {name}
{{
{struct_body}

    CUDA_CALLABLE {name}({forward_args})
    {forward_initializers}
    {{
    }}

    {name}& operator += (const {name}&) {{ return *this; }}

}};

static CUDA_CALLABLE void adj_{name}({reverse_args})
{{
{reverse_body}
}}
"""

cpu_function_template = """
// {filename}:{lineno}
static {return_type} {name}(
    {forward_args})
{{
{forward_body}
}}

// {filename}:{lineno}
static void adj_{name}(
    {reverse_args})
{{
{reverse_body}
}}

"""

cuda_function_template = """
// {filename}:{lineno}
static CUDA_CALLABLE {return_type} {name}(
    {forward_args})
{{
{forward_body}
}}

// {filename}:{lineno}
static CUDA_CALLABLE void adj_{name}(
    {reverse_args})
{{
{reverse_body}
}}

"""

cuda_kernel_template = """

extern "C" __global__ void {name}_cuda_kernel_forward(
    {forward_args})
{{
    size_t _idx = grid_index();
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

{forward_body}
}}

extern "C" __global__ void {name}_cuda_kernel_backward(
    {reverse_args})
{{
    size_t _idx = grid_index();
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

{reverse_body}
}}

"""

cpu_kernel_template = """

void {name}_cpu_kernel_forward(
    {forward_args})
{{
{forward_body}
}}

void {name}_cpu_kernel_backward(
    {reverse_args})
{{
{reverse_body}
}}

"""

cuda_module_template = """

extern "C" {{

// Python entry points
WP_API void {name}_cuda_forward(
    void* stream,
    {forward_args})
{{
    {name}_cuda_kernel_forward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>(
            {forward_params});
}}

WP_API void {name}_cuda_backward(
    void* stream,
    {reverse_args})
{{
    {name}_cuda_kernel_backward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>(
            {reverse_params});
}}

}} // extern C

"""

cpu_module_template = """

extern "C" {{

// Python CPU entry points
WP_API void {name}_cpu_forward(
    {forward_args})
{{
    set_launch_bounds(dim);

    for (size_t i=0; i < dim.size; ++i)
    {{
        s_threadIdx = i;

        {name}_cpu_kernel_forward(
            {forward_params});
    }}
}}

WP_API void {name}_cpu_backward(
    {reverse_args})
{{
    set_launch_bounds(dim);

    for (size_t i=0; i < dim.size; ++i)
    {{
        s_threadIdx = i;

        {name}_cpu_kernel_backward(
            {reverse_params});
    }}
}}

}} // extern C

"""

cuda_module_header_template = """

extern "C" {{

// Python CUDA entry points
WP_API void {name}_cuda_forward(
    void* stream,
    {forward_args});

WP_API void {name}_cuda_backward(
    void* stream,
    {reverse_args});

}} // extern C
"""

cpu_module_header_template = """

extern "C" {{

// Python CPU entry points
WP_API void {name}_cpu_forward(
    {forward_args});

WP_API void {name}_cpu_backward(
    {reverse_args});

}} // extern C
"""


# converts a constant Python value to equivalent C-repr
def constant_str(value):
    value_type = type(value)

    if value_type == bool:
        if value:
            return "true"
        else:
            return "false"

    elif value_type == str:
        # ensure constant strings are correctly escaped
        return '"' + str(value.encode("unicode-escape").decode()) + '"'

    elif isinstance(value, ctypes.Array):
        if value_type._wp_scalar_type_ == float16:
            # special case for float16, which is stored as uint16 in the ctypes.Array
            from warp.context import runtime

            scalar_value = runtime.core.half_bits_to_float
        else:
            scalar_value = lambda x: x

        # list of scalar initializer values
        initlist = []
        for i in range(value._length_):
            x = ctypes.Array.__getitem__(value, i)
            initlist.append(str(scalar_value(x)))

        dtypestr = f"wp::initializer_array<{value._length_},wp::{value._wp_scalar_type_.__name__}>"

        # construct value from initializer array, e.g. wp::initializer_array<4,wp::float32>{1.0, 2.0, 3.0, 4.0}
        return f"{dtypestr}{{{', '.join(initlist)}}}"

    else:
        # otherwise just convert constant to string
        return str(value)


def indent(args, stops=1):
    sep = ",\n"
    for i in range(stops):
        sep += "\t"

    # return sep + args.replace(", ", "," + sep)
    return sep.join(args)


# generates a C function name based on the python function name
def make_full_qualified_name(func):
    return re.sub("[^0-9a-zA-Z_]+", "", func.__qualname__.replace(".", "__"))


def codegen_struct(struct, device="cpu", indent_size=4):
    name = make_full_qualified_name(struct.cls)

    body = []
    indent_block = " " * indent_size
    for label, var in struct.vars.items():
        body.append(var.ctype() + " " + label + ";\n")

    forward_args = []
    reverse_args = []

    forward_initializers = []
    reverse_body = []

    # forward args
    for label, var in struct.vars.items():
        forward_args.append(f"{var.ctype()} const& {label} = {{}}")
        reverse_args.append(f"{var.ctype()} const&")

        prefix = f"{indent_block}," if forward_initializers else ":"
        forward_initializers.append(f"{indent_block}{prefix} {label}{{{label}}}\n")

    # reverse args
    for label, var in struct.vars.items():
        reverse_args.append(var.ctype() + " const& adj_" + label)

        reverse_body.append(f"{indent_block}adj_ret.{label} = adj_{label};\n")

    reverse_args.append(name + " & adj_ret")

    return struct_template.format(
        name=name,
        struct_body="".join([indent_block + l for l in body]),
        forward_args=indent(forward_args),
        forward_initializers="".join(forward_initializers),
        reverse_args=indent(reverse_args),
        reverse_body="".join(reverse_body),
    )


def codegen_func_forward_body(adj, device="cpu", indent=4):
    body = []
    indent_block = " " * indent

    for f in adj.blocks[0].body_forward:
        body += [f + "\n"]

    return "".join([indent_block + l for l in body])


def codegen_func_forward(adj, func_type="kernel", device="cpu"):
    s = ""

    # primal vars
    s += "    //---------\n"
    s += "    // primal vars\n"

    for var in adj.variables:
        if var.constant is None:
            s += "    " + var.ctype() + " var_" + str(var.label) + ";\n"
        else:
            s += "    const " + var.ctype() + " var_" + str(var.label) + " = " + constant_str(var.constant) + ";\n"

    # forward pass
    s += "    //---------\n"
    s += "    // forward\n"

    if device == "cpu":
        s += codegen_func_forward_body(adj, device=device, indent=4)

    elif device == "cuda":
        if func_type == "kernel":
            s += codegen_func_forward_body(adj, device=device, indent=8)
        else:
            s += codegen_func_forward_body(adj, device=device, indent=4)

    return s


def codegen_func_reverse_body(adj, device="cpu", indent=4):
    body = []
    indent_block = " " * indent

    # forward pass
    body += ["//---------\n"]
    body += ["// forward\n"]

    for f in adj.blocks[0].body_replay:
        body += [f + "\n"]

    # reverse pass
    body += ["//---------\n"]
    body += ["// reverse\n"]

    for l in reversed(adj.blocks[0].body_reverse):
        body += [l + "\n"]

    body += ["return;\n"]

    return "".join([indent_block + l for l in body])


def codegen_func_reverse(adj, func_type="kernel", device="cpu"):
    s = ""

    # primal vars
    s += "    //---------\n"
    s += "    // primal vars\n"

    for var in adj.variables:
        if var.constant is None:
            s += "    " + var.ctype() + " var_" + str(var.label) + ";\n"
        else:
            s += "    const " + var.ctype() + " var_" + str(var.label) + " = " + constant_str(var.constant) + ";\n"

    # dual vars
    s += "    //---------\n"
    s += "    // dual vars\n"

    for var in adj.variables:
        if isinstance(var.type, Struct):
            s += "    " + var.ctype() + " adj_" + str(var.label) + ";\n"
        else:
            s += "    " + var.ctype() + " adj_" + str(var.label) + "(0);\n"

    if device == "cpu":
        s += codegen_func_reverse_body(adj, device=device, indent=4)
    elif device == "cuda":
        if func_type == "kernel":
            s += codegen_func_reverse_body(adj, device=device, indent=8)
        else:
            s += codegen_func_reverse_body(adj, device=device, indent=4)
    else:
        raise ValueError("Device {} not supported for codegen".format(device))

    return s


def codegen_func(adj, device="cpu"):
    # forward header
    # return_type = "void"

    return_type = "void" if adj.return_var is None else adj.return_var.ctype()

    forward_args = []
    reverse_args = []

    # forward args
    for arg in adj.args:
        forward_args.append(arg.ctype() + " var_" + arg.label)
        reverse_args.append(arg.ctype() + " var_" + arg.label)

    # reverse args
    for arg in adj.args:
        # indexed array gradients are regular arrays
        if isinstance(arg.type, indexedarray):
            _arg = Var(arg.label, array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(_arg.ctype() + " & adj_" + arg.label)
        else:
            reverse_args.append(arg.ctype() + " & adj_" + arg.label)

    if return_type != "void":
        reverse_args.append(return_type + " & adj_ret")

    # codegen body
    forward_body = codegen_func_forward(adj, func_type="function", device=device)
    reverse_body = codegen_func_reverse(adj, func_type="function", device=device)

    if device == "cpu":
        template = cpu_function_template
    elif device == "cuda":
        template = cuda_function_template
    else:
        raise ValueError("Device {} is not supported".format(device))

    s = template.format(
        name=make_full_qualified_name(adj.func),
        return_type=return_type,
        forward_args=indent(forward_args),
        reverse_args=indent(reverse_args),
        forward_body=forward_body,
        reverse_body=reverse_body,
        filename=adj.filename,
        lineno=adj.fun_lineno,
    )

    return s


def codegen_kernel(kernel, device, options):
    # Update the module's options with the ones defined on the kernel, if any.
    options = dict(options)
    options.update(kernel.options)

    adj = kernel.adj

    forward_args = ["launch_bounds_t dim"]
    reverse_args = ["launch_bounds_t dim"]

    # forward args
    for arg in adj.args:
        forward_args.append(arg.ctype() + " var_" + arg.label)
        reverse_args.append(arg.ctype() + " var_" + arg.label)

    # reverse args
    for arg in adj.args:
        # indexed array gradients are regular arrays
        if isinstance(arg.type, indexedarray):
            _arg = Var(arg.label, array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(_arg.ctype() + " adj_" + arg.label)
        else:
            reverse_args.append(arg.ctype() + " adj_" + arg.label)

    # codegen body
    forward_body = codegen_func_forward(adj, func_type="kernel", device=device)

    if options["enable_backward"]:
        reverse_body = codegen_func_reverse(adj, func_type="kernel", device=device)
    else:
        reverse_body = ""

    if device == "cpu":
        template = cpu_kernel_template
    elif device == "cuda":
        template = cuda_kernel_template
    else:
        raise ValueError("Device {} is not supported".format(device))

    s = template.format(
        name=kernel.get_mangled_name(),
        forward_args=indent(forward_args),
        reverse_args=indent(reverse_args),
        forward_body=forward_body,
        reverse_body=reverse_body,
    )

    return s


def codegen_module(kernel, device="cpu"):
    adj = kernel.adj

    # build forward signature
    forward_args = ["launch_bounds_t dim"]
    forward_params = ["dim"]

    for arg in adj.args:
        if hasattr(arg.type, "_wp_generic_type_str_"):
            # vectors and matrices are passed from Python by pointer
            forward_args.append(f"const {arg.ctype()}* var_" + arg.label)
            forward_params.append(f"*var_{arg.label}")
        else:
            forward_args.append(f"{arg.ctype()} var_{arg.label}")
            forward_params.append("var_" + arg.label)

    # build reverse signature
    reverse_args = [*forward_args]
    reverse_params = [*forward_params]

    for arg in adj.args:
        if isinstance(arg.type, indexedarray):
            # indexed array gradients are regular arrays
            _arg = Var(arg.label, array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(f"const {_arg.ctype()} adj_{arg.label}")
            reverse_params.append(f"adj_{_arg.label}")
        elif hasattr(arg.type, "_wp_generic_type_str_"):
            # vectors and matrices are passed from Python by pointer
            reverse_args.append(f"const {arg.ctype()}* adj_{arg.label}")
            reverse_params.append(f"*adj_{arg.label}")
        else:
            reverse_args.append(f"{arg.ctype()} adj_{arg.label}")
            reverse_params.append(f"adj_{arg.label}")

    if device == "cpu":
        template = cpu_module_template
    elif device == "cuda":
        template = cuda_module_template
    else:
        raise ValueError("Device {} is not supported".format(device))

    s = template.format(
        name=kernel.get_mangled_name(),
        forward_args=indent(forward_args),
        reverse_args=indent(reverse_args),
        forward_params=indent(forward_params, 3),
        reverse_params=indent(reverse_params, 3),
    )
    return s
