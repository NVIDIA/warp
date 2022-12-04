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
import imp
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


class StructInstance:
    def __init__(self, struct: Struct):
        self.__dict__['_struct_'] = struct
        self.__dict__['_c_struct_'] = struct.ctype()

    def __setattr__(self, name, value):
        assert name in self._struct_.vars, "invalid struct member variable {}".format(name)
        if isinstance(self._struct_.vars[name].type, array):
            if value is None:
                # create array with null pointer
                setattr(self._c_struct_, name, array_t())
            else:                    
                # wp.array
                assert isinstance(value, array)
                assert value.dtype == self._struct_.vars[name].type.dtype, "assign to struct member variable {} failed, expected type {}, got type {}".format(name, self._struct_.vars[name].type.dtype, value.dtype)
                setattr(self._c_struct_, name, value.__ctype__())
        elif issubclass(self._struct_.vars[name].type, ctypes.Array):
            # array type e.g. vec3
            setattr(self._c_struct_, name, self._struct_.vars[name].type(*value))
        else:
            # primitive type
            setattr(self._c_struct_, name, self._struct_.vars[name].type._type_(value))

        self.__dict__[name] = value

    def __repr__(self):
        lines = []
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                lines.append(' ' * 4 + f"{k}: {v.__repr__()}")
        return "StructInstance(\n" + "\n".join(lines) + "\n)\n"


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
            elif issubclass(var.type, ctypes.Array):
                fields.append((label, var.type))
            else:
                fields.append((label, var.type._type_))

        class StructType(ctypes.Structure):
            # if struct is empty, add a dummy field to avoid launch errors on CPU device ("ffi_prep_cif failed")
            _fields_ = fields or [("_dummy_", ctypes.c_int)]

        self.ctype = StructType

        if (module):
            module.register_struct(self)

    def __call__(self):
        '''
        This function returns s = StructInstance(self)
        s uses self.cls as template.
        To enable autocomplete on s, we inherit from self.cls.
        For example,

        @wp.struct
        class A:
            # annotations
            ...

        The type annotations are inherited in A(), allowing autocomplete in kernels
        '''
        # return StructInstance(self)

        class NewStructInstance(self.cls, StructInstance):
            def __init__(inst):
                StructInstance.__init__(inst, self)
        return NewStructInstance()

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
        if (isinstance(self.type, array)):
            return f"array_t<{str(self.type.dtype.__name__)}>"
        if (isinstance(self.type, Struct)):
            return make_full_qualified_name(self.type.cls)
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

    def __init__(adj, func):

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
        adj.arg_types = typing.get_type_hints(func)
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

        adj.symbols = {}            # map from symbols to adjoint variables
        adj.variables = []          # list of local variables (in order)

        adj.cond = None             # condition variable if in branch
        adj.return_var = None       # return type for function or kernel

        # blocks
        adj.blocks = [Block()]

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
                lineno = adj.lineno+adj.fun_lineno
                line = adj.source.splitlines()[adj.lineno]
                msg = f"Error while parsing function \"{adj.fun_name}\" at {adj.filename}:{lineno}:\n{line}\n"
                ex, data, traceback = sys.exc_info()
                e = ex("".join([msg] + list(data.args))).with_traceback(traceback)
            finally:
                raise e

        for a in adj.args:
            if isinstance(a.type, Struct):
                builder.build_struct(a.type)

    # code generation methods
    def format_template(adj, template, input_vars, output_var):

        # output var is always the 0th index
        args = [output_var] + input_vars
        s = template.format(*args)

        return s

    # generates a comma separated list of args
    def format_args(adj, prefix, args):
        s = ""
        sep = ""

        for a in args:
            if type(a) == warp.context.Function:
                
                # functions don't have a var_ prefix so strip it off here
                if (prefix == "var_"):
                    s += sep + a.key
                else:
                    s += sep + prefix + a.key

            else:
                s += sep + prefix + str(a)

            sep = ", "

        return s

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

    def add_var(adj, type=None, constant=None):
        index = len(adj.variables)

        # allocate new variable
        v = Var(str(index), type=type, constant=constant)

        adj.variables.append(v)
        
        adj.blocks[-1].vars.append(v)

        return v

    # append a statement to the forward pass
    def add_forward(adj, statement, replay=None, skip_replay=False):

        adj.blocks[-1].body_forward.append(adj.prefix + statement)

        if skip_replay == False:

            if (replay):
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

    def add_load(adj, input):

        output = adj.add_var(input.type)

        adj.add_forward("var_{} = {};".format(output, input))
        adj.add_reverse("adj_{} += adj_{};".format(input, output))

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
        command = "var_" + str(output) + " = " + (" " + op_string + " ").join(["var_" + str(expr) for expr in exprs]) + ";"
        adj.add_forward(command)

        return output

    def add_call(adj, func, inputs, min_outputs=None):

        # if func is overloaded then perform overload resolution here
        # we validate argument types before they go to generated native code
        resolved_func = None

        for f in func.overloads:
            match = True

            # skip type checking for variadic functions
            if not f.variadic:

                # check argument counts match (todo: default arguments?)
                if len(f.input_types) != len(inputs):
                    match = False
                    continue

                # check argument types equal
                for i, a in enumerate(f.input_types.values()):
                    
                    # if arg type registered as Any, treat as 
                    # template allowing any type to match
                    if a == Any:
                        continue

                    # handle function refs as a special case
                    if a == Callable and type(inputs[i]) is warp.context.Function:
                        continue

                    # otherwise check arg type matches input variable type
                    if not types_equal(a, inputs[i].type):
                        match = False
                        break

            # check output dimensions match expectations
            if min_outputs:

                try:
                    value_type = f.value_func(inputs)
                    if len(value_type) != min_outputs:
                        match = False
                        continue
                except Exception as e:
                    
                    # value func may fail if the user has given 
                    # incorrect args, so we need to catch this
                    match = False
                    continue

            # found a match, use it
            if (match):
                resolved_func = f
                break

        if resolved_func == None:
            
            arg_types = []

            for x in inputs:
                if isinstance(x, Var):
                    # shorten Warp primitive type names
                    if x.type.__module__ == "warp.types":
                        arg_types.append(x.type.__name__)
                    else:
                        arg_types.append(x.type.__module__ + "." + x.type.__name__)
                
                if isinstance(x, warp.context.Function):
                    arg_types.append("function")

            raise Exception(f"Couldn't find function overload for '{func.key}' that matched inputs with types: [{', '.join(arg_types)}]")

        else:
            func = resolved_func


        # if it is a user-function then build it recursively
        if not func.is_builtin():
            adj.builder.build_function(func)

        # evaluate the function type based on inputs
        value_type = func.value_func(inputs)

        # handle expression (zero output), e.g.: void do_something();
        if (value_type == None):

            forward_call = func.namespace + "{}({});".format(func.key, adj.format_args("var_", inputs))
            
            if func.skip_replay:
                adj.add_forward(forward_call, replay="//" + forward_call)
            else:
                adj.add_forward(forward_call)

            if (len(inputs)):
                reverse_call = func.namespace + "{}({}, {});".format("adj_" + func.key, adj.format_args("var_", inputs), adj.format_args("adj_", inputs))
                adj.add_reverse(reverse_call)

            return None

        # handle multiple value functions
        elif (isinstance(value_type, list)):

            output = [adj.add_var(v) for v in value_type]
            forward_call = func.namespace + "{}({});".format(func.key, adj.format_args("var_", inputs+output))
            adj.add_forward(forward_call)

            if (len(inputs)):
                reverse_call = func.namespace + "{}({}, {}, {});".format(
                    "adj_" + func.key, adj.format_args("var_", inputs+output), adj.format_args("adj_", inputs), adj.format_args("adj_", output))
                adj.add_reverse(reverse_call)

            if len(output) == 1:
                return output[0]

            return output

        # handle simple function (one output)
        else:

            output = adj.add_var(func.value_func(inputs))

            forward_call = "var_{} = ".format(output) + func.namespace + "{}({});".format(func.key, adj.format_args("var_", inputs))

            if func.skip_replay:
                adj.add_forward(forward_call, replay="//" + forward_call)
            else:
                adj.add_forward(forward_call)
            
            if (len(inputs)):
                reverse_call = func.namespace + "{}({}, {}, {});".format(
                    "adj_" + func.key, adj.format_args("var_", inputs), adj.format_args("adj_", inputs), adj.format_args("adj_", [output]))
                adj.add_reverse(reverse_call)

            return output

    def add_return(adj, var):

        if (var == None):
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
             reverse.append(adj.prefix + f"\tadj_{i} = 0;")

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

        # evaulate condition in it's own block
        # so we can control replay
        cond_block = adj.begin_block()
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
             reverse.append(f"adj_{i} = 0;")

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


    def eval(adj, node):

        if hasattr(node, "lineno"):
            adj.set_lineno(node.lineno-1)

        # top level entry point for a function
        if (isinstance(node, ast.FunctionDef)):

            out = None
            for f in node.body:
                out = adj.eval(f)

            if 'return' in adj.symbols and adj.symbols['return'] is not None:
                out = adj.symbols['return']
            else:
                out = None
                
            return out

        # if statement
        elif (isinstance(node, ast.If)):

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
            if (len(node.orelse) > 0):

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


            return None

        elif (isinstance(node, ast.Compare)):
            # node.left, node.ops (list of ops), node.comparators (things to compare to)
            # e.g. (left ops[0] node.comparators[0]) ops[1] node.comparators[1]

            left = adj.eval(node.left)
            comps = [adj.eval(comp) for comp in node.comparators]
            op_strings = [builtin_operators[type(op)] for op in node.ops]

            out = adj.add_comp(op_strings, left, comps)

            return out

        elif (isinstance(node, ast.BoolOp)):
            # op, expr list values

            op = node.op
            if isinstance(op, ast.And):
                func = "&&"
            elif isinstance(op, ast.Or):
                func = "||"
            else:
                raise KeyError("Op {} is not supported".format(op))

            out = adj.add_bool_op(func, [adj.eval(expr) for expr in node.values])
            return out

        elif (isinstance(node, ast.Name)):
            # lookup symbol, if it has already been assigned to a variable then return the existing mapping
            if node.id in adj.symbols:
                return adj.symbols[node.id]

            # try and resolve the name using the functions globals context (used to lookup constants + functions)
            elif node.id in adj.func.__globals__:
                obj = adj.func.__globals__[node.id]
                
                if isinstance(obj, warp.constant):
                    # evaluate constant
                    out = adj.add_constant(obj.val)
                    adj.symbols[node.id] = out
                    return out

                elif isinstance(obj, warp.context.Function):
                    # pass back ref. to function (will be converted to name during function call)
                    return obj

                else:
                    raise TypeError(f"'{node.id}' is not a local variable, function, or warp.constant")
                
            else:
                raise KeyError("Referencing undefined symbol: " + str(node.id))

        elif (isinstance(node, ast.Attribute)):
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
                out = Var(attr_name, attr_type)

                return out
            else:

                # try and resolve to either a wp.constant
                # or a wp.func object
                obj = attribute_to_val(node, adj.func.__globals__)
                
                if isinstance(obj, warp.constant):
                    out = adj.add_constant(obj.val)
                    adj.symbols[key] = out          # if referencing a constant
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
                    out = Var(attr_name, attr_type)
                    
                    return out
                else:
                    raise TypeError(f"'{key}' is not a local variable, warp function, nested attribute, or warp constant")


        elif (isinstance(node, ast.Str)):

            # string constant
            return adj.add_constant(node.s)

        elif (isinstance(node, ast.Num)):

            # lookup constant, if it has already been assigned then return existing var
            key = (node.n, type(node.n))

            if (key in adj.symbols):
                return adj.symbols[key]
            else:
                out = adj.add_constant(node.n)
                adj.symbols[key] = out
                return out


        elif (isinstance(node, ast.BinOp)):
            # evaluate binary operator arguments
            left = adj.eval(node.left)
            right = adj.eval(node.right)

            name = builtin_operators[type(node.op)]
            func = warp.context.builtin_functions[name]

            out = adj.add_call(func, [left, right])
            return out

        elif (isinstance(node, ast.UnaryOp)):
            # evaluate unary op arguments
            arg = adj.eval(node.operand)

            name = builtin_operators[type(node.op)]
            func = warp.context.builtin_functions[name]

            out = adj.add_call(func, [arg])
            return out

        elif (isinstance(node, ast.While)):

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

                    if (warp.config.verbose):
                        print("Warning: detected mutated variable {} during a dynamic for-loop, this is a non-differentiable operation".format(sym))

                    if (var1.constant is not None):
                        raise Exception("Error mutating a constant {} inside a dynamic loop, use the following syntax: pi = float(3.141) to declare a dynamic variable".format(sym))
                    
                    # overwrite the old variable value (violates SSA)
                    adj.add_call(warp.context.builtin_functions["copy"], [var1, var2])

                    # reset the symbol to point to the original variable
                    adj.symbols[sym] = var1

            
            adj.end_while()


        elif (isinstance(node, ast.For)):

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
                    if isinstance(obj, warp.constant):
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
                    if isinstance(obj, warp.constant):
                        return obj.val
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

                if (is_constant):

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
                    max_iters = abs(end-start)//abs(step)
                    max_unroll = adj.builder.options["max_unroll"]

                    if max_iters > max_unroll:

                        if (warp.config.verbose):
                            print(f"Warning: fixed-size loop count of {max_iters} is larger than the module 'max_unroll' limit of {max_unroll}, will generate dynamic loop.")
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

                        if (warp.config.verbose):
                            print("Warning: detected mutated variable {} during a dynamic for-loop, this is a non-differentiable operation".format(sym))

                        if (var1.constant is not None):
                            raise Exception("Error mutating a constant {} inside a dynamic loop, use the following syntax: pi = float(3.141) to declare a dynamic variable".format(sym))
                        
                        # overwrite the old variable value (violates SSA)
                        adj.add_call(warp.context.builtin_functions["copy"], [var1, var2])

                        # reset the symbol to point to the original variable
                        adj.symbols[sym] = var1

                adj.end_for(iter)

        elif (isinstance(node, ast.Expr)):
            return adj.eval(node.value)

        elif (isinstance(node, ast.Call)):

            name = None
            
            # try and lookup function in globals by
            # resolving path (e.g.: module.submodule.attr) 
            func, path = adj.resolve_path(node.func)

            if isinstance(func, warp.context.Function) == False:

                if len(path) == 0:
                    raise RuntimeError(f"Unrecognized syntax for function call, path not valid: '{node.func}'")

                # try and lookup function in builtins, this allows users to avoid 
                # using "wp." prefix, and also handles type constructors
                # e.g.: wp.vec3 which aren't explicitly function objects
                attr = path[-1]
                if attr in warp.context.builtin_functions:
                    func = warp.context.builtin_functions[attr]
                else:
                    raise RuntimeError(f"Could not find function {'.'.join(path)} as a built-in or user-defined function. Note that user functions must be annotated with a @wp.func decorator to be called from a kernel.")

            args = []

            # eval all arguments
            for arg in node.args:
                var = adj.eval(arg)
                args.append(var)

            # get expected return count, e.g.: for multi-assignment
            min_outputs = None
            if hasattr(node, "expects"):
                min_outputs = node.expects

            # add var with value type from the function
            out = adj.add_call(func, args, min_outputs)
            return out

        elif (isinstance(node, ast.Index)):
            # the ast.Index node appears in 3.7 versions
            # when performing array slices, e.g.: x = arr[i]
            # but in version 3.8 and higher it does not appear
            return adj.eval(node.value)

        elif (isinstance(node, ast.Subscript)):

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

            if isinstance(target.type, array):
                
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

        elif (isinstance(node, ast.Assign)):

            # handle the case where we are assigning multiple output variables
            if (isinstance(node.targets[0], ast.Tuple)):

                # record the expected number of outputs on the node
                # we do this so we can decide which function to
                # call based on the number of expected outputs
                if isinstance(node.value, ast.Call):
                    node.value.expects = len(node.targets[0].elts)

                # evaluate values
                out = adj.eval(node.value)

                names = []
                for v in node.targets[0].elts:
                    if (isinstance(v, ast.Name)):
                        names.append(v.id)
                    else:
                        raise RuntimeError("Multiple return functions can only assign to simple variables, e.g.: x, y = func()")

                if len(names) != len(out):
                    raise RuntimeError("Multiple return functions need to receive all their output values, incorrect number of values to unpack (expected {}, got {})".format(len(out), len(names)))
                
                for name, rhs in zip(names, out):
                    if (name in adj.symbols):
                        if not types_equal(rhs.type, adj.symbols[name].type):
                            raise TypeError("Error, assigning to existing symbol {} ({}) with different type ({})".format(name, adj.symbols[name].type, rhs.type))

                    adj.symbols[name] = rhs

                return out

            # handles the case where we are assigning to an array index (e.g.: arr[i] = 2.0)
            elif (isinstance(node.targets[0], ast.Subscript)):
                
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

                if (isinstance(target.type, array)):
                    adj.add_call(warp.context.builtin_functions["store"], [target, *indices, value])
                else:
                    raise RuntimeError("Can only subscript assign array types")

                return None

            elif (isinstance(node.targets[0], ast.Name)):

                # symbol name
                name = node.targets[0].id

                # evaluate rhs
                rhs = adj.eval(node.value)

                # check type matches if symbol already defined
                if (name in adj.symbols):
                
                    if (rhs.type != adj.symbols[name].type):
                        raise TypeError("Error, assigning to existing symbol {} ({}) with different type ({})".format(name, adj.symbols[name].type, rhs.type))

                # handle simple assignment case (a = b), where we generate a value copy rather than reference
                if isinstance(node.value, ast.Name):
                    out = adj.add_var(rhs.type)
                    adj.add_call(warp.context.builtin_functions["copy"], [out, rhs])
                else:
                    out = rhs

                # update symbol map (assumes lhs is a Name node)
                adj.symbols[name] = out
                return out

            elif (isinstance(node.targets[0], ast.Attribute)):
                raise RuntimeError("Error, assignment to member variables is not currently support (structs are immutable)")

            else:
                raise RuntimeError("Error, unsupported assignment statement.")

        elif (isinstance(node, ast.Return)):
            cond = adj.cond

            out = adj.eval(node.value)
            adj.symbols['return'] = out

            if out is not None:        # set return type of function
                return_var = out
                if adj.return_var is not None and adj.return_var.ctype() != return_var.ctype():
                    raise TypeError(f"Error, function returned different types, previous: {adj.return_var.ctype()}, new {return_var.ctype()}")
                adj.return_var = return_var

            adj.add_return(out)

            return out

        elif (isinstance(node, ast.AugAssign)):
            
            # convert inplace operations (+=, -=, etc) to ssa form, e.g.: c = a + b
            left = adj.eval(node.target)
            right = adj.eval(node.value)

            # lookup
            name = builtin_operators[type(node.op)]
            func = warp.context.builtin_functions[name]

            out = adj.add_call(func, [left, right])

            # update symbol map
            adj.symbols[node.target.id] = out

        elif (isinstance(node, ast.NameConstant)):
            if node.value == True:
                out = adj.add_constant(True)
            elif node.value == False:
                out = adj.add_constant(False)
            elif node.value == None:
                raise TypeError("None type unsupported")

            return out

        elif node is None:
            return None
        else:
            raise Exception("Error, ast node of type {} not supported".format(type(node)))



    # helper to evaluate expressions of the form
    # obj1.obj2.obj3.attr in the function's global scope
    def resolve_path(adj, node):

        modules = []

        while isinstance(node, ast.Attribute):
            modules.append(node.attr)
            node = node.value

        if (isinstance(node, ast.Name)):
            modules.append(node.id)

        # reverse list since ast presents it backward order
        path = [*reversed(modules)]

        if len(path) == 0:
            return None, path

        # try and evaluate object path
        try:
            func = eval(".".join(path), adj.func.__globals__)
            return func, path
        except Exception as e:
            return None, path
        

    # annotate generated code with the original source code line
    def set_lineno(adj, lineno):
        if adj.lineno is None or adj.lineno != lineno:
            line = lineno + adj.fun_lineno
            source = adj.raw_source[lineno].strip().ljust(70)
            adj.add_forward(f'// {source}       <L {line}>')
            adj.add_reverse(f'// adj: {source}  <L {line}>')
        adj.lineno = lineno
        

#----------------
# code generation

cpu_module_header = '''
#include "../native/builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

using namespace wp;

'''

cuda_module_header = '''
#include "../native/builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)


using namespace wp;

'''

struct_template = '''
struct {name}
{{
{struct_body}
}};

'''

cpu_function_template = '''
// {filename}:{lineno}
static {return_type} {name}({forward_args})
{{
{forward_body}
}}

// {filename}:{lineno}
static void adj_{name}({reverse_args})
{{
{reverse_body}
}}

'''

cuda_function_template = '''
// {filename}:{lineno}
static CUDA_CALLABLE {return_type} {name}({forward_args})
{{
{forward_body}
}}

// {filename}:{lineno}
static CUDA_CALLABLE void adj_{name}({reverse_args})
{{
{reverse_body}
}}

'''

cuda_kernel_template = '''

extern "C" __global__ void {name}_cuda_kernel_forward({forward_args})
{{
    int _idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

{forward_body}
}}

extern "C" __global__ void {name}_cuda_kernel_backward({reverse_args})
{{
    int _idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

{reverse_body}
}}

'''

cpu_kernel_template = '''

void {name}_cpu_kernel_forward({forward_args})
{{
{forward_body}
}}

void {name}_cpu_kernel_backward({reverse_args})
{{
{reverse_body}
}}

'''

cuda_module_template = '''

extern "C" {{

// Python entry points
WP_API void {name}_cuda_forward(void* stream, {forward_args})
{{
    {name}_cuda_kernel_forward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>({forward_params});
}}

WP_API void {name}_cuda_backward(void* stream, {reverse_args})
{{
    {name}_cuda_kernel_backward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>({reverse_params});
}}

}} // extern C

'''

cpu_module_template = '''

extern "C" {{

// Python CPU entry points
WP_API void {name}_cpu_forward({forward_args})
{{
    set_launch_bounds(dim);

    for (int i=0; i < dim.size; ++i)
    {{
        s_threadIdx = i;

        {name}_cpu_kernel_forward({forward_params});
    }}
}}

WP_API void {name}_cpu_backward({reverse_args})
{{
    set_launch_bounds(dim);

    for (int i=0; i < dim.size; ++i)
    {{
        s_threadIdx = i;

        {name}_cpu_kernel_backward({reverse_params});
    }}
}}

}} // extern C

'''

cuda_module_header_template = '''

extern "C" {{

// Python CUDA entry points
WP_API void {name}_cuda_forward(void* stream, {forward_args});

WP_API void {name}_cuda_backward(void* stream, {reverse_args});

}} // extern C
'''

cpu_module_header_template = '''

extern "C" {{

// Python CPU entry points
WP_API void {name}_cpu_forward({forward_args});

WP_API void {name}_cpu_backward({reverse_args});

}} // extern C
'''

# converts a constant Python value to equivalent C-repr
def constant_str(value):
    
    if type(value) == bool:
        if value:
            return "true"
        else:
            return "false"

    elif type(value) == str:
        # ensure constant strings are correctly escaped
        return "\"" + str(value.encode("unicode-escape").decode()) + "\""

    elif isinstance(value, ctypes.Array):
        return "{" + ", ".join(map(str, value)) + "}"

    else:
        # otherwise just convert constant to string
        return str(value)

def indent(args, stops=1):
    sep = ",\n"
    for i in range(stops):
        sep += "\t"

    #return sep + args.replace(", ", "," + sep)
    return sep.join(args)

# generates a C function name based on the python function name
def make_full_qualified_name(func):
    return re.sub('[^0-9a-zA-Z_]+', '', func.__qualname__.replace('.', '__'))

def codegen_struct(struct, indent=4):
    body = []
    indent_block = " " * indent
    for label, var in struct.vars.items():
        assert not (isinstance(var.type, Struct))

        body.append(var.ctype() + " " + label + ";\n")

    return struct_template.format(
        name=make_full_qualified_name(struct.cls),
        struct_body="".join([indent_block + l for l in body])
    )


def codegen_func_forward_body(adj, device='cpu', indent=4):
    body = []
    indent_block = " " * indent

    for f in adj.blocks[0].body_forward:
        body += [f + "\n"]

    return "".join([indent_block + l for l in body])


def codegen_func_forward(adj, func_type='kernel', device='cpu'):
    s = ""

    # primal vars
    s += "    //---------\n"
    s += "    // primal vars\n"

    for var in adj.variables:    
        if var.constant == None:
            s += "    " + var.ctype() + " var_" + str(var.label) + ";\n"
        else:
            s += "    const " + var.ctype() + " var_" + str(var.label) + " = " + constant_str(var.constant) + ";\n"


    # forward pass
    s += "    //---------\n"
    s += "    // forward\n"

    if device == 'cpu':
        s += codegen_func_forward_body(adj, device=device, indent=4)

    elif device == 'cuda':
        if func_type == 'kernel':
            s += codegen_func_forward_body(adj, device=device, indent=8)
        else:
            s += codegen_func_forward_body(adj, device=device, indent=4)

    return s


def codegen_func_reverse_body(adj, device='cpu', indent=4):
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


def codegen_func_reverse(adj, func_type='kernel', device='cpu'):
    s = ""

    # primal vars
    s += "    //---------\n"
    s += "    // primal vars\n"

    for var in adj.variables:
        if var.constant == None:
            s += "    " + var.ctype() + " var_" + str(var.label) + ";\n"
        else:
            s += "    const " + var.ctype() + " var_" + str(var.label) + " = " + constant_str(var.constant) + ";\n"

    # dual vars
    s += "    //---------\n"
    s += "    // dual vars\n"

    for var in adj.variables:
        s += "    " + var.ctype() + " adj_" + str(var.label) + " = 0;\n"

    if device == 'cpu':
        s += codegen_func_reverse_body(adj, device=device, indent=4)
    elif device == 'cuda':
        if func_type == 'kernel':
            s += codegen_func_reverse_body(adj, device=device, indent=8)
        else:
            s += codegen_func_reverse_body(adj, device=device, indent=4)
    else:
        raise ValueError("Device {} not supported for codegen".format(device))

    return s


def codegen_func(adj, device='cpu'):

    # forward header
    # return_type = "void"

    return_type = 'void' if adj.return_var is None else adj.return_var.ctype()

    forward_args = []
    reverse_args = []

    # forward args
    for arg in adj.args:
        forward_args.append(arg.ctype() + " var_" + arg.label)
        reverse_args.append(arg.ctype() + " var_" + arg.label)

    # reverse args
    for arg in adj.args:
        reverse_args.append(arg.ctype() + " & adj_" + arg.label)
    
    if return_type != 'void':
        reverse_args.append(return_type + " & adj_ret")

    # codegen body
    forward_body = codegen_func_forward(adj, func_type='function', device=device)
    reverse_body = codegen_func_reverse(adj, func_type='function', device=device)

    if device == 'cpu':
        template = cpu_function_template
    elif device == 'cuda':
        template = cuda_function_template
    else:
        raise ValueError("Device {} is not supported".format(device))

    s = template.format(name=make_full_qualified_name(adj.func),
                        return_type=return_type,
                        forward_args=indent(forward_args),
                        reverse_args=indent(reverse_args),
                        forward_body=forward_body,
                        reverse_body=reverse_body,
                        filename=adj.filename,
                        lineno=adj.fun_lineno)

    return s


def codegen_kernel(kernel, device, options):

    adj = kernel.adj

    forward_args = ["launch_bounds_t dim"]
    reverse_args = ["launch_bounds_t dim"]

    # forward args
    for arg in adj.args:
        forward_args.append(arg.ctype() + " var_" + arg.label)
        reverse_args.append(arg.ctype() + " var_" + arg.label)

    # reverse args
    for arg in adj.args:
        reverse_args.append(arg.ctype() + " adj_" + arg.label)

    # codegen body
    forward_body = codegen_func_forward(adj, func_type='kernel', device=device)

    if options["enable_backward"]:
        reverse_body = codegen_func_reverse(adj, func_type='kernel', device=device)
    else:
        reverse_body = ""


    if device == 'cpu':
        template = cpu_kernel_template
    elif device == 'cuda':
        template = cuda_kernel_template
    else:
        raise ValueError("Device {} is not supported".format(device))


    s = template.format(name=kernel.key,
                        forward_args=indent(forward_args),
                        reverse_args=indent(reverse_args),
                        forward_body=forward_body,
                        reverse_body=reverse_body)

    return s


def codegen_module(kernel, device='cpu'):

    adj = kernel.adj

    # build forward signature
    forward_args = ["launch_bounds_t dim"]
    forward_params = ["dim"]

    for arg in adj.args:
        forward_args.append(arg.ctype() + " var_" + arg.label)
        forward_params.append("var_" + arg.label)

    # build reverse signature
    reverse_args = [*forward_args]
    reverse_params = [*forward_params]

    for arg in adj.args:
        reverse_args.append(arg.ctype() + " adj_" + arg.label)
        reverse_params.append("adj_" + arg.label)

    if device == 'cpu':
        template = cpu_module_template
    elif device == 'cuda':
        template = cuda_module_template
    else:
        raise ValueError("Device {} is not supported".format(device))

    s = template.format(name=kernel.key,
                        forward_args=indent(forward_args),
                        reverse_args=indent(reverse_args),
                        forward_params=indent(forward_params, 3),
                        reverse_params=indent(reverse_params, 3))
    return s



