# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
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
            return str(self.type.dtype.__name__) + "*"
        else:
            return str(self.type.__name__)


#------------------------------------------------------------------------
# Source code transformer, this class takes a Python function and
# computes its adjoint using single-pass translation of the function's AST


class Adjoint:


    def __init__(adj, func):

        adj.func = func

        adj.symbols = {}     # map from symbols to adjoint variables
        adj.variables = []   # list of local variables (in order)
        adj.args = []        # list of function arguments (in order)

        adj.cond = None                # condition variable if in branch
        adj.return_var = None          # return type for function or kernel

        # build AST from function object
        adj.source = inspect.getsource(func)
        
        # ensures that indented class methods can be parsed as kernels
        adj.source = textwrap.dedent(adj.source)    

        # build AST
        adj.tree = ast.parse(adj.source)

        # parse argument types
        arg_types = typing.get_type_hints(func)

        # add variables and symbol map for each argument
        for name, t in arg_types.items():
            adj.symbols[name] = Var(name, t, False)

        # build ordered list of args
        for a in adj.tree.body[0].args.args:
            adj.args.append(adj.symbols[a.arg])

        # primal statements (allows different statements in replay)
        adj.body_forward = []
        adj.body_forward_replay = []
        adj.body_reverse = []

        adj.output = []

        adj.indent_count = 0
        adj.label_count = 0

    # generate function ssa form and adjoint
    def build(adj, builtin_fuctions, user_functions, options):

        adj.builtin_functions = builtin_fuctions
        adj.user_functions = user_functions
        adj.options = options

        # recursively evaluate function body
        adj.eval(adj.tree.body[0])


    # code generation methods
    def format_template(adj, template, input_vars, output_var):

        # output var is always the 0th index
        args = [output_var] + input_vars
        s = template.format(*args)

        return s

    # generates a comma separated list of args
    def format_args(adj, prefix, indices):
        args = ""
        sep = ""

        for i in indices:
            args += sep + prefix + str(i)
            sep = ", "

        return args

    def add_var(adj, type=None, constant=None):
        index = len(adj.variables)

        # allocate new variable
        v = Var(str(index), type=type, constant=constant)
        adj.variables.append(v)

        return v

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

    def add_call(adj, func, inputs):

        # if func is overloaded then perform overload resolution here, this is just to try and catch
        # argument errors before they go to generated native code

        if (isinstance(func, list)):
    
            resolved_func = None

            for f in func:
                match = True

                if (f.variadic == False):
                  
                    # check argument counts match (todo: default arguments?)
                    if len(f.input_types) != len(inputs):
                        match = False
                        continue

                    # check argument types equal
                    for i, a in enumerate(f.input_types.values()):
                        
                        # if arg type registered as None, treat as 
                        # template allowing any type to match
                        if inputs[i].type == Any:
                            continue

                        # otherwise check args match signature
                        if not types_equal(a, inputs[i].type):
                            match = False
                            break

                # found a match, use it
                if (match):
                    resolved_func = f
                    break

            if (resolved_func == None):
                arg_types = "".join(str(x.type) + ", " for x in inputs)

                raise Exception(f"Couldn't find function overload for {func[0].key} that matched inputs {arg_types}")
            else:
                func = resolved_func

        # expression (zero output), e.g.: void do_something();
        if (func.value_type(inputs) == None):

            forward_call = func.namespace + "{}({});".format(func.key, adj.format_args("var_", inputs))
            adj.add_forward(forward_call)

            if (len(inputs)):
                reverse_call = func.namespace + "{}({}, {});".format("adj_" + func.key, adj.format_args("var_", inputs), adj.format_args("adj_", inputs))
                adj.add_reverse(reverse_call)

            return None

        # function (one output)
        else:

            output = adj.add_var(func.value_type(inputs))

            forward_call = "var_{} = ".format(output) + func.namespace + "{}({});".format(func.key, adj.format_args("var_", inputs))
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

        adj.indent_count += 1

    def end_if(adj, cond):

        adj.indent_count -= 1

        adj.add_forward("}")
        adj.add_reverse(f"if (var_{cond}) {{")

    def begin_else(adj, cond):

        adj.add_forward(f"if (!var_{cond}) {{")
        adj.add_reverse("}")

        adj.indent_count += 1

    def end_else(adj, cond):

        adj.indent_count -= 1

        adj.add_forward("}")
        adj.add_reverse(f"if (!var_{cond}) {{")


    # define a for-loop
    def begin_for(adj, iter, start, end, step):

        # note that dynamic for-loops must not mutate any previous state, so we don't need to re-run them in the reverse pass        
        adj.add_forward(f"for (var_{iter}=var_{start}; cmp(var_{iter}, var_{end}, var_{step}); var_{iter} += var_{step}) {{""", statement_replay="if (false) {")
        adj.add_reverse("}")

        adj.indent_count += 1

    def end_for(adj, iter, start, end, step):

        adj.indent_count -= 1

        # run loop in reverse order for gradient computation
        adj.add_forward("}")
        adj.add_reverse(f"for (var_{iter}=var_{end}-1; cmp(var_{iter}, var_{start}, -var_{step}); var_{iter} -= var_{step}) {{")

    # define a while loop, todo: reverse mode
    def begin_while(adj, cond):

        adj.add_forward("while (1) {")

        adj.indent_count += 1

        c = adj.eval(cond)
        adj.add_forward(f"if (var_{c} == false) break;")

    def end_while(adj):

        adj.indent_count -= 1

        adj.add_forward("}")


    # append a statement to the forward pass
    def add_forward(adj, statement, statement_replay=None):

        prefix = ""
        for i in range(adj.indent_count):
            prefix += "\t"

        adj.body_forward.append(prefix + statement)

        # allow for different statement in reverse kernel replay
        if (statement_replay):
            adj.body_forward_replay.append(prefix + statement_replay)
        else:
            adj.body_forward_replay.append(prefix + statement)

    # append a statement to the reverse pass
    def add_reverse(adj, statement):

        prefix = ""
        for i in range(adj.indent_count):
            prefix += "\t"

        adj.body_reverse.append(prefix + statement)

    def eval(adj, node):

        try:

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
                        out = adj.add_call(adj.builtin_functions["select"], [cond, var1, var2])
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
                        out = adj.add_call(adj.builtin_functions["select"], [cond, var2, var1])
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
                elif node.id in adj.func.__globals__:
                    obj = adj.func.__globals__[node.id]
                    if not isinstance(obj, warp.constant):
                        raise TypeError(f"'{node.id}' is not a local variable or of type warp.constant")
                    out = adj.add_constant(obj.val)
                    adj.symbols[node.id] = out
                    return out
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
                        return context[node.id]
                    elif isinstance(node, ast.Attribute):
                        return getattr(attribute_to_val(node.value, context), node.attr)
                    else:
                        raise RuntimeError(f"Failed to parse attribute")

                key = attribute_to_str(node)

                if key in adj.symbols:
                    return adj.symbols[key]
                else:
                    obj = attribute_to_val(node, adj.func.__globals__)
                    if not isinstance(obj, warp.constant):
                        raise TypeError(f"'{key}' is not a local variable or of type warp.constant")
                    out = adj.add_constant(obj.val)
                    adj.symbols[key] = out
                    return out

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
                func = adj.builtin_functions[name]

                out = adj.add_call(func, [left, right])
                return out

            elif (isinstance(node, ast.UnaryOp)):
                # evaluate unary op arguments
                arg = adj.eval(node.operand)

                name = builtin_operators[type(node.op)]
                func = adj.builtin_functions[name]

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
                        adj.add_call(adj.builtin_functions["copy"], [var1, var2])

                        # reset the symbol to point to the original variable
                        adj.symbols[sym] = var1

                
                adj.end_while()


            elif (isinstance(node, ast.For)):

                unroll = True
                for a in node.iter.args:

                    # if all range() arguments are numeric constants we will unroll
                    # note that this only handles trivial constants, it will not unroll
                    # constant-time expressions (e.g.: range(0, 3*2))
                    if (isinstance(a, ast.Num) == False):
                        unroll = False
                        break

                if (unroll):

                    # range(end)
                    if len(node.iter.args) == 1:
                        start = 0
                        end = node.iter.args[0].n
                        step = 1

                    # range(start, end)
                    elif len(node.iter.args) == 2:
                        start = node.iter.args[0].n
                        end = node.iter.args[1].n
                        step = 1

                    # range(start, end, step)
                    elif len(node.iter.args) == 3:
                        start = node.iter.args[0].n
                        end = node.iter.args[1].n
                        step = node.iter.args[2].n

                    # test if we're above max unroll count
                    max_iters = abs(end-start)//abs(step)
                    max_unroll = adj.options["max_unroll"]

                    if max_iters > max_unroll:
                        unroll = False

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
              
                if unroll == False:

                    # dynamic loop, body must be side-effect free, i.e.: not
                    # overwrite memory locations used by previous operations

                    # range(end)
                    if len(node.iter.args) == 1:                        
                        start = adj.add_constant(0)
                        end = adj.eval(node.iter.args[0])
                        step = adj.add_constant(1)

                    # range(start, end)
                    elif len(node.iter.args) == 2:
                        start = adj.eval(node.iter.args[0])
                        end = adj.eval(node.iter.args[1])
                        step = adj.add_constant(1)

                    # range(start, end, step)
                    elif len(node.iter.args) == 3:
                        start = adj.eval(node.iter.args[0])
                        end = adj.eval(node.iter.args[1])
                        step = adj.eval(node.iter.args[2])

                    # add iterator variable
                    iter = adj.add_var(int)
                    adj.symbols[node.target.id] = iter

                    # save symbol table
                    symbols_prev = adj.symbols.copy()

                    adj.begin_for(iter, start, end, step)

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
                            adj.add_call(adj.builtin_functions["copy"], [var1, var2])

                            # reset the symbol to point to the original variable
                            adj.symbols[sym] = var1

                    adj.end_for(iter, start, end, step)

            elif (isinstance(node, ast.Expr)):
                return adj.eval(node.value)

            elif (isinstance(node, ast.Call)):

                name = None

                # determine if call is to a builtin (e.g.: wp.cos(x)), or to a free-func, e.g.: my_func(x)
                if (isinstance(node.func, ast.Attribute)):
                    name = node.func.attr
                elif (isinstance(node.func, ast.Name)):
                    name = node.func.id

                # built in function
                if name in adj.builtin_functions:
                    func = adj.builtin_functions[name]

                # user-defined function in this module
                elif name in adj.user_functions:
                    func = adj.user_functions[name]

                else:
                    raise KeyError("Could not find function {}".format(name))

                args = []

                # eval all arguments
                for arg in node.args:
                    var = adj.eval(arg)
                    args.append(var)

                # add var with value type from the function
                out = adj.add_call(func, args)
                return out

            elif (isinstance(node, ast.Index)):
                # the ast.Index node appears in 3.7 versions 
                # when performing array slices, e.g.: x = arr[i]
                # but in version 3.8 and higher it does not appear
                return adj.eval(node.value)

            elif (isinstance(node, ast.Subscript)):

                target = adj.eval(node.value)

                if isinstance(target.type, array):
                    
                    # handles the case where we are indexing into an array, e.g.: x = arr[i]
                    index = adj.eval(node.slice)
                    out = adj.add_call(adj.builtin_functions["load"], [target, index])
                    return out

                else:

                    # handles non-array types, e.g: vec3, mat33, etc
                    indices = []

                    if isinstance(node.slice, ast.Tuple):
                        # handles the M[i, j] case (Python 3.8.x upward)
                        for arg in node.slice.elts:
                            var = adj.eval(arg)
                            indices.append(var)

                    elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Tuple):
                        # handles the M[i, j] case (Python 3.7.x)
                        for arg in node.slice.value.elts:
                            var = adj.eval(arg)
                            indices.append(var)
                    else:
                        # simple expression
                        var = adj.eval(node.slice)
                        indices.append(var)

                    out = adj.add_call(adj.builtin_functions["index"], [target, *indices])
                    return out

            elif (isinstance(node, ast.Assign)):

                # handles the case where we are assigning to an array index (e.g.: arr[i] = 2.0)
                if (isinstance(node.targets[0], ast.Subscript)):
                    
                    target = adj.eval(node.targets[0].value)
                    index = adj.eval(node.targets[0].slice)
                    value = adj.eval(node.value)

                    if (isinstance(target.type, array)):
                        adj.add_call(adj.builtin_functions["store"], [target, index, value])
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
                            raise TypeError("error, assigning to existing symbol {} ({}) with different type ({})".format(name, adj.symbols[name].type, rhs.type))

                    # handle simple assignment case (a = b), where we generate a value copy rather than reference
                    if isinstance(node.value, ast.Name):
                        out = adj.add_var(rhs.type)
                        adj.add_call(adj.builtin_functions["copy"], [out, rhs])
                    else:
                        out = rhs

                    # update symbol map (assumes lhs is a Name node)
                    adj.symbols[name] = out
                    return out

            elif (isinstance(node, ast.Return)):
                cond = adj.cond  # None if not in branch, else branch boolean

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
                func = adj.builtin_functions[name]

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

        except Exception as e:

            # print error / line number
            lines = adj.source.splitlines()
            print("Error: {} while transforming node {} in func: {} at line: {} col: {}: \n    {}".format(e, type(node), adj.func.__name__, node.lineno, node.col_offset, lines[max(node.lineno-1, 0)]))
            raise


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

cpu_function_template = '''
static {return_type} {name}({forward_args})
{{
    {forward_body}
}}

static void adj_{name}({reverse_args})
{{
    {reverse_body}
}}

'''

cuda_function_template = '''
static CUDA_CALLABLE {return_type} {name}({forward_args})
{{
    {forward_body}
}}

static CUDA_CALLABLE void adj_{name}({reverse_args})
{{
    {reverse_body}
}}

'''

cuda_kernel_template = '''

extern "C" __global__ void {name}_cuda_kernel_forward({forward_args})
{{
    {forward_body}
}}

extern "C" __global__ void {name}_cuda_kernel_backward({reverse_args})
{{
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
    {name}_cuda_kernel_forward<<<(dim + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>({forward_params});
}}

WP_API void {name}_cuda_backward(void* stream, {reverse_args})
{{
    {name}_cuda_kernel_backward<<<(dim + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>({reverse_params});
}}

}} // extern C

'''

cpu_module_template = '''

// Python CPU entry points
WP_API void {name}_cpu_forward({forward_args})
{{
    for (int i=0; i < dim; ++i)
    {{
        s_threadIdx = i;

        {name}_cpu_kernel_forward({forward_params});
    }}
}}

WP_API void {name}_cpu_backward({reverse_args})
{{
    for (int i=0; i < dim; ++i)
    {{
        s_threadIdx = i;

        {name}_cpu_kernel_backward({reverse_params});
    }}
}}

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
    sep = "\n"
    for i in range(stops):
        sep += "\t"

    return sep + args.replace(", ", "," + sep)


def codegen_func_forward_body(adj, device='cpu', indent=4):
    body = []
    indent_block = " " * indent

    for f in adj.body_forward:
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
            s += "    int var_idx = blockDim.x * blockIdx.x + threadIdx.x;\n"
            s += "    if (var_idx < dim) {\n"

            s += codegen_func_forward_body(adj, device=device, indent=8)

            s += "    }\n"
        else:
            s += codegen_func_forward_body(adj, device=device, indent=4)

    return s


def codegen_func_reverse_body(adj, device='cpu', indent=4):
    body = []
    indent_block = " " * indent

    # forward pass
    body += ["//---------\n"]
    body += ["// forward\n"]

    for f in adj.body_forward_replay:
        body += [f + "\n"]

    # reverse pass
    body += ["//---------\n"]
    body += ["// reverse\n"]

    for l in reversed(adj.body_reverse):
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
            s += "    int var_idx = blockDim.x * blockIdx.x + threadIdx.x;\n"
            s += "    if (var_idx < dim) {\n"
            s += codegen_func_reverse_body(adj, device=device, indent=8)
            s += "    }\n"
        else:
            s += codegen_func_reverse_body(adj, device=device, indent=4)
    else:
        raise ValueError("Device {} not supported for codegen".format(device))

    return s


def codegen_func(adj, device='cpu'):

    # forward header
    # return_type = "void"

    return_type = 'void' if adj.return_var is None else adj.return_var.ctype()

    forward_args = ""
    reverse_args = ""
    # s = ""

    # forward args
    sep = ""
    for arg in adj.args:
        forward_args += sep + arg.ctype() + " var_" + arg.label
        reverse_args += sep + arg.ctype() + " var_" + arg.label
        sep = ", "

    # reverse args
    sep = ","
    for arg in adj.args:
        if "*" in arg.ctype():
            reverse_args += sep + arg.ctype() + " adj_" + arg.label
        else:
            reverse_args += sep + arg.ctype() + " & adj_" + arg.label
        sep = ", "

    reverse_args += sep + return_type + " & adj_ret"

    # codegen body
    forward_body = codegen_func_forward(adj, func_type='function', device=device)
    reverse_body = codegen_func_reverse(adj, func_type='function', device=device)

    if device == 'cpu':
        template = cpu_function_template
    elif device == 'cuda':
        template = cuda_function_template
    else:
        raise ValueError("Device {} is not supported".format(device))

    s = template.format(name=adj.func.__name__,
                        return_type=return_type,
                        forward_args=indent(forward_args),
                        reverse_args=indent(reverse_args),
                        forward_body=forward_body,
                        reverse_body=reverse_body)

    return s


def codegen_kernel(kernel, device='cpu'):

    adj = kernel.adj

    forward_args = "int dim"
    reverse_args = "int dim"

    # forward args
    sep = ","
    for arg in adj.args:
        forward_args += sep + arg.ctype() + " var_" + arg.label
        reverse_args += sep + arg.ctype() + " var_" + arg.label
        sep = ", "

    # reverse args
    sep = ","
    for arg in adj.args:
        reverse_args += sep + arg.ctype() + " adj_" + arg.label
        sep = ", "

    # codegen body
    forward_body = codegen_func_forward(adj, func_type='kernel', device=device)
    reverse_body = codegen_func_reverse(adj, func_type='kernel', device=device)


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
    forward_args = "int dim"
    forward_params = "dim"

    sep = ","
    for arg in adj.args:
        if (isinstance(arg.type, array)):
            forward_args += sep + "wp::array var_" + arg.label
            forward_params += sep + "cast<" + arg.ctype() + ">(var_" + arg.label + ")"
        else:
            forward_args += sep + arg.ctype() + " var_" + arg.label
            forward_params += sep + "var_" + arg.label

        sep = ", "


    # build reverse signature
    reverse_args = forward_args
    reverse_params = forward_params

    sep = ","
    for arg in adj.args:
        if (isinstance(arg.type, array)):
            reverse_args += sep + "wp::array adj_" + arg.label
            reverse_params += sep + "cast<" + arg.ctype() + ">(adj_" + arg.label + ")"
        else:
            reverse_args += sep + arg.ctype() + " adj_" + arg.label
            reverse_params += sep + "adj_" + arg.label

        sep = ", "

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


def codegen_module_decl(kernel, device='cpu'):

    adj = kernel.adj

    # build forward signature
    forward_args = "int dim"
    forward_params = "dim"

    sep = ","
    for arg in adj.args:
        if (isinstance(arg.type, array)):
            forward_args += sep + "wp::array var_" + arg.label
            forward_params += sep + "cast<" + arg.ctype() + ">(var_" + arg.label + ")"
        else:
            forward_args += sep + arg.ctype() + " var_" + arg.label
            forward_params += sep + "var_" + arg.label

        sep = ", "

    # build reverse signature
    reverse_args = forward_args
    reverse_params = forward_params

    sep = ","
    for arg in adj.args:
        if (isinstance(arg.type, array)):
            reverse_args += sep + "wp::array adj_" + arg.label
            reverse_params += sep + "cast<" + arg.ctype() + ">(adj_" + arg.label + ")"
        else:
            reverse_args += sep + arg.ctype() + " adj_" + arg.label
            reverse_params += sep + "adj_" + arg.label

        sep = ", "

    if device == 'cpu':
        template = cpu_module_header_template
    elif device == 'cuda':
        template = cuda_module_header_template
    else:
        raise ValueError("Device {} is not supported".format(device))

    s = template.format(name=kernel.key, forward_args=indent(forward_args), reverse_args=indent(reverse_args))
    return s

