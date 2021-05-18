import os
import sys
import imp
import ast
import math
import inspect
import typing
import weakref
import ctypes
import numpy as np

import copy

from oglang.types import *


# map operator to function
builtin_operators = {}

builtin_operators[ast.Add] = "add"
builtin_operators[ast.Sub] = "sub"
builtin_operators[ast.Mult] = "mul"
builtin_operators[ast.Div] = "div"
builtin_operators[ast.FloorDiv] = "div"
builtin_operators[ast.Mod] = "mod"

builtin_operators[ast.Gt] = ">"
builtin_operators[ast.Lt] = "<"
builtin_operators[ast.GtE] = ">="
builtin_operators[ast.LtE] = "<="
builtin_operators[ast.Eq] = "=="
builtin_operators[ast.NotEq] = "!="


class Var:
    def __init__(adj, label, type, requires_grad=False, constant=None):

        # convert built-in types to og types
        if (type == float):
            type = float32
        elif (type == int):
            type = int32

        adj.label = label
        adj.type = type
        adj.requires_grad = requires_grad
        adj.constant = constant

    def __str__(adj):
        return adj.label

    def ctype(self):
        if (isinstance(self.type, array)):
            if self.type.dtype == vec3:
                return str("og::" + self.type.dtype.__name__) + "*"

            return str(self.type.dtype.__name__) + "*"
        elif self.type == vec3:
            return "og::" + str(self.type.__name__)
        else:
            return str(self.type.__name__)


#--------------------
# Storage class for partial AST up to a return statement.


class Stmt:
    def __init__(self, cond, forward, forward_replay, reverse, ret_forward, ret_line):
        self.cond = cond               # condition, can be None
        self.forward = forward         # all forward code outside of conditional branch *since last return*
        self.forward_replay = forward_replay
        self.reverse = reverse         # all reverse code including the reverse of any code in ret_forward

        self.ret_forward = ret_forward           # all forward commands in the return statement except the actual return statement
        self.ret_line = ret_line                 # actual return statement


#------------------------------------------------------------------------
# Source code transformer, this class takes a Python function and
# computes its adjoint using single-pass translation of the function's AST


class Adjoint:


    def __init__(adj, func, builtin_fuctions, user_functions, device='cpu'):

        adj.func = func
        adj.device = device
        adj.builtin_functions = builtin_fuctions
        adj.user_functions = user_functions

        adj.symbols = {}     # map from symbols to adjoint variables
        adj.variables = []   # list of local variables (in order)
        adj.args = []        # list of function arguments (in order)

        adj.cond = None                # condition variable if in branch
        adj.return_var = None          # return type for function or kernel

        # build AST from function object
        adj.source = inspect.getsource(func)
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

        #adj.add_forward("var_{} = {};".format(output, n))
        return output

    def add_load(adj, input):

        output = adj.add_var(input.type)

        adj.add_forward("var_{} = {};".format(output, input))
        adj.add_reverse("adj_{} += adj_{};".format(input, output))

        return output

    def add_operator(adj, op, inputs):

        # todo: just using first input as the output type, would need some
        # type inference here to support things like vec3 = float*vec3

        output = adj.add_var(inputs[0].type)

        transformer = builtin_operators[op.__class__]

        for t in transformer.forward():
            adj.add_forward(adj.format_template(t, inputs, output))

        for t in transformer.reverse():
            adj.add_reverse(adj.format_template(t, inputs, output))

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

    def add_call(adj, func, inputs, prefix='og::'):
        # expression (zero output), e.g.: tid()
        if (func.value_type(inputs) == None):

            forward_call = prefix + "{}({});".format(func.key, adj.format_args("var_", inputs))
            adj.add_forward(forward_call)

            if (len(inputs)):
                reverse_call = prefix + "{}({}, {});".format("adj_" + func.key, adj.format_args("var_", inputs), adj.format_args("adj_", inputs))
                adj.add_reverse(reverse_call)

            return None

        # function (one output)
        else:

            output = adj.add_var(func.value_type(inputs))

            forward_call = "var_{} = ".format(output) + prefix + "{}({});".format(func.key, adj.format_args("var_", inputs))
            adj.add_forward(forward_call)

            if (len(inputs)):
                reverse_call = prefix + "{}({}, {}, {});".format(
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
        adj.add_reverse("if (var_{}) {{".format(cond))

    # define a for-loop
    def begin_for(adj, iter, start, end):

        # note that dynamic for-loops must not mutate any previous state, so we don't need to re-run them in the reverse pass
        adj.add_forward("for (var_{0}=var_{1}; var_{0} < var_{2}; ++var_{0}) {{".format(iter, start, end), "if (false) {")
        adj.add_reverse("}")

        adj.indent_count += 1

    def end_for(adj, iter, start, end):

        adj.indent_count -= 1

        adj.add_forward("}")
        adj.add_reverse("for (var_{0}=var_{2}-1; var_{0} >= var_{1}; --var_{0}) {{".format(iter, start, end))

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

            if (isinstance(node, ast.FunctionDef)):

                out = None
                for f in node.body:
                    out = adj.eval(f)

                if 'return' in adj.symbols and adj.symbols['return'] is not None:
                    out = adj.symbols['return']
                    stmt = Stmt(None, adj.body_forward, adj.body_forward_replay, reversed(adj.body_reverse), [], "")
                    adj.output.append(stmt)
                else:
                    stmt = Stmt(None, adj.body_forward, adj.body_forward_replay, reversed(adj.body_reverse), [], "")
                    adj.output.append(stmt)

                return out

            elif (isinstance(node, ast.If)):         # if statement

                if len(node.orelse) != 0:
                    raise SyntaxError("Else statements not currently supported")

                if len(node.body) == 0:
                    return None

                # save symbol map
                symbols_prev = adj.symbols.copy()

                # eval condition
                cond = adj.eval(node.test)

                # eval body
                adj.begin_if(cond)

                for stmt in node.body:
                    adj.eval(stmt)

                adj.end_if(cond)

                # detect symbols with conflicting definitions (assigned inside the branch)
                for items in symbols_prev.items():

                    sym = items[0]
                    var1 = items[1]
                    var2 = adj.symbols[sym]

                    if var1 != var2:
                        # insert a phi function that
                        # selects var1, var2 based on cond
                        out = adj.add_call(adj.builtin_functions["select"], [cond, var1, var2])
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
                # op, expr list values (e.g. and and a list of things anded together)

                op = node.op
                if isinstance(op, ast.And):
                    func = "&&"
                elif isinstance(op, ast.Or):
                    func = "||"
                else:
                    raise KeyError("Op {} is not supported".format(op))

                out = adj.add_bool_op(func, [adj.eval(expr) for expr in node.values])

                # import pdb
                # pdb.set_trace()

                return out

            elif (isinstance(node, ast.Name)):
                # lookup symbol, if it has already been assigned to a variable then return the existing mapping
                if (node.id in adj.symbols):
                    return adj.symbols[node.id]
                else:
                    raise KeyError("Referencing undefined symbol: " + str(node.id))

            elif (isinstance(node, ast.Num)):

                # lookup constant, if it has already been assigned then return existing var
                # currently disabled, since assigning constant in a branch means it 
                key = (node.n, type(node.n))

                if (key in adj.symbols):
                    return adj.symbols[key]
                else:
                    out = adj.add_constant(node.n)
                    adj.symbols[key] = out
                    return out

                #out = adj.add_constant(node.n)
                #return out

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

                out = adj.add_operator(node.op, [arg])
                return out

            elif (isinstance(node, ast.For)):

                if (len(node.iter.args) != 2):
                    raise Exception("For loop ranges must be of form range(start, end) with both start and end specified and no skip specifier.")

                # check if loop range is compile time constant
                unroll = True
                for a in node.iter.args:
                    if (isinstance(a, ast.Num) == False):
                        unroll = False
                        break

                if (unroll):

                    # constant loop, unroll
                    start = node.iter.args[0].n
                    end = node.iter.args[1].n

                    for i in range(start, end):

                        var_iter = adj.add_constant(i)
                        adj.symbols[node.target.id] = var_iter

                        # eval body
                        for s in node.body:
                            adj.eval(s)
                else:

                    # dynamic loop, body must be side-effect free, i.e.: not
                    # overwrite memory locations used by previous operations
                    start = adj.eval(node.iter.args[0])
                    end = adj.eval(node.iter.args[1])

                    # add iterator variable
                    iter = adj.add_var(int)
                    adj.symbols[node.target.id] = iter

                    adj.begin_for(iter, start, end)

                    # eval body
                    for s in node.body:
                        adj.eval(s)

                    adj.end_for(iter, start, end)

            elif (isinstance(node, ast.Expr)):
                return adj.eval(node.value)

            elif (isinstance(node, ast.Call)):

                name = None

                # determine if call is to a builtin (e.g.: og.cos(x)), or to a free-func, e.g.: my_func(x)
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
                out = adj.add_call(func, args, prefix=func.namespace)
                return out

            elif (isinstance(node, ast.Subscript)):
                target = adj.eval(node.value)

                indices = []

                if isinstance(node.slice.value, ast.Tuple):
                    # handles the M[i, j] case
                    for arg in node.slice.value.elts:
                        var = adj.eval(arg)
                        indices.append(var)
                else:
                    # simple expression
                    var = adj.eval(node.slice.value)
                    indices.append(var)

                out = adj.add_call(adj.builtin_functions["index"], [target, *indices])
                return out

            elif (isinstance(node, ast.Assign)):
                # if adj.cond is not None:
                #     raise SyntaxError("error, cannot assign variables in a conditional branch")

                # evaluate rhs
                out = adj.eval(node.value)

                # update symbol map (assumes lhs is a Name node)
                adj.symbols[node.targets[0].id] = out
                return out

            elif (isinstance(node, ast.Return)):
                cond = adj.cond  # None if not in branch, else branch boolean

                out = adj.eval(node.value)
                adj.symbols['return'] = out

                if out is not None:        # set return type of function
                    return_var = out
                    if adj.return_var is not None and adj.return_var.ctype() != return_var.ctype():
                        raise TypeError("error, function returned different types")
                    adj.return_var = return_var

                adj.add_return(out)

                return out
            elif node is None:
                return None
            else:
                print("[WARNING] ast node of type {} not supported".format(type(node)))

        except Exception as e:

            # print error / line number
            lines = adj.source.splitlines()
            print("Error: {} while transforming node {} in func: {} at line: {} col: {}: \n    {}".format(e, type(node), adj.func.__name__, node.lineno, node.col_offset, lines[max(node.lineno-1, 0)]))
            raise


#----------------
# code generation

cpu_module_header = '''
#include "../native/core.h"

// avoid namespacing of float type for casting to float type, this is to avoid og::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

using namespace og;

'''

cuda_module_header = '''
#include "../native/core.h"

// avoid namespacing of float type for casting to float type, this is to avoid og::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

using namespace og;

'''

cpu_function_template = '''
static {return_type} {name}({forward_args})
{{
    {forward_body}
}}

static void adj_{name}({forward_args}, {reverse_args})
{{
    {reverse_body}
}}

'''

cuda_function_template = '''
static CUDA_CALLABLE {return_type} {name}({forward_args})
{{
    {forward_body}
}}

static CUDA_CALLABLE void adj_{name}({forward_args}, {reverse_args})
{{
    {reverse_body}
}}

'''

cuda_kernel_template = '''

__global__ void {name}_cuda_kernel_forward(int dim, {forward_args})
{{
    {forward_body}
}}

__global__ void {name}_cuda_kernel_backward(int dim, {forward_args}, {reverse_args})
{{
    {reverse_body}
}}

'''

cpu_kernel_template = '''

void {name}_cpu_kernel_forward({forward_args})
{{
    {forward_body}
}}

void {name}_cpu_kernel_backward({forward_args}, {reverse_args})
{{
    {reverse_body}
}}

'''

cuda_module_template = '''

extern "C" {{

// Python entry points
OG_API void {name}_cuda_forward(int dim, {forward_args})
{{
    {name}_cuda_kernel_forward<<<(dim + 256 - 1) / 256, 256>>>(dim, {forward_params});
}}

OG_API void {name}_cuda_backward(int dim, {forward_args}, {reverse_args})
{{
    {name}_cuda_kernel_backward<<<(dim + 256 - 1) / 256, 256>>>(dim, {forward_params}, {reverse_params});
}}

}} // extern C

'''

cpu_module_template = '''

// Python CPU entry points
OG_API void {name}_cpu_forward(int dim, {forward_args})
{{
    for (int i=0; i < dim; ++i)
    {{
        s_threadIdx = i;

        {name}_cpu_kernel_forward({forward_params});
    }}
}}

OG_API void {name}_cpu_backward(int dim, {forward_args}, {reverse_args})
{{
    for (int i=0; i < dim; ++i)
    {{
        s_threadIdx = i;

        {name}_cpu_kernel_backward({forward_params}, {reverse_params});
    }}
}}

'''

cuda_module_header_template = '''

extern "C" {{

// Python CUDA entry points
OG_API void {name}_cuda_forward(int dim, {forward_args});

OG_API void {name}_cuda_backward(int dim, {forward_args}, {reverse_args});

}} // extern C
'''

cpu_module_header_template = '''

extern "C" {{

// Python CPU entry points
OG_API void {name}_cpu_forward(int dim, {forward_args});

OG_API void {name}_cpu_backward(int dim, {forward_args}, {reverse_args});

}} // extern C
'''


def indent(args, stops=1):
    sep = "\n"
    for i in range(stops):
        sep += "\t"

    return sep + args.replace(", ", "," + sep)


def codegen_func_forward_body(adj, device='cpu', indent=4):
    body = []
    indent_block = " " * indent

    for stmt in adj.output:
        for f in stmt.forward:
            body += [f + "\n"]

        if stmt.cond is not None:
            body += ["if (" + str(stmt.cond) + ") {\n"]
            for l in stmt.ret_forward:
                body += [indent_block + l + "\n"]

            body += [indent_block + stmt.ret_line + "\n"]
            body += ["}\n"]
        else:
            for l in stmt.ret_forward:
                body += [l + "\n"]

            body += [stmt.ret_line + "\n"]

            break  # break once unconditional return is encountered

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
            s += "    const " + var.ctype() + " var_" + str(var.label) + " = " + str(var.constant) + ";\n"


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

    for stmt in adj.output:
        # forward pass
        body += ["//---------\n"]
        body += ["// forward\n"]

        for f in stmt.forward_replay:
            body += [f + "\n"]

        if stmt.cond is not None:
            body += ["if (" + str(stmt.cond) + ") {\n"]
            for l in stmt.ret_forward:
                body += [indent_block + l + "\n"]

            # reverse pass
            body += [indent_block + "//---------\n"]
            body += [indent_block + "// reverse\n"]

            for l in stmt.reverse:
                body += [indent_block + l + "\n"]

            body += [indent_block + "return;\n"]
            body += ["}\n"]
        else:
            for l in stmt.ret_forward:
                body += [l + "\n"]

            # reverse pass
            body += ["//---------\n"]
            body += ["// reverse\n"]

            for l in stmt.reverse:
                body += [l + "\n"]

            body += ["return;\n"]
            break  # break once unconditional return is encountered

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
            s += "    const " + var.ctype() + " var_" + str(var.label) + " = " + str(var.constant) + ";\n"

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

    # s = "{} {}_forward(".format(return_type, adj.func.__name__)

    # sep = ""
    # for arg in adj.args:
    #     if (arg.label != 'return'):
    #         s += sep + str(arg.type.__name__) + " var_" + arg.label
    #         sep = ", "

    # reverse header
    # s = "void {}_reverse(".format(adj.func.__name__)

    # return s

    forward_args = ""
    reverse_args = ""
    # s = ""

    # forward args
    sep = ""
    for arg in adj.args:
        forward_args += sep + arg.ctype() + " var_" + arg.label
        sep = ", "

    # reverse args
    sep = ""
    for arg in adj.args:
        if "*" in arg.ctype():
            reverse_args += sep + arg.ctype() + " adj_" + arg.label
        else:
            reverse_args += sep + arg.ctype() + " & adj_" + arg.label
        sep = ", "

    reverse_args += sep + return_type + " & adj_ret"

    # reverse args

    # add primal version of parameters
    # sep = ""
    # for var in adj.args:
    #     if (var.label != 'return'):
    #         s += sep + var.ctype() + " var_" + var.label
    #         sep = ", "

    # # add adjoint version of parameters
    # for var in adj.args:
    #     if (var.label != 'return'):
    #         s += sep + var.ctype() + "& adj_" + var.label
    #         sep = ", "

    # # add adjoint of output
    # if ('return' in adj.symbols and adj.symbols['return'] != None):
    #     s += sep + str(adj.symbols['return'].type.__name__) + " adj_" + str(adj.symbols['return'])

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


def codegen_kernel(adj, device='cpu'):

    forward_args = ""
    reverse_args = ""

    # forward args
    sep = ""
    for arg in adj.args:
        forward_args += sep + arg.ctype() + " var_" + arg.label
        sep = ", "

    # reverse args
    sep = ""
    for arg in adj.args:
        reverse_args += sep + arg.ctype() + " adj_" + arg.label
        sep = ", "

    # codegen body
    forward_body = codegen_func_forward(adj, func_type='kernel', device=device)
    reverse_body = codegen_func_reverse(adj, func_type='kernel', device=device)

    # import pdb
    # pdb.set_trace()

    if device == 'cpu':
        template = cpu_kernel_template
    elif device == 'cuda':
        template = cuda_kernel_template
    else:
        raise ValueError("Device {} is not supported".format(device))

    s = template.format(name=adj.func.__name__,
                        forward_args=indent(forward_args),
                        reverse_args=indent(reverse_args),
                        forward_body=forward_body,
                        reverse_body=reverse_body)

    return s


def codegen_module(adj, device='cpu'):

    forward_args = ""
    reverse_args = ""

    forward_params = ""
    reverse_params = ""

    sep = ""
    for arg in adj.args:
        if (isinstance(arg.type, array)):
            forward_args += sep + "og::array var_" + arg.label
            forward_params += sep + "cast<" + arg.ctype() + ">(var_" + arg.label + ")"
        else:
            forward_args += sep + arg.ctype() + " var_" + arg.label
            forward_params += sep + "var_" + arg.label

        sep = ", "

    sep = ""
    for arg in adj.args:
        if (isinstance(arg.type, array)):
            reverse_args += sep + "og::array adj_" + arg.label
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

    s = template.format(name=adj.func.__name__,
                        forward_args=indent(forward_args),
                        reverse_args=indent(reverse_args),
                        forward_params=indent(forward_params, 3),
                        reverse_params=indent(reverse_params, 3))
    return s


def codegen_module_decl(adj, device='cpu'):

    forward_args = ""
    reverse_args = ""

    forward_params = ""
    reverse_params = ""

    sep = ""
    for arg in adj.args:
        if (isinstance(arg.type, array)):
            forward_args += sep + "og::array var_" + arg.label
            forward_params += sep + "cast<" + arg.ctype() + ">(var_" + arg.label + ")"
        else:
            forward_args += sep + arg.ctype() + " var_" + arg.label
            forward_params += sep + "var_" + arg.label

        sep = ", "

    sep = ""
    for arg in adj.args:
        if (isinstance(arg.type, array)):
            reverse_args += sep + "og::array adj_" + arg.label
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

    s = template.format(name=adj.func.__name__, forward_args=indent(forward_args), reverse_args=indent(reverse_args))
    return s












# def matmul(tape, m, n, k, t1, t2, A, B, C, adapter):
    
#     if (adapter == 'cpu'):
#         threads = 1
#     else:
#         threads = 256   # should match the threadblock size

#     tape.launch(
#         func=og.eval_dense_gemm,
#         dim=threads,
#         inputs=[
#             m,
#             n,
#             k,
#             t1,
#             t2,
#             A,
#             B,
#         ],
#         outputs=[
#             C
#         ],
#         adapter=adapter,
#         preserve_output=False)


# def matmul_batched(tape, batch_count, m, n, k, t1, t2, A_start, B_start, C_start, A, B, C, adapter):
    
#     if (adapter == 'cpu'):
#         threads = batch_count
#     else:
#         threads = 256*batch_count   # must match the threadblock size used in adjoint.py

#     tape.launch(
#         func=og.eval_dense_gemm_batched,
#         dim=threads,
#         inputs=[
#             m,
#             n,
#             k,
#             t1,
#             t2,
#             A_start,
#             B_start,
#             C_start,
#             A,
#             B,
#         ],
#         outputs=[
#             C
#         ],
#         adapter=adapter,
#         preserve_output=False)