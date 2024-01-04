# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ast
import ctypes
import gc
import hashlib
import inspect
import io
import os
import platform
import sys
import types
from copy import copy as shallowcopy
from types import ModuleType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import warp
import warp.build
import warp.codegen
import warp.config

# represents either a built-in or user-defined function


def create_value_func(type):
    def value_func(args, kwds, templates):
        return type

    return value_func


def get_function_args(func):
    """Ensures that all function arguments are annotated and returns a dictionary mapping from argument name to its type."""
    import inspect

    argspec = inspect.getfullargspec(func)

    # use source-level argument annotations
    if len(argspec.annotations) < len(argspec.args):
        raise RuntimeError(f"Incomplete argument annotations on function {func.__qualname__}")
    return argspec.annotations


class Function:
    def __init__(
        self,
        func,
        key,
        namespace,
        input_types=None,
        value_func=None,
        template_func=None,
        module=None,
        variadic=False,
        initializer_list_func=None,
        export=False,
        doc="",
        group="",
        hidden=False,
        skip_replay=False,
        missing_grad=False,
        generic=False,
        native_func=None,
        defaults=None,
        custom_replay_func=None,
        native_snippet=None,
        adj_native_snippet=None,
        skip_forward_codegen=False,
        skip_reverse_codegen=False,
        custom_reverse_num_input_args=-1,
        custom_reverse_mode=False,
        overloaded_annotations=None,
        code_transformers=[],
        skip_adding_overload=False,
        require_original_output_arg=False,
    ):
        self.func = func  # points to Python function decorated with @wp.func, may be None for builtins
        self.key = key
        self.namespace = namespace
        self.value_func = value_func  # a function that takes a list of args and a list of templates and returns the value type, e.g.: load(array, index) returns the type of value being loaded
        self.template_func = template_func
        self.input_types = {}
        self.export = export
        self.doc = doc
        self.group = group
        self.module = module
        self.variadic = variadic  # function can take arbitrary number of inputs, e.g.: printf()
        self.defaults = defaults
        # Function instance for a custom implementation of the replay pass
        self.custom_replay_func = custom_replay_func
        self.native_snippet = native_snippet
        self.adj_native_snippet = adj_native_snippet
        self.custom_grad_func = None
        self.require_original_output_arg = require_original_output_arg

        if initializer_list_func is None:
            self.initializer_list_func = lambda x, y: False
        else:
            self.initializer_list_func = (
                initializer_list_func  # True if the arguments should be emitted as an initializer list in the c++ code
            )
        self.hidden = hidden  # function will not be listed in docs
        self.skip_replay = (
            skip_replay  # whether or not operation will be performed during the forward replay in the backward pass
        )
        self.missing_grad = missing_grad  # whether or not builtin is missing a corresponding adjoint
        self.generic = generic

        # allow registering builtin functions with a different name in Python from the native code
        if native_func is None:
            self.native_func = key
        else:
            self.native_func = native_func

        if func:
            # user-defined function

            # generic and concrete overload lookups by type signature
            self.user_templates = {}
            self.user_overloads = {}

            # user defined (Python) function
            self.adj = warp.codegen.Adjoint(
                func,
                is_user_function=True,
                skip_forward_codegen=skip_forward_codegen,
                skip_reverse_codegen=skip_reverse_codegen,
                custom_reverse_num_input_args=custom_reverse_num_input_args,
                custom_reverse_mode=custom_reverse_mode,
                overload_annotations=overloaded_annotations,
                transformers=code_transformers,
            )

            # record input types
            for name, type in self.adj.arg_types.items():
                if name == "return":
                    self.value_func = create_value_func(type)

                else:
                    self.input_types[name] = type

        else:
            # builtin function

            # embedded linked list of all overloads
            # the builtin_functions dictionary holds
            # the list head for a given key (func name)
            self.overloads = []

            # builtin (native) function, canonicalize argument types
            for k, v in input_types.items():
                self.input_types[k] = warp.types.type_to_warp(v)

            # cache mangled name
            if self.is_simple():
                self.mangled_name = self.mangle()
            else:
                self.mangled_name = None

        if not skip_adding_overload:
            self.add_overload(self)

        # add to current module
        if module:
            module.register_function(self, skip_adding_overload)

    def __call__(self, *args, **kwargs):
        # handles calling a builtin (native) function
        # as if it was a Python function, i.e.: from
        # within the CPython interpreter rather than
        # from within a kernel (experimental).

        if self.is_builtin() and self.mangled_name:
            # For each of this function's existing overloads, we attempt to pack
            # the given arguments into the C types expected by the corresponding
            # parameters, and we rinse and repeat until we get a match.
            for overload in self.overloads:
                if overload.generic:
                    continue

                success, return_value = call_builtin(overload, *args)
                if success:
                    return return_value

            # overload resolution or call failed
            raise RuntimeError(
                f"Couldn't find a function '{self.key}' compatible with "
                f"the arguments '{', '.join(type(x).__name__ for x in args)}'"
            )

        if hasattr(self, "user_overloads") and len(self.user_overloads):
            # user-defined function with overloads

            if len(kwargs):
                raise RuntimeError(
                    f"Error calling function '{self.key}', keyword arguments are not supported for user-defined overloads."
                )

            # try and find a matching overload
            for overload in self.user_overloads.values():
                if len(overload.input_types) != len(args):
                    continue
                template_types = list(overload.input_types.values())
                arg_names = list(overload.input_types.keys())
                try:
                    # attempt to unify argument types with function template types
                    warp.types.infer_argument_types(args, template_types, arg_names)
                    return overload.func(*args)
                except Exception:
                    continue

            raise RuntimeError(f"Error calling function '{self.key}', no overload found for arguments {args}")

        # user-defined function with no overloads
        if self.func is None:
            raise RuntimeError(f"Error calling function '{self.key}', function is undefined")

        # this function has no overloads, call it like a plain Python function
        return self.func(*args, **kwargs)

    def is_builtin(self):
        return self.func is None

    def is_simple(self):
        if self.variadic:
            return False

        # only export simple types that don't use arrays
        for k, v in self.input_types.items():
            if isinstance(v, warp.array) or v == Any or v == Callable or v == Tuple:
                return False

        return_type = ""

        try:
            # todo: construct a default value for each of the functions args
            # so we can generate the return type for overloaded functions
            return_type = type_str(self.value_func(None, None, None))
        except Exception:
            return False

        if return_type.startswith("Tuple"):
            return False

        return True

    def mangle(self):
        # builds a mangled name for the C-exported
        # function, e.g.: builtin_normalize_vec3()

        name = "builtin_" + self.key

        types = []
        for t in self.input_types.values():
            types.append(t.__name__)

        return "_".join([name, *types])

    def add_overload(self, f):
        if self.is_builtin():
            # todo: note that it is an error to add two functions
            # with the exact same signature as this would cause compile
            # errors during compile time. We should check here if there
            # is a previously created function with the same signature
            self.overloads.append(f)

            # make sure variadic overloads appear last so non variadic
            # ones are matched first:
            self.overloads.sort(key=lambda f: f.variadic)

        else:
            # get function signature based on the input types
            sig = warp.types.get_signature(
                f.input_types.values(), func_name=f.key, arg_names=list(f.input_types.keys())
            )

            # check if generic
            if warp.types.is_generic_signature(sig):
                if sig in self.user_templates:
                    raise RuntimeError(
                        f"Duplicate generic function overload {self.key} with arguments {f.input_types.values()}"
                    )
                self.user_templates[sig] = f
            else:
                if sig in self.user_overloads:
                    raise RuntimeError(
                        f"Duplicate function overload {self.key} with arguments {f.input_types.values()}"
                    )
                self.user_overloads[sig] = f

    def get_overload(self, arg_types):
        assert not self.is_builtin()

        sig = warp.types.get_signature(arg_types, func_name=self.key)

        f = self.user_overloads.get(sig)
        if f is not None:
            return f
        else:
            for f in self.user_templates.values():
                if len(f.input_types) != len(arg_types):
                    continue

                # try to match the given types to the function template types
                template_types = list(f.input_types.values())
                args_matched = True

                for i in range(len(arg_types)):
                    if not warp.types.type_matches_template(arg_types[i], template_types[i]):
                        args_matched = False
                        break

                if args_matched:
                    # instantiate this function with the specified argument types

                    arg_names = f.input_types.keys()
                    overload_annotations = dict(zip(arg_names, arg_types))

                    ovl = shallowcopy(f)
                    ovl.adj = warp.codegen.Adjoint(f.func, overload_annotations)
                    ovl.input_types = overload_annotations
                    ovl.value_func = None

                    self.user_overloads[sig] = ovl

                    return ovl

            # failed  to find overload
            return None

    def __repr__(self):
        inputs_str = ", ".join([f"{k}: {warp.types.type_repr(v)}" for k, v in self.input_types.items()])
        return f"<Function {self.key}({inputs_str})>"


def call_builtin(func: Function, *params) -> Tuple[bool, Any]:
    uses_non_warp_array_type = False

    # Retrieve the built-in function from Warp's dll.
    c_func = getattr(warp.context.runtime.core, func.mangled_name)

    # Try gathering the parameters that the function expects and pack them
    # into their corresponding C types.
    c_params = []
    for i, (_, arg_type) in enumerate(func.input_types.items()):
        param = params[i]

        try:
            iter(param)
        except TypeError:
            is_array = False
        else:
            is_array = True

        if is_array:
            if not issubclass(arg_type, ctypes.Array):
                return (False, None)

            # The argument expects a built-in Warp type like a vector or a matrix.

            c_param = None

            if isinstance(param, ctypes.Array):
                # The given parameter is also a built-in Warp type, so we only need
                # to make sure that it matches with the argument.
                if not warp.types.types_equal(type(param), arg_type):
                    return (False, None)

                if isinstance(param, arg_type):
                    c_param = param
                else:
                    # Cast the value to its argument type to make sure that it
                    # can be assigned to the field of the `Param` struct.
                    # This could error otherwise when, for example, the field type
                    # is set to `vec3i` while the value is of type `vector(length=3, dtype=int)`,
                    # even though both types are semantically identical.
                    c_param = arg_type(param)
            else:
                # Flatten the parameter values into a flat 1-D array.
                arr = []
                ndim = 1
                stack = [(0, param)]
                while stack:
                    depth, elem = stack.pop(0)
                    try:
                        # If `elem` is a sequence, then it should be possible
                        # to add its elements to the stack for later processing.
                        stack.extend((depth + 1, x) for x in elem)
                    except TypeError:
                        # Since `elem` doesn't seem to be a sequence,
                        # we must have a leaf value that we need to add to our
                        # resulting array.
                        arr.append(elem)
                        ndim = max(depth, ndim)

                assert ndim > 0

                # Ensure that if the given parameter value is, say, a 2-D array,
                # then we try to resolve it against a matrix argument rather than
                # a vector.
                if ndim > len(arg_type._shape_):
                    return (False, None)

                elem_count = len(arr)
                if elem_count != arg_type._length_:
                    return (False, None)

                # Retrieve the element type of the sequence while ensuring
                # that it's homogeneous.
                elem_type = type(arr[0])
                for i in range(1, elem_count):
                    if type(arr[i]) is not elem_type:
                        raise ValueError("All array elements must share the same type.")

                expected_elem_type = arg_type._wp_scalar_type_
                if not (
                    elem_type is expected_elem_type
                    or (elem_type is float and expected_elem_type is warp.types.float32)
                    or (elem_type is int and expected_elem_type is warp.types.int32)
                    or (
                        issubclass(elem_type, np.number)
                        and warp.types.np_dtype_to_warp_type[np.dtype(elem_type)] is expected_elem_type
                    )
                ):
                    # The parameter value has a type not matching the type defined
                    # for the corresponding argument.
                    return (False, None)

                if elem_type in warp.types.int_types:
                    # Pass the value through the expected integer type
                    # in order to evaluate any integer wrapping.
                    # For example `uint8(-1)` should result in the value `-255`.
                    arr = tuple(elem_type._type_(x.value).value for x in arr)
                elif elem_type in warp.types.float_types:
                    # Extract the floating-point values.
                    arr = tuple(x.value for x in arr)

                c_param = arg_type()
                if warp.types.type_is_matrix(arg_type):
                    rows, cols = arg_type._shape_
                    for i in range(rows):
                        idx_start = i * cols
                        idx_end = idx_start + cols
                        c_param[i] = arr[idx_start:idx_end]
                else:
                    c_param[:] = arr

                uses_non_warp_array_type = True

            c_params.append(ctypes.byref(c_param))
        else:
            if issubclass(arg_type, ctypes.Array):
                return (False, None)

            if not (
                isinstance(param, arg_type)
                or (type(param) is float and arg_type is warp.types.float32)
                or (type(param) is int and arg_type is warp.types.int32)
                or warp.types.np_dtype_to_warp_type.get(getattr(param, "dtype", None)) is arg_type
            ):
                return (False, None)

            if type(param) in warp.types.scalar_types:
                param = param.value

            # try to pack as a scalar type
            if arg_type == warp.types.float16:
                c_params.append(arg_type._type_(warp.types.float_to_half_bits(param)))
            else:
                c_params.append(arg_type._type_(param))

    # returns the corresponding ctype for a scalar or vector warp type
    value_type = func.value_func(None, None, None)
    if value_type == float:
        value_ctype = ctypes.c_float
    elif value_type == int:
        value_ctype = ctypes.c_int32
    elif issubclass(value_type, (ctypes.Array, ctypes.Structure)):
        value_ctype = value_type
    else:
        # scalar type
        value_ctype = value_type._type_

    # construct return value (passed by address)
    ret = value_ctype()
    ret_addr = ctypes.c_void_p(ctypes.addressof(ret))
    c_params.append(ret_addr)

    # Call the built-in function from Warp's dll.
    c_func(*c_params)

    # TODO: uncomment when we have a way to print warning messages only once.
    # if uses_non_warp_array_type:
    #     warp.utils.warn(
    #         "Support for built-in functions called with non-Warp array types, "
    #         "such as lists, tuples, NumPy arrays, and others, will be dropped "
    #         "in the future. Use a Warp type such as `wp.vec`, `wp.mat`, "
    #         "`wp.quat`, or `wp.transform`.",
    #         DeprecationWarning,
    #         stacklevel=3
    #     )

    if issubclass(value_ctype, ctypes.Array) or issubclass(value_ctype, ctypes.Structure):
        # return vector types as ctypes
        return (True, ret)

    if value_type == warp.types.float16:
        return (True, warp.types.half_bits_to_float(ret.value))

    # return scalar types as int/float
    return (True, ret.value)


class KernelHooks:
    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward


# caches source and compiled entry points for a kernel (will be populated after module loads)
class Kernel:
    def __init__(self, func, key=None, module=None, options=None, code_transformers=[]):
        self.func = func

        if module is None:
            self.module = get_module(func.__module__)
        else:
            self.module = module

        if key is None:
            unique_key = self.module.generate_unique_kernel_key(func.__name__)
            self.key = unique_key
        else:
            self.key = key

        self.options = {} if options is None else options

        self.adj = warp.codegen.Adjoint(func, transformers=code_transformers)

        # check if generic
        self.is_generic = False
        for arg_type in self.adj.arg_types.values():
            if warp.types.type_is_generic(arg_type):
                self.is_generic = True
                break

        # unique signature (used to differentiate instances of generic kernels during codegen)
        self.sig = ""

        # known overloads for generic kernels, indexed by type signature
        self.overloads = {}

        # argument indices by name
        self.arg_indices = dict((a.label, i) for i, a in enumerate(self.adj.args))

        if self.module:
            self.module.register_kernel(self)

    def infer_argument_types(self, args):
        template_types = list(self.adj.arg_types.values())

        if len(args) != len(template_types):
            raise RuntimeError(f"Invalid number of arguments for kernel {self.key}")

        arg_names = list(self.adj.arg_types.keys())

        return warp.types.infer_argument_types(args, template_types, arg_names)

    def add_overload(self, arg_types):
        if len(arg_types) != len(self.adj.arg_types):
            raise RuntimeError(f"Invalid number of arguments for kernel {self.key}")

        arg_names = list(self.adj.arg_types.keys())
        template_types = list(self.adj.arg_types.values())

        # make sure all argument types are concrete and match the kernel parameters
        for i in range(len(arg_types)):
            if not warp.types.type_matches_template(arg_types[i], template_types[i]):
                if warp.types.type_is_generic(arg_types[i]):
                    raise TypeError(
                        f"Kernel {self.key} argument '{arg_names[i]}' cannot be generic, got {arg_types[i]}"
                    )
                else:
                    raise TypeError(
                        f"Kernel {self.key} argument '{arg_names[i]}' type mismatch: expected {template_types[i]}, got {arg_types[i]}"
                    )

        # get a type signature from the given argument types
        sig = warp.types.get_signature(arg_types, func_name=self.key)
        if sig in self.overloads:
            raise RuntimeError(
                f"Duplicate overload for kernel {self.key}, an overload with the given arguments already exists"
            )

        overload_annotations = dict(zip(arg_names, arg_types))

        # instantiate this kernel with the given argument types
        ovl = shallowcopy(self)
        ovl.adj = warp.codegen.Adjoint(self.func, overload_annotations)
        ovl.is_generic = False
        ovl.overloads = {}
        ovl.sig = sig

        self.overloads[sig] = ovl

        self.module.unload()

        return ovl

    def get_overload(self, arg_types):
        sig = warp.types.get_signature(arg_types, func_name=self.key)

        ovl = self.overloads.get(sig)
        if ovl is not None:
            return ovl
        else:
            return self.add_overload(arg_types)

    def get_mangled_name(self):
        if self.sig:
            return f"{self.key}_{self.sig}"
        else:
            return self.key


# ----------------------


# decorator to register function, @func
def func(f):
    name = warp.codegen.make_full_qualified_name(f)

    m = get_module(f.__module__)
    Function(
        func=f, key=name, namespace="", module=m, value_func=None
    )  # value_type not known yet, will be inferred during Adjoint.build()

    # return the top of the list of overloads for this key
    return m.functions[name]


def func_native(snippet, adj_snippet=None):
    """
    Decorator to register native code snippet, @func_native
    """

    def snippet_func(f):
        name = warp.codegen.make_full_qualified_name(f)

        m = get_module(f.__module__)
        func = Function(
            func=f, key=name, namespace="", module=m, native_snippet=snippet, adj_native_snippet=adj_snippet
        )  # cuda snippets do not have a return value_type

        return m.functions[name]

    return snippet_func


def func_grad(forward_fn):
    """
    Decorator to register a custom gradient function for a given forward function.
    The function signature must correspond to one of the function overloads in the following way:
    the first part of the input arguments are the original input variables with the same types as their
    corresponding arguments in the original function, and the second part of the input arguments are the
    adjoint variables of the output variables (if available) of the original function with the same types as the
    output variables. The function must not return anything.
    """

    def wrapper(grad_fn):
        generic = any(warp.types.type_is_generic(x) for x in forward_fn.input_types.values())
        if generic:
            raise RuntimeError(
                f"Cannot define custom grad definition for {forward_fn.key} since functions with generic input arguments are not yet supported."
            )

        reverse_args = {}
        reverse_args.update(forward_fn.input_types)

        # create temporary Adjoint instance to analyze the function signature
        adj = warp.codegen.Adjoint(
            grad_fn, skip_forward_codegen=True, skip_reverse_codegen=False, transformers=forward_fn.adj.transformers
        )

        from warp.types import types_equal

        grad_args = adj.args
        grad_sig = warp.types.get_signature([arg.type for arg in grad_args], func_name=forward_fn.key)

        generic = any(warp.types.type_is_generic(x.type) for x in grad_args)
        if generic:
            raise RuntimeError(
                f"Cannot define custom grad definition for {forward_fn.key} since the provided grad function has generic input arguments."
            )

        def match_function(f):
            # check whether the function overload f matches the signature of the provided gradient function
            if not hasattr(f.adj, "return_var"):
                f.adj.build(None)
            expected_args = list(f.input_types.items())
            if f.adj.return_var is not None:
                expected_args += [(f"adj_ret_{var.label}", var.type) for var in f.adj.return_var]
            if len(grad_args) != len(expected_args):
                return False
            if any(not types_equal(a.type, exp_type) for a, (_, exp_type) in zip(grad_args, expected_args)):
                return False
            return True

        def add_custom_grad(f: Function):
            # register custom gradient function
            f.custom_grad_func = Function(
                grad_fn,
                key=f.key,
                namespace=f.namespace,
                input_types=reverse_args,
                value_func=None,
                module=f.module,
                template_func=f.template_func,
                skip_forward_codegen=True,
                custom_reverse_mode=True,
                custom_reverse_num_input_args=len(f.input_types),
                skip_adding_overload=False,
                code_transformers=f.adj.transformers,
            )
            f.adj.skip_reverse_codegen = True

        if hasattr(forward_fn, "user_overloads") and len(forward_fn.user_overloads):
            # find matching overload for which this grad function is defined
            for sig, f in forward_fn.user_overloads.items():
                if not grad_sig.startswith(sig):
                    continue
                if match_function(f):
                    add_custom_grad(f)
                    return
            raise RuntimeError(
                f"No function overload found for gradient function {grad_fn.__qualname__} for function {forward_fn.key}"
            )
        else:
            # resolve return variables
            forward_fn.adj.build(None)

            expected_args = list(forward_fn.input_types.items())
            if forward_fn.adj.return_var is not None:
                expected_args += [(f"adj_ret_{var.label}", var.type) for var in forward_fn.adj.return_var]

            # check if the signature matches this function
            if match_function(forward_fn):
                add_custom_grad(forward_fn)
            else:
                raise RuntimeError(
                    f"Gradient function {grad_fn.__qualname__} for function {forward_fn.key} has an incorrect signature. The arguments must match the "
                    "forward function arguments plus the adjoint variables corresponding to the return variables:"
                    f"\n{', '.join(map(lambda nt: f'{nt[0]}: {nt[1].__name__}', expected_args))}"
                )

    return wrapper


def func_replay(forward_fn):
    """
    Decorator to register a custom replay function for a given forward function.
    The replay function is the function version that is called in the forward phase of the backward pass (replay mode) and corresponds to the forward function by default.
    The provided function has to match the signature of one of the original forward function overloads.
    """

    def wrapper(replay_fn):
        generic = any(warp.types.type_is_generic(x) for x in forward_fn.input_types.values())
        if generic:
            raise RuntimeError(
                f"Cannot define custom replay definition for {forward_fn.key} since functions with generic input arguments are not yet supported."
            )

        args = get_function_args(replay_fn)
        arg_types = list(args.values())
        generic = any(warp.types.type_is_generic(x) for x in arg_types)
        if generic:
            raise RuntimeError(
                f"Cannot define custom replay definition for {forward_fn.key} since the provided replay function has generic input arguments."
            )

        f = forward_fn.get_overload(arg_types)
        if f is None:
            inputs_str = ", ".join([f"{k}: {v.__name__}" for k, v in args.items()])
            raise RuntimeError(
                f"Could not find forward definition of function {forward_fn.key} that matches custom replay definition with arguments:\n{inputs_str}"
            )
        f.custom_replay_func = Function(
            replay_fn,
            key=f"replay_{f.key}",
            namespace=f.namespace,
            input_types=f.input_types,
            value_func=f.value_func,
            module=f.module,
            template_func=f.template_func,
            skip_reverse_codegen=True,
            skip_adding_overload=True,
            code_transformers=f.adj.transformers,
        )

    return wrapper


# decorator to register kernel, @kernel, custom_name may be a string
# that creates a kernel with a different name from the actual function
def kernel(f=None, *, enable_backward=None):
    def wrapper(f, *args, **kwargs):
        options = {}

        if enable_backward is not None:
            options["enable_backward"] = enable_backward

        m = get_module(f.__module__)
        k = Kernel(
            func=f,
            key=warp.codegen.make_full_qualified_name(f),
            module=m,
            options=options,
        )
        return k

    if f is None:
        # Arguments were passed to the decorator.
        return wrapper

    return wrapper(f)


# decorator to register struct, @struct
def struct(c):
    m = get_module(c.__module__)
    s = warp.codegen.Struct(cls=c, key=warp.codegen.make_full_qualified_name(c), module=m)

    return s


# overload a kernel with the given argument types
def overload(kernel, arg_types=None):
    if isinstance(kernel, Kernel):
        # handle cases where user calls us directly, e.g. wp.overload(kernel, [args...])

        if not kernel.is_generic:
            raise RuntimeError(f"Only generic kernels can be overloaded.  Kernel {kernel.key} is not generic")

        if isinstance(arg_types, list):
            arg_list = arg_types
        elif isinstance(arg_types, dict):
            # substitute named args
            arg_list = [a.type for a in kernel.adj.args]
            for arg_name, arg_type in arg_types.items():
                idx = kernel.arg_indices.get(arg_name)
                if idx is None:
                    raise RuntimeError(f"Invalid argument name '{arg_name}' in overload of kernel {kernel.key}")
                arg_list[idx] = arg_type
        elif arg_types is None:
            arg_list = []
        else:
            raise TypeError("Kernel overload types must be given in a list or dict")

        # return new kernel overload
        return kernel.add_overload(arg_list)

    elif isinstance(kernel, types.FunctionType):
        # handle cases where user calls us as a function decorator (@wp.overload)

        # ensure this function name corresponds to a kernel
        fn = kernel
        module = get_module(fn.__module__)
        kernel = module.kernels.get(fn.__name__)
        if kernel is None:
            raise RuntimeError(f"Failed to find a kernel named '{fn.__name__}' in module {fn.__module__}")

        if not kernel.is_generic:
            raise RuntimeError(f"Only generic kernels can be overloaded.  Kernel {kernel.key} is not generic")

        # ensure the function is defined without a body, only ellipsis (...), pass, or a string expression
        # TODO: show we allow defining a new body for kernel overloads?
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        assert isinstance(tree, ast.Module)
        assert isinstance(tree.body[0], ast.FunctionDef)
        func_body = tree.body[0].body
        for node in func_body:
            if isinstance(node, ast.Pass):
                continue
            elif isinstance(node, ast.Expr) and isinstance(node.value, (ast.Str, ast.Ellipsis)):
                continue
            raise RuntimeError(
                "Illegal statement in kernel overload definition.  Only pass, ellipsis (...), comments, or docstrings are allowed"
            )

        # ensure all arguments are annotated
        argspec = inspect.getfullargspec(fn)
        if len(argspec.annotations) < len(argspec.args):
            raise RuntimeError(f"Incomplete argument annotations on kernel overload {fn.__name__}")

        # get type annotation list
        arg_list = []
        for arg_name, arg_type in argspec.annotations.items():
            if arg_name != "return":
                arg_list.append(arg_type)

        # add new overload, but we must return the original kernel from @wp.overload decorator!
        kernel.add_overload(arg_list)
        return kernel

    else:
        raise RuntimeError("wp.overload() called with invalid argument!")


builtin_functions = {}


def add_builtin(
    key,
    input_types={},
    value_type=None,
    value_func=None,
    template_func=None,
    doc="",
    namespace="wp::",
    variadic=False,
    initializer_list_func=None,
    export=True,
    group="Other",
    hidden=False,
    skip_replay=False,
    missing_grad=False,
    native_func=None,
    defaults=None,
    require_original_output_arg=False,
):
    # wrap simple single-type functions with a value_func()
    if value_func is None:

        def value_func(args, kwds, templates):
            return value_type

    if initializer_list_func is None:

        def initializer_list_func(args, templates):
            return False

    if defaults is None:
        defaults = {}

    # Add specialized versions of this builtin if it's generic by matching arguments against
    # hard coded types. We do this so you can use hard coded warp types outside kernels:
    generic = any(warp.types.type_is_generic(x) for x in input_types.values())
    if generic and export:
        # get a list of existing generic vector types (includes matrices and stuff)
        # so we can match arguments against them:
        generic_vtypes = [x for x in warp.types.vector_types if hasattr(x, "_wp_generic_type_str_")]

        # deduplicate identical types:
        def typekey(t):
            return f"{t._wp_generic_type_str_}_{t._wp_type_params_}"

        typedict = {typekey(t): t for t in generic_vtypes}
        generic_vtypes = [typedict[k] for k in sorted(typedict.keys())]

        # collect the parent type names of all the generic arguments:
        def generic_names(l):
            for t in l:
                if hasattr(t, "_wp_generic_type_str_"):
                    yield t._wp_generic_type_str_
                elif warp.types.type_is_generic_scalar(t):
                    yield t.__name__

        genericset = set(generic_names(input_types.values()))

        # for each of those type names, get a list of all hard coded types derived
        # from them:
        def derived(name):
            if name == "Float":
                return warp.types.float_types
            elif name == "Scalar":
                return warp.types.scalar_types
            elif name == "Int":
                return warp.types.int_types
            return [x for x in generic_vtypes if x._wp_generic_type_str_ == name]

        gtypes = {k: derived(k) for k in genericset}

        # find the scalar data types supported by all the arguments by intersecting
        # sets:
        def scalar_type(t):
            if t in warp.types.scalar_types:
                return t
            return [p for p in t._wp_type_params_ if p in warp.types.scalar_types][0]

        scalartypes = [{scalar_type(x) for x in gtypes[k]} for k in gtypes.keys()]
        if scalartypes:
            scalartypes = scalartypes.pop().intersection(*scalartypes)

        scalartypes = list(scalartypes)
        scalartypes.sort(key=str)

        # generate function calls for each of these scalar types:
        for stype in scalartypes:
            # find concrete types for this scalar type (eg if the scalar type is float32
            # this dict will look something like this:
            # {"vec":[wp.vec2,wp.vec3,wp.vec4], "mat":[wp.mat22,wp.mat33,wp.mat44]})
            consistenttypes = {k: [x for x in v if scalar_type(x) == stype] for k, v in gtypes.items()}

            def typelist(param):
                if warp.types.type_is_generic_scalar(param):
                    return [stype]
                if hasattr(param, "_wp_generic_type_str_"):
                    l = consistenttypes[param._wp_generic_type_str_]
                    return [x for x in l if warp.types.types_equal(param, x, match_generic=True)]
                return [param]

            # gotta try generating function calls for all combinations of these argument types
            # now.
            import itertools

            typelists = [typelist(param) for param in input_types.values()]
            for argtypes in itertools.product(*typelists):
                # Some of these argument lists won't work, eg if the function is mul(), we won't be
                # able to do a matrix vector multiplication for a mat22 and a vec3, so we call value_func
                # on the generated argument list and skip generation if it fails.
                # This also gives us the return type, which we keep for later:
                try:
                    return_type = value_func(argtypes, {}, [])
                except Exception:
                    continue

                # The return_type might just be vector_t(length=3,dtype=wp.float32), so we've got to match that
                # in the list of hard coded types so it knows it's returning one of them:
                if hasattr(return_type, "_wp_generic_type_str_"):
                    return_type_match = [
                        x
                        for x in generic_vtypes
                        if x._wp_generic_type_str_ == return_type._wp_generic_type_str_
                        and x._wp_type_params_ == return_type._wp_type_params_
                    ]
                    if not return_type_match:
                        continue
                    return_type = return_type_match[0]

                # finally we can generate a function call for these concrete types:
                add_builtin(
                    key,
                    input_types=dict(zip(input_types.keys(), argtypes)),
                    value_type=return_type,
                    doc=doc,
                    namespace=namespace,
                    variadic=variadic,
                    initializer_list_func=initializer_list_func,
                    export=export,
                    group=group,
                    hidden=True,
                    skip_replay=skip_replay,
                    missing_grad=missing_grad,
                    require_original_output_arg=require_original_output_arg,
                )

    func = Function(
        func=None,
        key=key,
        namespace=namespace,
        input_types=input_types,
        value_func=value_func,
        template_func=template_func,
        variadic=variadic,
        initializer_list_func=initializer_list_func,
        export=export,
        doc=doc,
        group=group,
        hidden=hidden,
        skip_replay=skip_replay,
        missing_grad=missing_grad,
        generic=generic,
        native_func=native_func,
        defaults=defaults,
        require_original_output_arg=require_original_output_arg,
    )

    if key in builtin_functions:
        builtin_functions[key].add_overload(func)
    else:
        builtin_functions[key] = func

        # export means the function will be added to the `warp` module namespace
        # so that users can call it directly from the Python interpreter
        if export:
            if hasattr(warp, key):
                # check that we haven't already created something at this location
                # if it's just an overload stub for auto-complete then overwrite it
                if getattr(warp, key).__name__ != "_overload_dummy":
                    raise RuntimeError(
                        f"Trying to register builtin function '{key}' that would overwrite existing object."
                    )

            setattr(warp, key, func)


# global dictionary of modules
user_modules = {}


def get_module(name):
    # some modules might be manually imported using `importlib` without being
    # registered into `sys.modules`
    parent = sys.modules.get(name, None)
    parent_loader = None if parent is None else parent.__loader__

    if name in user_modules:
        # check if the Warp module was created using a different loader object
        # if so, we assume the file has changed and we recreate the module to
        # clear out old kernels / functions
        if user_modules[name].loader is not parent_loader:
            old_module = user_modules[name]

            # Unload the old module and recursively unload all of its dependents.
            # This ensures that dependent modules will be re-hashed and reloaded on next launch.
            # The visited set tracks modules already visited to avoid circular references.
            def unload_recursive(module, visited):
                module.unload()
                visited.add(module)
                for d in module.dependents:
                    if d not in visited:
                        unload_recursive(d, visited)

            unload_recursive(old_module, visited=set())

            # clear out old kernels, funcs, struct definitions
            old_module.kernels = {}
            old_module.functions = {}
            old_module.constants = []
            old_module.structs = {}
            old_module.loader = parent_loader

        return user_modules[name]

    else:
        # else Warp module didn't exist yet, so create a new one
        user_modules[name] = warp.context.Module(name, parent_loader)
        return user_modules[name]


class ModuleBuilder:
    def __init__(self, module, options):
        self.functions = {}
        self.structs = {}
        self.options = options
        self.module = module

        # build all functions declared in the module
        for func in module.functions.values():
            for f in func.user_overloads.values():
                self.build_function(f)
                if f.custom_replay_func is not None:
                    self.build_function(f.custom_replay_func)

        # build all kernel entry points
        for kernel in module.kernels.values():
            if not kernel.is_generic:
                self.build_kernel(kernel)
            else:
                for k in kernel.overloads.values():
                    self.build_kernel(k)

    def build_struct_recursive(self, struct: warp.codegen.Struct):
        structs = []

        stack = [struct]
        while stack:
            s = stack.pop()

            structs.append(s)

            for var in s.vars.values():
                if isinstance(var.type, warp.codegen.Struct):
                    stack.append(var.type)
                elif isinstance(var.type, warp.types.array) and isinstance(var.type.dtype, warp.codegen.Struct):
                    stack.append(var.type.dtype)

        # Build them in reverse to generate a correct dependency order.
        for s in reversed(structs):
            self.build_struct(s)

    def build_struct(self, struct):
        self.structs[struct] = None

    def build_kernel(self, kernel):
        kernel.adj.build(self)

        if kernel.adj.return_var is not None:
            if kernel.adj.return_var.ctype() != "void":
                raise TypeError(f"Error, kernels can't have return values, got: {kernel.adj.return_var}")

    def build_function(self, func):
        if func in self.functions:
            return
        else:
            func.adj.build(self)

            # complete the function return type after we have analyzed it (inferred from return statement in ast)
            if not func.value_func:

                def wrap(adj):
                    def value_type(arg_types, kwds, templates):
                        if adj.return_var is None or len(adj.return_var) == 0:
                            return None
                        if len(adj.return_var) == 1:
                            return adj.return_var[0].type
                        else:
                            return [v.type for v in adj.return_var]

                    return value_type

                func.value_func = wrap(func.adj)

            # use dict to preserve import order
            self.functions[func] = None

    def codegen(self, device):
        source = ""

        # code-gen structs
        for struct in self.structs.keys():
            source += warp.codegen.codegen_struct(struct)

        # code-gen all imported functions
        for func in self.functions.keys():
            if func.native_snippet is None:
                source += warp.codegen.codegen_func(
                    func.adj, c_func_name=func.native_func, device=device, options=self.options
                )
            else:
                source += warp.codegen.codegen_snippet(
                    func.adj, name=func.key, snippet=func.native_snippet, adj_snippet=func.adj_native_snippet
                )

        for kernel in self.module.kernels.values():
            # each kernel gets an entry point in the module
            if not kernel.is_generic:
                source += warp.codegen.codegen_kernel(kernel, device=device, options=self.options)
                source += warp.codegen.codegen_module(kernel, device=device)
            else:
                for k in kernel.overloads.values():
                    source += warp.codegen.codegen_kernel(k, device=device, options=self.options)
                    source += warp.codegen.codegen_module(k, device=device)

        # add headers
        if device == "cpu":
            source = warp.codegen.cpu_module_header + source
        else:
            source = warp.codegen.cuda_module_header + source

        return source


# -----------------------------------------------------
# stores all functions and kernels for a Python module
# creates a hash of the function to use for checking
# build cache


class Module:
    def __init__(self, name, loader):
        self.name = name
        self.loader = loader

        self.kernels = {}
        self.functions = {}
        self.constants = []
        self.structs = {}

        self.cpu_module = None
        self.cuda_modules = {}  # module lookup by CUDA context

        self.cpu_build_failed = False
        self.cuda_build_failed = False

        self.options = {
            "max_unroll": 16,
            "enable_backward": warp.config.enable_backward,
            "fast_math": False,
            "cuda_output": None,  # supported values: "ptx", "cubin", or None (automatic)
            "mode": warp.config.mode,
        }

        # kernel hook lookup per device
        # hooks are stored with the module so they can be easily cleared when the module is reloaded.
        # -> See ``Module.get_kernel_hooks()``
        self.kernel_hooks = {}

        # Module dependencies are determined by scanning each function
        # and kernel for references to external functions and structs.
        #
        # When a referenced module is modified, all of its dependents need to be reloaded
        # on the next launch.  To detect this, a module's hash recursively includes
        # all of its references.
        # -> See ``Module.hash_module()``
        #
        # The dependency mechanism works for both static and dynamic (runtime) modifications.
        # When a module is reloaded at runtime, we recursively unload all of its
        # dependents, so that they will be re-hashed and reloaded on the next launch.
        # -> See ``get_module()``

        self.references = set()  # modules whose content we depend on
        self.dependents = set()  # modules that depend on our content

        # Since module hashing is recursive, we improve performance by caching the hash of the
        # module contents (kernel source, function source, and struct source).
        # After all kernels, functions, and structs are added to the module (usually at import time),
        # the content hash doesn't change.
        # -> See ``Module.hash_module_recursive()``

        self.content_hash = None

        # number of times module auto-generates kernel key for user
        # used to ensure unique kernel keys
        self.count = 0

    def register_struct(self, struct):
        self.structs[struct.key] = struct

        # for a reload of module on next launch
        self.unload()

    def register_kernel(self, kernel):
        self.kernels[kernel.key] = kernel

        self.find_references(kernel.adj)

        # for a reload of module on next launch
        self.unload()

    def register_function(self, func, skip_adding_overload=False):
        if func.key not in self.functions:
            self.functions[func.key] = func
        else:
            # Check whether the new function's signature match any that has
            # already been registered. If so, then we simply override it, as
            # Python would do it, otherwise we register it as a new overload.
            func_existing = self.functions[func.key]
            sig = warp.types.get_signature(
                func.input_types.values(),
                func_name=func.key,
                arg_names=list(func.input_types.keys()),
            )
            sig_existing = warp.types.get_signature(
                func_existing.input_types.values(),
                func_name=func_existing.key,
                arg_names=list(func_existing.input_types.keys()),
            )
            if sig == sig_existing:
                self.functions[func.key] = func
            elif not skip_adding_overload:
                func_existing.add_overload(func)

        self.find_references(func.adj)

        # for a reload of module on next launch
        self.unload()

    def generate_unique_kernel_key(self, key):
        unique_key = f"{key}_{self.count}"
        self.count += 1
        return unique_key

    # collect all referenced functions / structs
    # given the AST of a function or kernel
    def find_references(self, adj):
        def add_ref(ref):
            if ref is not self:
                self.references.add(ref)
                ref.dependents.add(self)

        # scan for function calls
        for node in ast.walk(adj.tree):
            if isinstance(node, ast.Call):
                try:
                    # try to resolve the function
                    func, _ = adj.resolve_static_expression(node.func, eval_types=False)

                    # if this is a user-defined function, add a module reference
                    if isinstance(func, warp.context.Function) and func.module is not None:
                        add_ref(func.module)

                except Exception:
                    # Lookups may fail for builtins, but that's ok.
                    # Lookups may also fail for functions in this module that haven't been imported yet,
                    # and that's ok too (not an external reference).
                    pass

        # scan for structs
        for arg in adj.args:
            if isinstance(arg.type, warp.codegen.Struct) and arg.type.module is not None:
                add_ref(arg.type.module)

    def hash_module(self):
        def get_annotations(obj: Any) -> Mapping[str, Any]:
            """Alternative to `inspect.get_annotations()` for Python 3.9 and older."""
            # See https://docs.python.org/3/howto/annotations.html#accessing-the-annotations-dict-of-an-object-in-python-3-9-and-older
            if isinstance(obj, type):
                return obj.__dict__.get("__annotations__", {})

            return getattr(obj, "__annotations__", {})

        def get_type_name(type_hint):
            if isinstance(type_hint, warp.codegen.Struct):
                return get_type_name(type_hint.cls)
            return type_hint

        def hash_recursive(module, visited):
            # Hash this module, including all referenced modules recursively.
            # The visited set tracks modules already visited to avoid circular references.

            # check if we need to update the content hash
            if not module.content_hash:
                # recompute content hash
                ch = hashlib.sha256()

                # struct source
                for struct in module.structs.values():
                    s = ",".join(
                        "{}: {}".format(name, get_type_name(type_hint))
                        for name, type_hint in get_annotations(struct.cls).items()
                    )
                    ch.update(bytes(s, "utf-8"))

                # functions source
                for func in module.functions.values():
                    s = func.adj.source
                    ch.update(bytes(s, "utf-8"))

                    if func.custom_grad_func:
                        s = func.custom_grad_func.adj.source
                        ch.update(bytes(s, "utf-8"))
                    if func.custom_replay_func:
                        s = func.custom_replay_func.adj.source

                    # cache func arg types
                    for arg, arg_type in func.adj.arg_types.items():
                        s = f"{arg}: {get_type_name(arg_type)}"
                        ch.update(bytes(s, "utf-8"))

                # kernel source
                for kernel in module.kernels.values():
                    ch.update(bytes(kernel.adj.source, "utf-8"))
                    # cache kernel arg types
                    for arg, arg_type in kernel.adj.arg_types.items():
                        s = f"{arg}: {get_type_name(arg_type)}"
                        ch.update(bytes(s, "utf-8"))
                    # for generic kernels the Python source is always the same,
                    # but we hash the type signatures of all the overloads
                    if kernel.is_generic:
                        for sig in sorted(kernel.overloads.keys()):
                            ch.update(bytes(sig, "utf-8"))

                module.content_hash = ch.digest()

            h = hashlib.sha256()

            # content hash
            h.update(module.content_hash)

            # configuration parameters
            for k in sorted(module.options.keys()):
                s = f"{k}={module.options[k]}"
                h.update(bytes(s, "utf-8"))

            # ensure to trigger recompilation if flags affecting kernel compilation are changed
            if warp.config.verify_fp:
                h.update(bytes("verify_fp", "utf-8"))

            h.update(bytes(warp.config.mode, "utf-8"))

            # compile-time constants (global)
            if warp.types._constant_hash:
                h.update(warp.types._constant_hash.digest())

            # recurse on references
            visited.add(module)

            sorted_deps = sorted(module.references, key=lambda m: m.name)
            for dep in sorted_deps:
                if dep not in visited:
                    dep_hash = hash_recursive(dep, visited)
                    h.update(dep_hash)

            return h.digest()

        return hash_recursive(self, visited=set())

    def load(self, device):
        from warp.utils import ScopedTimer

        device = get_device(device)

        if device.is_cpu:
            # check if already loaded
            if self.cpu_module:
                return True
            # avoid repeated build attempts
            if self.cpu_build_failed:
                return False
            if not warp.is_cpu_available():
                raise RuntimeError("Failed to build CPU module because no CPU buildchain was found")
        else:
            # check if already loaded
            if device.context in self.cuda_modules:
                return True
            # avoid repeated build attempts
            if self.cuda_build_failed:
                return False
            if not warp.is_cuda_available():
                raise RuntimeError("Failed to build CUDA module because CUDA is not available")

        with ScopedTimer(f"Module {self.name} load on device '{device}'", active=not warp.config.quiet):
            build_path = warp.build.kernel_bin_dir
            gen_path = warp.build.kernel_gen_dir

            if not os.path.exists(build_path):
                os.makedirs(build_path)
            if not os.path.exists(gen_path):
                os.makedirs(gen_path)

            module_name = "wp_" + self.name
            module_path = os.path.join(build_path, module_name)
            module_hash = self.hash_module()

            builder = ModuleBuilder(self, self.options)

            if device.is_cpu:
                obj_path = os.path.join(build_path, module_name)
                obj_path = obj_path + ".o"
                cpu_hash_path = module_path + ".cpu.hash"

                # check cache
                if warp.config.cache_kernels and os.path.isfile(cpu_hash_path) and os.path.isfile(obj_path):
                    with open(cpu_hash_path, "rb") as f:
                        cache_hash = f.read()

                    if cache_hash == module_hash:
                        runtime.llvm.load_obj(obj_path.encode("utf-8"), module_name.encode("utf-8"))
                        self.cpu_module = module_name
                        return True

                # build
                try:
                    cpp_path = os.path.join(gen_path, module_name + ".cpp")

                    # write cpp sources
                    cpp_source = builder.codegen("cpu")

                    cpp_file = open(cpp_path, "w")
                    cpp_file.write(cpp_source)
                    cpp_file.close()

                    # build object code
                    with ScopedTimer("Compile x86", active=warp.config.verbose):
                        warp.build.build_cpu(
                            obj_path,
                            cpp_path,
                            mode=self.options["mode"],
                            fast_math=self.options["fast_math"],
                            verify_fp=warp.config.verify_fp,
                        )

                    # update cpu hash
                    with open(cpu_hash_path, "wb") as f:
                        f.write(module_hash)

                    # load the object code
                    runtime.llvm.load_obj(obj_path.encode("utf-8"), module_name.encode("utf-8"))
                    self.cpu_module = module_name

                except Exception as e:
                    self.cpu_build_failed = True
                    raise (e)

            elif device.is_cuda:
                # determine whether to use PTX or CUBIN
                if device.is_cubin_supported:
                    # get user preference specified either per module or globally
                    preferred_cuda_output = self.options.get("cuda_output") or warp.config.cuda_output
                    if preferred_cuda_output is not None:
                        use_ptx = preferred_cuda_output == "ptx"
                    else:
                        # determine automatically: older drivers may not be able to handle PTX generated using newer
                        # CUDA Toolkits, in which case we fall back on generating CUBIN modules
                        use_ptx = runtime.driver_version >= runtime.toolkit_version
                else:
                    # CUBIN not an option, must use PTX (e.g. CUDA Toolkit too old)
                    use_ptx = True

                if use_ptx:
                    output_arch = min(device.arch, warp.config.ptx_target_arch)
                    output_path = module_path + f".sm{output_arch}.ptx"
                else:
                    output_arch = device.arch
                    output_path = module_path + f".sm{output_arch}.cubin"

                cuda_hash_path = module_path + f".sm{output_arch}.hash"

                # check cache
                if warp.config.cache_kernels and os.path.isfile(cuda_hash_path) and os.path.isfile(output_path):
                    with open(cuda_hash_path, "rb") as f:
                        cache_hash = f.read()

                    if cache_hash == module_hash:
                        cuda_module = warp.build.load_cuda(output_path, device)
                        if cuda_module is not None:
                            self.cuda_modules[device.context] = cuda_module
                            return True

                # build
                try:
                    cu_path = os.path.join(gen_path, module_name + ".cu")

                    # write cuda sources
                    cu_source = builder.codegen("cuda")

                    cu_file = open(cu_path, "w")
                    cu_file.write(cu_source)
                    cu_file.close()

                    # generate PTX or CUBIN
                    with ScopedTimer("Compile CUDA", active=warp.config.verbose):
                        warp.build.build_cuda(
                            cu_path,
                            output_arch,
                            output_path,
                            config=self.options["mode"],
                            fast_math=self.options["fast_math"],
                            verify_fp=warp.config.verify_fp,
                        )

                    # update cuda hash
                    with open(cuda_hash_path, "wb") as f:
                        f.write(module_hash)

                    # load the module
                    cuda_module = warp.build.load_cuda(output_path, device)
                    if cuda_module is not None:
                        self.cuda_modules[device.context] = cuda_module
                    else:
                        raise Exception("Failed to load CUDA module")

                except Exception as e:
                    self.cuda_build_failed = True
                    raise (e)

            return True

    def unload(self):
        if self.cpu_module:
            runtime.llvm.unload_obj(self.cpu_module.encode("utf-8"))
            self.cpu_module = None

        # need to unload the CUDA module from all CUDA contexts where it is loaded
        # note: we ensure that this doesn't change the current CUDA context
        if self.cuda_modules:
            saved_context = runtime.core.cuda_context_get_current()
            for context, module in self.cuda_modules.items():
                runtime.core.cuda_unload_module(context, module)
            runtime.core.cuda_context_set_current(saved_context)
            self.cuda_modules = {}

        # clear kernel hooks
        self.kernel_hooks = {}

        # clear content hash
        self.content_hash = None

    # lookup and cache kernel entry points based on name, called after compilation / module load
    def get_kernel_hooks(self, kernel, device):
        # get all hooks for this device
        device_hooks = self.kernel_hooks.get(device.context)
        if device_hooks is None:
            self.kernel_hooks[device.context] = device_hooks = {}

        # look up this kernel
        hooks = device_hooks.get(kernel)
        if hooks is not None:
            return hooks

        name = kernel.get_mangled_name()

        if device.is_cpu:
            func = ctypes.CFUNCTYPE(None)
            forward = func(
                runtime.llvm.lookup(self.cpu_module.encode("utf-8"), (name + "_cpu_forward").encode("utf-8"))
            )
            backward = func(
                runtime.llvm.lookup(self.cpu_module.encode("utf-8"), (name + "_cpu_backward").encode("utf-8"))
            )
        else:
            cu_module = self.cuda_modules[device.context]
            forward = runtime.core.cuda_get_kernel(
                device.context, cu_module, (name + "_cuda_kernel_forward").encode("utf-8")
            )
            backward = runtime.core.cuda_get_kernel(
                device.context, cu_module, (name + "_cuda_kernel_backward").encode("utf-8")
            )

        hooks = KernelHooks(forward, backward)
        device_hooks[kernel] = hooks
        return hooks


# -------------------------------------------
# execution context


# a simple allocator
# TODO: use a pooled allocator to avoid hitting the system allocator
class Allocator:
    def __init__(self, device):
        self.device = device

    def alloc(self, size_in_bytes, pinned=False):
        if self.device.is_cuda:
            if self.device.is_capturing:
                raise RuntimeError(f"Cannot allocate memory on device {self} while graph capture is active")
            return runtime.core.alloc_device(self.device.context, size_in_bytes)
        elif self.device.is_cpu:
            if pinned:
                return runtime.core.alloc_pinned(size_in_bytes)
            else:
                return runtime.core.alloc_host(size_in_bytes)

    def free(self, ptr, size_in_bytes, pinned=False):
        if self.device.is_cuda:
            if self.device.is_capturing:
                raise RuntimeError(f"Cannot free memory on device {self} while graph capture is active")
            return runtime.core.free_device(self.device.context, ptr)
        elif self.device.is_cpu:
            if pinned:
                return runtime.core.free_pinned(ptr)
            else:
                return runtime.core.free_host(ptr)


class ContextGuard:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        if self.device.is_cuda:
            runtime.core.cuda_context_push_current(self.device.context)
        elif is_cuda_driver_initialized():
            self.saved_context = runtime.core.cuda_context_get_current()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device.is_cuda:
            runtime.core.cuda_context_pop_current()
        elif is_cuda_driver_initialized():
            runtime.core.cuda_context_set_current(self.saved_context)


class Stream:
    def __init__(self, device=None, **kwargs):
        self.owner = False

        # we can't use get_device() if called during init, but we can use an explicit Device arg
        if runtime is not None:
            device = runtime.get_device(device)
        elif not isinstance(device, Device):
            raise RuntimeError(
                "A device object is required when creating a stream before or during Warp initialization"
            )

        if not device.is_cuda:
            raise RuntimeError(f"Device {device} is not a CUDA device")

        # we pass cuda_stream through kwargs because cuda_stream=None is actually a valid value (CUDA default stream)
        if "cuda_stream" in kwargs:
            self.cuda_stream = kwargs["cuda_stream"]
        else:
            self.cuda_stream = device.runtime.core.cuda_stream_create(device.context)
            if not self.cuda_stream:
                raise RuntimeError(f"Failed to create stream on device {device}")
            self.owner = True

        self.device = device

    def __del__(self):
        if self.owner:
            runtime.core.cuda_stream_destroy(self.device.context, self.cuda_stream)

    def record_event(self, event=None):
        if event is None:
            event = Event(self.device)
        elif event.device != self.device:
            raise RuntimeError(
                f"Event from device {event.device} cannot be recorded on stream from device {self.device}"
            )

        runtime.core.cuda_event_record(self.device.context, event.cuda_event, self.cuda_stream)

        return event

    def wait_event(self, event):
        runtime.core.cuda_stream_wait_event(self.device.context, self.cuda_stream, event.cuda_event)

    def wait_stream(self, other_stream, event=None):
        if event is None:
            event = Event(other_stream.device)

        runtime.core.cuda_stream_wait_stream(
            self.device.context, self.cuda_stream, other_stream.cuda_stream, event.cuda_event
        )


class Event:
    # event creation flags
    class Flags:
        DEFAULT = 0x0
        BLOCKING_SYNC = 0x1
        DISABLE_TIMING = 0x2

    def __init__(self, device=None, cuda_event=None, enable_timing=False):
        self.owner = False

        device = get_device(device)
        if not device.is_cuda:
            raise RuntimeError(f"Device {device} is not a CUDA device")

        self.device = device

        if cuda_event is not None:
            self.cuda_event = cuda_event
        else:
            flags = Event.Flags.DEFAULT
            if not enable_timing:
                flags |= Event.Flags.DISABLE_TIMING
            self.cuda_event = runtime.core.cuda_event_create(device.context, flags)
            if not self.cuda_event:
                raise RuntimeError(f"Failed to create event on device {device}")
            self.owner = True

    def __del__(self):
        if self.owner:
            runtime.core.cuda_event_destroy(self.device.context, self.cuda_event)


class Device:
    def __init__(self, runtime, alias, ordinal=-1, is_primary=False, context=None):
        self.runtime = runtime
        self.alias = alias
        self.ordinal = ordinal
        self.is_primary = is_primary

        # context can be None to avoid acquiring primary contexts until the device is used
        self._context = context

        # if the device context is not primary, it cannot be None
        if ordinal != -1 and not is_primary:
            assert context is not None

        # streams will be created when context is acquired
        self._stream = None
        self.null_stream = None

        # indicates whether CUDA graph capture is active for this device
        self.is_capturing = False

        self.allocator = Allocator(self)
        self.context_guard = ContextGuard(self)

        if self.ordinal == -1:
            # CPU device
            self.name = platform.processor() or "CPU"
            self.arch = 0
            self.is_uva = False
            self.is_cubin_supported = False
            self.is_mempool_supported = False

            # TODO: add more device-specific dispatch functions
            self.memset = runtime.core.memset_host
            self.memtile = runtime.core.memtile_host

        elif ordinal >= 0 and ordinal < runtime.core.cuda_device_get_count():
            # CUDA device
            self.name = runtime.core.cuda_device_get_name(ordinal).decode()
            self.arch = runtime.core.cuda_device_get_arch(ordinal)
            self.is_uva = runtime.core.cuda_device_is_uva(ordinal)
            # check whether our NVRTC can generate CUBINs for this architecture
            self.is_cubin_supported = self.arch in runtime.nvrtc_supported_archs
            self.is_mempool_supported = runtime.core.cuda_device_is_memory_pool_supported(ordinal)

            # Warn the user of a possible misconfiguration of their system
            if not self.is_mempool_supported:
                warp.utils.warn(
                    f"Support for stream ordered memory allocators was not detected on device {ordinal}. "
                    "This can prevent the use of graphs and/or result in poor performance. "
                    "Is the UVM driver enabled?"
                )

            # initialize streams unless context acquisition is postponed
            if self._context is not None:
                self.init_streams()

            # TODO: add more device-specific dispatch functions
            self.memset = lambda ptr, value, size: runtime.core.memset_device(self.context, ptr, value, size)
            self.memtile = lambda ptr, src, srcsize, reps: runtime.core.memtile_device(
                self.context, ptr, src, srcsize, reps
            )

        else:
            raise RuntimeError(f"Invalid device ordinal ({ordinal})'")

    def init_streams(self):
        # create a stream for asynchronous work
        self.stream = Stream(self)

        # CUDA default stream for some synchronous operations
        self.null_stream = Stream(self, cuda_stream=None)

    @property
    def is_cpu(self):
        return self.ordinal < 0

    @property
    def is_cuda(self):
        return self.ordinal >= 0

    @property
    def context(self):
        if self._context is not None:
            return self._context
        elif self.is_primary:
            # acquire primary context on demand
            self._context = self.runtime.core.cuda_device_primary_context_retain(self.ordinal)
            if self._context is None:
                raise RuntimeError(f"Failed to acquire primary context for device {self}")
            self.runtime.context_map[self._context] = self
            # initialize streams
            self.init_streams()
        return self._context

    @property
    def has_context(self):
        return self._context is not None

    @property
    def stream(self):
        if self.context:
            return self._stream
        else:
            raise RuntimeError(f"Device {self} is not a CUDA device")

    @stream.setter
    def stream(self, s):
        if self.is_cuda:
            if s.device != self:
                raise RuntimeError(f"Stream from device {s.device} cannot be used on device {self}")
            self._stream = s
            self.runtime.core.cuda_context_set_stream(self.context, s.cuda_stream)
        else:
            raise RuntimeError(f"Device {self} is not a CUDA device")

    @property
    def has_stream(self):
        return self._stream is not None

    def __str__(self):
        return self.alias

    def __repr__(self):
        return f"'{self.alias}'"

    def __eq__(self, other):
        if self is other:
            return True
        elif isinstance(other, Device):
            return self.context == other.context
        elif isinstance(other, str):
            if other == "cuda":
                return self == self.runtime.get_current_cuda_device()
            else:
                return other == self.alias
        else:
            return False

    def make_current(self):
        if self.context is not None:
            self.runtime.core.cuda_context_set_current(self.context)

    def can_access(self, other):
        other = self.runtime.get_device(other)
        if self.context == other.context:
            return True
        elif self.context is not None and other.context is not None:
            return bool(self.runtime.core.cuda_context_can_access_peer(self.context, other.context))
        else:
            return False


""" Meta-type for arguments that can be resolved to a concrete Device.
"""
Devicelike = Union[Device, str, None]


class Graph:
    def __init__(self, device: Device, exec: ctypes.c_void_p):
        self.device = device
        self.exec = exec

    def __del__(self):
        # use CUDA context guard to avoid side effects during garbage collection
        with self.device.context_guard:
            runtime.core.cuda_graph_destroy(self.device.context, self.exec)


class Runtime:
    def __init__(self):
        bin_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin")

        if os.name == "nt":
            if sys.version_info[0] > 3 or sys.version_info[0] == 3 and sys.version_info[1] >= 8:
                # Python >= 3.8 this method to add dll search paths
                os.add_dll_directory(bin_path)

            else:
                # Python < 3.8 we add dll directory to path
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

            warp_lib = os.path.join(bin_path, "warp.dll")
            llvm_lib = os.path.join(bin_path, "warp-clang.dll")

        elif sys.platform == "darwin":
            warp_lib = os.path.join(bin_path, "libwarp.dylib")
            llvm_lib = os.path.join(bin_path, "libwarp-clang.dylib")

        else:
            warp_lib = os.path.join(bin_path, "warp.so")
            llvm_lib = os.path.join(bin_path, "warp-clang.so")

        self.core = self.load_dll(warp_lib)

        if os.path.exists(llvm_lib):
            self.llvm = self.load_dll(llvm_lib)
            # setup c-types for warp-clang.dll
            self.llvm.lookup.restype = ctypes.c_uint64
        else:
            self.llvm = None

        # setup c-types for warp.dll
        self.core.alloc_host.argtypes = [ctypes.c_size_t]
        self.core.alloc_host.restype = ctypes.c_void_p
        self.core.alloc_pinned.argtypes = [ctypes.c_size_t]
        self.core.alloc_pinned.restype = ctypes.c_void_p
        self.core.alloc_device.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.core.alloc_device.restype = ctypes.c_void_p

        self.core.float_to_half_bits.argtypes = [ctypes.c_float]
        self.core.float_to_half_bits.restype = ctypes.c_uint16
        self.core.half_bits_to_float.argtypes = [ctypes.c_uint16]
        self.core.half_bits_to_float.restype = ctypes.c_float

        self.core.free_host.argtypes = [ctypes.c_void_p]
        self.core.free_host.restype = None
        self.core.free_pinned.argtypes = [ctypes.c_void_p]
        self.core.free_pinned.restype = None
        self.core.free_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.free_device.restype = None

        self.core.memset_host.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.core.memset_host.restype = None
        self.core.memset_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.core.memset_device.restype = None

        self.core.memtile_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        self.core.memtile_host.restype = None
        self.core.memtile_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self.core.memtile_device.restype = None

        self.core.memcpy_h2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.core.memcpy_h2h.restype = None
        self.core.memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.core.memcpy_h2d.restype = None
        self.core.memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.core.memcpy_d2h.restype = None
        self.core.memcpy_d2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.core.memcpy_d2d.restype = None
        self.core.memcpy_peer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.core.memcpy_peer.restype = None

        self.core.array_copy_host.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.core.array_copy_host.restype = ctypes.c_size_t
        self.core.array_copy_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.core.array_copy_device.restype = ctypes.c_size_t

        self.core.array_fill_host.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.core.array_fill_host.restype = None
        self.core.array_fill_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.core.array_fill_device.restype = None

        self.core.array_sum_double_host.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.core.array_sum_float_host.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.core.array_sum_double_device.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.core.array_sum_float_device.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.core.array_inner_double_host.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.core.array_inner_float_host.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.core.array_inner_double_device.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.core.array_inner_float_device.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.core.array_scan_int_host.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_bool]
        self.core.array_scan_float_host.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_bool]
        self.core.array_scan_int_device.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_bool]
        self.core.array_scan_float_device.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_bool]

        self.core.radix_sort_pairs_int_host.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
        self.core.radix_sort_pairs_int_device.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

        self.core.runlength_encode_int_host.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        self.core.runlength_encode_int_device.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
        ]

        self.core.bvh_create_host.restype = ctypes.c_uint64
        self.core.bvh_create_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

        self.core.bvh_create_device.restype = ctypes.c_uint64
        self.core.bvh_create_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

        self.core.bvh_destroy_host.argtypes = [ctypes.c_uint64]
        self.core.bvh_destroy_device.argtypes = [ctypes.c_uint64]

        self.core.bvh_refit_host.argtypes = [ctypes.c_uint64]
        self.core.bvh_refit_device.argtypes = [ctypes.c_uint64]

        self.core.mesh_create_host.restype = ctypes.c_uint64
        self.core.mesh_create_host.argtypes = [
            warp.types.array_t,
            warp.types.array_t,
            warp.types.array_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.core.mesh_create_device.restype = ctypes.c_uint64
        self.core.mesh_create_device.argtypes = [
            ctypes.c_void_p,
            warp.types.array_t,
            warp.types.array_t,
            warp.types.array_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.core.mesh_destroy_host.argtypes = [ctypes.c_uint64]
        self.core.mesh_destroy_device.argtypes = [ctypes.c_uint64]

        self.core.mesh_refit_host.argtypes = [ctypes.c_uint64]
        self.core.mesh_refit_device.argtypes = [ctypes.c_uint64]

        self.core.hash_grid_create_host.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.core.hash_grid_create_host.restype = ctypes.c_uint64
        self.core.hash_grid_destroy_host.argtypes = [ctypes.c_uint64]
        self.core.hash_grid_update_host.argtypes = [ctypes.c_uint64, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
        self.core.hash_grid_reserve_host.argtypes = [ctypes.c_uint64, ctypes.c_int]

        self.core.hash_grid_create_device.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.core.hash_grid_create_device.restype = ctypes.c_uint64
        self.core.hash_grid_destroy_device.argtypes = [ctypes.c_uint64]
        self.core.hash_grid_update_device.argtypes = [ctypes.c_uint64, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
        self.core.hash_grid_reserve_device.argtypes = [ctypes.c_uint64, ctypes.c_int]

        self.core.cutlass_gemm.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_int,
        ]
        self.core.cutlass_gemm.restypes = ctypes.c_bool

        self.core.volume_create_host.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.core.volume_create_host.restype = ctypes.c_uint64
        self.core.volume_get_buffer_info_host.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.core.volume_get_tiles_host.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.core.volume_destroy_host.argtypes = [ctypes.c_uint64]

        self.core.volume_create_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        self.core.volume_create_device.restype = ctypes.c_uint64
        self.core.volume_f_from_tiles_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_bool,
        ]
        self.core.volume_f_from_tiles_device.restype = ctypes.c_uint64
        self.core.volume_v_from_tiles_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_bool,
        ]
        self.core.volume_v_from_tiles_device.restype = ctypes.c_uint64
        self.core.volume_i_from_tiles_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_bool,
        ]
        self.core.volume_i_from_tiles_device.restype = ctypes.c_uint64
        self.core.volume_get_buffer_info_device.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.core.volume_get_tiles_device.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.core.volume_destroy_device.argtypes = [ctypes.c_uint64]

        self.core.volume_get_voxel_size.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        bsr_matrix_from_triplets_argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self.core.bsr_matrix_from_triplets_float_host.argtypes = bsr_matrix_from_triplets_argtypes
        self.core.bsr_matrix_from_triplets_double_host.argtypes = bsr_matrix_from_triplets_argtypes
        self.core.bsr_matrix_from_triplets_float_device.argtypes = bsr_matrix_from_triplets_argtypes
        self.core.bsr_matrix_from_triplets_double_device.argtypes = bsr_matrix_from_triplets_argtypes

        self.core.bsr_matrix_from_triplets_float_host.restype = ctypes.c_int
        self.core.bsr_matrix_from_triplets_double_host.restype = ctypes.c_int
        self.core.bsr_matrix_from_triplets_float_device.restype = ctypes.c_int
        self.core.bsr_matrix_from_triplets_double_device.restype = ctypes.c_int

        bsr_transpose_argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self.core.bsr_transpose_float_host.argtypes = bsr_transpose_argtypes
        self.core.bsr_transpose_double_host.argtypes = bsr_transpose_argtypes
        self.core.bsr_transpose_float_device.argtypes = bsr_transpose_argtypes
        self.core.bsr_transpose_double_device.argtypes = bsr_transpose_argtypes

        self.core.is_cuda_enabled.argtypes = None
        self.core.is_cuda_enabled.restype = ctypes.c_int
        self.core.is_cuda_compatibility_enabled.argtypes = None
        self.core.is_cuda_compatibility_enabled.restype = ctypes.c_int
        self.core.is_cutlass_enabled.argtypes = None
        self.core.is_cutlass_enabled.restype = ctypes.c_int

        self.core.cuda_driver_version.argtypes = None
        self.core.cuda_driver_version.restype = ctypes.c_int
        self.core.cuda_toolkit_version.argtypes = None
        self.core.cuda_toolkit_version.restype = ctypes.c_int
        self.core.cuda_driver_is_initialized.argtypes = None
        self.core.cuda_driver_is_initialized.restype = ctypes.c_bool

        self.core.nvrtc_supported_arch_count.argtypes = None
        self.core.nvrtc_supported_arch_count.restype = ctypes.c_int
        self.core.nvrtc_supported_archs.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.core.nvrtc_supported_archs.restype = None

        self.core.cuda_device_get_count.argtypes = None
        self.core.cuda_device_get_count.restype = ctypes.c_int
        self.core.cuda_device_primary_context_retain.argtypes = [ctypes.c_int]
        self.core.cuda_device_primary_context_retain.restype = ctypes.c_void_p
        self.core.cuda_device_get_name.argtypes = [ctypes.c_int]
        self.core.cuda_device_get_name.restype = ctypes.c_char_p
        self.core.cuda_device_get_arch.argtypes = [ctypes.c_int]
        self.core.cuda_device_get_arch.restype = ctypes.c_int
        self.core.cuda_device_is_uva.argtypes = [ctypes.c_int]
        self.core.cuda_device_is_uva.restype = ctypes.c_int

        self.core.cuda_context_get_current.argtypes = None
        self.core.cuda_context_get_current.restype = ctypes.c_void_p
        self.core.cuda_context_set_current.argtypes = [ctypes.c_void_p]
        self.core.cuda_context_set_current.restype = None
        self.core.cuda_context_push_current.argtypes = [ctypes.c_void_p]
        self.core.cuda_context_push_current.restype = None
        self.core.cuda_context_pop_current.argtypes = None
        self.core.cuda_context_pop_current.restype = None
        self.core.cuda_context_create.argtypes = [ctypes.c_int]
        self.core.cuda_context_create.restype = ctypes.c_void_p
        self.core.cuda_context_destroy.argtypes = [ctypes.c_void_p]
        self.core.cuda_context_destroy.restype = None
        self.core.cuda_context_synchronize.argtypes = [ctypes.c_void_p]
        self.core.cuda_context_synchronize.restype = None
        self.core.cuda_context_check.argtypes = [ctypes.c_void_p]
        self.core.cuda_context_check.restype = ctypes.c_uint64

        self.core.cuda_context_get_device_ordinal.argtypes = [ctypes.c_void_p]
        self.core.cuda_context_get_device_ordinal.restype = ctypes.c_int
        self.core.cuda_context_is_primary.argtypes = [ctypes.c_void_p]
        self.core.cuda_context_is_primary.restype = ctypes.c_int
        self.core.cuda_context_get_stream.argtypes = [ctypes.c_void_p]
        self.core.cuda_context_get_stream.restype = ctypes.c_void_p
        self.core.cuda_context_set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_context_set_stream.restype = None
        self.core.cuda_context_can_access_peer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_context_can_access_peer.restype = ctypes.c_int

        self.core.cuda_stream_create.argtypes = [ctypes.c_void_p]
        self.core.cuda_stream_create.restype = ctypes.c_void_p
        self.core.cuda_stream_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_stream_destroy.restype = None
        self.core.cuda_stream_synchronize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_stream_synchronize.restype = None
        self.core.cuda_stream_wait_event.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_stream_wait_event.restype = None
        self.core.cuda_stream_wait_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.core.cuda_stream_wait_stream.restype = None

        self.core.cuda_event_create.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self.core.cuda_event_create.restype = ctypes.c_void_p
        self.core.cuda_event_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_event_destroy.restype = None
        self.core.cuda_event_record.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_event_record.restype = None

        self.core.cuda_graph_begin_capture.argtypes = [ctypes.c_void_p]
        self.core.cuda_graph_begin_capture.restype = None
        self.core.cuda_graph_end_capture.argtypes = [ctypes.c_void_p]
        self.core.cuda_graph_end_capture.restype = ctypes.c_void_p
        self.core.cuda_graph_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_graph_launch.restype = None
        self.core.cuda_graph_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_graph_destroy.restype = None

        self.core.cuda_compile_program.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_char_p,
        ]
        self.core.cuda_compile_program.restype = ctypes.c_size_t

        self.core.cuda_load_module.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.core.cuda_load_module.restype = ctypes.c_void_p

        self.core.cuda_unload_module.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_unload_module.restype = None

        self.core.cuda_get_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        self.core.cuda_get_kernel.restype = ctypes.c_void_p

        self.core.cuda_launch_kernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.core.cuda_launch_kernel.restype = ctypes.c_size_t

        self.core.cuda_graphics_map.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_graphics_map.restype = None
        self.core.cuda_graphics_unmap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_graphics_unmap.restype = None
        self.core.cuda_graphics_device_ptr_and_size.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.core.cuda_graphics_device_ptr_and_size.restype = None
        self.core.cuda_graphics_register_gl_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint]
        self.core.cuda_graphics_register_gl_buffer.restype = ctypes.c_void_p
        self.core.cuda_graphics_unregister_resource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.core.cuda_graphics_unregister_resource.restype = None

        self.core.init.restype = ctypes.c_int

        error = self.core.init()

        if error != 0:
            raise Exception("Warp initialization failed")

        self.device_map = {}  # device lookup by alias
        self.context_map = {}  # device lookup by context

        # register CPU device
        cpu_name = platform.processor()
        if not cpu_name:
            cpu_name = "CPU"
        self.cpu_device = Device(self, "cpu")
        self.device_map["cpu"] = self.cpu_device
        self.context_map[None] = self.cpu_device

        cuda_device_count = self.core.cuda_device_get_count()

        if cuda_device_count > 0:
            # get CUDA Toolkit and driver versions
            self.toolkit_version = self.core.cuda_toolkit_version()
            self.driver_version = self.core.cuda_driver_version()

            # get all architectures supported by NVRTC
            num_archs = self.core.nvrtc_supported_arch_count()
            if num_archs > 0:
                archs = (ctypes.c_int * num_archs)()
                self.core.nvrtc_supported_archs(archs)
                self.nvrtc_supported_archs = list(archs)
            else:
                self.nvrtc_supported_archs = []

        # register CUDA devices
        self.cuda_devices = []
        self.cuda_primary_devices = []
        for i in range(cuda_device_count):
            alias = f"cuda:{i}"
            device = Device(self, alias, ordinal=i, is_primary=True)
            self.cuda_devices.append(device)
            self.cuda_primary_devices.append(device)
            self.device_map[alias] = device

        # set default device
        if cuda_device_count > 0:
            if self.core.cuda_context_get_current() is not None:
                self.set_default_device("cuda")
            else:
                self.set_default_device("cuda:0")
        else:
            # CUDA not available
            self.set_default_device("cpu")

        # initialize kernel cache
        warp.build.init_kernel_cache(warp.config.kernel_cache_dir)

        # print device and version information
        if not warp.config.quiet:
            print(f"Warp {warp.config.version} initialized:")
            if cuda_device_count > 0:
                toolkit_version = (self.toolkit_version // 1000, (self.toolkit_version % 1000) // 10)
                driver_version = (self.driver_version // 1000, (self.driver_version % 1000) // 10)
                print(
                    f"   CUDA Toolkit: {toolkit_version[0]}.{toolkit_version[1]}, Driver: {driver_version[0]}.{driver_version[1]}"
                )
            else:
                if self.core.is_cuda_enabled():
                    # Warp was compiled with CUDA support, but no devices are available
                    print("   CUDA devices not available")
                else:
                    # Warp was compiled without CUDA support
                    print("   CUDA support not enabled in this build")
            print("   Devices:")
            print(f'     "{self.cpu_device.alias}"    | {self.cpu_device.name}')
            for cuda_device in self.cuda_devices:
                print(f'     "{cuda_device.alias}" | {cuda_device.name} (sm_{cuda_device.arch})')
            print(f"   Kernel cache: {warp.config.kernel_cache_dir}")

        # CUDA compatibility check
        if cuda_device_count > 0 and not self.core.is_cuda_compatibility_enabled():
            if self.driver_version < self.toolkit_version:
                print("******************************************************************")
                print("* WARNING:                                                       *")
                print("*   Warp was compiled without CUDA compatibility support         *")
                print("*   (quick build).  The CUDA Toolkit version used to build       *")
                print("*   Warp is not fully supported by the current driver.           *")
                print("*   Some CUDA functionality may not work correctly!              *")
                print("*   Update the driver or rebuild Warp without the --quick flag.  *")
                print("******************************************************************")

        # global tape
        self.tape = None

    def load_dll(self, dll_path):
        try:
            if sys.version_info[0] > 3 or sys.version_info[0] == 3 and sys.version_info[1] >= 8:
                dll = ctypes.CDLL(dll_path, winmode=0)
            else:
                dll = ctypes.CDLL(dll_path)
        except OSError as e:
            if "GLIBCXX" in str(e):
                raise RuntimeError(
                    f"Failed to load the shared library '{dll_path}'.\n"
                    "The execution environment's libstdc++ runtime is older than the version the Warp library was built for.\n"
                    "See https://nvidia.github.io/warp/_build/html/installation.html#conda-environments for details."
                ) from e
            else:
                raise RuntimeError(f"Failed to load the shared library '{dll_path}'") from e
        return dll

    def get_device(self, ident: Devicelike = None) -> Device:
        if isinstance(ident, Device):
            return ident
        elif ident is None:
            return self.default_device
        elif isinstance(ident, str):
            if ident == "cuda":
                return self.get_current_cuda_device()
            else:
                return self.device_map[ident]
        else:
            raise RuntimeError(f"Unable to resolve device from argument of type {type(ident)}")

    def set_default_device(self, ident: Devicelike):
        self.default_device = self.get_device(ident)

    def get_current_cuda_device(self):
        current_context = self.core.cuda_context_get_current()
        if current_context is not None:
            current_device = self.context_map.get(current_context)
            if current_device is not None:
                # this is a known device
                return current_device
            elif self.core.cuda_context_is_primary(current_context):
                # this is a primary context that we haven't used yet
                ordinal = self.core.cuda_context_get_device_ordinal(current_context)
                device = self.cuda_devices[ordinal]
                self.context_map[current_context] = device
                return device
            else:
                # this is an unseen non-primary context, register it as a new device with a unique alias
                alias = f"cuda!{current_context:x}"
                return self.map_cuda_device(alias, current_context)
        elif self.default_device.is_cuda:
            return self.default_device
        elif self.cuda_devices:
            return self.cuda_devices[0]
        else:
            raise RuntimeError("CUDA is not available")

    def rename_device(self, device, alias):
        del self.device_map[device.alias]
        device.alias = alias
        self.device_map[alias] = device
        return device

    def map_cuda_device(self, alias, context=None) -> Device:
        if context is None:
            context = self.core.cuda_context_get_current()
            if context is None:
                raise RuntimeError(f"Unable to determine CUDA context for device alias '{alias}'")

        # check if this alias already exists
        if alias in self.device_map:
            device = self.device_map[alias]
            if context == device.context:
                # device already exists with the same alias, that's fine
                return device
            else:
                raise RuntimeError(f"Device alias '{alias}' already exists")

        # check if this context already has an associated Warp device
        if context in self.context_map:
            # rename the device
            device = self.context_map[context]
            return self.rename_device(device, alias)
        else:
            # it's an unmapped context

            # get the device ordinal
            ordinal = self.core.cuda_context_get_device_ordinal(context)

            # check if this is a primary context (we could get here if it's a device that hasn't been used yet)
            if self.core.cuda_context_is_primary(context):
                # rename the device
                device = self.cuda_primary_devices[ordinal]
                return self.rename_device(device, alias)
            else:
                # create a new Warp device for this context
                device = Device(self, alias, ordinal=ordinal, is_primary=False, context=context)

                self.device_map[alias] = device
                self.context_map[context] = device
                self.cuda_devices.append(device)

                return device

    def unmap_cuda_device(self, alias):
        device = self.device_map.get(alias)

        # make sure the alias refers to a CUDA device
        if device is None or not device.is_cuda:
            raise RuntimeError(f"Invalid CUDA device alias '{alias}'")

        del self.device_map[alias]
        del self.context_map[device.context]
        self.cuda_devices.remove(device)

    def verify_cuda_device(self, device: Devicelike = None):
        if warp.config.verify_cuda:
            device = runtime.get_device(device)
            if not device.is_cuda:
                return

            err = self.core.cuda_context_check(device.context)
            if err != 0:
                raise RuntimeError(f"CUDA error detected: {err}")


def assert_initialized():
    assert runtime is not None, "Warp not initialized, call wp.init() before use"


# global entry points
def is_cpu_available():
    return runtime.llvm


def is_cuda_available():
    return get_cuda_device_count() > 0


def is_device_available(device):
    return device in get_devices()


def is_cuda_driver_initialized() -> bool:
    """Returns ``True`` if the CUDA driver is initialized.

    This is a stricter test than ``is_cuda_available()`` since a CUDA driver
    call to ``cuCtxGetCurrent`` is made, and the result is compared to
    `CUDA_SUCCESS`. Note that `CUDA_SUCCESS` is returned by ``cuCtxGetCurrent``
    even if there is no context bound to the calling CPU thread.

    This can be helpful in cases in which ``cuInit()`` was called before a fork.
    """
    assert_initialized()

    return runtime.core.cuda_driver_is_initialized()


def get_devices() -> List[Device]:
    """Returns a list of devices supported in this environment."""

    assert_initialized()

    devices = []
    if is_cpu_available():
        devices.append(runtime.cpu_device)
    for cuda_device in runtime.cuda_devices:
        devices.append(cuda_device)
    return devices


def get_cuda_device_count() -> int:
    """Returns the number of CUDA devices supported in this environment."""

    assert_initialized()

    return len(runtime.cuda_devices)


def get_cuda_device(ordinal: Union[int, None] = None) -> Device:
    """Returns the CUDA device with the given ordinal or the current CUDA device if ordinal is None."""

    assert_initialized()

    if ordinal is None:
        return runtime.get_current_cuda_device()
    else:
        return runtime.cuda_devices[ordinal]


def get_cuda_devices() -> List[Device]:
    """Returns a list of CUDA devices supported in this environment."""

    assert_initialized()

    return runtime.cuda_devices


def get_preferred_device() -> Device:
    """Returns the preferred compute device, CUDA if available and CPU otherwise."""

    assert_initialized()

    if is_cuda_available():
        return runtime.cuda_devices[0]
    elif is_cpu_available():
        return runtime.cpu_device
    else:
        return None


def get_device(ident: Devicelike = None) -> Device:
    """Returns the device identified by the argument."""

    assert_initialized()

    return runtime.get_device(ident)


def set_device(ident: Devicelike):
    """Sets the target device identified by the argument."""

    assert_initialized()

    device = runtime.get_device(ident)
    runtime.set_default_device(device)
    device.make_current()


def map_cuda_device(alias: str, context: ctypes.c_void_p = None) -> Device:
    """Assign a device alias to a CUDA context.

    This function can be used to create a wp.Device for an external CUDA context.
    If a wp.Device already exists for the given context, it's alias will change to the given value.

    Args:
        alias: A unique string to identify the device.
        context: A CUDA context pointer (CUcontext).  If None, the currently bound CUDA context will be used.

    Returns:
        The associated wp.Device.
    """

    assert_initialized()

    return runtime.map_cuda_device(alias, context)


def unmap_cuda_device(alias: str):
    """Remove a CUDA device with the given alias."""

    assert_initialized()

    runtime.unmap_cuda_device(alias)


def get_stream(device: Devicelike = None) -> Stream:
    """Return the stream currently used by the given device"""

    return get_device(device).stream


def set_stream(stream, device: Devicelike = None):
    """Set the stream to be used by the given device.

    If this is an external stream, caller is responsible for guaranteeing the lifetime of the stream.
    Consider using wp.ScopedStream instead.
    """

    get_device(device).stream = stream


def record_event(event: Event = None):
    """Record a CUDA event on the current stream.

    Args:
        event: Event to record. If None, a new Event will be created.

    Returns:
        The recorded event.
    """

    return get_stream().record_event(event)


def wait_event(event: Event):
    """Make the current stream wait for a CUDA event.

    Args:
        event: Event to wait for.
    """

    get_stream().wait_event(event)


def wait_stream(stream: Stream, event: Event = None):
    """Make the current stream wait for another CUDA stream to complete its work.

    Args:
        event: Event to be used.  If None, a new Event will be created.
    """

    get_stream().wait_stream(stream, event=event)


class RegisteredGLBuffer:
    """
    Helper object to register a GL buffer with CUDA so that it can be mapped to a Warp array.
    """

    # Specifies no hints about how this resource will be used.
    # It is therefore assumed that this resource will be
    # read from and written to by CUDA. This is the default value.
    NONE = 0x00

    # Specifies that CUDA will not write to this resource.
    READ_ONLY = 0x01

    # Specifies that CUDA will not read from this resource and will write over the
    # entire contents of the resource, so none of the data previously
    # stored in the resource will be preserved.
    WRITE_DISCARD = 0x02

    def __init__(self, gl_buffer_id: int, device: Devicelike = None, flags: int = NONE):
        """Create a new RegisteredGLBuffer object.

        Args:
            gl_buffer_id: The OpenGL buffer id (GLuint).
            device: The device to register the buffer with.  If None, the current device will be used.
            flags: A combination of the flags constants.
        """
        self.gl_buffer_id = gl_buffer_id
        self.device = get_device(device)
        self.context = self.device.context
        self.resource = runtime.core.cuda_graphics_register_gl_buffer(self.context, gl_buffer_id, flags)

    def __del__(self):
        runtime.core.cuda_graphics_unregister_resource(self.context, self.resource)

    def map(self, dtype, shape) -> warp.array:
        """Map the OpenGL buffer to a Warp array.

        Args:
            dtype: The type of each element in the array.
            shape: The shape of the array.

        Returns:
            A Warp array object representing the mapped OpenGL buffer.
        """
        runtime.core.cuda_graphics_map(self.context, self.resource)
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t)
        ptr = ctypes.c_uint64(0)
        size = ctypes.c_size_t(0)
        runtime.core.cuda_graphics_device_ptr_and_size(
            self.context, self.resource, ctypes.byref(ptr), ctypes.byref(size)
        )
        return warp.array(ptr=ptr.value, dtype=dtype, shape=shape, device=self.device, owner=False)

    def unmap(self):
        """Unmap the OpenGL buffer."""
        runtime.core.cuda_graphics_unmap(self.context, self.resource)


def zeros(
    shape: Tuple = None,
    dtype=float,
    device: Devicelike = None,
    requires_grad: bool = False,
    pinned: bool = False,
    **kwargs,
) -> warp.array:
    """Return a zero-initialized array

    Args:
        shape: Array dimensions
        dtype: Type of each element, e.g.: warp.vec3, warp.mat33, etc
        device: Device that array will live on
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    arr = empty(shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, pinned=pinned, **kwargs)

    # use the CUDA default stream for synchronous behaviour with other streams
    with warp.ScopedStream(arr.device.null_stream):
        arr.zero_()

    return arr


def zeros_like(
    src: warp.array, device: Devicelike = None, requires_grad: bool = None, pinned: bool = None
) -> warp.array:
    """Return a zero-initialized array with the same type and dimension of another array

    Args:
        src: The template array to use for shape, data type, and device
        device: The device where the new array will be created (defaults to src.device)
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    arr = empty_like(src, device=device, requires_grad=requires_grad, pinned=pinned)

    arr.zero_()

    return arr


def full(
    shape: Tuple = None,
    value=0,
    dtype=Any,
    device: Devicelike = None,
    requires_grad: bool = False,
    pinned: bool = False,
    **kwargs,
) -> warp.array:
    """Return an array with all elements initialized to the given value

    Args:
        shape: Array dimensions
        value: Element value
        dtype: Type of each element, e.g.: float, warp.vec3, warp.mat33, etc
        device: Device that array will live on
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    if dtype == Any:
        # determine dtype from value
        value_type = type(value)
        if value_type == int:
            dtype = warp.int32
        elif value_type == float:
            dtype = warp.float32
        elif value_type in warp.types.scalar_types or hasattr(value_type, "_wp_scalar_type_"):
            dtype = value_type
        elif isinstance(value, warp.codegen.StructInstance):
            dtype = value._cls
        elif hasattr(value, "__len__"):
            # a sequence, assume it's a vector or matrix value
            try:
                # try to convert to a numpy array first
                na = np.array(value, copy=False)
            except Exception as e:
                raise ValueError(f"Failed to interpret the value as a vector or matrix: {e}")

            # determine the scalar type
            scalar_type = warp.types.np_dtype_to_warp_type.get(na.dtype)
            if scalar_type is None:
                raise ValueError(f"Failed to convert {na.dtype} to a Warp data type")

            # determine if vector or matrix
            if na.ndim == 1:
                dtype = warp.types.vector(na.size, scalar_type)
            elif na.ndim == 2:
                dtype = warp.types.matrix(na.shape, scalar_type)
            else:
                raise ValueError("Values with more than two dimensions are not supported")
        else:
            raise ValueError(f"Invalid value type for Warp array: {value_type}")

    arr = empty(shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, pinned=pinned, **kwargs)

    # use the CUDA default stream for synchronous behaviour with other streams
    with warp.ScopedStream(arr.device.null_stream):
        arr.fill_(value)

    return arr


def full_like(
    src: warp.array, value: Any, device: Devicelike = None, requires_grad: bool = None, pinned: bool = None
) -> warp.array:
    """Return an array with all elements initialized to the given value with the same type and dimension of another array

    Args:
        src: The template array to use for shape, data type, and device
        value: Element value
        device: The device where the new array will be created (defaults to src.device)
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    arr = empty_like(src, device=device, requires_grad=requires_grad, pinned=pinned)

    arr.fill_(value)

    return arr


def clone(src: warp.array, device: Devicelike = None, requires_grad: bool = None, pinned: bool = None) -> warp.array:
    """Clone an existing array, allocates a copy of the src memory

    Args:
        src: The source array to copy
        device: The device where the new array will be created (defaults to src.device)
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    arr = empty_like(src, device=device, requires_grad=requires_grad, pinned=pinned)

    warp.copy(arr, src)

    return arr


def empty(
    shape: Tuple = None,
    dtype=float,
    device: Devicelike = None,
    requires_grad: bool = False,
    pinned: bool = False,
    **kwargs,
) -> warp.array:
    """Returns an uninitialized array

    Args:
        shape: Array dimensions
        dtype: Type of each element, e.g.: `warp.vec3`, `warp.mat33`, etc
        device: Device that array will live on
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    # backwards compatibility for case where users called wp.empty(n=length, ...)
    if "n" in kwargs:
        shape = (kwargs["n"],)
        del kwargs["n"]

    # ensure shape is specified, even if creating a zero-sized array
    if shape is None:
        shape = 0

    return warp.array(shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, pinned=pinned, **kwargs)


def empty_like(
    src: warp.array, device: Devicelike = None, requires_grad: bool = None, pinned: bool = None
) -> warp.array:
    """Return an uninitialized array with the same type and dimension of another array

    Args:
        src: The template array to use for shape, data type, and device
        device: The device where the new array will be created (defaults to src.device)
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    if device is None:
        device = src.device

    if requires_grad is None:
        if hasattr(src, "requires_grad"):
            requires_grad = src.requires_grad
        else:
            requires_grad = False

    if pinned is None:
        if hasattr(src, "pinned"):
            pinned = src.pinned
        else:
            pinned = False

    arr = empty(shape=src.shape, dtype=src.dtype, device=device, requires_grad=requires_grad, pinned=pinned)
    return arr


def from_numpy(
    arr: np.ndarray,
    dtype: Optional[type] = None,
    shape: Optional[Sequence[int]] = None,
    device: Optional[Devicelike] = None,
    requires_grad: bool = False,
) -> warp.array:
    if dtype is None:
        base_type = warp.types.np_dtype_to_warp_type.get(arr.dtype)
        if base_type is None:
            raise RuntimeError("Unsupported NumPy data type '{}'.".format(arr.dtype))

        dim_count = len(arr.shape)
        if dim_count == 2:
            dtype = warp.types.vector(length=arr.shape[1], dtype=base_type)
        elif dim_count == 3:
            dtype = warp.types.matrix(shape=(arr.shape[1], arr.shape[2]), dtype=base_type)
        else:
            dtype = base_type

    return warp.array(
        data=arr,
        dtype=dtype,
        shape=shape,
        owner=False,
        device=device,
        requires_grad=requires_grad,
    )


# given a kernel destination argument type and a value convert
#  to a c-type that can be passed to a kernel
def pack_arg(kernel, arg_type, arg_name, value, device, adjoint=False):
    if warp.types.is_array(arg_type):
        if value is None:
            # allow for NULL arrays
            return arg_type.__ctype__()

        else:
            # check for array type
            # - in forward passes, array types have to match
            # - in backward passes, indexed array gradients are regular arrays
            if adjoint:
                array_matches = isinstance(value, warp.array)
            else:
                array_matches = type(value) is type(arg_type)

            if not array_matches:
                adj = "adjoint " if adjoint else ""
                raise RuntimeError(
                    f"Error launching kernel '{kernel.key}', {adj}argument '{arg_name}' expects an array of type {type(arg_type)}, but passed value has type {type(value)}."
                )

            # check subtype
            if not warp.types.types_equal(value.dtype, arg_type.dtype):
                adj = "adjoint " if adjoint else ""
                raise RuntimeError(
                    f"Error launching kernel '{kernel.key}', {adj}argument '{arg_name}' expects an array with dtype={arg_type.dtype} but passed array has dtype={value.dtype}."
                )

            # check dimensions
            if value.ndim != arg_type.ndim:
                adj = "adjoint " if adjoint else ""
                raise RuntimeError(
                    f"Error launching kernel '{kernel.key}', {adj}argument '{arg_name}' expects an array with {arg_type.ndim} dimension(s) but the passed array has {value.ndim} dimension(s)."
                )

            # check device
            # if a.device != device and not device.can_access(a.device):
            if value.device != device:
                raise RuntimeError(
                    f"Error launching kernel '{kernel.key}', trying to launch on device='{device}', but input array for argument '{arg_name}' is on device={value.device}."
                )

            return value.__ctype__()

    elif isinstance(arg_type, warp.codegen.Struct):
        assert value is not None
        return value.__ctype__()

    # try to convert to a value type (vec3, mat33, etc)
    elif issubclass(arg_type, ctypes.Array):
        if warp.types.types_equal(type(value), arg_type):
            return value
        else:
            # try constructing the required value from the argument (handles tuple / list, Gf.Vec3 case)
            try:
                return arg_type(value)
            except Exception:
                raise ValueError(f"Failed to convert argument for param {arg_name} to {type_str(arg_type)}")

    elif isinstance(value, bool):
        return ctypes.c_bool(value)

    elif isinstance(value, arg_type):
        try:
            # try to pack as a scalar type
            if arg_type is warp.types.float16:
                return arg_type._type_(warp.types.float_to_half_bits(value.value))
            else:
                return arg_type._type_(value.value)
        except Exception:
            raise RuntimeError(
                "Error launching kernel, unable to pack kernel parameter type "
                f"{type(value)} for param {arg_name}, expected {arg_type}"
            )

    else:
        try:
            # try to pack as a scalar type
            if arg_type is warp.types.float16:
                return arg_type._type_(warp.types.float_to_half_bits(value))
            else:
                return arg_type._type_(value)
        except Exception as e:
            print(e)
            raise RuntimeError(
                "Error launching kernel, unable to pack kernel parameter type "
                f"{type(value)} for param {arg_name}, expected {arg_type}"
            )


# represents all data required for a kernel launch
# so that launches can be replayed quickly, use `wp.launch(..., record_cmd=True)`
class Launch:
    def __init__(self, kernel, device, hooks=None, params=None, params_addr=None, bounds=None, max_blocks=0):
        # if not specified look up hooks
        if not hooks:
            module = kernel.module
            if not module.load(device):
                return

            hooks = module.get_kernel_hooks(kernel, device)

        # if not specified set a zero bound
        if not bounds:
            bounds = warp.types.launch_bounds_t(0)

        # if not specified then build a list of default value params for args
        if not params:
            params = []
            params.append(bounds)

            for a in kernel.adj.args:
                if isinstance(a.type, warp.types.array):
                    params.append(a.type.__ctype__())
                elif isinstance(a.type, warp.codegen.Struct):
                    params.append(a.type().__ctype__())
                else:
                    params.append(pack_arg(kernel, a.type, a.label, 0, device, False))

            kernel_args = [ctypes.c_void_p(ctypes.addressof(x)) for x in params]
            kernel_params = (ctypes.c_void_p * len(kernel_args))(*kernel_args)

            params_addr = kernel_params

        self.kernel = kernel
        self.hooks = hooks
        self.params = params
        self.params_addr = params_addr
        self.device = device
        self.bounds = bounds
        self.max_blocks = max_blocks

    def set_dim(self, dim):
        self.bounds = warp.types.launch_bounds_t(dim)

        # launch bounds always at index 0
        self.params[0] = self.bounds

        # for CUDA kernels we need to update the address to each arg
        if self.params_addr:
            self.params_addr[0] = ctypes.c_void_p(ctypes.addressof(self.bounds))

    # set kernel param at an index, will convert to ctype as necessary
    def set_param_at_index(self, index, value):
        arg_type = self.kernel.adj.args[index].type
        arg_name = self.kernel.adj.args[index].label

        carg = pack_arg(self.kernel, arg_type, arg_name, value, self.device, False)

        self.params[index + 1] = carg

        # for CUDA kernels we need to update the address to each arg
        if self.params_addr:
            self.params_addr[index + 1] = ctypes.c_void_p(ctypes.addressof(carg))

    # set kernel param at an index without any type conversion
    # args must be passed as ctypes or basic int / float types
    def set_param_at_index_from_ctype(self, index, value):
        if isinstance(value, ctypes.Structure):
            # not sure how to directly assign struct->struct without reallocating using ctypes
            self.params[index + 1] = value

            # for CUDA kernels we need to update the address to each arg
            if self.params_addr:
                self.params_addr[index + 1] = ctypes.c_void_p(ctypes.addressof(value))

        else:
            self.params[index + 1].__init__(value)

    # set kernel param by argument name
    def set_param_by_name(self, name, value):
        for i, arg in enumerate(self.kernel.adj.args):
            if arg.label == name:
                self.set_param_at_index(i, value)

    # set kernel param by argument name with no type conversions
    def set_param_by_name_from_ctype(self, name, value):
        # lookup argument index
        for i, arg in enumerate(self.kernel.adj.args):
            if arg.label == name:
                self.set_param_at_index_from_ctype(i, value)

    # set all params
    def set_params(self, values):
        for i, v in enumerate(values):
            self.set_param_at_index(i, v)

    # set all params without performing type-conversions
    def set_params_from_ctypes(self, values):
        for i, v in enumerate(values):
            self.set_param_at_index_from_ctype(i, v)

    def launch(self) -> Any:
        if self.device.is_cpu:
            self.hooks.forward(*self.params)
        else:
            runtime.core.cuda_launch_kernel(
                self.device.context, self.hooks.forward, self.bounds.size, self.max_blocks, self.params_addr
            )


def launch(
    kernel,
    dim: Tuple[int],
    inputs: List,
    outputs: List = [],
    adj_inputs: List = [],
    adj_outputs: List = [],
    device: Devicelike = None,
    stream: Stream = None,
    adjoint=False,
    record_tape=True,
    record_cmd=False,
    max_blocks=0,
):
    """Launch a Warp kernel on the target device

    Kernel launches are asynchronous with respect to the calling Python thread.

    Args:
        kernel: The name of a Warp kernel function, decorated with the ``@wp.kernel`` decorator
        dim: The number of threads to launch the kernel, can be an integer, or a Tuple of ints with max of 4 dimensions
        inputs: The input parameters to the kernel
        outputs: The output parameters (optional)
        adj_inputs: The adjoint inputs (optional)
        adj_outputs: The adjoint outputs (optional)
        device: The device to launch on (optional)
        stream: The stream to launch on (optional)
        adjoint: Whether to run forward or backward pass (typically use False)
        record_tape: When true the launch will be recorded the global wp.Tape() object when present
        record_cmd: When True the launch will be returned as a ``Launch`` command object, the launch will not occur until the user calls ``cmd.launch()``
        max_blocks: The maximum number of CUDA thread blocks to use. Only has an effect for CUDA kernel launches.
            If negative or zero, the maximum hardware value will be used.
    """

    assert_initialized()

    # if stream is specified, use the associated device
    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)

    # check function is a Kernel
    if not isinstance(kernel, Kernel):
        raise RuntimeError("Error launching kernel, can only launch functions decorated with @wp.kernel.")

    # debugging aid
    if warp.config.print_launches:
        print(f"kernel: {kernel.key} dim: {dim} inputs: {inputs} outputs: {outputs} device: {device}")

    # construct launch bounds
    bounds = warp.types.launch_bounds_t(dim)

    if bounds.size > 0:
        # first param is the number of threads
        params = []
        params.append(bounds)

        # converts arguments to kernel's expected ctypes and packs into params
        def pack_args(args, params, adjoint=False):
            for i, a in enumerate(args):
                arg_type = kernel.adj.args[i].type
                arg_name = kernel.adj.args[i].label

                params.append(pack_arg(kernel, arg_type, arg_name, a, device, adjoint))

        fwd_args = inputs + outputs
        adj_args = adj_inputs + adj_outputs

        if (len(fwd_args)) != (len(kernel.adj.args)):
            raise RuntimeError(
                f"Error launching kernel '{kernel.key}', passed {len(fwd_args)} arguments but kernel requires {len(kernel.adj.args)}."
            )

        # if it's a generic kernel, infer the required overload from the arguments
        if kernel.is_generic:
            fwd_types = kernel.infer_argument_types(fwd_args)
            kernel = kernel.get_overload(fwd_types)

        # delay load modules, including new overload if needed
        module = kernel.module
        if not module.load(device):
            return

        # late bind
        hooks = module.get_kernel_hooks(kernel, device)

        pack_args(fwd_args, params)
        pack_args(adj_args, params, adjoint=True)

        # run kernel
        if device.is_cpu:
            if adjoint:
                if hooks.backward is None:
                    raise RuntimeError(
                        f"Failed to find backward kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
                    )

                hooks.backward(*params)

            else:
                if hooks.forward is None:
                    raise RuntimeError(
                        f"Failed to find forward kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
                    )

                if record_cmd:
                    launch = Launch(
                        kernel=kernel, hooks=hooks, params=params, params_addr=None, bounds=bounds, device=device
                    )
                    return launch
                else:
                    hooks.forward(*params)

        else:
            kernel_args = [ctypes.c_void_p(ctypes.addressof(x)) for x in params]
            kernel_params = (ctypes.c_void_p * len(kernel_args))(*kernel_args)

            with warp.ScopedStream(stream):
                if adjoint:
                    if hooks.backward is None:
                        raise RuntimeError(
                            f"Failed to find backward kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
                        )

                    runtime.core.cuda_launch_kernel(
                        device.context, hooks.backward, bounds.size, max_blocks, kernel_params
                    )

                else:
                    if hooks.forward is None:
                        raise RuntimeError(
                            f"Failed to find forward kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
                        )

                    if record_cmd:
                        launch = Launch(
                            kernel=kernel,
                            hooks=hooks,
                            params=params,
                            params_addr=kernel_params,
                            bounds=bounds,
                            device=device,
                        )
                        return launch

                    else:
                        # launch
                        runtime.core.cuda_launch_kernel(
                            device.context, hooks.forward, bounds.size, max_blocks, kernel_params
                        )

                try:
                    runtime.verify_cuda_device(device)
                except Exception as e:
                    print(f"Error launching kernel: {kernel.key} on device {device}")
                    raise e

    # record on tape if one is active
    if runtime.tape and record_tape:
        runtime.tape.record_launch(kernel, dim, max_blocks, inputs, outputs, device)


def synchronize():
    """Manually synchronize the calling CPU thread with any outstanding CUDA work on all devices

    This method allows the host application code to ensure that any kernel launches
    or memory copies have completed.
    """

    if is_cuda_driver_initialized():
        # save the original context to avoid side effects
        saved_context = runtime.core.cuda_context_get_current()

        # TODO: only synchronize devices that have outstanding work
        for device in runtime.cuda_devices:
            # avoid creating primary context if the device has not been used yet
            if device.has_context:
                if device.is_capturing:
                    raise RuntimeError(f"Cannot synchronize device {device} while graph capture is active")

                runtime.core.cuda_context_synchronize(device.context)

        # restore the original context to avoid side effects
        runtime.core.cuda_context_set_current(saved_context)


def synchronize_device(device: Devicelike = None):
    """Manually synchronize the calling CPU thread with any outstanding CUDA work on the specified device

    This method allows the host application code to ensure that any kernel launches
    or memory copies have completed.

    Args:
        device: Device to synchronize.  If None, synchronize the current CUDA device.
    """

    device = runtime.get_device(device)
    if device.is_cuda:
        if device.is_capturing:
            raise RuntimeError(f"Cannot synchronize device {device} while graph capture is active")

        runtime.core.cuda_context_synchronize(device.context)


def synchronize_stream(stream_or_device=None):
    """Manually synchronize the calling CPU thread with any outstanding CUDA work on the specified stream.

    Args:
        stream_or_device: `wp.Stream` or a device.  If the argument is a device, synchronize the device's current stream.
    """

    if isinstance(stream_or_device, Stream):
        stream = stream_or_device
    else:
        stream = runtime.get_device(stream_or_device).stream

    runtime.core.cuda_stream_synchronize(stream.device.context, stream.cuda_stream)


def force_load(device: Union[Device, str, List[Device], List[str]] = None, modules: List[Module] = None):
    """Force user-defined kernels to be compiled and loaded

    Args:
        device: The device or list of devices to load the modules on.  If None, load on all devices.
        modules: List of modules to load.  If None, load all imported modules.
    """

    if is_cuda_driver_initialized():
        # save original context to avoid side effects
        saved_context = runtime.core.cuda_context_get_current()

    if device is None:
        devices = get_devices()
    elif isinstance(device, list):
        devices = [get_device(device_item) for device_item in device]
    else:
        devices = [get_device(device)]

    if modules is None:
        modules = user_modules.values()

    for d in devices:
        for m in modules:
            m.load(d)

    if is_cuda_available():
        # restore original context to avoid side effects
        runtime.core.cuda_context_set_current(saved_context)


def load_module(
    module: Union[Module, ModuleType, str] = None, device: Union[Device, str] = None, recursive: bool = False
):
    """Force user-defined module to be compiled and loaded

    Args:
        module: The module to load.  If None, load the current module.
        device: The device to load the modules on.  If None, load on all devices.
        recursive: Whether to load submodules.  E.g., if the given module is `warp.sim`, this will also load `warp.sim.model`, `warp.sim.articulation`, etc.

    Note: A module must be imported before it can be loaded by this function.
    """

    if module is None:
        # if module not specified, use the module that called us
        module = inspect.getmodule(inspect.stack()[1][0])
        module_name = module.__name__
    elif isinstance(module, Module):
        module_name = module.name
    elif isinstance(module, ModuleType):
        module_name = module.__name__
    elif isinstance(module, str):
        module_name = module
    else:
        raise TypeError(f"Argument must be a module, got {type(module)}")

    modules = []

    # add the given module, if found
    m = user_modules.get(module_name)
    if m is not None:
        modules.append(m)

    # add submodules, if recursive
    if recursive:
        prefix = module_name + "."
        for name, mod in user_modules.items():
            if name.startswith(prefix):
                modules.append(mod)

    force_load(device=device, modules=modules)


def set_module_options(options: Dict[str, Any], module: Optional[Any] = None):
    """Set options for the current module.

    Options can be used to control runtime compilation and code-generation
    for the current module individually. Available options are listed below.

    * **mode**: The compilation mode to use, can be "debug", or "release", defaults to the value of ``warp.config.mode``.
    * **max_unroll**: The maximum fixed-size loop to unroll (default 16)

    Args:

        options: Set of key-value option pairs
    """

    if module is None:
        m = inspect.getmodule(inspect.stack()[1][0])
    else:
        m = module

    get_module(m.__name__).options.update(options)
    get_module(m.__name__).unload()


def get_module_options(module: Optional[Any] = None) -> Dict[str, Any]:
    """Returns a list of options for the current module."""
    if module is None:
        m = inspect.getmodule(inspect.stack()[1][0])
    else:
        m = module

    return get_module(m.__name__).options


def capture_begin(device: Devicelike = None, stream=None, force_module_load=None):
    """Begin capture of a CUDA graph

    Captures all subsequent kernel launches and memory operations on CUDA devices.
    This can be used to record large numbers of kernels and replay them with low-overhead.

    Args:

        device: The device to capture on, if None the current CUDA device will be used
        stream: The CUDA stream to capture on
        force_module_load: Whether or not to force loading of all kernels before capture, in general it is better to use :func:`~warp.load_module()` to selectively load kernels.

    """

    if force_module_load is None:
        force_module_load = warp.config.graph_capture_module_load_default

    if warp.config.verify_cuda:
        raise RuntimeError("Cannot use CUDA error verification during graph capture")

    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("Must be a CUDA device")

    if force_module_load:
        force_load(device)

    device.is_capturing = True

    # disable garbage collection to avoid older allocations getting collected during graph capture
    gc.disable()

    with warp.ScopedStream(stream):
        runtime.core.cuda_graph_begin_capture(device.context)


def capture_end(device: Devicelike = None, stream=None) -> Graph:
    """Ends the capture of a CUDA graph

    Returns:
        A handle to a CUDA graph object that can be launched with :func:`~warp.capture_launch()`
    """

    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("Must be a CUDA device")

    with warp.ScopedStream(stream):
        graph = runtime.core.cuda_graph_end_capture(device.context)

    device.is_capturing = False

    # re-enable GC
    gc.enable()

    if graph is None:
        raise RuntimeError(
            "Error occurred during CUDA graph capture. This could be due to an unintended allocation or CPU/GPU synchronization event."
        )
    else:
        return Graph(device, graph)


def capture_launch(graph: Graph, stream: Stream = None):
    """Launch a previously captured CUDA graph

    Args:
        graph: A Graph as returned by :func:`~warp.capture_end()`
        stream: A Stream to launch the graph on (optional)
    """

    if stream is not None:
        if stream.device != graph.device:
            raise RuntimeError(f"Cannot launch graph from device {graph.device} on stream from device {stream.device}")
        device = stream.device
    else:
        device = graph.device

    with warp.ScopedStream(stream):
        runtime.core.cuda_graph_launch(device.context, graph.exec)


def copy(
    dest: warp.array, src: warp.array, dest_offset: int = 0, src_offset: int = 0, count: int = 0, stream: Stream = None
):
    """Copy array contents from src to dest

    Args:
        dest: Destination array, must be at least as big as source buffer
        src: Source array
        dest_offset: Element offset in the destination array
        src_offset: Element offset in the source array
        count: Number of array elements to copy (will copy all elements if set to 0)
        stream: The stream on which to perform the copy (optional)

    """

    if not warp.types.is_array(src) or not warp.types.is_array(dest):
        raise RuntimeError("Copy source and destination must be arrays")

    # backwards compatibility, if count is zero then copy entire src array
    if count <= 0:
        count = src.size

    if count == 0:
        return

    # copying non-contiguous arrays requires that they are on the same device
    if not (src.is_contiguous and dest.is_contiguous) and src.device != dest.device:
        if dest.is_contiguous:
            # make a contiguous copy of the source array
            src = src.contiguous()
        else:
            # make a copy of the source array on the destination device
            src = src.to(dest.device)

    if src.is_contiguous and dest.is_contiguous:
        bytes_to_copy = count * warp.types.type_size_in_bytes(src.dtype)

        src_size_in_bytes = src.size * warp.types.type_size_in_bytes(src.dtype)
        dst_size_in_bytes = dest.size * warp.types.type_size_in_bytes(dest.dtype)

        src_offset_in_bytes = src_offset * warp.types.type_size_in_bytes(src.dtype)
        dst_offset_in_bytes = dest_offset * warp.types.type_size_in_bytes(dest.dtype)

        src_ptr = src.ptr + src_offset_in_bytes
        dst_ptr = dest.ptr + dst_offset_in_bytes

        if src_offset_in_bytes + bytes_to_copy > src_size_in_bytes:
            raise RuntimeError(
                f"Trying to copy source buffer with size ({bytes_to_copy}) from offset ({src_offset_in_bytes}) is larger than source size ({src_size_in_bytes})"
            )

        if dst_offset_in_bytes + bytes_to_copy > dst_size_in_bytes:
            raise RuntimeError(
                f"Trying to copy source buffer with size ({bytes_to_copy}) to offset ({dst_offset_in_bytes}) is larger than destination size ({dst_size_in_bytes})"
            )

        if src.device.is_cpu and dest.device.is_cpu:
            runtime.core.memcpy_h2h(dst_ptr, src_ptr, bytes_to_copy)
        else:
            # figure out the CUDA context/stream for the copy
            if stream is not None:
                copy_device = stream.device
            elif dest.device.is_cuda:
                copy_device = dest.device
            else:
                copy_device = src.device

            with warp.ScopedStream(stream):
                if src.device.is_cpu and dest.device.is_cuda:
                    runtime.core.memcpy_h2d(copy_device.context, dst_ptr, src_ptr, bytes_to_copy)
                elif src.device.is_cuda and dest.device.is_cpu:
                    runtime.core.memcpy_d2h(copy_device.context, dst_ptr, src_ptr, bytes_to_copy)
                elif src.device.is_cuda and dest.device.is_cuda:
                    if src.device == dest.device:
                        runtime.core.memcpy_d2d(copy_device.context, dst_ptr, src_ptr, bytes_to_copy)
                    else:
                        runtime.core.memcpy_peer(copy_device.context, dst_ptr, src_ptr, bytes_to_copy)
                else:
                    raise RuntimeError("Unexpected source and destination combination")

    else:
        # handle non-contiguous and indexed arrays

        if src.shape != dest.shape:
            raise RuntimeError("Incompatible array shapes")

        src_elem_size = warp.types.type_size_in_bytes(src.dtype)
        dst_elem_size = warp.types.type_size_in_bytes(dest.dtype)

        if src_elem_size != dst_elem_size:
            raise RuntimeError("Incompatible array data types")

        # can't copy to/from fabric arrays of arrays, because they are jagged arrays of arbitrary lengths
        # TODO?
        if (
            isinstance(src, (warp.fabricarray, warp.indexedfabricarray))
            and src.ndim > 1
            or isinstance(dest, (warp.fabricarray, warp.indexedfabricarray))
            and dest.ndim > 1
        ):
            raise RuntimeError("Copying to/from Fabric arrays of arrays is not supported")

        src_desc = src.__ctype__()
        dst_desc = dest.__ctype__()
        src_ptr = ctypes.pointer(src_desc)
        dst_ptr = ctypes.pointer(dst_desc)
        src_type = warp.types.array_type_id(src)
        dst_type = warp.types.array_type_id(dest)

        if src.device.is_cuda:
            with warp.ScopedStream(stream):
                runtime.core.array_copy_device(src.device.context, dst_ptr, src_ptr, dst_type, src_type, src_elem_size)
        else:
            runtime.core.array_copy_host(dst_ptr, src_ptr, dst_type, src_type, src_elem_size)

    # copy gradient, if needed
    if hasattr(src, "grad") and src.grad is not None and hasattr(dest, "grad") and dest.grad is not None:
        copy(dest.grad, src.grad, stream=stream)


def type_str(t):
    if t is None:
        return "None"
    elif t == Any:
        return "Any"
    elif t == Callable:
        return "Callable"
    elif t == Tuple[int, int]:
        return "Tuple[int, int]"
    elif isinstance(t, int):
        return str(t)
    elif isinstance(t, List):
        return "Tuple[" + ", ".join(map(type_str, t)) + "]"
    elif isinstance(t, warp.array):
        return f"Array[{type_str(t.dtype)}]"
    elif isinstance(t, warp.indexedarray):
        return f"IndexedArray[{type_str(t.dtype)}]"
    elif isinstance(t, warp.fabricarray):
        return f"FabricArray[{type_str(t.dtype)}]"
    elif isinstance(t, warp.indexedfabricarray):
        return f"IndexedFabricArray[{type_str(t.dtype)}]"
    elif hasattr(t, "_wp_generic_type_str_"):
        generic_type = t._wp_generic_type_str_

        # for concrete vec/mat types use the short name
        if t in warp.types.vector_types:
            return t.__name__

        # for generic vector / matrix type use a Generic type hint
        if generic_type == "vec_t":
            # return f"Vector"
            return f"Vector[{type_str(t._wp_type_params_[0])},{type_str(t._wp_scalar_type_)}]"
        elif generic_type == "quat_t":
            # return f"Quaternion"
            return f"Quaternion[{type_str(t._wp_scalar_type_)}]"
        elif generic_type == "mat_t":
            # return f"Matrix"
            return f"Matrix[{type_str(t._wp_type_params_[0])},{type_str(t._wp_type_params_[1])},{type_str(t._wp_scalar_type_)}]"
        elif generic_type == "transform_t":
            # return f"Transformation"
            return f"Transformation[{type_str(t._wp_scalar_type_)}]"
        else:
            raise TypeError("Invalid vector or matrix dimensions")
    else:
        return t.__name__


def print_function(f, file, noentry=False):  # pragma: no cover
    """Writes a function definition to a file for use in reST documentation

    Args:
        f: The function being written
        file: The file object for output
        noentry: If True, then the :noindex: and :nocontentsentry: directive
          options will be added

    Returns:
        A bool indicating True if f was written to file
    """

    if f.hidden:
        return False

    args = ", ".join(f"{k}: {type_str(v)}" for k, v in f.input_types.items())

    return_type = ""

    try:
        # todo: construct a default value for each of the functions args
        # so we can generate the return type for overloaded functions
        return_type = " -> " + type_str(f.value_func(None, None, None))
    except Exception:
        pass

    print(f".. function:: {f.key}({args}){return_type}", file=file)
    if noentry:
        print("   :noindex:", file=file)
        print("   :nocontentsentry:", file=file)
    print("", file=file)

    if f.doc != "":
        if not f.missing_grad:
            print(f"   {f.doc}", file=file)
        else:
            print(f"   {f.doc} [1]_", file=file)
        print("", file=file)

    print(file=file)

    return True


def export_functions_rst(file):  # pragma: no cover
    header = (
        "..\n"
        "   Autogenerated File - Do not edit. Run build_docs.py to generate.\n"
        "\n"
        ".. functions:\n"
        ".. currentmodule:: warp\n"
        "\n"
        "Kernel Reference\n"
        "================"
    )

    print(header, file=file)

    # type definitions of all functions by group
    print("\nScalar Types", file=file)
    print("------------", file=file)

    for t in warp.types.scalar_types:
        print(f".. class:: {t.__name__}", file=file)
    # Manually add wp.bool since it's inconvenient to add to wp.types.scalar_types:
    print(f".. class:: {warp.types.bool.__name__}", file=file)

    print("\n\nVector Types", file=file)
    print("------------", file=file)

    for t in warp.types.vector_types:
        print(f".. class:: {t.__name__}", file=file)

    print("\nGeneric Types", file=file)
    print("-------------", file=file)

    print(".. class:: Int", file=file)
    print(".. class:: Float", file=file)
    print(".. class:: Scalar", file=file)
    print(".. class:: Vector", file=file)
    print(".. class:: Matrix", file=file)
    print(".. class:: Quaternion", file=file)
    print(".. class:: Transformation", file=file)
    print(".. class:: Array", file=file)

    print("\nQuery Types", file=file)
    print("-------------", file=file)
    print(".. autoclass:: bvh_query_t", file=file)
    print(".. autoclass:: hash_grid_query_t", file=file)
    print(".. autoclass:: mesh_query_aabb_t", file=file)
    print(".. autoclass:: mesh_query_point_t", file=file)
    print(".. autoclass:: mesh_query_ray_t", file=file)

    # build dictionary of all functions by group
    groups = {}

    for k, f in builtin_functions.items():
        # build dict of groups
        if f.group not in groups:
            groups[f.group] = []

        # append all overloads to the group
        for o in f.overloads:
            groups[f.group].append(o)

    # Keep track of what function names have been written
    written_functions = {}

    for k, g in groups.items():
        print("\n", file=file)
        print(k, file=file)
        print("---------------", file=file)

        for f in g:
            if f.key in written_functions:
                # Add :noindex: + :nocontentsentry: since Sphinx gets confused
                print_function(f, file=file, noentry=True)
            else:
                if print_function(f, file=file):
                    written_functions[f.key] = []

    # footnotes
    print(".. rubric:: Footnotes", file=file)
    print(".. [1] Note: function gradients not implemented for backpropagation.", file=file)


def export_stubs(file):  # pragma: no cover
    """Generates stub file for auto-complete of builtin functions"""

    import textwrap

    print(
        "# Autogenerated file, do not edit, this file provides stubs for builtins autocomplete in VSCode, PyCharm, etc",
        file=file,
    )
    print("", file=file)
    print("from typing import Any", file=file)
    print("from typing import Tuple", file=file)
    print("from typing import Callable", file=file)
    print("from typing import TypeVar", file=file)
    print("from typing import Generic", file=file)
    print("from typing import overload as over", file=file)
    print(file=file)

    # type hints, these need to be mirrored into the stubs file
    print('Length = TypeVar("Length", bound=int)', file=file)
    print('Rows = TypeVar("Rows", bound=int)', file=file)
    print('Cols = TypeVar("Cols", bound=int)', file=file)
    print('DType = TypeVar("DType")', file=file)

    print('Int = TypeVar("Int")', file=file)
    print('Float = TypeVar("Float")', file=file)
    print('Scalar = TypeVar("Scalar")', file=file)
    print("Vector = Generic[Length, Scalar]", file=file)
    print("Matrix = Generic[Rows, Cols, Scalar]", file=file)
    print("Quaternion = Generic[Float]", file=file)
    print("Transformation = Generic[Float]", file=file)
    print("Array = Generic[DType]", file=file)
    print("FabricArray = Generic[DType]", file=file)
    print("IndexedFabricArray = Generic[DType]", file=file)

    # prepend __init__.py
    with open(os.path.join(os.path.dirname(file.name), "__init__.py")) as header_file:
        # strip comment lines
        lines = [line for line in header_file if not line.startswith("#")]
        header = "".join(lines)

    print(header, file=file)
    print(file=file)

    for k, g in builtin_functions.items():
        for f in g.overloads:
            args = ", ".join(f"{k}: {type_str(v)}" for k, v in f.input_types.items())

            return_str = ""

            if not f.export or f.hidden:  # or f.generic:
                continue

            try:
                # todo: construct a default value for each of the functions args
                # so we can generate the return type for overloaded functions
                return_type = f.value_func(None, None, None)
                if return_type:
                    return_str = " -> " + type_str(return_type)

            except Exception:
                pass

            print("@over", file=file)
            print(f"def {f.key}({args}){return_str}:", file=file)
            print('    """', file=file)
            print(textwrap.indent(text=f.doc, prefix="    "), file=file)
            print('    """', file=file)
            print("    ...\n\n", file=file)


def export_builtins(file: io.TextIOBase):  # pragma: no cover
    def ctype_arg_str(t):
        if isinstance(t, int):
            return "int"
        elif isinstance(t, float):
            return "float"
        elif t in warp.types.vector_types:
            return f"{t.__name__}&"
        else:
            return t.__name__

    def ctype_ret_str(t):
        if isinstance(t, int):
            return "int"
        elif isinstance(t, float):
            return "float"
        else:
            return t.__name__

    file.write("namespace wp {\n\n")
    file.write('extern "C" {\n\n')

    for k, g in builtin_functions.items():
        for f in g.overloads:
            if not f.export or f.generic:
                continue

            simple = True
            for k, v in f.input_types.items():
                if isinstance(v, warp.array) or v == Any or v == Callable or v == Tuple:
                    simple = False
                    break

            # only export simple types that don't use arrays
            # or templated types
            if not simple or f.variadic:
                continue

            args = ", ".join(f"{ctype_arg_str(v)} {k}" for k, v in f.input_types.items())
            params = ", ".join(f.input_types.keys())

            return_type = ""

            try:
                # todo: construct a default value for each of the functions args
                # so we can generate the return type for overloaded functions
                return_type = ctype_ret_str(f.value_func(None, None, None))
            except Exception:
                continue

            if return_type.startswith("Tuple"):
                continue

            if args == "":
                file.write(f"WP_API void {f.mangled_name}({return_type}* ret) {{ *ret = wp::{f.key}({params}); }}\n")
            elif return_type == "None":
                file.write(f"WP_API void {f.mangled_name}({args}) {{ wp::{f.key}({params}); }}\n")
            else:
                file.write(
                    f"WP_API void {f.mangled_name}({args}, {return_type}* ret) {{ *ret = wp::{f.key}({params}); }}\n"
                )

    file.write('\n}  // extern "C"\n\n')
    file.write("}  // namespace wp\n")


# initialize global runtime
runtime = None


def init():
    """Initialize the Warp runtime. This function must be called before any other API call. If an error occurs an exception will be raised."""
    global runtime

    if runtime is None:
        runtime = Runtime()
