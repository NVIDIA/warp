# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import ast
import ctypes
import functools
import hashlib
import inspect
import io
import itertools
import json
import operator
import os
import platform
import sys
import types
import typing
import weakref
from copy import copy as shallowcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import numpy as np

import warp
import warp.build
import warp.codegen
import warp.config
from warp.types import Array, launch_bounds_t

# represents either a built-in or user-defined function


def create_value_func(type):
    def value_func(arg_types, arg_values):
        hint_origin = getattr(type, "__origin__", None)
        if hint_origin is not None and issubclass(hint_origin, typing.Tuple):
            return type.__args__

        return type

    return value_func


def get_function_args(func):
    """Ensures that all function arguments are annotated and returns a dictionary mapping from argument name to its type."""
    argspec = warp.codegen.get_full_arg_spec(func)

    # use source-level argument annotations
    if len(argspec.annotations) < len(argspec.args):
        raise RuntimeError(f"Incomplete argument annotations on function {func.__qualname__}")
    return argspec.annotations


complex_type_hints = (Any, Callable, Tuple)
sequence_types = (list, tuple)

function_key_counts: dict[str, int] = {}


def generate_unique_function_identifier(key: str) -> str:
    # Generate unique identifiers for user-defined functions in native code.
    # - Prevents conflicts when a function is redefined and old versions are still in use.
    # - Prevents conflicts between multiple closures returned from the same function.
    # - Prevents conflicts between identically named functions from different modules.
    #
    # Currently, we generate a unique id when a new Function is created, which produces
    # globally unique identifiers.
    #
    # NOTE:
    #   We could move this to the Module class for generating unique identifiers at module scope,
    #   but then we need another solution for preventing conflicts across modules (e.g., different namespaces).
    #   That would requires more Python code, generate more native code, and would be slightly slower
    #   with no clear advantages over globally-unique identifiers (non-global shared state is still shared state).
    #
    # TODO:
    #   Kernels and structs use unique identifiers based on their hash.  Using hash-based identifiers
    #   for functions would allow filtering out duplicate identical functions during codegen,
    #   like we do with kernels and structs.  This is worth investigating further, but might require
    #   additional refactoring.  For example, the code that deals with custom gradient and replay functions
    #   requires matching function names, but these special functions get created before the hash
    #   for the parent function can be computed.  In addition to these complications, computing hashes
    #   for all function instances would increase the cost of module hashing when generic functions
    #   are involved (currently we only hash the generic templates, which is sufficient).

    unique_id = function_key_counts.get(key, 0)
    function_key_counts[key] = unique_id + 1
    return f"{key}_{unique_id}"


class Function:
    def __init__(
        self,
        func: Callable | None,
        key: str,
        namespace: str,
        input_types: dict[str, type | TypeVar] | None = None,
        value_type: type | None = None,
        value_func: Callable[[Mapping[str, type], Mapping[str, Any]], type] | None = None,
        export_func: Callable[[dict[str, type]], dict[str, type]] | None = None,
        dispatch_func: Callable | None = None,
        lto_dispatch_func: Callable | None = None,
        module: Module | None = None,
        variadic: bool = False,
        initializer_list_func: Callable[[dict[str, Any], type], bool] | None = None,
        export: bool = False,
        doc: str = "",
        group: str = "",
        hidden: bool = False,
        skip_replay: bool = False,
        missing_grad: bool = False,
        generic: bool = False,
        native_func: str | None = None,
        defaults: dict[str, Any] | None = None,
        custom_replay_func: Function | None = None,
        native_snippet: str | None = None,
        adj_native_snippet: str | None = None,
        replay_snippet: str | None = None,
        skip_forward_codegen: bool = False,
        skip_reverse_codegen: bool = False,
        custom_reverse_num_input_args: int = -1,
        custom_reverse_mode: bool = False,
        overloaded_annotations: dict[str, type] | None = None,
        code_transformers: list[ast.NodeTransformer] | None = None,
        skip_adding_overload: bool = False,
        require_original_output_arg: bool = False,
        scope_locals: dict[str, Any] | None = None,
    ):
        if code_transformers is None:
            code_transformers = []

        self.func = func  # points to Python function decorated with @wp.func, may be None for builtins
        self.key = key
        self.namespace = namespace
        self.value_type = value_type
        self.value_func = value_func  # a function that takes a list of args and a list of templates and returns the value type, e.g.: load(array, index) returns the type of value being loaded
        self.export_func = export_func
        self.dispatch_func = dispatch_func
        self.lto_dispatch_func = lto_dispatch_func
        self.input_types = {}
        self.export = export
        self.doc = doc
        self.group = group
        self.module = module
        self.variadic = variadic  # function can take arbitrary number of inputs, e.g.: printf()
        self.defaults = {} if defaults is None else defaults
        # Function instance for a custom implementation of the replay pass
        self.custom_replay_func = custom_replay_func
        self.native_snippet = native_snippet
        self.adj_native_snippet = adj_native_snippet
        self.replay_snippet = replay_snippet
        self.custom_grad_func: Function | None = None
        self.require_original_output_arg = require_original_output_arg
        self.generic_parent = None  # generic function that was used to instantiate this overload

        if initializer_list_func is None:
            self.initializer_list_func = lambda x, y: False
        else:
            self.initializer_list_func = (
                initializer_list_func  # True if the arguments should be emitted as an initializer list in the c++ code
            )
        self.hidden = hidden  # function will not be listed in docs
        self.skip_replay = (
            skip_replay  # whether operation will be performed during the forward replay in the backward pass
        )
        self.missing_grad = missing_grad  # whether builtin is missing a corresponding adjoint
        self.generic = generic
        self.mangled_name: str | None = None

        # allow registering functions with a different name in Python and native code
        if native_func is None:
            if func is None:
                # builtin function
                self.native_func = key
            else:
                # user functions need unique identifiers to avoid conflicts
                self.native_func = generate_unique_function_identifier(key)
        else:
            self.native_func = native_func

        if func:
            # user-defined function

            # generic and concrete overload lookups by type signature
            self.user_templates: dict[str, Function] = {}
            self.user_overloads: dict[str, Function] = {}

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

            # Record any default parameter values.
            if not self.defaults:
                signature = inspect.signature(func)
                self.defaults = {k: v.default for k, v in signature.parameters.items() if v.default is not v.empty}

        else:
            # builtin function

            # embedded linked list of all overloads
            # the builtin_functions dictionary holds the list head for a given key (func name)
            self.overloads: list[Function] = []

            # builtin (native) function, canonicalize argument types
            if input_types is not None:
                for k, v in input_types.items():
                    self.input_types[k] = warp.types.type_to_warp(v)

            # cache mangled name
            if self.export and self.is_simple():
                self.mangled_name = self.mangle()

        if not skip_adding_overload:
            self.add_overload(self)

        # Store a description of the function's signature that can be used
        # to resolve a bunch of positional/keyword/variadic arguments against,
        # in a way that is compatible with Python's semantics.
        signature_params = []
        signature_default_param_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        for raw_param_name in self.input_types.keys():
            if raw_param_name.startswith("**"):
                param_name = raw_param_name[2:]
                param_kind = inspect.Parameter.VAR_KEYWORD
            elif raw_param_name.startswith("*"):
                param_name = raw_param_name[1:]
                param_kind = inspect.Parameter.VAR_POSITIONAL

                # Once a variadic argument like `*args` is found, any following
                # arguments need to be passed using keywords.
                signature_default_param_kind = inspect.Parameter.KEYWORD_ONLY
            else:
                param_name = raw_param_name
                param_kind = signature_default_param_kind

            param = inspect.Parameter(
                param_name, param_kind, default=self.defaults.get(param_name, inspect.Parameter.empty)
            )
            signature_params.append(param)
        self.signature = inspect.Signature(signature_params)

        # scope for resolving overloads, the locals() where the function is defined
        if scope_locals is None:
            scope_locals = inspect.currentframe().f_back.f_locals

        # add to current module
        if module:
            module.register_function(self, scope_locals, skip_adding_overload)

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

                try:
                    # Try to bind the given arguments to the function's signature.
                    # This is not checking whether the argument types are matching,
                    # rather it's just assigning each argument to the corresponding
                    # function parameter.
                    bound_args = self.signature.bind(*args, **kwargs)
                except TypeError:
                    continue

                if self.defaults:
                    # Populate the bound arguments with any default values.
                    default_args = {k: v for k, v in self.defaults.items() if k not in bound_args.arguments}
                    warp.codegen.apply_defaults(bound_args, default_args)

                bound_args = tuple(bound_args.arguments.values())

                success, return_value = call_builtin(overload, bound_args)
                if success:
                    return return_value

            # overload resolution or call failed
            raise RuntimeError(
                f"Couldn't find a function '{self.key}' compatible with "
                f"the arguments '{', '.join(type(x).__name__ for x in args)}'"
            )

        if hasattr(self, "user_overloads") and len(self.user_overloads):
            # user-defined function with overloads
            bound_args = self.signature.bind(*args, **kwargs)
            if self.defaults:
                warp.codegen.apply_defaults(bound_args, self.defaults)

            arguments = tuple(bound_args.arguments.values())

            # try and find a matching overload
            for overload in self.user_overloads.values():
                if len(overload.input_types) != len(arguments):
                    continue
                template_types = list(overload.input_types.values())
                arg_names = list(overload.input_types.keys())
                try:
                    # attempt to unify argument types with function template types
                    warp.types.infer_argument_types(arguments, template_types, arg_names)
                    return overload.func(*arguments)
                except Exception:
                    continue

            raise RuntimeError(f"Error calling function '{self.key}', no overload found for arguments {args}")

        # user-defined function with no overloads
        if self.func is None:
            raise RuntimeError(f"Error calling function '{self.key}', function is undefined")

        # this function has no overloads, call it like a plain Python function
        return self.func(*args, **kwargs)

    def is_builtin(self) -> bool:
        return self.func is None

    def is_simple(self) -> bool:
        if self.variadic:
            return False

        # only export simple types that don't use arrays
        for v in self.input_types.values():
            if warp.types.is_array(v) or v in complex_type_hints:
                return False

        return True

    def mangle(self) -> str:
        """Build a mangled name for the C-exported function, e.g.: `builtin_normalize_vec3()`."""

        name = "builtin_" + self.key

        # Runtime arguments that are to be passed to the function, not its template signature.
        if self.export_func is not None:
            func_args = self.export_func(self.input_types)
        else:
            func_args = self.input_types

        types = []
        for t in func_args.values():
            types.append(t.__name__)

        return "_".join([name, *types])

    def add_overload(self, f: Function) -> None:
        if self.is_builtin():
            # todo: note that it is an error to add two functions
            # with the exact same signature as this would cause compile
            # errors during compile time. We should check here if there
            # is a previously created function with the same signature
            self.overloads.append(f)

            # make sure variadic overloads appear last so non variadic
            # ones are matched first:
            self.overloads.sort(key=operator.attrgetter("variadic"))

        else:
            # get function signature based on the input types
            sig = warp.types.get_signature(
                list(f.input_types.values()), func_name=f.key, arg_names=list(f.input_types.keys())
            )

            # check if generic
            if warp.types.is_generic_signature(sig):
                self.user_templates[sig] = f
            else:
                self.user_overloads[sig] = f

    def get_overload(self, arg_types: list[type], kwarg_types: Mapping[str, type]) -> Function | None:
        assert not self.is_builtin()

        for f in self.user_overloads.values():
            if warp.codegen.func_match_args(f, arg_types, kwarg_types):
                return f

        for f in self.user_templates.values():
            if not warp.codegen.func_match_args(f, arg_types, kwarg_types):
                continue

            acceptable_arg_num = len(f.input_types) - len(f.defaults) <= len(arg_types) <= len(f.input_types)
            if not acceptable_arg_num:
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
                # add defaults
                for k, d in f.defaults.items():
                    if k not in overload_annotations:
                        overload_annotations[k] = warp.codegen.strip_reference(warp.codegen.get_arg_type(d))

                ovl = shallowcopy(f)
                ovl.adj = warp.codegen.Adjoint(f.func, overload_annotations)
                ovl.input_types = overload_annotations
                ovl.value_func = None
                ovl.generic_parent = f

                sig = warp.types.get_signature(arg_types, func_name=self.key)
                self.user_overloads[sig] = ovl

                return ovl

        # failed  to find overload
        return None

    def __repr__(self):
        inputs_str = ", ".join([f"{k}: {warp.types.type_repr(v)}" for k, v in self.input_types.items()])
        return f"<Function {self.key}({inputs_str})>"


def get_builtin_type(return_type: type) -> type:
    # The return_type might just be vector_t(length=3,dtype=wp.float32), so we've got to match that
    # in the list of hard coded types so it knows it's returning one of them:
    if hasattr(return_type, "_wp_generic_type_hint_"):
        return_type_match = tuple(
            x
            for x in generic_vtypes
            if x._wp_generic_type_hint_ == return_type._wp_generic_type_hint_
            and x._wp_type_params_ == return_type._wp_type_params_
        )
        if not return_type_match:
            raise RuntimeError("No match")

        return return_type_match[0]

    return return_type


def extract_return_value(value_type: type, value_ctype: type, ret: Any) -> Any:
    if issubclass(value_ctype, ctypes.Array) or issubclass(value_ctype, ctypes.Structure):
        # return vector types as ctypes
        return ret

    if value_type is warp.types.float16:
        return warp.types.half_bits_to_float(ret.value)

    return ret.value


def call_builtin(func: Function, params: tuple) -> tuple[bool, Any]:
    uses_non_warp_array_type = False

    init()

    # Retrieve the built-in function from Warp's dll.
    c_func = getattr(warp.context.runtime.core, func.mangled_name)

    # Runtime arguments that are to be passed to the function, not its template signature.
    if func.export_func is not None:
        func_args = func.export_func(func.input_types)
    else:
        func_args = func.input_types

    value_type = func.value_func(None, None)

    # Try gathering the parameters that the function expects and pack them
    # into their corresponding C types.
    c_params = []
    for i, (_, arg_type) in enumerate(func_args.items()):
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

                # Retrieve the element type of the sequence while ensuring that it's homogeneous.
                elem_type = type(arr[0])
                for array_index in range(1, elem_count):
                    if type(arr[array_index]) is not elem_type:
                        raise ValueError("All array elements must share the same type.")

                expected_elem_type = arg_type._wp_scalar_type_
                if not (
                    elem_type is expected_elem_type
                    or (elem_type is float and expected_elem_type is warp.types.float32)
                    or (elem_type is int and expected_elem_type is warp.types.int32)
                    or (elem_type is bool and expected_elem_type is warp.types.bool)
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
                    for row_index in range(rows):
                        idx_start = row_index * cols
                        idx_end = idx_start + cols
                        c_param[row_index] = arr[idx_start:idx_end]
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
                or (type(param) is bool and arg_type is warp.types.bool)
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

    # Retrieve the return type.
    value_type = func.value_func(None, None)

    if value_type is not None:
        if not isinstance(value_type, Sequence):
            value_type = (value_type,)

        value_ctype = tuple(warp.types.type_ctype(x) for x in value_type)
        ret = tuple(x() for x in value_ctype)
        ret_addr = tuple(ctypes.c_void_p(ctypes.addressof(x)) for x in ret)

        c_params.extend(ret_addr)

    # Call the built-in function from Warp's dll.
    c_func(*c_params)

    if uses_non_warp_array_type:
        warp.utils.warn(
            "Support for built-in functions called with non-Warp array types, "
            "such as lists, tuples, NumPy arrays, and others, will be dropped "
            "in the future. Use a Warp type such as `wp.vec`, `wp.mat`, "
            "`wp.quat`, or `wp.transform`.",
            DeprecationWarning,
            stacklevel=3,
        )

    if value_type is None:
        return (True, None)

    return_value = tuple(extract_return_value(x, y, z) for x, y, z in zip(value_type, value_ctype, ret))
    if len(return_value) == 1:
        return_value = return_value[0]

    return (True, return_value)


class KernelHooks:
    def __init__(self, forward, backward, forward_smem_bytes=0, backward_smem_bytes=0):
        self.forward = forward
        self.backward = backward

        self.forward_smem_bytes = forward_smem_bytes
        self.backward_smem_bytes = backward_smem_bytes


# caches source and compiled entry points for a kernel (will be populated after module loads)
class Kernel:
    def __init__(self, func, key=None, module=None, options=None, code_transformers=None):
        self.func = func

        if module is None:
            self.module = get_module(func.__module__)
        else:
            self.module = module

        if key is None:
            self.key = warp.codegen.make_full_qualified_name(func)
        else:
            self.key = key

        self.options = {} if options is None else options

        if code_transformers is None:
            code_transformers = []

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

        # generic kernel that was used to instantiate this overload
        self.generic_parent = None

        # argument indices by name
        self.arg_indices = {a.label: i for i, a in enumerate(self.adj.args)}

        # hash will be computed when the module is built
        self.hash = None

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

        # get a type signature from the given argument types
        sig = warp.types.get_signature(arg_types, func_name=self.key)
        ovl = self.overloads.get(sig)
        if ovl is not None:
            # return the existing overload matching the signature
            return ovl

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

        overload_annotations = dict(zip(arg_names, arg_types))

        # instantiate this kernel with the given argument types
        ovl = shallowcopy(self)
        ovl.adj = warp.codegen.Adjoint(self.func, overload_annotations)
        ovl.is_generic = False
        ovl.overloads = {}
        ovl.sig = sig
        ovl.generic_parent = self

        self.overloads[sig] = ovl

        self.module.mark_modified()

        return ovl

    def get_overload(self, arg_types):
        sig = warp.types.get_signature(arg_types, func_name=self.key)
        return self.overloads.get(sig)

    def get_mangled_name(self):
        if self.hash is None:
            raise RuntimeError(f"Missing hash for kernel {self.key} in module {self.module.name}")

        # TODO: allow customizing the number of hash characters used
        hash_suffix = self.hash.hex()[:8]

        return f"{self.key}_{hash_suffix}"

    def __call__(self, *args, **kwargs):
        # we implement this function only to ensure Kernel is a callable object
        # so that we can document Warp kernels in the same way as Python functions
        # annotated by @wp.kernel (see functools.update_wrapper())
        raise NotImplementedError("Kernel.__call__() is not implemented, please use wp.launch() instead")


# ----------------------


# decorator to register function, @func
def func(f: Callable | None = None, *, name: str | None = None):
    def wrapper(f, *args, **kwargs):
        if name is None:
            key = warp.codegen.make_full_qualified_name(f)
        else:
            key = name

        scope_locals = inspect.currentframe().f_back.f_back.f_locals

        m = get_module(f.__module__)
        doc = getattr(f, "__doc__", "") or ""
        Function(
            func=f,
            key=key,
            namespace="",
            module=m,
            value_func=None,
            scope_locals=scope_locals,
            doc=doc.strip(),
        )  # value_type not known yet, will be inferred during Adjoint.build()

        # use the top of the list of overloads for this key
        g = m.functions[key]
        # copy over the function attributes, including docstring
        return functools.update_wrapper(g, f)

    if f is None:
        # Arguments were passed to the decorator.
        return wrapper

    return wrapper(f)


def func_native(snippet: str, adj_snippet: str | None = None, replay_snippet: str | None = None):
    """
    Decorator to register native code snippet, @func_native
    """

    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        scope_locals = {}
    else:
        scope_locals = frame.f_back.f_locals

    def snippet_func(f: Callable) -> Callable:
        name = warp.codegen.make_full_qualified_name(f)

        m = get_module(f.__module__)
        Function(
            func=f,
            key=name,
            namespace="",
            module=m,
            native_snippet=snippet,
            adj_native_snippet=adj_snippet,
            replay_snippet=replay_snippet,
            scope_locals=scope_locals,
        )  # value_type not known yet, will be inferred during Adjoint.build()
        g = m.functions[name]
        # copy over the function attributes, including docstring
        return functools.update_wrapper(g, f)

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
                # we have to temporarily build this function to figure out its return type(s);
                # note that we do not have a ModuleBuilder instance here at this wrapping stage, hence we
                # have to create a dummy builder
                builder = ModuleBuilder(Module("dummy", None), f.module.options)
                f.adj.build(builder)
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
                native_func=f.native_func,
                namespace=f.namespace,
                input_types=reverse_args,
                value_func=None,
                module=f.module,
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
                    return grad_fn
            raise RuntimeError(
                f"No function overload found for gradient function {grad_fn.__qualname__} for function {forward_fn.key}"
            )
        else:
            # resolve return variables
            forward_fn.adj.build(None, forward_fn.module.options)

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
                    f"\n{', '.join(f'{nt[0]}: {nt[1].__name__}' for nt in expected_args)}"
                )

        return grad_fn

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

        f = forward_fn.get_overload(arg_types, {})
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
            export_func=f.export_func,
            dispatch_func=f.dispatch_func,
            module=f.module,
            skip_reverse_codegen=True,
            skip_adding_overload=True,
            code_transformers=f.adj.transformers,
        )
        return replay_fn

    return wrapper


def kernel(
    f: Callable | None = None,
    *,
    enable_backward: bool | None = None,
    module: Module | Literal["unique"] | None = None,
):
    """
    Decorator to register a Warp kernel from a Python function.
    The function must be defined with type annotations for all arguments.
    The function must not return anything.

    Example::

        @wp.kernel
        def my_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
            tid = wp.tid()
            b[tid] = a[tid] + 1.0


        @wp.kernel(enable_backward=False)
        def my_kernel_no_backward(a: wp.array(dtype=float, ndim=2), x: float):
            # the backward pass will not be generated
            i, j = wp.tid()
            a[i, j] = x


        @wp.kernel(module="unique")
        def my_kernel_unique_module(a: wp.array(dtype=float), b: wp.array(dtype=float)):
            # the kernel will be registered in new unique module created just for this
            # kernel and its dependent functions and structs
            tid = wp.tid()
            b[tid] = a[tid] + 1.0

    Args:
        f: The function to be registered as a kernel.
        enable_backward: If False, the backward pass will not be generated.
        module: The :class:`warp.context.Module` to which the kernel belongs. Alternatively, if a string `"unique"` is provided, the kernel is assigned to a new module named after the kernel name and hash. If None, the module is inferred from the function's module.

    Returns:
        The registered kernel.
    """

    def wrapper(f, *args, **kwargs):
        options = {}

        if enable_backward is not None:
            options["enable_backward"] = enable_backward

        if module is None:
            m = get_module(f.__module__)
        elif module == "unique":
            m = Module(f.__name__, None)
        else:
            m = module
        k = Kernel(
            func=f,
            key=warp.codegen.make_full_qualified_name(f),
            module=m,
            options=options,
        )
        if module == "unique":
            # add the hash to the module name
            hasher = warp.context.ModuleHasher(m)
            k.module.name = f"{k.key}_{hasher.module_hash.hex()[:8]}"

        k = functools.update_wrapper(k, f)
        return k

    if f is None:
        # Arguments were passed to the decorator.
        return wrapper

    return wrapper(f)


# decorator to register struct, @struct
def struct(c: type):
    m = get_module(c.__module__)
    s = warp.codegen.Struct(key=warp.codegen.make_full_qualified_name(c), cls=c, module=m)
    s = functools.update_wrapper(s, c)
    return s


def overload(kernel, arg_types=Union[None, Dict[str, Any], List[Any]]):
    """Overload a generic kernel with the given argument types.

    Can be called directly or used as a function decorator.

    Args:
        kernel: The generic kernel to be instantiated with concrete types.
        arg_types: A list of concrete argument types for the kernel or a
            dictionary specifying generic argument names as keys and concrete
            types as variables.
    """
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
        kernel = module.find_kernel(fn)
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
        argspec = warp.codegen.get_full_arg_spec(fn)
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


# native functions that are part of the Warp API
builtin_functions: dict[str, Function] = {}


def get_generic_vtypes():
    # get a list of existing generic vector types (includes matrices and stuff)
    # so we can match arguments against them:
    generic_vtypes = tuple(x for x in warp.types.vector_types if hasattr(x, "_wp_generic_type_str_"))

    # deduplicate identical types:
    typedict = {(t._wp_generic_type_str_, str(t._wp_type_params_)): t for t in generic_vtypes}
    return tuple(typedict[k] for k in sorted(typedict.keys()))


generic_vtypes = get_generic_vtypes()


scalar_types = {}
scalar_types.update({x: x for x in warp.types.scalar_types})
scalar_types.update({x: x._wp_scalar_type_ for x in warp.types.vector_types})


def add_builtin(
    key: str,
    input_types: dict[str, type | TypeVar] | None = None,
    constraint: Callable[[Mapping[str, type]], bool] | None = None,
    value_type: type | None = None,
    value_func: Callable | None = None,
    export_func: Callable | None = None,
    dispatch_func: Callable | None = None,
    lto_dispatch_func: Callable | None = None,
    doc: str = "",
    namespace: str = "wp::",
    variadic: bool = False,
    initializer_list_func=None,
    export: bool = True,
    group: str = "Other",
    hidden: bool = False,
    skip_replay: bool = False,
    missing_grad: bool = False,
    native_func: str | None = None,
    defaults: dict[str, Any] | None = None,
    require_original_output_arg: bool = False,
):
    """Main entry point to register a new built-in function.

    Args:
        key: Function name. Multiple overloaded functions can be registered
            under the same name as long as their signature differ.
        input_types: Signature of the user-facing function.
            Variadic arguments are supported by prefixing the parameter names
            with asterisks as in `*args` and `**kwargs`. Generic arguments are
            supported with types such as `Any`, `Float`, `Scalar`, etc.
        constraint: For functions that define generic arguments and
            are to be exported, this callback is used to specify whether some
            combination of inferred arguments are valid or not.
        value_type: Type returned by the function.
        value_func: Callback used to specify the return type when
            `value_type` isn't enough.
        export_func: Callback used during the context stage to specify
            the signature of the underlying C++ function, not accounting for
            the template parameters.
            If not provided, `input_types` is used.
        dispatch_func: Callback used during the codegen stage to specify
            the runtime and template arguments to be passed to the underlying C++
            function. In other words, this allows defining a mapping between
            the signatures of the user-facing and the C++ functions, and even to
            dynamically create new arguments on the fly.
            The arguments returned must be of type `codegen.Var`.
            If not provided, all arguments passed by the users when calling
            the built-in are passed as-is as runtime arguments to the C++ function.
        lto_dispatch_func: Same as dispatch_func, but takes an 'option' dict
            as extra argument (indicating tile_size and target architecture) and returns
            an LTO-IR buffer as extra return value
        doc: Used to generate the Python's docstring and the HTML documentation.
        namespace: Namespace for the underlying C++ function.
        variadic: Whether the function declares variadic arguments.
        initializer_list_func: Callback to determine whether to use the
            initializer list syntax when passing the arguments to the underlying
            C++ function.
        export: Whether the function is to be exposed to the Python
            interpreter so that it becomes available from within the `warp`
            module.
        group: Classification used for the documentation.
        hidden: Whether to add that function into the documentation.
        skip_replay: Whether operation will be performed during
            the forward replay in the backward pass.
        missing_grad: Whether the function is missing a corresponding adjoint.
        native_func: Name of the underlying C++ function.
        defaults: Default values for the parameters defined in `input_types`.
        require_original_output_arg: Used during the codegen stage to
            specify whether an adjoint parameter corresponding to the return
            value should be included in the signature of the backward function.
    """
    if input_types is None:
        input_types = {}

    # wrap simple single-type functions with a value_func()
    if value_func is None:

        def value_func(arg_types, arg_values):
            return value_type

    if initializer_list_func is None:

        def initializer_list_func(args, return_type):
            return False

    if defaults is None:
        defaults = {}

    # Add specialized versions of this builtin if it's generic by matching arguments against
    # hard coded types. We do this so you can use hard coded warp types outside kernels:
    if export_func is not None:
        func_arg_types = export_func(input_types)
    else:
        func_arg_types = input_types

    generic = False
    for x in func_arg_types.values():
        if warp.types.type_is_generic(x):
            generic = True
            break

    if generic and export:
        # collect the parent type names of all the generic arguments:
        genericset = set()
        for t in func_arg_types.values():
            if hasattr(t, "_wp_generic_type_hint_"):
                genericset.add(t._wp_generic_type_hint_)
            elif warp.types.type_is_generic_scalar(t):
                genericset.add(t)

        # for each of those type names, get a list of all hard coded types derived
        # from them:
        gtypes = []
        for t in genericset:
            if t is warp.types.Float:
                value = warp.types.float_types
            elif t == warp.types.Scalar:
                value = warp.types.scalar_types
            elif t == warp.types.Int:
                value = warp.types.int_types
            else:
                value = tuple(x for x in generic_vtypes if x._wp_generic_type_hint_ == t)

            gtypes.append((t, value))

        # find the scalar data types supported by all the arguments by intersecting
        # sets:
        scalartypes = tuple({scalar_types[x] for x in v} for _, v in gtypes)
        if scalartypes:
            scalartypes = set.intersection(*scalartypes)
        scalartypes = sorted(scalartypes, key=str)

        # generate function calls for each of these scalar types:
        for stype in scalartypes:
            # find concrete types for this scalar type (eg if the scalar type is float32
            # this dict will look something like this:
            # {"vec":[wp.vec2,wp.vec3,wp.vec4], "mat":[wp.mat22,wp.mat33,wp.mat44]})
            consistenttypes = {k: tuple(x for x in v if scalar_types[x] == stype) for k, v in gtypes}

            # gotta try generating function calls for all combinations of these argument types
            # now.
            typelists = []
            for param in input_types.values():
                if warp.types.type_is_generic_scalar(param):
                    l = (stype,)
                elif hasattr(param, "_wp_generic_type_hint_"):
                    l = tuple(
                        x
                        for x in consistenttypes[param._wp_generic_type_hint_]
                        if warp.types.types_equal(param, x, match_generic=True)
                    )
                else:
                    l = (param,)

                typelists.append(l)

            for arg_types in itertools.product(*typelists):
                concrete_arg_types = dict(zip(input_types.keys(), arg_types))

                # Some of these argument lists won't work, eg if the function is mul(), we won't be
                # able to do a matrix vector multiplication for a mat22 and a vec3. The `constraint`
                # function determines which combinations are valid:
                if constraint:
                    if constraint(concrete_arg_types) is False:
                        continue

                return_type = value_func(concrete_arg_types, None)

                try:
                    if isinstance(return_type, Sequence):
                        return_type = tuple(get_builtin_type(x) for x in return_type)
                    else:
                        return_type = get_builtin_type(return_type)
                except RuntimeError:
                    continue

                # finally we can generate a function call for these concrete types:
                add_builtin(
                    key,
                    input_types=concrete_arg_types,
                    value_type=return_type,
                    value_func=value_func if return_type is Any else None,
                    export_func=export_func,
                    dispatch_func=dispatch_func,
                    lto_dispatch_func=lto_dispatch_func,
                    doc=doc,
                    namespace=namespace,
                    variadic=variadic,
                    initializer_list_func=initializer_list_func,
                    export=export,
                    group=group,
                    hidden=True,
                    skip_replay=skip_replay,
                    missing_grad=missing_grad,
                    defaults=defaults,
                    require_original_output_arg=require_original_output_arg,
                )

    func = Function(
        func=None,
        key=key,
        namespace=namespace,
        input_types=input_types,
        value_type=value_type,
        value_func=value_func,
        export_func=export_func,
        dispatch_func=dispatch_func,
        lto_dispatch_func=lto_dispatch_func,
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


def register_api_function(
    function: Function,
    group: str = "Other",
    hidden: bool = False,
):
    """Main entry point to register a Warp Python function to be part of the Warp API and appear in the documentation.

    Args:
        function: Warp function to be registered.
        group: Classification used for the documentation.
        hidden: Whether to add that function into the documentation.
    """
    function.group = group
    function.hidden = hidden
    builtin_functions[function.key] = function


# global dictionary of modules
user_modules: dict[str, Module] = {}


def get_module(name: str) -> Module:
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
            old_module.structs = {}
            old_module.loader = parent_loader

        return user_modules[name]

    else:
        # else Warp module didn't exist yet, so create a new one
        user_modules[name] = warp.context.Module(name, parent_loader)
        return user_modules[name]


# ModuleHasher computes the module hash based on all the kernels, module options,
# and build configuration.  For each kernel, it computes a deep hash by recursively
# hashing all referenced functions, structs, and constants, even those defined in
# other modules.  The module hash is computed in the constructor and can be retrieved
# using get_module_hash().  In addition, the ModuleHasher takes care of filtering out
# duplicate kernels for codegen (see get_unique_kernels()).
class ModuleHasher:
    def __init__(self, module):
        # cache function hashes to avoid hashing multiple times
        self.function_hashes = {}  # (function: hash)

        # avoid recursive spiral of doom (e.g., function calling an overload of itself)
        self.functions_in_progress = set()

        # all unique kernels for codegen, filtered by hash
        self.unique_kernels = {}  # (hash: kernel)

        # start hashing the module
        ch = hashlib.sha256()

        # hash all non-generic kernels
        for kernel in module.live_kernels:
            if kernel.is_generic:
                for ovl in kernel.overloads.values():
                    if not ovl.adj.skip_build:
                        ovl.hash = self.hash_kernel(ovl)
            else:
                if not kernel.adj.skip_build:
                    kernel.hash = self.hash_kernel(kernel)

        # include all unique kernels in the module hash
        for kernel_hash in sorted(self.unique_kernels.keys()):
            ch.update(kernel_hash)

        # configuration parameters
        for opt in sorted(module.options.keys()):
            s = f"{opt}:{module.options[opt]}"
            ch.update(bytes(s, "utf-8"))

        # ensure to trigger recompilation if flags affecting kernel compilation are changed
        if warp.config.verify_fp:
            ch.update(bytes("verify_fp", "utf-8"))

        # line directives, e.g. for Nsight Compute
        ch.update(bytes(ctypes.c_int(warp.config.line_directives)))

        # build config
        ch.update(bytes(warp.config.mode, "utf-8"))

        # save the module hash
        self.module_hash = ch.digest()

    def hash_kernel(self, kernel: Kernel) -> bytes:
        # NOTE: We only hash non-generic kernels, so we don't traverse kernel overloads here.

        ch = hashlib.sha256()

        ch.update(bytes(kernel.key, "utf-8"))
        ch.update(self.hash_adjoint(kernel.adj))

        h = ch.digest()

        self.unique_kernels[h] = kernel

        return h

    def hash_function(self, func: Function) -> bytes:
        # NOTE: This method hashes all possible overloads that a function call could resolve to.
        # The exact overload will be resolved at build time, when the argument types are known.

        h = self.function_hashes.get(func)
        if h is not None:
            return h

        self.functions_in_progress.add(func)

        ch = hashlib.sha256()

        ch.update(bytes(func.key, "utf-8"))

        # include all concrete and generic overloads
        overloads: dict[str, Function] = {**func.user_overloads, **func.user_templates}
        for sig in sorted(overloads.keys()):
            ovl = overloads[sig]

            # skip instantiations of generic functions
            if ovl.generic_parent is not None:
                continue

            # adjoint
            ch.update(self.hash_adjoint(ovl.adj))

            # custom bits
            if ovl.custom_grad_func:
                ch.update(self.hash_adjoint(ovl.custom_grad_func.adj))
            if ovl.custom_replay_func:
                ch.update(self.hash_adjoint(ovl.custom_replay_func.adj))
            if ovl.replay_snippet:
                ch.update(bytes(ovl.replay_snippet, "utf-8"))
            if ovl.native_snippet:
                ch.update(bytes(ovl.native_snippet, "utf-8"))
            if ovl.adj_native_snippet:
                ch.update(bytes(ovl.adj_native_snippet, "utf-8"))

        h = ch.digest()

        self.function_hashes[func] = h

        self.functions_in_progress.remove(func)

        return h

    def hash_adjoint(self, adj: warp.codegen.Adjoint) -> bytes:
        # NOTE: We don't cache adjoint hashes, because adjoints are always unique.
        # Even instances of generic kernels and functions have unique adjoints with
        # different argument types.

        ch = hashlib.sha256()

        # source
        ch.update(bytes(adj.source, "utf-8"))

        # args
        for arg, arg_type in adj.arg_types.items():
            s = f"{arg}:{warp.types.get_type_code(arg_type)}"
            ch.update(bytes(s, "utf-8"))

            # hash struct types
            if isinstance(arg_type, warp.codegen.Struct):
                ch.update(arg_type.hash)
            elif warp.types.is_array(arg_type) and isinstance(arg_type.dtype, warp.codegen.Struct):
                ch.update(arg_type.dtype.hash)

        # find referenced constants, types, and functions
        constants, types, functions = adj.get_references()

        # hash referenced constants
        for name, value in constants.items():
            ch.update(bytes(name, "utf-8"))
            ch.update(self.get_constant_bytes(value))

        # hash wp.static() expressions that were evaluated at declaration time
        for k, v in adj.static_expressions.items():
            ch.update(bytes(k, "utf-8"))
            if isinstance(v, Function):
                if v not in self.functions_in_progress:
                    ch.update(self.hash_function(v))
            else:
                ch.update(self.get_constant_bytes(v))

        # hash referenced types
        for t in types.keys():
            ch.update(bytes(warp.types.get_type_code(t), "utf-8"))

        # hash referenced functions
        for f in functions.keys():
            if f not in self.functions_in_progress:
                ch.update(self.hash_function(f))

        return ch.digest()

    def get_constant_bytes(self, value) -> bytes:
        if isinstance(value, int):
            # this also handles builtins.bool
            return bytes(ctypes.c_int(value))
        elif isinstance(value, float):
            return bytes(ctypes.c_float(value))
        elif isinstance(value, warp.types.float16):
            # float16 is a special case
            return bytes(ctypes.c_float(value.value))
        elif isinstance(value, tuple(warp.types.scalar_and_bool_types)):
            return bytes(value._type_(value.value))
        elif hasattr(value, "_wp_scalar_type_"):
            return bytes(value)
        elif isinstance(value, warp.codegen.StructInstance):
            return bytes(value._ctype)
        else:
            raise TypeError(f"Invalid constant type: {type(value)}")

    def get_module_hash(self) -> bytes:
        return self.module_hash

    def get_unique_kernels(self):
        return self.unique_kernels.values()


class ModuleBuilder:
    def __init__(self, module, options, hasher=None):
        self.functions = {}
        self.structs = {}
        self.options = options
        self.module = module
        self.deferred_functions = []
        self.fatbins = {}  # map from <some identifier> to fatbins, to add at link time
        self.ltoirs = {}  # map from lto symbol to lto binary
        self.ltoirs_decl = {}  # map from lto symbol to lto forward declaration
        self.shared_memory_bytes = {}  # map from lto symbol to shared memory requirements

        if hasher is None:
            hasher = ModuleHasher(module)

        # build all unique kernels
        self.kernels = hasher.get_unique_kernels()
        for kernel in self.kernels:
            self.build_kernel(kernel)

        # build deferred functions
        for func in self.deferred_functions:
            self.build_function(func)

    def build_struct_recursive(self, struct: warp.codegen.Struct):
        structs = []

        stack = [struct]
        while stack:
            s = stack.pop()

            structs.append(s)

            for var in s.vars.values():
                if isinstance(var.type, warp.codegen.Struct):
                    stack.append(var.type)
                elif warp.types.is_array(var.type) and isinstance(var.type.dtype, warp.codegen.Struct):
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
                    def value_type(arg_types, arg_values):
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

    def build_meta(self):
        meta = {}

        for kernel in self.kernels:
            name = kernel.get_mangled_name()

            meta[name + "_cuda_kernel_forward_smem_bytes"] = kernel.adj.get_total_required_shared()
            meta[name + "_cuda_kernel_backward_smem_bytes"] = kernel.adj.get_total_required_shared() * 2

        return meta

    def codegen(self, device):
        source = ""

        # code-gen LTO forward declarations
        source += 'extern "C" {\n'
        for fwd in self.ltoirs_decl.values():
            source += fwd + "\n"
        source += "}\n"

        # code-gen structs
        visited_structs = set()
        for struct in self.structs.keys():
            # avoid emitting duplicates
            if struct.hash not in visited_structs:
                source += warp.codegen.codegen_struct(struct)
                visited_structs.add(struct.hash)

        # code-gen all imported functions
        for func in self.functions.keys():
            if func.native_snippet is None:
                source += warp.codegen.codegen_func(
                    func.adj, c_func_name=func.native_func, device=device, options=self.options
                )
            else:
                source += warp.codegen.codegen_snippet(
                    func.adj,
                    name=func.native_func,
                    snippet=func.native_snippet,
                    adj_snippet=func.adj_native_snippet,
                    replay_snippet=func.replay_snippet,
                )

        for kernel in self.kernels:
            source += warp.codegen.codegen_kernel(kernel, device=device, options=self.options)
            source += warp.codegen.codegen_module(kernel, device=device, options=self.options)

        # add headers
        if device == "cpu":
            source = warp.codegen.cpu_module_header.format(block_dim=self.options["block_dim"]) + source
        else:
            source = warp.codegen.cuda_module_header.format(block_dim=self.options["block_dim"]) + source

        return source


# ModuleExec holds the compiled executable code for a specific device.
# It can be used to obtain kernel hooks on that device and serves
# as a reference-counted wrapper of the loaded module.
# Clients can keep a reference to a ModuleExec object to prevent the
# executable code from being unloaded prematurely.
# For example, the Graph class retains references to all the CUDA modules
# needed by a graph.  This ensures that graphs remain valid even if
# the original Modules get reloaded.
class ModuleExec:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.handle = None
        return instance

    def __init__(self, handle, module_hash, device, meta):
        self.handle = handle
        self.module_hash = module_hash
        self.device = device
        self.kernel_hooks = {}
        self.meta = meta

    # release the loaded module
    def __del__(self):
        if self.handle is not None:
            if self.device.is_cuda:
                # use CUDA context guard to avoid side effects during garbage collection
                with self.device.context_guard:
                    runtime.core.cuda_unload_module(self.device.context, self.handle)
            else:
                runtime.llvm.unload_obj(self.handle.encode("utf-8"))

    # lookup and cache kernel entry points
    def get_kernel_hooks(self, kernel) -> KernelHooks:
        # Use kernel.adj as a unique key for cache lookups instead of the kernel itself.
        # This avoids holding a reference to the kernel and is faster than using
        # a WeakKeyDictionary with kernels as keys.
        hooks = self.kernel_hooks.get(kernel.adj)
        if hooks is not None:
            return hooks

        name = kernel.get_mangled_name()

        options = dict(kernel.module.options)
        options.update(kernel.options)

        if self.device.is_cuda:
            forward_name = name + "_cuda_kernel_forward"
            forward_kernel = runtime.core.cuda_get_kernel(
                self.device.context, self.handle, forward_name.encode("utf-8")
            )

            if options["enable_backward"]:
                backward_name = name + "_cuda_kernel_backward"
                backward_kernel = runtime.core.cuda_get_kernel(
                    self.device.context, self.handle, backward_name.encode("utf-8")
                )
            else:
                backward_kernel = None

            # look up the required shared memory size for each kernel from module metadata
            forward_smem_bytes = self.meta[forward_name + "_smem_bytes"]
            backward_smem_bytes = self.meta[backward_name + "_smem_bytes"] if options["enable_backward"] else 0

            # configure kernels maximum shared memory size
            max_smem_bytes = runtime.core.cuda_get_max_shared_memory(self.device.context)

            if not runtime.core.cuda_configure_kernel_shared_memory(forward_kernel, forward_smem_bytes):
                print(
                    f"Warning: Failed to configure kernel dynamic shared memory for this device, tried to configure {forward_name} kernel for {forward_smem_bytes} bytes, but maximum available is {max_smem_bytes}"
                )

            if options["enable_backward"] and not runtime.core.cuda_configure_kernel_shared_memory(
                backward_kernel, backward_smem_bytes
            ):
                print(
                    f"Warning: Failed to configure kernel dynamic shared memory for this device, tried to configure {backward_name} kernel for {backward_smem_bytes} bytes, but maximum available is {max_smem_bytes}"
                )

            hooks = KernelHooks(forward_kernel, backward_kernel, forward_smem_bytes, backward_smem_bytes)

        else:
            func = ctypes.CFUNCTYPE(None)
            forward = (
                func(runtime.llvm.lookup(self.handle.encode("utf-8"), (name + "_cpu_forward").encode("utf-8"))) or None
            )

            if options["enable_backward"]:
                backward = (
                    func(runtime.llvm.lookup(self.handle.encode("utf-8"), (name + "_cpu_backward").encode("utf-8")))
                    or None
                )
            else:
                backward = None

            hooks = KernelHooks(forward, backward)

        self.kernel_hooks[kernel.adj] = hooks
        return hooks


# -----------------------------------------------------
# stores all functions and kernels for a Python module
# creates a hash of the function to use for checking
# build cache
class Module:
    def __init__(self, name: str | None, loader=None):
        self.name = name if name is not None else "None"

        self.loader = loader

        # lookup the latest versions of kernels, functions, and structs by key
        self.kernels = {}  # (key: kernel)
        self.functions = {}  # (key: function)
        self.structs = {}  # (key: struct)

        # Set of all "live" kernels in this module, i.e., kernels that still have references.
        # We keep a weak reference to every kernel ever created in this module and rely on Python GC
        # to release kernels that no longer have any references (in user code or internal bookkeeping).
        # The difference between `live_kernels` and `kernels` is that `live_kernels` may contain
        # multiple kernels with the same key (which is essential to support closures), while `kernels`
        # only holds the latest kernel for each key.  When the module is built, we compute the hash
        # of each kernel in `live_kernels` and filter out duplicates for codegen.
        self._live_kernels = weakref.WeakSet()

        # executable modules currently loaded
        self.execs = {}  # ((device.context, blockdim): ModuleExec)

        # set of device contexts where the build has failed
        self.failed_builds = set()

        # hash data, including the module hash. Module may store multiple hashes (one per block_dim used)
        self.hashers = {}

        # LLVM executable modules are identified using strings.  Since it's possible for multiple
        # executable versions to be loaded at the same time, we need a way to ensure uniqueness.
        # A unique handle is created from the module name and this auto-incremented integer id.
        # NOTE: The module hash is not sufficient for uniqueness in rare cases where a module
        # is retained and later reloaded with the same hash.
        self.cpu_exec_id = 0

        self.options = {
            "max_unroll": warp.config.max_unroll,
            "enable_backward": warp.config.enable_backward,
            "fast_math": False,
            "fuse_fp": True,
            "lineinfo": warp.config.lineinfo,
            "cuda_output": None,  # supported values: "ptx", "cubin", or None (automatic)
            "mode": warp.config.mode,
            "block_dim": 256,
            "compile_time_trace": warp.config.compile_time_trace,
        }

        # Module dependencies are determined by scanning each function
        # and kernel for references to external functions and structs.
        #
        # The dependency mechanism works for both static and dynamic (runtime) modifications.
        # When a module is reloaded at runtime, we recursively unload all of its
        # dependents, so that they will be re-hashed and reloaded on the next launch.
        # -> See ``get_module()``

        self.references = set()  # modules whose content we depend on
        self.dependents = set()  # modules that depend on our content

    def register_struct(self, struct):
        self.structs[struct.key] = struct

        # for a reload of module on next launch
        self.mark_modified()

    def register_kernel(self, kernel):
        # keep a reference to the latest version
        self.kernels[kernel.key] = kernel

        # track all kernel objects, even if they are duplicates
        self._live_kernels.add(kernel)

        self.find_references(kernel.adj)

        # for a reload of module on next launch
        self.mark_modified()

    def register_function(self, func, scope_locals, skip_adding_overload=False):
        # check for another Function with the same name in the same scope
        obj = scope_locals.get(func.func.__name__)
        if isinstance(obj, Function):
            func_existing = obj
        else:
            func_existing = None

        # keep a reference to the latest version
        self.functions[func.key] = func_existing or func

        if func_existing:
            # Check whether the new function's signature match any that has
            # already been registered. If so, then we simply override it, as
            # Python would do it, otherwise we register it as a new overload.
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
                # replace the top-level function, but keep existing overloads

                # copy generic overloads
                func.user_templates = func_existing.user_templates.copy()

                # copy concrete overloads
                if warp.types.is_generic_signature(sig):
                    # skip overloads that were instantiated from the function being replaced
                    for k, v in func_existing.user_overloads.items():
                        if v.generic_parent != func_existing:
                            func.user_overloads[k] = v
                    func.user_templates[sig] = func
                else:
                    func.user_overloads = func_existing.user_overloads.copy()
                    func.user_overloads[sig] = func

                self.functions[func.key] = func
            elif not skip_adding_overload:
                # check if this is a generic overload that replaces an existing one
                if warp.types.is_generic_signature(sig):
                    old_generic = func_existing.user_templates.get(sig)
                    if old_generic is not None:
                        # purge any concrete overloads that were instantiated from the old one
                        for k, v in list(func_existing.user_overloads.items()):
                            if v.generic_parent == old_generic:
                                del func_existing.user_overloads[k]
                func_existing.add_overload(func)

        self.find_references(func.adj)

        # for a reload of module on next launch
        self.mark_modified()

    @property
    def live_kernels(self):
        # Return a list of kernels that still have references.
        # We return a regular list instead of the WeakSet to avoid undesirable issues
        # if kernels are garbage collected before the caller is done using this list.
        # Note that we should avoid retaining strong references to kernels unnecessarily
        # so that Python GC can release kernels that no longer have user references.
        # It is tempting to call gc.collect() here to force garbage collection,
        # but this can have undesirable consequences (e.g., GC during graph capture),
        # so we should avoid it as a general rule.  Instead, we rely on Python's
        # reference counting GC to collect kernels that have gone out of scope.
        return list(self._live_kernels)

    # find kernel corresponding to a Python function
    def find_kernel(self, func):
        qualname = warp.codegen.make_full_qualified_name(func)
        return self.kernels.get(qualname)

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
        # compute latest hash
        block_dim = self.options["block_dim"]
        self.hashers[block_dim] = ModuleHasher(self)
        return self.hashers[block_dim].get_module_hash()

    def load(self, device, block_dim=None) -> ModuleExec:
        device = runtime.get_device(device)

        # update module options if launching with a new block dim
        if block_dim is not None:
            self.options["block_dim"] = block_dim

        active_block_dim = self.options["block_dim"]

        # compute the hash if needed
        if active_block_dim not in self.hashers:
            self.hashers[active_block_dim] = ModuleHasher(self)

        # check if executable module is already loaded and not stale
        exec = self.execs.get((device.context, active_block_dim))
        if exec is not None:
            if exec.module_hash == self.hashers[active_block_dim].get_module_hash():
                return exec

        # quietly avoid repeated build attempts to reduce error spew
        if device.context in self.failed_builds:
            return None

        module_name = "wp_" + self.name
        module_hash = self.hashers[active_block_dim].get_module_hash()

        # use a unique module path using the module short hash
        module_name_short = f"{module_name}_{module_hash.hex()[:7]}"
        module_dir = os.path.join(warp.config.kernel_cache_dir, module_name_short)

        with warp.ScopedTimer(
            f"Module {self.name} {module_hash.hex()[:7]} load on device '{device}'", active=not warp.config.quiet
        ) as module_load_timer:
            # -----------------------------------------------------------
            # determine output paths
            if device.is_cpu:
                output_name = f"{module_name_short}.o"
                output_arch = None

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
                    # use the default PTX arch if the device supports it
                    if warp.config.ptx_target_arch is not None:
                        output_arch = min(device.arch, warp.config.ptx_target_arch)
                    else:
                        output_arch = min(device.arch, runtime.default_ptx_arch)
                    output_name = f"{module_name_short}.sm{output_arch}.ptx"
                else:
                    output_arch = device.arch
                    output_name = f"{module_name_short}.sm{output_arch}.cubin"

            # final object binary path
            binary_path = os.path.join(module_dir, output_name)

            # -----------------------------------------------------------
            # check cache and build if necessary

            build_dir = None

            # we always want to build if binary doesn't exist yet
            # and we want to rebuild if we are not caching kernels or if we are tracking array access
            if (
                not os.path.exists(binary_path)
                or not warp.config.cache_kernels
                or warp.config.verify_autograd_array_access
            ):
                builder_options = {
                    **self.options,
                    # Some of the tile codegen, such as cuFFTDx and cuBLASDx, requires knowledge of the target arch
                    "output_arch": output_arch,
                }
                builder = ModuleBuilder(self, builder_options, hasher=self.hashers[active_block_dim])

                # create a temporary (process unique) dir for build outputs before moving to the binary dir
                build_dir = os.path.join(
                    warp.config.kernel_cache_dir, f"{module_name}_{module_hash.hex()[:7]}_p{os.getpid()}"
                )

                # dir may exist from previous attempts / runs / archs
                Path(build_dir).mkdir(parents=True, exist_ok=True)

                module_load_timer.extra_msg = " (compiled)"  # For wp.ScopedTimer informational purposes

                # build CPU
                if device.is_cpu:
                    # build
                    try:
                        source_code_path = os.path.join(build_dir, f"{module_name_short}.cpp")

                        # write cpp sources
                        cpp_source = builder.codegen("cpu")

                        with open(source_code_path, "w") as cpp_file:
                            cpp_file.write(cpp_source)

                        output_path = os.path.join(build_dir, output_name)

                        # build object code
                        with warp.ScopedTimer("Compile x86", active=warp.config.verbose):
                            warp.build.build_cpu(
                                output_path,
                                source_code_path,
                                mode=self.options["mode"],
                                fast_math=self.options["fast_math"],
                                verify_fp=warp.config.verify_fp,
                                fuse_fp=self.options["fuse_fp"],
                            )

                    except Exception as e:
                        self.failed_builds.add(None)
                        module_load_timer.extra_msg = " (error)"
                        raise (e)

                elif device.is_cuda:
                    # build
                    try:
                        source_code_path = os.path.join(build_dir, f"{module_name_short}.cu")

                        # write cuda sources
                        cu_source = builder.codegen("cuda")

                        with open(source_code_path, "w") as cu_file:
                            cu_file.write(cu_source)

                        output_path = os.path.join(build_dir, output_name)

                        # generate PTX or CUBIN
                        with warp.ScopedTimer("Compile CUDA", active=warp.config.verbose):
                            warp.build.build_cuda(
                                source_code_path,
                                output_arch,
                                output_path,
                                config=self.options["mode"],
                                verify_fp=warp.config.verify_fp,
                                fast_math=self.options["fast_math"],
                                fuse_fp=self.options["fuse_fp"],
                                lineinfo=self.options["lineinfo"],
                                compile_time_trace=self.options["compile_time_trace"],
                                ltoirs=builder.ltoirs.values(),
                                fatbins=builder.fatbins.values(),
                            )

                    except Exception as e:
                        self.failed_builds.add(device.context)
                        module_load_timer.extra_msg = " (error)"
                        raise (e)

                # ------------------------------------------------------------
                # build meta data

                meta = builder.build_meta()
                meta_path = os.path.join(build_dir, f"{module_name_short}.meta")

                with open(meta_path, "w") as meta_file:
                    json.dump(meta, meta_file)

                # -----------------------------------------------------------
                # update cache

                # try to move process outputs to cache
                warp.build.safe_rename(build_dir, module_dir)

                if os.path.exists(module_dir):
                    if not os.path.exists(binary_path):
                        # copy our output file to the destination module
                        # this is necessary in case different processes
                        # have different GPU architectures / devices
                        try:
                            os.rename(output_path, binary_path)
                        except (OSError, FileExistsError):
                            # another process likely updated the module dir first
                            pass

                    try:
                        final_source_path = os.path.join(module_dir, os.path.basename(source_code_path))
                        if not os.path.exists(final_source_path):
                            os.rename(source_code_path, final_source_path)
                    except (OSError, FileExistsError):
                        # another process likely updated the module dir first
                        pass
                    except Exception as e:
                        # We don't need source_code_path to be copied successfully to proceed, so warn and keep running
                        warp.utils.warn(f"Exception when renaming {source_code_path}: {e}")
            else:
                module_load_timer.extra_msg = " (cached)"  # For wp.ScopedTimer informational purposes

            # -----------------------------------------------------------
            # Load CPU or CUDA binary

            meta_path = os.path.join(module_dir, f"{module_name_short}.meta")
            with open(meta_path) as meta_file:
                meta = json.load(meta_file)

            if device.is_cpu:
                # LLVM modules are identified using strings, so we need to ensure uniqueness
                module_handle = f"{module_name}_{self.cpu_exec_id}"
                self.cpu_exec_id += 1
                runtime.llvm.load_obj(binary_path.encode("utf-8"), module_handle.encode("utf-8"))
                module_exec = ModuleExec(module_handle, module_hash, device, meta)
                self.execs[(None, active_block_dim)] = module_exec

            elif device.is_cuda:
                cuda_module = warp.build.load_cuda(binary_path, device)
                if cuda_module is not None:
                    module_exec = ModuleExec(cuda_module, module_hash, device, meta)
                    self.execs[(device.context, active_block_dim)] = module_exec
                else:
                    module_load_timer.extra_msg = " (error)"
                    raise Exception(f"Failed to load CUDA module '{self.name}'")

            if build_dir:
                import shutil

                # clean up build_dir used for this process regardless
                shutil.rmtree(build_dir, ignore_errors=True)

        return module_exec

    def unload(self):
        # force rehashing on next load
        self.mark_modified()

        # clear loaded modules
        self.execs = {}

    def mark_modified(self):
        # clear hash data
        self.hashers = {}

        # clear build failures
        self.failed_builds = set()

    # lookup kernel entry points based on name, called after compilation / module load
    def get_kernel_hooks(self, kernel, device: Device) -> KernelHooks:
        module_exec = self.execs.get((device.context, self.options["block_dim"]))
        if module_exec is not None:
            return module_exec.get_kernel_hooks(kernel)
        else:
            raise RuntimeError(f"Module is not loaded on device {device}")


# -------------------------------------------
# execution context


class CpuDefaultAllocator:
    def __init__(self, device):
        assert device.is_cpu
        self.deleter = lambda ptr, size: self.free(ptr, size)

    def alloc(self, size_in_bytes):
        ptr = runtime.core.alloc_host(size_in_bytes)
        if not ptr:
            raise RuntimeError(f"Failed to allocate {size_in_bytes} bytes on device '{self.device}'")
        return ptr

    def free(self, ptr, size_in_bytes):
        runtime.core.free_host(ptr)


class CpuPinnedAllocator:
    def __init__(self, device):
        assert device.is_cpu
        self.deleter = lambda ptr, size: self.free(ptr, size)

    def alloc(self, size_in_bytes):
        ptr = runtime.core.alloc_pinned(size_in_bytes)
        if not ptr:
            raise RuntimeError(f"Failed to allocate {size_in_bytes} bytes on device '{self.device}'")
        return ptr

    def free(self, ptr, size_in_bytes):
        runtime.core.free_pinned(ptr)


class CudaDefaultAllocator:
    def __init__(self, device):
        assert device.is_cuda
        self.device = device
        self.deleter = lambda ptr, size: self.free(ptr, size)

    def alloc(self, size_in_bytes):
        ptr = runtime.core.alloc_device_default(self.device.context, size_in_bytes)
        # If the allocation fails, check if graph capture is active to raise an informative error.
        # We delay the capture check to avoid overhead.
        if not ptr:
            reason = ""
            if self.device.is_capturing:
                if not self.device.is_mempool_supported:
                    reason = (
                        ":  "
                        f"Failed to allocate memory during graph capture because memory pools are not supported "
                        f"on device '{self.device}'.  Try pre-allocating memory before capture begins."
                    )
                elif not self.device.is_mempool_enabled:
                    reason = (
                        ":  "
                        f"Failed to allocate memory during graph capture because memory pools are not enabled "
                        f"on device '{self.device}'.  Try calling wp.set_mempool_enabled('{self.device}', True) before capture begins."
                    )
            raise RuntimeError(f"Failed to allocate {size_in_bytes} bytes on device '{self.device}'{reason}")
        return ptr

    def free(self, ptr, size_in_bytes):
        runtime.core.free_device_default(self.device.context, ptr)


class CudaMempoolAllocator:
    def __init__(self, device):
        assert device.is_cuda
        assert device.is_mempool_supported
        self.device = device
        self.deleter = lambda ptr, size: self.free(ptr, size)

    def alloc(self, size_in_bytes):
        ptr = runtime.core.alloc_device_async(self.device.context, size_in_bytes)
        if not ptr:
            raise RuntimeError(f"Failed to allocate {size_in_bytes} bytes on device '{self.device}'")
        return ptr

    def free(self, ptr, size_in_bytes):
        runtime.core.free_device_async(self.device.context, ptr)


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


class Event:
    """A CUDA event that can be recorded onto a stream.

    Events can be used for device-side synchronization, which do not block
    the host thread.
    """

    # event creation flags
    class Flags:
        DEFAULT = 0x0
        BLOCKING_SYNC = 0x1
        DISABLE_TIMING = 0x2
        INTERPROCESS = 0x4

    def __new__(cls, *args, **kwargs):
        """Creates a new event instance."""
        instance = super().__new__(cls)
        instance.owner = False
        return instance

    def __init__(
        self, device: Devicelike = None, cuda_event=None, enable_timing: bool = False, interprocess: bool = False
    ):
        """Initializes the event on a CUDA device.

        Args:
            device: The CUDA device whose streams this event may be recorded onto.
              If ``None``, then the current default device will be used.
            cuda_event: A pointer to a previously allocated CUDA event. If
              `None`, then a new event will be allocated on the associated device.
            enable_timing: If ``True`` this event will record timing data.
              :func:`~warp.get_event_elapsed_time` can be used to measure the
              time between two events created with ``enable_timing=True`` and
              recorded onto streams.
            interprocess: If ``True`` this event may be used as an interprocess event.

        Raises:
            RuntimeError: The event could not be created.
            ValueError: The combination of ``enable_timing=True`` and
                ``interprocess=True`` is not allowed.
        """

        device = get_device(device)
        if not device.is_cuda:
            raise RuntimeError(f"Device {device} is not a CUDA device")

        self.device = device
        self.enable_timing = enable_timing

        if cuda_event is not None:
            self.cuda_event = cuda_event
        else:
            flags = Event.Flags.DEFAULT
            if not enable_timing:
                flags |= Event.Flags.DISABLE_TIMING
            if interprocess:
                if enable_timing:
                    raise ValueError("The combination of 'enable_timing=True' and 'interprocess=True' is not allowed.")
                flags |= Event.Flags.INTERPROCESS

            self.cuda_event = runtime.core.cuda_event_create(device.context, flags)
            if not self.cuda_event:
                raise RuntimeError(f"Failed to create event on device {device}")
            self.owner = True

    def ipc_handle(self) -> bytes:
        """Return a CUDA IPC handle of the event as a 64-byte ``bytes`` object.

        The event must have been created with ``interprocess=True`` in order to
        obtain a valid interprocess handle.

        IPC is currently only supported on Linux.

        Example:
            Create an event and get its IPC handle::

                e1 = wp.Event(interprocess=True)
                event_handle = e1.ipc_handle()

        Raises:
            RuntimeError: Device does not support IPC.
        """

        if self.device.is_ipc_supported is not False:
            # Allocate a buffer for the data (64-element char array)
            ipc_handle_buffer = (ctypes.c_char * 64)()

            warp.context.runtime.core.cuda_ipc_get_event_handle(self.device.context, self.cuda_event, ipc_handle_buffer)

            if ipc_handle_buffer.raw == bytes(64):
                warp.utils.warn("IPC event handle appears to be invalid. Was interprocess=True used?")

            return ipc_handle_buffer.raw

        else:
            raise RuntimeError(f"Device {self.device} does not support IPC.")

    @property
    def is_complete(self) -> bool:
        """A boolean indicating whether all work on the stream when the event was recorded has completed.

        This property may not be accessed during a graph capture on any stream.
        """

        result_code = runtime.core.cuda_event_query(self.cuda_event)

        return result_code == 0

    def __del__(self):
        if not self.owner:
            return

        runtime.core.cuda_event_destroy(self.cuda_event)


class Stream:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.cuda_stream = None
        instance.owner = False
        return instance

    def __init__(self, device: Device | str | None = None, priority: int = 0, **kwargs):
        """Initialize the stream on a device with an optional specified priority.

        Args:
            device: The CUDA device on which this stream will be created.
            priority: An optional integer specifying the requested stream priority.
              Can be -1 (high priority) or 0 (low/default priority).
              Values outside this range will be clamped.
            cuda_stream (int): A optional external stream handle passed as an
              integer. The caller is responsible for ensuring that the external
              stream does not get destroyed while it is referenced by this
              object.

        Raises:
            RuntimeError: If function is called before Warp has completed
              initialization with a ``device`` that is not an instance of
              :class:`Device <warp.context.Device>`.
            RuntimeError: ``device`` is not a CUDA Device.
            RuntimeError: The stream could not be created on the device.
            TypeError: The requested stream priority is not an integer.
        """
        # event used internally for synchronization (cached to avoid creating temporary events)
        self._cached_event = None

        # we can't use get_device() if called during init, but we can use an explicit Device arg
        if runtime is not None:
            device = runtime.get_device(device)
        elif not isinstance(device, Device):
            raise RuntimeError(
                "A Device object is required when creating a stream before or during Warp initialization"
            )

        if not device.is_cuda:
            raise RuntimeError(f"Device {device} is not a CUDA device")

        self.device = device

        # we pass cuda_stream through kwargs because cuda_stream=None is actually a valid value (CUDA default stream)
        if "cuda_stream" in kwargs:
            self.cuda_stream = kwargs["cuda_stream"]
            device.runtime.core.cuda_stream_register(device.context, self.cuda_stream)
        else:
            if not isinstance(priority, int):
                raise TypeError("Stream priority must be an integer.")
            clamped_priority = max(-1, min(priority, 0))  # Only support two priority levels
            self.cuda_stream = device.runtime.core.cuda_stream_create(device.context, clamped_priority)

            if not self.cuda_stream:
                raise RuntimeError(f"Failed to create stream on device {device}")
            self.owner = True

    def __del__(self):
        if not self.cuda_stream:
            return

        if self.owner:
            runtime.core.cuda_stream_destroy(self.device.context, self.cuda_stream)
        else:
            runtime.core.cuda_stream_unregister(self.device.context, self.cuda_stream)

    @property
    def cached_event(self) -> Event:
        if self._cached_event is None:
            self._cached_event = Event(self.device)
        return self._cached_event

    def record_event(self, event: Event | None = None) -> Event:
        """Record an event onto the stream.

        Args:
            event: A warp.Event instance to be recorded onto the stream. If not
              provided, an :class:`~warp.Event` on the same device will be created.

        Raises:
            RuntimeError: The provided :class:`~warp.Event` is from a different device than
                the recording stream.
        """
        if event is None:
            event = Event(self.device)
        elif event.device != self.device:
            raise RuntimeError(
                f"Event from device {event.device} cannot be recorded on stream from device {self.device}"
            )

        runtime.core.cuda_event_record(event.cuda_event, self.cuda_stream, event.enable_timing)

        return event

    def wait_event(self, event: Event):
        """Makes all future work in this stream wait until `event` has completed.

        This function does not block the host thread.
        """
        runtime.core.cuda_stream_wait_event(self.cuda_stream, event.cuda_event)

    def wait_stream(self, other_stream: Stream, event: Event | None = None):
        """Records an event on `other_stream` and makes this stream wait on it.

        All work added to this stream after this function has been called will
        delay their execution until all preceding commands in `other_stream`
        have completed.

        This function does not block the host thread.

        Args:
            other_stream: The stream on which the calling stream will wait for
              previously issued commands to complete before executing subsequent
              commands.
            event: An optional :class:`Event` instance that will be used to
              record an event onto ``other_stream``. If ``None``, an internally
              managed :class:`Event` instance will be used.
        """

        if event is None:
            event = other_stream.cached_event

        runtime.core.cuda_stream_wait_stream(self.cuda_stream, other_stream.cuda_stream, event.cuda_event)

    @property
    def is_complete(self) -> bool:
        """A boolean indicating whether all work on the stream has completed.

        This property may not be accessed during a graph capture on any stream.
        """

        result_code = runtime.core.cuda_stream_query(self.cuda_stream)

        return result_code == 0

    @property
    def is_capturing(self) -> bool:
        """A boolean indicating whether a graph capture is currently ongoing on this stream."""
        return bool(runtime.core.cuda_stream_is_capturing(self.cuda_stream))

    @property
    def priority(self) -> int:
        """An integer representing the priority of the stream."""
        return runtime.core.cuda_stream_get_priority(self.cuda_stream)


class Device:
    """A device to allocate Warp arrays and to launch kernels on.

    Attributes:
        ordinal (int): A Warp-specific label for the device. ``-1`` for CPU devices.
        name (str): A label for the device. By default, CPU devices will be named according to the processor name,
            or ``"CPU"`` if the processor name cannot be determined.
        arch (int): The compute capability version number calculated as ``10 * major + minor``.
            ``0`` for CPU devices.
        sm_count (int): The number of streaming multiprocessors on the CUDA device.
            ``0`` for CPU devices.
        is_uva (bool): Indicates whether the device supports unified addressing.
            ``False`` for CPU devices.
        is_cubin_supported (bool): Indicates whether Warp's version of NVRTC can directly
            generate CUDA binary files (cubin) for this device's architecture. ``False`` for CPU devices.
        is_mempool_supported (bool): Indicates whether the device supports using the ``cuMemAllocAsync`` and
            ``cuMemPool`` family of APIs for stream-ordered memory allocations. ``False`` for CPU devices.
        is_ipc_supported (Optional[bool]): Indicates whether the device supports IPC.

            - ``True`` if supported.
            - ``False`` if not supported.
            - ``None`` if IPC support could not be determined (e.g. CUDA 11).

        is_primary (bool): Indicates whether this device's CUDA context is also the device's primary context.
        uuid (str): The UUID of the CUDA device. The UUID is in the same format used by ``nvidia-smi -L``.
            ``None`` for CPU devices.
        pci_bus_id (str): An identifier for the CUDA device in the format ``[domain]:[bus]:[device]``, in which
            ``domain``, ``bus``, and ``device`` are all hexadecimal values. ``None`` for CPU devices.
    """

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

        # maps streams to started graph captures
        self.captures = {}

        self.context_guard = ContextGuard(self)

        if self.ordinal == -1:
            # CPU device
            self.name = platform.processor() or "CPU"
            self.arch = 0
            self.sm_count = 0
            self.is_uva = False
            self.is_mempool_supported = False
            self.is_mempool_enabled = False
            self.is_ipc_supported = False  # TODO: Support IPC for CPU arrays
            self.is_cubin_supported = False
            self.uuid = None
            self.pci_bus_id = None

            # TODO: add more device-specific dispatch functions
            self.memset = runtime.core.memset_host
            self.memtile = runtime.core.memtile_host

            self.default_allocator = CpuDefaultAllocator(self)
            self.pinned_allocator = CpuPinnedAllocator(self)

        elif ordinal >= 0 and ordinal < runtime.core.cuda_device_get_count():
            # CUDA device
            self.name = runtime.core.cuda_device_get_name(ordinal).decode()
            self.arch = runtime.core.cuda_device_get_arch(ordinal)
            self.sm_count = runtime.core.cuda_device_get_sm_count(ordinal)
            self.is_uva = runtime.core.cuda_device_is_uva(ordinal) > 0
            self.is_mempool_supported = runtime.core.cuda_device_is_mempool_supported(ordinal) > 0
            if platform.system() == "Linux":
                # Use None when IPC support cannot be determined
                ipc_support_api_query = runtime.core.cuda_device_is_ipc_supported(ordinal)
                self.is_ipc_supported = bool(ipc_support_api_query) if ipc_support_api_query >= 0 else None
            else:
                self.is_ipc_supported = False
            if warp.config.enable_mempools_at_init:
                # enable if supported
                self.is_mempool_enabled = self.is_mempool_supported
            else:
                # disable by default
                self.is_mempool_enabled = False

            uuid_buffer = (ctypes.c_char * 16)()
            runtime.core.cuda_device_get_uuid(ordinal, uuid_buffer)
            uuid_byte_str = bytes(uuid_buffer).hex()
            self.uuid = f"GPU-{uuid_byte_str[0:8]}-{uuid_byte_str[8:12]}-{uuid_byte_str[12:16]}-{uuid_byte_str[16:20]}-{uuid_byte_str[20:]}"

            pci_domain_id = runtime.core.cuda_device_get_pci_domain_id(ordinal)
            pci_bus_id = runtime.core.cuda_device_get_pci_bus_id(ordinal)
            pci_device_id = runtime.core.cuda_device_get_pci_device_id(ordinal)
            # This is (mis)named to correspond to the naming of cudaDeviceGetPCIBusId
            self.pci_bus_id = f"{pci_domain_id:08X}:{pci_bus_id:02X}:{pci_device_id:02X}"

            self.default_allocator = CudaDefaultAllocator(self)
            if self.is_mempool_supported:
                self.mempool_allocator = CudaMempoolAllocator(self)
            else:
                self.mempool_allocator = None

            # set current allocator
            if self.is_mempool_enabled:
                self.current_allocator = self.mempool_allocator
            else:
                self.current_allocator = self.default_allocator

            # check whether our NVRTC can generate CUBINs for this architecture
            self.is_cubin_supported = self.arch in runtime.nvrtc_supported_archs

            # initialize streams unless context acquisition is postponed
            if self._context is not None:
                self._init_streams()

            # TODO: add more device-specific dispatch functions
            self.memset = lambda ptr, value, size: runtime.core.memset_device(self.context, ptr, value, size)
            self.memtile = lambda ptr, src, srcsize, reps: runtime.core.memtile_device(
                self.context, ptr, src, srcsize, reps
            )

        else:
            raise RuntimeError(f"Invalid device ordinal ({ordinal})'")

    def get_allocator(self, pinned: bool = False):
        """Get the memory allocator for this device.

        Args:
            pinned: If ``True``, an allocator for pinned memory will be
              returned. Only applicable when this device is a CPU device.
        """
        if self.is_cuda:
            return self.current_allocator
        else:
            if pinned:
                return self.pinned_allocator
            else:
                return self.default_allocator

    def _init_streams(self):
        """Initializes the device's current stream and the device's null stream."""
        # create a stream for asynchronous work
        self.set_stream(Stream(self))

        # CUDA default stream for some synchronous operations
        self.null_stream = Stream(self, cuda_stream=None)

    @property
    def is_cpu(self) -> bool:
        """A boolean indicating whether the device is a CPU device."""
        return self.ordinal < 0

    @property
    def is_cuda(self) -> bool:
        """A boolean indicating whether the device is a CUDA device."""
        return self.ordinal >= 0

    @property
    def is_capturing(self) -> bool:
        """A boolean indicating whether this device's default stream is currently capturing a graph."""
        if self.is_cuda and self.stream is not None:
            # There is no CUDA API to check if graph capture was started on a device, so we
            # can't tell if a capture was started by external code on a different stream.
            # The best we can do is check whether a graph capture was started by Warp on this
            # device and whether the current stream is capturing.
            return self.captures or self.stream.is_capturing
        else:
            return False

    @property
    def context(self):
        """The context associated with the device."""
        if self._context is not None:
            return self._context
        elif self.is_primary:
            # acquire primary context on demand
            prev_context = runtime.core.cuda_context_get_current()
            self._context = self.runtime.core.cuda_device_get_primary_context(self.ordinal)
            if self._context is None:
                runtime.core.cuda_context_set_current(prev_context)
                raise RuntimeError(f"Failed to acquire primary context for device {self}")
            self.runtime.context_map[self._context] = self
            # initialize streams
            self._init_streams()
            runtime.core.cuda_context_set_current(prev_context)
        return self._context

    @property
    def has_context(self) -> bool:
        """A boolean indicating whether the device has a CUDA context associated with it."""
        return self._context is not None

    @property
    def stream(self) -> Stream:
        """The stream associated with a CUDA device.

        Raises:
            RuntimeError: The device is not a CUDA device.
        """
        if self.context:
            return self._stream
        else:
            raise RuntimeError(f"Device {self} is not a CUDA device")

    @stream.setter
    def stream(self, stream):
        self.set_stream(stream)

    def set_stream(self, stream: Stream, sync: bool = True) -> None:
        """Set the current stream for this CUDA device.

        The current stream will be used by default for all kernel launches and
        memory operations on this device.

        If this is an external stream, the caller is responsible for
        guaranteeing the lifetime of the stream.

        Consider using :class:`warp.ScopedStream` instead.

        Args:
            stream: The stream to set as this device's current stream.
            sync: If ``True``, then ``stream`` will perform a device-side
              synchronization with the device's previous current stream.
        """
        if self.is_cuda:
            if stream.device != self:
                raise RuntimeError(f"Stream from device {stream.device} cannot be used on device {self}")

            self.runtime.core.cuda_context_set_stream(self.context, stream.cuda_stream, int(sync))
            self._stream = stream
        else:
            raise RuntimeError(f"Device {self} is not a CUDA device")

    @property
    def has_stream(self) -> bool:
        """A boolean indicating whether the device has a stream associated with it."""
        return self._stream is not None

    @property
    def total_memory(self) -> int:
        """The total amount of device memory available in bytes.

        This function is currently only implemented for CUDA devices. 0 will be returned if called on a CPU device.
        """
        if self.is_cuda:
            total_mem = ctypes.c_size_t()
            self.runtime.core.cuda_device_get_memory_info(self.ordinal, None, ctypes.byref(total_mem))
            return total_mem.value
        else:
            # TODO: cpu
            return 0

    @property
    def free_memory(self) -> int:
        """The amount of memory on the device that is free according to the OS in bytes.

        This function is currently only implemented for CUDA devices. 0 will be returned if called on a CPU device.
        """
        if self.is_cuda:
            free_mem = ctypes.c_size_t()
            self.runtime.core.cuda_device_get_memory_info(self.ordinal, ctypes.byref(free_mem), None)
            return free_mem.value
        else:
            # TODO: cpu
            return 0

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
        # TODO: this function should be redesigned in terms of (device, resource).
        # - a device can access any resource on the same device
        # - a CUDA device can access pinned memory on the host
        # - a CUDA device can access regular allocations on a peer device if peer access is enabled
        # - a CUDA device can access mempool allocations on a peer device if mempool access is enabled
        other = self.runtime.get_device(other)
        if self.context == other.context:
            return True
        else:
            return False


""" Meta-type for arguments that can be resolved to a concrete Device.
"""
Devicelike = Union[Device, str, None]


class Graph:
    def __init__(self, device: Device, capture_id: int):
        self.device = device
        self.capture_id = capture_id
        self.module_execs: set[ModuleExec] = set()
        self.graph_exec: ctypes.c_void_p | None = None

        self.graph: ctypes.c_void_p | None = None
        self.has_conditional = (
            False  # Track if there are conditional nodes in the graph since they are not allowed in child graphs
        )

    def __del__(self):
        if not hasattr(self, "graph") or not hasattr(self, "device") or not self.graph:
            return

        # use CUDA context guard to avoid side effects during garbage collection
        with self.device.context_guard:
            runtime.core.cuda_graph_destroy(self.device.context, self.graph)
            if hasattr(self, "graph_exec") and self.graph_exec is not None:
                runtime.core.cuda_graph_exec_destroy(self.device.context, self.graph_exec)

    # retain executable CUDA modules used by this graph, which prevents them from being unloaded
    def retain_module_exec(self, module_exec: ModuleExec):
        self.module_execs.add(module_exec)


class Runtime:
    def __init__(self):
        if sys.version_info < (3, 9):
            warp.utils.warn(f"Python 3.9 or newer is recommended for running Warp, detected {sys.version_info}")

        bin_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin")

        if os.name == "nt":
            # Python >= 3.8 this method to add dll search paths
            os.add_dll_directory(bin_path)

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

        # maps capture ids to graphs
        self.captures = {}

        # setup c-types for warp.dll
        try:
            self.core.get_error_string.argtypes = []
            self.core.get_error_string.restype = ctypes.c_char_p
            self.core.set_error_output_enabled.argtypes = [ctypes.c_int]
            self.core.set_error_output_enabled.restype = None
            self.core.is_error_output_enabled.argtypes = []
            self.core.is_error_output_enabled.restype = ctypes.c_int

            self.core.alloc_host.argtypes = [ctypes.c_size_t]
            self.core.alloc_host.restype = ctypes.c_void_p
            self.core.alloc_pinned.argtypes = [ctypes.c_size_t]
            self.core.alloc_pinned.restype = ctypes.c_void_p
            self.core.alloc_device.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self.core.alloc_device.restype = ctypes.c_void_p
            self.core.alloc_device_default.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self.core.alloc_device_default.restype = ctypes.c_void_p
            self.core.alloc_device_async.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self.core.alloc_device_async.restype = ctypes.c_void_p

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
            self.core.free_device_default.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.free_device_default.restype = None
            self.core.free_device_async.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.free_device_async.restype = None

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
            self.core.memcpy_h2h.restype = ctypes.c_bool
            self.core.memcpy_h2d.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            self.core.memcpy_h2d.restype = ctypes.c_bool
            self.core.memcpy_d2h.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            self.core.memcpy_d2h.restype = ctypes.c_bool
            self.core.memcpy_d2d.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            self.core.memcpy_d2d.restype = ctypes.c_bool
            self.core.memcpy_p2p.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            self.core.memcpy_p2p.restype = ctypes.c_bool

            self.core.array_copy_host.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self.core.array_copy_host.restype = ctypes.c_bool
            self.core.array_copy_device.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self.core.array_copy_device.restype = ctypes.c_bool

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

            self.core.radix_sort_pairs_float_host.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
            self.core.radix_sort_pairs_float_device.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

            self.core.radix_sort_pairs_int64_host.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
            self.core.radix_sort_pairs_int64_device.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

            self.core.segmented_sort_pairs_int_host.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int,
            ]
            self.core.segmented_sort_pairs_int_device.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int,
            ]

            self.core.segmented_sort_pairs_float_host.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int,
            ]
            self.core.segmented_sort_pairs_float_device.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int,
            ]

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
            self.core.bvh_create_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

            self.core.bvh_create_device.restype = ctypes.c_uint64
            self.core.bvh_create_device.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]

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
                ctypes.c_int,
            ]

            self.core.mesh_destroy_host.argtypes = [ctypes.c_uint64]
            self.core.mesh_destroy_device.argtypes = [ctypes.c_uint64]

            self.core.mesh_refit_host.argtypes = [ctypes.c_uint64]
            self.core.mesh_refit_device.argtypes = [ctypes.c_uint64]

            self.core.mesh_set_points_host.argtypes = [ctypes.c_uint64, warp.types.array_t]
            self.core.mesh_set_points_device.argtypes = [ctypes.c_uint64, warp.types.array_t]

            self.core.mesh_set_velocities_host.argtypes = [ctypes.c_uint64, warp.types.array_t]
            self.core.mesh_set_velocities_device.argtypes = [ctypes.c_uint64, warp.types.array_t]

            self.core.hash_grid_create_host.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            self.core.hash_grid_create_host.restype = ctypes.c_uint64
            self.core.hash_grid_destroy_host.argtypes = [ctypes.c_uint64]
            self.core.hash_grid_update_host.argtypes = [ctypes.c_uint64, ctypes.c_float, ctypes.c_void_p]
            self.core.hash_grid_reserve_host.argtypes = [ctypes.c_uint64, ctypes.c_int]

            self.core.hash_grid_create_device.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            self.core.hash_grid_create_device.restype = ctypes.c_uint64
            self.core.hash_grid_destroy_device.argtypes = [ctypes.c_uint64]
            self.core.hash_grid_update_device.argtypes = [ctypes.c_uint64, ctypes.c_float, ctypes.c_void_p]
            self.core.hash_grid_reserve_device.argtypes = [ctypes.c_uint64, ctypes.c_int]

            self.core.volume_create_host.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_bool, ctypes.c_bool]
            self.core.volume_create_host.restype = ctypes.c_uint64
            self.core.volume_get_tiles_host.argtypes = [
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            self.core.volume_get_voxels_host.argtypes = [
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            self.core.volume_destroy_host.argtypes = [ctypes.c_uint64]

            self.core.volume_create_device.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_bool,
                ctypes.c_bool,
            ]
            self.core.volume_create_device.restype = ctypes.c_uint64
            self.core.volume_get_tiles_device.argtypes = [
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            self.core.volume_get_voxels_device.argtypes = [
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            self.core.volume_destroy_device.argtypes = [ctypes.c_uint64]

            self.core.volume_from_tiles_device.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_float * 9,
                ctypes.c_float * 3,
                ctypes.c_bool,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_char_p,
            ]
            self.core.volume_from_tiles_device.restype = ctypes.c_uint64
            self.core.volume_index_from_tiles_device.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_float * 9,
                ctypes.c_float * 3,
                ctypes.c_bool,
            ]
            self.core.volume_index_from_tiles_device.restype = ctypes.c_uint64
            self.core.volume_from_active_voxels_device.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_float * 9,
                ctypes.c_float * 3,
                ctypes.c_bool,
            ]
            self.core.volume_from_active_voxels_device.restype = ctypes.c_uint64

            self.core.volume_get_buffer_info.argtypes = [
                ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            self.core.volume_get_voxel_size.argtypes = [
                ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            self.core.volume_get_tile_and_voxel_count.argtypes = [
                ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            self.core.volume_get_grid_info.argtypes = [
                ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_float * 3,
                ctypes.c_float * 9,
                ctypes.c_char * 16,
            ]
            self.core.volume_get_grid_info.restype = ctypes.c_char_p
            self.core.volume_get_blind_data_count.argtypes = [
                ctypes.c_uint64,
            ]
            self.core.volume_get_blind_data_count.restype = ctypes.c_uint64
            self.core.volume_get_blind_data_info.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_char * 16,
            ]
            self.core.volume_get_blind_data_info.restype = ctypes.c_char_p

            bsr_matrix_from_triplets_argtypes = [
                ctypes.c_int,  # rows_per_bock
                ctypes.c_int,  # cols_per_blocks
                ctypes.c_int,  # row_count
                ctypes.c_int,  # tpl_nnz
                ctypes.POINTER(ctypes.c_int),  # tpl_rows
                ctypes.POINTER(ctypes.c_int),  # tpl_cols
                ctypes.c_void_p,  # tpl_values
                ctypes.c_bool,  # prune_numerical_zeros
                ctypes.c_bool,  # masked
                ctypes.POINTER(ctypes.c_int),  # bsr_offsets
                ctypes.POINTER(ctypes.c_int),  # bsr_columns
                ctypes.c_void_p,  # bsr_values
                ctypes.POINTER(ctypes.c_int),  # bsr_nnz
                ctypes.c_void_p,  # bsr_nnz_event
            ]

            self.core.bsr_matrix_from_triplets_float_host.argtypes = bsr_matrix_from_triplets_argtypes
            self.core.bsr_matrix_from_triplets_double_host.argtypes = bsr_matrix_from_triplets_argtypes
            self.core.bsr_matrix_from_triplets_float_device.argtypes = bsr_matrix_from_triplets_argtypes
            self.core.bsr_matrix_from_triplets_double_device.argtypes = bsr_matrix_from_triplets_argtypes

            bsr_transpose_argtypes = [
                ctypes.c_int,  # rows_per_bock
                ctypes.c_int,  # cols_per_blocks
                ctypes.c_int,  # row_count
                ctypes.c_int,  # col count
                ctypes.c_int,  # nnz
                ctypes.POINTER(ctypes.c_int),  # transposed_bsr_offsets
                ctypes.POINTER(ctypes.c_int),  # transposed_bsr_columns
                ctypes.c_void_p,  # bsr_values
                ctypes.POINTER(ctypes.c_int),  # transposed_bsr_offsets
                ctypes.POINTER(ctypes.c_int),  # transposed_bsr_columns
                ctypes.c_void_p,  # transposed_bsr_values
            ]
            self.core.bsr_transpose_float_host.argtypes = bsr_transpose_argtypes
            self.core.bsr_transpose_double_host.argtypes = bsr_transpose_argtypes
            self.core.bsr_transpose_float_device.argtypes = bsr_transpose_argtypes
            self.core.bsr_transpose_double_device.argtypes = bsr_transpose_argtypes

            self.core.is_cuda_enabled.argtypes = None
            self.core.is_cuda_enabled.restype = ctypes.c_int
            self.core.is_cuda_compatibility_enabled.argtypes = None
            self.core.is_cuda_compatibility_enabled.restype = ctypes.c_int
            self.core.is_mathdx_enabled.argtypes = None
            self.core.is_mathdx_enabled.restype = ctypes.c_int

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
            self.core.cuda_device_get_primary_context.argtypes = [ctypes.c_int]
            self.core.cuda_device_get_primary_context.restype = ctypes.c_void_p
            self.core.cuda_device_get_name.argtypes = [ctypes.c_int]
            self.core.cuda_device_get_name.restype = ctypes.c_char_p
            self.core.cuda_device_get_arch.argtypes = [ctypes.c_int]
            self.core.cuda_device_get_arch.restype = ctypes.c_int
            self.core.cuda_device_get_sm_count.argtypes = [ctypes.c_int]
            self.core.cuda_device_get_sm_count.restype = ctypes.c_int
            self.core.cuda_device_is_uva.argtypes = [ctypes.c_int]
            self.core.cuda_device_is_uva.restype = ctypes.c_int
            self.core.cuda_device_is_mempool_supported.argtypes = [ctypes.c_int]
            self.core.cuda_device_is_mempool_supported.restype = ctypes.c_int
            self.core.cuda_device_is_ipc_supported.argtypes = [ctypes.c_int]
            self.core.cuda_device_is_ipc_supported.restype = ctypes.c_int
            self.core.cuda_device_set_mempool_release_threshold.argtypes = [ctypes.c_int, ctypes.c_uint64]
            self.core.cuda_device_set_mempool_release_threshold.restype = ctypes.c_int
            self.core.cuda_device_get_mempool_release_threshold.argtypes = [ctypes.c_int]
            self.core.cuda_device_get_mempool_release_threshold.restype = ctypes.c_uint64
            self.core.cuda_device_get_mempool_used_mem_current.argtypes = [ctypes.c_int]
            self.core.cuda_device_get_mempool_used_mem_current.restype = ctypes.c_uint64
            self.core.cuda_device_get_mempool_used_mem_high.argtypes = [ctypes.c_int]
            self.core.cuda_device_get_mempool_used_mem_high.restype = ctypes.c_uint64
            self.core.cuda_device_get_memory_info.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_device_get_memory_info.restype = None
            self.core.cuda_device_get_uuid.argtypes = [ctypes.c_int, ctypes.c_char * 16]
            self.core.cuda_device_get_uuid.restype = None
            self.core.cuda_device_get_pci_domain_id.argtypes = [ctypes.c_int]
            self.core.cuda_device_get_pci_domain_id.restype = ctypes.c_int
            self.core.cuda_device_get_pci_bus_id.argtypes = [ctypes.c_int]
            self.core.cuda_device_get_pci_bus_id.restype = ctypes.c_int
            self.core.cuda_device_get_pci_device_id.argtypes = [ctypes.c_int]
            self.core.cuda_device_get_pci_device_id.restype = ctypes.c_int

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
            self.core.cuda_context_set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            self.core.cuda_context_set_stream.restype = None

            # peer access
            self.core.cuda_is_peer_access_supported.argtypes = [ctypes.c_int, ctypes.c_int]
            self.core.cuda_is_peer_access_supported.restype = ctypes.c_int
            self.core.cuda_is_peer_access_enabled.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_is_peer_access_enabled.restype = ctypes.c_int
            self.core.cuda_set_peer_access_enabled.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            self.core.cuda_set_peer_access_enabled.restype = ctypes.c_int
            self.core.cuda_is_mempool_access_enabled.argtypes = [ctypes.c_int, ctypes.c_int]
            self.core.cuda_is_mempool_access_enabled.restype = ctypes.c_int
            self.core.cuda_set_mempool_access_enabled.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            self.core.cuda_set_mempool_access_enabled.restype = ctypes.c_int

            # inter-process communication
            self.core.cuda_ipc_get_mem_handle.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
            self.core.cuda_ipc_get_mem_handle.restype = None
            self.core.cuda_ipc_open_mem_handle.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
            self.core.cuda_ipc_open_mem_handle.restype = ctypes.c_void_p
            self.core.cuda_ipc_close_mem_handle.argtypes = [ctypes.c_void_p]
            self.core.cuda_ipc_close_mem_handle.restype = None
            self.core.cuda_ipc_get_event_handle.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_char),
            ]
            self.core.cuda_ipc_get_event_handle.restype = None
            self.core.cuda_ipc_open_event_handle.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
            self.core.cuda_ipc_open_event_handle.restype = ctypes.c_void_p

            self.core.cuda_stream_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.core.cuda_stream_create.restype = ctypes.c_void_p
            self.core.cuda_stream_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_stream_destroy.restype = None
            self.core.cuda_stream_query.argtypes = [ctypes.c_void_p]
            self.core.cuda_stream_query.restype = ctypes.c_int
            self.core.cuda_stream_register.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_stream_register.restype = None
            self.core.cuda_stream_unregister.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_stream_unregister.restype = None
            self.core.cuda_stream_synchronize.argtypes = [ctypes.c_void_p]
            self.core.cuda_stream_synchronize.restype = None
            self.core.cuda_stream_wait_event.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_stream_wait_event.restype = None
            self.core.cuda_stream_wait_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_stream_wait_stream.restype = None
            self.core.cuda_stream_is_capturing.argtypes = [ctypes.c_void_p]
            self.core.cuda_stream_is_capturing.restype = ctypes.c_int
            self.core.cuda_stream_get_capture_id.argtypes = [ctypes.c_void_p]
            self.core.cuda_stream_get_capture_id.restype = ctypes.c_uint64
            self.core.cuda_stream_get_priority.argtypes = [ctypes.c_void_p]
            self.core.cuda_stream_get_priority.restype = ctypes.c_int

            self.core.cuda_event_create.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            self.core.cuda_event_create.restype = ctypes.c_void_p
            self.core.cuda_event_destroy.argtypes = [ctypes.c_void_p]
            self.core.cuda_event_destroy.restype = None
            self.core.cuda_event_query.argtypes = [ctypes.c_void_p]
            self.core.cuda_event_query.restype = ctypes.c_int
            self.core.cuda_event_record.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.core.cuda_event_record.restype = None
            self.core.cuda_event_synchronize.argtypes = [ctypes.c_void_p]
            self.core.cuda_event_synchronize.restype = None
            self.core.cuda_event_elapsed_time.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_event_elapsed_time.restype = ctypes.c_float

            self.core.cuda_graph_begin_capture.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            self.core.cuda_graph_begin_capture.restype = ctypes.c_bool
            self.core.cuda_graph_end_capture.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self.core.cuda_graph_end_capture.restype = ctypes.c_bool

            self.core.cuda_graph_create_exec.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self.core.cuda_graph_create_exec.restype = ctypes.c_bool

            self.core.cuda_graph_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_graph_launch.restype = ctypes.c_bool
            self.core.cuda_graph_exec_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_graph_exec_destroy.restype = ctypes.c_bool

            self.core.cuda_graph_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_graph_destroy.restype = ctypes.c_bool

            self.core.cuda_graph_insert_if_else.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self.core.cuda_graph_insert_if_else.restype = ctypes.c_bool

            self.core.cuda_graph_insert_while.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            self.core.cuda_graph_insert_while.restype = ctypes.c_bool

            self.core.cuda_graph_set_condition.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_uint64,
            ]
            self.core.cuda_graph_set_condition.restype = ctypes.c_bool

            self.core.cuda_graph_pause_capture.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self.core.cuda_graph_pause_capture.restype = ctypes.c_bool

            self.core.cuda_graph_resume_capture.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self.core.cuda_graph_resume_capture.restype = ctypes.c_bool

            self.core.cuda_graph_insert_child_graph.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self.core.cuda_graph_insert_child_graph.restype = ctypes.c_bool

            self.core.cuda_compile_program.argtypes = [
                ctypes.c_char_p,  # cuda_src
                ctypes.c_char_p,  # program name
                ctypes.c_int,  # arch
                ctypes.c_char_p,  # include_dir
                ctypes.c_int,  # num_cuda_include_dirs
                ctypes.POINTER(ctypes.c_char_p),  # cuda include dirs
                ctypes.c_bool,  # debug
                ctypes.c_bool,  # verbose
                ctypes.c_bool,  # verify_fp
                ctypes.c_bool,  # fast_math
                ctypes.c_bool,  # fuse_fp
                ctypes.c_bool,  # lineinfo
                ctypes.c_bool,  # compile_time_trace
                ctypes.c_char_p,  # output_path
                ctypes.c_size_t,  # num_ltoirs
                ctypes.POINTER(ctypes.c_char_p),  # ltoirs
                ctypes.POINTER(ctypes.c_size_t),  # ltoir_sizes
                ctypes.POINTER(ctypes.c_int),  # ltoir_input_types, each of type nvJitLinkInputType
            ]
            self.core.cuda_compile_program.restype = ctypes.c_size_t

            self.core.cuda_compile_fft.argtypes = [
                ctypes.c_char_p,  # lto
                ctypes.c_char_p,  # function name
                ctypes.c_int,  # num include dirs
                ctypes.POINTER(ctypes.c_char_p),  # include dirs
                ctypes.c_char_p,  # mathdx include dir
                ctypes.c_int,  # arch
                ctypes.c_int,  # size
                ctypes.c_int,  # ept
                ctypes.c_int,  # direction
                ctypes.c_int,  # precision
                ctypes.POINTER(ctypes.c_int),  # smem (out)
            ]
            self.core.cuda_compile_fft.restype = ctypes.c_bool

            self.core.cuda_compile_dot.argtypes = [
                ctypes.c_char_p,  # lto
                ctypes.c_char_p,  # function name
                ctypes.c_int,  # num include dirs
                ctypes.POINTER(ctypes.c_char_p),  # include dirs
                ctypes.c_char_p,  # mathdx include dir
                ctypes.c_int,  # arch
                ctypes.c_int,  # M
                ctypes.c_int,  # N
                ctypes.c_int,  # K
                ctypes.c_int,  # a_precision
                ctypes.c_int,  # b_precision
                ctypes.c_int,  # c_precision
                ctypes.c_int,  # type
                ctypes.c_int,  # a_arrangement
                ctypes.c_int,  # b_arrangement
                ctypes.c_int,  # c_arrangement
                ctypes.c_int,  # num threads
            ]
            self.core.cuda_compile_dot.restype = ctypes.c_bool

            self.core.cuda_compile_solver.argtypes = [
                ctypes.c_char_p,  # universal fatbin
                ctypes.c_char_p,  # lto
                ctypes.c_char_p,  # function name
                ctypes.c_int,  # num include dirs
                ctypes.POINTER(ctypes.c_char_p),  # include dirs
                ctypes.c_char_p,  # mathdx include dir
                ctypes.c_int,  # arch
                ctypes.c_int,  # M
                ctypes.c_int,  # N
                ctypes.c_int,  # precision
                ctypes.c_int,  # fill_mode
                ctypes.c_int,  # num threads
            ]
            self.core.cuda_compile_fft.restype = ctypes.c_bool

            self.core.cuda_load_module.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.core.cuda_load_module.restype = ctypes.c_void_p

            self.core.cuda_unload_module.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.core.cuda_unload_module.restype = None

            self.core.cuda_get_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
            self.core.cuda_get_kernel.restype = ctypes.c_void_p

            self.core.cuda_get_max_shared_memory.argtypes = [ctypes.c_void_p]
            self.core.cuda_get_max_shared_memory.restype = ctypes.c_int

            self.core.cuda_configure_kernel_shared_memory.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.core.cuda_configure_kernel_shared_memory.restype = ctypes.c_bool

            self.core.cuda_launch_kernel.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
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

            self.core.cuda_timing_begin.argtypes = [ctypes.c_int]
            self.core.cuda_timing_begin.restype = None
            self.core.cuda_timing_get_result_count.argtypes = []
            self.core.cuda_timing_get_result_count.restype = int
            self.core.cuda_timing_end.argtypes = []
            self.core.cuda_timing_end.restype = None

            self.core.graph_coloring.argtypes = [
                ctypes.c_int,
                warp.types.array_t,
                ctypes.c_int,
                warp.types.array_t,
            ]
            self.core.graph_coloring.restype = ctypes.c_int

            self.core.balance_coloring.argtypes = [
                ctypes.c_int,
                warp.types.array_t,
                ctypes.c_int,
                ctypes.c_float,
                warp.types.array_t,
            ]
            self.core.balance_coloring.restype = ctypes.c_float

            self.core.init.restype = ctypes.c_int

        except AttributeError as e:
            raise RuntimeError(f"Setting C-types for {warp_lib} failed. It may need rebuilding.") from e

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

        self.is_cuda_enabled = bool(self.core.is_cuda_enabled())
        self.is_cuda_compatibility_enabled = bool(self.core.is_cuda_compatibility_enabled())

        self.toolkit_version = None  # CTK version used to build the core lib
        self.driver_version = None  # installed driver version
        self.min_driver_version = None  # minimum required driver version

        self.cuda_devices = []
        self.cuda_primary_devices = []

        cuda_device_count = 0

        if self.is_cuda_enabled:
            # get CUDA Toolkit and driver versions
            toolkit_version = self.core.cuda_toolkit_version()
            driver_version = self.core.cuda_driver_version()

            # save versions as tuples, e.g., (12, 4)
            self.toolkit_version = (toolkit_version // 1000, (toolkit_version % 1000) // 10)
            self.driver_version = (driver_version // 1000, (driver_version % 1000) // 10)

            # determine minimum required driver version
            if self.is_cuda_compatibility_enabled:
                # we can rely on minor version compatibility, but 11.4 is the absolute minimum required from the driver
                if self.toolkit_version[0] > 11:
                    self.min_driver_version = (self.toolkit_version[0], 0)
                else:
                    self.min_driver_version = (11, 4)
            else:
                # we can't rely on minor version compatibility, so the driver can't be older than the toolkit
                self.min_driver_version = self.toolkit_version

            # determine if the installed driver is sufficient
            if self.driver_version >= self.min_driver_version:
                # get all architectures supported by NVRTC
                num_archs = self.core.nvrtc_supported_arch_count()
                if num_archs > 0:
                    archs = (ctypes.c_int * num_archs)()
                    self.core.nvrtc_supported_archs(archs)
                    self.nvrtc_supported_archs = set(archs)
                else:
                    self.nvrtc_supported_archs = set()

                # get CUDA device count
                cuda_device_count = self.core.cuda_device_get_count()

                # register primary CUDA devices
                for i in range(cuda_device_count):
                    alias = f"cuda:{i}"
                    device = Device(self, alias, ordinal=i, is_primary=True)
                    self.cuda_devices.append(device)
                    self.cuda_primary_devices.append(device)
                    self.device_map[alias] = device

                # count known non-primary contexts on each physical device so we can
                # give them reasonable aliases (e.g., "cuda:0.0", "cuda:0.1")
                self.cuda_custom_context_count = [0] * cuda_device_count

        # set default device
        if cuda_device_count > 0:
            # stick with the current cuda context, if one is bound
            initial_context = self.core.cuda_context_get_current()
            if initial_context is not None:
                self.set_default_device("cuda")
                # if this is a non-primary context that was just registered, update the device count
                cuda_device_count = len(self.cuda_devices)
            else:
                self.set_default_device("cuda:0")

            # the minimum PTX architecture that supports all of Warp's features
            self.default_ptx_arch = 75

            # Update the default PTX architecture based on devices present in the system.
            # Use the lowest architecture among devices that meet the minimum architecture requirement.
            # Devices below the required minimum will use the highest architecture they support.
            eligible_archs = [d.arch for d in self.cuda_devices if d.arch >= self.default_ptx_arch]
            if eligible_archs:
                self.default_ptx_arch = min(eligible_archs)
        else:
            # CUDA not available
            self.set_default_device("cpu")
            self.default_ptx_arch = None

        # initialize kernel cache
        warp.build.init_kernel_cache(warp.config.kernel_cache_dir)

        # global tape
        self.tape = None

        # print device and version information
        if not warp.config.quiet:
            greeting = []

            greeting.append(f"Warp {warp.config.version} initialized:")

            # Add git commit hash to greeting if available
            if warp.config._git_commit_hash is not None:
                greeting.append(f"   Git commit: {warp.config._git_commit_hash}")

            if cuda_device_count > 0:
                # print CUDA version info
                greeting.append(
                    f"   CUDA Toolkit {self.toolkit_version[0]}.{self.toolkit_version[1]}, Driver {self.driver_version[0]}.{self.driver_version[1]}"
                )
            else:
                # briefly explain lack of CUDA devices
                if not self.is_cuda_enabled:
                    # Warp was compiled without CUDA support
                    greeting.append("   CUDA not enabled in this build")
                elif self.driver_version < self.min_driver_version:
                    # insufficient CUDA driver version
                    greeting.append(
                        f"   CUDA Toolkit {self.toolkit_version[0]}.{self.toolkit_version[1]}, Driver {self.driver_version[0]}.{self.driver_version[1]}"
                        " (insufficient CUDA driver version!)"
                    )
                else:
                    # CUDA is supported, but no devices are available
                    greeting.append("   CUDA devices not available")
            greeting.append("   Devices:")
            alias_str = f'"{self.cpu_device.alias}"'
            name_str = f'"{self.cpu_device.name}"'
            greeting.append(f"     {alias_str:10s} : {name_str}")
            for cuda_device in self.cuda_devices:
                alias_str = f'"{cuda_device.alias}"'
                if cuda_device.is_primary:
                    name_str = f'"{cuda_device.name}"'
                    arch_str = f"sm_{cuda_device.arch}"
                    mem_str = f"{cuda_device.total_memory / 1024 / 1024 / 1024:.0f} GiB"
                    if cuda_device.is_mempool_supported:
                        if cuda_device.is_mempool_enabled:
                            mempool_str = "mempool enabled"
                        else:
                            mempool_str = "mempool supported"
                    else:
                        mempool_str = "mempool not supported"
                    greeting.append(f"     {alias_str:10s} : {name_str} ({mem_str}, {arch_str}, {mempool_str})")
                else:
                    primary_alias_str = f'"{self.cuda_primary_devices[cuda_device.ordinal].alias}"'
                    greeting.append(f"     {alias_str:10s} : Non-primary context on device {primary_alias_str}")
            if cuda_device_count > 1:
                # check peer access support
                access_matrix = []
                all_accessible = True
                none_accessible = True
                for i in range(cuda_device_count):
                    target_device = self.cuda_devices[i]
                    access_vector = []
                    for j in range(cuda_device_count):
                        if i == j:
                            access_vector.append(1)
                        else:
                            peer_device = self.cuda_devices[j]
                            can_access = self.core.cuda_is_peer_access_supported(
                                target_device.ordinal, peer_device.ordinal
                            )
                            access_vector.append(can_access)
                            all_accessible = all_accessible and can_access
                            none_accessible = none_accessible and not can_access
                    access_matrix.append(access_vector)
                greeting.append("   CUDA peer access:")
                if all_accessible:
                    greeting.append("     Supported fully (all-directional)")
                elif none_accessible:
                    greeting.append("     Not supported")
                else:
                    greeting.append("     Supported partially (see access matrix)")
                    # print access matrix
                    for i in range(cuda_device_count):
                        alias_str = f'"{self.cuda_devices[i].alias}"'
                        greeting.append(f"     {alias_str:10s} : {access_matrix[i]}")
            greeting.append("   Kernel cache:")
            greeting.append(f"     {warp.config.kernel_cache_dir}")

            print("\n".join(greeting))

        if cuda_device_count > 0:
            # ensure initialization did not change the initial context (e.g. querying available memory)
            self.core.cuda_context_set_current(initial_context)

            # detect possible misconfiguration of the system
            devices_without_uva = []
            devices_without_mempool = []
            for cuda_device in self.cuda_primary_devices:
                if not cuda_device.is_uva:
                    devices_without_uva.append(cuda_device)
                if not cuda_device.is_mempool_supported:
                    devices_without_mempool.append(cuda_device)

            if devices_without_uva:
                # This should not happen on any system officially supported by Warp.  UVA is not available
                # on 32-bit Windows, which we don't support.  Nonetheless, we should check and report a
                # warning out of abundance of caution.  It may help with debugging a broken VM setup etc.
                warp.utils.warn(
                    f"\n   Support for Unified Virtual Addressing (UVA) was not detected on devices {devices_without_uva}."
                )
            if devices_without_mempool:
                warp.utils.warn(
                    f"\n   Support for CUDA memory pools was not detected on devices {devices_without_mempool}."
                    "\n   This prevents memory allocations in CUDA graphs and may result in poor performance."
                    "\n   Is the UVM driver enabled?"
                )

        elif self.is_cuda_enabled:
            # Report a warning about insufficient driver version.  The warning should appear even in quiet mode
            # when the greeting message is suppressed.  Also try to provide guidance for resolving the situation.
            if self.driver_version < self.min_driver_version:
                msg = []
                msg.append("\n   Insufficient CUDA driver version.")
                msg.append(
                    f"The minimum required CUDA driver version is {self.min_driver_version[0]}.{self.min_driver_version[1]}, "
                    f"but the installed CUDA driver version is {self.driver_version[0]}.{self.driver_version[1]}."
                )
                msg.append("Visit https://github.com/NVIDIA/warp/blob/main/README.md#installing for guidance.")
                warp.utils.warn("\n   ".join(msg))

    def get_error_string(self):
        return self.core.get_error_string().decode("utf-8")

    def load_dll(self, dll_path):
        try:
            dll = ctypes.CDLL(dll_path, winmode=0)
        except OSError as e:
            if "GLIBCXX" in str(e):
                raise RuntimeError(
                    f"Failed to load the shared library '{dll_path}'.\n"
                    "The execution environment's libstdc++ runtime is older than the version the Warp library was built for.\n"
                    "See https://nvidia.github.io/warp/installation.html#conda-environments for details."
                ) from e
            else:
                raise RuntimeError(f"Failed to load the shared library '{dll_path}'") from e
        return dll

    def get_device(self, ident: Devicelike = None) -> Device:
        # special cases
        if type(ident) is Device:
            return ident
        elif ident is None:
            return self.default_device

        # string lookup
        device = self.device_map.get(ident)
        if device is not None:
            return device
        elif ident == "cuda":
            return self.get_current_cuda_device()

        raise ValueError(f"Invalid device identifier: {ident}")

    def set_default_device(self, ident: Devicelike) -> None:
        self.default_device = self.get_device(ident)

    def get_current_cuda_device(self) -> Device:
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
                ordinal = self.core.cuda_context_get_device_ordinal(current_context)
                alias = f"cuda:{ordinal}.{self.cuda_custom_context_count[ordinal]}"
                self.cuda_custom_context_count[ordinal] += 1
                return self.map_cuda_device(alias, current_context)
        elif self.default_device.is_cuda:
            return self.default_device
        elif self.cuda_devices:
            return self.cuda_devices[0]
        else:
            # CUDA is not available
            if not self.is_cuda_enabled:
                raise RuntimeError('"cuda" device requested but this build of Warp does not support CUDA')
            else:
                raise RuntimeError('"cuda" device requested but CUDA is not supported by the hardware or driver')

    def rename_device(self, device, alias) -> Device:
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

    def unmap_cuda_device(self, alias) -> None:
        device = self.device_map.get(alias)

        # make sure the alias refers to a CUDA device
        if device is None or not device.is_cuda:
            raise RuntimeError(f"Invalid CUDA device alias '{alias}'")

        del self.device_map[alias]
        del self.context_map[device.context]
        self.cuda_devices.remove(device)

    def verify_cuda_device(self, device: Devicelike = None) -> None:
        if warp.config.verify_cuda:
            device = runtime.get_device(device)
            if not device.is_cuda:
                return

            err = self.core.cuda_context_check(device.context)
            if err != 0:
                raise RuntimeError(f"CUDA error detected: {err}")


# global entry points
def is_cpu_available() -> bool:
    init()

    return runtime.llvm is not None


def is_cuda_available() -> bool:
    return get_cuda_device_count() > 0


def is_device_available(device: Device) -> bool:
    return device in get_devices()


def is_cuda_driver_initialized() -> bool:
    """Returns ``True`` if the CUDA driver is initialized.

    This is a stricter test than ``is_cuda_available()`` since a CUDA driver
    call to ``cuCtxGetCurrent`` is made, and the result is compared to
    `CUDA_SUCCESS`. Note that `CUDA_SUCCESS` is returned by ``cuCtxGetCurrent``
    even if there is no context bound to the calling CPU thread.

    This can be helpful in cases in which ``cuInit()`` was called before a fork.
    """
    init()

    return runtime.core.cuda_driver_is_initialized()


def get_devices() -> list[Device]:
    """Returns a list of devices supported in this environment."""

    init()

    devices = []
    if is_cpu_available():
        devices.append(runtime.cpu_device)
    for cuda_device in runtime.cuda_devices:
        devices.append(cuda_device)
    return devices


def get_cuda_device_count() -> int:
    """Returns the number of CUDA devices supported in this environment."""

    init()

    return len(runtime.cuda_devices)


def get_cuda_device(ordinal: int | None = None) -> Device:
    """Returns the CUDA device with the given ordinal or the current CUDA device if ordinal is None."""

    init()

    if ordinal is None:
        return runtime.get_current_cuda_device()
    else:
        return runtime.cuda_devices[ordinal]


def get_cuda_devices() -> list[Device]:
    """Returns a list of CUDA devices supported in this environment."""

    init()

    return runtime.cuda_devices


def get_preferred_device() -> Device:
    """Returns the preferred compute device, ``cuda:0`` if available and ``cpu`` otherwise."""

    init()

    if is_cuda_available():
        return runtime.cuda_devices[0]
    elif is_cpu_available():
        return runtime.cpu_device
    else:
        return None


def get_device(ident: Devicelike = None) -> Device:
    """Returns the device identified by the argument."""

    init()

    return runtime.get_device(ident)


def set_device(ident: Devicelike) -> None:
    """Sets the default device identified by the argument."""

    init()

    device = runtime.get_device(ident)
    runtime.set_default_device(device)
    device.make_current()


def map_cuda_device(alias: str, context: ctypes.c_void_p | None = None) -> Device:
    """Assign a device alias to a CUDA context.

    This function can be used to create a wp.Device for an external CUDA context.
    If a wp.Device already exists for the given context, it's alias will change to the given value.

    Args:
        alias: A unique string to identify the device.
        context: A CUDA context pointer (CUcontext).  If None, the currently bound CUDA context will be used.

    Returns:
        The associated wp.Device.
    """

    init()

    return runtime.map_cuda_device(alias, context)


def unmap_cuda_device(alias: str) -> None:
    """Remove a CUDA device with the given alias."""

    init()

    runtime.unmap_cuda_device(alias)


def is_mempool_supported(device: Devicelike) -> bool:
    """Check if CUDA memory pool allocators are available on the device.

    Parameters:
        device: The :class:`Device <warp.context.Device>` or device identifier
          for which the query is to be performed.
          If ``None``, the default device will be used.
    """

    init()

    device = runtime.get_device(device)

    return device.is_mempool_supported


def is_mempool_enabled(device: Devicelike) -> bool:
    """Check if CUDA memory pool allocators are enabled on the device.

    Parameters:
        device: The :class:`Device <warp.context.Device>` or device identifier
          for which the query is to be performed.
          If ``None``, the default device will be used.
    """

    init()

    device = runtime.get_device(device)

    return device.is_mempool_enabled


def set_mempool_enabled(device: Devicelike, enable: bool) -> None:
    """Enable or disable CUDA memory pool allocators on the device.

    Pooled allocators are typically faster and allow allocating memory during graph capture.

    They should generally be enabled, but there is a rare caveat.  Copying data between different GPUs
    may fail during graph capture if the memory was allocated using pooled allocators and memory pool
    access is not enabled between the two GPUs.  This is an internal CUDA limitation that is not related
    to Warp.  The preferred solution is to enable memory pool access using :func:`set_mempool_access_enabled`.
    If peer access is not supported, then the default CUDA allocators must be used to pre-allocate the memory
    prior to graph capture.

    Parameters:
        device: The :class:`Device <warp.context.Device>` or device identifier
          for which the operation is to be performed.
          If ``None``, the default device will be used.
    """

    init()

    device = runtime.get_device(device)

    if device.is_cuda:
        if enable:
            if not device.is_mempool_supported:
                raise RuntimeError(f"Device {device} does not support memory pools")
            device.current_allocator = device.mempool_allocator
            device.is_mempool_enabled = True
        else:
            device.current_allocator = device.default_allocator
            device.is_mempool_enabled = False
    else:
        if enable:
            raise ValueError("Memory pools are only supported on CUDA devices")


def set_mempool_release_threshold(device: Devicelike, threshold: int | float) -> None:
    """Set the CUDA memory pool release threshold on the device.

    This is the amount of reserved memory to hold onto before trying to release memory back to the OS.
    When more than this amount of bytes is held by the memory pool, the allocator will try to release
    memory back to the OS on the next call to stream, event, or device synchronize.

    Values between 0 and 1 are interpreted as fractions of available memory.  For example, 0.5 means
    half of the device's physical memory.  Greater values are interpreted as an absolute number of bytes.
    For example, 1024**3 means one GiB of memory.

    Parameters:
        device: The :class:`Device <warp.context.Device>` or device identifier
          for which the operation is to be performed.
          If ``None``, the default device will be used.
        threshold: An integer representing a number of bytes, or a ``float`` between 0 and 1,
          specifying the desired release threshold.

    Raises:
        ValueError: If ``device`` is not a CUDA device.
        RuntimeError: If ``device`` is a CUDA device, but does not support memory pools.
        RuntimeError: Failed to set the memory pool release threshold.
    """

    init()

    device = runtime.get_device(device)

    if not device.is_cuda:
        raise ValueError("Memory pools are only supported on CUDA devices")

    if not device.is_mempool_supported:
        raise RuntimeError(f"Device {device} does not support memory pools")

    if threshold < 0:
        threshold = 0
    elif threshold > 0 and threshold <= 1:
        threshold = int(threshold * device.total_memory)

    if not runtime.core.cuda_device_set_mempool_release_threshold(device.ordinal, threshold):
        raise RuntimeError(f"Failed to set memory pool release threshold for device {device}")


def get_mempool_release_threshold(device: Devicelike = None) -> int:
    """Get the CUDA memory pool release threshold on the device.

    Parameters:
        device: The :class:`Device <warp.context.Device>` or device identifier
          for which the query is to be performed.
          If ``None``, the default device will be used.

    Returns:
        The memory pool release threshold in bytes.

    Raises:
        ValueError: If ``device`` is not a CUDA device.
        RuntimeError: If ``device`` is a CUDA device, but does not support memory pools.
    """

    init()

    device = runtime.get_device(device)

    if not device.is_cuda:
        raise ValueError("Memory pools are only supported on CUDA devices")

    if not device.is_mempool_supported:
        raise RuntimeError(f"Device {device} does not support memory pools")

    return runtime.core.cuda_device_get_mempool_release_threshold(device.ordinal)


def get_mempool_used_mem_current(device: Devicelike = None) -> int:
    """Get the amount of memory from the device's memory pool that is currently in use by the application.

    Parameters:
        device: The :class:`Device <warp.context.Device>` or device identifier
          for which the query is to be performed.
          If ``None``, the default device will be used.

    Returns:
        The amount of memory used in bytes.

    Raises:
        ValueError: If ``device`` is not a CUDA device.
        RuntimeError: If ``device`` is a CUDA device, but does not support memory pools.
    """

    init()

    device = runtime.get_device(device)

    if not device.is_cuda:
        raise ValueError("Memory pools are only supported on CUDA devices")

    if not device.is_mempool_supported:
        raise RuntimeError(f"Device {device} does not support memory pools")

    return runtime.core.cuda_device_get_mempool_used_mem_current(device.ordinal)


def get_mempool_used_mem_high(device: Devicelike = None) -> int:
    """Get the application's memory usage high-water mark from the device's CUDA memory pool.

    Parameters:
        device: The :class:`Device <warp.context.Device>` or device identifier
          for which the query is to be performed.
          If ``None``, the default device will be used.

    Returns:
        The high-water mark of memory used from the memory pool in bytes.

    Raises:
        ValueError: If ``device`` is not a CUDA device.
        RuntimeError: If ``device`` is a CUDA device, but does not support memory pools.
    """

    init()

    device = runtime.get_device(device)

    if not device.is_cuda:
        raise ValueError("Memory pools are only supported on CUDA devices")

    if not device.is_mempool_supported:
        raise RuntimeError(f"Device {device} does not support memory pools")

    return runtime.core.cuda_device_get_mempool_used_mem_high(device.ordinal)


def is_peer_access_supported(target_device: Devicelike, peer_device: Devicelike) -> bool:
    """Check if `peer_device` can directly access the memory of `target_device` on this system.

    This applies to memory allocated using default CUDA allocators.  For memory allocated using
    CUDA pooled allocators, use :func:`is_mempool_access_supported()`.

    Returns:
        A Boolean value indicating if this peer access is supported by the system.
    """

    init()

    target_device = runtime.get_device(target_device)
    peer_device = runtime.get_device(peer_device)

    if not target_device.is_cuda or not peer_device.is_cuda:
        return False

    return bool(runtime.core.cuda_is_peer_access_supported(target_device.ordinal, peer_device.ordinal))


def is_peer_access_enabled(target_device: Devicelike, peer_device: Devicelike) -> bool:
    """Check if `peer_device` can currently access the memory of `target_device`.

    This applies to memory allocated using default CUDA allocators.  For memory allocated using
    CUDA pooled allocators, use :func:`is_mempool_access_enabled()`.

    Returns:
        A Boolean value indicating if this peer access is currently enabled.
    """

    init()

    target_device = runtime.get_device(target_device)
    peer_device = runtime.get_device(peer_device)

    if not target_device.is_cuda or not peer_device.is_cuda:
        return False

    return bool(runtime.core.cuda_is_peer_access_enabled(target_device.context, peer_device.context))


def set_peer_access_enabled(target_device: Devicelike, peer_device: Devicelike, enable: bool) -> None:
    """Enable or disable direct access from `peer_device` to the memory of `target_device`.

    Enabling peer access can improve the speed of peer-to-peer memory transfers, but can have
    a negative impact on memory consumption and allocation performance.

    This applies to memory allocated using default CUDA allocators.  For memory allocated using
    CUDA pooled allocators, use :func:`set_mempool_access_enabled()`.
    """

    init()

    target_device = runtime.get_device(target_device)
    peer_device = runtime.get_device(peer_device)

    if not target_device.is_cuda or not peer_device.is_cuda:
        if enable:
            raise ValueError("Peer access is only supported between CUDA devices")
        else:
            return

    if not is_peer_access_supported(target_device, peer_device):
        if enable:
            raise RuntimeError(f"Device {peer_device} cannot access device {target_device}")
        else:
            return

    if not runtime.core.cuda_set_peer_access_enabled(target_device.context, peer_device.context, int(enable)):
        action = "enable" if enable else "disable"
        raise RuntimeError(f"Failed to {action} peer access from device {peer_device} to device {target_device}")


def is_mempool_access_supported(target_device: Devicelike, peer_device: Devicelike) -> bool:
    """Check if `peer_device` can directly access the memory pool of `target_device`.

    If mempool access is possible, it can be managed using :func:`set_mempool_access_enabled()`
    and :func:`is_mempool_access_enabled()`.

    Returns:
        A Boolean value indicating if this memory pool access is supported by the system.
    """

    init()

    target_device = runtime.get_device(target_device)
    peer_device = runtime.get_device(peer_device)

    return target_device.is_mempool_supported and is_peer_access_supported(target_device, peer_device)


def is_mempool_access_enabled(target_device: Devicelike, peer_device: Devicelike) -> bool:
    """Check if `peer_device` can currently access the memory pool of `target_device`.

    This applies to memory allocated using CUDA pooled allocators.  For memory allocated using
    default CUDA allocators, use :func:`is_peer_access_enabled()`.

    Returns:
        A Boolean value indicating if this peer access is currently enabled.
    """

    init()

    target_device = runtime.get_device(target_device)
    peer_device = runtime.get_device(peer_device)

    if not peer_device.is_cuda or not target_device.is_cuda or not target_device.is_mempool_supported:
        return False

    return bool(runtime.core.cuda_is_mempool_access_enabled(target_device.ordinal, peer_device.ordinal))


def set_mempool_access_enabled(target_device: Devicelike, peer_device: Devicelike, enable: bool) -> None:
    """Enable or disable access from `peer_device` to the memory pool of `target_device`.

    This applies to memory allocated using CUDA pooled allocators.  For memory allocated using
    default CUDA allocators, use :func:`set_peer_access_enabled()`.
    """

    init()

    target_device = runtime.get_device(target_device)
    peer_device = runtime.get_device(peer_device)

    if not target_device.is_cuda or not peer_device.is_cuda:
        if enable:
            raise ValueError("Memory pool access is only supported between CUDA devices")
        else:
            return

    if not target_device.is_mempool_supported:
        if enable:
            raise RuntimeError(f"Device {target_device} does not support memory pools")
        else:
            return

    if not is_peer_access_supported(target_device, peer_device):
        if enable:
            raise RuntimeError(f"Device {peer_device} cannot access device {target_device}")
        else:
            return

    if not runtime.core.cuda_set_mempool_access_enabled(target_device.ordinal, peer_device.ordinal, int(enable)):
        action = "enable" if enable else "disable"
        raise RuntimeError(f"Failed to {action} memory pool access from device {peer_device} to device {target_device}")


def get_stream(device: Devicelike = None) -> Stream:
    """Return the stream currently used by the given device.

    Args:
        device: An optional :class:`Device` instance or device alias
          (e.g. "cuda:0") for which the current stream will be returned.
          If ``None``, the default device will be used.

    Raises:
        RuntimeError: The device is not a CUDA device.
    """

    return get_device(device).stream


def set_stream(stream: Stream, device: Devicelike = None, sync: bool = False) -> None:
    """Convenience function for calling :meth:`Device.set_stream` on the given ``device``.

    Args:
        device: An optional :class:`Device` instance or device alias
          (e.g. "cuda:0") for which the current stream is to be replaced with
          ``stream``. If ``None``, the default device will be used.
        stream: The stream to set as this device's current stream.
        sync: If ``True``, then ``stream`` will perform a device-side
          synchronization with the device's previous current stream.
    """

    get_device(device).set_stream(stream, sync=sync)


def record_event(event: Event | None = None):
    """Convenience function for calling :meth:`Stream.record_event` on the current stream.

    Args:
        event: :class:`Event` instance to record. If ``None``, a new :class:`Event`
          instance will be created.

    Returns:
        The recorded event.
    """

    return get_stream().record_event(event)


def wait_event(event: Event):
    """Convenience function for calling :meth:`Stream.wait_event` on the current stream.

    Args:
        event: :class:`Event` instance to wait for.
    """

    get_stream().wait_event(event)


def get_event_elapsed_time(start_event: Event, end_event: Event, synchronize: bool = True):
    """Get the elapsed time between two recorded events.

    Both events must have been previously recorded with
    :func:`~warp.record_event()` or :meth:`warp.Stream.record_event()`.

    If ``synchronize`` is False, the caller must ensure that device execution has reached ``end_event``
    prior to calling ``get_event_elapsed_time()``.

    Args:
        start_event: The start event.
        end_event: The end event.
        synchronize: Whether Warp should synchronize on the ``end_event``.

    Returns:
        The elapsed time in milliseconds with a resolution about 0.5 ms.
    """

    # ensure the end_event is reached
    if synchronize:
        synchronize_event(end_event)

    return runtime.core.cuda_event_elapsed_time(start_event.cuda_event, end_event.cuda_event)


def wait_stream(other_stream: Stream, event: Event | None = None):
    """Convenience function for calling :meth:`Stream.wait_stream` on the current stream.

    Args:
        other_stream: The stream on which the calling stream will wait for
          previously issued commands to complete before executing subsequent
          commands.
        event: An optional :class:`Event` instance that will be used to
          record an event onto ``other_stream``. If ``None``, an internally
          managed :class:`Event` instance will be used.
    """

    get_stream().wait_stream(other_stream, event=event)


class RegisteredGLBuffer:
    """
    Helper class to register a GL buffer with CUDA so that it can be mapped to a Warp array.

    Example usage::

        import warp as wp
        import numpy as np
        from pyglet.gl import *

        wp.init()

        # create a GL buffer
        gl_buffer_id = GLuint()
        glGenBuffers(1, gl_buffer_id)

        # copy some data to the GL buffer
        glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_id)
        gl_data = np.arange(1024, dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, gl_data.nbytes, gl_data.ctypes.data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # register the GL buffer with CUDA
        cuda_gl_buffer = wp.RegisteredGLBuffer(gl_buffer_id)

        # map the GL buffer to a Warp array
        arr = cuda_gl_buffer.map(dtype=wp.float32, shape=(1024,))
        # launch a Warp kernel to manipulate or read the array
        wp.launch(my_kernel, dim=1024, inputs=[arr])
        # unmap the GL buffer
        cuda_gl_buffer.unmap()
    """

    NONE = 0x00
    """
    Flag that specifies no hints about how this resource will be used.
    It is therefore assumed that this resource will be
    read from and written to by CUDA. This is the default value.
    """

    READ_ONLY = 0x01
    """
    Flag that specifies that CUDA will not write to this resource.
    """

    WRITE_DISCARD = 0x02
    """
    Flag that specifies that CUDA will not read from this resource and will write over the
    entire contents of the resource, so none of the data previously
    stored in the resource will be preserved.
    """

    __fallback_warning_shown = False

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.resource = None
        return instance

    def __init__(self, gl_buffer_id: int, device: Devicelike = None, flags: int = NONE, fallback_to_copy: bool = True):
        """
        Args:
            gl_buffer_id: The OpenGL buffer id (GLuint).
            device: The device to register the buffer with.  If None, the current device will be used.
            flags: A combination of the flags constants :attr:`NONE`, :attr:`READ_ONLY`, and :attr:`WRITE_DISCARD`.
            fallback_to_copy: If True and CUDA/OpenGL interop is not available, fall back to copy operations between the Warp array and the OpenGL buffer. Otherwise, a ``RuntimeError`` will be raised.

        Note:

            The ``fallback_to_copy`` option (to use copy operations if CUDA graphics interop functionality is not available) requires pyglet version 2.0 or later. Install via ``pip install pyglet==2.*``.
        """
        self.gl_buffer_id = gl_buffer_id
        self.device = get_device(device)
        self.context = self.device.context
        self.flags = flags
        self.fallback_to_copy = fallback_to_copy
        self.resource = runtime.core.cuda_graphics_register_gl_buffer(self.context, gl_buffer_id, flags)
        if self.resource is None:
            if self.fallback_to_copy:
                self.warp_buffer = None
                self.warp_buffer_cpu = None
                if not RegisteredGLBuffer.__fallback_warning_shown:
                    warp.utils.warn(
                        "Could not register GL buffer since CUDA/OpenGL interoperability is not available. Falling back to copy operations between the Warp array and the OpenGL buffer.",
                    )
                    RegisteredGLBuffer.__fallback_warning_shown = True
            else:
                raise RuntimeError(f"Failed to register OpenGL buffer {gl_buffer_id} with CUDA")

    def __del__(self):
        if not self.resource:
            return

        # use CUDA context guard to avoid side effects during garbage collection
        with self.device.context_guard:
            runtime.core.cuda_graphics_unregister_resource(self.context, self.resource)

    def map(self, dtype, shape) -> warp.array:
        """Map the OpenGL buffer to a Warp array.

        Args:
            dtype: The type of each element in the array.
            shape: The shape of the array.

        Returns:
            A Warp array object representing the mapped OpenGL buffer.
        """
        if self.resource is not None:
            runtime.core.cuda_graphics_map(self.context, self.resource)
            ptr = ctypes.c_uint64(0)
            size = ctypes.c_size_t(0)
            runtime.core.cuda_graphics_device_ptr_and_size(
                self.context, self.resource, ctypes.byref(ptr), ctypes.byref(size)
            )
            return warp.array(ptr=ptr.value, dtype=dtype, shape=shape, device=self.device)
        elif self.fallback_to_copy:
            if self.warp_buffer is None or self.warp_buffer.dtype != dtype or self.warp_buffer.shape != shape:
                self.warp_buffer = warp.empty(shape, dtype, device=self.device)
                self.warp_buffer_cpu = warp.empty(shape, dtype, device="cpu", pinned=True)

            if self.flags == self.READ_ONLY or self.flags == self.NONE:
                # copy from OpenGL buffer to Warp array
                from pyglet import gl

                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.gl_buffer_id)
                nbytes = self.warp_buffer.size * warp.types.type_size_in_bytes(dtype)
                gl.glGetBufferSubData(gl.GL_ARRAY_BUFFER, 0, nbytes, self.warp_buffer_cpu.ptr)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                warp.copy(self.warp_buffer, self.warp_buffer_cpu)
            return self.warp_buffer

        return None

    def unmap(self):
        """Unmap the OpenGL buffer."""
        if self.resource is not None:
            runtime.core.cuda_graphics_unmap(self.context, self.resource)
        elif self.fallback_to_copy:
            if self.warp_buffer is None:
                raise RuntimeError("RegisteredGLBuffer first has to be mapped")

            if self.flags == self.WRITE_DISCARD or self.flags == self.NONE:
                # copy from Warp array to OpenGL buffer
                from pyglet import gl

                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.gl_buffer_id)
                buffer = self.warp_buffer.numpy()
                gl.glBufferData(gl.GL_ARRAY_BUFFER, buffer.nbytes, buffer.ctypes.data, gl.GL_DYNAMIC_DRAW)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)


def zeros(
    shape: int | tuple[int, ...] | list[int] | None = None,
    dtype: type = float,
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

    arr.zero_()

    return arr


def zeros_like(
    src: Array, device: Devicelike = None, requires_grad: bool | None = None, pinned: bool | None = None
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


def ones(
    shape: int | tuple[int, ...] | list[int] | None = None,
    dtype: type = float,
    device: Devicelike = None,
    requires_grad: bool = False,
    pinned: bool = False,
    **kwargs,
) -> warp.array:
    """Return a one-initialized array

    Args:
        shape: Array dimensions
        dtype: Type of each element, e.g.: warp.vec3, warp.mat33, etc
        device: Device that array will live on
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    return full(shape=shape, value=1, dtype=dtype, device=device, requires_grad=requires_grad, pinned=pinned, **kwargs)


def ones_like(
    src: Array, device: Devicelike = None, requires_grad: bool | None = None, pinned: bool | None = None
) -> warp.array:
    """Return a one-initialized array with the same type and dimension of another array

    Args:
        src: The template array to use for shape, data type, and device
        device: The device where the new array will be created (defaults to src.device)
        requires_grad: Whether the array will be tracked for back propagation
        pinned: Whether the array uses pinned host memory (only applicable to CPU arrays)

    Returns:
        A warp.array object representing the allocation
    """

    return full_like(src, 1, device=device, requires_grad=requires_grad, pinned=pinned)


def full(
    shape: int | tuple[int, ...] | list[int] | None = None,
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
        elif value_type == bool:
            dtype = warp.bool
        elif value_type in warp.types.scalar_types or hasattr(value_type, "_wp_scalar_type_"):
            dtype = value_type
        elif isinstance(value, warp.codegen.StructInstance):
            dtype = value._cls
        elif hasattr(value, "__len__"):
            # a sequence, assume it's a vector or matrix value
            try:
                # try to convert to a numpy array first
                na = np.asarray(value)
            except Exception as e:
                raise ValueError(f"Failed to interpret the value as a vector or matrix: {e}") from e

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

    arr.fill_(value)

    return arr


def full_like(
    src: Array,
    value: Any,
    device: Devicelike = None,
    requires_grad: bool | None = None,
    pinned: bool | None = None,
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


def clone(
    src: warp.array, device: Devicelike = None, requires_grad: bool | None = None, pinned: bool | None = None
) -> warp.array:
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
    shape: int | tuple[int, ...] | list[int] | None = None,
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
    src: Array, device: Devicelike = None, requires_grad: bool | None = None, pinned: bool | None = None
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
    dtype: type | None = None,
    shape: Sequence[int] | None = None,
    device: Devicelike | None = None,
    requires_grad: bool = False,
) -> warp.array:
    """Returns a Warp array created from a NumPy array.

    Args:
        arr: The NumPy array providing the data to construct the Warp array.
        dtype: The data type of the new Warp array. If this is not provided, the data type will be inferred.
        shape: The shape of the Warp array.
        device: The device on which the Warp array will be constructed.
        requires_grad: Whether gradients will be tracked for this array.

    Raises:
        RuntimeError: The data type of the NumPy array is not supported.
    """
    if dtype is None:
        base_type = warp.types.np_dtype_to_warp_type.get(arr.dtype)
        if base_type is None:
            raise RuntimeError(f"Unsupported NumPy data type '{arr.dtype}'.")

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
        device=device,
        requires_grad=requires_grad,
    )


def event_from_ipc_handle(handle, device: Devicelike = None) -> Event:
    """Create an event from an IPC handle.

    Args:
        handle: The interprocess event handle for an existing CUDA event.
        device (Devicelike): Device to associate with the array.

    Returns:
        An event created from the interprocess event handle ``handle``.

    Raises:
        RuntimeError: IPC is not supported on ``device``.
    """

    try:
        # Performance note: try first, ask questions later
        device = warp.context.runtime.get_device(device)
    except Exception:
        # Fallback to using the public API for retrieving the device,
        # which takes take of initializing Warp if needed.
        device = warp.context.get_device(device)

    if device.is_ipc_supported is False:
        raise RuntimeError(f"IPC is not supported on device {device}.")

    event = Event(
        device=device, cuda_event=warp.context.runtime.core.cuda_ipc_open_event_handle(device.context, handle)
    )
    # Events created from IPC handles must be freed with cuEventDestroy
    event.owner = True

    return event


# given a kernel destination argument type and a value convert
#  to a c-type that can be passed to a kernel
def pack_arg(kernel, arg_type, arg_name, value, device, adjoint=False):
    if warp.types.is_array(arg_type):
        if value is None:
            # allow for NULL arrays
            return arg_type.__ctype__()

        elif isinstance(value, warp.types.array_t):
            # accept array descriptors verbatim
            return value

        else:
            # check for array type
            # - in forward passes, array types have to match
            # - in backward passes, indexed array gradients are regular arrays
            if adjoint:
                array_matches = isinstance(value, warp.array)
            else:
                array_matches = type(value) is type(arg_type)

            if not array_matches:
                # if a regular Warp array is required, try converting from __cuda_array_interface__ or __array_interface__
                if isinstance(arg_type, warp.array):
                    if device.is_cuda:
                        # check for __cuda_array_interface__
                        try:
                            interface = value.__cuda_array_interface__
                        except AttributeError:
                            pass
                        else:
                            return warp.types.array_ctype_from_interface(interface, dtype=arg_type.dtype, owner=value)
                    else:
                        # check for __array_interface__
                        try:
                            interface = value.__array_interface__
                        except AttributeError:
                            pass
                        else:
                            return warp.types.array_ctype_from_interface(interface, dtype=arg_type.dtype, owner=value)
                        # check for __array__() method, e.g. Torch CPU tensors
                        try:
                            interface = value.__array__().__array_interface__
                        except AttributeError:
                            pass
                        else:
                            return warp.types.array_ctype_from_interface(interface, dtype=arg_type.dtype, owner=value)

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
        # simple value types don't have gradient arrays, but native built-in signatures still expect a non-null adjoint value of the correct type
        if value is None and adjoint:
            return arg_type(0)
        if warp.types.types_equal(type(value), arg_type):
            return value
        else:
            # try constructing the required value from the argument (handles tuple / list, Gf.Vec3 case)
            try:
                return arg_type(value)
            except Exception as e:
                raise ValueError(f"Failed to convert argument for param {arg_name} to {type_str(arg_type)}") from e

    elif isinstance(value, arg_type):
        try:
            # try to pack as a scalar type
            if arg_type is warp.types.float16:
                return arg_type._type_(warp.types.float_to_half_bits(value.value))
            else:
                return arg_type._type_(value.value)
        except Exception as e:
            raise RuntimeError(
                "Error launching kernel, unable to pack kernel parameter type "
                f"{type(value)} for param {arg_name}, expected {arg_type}"
            ) from e

    else:
        # scalar args don't have gradient arrays, but native built-in signatures still expect a non-null scalar adjoint
        if value is None and adjoint:
            return arg_type._type_(0)
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
            ) from e


class Launch:
    """Represents all data required for a kernel launch so that launches can be replayed quickly.

    Users should not directly instantiate this class, instead use
    ``wp.launch(..., record_cmd=True)`` to record a launch.
    """

    def __init__(
        self,
        kernel,
        device: Device,
        hooks: KernelHooks | None = None,
        params: Sequence[Any] | None = None,
        params_addr: Sequence[ctypes.c_void_p] | None = None,
        bounds: launch_bounds_t | None = None,
        max_blocks: int = 0,
        block_dim: int = 256,
        adjoint: bool = False,
    ):
        # retain the module executable so it doesn't get unloaded
        self.module_exec = kernel.module.load(device)
        if not self.module_exec:
            raise RuntimeError(f"Failed to load module {kernel.module.name} on device {device}")

        # if not specified look up hooks
        if not hooks:
            hooks = self.module_exec.get_kernel_hooks(kernel)

        # if not specified set a zero bound
        if not bounds:
            bounds = launch_bounds_t(0)

        # if not specified then build a list of default value params for args
        if not params:
            params = []
            params.append(bounds)

            # Pack forward parameters
            for a in kernel.adj.args:
                if isinstance(a.type, warp.types.array):
                    params.append(a.type.__ctype__())
                elif isinstance(a.type, warp.codegen.Struct):
                    params.append(a.type().__ctype__())
                else:
                    params.append(pack_arg(kernel, a.type, a.label, 0, device, False))

            # Pack adjoint parameters if adjoint=True
            if adjoint:
                for a in kernel.adj.args:
                    if isinstance(a.type, warp.types.array):
                        params.append(a.type.__ctype__())
                    elif isinstance(a.type, warp.codegen.Struct):
                        params.append(a.type().__ctype__())
                    else:
                        # For primitive types in adjoint mode, initialize with 0
                        params.append(pack_arg(kernel, a.type, a.label, 0, device, True))

            # Create array of parameter addresses
            kernel_args = [ctypes.c_void_p(ctypes.addressof(x)) for x in params]
            kernel_params = (ctypes.c_void_p * len(kernel_args))(*kernel_args)

            params_addr = kernel_params

        self.kernel = kernel
        self.hooks = hooks
        self.params = params
        self.params_addr = params_addr
        self.device: Device = device
        """The device to launch on.
        This should not be changed after the launch object is created.
        """

        self.bounds: launch_bounds_t = bounds
        """The launch bounds. Update with :meth:`set_dim`."""

        self.max_blocks: int = max_blocks
        """The maximum number of CUDA thread blocks to use."""

        self.block_dim: int = block_dim
        """The number of threads per block."""

        self.adjoint: bool = adjoint
        """Whether to run the adjoint kernel instead of the forward kernel."""

    def set_dim(self, dim: int | list[int] | tuple[int, ...]):
        """Set the launch dimensions.

        Args:
            dim: The dimensions of the launch.
        """
        self.bounds = launch_bounds_t(dim)

        # launch bounds always at index 0
        self.params[0] = self.bounds

        # for CUDA kernels we need to update the address to each arg
        if self.params_addr:
            self.params_addr[0] = ctypes.c_void_p(ctypes.addressof(self.bounds))

    def set_param_at_index(self, index: int, value: Any, adjoint: bool = False):
        """Set a kernel parameter at an index.

        Args:
            index: The index of the param to set.
            value: The value to set the param to.
        """
        arg_type = self.kernel.adj.args[index].type
        arg_name = self.kernel.adj.args[index].label

        carg = pack_arg(self.kernel, arg_type, arg_name, value, self.device, adjoint)

        if adjoint:
            params_index = index + len(self.kernel.adj.args) + 1
        else:
            params_index = index + 1

        self.params[params_index] = carg

        # for CUDA kernels we need to update the address to each arg
        if self.params_addr:
            self.params_addr[params_index] = ctypes.c_void_p(ctypes.addressof(carg))

    def set_param_at_index_from_ctype(self, index: int, value: ctypes.Structure | int | float):
        """Set a kernel parameter at an index without any type conversion.

        Args:
            index: The index of the param to set.
            value: The value to set the param to.
        """
        if isinstance(value, ctypes.Structure):
            # not sure how to directly assign struct->struct without reallocating using ctypes
            self.params[index + 1] = value

            # for CUDA kernels we need to update the address to each arg
            if self.params_addr:
                self.params_addr[index + 1] = ctypes.c_void_p(ctypes.addressof(value))

        else:
            self.params[index + 1].__init__(value)

    def set_param_by_name(self, name: str, value: Any, adjoint: bool = False):
        """Set a kernel parameter by argument name.

        Args:
            name: The name of the argument to set.
            value: The value to set the argument to.
            adjoint: If ``True``, set the adjoint of this parameter instead of the forward parameter.
        """
        for i, arg in enumerate(self.kernel.adj.args):
            if arg.label == name:
                self.set_param_at_index(i, value, adjoint)
                return

        raise ValueError(f"Argument '{name}' not found in kernel '{self.kernel.key}'")

    def set_param_by_name_from_ctype(self, name: str, value: ctypes.Structure):
        """Set a kernel parameter by argument name with no type conversions.

        Args:
            name: The name of the argument to set.
            value: The value to set the argument to.
        """
        # lookup argument index
        for i, arg in enumerate(self.kernel.adj.args):
            if arg.label == name:
                self.set_param_at_index_from_ctype(i, value)

    def set_params(self, values: Sequence[Any]):
        """Set all parameters.

        Args:
            values: A list of values to set the params to.
        """
        for i, v in enumerate(values):
            self.set_param_at_index(i, v)

    def set_params_from_ctypes(self, values: Sequence[ctypes.Structure]):
        """Set all parameters without performing type-conversions.

        Args:
            values: A list of ctypes or basic int / float types.
        """
        for i, v in enumerate(values):
            self.set_param_at_index_from_ctype(i, v)

    def launch(self, stream: Stream | None = None) -> None:
        """Launch the kernel.

        Args:
            stream: The stream to launch on.
        """
        if self.device.is_cpu:
            if self.adjoint:
                self.hooks.backward(*self.params)
            else:
                self.hooks.forward(*self.params)
        else:
            if stream is None:
                stream = self.device.stream

            # If the stream is capturing, we retain the CUDA module so that it doesn't get unloaded
            # before the captured graph is released.
            if runtime.core.cuda_stream_is_capturing(stream.cuda_stream):
                capture_id = runtime.core.cuda_stream_get_capture_id(stream.cuda_stream)
                graph = runtime.captures.get(capture_id)
                if graph is not None:
                    graph.retain_module_exec(self.module_exec)

            if self.adjoint:
                runtime.core.cuda_launch_kernel(
                    self.device.context,
                    self.hooks.backward,
                    self.bounds.size,
                    self.max_blocks,
                    self.block_dim,
                    self.hooks.backward_smem_bytes,
                    self.params_addr,
                    stream.cuda_stream,
                )
            else:
                runtime.core.cuda_launch_kernel(
                    self.device.context,
                    self.hooks.forward,
                    self.bounds.size,
                    self.max_blocks,
                    self.block_dim,
                    self.hooks.forward_smem_bytes,
                    self.params_addr,
                    stream.cuda_stream,
                )


def launch(
    kernel,
    dim: int | Sequence[int],
    inputs: Sequence = [],
    outputs: Sequence = [],
    adj_inputs: Sequence = [],
    adj_outputs: Sequence = [],
    device: Devicelike = None,
    stream: Stream | None = None,
    adjoint: bool = False,
    record_tape: bool = True,
    record_cmd: bool = False,
    max_blocks: int = 0,
    block_dim: int = 256,
):
    """Launch a Warp kernel on the target device

    Kernel launches are asynchronous with respect to the calling Python thread.

    Args:
        kernel: The name of a Warp kernel function, decorated with the ``@wp.kernel`` decorator
        dim: The number of threads to launch the kernel, can be an integer or a
          sequence of integers with a maximum of 4 dimensions.
        inputs: The input parameters to the kernel (optional)
        outputs: The output parameters (optional)
        adj_inputs: The adjoint inputs (optional)
        adj_outputs: The adjoint outputs (optional)
        device: The device to launch on.
        stream: The stream to launch on.
        adjoint: Whether to run forward or backward pass (typically use ``False``).
        record_tape: When ``True``, the launch will be recorded the global
          :class:`wp.Tape() <warp.Tape>` object when present.
        record_cmd: When ``True``, the launch will return a :class:`Launch`
          object. The launch will not occur until the user calls
          :meth:`Launch.launch()`.
        max_blocks: The maximum number of CUDA thread blocks to use.
          Only has an effect for CUDA kernel launches.
          If negative or zero, the maximum hardware value will be used.
        block_dim: The number of threads per block (always 1 for "cpu" devices).
    """

    init()

    # if stream is specified, use the associated device
    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)

    if device == "cpu":
        block_dim = 1

    # check function is a Kernel
    if not isinstance(kernel, Kernel):
        raise RuntimeError("Error launching kernel, can only launch functions decorated with @wp.kernel.")

    # debugging aid
    if warp.config.print_launches:
        print(f"kernel: {kernel.key} dim: {dim} inputs: {inputs} outputs: {outputs} device: {device}")

    # construct launch bounds
    bounds = launch_bounds_t(dim)

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

        fwd_args = []
        fwd_args.extend(inputs)
        fwd_args.extend(outputs)

        adj_args = []
        adj_args.extend(adj_inputs)
        adj_args.extend(adj_outputs)

        if (len(fwd_args)) != (len(kernel.adj.args)):
            raise RuntimeError(
                f"Error launching kernel '{kernel.key}', passed {len(fwd_args)} arguments but kernel requires {len(kernel.adj.args)}."
            )

        # if it's a generic kernel, infer the required overload from the arguments
        if kernel.is_generic:
            fwd_types = kernel.infer_argument_types(fwd_args)
            kernel = kernel.add_overload(fwd_types)

        # delay load modules, including new overload if needed
        try:
            module_exec = kernel.module.load(device, block_dim)
        except Exception:
            kernel.adj.skip_build = True
            raise

        if not module_exec:
            return

        # late bind
        hooks = module_exec.get_kernel_hooks(kernel)

        pack_args(fwd_args, params)
        pack_args(adj_args, params, adjoint=True)

        # run kernel
        if device.is_cpu:
            if adjoint:
                if hooks.backward is None:
                    raise RuntimeError(
                        f"Failed to find backward kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
                    )

                if record_cmd:
                    launch = Launch(
                        kernel=kernel,
                        hooks=hooks,
                        params=params,
                        params_addr=None,
                        bounds=bounds,
                        device=device,
                        adjoint=adjoint,
                    )
                    return launch
                hooks.backward(*params)

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
                        params_addr=None,
                        bounds=bounds,
                        device=device,
                        adjoint=adjoint,
                    )
                    return launch
                else:
                    hooks.forward(*params)

        else:
            kernel_args = [ctypes.c_void_p(ctypes.addressof(x)) for x in params]
            kernel_params = (ctypes.c_void_p * len(kernel_args))(*kernel_args)

            if stream is None:
                stream = device.stream

            # If the stream is capturing, we retain the CUDA module so that it doesn't get unloaded
            # before the captured graph is released.
            if runtime.core.cuda_stream_is_capturing(stream.cuda_stream):
                capture_id = runtime.core.cuda_stream_get_capture_id(stream.cuda_stream)
                graph = runtime.captures.get(capture_id)
                if graph is not None:
                    graph.retain_module_exec(module_exec)

            if adjoint:
                if hooks.backward is None:
                    raise RuntimeError(
                        f"Failed to find backward kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
                    )

                if record_cmd:
                    launch = Launch(
                        kernel=kernel,
                        hooks=hooks,
                        params=params,
                        params_addr=kernel_params,
                        bounds=bounds,
                        device=device,
                        max_blocks=max_blocks,
                        block_dim=block_dim,
                        adjoint=adjoint,
                    )
                    return launch
                else:
                    runtime.core.cuda_launch_kernel(
                        device.context,
                        hooks.backward,
                        bounds.size,
                        max_blocks,
                        block_dim,
                        hooks.backward_smem_bytes,
                        kernel_params,
                        stream.cuda_stream,
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
                        max_blocks=max_blocks,
                        block_dim=block_dim,
                    )
                    return launch
                else:
                    # launch
                    runtime.core.cuda_launch_kernel(
                        device.context,
                        hooks.forward,
                        bounds.size,
                        max_blocks,
                        block_dim,
                        hooks.forward_smem_bytes,
                        kernel_params,
                        stream.cuda_stream,
                    )

            try:
                runtime.verify_cuda_device(device)
            except Exception as e:
                print(f"Error launching kernel: {kernel.key} on device {device}")
                raise e

    # record on tape if one is active
    if runtime.tape and record_tape:
        # record file, lineno, func as metadata
        frame = inspect.currentframe().f_back
        caller = {"file": frame.f_code.co_filename, "lineno": frame.f_lineno, "func": frame.f_code.co_name}
        runtime.tape.record_launch(
            kernel, dim, max_blocks, inputs, outputs, device, block_dim, metadata={"caller": caller}
        )

        # detect illegal inter-kernel read/write access patterns if verification flag is set
        if warp.config.verify_autograd_array_access:
            runtime.tape._check_kernel_array_access(kernel, fwd_args)


def launch_tiled(*args, **kwargs):
    """A helper method for launching a grid with an extra trailing dimension equal to the block size.

    For example, to launch a 2D grid, where each element has 64 threads assigned you would use the following:

    .. code-block:: python

        wp.launch_tiled(kernel, [M, N], inputs=[...], block_dim=64)

    Which is equivalent to the following:

    .. code-block:: python

        wp.launch(kernel, [M, N, 64], inputs=[...], block_dim=64)

    Inside your kernel code you can retrieve the first two indices of the thread as usual, ignoring the implicit third dimension if desired:

    .. code-block:: python

        @wp.kernel
        def compute()

            i, j = wp.tid()

            ...
    """

    # promote dim to a list in case it was passed as a scalar or tuple
    if "dim" not in kwargs:
        raise RuntimeError("Launch dimensions 'dim' argument should be passed via. keyword args for wp.launch_tiled()")

    if "block_dim" not in kwargs:
        raise RuntimeError(
            "Launch block dimension 'block_dim' argument should be passed via. keyword args for wp.launch_tiled()"
        )

    if "device" in kwargs:
        device = kwargs["device"]
    else:
        # todo: this doesn't consider the case where device
        # is passed through positional args
        device = None

    # force the block_dim to 1 if running on "cpu"
    device = runtime.get_device(device)
    if device.is_cpu:
        kwargs["block_dim"] = 1

    dim = kwargs["dim"]
    if not isinstance(dim, list):
        dim = list(dim) if isinstance(dim, tuple) else [dim]

    if len(dim) > 3:
        raise RuntimeError("wp.launch_tiled() requires a grid with fewer than 4 dimensions")

    # add trailing dimension
    kwargs["dim"] = [*dim, kwargs["block_dim"]]

    # forward to original launch method
    return launch(*args, **kwargs)


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
    """Synchronize the calling CPU thread with any outstanding CUDA work on the specified device

    This function allows the host application code to ensure that all kernel launches
    and memory copies have completed on the device.

    Args:
        device: Device to synchronize.
    """

    device = runtime.get_device(device)
    if device.is_cuda:
        if device.is_capturing:
            raise RuntimeError(f"Cannot synchronize device {device} while graph capture is active")

        runtime.core.cuda_context_synchronize(device.context)


def synchronize_stream(stream_or_device: Stream | Devicelike | None = None):
    """Synchronize the calling CPU thread with any outstanding CUDA work on the specified stream.

    This function allows the host application code to ensure that all kernel launches
    and memory copies have completed on the stream.

    Args:
        stream_or_device: `wp.Stream` or a device.  If the argument is a device, synchronize the device's current stream.
    """

    if isinstance(stream_or_device, Stream):
        stream = stream_or_device
    else:
        stream = runtime.get_device(stream_or_device).stream

    runtime.core.cuda_stream_synchronize(stream.cuda_stream)


def synchronize_event(event: Event):
    """Synchronize the calling CPU thread with an event recorded on a CUDA stream.

    This function allows the host application code to ensure that a specific synchronization point was reached.

    Args:
        event: Event to wait for.
    """

    runtime.core.cuda_event_synchronize(event.cuda_event)


def force_load(device: Device | str | list[Device] | list[str] | None = None, modules: list[Module] | None = None):
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
    module: Module | types.ModuleType | str | None = None, device: Device | str | None = None, recursive: bool = False
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
    elif isinstance(module, types.ModuleType):
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


def set_module_options(options: dict[str, Any], module: Any = None):
    """Set options for the current module.

    Options can be used to control runtime compilation and code-generation
    for the current module individually. Available options are listed below.

    * **mode**: The compilation mode to use, can be "debug", or "release", defaults to the value of ``warp.config.mode``.
    * **max_unroll**: The maximum fixed-size loop to unroll, defaults to the value of ``warp.config.max_unroll``.
    * **block_dim**: The default number of threads to assign to each block

    Args:

        options: Set of key-value option pairs
    """

    if module is None:
        m = inspect.getmodule(inspect.stack()[1][0])
    else:
        m = module

    get_module(m.__name__).options.update(options)
    get_module(m.__name__).mark_modified()


def get_module_options(module: Any = None) -> dict[str, Any]:
    """Returns a list of options for the current module."""
    if module is None:
        m = inspect.getmodule(inspect.stack()[1][0])
    else:
        m = module

    return get_module(m.__name__).options


def capture_begin(
    device: Devicelike = None,
    stream: Stream | None = None,
    force_module_load: bool | None = None,
    external: bool = False,
):
    """Begin capture of a CUDA graph

    Captures all subsequent kernel launches and memory operations on CUDA devices.
    This can be used to record large numbers of kernels and replay them with low overhead.

    If `device` is specified, the capture will begin on the CUDA stream currently
    associated with the device.  If `stream` is specified, the capture will begin
    on the given stream.  If both are omitted, the capture will begin on the current
    stream of the current device.

    Args:
        device: The CUDA device to capture on
        stream: The CUDA stream to capture on
        force_module_load: Whether to force loading of all kernels before capture.
          In general it is better to use :func:`~warp.load_module()` to selectively load kernels.
          When running with CUDA drivers that support CUDA 12.3 or newer, this option is not recommended to be set to
          ``True`` because kernels can be loaded during graph capture on more recent drivers. If this argument is
          ``None``, then the behavior inherits from ``wp.config.enable_graph_capture_module_load_by_default`` if the
          driver is older than CUDA 12.3.
        external: Whether the capture was already started externally

    """

    if force_module_load is None:
        if runtime.driver_version >= (12, 3):
            # Driver versions 12.3 and can compile modules during graph capture
            force_module_load = False
        else:
            force_module_load = warp.config.enable_graph_capture_module_load_by_default

    if warp.config.verify_cuda:
        raise RuntimeError("Cannot use CUDA error verification during graph capture")

    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("Must be a CUDA device")
        stream = device.stream

    if external:
        # make sure the stream is already capturing
        if not stream.is_capturing:
            raise RuntimeError("External capture reported, but the stream is not capturing")
    else:
        # make sure the stream is not capturing yet
        if stream.is_capturing:
            raise RuntimeError("Graph capture already in progress on this stream")

        if force_module_load:
            force_load(device)

    if not runtime.core.cuda_graph_begin_capture(device.context, stream.cuda_stream, int(external)):
        raise RuntimeError(runtime.get_error_string())

    capture_id = runtime.core.cuda_stream_get_capture_id(stream.cuda_stream)
    graph = Graph(device, capture_id)

    # add to ongoing captures on the device
    device.captures[stream] = graph

    # add to lookup table by globally unique capture id
    runtime.captures[capture_id] = graph


def capture_end(device: Devicelike = None, stream: Stream | None = None) -> Graph:
    """End the capture of a CUDA graph.

    Args:
        device: The CUDA device where capture began
        stream: The CUDA stream where capture began

    Returns:
        A :class:`Graph` object that can be launched with :func:`~warp.capture_launch()`
    """

    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("Must be a CUDA device")
        stream = device.stream

    # get the graph being captured
    graph = device.captures.get(stream)

    if graph is None:
        raise RuntimeError("Graph capture is not active on this stream")

    del device.captures[stream]
    del runtime.captures[graph.capture_id]

    # get the graph executable
    g = ctypes.c_void_p()
    result = runtime.core.cuda_graph_end_capture(device.context, stream.cuda_stream, ctypes.byref(g))

    if not result:
        # A concrete error should've already been reported, so we don't need to go into details here
        raise RuntimeError(f"CUDA graph capture failed. {runtime.get_error_string()}")

    # set the graph executable
    graph.graph = g
    graph.graph_exec = None  # Lazy initialization

    return graph


def assert_conditional_graph_support():
    if runtime is None:
        init()

    if runtime.toolkit_version < (12, 4):
        raise RuntimeError("Warp must be built with CUDA Toolkit 12.4+ to enable conditional graph nodes")

    if runtime.driver_version < (12, 4):
        raise RuntimeError("Conditional graph nodes require CUDA driver 12.4+")


def capture_pause(device: Devicelike = None, stream: Stream | None = None) -> ctypes.c_void_p:
    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("Must be a CUDA device")
        stream = device.stream

    graph = ctypes.c_void_p()
    if not runtime.core.cuda_graph_pause_capture(device.context, stream.cuda_stream, ctypes.byref(graph)):
        raise RuntimeError(runtime.get_error_string())

    return graph


def capture_resume(graph: ctypes.c_void_p, device: Devicelike = None, stream: Stream | None = None):
    if stream is not None:
        device = stream.device
    else:
        device = runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("Must be a CUDA device")
        stream = device.stream

    if not runtime.core.cuda_graph_resume_capture(device.context, stream.cuda_stream, graph):
        raise RuntimeError(runtime.get_error_string())


# reusable pinned readback buffer for conditions
condition_host = None


def capture_if(
    condition: warp.array(dtype=int),
    on_true: Callable | Graph | None = None,
    on_false: Callable | Graph | None = None,
    stream: Stream = None,
    **kwargs,
):
    """Create a dynamic branch based on a condition.

    The condition value is retrieved from the first element of the ``condition`` array.

    This function is particularly useful with CUDA graphs, but can be used without graph capture as well.
    CUDA 12.4+ is required to take advantage of conditional graph nodes for dynamic control flow.

    Args:
        condition: Warp array holding the condition value.
        on_true: A callback function or :class:`Graph` to execute if the condition is True.
        on_false: A callback function or :class:`Graph` to execute if the condition is False.
        stream: The CUDA stream where the condition was written. If None, use the current stream on the device where ``condition`` resides.

    Any additional keyword arguments are forwarded to the callback functions.
    """

    # if neither the IF branch nor the ELSE branch is specified, it's a no-op
    if on_true is None and on_false is None:
        return

    # check condition data type
    if not isinstance(condition, warp.array) or condition.dtype is not warp.int32:
        raise TypeError("Condition must be a Warp array of int32 with a single element")

    device = condition.device

    # determine the stream and whether a graph capture is active
    if device.is_cuda:
        if stream is None:
            stream = device.stream
        graph = device.captures.get(stream)
    else:
        graph = None

    if graph is None:
        # if no graph is active, just execute the correct branch directly
        if device.is_cuda:
            # use a pinned buffer for condition readback to host
            global condition_host
            if condition_host is None:
                condition_host = warp.empty(1, dtype=int, device="cpu", pinned=True)
            warp.copy(condition_host, condition, stream=stream)
            warp.synchronize_stream(stream)
            condition_value = bool(ctypes.cast(condition_host.ptr, ctypes.POINTER(ctypes.c_int32)).contents)
        else:
            condition_value = bool(ctypes.cast(condition.ptr, ctypes.POINTER(ctypes.c_int32)).contents)

        if condition_value:
            if on_true is not None:
                if isinstance(on_true, Callable):
                    on_true(**kwargs)
                elif isinstance(on_true, Graph):
                    capture_launch(on_true, stream=stream)
                else:
                    raise TypeError("on_true must be a Callable or a Graph")
        else:
            if on_false is not None:
                if isinstance(on_false, Callable):
                    on_false(**kwargs)
                elif isinstance(on_false, Graph):
                    capture_launch(on_false, stream=stream)
                else:
                    raise TypeError("on_false must be a Callable or a Graph")

        return

    graph.has_conditional = True

    # ensure conditional graph nodes are supported
    assert_conditional_graph_support()

    # insert conditional node
    graph_on_true = ctypes.c_void_p()
    graph_on_false = ctypes.c_void_p()
    if not runtime.core.cuda_graph_insert_if_else(
        device.context,
        stream.cuda_stream,
        ctypes.cast(condition.ptr, ctypes.POINTER(ctypes.c_int32)),
        None if on_true is None else ctypes.byref(graph_on_true),
        None if on_false is None else ctypes.byref(graph_on_false),
    ):
        raise RuntimeError(runtime.get_error_string())

    # pause capturing parent graph
    main_graph = capture_pause(stream=stream)

    # capture if-graph
    if on_true is not None:
        capture_resume(graph_on_true, stream=stream)
        if isinstance(on_true, Callable):
            on_true(**kwargs)
        elif isinstance(on_true, Graph):
            if on_true.has_conditional:
                raise RuntimeError(
                    "The on_true graph contains conditional nodes, which are not allowed in child graphs"
                )
            if not runtime.core.cuda_graph_insert_child_graph(
                device.context,
                stream.cuda_stream,
                on_true.graph,
            ):
                raise RuntimeError(runtime.get_error_string())
        else:
            raise TypeError("on_true must be a Callable or a Graph")
        capture_pause(stream=stream)

    # capture else-graph
    if on_false is not None:
        capture_resume(graph_on_false, stream=stream)
        if isinstance(on_false, Callable):
            on_false(**kwargs)
        elif isinstance(on_false, Graph):
            if on_false.has_conditional:
                raise RuntimeError(
                    "The on_false graph contains conditional nodes, which are not allowed in child graphs"
                )
            if not runtime.core.cuda_graph_insert_child_graph(
                device.context,
                stream.cuda_stream,
                on_false.graph,
            ):
                raise RuntimeError(runtime.get_error_string())
        else:
            raise TypeError("on_false must be a Callable or a Graph")
        capture_pause(stream=stream)

    # resume capturing parent graph
    capture_resume(main_graph, stream=stream)


def capture_while(condition: warp.array(dtype=int), while_body: Callable | Graph, stream: Stream = None, **kwargs):
    """Create a dynamic loop based on a condition.

    The condition value is retrieved from the first element of the ``condition`` array.

    The ``while_body`` callback is responsible for updating the condition value so the loop can terminate.

    This function is particularly useful with CUDA graphs, but can be used without graph capture as well.
    CUDA 12.4+ is required to take advantage of conditional graph nodes for dynamic control flow.

    Args:
        condition: Warp array holding the condition value.
        while_body: A callback function or :class:`Graph` to execute while the loop condition is True.
        stream: The CUDA stream where the condition was written. If None, use the current stream on the device where ``condition`` resides.

    Any additional keyword arguments are forwarded to the callback function.
    """

    # check condition data type
    if not isinstance(condition, warp.array) or condition.dtype is not warp.int32:
        raise TypeError("Condition must be a Warp array of int32 with a single element")

    device = condition.device

    # determine the stream and whether a graph capture is active
    if device.is_cuda:
        if stream is None:
            stream = device.stream
        graph = device.captures.get(stream)
    else:
        graph = None

    if graph is None:
        # since no graph is active, just execute the kernels directly
        while True:
            if device.is_cuda:
                # use a pinned buffer for condition readback to host
                global condition_host
                if condition_host is None:
                    condition_host = warp.empty(1, dtype=int, device="cpu", pinned=True)
                warp.copy(condition_host, condition, stream=stream)
                warp.synchronize_stream(stream)
                condition_value = bool(ctypes.cast(condition_host.ptr, ctypes.POINTER(ctypes.c_int32)).contents)
            else:
                condition_value = bool(ctypes.cast(condition.ptr, ctypes.POINTER(ctypes.c_int32)).contents)

            if condition_value:
                if isinstance(while_body, Callable):
                    while_body(**kwargs)
                elif isinstance(while_body, Graph):
                    capture_launch(while_body, stream=stream)
                else:
                    raise TypeError("while_body must be a callable or a graph")

            else:
                break

        return

    graph.has_conditional = True

    # ensure conditional graph nodes are supported
    assert_conditional_graph_support()

    # insert conditional while-node
    body_graph = ctypes.c_void_p()
    cond_handle = ctypes.c_uint64()
    if not runtime.core.cuda_graph_insert_while(
        device.context,
        stream.cuda_stream,
        ctypes.cast(condition.ptr, ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(body_graph),
        ctypes.byref(cond_handle),
    ):
        raise RuntimeError(runtime.get_error_string())

    # pause capturing parent graph and start capturing child graph
    main_graph = capture_pause(stream=stream)
    capture_resume(body_graph, stream=stream)

    # capture while-body
    if isinstance(while_body, Callable):
        while_body(**kwargs)
    elif isinstance(while_body, Graph):
        if while_body.has_conditional:
            raise RuntimeError("The body graph contains conditional nodes, which are not allowed in child graphs")

        if not runtime.core.cuda_graph_insert_child_graph(
            device.context,
            stream.cuda_stream,
            while_body.graph,
        ):
            raise RuntimeError(runtime.get_error_string())
    else:
        raise RuntimeError(runtime.get_error_string())

    # update condition
    if not runtime.core.cuda_graph_set_condition(
        device.context,
        stream.cuda_stream,
        ctypes.cast(condition.ptr, ctypes.POINTER(ctypes.c_int32)),
        cond_handle,
    ):
        raise RuntimeError(runtime.get_error_string())

    # stop capturing child graph and resume capturing parent graph
    capture_pause(stream=stream)
    capture_resume(main_graph, stream=stream)


def capture_launch(graph: Graph, stream: Stream | None = None):
    """Launch a previously captured CUDA graph

    Args:
        graph: A :class:`Graph` as returned by :func:`~warp.capture_end()`
        stream: A :class:`Stream` to launch the graph on
    """

    if stream is not None:
        if stream.device != graph.device:
            raise RuntimeError(f"Cannot launch graph from device {graph.device} on stream from device {stream.device}")
        device = stream.device
    else:
        device = graph.device
        stream = device.stream

    if graph.graph_exec is None:
        g = ctypes.c_void_p()
        result = runtime.core.cuda_graph_create_exec(graph.device.context, graph.graph, ctypes.byref(g))
        if not result:
            raise RuntimeError(f"Graph creation error: {runtime.get_error_string()}")
        graph.graph_exec = g

    if not runtime.core.cuda_graph_launch(graph.graph_exec, stream.cuda_stream):
        raise RuntimeError(f"Graph launch error: {runtime.get_error_string()}")


def copy(
    dest: warp.array,
    src: warp.array,
    dest_offset: int = 0,
    src_offset: int = 0,
    count: int = 0,
    stream: Stream | None = None,
):
    """Copy array contents from `src` to `dest`.

    Args:
        dest: Destination array, must be at least as large as source buffer
        src: Source array
        dest_offset: Element offset in the destination array
        src_offset: Element offset in the source array
        count: Number of array elements to copy (will copy all elements if set to 0)
        stream: The stream on which to perform the copy

    The stream, if specified, can be from any device.  If the stream is omitted, then Warp selects a stream based on the following rules:
    (1) If the destination array is on a CUDA device, use the current stream on the destination device.
    (2) Otherwise, if the source array is on a CUDA device, use the current stream on the source device.

    If neither source nor destination are on a CUDA device, no stream is used for the copy.
    """

    from warp.context import runtime

    if not warp.types.is_array(src) or not warp.types.is_array(dest):
        raise RuntimeError("Copy source and destination must be arrays")

    # backwards compatibility, if count is zero then copy entire src array
    if count <= 0:
        count = src.size

    if count == 0:
        return

    # figure out the stream for the copy
    if stream is None:
        if dest.device.is_cuda:
            stream = dest.device.stream
        elif src.device.is_cuda:
            stream = src.device.stream

    # Copying between different devices requires contiguous arrays.  If the arrays
    # are not contiguous, we must use temporary staging buffers for the transfer.
    # TODO: We can skip the staging if device access is enabled.
    if src.device != dest.device:
        # If the source is not contiguous, make a contiguous copy on the source device.
        if not src.is_contiguous:
            # FIXME: We can't use a temporary CPU allocation during graph capture,
            # because launching the graph will crash after the allocation is
            # garbage-collected.
            if src.device.is_cpu and stream.is_capturing:
                raise RuntimeError("Failed to allocate a CPU staging buffer during graph capture")
            # This involves an allocation and a kernel launch, which must run on the source device.
            if src.device.is_cuda and stream != src.device.stream:
                src.device.stream.wait_stream(stream)
                src = src.contiguous()
                stream.wait_stream(src.device.stream)
            else:
                src = src.contiguous()

        # The source is now contiguous.  If the destination is not contiguous,
        # clone a contiguous copy on the destination device.
        if not dest.is_contiguous:
            # FIXME: We can't use a temporary CPU allocation during graph capture,
            # because launching the graph will crash after the allocation is
            # garbage-collected.
            if dest.device.is_cpu and stream.is_capturing:
                raise RuntimeError("Failed to allocate a CPU staging buffer during graph capture")
            # The allocation must run on the destination device
            if dest.device.is_cuda and stream != dest.device.stream:
                dest.device.stream.wait_stream(stream)
                tmp = empty_like(src, device=dest.device)
                stream.wait_stream(dest.device.stream)
            else:
                tmp = empty_like(src, device=dest.device)
            # Run the copy on the stream given by the caller
            copy(tmp, src, stream=stream)
            src = tmp

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

        if dest.device.is_cuda:
            if src.device.is_cuda:
                if src.device == dest.device:
                    result = runtime.core.memcpy_d2d(
                        dest.device.context, dst_ptr, src_ptr, bytes_to_copy, stream.cuda_stream
                    )
                else:
                    result = runtime.core.memcpy_p2p(
                        dest.device.context, dst_ptr, src.device.context, src_ptr, bytes_to_copy, stream.cuda_stream
                    )
            else:
                result = runtime.core.memcpy_h2d(
                    dest.device.context, dst_ptr, src_ptr, bytes_to_copy, stream.cuda_stream
                )
        else:
            if src.device.is_cuda:
                result = runtime.core.memcpy_d2h(
                    src.device.context, dst_ptr, src_ptr, bytes_to_copy, stream.cuda_stream
                )
            else:
                result = runtime.core.memcpy_h2h(dst_ptr, src_ptr, bytes_to_copy)

        if not result:
            raise RuntimeError(f"Warp copy error: {runtime.get_error_string()}")

    else:
        # handle non-contiguous arrays

        if src.shape != dest.shape:
            raise RuntimeError("Incompatible array shapes")

        src_elem_size = warp.types.type_size_in_bytes(src.dtype)
        dst_elem_size = warp.types.type_size_in_bytes(dest.dtype)

        if src_elem_size != dst_elem_size:
            raise RuntimeError("Incompatible array data types")

        # can't copy to/from fabric arrays of arrays, because they are jagged arrays of arbitrary lengths
        # TODO?
        if (isinstance(src, (warp.fabricarray, warp.indexedfabricarray)) and src.ndim > 1) or (
            isinstance(dest, (warp.fabricarray, warp.indexedfabricarray)) and dest.ndim > 1
        ):
            raise RuntimeError("Copying to/from Fabric arrays of arrays is not supported")

        src_desc = src.__ctype__()
        dst_desc = dest.__ctype__()
        src_ptr = ctypes.pointer(src_desc)
        dst_ptr = ctypes.pointer(dst_desc)
        src_type = warp.types.array_type_id(src)
        dst_type = warp.types.array_type_id(dest)

        if dest.device.is_cuda:
            # This work involves a kernel launch, so it must run on the destination device.
            # If the copy stream is different, we need to synchronize it.
            if stream == dest.device.stream:
                result = runtime.core.array_copy_device(
                    dest.device.context, dst_ptr, src_ptr, dst_type, src_type, src_elem_size
                )
            else:
                dest.device.stream.wait_stream(stream)
                result = runtime.core.array_copy_device(
                    dest.device.context, dst_ptr, src_ptr, dst_type, src_type, src_elem_size
                )
                stream.wait_stream(dest.device.stream)
        else:
            result = runtime.core.array_copy_host(dst_ptr, src_ptr, dst_type, src_type, src_elem_size)

        if not result:
            raise RuntimeError(f"Warp copy error: {runtime.get_error_string()}")

    # copy gradient, if needed
    if hasattr(src, "grad") and src.grad is not None and hasattr(dest, "grad") and dest.grad is not None:
        copy(dest.grad, src.grad, dest_offset=dest_offset, src_offset=src_offset, count=count, stream=stream)

        if runtime.tape:
            runtime.tape.record_func(
                backward=lambda: adj_copy(
                    dest.grad, src.grad, dest_offset=dest_offset, src_offset=src_offset, count=count, stream=stream
                ),
                arrays=[dest, src],
            )
            if warp.config.verify_autograd_array_access:
                dest.mark_write()
                src.mark_read()


def adj_copy(
    adj_dest: warp.array, adj_src: warp.array, dest_offset: int, src_offset: int, count: int, stream: Stream = None
):
    """Copy adjoint operation for wp.copy() calls on the tape.

    Args:
        adj_dest: Destination array adjoint
        adj_src: Source array adjoint
        stream: The stream on which the copy was performed in the forward pass
    """
    copy(adj_src, adj_dest, dest_offset=dest_offset, src_offset=src_offset, count=count, stream=stream)


def type_str(t):
    if t is None:
        return "None"
    elif t == Any:
        return "Any"
    elif t == Callable:
        return "Callable"
    elif isinstance(t, int):
        return str(t)
    elif isinstance(t, (List, tuple)):
        return "Tuple[" + ", ".join(map(type_str, t)) + "]"
    elif isinstance(t, warp.array):
        return f"Array[{type_str(t.dtype)}]"
    elif isinstance(t, warp.indexedarray):
        return f"IndexedArray[{type_str(t.dtype)}]"
    elif isinstance(t, warp.fabricarray):
        return f"FabricArray[{type_str(t.dtype)}]"
    elif isinstance(t, warp.indexedfabricarray):
        return f"IndexedFabricArray[{type_str(t.dtype)}]"
    elif hasattr(t, "_wp_generic_type_hint_"):
        generic_type = t._wp_generic_type_hint_

        # for concrete vec/mat types use the short name
        if t in warp.types.vector_types:
            return t.__name__

        # for generic vector / matrix type use a Generic type hint
        if generic_type == warp.types.Vector:
            # return f"Vector"
            return f"Vector[{type_str(t._wp_type_params_[0])},{type_str(t._wp_scalar_type_)}]"
        elif generic_type == warp.types.Quaternion:
            # return f"Quaternion"
            return f"Quaternion[{type_str(t._wp_scalar_type_)}]"
        elif generic_type == warp.types.Matrix:
            # return f"Matrix"
            return f"Matrix[{type_str(t._wp_type_params_[0])},{type_str(t._wp_type_params_[1])},{type_str(t._wp_scalar_type_)}]"
        elif generic_type == warp.types.Transformation:
            # return f"Transformation"
            return f"Transformation[{type_str(t._wp_scalar_type_)}]"

        raise TypeError("Invalid vector or matrix dimensions")
    elif get_origin(t) in (list, tuple):
        args_repr = ", ".join(type_str(x) for x in get_args(t))
        return f"{t._name}[{args_repr}]"
    elif t is Ellipsis:
        return "..."
    elif warp.types.is_tile(t):
        return f"Tile[{type_str(t.dtype)},{type_str(t.shape)}]"

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
        return_type = " -> " + type_str(f.value_func(None, None))
    except Exception:
        pass

    print(f".. py:function:: {f.key}({args}){return_type}", file=file)
    if noentry:
        print("    :noindex:", file=file)
        print("    :nocontentsentry:", file=file)
    print("", file=file)

    if f.doc != "":
        if not f.missing_grad:
            print(f"    {f.doc}", file=file)
        else:
            print(f"    {f.doc} [1]_", file=file)
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

    # build dictionary of all functions by group
    groups = {}

    functions = list(builtin_functions.values())

    for f in functions:
        # build dict of groups
        if f.group not in groups:
            groups[f.group] = []

        if hasattr(f, "overloads"):
            # append all overloads to the group
            for o in f.overloads:
                groups[f.group].append(o)
        else:
            groups[f.group].append(f)

    # Keep track of what function and query types have been written
    written_functions = set()
    written_query_types = set()

    query_types = (
        ("bvh_query", "BvhQuery"),
        ("mesh_query_aabb", "MeshQueryAABB"),
        ("mesh_query_point", "MeshQueryPoint"),
        ("mesh_query_ray", "MeshQueryRay"),
        ("hash_grid_query", "HashGridQuery"),
    )

    for k, g in groups.items():
        print("\n", file=file)
        print(k, file=file)
        print("---------------", file=file)

        for f in g:
            if f.func:
                # f is a Warp function written in Python, we can use autofunction
                print(f".. autofunction:: {f.func.__module__}.{f.key}", file=file)
                continue
            for f_prefix, query_type in query_types:
                if f.key.startswith(f_prefix) and query_type not in written_query_types:
                    print(f".. autoclass:: {query_type}", file=file)
                    written_query_types.add(query_type)
                    break

            if f.key in written_functions:
                # Add :noindex: + :nocontentsentry: since Sphinx gets confused
                print_function(f, file=file, noentry=True)
            else:
                if print_function(f, file=file):
                    written_functions.add(f.key)

    # footnotes
    print(".. rubric:: Footnotes", file=file)
    print(".. [1] Function gradients have not been implemented for backpropagation.", file=file)


def export_stubs(file):  # pragma: no cover
    """Generates stub file for auto-complete of builtin functions"""

    # Add copyright notice
    print(
        """# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
""",
        file=file,
    )

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
    print('Shape = TypeVar("Shape")', file=file)

    print("Vector = Generic[Length, Scalar]", file=file)
    print("Matrix = Generic[Rows, Cols, Scalar]", file=file)
    print("Quaternion = Generic[Float]", file=file)
    print("Transformation = Generic[Float]", file=file)
    print("Array = Generic[DType]", file=file)
    print("FabricArray = Generic[DType]", file=file)
    print("IndexedFabricArray = Generic[DType]", file=file)
    print("Tile = Generic[DType, Shape]", file=file)

    # prepend __init__.py
    with open(os.path.join(os.path.dirname(file.name), "__init__.py")) as header_file:
        # strip comment lines
        lines = [line for line in header_file if not line.startswith("#")]
        header = "".join(lines)

    print(header, file=file)
    print(file=file)

    def add_stub(f):
        args = ", ".join(f"{k}: {type_str(v)}" for k, v in f.input_types.items())

        return_str = ""

        if f.hidden:  # or f.generic:
            return

        return_type = f.value_type
        if f.value_func:
            return_type = f.value_func(None, None)
        if return_type:
            return_str = " -> " + type_str(return_type)

        print("@over", file=file)
        print(f"def {f.key}({args}){return_str}:", file=file)
        print(f'    """{f.doc}', file=file)
        print('    """', file=file)
        print("    ...\n\n", file=file)

    for g in builtin_functions.values():
        if hasattr(g, "overloads"):
            for f in g.overloads:
                add_stub(f)
        else:
            add_stub(g)


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
        return get_builtin_type(t).__name__

    file.write("namespace wp {\n\n")
    file.write('extern "C" {\n\n')

    for k, g in builtin_functions.items():
        if not hasattr(g, "overloads"):
            continue
        for f in g.overloads:
            if not f.export or f.generic:
                continue

            # only export simple types that don't use arrays
            # or templated types
            if not f.is_simple():
                continue

            # Runtime arguments that are to be passed to the function, not its template signature.
            if f.export_func is not None:
                func_args = f.export_func(f.input_types)
            else:
                func_args = f.input_types

            # todo: construct a default value for each of the functions args
            # so we can generate the return type for overloaded functions
            return_type = f.value_func(func_args, None)

            args = ", ".join(f"{ctype_arg_str(v)} {k}" for k, v in func_args.items())
            params = ", ".join(func_args.keys())

            if return_type is None:
                # void function
                file.write(f"WP_API void {f.mangled_name}({args}) {{ wp::{f.key}({params}); }}\n")
            elif isinstance(return_type, tuple) and len(return_type) > 1:
                # multiple return value function using output parameters
                outputs = tuple(f"{ctype_ret_str(x)}& ret_{i}" for i, x in enumerate(return_type))
                output_params = ", ".join(f"ret_{i}" for i in range(len(outputs)))
                if args:
                    file.write(
                        f"WP_API void {f.mangled_name}({args}, {', '.join(outputs)}) {{ wp::{f.key}({params}, {output_params}); }}\n"
                    )
                else:
                    file.write(
                        f"WP_API void {f.mangled_name}({', '.join(outputs)}) {{ wp::{f.key}({params}, {output_params}); }}\n"
                    )
            else:
                # single return value function
                try:
                    return_str = ctype_ret_str(return_type)
                except Exception:
                    continue

                if args:
                    file.write(
                        f"WP_API void {f.mangled_name}({args}, {return_str}* ret) {{ *ret = wp::{f.key}({params}); }}\n"
                    )
                else:
                    file.write(f"WP_API void {f.mangled_name}({return_str}* ret) {{ *ret = wp::{f.key}({params}); }}\n")

    file.write('\n}  // extern "C"\n\n')
    file.write("}  // namespace wp\n")


# initialize global runtime
runtime = None


def init():
    """Initialize the Warp runtime. This function must be called before any other API call. If an error occurs an exception will be raised."""
    global runtime

    if runtime is None:
        runtime = Runtime()
