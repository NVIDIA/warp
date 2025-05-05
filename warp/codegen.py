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
import builtins
import ctypes
import functools
import hashlib
import inspect
import math
import re
import sys
import textwrap
import types
from typing import Any, Callable, ClassVar, Mapping, Sequence, get_args, get_origin

import warp.config
from warp.types import *

# used as a globally accessible copy
# of current compile options (block_dim) etc
options = {}


class WarpCodegenError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)


class WarpCodegenTypeError(TypeError):
    def __init__(self, message):
        super().__init__(message)


class WarpCodegenAttributeError(AttributeError):
    def __init__(self, message):
        super().__init__(message)


class WarpCodegenKeyError(KeyError):
    def __init__(self, message):
        super().__init__(message)


# map operator to function name
builtin_operators: dict[type[ast.AST], str] = {}

# see https://www.ics.uci.edu/~pattis/ICS-31/lectures/opexp.pdf for a
# nice overview of python operators

builtin_operators[ast.Add] = "add"
builtin_operators[ast.Sub] = "sub"
builtin_operators[ast.Mult] = "mul"
builtin_operators[ast.MatMult] = "mul"
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

builtin_operators[ast.BitAnd] = "bit_and"
builtin_operators[ast.BitOr] = "bit_or"
builtin_operators[ast.BitXor] = "bit_xor"
builtin_operators[ast.Invert] = "invert"
builtin_operators[ast.LShift] = "lshift"
builtin_operators[ast.RShift] = "rshift"

comparison_chain_strings = [
    builtin_operators[ast.Gt],
    builtin_operators[ast.Lt],
    builtin_operators[ast.LtE],
    builtin_operators[ast.GtE],
    builtin_operators[ast.Eq],
    builtin_operators[ast.NotEq],
]


def values_check_equal(a, b):
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        if len(a) != len(b):
            return False

        return all(x == y for x, y in zip(a, b))

    return a == b


def op_str_is_chainable(op: str) -> builtins.bool:
    return op in comparison_chain_strings


def get_closure_cell_contents(obj):
    """Retrieve a closure's cell contents or `None` if it's empty."""
    try:
        return obj.cell_contents
    except ValueError:
        pass

    return None


def eval_annotations(annotations: Mapping[str, Any], obj: Any) -> Mapping[str, Any]:
    """Un-stringize annotations caused by `from __future__ import annotations` of PEP 563."""
    # Implementation backported from `inspect.get_annotations()` for Python 3.9 and older.
    if not annotations:
        return {}

    if not any(isinstance(x, str) for x in annotations.values()):
        # No annotation to un-stringize.
        return annotations

    if isinstance(obj, type):
        # class
        globals = {}
        module_name = getattr(obj, "__module__", None)
        if module_name:
            module = sys.modules.get(module_name, None)
            if module:
                globals = getattr(module, "__dict__", {})
        locals = dict(vars(obj))
        unwrap = obj
    elif isinstance(obj, types.ModuleType):
        # module
        globals = obj.__dict__
        locals = {}
        unwrap = None
    elif callable(obj):
        # function
        globals = getattr(obj, "__globals__", {})
        # Capture the variables from the surrounding scope.
        closure_vars = zip(
            obj.__code__.co_freevars, tuple(get_closure_cell_contents(x) for x in (obj.__closure__ or ()))
        )
        locals = {k: v for k, v in closure_vars if v is not None}
        unwrap = obj
    else:
        raise TypeError(f"{obj!r} is not a module, class, or callable.")

    if unwrap is not None:
        while True:
            if hasattr(unwrap, "__wrapped__"):
                unwrap = unwrap.__wrapped__
                continue
            if isinstance(unwrap, functools.partial):
                unwrap = unwrap.func
                continue
            break
        if hasattr(unwrap, "__globals__"):
            globals = unwrap.__globals__

    # "Inject" type parameters into the local namespace
    # (unless they are shadowed by assignments *in* the local namespace),
    # as a way of emulating annotation scopes when calling `eval()`
    type_params = getattr(obj, "__type_params__", ())
    if type_params:
        locals = {param.__name__: param for param in type_params} | locals

    return {k: v if not isinstance(v, str) else eval(v, globals, locals) for k, v in annotations.items()}


def get_annotations(obj: Any) -> Mapping[str, Any]:
    """Same as `inspect.get_annotations()` but always returning un-stringized annotations."""
    # This backports `inspect.get_annotations()` for Python 3.9 and older.
    # See https://docs.python.org/3/howto/annotations.html#accessing-the-annotations-dict-of-an-object-in-python-3-9-and-older
    if isinstance(obj, type):
        annotations = obj.__dict__.get("__annotations__", {})
    else:
        annotations = getattr(obj, "__annotations__", {})

    # Evaluating annotations can be done using the `eval_str` parameter with
    # the official function from the `inspect` module.
    return eval_annotations(annotations, obj)


def get_full_arg_spec(func: Callable) -> inspect.FullArgSpec:
    """Same as `inspect.getfullargspec()` but always returning un-stringized annotations."""
    # See https://docs.python.org/3/howto/annotations.html#manually-un-stringizing-stringized-annotations
    spec = inspect.getfullargspec(func)
    return spec._replace(annotations=eval_annotations(spec.annotations, func))


def struct_instance_repr_recursive(inst: StructInstance, depth: int, use_repr: bool) -> str:
    indent = "\t"

    # handle empty structs
    if len(inst._cls.vars) == 0:
        return f"{inst._cls.key}()"

    lines = []
    lines.append(f"{inst._cls.key}(")

    for field_name, _ in inst._cls.ctype._fields_:
        field_value = getattr(inst, field_name, None)

        if isinstance(field_value, StructInstance):
            field_value = struct_instance_repr_recursive(field_value, depth + 1, use_repr)

        if use_repr:
            lines.append(f"{indent * (depth + 1)}{field_name}={field_value!r},")
        else:
            lines.append(f"{indent * (depth + 1)}{field_name}={field_value!s},")

    lines.append(f"{indent * depth})")
    return "\n".join(lines)


class StructInstance:
    def __init__(self, cls: Struct, ctype):
        super().__setattr__("_cls", cls)

        # maintain a c-types object for the top-level instance the struct
        if not ctype:
            super().__setattr__("_ctype", cls.ctype())
        else:
            super().__setattr__("_ctype", ctype)

        # create Python attributes for each of the struct's variables
        for field, var in cls.vars.items():
            if isinstance(var.type, warp.codegen.Struct):
                self.__dict__[field] = var.type.instance_type(ctype=getattr(self._ctype, field))
            elif isinstance(var.type, warp.types.array):
                self.__dict__[field] = None
            else:
                self.__dict__[field] = var.type()

    def __getattribute__(self, name):
        cls = super().__getattribute__("_cls")
        if name == "native_name":
            return cls.native_name

        var = cls.vars.get(name)
        if var is not None:
            if isinstance(var.type, type) and issubclass(var.type, ctypes.Array):
                # Each field stored in a `StructInstance` is exposed as
                # a standard Python attribute but also has a `ctypes`
                # equivalent that is being updated in `__setattr__`.
                # However, when assigning in place an object such as a vec/mat
                # (e.g.: `my_struct.my_vec[0] = 1.23`), the `__setattr__` method
                # from `StructInstance` isn't called, and the synchronization
                # mechanism has no chance of updating the underlying ctype data.
                # As a workaround, we catch here all attempts at accessing such
                # objects and directly return their underlying ctype since
                # the Python-facing Warp vectors and matrices are implemented
                # using `ctypes.Array` anyways.
                return getattr(self._ctype, name)

        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name not in self._cls.vars:
            raise RuntimeError(f"Trying to set Warp struct attribute that does not exist {name}")

        var = self._cls.vars[name]

        # update our ctype flat copy
        if isinstance(var.type, array):
            if value is None:
                # create array with null pointer
                setattr(self._ctype, name, array_t())
            else:
                # wp.array
                assert isinstance(value, array)
                assert types_equal(value.dtype, var.type.dtype), (
                    f"assign to struct member variable {name} failed, expected type {type_repr(var.type.dtype)}, got type {type_repr(value.dtype)}"
                )
                setattr(self._ctype, name, value.__ctype__())

        elif isinstance(var.type, Struct):
            # assign structs by-value, otherwise we would have problematic cases transferring ownership
            # of the underlying ctypes data between shared Python struct instances

            if not isinstance(value, StructInstance):
                raise RuntimeError(
                    f"Trying to assign a non-structure value to a struct attribute with type: {self._cls.key}"
                )

            # destination attribution on self
            dest = getattr(self, name)

            if dest._cls.key is not value._cls.key:
                raise RuntimeError(
                    f"Trying to assign a structure of type {value._cls.key} to an attribute of {self._cls.key}"
                )

            # update all nested ctype vars by deep copy
            for n in dest._cls.vars:
                setattr(dest, n, getattr(value, n))

            # early return to avoid updating our Python StructInstance
            return

        elif issubclass(var.type, ctypes.Array):
            # vector/matrix type, e.g. vec3
            if value is None:
                setattr(self._ctype, name, var.type())
            elif types_equal(type(value), var.type):
                setattr(self._ctype, name, value)
            else:
                # conversion from list/tuple, ndarray, etc.
                setattr(self._ctype, name, var.type(value))

        else:
            # primitive type
            if value is None:
                # zero initialize
                setattr(self._ctype, name, var.type._type_())
            else:
                if hasattr(value, "_type_"):
                    # assigning warp type value (e.g.: wp.float32)
                    value = value.value
                # float16 needs conversion to uint16 bits
                if var.type == warp.float16:
                    setattr(self._ctype, name, float_to_half_bits(value))
                else:
                    setattr(self._ctype, name, value)

        # update Python instance
        super().__setattr__(name, value)

    def __ctype__(self):
        return self._ctype

    def __repr__(self):
        return struct_instance_repr_recursive(self, 0, use_repr=True)

    def __str__(self):
        return struct_instance_repr_recursive(self, 0, use_repr=False)

    def to(self, device):
        """Copies this struct with all array members moved onto the given device.

        Arrays already living on the desired device are referenced as-is, while
        arrays being moved are copied.
        """
        out = self._cls()
        stack = [(self, out, k, v) for k, v in self._cls.vars.items()]
        while stack:
            src, dst, name, var = stack.pop()
            value = getattr(src, name)
            if isinstance(var.type, array):
                # array_t
                setattr(dst, name, value.to(device))
            elif isinstance(var.type, Struct):
                # nested struct
                new_struct = value._cls()
                setattr(dst, name, new_struct)
                # The call to `setattr()` just above makes a copy of `new_struct`
                # so we need to reference that new instance of the struct.
                new_struct = getattr(dst, name)
                stack.extend((value, new_struct, k, v) for k, v in value._cls.vars.items())
            else:
                setattr(dst, name, value)

        return out

    # type description used in numpy structured arrays
    def numpy_dtype(self):
        return self._cls.numpy_dtype()

    # value usable in numpy structured arrays of .numpy_dtype(), e.g. (42, 13.37, [1.0, 2.0, 3.0])
    def numpy_value(self):
        npvalue = []
        for name, var in self._cls.vars.items():
            # get the attribute value
            value = getattr(self._ctype, name)

            if isinstance(var.type, array):
                # array_t
                npvalue.append(value.numpy_value())
            elif isinstance(var.type, Struct):
                # nested struct
                npvalue.append(value.numpy_value())
            elif issubclass(var.type, ctypes.Array):
                if len(var.type._shape_) == 1:
                    # vector
                    npvalue.append(list(value))
                else:
                    # matrix
                    npvalue.append([list(row) for row in value])
            else:
                # scalar
                if var.type == warp.float16:
                    npvalue.append(half_bits_to_float(value))
                else:
                    npvalue.append(value)

        return tuple(npvalue)


class Struct:
    hash: bytes

    def __init__(self, key: str, cls: type, module: warp.context.Module):
        self.key = key
        self.cls = cls
        self.module = module
        self.vars: dict[str, Var] = {}

        if isinstance(self.cls, Sequence):
            raise RuntimeError("Warp structs must be defined as base classes")

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
                # HACK: fp16 requires conversion functions from warp.so
                if var.type is warp.float16:
                    warp.init()
                fields.append((label, var.type._type_))

        class StructType(ctypes.Structure):
            # if struct is empty, add a dummy field to avoid launch errors on CPU device ("ffi_prep_cif failed")
            _fields_ = fields or [("_dummy_", ctypes.c_byte)]

        self.ctype = StructType

        # Compute the hash.  We can cache the hash because it's static, even with nested structs.
        # All field types are specified in the annotations, so they're resolved at declaration time.
        ch = hashlib.sha256()

        ch.update(bytes(self.key, "utf-8"))

        for name, type_hint in annotations.items():
            s = f"{name}:{warp.types.get_type_code(type_hint)}"
            ch.update(bytes(s, "utf-8"))

            # recurse on nested structs
            if isinstance(type_hint, Struct):
                ch.update(type_hint.hash)

        self.hash = ch.digest()

        # generate unique identifier for structs in native code
        hash_suffix = f"{self.hash.hex()[:8]}"
        self.native_name = f"{self.key}_{hash_suffix}"

        # create default constructor (zero-initialize)
        self.default_constructor = warp.context.Function(
            func=None,
            key=self.native_name,
            namespace="",
            value_func=lambda *_: self,
            input_types={},
            initializer_list_func=lambda *_: False,
            native_func=self.native_name,
        )

        # build a constructor that takes each param as a value
        input_types = {label: var.type for label, var in self.vars.items()}

        self.value_constructor = warp.context.Function(
            func=None,
            key=self.native_name,
            namespace="",
            value_func=lambda *_: self,
            input_types=input_types,
            initializer_list_func=lambda *_: False,
            native_func=self.native_name,
        )

        self.default_constructor.add_overload(self.value_constructor)

        if isinstance(module, warp.context.Module):
            module.register_struct(self)

        # Define class for instances of this struct
        # To enable autocomplete on s, we inherit from self.cls.
        # For example,

        # @wp.struct
        # class A:
        #     # annotations
        #     ...

        # The type annotations are inherited in A(), allowing autocomplete in kernels
        class NewStructInstance(self.cls, StructInstance):
            def __init__(inst, ctype=None):
                StructInstance.__init__(inst, self, ctype)

        # make sure warp.types.get_type_code works with this StructInstance
        NewStructInstance.cls = self.cls
        NewStructInstance.native_name = self.native_name

        self.instance_type = NewStructInstance

    def __call__(self):
        """
        This function returns s = StructInstance(self)
        s uses self.cls as template.
        """
        return self.instance_type()

    def initializer(self):
        return self.default_constructor

    # return structured NumPy dtype, including field names, formats, and offsets
    def numpy_dtype(self):
        names = []
        formats = []
        offsets = []
        for name, var in self.vars.items():
            names.append(name)
            offsets.append(getattr(self.ctype, name).offset)
            if isinstance(var.type, array):
                # array_t
                formats.append(array_t.numpy_dtype())
            elif isinstance(var.type, Struct):
                # nested struct
                formats.append(var.type.numpy_dtype())
            elif issubclass(var.type, ctypes.Array):
                scalar_typestr = type_typestr(var.type._wp_scalar_type_)
                if len(var.type._shape_) == 1:
                    # vector
                    formats.append(f"{var.type._length_}{scalar_typestr}")
                else:
                    # matrix
                    formats.append(f"{var.type._shape_}{scalar_typestr}")
            else:
                # scalar
                formats.append(type_typestr(var.type))

        return {"names": names, "formats": formats, "offsets": offsets, "itemsize": ctypes.sizeof(self.ctype)}

    # constructs a Warp struct instance from a pointer to the ctype
    def from_ptr(self, ptr):
        if not ptr:
            raise RuntimeError("NULL pointer exception")

        # create a new struct instance
        instance = self()

        for name, var in self.vars.items():
            offset = getattr(self.ctype, name).offset
            if isinstance(var.type, array):
                # We could reconstruct wp.array from array_t, but it's problematic.
                # There's no guarantee that the original wp.array is still allocated and
                # no easy way to make a backref.
                # Instead, we just create a stub annotation, which is not a fully usable array object.
                setattr(instance, name, array(dtype=var.type.dtype, ndim=var.type.ndim))
            elif isinstance(var.type, Struct):
                # nested struct
                value = var.type.from_ptr(ptr + offset)
                setattr(instance, name, value)
            elif issubclass(var.type, ctypes.Array):
                # vector/matrix
                value = var.type.from_ptr(ptr + offset)
                setattr(instance, name, value)
            else:
                # scalar
                cvalue = ctypes.cast(ptr + offset, ctypes.POINTER(var.type._type_)).contents
                if var.type == warp.float16:
                    setattr(instance, name, half_bits_to_float(cvalue))
                else:
                    setattr(instance, name, cvalue.value)

        return instance


class Reference:
    def __init__(self, value_type):
        self.value_type = value_type


def is_reference(type: Any) -> builtins.bool:
    return isinstance(type, Reference)


def strip_reference(arg: Any) -> Any:
    if is_reference(arg):
        return arg.value_type
    else:
        return arg


def compute_type_str(base_name, template_params):
    if not template_params:
        return base_name

    def param2str(p):
        if isinstance(p, int):
            return str(p)
        elif hasattr(p, "_type_"):
            if p.__name__ == "bool":
                return "bool"
            else:
                return f"wp::{p.__name__}"
        elif is_tile(p):
            return p.ctype()

        return p.__name__

    return f"{base_name}<{','.join(map(param2str, template_params))}>"


class Var:
    def __init__(
        self,
        label: str,
        type: type,
        requires_grad: builtins.bool = False,
        constant: builtins.bool | None = None,
        prefix: builtins.bool = True,
        relative_lineno: int | None = None,
    ):
        # convert built-in types to wp types
        if type == float:
            type = float32
        elif type == int:
            type = int32
        elif type == builtins.bool:
            type = bool

        self.label = label
        self.type = type
        self.requires_grad = requires_grad
        self.constant = constant
        self.prefix = prefix

        # records whether this Var has been read from in a kernel function (array only)
        self.is_read = False
        # records whether this Var has been written to in a kernel function (array only)
        self.is_write = False

        # used to associate a view array Var with its parent array Var
        self.parent = None

        # Used to associate the variable with the Python statement that resulted in it being created.
        self.relative_lineno = relative_lineno

    def __str__(self):
        return self.label

    @staticmethod
    def type_to_ctype(t: type, value_type: builtins.bool = False) -> str:
        if is_array(t):
            if hasattr(t.dtype, "_wp_generic_type_str_"):
                dtypestr = compute_type_str(f"wp::{t.dtype._wp_generic_type_str_}", t.dtype._wp_type_params_)
            elif isinstance(t.dtype, Struct):
                dtypestr = t.dtype.native_name
            elif t.dtype.__name__ in ("bool", "int", "float"):
                dtypestr = t.dtype.__name__
            else:
                dtypestr = f"wp::{t.dtype.__name__}"
            classstr = f"wp::{type(t).__name__}"
            return f"{classstr}_t<{dtypestr}>"
        elif is_tile(t):
            return t.ctype()
        elif isinstance(t, Struct):
            return t.native_name
        elif isinstance(t, type) and issubclass(t, StructInstance):
            # ensure the actual Struct name is used instead of "NewStructInstance"
            return t.native_name
        elif is_reference(t):
            if not value_type:
                return Var.type_to_ctype(t.value_type) + "*"
            else:
                return Var.type_to_ctype(t.value_type)
        elif hasattr(t, "_wp_generic_type_str_"):
            return compute_type_str(f"wp::{t._wp_generic_type_str_}", t._wp_type_params_)
        elif t.__name__ in ("bool", "int", "float"):
            return t.__name__
        else:
            return f"wp::{t.__name__}"

    def ctype(self, value_type: builtins.bool = False) -> str:
        return Var.type_to_ctype(self.type, value_type)

    def emit(self, prefix: str = "var"):
        if self.prefix:
            return f"{prefix}_{self.label}"
        else:
            return self.label

    def emit_adj(self):
        return self.emit("adj")

    def mark_read(self):
        """Marks this Var as having been read from in a kernel (array only)."""
        if not is_array(self.type):
            return

        self.is_read = True

        # recursively update all parent states
        parent = self.parent
        while parent is not None:
            parent.is_read = True
            parent = parent.parent

    def mark_write(self, **kwargs):
        """Marks this Var has having been written to in a kernel (array only)."""
        if not is_array(self.type):
            return

        # detect if we are writing to an array after reading from it within the same kernel
        if self.is_read and warp.config.verify_autograd_array_access:
            if "kernel_name" and "filename" and "lineno" in kwargs:
                print(
                    f"Warning: Array passed to argument {self.label} in kernel {kwargs['kernel_name']} at {kwargs['filename']}:{kwargs['lineno']} is being written to after it has been read from within the same kernel. This may corrupt gradient computation in the backward pass."
                )
            else:
                print(
                    f"Warning: Array {self} is being written to after it has been read from within the same kernel. This may corrupt gradient computation in the backward pass."
                )
        self.is_write = True

        # recursively update all parent states
        parent = self.parent
        while parent is not None:
            parent.is_write = True
            parent = parent.parent


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


def apply_defaults(
    bound_args: inspect.BoundArguments,
    values: Mapping[str, Any],
):
    # Similar to Python's `inspect.BoundArguments.apply_defaults()`
    # but with the possibility to pass an augmented set of default values.
    arguments = bound_args.arguments
    new_arguments = []
    for name in bound_args._signature.parameters.keys():
        try:
            new_arguments.append((name, arguments[name]))
        except KeyError:
            if name in values:
                new_arguments.append((name, values[name]))

    bound_args.arguments = dict(new_arguments)


def func_match_args(func, arg_types, kwarg_types):
    try:
        # Try to bind the given arguments to the function's signature.
        # This is not checking whether the argument types are matching,
        # rather it's just assigning each argument to the corresponding
        # function parameter.
        bound_arg_types = func.signature.bind(*arg_types, **kwarg_types)
    except TypeError:
        return False

    # Populate the bound arguments with any default values.
    default_arg_types = {
        k: None if v is None else get_arg_type(v)
        for k, v in func.defaults.items()
        if k not in bound_arg_types.arguments
    }
    apply_defaults(bound_arg_types, default_arg_types)
    bound_arg_types = tuple(bound_arg_types.arguments.values())

    # Check the given argument types against the ones defined on the function.
    for bound_arg_type, func_arg_type in zip(bound_arg_types, func.input_types.values()):
        # Let the `value_func` callback infer the type.
        if bound_arg_type is None:
            continue

        # if arg type registered as Any, treat as
        # template allowing any type to match
        if func_arg_type == Any:
            continue

        # handle function refs as a special case
        if func_arg_type == Callable and isinstance(bound_arg_type, warp.context.Function):
            continue

        # check arg type matches input variable type
        if not types_equal(func_arg_type, strip_reference(bound_arg_type), match_generic=True):
            return False

    return True


def get_arg_type(arg: Union[Var, Any]) -> type:
    if isinstance(arg, str):
        return str

    if isinstance(arg, Sequence):
        return tuple(get_arg_type(x) for x in arg)

    if isinstance(arg, (type, warp.context.Function)):
        return arg

    if isinstance(arg, Var):
        return arg.type

    return type(arg)


def get_arg_value(arg: Any) -> Any:
    if isinstance(arg, Sequence):
        return tuple(get_arg_value(x) for x in arg)

    if isinstance(arg, (type, warp.context.Function)):
        return arg

    if isinstance(arg, Var):
        return arg.constant

    return arg


class Adjoint:
    # Source code transformer, this class takes a Python function and
    # generates forward and backward SSA forms of the function instructions

    def __init__(
        adj,
        func: Callable[..., Any],
        overload_annotations=None,
        is_user_function=False,
        skip_forward_codegen=False,
        skip_reverse_codegen=False,
        custom_reverse_mode=False,
        custom_reverse_num_input_args=-1,
        transformers: list[ast.NodeTransformer] | None = None,
    ):
        adj.func = func

        adj.is_user_function = is_user_function

        # whether the generation of the forward code is skipped for this function
        adj.skip_forward_codegen = skip_forward_codegen
        # whether the generation of the adjoint code is skipped for this function
        adj.skip_reverse_codegen = skip_reverse_codegen

        # extract name of source file
        adj.filename = inspect.getsourcefile(func) or "unknown source file"
        # get source file line number where function starts
        try:
            _, adj.fun_lineno = inspect.getsourcelines(func)
        except OSError as e:
            raise RuntimeError(
                "Directly evaluating Warp code defined as a string using `exec()` is not supported, "
                "please save it on a file and use `importlib` if needed."
            ) from e

        # Indicates where the function definition starts (excludes decorators)
        adj.fun_def_lineno = None

        # get function source code
        adj.source = inspect.getsource(func)
        # ensures that indented class methods can be parsed as kernels
        adj.source = textwrap.dedent(adj.source)

        adj.source_lines = adj.source.splitlines()

        if transformers is None:
            transformers = []

        # build AST and apply node transformers
        adj.tree = ast.parse(adj.source)
        adj.transformers = transformers
        for transformer in transformers:
            adj.tree = transformer.visit(adj.tree)

        adj.fun_name = adj.tree.body[0].name

        # for keeping track of line number in function code
        adj.lineno = None

        # whether the forward code shall be used for the reverse pass and a custom
        # function signature is applied to the reverse version of the function
        adj.custom_reverse_mode = custom_reverse_mode
        # the number of function arguments that pertain to the forward function
        # input arguments (i.e. the number of arguments that are not adjoint arguments)
        adj.custom_reverse_num_input_args = custom_reverse_num_input_args

        # parse argument types
        argspec = get_full_arg_spec(func)

        # ensure all arguments are annotated
        if overload_annotations is None:
            # use source-level argument annotations
            if len(argspec.annotations) < len(argspec.args):
                raise WarpCodegenError(f"Incomplete argument annotations on function {adj.fun_name}")
            adj.arg_types = {k: v for k, v in argspec.annotations.items() if not (k == "return" and v is None)}
        else:
            # use overload argument annotations
            for arg_name in argspec.args:
                if arg_name not in overload_annotations:
                    raise WarpCodegenError(f"Incomplete overload annotations for function {adj.fun_name}")
            adj.arg_types = overload_annotations.copy()

        adj.args = []
        adj.symbols = {}

        for name, type in adj.arg_types.items():
            # skip return hint
            if name == "return":
                continue

            # add variable for argument
            arg = Var(name, type, False)
            adj.args.append(arg)

            # pre-populate symbol dictionary with function argument names
            # this is to avoid registering false references to overshadowed modules
            adj.symbols[name] = arg

        # try to replace static expressions by their constant result if the
        # expression can be evaluated at declaration time
        adj.static_expressions: dict[str, Any] = {}
        if "static" in adj.source:
            adj.replace_static_expressions()

        # There are cases where a same module might be rebuilt multiple times,
        # for example when kernels are nested inside of functions, or when
        # a kernel's launch raises an exception. Ideally we'd always want to
        # avoid rebuilding kernels but some corner cases seem to depend on it,
        # so we only avoid rebuilding kernels that errored out to give a chance
        # for unit testing errors being spit out from kernels.
        adj.skip_build = False

    # allocate extra space for a function call that requires its
    # own shared memory space, we treat shared memory as a stack
    # where each function pushes and pops space off, the extra
    # quantity is the 'roofline' amount required for the entire kernel
    def alloc_shared_extra(adj, num_bytes):
        adj.max_required_extra_shared_memory = max(adj.max_required_extra_shared_memory, num_bytes)

    # returns the total number of bytes for a function
    # based on it's own requirements + worst case
    # requirements of any dependent functions
    def get_total_required_shared(adj):
        total_shared = 0

        for var in adj.variables:
            if is_tile(var.type) and var.type.storage == "shared" and var.type.owner:
                total_shared += var.type.size_in_bytes()

        return total_shared + adj.max_required_extra_shared_memory

    # generate function ssa form and adjoint
    def build(adj, builder, default_builder_options=None):
        # arg Var read/write flags are held during module rebuilds, so we reset here even when skipping a build
        for arg in adj.args:
            arg.is_read = False
            arg.is_write = False

        if adj.skip_build:
            return

        adj.builder = builder

        if default_builder_options is None:
            default_builder_options = {}

        if adj.builder:
            adj.builder_options = adj.builder.options
        else:
            adj.builder_options = default_builder_options

        global options
        options = adj.builder_options

        adj.symbols = {}  # map from symbols to adjoint variables
        adj.variables = []  # list of local variables (in order)

        adj.return_var = None  # return type for function or kernel
        adj.loop_symbols = []  # symbols at the start of each loop
        adj.loop_const_iter_symbols = (
            set()
        )  # constant iteration variables for static loops (mutating them does not raise an error)

        # blocks
        adj.blocks = [Block()]
        adj.loop_blocks = []

        # holds current indent level
        adj.indentation = ""

        # used to generate new label indices
        adj.label_count = 0

        # tracks how much additional shared memory is required by any dependent function calls
        adj.max_required_extra_shared_memory = 0

        # update symbol map for each argument
        for a in adj.args:
            adj.symbols[a.label] = a

        # recursively evaluate function body
        try:
            adj.eval(adj.tree.body[0])
        except Exception:
            try:
                lineno = adj.lineno + adj.fun_lineno
                line = adj.source_lines[adj.lineno]
                msg = f'Error while parsing function "{adj.fun_name}" at {adj.filename}:{lineno}:\n{line}\n'
                ex, data, traceback = sys.exc_info()
                e = ex(";".join([msg] + [str(a) for a in data.args])).with_traceback(traceback)
            finally:
                adj.skip_build = True
                adj.builder = None
                raise e

        if builder is not None:
            for a in adj.args:
                if isinstance(a.type, Struct):
                    builder.build_struct_recursive(a.type)
                elif isinstance(a.type, warp.types.array) and isinstance(a.type.dtype, Struct):
                    builder.build_struct_recursive(a.type.dtype)

            # release builder reference for GC
            adj.builder = None

    # code generation methods
    def format_template(adj, template, input_vars, output_var):
        # output var is always the 0th index
        args = [output_var, *input_vars]
        s = template.format(*args)

        return s

    # generates a list of formatted args
    def format_args(adj, prefix, args):
        arg_strs = []

        for a in args:
            if isinstance(a, warp.context.Function):
                # functions don't have a var_ prefix so strip it off here
                if prefix == "var":
                    arg_strs.append(f"{a.namespace}{a.native_func}")
                else:
                    arg_strs.append(f"{a.namespace}{prefix}_{a.native_func}")
            elif is_reference(a.type):
                arg_strs.append(f"{prefix}_{a}")
            elif isinstance(a, Var):
                arg_strs.append(a.emit(prefix))
            else:
                raise WarpCodegenTypeError(f"Arguments must be variables or functions, got {type(a)}")

        return arg_strs

    # generates argument string for a forward function call
    def format_forward_call_args(adj, args, use_initializer_list):
        arg_str = ", ".join(adj.format_args("var", args))
        if use_initializer_list:
            return f"{{{arg_str}}}"
        return arg_str

    # generates argument string for a reverse function call
    def format_reverse_call_args(
        adj,
        args_var,
        args,
        args_out,
        use_initializer_list,
        has_output_args=True,
        require_original_output_arg=False,
    ):
        formatted_var = adj.format_args("var", args_var)
        formatted_out = []
        if has_output_args and (require_original_output_arg or len(args_out) > 1):
            formatted_out = adj.format_args("var", args_out)
        formatted_var_adj = adj.format_args(
            "&adj" if use_initializer_list else "adj",
            args,
        )
        formatted_out_adj = adj.format_args("adj", args_out)

        if len(formatted_var_adj) == 0 and len(formatted_out_adj) == 0:
            # there are no adjoint arguments, so we don't need to call the reverse function
            return None

        if use_initializer_list:
            var_str = f"{{{', '.join(formatted_var)}}}"
            out_str = f"{{{', '.join(formatted_out)}}}"
            adj_str = f"{{{', '.join(formatted_var_adj)}}}"
            out_adj_str = ", ".join(formatted_out_adj)
            if len(args_out) > 1:
                arg_str = ", ".join([var_str, out_str, adj_str, out_adj_str])
            else:
                arg_str = ", ".join([var_str, adj_str, out_adj_str])
        else:
            arg_str = ", ".join(formatted_var + formatted_out + formatted_var_adj + formatted_out_adj)
        return arg_str

    def indent(adj):
        adj.indentation = adj.indentation + "    "

    def dedent(adj):
        adj.indentation = adj.indentation[:-4]

    def begin_block(adj, name="block"):
        b = Block()

        # give block a unique id
        b.label = name + "_" + str(adj.label_count)
        adj.label_count += 1

        adj.blocks.append(b)
        return b

    def end_block(adj):
        return adj.blocks.pop()

    def add_var(adj, type=None, constant=None):
        index = len(adj.variables)
        name = str(index)

        # allocate new variable
        v = Var(name, type=type, constant=constant, relative_lineno=adj.lineno)

        adj.variables.append(v)

        adj.blocks[-1].vars.append(v)

        return v

    def register_var(adj, var):
        # We sometimes initialize `Var` instances that might be thrown away
        # afterwards, so this method allows to defer their registration among
        # the list of primal vars until later on, instead of registering them
        # immediately if we were to use `adj.add_var()` or `adj.add_constant()`.

        if isinstance(var, (Reference, warp.context.Function)):
            return var

        if isinstance(var, int):
            return adj.add_constant(var)

        if var.label is None:
            return adj.add_var(var.type, var.constant)

        return var

    def get_line_directive(adj, statement: str, relative_lineno: int | None = None) -> str | None:
        """Get a line directive for the given statement.

        Args:
            statement: The statement to get the line directive for.
            relative_lineno: The line number of the statement relative to the function.

        Returns:
            A line directive for the given statement, or None if no line directive is needed.
        """

        # lineinfo is enabled by default in debug mode regardless of the builder option, don't want to unnecessarily
        # emit line directives in generated code if it's not being compiled with line information
        lineinfo_enabled = (
            adj.builder_options.get("lineinfo", False) or adj.builder_options.get("mode", "release") == "debug"
        )

        if relative_lineno is not None and lineinfo_enabled and warp.config.line_directives:
            is_comment = statement.strip().startswith("//")
            if not is_comment:
                line = relative_lineno + adj.fun_lineno
                # Convert backslashes to forward slashes for CUDA compatibility
                normalized_path = adj.filename.replace("\\", "/")
                return f'#line {line} "{normalized_path}"'
        return None

    def add_forward(adj, statement: str, replay: str | None = None, skip_replay: builtins.bool = False) -> None:
        """Append a statement to the forward pass."""

        if line_directive := adj.get_line_directive(statement, adj.lineno):
            adj.blocks[-1].body_forward.append(line_directive)

        adj.blocks[-1].body_forward.append(adj.indentation + statement)

        if not skip_replay:
            if line_directive:
                adj.blocks[-1].body_replay.append(line_directive)

            if replay:
                # if custom replay specified then output it
                adj.blocks[-1].body_replay.append(adj.indentation + replay)
            else:
                # by default just replay the original statement
                adj.blocks[-1].body_replay.append(adj.indentation + statement)

    # append a statement to the reverse pass
    def add_reverse(adj, statement: str) -> None:
        """Append a statement to the reverse pass."""

        adj.blocks[-1].body_reverse.append(adj.indentation + statement)

        if line_directive := adj.get_line_directive(statement, adj.lineno):
            adj.blocks[-1].body_reverse.append(line_directive)

    def add_constant(adj, n):
        output = adj.add_var(type=type(n), constant=n)
        return output

    def load(adj, var):
        if is_reference(var.type):
            var = adj.add_builtin_call("load", [var])
        return var

    def add_comp(adj, op_strings, left, comps):
        output = adj.add_var(builtins.bool)

        left = adj.load(left)
        s = output.emit() + " = " + ("(" * len(comps)) + left.emit() + " "

        prev_comp_var = None

        for op, comp in zip(op_strings, comps):
            comp_chainable = op_str_is_chainable(op)
            if comp_chainable and prev_comp_var:
                # We restrict chaining to operands of the same type
                if prev_comp_var.type is comp.type:
                    prev_comp_var = adj.load(prev_comp_var)
                    comp_var = adj.load(comp)
                    s += "&& (" + prev_comp_var.emit() + " " + op + " " + comp_var.emit() + ")) "
                else:
                    raise WarpCodegenTypeError(
                        f"Cannot chain comparisons of unequal types: {prev_comp_var.type} {op} {comp.type}."
                    )
            else:
                comp_var = adj.load(comp)
                s += op + " " + comp_var.emit() + ") "

            prev_comp_var = comp_var

        s = s.rstrip() + ";"

        adj.add_forward(s)

        return output

    def add_bool_op(adj, op_string, exprs):
        exprs = [adj.load(expr) for expr in exprs]
        output = adj.add_var(builtins.bool)
        command = output.emit() + " = " + (" " + op_string + " ").join([expr.emit() for expr in exprs]) + ";"
        adj.add_forward(command)

        return output

    def resolve_func(adj, func, arg_types, kwarg_types, min_outputs):
        if not func.is_builtin():
            # user-defined function
            overload = func.get_overload(arg_types, kwarg_types)
            if overload is not None:
                return overload
        else:
            # if func is overloaded then perform overload resolution here
            # we validate argument types before they go to generated native code
            for f in func.overloads:
                # skip type checking for variadic functions
                if not f.variadic:
                    # check argument counts match are compatible (may be some default args)
                    if len(f.input_types) < len(arg_types) + len(kwarg_types):
                        continue

                    if not func_match_args(f, arg_types, kwarg_types):
                        continue

                # check output dimensions match expectations
                if min_outputs:
                    value_type = f.value_func(None, None)
                    if not isinstance(value_type, Sequence) or len(value_type) != min_outputs:
                        continue

                # found a match, use it
                return f

        # unresolved function, report error
        arg_type_reprs = []

        for x in arg_types:
            if isinstance(x, warp.context.Function):
                arg_type_reprs.append("function")
            else:
                # shorten Warp primitive type names
                if isinstance(x, Sequence):
                    if len(x) != 1:
                        raise WarpCodegenError("Argument must not be the result from a multi-valued function")
                    arg_type = x[0]
                else:
                    arg_type = x

                arg_type_reprs.append(type_repr(arg_type))

        raise WarpCodegenError(
            f"Couldn't find function overload for '{func.key}' that matched inputs with types: [{', '.join(arg_type_reprs)}]"
        )

    def add_call(adj, func, args, kwargs, type_args, min_outputs=None):
        # Extract the types and values passed as arguments to the function call.
        arg_types = tuple(strip_reference(get_arg_type(x)) for x in args)
        kwarg_types = {k: strip_reference(get_arg_type(v)) for k, v in kwargs.items()}

        # Resolve the exact function signature among any existing overload.
        func = adj.resolve_func(func, arg_types, kwarg_types, min_outputs)

        # Bind the positional and keyword arguments to the function's signature
        # in order to process them as Python does it.
        bound_args: inspect.BoundArguments = func.signature.bind(*args, **kwargs)

        # Type args are the “compile time” argument values we get from codegen.
        # For example, when calling `wp.vec3f(...)` from within a kernel,
        # this translates in fact to calling the `vector()` built-in augmented
        # with the type args `length=3, dtype=float`.
        # Eventually, these need to be passed to the underlying C++ function,
        # so we update the arguments with the type args here.
        if type_args:
            for arg in type_args:
                if arg in bound_args.arguments:
                    # In case of conflict, ideally we'd throw an error since
                    # what comes from codegen should be the source of truth
                    # and users also passing the same value as an argument
                    # is redundant (e.g.: `wp.mat22(shape=(2, 2))`).
                    # However, for backward compatibility, we allow that form
                    # as long as the values are equal.
                    if values_check_equal(get_arg_value(bound_args.arguments[arg]), type_args[arg]):
                        continue

                    raise RuntimeError(
                        f"Remove the extraneous `{arg}` parameter "
                        f"when calling the templated version of "
                        f"`wp.{func.native_func}()`"
                    )

            type_vars = {k: Var(None, type=type(v), constant=v) for k, v in type_args.items()}
            apply_defaults(bound_args, type_vars)

        if func.defaults:
            default_vars = {
                k: Var(None, type=type(v), constant=v)
                for k, v in func.defaults.items()
                if k not in bound_args.arguments and v is not None
            }
            apply_defaults(bound_args, default_vars)

        bound_args = bound_args.arguments

        # if it is a user-function then build it recursively
        if not func.is_builtin() and func not in adj.builder.functions:
            adj.builder.build_function(func)
            # add custom grad, replay functions to the list of functions
            # to be built later (invalid code could be generated if we built them now)
            # so that they are not missed when only the forward function is imported
            # from another module
            if func.custom_grad_func:
                adj.builder.deferred_functions.append(func.custom_grad_func)
            if func.custom_replay_func:
                adj.builder.deferred_functions.append(func.custom_replay_func)

        # Resolve the return value based on the types and values of the given arguments.
        bound_arg_types = {k: get_arg_type(v) for k, v in bound_args.items()}
        bound_arg_values = {k: get_arg_value(v) for k, v in bound_args.items()}
        return_type = func.value_func(
            {k: strip_reference(v) for k, v in bound_arg_types.items()},
            bound_arg_values,
        )

        # immediately allocate output variables so we can pass them into the dispatch method
        if return_type is None:
            # void function
            output = None
            output_list = []
        elif not isinstance(return_type, Sequence) or len(return_type) == 1:
            # single return value function
            if isinstance(return_type, Sequence):
                return_type = return_type[0]
            output = adj.add_var(return_type)
            output_list = [output]
        else:
            # multiple return value function
            output = [adj.add_var(v) for v in return_type]
            output_list = output

        # If we have a built-in that requires special handling to dispatch
        # the arguments to the underlying C++ function, then we can resolve
        # these using the `dispatch_func`. Since this is only called from
        # within codegen, we pass it directly `codegen.Var` objects,
        # which allows for some more advanced resolution to be performed,
        # for example by checking whether an argument corresponds to
        # a literal value or references a variable.
        extra_shared_memory = 0
        if func.lto_dispatch_func is not None:
            func_args, template_args, ltoirs, extra_shared_memory = func.lto_dispatch_func(
                func.input_types, return_type, output_list, bound_args, options=adj.builder_options, builder=adj.builder
            )
        elif func.dispatch_func is not None:
            func_args, template_args = func.dispatch_func(func.input_types, return_type, bound_args)
        else:
            func_args = tuple(bound_args.values())
            template_args = ()

        func_args = tuple(adj.register_var(x) for x in func_args)
        func_name = compute_type_str(func.native_func, template_args)
        use_initializer_list = func.initializer_list_func(bound_args, return_type)

        fwd_args = []
        for func_arg in func_args:
            if not isinstance(func_arg, (Reference, warp.context.Function)):
                func_arg_var = adj.load(func_arg)
            else:
                func_arg_var = func_arg

            # if the argument is a function (and not a builtin), then build it recursively
            if isinstance(func_arg_var, warp.context.Function) and not func_arg_var.is_builtin():
                adj.builder.build_function(func_arg_var)

            fwd_args.append(strip_reference(func_arg_var))

        if return_type is None:
            # handles expression (zero output) functions, e.g.: void do_something();
            forward_call = (
                f"{func.namespace}{func_name}({adj.format_forward_call_args(fwd_args, use_initializer_list)});"
            )
            replay_call = forward_call
            if func.custom_replay_func is not None or func.replay_snippet is not None:
                replay_call = f"{func.namespace}replay_{func_name}({adj.format_forward_call_args(fwd_args, use_initializer_list)});"

        elif not isinstance(return_type, Sequence) or len(return_type) == 1:
            # handle simple function (one output)
            forward_call = f"var_{output} = {func.namespace}{func_name}({adj.format_forward_call_args(fwd_args, use_initializer_list)});"
            replay_call = forward_call
            if func.custom_replay_func is not None:
                replay_call = f"var_{output} = {func.namespace}replay_{func_name}({adj.format_forward_call_args(fwd_args, use_initializer_list)});"

        else:
            # handle multiple value functions
            forward_call = (
                f"{func.namespace}{func_name}({adj.format_forward_call_args(fwd_args + output, use_initializer_list)});"
            )
            replay_call = forward_call

        if func.skip_replay:
            adj.add_forward(forward_call, replay="// " + replay_call)
        else:
            adj.add_forward(forward_call, replay=replay_call)

        if not func.missing_grad and len(func_args):
            adj_args = tuple(strip_reference(x) for x in func_args)
            reverse_has_output_args = (
                func.require_original_output_arg or len(output_list) > 1
            ) and func.custom_grad_func is None
            arg_str = adj.format_reverse_call_args(
                fwd_args,
                adj_args,
                output_list,
                use_initializer_list,
                has_output_args=reverse_has_output_args,
                require_original_output_arg=func.require_original_output_arg,
            )
            if arg_str is not None:
                reverse_call = f"{func.namespace}adj_{func.native_func}({arg_str});"
                adj.add_reverse(reverse_call)

        # update our smem roofline requirements based on any
        # shared memory required by the dependent function call
        if not func.is_builtin():
            adj.alloc_shared_extra(func.adj.get_total_required_shared() + extra_shared_memory)
        else:
            adj.alloc_shared_extra(extra_shared_memory)

        return output

    def add_builtin_call(adj, func_name, args, min_outputs=None):
        func = warp.context.builtin_functions[func_name]
        return adj.add_call(func, args, {}, {}, min_outputs=min_outputs)

    def add_return(adj, var):
        if var is None or len(var) == 0:
            # NOTE: If this kernel gets compiled for a CUDA device, then we need
            # to convert the return; into a continue; in codegen_func_forward()
            adj.add_forward("return;", f"goto label{adj.label_count};")
        elif len(var) == 1:
            adj.add_forward(f"return {var[0].emit()};", f"goto label{adj.label_count};")
            adj.add_reverse("adj_" + str(var[0]) + " += adj_ret;")
        else:
            for i, v in enumerate(var):
                adj.add_forward(f"ret_{i} = {v.emit()};")
                adj.add_reverse(f"adj_{v} += adj_ret_{i};")
            adj.add_forward("return;", f"goto label{adj.label_count};")

        adj.add_reverse(f"label{adj.label_count}:;")

        adj.label_count += 1

    # define an if statement
    def begin_if(adj, cond):
        cond = adj.load(cond)
        adj.add_forward(f"if ({cond.emit()}) {{")
        adj.add_reverse("}")

        adj.indent()

    def end_if(adj, cond):
        adj.dedent()

        adj.add_forward("}")
        cond = adj.load(cond)
        adj.add_reverse(f"if ({cond.emit()}) {{")

    def begin_else(adj, cond):
        cond = adj.load(cond)
        adj.add_forward(f"if (!{cond.emit()}) {{")
        adj.add_reverse("}")

        adj.indent()

    def end_else(adj, cond):
        adj.dedent()

        adj.add_forward("}")
        cond = adj.load(cond)
        adj.add_reverse(f"if (!{cond.emit()}) {{")

    # define a for-loop
    def begin_for(adj, iter):
        cond_block = adj.begin_block("for")
        adj.loop_blocks.append(cond_block)
        adj.add_forward(f"start_{cond_block.label}:;")
        adj.indent()

        # evaluate cond
        adj.add_forward(f"if (iter_cmp({iter.emit()}) == 0) goto end_{cond_block.label};")

        # evaluate iter
        val = adj.add_builtin_call("iter_next", [iter])

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

        adj.add_forward(f"goto start_{cond_block.label};", skip_replay=True)

        adj.dedent()
        adj.add_forward(f"end_{cond_block.label}:;", skip_replay=True)

        ####################
        # reverse pass

        reverse = []

        # reverse iterator
        reverse.append(adj.indentation + f"{iter.emit()} = wp::iter_reverse({iter.emit()});")

        for i in cond_block.body_forward:
            reverse.append(i)

        # zero adjoints
        for i in body_block.vars:
            if is_tile(i.type):
                if i.type.owner:
                    reverse.append(adj.indentation + f"\t{i.emit_adj()}.grad_zero();")
            else:
                reverse.append(adj.indentation + f"\t{i.emit_adj()} = {{}};")

        # replay
        for i in body_block.body_replay:
            reverse.append(i)

        # reverse
        for i in reversed(body_block.body_reverse):
            reverse.append(i)

        reverse.append(adj.indentation + f"\tgoto start_{cond_block.label};")
        reverse.append(adj.indentation + f"end_{cond_block.label}:;")

        adj.blocks[-1].body_reverse.extend(reversed(reverse))

    # define a while loop
    def begin_while(adj, cond):
        # evaluate condition in its own block
        # so we can control replay
        cond_block = adj.begin_block("while")
        adj.loop_blocks.append(cond_block)
        cond_block.body_forward.append(f"start_{cond_block.label}:;")

        c = adj.eval(cond)
        c = adj.load(c)

        cond_block.body_forward.append(f"if (({c.emit()}) == false) goto end_{cond_block.label};")

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

        adj.blocks[-1].body_forward.append(f"goto start_{cond_block.label};")
        adj.blocks[-1].body_forward.append(f"end_{cond_block.label}:;")

        ####################
        # reverse pass
        reverse = []

        # cond
        for i in cond_block.body_forward:
            reverse.append(i)

        # zero adjoints of local vars
        for i in body_block.vars:
            reverse.append(f"{i.emit_adj()} = {{}};")

        # replay
        for i in body_block.body_replay:
            reverse.append(i)

        # reverse
        for i in reversed(body_block.body_reverse):
            reverse.append(i)

        reverse.append(f"goto start_{cond_block.label};")
        reverse.append(f"end_{cond_block.label}:;")

        # output
        adj.blocks[-1].body_reverse.extend(reversed(reverse))

    def emit_FunctionDef(adj, node):
        adj.fun_def_lineno = node.lineno

        for f in node.body:
            # Skip variable creation for standalone constants, including docstrings
            if isinstance(f, ast.Expr) and isinstance(f.value, ast.Constant):
                continue
            adj.eval(f)

        if adj.return_var is not None and len(adj.return_var) == 1:
            if not isinstance(node.body[-1], ast.Return):
                adj.add_forward("return {};", skip_replay=True)

        # native function case: return type is specified, eg -> int or -> wp.float32
        is_func_native = False
        if node.decorator_list is not None and len(node.decorator_list) == 1:
            obj = node.decorator_list[0]
            if isinstance(obj, ast.Call):
                if isinstance(obj.func, ast.Attribute):
                    if obj.func.attr == "func_native":
                        is_func_native = True
        if is_func_native and node.returns is not None:
            if isinstance(node.returns, ast.Name):  # python built-in type
                var = Var(label="return_type", type=eval(node.returns.id))
            elif isinstance(node.returns, ast.Attribute):  # warp type
                var = Var(label="return_type", type=eval(node.returns.attr))
            else:
                raise WarpCodegenTypeError("Native function return type not recognized")
            adj.return_var = (var,)

    def emit_If(adj, node):
        if len(node.body) == 0:
            return None

        # eval condition
        cond = adj.eval(node.test)

        if cond.constant is not None:
            # resolve constant condition
            if cond.constant:
                for stmt in node.body:
                    adj.eval(stmt)
            else:
                for stmt in node.orelse:
                    adj.eval(stmt)
            return None

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
                out = adj.add_builtin_call("where", [cond, var2, var1])
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
                out = adj.add_builtin_call("where", [cond, var1, var2])
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
            raise WarpCodegenKeyError(f"Op {op} is not supported")

        return adj.add_bool_op(func, [adj.eval(expr) for expr in node.values])

    def emit_Name(adj, node):
        # lookup symbol, if it has already been assigned to a variable then return the existing mapping
        if node.id in adj.symbols:
            return adj.symbols[node.id]

        obj = adj.resolve_external_reference(node.id)

        if obj is None:
            raise WarpCodegenKeyError("Referencing undefined symbol: " + str(node.id))

        if warp.types.is_value(obj):
            # evaluate constant
            out = adj.add_constant(obj)
            adj.symbols[node.id] = out
            return out

        # the named object is either a function, class name, or module
        # pass it back to the caller for processing
        if isinstance(obj, warp.context.Function):
            return obj
        if isinstance(obj, type):
            return obj
        if isinstance(obj, types.ModuleType):
            return obj

        raise TypeError(f"Invalid external reference type: {type(obj)}")

    @staticmethod
    def resolve_type_attribute(var_type: type, attr: str):
        if isinstance(var_type, type) and type_is_value(var_type):
            if attr == "dtype":
                return type_scalar_type(var_type)
            elif attr == "length":
                return type_length(var_type)

        return getattr(var_type, attr, None)

    def vector_component_index(adj, component, vector_type):
        if len(component) != 1:
            raise WarpCodegenAttributeError(f"Vector swizzle must be single character, got .{component}")

        dim = vector_type._shape_[0]
        swizzles = "xyzw"[0:dim]
        if component not in swizzles:
            raise WarpCodegenAttributeError(
                f"Vector swizzle for {vector_type} must be one of {swizzles}, got {component}"
            )

        index = swizzles.index(component)
        index = adj.add_constant(index)
        return index

    @staticmethod
    def is_differentiable_value_type(var_type):
        # checks that the argument type is a value type (i.e, not an array)
        # possibly holding differentiable values (for which gradients must be accumulated)
        return type_scalar_type(var_type) in float_types or isinstance(var_type, Struct)

    def emit_Attribute(adj, node):
        if hasattr(node, "is_adjoint"):
            node.value.is_adjoint = True

        aggregate = adj.eval(node.value)

        try:
            if isinstance(aggregate, types.ModuleType) or isinstance(aggregate, type):
                out = getattr(aggregate, node.attr)

                if warp.types.is_value(out):
                    return adj.add_constant(out)

                return out

            if hasattr(node, "is_adjoint"):
                # create a Var that points to the struct attribute, i.e.: directly generates `struct.attr` when used
                attr_name = aggregate.label + "." + node.attr
                attr_type = aggregate.type.vars[node.attr].type

                return Var(attr_name, attr_type)

            aggregate_type = strip_reference(aggregate.type)

            # reading a vector or quaternion component
            if type_is_vector(aggregate_type) or type_is_quaternion(aggregate_type):
                index = adj.vector_component_index(node.attr, aggregate_type)

                return adj.add_builtin_call("extract", [aggregate, index])

            else:
                attr_type = Reference(aggregate_type.vars[node.attr].type)
                attr = adj.add_var(attr_type)

                if is_reference(aggregate.type):
                    adj.add_forward(f"{attr.emit()} = &({aggregate.emit()}->{node.attr});")
                else:
                    adj.add_forward(f"{attr.emit()} = &({aggregate.emit()}.{node.attr});")

                if adj.is_differentiable_value_type(strip_reference(attr_type)):
                    adj.add_reverse(f"{aggregate.emit_adj()}.{node.attr} += {attr.emit_adj()};")
                else:
                    adj.add_reverse(f"{aggregate.emit_adj()}.{node.attr} = {attr.emit_adj()};")

                return attr

        except (KeyError, AttributeError) as e:
            # Try resolving as type attribute
            aggregate_type = strip_reference(aggregate.type) if isinstance(aggregate, Var) else aggregate

            type_attribute = adj.resolve_type_attribute(aggregate_type, node.attr)
            if type_attribute is not None:
                return type_attribute

            if isinstance(aggregate, Var):
                raise WarpCodegenAttributeError(
                    f"Error, `{node.attr}` is not an attribute of '{node.value.id}' ({type_repr(aggregate.type)})"
                ) from e
            raise WarpCodegenAttributeError(f"Error, `{node.attr}` is not an attribute of '{aggregate}'") from e

    def emit_Assert(adj, node):
        # eval condition
        cond = adj.eval(node.test)
        cond = adj.load(cond)

        source_segment = ast.get_source_segment(adj.source, node)
        # If a message was provided with the assert, " marks can interfere with the generated code
        escaped_segment = source_segment.replace('"', '\\"')

        adj.add_forward(f'assert(("{escaped_segment}",{cond.emit()}));')

    def emit_Constant(adj, node):
        if node.value is None:
            raise WarpCodegenTypeError("None type unsupported")
        else:
            return adj.add_constant(node.value)

    def emit_BinOp(adj, node):
        # evaluate binary operator arguments

        if warp.config.verify_autograd_array_access:
            # array overwrite tracking: in-place operators are a special case
            # x[tid] = x[tid] + 1 is a read followed by a write, but we only want to record the write
            # so we save the current arg read flags and restore them after lhs eval
            is_read_states = []
            for arg in adj.args:
                is_read_states.append(arg.is_read)

        # evaluate lhs binary operator argument
        left = adj.eval(node.left)

        if warp.config.verify_autograd_array_access:
            # restore arg read flags
            for i, arg in enumerate(adj.args):
                arg.is_read = is_read_states[i]

        # evaluate rhs binary operator argument
        right = adj.eval(node.right)

        name = builtin_operators[type(node.op)]

        try:
            # Check if there is any user-defined overload for this operator
            user_func = adj.resolve_external_reference(name)
            if isinstance(user_func, warp.context.Function):
                return adj.add_call(user_func, (left, right), {}, {})
        except WarpCodegenError:
            pass

        return adj.add_builtin_call(name, [left, right])

    def emit_UnaryOp(adj, node):
        # evaluate unary op arguments
        arg = adj.eval(node.operand)

        # evaluate expression to a compile-time constant if arg is a constant
        if arg.constant is not None and math.isfinite(arg.constant):
            if isinstance(node.op, ast.USub):
                return adj.add_constant(-arg.constant)

        name = builtin_operators[type(node.op)]

        return adj.add_builtin_call(name, [arg])

    def materialize_redefinitions(adj, symbols):
        # detect symbols with conflicting definitions (assigned inside the for loop)
        for items in symbols.items():
            sym = items[0]
            if adj.is_constant_iter_symbol(sym):
                # ignore constant overwriting in for-loops if it is a loop iterator
                # (it is no problem to unroll static loops multiple times in sequence)
                continue

            var1 = items[1]
            var2 = adj.symbols[sym]

            if var1 != var2:
                if warp.config.verbose and not adj.custom_reverse_mode:
                    lineno = adj.lineno + adj.fun_lineno
                    line = adj.source_lines[adj.lineno]
                    msg = f'Warning: detected mutated variable {sym} during a dynamic for-loop in function "{adj.fun_name}" at {adj.filename}:{lineno}: this may not be a differentiable operation.\n{line}\n'
                    print(msg)

                if var1.constant is not None:
                    raise WarpCodegenError(
                        f"Error mutating a constant {sym} inside a dynamic loop, use the following syntax: pi = float(3.141) to declare a dynamic variable"
                    )

                # overwrite the old variable value (violates SSA)
                adj.add_builtin_call("assign", [var1, var2])

                # reset the symbol to point to the original variable
                adj.symbols[sym] = var1

    def emit_While(adj, node):
        adj.begin_while(node.test)

        adj.loop_symbols.append(adj.symbols.copy())

        # eval body
        for s in node.body:
            adj.eval(s)

        adj.materialize_redefinitions(adj.loop_symbols[-1])
        adj.loop_symbols.pop()

        adj.end_while()

    def eval_num(adj, a):
        if isinstance(a, ast.Constant):
            return True, a.value
        if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Constant):
            # Negative constant
            return True, -a.operand.value

        # try and resolve the expression to an object
        # e.g.: wp.constant in the globals scope
        obj, _ = adj.resolve_static_expression(a)

        if obj is None:
            obj = adj.eval(a)

        if isinstance(obj, Var) and obj.constant is not None:
            obj = obj.constant

        return warp.types.is_int(obj), obj

    # detects whether a loop contains a break (or continue) statement
    def contains_break(adj, body):
        for s in body:
            if isinstance(s, ast.Break):
                return True
            elif isinstance(s, ast.Continue):
                return True
            elif isinstance(s, ast.If):
                if adj.contains_break(s.body):
                    return True
                if adj.contains_break(s.orelse):
                    return True
            else:
                # note that nested for or while loops containing a break statement
                # do not affect the current loop
                pass

        return False

    # returns a constant range() if unrollable, otherwise None
    def get_unroll_range(adj, loop):
        if (
            not isinstance(loop.iter, ast.Call)
            or not isinstance(loop.iter.func, ast.Name)
            or loop.iter.func.id != "range"
            or len(loop.iter.args) == 0
            or len(loop.iter.args) > 3
        ):
            return None

        # if all range() arguments are numeric constants we will unroll
        # note that this only handles trivial constants, it will not unroll
        # constant compile-time expressions e.g.: range(0, 3*2)

        # Evaluate the arguments and check that they are numeric constants
        # It is important to do that in one pass, so that if evaluating these arguments have side effects
        # the code does not get generated more than once
        range_args = [adj.eval_num(arg) for arg in loop.iter.args]
        arg_is_numeric, arg_values = zip(*range_args)

        if all(arg_is_numeric):
            # All argument are numeric constants

            # range(end)
            if len(loop.iter.args) == 1:
                start = 0
                end = arg_values[0]
                step = 1

            # range(start, end)
            elif len(loop.iter.args) == 2:
                start = arg_values[0]
                end = arg_values[1]
                step = 1

            # range(start, end, step)
            elif len(loop.iter.args) == 3:
                start = arg_values[0]
                end = arg_values[1]
                step = arg_values[2]

            # test if we're above max unroll count
            max_iters = abs(end - start) // abs(step)

            if "max_unroll" in adj.builder_options:
                max_unroll = adj.builder_options["max_unroll"]
            else:
                max_unroll = warp.config.max_unroll

            ok_to_unroll = True

            if max_iters > max_unroll:
                if warp.config.verbose:
                    print(
                        f"Warning: fixed-size loop count of {max_iters} is larger than the module 'max_unroll' limit of {max_unroll}, will generate dynamic loop."
                    )
                ok_to_unroll = False

            elif adj.contains_break(loop.body):
                if warp.config.verbose:
                    print("Warning: 'break' or 'continue' found in loop body, will generate dynamic loop.")
                ok_to_unroll = False

            if ok_to_unroll:
                return range(start, end, step)

        # Unroll is not possible, range needs to be valuated dynamically
        range_call = adj.add_builtin_call(
            "range",
            [adj.add_constant(val) if is_numeric else val for is_numeric, val in range_args],
        )
        return range_call

    def record_constant_iter_symbol(adj, sym):
        adj.loop_const_iter_symbols.add(sym)

    def is_constant_iter_symbol(adj, sym):
        return sym in adj.loop_const_iter_symbols

    def emit_For(adj, node):
        # try and unroll simple range() statements that use constant args
        unroll_range = adj.get_unroll_range(node)

        if isinstance(unroll_range, range):
            const_iter_sym = node.target.id
            # prevent constant conflicts in `materialize_redefinitions()`
            adj.record_constant_iter_symbol(const_iter_sym)

            # unroll static for-loop
            for i in unroll_range:
                const_iter = adj.add_constant(i)
                adj.symbols[const_iter_sym] = const_iter

                # eval body
                for s in node.body:
                    adj.eval(s)

        # otherwise generate a dynamic loop
        else:
            # evaluate the Iterable -- only if not previously evaluated when trying to unroll
            if unroll_range is not None:
                # Range has already been evaluated when trying to unroll, do not re-evaluate
                iter = unroll_range
            else:
                iter = adj.eval(node.iter)

            adj.symbols[node.target.id] = adj.begin_for(iter)

            # for loops should be side-effect free, here we store a copy
            adj.loop_symbols.append(adj.symbols.copy())

            # eval body
            for s in node.body:
                adj.eval(s)

            adj.materialize_redefinitions(adj.loop_symbols[-1])
            adj.loop_symbols.pop()

            adj.end_for(iter)

    def emit_Break(adj, node):
        adj.materialize_redefinitions(adj.loop_symbols[-1])

        adj.add_forward(f"goto end_{adj.loop_blocks[-1].label};")

    def emit_Continue(adj, node):
        adj.materialize_redefinitions(adj.loop_symbols[-1])

        adj.add_forward(f"goto start_{adj.loop_blocks[-1].label};")

    def emit_Expr(adj, node):
        return adj.eval(node.value)

    def check_tid_in_func_error(adj, node):
        if adj.is_user_function:
            if hasattr(node.func, "attr") and node.func.attr == "tid":
                lineno = adj.lineno + adj.fun_lineno
                line = adj.source_lines[adj.lineno]
                raise WarpCodegenError(
                    "tid() may only be called from a Warp kernel, not a Warp function. "
                    "Instead, obtain the indices from a @wp.kernel and pass them as "
                    f"arguments to the function {adj.fun_name}, {adj.filename}:{lineno}:\n{line}\n"
                )

    def resolve_arg(adj, arg):
        # Always try to start with evaluating the argument since it can help
        # detecting some issues such as global variables being accessed.
        try:
            var = adj.eval(arg)
        except (WarpCodegenError, WarpCodegenKeyError) as e:
            error = e
        else:
            error = None

        # Check if we can resolve the argument as a static expression.
        # If not, return the variable resulting from evaluating the argument.
        expr, _ = adj.resolve_static_expression(arg)
        if expr is None:
            if error is not None:
                raise error

            return var

        if isinstance(expr, (type, Var, warp.context.Function)):
            return expr

        return adj.add_constant(expr)

    def emit_Call(adj, node):
        adj.check_tid_in_func_error(node)

        # try and lookup function in globals by
        # resolving path (e.g.: module.submodule.attr)
        if hasattr(node.func, "warp_func"):
            func = node.func.warp_func
            path = []
        else:
            func, path = adj.resolve_static_expression(node.func)
        if func is None:
            func = adj.eval(node.func)

        if adj.is_static_expression(func):
            # try to evaluate wp.static() expressions
            obj, _ = adj.evaluate_static_expression(node)
            if obj is not None:
                if isinstance(obj, warp.context.Function):
                    # special handling for wp.static() evaluating to a function
                    return obj
                else:
                    out = adj.add_constant(obj)
                    return out

        type_args = {}

        if len(path) > 0 and not isinstance(func, warp.context.Function):
            attr = path[-1]
            caller = func
            func = None

            # try and lookup function name in builtins (e.g.: using `dot` directly without wp prefix)
            if attr in warp.context.builtin_functions:
                func = warp.context.builtin_functions[attr]

            # vector class type e.g.: wp.vec3f constructor
            if func is None and hasattr(caller, "_wp_generic_type_str_"):
                func = warp.context.builtin_functions.get(caller._wp_constructor_)

            # scalar class type e.g.: wp.int8 constructor
            if func is None and hasattr(caller, "__name__") and caller.__name__ in warp.context.builtin_functions:
                func = warp.context.builtin_functions.get(caller.__name__)

            # struct constructor
            if func is None and isinstance(caller, Struct):
                adj.builder.build_struct_recursive(caller)
                if node.args or node.keywords:
                    func = caller.value_constructor
                else:
                    func = caller.default_constructor

            if hasattr(caller, "_wp_type_args_"):
                type_args = caller._wp_type_args_

            if func is None:
                raise WarpCodegenError(
                    f"Could not find function {'.'.join(path)} as a built-in or user-defined function. Note that user functions must be annotated with a @wp.func decorator to be called from a kernel."
                )

        # Check if any argument correspond to an unsupported construct.
        # Tuples are supported in the context of assigning multiple variables
        # at once, but not in place of vectors when calling built-ins like
        # `wp.length((1, 2, 3))`.
        # Therefore, we need to catch this specific case here instead of
        # more generally in `adj.eval()`.
        for arg in node.args:
            if isinstance(arg, ast.Tuple):
                raise WarpCodegenError(
                    "Tuple constructs are not supported in kernels. Use vectors like `wp.vec3()` instead."
                )

        # get expected return count, e.g.: for multi-assignment
        min_outputs = None
        if hasattr(node, "expects"):
            min_outputs = node.expects

        # Evaluate all positional and keywords arguments.
        args = tuple(adj.resolve_arg(x) for x in node.args)
        kwargs = {x.arg: adj.resolve_arg(x.value) for x in node.keywords}

        # add the call and build the callee adjoint if needed (func.adj)
        out = adj.add_call(func, args, kwargs, type_args, min_outputs=min_outputs)

        if warp.config.verify_autograd_array_access:
            # Extract the types and values passed as arguments to the function call.
            arg_types = tuple(strip_reference(get_arg_type(x)) for x in args)
            kwarg_types = {k: strip_reference(get_arg_type(v)) for k, v in kwargs.items()}

            # Resolve the exact function signature among any existing overload.
            resolved_func = adj.resolve_func(func, arg_types, kwarg_types, min_outputs)

            # update arg read/write states according to what happens to that arg in the called function
            if hasattr(resolved_func, "adj"):
                for i, arg in enumerate(args):
                    if resolved_func.adj.args[i].is_write:
                        kernel_name = adj.fun_name
                        filename = adj.filename
                        lineno = adj.lineno + adj.fun_lineno
                        arg.mark_write(kernel_name=kernel_name, filename=filename, lineno=lineno)
                    if resolved_func.adj.args[i].is_read:
                        arg.mark_read()

        return out

    def emit_Index(adj, node):
        # the ast.Index node appears in 3.7 versions
        # when performing array slices, e.g.: x = arr[i]
        # but in version 3.8 and higher it does not appear

        if hasattr(node, "is_adjoint"):
            node.value.is_adjoint = True

        return adj.eval(node.value)

    # returns the object being indexed, and the list of indices
    def eval_subscript(adj, node):
        # We want to coalesce multi-dimensional array indexing into a single operation. This needs to deal with expressions like `a[i][j][x][y]` where `a` is a 2D array of matrices,
        # and essentially rewrite it into `a[i, j][x][y]`. Since the AST observes the indexing right-to-left, and we don't want to evaluate the index expressions prematurely,
        # this requires a first loop to check if this `node` only performs indexing on the array, and a second loop to evaluate and collect index variables.
        root = node
        count = 0
        array = None
        while isinstance(root, ast.Subscript):
            if isinstance(root.slice, ast.Tuple):
                # handles the x[i, j] case (Python 3.8.x upward)
                count += len(root.slice.elts)
            elif isinstance(root.slice, ast.Index) and isinstance(root.slice.value, ast.Tuple):
                # handles the x[i, j] case (Python 3.7.x)
                count += len(root.slice.value.elts)
            else:
                # simple expression, e.g.: x[i]
                count += 1

            if isinstance(root.value, ast.Name):
                symbol = adj.emit_Name(root.value)
                symbol_type = strip_reference(symbol.type)
                if is_array(symbol_type):
                    array = symbol
                    break

            root = root.value

        # If not all indices index into the array, just evaluate the right-most indexing operation.
        if not array or (count > array.type.ndim):
            count = 1

        indices = []
        root = node
        while len(indices) < count:
            if isinstance(root.slice, ast.Tuple):
                ij = [adj.eval(arg) for arg in root.slice.elts]
            elif isinstance(root.slice, ast.Index) and isinstance(root.slice.value, ast.Tuple):
                ij = [adj.eval(arg) for arg in root.slice.value.elts]
            else:
                ij = [adj.eval(root.slice)]

            indices = ij + indices  # prepend

            root = root.value

        target = adj.eval(root)

        return target, indices

    def emit_Subscript(adj, node):
        if hasattr(node.value, "attr") and node.value.attr == "adjoint":
            # handle adjoint of a variable, i.e. wp.adjoint[var]
            node.slice.is_adjoint = True
            var = adj.eval(node.slice)
            var_name = var.label
            var = Var(f"adj_{var_name}", type=var.type, constant=None, prefix=False)
            return var

        target, indices = adj.eval_subscript(node)

        target_type = strip_reference(target.type)
        if is_array(target_type):
            if len(indices) == target_type.ndim:
                # handles array loads (where each dimension has an index specified)
                out = adj.add_builtin_call("address", [target, *indices])

                if warp.config.verify_autograd_array_access:
                    target.mark_read()

            else:
                # handles array views (fewer indices than dimensions)
                out = adj.add_builtin_call("view", [target, *indices])

                if warp.config.verify_autograd_array_access:
                    # store reference to target Var to propagate downstream read/write state back to root arg Var
                    out.parent = target

                    # view arg inherits target Var's read/write states
                    out.is_read = target.is_read
                    out.is_write = target.is_write

        elif is_tile(target_type):
            if len(indices) == len(target_type.shape):
                # handles extracting a single element from a tile
                out = adj.add_builtin_call("tile_extract", [target, *indices])
            elif len(indices) < len(target_type.shape):
                # handles tile views
                out = adj.add_builtin_call("tile_view", [target, indices])
            else:
                raise RuntimeError(
                    f"Incorrect number of indices specified for a tile view/extract, got {len(indices)} indices for a {len(target_type.shape)} dimensional tile."
                )

        else:
            # handles non-array type indexing, e.g: vec3, mat33, etc
            out = adj.add_builtin_call("extract", [target, *indices])

        return out

    def emit_Assign(adj, node):
        if len(node.targets) != 1:
            raise WarpCodegenError("Assigning the same value to multiple variables is not supported")

        lhs = node.targets[0]

        if not isinstance(lhs, ast.Tuple):
            # Check if the rhs corresponds to an unsupported construct.
            # Tuples are supported in the context of assigning multiple variables
            # at once, but not for simple assignments like `x = (1, 2, 3)`.
            # Therefore, we need to catch this specific case here instead of
            # more generally in `adj.eval()`.
            if isinstance(node.value, ast.List):
                raise WarpCodegenError(
                    "List constructs are not supported in kernels. Use vectors like `wp.vec3()` for small collections instead."
                )
            elif isinstance(node.value, ast.Tuple):
                raise WarpCodegenError(
                    "Tuple constructs are not supported in kernels. Use vectors like `wp.vec3()` for small collections instead."
                )

        # handle the case where we are assigning multiple output variables
        if isinstance(lhs, ast.Tuple):
            # record the expected number of outputs on the node
            # we do this so we can decide which function to
            # call based on the number of expected outputs
            if isinstance(node.value, ast.Call):
                node.value.expects = len(lhs.elts)

            # evaluate values
            if isinstance(node.value, ast.Tuple):
                out = [adj.eval(v) for v in node.value.elts]
            else:
                out = adj.eval(node.value)

            names = []
            for v in lhs.elts:
                if isinstance(v, ast.Name):
                    names.append(v.id)
                else:
                    raise WarpCodegenError(
                        "Multiple return functions can only assign to simple variables, e.g.: x, y = func()"
                    )

            if len(names) != len(out):
                raise WarpCodegenError(
                    f"Multiple return functions need to receive all their output values, incorrect number of values to unpack (expected {len(out)}, got {len(names)})"
                )

            for name, rhs in zip(names, out):
                if name in adj.symbols:
                    if not types_equal(rhs.type, adj.symbols[name].type):
                        raise WarpCodegenTypeError(
                            f"Error, assigning to existing symbol {name} ({adj.symbols[name].type}) with different type ({rhs.type})"
                        )

                adj.symbols[name] = rhs

        # handles the case where we are assigning to an array index (e.g.: arr[i] = 2.0)
        elif isinstance(lhs, ast.Subscript):
            rhs = adj.eval(node.value)

            if hasattr(lhs.value, "attr") and lhs.value.attr == "adjoint":
                # handle adjoint of a variable, i.e. wp.adjoint[var]
                lhs.slice.is_adjoint = True
                src_var = adj.eval(lhs.slice)
                var = Var(f"adj_{src_var.label}", type=src_var.type, constant=None, prefix=False)
                adj.add_forward(f"{var.emit()} = {rhs.emit()};")
                return

            target, indices = adj.eval_subscript(lhs)

            target_type = strip_reference(target.type)

            if is_array(target_type):
                adj.add_builtin_call("array_store", [target, *indices, rhs])

                if warp.config.verify_autograd_array_access:
                    kernel_name = adj.fun_name
                    filename = adj.filename
                    lineno = adj.lineno + adj.fun_lineno

                    target.mark_write(kernel_name=kernel_name, filename=filename, lineno=lineno)

            elif is_tile(target_type):
                adj.add_builtin_call("assign", [target, *indices, rhs])

            elif type_is_vector(target_type) or type_is_quaternion(target_type) or type_is_matrix(target_type):
                # recursively unwind AST, stopping at penultimate node
                node = lhs
                while hasattr(node, "value"):
                    if hasattr(node.value, "value"):
                        node = node.value
                    else:
                        break
                # lhs is updating a variable adjoint (i.e. wp.adjoint[var])
                if hasattr(node, "attr") and node.attr == "adjoint":
                    attr = adj.add_builtin_call("index", [target, *indices])
                    adj.add_builtin_call("store", [attr, rhs])
                    return

                # TODO: array vec component case
                if is_reference(target.type):
                    attr = adj.add_builtin_call("indexref", [target, *indices])
                    adj.add_builtin_call("store", [attr, rhs])

                    if warp.config.verbose and not adj.custom_reverse_mode:
                        lineno = adj.lineno + adj.fun_lineno
                        line = adj.source_lines[adj.lineno]
                        node_source = adj.get_node_source(lhs.value)
                        print(
                            f"Warning: mutating {node_source} in function {adj.fun_name} at {adj.filename}:{lineno}: this is a non-differentiable operation.\n{line}\n"
                        )
                else:
                    if warp.config.enable_vector_component_overwrites:
                        out = adj.add_builtin_call("assign_copy", [target, *indices, rhs])

                        # re-point target symbol to out var
                        for id in adj.symbols:
                            if adj.symbols[id] == target:
                                adj.symbols[id] = out
                                break
                    else:
                        adj.add_builtin_call("assign_inplace", [target, *indices, rhs])

            else:
                raise WarpCodegenError(
                    f"Can only subscript assign array, vector, quaternion, and matrix types, got {target_type}"
                )

        elif isinstance(lhs, ast.Name):
            # symbol name
            name = lhs.id

            # evaluate rhs
            rhs = adj.eval(node.value)

            # check type matches if symbol already defined
            if name in adj.symbols:
                if not types_equal(strip_reference(rhs.type), adj.symbols[name].type):
                    raise WarpCodegenTypeError(
                        f"Error, assigning to existing symbol {name} ({adj.symbols[name].type}) with different type ({rhs.type})"
                    )

            # handle simple assignment case (a = b), where we generate a value copy rather than reference
            if isinstance(node.value, ast.Name) or is_reference(rhs.type):
                out = adj.add_builtin_call("copy", [rhs])
            else:
                out = rhs

            # update symbol map (assumes lhs is a Name node)
            adj.symbols[name] = out

        elif isinstance(lhs, ast.Attribute):
            rhs = adj.eval(node.value)
            aggregate = adj.eval(lhs.value)
            aggregate_type = strip_reference(aggregate.type)

            # assigning to a vector or quaternion component
            if type_is_vector(aggregate_type) or type_is_quaternion(aggregate_type):
                index = adj.vector_component_index(lhs.attr, aggregate_type)

                if is_reference(aggregate.type):
                    attr = adj.add_builtin_call("indexref", [aggregate, index])
                    adj.add_builtin_call("store", [attr, rhs])
                else:
                    if warp.config.enable_vector_component_overwrites:
                        out = adj.add_builtin_call("assign_copy", [aggregate, index, rhs])

                        # re-point target symbol to out var
                        for id in adj.symbols:
                            if adj.symbols[id] == aggregate:
                                adj.symbols[id] = out
                                break
                    else:
                        adj.add_builtin_call("assign_inplace", [aggregate, index, rhs])

            else:
                attr = adj.emit_Attribute(lhs)
                if is_reference(attr.type):
                    adj.add_builtin_call("store", [attr, rhs])
                else:
                    adj.add_builtin_call("assign", [attr, rhs])

                if warp.config.verbose and not adj.custom_reverse_mode:
                    lineno = adj.lineno + adj.fun_lineno
                    line = adj.source_lines[adj.lineno]
                    msg = f'Warning: detected mutated struct {attr.label} during function "{adj.fun_name}" at {adj.filename}:{lineno}: this is a non-differentiable operation.\n{line}\n'
                    print(msg)

        else:
            raise WarpCodegenError("Error, unsupported assignment statement.")

    def emit_Return(adj, node):
        if node.value is None:
            var = None
        elif isinstance(node.value, ast.Tuple):
            var = tuple(adj.eval(arg) for arg in node.value.elts)
        else:
            var = (adj.eval(node.value),)

        if adj.return_var is not None:
            old_ctypes = tuple(v.ctype(value_type=True) for v in adj.return_var)
            new_ctypes = tuple(v.ctype(value_type=True) for v in var)
            if old_ctypes != new_ctypes:
                raise WarpCodegenTypeError(
                    f"Error, function returned different types, previous: [{', '.join(old_ctypes)}], new [{', '.join(new_ctypes)}]"
                )

        if var is not None:
            adj.return_var = ()
            for ret in var:
                if is_reference(ret.type):
                    ret_var = adj.add_builtin_call("copy", [ret])
                else:
                    ret_var = ret
                adj.return_var += (ret_var,)

        adj.add_return(adj.return_var)

    def emit_AugAssign(adj, node):
        lhs = node.target

        # replace augmented assignment with assignment statement + binary op (default behaviour)
        def make_new_assign_statement():
            new_node = ast.Assign(targets=[lhs], value=ast.BinOp(lhs, node.op, node.value))
            adj.eval(new_node)

        if isinstance(lhs, ast.Subscript):
            rhs = adj.eval(node.value)

            # wp.adjoint[var] appears in custom grad functions, and does not require
            # special consideration in the AugAssign case
            if hasattr(lhs.value, "attr") and lhs.value.attr == "adjoint":
                make_new_assign_statement()
                return

            target, indices = adj.eval_subscript(lhs)

            target_type = strip_reference(target.type)

            if is_array(target_type):
                # target_types int8, uint8, int16, uint16 are not suitable for atomic array accumulation
                if target_type.dtype in warp.types.non_atomic_types:
                    make_new_assign_statement()
                    return

                # the same holds true for vecs/mats/quats that are composed of these types
                if (
                    type_is_vector(target_type.dtype)
                    or type_is_quaternion(target_type.dtype)
                    or type_is_matrix(target_type.dtype)
                ):
                    dtype = getattr(target_type.dtype, "_wp_scalar_type_", None)
                    if dtype in warp.types.non_atomic_types:
                        make_new_assign_statement()
                        return

                kernel_name = adj.fun_name
                filename = adj.filename
                lineno = adj.lineno + adj.fun_lineno

                if isinstance(node.op, ast.Add):
                    adj.add_builtin_call("atomic_add", [target, *indices, rhs])

                    if warp.config.verify_autograd_array_access:
                        target.mark_write(kernel_name=kernel_name, filename=filename, lineno=lineno)

                elif isinstance(node.op, ast.Sub):
                    adj.add_builtin_call("atomic_sub", [target, *indices, rhs])

                    if warp.config.verify_autograd_array_access:
                        target.mark_write(kernel_name=kernel_name, filename=filename, lineno=lineno)
                else:
                    if warp.config.verbose:
                        print(f"Warning: in-place op {node.op} is not differentiable")
                    make_new_assign_statement()
                    return

            elif type_is_vector(target_type) or type_is_quaternion(target_type) or type_is_matrix(target_type):
                if isinstance(node.op, ast.Add):
                    adj.add_builtin_call("add_inplace", [target, *indices, rhs])
                elif isinstance(node.op, ast.Sub):
                    adj.add_builtin_call("sub_inplace", [target, *indices, rhs])
                else:
                    if warp.config.verbose:
                        print(f"Warning: in-place op {node.op} is not differentiable")
                    make_new_assign_statement()
                    return

            elif is_tile(target.type):
                if isinstance(node.op, ast.Add):
                    adj.add_builtin_call("tile_add_inplace", [target, *indices, rhs])
                elif isinstance(node.op, ast.Sub):
                    adj.add_builtin_call("tile_sub_inplace", [target, *indices, rhs])
                else:
                    if warp.config.verbose:
                        print(f"Warning: in-place op {node.op} is not differentiable")
                    make_new_assign_statement()
                    return

            else:
                raise WarpCodegenError("Can only subscript in-place assign array, vector, quaternion, and matrix types")

        elif isinstance(lhs, ast.Name):
            target = adj.eval(node.target)
            rhs = adj.eval(node.value)

            if is_tile(target.type) and is_tile(rhs.type):
                if isinstance(node.op, ast.Add):
                    adj.add_builtin_call("add_inplace", [target, rhs])
                elif isinstance(node.op, ast.Sub):
                    adj.add_builtin_call("sub_inplace", [target, rhs])
                else:
                    make_new_assign_statement()
                    return
            else:
                make_new_assign_statement()
                return

        # TODO
        elif isinstance(lhs, ast.Attribute):
            make_new_assign_statement()
            return

        else:
            make_new_assign_statement()
            return

    def emit_Tuple(adj, node):
        # LHS for expressions, such as i, j, k = 1, 2, 3
        return tuple(adj.eval(x) for x in node.elts)

    def emit_Pass(adj, node):
        pass

    node_visitors: ClassVar[dict[type[ast.AST], Callable]] = {
        ast.FunctionDef: emit_FunctionDef,
        ast.If: emit_If,
        ast.Compare: emit_Compare,
        ast.BoolOp: emit_BoolOp,
        ast.Name: emit_Name,
        ast.Attribute: emit_Attribute,
        ast.Constant: emit_Constant,
        ast.BinOp: emit_BinOp,
        ast.UnaryOp: emit_UnaryOp,
        ast.While: emit_While,
        ast.For: emit_For,
        ast.Break: emit_Break,
        ast.Continue: emit_Continue,
        ast.Expr: emit_Expr,
        ast.Call: emit_Call,
        ast.Index: emit_Index,  # Deprecated in 3.9
        ast.Subscript: emit_Subscript,
        ast.Assign: emit_Assign,
        ast.Return: emit_Return,
        ast.AugAssign: emit_AugAssign,
        ast.Tuple: emit_Tuple,
        ast.Pass: emit_Pass,
        ast.Assert: emit_Assert,
    }

    def eval(adj, node):
        if hasattr(node, "lineno"):
            adj.set_lineno(node.lineno - 1)

        try:
            emit_node = adj.node_visitors[type(node)]
        except KeyError as e:
            type_name = type(node).__name__
            namespace = "ast." if isinstance(node, ast.AST) else ""
            raise WarpCodegenError(f"Construct `{namespace}{type_name}` not supported in kernels.") from e

        return emit_node(adj, node)

    # helper to evaluate expressions of the form
    # obj1.obj2.obj3.attr in the function's global scope
    def resolve_path(adj, path):
        if len(path) == 0:
            return None

        # if root is overshadowed by local symbols, bail out
        if path[0] in adj.symbols:
            return None

        # look up in closure/global variables
        expr = adj.resolve_external_reference(path[0])

        # Support Warp types in kernels without the module suffix (e.g. v = vec3(0.0,0.2,0.4)):
        if expr is None:
            expr = getattr(warp, path[0], None)

        # look up in builtins
        if expr is None:
            expr = __builtins__.get(path[0])

        if expr is not None:
            for i in range(1, len(path)):
                if hasattr(expr, path[i]):
                    expr = getattr(expr, path[i])

        return expr

    # retrieves a dictionary of all closure and global variables and their values
    # to be used in the evaluation context of wp.static() expressions
    def get_static_evaluation_context(adj):
        closure_vars = dict(
            zip(
                adj.func.__code__.co_freevars,
                [c.cell_contents for c in (adj.func.__closure__ or [])],
            )
        )

        vars_dict = {}
        vars_dict.update(adj.func.__globals__)
        # variables captured in closure have precedence over global vars
        vars_dict.update(closure_vars)

        return vars_dict

    def is_static_expression(adj, func):
        return (
            isinstance(func, types.FunctionType)
            and func.__module__ == "warp.builtins"
            and func.__qualname__ == "static"
        )

    # verify the return type of a wp.static() expression is supported inside a Warp kernel
    def verify_static_return_value(adj, value):
        if value is None:
            raise ValueError("None is returned")
        if warp.types.is_value(value):
            return True
        if warp.types.is_array(value):
            # more useful explanation for the common case of creating a Warp array
            raise ValueError("a Warp array cannot be created inside Warp kernels")
        if isinstance(value, str):
            # we want to support cases such as `print(wp.static("test"))`
            return True
        if isinstance(value, warp.context.Function):
            return True

        def verify_struct(s: StructInstance, attr_path: List[str]):
            for key in s._cls.vars.keys():
                v = getattr(s, key)
                if issubclass(type(v), StructInstance):
                    verify_struct(v, [*attr_path, key])
                else:
                    try:
                        adj.verify_static_return_value(v)
                    except ValueError as e:
                        raise ValueError(
                            f"the returned Warp struct contains a data type that cannot be constructed inside Warp kernels: {e} at {value._cls.key}.{'.'.join(attr_path)}"
                        ) from e

        if issubclass(type(value), StructInstance):
            return verify_struct(value, [])

        raise ValueError(f"value of type {type(value)} cannot be constructed inside Warp kernels")

    # find the source code string of an AST node
    def extract_node_source(adj, node) -> str | None:
        if not hasattr(node, "lineno") or not hasattr(node, "col_offset"):
            return None

        start_line = node.lineno - 1  # line numbers start at 1
        start_col = node.col_offset

        if hasattr(node, "end_lineno") and hasattr(node, "end_col_offset"):
            end_line = node.end_lineno - 1
            end_col = node.end_col_offset
        else:
            # fallback for Python versions before 3.8
            # we have to find the end line and column manually
            end_line = start_line
            end_col = start_col
            parenthesis_count = 1
            for lineno in range(start_line, len(adj.source_lines)):
                if lineno == start_line:
                    c_start = start_col
                else:
                    c_start = 0
                line = adj.source_lines[lineno]
                for i in range(c_start, len(line)):
                    c = line[i]
                    if c == "(":
                        parenthesis_count += 1
                    elif c == ")":
                        parenthesis_count -= 1
                        if parenthesis_count == 0:
                            end_col = i
                            end_line = lineno
                            break
                if parenthesis_count == 0:
                    break

        if start_line == end_line:
            # single-line expression
            return adj.source_lines[start_line][start_col:end_col]
        else:
            # multi-line expression
            lines = []
            # first line (from start_col to the end)
            lines.append(adj.source_lines[start_line][start_col:])
            # middle lines (entire lines)
            lines.extend(adj.source_lines[start_line + 1 : end_line])
            # last line (from the start to end_col)
            lines.append(adj.source_lines[end_line][:end_col])
            return "\n".join(lines).strip()

    # handles a wp.static() expression and returns the resulting object and a string representing the code
    # of the static expression
    def evaluate_static_expression(adj, node) -> Tuple[Any, str]:
        if len(node.args) == 1:
            static_code = adj.extract_node_source(node.args[0])
        elif len(node.keywords) == 1:
            static_code = adj.extract_node_source(node.keywords[0])
        else:
            raise WarpCodegenError("warp.static() requires a single argument or keyword")
        if static_code is None:
            raise WarpCodegenError("Error extracting source code from wp.static() expression")

        # Since this is an expression, we can enforce it to be defined on a single line.
        static_code = static_code.replace("\n", "")

        vars_dict = adj.get_static_evaluation_context()
        # add constant variables to the static call context
        constant_vars = {k: v.constant for k, v in adj.symbols.items() if isinstance(v, Var) and v.constant is not None}
        vars_dict.update(constant_vars)

        # Replace all constant `len()` expressions with their value.
        if "len" in static_code:

            def eval_len(obj):
                if type_is_vector(obj):
                    return obj._length_
                elif type_is_quaternion(obj):
                    return obj._length_
                elif type_is_matrix(obj):
                    return obj._shape_[0]
                elif type_is_transformation(obj):
                    return obj._length_
                elif is_tile(obj):
                    return obj.shape[0]

                return len(obj)

            len_expr_ctx = vars_dict.copy()
            constant_types = {k: v.type for k, v in adj.symbols.items() if isinstance(v, Var) and v.type is not None}
            len_expr_ctx.update(constant_types)
            len_expr_ctx.update({"len": eval_len})

            # We want to replace the expression code in-place,
            # so reparse it to get the correct column info.
            len_value_locs: List[Tuple[int, int, int]] = []
            expr_tree = ast.parse(static_code)
            assert len(expr_tree.body) == 1 and isinstance(expr_tree.body[0], ast.Expr)
            expr_root = expr_tree.body[0].value
            for expr_node in ast.walk(expr_root):
                if (
                    isinstance(expr_node, ast.Call)
                    and getattr(expr_node.func, "id", None) == "len"
                    and len(expr_node.args) == 1
                ):
                    len_expr = static_code[expr_node.col_offset : expr_node.end_col_offset]
                    try:
                        len_value = eval(len_expr, len_expr_ctx)
                    except Exception:
                        pass
                    else:
                        len_value_locs.append((len_value, expr_node.col_offset, expr_node.end_col_offset))

            if len_value_locs:
                new_static_code = ""
                loc = 0
                for value, start, end in len_value_locs:
                    new_static_code += f"{static_code[loc:start]}{value}"
                    loc = end

                new_static_code += static_code[len_value_locs[-1][2] :]
                static_code = new_static_code

        try:
            value = eval(static_code, vars_dict)
            if warp.config.verbose:
                print(f"Evaluated static command: {static_code} = {value}")
        except NameError as e:
            raise WarpCodegenError(
                f"Error evaluating static expression: {e}. Make sure all variables used in the static expression are constant."
            ) from e
        except Exception as e:
            raise WarpCodegenError(
                f"Error evaluating static expression: {e} while evaluating the following code generated from the static expression:\n{static_code}"
            ) from e

        try:
            adj.verify_static_return_value(value)
        except ValueError as e:
            raise WarpCodegenError(
                f"Static expression returns an unsupported value: {e} while evaluating the following code generated from the static expression:\n{static_code}"
            ) from e

        return value, static_code

    # try to replace wp.static() expressions by their evaluated value if the
    # expression can be evaluated
    def replace_static_expressions(adj):
        class StaticExpressionReplacer(ast.NodeTransformer):
            def visit_Call(self, node):
                func, _ = adj.resolve_static_expression(node.func, eval_types=False)
                if adj.is_static_expression(func):
                    try:
                        # the static expression will execute as long as the static expression is valid and
                        # only depends on global or captured variables
                        obj, code = adj.evaluate_static_expression(node)
                        if code is not None:
                            adj.static_expressions[code] = obj
                            if isinstance(obj, warp.context.Function):
                                name_node = ast.Name("__warp_func__")
                                # we add a pointer to the Warp function here so that we can refer to it later at
                                # codegen time (note that the function key itself is not sufficient to uniquely
                                # identify the function, as the function may be redefined between the current time
                                # of wp.static() declaration and the time of codegen during module building)
                                name_node.warp_func = obj
                                return ast.copy_location(name_node, node)
                            else:
                                return ast.copy_location(ast.Constant(value=obj), node)
                    except Exception:
                        # Ignoring failing static expressions should generally not be an issue because only
                        # one of these cases should be possible:
                        #   1) the static expression itself is invalid code, in which case the module cannot be
                        #      built all,
                        #   2) the static expression contains a reference to a local (even if constant) variable
                        #      (and is therefore not executable and raises this exception), in which
                        #      case changing the constant, or the code affecting this constant, would lead to
                        #      a different module hash anyway.
                        pass

                return self.generic_visit(node)

        adj.tree = StaticExpressionReplacer().visit(adj.tree)

    # Evaluates a static expression that does not depend on runtime values
    # if eval_types is True, try resolving the path using evaluated type information as well
    def resolve_static_expression(adj, root_node, eval_types=True):
        attributes = []

        node = root_node
        while isinstance(node, ast.Attribute):
            attributes.append(node.attr)
            node = node.value

        if eval_types and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # support for operators returning modules
            # i.e. operator_name(*operator_args).x.y.z
            operator_args = node.args
            operator_name = node.func.id

            if operator_name == "type":
                if len(operator_args) != 1:
                    raise WarpCodegenError(f"type() operator expects exactly one argument, got {len(operator_args)}")

                # type() operator
                var = adj.eval(operator_args[0])

                if isinstance(var, Var):
                    var_type = strip_reference(var.type)
                    # Allow accessing type attributes, for instance array.dtype
                    while attributes:
                        attr_name = attributes.pop()
                        var_type, prev_type = adj.resolve_type_attribute(var_type, attr_name), var_type

                        if var_type is None:
                            raise WarpCodegenAttributeError(
                                f"{attr_name} is not an attribute of {type_repr(prev_type)}"
                            )

                    return var_type, [str(var_type)]
                else:
                    raise WarpCodegenError(f"Cannot deduce the type of {var}")

        # reverse list since ast presents it in backward order
        path = [*reversed(attributes)]
        if isinstance(node, ast.Name):
            path.insert(0, node.id)

        # Try resolving path from captured context
        captured_obj = adj.resolve_path(path)
        if captured_obj is not None:
            return captured_obj, path

        return None, path

    def resolve_external_reference(adj, name: str):
        try:
            # look up in closure variables
            idx = adj.func.__code__.co_freevars.index(name)
            obj = adj.func.__closure__[idx].cell_contents
        except ValueError:
            # look up in global variables
            obj = adj.func.__globals__.get(name)
        return obj

    # annotate generated code with the original source code line
    def set_lineno(adj, lineno):
        if adj.lineno is None or adj.lineno != lineno:
            line = lineno + adj.fun_lineno
            source = adj.source_lines[lineno].strip().ljust(80 - len(adj.indentation), " ")
            adj.add_forward(f"// {source}       <L {line}>")
            adj.add_reverse(f"// adj: {source}  <L {line}>")
        adj.lineno = lineno

    def get_node_source(adj, node):
        # return the Python code corresponding to the given AST node
        return ast.get_source_segment(adj.source, node)

    def get_references(adj) -> tuple[dict[str, Any], dict[Any, Any], dict[warp.context.Function, Any]]:
        """Traverses ``adj.tree`` and returns referenced constants, types, and user-defined functions."""

        local_variables = set()  # Track local variables appearing on the LHS so we know when variables are shadowed

        constants: dict[str, Any] = {}
        types: dict[Struct | type, Any] = {}
        functions: dict[warp.context.Function, Any] = {}

        for node in ast.walk(adj.tree):
            if isinstance(node, ast.Name) and node.id not in local_variables:
                # look up in closure/global variables
                obj = adj.resolve_external_reference(node.id)
                if warp.types.is_value(obj):
                    constants[node.id] = obj

            elif isinstance(node, ast.Attribute):
                obj, path = adj.resolve_static_expression(node, eval_types=False)
                if warp.types.is_value(obj):
                    constants[".".join(path)] = obj

            elif isinstance(node, ast.Call):
                func, _ = adj.resolve_static_expression(node.func, eval_types=False)
                if isinstance(func, warp.context.Function) and not func.is_builtin():
                    # calling user-defined function
                    functions[func] = None
                elif isinstance(func, Struct):
                    # calling struct constructor
                    types[func] = None
                elif isinstance(func, type) and warp.types.type_is_value(func):
                    # calling value type constructor
                    types[func] = None

            elif isinstance(node, ast.Assign):
                # Add the LHS names to the local_variables so we know any subsequent uses are shadowed
                lhs = node.targets[0]
                if isinstance(lhs, ast.Tuple):
                    for v in lhs.elts:
                        if isinstance(v, ast.Name):
                            local_variables.add(v.id)
                elif isinstance(lhs, ast.Name):
                    local_variables.add(lhs.id)

        return constants, types, functions


# ----------------
# code generation

cpu_module_header = """
#define WP_TILE_BLOCK_DIM {block_dim}
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(task_index, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, task_index, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, task_index, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, task_index, dim)

#define builtin_block_dim() wp::block_dim()

"""

cuda_module_header = """
#define WP_TILE_BLOCK_DIM {block_dim}
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

"""

struct_template = """
struct {name}
{{
{struct_body}

    {defaulted_constructor_def}
    CUDA_CALLABLE {name}({forward_args})
    {forward_initializers}
    {{
    }}

    CUDA_CALLABLE {name}& operator += (const {name}& rhs)
    {{{prefix_add_body}
        return *this;}}

}};

static CUDA_CALLABLE void adj_{name}({reverse_args})
{{
{reverse_body}}}

CUDA_CALLABLE void adj_atomic_add({name}* p, {name} t)
{{
{atomic_add_body}}}


"""

cpu_forward_function_template = """
// {filename}:{lineno}
static {return_type} {name}(
    {forward_args})
{{
{forward_body}}}

"""

cpu_reverse_function_template = """
// {filename}:{lineno}
static void adj_{name}(
    {reverse_args})
{{
{reverse_body}}}

"""

cuda_forward_function_template = """
// {filename}:{lineno}
{line_directive}static CUDA_CALLABLE {return_type} {name}(
    {forward_args})
{{
{forward_body}{line_directive}}}

"""

cuda_reverse_function_template = """
// {filename}:{lineno}
{line_directive}static CUDA_CALLABLE void adj_{name}(
    {reverse_args})
{{
{reverse_body}{line_directive}}}

"""

cuda_kernel_template_forward = """

{line_directive}extern "C" __global__ void {name}_cuda_kernel_forward(
    {forward_args})
{{
{line_directive}    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
{line_directive}         _idx < dim.size;
{line_directive}         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {{
        // reset shared memory allocator
{line_directive}        wp::tile_alloc_shared(0, true);

{forward_body}{line_directive}    }}
{line_directive}}}

"""

cuda_kernel_template_backward = """

{line_directive}extern "C" __global__ void {name}_cuda_kernel_backward(
    {reverse_args})
{{
{line_directive}    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
{line_directive}         _idx < dim.size;
{line_directive}         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {{
        // reset shared memory allocator
{line_directive}        wp::tile_alloc_shared(0, true);

{reverse_body}{line_directive}    }}
{line_directive}}}

"""

cpu_kernel_template_forward = """

void {name}_cpu_kernel_forward(
    {forward_args})
{{
{forward_body}}}

"""

cpu_kernel_template_backward = """

void {name}_cpu_kernel_backward(
    {reverse_args})
{{
{reverse_body}}}

"""

cpu_module_template_forward = """

extern "C" {{

// Python CPU entry points
WP_API void {name}_cpu_forward(
    {forward_args})
{{
for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {{
        // init shared memory allocator
        wp::tile_alloc_shared(0, true);

        {name}_cpu_kernel_forward(
            {forward_params});

        // check shared memory allocator
        wp::tile_alloc_shared(0, false, true);

    }}
}}

}} // extern C

"""

cpu_module_template_backward = """

extern "C" {{

WP_API void {name}_cpu_backward(
    {reverse_args})
{{
    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {{
        // initialize shared memory allocator
        wp::tile_alloc_shared(0, true);

        {name}_cpu_kernel_backward(
            {reverse_params});

        // check shared memory allocator
        wp::tile_alloc_shared(0, false, true);
    }}
}}

}} // extern C

"""


# converts a constant Python value to equivalent C-repr
def constant_str(value):
    value_type = type(value)

    if value_type == bool or value_type == builtins.bool:
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

            def scalar_value(x):
                return x

        # list of scalar initializer values
        initlist = []
        for i in range(value._length_):
            x = ctypes.Array.__getitem__(value, i)
            initlist.append(str(scalar_value(x)).lower())

        if value._wp_scalar_type_ is bool:
            dtypestr = f"wp::initializer_array<{value._length_},{value._wp_scalar_type_.__name__}>"
        else:
            dtypestr = f"wp::initializer_array<{value._length_},wp::{value._wp_scalar_type_.__name__}>"

        # construct value from initializer array, e.g. wp::initializer_array<4,wp::float32>{1.0, 2.0, 3.0, 4.0}
        return f"{dtypestr}{{{', '.join(initlist)}}}"

    elif value_type in warp.types.scalar_types:
        # make sure we emit the value of objects, e.g. uint32
        return str(value.value)

    elif issubclass(value_type, warp.codegen.StructInstance):
        # constant struct instance
        arg_strs = []
        for key, var in value._cls.vars.items():
            attr = getattr(value, key)
            arg_strs.append(f"{Var.type_to_ctype(var.type)}({constant_str(attr)})")
        arg_str = ", ".join(arg_strs)
        return f"{value.native_name}({arg_str})"

    elif value == math.inf:
        return "INFINITY"

    elif math.isnan(value):
        return "NAN"

    else:
        # otherwise just convert constant to string
        return str(value)


def indent(args, stops=1):
    sep = ",\n"
    for _i in range(stops):
        sep += "    "

    # return sep + args.replace(", ", "," + sep)
    return sep.join(args)


# generates a C function name based on the python function name
def make_full_qualified_name(func: Union[str, Callable]) -> str:
    if not isinstance(func, str):
        func = func.__qualname__
    return re.sub("[^0-9a-zA-Z_]+", "", func.replace(".", "__"))


def codegen_struct(struct, device="cpu", indent_size=4):
    name = struct.native_name

    body = []
    indent_block = " " * indent_size

    if len(struct.vars) > 0:
        for label, var in struct.vars.items():
            body.append(var.ctype() + " " + label + ";\n")
    else:
        # for empty structs, emit the dummy attribute to avoid any compiler-specific alignment issues
        body.append("char _dummy_;\n")

    forward_args = []
    reverse_args = []

    forward_initializers = []
    reverse_body = []
    atomic_add_body = []
    prefix_add_body = []

    # forward args
    for label, var in struct.vars.items():
        var_ctype = var.ctype()
        default_arg_def = " = {}" if forward_args else ""
        forward_args.append(f"{var_ctype} const& {label}{default_arg_def}")
        reverse_args.append(f"{var_ctype} const&")

        namespace = "wp::" if var_ctype.startswith("wp::") or var_ctype == "bool" else ""
        atomic_add_body.append(f"{indent_block}{namespace}adj_atomic_add(&p->{label}, t.{label});\n")

        prefix = f"{indent_block}," if forward_initializers else ":"
        forward_initializers.append(f"{indent_block}{prefix} {label}{{{label}}}\n")

    # prefix-add operator
    for label, var in struct.vars.items():
        if not is_array(var.type):
            prefix_add_body.append(f"{indent_block}{label} += rhs.{label};\n")

    # reverse args
    for label, var in struct.vars.items():
        reverse_args.append(var.ctype() + " & adj_" + label)
        if is_array(var.type):
            reverse_body.append(f"{indent_block}adj_{label} = adj_ret.{label};\n")
        else:
            reverse_body.append(f"{indent_block}adj_{label} += adj_ret.{label};\n")

    reverse_args.append(name + " & adj_ret")

    # explicitly defaulted default constructor if no default constructor has been defined
    defaulted_constructor_def = f"{name}() = default;" if forward_args else ""

    return struct_template.format(
        name=name,
        struct_body="".join([indent_block + l for l in body]),
        forward_args=indent(forward_args),
        forward_initializers="".join(forward_initializers),
        reverse_args=indent(reverse_args),
        reverse_body="".join(reverse_body),
        prefix_add_body="".join(prefix_add_body),
        atomic_add_body="".join(atomic_add_body),
        defaulted_constructor_def=defaulted_constructor_def,
    )


def codegen_func_forward(adj, func_type="kernel", device="cpu"):
    if device == "cpu":
        indent = 4
    elif device == "cuda":
        if func_type == "kernel":
            indent = 8
        else:
            indent = 4
    else:
        raise ValueError(f"Device {device} not supported for codegen")

    indent_block = " " * indent

    # primal vars
    lines = []
    lines += ["//---------\n"]
    lines += ["// primal vars\n"]

    for var in adj.variables:
        if is_tile(var.type):
            lines += [f"{var.ctype()} {var.emit()} = {var.type.cinit(requires_grad=False)};\n"]
        elif var.constant is None:
            lines += [f"{var.ctype()} {var.emit()};\n"]
        else:
            lines += [f"const {var.ctype()} {var.emit()} = {constant_str(var.constant)};\n"]

        if line_directive := adj.get_line_directive(lines[-1], var.relative_lineno):
            lines.insert(-1, f"{line_directive}\n")

    # forward pass
    lines += ["//---------\n"]
    lines += ["// forward\n"]

    for f in adj.blocks[0].body_forward:
        if func_type == "kernel" and device == "cuda" and f.lstrip().startswith("return;"):
            # Use of grid-stride loops in CUDA kernels requires that we convert return; to continue;
            lines += [f.replace("return;", "continue;") + "\n"]
        else:
            lines += [f + "\n"]

    return "".join(l.lstrip() if l.lstrip().startswith("#line") else indent_block + l for l in lines)


def codegen_func_reverse(adj, func_type="kernel", device="cpu"):
    if device == "cpu":
        indent = 4
    elif device == "cuda":
        if func_type == "kernel":
            indent = 8
        else:
            indent = 4
    else:
        raise ValueError(f"Device {device} not supported for codegen")

    indent_block = " " * indent

    lines = []

    # primal vars
    lines += ["//---------\n"]
    lines += ["// primal vars\n"]

    for var in adj.variables:
        if is_tile(var.type):
            lines += [f"{var.ctype()} {var.emit()} = {var.type.cinit(requires_grad=True)};\n"]
        elif var.constant is None:
            lines += [f"{var.ctype()} {var.emit()};\n"]
        else:
            lines += [f"const {var.ctype()} {var.emit()} = {constant_str(var.constant)};\n"]

        if line_directive := adj.get_line_directive(lines[-1], var.relative_lineno):
            lines.insert(-1, f"{line_directive}\n")

    # dual vars
    lines += ["//---------\n"]
    lines += ["// dual vars\n"]

    for var in adj.variables:
        name = var.emit_adj()
        ctype = var.ctype(value_type=True)

        if is_tile(var.type):
            if var.type.storage == "register":
                lines += [
                    f"{var.type.ctype()} {name}(0.0);\n"
                ]  # reverse mode tiles alias the forward vars since shared tiles store both primal/dual vars together
            elif var.type.storage == "shared":
                lines += [
                    f"{var.type.ctype()}& {name} = {var.emit()};\n"
                ]  # reverse mode tiles alias the forward vars since shared tiles store both primal/dual vars together
        else:
            lines += [f"{ctype} {name} = {{}};\n"]

        if line_directive := adj.get_line_directive(lines[-1], var.relative_lineno):
            lines.insert(-1, f"{line_directive}\n")

    # forward pass
    lines += ["//---------\n"]
    lines += ["// forward\n"]

    for f in adj.blocks[0].body_replay:
        lines += [f + "\n"]

    # reverse pass
    lines += ["//---------\n"]
    lines += ["// reverse\n"]

    for l in reversed(adj.blocks[0].body_reverse):
        lines += [l + "\n"]

    # In grid-stride kernels the reverse body is in a for loop
    if device == "cuda" and func_type == "kernel":
        lines += ["continue;\n"]
    else:
        lines += ["return;\n"]

    return "".join(l.lstrip() if l.lstrip().startswith("#line") else indent_block + l for l in lines)


def codegen_func(adj, c_func_name: str, device="cpu", options=None):
    if options is None:
        options = {}

    if adj.return_var is not None and "return" in adj.arg_types:
        if get_origin(adj.arg_types["return"]) is tuple:
            if len(get_args(adj.arg_types["return"])) != len(adj.return_var):
                raise WarpCodegenError(
                    f"The function `{adj.fun_name}` has its return type "
                    f"annotated as a tuple of {len(get_args(adj.arg_types['return']))} elements "
                    f"but the code returns {len(adj.return_var)} values."
                )
            elif not types_equal(adj.arg_types["return"], tuple(x.type for x in adj.return_var)):
                raise WarpCodegenError(
                    f"The function `{adj.fun_name}` has its return type "
                    f"annotated as `{warp.context.type_str(adj.arg_types['return'])}` "
                    f"but the code returns a tuple with types `({', '.join(warp.context.type_str(x.type) for x in adj.return_var)})`."
                )
        elif len(adj.return_var) > 1 and get_origin(adj.arg_types["return"]) is not tuple:
            raise WarpCodegenError(
                f"The function `{adj.fun_name}` has its return type "
                f"annotated as `{warp.context.type_str(adj.arg_types['return'])}` "
                f"but the code returns {len(adj.return_var)} values."
            )
        elif not types_equal(adj.arg_types["return"], adj.return_var[0].type):
            raise WarpCodegenError(
                f"The function `{adj.fun_name}` has its return type "
                f"annotated as `{warp.context.type_str(adj.arg_types['return'])}` "
                f"but the code returns a value of type `{warp.context.type_str(adj.return_var[0].type)}`."
            )

    # Build line directive for function definition (subtract 1 to account for 1-indexing of AST line numbers)
    # This is used as a catch-all C-to-Python source line mapping for any code that does not have
    # a direct mapping to a Python source line.
    func_line_directive = ""
    if line_directive := adj.get_line_directive("", adj.fun_def_lineno - 1):
        func_line_directive = f"{line_directive}\n"

    # forward header
    if adj.return_var is not None and len(adj.return_var) == 1:
        return_type = adj.return_var[0].ctype()
    else:
        return_type = "void"

    has_multiple_outputs = adj.return_var is not None and len(adj.return_var) != 1

    forward_args = []
    reverse_args = []

    # forward args
    for i, arg in enumerate(adj.args):
        s = f"{arg.ctype()} {arg.emit()}"
        forward_args.append(s)
        if not adj.custom_reverse_mode or i < adj.custom_reverse_num_input_args:
            reverse_args.append(s)
    if has_multiple_outputs:
        for i, arg in enumerate(adj.return_var):
            forward_args.append(arg.ctype() + " & ret_" + str(i))
            reverse_args.append(arg.ctype() + " & ret_" + str(i))

    # reverse args
    for i, arg in enumerate(adj.args):
        if adj.custom_reverse_mode and i >= adj.custom_reverse_num_input_args:
            break
        # indexed array gradients are regular arrays
        if isinstance(arg.type, indexedarray):
            _arg = Var(arg.label, array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(_arg.ctype() + " & adj_" + arg.label)
        else:
            reverse_args.append(arg.ctype() + " & adj_" + arg.label)
    if has_multiple_outputs:
        for i, arg in enumerate(adj.return_var):
            reverse_args.append(arg.ctype() + " & adj_ret_" + str(i))
    elif return_type != "void":
        reverse_args.append(return_type + " & adj_ret")
    # custom output reverse args (user-declared)
    if adj.custom_reverse_mode:
        for arg in adj.args[adj.custom_reverse_num_input_args :]:
            reverse_args.append(f"{arg.ctype()} & {arg.emit()}")

    if device == "cpu":
        forward_template = cpu_forward_function_template
        reverse_template = cpu_reverse_function_template
    elif device == "cuda":
        forward_template = cuda_forward_function_template
        reverse_template = cuda_reverse_function_template
    else:
        raise ValueError(f"Device {device} is not supported")

    # codegen body
    forward_body = codegen_func_forward(adj, func_type="function", device=device)

    s = ""
    if not adj.skip_forward_codegen:
        s += forward_template.format(
            name=c_func_name,
            return_type=return_type,
            forward_args=indent(forward_args),
            forward_body=forward_body,
            filename=adj.filename,
            lineno=adj.fun_lineno,
            line_directive=func_line_directive,
        )

    if not adj.skip_reverse_codegen:
        if adj.custom_reverse_mode:
            reverse_body = "\t// user-defined adjoint code\n" + forward_body
        else:
            if options.get("enable_backward", True):
                reverse_body = codegen_func_reverse(adj, func_type="function", device=device)
            else:
                reverse_body = '\t// reverse mode disabled (module option "enable_backward" is False)\n'
        s += reverse_template.format(
            name=c_func_name,
            return_type=return_type,
            reverse_args=indent(reverse_args),
            forward_body=forward_body,
            reverse_body=reverse_body,
            filename=adj.filename,
            lineno=adj.fun_lineno,
            line_directive=func_line_directive,
        )

    return s


def codegen_snippet(adj, name, snippet, adj_snippet, replay_snippet):
    if adj.return_var is not None and len(adj.return_var) == 1:
        return_type = adj.return_var[0].ctype()
    else:
        return_type = "void"

    forward_args = []
    reverse_args = []

    # forward args
    for _i, arg in enumerate(adj.args):
        s = f"{arg.ctype()} {arg.emit().replace('var_', '')}"
        forward_args.append(s)
        reverse_args.append(s)

    # reverse args
    for _i, arg in enumerate(adj.args):
        if isinstance(arg.type, indexedarray):
            _arg = Var(arg.label, array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(_arg.ctype() + " & adj_" + arg.label)
        else:
            reverse_args.append(arg.ctype() + " & adj_" + arg.label)
    if return_type != "void":
        reverse_args.append(return_type + " & adj_ret")

    forward_template = cuda_forward_function_template
    replay_template = cuda_forward_function_template
    reverse_template = cuda_reverse_function_template

    s = ""
    s += forward_template.format(
        name=name,
        return_type=return_type,
        forward_args=indent(forward_args),
        forward_body=snippet,
        filename=adj.filename,
        lineno=adj.fun_lineno,
        line_directive="",
    )

    if replay_snippet is not None:
        s += replay_template.format(
            name="replay_" + name,
            return_type=return_type,
            forward_args=indent(forward_args),
            forward_body=replay_snippet,
            filename=adj.filename,
            lineno=adj.fun_lineno,
            line_directive="",
        )

    if adj_snippet:
        reverse_body = adj_snippet
    else:
        reverse_body = ""

    s += reverse_template.format(
        name=name,
        return_type=return_type,
        reverse_args=indent(reverse_args),
        forward_body=snippet,
        reverse_body=reverse_body,
        filename=adj.filename,
        lineno=adj.fun_lineno,
        line_directive="",
    )

    return s


def codegen_kernel(kernel, device, options):
    # Update the module's options with the ones defined on the kernel, if any.
    options = dict(options)
    options.update(kernel.options)

    adj = kernel.adj

    # Build line directive for function definition (subtract 1 to account for 1-indexing of AST line numbers)
    # This is used as a catch-all C-to-Python source line mapping for any code that does not have
    # a direct mapping to a Python source line.
    func_line_directive = ""
    if line_directive := adj.get_line_directive("", adj.fun_def_lineno - 1):
        func_line_directive = f"{line_directive}\n"

    if device == "cpu":
        template_forward = cpu_kernel_template_forward
        template_backward = cpu_kernel_template_backward
    elif device == "cuda":
        template_forward = cuda_kernel_template_forward
        template_backward = cuda_kernel_template_backward
    else:
        raise ValueError(f"Device {device} is not supported")

    template = ""
    template_fmt_args = {
        "name": kernel.get_mangled_name(),
    }

    # build forward signature
    forward_args = ["wp::launch_bounds_t dim"]
    if device == "cpu":
        forward_args.append("size_t task_index")

    for arg in adj.args:
        forward_args.append(arg.ctype() + " var_" + arg.label)

    forward_body = codegen_func_forward(adj, func_type="kernel", device=device)
    template_fmt_args.update(
        {
            "forward_args": indent(forward_args),
            "forward_body": forward_body,
            "line_directive": func_line_directive,
        }
    )
    template += template_forward

    if options["enable_backward"]:
        # build reverse signature
        reverse_args = ["wp::launch_bounds_t dim"]
        if device == "cpu":
            reverse_args.append("size_t task_index")

        for arg in adj.args:
            reverse_args.append(arg.ctype() + " var_" + arg.label)

        for arg in adj.args:
            # indexed array gradients are regular arrays
            if isinstance(arg.type, indexedarray):
                _arg = Var(arg.label, array(dtype=arg.type.dtype, ndim=arg.type.ndim))
                reverse_args.append(_arg.ctype() + " adj_" + arg.label)
            else:
                reverse_args.append(arg.ctype() + " adj_" + arg.label)

        reverse_body = codegen_func_reverse(adj, func_type="kernel", device=device)
        template_fmt_args.update(
            {
                "reverse_args": indent(reverse_args),
                "reverse_body": reverse_body,
            }
        )
        template += template_backward

    s = template.format(**template_fmt_args)
    return s


def codegen_module(kernel, device, options):
    if device != "cpu":
        return ""

    # Update the module's options with the ones defined on the kernel, if any.
    options = dict(options)
    options.update(kernel.options)

    adj = kernel.adj

    template = ""
    template_fmt_args = {
        "name": kernel.get_mangled_name(),
    }

    # build forward signature
    forward_args = ["wp::launch_bounds_t dim"]
    forward_params = ["dim", "task_index"]

    for arg in adj.args:
        if hasattr(arg.type, "_wp_generic_type_str_"):
            # vectors and matrices are passed from Python by pointer
            forward_args.append(f"const {arg.ctype()}* var_" + arg.label)
            forward_params.append(f"*var_{arg.label}")
        else:
            forward_args.append(f"{arg.ctype()} var_{arg.label}")
            forward_params.append("var_" + arg.label)

    template_fmt_args.update(
        {
            "forward_args": indent(forward_args),
            "forward_params": indent(forward_params, 3),
        }
    )
    template += cpu_module_template_forward

    if options["enable_backward"]:
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

        template_fmt_args.update(
            {
                "reverse_args": indent(reverse_args),
                "reverse_params": indent(reverse_params, 3),
            }
        )
        template += cpu_module_template_backward

    s = template.format(**template_fmt_args)
    return s
