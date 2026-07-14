# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import builtins
import contextlib
import ctypes
import enum
import functools
import hashlib
import inspect
import itertools
import linecache
import math
import re
import textwrap
import threading
import types
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import copy as shallowcopy
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, NamedTuple, get_args, get_origin

import warp.config
from warp._src.deterministic import DeterministicCodegen
from warp._src.logger import log_debug, log_warning
from warp._src.types import *

# used as a globally accessible copy
# of current compile options (block_dim) etc
options = {}


def _escape_line_directive_filename(filename: str) -> str:
    """Return ``filename`` escaped for the quoted filename field of a C/CUDA ``#line`` directive."""

    escaped = []
    for c in filename.replace("\\", "/"):
        if c == '"':
            escaped.append('\\"')
        elif c == "\n":
            escaped.append("\\n")
        elif c == "\r":
            escaped.append("\\r")
        elif c == "\t":
            escaped.append("\\t")
        elif ord(c) < 32 or ord(c) == 127:
            # Use fixed-width octal so following filename characters cannot be consumed by the escape.
            escaped.append(f"\\{ord(c):03o}")
        else:
            escaped.append(c)

    return "".join(escaped)


def get_node_name_safe(node):
    """Safely get a string representation of an AST node for error messages.

    This handles different AST node types (Name, Subscript, etc.) without
    raising AttributeError when accessing attributes that may not exist.
    """
    if hasattr(node, "id"):
        return node.id
    elif hasattr(node, "value") and hasattr(node, "slice"):
        # Subscript node like inputs[tid]
        base_name = get_node_name_safe(node.value)
        return f"{base_name}[...]"
    else:
        return f"<{type(node).__name__}>"


class WarpCodegenError(RuntimeError):
    """General error during Warp kernel code generation."""

    def __init__(self, message):
        super().__init__(message)


class WarpCodegenTypeError(TypeError):
    """Type error during Warp kernel code generation."""

    def __init__(self, message):
        super().__init__(message)


class WarpCodegenAttributeError(AttributeError):
    """Attribute error during Warp kernel code generation."""

    def __init__(self, message):
        super().__init__(message)


class WarpCodegenIndexError(IndexError):
    """Index error during Warp kernel code generation."""

    def __init__(self, message):
        super().__init__(message)


class WarpCodegenKeyError(KeyError):
    """Key error during Warp kernel code generation."""

    def __init__(self, message):
        super().__init__(message)


class WarpCodegenValueError(ValueError):
    """Value error during Warp kernel code generation."""

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


def values_check_equal(a, b):
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        if len(a) != len(b):
            return False

        return all(x == y for x, y in zip(a, b, strict=True))

    return a == b


def get_closure_vars(func: Callable) -> dict[str, Any]:
    """Return a dict of the function's closure variables, skipping empty cells."""
    result = {}
    closure = func.__closure__
    if closure is not None:
        for name, cell in zip(func.__code__.co_freevars, closure, strict=True):
            try:
                result[name] = cell.cell_contents
            except ValueError:
                pass
    return result


def resolve_closure_or_global(func: Callable, name: str):
    """Look up *name* in the function's closure variables, falling back to globals."""
    closure_vars = get_closure_vars(func)
    if name in closure_vars:
        return closure_vars[name]
    return func.__globals__.get(name)


def get_annotations(obj: Any) -> Mapping[str, Any]:
    """Same as `inspect.get_annotations()` but always returning un-stringized annotations."""
    return inspect.get_annotations(obj, eval_str=True)


def get_full_arg_spec(func: Callable) -> inspect.FullArgSpec:
    """Same as `inspect.getfullargspec()` but always returning un-stringized annotations."""
    spec = inspect.getfullargspec(func)
    closure_vars = get_closure_vars(func)
    return spec._replace(annotations=inspect.get_annotations(func, eval_str=True, locals=closure_vars))


def struct_instance_repr_recursive(inst: StructInstance, depth: int, use_repr: bool) -> str:
    indent = "\t"

    # handle empty structs
    if len(inst._cls.vars) == 0:
        return f"{inst._cls.key}()"

    lines = []
    lines.append(f"{inst._cls.key}(")

    for field_name, _ in inst._cls.ctype._fields_:
        field_value = getattr(inst, field_name, None)

        if is_struct(field_value):
            field_value = struct_instance_repr_recursive(field_value, depth + 1, use_repr)

        if use_repr:
            lines.append(f"{indent * (depth + 1)}{field_name}={field_value!r},")
        else:
            lines.append(f"{indent * (depth + 1)}{field_name}={field_value!s},")

    lines.append(f"{indent * depth})")
    return "\n".join(lines)


class StructInstance:
    def __init__(self, ctype):
        # maintain a c-types object for the top-level instance the struct
        super().__setattr__("_ctype", ctype)

        # create Python attributes for each of the struct's variables
        for k, cst in type(self)._constructors:
            self.__dict__[k] = cst(ctype)

    def __setattr__(self, name, value):
        try:
            self._setters[name](self, value)
        except KeyError as err:
            raise RuntimeError(f"Trying to set Warp struct attribute that does not exist {name}") from err

    def __ctype__(self):
        return self._ctype

    def __repr__(self):
        return struct_instance_repr_recursive(self, 0, use_repr=True)

    def __str__(self):
        return struct_instance_repr_recursive(self, 0, use_repr=False)

    def assign(self, value):
        """Assigns the values of another struct instance to this one."""
        if not is_struct(value):
            raise RuntimeError(
                f"Trying to assign a non-structure value to a struct attribute with type: {self._cls.key}"
            )

        if self._cls.key is not value._cls.key:
            raise RuntimeError(
                f"Trying to assign a structure of type {value._cls.key} to an attribute of {self._cls.key}"
            )

        # update all nested ctype vars by deep copy
        for n in self._cls.vars:
            setattr(self, n, getattr(value, n))

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
            if matches_array_class(var.type, array):
                # array_t
                setattr(dst, name, value.to(device))
            elif matches_array_class(var.type, indexedarray):
                # indexedarray_t
                # `.to` returns an array if on different device, force to identity indexedarray
                cloned = value.to(device)
                setattr(dst, name, cloned if isinstance(cloned, indexedarray) else indexedarray(cloned))
            elif isinstance(var.type, Struct):
                # nested struct
                new_struct = var.type()
                setattr(dst, name, new_struct)
                # The call to `setattr()` just above makes a copy of `new_struct`
                # so we need to reference that new instance of the struct.
                new_struct = getattr(dst, name)
                stack.extend((value, new_struct, k, v) for k, v in var.type.vars.items())
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

            if matches_array_class(var.type, array):
                # array_t
                npvalue.append(value.numpy_value())
            elif matches_array_class(var.type, indexedarray):
                # indexedarray_t
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
                elif var.type == warp.bfloat16:
                    npvalue.append(bfloat16_bits_to_float(value))
                else:
                    npvalue.append(value)

        return tuple(npvalue)


def _is_tid_call(node) -> bool:
    """Return True if ``node`` is an AST call to ``wp.tid()``."""
    return isinstance(node, ast.Call) and hasattr(node.func, "attr") and node.func.attr == "tid"


def iter_ast_nodes_of_types(root: ast.AST, *types: type):
    """Like ``(n for n in ast.walk(root) if type(n) in types)`` but faster.

    Inlines ``ast.walk``'s field iteration over a ``deque``, preserving its
    breadth-first order, so it is a drop-in even where order matters. Exact-type
    match; AST node classes are never subclassed in practice.
    """
    todo = deque((root,))
    while todo:
        node = todo.popleft()
        if type(node) in types:
            yield node
        for field in node._fields:
            value = getattr(node, field, None)
            if type(value) is list:
                for child in value:
                    if isinstance(child, ast.AST):
                        todo.append(child)
            elif isinstance(value, ast.AST):
                todo.append(value)


def _is_texture_type(var_type: type) -> bool:
    """Check if var_type is a Texture subclass (Texture2D, Texture3D, etc.)."""
    from warp._src.texture import Texture  # noqa: PLC0415

    try:
        return issubclass(var_type, Texture)
    except TypeError:
        return False


def _make_struct_field_constructor(field: str, var_type: type):
    if isinstance(var_type, Struct):
        return lambda ctype: var_type.instance_type(ctype=getattr(ctype, field))
    elif matches_array_class(var_type, warp._src.types.array):
        return lambda ctype: None
    elif matches_array_class(var_type, warp._src.types.indexedarray):
        return lambda ctype: None
    elif _is_texture_type(var_type):
        return lambda ctype: None
    elif issubclass(var_type, ctypes.Array):
        # for vector/matrices, the Python attribute aliases the ctype one
        return lambda ctype: getattr(ctype, field)
    else:
        return lambda ctype: var_type()


def _make_struct_field_setter(cls, field: str, var_type: type):
    def set_array_value(inst, value):
        if value is None:
            # create array with null pointer
            setattr(inst._ctype, field, array_t())
        else:
            # wp.array
            if not isinstance(value, array):
                raise TypeError(f"Struct field '{field}' expects a Warp array, got {type(value).__name__}")
            if not types_equal(value.dtype, var_type.dtype):
                raise TypeError(
                    f"Struct field '{field}' expects dtype {type_repr(var_type.dtype)}, got {type_repr(value.dtype)}"
                )
            setattr(inst._ctype, field, value.__ctype__())

        # Keep gradient buffers alive while the struct's native array
        # descriptor may reference them. Clear any previous keepalive when
        # this field no longer points at a grad-tracked array.
        grad_attr = "_" + field + "_grad"
        if value is not None and value.requires_grad:
            cls.__setattr__(inst, grad_attr, value.grad)
        else:
            # clear any previous keepalive
            cls.__setattr__(inst, grad_attr, None)

        cls.__setattr__(inst, field, value)

    def set_indexedarray_value(inst, value):
        if value is None:
            setattr(inst._ctype, field, var_type.__ctype__())
        else:
            if not isinstance(value, indexedarray):
                raise TypeError(f"Struct field '{field}' expects a Warp indexed array, got {type(value).__name__}")
            if not types_equal(value.dtype, var_type.dtype):
                raise TypeError(
                    f"Struct field '{field}' expects dtype {type_repr(var_type.dtype)}, got {type_repr(value.dtype)}"
                )
            setattr(inst._ctype, field, value.__ctype__())

        # workaround to prevent gradient buffers being garbage collected
        # (indexedarray_t embeds an array_t)
        grad_attr = "_" + field + "_grad"
        if value is not None and value.data is not None and value.data.requires_grad:
            cls.__setattr__(inst, grad_attr, value.data.grad)
        else:
            # clear any previous keepalive
            cls.__setattr__(inst, grad_attr, None)

        cls.__setattr__(inst, field, value)

    def set_struct_value(inst, value):
        getattr(inst, field).assign(value)

    def set_vector_value(inst, value):
        # vector/matrix type, e.g. vec3
        if value is None:
            setattr(inst._ctype, field, var_type())
        elif type(value) is var_type:
            setattr(inst._ctype, field, value)
        else:
            if is_scalar(value):
                log_warning(
                    f"Implicit conversion from a scalar type to the composite type "
                    f"`{type_repr(var_type)}` for struct field '{field}' is deprecated. "
                    f"Use an explicit conversion, e.g.: `{type_repr(var_type)}(...)`.",
                    category=DeprecationWarning,
                    stacklevel=3,
                )
            # conversion from list/tuple, ndarray, etc.
            setattr(inst._ctype, field, var_type(value))

        # no need to update the Python attribute,
        # it's already aliasing the ctype one

    def set_primitive_value(inst, value):
        # primitive type
        if value is None:
            # zero initialize
            setattr(inst._ctype, field, var_type._type_())
            cls.__setattr__(inst, field, var_type())
        else:
            is_warp_scalar = hasattr(value, "_type_")
            if is_warp_scalar:
                # assigning warp type value (e.g.: wp.float32)
                value = value.value
            # float16/bfloat16 needs conversion to uint16 bits
            if var_type == warp.float16:
                setattr(inst._ctype, field, float_to_half_bits(value))
            elif var_type == warp.bfloat16:
                setattr(inst._ctype, field, float_to_bfloat16_bits(value))
            else:
                setattr(inst._ctype, field, value)

            # Re-wrap in the Warp scalar type so the Python attribute preserves
            # the declared type (e.g. wp.uint8) instead of decaying to plain
            # int/float, but only when the caller passed a Warp scalar.
            cls.__setattr__(inst, field, var_type(value) if is_warp_scalar else value)

    def set_texture_value(inst, value):
        # Texture2D, Texture3D, etc.
        if value is None:
            # create texture with null/default handle
            setattr(inst._ctype, field, var_type._wp_ctype_())
        else:
            # texture instance
            setattr(inst._ctype, field, value.__ctype__())

        cls.__setattr__(inst, field, value)

    if matches_array_class(var_type, array):
        return set_array_value
    elif matches_array_class(var_type, indexedarray):
        return set_indexedarray_value
    elif isinstance(var_type, Struct):
        return set_struct_value
    elif _is_texture_type(var_type):
        return set_texture_value
    elif issubclass(var_type, ctypes.Array):
        return set_vector_value
    else:
        return set_primitive_value


class Struct:
    hash: bytes

    def __init__(self, key: str, cls: type, module: warp._src.context.Module):
        self.key = key
        self.cls = cls
        self.module = module
        self.vars: dict[str, Var] = {}

        if isinstance(self.cls, Sequence):
            raise RuntimeError("Warp structs must be defined as base classes")

        annotations = get_annotations(self.cls)
        for label, type_ in annotations.items():
            self.vars[label] = Var(label, type_)

        fields = []
        for label, var in self.vars.items():
            if matches_array_class(var.type, array):
                fields.append((label, array_t))
            elif matches_array_class(var.type, indexedarray):
                fields.append((label, indexedarray_t))
            elif isinstance(var.type, Struct):
                fields.append((label, var.type.ctype))
            elif issubclass(var.type, ctypes.Array):
                fields.append((label, var.type))
            elif _is_texture_type(var.type):
                fields.append((label, var.type._wp_ctype_))
            else:
                # HACK: fp16/bf16 requires conversion functions from warp.so
                if var.type is warp.float16 or var.type is warp.bfloat16:
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
            s = f"{name}:{warp._src.types.get_type_code(type_hint)}"
            ch.update(bytes(s, "utf-8"))

            # recurse on nested structs
            if isinstance(type_hint, Struct):
                ch.update(type_hint.hash)

        self.hash = ch.digest()

        # generate unique identifier for structs in native code
        hash_suffix = f"{self.hash.hex()[:8]}"
        self.native_name = f"{self.key}_{hash_suffix}"

        # create default constructor (zero-initialize)
        self.default_constructor = warp._src.context.Function(
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

        self.value_constructor = warp._src.context.Function(
            func=None,
            key=self.native_name,
            namespace="",
            value_func=lambda *_: self,
            input_types=input_types,
            initializer_list_func=lambda *_: False,
            native_func=self.native_name,
        )

        self.default_constructor.add_overload(self.value_constructor)

        if isinstance(module, warp._src.context.Module):
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
            cls: ClassVar[type] = self.cls
            native_name: ClassVar[str] = self.native_name

            _cls: ClassVar[type] = self
            _constructors: ClassVar[list[tuple[str, Callable]]] = [
                (field, _make_struct_field_constructor(field, var.type)) for field, var in self.vars.items()
            ]
            _setters: ClassVar[dict[str, Callable]] = {
                field: _make_struct_field_setter(self.cls, field, var.type) for field, var in self.vars.items()
            }

            def __init__(inst, ctype=None):
                StructInstance.__init__(inst, ctype or self.ctype())

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
            if matches_array_class(var.type, array):
                # array_t
                formats.append(array_t.numpy_dtype())
            elif matches_array_class(var.type, indexedarray):
                # indexedarray_t
                formats.append(indexedarray_t.numpy_dtype())
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
            if matches_array_class(var.type, array):
                # We could reconstruct wp.array from array_t, but it's problematic.
                # There's no guarantee that the original wp.array is still allocated and
                # no easy way to make a backref.
                # Instead, we just create a stub annotation, which is not a fully usable array object.
                setattr(instance, name, array(dtype=var.type.dtype, ndim=var.type.ndim))
            elif matches_array_class(var.type, indexedarray):
                # Same as regular arrays: return an annotation stub only.
                setattr(instance, name, indexedarray(dtype=var.type.dtype, ndim=var.type.ndim))
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
                    setattr(instance, name, half_bits_to_float(cvalue.value))
                elif var.type == warp.bfloat16:
                    setattr(instance, name, bfloat16_bits_to_float(cvalue.value))
                else:
                    setattr(instance, name, cvalue.value)

        return instance


def compute_type_str(base_name, template_params):
    if not template_params:
        return base_name

    def param2str(p):
        if isinstance(p, builtins.bool):
            return "true" if p else "false"
        if isinstance(p, int):
            return str(p)
        elif hasattr(p, "_wp_generic_type_str_"):
            return compute_type_str(f"wp::{p._wp_generic_type_str_}", p._wp_type_params_)
        elif hasattr(p, "_type_"):
            if p.__name__ == "bool":
                return "bool"
            else:
                return f"wp::{p.__name__}"
        elif is_tile(p):
            return p.ctype()
        elif isinstance(p, Struct):
            return p.native_name

        return p.__name__

    return f"{base_name}<{', '.join(map(param2str, template_params))}>"


@dataclass(frozen=True)
class _LValueStep:
    """One access step from an addressable root to a nested lvalue."""

    kind: Literal["field", "array", "view", "index"]
    payload: str | tuple[Var, ...]
    pointer_cast: str = ""
    adjoint_pointer_cast: str = ""


@dataclass(frozen=True)
class _LValueOrigin:
    """A recursive path from an addressable root to a ``Reference(T)`` value."""

    root: Var
    root_is_ref_parameter: bool = False
    steps: tuple[_LValueStep, ...] = ()

    @classmethod
    def from_ref_parameter(cls, parameter: Var) -> _LValueOrigin:
        return cls(parameter, root_is_ref_parameter=True)

    @classmethod
    def from_local(cls, local: Var) -> _LValueOrigin:
        return cls(local)

    def extend(self, step: _LValueStep) -> _LValueOrigin:
        return _LValueOrigin(
            self.root,
            root_is_ref_parameter=self.root_is_ref_parameter,
            steps=(*self.steps, step),
        )

    def extend_field(
        self,
        field_label: str,
        pointer_cast: str = "",
        adjoint_pointer_cast: str = "",
    ) -> _LValueOrigin:
        return self.extend(
            _LValueStep(
                "field",
                field_label,
                pointer_cast=pointer_cast,
                adjoint_pointer_cast=adjoint_pointer_cast,
            )
        )

    def extend_array(self, indices: Sequence[Var]) -> _LValueOrigin:
        return self.extend(_LValueStep("array", tuple(indices)))

    def extend_view(self, indices: Sequence[Var]) -> _LValueOrigin:
        return self.extend(_LValueStep("view", tuple(indices)))

    def extend_index(self, indices: Var | Sequence[Var]) -> _LValueOrigin:
        if isinstance(indices, Var):
            indices = (indices,)
        return self.extend(_LValueStep("index", tuple(indices)))

    @staticmethod
    def _emit_indices(step: _LValueStep) -> str:
        assert isinstance(step.payload, tuple)
        return ", ".join(x.emit() for x in step.payload)

    @classmethod
    def _emit_step_lvalue(cls, base_expr: str, step: _LValueStep) -> str:
        if step.kind == "field":
            return f"({base_expr}).{step.payload}"
        if step.kind == "array":
            return f"(*wp::address({base_expr}, {cls._emit_indices(step)}))"
        if step.kind == "view":
            return f"wp::view({base_expr}, {cls._emit_indices(step)})"
        if step.kind == "index":
            return f"(*wp::index({base_expr}, {cls._emit_indices(step)}))"
        raise WarpCodegenError(f"Unsupported reference lvalue access kind: {step.kind}")

    @classmethod
    def _emit_step_pointer(cls, base_expr: str, step: _LValueStep, adjoint: bool = False) -> str:
        pointer_cast = step.adjoint_pointer_cast if adjoint else step.pointer_cast
        if step.kind == "array":
            return f"{pointer_cast}wp::address({base_expr}, {cls._emit_indices(step)})"
        if step.kind == "index":
            return f"{pointer_cast}wp::index({base_expr}, {cls._emit_indices(step)})"
        return f"{pointer_cast}&({cls._emit_step_lvalue(base_expr, step)})"

    def emit_lvalue(self, adjoint: bool = False) -> str:
        if self.root_is_ref_parameter:
            root_expr = self.root.emit("adj") if adjoint else self.root.emit()
            expr = f"*({root_expr})"
        else:
            expr = self.root.emit("adj" if adjoint else "var")

        for step in self.steps:
            expr = self._emit_step_lvalue(expr, step)

        return expr

    @classmethod
    def _emit_materialized_view(cls, adj, prelude: list[str], base_expr: str, step: _LValueStep) -> str:
        name = f"_wp_ref_view_{adj.label_count}"
        adj.label_count += 1
        prelude.append(f"auto {name} = {cls._emit_step_lvalue(base_expr, step)};")
        return name

    def emit_pointer(self, adjoint: bool = False, adj=None, prelude: list[str] | None = None) -> str:
        if not self.steps:
            if self.root_is_ref_parameter:
                # Root is already a pointer — return it directly.
                root_expr = self.root.emit("adj") if adjoint else self.root.emit()
                return root_expr
            return f"&({self.emit_lvalue(adjoint)})"

        # Walk all but the last step as lvalue expressions, then let the
        # terminal step emit its pointer (avoids &(*ptr) for array steps).
        if self.root_is_ref_parameter:
            root_expr = self.root.emit("adj") if adjoint else self.root.emit()
            expr = f"*({root_expr})"
        else:
            expr = self.root.emit("adj" if adjoint else "var")

        for step in self.steps[:-1]:
            if prelude is not None and step.kind == "view":
                expr = self._emit_materialized_view(adj, prelude, expr, step)
            else:
                expr = self._emit_step_lvalue(expr, step)

        return self._emit_step_pointer(expr, self.steps[-1], adjoint=adjoint)

    def emit_adjoint_pointer(self, adj, var: Var, prelude: list[str] | None = None) -> str:
        return self.emit_pointer(adjoint=True, adj=adj, prelude=prelude)


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

        # For Reference(T) vars, records the lvalue provenance used to derive
        # the adjoint storage pointer on demand.
        self.ref_origin: _LValueOrigin | None = None

        # Used to associate the variable with the Python statement that resulted in it being created.
        self.relative_lineno = relative_lineno

    def __str__(self):
        return self.label

    @staticmethod
    def dtype_to_ctype(t: type) -> str:
        if hasattr(t, "_wp_generic_type_str_"):
            return compute_type_str(f"wp::{t._wp_generic_type_str_}", t._wp_type_params_)
        elif isinstance(t, Struct):
            return t.native_name
        elif hasattr(t, "_wp_native_name_"):
            return f"wp::{t._wp_native_name_}"
        elif t.__name__ in ("bool", "int", "float"):
            return t.__name__

        return f"wp::{t.__name__}"

    @staticmethod
    def type_to_ctype(t: type, value_type: builtins.bool = False) -> str:
        if isinstance(t, fixedarray):
            template_args = (str(t.size), Var.dtype_to_ctype(t.dtype))
            dtypestr = ", ".join(template_args)
            classstr = f"wp::{type(t).__name__}"
            return f"{classstr}_t<{dtypestr}>"
        elif is_array(t):
            dtypestr = Var.dtype_to_ctype(t.dtype)
            classstr = f"wp::{concrete_array_type(t).__name__}"
            return f"{classstr}_t<{dtypestr}>"
        elif get_origin(t) is tuple:
            dtypestr = ", ".join(Var.dtype_to_ctype(x) for x in get_args(t))
            return f"wp::tuple_t<{dtypestr}>"
        elif is_tuple(t):
            dtypestr = ", ".join(Var.dtype_to_ctype(x) for x in t.types)
            classstr = f"wp::{type(t).__name__}"
            return f"{classstr}<{dtypestr}>"
        elif is_tile(t):
            return t.ctype()
        elif is_tile_stack(t):
            return t.ctype()
        elif isinstance(t, type) and issubclass(t, StructInstance):
            # ensure the actual Struct name is used instead of "NewStructInstance"
            return t.native_name
        elif is_reference(t):
            if not value_type:
                return Var.type_to_ctype(t.value_type) + "*"

            return Var.type_to_ctype(t.value_type)

        return Var.dtype_to_ctype(t)

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
        if self.is_read and warp._src.codegen.options.get("verify_autograd_array_access", False):
            if "kernel_name" in kwargs and "filename" in kwargs and "lineno" in kwargs:
                log_warning(
                    f"Array passed to argument {self.label} in kernel {kwargs['kernel_name']} at {kwargs['filename']}:{kwargs['lineno']} is being written to after it has been read from within the same kernel. This may corrupt gradient computation in the backward pass."
                )
            else:
                log_warning(
                    f"Array {self} is being written to after it has been read from within the same kernel. This may corrupt gradient computation in the backward pass."
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
        if name in arguments:
            new_arguments.append((name, arguments[name]))
        elif name in values:
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
    for bound_arg_type, func_arg_type in zip(bound_arg_types, func.input_types.values(), strict=True):
        # Let the `value_func` callback infer the type.
        if bound_arg_type is None:
            continue

        # if arg type registered as Any, treat as
        # template allowing any type to match
        if func_arg_type is Any:
            continue

        # Function parameters are type-erased during overload matching. User
        # functions spell those slots with `wp.Function`; some existing
        # builtins spell them with `Callable` for operator arguments.
        if isinstance(bound_arg_type, warp._src.context.Function) and (
            warp._src.types.is_warp_function_annotation(func_arg_type)
            or (func.is_builtin() and warp._src.types.is_builtin_callable_annotation(func_arg_type))
        ):
            continue

        bound_arg_type_stripped = strip_reference(bound_arg_type)

        # Strip Reference(T) from the parameter so that an argument of type T matches.
        param_type = strip_reference(func_arg_type)

        # Handle array polymorphism (e.g., passing a fixed array to a function taking an array).
        func_concrete = concrete_array_type(param_type)
        bound_concrete = concrete_array_type(bound_arg_type_stripped)
        if (
            is_array(param_type)
            and (issubclass(func_concrete, bound_concrete) or issubclass(bound_concrete, func_concrete))
            and types_equal_generic(param_type.dtype, bound_arg_type_stripped.dtype, match_generic=True)
        ):
            continue

        # check arg type matches input variable type
        if not types_equal_generic(param_type, bound_arg_type_stripped):
            return False

    return True


def is_regular_builtin_callable_target(func):
    """Return whether ``func`` is a simple built-in function target.

    Function parameters can specialize on built-ins that lower directly through
    the normal built-in function path. Built-ins that need variadic handling,
    dispatch callbacks, replay suppression, or LTO dispatch are excluded because
    they need special codegen behavior that cannot be represented by replacing a
    callable parameter with a direct function call.

    Args:
        func: Function object to check.

    Returns:
        ``True`` if every overload of ``func`` is supported as a function target.
    """

    if not isinstance(func, warp._src.context.Function) or not func.is_builtin():
        return False

    overloads = getattr(func, "overloads", None) or (func,)
    for overload in overloads:
        if (
            not overload.is_builtin()
            or overload.variadic
            or overload.dispatch_func is not None
            or overload.lto_dispatch_func is not None
            or overload.skip_replay
        ):
            return False

    return True


def get_callable_arg_values(func, bound_args):
    """Return concrete function targets for ``wp.Function`` parameters.

    ``bound_args`` already includes defaults. A non-empty result means the call
    needs a specialized clone where callable parameter names resolve directly to
    function objects during codegen instead of runtime variables.

    Args:
        func: Function being called.
        bound_args: Bound argument values for the call, including defaults.

    Returns:
        A mapping from function parameter names to concrete Warp functions, or
        ``None`` when the call has no concrete function targets.
    """

    if func.is_builtin():
        return None

    callable_arg_values = {}

    for name, value in bound_args.items():
        if not warp._src.types.is_warp_function_annotation(func.input_types.get(name)):
            continue

        if not isinstance(value, warp._src.context.Function):
            continue

        if value.is_builtin() and not is_regular_builtin_callable_target(value):
            raise WarpCodegenError(
                "wp.Function parameters support user-defined Warp functions and simple built-in Warp functions "
                "such as wp.sin() and wp.min(), "
                f"but parameter '{name}' of '{func.key}' received unsupported built-in function '{value.key}'."
            )

        callable_arg_values[name] = value

    if callable_arg_values:
        return callable_arg_values

    return None


def get_default_arg_value(func, name, value):
    """Return the codegen value for a default argument.

    Function defaults are specialization inputs, not runtime constants, so they
    stay as raw function objects. Other defaults are represented as constant
    variables and emitted through the regular default-argument path.

    Args:
        func: Function that owns the default argument.
        name: Parameter name for ``value``.
        value: Python default value from the function signature.

    Returns:
        A Warp function for function defaults, otherwise a constant ``Var``.
    """

    if warp._src.types.is_warp_function_annotation(func.input_types.get(name)) and isinstance(
        value, warp._src.context.Function
    ):
        # Function defaults need the same specialization path as explicit
        # function arguments.
        return value

    return Var(None, type=type(value), constant=value)


def bind_call_arg_nodes(func, call_node):
    """Bind a call AST to ``func`` and return AST/default arguments by name."""

    try:
        bound_args = func.signature.bind(*call_node.args, **{kw.arg: kw.value for kw in call_node.keywords})
    except TypeError:
        return {}

    default_args = {k: v for k, v in func.defaults.items() if k not in bound_args.arguments and v is not None}
    apply_defaults(bound_args, default_args)
    return bound_args.arguments


def resolve_callable_arg_target(adj, arg_node, callable_arg_values=None):
    """Resolve a callable argument node or default to a concrete Warp function.

    Args:
        adj: Adjoint whose symbols and globals should be used for resolution.
        arg_node: AST node or default value bound to a function parameter.
        callable_arg_values: Specialized function targets already bound in the
            caller, keyed by parameter name.

    Returns:
        The resolved Warp function, or the unresolved object when resolution
        does not produce a function.
    """

    if isinstance(arg_node, warp._src.context.Function):
        return arg_node

    if callable_arg_values and isinstance(arg_node, ast.Name):
        callable_func = callable_arg_values.get(arg_node.id)
        if callable_func is not None:
            return callable_func

    callable_func, _ = adj.resolve_static_expression(arg_node, eval_types=False)
    return callable_func


_UNRESOLVED_CALL_ARG = object()


def resolve_call_arg_type(adj, arg_node, callable_arg_values=None):
    """Best-effort static type resolution for call arguments during reference scans."""

    if isinstance(arg_node, ast.Name):
        if callable_arg_values:
            callable_func = callable_arg_values.get(arg_node.id)
            if callable_func is not None:
                return get_arg_type(callable_func)

        symbol = adj.symbols.get(arg_node.id)
        if symbol is not None:
            return get_arg_type(symbol)

        obj = adj.resolve_external_reference(arg_node.id)
        if obj is not None:
            return get_arg_type(obj)

        return _UNRESOLVED_CALL_ARG

    if isinstance(arg_node, ast.Attribute):
        obj, _ = adj.resolve_static_expression(arg_node, eval_types=False)
        if obj is not None:
            return get_arg_type(obj)

        return _UNRESOLVED_CALL_ARG

    if isinstance(arg_node, ast.Constant):
        return get_arg_type(arg_node.value)

    try:
        return get_arg_type(ast.literal_eval(arg_node))
    except (TypeError, ValueError):
        return _UNRESOLVED_CALL_ARG


def resolve_call_arg_types(adj, call_node, callable_arg_values=None):
    """Return static call argument types and whether every type was resolved."""

    arg_types = []
    resolved = True

    for arg_node in call_node.args:
        if isinstance(arg_node, ast.Starred):
            resolved = False
            continue

        arg_type = resolve_call_arg_type(adj, arg_node, callable_arg_values)
        if arg_type is _UNRESOLVED_CALL_ARG:
            resolved = False
        else:
            arg_types.append(arg_type)

    kwarg_types = {}
    for kw_node in call_node.keywords:
        if kw_node.arg is None:
            resolved = False
            continue

        arg_type = resolve_call_arg_type(adj, kw_node.value, callable_arg_values)
        if arg_type is _UNRESOLVED_CALL_ARG:
            resolved = False
        else:
            kwarg_types[kw_node.arg] = arg_type

    return tuple(arg_types), kwarg_types, resolved


def iter_call_func_overload_candidates(func, call_node):
    """Yield overloads whose signatures can bind ``call_node`` arguments."""

    overloads = []
    if hasattr(func, "user_overloads"):
        overloads.extend(func.user_overloads.values())
    if hasattr(func, "user_templates"):
        overloads.extend(func.user_templates.values())
    if not overloads:
        overloads.append(func)

    yielded = set()
    for overload in overloads:
        if overload in yielded:
            continue

        yielded.add(overload)
        if bind_call_arg_nodes(overload, call_node):
            yield overload


def resolve_grad_call_reference_func(adj, func_node, callable_arg_values=None):
    """Return the function wrapped by a ``wp.grad(...)`` callee, if any."""

    if not isinstance(func_node, ast.Call) or len(func_node.args) != 1 or func_node.keywords:
        return None

    grad_func, _ = adj.resolve_static_expression(func_node.func, eval_types=False)
    if not adj.is_grad_expression(grad_func):
        return None

    target_func = resolve_callable_arg_target(adj, func_node.args[0], callable_arg_values)
    if isinstance(target_func, warp._src.context.Function):
        return target_func

    return None


def resolve_reference_call_func(adj, call_node, callable_arg_values=None):
    """Resolve the Warp function called by ``call_node`` during reference scans."""

    if callable_arg_values is None:
        callable_arg_values = {}

    func, _ = adj.resolve_static_expression(call_node.func, eval_types=False)
    if func is None and isinstance(call_node.func, ast.Name):
        func = callable_arg_values.get(call_node.func.id)

    if func is None:
        func = resolve_grad_call_reference_func(adj, call_node.func, callable_arg_values)
    elif isinstance(func, warp._src.context.GradWrapper):
        func = func.func

    return func


def iter_call_callable_arg_targets(adj, func, call_node, callable_arg_values=None):
    """Yield Warp function targets passed to ``wp.Function`` parameters.

    Args:
        adj: Adjoint whose symbols and globals should be used for resolution.
        func: Function object referenced by the call.
        call_node: AST call node whose arguments may include function targets.
        callable_arg_values: Specialized function targets already bound in the
            caller, keyed by parameter name.

    Yields:
        Concrete Warp functions supplied to function parameters by explicit
        arguments or defaults.
    """

    if not isinstance(func, warp._src.context.Function) or func.is_builtin():
        return

    arg_types, kwarg_types, resolved = resolve_call_arg_types(adj, call_node, callable_arg_values)
    if resolved:
        overload = func.get_overload(arg_types, kwarg_types)
        func_candidates = (overload or func,)
    else:
        func_candidates = iter_call_func_overload_candidates(func, call_node)

    yielded = set()
    for func_candidate in func_candidates:
        bound_arg_nodes = bind_call_arg_nodes(func_candidate, call_node)

        for arg_name, arg_node in bound_arg_nodes.items():
            if not warp._src.types.is_warp_function_annotation(func_candidate.input_types.get(arg_name)):
                continue

            callable_func = resolve_callable_arg_target(adj, arg_node, callable_arg_values)
            if isinstance(callable_func, warp._src.context.Function) and callable_func not in yielded:
                yielded.add(callable_func)
                yield callable_func


def specialize_callable_func(func, callable_arg_values):
    """Clone ``func`` for a concrete set of function parameter targets.

    Function targets affect generated code but are omitted from the native C++
    signature, so each target set needs a cached specialization with a distinct
    native function name. The clone keeps the original Python source and arg
    types while storing the concrete function targets on the new adjoint.

    Args:
        func: User-defined function to specialize.
        callable_arg_values: Mapping from function parameter names to concrete
            Warp functions.

    Returns:
        A cached specialized clone of ``func`` for ``callable_arg_values``.
    """

    if func.custom_grad_func is not None or func.custom_replay_func is not None:
        raise WarpCodegenError(
            "wp.Function parameters are not supported on functions with custom gradients or replay functions: "
            f"'{func.key}'"
        )

    specialization_key = tuple(
        (name, callable_arg_values[name]) for name in func.input_types if name in callable_arg_values
    )

    specializations = getattr(func, "_callable_specializations", None)
    if specializations is None:
        specializations = {}
        func._callable_specializations = specializations

    specialized_func = specializations.get(specialization_key)
    if specialized_func is not None:
        return specialized_func

    # The callable targets are inlined by name while being omitted from the C++
    # function parameters, so each target set needs a distinct native name.
    suffix_hash = hashlib.sha256()
    suffix_hash.update(bytes(func.native_func, "utf-8"))
    for name, callable_func in specialization_key:
        suffix_hash.update(bytes(name, "utf-8"))
        suffix_hash.update(bytes(callable_func.key, "utf-8"))
        suffix_hash.update(bytes(callable_func.native_func, "utf-8"))

    specialized_func = shallowcopy(func)
    # Specialization clones should not share the parent specialization cache.
    specialized_func.__dict__.pop("_callable_specializations", None)
    specialized_func.native_func = f"{func.native_func}_callable_{suffix_hash.hexdigest()[:12]}"
    specialized_func.value_func = None
    specialized_func.adj = Adjoint(
        func.func,
        overload_annotations=func.adj.arg_types,
        is_user_function=func.adj.is_user_function,
        skip_forward_codegen=func.adj.skip_forward_codegen,
        skip_reverse_codegen=func.adj.skip_reverse_codegen,
        custom_reverse_mode=func.adj.custom_reverse_mode,
        custom_reverse_num_input_args=func.adj.custom_reverse_num_input_args,
        transformers=func.adj.transformers,
        source=func.adj.source,
    )
    specialized_func.adj.callable_arg_values = dict(callable_arg_values)
    specialized_func.adj.used_by_backward_kernel = func.adj.used_by_backward_kernel
    specialized_func.adj.force_adjoint_codegen = func.adj.force_adjoint_codegen

    specializations[specialization_key] = specialized_func
    return specialized_func


def get_arg_type(arg: Var | Any) -> type:
    arg = strip_reference(arg)

    # `Any` marks an unspecialized generic parameter. Return it unchanged so
    # downstream signature logic can treat it as generic. Before Python 3.11
    # `Any` is a `typing._SpecialForm` instance rather than a `type`, so the
    # `isinstance(arg, type)` check below would otherwise fall through to
    # `type(arg)` and yield `typing._SpecialForm`.
    if arg is Any:
        return Any

    if isinstance(arg, str):
        return str

    if is_struct(arg):
        return arg._cls

    if isinstance(
        arg,
        (
            type,
            *array_types,
            warp._src.types._ArrayAnnotationBase,
            warp._src.codegen.Struct,
            warp._src.context.Function,
            tuple_t,
            slice_t,
            tile,
            tile_stack,
        ),
    ):
        return arg

    if isinstance(arg, Var):
        if get_origin(arg.type) is tuple:
            return get_args(arg.type)

        return get_arg_type(arg.type)

    if isinstance(arg, Sequence):
        return tuple(get_arg_type(x) for x in arg)

    if get_origin(arg) is tuple:
        return tuple(get_arg_type(x) for x in get_args(arg))

    return type(arg)


def get_arg_value(arg: Any) -> Any:
    arg = strip_reference(arg)

    if isinstance(arg, (type, str, warp._src.context.Function)):
        return arg

    if isinstance(arg, Sequence):
        return tuple(get_arg_value(x) for x in arg)

    if isinstance(arg, Var):
        if is_tuple(arg.type):
            return tuple(get_arg_value(x) for x in arg.type.values)

        if arg.constant is not None:
            return arg.constant

    return arg


# Re-entrant lock guarding mutation and reading of shared per-Adjoint
# state. ``@wp.func`` helpers share one ``Adjoint`` object across every
# module that references them; ``Adjoint.build`` rewrites ``adj.blocks``,
# ``adj.symbols``, ``adj.variables``, ``adj.deferred_static_expressions``
# from scratch, and ``ModuleBuilder.codegen`` later reads those same
# fields. Holding this lock across the full build+emit window in
# ``Module._compile`` stops a parallel ``Module.load`` (e.g. from
# ``wp.force_load(max_workers > 1)``) from clobbering the state mid-emit.
# Re-entrant so ``Module._compile`` -> ``ModuleBuilder.build_kernel`` ->
# ``Adjoint.build`` on the same thread doesn't deadlock.
_codegen_lock = threading.RLock()


def synchronized(rlock: threading.RLock | None = None):
    """Decorator that serializes calls to the wrapped function under a re-entrant lock.

    ``@synchronized()`` mints a fresh private ``RLock`` for this decoration.
    ``@synchronized(rlock)`` uses the given ``RLock``; pass the same lock to
    every decoration that must mutually exclude. The lock is re-entrant, so
    nested calls from the same thread do not deadlock.
    """
    if rlock is None:
        rlock = threading.RLock()

    def decorator(func):
        @functools.wraps(func)
        def locked_call(*args, **kwargs):
            with rlock:
                return func(*args, **kwargs)

        return locked_call

    return decorator


class SlotAccessPlan(NamedTuple):
    """Result of classifying an array-rooted composite-component write LHS.

    Built by ``Adjoint._classify_slot_access`` without emitting any IR. The
    AST nodes in ``array_indices_ast`` and the AST nodes interleaved in
    ``access_parts`` are evaluated by the caller only after the access is
    accepted, so a declined write never pollutes the IR.
    """

    root_var: Var
    array_indices_ast: list  # list[ast.expr]: outer array subscripts, left-to-right
    access_parts: list  # list[str | ast.expr]: str text segments + index AST nodes
    slot_type: object  # informational leaf type; already validated against rhs_type


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
        source: str | None = None,
    ):
        adj.func = func

        adj.is_user_function = is_user_function

        # whether the generation of the forward code is skipped for this function
        adj.skip_forward_codegen = skip_forward_codegen
        # whether the generation of the adjoint code is skipped for this function
        adj.skip_reverse_codegen = skip_reverse_codegen
        # Whether this function is used by a kernel that has has the backward pass enabled.
        adj.used_by_backward_kernel = False
        # Whether to force adjoint code generation regardless of enable_backward setting.
        # This is used by warp.grad() to ensure the adjoint exists even in forward-only modules.
        adj.force_adjoint_codegen = False
        # Whether this function uses warp.grad() calls. Such functions are generated in a
        # separate pass after adjoints, and don't have their own adjoints generated.
        adj.uses_grad_call = False

        # extract name of source file
        adj.filename = inspect.getsourcefile(func) or "unknown source file"
        # get source file line number where function starts
        adj.fun_lineno = 0
        if source is None:
            adj.source, adj.fun_lineno, adj.tree = adj.extract_function_source(func)
        else:
            # ensures that indented class methods can be parsed as kernels
            adj.source = textwrap.dedent(source)
            adj.tree = ast.parse(adj.source)

        assert adj.source is not None, f"Failed to extract source code for function {func.__name__}"

        # Indicates where the function definition starts (excludes decorators)
        adj.fun_def_lineno = None

        adj.source_lines = adj.source.splitlines()

        if transformers is None:
            transformers = []

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
        # Names of parameters declared as wp.ref[T]. These are Reference(T) vars in
        # both adj.args and adj.symbols; generated C++ treats them as pointer
        # IR while Python-level semantics expose pass-by-reference.
        adj.ref_params: dict[str, Var] = {}

        for name, type in adj.arg_types.items():
            # skip return hint
            if name == "return":
                continue

            # add variable for argument
            arg = Var(name, type, requires_grad=False)
            adj.args.append(arg)

            if is_reference(type):
                adj.symbols[name] = arg
                adj.ref_params[name] = arg
                arg.ref_origin = _LValueOrigin.from_ref_parameter(arg)
            else:
                if is_array(type):
                    arg.ref_origin = _LValueOrigin.from_local(arg)
                # pre-populate symbol dictionary with function argument names
                # this is to avoid registering false references to overshadowed modules
                adj.symbols[name] = arg

        # Indicates whether there are unresolved static expressions in the function.
        # These stem from wp.static() expressions that could not be evaluated at declaration time.
        # This will signal to the module builder that this module needs to be rebuilt even if the module hash is unchanged.
        adj.has_unresolved_static_expressions = False

        # wp.static() expressions resolved at declaration time (replace_static_expressions),
        # keyed by source code string. Used for hashing. Immutable after __init__.
        adj.resolved_static_expressions: dict[str, Any] = {}
        if "static" in adj.source:
            adj.replace_static_expressions()

        # wp.static() expressions resolved during codegen (emit_Call) for expressions that
        # depend on loop variables; reset at the start of each build(). Used for hashing.
        # This is a list (not a dict) because the same source expression can evaluate to
        # different values across loop iterations — e.g. wp.static(values[i]) always has
        # key "values[i]" but a different value per iteration.
        adj.deferred_static_expressions: list[tuple[str, Any]] = []

        # There are cases where a same module might be rebuilt multiple times,
        # for example when kernels are nested inside of functions, or when
        # a kernel's launch raises an exception. Ideally we'd always want to
        # avoid rebuilding kernels but some corner cases seem to depend on it,
        # so we only avoid rebuilding kernels that errored out to give a chance
        # for unit testing errors being spit out from kernels.
        adj.skip_build = False

        # Feature-specific deterministic lowering state.  Keep its policy and
        # helper metadata behind a small integration object so the core codegen
        # paths do not need to coordinate deterministic internals directly.
        adj.deterministic = DeterministicCodegen(adj)

        # Cache of reference-candidate AST nodes, materialized once by ``reference_nodes()``.
        # Reset to None if ``adj.tree`` is ever mutated after the cache is populated.
        adj._reference_nodes = None

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
            elif is_tile_stack(var.type):
                total_shared += var.type.size_in_bytes()

        return total_shared + adj.max_required_extra_shared_memory

    @staticmethod
    def extract_function_source(func: Callable) -> tuple[str, int, ast.Module]:
        """Extract a function's source as ``inspect.getsourcelines`` would, but faster.

        Uses a ``co_lines()``-based heuristic to find the function's source slice
        without tokenizing, then verifies the slice by parsing it and checking
        that the parsed block starts with the target function. On any parse or
        validation failure the implementation falls back transparently to
        ``inspect.getsourcelines`` — so the only observable difference from the
        upstream behavior is throughput.

        Returns ``(source, fun_lineno, tree)`` where ``source`` is dedented and
        ``tree == ast.parse(source)``. Callers can rely on ``tree.body[0]`` being
        a ``FunctionDef`` / ``AsyncFunctionDef`` with ``name == func.__code__.co_name``.

        ``inspect.unwrap`` is applied before reading ``__code__`` so the fast path
        matches ``inspect.getsourcelines``'s semantics for ``__wrapped__`` chains
        (e.g. ``functools.wraps``-decorated functions).

        **Correctness.** The fast path's slice either parses to the target
        function or it is rejected. The argument:

        - The slice always covers ``[co_firstlineno, max_line]`` where
          ``max_line = max(line for *,*,line in code.co_lines())``. By PEP 626
          every bytecode instruction has a line in ``co_lines()``, so the slice
          contains every executable statement of ``func``.
        - If ``ast.parse`` succeeds and ``body[0]`` is a ``FunctionDef`` /
          ``AsyncFunctionDef`` whose ``name == code.co_name``, the parsed block
          starts at the target function and its body contains every executable
          statement of ``func``. Any content the forward walk overshot into
          ``body[1:]`` is ignored by downstream code, which only ever reads
          ``body[0]``.
        - The slice can be *wrong* only if the forward walk stops *inside* the
          function body. The walk stops at the first non-blank line at indent
          ``<= base_indent`` (a comment at that indent counts — it terminates
          the suite just as code would), and it only ever scans forward from
          ``max_line``, the last line carrying bytecode. So by Python's grammar
          such a stop can land inside the function only if the line lies within
          an unclosed multi-line string or bracket expression that began at or
          before ``max_line``. Either case leaves the slice ending in an
          unclosed token, so ``ast.parse`` raises ``SyntaxError``.
        - ``SyntaxError`` triggers the fallback to
          :meth:`_inspect_extract_function_source`, which is the upstream
          ``inspect.getsourcelines`` behavior.

        So: parse + target-function validation success ⟹ correct slice; parse or
        validation failure ⟹ fallback. The slow path is also parsed; if *that*
        fails, the function itself has a syntax error and we let it propagate.
        """
        try:
            code = inspect.unwrap(func).__code__
        except (AttributeError, ValueError):
            code = None
        if code is not None:
            fast = Adjoint._try_extract_function_source(code)
            if fast is not None:
                fast_source, fast_lineno = fast
                dedented = textwrap.dedent(fast_source)
                try:
                    tree = ast.parse(dedented)
                except SyntaxError:
                    pass
                else:
                    if (
                        tree.body
                        and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
                        and tree.body[0].name == code.co_name
                    ):
                        return dedented, fast_lineno, tree
        source, fun_lineno = Adjoint._inspect_extract_function_source(func)
        dedented = textwrap.dedent(source)
        return dedented, fun_lineno, ast.parse(dedented)

    @staticmethod
    def _inspect_extract_function_source(func: Callable) -> tuple[str, int]:
        """Tokenizer-driven extraction via ``inspect.getsourcelines``. Default slow
        path for :meth:`extract_function_source` and the recovery path used when
        the fast extractor produces an unparsable slice.
        """
        try:
            source_lines, fun_lineno = inspect.getsourcelines(func)
        except OSError as e:
            raise RuntimeError(
                "Directly evaluating Warp code defined as a string using `exec()` is not supported, "
                "please save it to a file and use `importlib` if needed."
            ) from e
        return "".join(source_lines), fun_lineno

    @staticmethod
    def _try_extract_function_source(code: types.CodeType) -> tuple[str, int] | None:
        """Best-effort extraction of a function's source slice via ``co_lines()`` + linecache.

        Returns ``None`` when the file isn't in ``linecache``, ``co_firstlineno``
        is out of range, the def line is blank, or ``max_line < co_firstlineno``.
        Otherwise returns ``(source, co_firstlineno)`` where ``source`` covers
        ``[co_firstlineno, end]`` for some ``end >= max_line``. The slice is *not*
        guaranteed parseable — the parse-time fallback in
        :meth:`extract_function_source` catches every heuristic miss (see that
        method's docstring for the proof).
        """
        if not (code.co_filename.startswith("<") and code.co_filename.endswith(">")):
            linecache.checkcache(code.co_filename)
        lines = linecache.getlines(code.co_filename)
        start = code.co_firstlineno - 1
        if not lines or not (0 <= start < len(lines)):
            return None
        first = lines[start]
        stripped = first.lstrip()
        if not stripped:
            return None
        base_indent = len(first) - len(stripped)

        max_line = max((ln for _, _, ln in code.co_lines() if ln is not None), default=0)
        if max_line < code.co_firstlineno:
            return None

        # End at the first non-blank line at indent <= base_indent past max_line.
        end = max_line
        while end < len(lines):
            s = lines[end].lstrip()
            if s and len(lines[end]) - len(s) <= base_indent:
                break
            end += 1
        # inspect.getblock excludes trailing blanks; match that.
        while end > max_line and not lines[end - 1].strip():
            end -= 1

        return "".join(lines[start:end]), code.co_firstlineno

    # generate function ssa form and adjoint
    @synchronized(_codegen_lock)
    def build(adj, builder, default_builder_options=None, callable_arg_values=None):
        # arg Var read/write flags are held during module rebuilds, so we reset here even when skipping a build
        for arg in adj.args:
            arg.is_read = False
            arg.is_write = False

        if adj.skip_build:
            return

        if callable_arg_values is None:
            callable_arg_values = getattr(adj, "callable_arg_values", None)

        adj.builder = builder

        if default_builder_options is None:
            default_builder_options = {}

        if adj.builder:
            adj.builder_options = adj.builder.options
        else:
            adj.builder_options = default_builder_options

        global options
        options = adj.builder_options

        adj.deterministic.begin_build(adj.builder_options)

        adj.symbols = {}  # map from symbols to adjoint variables
        adj.variables = []  # list of local variables (in order)
        adj.deferred_static_expressions = []

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

        # Function-specialized functions replace selected argument Vars with
        # Function objects so calls like `op(x)` resolve statically.
        for a in adj.args:
            if callable_arg_values is not None and a.label in callable_arg_values:
                adj.symbols[a.label] = callable_arg_values[a.label]
            else:
                adj.symbols[a.label] = a

        # recursively evaluate function body
        try:
            adj.eval(adj.tree.body[0])

            # After evaluating the whole function we can validate the return
            # type. This needs to happen before actually writing the generated
            # code to the CUDA/C++ file to avoid a broken function from affecting
            # the compilation of valid ones.
            adj._validate_return_type()

        except Exception as original_exc:
            try:
                lineno = adj.lineno + adj.fun_lineno
                line = adj.source_lines[adj.lineno]
                msg = f'Error while parsing function "{adj.fun_name}" at {adj.filename}:{lineno}:\n{line}\n'

                # Combine the new message with the original exception's arguments
                new_args = (";".join([msg] + [str(a) for a in original_exc.args]),)

                # Enhance the original exception with parser context before re-raising.
                # 'from None' is used to suppress Python's chained exceptions for a cleaner error output.
                raise type(original_exc)(*new_args).with_traceback(original_exc.__traceback__) from None
            finally:
                adj.skip_build = True
                adj.builder = None

        if builder is not None:
            for a in adj.args:
                if isinstance(a.type, Struct):
                    builder.build_struct_recursive(a.type)
                elif warp._src.types.matches_array_class(a.type, warp._src.types.array) and isinstance(
                    a.type.dtype, Struct
                ):
                    builder.build_struct_recursive(a.type.dtype)

            # release builder reference for GC
            adj.builder = None

    def _validate_return_type(adj):
        """Validate function return type annotation against actual return values.

        This validation happens during build() (before C++ code generation) to catch
        errors early and prevent module contamination. If validation fails here,
        the function is marked as skip_build and won't emit any C++ code.
        """
        if adj.return_var is not None and "return" in adj.arg_types:
            if get_origin(adj.arg_types["return"]) is tuple:
                if len(get_args(adj.arg_types["return"])) != len(adj.return_var):
                    raise WarpCodegenError(
                        f"The function `{adj.fun_name}` has its return type "
                        f"annotated as a tuple of {len(get_args(adj.arg_types['return']))} elements "
                        f"but the code returns {len(adj.return_var)} values."
                    )
                elif not types_equal_generic(adj.arg_types["return"], tuple(x.type for x in adj.return_var)):
                    raise WarpCodegenError(
                        f"The function `{adj.fun_name}` has its return type "
                        f"annotated as `{warp._src.context.type_str(adj.arg_types['return'])}` "
                        f"but the code returns a tuple with types `({', '.join(warp._src.context.type_str(x.type) for x in adj.return_var)})`."
                    )
            elif len(adj.return_var) > 1 and get_origin(adj.arg_types["return"]) is not tuple:
                raise WarpCodegenError(
                    f"The function `{adj.fun_name}` has its return type "
                    f"annotated as `{warp._src.context.type_str(adj.arg_types['return'])}` "
                    f"but the code returns {len(adj.return_var)} values."
                )
            elif (
                isinstance(adj.return_var[0].type, warp._src.types.fixedarray)
                and type(adj.arg_types["return"]) is warp._src.types.array
            ):
                # If the return statement yields a `fixedarray` while the function is annotated
                # to return a standard `array`, then raise an error since the `fixedarray` storage
                # allocated on the stack will be freed once the function exits, meaning that the
                # resulting `array` instance will point to an invalid data.
                raise WarpCodegenError(
                    f"The function `{adj.fun_name}` returns a fixed-size array "
                    f"whereas it has its return type annotated as "
                    f"`{warp._src.context.type_str(adj.arg_types['return'])}`."
                )
            elif not types_equal(adj.arg_types["return"], adj.return_var[0].type):
                raise WarpCodegenError(
                    f"The function `{adj.fun_name}` has its return type "
                    f"annotated as `{warp._src.context.type_str(adj.arg_types['return'])}` "
                    f"but the code returns a value of type `{warp._src.context.type_str(adj.return_var[0].type)}`."
                )

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
            if isinstance(a, str):
                arg_strs.append(a)
            elif isinstance(a, warp._src.context.Function):
                # functions don't have a var_ prefix so strip it off here
                if prefix == "var":
                    arg_strs.append(f"{a.namespace}{a.native_func}")
                else:
                    arg_strs.append(f"{a.namespace}{prefix}_{a.native_func}")
            elif is_reference(a.type):
                arg_strs.append(a.emit(prefix))
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
        extra_args=None,
        has_output_args=True,
        require_original_output_arg=False,
    ):
        if extra_args is None:
            extra_args = []
        formatted_var = adj.format_args("var", args_var)
        formatted_extra = adj.format_args("var", extra_args)
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
            extra_str = ", ".join(formatted_extra)
            out_str = f"{{{', '.join(formatted_out)}}}"
            adj_str = f"{{{', '.join(formatted_var_adj)}}}"
            out_adj_str = ", ".join(formatted_out_adj)
            if len(args_out) > 1:
                arg_str = ", ".join(x for x in [var_str, extra_str, out_str, adj_str, out_adj_str] if x)
            else:
                arg_str = ", ".join(x for x in [var_str, extra_str, adj_str, out_adj_str] if x)
        else:
            arg_str = ", ".join(formatted_var + formatted_extra + formatted_out + formatted_var_adj + formatted_out_adj)
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

        if isinstance(var, (Reference, warp._src.context.Function)):
            return var

        if isinstance(var, int):
            return adj.add_constant(var)

        if isinstance(var, float):
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

        if adj.filename == "unknown source file" or adj.fun_lineno == 0:
            # Early return if function is not associated with a source file or is otherwise invalid
            # TODO: Get line directives working with wp.map() functions
            return None

        # lineinfo is enabled by default in debug mode regardless of the builder option, don't want to unnecessarily
        # emit line directives in generated code if it's not being compiled with line information
        build_mode = adj.builder_options.get("mode", "release")

        lineinfo_enabled = adj.builder_options.get("lineinfo", False) or build_mode == "debug"

        if relative_lineno is not None and lineinfo_enabled and adj.builder_options.get("line_directives", True):
            is_comment = statement.strip().startswith("//")
            if not is_comment:
                line = relative_lineno + adj.fun_lineno
                escaped_path = _escape_line_directive_filename(adj.filename)
                return f'#line {line} "{escaped_path}"'
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
        output = adj.add_var(type=get_arg_type(n), constant=n)
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

        for op, comp in zip(op_strings, comps, strict=True):
            if prev_comp_var:
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

        for x in itertools.chain(arg_types, kwarg_types.values()):
            if isinstance(x, warp._src.context.Function):
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

    @staticmethod
    def has_manual_ref_adjoint(func):
        if func.custom_grad_func is not None:
            return True

        return func.native_snippet is not None and func.adj_native_snippet is not None

    def emit_ref_adjoint_pointer(adj, var, prelude: list[str] | None = None):
        if var.ref_origin is None:
            raise WarpCodegenError(
                f"Internal error: missing adjoint storage expression for reference argument '{var.label}'"
            )

        return var.ref_origin.emit_adjoint_pointer(adj, var, prelude=prelude)

    def reference_origin_for_var(adj, var):
        """Return the lvalue origin for *var*, or ``None`` if it is not addressable.

        For Reference(T) vars the origin is stored directly on the var.
        For value-typed vars:
        - Named locals (in adj.symbols) are copies; their lvalue is their own
          storage regardless of any ref_origin inherited from the expression
          they were initialised from.
        - Anonymous temporaries propagate ref_origin (enabling chained access
          like ``arr[i].vec.y`` where intermediate results are never named).
        """
        if not isinstance(var, Var):
            return None

        if is_reference(var.type):
            return var.ref_origin  # Reference: stored origin (may be None)

        # Array-typed vars are descriptors, so named aliases and views keep
        # the provenance of the array storage they reference.
        if is_array(var.type) and var.ref_origin is not None:
            return var.ref_origin

        # Value-typed: named locals are copies — address their own storage.
        for symbol in adj.symbols.values():
            if symbol is var:
                return _LValueOrigin.from_local(var)

        # Anonymous temp: propagate ref_origin for chaining (may be None).
        return var.ref_origin

    def emit_addressable_reference(adj, node, var=None, expected_type=None, purpose="reference argument"):
        _err = (
            f"{purpose} requires an addressable expression "
            "(local variable, array element, struct field, or wp.ref[T] parameter)"
        )

        if node is None and var is None:
            raise WarpCodegenError(_err)

        if var is None:
            if not isinstance(node, (ast.Name, ast.Subscript, ast.Attribute)):
                raise WarpCodegenError(_err)
            var = adj.eval(node)

        if not isinstance(var, Var):
            raise WarpCodegenError(_err)

        if isinstance(node, ast.Name) and var.constant is not None:
            constructor = type_repr(var.type)
            if var.type in scalar_types:
                constructor = f"wp.{constructor}"
            raise WarpCodegenError(
                f"Error taking a mutable reference to constant local '{node.id}', use the following syntax: "
                f"{node.id} = {constructor}({var.constant!r}) to declare a dynamic variable"
            )

        if is_reference(var.type):
            # Already a Reference(T): struct fields, array elements, ref params.
            if var.ref_origin is None:
                raise WarpCodegenError(_err)
            ref_var = var
        elif var.ref_origin is not None and not isinstance(node, ast.Name):
            # Value-typed SSA with a traced lvalue origin from a direct expression
            # (ast.Attribute or ast.Subscript). Named local bindings (ast.Name) are
            # copies — their storage is addressed via Branch 3, not through the origin.
            ref_var = Var(var.ref_origin.emit_pointer(), Reference(var.type), prefix=False)
            ref_var.ref_origin = var.ref_origin
        elif isinstance(node, ast.Name):
            # Plain value-typed local variable (including named bindings of extracted
            # components): var came from adj.symbols[node.id] and its storage is
            # directly addressable. Named variables are copies, never aliases.
            ref_origin = _LValueOrigin.from_local(var)
            ref_var = Var(ref_origin.emit_pointer(), Reference(var.type), prefix=False)
            ref_var.ref_origin = ref_origin
        else:
            raise WarpCodegenError(_err)

        if expected_type is not None and not types_equal(ref_var.type.value_type, expected_type.value_type):
            raise WarpCodegenTypeError(
                f"{purpose} expects {type_repr(expected_type.value_type)}, got {type_repr(ref_var.type.value_type)}"
            )

        return ref_var

    def resolve_call(adj, func, args, kwargs, type_args=None, min_outputs=None, arg_nodes=None, kwarg_nodes=None):
        # Extract the types and values passed as arguments to the function call.
        arg_types = tuple(get_arg_type(x) for x in args)
        kwarg_types = {k: get_arg_type(v) for k, v in kwargs.items()}

        # Resolve the exact function signature among any existing overload.
        func = adj.resolve_func(func, arg_types, kwarg_types, min_outputs)

        # Bind the positional and keyword arguments to the function's signature
        # in order to process them as Python does it.
        bound_args: inspect.BoundArguments = func.signature.bind(*args, **kwargs)
        if arg_nodes is not None or kwarg_nodes is not None:
            bound_arg_nodes = func.signature.bind(*(arg_nodes or ()), **(kwarg_nodes or {})).arguments
        else:
            bound_arg_nodes = {}

        if type_args is None:
            type_args = {}

        # Type args are the "compile time" argument values we get from codegen.
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
                k: get_default_arg_value(func, k, v)
                for k, v in func.defaults.items()
                if k not in bound_args.arguments and v is not None
            }
            apply_defaults(bound_args, default_vars)

        bound_args = bound_args.arguments
        callable_arg_values = get_callable_arg_values(func, bound_args)
        if callable_arg_values is not None:
            func = specialize_callable_func(func, callable_arg_values)

        return func, bound_args, bound_arg_nodes

    def add_call(
        adj,
        func,
        args,
        kwargs,
        type_args,
        min_outputs=None,
        return_value_used=True,
        arg_nodes=None,
        kwarg_nodes=None,
        return_resolved=False,
    ):
        func, bound_args, bound_arg_nodes = adj.resolve_call(
            func,
            args,
            kwargs,
            type_args,
            min_outputs,
            arg_nodes=arg_nodes,
            kwarg_nodes=kwarg_nodes,
        )

        def return_value(output):
            if return_resolved:
                return output, func, bound_args
            return output

        # Constant precision preservation: when calling a 64-bit scalar type
        # constructor with a single compile-time constant argument, emit
        # a variable of the target type initialized directly from the
        # literal value.  This avoids precision loss from the intermediate
        # variable being narrowed to int32/float32.
        # The variable is NOT marked as a constant (via add_var's constant=
        # parameter) because emit_Assign maps symbols directly to the Var
        # returned here, and a const-qualified C++ variable cannot be passed
        # by non-const reference to functions that write through it.
        if (
            func.is_builtin()
            and func.value_type in (float64, int64, uint64)
            and func.native_func == func.value_type.__name__
            and len(bound_args) == 1
        ):
            arg = next(iter(bound_args.values()))
            if isinstance(arg, Var) and arg.constant is not None:
                raw = arg.constant
                # Unwrap Warp scalar type instances to their raw Python value
                if type(raw) in warp._src.types.scalar_types:
                    raw = raw.value
                if isinstance(raw, builtins.int) or (isinstance(raw, builtins.float) and not math.isnan(raw)):
                    # Repurpose the original arg variable: change its type to
                    # the 64-bit target and clear its constant so it emits as
                    # an uninitialized variable of the correct type, avoiding
                    # compiler warnings from narrowing a 64-bit literal.
                    arg.type = func.value_type
                    arg.constant = None
                    adj.add_forward(f"var_{arg} = {constant_str(func.value_type(raw))};")
                    return return_value(arg)

        # if it is a user-function then build it recursively
        if not func.is_builtin():
            # If the function called is a user function,
            # we need to ensure its adjoint is also being generated.
            if adj.used_by_backward_kernel:
                func.adj.used_by_backward_kernel = True
            if adj.force_adjoint_codegen:
                func.adj.force_adjoint_codegen = True

            if adj.builder is None:
                func.build(None, adj.builder_options)

            elif func not in adj.builder.functions:
                adj.builder.build_function(func)
                # add custom grad, replay functions to the list of functions
                # to be built later (invalid code could be generated if we built them now)
                # so that they are not missed when only the forward function is imported
                # from another module
                if func.custom_grad_func:
                    adj.builder.deferred_functions.append(func.custom_grad_func)
                if func.custom_replay_func:
                    adj.builder.deferred_functions.append(func.custom_replay_func)

            adj.deterministic.include_function_call(func, bound_args)

        # Resolve the return value based on the types and values of the given arguments.
        bound_arg_types = {k: get_arg_type(v) for k, v in bound_args.items()}
        bound_arg_values = {k: get_arg_value(v) for k, v in bound_args.items()}

        return_type = func.value_func(
            {k: strip_reference(v) for k, v in bound_arg_types.items()},
            bound_arg_values,
        )

        # Handle the special case where a Var instance is returned from the `value_func`
        # callback, in which case we replace the call with a reference to that variable.
        if isinstance(return_type, Var):
            return return_value(adj.register_var(return_type))
        elif isinstance(return_type, Sequence) and all(isinstance(x, Var) for x in return_type):
            return return_value(tuple(adj.register_var(x) for x in return_type))

        if get_origin(return_type) is tuple:
            types = get_args(return_type)
            return_type = warp._src.types.tuple_t(types=types, values=(None,) * len(types))

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

        # Deterministic mode: intercept atomic builtins and emit scatter or
        # two-pass code instead of the normal atomic call.
        det_output = adj.deterministic.emit_atomic_call(
            func, bound_args, return_type, output, output_list, return_value_used=return_value_used
        )
        if det_output is not None:
            return return_value(det_output)

        # If we have a built-in that requires special handling to dispatch
        # the arguments to the underlying C++ function, then we can resolve
        # these using the `dispatch_func`. Since this is only called from
        # within codegen, we pass it directly `codegen.Var` objects,
        # which allows for some more advanced resolution to be performed,
        # for example by checking whether an argument corresponds to
        # a literal value or references a variable.
        extra_shared_memory = 0
        func_arg_names = tuple(bound_args.keys())
        if func.lto_dispatch_func is not None:
            func_args, template_args, _ltoirs, extra_shared_memory = func.lto_dispatch_func(
                func.input_types, return_type, output_list, bound_args, options=adj.builder_options, builder=adj.builder
            )
        elif func.dispatch_func is not None:
            func_args, template_args = func.dispatch_func(func.input_types, return_type, bound_args)
        else:
            func_arg_names = tuple(
                name
                for name in bound_args
                if func.is_builtin() or not warp._src.types.is_warp_function_annotation(func.input_types.get(name))
            )
            func_args = tuple(bound_args[name] for name in func_arg_names)
            # Function parameters are specialization inputs, not C++ arguments.
            template_args = ()

        func_args = tuple(adj.register_var(x) for x in func_args)
        func_name = compute_type_str(func.native_func, template_args)
        use_initializer_list = func.initializer_list_func(bound_args, return_type)

        # For user functions, check which parameters are wp.ref[T] / Reference(T)
        # so we can pass pointer-shaped IR through without creating copies.
        parameter_types = func.input_types if not func.is_builtin() else {}
        func_has_ref_params = any(is_reference(t) for t in parameter_types.values())

        fwd_args = []
        reverse_adj_args = []
        reverse_prelude = []
        for i, func_arg in enumerate(func_args):
            arg_name = func_arg_names[i] if i < len(func_arg_names) else None
            parameter_type = parameter_types.get(arg_name)
            parameter_is_ref = is_reference(parameter_type)

            if parameter_is_ref:
                arg_node = bound_arg_nodes.get(arg_name)
                func_arg_var = adj.emit_addressable_reference(
                    arg_node,
                    var=func_arg,
                    expected_type=parameter_type,
                    purpose=f"wp.ref[{type_repr(parameter_type.value_type)}] parameter",
                )
                reverse_arg_var = Var(
                    adj.emit_ref_adjoint_pointer(func_arg_var, prelude=reverse_prelude),
                    parameter_type,
                    prefix=False,
                )
            elif not isinstance(func_arg, (Reference, warp._src.context.Function)):
                func_arg_var = adj.load(func_arg)
                reverse_arg_var = strip_reference(func_arg)
            else:
                func_arg_var = func_arg
                reverse_arg_var = strip_reference(func_arg)

            # if the argument is a function (and not a builtin), then build it recursively
            if isinstance(func_arg_var, warp._src.context.Function) and not func_arg_var.is_builtin():
                if adj.used_by_backward_kernel:
                    func_arg_var.adj.used_by_backward_kernel = True
                if adj.force_adjoint_codegen:
                    func_arg_var.adj.force_adjoint_codegen = True

                if adj.builder is None:
                    func_arg_var.build(None, adj.builder_options)
                else:
                    adj.builder.build_function(func_arg_var)

            if parameter_is_ref:
                fwd_args.append(func_arg_var)
            else:
                fwd_args.append(strip_reference(func_arg_var))
            reverse_adj_args.append(reverse_arg_var)

        det_args = adj.deterministic.call_args(func, bound_args)
        replay_det_args = adj.deterministic.replay_call_args(func, bound_args, det_args)

        if return_type is None:
            # handles expression (zero output) functions, e.g.: void do_something();
            forward_call = f"{func.namespace}{func_name}({adj.format_forward_call_args(fwd_args + det_args, use_initializer_list)});"
            replay_call = forward_call
            if func.custom_replay_func is not None or func.replay_snippet is not None:
                replay_call = f"{func.namespace}replay_{func_name}({adj.format_forward_call_args(fwd_args + replay_det_args, use_initializer_list)});"

        elif not isinstance(return_type, Sequence) or len(return_type) == 1:
            # handle simple function (one output)
            forward_call = f"var_{output} = {func.namespace}{func_name}({adj.format_forward_call_args(fwd_args + det_args, use_initializer_list)});"
            replay_call = forward_call
            if func.custom_replay_func is not None:
                replay_call = f"var_{output} = {func.namespace}replay_{func_name}({adj.format_forward_call_args(fwd_args + replay_det_args, use_initializer_list)});"

        else:
            # handle multiple value functions
            forward_call = f"{func.namespace}{func_name}({adj.format_forward_call_args(fwd_args + det_args + output, use_initializer_list)});"
            replay_call = forward_call

        forward_call, replay_call = adj.deterministic.wrap_unintercepted_side_effect_atomic(
            func, forward_call, replay_call, return_value_used=return_value_used
        )

        if func.skip_replay:
            adj.add_forward(forward_call, replay="// " + replay_call)
        else:
            adj.add_forward(forward_call, replay=replay_call)

        # Skip reverse call generation for functions that use warp.grad() - they don't have
        # meaningful adjoints (the gradient of a gradient call is not supported).
        skip_reverse = not func.is_builtin() and func.adj.uses_grad_call
        # Higher-order built-ins pass callable args to native adjoint helpers.
        # Only generate the reverse call when those callables have adjoints.
        has_nondifferentiable_callable_arg = any(
            isinstance(arg, warp._src.context.Function) and not arg.is_differentiable for arg in func_args
        )

        if func.is_differentiable and func_args and not skip_reverse and not has_nondifferentiable_callable_arg:
            # Ref-param functions are not automatically differentiable: a silent
            # skip would produce wrong gradients, so raise instead.
            if func_has_ref_params and adj.used_by_backward_kernel and not adj.has_manual_ref_adjoint(func):
                raise WarpCodegenError(
                    f"Cannot call '{func.key}' with wp.ref[T] parameters from a backward-enabled "
                    "kernel. Use enable_backward=False on the kernel, or switch to @wp.func_native "
                    "with adj_snippet for a manually written adjoint."
                )
            adj_args = tuple(reverse_adj_args)
            reverse_has_output_args = (
                func.require_original_output_arg or len(output_list) > 1
            ) and func.custom_grad_func is None
            reverse_extra_args = adj.deterministic.reverse_call_args(func, bound_args, det_args)
            arg_str = adj.format_reverse_call_args(
                fwd_args,
                adj_args,
                output_list,
                use_initializer_list,
                extra_args=reverse_extra_args,
                has_output_args=reverse_has_output_args,
                require_original_output_arg=func.require_original_output_arg,
            )
            if arg_str is not None:
                if func.lto_dispatch_func is not None:
                    adj_func_name = compute_type_str(func.native_func, template_args)
                else:
                    adj_func_name = func.native_func
                reverse_call = f"{func.namespace}adj_{adj_func_name}({arg_str});"
                if adj.deterministic.enabled and func.is_builtin() and func.key == "address":
                    reverse_call = adj.deterministic.adjoint_address_call(fwd_args, output_list, reverse_call)
                adj.add_reverse(reverse_call)
                for statement in reversed(reverse_prelude):
                    adj.add_reverse(statement)

        # update our smem roofline requirements based on any
        # shared memory required by the dependent function call
        if not func.is_builtin():
            adj.alloc_shared_extra(func.adj.get_total_required_shared() + extra_shared_memory)
        else:
            adj.alloc_shared_extra(extra_shared_memory)

        return return_value(output)

    def add_builtin_call(adj, func_name, args, min_outputs=None, return_value_used=True):
        func = warp._src.context.builtin_functions[func_name]
        return adj.add_call(func, args, {}, {}, min_outputs=min_outputs, return_value_used=return_value_used)

    def add_grad_call(adj, func, args, kwargs):
        """Generate code for calling the gradient of a function via warp.grad().

        This generates inline code in the forward pass that:
        1. Creates local variables for adjoint inputs (initialized to 0)
        2. Creates local variable(s) for adjoint output (initialized to 1.0)
        3. Calls the function's auto-generated adjoint
        4. Returns the adjoint inputs as a tuple (or single value if 1 input)

        Note:
            This gradient call is forward-only and does NOT participate in automatic
            differentiation.
        """
        func, bound_args, _ = adj.resolve_call(func, args, kwargs)

        if not func.is_differentiable:
            raise WarpCodegenError(f"Cannot compute gradient of non-differentiable function '{func.key}'")

        # Mark this function as using grad() - it will be generated in a later pass
        # and won't have its own adjoint generated.
        # Exception: Custom gradient functions (custom_reverse_mode=True) ARE adjoints,
        # so they don't need their own adjoints and should be generated in the adjoint pass.
        if not adj.custom_reverse_mode:
            adj.uses_grad_call = True

        # Warn if this kernel has backward enabled, since the gradient call
        # is forward-only and won't participate in automatic differentiation.
        if adj.used_by_backward_kernel:
            log_warning(
                f'grad() call for function "{func.key}" is used in a kernel with enable_backward=True. The gradient call does NOT participate in automatic differentiation - gradients will not flow through this call in the backward pass.'
            )

        # Ensure the function is built so its adjoint code exists.
        if not func.is_builtin():
            # Force adjoint code generation for the function.
            func.adj.force_adjoint_codegen = True

            # Build the function if not already built
            if adj.builder is None:
                func.build(None, adj.builder_options)
            elif func not in adj.builder.functions:
                adj.builder.build_function(func)

        # Get return type
        bound_arg_types = {k: get_arg_type(v) for k, v in bound_args.items()}
        bound_arg_values = {k: get_arg_value(v) for k, v in bound_args.items()}
        return_type = func.value_func(
            {k: strip_reference(v) for k, v in bound_arg_types.items()},
            bound_arg_values,
        )

        if return_type is None:
            raise WarpCodegenError(f"Cannot compute gradient of void function '{func.key}'")

        # Function parameters are specialization inputs, not native function
        # arguments, and their adjoints are not representable.
        input_types = {
            name: typ for name, typ in func.input_types.items() if not warp._src.types.is_warp_function_annotation(typ)
        }

        # Load input arguments into variables
        fwd_args_loaded = [adj.load(bound_args[name]) for name in input_types.keys()]

        # Determine return type structure
        if isinstance(return_type, Sequence):
            return_types = list(return_type)
        else:
            return_types = [return_type]

        # Determine if we need to include the forward output in the adjoint call.
        # Builtins with require_original_output_arg=True or multiple outputs need it.
        # User functions don't include it (unless multiple outputs).
        include_fwd_output = func.require_original_output_arg or len(return_types) > 1

        # Call the forward function first to get the output value(s)
        # (needed for adjoint functions that require the original output)
        fwd_output_vars = []
        if include_fwd_output:
            if len(return_types) == 1:
                # Single return value
                fwd_out = adj.add_var(return_types[0])
                fwd_args_str = ", ".join(arg.emit() for arg in fwd_args_loaded)
                adj.add_forward(f"{fwd_out.emit()} = {func.namespace}{func.native_func}({fwd_args_str});")
                fwd_output_vars.append(fwd_out)
            else:
                # Multiple return values
                for ret_type in return_types:
                    fwd_out = adj.add_var(ret_type)
                    fwd_output_vars.append(fwd_out)
                fwd_args_str = ", ".join(arg.emit() for arg in fwd_args_loaded)
                out_args_str = ", ".join(v.emit() for v in fwd_output_vars)
                adj.add_forward(f"{func.namespace}{func.native_func}({fwd_args_str}, {out_args_str});")

        # Create local variables for adjoint inputs initialized to 0
        adj_input_vars = []
        for typ in input_types.values():
            local_var = adj.add_var(typ)
            adj.add_forward(f"{local_var.emit()} = {{}};")  # Zero-initialize
            adj_input_vars.append(local_var)

        # Create local variable(s) for adjoint return initialized to 1.0
        adj_ret_vars = []
        for ret_type in return_types:
            local_var = adj.add_var(ret_type)
            ctype = Var.type_to_ctype(ret_type)
            adj.add_forward(f"{local_var.emit()} = {ctype}(1);")
            adj_ret_vars.append(local_var)

        # Build arguments for calling the adjoint function.
        # Signature varies:
        # - Builtins: adj_func(primal_inputs..., primal_output, adj_inputs..., adj_ret)
        # - User funcs: adj_func(primal_inputs..., adj_inputs..., adj_ret)
        adj_call_args = []
        for arg in fwd_args_loaded:
            adj_call_args.append(arg.emit())
        if include_fwd_output:
            for fwd_out in fwd_output_vars:
                adj_call_args.append(fwd_out.emit())
        for local_var in adj_input_vars:
            adj_call_args.append(local_var.emit())
        for local_var in adj_ret_vars:
            adj_call_args.append(local_var.emit())

        adj_call_args_str = ", ".join(adj_call_args)

        # Call the function's adjoint (accumulates into local vars)
        adj.add_forward(f"{func.namespace}adj_{func.native_func}({adj_call_args_str});")

        # Return the accumulated adjoint(s)
        if len(adj_input_vars) == 1:
            return adj_input_vars[0]
        else:
            # Return as a tuple
            result_var = adj.add_var(
                warp._src.types.tuple_t(
                    types=tuple(input_types.values()),
                    values=(None,) * len(input_types),
                )
            )
            ctypes_str = ", ".join(Var.type_to_ctype(t) for t in input_types.values())
            adj_returns = ", ".join(v.emit() for v in adj_input_vars)
            adj.add_forward(f"{result_var.emit()} = wp::tuple_t<{ctypes_str}>({adj_returns});")
            return result_var

    def add_return(adj, var):
        if var is None or len(var) == 0:
            # Rewritten to `continue;` for CUDA grid-stride kernels by codegen_func_forward().
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
                # Non-owner tile adjoints are alias handles - their grad pointer
                # references storage owned by another tile (typically a
                # loop-invariant accumulator). Resetting here would corrupt that
                # storage and null the alias pointers in this local handle.
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
            if is_tile(i.type):
                if i.type.owner:
                    reverse.append(f"{i.emit_adj()}.grad_zero();")
                # Non-owner tile adjoints are alias handles - their grad pointer
                # references storage owned by another tile (typically a
                # loop-invariant accumulator). Resetting here would corrupt that
                # storage and null the alias pointers in this local handle.
            else:
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
        if is_func_native and "return" in adj.arg_types:
            ret_type = adj.arg_types["return"]
            if not (type_is_value(ret_type) or is_array(ret_type)):
                raise WarpCodegenError(
                    f"Native function '{adj.fun_name}' has unsupported return type `{ret_type}`. "
                    f"Expected a Warp scalar, vector, matrix, quaternion, array, or fixedarray type."
                )
            var = Var(label="return_type", type=ret_type)
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

        # save the symbol map from before the conditional
        symbols_prev = adj.symbols.copy()

        # eval the 'if' body as `if (cond)`
        adj.begin_if(cond)

        for stmt in node.body:
            adj.eval(stmt)

        adj.end_if(cond)

        # capture the symbol versions produced by the 'if' branch, then restore the
        # pre-conditional symbol map so the 'else' branch is lowered independently.
        #
        # This matters for variables that are first assigned inside the 'if' branch
        # (i.e. they have no version prior to the conditional): lowering the 'else'
        # branch on top of the 'if' branch's symbol map would let those branch-local
        # versions be referenced from inside the 'else' branch (e.g. as the "previous"
        # operand of a nested phi/select), even though they are never assigned on the
        # 'else' path. That produces selects over locals that are only defined on the
        # opposite path, which miscompiles on CUDA under register pressure and
        # corrupts the merged value.
        symbols_if = adj.symbols
        adj.symbols = symbols_prev.copy()

        # evaluate 'else' statement as if (!cond)
        if len(node.orelse) > 0:
            adj.begin_else(cond)

            for stmt in node.orelse:
                adj.eval(stmt)

            adj.end_else(cond)

        symbols_else = adj.symbols

        # detect symbols with conflicting definitions across the two branches and
        # resolve them with a phi (select) function based on `cond`
        merged = symbols_prev.copy()
        for sym in dict.fromkeys((*symbols_if.keys(), *symbols_else.keys())):
            prev_var = symbols_prev.get(sym)
            if_var = symbols_if.get(sym, prev_var)
            else_var = symbols_else.get(sym, prev_var)

            if if_var is else_var:
                # not modified relative to the shared pre-conditional version
                if if_var is not None:
                    merged[sym] = if_var
            elif if_var is None:
                # only assigned on the 'else' path
                merged[sym] = else_var
            elif else_var is None:
                # only assigned on the 'if' path
                merged[sym] = if_var
            else:
                # insert a phi function that selects the 'if'/'else' version based on cond
                merged[sym] = adj.add_builtin_call("where", [cond, if_var, else_var])

        adj.symbols = merged

    def emit_IfExp(adj, node):
        cond = adj.eval(node.test)

        if cond.constant is not None:
            return adj.eval(node.body) if cond.constant else adj.eval(node.orelse)

        adj.begin_if(cond)
        body = adj.eval(node.body)
        body = adj.load(body)
        adj.end_if(cond)

        adj.begin_else(cond)
        orelse = adj.eval(node.orelse)
        orelse = adj.load(orelse)
        adj.end_else(cond)

        return adj.add_builtin_call("where", [cond, body, orelse])

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
            is_and = True
        elif isinstance(op, ast.Or):
            is_and = False
        else:
            raise WarpCodegenKeyError(f"Op {op} is not supported")

        # Short-circuit evaluation: only evaluate subsequent operands
        # if the result so far permits it (true for 'and', false for 'or').
        output = adj.add_var(builtins.bool)
        first = adj.eval(node.values[0])
        first = adj.load(first)
        adj.add_forward(f"{output.emit()} = {first.emit()};")

        for expr in node.values[1:]:
            # Guard: only evaluate next operand if short-circuit condition holds
            if is_and:
                adj.add_forward(f"if ({output.emit()}) {{")
                adj.add_reverse("}")
            else:
                adj.add_forward(f"if (!{output.emit()}) {{")
                adj.add_reverse("}")
            adj.indent()

            val = adj.eval(expr)
            val = adj.load(val)
            op_str = "&&" if is_and else "||"
            adj.add_forward(f"{output.emit()} = {output.emit()} {op_str} {val.emit()};")

            adj.dedent()
            adj.add_forward("}")
            if is_and:
                adj.add_reverse(f"if ({output.emit()}) {{")
            else:
                adj.add_reverse(f"if (!{output.emit()}) {{")

        return output

    def emit_Name(adj, node):
        # a static expression that resolved to a Warp function is rewritten to an `__warp_func__`
        # Name node carrying the function on `warp_func` (see StaticExpressionReplacer). Return it
        # directly so it can be bound to a local, e.g. `func = wp.static(...)`.
        if hasattr(node, "warp_func"):
            return node.warp_func

        # lookup symbol, if it has already been assigned to a variable then return the existing mapping
        if node.id in adj.symbols:
            return adj.symbols[node.id]

        obj = adj.resolve_external_reference(node.id)

        if obj is None:
            raise WarpCodegenKeyError("Referencing undefined symbol: " + str(node.id))

        if warp._src.types.is_value(obj):
            # evaluate constant
            out = adj.add_constant(obj)
            adj.symbols[node.id] = out
            return out

        # the named object is either a function, class name, or module
        # pass it back to the caller for processing
        if isinstance(obj, warp._src.context.Function):
            return obj
        if isinstance(obj, type):
            return obj
        if isinstance(obj, Struct):
            if adj.builder is not None:
                adj.builder.build_struct_recursive(obj)
            return obj
        if isinstance(obj, types.ModuleType):
            return obj

        raise TypeError(f"Invalid external reference type: {type(obj)}")

    @staticmethod
    def resolve_type_attribute(var_type: type, attr: str):
        if type_is_value(var_type):
            if attr == "dtype":
                return type_scalar_type(var_type)
            elif attr == "length":
                return type_size(var_type)

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

    def transform_component(adj, component):
        if len(component) != 1:
            raise WarpCodegenAttributeError(f"Transform attribute must be single character, got .{component}")

        if component not in ("p", "q"):
            raise WarpCodegenAttributeError(f"Attribute for transformation must be either 'p' or 'q', got {component}")

        return component

    @staticmethod
    def is_differentiable_value_type(var_type):
        # checks that the argument type is a value type (i.e, not an array)
        # possibly holding differentiable values (for which gradients must be accumulated)
        return type_scalar_type(var_type) in float_types or isinstance(var_type, Struct)

    def emit_Attribute(adj, node, aggregate=None):
        if hasattr(node, "is_adjoint"):
            node.value.is_adjoint = True

        if aggregate is None:
            aggregate = adj.eval(node.value)

        try:
            if isinstance(aggregate, Var) and aggregate.constant is not None:
                # this case may occur when the attribute is a constant, e.g.: `IntEnum.A.value`
                return aggregate

            if isinstance(aggregate, types.ModuleType) or isinstance(aggregate, type):
                out = getattr(aggregate, node.attr)

                if isinstance(out, (enum.IntEnum, enum.IntFlag)):
                    return adj.add_constant(int(out))

                if warp._src.types.is_value(out):
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
                out = adj.add_builtin_call("extract", [aggregate, index])

                if origin := adj.reference_origin_for_var(aggregate):
                    out.ref_origin = origin.extend_index(index)

                return out

            elif type_is_transformation(aggregate_type):
                component = adj.transform_component(node.attr)

                if component == "p":
                    out = adj.add_builtin_call("transform_get_translation", [aggregate])
                else:
                    out = adj.add_builtin_call("transform_get_rotation", [aggregate])

                if origin := adj.reference_origin_for_var(aggregate):
                    out.ref_origin = origin.extend_field(component)

                return out

            else:
                attr_var = aggregate_type.vars[node.attr]

                # represent pointer types as uint64
                if isinstance(attr_var.type, pointer_t):
                    cast = f"({Var.dtype_to_ctype(uint64)}*)"
                    adj_cast = f"({Var.dtype_to_ctype(attr_var.type.dtype)}*)"
                    attr_type = Reference(uint64)
                else:
                    cast = ""
                    adj_cast = ""
                    attr_type = Reference(attr_var.type)

                attr = adj.add_var(attr_type)
                # Array descriptor fields, such as ``shape``, must be taken from
                # the descriptor Var itself; replaying an origin may reconstruct
                # an rvalue view like ``wp::view(...).shape``.
                origin = None if is_array(aggregate_type) else adj.reference_origin_for_var(aggregate)

                if origin is not None:
                    attr.ref_origin = origin.extend_field(
                        attr_var.label,
                        pointer_cast=cast,
                        adjoint_pointer_cast=adj_cast,
                    )
                    adj.add_forward(f"{attr.emit()} = {attr.ref_origin.emit_pointer()};")
                elif is_reference(aggregate.type):
                    adj.add_forward(f"{attr.emit()} = {cast}&({aggregate.emit()}->{attr_var.label});")
                else:
                    adj.add_forward(f"{attr.emit()} = {cast}&({aggregate.emit()}.{attr_var.label});")

                adj.deterministic.propagate_attribute_reference(attr, aggregate, attr_var, attr_type)

                if adj.is_differentiable_value_type(strip_reference(attr_type)):
                    adj.add_reverse(f"{aggregate.emit_adj()}.{attr_var.label} += {adj_cast}{attr.emit_adj()};")
                else:
                    adj.add_reverse(f"{aggregate.emit_adj()}.{attr_var.label} = {adj_cast}{attr.emit_adj()};")

                return attr

        except (KeyError, AttributeError) as e:
            # Try resolving as type attribute
            aggregate_type = strip_reference(aggregate.type) if isinstance(aggregate, Var) else aggregate

            type_attribute = adj.resolve_type_attribute(aggregate_type, node.attr)
            if type_attribute is not None:
                return type_attribute

            if isinstance(aggregate, Var):
                node_name = get_node_name_safe(node.value)
                raise WarpCodegenAttributeError(
                    f"Error, `{node.attr}` is not an attribute of '{node_name}' ({type_repr(aggregate.type)})"
                ) from e
            raise WarpCodegenAttributeError(f"Error, `{node.attr}` is not an attribute of '{aggregate}'") from e

    def emit_Assert(adj, node):
        # eval condition
        cond = adj.eval(node.test)
        cond = adj.load(cond)

        source_segment = ast.get_source_segment(adj.source, node)
        # If a message was provided with the assert, " marks can interfere with the generated code
        escaped_segment = source_segment.replace('"', '\\"')

        adj.add_forward(f'assert(((void)"{escaped_segment}",{cond.emit()}));')

    def emit_Constant(adj, node):
        if node.value is None:
            raise WarpCodegenTypeError("None type unsupported")
        else:
            return adj.add_constant(node.value)

    def emit_BinOp(adj, node):
        # evaluate binary operator arguments

        if adj.builder_options.get("verify_autograd_array_access", False):
            # array overwrite tracking: in-place operators are a special case
            # x[tid] = x[tid] + 1 is a read followed by a write, but we only want to record the write
            # so we save the current arg read flags and restore them after lhs eval
            is_read_states = []
            for arg in adj.args:
                is_read_states.append(arg.is_read)

        # evaluate lhs binary operator argument
        left = adj.eval(node.left)

        if adj.builder_options.get("verify_autograd_array_access", False):
            # restore arg read flags
            for i, arg in enumerate(adj.args):
                arg.is_read = is_read_states[i]

        # evaluate rhs binary operator argument
        right = adj.eval(node.right)

        name = builtin_operators[type(node.op)]

        try:
            # Check if there is any user-defined overload for this operator
            user_func = adj.resolve_external_reference(name)
            if isinstance(user_func, warp._src.context.Function):
                return adj.add_call(user_func, (left, right), {}, {})
        except WarpCodegenError:
            pass

        return adj.add_builtin_call(name, [left, right])

    def emit_UnaryOp(adj, node):
        # evaluate unary op arguments
        arg = adj.eval(node.operand)

        # evaluate expression to a compile-time constant if arg is a constant
        if isinstance(arg.constant, (builtins.int, builtins.float)):
            if isinstance(node.op, ast.USub):
                # When the operand is a literal constant (ast.Constant), the
                # arg Var was just created and is not referenced by any symbol,
                # so we can safely repurpose it with the negated value to avoid
                # emitting a dead variable with a potentially truncating
                # constant initializer (e.g. int32 assigned a 64-bit literal).
                if isinstance(node.operand, ast.Constant):
                    arg.constant = -arg.constant
                    return arg
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
                if not adj.custom_reverse_mode:
                    lineno = adj.lineno + adj.fun_lineno
                    line = adj.source_lines[adj.lineno]
                    msg = f'Warning: detected mutated variable {sym} during a dynamic for-loop in function "{adj.fun_name}" at {adj.filename}:{lineno}: this may not be a differentiable operation.\n{line}'
                    log_debug(msg)

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

        return warp._src.types.is_int(obj), obj

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
        arg_is_numeric, arg_values = zip(*range_args, strict=True)

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

            max_unroll = adj.builder_options.get("max_unroll", 16)

            ok_to_unroll = True

            contains_static = False
            for node in ast.walk(loop):
                if not isinstance(node, ast.Call):
                    continue
                try:
                    func, _ = adj.resolve_static_expression(node.func, eval_types=False)
                except Exception:
                    continue
                if adj.is_static_expression(func):
                    contains_static = True
                    break

            # Always unroll if the loop contains static expressions
            if contains_static:
                # Forced unrolling for loops with static expressions regardless of max_unroll
                if max_iters > max_unroll:
                    log_debug(
                        f"Notice: Forcing unroll of loop with {max_iters} iterations because it contains wp.static expressions."
                    )
                return range(start, end, step)

            # Apply max_unroll check only for regular loops (no static expressions)
            if max_iters > max_unroll:
                log_debug(
                    f"Warning: fixed-size loop count of {max_iters} is larger than the module 'max_unroll' limit of {max_unroll}, will generate dynamic loop."
                )
                ok_to_unroll = False

            elif adj.contains_break(loop.body):
                log_debug("Warning: 'break' or 'continue' found in loop body, will generate dynamic loop.")
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
        if isinstance(node.value, ast.Call):
            return adj.emit_Call(node.value, return_value_used=False)
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

        if isinstance(expr, (type, Struct, Var, warp._src.context.Function)):
            return expr

        if isinstance(expr, (enum.IntEnum, enum.IntFlag)):
            return adj.add_constant(int(expr))

        return adj.add_constant(expr)

    def eval_const_slice_component(adj, node) -> int:
        """Evaluate a slice component, returning its constant value."""
        var = adj.eval(node)

        if not isinstance(var, Var) or var.constant is None:
            raise WarpCodegenValueError("Slice component must be a compile-time constant.")

        if not isinstance(var.constant, int):
            raise WarpCodegenTypeError("Slice component must be an integer.")

        return var.constant

    def eval_const_slice(adj, node, length) -> tuple[int, int, int]:
        """Evaluate a slice, returning its constant components."""
        step = 1 if node.step is None else adj.eval_const_slice_component(node.step)
        if step == 0:
            raise WarpCodegenValueError("Slice step cannot be zero.")

        if node.lower is None:
            start = length - 1 if step < 0 else 0
        else:
            start = adj.eval_const_slice_component(node.lower)
            if length is not None:
                start = min(max(start, -length), length)
                start = start + length if start < 0 else start

        if node.upper is None:
            stop = -1 if step < 0 else length
        else:
            stop = adj.eval_const_slice_component(node.upper)
            if length is not None:
                stop = min(max(stop, -length), length)
                stop = stop + length if stop < 0 else stop

        return (start, stop, step)

    def unpack_starred(adj, node):
        """Unpack a starred expression into individual elements."""
        value = node.value

        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Slice):
            target = adj.eval(value.value)
            target_type = strip_reference(target.type)

            if hasattr(target_type, "_wp_generic_type_hint_"):
                # Compound slicing.
                length = target_type._shape_[0]
                bounds = adj.eval_const_slice(value.slice, length)
            elif is_array(target_type):
                # Array slicing.
                if target_type.ndim != 1:
                    raise WarpCodegenValueError(
                        f"Starred expressions with slices are only supported for 1D arrays, got {target_type.ndim}D array."
                    )

                length = None
                bounds = adj.eval_const_slice(value.slice, length)

                # Arrays don't have a length known at compile-time so the slice bounds need to be fully defined:
                # - The upper bound needs to be provided.
                # - Only non-negative indices can be passed.

                if value.slice.upper is None:
                    raise WarpCodegenValueError(
                        "Starred expression on arrays requires explicit upper bound (e.g., *array[0:3], not *array[0:])."
                    )

                if bounds[0] < 0 or bounds[1] < 0:
                    raise WarpCodegenValueError("Starred expression slice bounds cannot be negative for arrays.")

                if bounds[2] < 0:
                    raise WarpCodegenValueError("Starred expression slice step cannot be negative for arrays.")
            else:
                raise WarpCodegenTypeError(
                    f"Starred expressions with slices are only supported for arrays, vectors, quaternions, and matrices. "
                    f"Got {type_repr(target_type)}."
                )
        else:
            target = adj.eval(value)
            target_type = strip_reference(target.type)

            if hasattr(target_type, "_wp_generic_type_hint_"):
                # Whole composite type.
                bounds = (0, target_type._shape_[0], 1)
            elif is_array(target_type):
                raise WarpCodegenTypeError(
                    "Starred expressions apply to arrays only if they are sliced with bounds known at compile-time."
                )
            else:
                raise WarpCodegenTypeError(
                    f"Starred expressions are only supported for arrays (with slice), vectors, quaternions, and matrices. "
                    f"Got {type_repr(target_type)}."
                )

        if is_array(target_type):
            builtin_name = "address"
            if adj.builder_options.get("verify_autograd_array_access", False):
                target.mark_read()
        else:
            builtin_name = "extract"

        elements = tuple(adj.add_builtin_call(builtin_name, (target, adj.add_constant(i))) for i in range(*bounds))
        if not elements:
            raise WarpCodegenError("Starred expression results in empty sequence.")

        return elements

    def emit_address_of(adj, node):
        """Handle wp.address_of(expr) -> wp.uint64 as a codegen special form."""
        if len(node.args) != 1 or node.keywords:
            raise WarpCodegenError("wp.address_of() takes exactly one positional argument")

        arg_node = node.args[0]
        var = adj.emit_addressable_reference(arg_node, purpose="wp.address_of()")

        ctype_uint64 = Var.type_to_ctype(warp.uint64)  # "wp::uint64"
        addr_expr = f"({ctype_uint64})({var.emit()})"

        result = adj.add_var(warp.uint64)
        adj.add_forward(f"{result.emit()} = {addr_expr};")
        return result

    def emit_Call(adj, node, return_value_used=True):
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

        # wp.address_of() is a codegen-only special form; intercept it here
        # regardless of how it was resolved (via module path or direct ref).
        if func is warp.address_of:
            return adj.emit_address_of(node)

        if adj.is_static_expression(func):
            # try to evaluate wp.static() expressions
            obj, code = adj.evaluate_static_expression(node)
            if obj is not None:
                adj.deferred_static_expressions.append((code, obj))
                if isinstance(obj, warp._src.context.Function):
                    # special handling for wp.static() evaluating to a function
                    return obj
                else:
                    out = adj.add_constant(obj)
                    return out

        # Check if this is a warp.grad() call
        if adj.is_grad_expression(func):
            # warp.grad(some_func) should return a GradWrapper
            if len(node.args) != 1 or node.keywords:
                raise WarpCodegenError("grad() expects exactly one function argument")

            target_func = adj.resolve_arg(node.args[0])
            if not isinstance(target_func, warp._src.context.Function):
                raise WarpCodegenError(f"grad() expects a Warp function, got {type(target_func).__name__}")

            return warp._src.context.GradWrapper(target_func)

        # Check if we're calling a GradWrapper (result of warp.grad(func))
        if isinstance(func, warp._src.context.GradWrapper):
            # Evaluate arguments and generate gradient call
            args = tuple(adj.resolve_arg(x) for x in node.args)
            kwargs = {x.arg: adj.resolve_arg(x.value) for x in node.keywords}
            return adj.add_grad_call(func.func, args, kwargs)

        type_args = {}

        if len(path) > 0 and not isinstance(func, warp._src.context.Function):
            attr = path[-1]
            caller = func
            func = None

            # Handle module callers: check if attribute exists on the module
            if isinstance(caller, types.ModuleType):
                if hasattr(caller, attr):
                    # Attribute exists on the module - use it
                    func = getattr(caller, attr)
                    if not isinstance(func, warp._src.context.Function):
                        # It's not a Function, might be a type - let subsequent logic handle it
                        caller = func
                        func = None
                elif caller is warp and attr in warp._src.context.builtin_functions:
                    # Fallback: for the warp module, check builtin_functions
                    # (builtins like tid() are not actual attributes of the warp module)
                    func = warp._src.context.builtin_functions[attr]
                else:
                    # Attribute doesn't exist on this module
                    raise WarpCodegenAttributeError(
                        f"Could not find function {'.'.join(path)} as a built-in or user-defined function. "
                        "Note that user functions must be annotated with a @wp.func decorator to be called from a kernel."
                    )

            # try and lookup function name in builtins (e.g.: using `dot` directly without wp prefix)
            if func is None and attr in warp._src.context.builtin_functions:
                func = warp._src.context.builtin_functions[attr]

            # vector class type e.g.: wp.vec3f constructor
            if func is None and hasattr(caller, "_wp_generic_type_str_"):
                func = warp._src.context.builtin_functions.get(caller._wp_constructor_)

            # scalar class type e.g.: wp.int8 constructor
            if func is None and hasattr(caller, "__name__") and caller.__name__ in warp._src.context.builtin_functions:
                func = warp._src.context.builtin_functions.get(caller.__name__)

            # struct constructor
            if func is None and isinstance(caller, Struct):
                if adj.builder is not None:
                    adj.builder.build_struct_recursive(caller)
                if node.args or node.keywords:
                    func = caller.value_constructor
                else:
                    func = caller.default_constructor

            # lambda function
            if func is None and getattr(caller, "__name__", None) == "<lambda>":
                raise NotImplementedError("Lambda expressions are not yet supported")

            if hasattr(caller, "_wp_type_args_"):
                type_args = caller._wp_type_args_

            if func is None:
                raise WarpCodegenError(
                    f"Could not find function {'.'.join(path)} as a built-in or user-defined function. Note that user functions must be annotated with a @wp.func decorator to be called from a kernel."
                )

        # get expected return count, e.g.: for multi-assignment
        min_outputs = None
        if hasattr(node, "expects"):
            min_outputs = node.expects

        # Evaluate positional arguments.
        args = []
        arg_nodes = []
        for x in node.args:
            if isinstance(x, ast.Starred):
                # Handle starred expressions by unpacking them into multiple arguments.
                unpacked = adj.unpack_starred(x)
                args.extend(unpacked)
                arg_nodes.extend([None] * len(unpacked))
            else:
                args.append(adj.resolve_arg(x))
                arg_nodes.append(x)

        # Evaluate keyword arguments.
        kwargs = {x.arg: adj.resolve_arg(x.value) for x in node.keywords}
        kwarg_nodes = {x.arg: x.value for x in node.keywords}

        if adj.builder_options.get("verify_autograd_array_access", False):
            out, resolved_func, resolved_bound_args = adj.add_call(
                func,
                args,
                kwargs,
                type_args,
                min_outputs=min_outputs,
                return_value_used=return_value_used,
                arg_nodes=arg_nodes,
                kwarg_nodes=kwarg_nodes,
                return_resolved=True,
            )

            # update arg read/write states according to what happens to that arg in the called function
            if hasattr(resolved_func, "adj"):
                resolved_args_by_name = {arg.label: arg for arg in resolved_func.adj.args}
                for name, arg in resolved_bound_args.items():
                    if warp._src.types.is_warp_function_annotation(resolved_func.input_types.get(name)):
                        continue

                    resolved_arg = resolved_args_by_name.get(name)
                    if resolved_arg is None or not isinstance(arg, Var):
                        continue

                    if resolved_arg.is_write:
                        kernel_name = adj.fun_name
                        filename = adj.filename
                        lineno = adj.lineno + adj.fun_lineno
                        arg.mark_write(kernel_name=kernel_name, filename=filename, lineno=lineno)
                    if resolved_arg.is_read:
                        arg.mark_read()
        else:
            out = adj.add_call(
                func,
                args,
                kwargs,
                type_args,
                min_outputs=min_outputs,
                return_value_used=return_value_used,
                arg_nodes=arg_nodes,
                kwarg_nodes=kwarg_nodes,
            )

        return out

    def eval_indices(adj, target_type, indices):
        nodes = indices
        if type_is_composite(target_type):
            indices = []
            for dim, node in enumerate(nodes):
                if isinstance(node, ast.Slice):
                    # In the context of slicing a vec/mat type, indices are expected
                    # to be compile-time constants, hence we can infer the actual slice
                    # bounds also at compile-time.
                    length = target_type._shape_[dim]
                    bounds = adj.eval_const_slice(node, length)
                    slice = adj.add_builtin_call("slice", bounds)
                    indices.append(slice)
                else:
                    indices.append(adj.eval(node))

            return tuple(indices)
        else:
            return tuple(adj.eval(x) for x in nodes)

    def emit_indexing(adj, target, indices):
        target_type = strip_reference(target.type)
        indices = adj.eval_indices(target_type, indices)

        if is_array(target_type):
            if len(indices) == target_type.ndim and all(
                warp._src.types.type_is_int(strip_reference(x.type)) for x in indices
            ):
                # handles array loads (where each dimension has an index specified)
                out = adj.add_builtin_call("address", [target, *indices])
                if origin := adj.reference_origin_for_var(target):
                    out.ref_origin = origin.extend_array(indices)

                if adj.builder_options.get("verify_autograd_array_access", False):
                    target.mark_read()

            else:
                # Keep the original source-level indices for deterministic view
                # tracking before the plain array path rewrites integer indices
                # into slice arguments for the native view builtin.
                view_indices = tuple(indices)

                if warp._src.types.matches_array_class(target_type, warp._src.types.array):
                    # In order to reduce the number of overloads needed in the C
                    # implementation to support combinations of int/slice indices,
                    # we convert all integer indices into slices, and set their
                    # step to 0 if they are representing an integer index.
                    new_indices = []
                    for idx in indices:
                        if not warp._src.types.is_slice(strip_reference(idx.type)):
                            new_idx = adj.add_builtin_call("slice", (idx, idx, 0))
                            new_indices.append(new_idx)
                        else:
                            new_indices.append(idx)

                    indices = new_indices

                # handles array views (fewer indices than dimensions)
                out = adj.add_builtin_call("view", [target, *indices])
                adj.deterministic.track_view(out, target, view_indices)
                if origin := adj.reference_origin_for_var(target):
                    out.ref_origin = origin.extend_view(indices)

                if adj.builder_options.get("verify_autograd_array_access", False):
                    # store reference to target Var to propagate downstream read/write state back to root arg Var
                    out.parent = target

                    # view arg inherits target Var's read/write states
                    out.is_read = target.is_read
                    out.is_write = target.is_write

        elif is_tile(target_type):
            if len(indices) >= len(target_type.shape):  # equality for scalars, inequality for composite types
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
            if origin := adj.reference_origin_for_var(target):
                index_types_are_int = all(warp._src.types.type_is_int(strip_reference(x.type)) for x in indices)
                if type_is_matrix(target_type) and len(indices) in (1, 2) and index_types_are_int:
                    out.ref_origin = origin.extend_index(indices)
                elif (
                    (
                        type_is_vector(target_type)
                        or type_is_quaternion(target_type)
                        or type_is_transformation(target_type)
                    )
                    and len(indices) == 1
                    and index_types_are_int
                ):
                    out.ref_origin = origin.extend_index(indices[0])

        return out

    # from a list of lists of indices, strip the first `count` indices
    @staticmethod
    def strip_indices(indices, count):
        dim = count
        while count > 0:
            ij = indices[0]
            indices = indices[1:]
            count -= len(ij)

        # report straddling like in `arr2d[0][1,2]` as a syntax error
        if count < 0:
            raise WarpCodegenError(
                f"Incorrect number of indices specified for array indexing, got {dim - count} indices for a {dim} dimensional array."
            )

        return indices

    def recurse_subscript(adj, node, indices):
        if isinstance(node, ast.Name):
            target = adj.eval(node)
            return target, indices

        if isinstance(node, ast.Subscript):
            if hasattr(node.value, "attr") and node.value.attr == "adjoint":
                return adj.eval(node), indices

            if isinstance(node.slice, ast.Tuple):
                ij = node.slice.elts
            else:
                ij = [node.slice]

            indices = [ij, *indices]  # prepend

            target, indices = adj.recurse_subscript(node.value, indices)

            target_type = strip_reference(target.type)
            if is_array(target_type):
                flat_indices = [i for ij in indices for i in ij]
                if len(flat_indices) > target_type.ndim:
                    target = adj.emit_indexing(target, flat_indices[: target_type.ndim])
                    indices = adj.strip_indices(indices, target_type.ndim)

            return target, indices

        target = adj.eval(node)
        return target, indices

    # returns the object being indexed, and the list of indices
    def eval_subscript(adj, node):
        target, indices = adj.recurse_subscript(node, [])
        flat_indices = [i for ij in indices for i in ij]
        return target, flat_indices

    def emit_Subscript(adj, node):
        if hasattr(node.value, "attr") and node.value.attr == "adjoint":
            # handle adjoint of a variable, i.e. wp.adjoint[var]
            node.slice.is_adjoint = True
            var = adj.eval(node.slice)
            var_name = var.label
            var = Var(f"adj_{var_name}", type=var.type, constant=None, prefix=False)
            adj.deterministic.mark_adjoint_target(var, var_name)
            return var

        target, indices = adj.eval_subscript(node)

        return adj.emit_indexing(target, indices)

    def emit_Slice(adj, node):
        start = SLICE_BEGIN if node.lower is None else adj.eval(node.lower)
        stop = SLICE_END if node.upper is None else adj.eval(node.upper)
        step = 1 if node.step is None else adj.eval(node.step)
        return adj.add_builtin_call("slice", (start, stop, step))

    def store_ref_param_value(adj, name, value, *, augmented=False):
        ref_var = adj.ref_params[name]
        value = adj.load(value)
        value_type = value.type if isinstance(value, Var) else None
        if value_type is not None and not types_equal(value_type, ref_var.type.value_type):
            if augmented:
                raise WarpCodegenTypeError(
                    f"Error, augmented assignment to ref parameter `{name}` ({ref_var.type.value_type}) "
                    f"produces different type ({value_type})"
                )
            raise WarpCodegenTypeError(
                f"Error, assigning to ref parameter '{name}' ({ref_var.type.value_type}) "
                f"with value of different type ({value_type})"
            )

        adj.add_forward(f"*{ref_var.emit()} = {value.emit()};")

    def emit_Assign(adj, node):
        if len(node.targets) != 1:
            raise WarpCodegenError("Assigning the same value to multiple variables is not supported")

        # Check if the rhs corresponds to an unsupported construct.
        # Tuples are supported in the context of assigning multiple variables
        # at once, but not for simple assignments like `x = (1, 2, 3)`.
        # Therefore, we need to catch this specific case here instead of
        # more generally in `adj.eval()`.
        if isinstance(node.value, ast.List):
            raise WarpCodegenError(
                "List constructs are not supported in kernels. Use vectors like `wp.vec3()` for small fixed-size collections, or `wp.zeros(shape=N, dtype=...)` for stack-allocated arrays."
            )

        lhs = node.targets[0]

        if isinstance(lhs, ast.Tuple) and isinstance(node.value, ast.Call):
            # record the expected number of outputs on the node
            # we do this so we can decide which function to
            # call based on the number of expected outputs
            node.value.expects = len(lhs.elts)

        # evaluate rhs
        if isinstance(lhs, ast.Tuple) and isinstance(node.value, ast.Tuple):
            rhs = [adj.eval(v) for v in node.value.elts]
        else:
            rhs = adj.eval(node.value)

        # handle the case where we are assigning multiple output variables
        if isinstance(lhs, ast.Tuple):
            subtype = getattr(rhs, "type", None)

            if isinstance(subtype, warp._src.types.tuple_t):
                if len(rhs.type.types) != len(lhs.elts):
                    raise WarpCodegenError(
                        f"Invalid number of values to unpack (expected {len(lhs.elts)}, got {len(rhs.type.types)})."
                    )
                rhs = tuple(adj.add_builtin_call("extract", (rhs, adj.add_constant(i))) for i in range(len(lhs.elts)))

            names = []
            for v in lhs.elts:
                if isinstance(v, ast.Name):
                    names.append(v.id)
                else:
                    raise WarpCodegenError(
                        "Multiple return functions can only assign to simple variables, e.g.: x, y = func()"
                    )

            if len(names) != len(rhs):
                raise WarpCodegenError(
                    f"Multiple return functions need to receive all their output values, incorrect number of values to unpack (expected {len(rhs)}, got {len(names)})"
                )

            if any(name in adj.ref_params for name in names):
                # Snapshot reference-valued RHS entries before binding or
                # storing any ref target so tuple assignment keeps its
                # simultaneous semantics, e.g. `old, x = x, 2.0`.
                out = tuple(
                    adj.load(value) if isinstance(value, Var) and is_reference(value.type) else value for value in rhs
                )
            else:
                out = rhs

            for name, rhs in zip(names, out, strict=True):
                # A tuple-unpack target that is a wp.ref[T] parameter mutates
                # the referenced storage in place, like a scalar `name = rhs`
                # assignment, rather than rebinding the symbol to a value of a
                # different type (the parameter's type is Reference(T), not T).
                if name in adj.ref_params:
                    adj.store_ref_param_value(name, rhs)
                    continue

                if name in adj.symbols:
                    if isinstance(adj.symbols[name], warp._src.context.GradWrapper):
                        raise WarpCodegenError(
                            f"Cannot reassign local '{name}' after binding it to wp.grad(...). Warp treats "
                            "wp.grad(...) results as static function handles; use a different local variable "
                            "for the new value."
                        )

                    if not types_equal(rhs.type, adj.symbols[name].type):
                        raise WarpCodegenTypeError(
                            f"Error, assigning to existing symbol {name} ({adj.symbols[name].type}) with different type ({rhs.type})"
                        )

                adj.symbols[name] = rhs

        # handles the case where we are assigning to an array index (e.g.: arr[i] = 2.0)
        elif isinstance(lhs, ast.Subscript):
            if hasattr(lhs.value, "attr") and lhs.value.attr == "adjoint":
                # handle adjoint of a variable, i.e. wp.adjoint[var]
                lhs.slice.is_adjoint = True
                src_var = adj.eval(lhs.slice)
                var = Var(f"adj_{src_var.label}", type=src_var.type, constant=None, prefix=False)
                adj.add_forward(f"{var.emit()} = {rhs.emit()};")
                return

            # Fast path: array-rooted composite-component write -> single-slot
            # store with a correct adjoint (the legacy path's adjoint is a no-op).
            # wp.adjoint[var] is handled by the intercept above and never reaches here.
            if adj._try_lower_array_slot_write(lhs, rhs):
                return

            target, indices = adj.eval_subscript(lhs)
            target_type = strip_reference(target.type)
            indices = adj.eval_indices(target_type, indices)
            adj._store_subscript(lhs, target, indices, rhs)

        elif isinstance(lhs, ast.Name):
            # symbol name
            name = lhs.id

            # If this name is a wp.ref[T] parameter, emit a direct mutation
            # of the referenced storage rather than creating a new SSA variable.
            if name in adj.ref_params:
                adj.store_ref_param_value(name, rhs)
                return

            # handle GradWrapper specially - just store it in symbols for later use
            # this allows patterns like: func_handle = warp.grad(square); func_handle(x)
            if isinstance(rhs, warp._src.context.GradWrapper):
                adj.symbols[name] = rhs
                return

            # handle Warp functions specially - bind the name directly to the function
            # so later calls through the local resolve to it. This covers `f = my_func`,
            # `f = mod.func`, and `f = wp.static(...)` returning a function. Without this the
            # function would be routed through Var-shaped logic and miscompiled.
            if isinstance(rhs, warp._src.context.Function):
                if name in adj.symbols and isinstance(adj.symbols[name], warp._src.context.Function):
                    if adj.symbols[name] is not rhs:
                        raise WarpCodegenError(
                            f"Error, rebinding function-valued local '{name}' to a different function is not "
                            "supported. Warp does not have function pointers, so a local bound to a function "
                            "must refer to the same function throughout the kernel."
                        )
                adj.symbols[name] = rhs
                return

            # check type matches if symbol already defined
            if name in adj.symbols:
                # a local previously bound to a function cannot be reassigned to a value. Warp does
                # not have function pointers, so a mixed function/value local has no codegen-able
                # meaning. Guard here before the generic type check below, which would otherwise read
                # `adj.symbols[name].type` and fail with an opaque AttributeError on the Function.
                if isinstance(adj.symbols[name], warp._src.context.Function):
                    raise WarpCodegenError(
                        f"Error, rebinding function-valued local '{name}' to a non-function value is not "
                        "supported. Warp does not have function pointers, so a local bound to a function "
                        "must refer to a function throughout the kernel."
                    )

                if isinstance(adj.symbols[name], warp._src.context.GradWrapper):
                    raise WarpCodegenError(
                        f"Cannot reassign local '{name}' after binding it to wp.grad(...). Warp treats "
                        "wp.grad(...) results as static function handles; use a different local variable "
                        "for the new value."
                    )

                if not types_equal(strip_reference(rhs.type), adj.symbols[name].type):
                    raise WarpCodegenTypeError(
                        f"Error, assigning to existing symbol {name} ({adj.symbols[name].type}) with different type ({rhs.type})"
                    )

            if isinstance(node.value, ast.Tuple):
                out = rhs
            elif isinstance(rhs, Sequence):
                out = adj.add_builtin_call("tuple", rhs)
            elif isinstance(node.value, ast.Name) or is_reference(rhs.type):
                out = adj.add_builtin_call("copy", [rhs])
            else:
                out = rhs

            if isinstance(out, Var) and is_array(out.type):
                out.ref_origin = getattr(rhs, "ref_origin", None)

            # update symbol map (assumes lhs is a Name node)
            adj.symbols[name] = out

        elif isinstance(lhs, ast.Attribute):
            # Fast path: array-rooted composite-component write (see _try_lower_array_slot_write).
            # wp.adjoint[var].field reaches here but declines because root name "wp" is not a symbol.
            if adj._try_lower_array_slot_write(lhs, rhs):
                return
            adj._store_attribute(lhs, adj.resolve_attribute_store_aggregate(lhs.value), rhs)

        else:
            raise WarpCodegenError("Error, unsupported assignment statement.")

    def _try_lower_array_slot_write(adj, lhs, rhs):
        """Intercept array-rooted composite-component writes and emit
        direct slot access. Returns True if the write was handled.

        For ``arr[i].y = rhs`` on a ``wp.array(dtype=wp.vec3)``, emits:

        - Forward: ``wp::index(arr, i).c[1] = rhs;`` (one scalar-sized store).
        - Reverse: ``wp::adj_array_store_slot(arr, adj_arr, adj_rhs,
          [&](auto& _e) -> auto& { return _e.c[1]; }, i);`` — the native
          helper reads the slot's accumulated adjoint into ``adj_rhs`` and
          zeros it (overwrite semantic), or uses ``buf.grad`` as a fallback
          source when no adjoint array is passed by the tape.

        Motivation: the generic path lowers a composite-component write as a
        whole-element load / ``assign_copy`` / ``array_store`` chain whose
        reverse pass has a **no-op adjoint** — gradients are silently dropped,
        not merely slow. This slot-level lowering stores only the touched
        scalar/sub-composite, giving correct gradients at single-slot cost
        (the generic chain also costs up to ~10x more for large composite
        dtypes such as ``mat44``).

        Shapes accepted on this fast path (where ``SCALAR`` means the
        element's scalar dtype, e.g. ``float32``; ``COMPOSITE`` means a
        ``vec``/``quat``/``mat``/``transform``):

          - ``arr[i].x`` — vec/quat component via attribute (leaf SCALAR).
          - ``arr[i][k]`` — vec/quat scalar subscript (leaf SCALAR).
          - ``arr[i][r, c]`` — mat element subscript (leaf SCALAR).
          - ``arr[i].p`` / ``arr[i].q`` — transform translation / rotation
            (leaf is COMPOSITE: vec3 or quat).
          - ``arr[i].field`` — struct field (leaf SCALAR or COMPOSITE).
          - ``arr[i].outer.inner.a`` — nested struct chains terminating in
            any of the above.
          - ``arr[i].inner.m[r, c]``, ``arr[i].v.y``, etc. — struct chains
            descending into a composite field.
          - Any of the above on 2D/3D/4D arrays.

        Declines (returns False) for:

          - Arrays other than plain ``wp.array`` (indexed, fabric, fixed):
            those have no ``adj_array_store_slot`` overload (the slot call
            would fail to compile for them).
          - Vec/quat/mat slices (writing a sub-vec, a row, or a sub-mat):
            the slot is a composite that isn't trivially addressable as
            a single reference via the lambda pattern.
          - Chains that traverse an array field of a struct
            (``state.v[i] = rhs`` where ``v`` is ``wp.array``): that's
            a plain array write, already handled by ``array_store``.
          - Non-Name roots (e.g. ``func_call().field``).
          - Dtypes that don't support atomic accumulation at the array
            level.
        """
        plan = adj._classify_slot_access(lhs, rhs.type)
        if plan is None:
            return False

        # ``src[i]`` arrives as ``address(src, i)``; route it through ``copy``
        # so the rhs has a working adjoint chain back to the source array
        # (``load`` would route through a nop adjoint and drop the read-side
        # gradient).
        if is_reference(rhs.type):
            rhs = adj.add_builtin_call("copy", [rhs])

        # Committed: evaluate indices in Python left-to-right order
        # (outer array subscripts first, then composite-chain subscripts).
        array_indices_cpp = ", ".join(adj.eval(n).emit() for n in plan.array_indices_ast)
        access_cpp = "".join(p if isinstance(p, str) else adj.eval(p).emit() for p in plan.access_parts)

        arr_cpp = plan.root_var.emit()
        adj_arr_cpp = plan.root_var.emit_adj()
        rhs_cpp = rhs.emit()
        adj_rhs_cpp = rhs.emit_adj()

        # Forward: one slot store (wrapped for deterministic atomic mode).
        slot_lvalue = f"wp::index({arr_cpp}, {array_indices_cpp}){access_cpp}"
        adj.add_forward(adj.deterministic.wrap_slot_store(slot_lvalue, rhs_cpp))

        # Reverse: single call to the slot-level adj_array_store variant,
        # with the composite-component access encoded as a short lambda.
        # The grad-routing (adj_buf vs buf.grad) and RETAIN_GRAD handling
        # live in the native helper, matching adj_array_store's existing
        # logic scoped to the single slot.
        adj.add_reverse(
            f"wp::adj_array_store_slot({arr_cpp}, {adj_arr_cpp}, {adj_rhs_cpp}, "
            f"[&](auto& _e) -> auto& {{ return _e{access_cpp}; }}, {array_indices_cpp});"
        )

        adj._mark_array_write(plan.root_var)
        return True

    def _classify_slot_access(adj, lhs, rhs_type):
        """Pure analysis of an array slot-write LHS — emits no IR.

        Returns a ``SlotAccessPlan`` if ``lhs`` is an array-rooted composite-
        component write the fast path can lower, else ``None``. Performs only
        AST-shape and type checks; the caller evaluates indices and emits IR
        after acceptance, so a decline never pollutes the IR.
        """
        # Walk LHS leaf-to-root, then validate root.
        chain = []
        node = lhs
        while isinstance(node, ast.Attribute | ast.Subscript):
            chain.append(node)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        # ``wp``-rooted chains (e.g. ``wp.adjoint[var].field``) decline here:
        # the module alias ``wp`` is never a key in ``adj.symbols``. Plain and
        # augmented ``wp.adjoint[var]`` writes are also intercepted upstream in
        # emit_Assign before this fast path is reached.
        if node.id not in adj.symbols:
            return None
        root_var = adj.symbols[node.id]
        # ``adj.symbols`` may also hold ``GradWrapper`` and other non-Var values.
        if not isinstance(root_var, Var):
            return None
        root_type = strip_reference(root_var.type)
        if not warp._src.types.matches_array_class(root_type, warp._src.types.array):
            return None
        if root_type.dtype in warp._src.types.non_atomic_types:
            return None

        # Consume ``ndim`` outermost subscripts as the array indices.
        chain.reverse()
        needed = root_type.ndim
        array_indices_ast = []
        consumed = 0
        for step in chain:
            if not isinstance(step, ast.Subscript) or needed == 0:
                break
            elts = list(step.slice.elts) if isinstance(step.slice, ast.Tuple) else [step.slice]
            if len(elts) > needed:
                return None
            array_indices_ast.extend(elts)
            needed -= len(elts)
            consumed += 1
        if needed != 0:
            return None
        remaining = chain[consumed:]
        if not remaining:
            # Whole-element write — not a composite-component write.
            return None

        # Walk the composite-component chain. ``access_parts`` interleaves
        # text segments (``str``) and AST nodes for subscript indices
        # (evaluated after acceptance, so a reject doesn't pollute IR).
        access_parts: list = []  # list[str | ast.expr]
        current_type = root_type.dtype
        # Member-access fragments below encode native layout: vec_t uses .c[i],
        # mat_t uses .data[r][c], quat_t uses named .x/.y/.z/.w, transform_t .p/.q.
        # The same access string drives both the forward store and the reverse lambda.
        for step in remaining:
            if isinstance(step, ast.Attribute):
                if type_is_vector(current_type):
                    dim = current_type._shape_[0]
                    swizzles = "xyzw"[:dim]
                    if len(step.attr) != 1 or step.attr not in swizzles:
                        return None
                    access_parts.append(f".c[{swizzles.index(step.attr)}]")
                    current_type = getattr(current_type, "_wp_scalar_type_", None)
                elif type_is_quaternion(current_type):
                    if step.attr not in ("x", "y", "z", "w"):
                        return None
                    # quat_t exposes named scalar fields .x/.y/.z/.w; vec_t uses .c[N].
                    access_parts.append(f".{step.attr}")
                    current_type = getattr(current_type, "_wp_scalar_type_", None)
                elif type_is_transformation(current_type):
                    scalar_t = getattr(current_type, "_wp_scalar_type_", None)
                    if scalar_t is None:
                        return None
                    if step.attr == "p":
                        access_parts.append(".p")
                        current_type = vector(length=3, dtype=scalar_t)
                    elif step.attr == "q":
                        access_parts.append(".q")
                        current_type = quaternion(dtype=scalar_t)
                    else:
                        return None
                elif isinstance(current_type, Struct):
                    if step.attr not in current_type.vars:
                        return None
                    access_parts.append(f".{step.attr}")
                    current_type = current_type.vars[step.attr].type
                    # Array-field chains (``state.v[i] = rhs``) are plain
                    # array writes; let the legacy path handle them.
                    if is_array(current_type):
                        return None
                else:
                    return None
            else:  # ast.Subscript
                if type_is_matrix(current_type):
                    if not isinstance(step.slice, ast.Tuple) or len(step.slice.elts) != 2:
                        return None
                    access_parts.extend([".data[", step.slice.elts[0], "][", step.slice.elts[1], "]"])
                    current_type = getattr(current_type, "_wp_scalar_type_", None)
                elif (
                    type_is_vector(current_type)
                    or type_is_quaternion(current_type)
                    or type_is_transformation(current_type)
                ):
                    if isinstance(step.slice, ast.Tuple):
                        return None
                    access_parts.extend(["[", step.slice, "]"])
                    current_type = getattr(current_type, "_wp_scalar_type_", None)
                else:
                    return None
            if current_type is None:
                return None

        slot_type = current_type
        if not types_equal(strip_reference(rhs_type), slot_type):
            return None

        return SlotAccessPlan(root_var, array_indices_ast, access_parts, slot_type)

    def _mark_array_write(adj, target):
        """Record an array write for the autograd access verifier.

        No-op unless ``builder_options['verify_autograd_array_access']`` is set.
        ``target`` must be the array ``Var`` being written.
        """
        if not adj.builder_options.get("verify_autograd_array_access", False):
            return
        target.mark_write(
            kernel_name=adj.fun_name,
            filename=adj.filename,
            lineno=adj.lineno + adj.fun_lineno,
        )

    def _store_subscript(adj, lhs, target, indices, rhs):
        """Store ``rhs`` into a subscript target using pre-evaluated ``target`` and ``indices``.

        Shared by ``emit_Assign`` and ``emit_AugAssign`` to avoid duplicating
        the store-dispatch logic for array, tile, and vector/matrix subscripts.
        """
        target_type = strip_reference(target.type)

        if is_array(target_type):
            # Deterministic two-pass mode must suppress normal array writes in
            # phase 0 so the counting pass does not introduce side effects.
            if adj.deterministic.needs_store_guard():
                adj.deterministic.add_array_store(target, indices, rhs)
            else:
                adj.add_builtin_call("array_store", [target, *indices, rhs])

            adj._mark_array_write(target)

        elif is_tile(target_type):
            adj.add_builtin_call("assign", [target, *indices, rhs])

        elif (
            type_is_vector(target_type)
            or type_is_quaternion(target_type)
            or type_is_matrix(target_type)
            or type_is_transformation(target_type)
        ):
            # recursively unwind AST, stopping at penultimate node
            root = lhs
            while hasattr(root.value, "value"):
                root = root.value
            # lhs is updating a variable adjoint (i.e. wp.adjoint[var])
            if hasattr(root, "attr") and root.attr == "adjoint":
                attr = adj.add_builtin_call("index", [target, *indices])
                adj.add_builtin_call("store", [attr, rhs])
                return

            # TODO: array vec component case
            if is_reference(target.type):
                attr = adj.add_builtin_call("indexref", [target, *indices])
                adj.add_builtin_call("store", [attr, rhs])

                if not adj.custom_reverse_mode:
                    lineno = adj.lineno + adj.fun_lineno
                    line = adj.source_lines[adj.lineno]
                    node_source = adj.get_node_source(lhs.value)
                    log_debug(
                        f"Warning: mutating {node_source} in function {adj.fun_name} at {adj.filename}:{lineno}: this is a non-differentiable operation.\n{line}"
                    )
            else:
                if adj.builder_options.get("enable_vector_component_overwrites", False):
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
                f"Can only subscript assign array, vector, quaternion, transformation, and matrix types, got {target_type}"
            )

    def resolve_attribute_store_aggregate(adj, node):
        """Resolve the aggregate that an attribute store/augmented-store writes into.

        For a nested struct chain rooted at a local variable (e.g. ``out.inner`` in
        ``out.inner.x = ...``), return a flat dotted member ``Var`` (``Var("0.inner",
        Inner)``) instead of evaluating it to a ``Reference``. Routing struct-field
        stores through a value-typed member Var lets them reuse the value-type
        assignment branch in :meth:`_store_attribute`, which emits correct root-to-leaf
        adjoints. Evaluating to a reference instead only flushes leaf-to-root, which is
        correct for reads but silently drops the write gradient (the parent's adjoint is
        never read back into the stored value).

        Falls back to :meth:`eval` for anything that is not a pure local-struct chain
        (array-rooted writes, vector/transform components, function-argument
        references), preserving the existing reference-based paths.
        """
        member = adj._resolve_struct_member_lvalue(node)
        if member is not None and isinstance(member.type, Struct):
            return member
        return adj.eval(node)

    def _resolve_struct_member_lvalue(adj, node):
        """Resolve a struct attribute chain rooted at a local variable to a flat dotted
        ``Var``, or return ``None`` if ``node`` is not such a chain.

        Each level must be a non-reference struct field so the result can be written as
        a direct ``a.b.c`` member access. Used by :meth:`resolve_attribute_store_aggregate`.
        """
        if not isinstance(node, ast.Attribute):
            return None

        if isinstance(node.value, ast.Name):
            base = adj.eval(node.value)
        else:
            base = adj._resolve_struct_member_lvalue(node.value)

        if not isinstance(base, Var) or is_reference(base.type) or not isinstance(base.type, Struct):
            return None
        if node.attr not in base.type.vars:
            return None

        attr_type = base.type.vars[node.attr].type
        return Var(f"{base.label}.{node.attr}", attr_type, prefix=base.prefix)

    def _store_attribute(adj, lhs, aggregate, rhs):
        """Store ``rhs`` into an attribute target using pre-evaluated ``aggregate``.

        Shared by ``emit_Assign`` and ``emit_AugAssign`` to avoid duplicating
        the store-dispatch logic for vector/quaternion/transform/struct attributes.
        """
        aggregate_type = strip_reference(aggregate.type)

        # assigning to a vector or quaternion component
        if type_is_vector(aggregate_type) or type_is_quaternion(aggregate_type):
            index = adj.vector_component_index(lhs.attr, aggregate_type)

            if is_reference(aggregate.type):
                attr = adj.add_builtin_call("indexref", [aggregate, index])
                adj.add_builtin_call("store", [attr, rhs])
            else:
                if adj.builder_options.get("enable_vector_component_overwrites", False):
                    out = adj.add_builtin_call("assign_copy", [aggregate, index, rhs])

                    # re-point target symbol to out var
                    for id in adj.symbols:
                        if adj.symbols[id] == aggregate:
                            adj.symbols[id] = out
                            break
                else:
                    adj.add_builtin_call("assign_inplace", [aggregate, index, rhs])

        elif type_is_transformation(aggregate_type):
            component = adj.transform_component(lhs.attr)

            # TODO: x[i,j].p = rhs case
            if is_reference(aggregate.type):
                raise WarpCodegenError(f"Error, assigning transform attribute {component} to an array element")

            if component == "p":
                return adj.add_builtin_call("transform_set_translation", [aggregate, rhs])
            else:
                return adj.add_builtin_call("transform_set_rotation", [aggregate, rhs])

        elif isinstance(aggregate_type, Struct) and not is_reference(aggregate.type):
            attr_var = aggregate_type.vars[lhs.attr]

            if is_reference(rhs.type):
                rhs = adj.add_builtin_call("copy", [rhs])

            if not types_equal(strip_reference(rhs.type), attr_var.type):
                raise WarpCodegenTypeError(
                    f"Error, assigning to struct field `{lhs.attr}` ({attr_var.type}) with different type ({rhs.type})"
                )

            adj.add_forward(f"{aggregate.emit()}.{attr_var.label} = {rhs.emit()};")

            adj.add_reverse(f"{aggregate.emit_adj()}.{attr_var.label} = {{}};")
            if adj.is_differentiable_value_type(attr_var.type):
                adj.add_reverse(f"{rhs.emit_adj()} += {aggregate.emit_adj()}.{attr_var.label};")

        else:
            attr = adj.emit_Attribute(lhs, aggregate=aggregate)
            if is_reference(attr.type):
                adj.add_builtin_call("store", [attr, rhs])
            else:
                adj.add_builtin_call("assign", [attr, rhs])

            if not adj.custom_reverse_mode:
                lineno = adj.lineno + adj.fun_lineno
                line = adj.source_lines[adj.lineno]
                msg = f'Warning: detected mutated struct {attr.label} during function "{adj.fun_name}" at {adj.filename}:{lineno}: this is a non-differentiable operation.\n{line}'
                log_debug(msg)

    def emit_Return(adj, node):
        if node.value is None:
            var = None
        elif isinstance(node.value, ast.Tuple):
            var = tuple(adj.eval(arg) for arg in node.value.elts)
        else:
            var = adj.eval(node.value)
            if not isinstance(var, list) and not isinstance(var, tuple):
                var = (var,)

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

    @contextlib.contextmanager
    def suppress_read_tracking(adj):
        """Save and restore ``is_read`` states on function arguments.

        Used by augmented assignment emission to prevent the LHS load from
        registering as a read in the autograd verification system, since the
        overall operation is a write, not a read.
        """
        if adj.builder_options.get("verify_autograd_array_access", False):
            is_read_states = [arg.is_read for arg in adj.args]
        else:
            is_read_states = None
        try:
            yield
        finally:
            if is_read_states is not None:
                for i, arg in enumerate(adj.args):
                    arg.is_read = is_read_states[i]

    def emit_AugAssign(adj, node):
        lhs = node.target

        # For simple name targets (x += expr), evaluate the RHS once and
        # apply the operation directly to avoid double-evaluation.
        if isinstance(lhs, ast.Name):
            rhs = adj.eval(node.value)
            target = adj.eval(lhs)
            if isinstance(target, warp._src.context.GradWrapper):
                raise WarpCodegenError(
                    f"Cannot reassign local '{lhs.id}' after binding it to wp.grad(...). Warp treats "
                    "wp.grad(...) results as static function handles; use a different local variable "
                    "for the new value."
                )

            # In-place tile ops mutate target directly; no symbol table update needed.
            if is_tile(target.type) and is_tile(rhs.type):
                if isinstance(node.op, ast.Add):
                    adj.add_builtin_call("add_inplace", [target, rhs])
                    return
                if isinstance(node.op, ast.Sub):
                    adj.add_builtin_call("sub_inplace", [target, rhs])
                    return
                if isinstance(node.op, ast.BitAnd):
                    adj.add_builtin_call("bit_and_inplace", [target, rhs])
                    return
                if isinstance(node.op, ast.BitOr):
                    adj.add_builtin_call("bit_or_inplace", [target, rhs])
                    return
                if isinstance(node.op, ast.BitXor):
                    adj.add_builtin_call("bit_xor_inplace", [target, rhs])
                    return

            # Non-inplace: produces a new value, rebind the symbol.
            # Check for user-defined operator overloads first (same as emit_BinOp).
            op_name = builtin_operators[type(node.op)]

            try:
                user_func = adj.resolve_external_reference(op_name)
                if isinstance(user_func, warp._src.context.Function):
                    result = adj.add_call(user_func, (target, rhs), {}, {})
            except WarpCodegenError:
                pass
            else:
                if isinstance(user_func, warp._src.context.Function):
                    if lhs.id in adj.ref_params:
                        adj.store_ref_param_value(lhs.id, result, augmented=True)
                    else:
                        adj.symbols[lhs.id] = result
                    return

            result = adj.add_builtin_call(op_name, [target, rhs])

            if lhs.id in adj.ref_params:
                # Write the computed result back into the referenced storage.
                adj.store_ref_param_value(lhs.id, result, augmented=True)
            else:
                # Validate type consistency (same as emit_Assign for Name targets).
                if lhs.id in adj.symbols:
                    if not types_equal(strip_reference(result.type), adj.symbols[lhs.id].type):
                        raise WarpCodegenTypeError(
                            f"Error, augmented assignment to `{lhs.id}` ({adj.symbols[lhs.id].type}) "
                            f"produces different type ({result.type})"
                        )

                adj.symbols[lhs.id] = result
            return

        # Evaluate RHS once for non-Name targets.
        rhs = adj.eval(node.value)

        def apply_op(current):
            """Compute ``current <op> rhs`` using user-defined overloads or builtins."""
            op_name = builtin_operators[type(node.op)]
            try:
                user_func = adj.resolve_external_reference(op_name)
                if isinstance(user_func, warp._src.context.Function):
                    return adj.add_call(user_func, (current, rhs), {}, {})
            except WarpCodegenError:
                pass
            return adj.add_builtin_call(op_name, [current, rhs])

        def augassign_subscript(target, indices):
            """Load current value via pre-evaluated target/indices, apply op, store back."""
            target_type = strip_reference(target.type)

            with adj.suppress_read_tracking():
                if is_reference(target.type):
                    current_ref = adj.add_builtin_call("indexref", [target, *indices])
                    current = adj.load(current_ref)
                elif is_array(target_type):
                    current = adj.add_builtin_call("address", [target, *indices])
                elif is_tile(target_type):
                    current = adj.add_builtin_call("tile_extract", [target, *indices])
                else:
                    current = adj.add_builtin_call("extract", [target, *indices])

            result = apply_op(current)
            adj._store_subscript(lhs, target, indices, result)

        def augassign_attribute():
            """Load current value of attribute target, apply op, store back."""
            aggregate = adj.resolve_attribute_store_aggregate(lhs.value)

            with adj.suppress_read_tracking():
                current = adj.emit_Attribute(lhs, aggregate=aggregate)

            result = apply_op(current)
            adj._store_attribute(lhs, aggregate, result)

        if isinstance(lhs, ast.Subscript):
            # wp.adjoint[var] appears in custom grad functions; handle the
            # adjoint store inline rather than through augassign_subscript.
            if hasattr(lhs.value, "attr") and lhs.value.attr == "adjoint":
                with adj.suppress_read_tracking():
                    current = adj.eval(lhs)

                result = apply_op(current)
                lhs.slice.is_adjoint = True
                src_var = adj.eval(lhs.slice)
                var = Var(f"adj_{src_var.label}", type=src_var.type, constant=None, prefix=False)
                adj.add_forward(f"{var.emit()} = {result.emit()};")
                return

            target, indices = adj.eval_subscript(lhs)

            target_type = strip_reference(target.type)
            indices = adj.eval_indices(target_type, indices)

            if is_array(target_type):
                # target_types int8, uint8, int16, uint16 are not suitable for atomic array accumulation
                if target_type.dtype in warp._src.types.non_atomic_types:
                    augassign_subscript(target, indices)
                    return

                # the same holds true for vecs/mats/quats that are composed of these types
                if (
                    type_is_vector(target_type.dtype)
                    or type_is_quaternion(target_type.dtype)
                    or type_is_matrix(target_type.dtype)
                    or type_is_transformation(target_type.dtype)
                ):
                    dtype = getattr(target_type.dtype, "_wp_scalar_type_", None)
                    if dtype in warp._src.types.non_atomic_types:
                        augassign_subscript(target, indices)
                        return

                # Array augmented assignment lowers to a Warp atomic, e.g.
                # ``arr[i] += value`` -> ``wp.atomic_add(arr, i, value)``.
                # The Python expression has no visible return value, so the
                # generated atomic return is intentionally discarded.
                if isinstance(node.op, ast.Add):
                    adj.add_builtin_call("atomic_add", [target, *indices, rhs], return_value_used=False)
                    adj._mark_array_write(target)
                elif isinstance(node.op, ast.Sub):
                    adj.add_builtin_call("atomic_sub", [target, *indices, rhs], return_value_used=False)
                    adj._mark_array_write(target)
                elif isinstance(node.op, ast.BitAnd):
                    adj.add_builtin_call("atomic_and", [target, *indices, rhs], return_value_used=False)
                    adj._mark_array_write(target)
                elif isinstance(node.op, ast.BitOr):
                    adj.add_builtin_call("atomic_or", [target, *indices, rhs], return_value_used=False)
                    adj._mark_array_write(target)
                elif isinstance(node.op, ast.BitXor):
                    adj.add_builtin_call("atomic_xor", [target, *indices, rhs], return_value_used=False)
                    adj._mark_array_write(target)
                else:
                    log_debug(f"Warning: in-place op {node.op} is not differentiable")
                    augassign_subscript(target, indices)
                    return

            elif (
                type_is_vector(target_type)
                or type_is_quaternion(target_type)
                or type_is_matrix(target_type)
                or type_is_transformation(target_type)
            ):
                if is_reference(target.type):
                    augassign_subscript(target, indices)
                    return

                if isinstance(node.op, ast.Add):
                    adj.add_builtin_call("add_inplace", [target, *indices, rhs])
                elif isinstance(node.op, ast.Sub):
                    adj.add_builtin_call("sub_inplace", [target, *indices, rhs])
                elif isinstance(node.op, ast.BitAnd):
                    adj.add_builtin_call("bit_and_inplace", [target, *indices, rhs])
                elif isinstance(node.op, ast.BitOr):
                    adj.add_builtin_call("bit_or_inplace", [target, *indices, rhs])
                elif isinstance(node.op, ast.BitXor):
                    adj.add_builtin_call("bit_xor_inplace", [target, *indices, rhs])
                else:
                    log_debug(f"Warning: in-place op {node.op} is not differentiable")
                    augassign_subscript(target, indices)
                    return

            elif is_tile(target.type):
                if isinstance(node.op, ast.Add):
                    adj.add_builtin_call("tile_add_inplace", [target, *indices, rhs])
                elif isinstance(node.op, ast.Sub):
                    adj.add_builtin_call("tile_sub_inplace", [target, *indices, rhs])
                elif isinstance(node.op, ast.BitAnd):
                    adj.add_builtin_call("tile_bit_and_inplace", [target, *indices, rhs])
                elif isinstance(node.op, ast.BitOr):
                    adj.add_builtin_call("tile_bit_or_inplace", [target, *indices, rhs])
                elif isinstance(node.op, ast.BitXor):
                    adj.add_builtin_call("tile_bit_xor_inplace", [target, *indices, rhs])
                else:
                    log_debug(f"Warning: in-place op {node.op} is not differentiable")
                    augassign_subscript(target, indices)
                    return

            else:
                raise WarpCodegenError("Can only subscript in-place assign array, vector, quaternion, and matrix types")

        elif isinstance(lhs, ast.Attribute):
            augassign_attribute()
            return

        else:
            raise WarpCodegenError("Error, unsupported target for augmented assignment.")

    def emit_Tuple(adj, node):
        elements = tuple(adj.eval(x) for x in node.elts)
        return adj.add_builtin_call("tuple", elements)

    def emit_Pass(adj, node):
        pass

    node_visitors: ClassVar[dict[type[ast.AST], Callable]] = {
        ast.FunctionDef: emit_FunctionDef,
        ast.If: emit_If,
        ast.IfExp: emit_IfExp,
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
        ast.Subscript: emit_Subscript,
        ast.Slice: emit_Slice,
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
                elif i < len(path) - 1:
                    # Intermediate attribute doesn't exist - path is invalid.
                    # Only the last element is allowed to be missing (e.g., for builtin functions
                    # like wp.tid() where 'tid' is not an attribute of the warp module but is
                    # looked up in builtin_functions by emit_Call).
                    return None

        return expr

    # retrieves a dictionary of all closure and global variables and their values
    # to be used in the evaluation context of wp.static() expressions
    def get_static_evaluation_context(adj):
        # variables captured in closure have precedence over global vars
        vars_dict = adj.func.__globals__ | get_closure_vars(adj.func)

        return vars_dict

    def is_static_expression(adj, func):
        return (
            isinstance(func, types.FunctionType)
            and func.__module__ == "warp._src.builtins"
            and func.__qualname__ == "static"
        )

    def is_grad_expression(adj, func):
        """Check if func is a warp.grad function."""
        return (
            isinstance(func, types.FunctionType)
            and func.__module__ == "warp._src.context"
            and func.__qualname__ == "grad"
        )

    # verify the return type of a wp.static() expression is supported inside a Warp kernel
    def verify_static_return_value(adj, value):
        if value is None:
            raise ValueError("None is returned")
        if warp._src.types.is_value(value):
            return True
        if warp._src.types.is_array(value):
            # more useful explanation for the common case of creating a Warp array
            raise ValueError("a Warp array cannot be created inside Warp kernels")
        if isinstance(value, str):
            # we want to support cases such as `print(wp.static("test"))`
            return True
        if isinstance(value, warp._src.context.Function):
            return True

        def verify_struct(s: StructInstance, attr_path: list[str]):
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
    @staticmethod
    def extract_node_source_from_lines(source_lines, node) -> str | None:
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
            for lineno in range(start_line, len(source_lines)):
                if lineno == start_line:
                    c_start = start_col
                else:
                    c_start = 0
                line = source_lines[lineno]
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
            return source_lines[start_line][start_col:end_col]
        else:
            # multi-line expression
            lines = []
            # first line (from start_col to the end)
            lines.append(source_lines[start_line][start_col:])
            # middle lines (entire lines)
            lines.extend(source_lines[start_line + 1 : end_line])
            # last line (from the start to end_col)
            lines.append(source_lines[end_line][:end_col])
            return "".join(lines).strip()

    @staticmethod
    def extract_lambda_source(func, only_body=False) -> str | None:
        try:
            source_lines = inspect.getsourcelines(func)[0]
            source_lines[0] = source_lines[0][source_lines[0].index("lambda") :]
        except OSError as e:
            raise WarpCodegenError(
                "Could not access lambda function source code. Please use a named function instead."
            ) from e
        source = "".join(source_lines)
        source = source[source.index("lambda") :].rstrip()
        # Remove trailing unbalanced parentheses
        while source.count("(") < source.count(")"):
            source = source[:-1]
        # extract lambda expression up until a comma, e.g. in the case of
        # "map(lambda a: (a + 2.0, a + 3.0), a, return_kernel=True)"
        si = max(source.rfind(")"), source.find(":"))
        ci = source.find(",", si)
        if ci != -1:
            source = source[:ci]
        tree = ast.parse(source)
        lambda_source = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                if only_body:
                    # extract the body of the lambda function
                    lambda_source = Adjoint.extract_node_source_from_lines(source_lines, node.body)
                    if lambda_source is not None and "\n" in lambda_source:
                        try:
                            # Probe parseability; missing outer parentheses are the only fixable case.
                            ast.parse(lambda_source, mode="eval")
                        except SyntaxError:
                            lambda_source = f"({lambda_source})"
                else:
                    # extract the entire lambda function
                    lambda_source = Adjoint.extract_node_source_from_lines(source_lines, node)
                    break
        return lambda_source

    def extract_node_source(adj, node) -> str | None:
        return adj.extract_node_source_from_lines(adj.source_lines, node)

    # handles a wp.static() expression and returns the resulting object and a string representing the code
    # of the static expression
    def evaluate_static_expression(adj, node) -> tuple[Any, str]:
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
        code_to_eval = static_code  # code to be evaluated

        vars_dict = adj.get_static_evaluation_context()
        # add constant variables to the static call context
        constant_vars = {k: v.constant for k, v in adj.symbols.items() if isinstance(v, Var) and v.constant is not None}
        vars_dict.update(constant_vars)

        # Replace all constant `len()` expressions with their value.
        if "len" in static_code:
            constant_types = {k: v.type for k, v in adj.symbols.items() if isinstance(v, Var) and v.type is not None}
            len_expr_ctx = vars_dict | constant_types | {"len": warp._src.types.type_length}

            # We want to replace the expression code in-place,
            # so reparse it to get the correct column info.
            len_value_locs: list[tuple[int, int, int]] = []
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
                code_to_eval = new_static_code

        try:
            value = eval(code_to_eval, vars_dict)
            if isinstance(value, (enum.IntEnum, enum.IntFlag)):
                value = int(value)
            log_debug(f"Evaluated static command: {static_code} = {value}")
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
        # ``visit_For`` and ``visit_Call`` below are the upstream
        # ``ast.NodeTransformer`` subclass's methods lifted into closures —
        # bodies are unchanged except for the trailing ``self.generic_visit(node)``,
        # which becomes ``_walk_children(node)`` in ``visit_For`` and ``None`` in
        # ``visit_Call`` (where ``None`` means "no replacement, recurse normally").
        # ``_walk_children`` replaces ``generic_visit``: same DFS over
        # ``node._fields``, but dispatching Calls/Fors inline by class identity
        # (no ``'visit_' + cls.__name__`` + ``getattr``) and mutating list
        # fields in place only when a replacement actually occurred.
        # Replacements are collected as ``(container, key, new_node)`` and
        # applied after the walk so the walk sees an unmutated tree.
        loop_vars = {}  # was: self.loop_vars
        replacements = []  # (container, key, new_node); applied after the walk

        def _walk_children(node):
            for field_name in node._fields:
                value = getattr(node, field_name, None)
                if value is None:
                    continue
                if type(value) is list:
                    for i, child in enumerate(value):
                        if not isinstance(child, ast.AST):
                            continue
                        cls = type(child)
                        if cls is ast.Call:
                            result = visit_Call(child)
                            if result is not None:
                                replacements.append((value, i, result))
                                continue
                        elif cls is ast.For:
                            visit_For(child)
                            continue
                        _walk_children(child)
                elif isinstance(value, ast.AST):
                    cls = type(value)
                    if cls is ast.Call:
                        result = visit_Call(value)
                        if result is not None:
                            replacements.append((node, field_name, result))
                            continue
                    elif cls is ast.For:
                        visit_For(value)
                        continue
                    _walk_children(value)

        def visit_For(node):
            # Track loop variable while visiting loop body (simple names only;
            # tuple unpacking like `for x, y in ...` is rare in Warp kernels)
            var_name = node.target.id if isinstance(node.target, ast.Name) else None
            if var_name:
                loop_vars[var_name] = loop_vars.get(var_name, 0) + 1
            _walk_children(node)  # was: self.generic_visit(node)
            if var_name:
                loop_vars[var_name] -= 1
                if loop_vars[var_name] == 0:
                    del loop_vars[var_name]

        def visit_Call(node):
            func, _ = adj.resolve_static_expression(node.func, eval_types=False)
            if adj.is_static_expression(func):
                # If the static expression references an enclosing loop variable,
                # defer evaluation to codegen time when the loop constant is available
                expr_node = node.args[0] if node.args else (node.keywords[0].value if node.keywords else None)
                if expr_node:
                    referenced = {n.id for n in ast.walk(expr_node) if isinstance(n, ast.Name)}
                    if referenced & loop_vars.keys():
                        adj.has_unresolved_static_expressions = True
                        return None  # was: return self.generic_visit(node)

                try:
                    # the static expression will execute as long as the static expression is valid and
                    # only depends on global or captured variables
                    obj, code = adj.evaluate_static_expression(node)
                    if code is not None:
                        adj.resolved_static_expressions[code] = obj
                        if isinstance(obj, warp._src.context.Function):
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
                    # In any case, we mark this Adjoint to have unresolvable static expressions.
                    # This will trigger a code generation step even if the module hash is unchanged.
                    adj.has_unresolved_static_expressions = True

            return None  # was: return self.generic_visit(node)

        # Walk the tree, then apply replacements in one pass. ``adj.tree`` is
        # always a Module, so we go straight into ``_walk_children``.
        _walk_children(adj.tree)
        for container, key, new_node in replacements:
            if isinstance(container, list):
                container[key] = new_node
            else:
                setattr(container, key, new_node)

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
            # resolve_path traverses a dotted name chain starting from a root
            # name — only valid when the root expression is actually a name.
            # A non-Name root (e.g. boxes[i].quat.w where boxes[i] is a
            # Subscript) has no static root to look up; calling resolve_path
            # with the bare attribute suffix would match warp module names
            # (e.g. 'quat' → warp.quat) and return the wrong object.
            captured_obj = adj.resolve_path(path)
            if captured_obj is not None:
                return captured_obj, path

        return None, path

    def resolve_external_reference(adj, name: str):
        return resolve_closure_or_global(adj.func, name)

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

    def reference_nodes(adj) -> tuple[ast.AST, ...]:
        """Return the cached ``Name``/``Attribute``/``Call``/``Assign`` nodes of ``adj.tree``.

        Both ``Adjoint.get_references`` (module hashing) and ``Module._find_references``
        (dependency tracking) walk the kernel AST to find references. They run at different
        times (hashing versus registration), so they cannot share the resolution of those
        nodes, but they can share the traversal: the tree is walked once here and the node
        tuple is reused, each caller resolving from it at its own time.

        Sharing the resolution would be wrong in either direction. Resolving at hash time and
        reusing the result for dependency tracking would miss the dependency edges of modules
        that are only reached transitively, so reloading such a module would not unload its
        dependents. Resolving at registration time and reusing the result for hashing would
        make a regular kernel's hash stale if a referenced global or constant is rebound
        before the kernel is first built.

        This cache assumes ``adj.tree`` is structurally final before the first call. Any code
        that mutates ``adj.tree`` afterwards must reset ``adj._reference_nodes`` to ``None``.
        """
        if adj._reference_nodes is None:
            adj._reference_nodes = tuple(
                iter_ast_nodes_of_types(adj.tree, ast.Name, ast.Attribute, ast.Call, ast.Assign)
            )
        return adj._reference_nodes

    def get_references(adj) -> tuple[dict[str, Any], dict[Any, Any], dict[warp._src.context.Function, Any]]:
        """Traverse ``adj.tree`` for referenced constants, types, and user-defined functions.

        As a side effect, also sets ``adj.kernel_dim`` (the thread-grid dimension inferred from
        ``wp.tid()``). It is folded into this traversal rather than walked separately because
        ``get_references`` already visits every ``Assign`` and runs for every adjoint during
        module hashing -- which precedes any code generation or launch that reads ``kernel_dim``.
        """

        local_variables = set()  # Track local variables appearing on the LHS so we know when variables are shadowed

        constants: dict[str, Any] = {}
        types: dict[Struct | type, Any] = {}
        functions: dict[warp._src.context.Function, Any] = {}
        max_dim = 0  # thread-grid dimension, inferred from wp.tid() unpack arity
        callable_arg_values = getattr(adj, "callable_arg_values", None) or {}

        # Shared single traversal (see reference_nodes); resolved here at hash time.
        for node in adj.reference_nodes():
            if isinstance(node, ast.Name) and node.id not in local_variables:
                # look up in closure/global variables
                obj = adj.resolve_external_reference(node.id)
                if warp._src.types.is_value(obj):
                    constants[node.id] = obj

            elif isinstance(node, ast.Attribute):
                obj, path = adj.resolve_static_expression(node, eval_types=False)
                if warp._src.types.is_value(obj):
                    constants[".".join(path)] = obj

            elif isinstance(node, ast.Call):
                func = resolve_reference_call_func(adj, node, callable_arg_values)

                if isinstance(func, warp._src.context.Function) and not func.is_builtin():
                    # calling user-defined function
                    functions[func] = None

                    # Function targets are passed as values, so they must be
                    # added explicitly to the function reference set. Built-in
                    # targets are hash inputs too, but they are filtered out by
                    # module dependency discovery because they have no module.
                    for callable_func in iter_call_callable_arg_targets(adj, func, node, callable_arg_values):
                        functions[callable_func] = None
                elif isinstance(func, Struct):
                    # calling struct constructor
                    types[func] = None
                elif warp._src.types.type_is_value(func):
                    # calling value type constructor
                    types[func] = None

            elif isinstance(node, ast.Assign):
                # Infer the thread-grid dimension from `i[, j, ...] = wp.tid()` unpack arity.
                if _is_tid_call(node.value):
                    target = node.targets[0]
                    max_dim = max(max_dim, len(target.elts) if isinstance(target, ast.Tuple) else 1)

                # A function bound to a local (`f = mod.func`) or to several locals via tuple
                # unpacking (`f, g = mod.a, mod.b`) is referenced only through the local(s)
                # afterwards, so it would otherwise be missed here and left out of the module
                # hash. Register each bound function explicitly to keep the hash sound.
                rhs_nodes = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
                for rhs_node in rhs_nodes:
                    rhs_func, _ = adj.resolve_static_expression(rhs_node, eval_types=False)
                    if isinstance(rhs_func, warp._src.context.Function) and not rhs_func.is_builtin():
                        functions[rhs_func] = None

                # Add the LHS names to the local_variables so we know any subsequent uses are shadowed
                lhs = node.targets[0]
                if isinstance(lhs, ast.Tuple):
                    for v in lhs.elts:
                        if isinstance(v, ast.Name):
                            local_variables.add(v.id)
                elif isinstance(lhs, ast.Name):
                    local_variables.add(lhs.id)

        adj.kernel_dim = max_dim if max_dim > 0 else 1
        return constants, types, functions


# ----------------
# code generation

cpu_module_header = """
#define WP_TILE_BLOCK_DIM {block_dim}
#define WP_NO_CRT
#include "builtin.h"
#include "deterministic.h"

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
#include "deterministic.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

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

// CUDA Thread Block Cluster shape declaration. Expands to __cluster_dims__
// only on devices that support clusters (compute capability 9.0+); otherwise
// expands to nothing so the same source compiles cleanly for any target arch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif

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
{tile_member_ops}
}};

static CUDA_CALLABLE void adj_{name}({reverse_args})
{{
{reverse_body}}}

// Required when compiling adjoints.
CUDA_CALLABLE {name} add(const {name}& a, const {name}& b)
{{
{add_body}
}}

CUDA_CALLABLE void adj_atomic_add({name}* p, {name} t)
{{
{atomic_add_body}}}

{tile_helper_body}

"""

tile_struct_member_ops_template = """

    CUDA_CALLABLE {name}& operator -= (const {name}& rhs)
    {{{prefix_sub_body}
        return *this;}}

    CUDA_CALLABLE {name} operator - () const
    {{
        {name} ret = *this;
{prefix_neg_body}
        return ret;
    }}
"""

tile_struct_helpers_template = """
// Required by tile templates. The overloads are found by ADL when tile.h is
// instantiated with a generated struct type.
CUDA_CALLABLE void adj_add(const {name}& a, const {name}& b, {name}& adj_a, {name}& adj_b, const {name}& adj_ret)
{{
    adj_a += adj_ret;
    adj_b += adj_ret;
}}

CUDA_CALLABLE {name} sub(const {name}& a, const {name}& b)
{{
    {name} ret = a;
    ret -= b;
    return ret;
}}

CUDA_CALLABLE void adj_sub(const {name}& a, const {name}& b, {name}& adj_a, {name}& adj_b, const {name}& adj_ret)
{{
    adj_a += adj_ret;
    adj_b -= adj_ret;
}}

CUDA_CALLABLE {name} atomic_add({name}* p, {name} t)
{{
    {name} old {{}};
{atomic_add_forward_body}
    return old;
}}

CUDA_CALLABLE {name} tile_atomic_add_value({name}* p, {name} t)
{{
    return atomic_add(p, t);
}}

CUDA_CALLABLE {name} tile_adj_atomic_add_value({name}* p, {name} t)
{{
    // Tile adjoint struct atomics accumulate only for side effects; callers
    // currently ignore the returned old value, so avoid a second atomic here.
    {name} old {{}};
    adj_atomic_add(p, t);
    return old;
}}

#if defined(__CUDA_ARCH__)
CUDA_CALLABLE {name} warp_shuffle_down({name} val, int offset, int mask)
{{
    {name} ret {{}};
{shuffle_down_body}
    return ret;
}}

CUDA_CALLABLE {name} warp_shuffle_xor({name} val, int lane_mask)
{{
    {name} ret {{}};
{shuffle_xor_body}
    return ret;
}}
#endif
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

# Lean (grid_stride=False) templates: 3D grid with a per-thread early return, no grid-stride loop.
# The index flattens blockIdx.{z,y,x}; the grid shape (and its uint32 cap) is built in wp_cuda_launch_kernel.
cuda_kernel_template_forward = """

{line_directive}extern "C" {launch_bounds_str}{cluster_dims_str}__global__ void {name}_cuda_kernel_forward(
    {forward_args})
{{
{line_directive}    wp::tile_shared_storage_t tile_mem;

{line_directive}    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
{line_directive}    if (_idx >= dim.size) return;
            // reset shared memory allocator
{line_directive}    wp::tile_shared_storage_t::init();

{forward_body}{line_directive}}}

"""

cuda_kernel_template_backward = """

{line_directive}extern "C" {launch_bounds_str}{cluster_dims_str}__global__ void {name}_cuda_kernel_backward(
    {reverse_args})
{{
{line_directive}    wp::tile_shared_storage_t tile_mem;

{line_directive}    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
{line_directive}    if (_idx >= dim.size) return;
            // reset shared memory allocator
{line_directive}    wp::tile_shared_storage_t::init();

{reverse_body}{line_directive}}}

"""

cuda_kernel_template_forward_grid_stride = """

{line_directive}extern "C" {launch_bounds_str}{cluster_dims_str}__global__ void {name}_cuda_kernel_forward(
    {forward_args})
{{
{line_directive}    wp::tile_shared_storage_t tile_mem;

{line_directive}    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
{line_directive}         _idx < dim.size;
{line_directive}         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {{
            // reset shared memory allocator
{line_directive}        wp::tile_shared_storage_t::init();

{forward_body}{line_directive}    }}
{line_directive}}}

"""

cuda_kernel_template_backward_grid_stride = """

{line_directive}extern "C" {launch_bounds_str}{cluster_dims_str}__global__ void {name}_cuda_kernel_backward(
    {reverse_args})
{{
{line_directive}    wp::tile_shared_storage_t tile_mem;

{line_directive}    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
{line_directive}         _idx < dim.size;
{line_directive}         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {{
            // reset shared memory allocator
{line_directive}        wp::tile_shared_storage_t::init();

{reverse_body}{line_directive}    }}
{line_directive}}}

"""

cpu_kernel_template_forward = """

void {name}_cpu_kernel_forward(
    {forward_args},
    wp_args_{name} *_wp_args)
{{
{forward_body}}}

"""

cpu_kernel_template_backward = """

void {name}_cpu_kernel_backward(
    {reverse_args},
    wp_args_{name} *_wp_args,
    wp_args_{name} *_wp_adj_args)
{{
{reverse_body}}}

"""

cpu_module_template_forward = """

extern "C" {{

// Python CPU entry points
WP_API void {name}_cpu_forward(
    wp::launch_bounds_t<{launch_ndim}> *dim,
    wp_args_{name} *_wp_args)
{{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {{
        {name}_cpu_kernel_forward(*dim, task_index, _wp_args);
    }}
}}

}} // extern C

"""

cpu_module_template_backward = """

extern "C" {{

WP_API void {name}_cpu_backward(
    wp::launch_bounds_t<{launch_ndim}> *dim,
    wp_args_{name} *_wp_args,
    wp_args_{name} *_wp_adj_args)
{{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {{
        {name}_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
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
            scalar_value = warp._src.context.runtime.core.wp_half_bits_to_float
        elif value_type._wp_scalar_type_ == bfloat16:
            # special case for bfloat16, which is stored as uint16 in the ctypes.Array
            scalar_value = warp._src.context.runtime.core.wp_bfloat16_bits_to_float
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

    elif value_type in warp._src.types.scalar_types:
        # Unwrap the raw value and handle special floats before applying
        # C++ literal suffixes for wide integer types.
        raw = value.value
        if isinstance(raw, (enum.IntEnum, enum.IntFlag)):
            raw = int(raw)
        elif isinstance(raw, builtins.float):
            if raw == math.inf:
                return "INFINITY"
            if raw == -math.inf:
                return "-INFINITY"
            if math.isnan(raw):
                return "NAN"
        s = str(raw)
        if isinstance(raw, builtins.int):
            if value_type is uint64:
                return s + "ull"
            elif value_type is int64:
                return s + "ll"
            elif value_type is uint32:
                return s + "u"
        return s

    elif issubclass(value_type, StructInstance):
        # constant struct instance
        arg_strs = []
        for key, var in value._cls.vars.items():
            attr = getattr(value, key)
            arg_strs.append(f"{Var.type_to_ctype(var.type)}({constant_str(attr)})")
        arg_str = ", ".join(arg_strs)
        return f"{value.native_name}({arg_str})"

    elif value == math.inf:
        return "INFINITY"

    elif value == -math.inf:
        return "-INFINITY"

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
def make_full_qualified_name(func: str | Callable) -> str:
    if not isinstance(func, str):
        func = func.__qualname__
    return re.sub("[^0-9a-zA-Z_]+", "", func.replace(".", "__"))


def codegen_struct(struct, device="cpu", indent_size=4, include_tile_helpers=False):
    name = struct.native_name

    body = []
    indent_block = " " * indent_size

    def field_type_supports_tile_value_ops(field_type):
        return type_is_value(field_type) or type_is_struct(field_type)

    def field_type_supports_tile_descriptor_shuffle(field_type):
        return is_array(field_type) and concrete_array_type(field_type) in (array, indexedarray)

    # Scalar leaf types that support additive accumulation: exactly the wp.atomic_add()
    # type set. Every field-wise operation (add, subtract, negate, reduction, atomic add)
    # accumulates a field only if its scalar type is here; other fields (arrays, bool,
    # narrow ints) ride along unchanged. This keeps the operations consistent with each
    # other and with Warp, which does not accumulate integral types (see the no-op
    # adj_atomic_add overloads in native code).
    accumulatable_scalar_types = (int32, int64, uint32, uint64, float32, float64, float16, bfloat16)

    def field_type_accumulates(field_type):
        # Nested structs recurse through their own (already-gated) helpers.
        if type_is_struct(field_type):
            return True
        if is_array(field_type):
            return False
        return type_scalar_type(field_type) in accumulatable_scalar_types

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
    atomic_add_forward_body = []
    prefix_add_body = []
    prefix_sub_body = []
    prefix_neg_body = []
    shuffle_down_body = []
    shuffle_xor_body = []

    # forward args
    for label, var in struct.vars.items():
        var_ctype = var.ctype()
        default_arg_def = " = {}" if forward_args else ""
        forward_args.append(f"{var_ctype} const& {label}{default_arg_def}")
        reverse_args.append(f"{var_ctype} const&")

        namespace = "wp::" if var_ctype.startswith("wp::") or var_ctype == "bool" else ""
        atomic_add_body.append(f"{indent_block}{namespace}adj_atomic_add(&p->{label}, t.{label});\n")
        if field_type_supports_tile_descriptor_shuffle(var.type):
            atomic_add_forward_body.append(f"{indent_block}old.{label} = p->{label};\n")
            shuffle_down_body.append(f"{indent_block}ret.{label} = wp::warp_shuffle_down(val.{label}, offset, mask);\n")
            shuffle_xor_body.append(f"{indent_block}ret.{label} = wp::warp_shuffle_xor(val.{label}, lane_mask);\n")
        elif field_type_supports_tile_value_ops(var.type):
            if field_type_accumulates(var.type):
                atomic_add_forward_body.append(
                    f"{indent_block}old.{label} = {namespace}atomic_add(&p->{label}, t.{label});\n"
                )
            else:
                # No CUDA atomic add for this scalar type (e.g. bool, [u]int8/16); the field
                # rides along with the struct value but is not accumulated.
                atomic_add_forward_body.append(f"{indent_block}old.{label} = p->{label};\n")
            shuffle_namespace = "" if type_is_struct(var.type) else "wp::"
            shuffle_down_body.append(
                f"{indent_block}ret.{label} = {shuffle_namespace}warp_shuffle_down(val.{label}, offset, mask);\n"
            )
            shuffle_xor_body.append(
                f"{indent_block}ret.{label} = {shuffle_namespace}warp_shuffle_xor(val.{label}, lane_mask);\n"
            )
        else:
            atomic_add_forward_body.append(f"{indent_block}old.{label} = p->{label};\n")
            shuffle_down_body.append(f"{indent_block}ret.{label} = val.{label};\n")
            shuffle_xor_body.append(f"{indent_block}ret.{label} = val.{label};\n")

        prefix = f"{indent_block}," if forward_initializers else ":"
        forward_initializers.append(f"{indent_block}{prefix} {label}{{{label}}}\n")

    # Field-wise arithmetic. A field participates in addition, subtraction, and negation
    # only if its scalar type supports value accumulation; other fields (arrays, bool,
    # narrow ints) ride along unchanged, keeping +, -, reductions, and atomic add
    # consistent rather than silently producing meaningless integer/bool arithmetic.
    for label, var in struct.vars.items():
        if field_type_accumulates(var.type):
            prefix_add_body.append(f"{indent_block}{label} += rhs.{label};\n")
            prefix_sub_body.append(f"{indent_block}{label} -= rhs.{label};\n")
            prefix_neg_body.append(f"{indent_block}ret.{label} = -ret.{label};\n")

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

    tile_member_ops = ""
    tile_helper_body = ""
    if include_tile_helpers:
        tile_member_ops = tile_struct_member_ops_template.format(
            name=name,
            prefix_sub_body="".join(prefix_sub_body),
            prefix_neg_body="".join(prefix_neg_body),
        )
        tile_helper_body = tile_struct_helpers_template.format(
            name=name,
            atomic_add_forward_body="".join(atomic_add_forward_body),
            shuffle_down_body="".join(shuffle_down_body),
            shuffle_xor_body="".join(shuffle_xor_body),
        )

    if include_tile_helpers:
        add_body = f"    {name} ret = a;\n    ret += b;\n    return ret;"
    else:
        add_body = f"    return {name}();"

    return struct_template.format(
        name=name,
        struct_body="".join([indent_block + l for l in body]),
        forward_args=indent(forward_args),
        forward_initializers="".join(forward_initializers),
        reverse_args=indent(reverse_args),
        reverse_body="".join(reverse_body),
        prefix_add_body="".join(prefix_add_body),
        atomic_add_body="".join(atomic_add_body),
        tile_member_ops=tile_member_ops,
        add_body=add_body,
        tile_helper_body=tile_helper_body,
        defaulted_constructor_def=defaulted_constructor_def,
    )


def codegen_func_forward(adj, func_type="kernel", device="cpu", grid_stride=False):
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

    # argument vars
    if device == "cpu" and func_type == "kernel":
        lines += ["//---------\n"]
        lines += ["// argument vars\n"]

        for var in adj.args:
            lines += [f"{var.ctype()} {var.emit()} = _wp_args->{var.label};\n"]

    # primal vars
    lines += ["//---------\n"]
    lines += ["// primal vars\n"]

    for var in adj.variables:
        if is_tile(var.type):
            lines += [f"{var.ctype()} {var.emit()} = {var.type.cinit(requires_grad=False)};\n"]
        elif is_tile_stack(var.type):
            lines += [f"{var.ctype()} {var.emit()} = {var.type.cinit()};\n"]
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
        if grid_stride and func_type == "kernel" and device == "cuda" and f.lstrip().startswith("return;"):
            lines += [f.replace("return;", "continue;") + "\n"]
        else:
            lines += [f + "\n"]

    return "".join(l.lstrip() if l.lstrip().startswith("#line") else indent_block + l for l in lines)


def codegen_func_reverse(adj, func_type="kernel", device="cpu", grid_stride=False):
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

    # argument vars
    if device == "cpu" and func_type == "kernel":
        lines += ["//---------\n"]
        lines += ["// argument vars\n"]

        for var in adj.args:
            lines += [f"{var.ctype()} {var.emit()} = _wp_args->{var.label};\n"]

        for var in adj.args:
            lines += [f"{var.ctype()} {var.emit_adj()} = _wp_adj_args->{var.label};\n"]

    # primal vars
    lines += ["//---------\n"]
    lines += ["// primal vars\n"]

    for var in adj.variables:
        if is_tile(var.type):
            lines += [f"{var.ctype()} {var.emit()} = {var.type.cinit(requires_grad=True)};\n"]
        elif is_tile_stack(var.type):
            lines += [f"{var.ctype()} {var.emit()} = {var.type.cinit()};\n"]
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
                    f"{var.type.ctype()} {name}{{}};\n"
                ]  # reverse mode tiles alias the forward vars since shared tiles store both primal/dual vars together
            elif var.type.storage == "shared":
                lines += [
                    f"{var.type.ctype()}& {name} = {var.emit()};\n"
                ]  # reverse mode tiles alias the forward vars since shared tiles store both primal/dual vars together
        elif is_tile_stack(var.type):
            # Adjoint pointers are intentionally uninitialized -- all adj_tile_stack_* stubs are empty no-ops.
            lines += [f"{var.ctype()} {name};\n"]
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

    if grid_stride and device == "cuda" and func_type == "kernel":
        lines += ["continue;\n"]
    else:
        lines += ["return;\n"]

    return "".join(l.lstrip() if l.lstrip().startswith("#line") else indent_block + l for l in lines)


def codegen_func(adj, c_func_name: str, device="cpu", options=None, forward_only=False, reverse_only=False):
    if options is None:
        options = {}

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

    # Tile parameters are emitted as C++ template parameters so that the
    # same @wp.func can accept tiles with any storage type (register or
    # shared) without requiring separate overloads.  The Python-level tile
    # annotation (e.g. wp.tile[float, M, N]) defaults to
    # register storage, but at the call site the tile may actually live in
    # shared memory.  By generating ``template<typename tile_t>`` instead
    # of the concrete ``tile_register_t<...>`` type, C++ template argument
    # deduction resolves the correct storage type automatically.
    #
    # Tile parameters are passed by non-const reference (not by value)
    # for two reasons: (1) owning shared tiles (tile_shared_t with
    # Owner=true) cannot be copied (static_assert in copy constructor),
    # and (2) adjoint built-ins like adj_tile_sum() expect non-const
    # Tile& parameters.
    #
    # This is a semantic change for register tiles, which were previously
    # passed by value.  The difference is observable for in-place tile
    # operations (e.g., a += b where both are tiles), which mutate the
    # parameter directly.  Simple rebinding (a = expr) creates a new C++
    # variable: for register tiles this is a full value copy, for shared
    # tiles a non-owning handle to the same shared memory (element-level
    # writes through either variable affect the same data).
    # The pass-by-reference behavior for in-place ops is intentional:
    # it matches the Python semantics where augmented assignment on a
    # mutable object modifies it in place.
    template_params = []

    # forward args
    for i, arg in enumerate(adj.args):
        if warp._src.types.is_warp_function_annotation(arg.type):
            continue
        if is_tile(arg.type) or is_tile_stack(arg.type):
            tname = f"tile_{arg.label}"
            template_params.append(tname)
            s = f"{tname}& {arg.emit()}"
        else:
            s = f"{arg.ctype()} {arg.emit()}"
        forward_args.append(s)
        if not adj.custom_reverse_mode or i < adj.custom_reverse_num_input_args:
            reverse_args.append(s)
    det_args = adj.deterministic.function_args()
    forward_args.extend(det_args)
    reverse_args.extend(det_args)
    if has_multiple_outputs:
        for i, arg in enumerate(adj.return_var):
            forward_args.append(arg.ctype() + " & ret_" + str(i))
            reverse_args.append(arg.ctype() + " & ret_" + str(i))

    # reverse args
    for i, arg in enumerate(adj.args):
        if warp._src.types.is_warp_function_annotation(arg.type):
            continue
        if adj.custom_reverse_mode and i >= adj.custom_reverse_num_input_args:
            break
        # indexed array gradients are regular arrays
        if matches_array_class(arg.type, indexedarray):
            _arg = Var(arg.label, array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(_arg.ctype() + " & adj_" + arg.label)
        elif is_tile(arg.type) or is_tile_stack(arg.type):
            tname = f"tile_{arg.label}"
            reverse_args.append(f"{tname} & adj_{arg.label}")
        elif is_reference(arg.type):
            reverse_args.append(arg.ctype() + " adj_" + arg.label)
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
            if is_tile(arg.type) or is_tile_stack(arg.type):
                tname = f"tile_{arg.label}"
                reverse_args.append(f"{tname} & {arg.emit()}")
            elif is_reference(arg.type):
                reverse_args.append(f"{arg.ctype()} {arg.emit()}")
            else:
                reverse_args.append(f"{arg.ctype()} & {arg.emit()}")

    # build template prefix for functions with tile parameters
    template_prefix = ""
    if template_params:
        template_prefix = "template<" + ", ".join(f"typename {t}" for t in template_params) + ">\n"

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
    if not adj.skip_forward_codegen and not reverse_only:
        s += template_prefix + forward_template.format(
            name=c_func_name,
            return_type=return_type,
            forward_args=indent(forward_args),
            forward_body=forward_body,
            filename=adj.filename,
            lineno=adj.fun_lineno,
            line_directive=func_line_directive,
        )

    if not adj.skip_reverse_codegen and not forward_only:
        if adj.custom_reverse_mode:
            reverse_body = "\t// user-defined adjoint code\n" + forward_body
        else:
            # Generate adjoint code if:
            # - enable_backward is True and the function is used by a backward kernel, OR
            # - force_adjoint_codegen is True (set by warp.grad() to ensure adjoint exists)
            # Note: Functions using warp.grad() won't have their adjoints called anyway
            # (the reverse call is skipped in add_call), so we can skip generating them.
            should_generate_adjoint = (
                options.get("enable_backward", True) and adj.used_by_backward_kernel
            ) or adj.force_adjoint_codegen
            should_generate_adjoint = should_generate_adjoint and not adj.uses_grad_call
            if should_generate_adjoint:
                reverse_body = codegen_func_reverse(adj, func_type="function", device=device)
            else:
                reverse_body = '\t// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")\n'
        s += template_prefix + reverse_template.format(
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


def codegen_snippet(adj, name, snippet, adj_snippet, replay_snippet, forward_only=False, reverse_only=False):
    if adj.return_var is not None and len(adj.return_var) == 1:
        return_type = adj.return_var[0].ctype()
    else:
        return_type = "void"

    forward_args = []
    reverse_args = []
    forward_ref_aliases = []
    reverse_ref_aliases = []

    # Tile parameters use C++ template parameters (matching codegen_func)
    # so that the same @wp.func_native can accept tiles with any storage
    # type (register or shared).  They are passed by non-const reference
    # for the same reasons as @wp.func: owning shared tiles cannot be
    # copied, and adjoint built-ins expect non-const Tile& parameters.
    template_params = []

    # forward args
    for _i, arg in enumerate(adj.args):
        if is_tile(arg.type):
            tname = f"tile_{arg.label}"
            template_params.append(tname)
            s = f"{tname}& {arg.emit().replace('var_', '')}"
        elif is_reference(arg.type):
            label = arg.emit().replace("var_", "")
            internal = f"_wp_ref_{label}"
            s = f"{arg.ctype()} {internal}"
            forward_ref_aliases.append(f"    {Var.type_to_ctype(arg.type.value_type)}& {label} = *{internal};\n")
            reverse_ref_aliases.append(f"    {Var.type_to_ctype(arg.type.value_type)}& {label} = *{internal};\n")
        else:
            s = f"{arg.ctype()} {arg.emit().replace('var_', '')}"
        forward_args.append(s)
        reverse_args.append(s)

    # reverse args
    for _i, arg in enumerate(adj.args):
        if matches_array_class(arg.type, indexedarray):
            _arg = Var(arg.label, array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(_arg.ctype() + " & adj_" + arg.label)
        elif is_tile(arg.type):
            reverse_args.append(f"tile_{arg.label} & adj_{arg.label}")
        elif is_reference(arg.type):
            internal = f"_wp_ref_adj_{arg.label}"
            reverse_args.append(f"{arg.ctype()} {internal}")
            reverse_ref_aliases.append(
                f"    {Var.type_to_ctype(arg.type.value_type)}& adj_{arg.label} = *{internal};\n"
            )
        else:
            reverse_args.append(arg.ctype() + " & adj_" + arg.label)
    if return_type != "void":
        reverse_args.append(return_type + " & adj_ret")

    # build template prefix for snippets with tile parameters
    template_prefix = ""
    if template_params:
        template_prefix = "template<" + ", ".join(f"typename {t}" for t in template_params) + ">\n"

    forward_ref_aliases_str = "".join(forward_ref_aliases)
    reverse_ref_aliases_str = "".join(reverse_ref_aliases)

    forward_template = cuda_forward_function_template
    replay_template = cuda_forward_function_template
    reverse_template = cuda_reverse_function_template

    s = ""

    # Pass 1: Forward and replay (both are "forward-like" functions)
    if not reverse_only:
        s += template_prefix + forward_template.format(
            name=name,
            return_type=return_type,
            forward_args=indent(forward_args),
            forward_body=forward_ref_aliases_str + snippet,
            filename=adj.filename,
            lineno=adj.fun_lineno,
            line_directive="",
        )

        if replay_snippet is not None:
            s += template_prefix + replay_template.format(
                name="replay_" + name,
                return_type=return_type,
                forward_args=indent(forward_args),
                forward_body=forward_ref_aliases_str + replay_snippet,
                filename=adj.filename,
                lineno=adj.fun_lineno,
                line_directive="",
            )

    # Pass 2: Reverse/adjoint only
    if not forward_only:
        if adj_snippet:
            reverse_body = reverse_ref_aliases_str + adj_snippet
        else:
            reverse_body = reverse_ref_aliases_str

        s += template_prefix + reverse_template.format(
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


def resolve_grid_stride(kernel_options: dict, default_grid_stride: builtins.bool) -> builtins.bool:
    """Resolve a kernel's effective ``grid_stride``: an explicit ``@wp.kernel(grid_stride=...)`` choice in
    ``kernel_options`` wins, otherwise the kernel inherits the resolved ``default_grid_stride``.
    """
    # ``bool`` is shadowed by ``warp.bool`` in this module's namespace; use ``builtins.bool`` so this
    # returns a Python bool (callers cache and compare it as one), not a Warp scalar.
    explicit = kernel_options.get("grid_stride")
    return builtins.bool(default_grid_stride if explicit is None else explicit)


def codegen_kernel(kernel, device, options):
    # Update the module's options with the ones defined on the kernel, if any.
    options = options | kernel.options

    adj = kernel.adj

    args_struct = ""
    if device == "cpu":
        args_struct = f"struct wp_args_{kernel.get_mangled_name()} {{\n"
        for i in adj.args:
            args_struct += f"    {i.ctype()} {i.label};\n"
        args_struct += "};\n"

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
        if kernel.grid_stride:
            template_forward = cuda_kernel_template_forward_grid_stride
            template_backward = cuda_kernel_template_backward_grid_stride
        else:
            template_forward = cuda_kernel_template_forward
            template_backward = cuda_kernel_template_backward
    else:
        raise ValueError(f"Device {device} is not supported")

    template = ""
    template_fmt_args = {
        "name": kernel.get_mangled_name(),
        "launch_ndim": kernel.adj.kernel_dim,
    }

    # Generate launch_bounds string for CUDA kernels
    launch_bounds_str = ""
    if device == "cuda" and "launch_bounds" in options:
        launch_bounds = options["launch_bounds"]
        if isinstance(launch_bounds, int):
            launch_bounds_str = f"__launch_bounds__({launch_bounds}) "
        elif isinstance(launch_bounds, (tuple, list)):
            if len(launch_bounds) == 1:
                launch_bounds_str = f"__launch_bounds__({launch_bounds[0]}) "
            elif len(launch_bounds) == 2:
                launch_bounds_str = f"__launch_bounds__({launch_bounds[0]}, {launch_bounds[1]}) "
            else:
                raise ValueError(f"launch_bounds must be an int or a tuple/list of 1-2 ints, got {launch_bounds}")
        else:
            raise ValueError(f"launch_bounds must be an int or a tuple/list of 1-2 ints, got {type(launch_bounds)}")

    # Generate cluster_dims string for CUDA kernels.
    # 1 is the implicit default and is treated as a no-op so that
    # kernels without cluster_dim produce byte-identical source to pre-feature.
    cluster_dims_str = ""
    if device == "cuda":
        cluster_dim = options.get("cluster_dim", 1)
        if cluster_dim != 1:
            cluster_dims_str = f"WP_CLUSTER_DIMS({cluster_dim}, 1, 1) "

    # build forward signature
    forward_args = [f"wp::launch_bounds_t<{adj.kernel_dim}> dim"]
    if device == "cpu":
        forward_args.append("size_t task_index")
    else:
        for arg in adj.args:
            forward_args.append(arg.ctype() + " var_" + arg.label)

        forward_args.extend(adj.deterministic.kernel_args())

    forward_body = ""
    forward_body += adj.deterministic.kernel_locals(device)
    forward_body += codegen_func_forward(adj, func_type="kernel", device=device, grid_stride=kernel.grid_stride)
    template_fmt_args.update(
        {
            "forward_args": indent(forward_args),
            "forward_body": forward_body,
            "line_directive": func_line_directive,
            "launch_bounds_str": launch_bounds_str,
            "cluster_dims_str": cluster_dims_str,
        }
    )
    template += template_forward

    if options["enable_backward"]:
        # build reverse signature
        reverse_args = [f"wp::launch_bounds_t<{adj.kernel_dim}> dim"]
        if device == "cpu":
            reverse_args.append("size_t task_index")
        else:
            for arg in adj.args:
                reverse_args.append(arg.ctype() + " var_" + arg.label)
            for arg in adj.args:
                # indexed array gradients are regular arrays
                if matches_array_class(arg.type, indexedarray):
                    _arg = Var(arg.label, array(dtype=arg.type.dtype, ndim=arg.type.ndim))
                    reverse_args.append(_arg.ctype() + " adj_" + arg.label)
                else:
                    reverse_args.append(arg.ctype() + " adj_" + arg.label)
            reverse_args.extend(adj.deterministic.kernel_args())

        reverse_body = ""
        reverse_body += adj.deterministic.kernel_locals(device)
        reverse_body += codegen_func_reverse(adj, func_type="kernel", device=device, grid_stride=kernel.grid_stride)
        template_fmt_args.update(
            {
                "reverse_args": indent(reverse_args),
                "reverse_body": reverse_body,
            }
        )
        template += template_backward

    s = template.format(**template_fmt_args)
    return args_struct + s


def codegen_module(kernel, device, options):
    if device != "cpu":
        return ""

    # Update the module's options with the ones defined on the kernel, if any.
    options = options | kernel.options

    template = ""
    template_fmt_args = {
        "name": kernel.get_mangled_name(),
        "launch_ndim": kernel.adj.kernel_dim,
    }

    template += cpu_module_template_forward

    if options["enable_backward"]:
        template += cpu_module_template_backward

    s = template.format(**template_fmt_args)
    return s
