# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation of the experimental external-compilation registration API."""

import ctypes as _ctypes
import hashlib as _hashlib
import re as _re
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass

from warp._src.context import Function as _Function
from warp._src.context import add_builtin as _add_builtin
from warp._src.context import builtin_functions as _builtin_functions
from warp._src.types import _native_value_types as _native_value_types
from warp._src.types import get_type_code as _get_type_code
from warp._src.types import type_repr as _type_repr
from warp._src.types import type_size_in_bytes as _type_size_in_bytes
from warp._src.types import type_to_warp as _type_to_warp

__all__ = ["add_builtin", "add_native_type"]

_NATIVE_NAME_RE = _re.compile(r"^(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*$")
_native_types_by_name = {}


@_dataclass(frozen=True)
class _NativeTypeInfo:
    native_name: str
    fields: tuple[tuple[str, type], ...] | None
    initializer: str | None
    size: int
    alignment: int
    field_offsets: tuple[tuple[str, int], ...]
    type_code: str


def _native_type_info_contract(info: _NativeTypeInfo) -> tuple:
    """Return reload-stable native type metadata for compatibility checks."""
    fields = None
    if info.fields is not None:
        fields = tuple((name, _get_type_code(dtype)) for name, dtype in info.fields)
    return (
        info.native_name,
        fields,
        info.initializer,
        info.size,
        info.alignment,
        info.field_offsets,
        info.type_code,
    )


def _split_native_name(native_name: str) -> tuple[str, str]:
    """Split an exact qualified C++ function name into namespace and function."""
    global_qualified = native_name.startswith("::")
    parts = native_name.removeprefix("::").split("::")
    function_name = parts.pop()
    namespace = "::".join(parts)
    if namespace:
        namespace += "::"
    if global_qualified:
        namespace = "::" + namespace
    return namespace, function_name


def _type_contract_code(dtype: type | None) -> str:
    """Return the stable Warp-visible identity of an extension value type."""
    if dtype is None:
        return "void"
    return _get_type_code(_type_to_warp(dtype))


def _signature_contract(input_types: _Mapping[str, type], value_type: type | None) -> tuple:
    """Return an ordered, reload-stable external builtin signature."""
    inputs = tuple((param, _type_contract_code(dtype)) for param, dtype in input_types.items())
    return inputs, _type_contract_code(value_type)


def add_builtin(
    name: str,
    input_types: _Mapping[str, type] | None = None,
    value_type: type | None = None,
    *,
    native_name: str | None = None,
    doc: str = "",
) -> _Function:
    """Register an external C++ function for use in Warp kernels.

    Experimental: this API may change without deprecation in future releases.

    The registered function is available in kernels as ``wp.<name>``. Calling
    it directly from Python is not supported.

    Args:
        name: Name used to call the function from Warp kernels.
        input_types: Ordered mapping from parameter names to Warp types.
        value_type: Warp return type, or ``None`` for a function returning
            ``void``.
        native_name: Exact qualified C++ function name. When omitted,
            ``wp::<name>`` is used.
        doc: Description of the function.

    Returns:
        The canonical Warp function containing the registered overload.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"name must be a valid Python identifier, got {name!r}")
    if input_types is None:
        input_types = {}
    if not isinstance(input_types, _Mapping):
        raise TypeError(f"input_types must be a mapping or None, got {type(input_types).__name__}")
    if not all(isinstance(param, str) and param.isidentifier() for param in input_types):
        raise ValueError("input_types keys must be valid Python identifiers")
    if not isinstance(doc, str):
        raise TypeError(f"doc must be a str, got {type(doc).__name__}")

    if native_name is None:
        native_name = f"wp::{name}"
    if not isinstance(native_name, str) or not _NATIVE_NAME_RE.fullmatch(native_name):
        raise ValueError(f"native_name must be a qualified C++ identifier, got {native_name!r}")

    canonical_input_types = {param: _type_to_warp(dtype) for param, dtype in input_types.items()}
    canonical_value_type = _type_to_warp(value_type) if value_type is not None else None
    signature = _signature_contract(canonical_input_types, canonical_value_type)
    namespace, native_func = _split_native_name(native_name)
    external_contract = (signature, namespace, native_func)

    existing = _builtin_functions.get(name)
    if existing is not None:
        for overload in existing.overloads:
            overload_signature = _signature_contract(overload.input_types, overload.value_type)
            if overload_signature[0] == signature[0]:
                if (
                    overload_signature[1] == signature[1]
                    and overload.namespace == namespace
                    and overload.native_func == native_func
                ):
                    overload._external_builtin_contract = external_contract
                    existing._has_external_builtin_contract = True
                    return existing
                raise RuntimeError(
                    f"Cannot register conflicting external builtin overload '{name}' "
                    f"with input types {canonical_input_types!r}"
                )

    _add_builtin(
        name,
        input_types=dict(input_types),
        value_type=value_type,
        doc=doc,
        namespace=namespace,
        export=False,
        hidden=True,
        is_differentiable=False,
        native_func=native_func,
    )
    function = _builtin_functions[name]
    for overload in function.overloads:
        if (
            _signature_contract(overload.input_types, overload.value_type) == signature
            and overload.namespace == namespace
            and overload.native_func == native_func
        ):
            overload._external_builtin_contract = external_contract
            function._has_external_builtin_contract = True
            break
    return function


def add_native_type(
    ctype: type[_ctypes.Structure],
    *,
    native_name: str,
    fields: _Mapping[str, type] | None = None,
    initializer: str | None = None,
) -> type[_ctypes.Structure]:
    """Register an external C++ value type for use in Warp kernels.

    Experimental: this API may change without deprecation in future releases.

    The Python type must be a :class:`ctypes.Structure` with the same size,
    alignment, and field layout as the C++ type. Warp verifies the size and
    alignment during compilation. CPU compilation also verifies exposed field
    offsets; CUDA compilation verifies exposed field sizes. The C++ type must
    be standard-layout and trivially copyable.

    Args:
        ctype: Host ABI description and Python value class.
        native_name: Exact qualified C++ type name.
        fields: Ordered mapping of public C++ data members to Warp types.
            Omit this for an opaque value type.
        initializer: Set to ``"aggregate"`` to allow construction from all
            fields in kernels. In this mode, ``fields`` must list every ctypes
            field in declaration order. Default construction is always
            allowed.

    Returns:
        ``ctype``, which is used directly as the Warp annotation and array
        ``dtype``.
    """
    if not isinstance(ctype, type) or not issubclass(ctype, _ctypes.Structure):
        raise TypeError(f"ctype must be a ctypes.Structure subclass, got {ctype!r}")
    if not isinstance(native_name, str) or not _NATIVE_NAME_RE.fullmatch(native_name):
        raise ValueError(f"native_name must be a qualified C++ identifier, got {native_name!r}")
    if fields is not None and not isinstance(fields, _Mapping):
        raise TypeError(f"fields must be a mapping or None, got {type(fields).__name__}")
    if initializer not in (None, "aggregate"):
        raise ValueError("initializer must be None or 'aggregate'")
    if initializer == "aggregate" and fields is None:
        raise ValueError("initializer='aggregate' requires exposed fields")

    declared_fields = {}
    for field in getattr(ctype, "_fields_", ()):
        if len(field) != 2:
            raise TypeError("Bit-field ctypes layouts are not supported for native value types")
        declared_fields[field[0]] = field[1]

    canonical_fields = None
    field_offsets = ()
    if fields is not None:
        canonical = []
        offsets = []
        for name, dtype in fields.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"fields keys must be valid C++ identifiers, got {name!r}")
            if name not in declared_fields:
                raise ValueError(f"Exposed field {name!r} is not present in {ctype.__name__}._fields_")
            canonical_dtype = _type_to_warp(dtype)
            try:
                dtype_size = _type_size_in_bytes(canonical_dtype)
                _ = _get_type_code(canonical_dtype)
            except Exception as e:
                raise TypeError(f"Invalid Warp type for native field {name!r}: {_type_repr(canonical_dtype)}") from e
            storage_size = _ctypes.sizeof(declared_fields[name])
            if dtype_size != storage_size:
                raise ValueError(
                    f"Native field {name!r} maps to {_type_repr(canonical_dtype)} ({dtype_size} bytes), "
                    f"but its ctypes storage is {storage_size} bytes"
                )
            canonical.append((name, canonical_dtype))
            offsets.append((name, getattr(ctype, name).offset))
        canonical_fields = tuple(canonical)
        field_offsets = tuple(offsets)

    if initializer == "aggregate":
        declared_field_names = tuple(declared_fields)
        exposed_field_names = tuple(name for name, _ in canonical_fields)
        if exposed_field_names != declared_field_names:
            raise ValueError(
                "initializer='aggregate' requires fields to list every ctypes field in declaration order; "
                f"expected {declared_field_names!r}, got {exposed_field_names!r}"
            )

    schema = _hashlib.sha256()
    schema.update(native_name.encode())
    schema.update(f":{_ctypes.sizeof(ctype)}:{_ctypes.alignment(ctype)}:{initializer}".encode())
    if canonical_fields is not None:
        for (name, dtype), (_, offset) in zip(canonical_fields, field_offsets, strict=True):
            schema.update(f":{name}:{_get_type_code(dtype)}:{offset}".encode())

    info = _NativeTypeInfo(
        native_name=native_name,
        fields=canonical_fields,
        initializer=initializer,
        size=_ctypes.sizeof(ctype),
        alignment=_ctypes.alignment(ctype),
        field_offsets=field_offsets,
        type_code=schema.hexdigest()[:16],
    )

    existing = getattr(ctype, "_wp_native_type_", None)
    if existing is not None:
        if _native_type_info_contract(existing) == _native_type_info_contract(info):
            _native_value_types.add(ctype)
            return ctype
        raise RuntimeError(f"Cannot register conflicting definitions for native type {ctype.__name__!r}")

    named_type = _native_types_by_name.get(native_name)
    if named_type is not None and named_type is not ctype:
        if _native_type_info_contract(named_type._wp_native_type_) != _native_type_info_contract(info):
            raise RuntimeError(
                f"Native C++ type name {native_name!r} is already registered with a different definition"
            )

    # Field metadata deliberately follows the same small protocol as Warp
    # structs, allowing existing attribute code generation to stay generic.
    from warp._src.codegen import Var as _Var  # noqa: PLC0415

    ctype._wp_native_type_ = info
    ctype._wp_native_vars_ = (
        {} if canonical_fields is None else {name: _Var(name, dtype) for name, dtype in canonical_fields}
    )
    _native_value_types.add(ctype)
    _native_types_by_name.setdefault(native_name, ctype)
    return ctype
