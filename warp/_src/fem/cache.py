# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ast
import bisect
import hashlib
import pickle
import re
import weakref
from typing import Any, Callable, ClassVar, Optional, Union

import warp as wp
from warp._src.codegen import Struct, StructInstance, get_annotations
from warp._src.fem.operator import Integrand
from warp._src.fem.types import Domain, Field
from warp._src.types import get_type_code, type_repr, type_scalar_type, type_size, type_size_in_bytes, type_to_warp
from warp._src.utils import warn

_wp_module_name_ = "warp.fem.cache"

_kernel_cache = {}
_struct_cache = {}
_func_cache = {}

_key_re = re.compile("[^0-9a-zA-Z_]+")


def _make_key(obj, suffix: Any, options: Optional[dict[str, Any]] = None):
    sorted_opts = tuple(sorted(options.items())) if options is not None else ()
    key = (
        obj.__module__,
        obj.__qualname__,
        suffix,
        sorted_opts,
    )
    return key


def _native_key(obj, key: Any):
    uid = hashlib.blake2b(pickle.dumps(key), digest_size=4).hexdigest()
    key = f"{obj.__name__}_{uid}"
    return _key_re.sub("", key)


def _arg_type_key(arg_type):
    if isinstance(arg_type, str):
        return arg_type
    if arg_type in (Field, Domain):
        return ""
    return get_type_code(type_to_warp(arg_type))


def _make_cache_key(func, key, argspec=None, allow_overloads: bool = True):
    if not allow_overloads:
        return key

    if argspec is None:
        annotations = get_annotations(func)
    else:
        annotations = argspec.annotations

    sig_key = (key, *((k, _arg_type_key(v)) for k, v in annotations.items()))
    return sig_key


def _register_function(
    func,
    key,
    module,
    **kwargs,
):
    # wp.Function will override existing func for a given key...
    # manually add back our overloads
    key = _native_key(func, key)
    existing = module.functions.get(key)
    new_fn = wp.Function(
        func=func,
        key=key,
        namespace="",
        module=module,
        **kwargs,
    )

    if existing:
        existing.add_overload(new_fn)
        module.functions[key] = existing
    return module.functions[key]


def get_func(func, suffix: Any, code_transformers=None, allow_overloads=False):
    key = _make_key(func, suffix)
    cache_key = _make_cache_key(func, key, allow_overloads=allow_overloads)

    if cache_key not in _func_cache:
        module = wp.get_module(func.__module__)
        _func_cache[cache_key] = _register_function(
            func,
            key,
            module,
            code_transformers=code_transformers,
        )

    return _func_cache[cache_key]


def dynamic_func(suffix: Any, code_transformers=None, allow_overloads=False):
    def wrap_func(func: Callable):
        return get_func(func, suffix=suffix, code_transformers=code_transformers, allow_overloads=allow_overloads)

    return wrap_func


def get_kernel(
    func,
    suffix: Any,
    kernel_options: dict[str, Any],
    allow_overloads=False,
):
    key = _make_key(func, suffix, kernel_options)
    cache_key = _make_cache_key(func, key, allow_overloads=allow_overloads)

    if cache_key not in _kernel_cache:
        kernel_key = _native_key(func, key)
        module_name = f"{func.__module__}.dyn.{kernel_key}"
        module = wp.get_module(module_name)
        module.options = wp.get_module(func.__module__).options | kernel_options
        _kernel_cache[cache_key] = wp.Kernel(func=func, key=kernel_key, module=module, options=kernel_options)

    return _kernel_cache[cache_key]


def dynamic_kernel(suffix: Any, kernel_options: Optional[dict[str, Any]] = None, allow_overloads=False):
    if kernel_options is None:
        kernel_options = {}

    def wrap_kernel(func: Callable):
        return get_kernel(func, suffix=suffix, kernel_options=kernel_options, allow_overloads=allow_overloads)

    return wrap_kernel


def get_struct(struct: type, suffix: Any):
    key = _make_key(struct, suffix)
    cache_key = key

    if cache_key not in _struct_cache:
        # used in codegen
        struct.__qualname__ = _native_key(struct, key)
        module = wp.get_module(struct.__module__)
        _struct_cache[cache_key] = Struct(
            key=struct.__qualname__,
            cls=struct,
            module=module,
        )

    return _struct_cache[cache_key]


def dynamic_struct(suffix: Any):
    def wrap_struct(struct: type):
        return get_struct(struct, suffix=suffix)

    return wrap_struct


def get_argument_struct(arg_types: dict[str, type]):
    class Args:
        pass

    annotations = get_annotations(Args)

    for name, arg_type in arg_types.items():
        setattr(Args, name, None)
        annotations[name] = arg_type

    try:
        Args.__annotations__ = annotations
    except AttributeError:
        Args.__dict__.__annotations__ = annotations

    suffix = tuple((name, _arg_type_key(arg_type)) for name, arg_type in annotations.items())
    return get_struct(Args, suffix=suffix)


def populate_argument_struct(value_struct: StructInstance, values: Optional[dict[str, Any]], func_name: str):
    if values is None:
        values = {}

    Args = value_struct._cls

    try:
        for k, v in values.items():
            setattr(value_struct, k, v)
    except Exception as err:
        if k not in Args.vars:
            raise ValueError(
                f"Passed value argument '{k}' does not match any of the function '{func_name}' parameters"
            ) from err
        raise ValueError(
            f"Passed value argument '{k}' of type '{type_repr(v)}' is incompatible with the function '{func_name}' parameter of type '{type_repr(Args.vars[k].type)}'"
        ) from err

    missing_values = Args.vars.keys() - values.keys()
    if missing_values:
        warn(
            f"Missing values for parameter(s) '{', '.join(missing_values)}' of the function '{func_name}', will be zero-initialized"
        )

    return value_struct


class ExpandStarredArgumentStruct(ast.NodeTransformer):
    def __init__(
        self,
        structs: dict[str, Struct],
    ):
        self._structs = structs

    @staticmethod
    def _build_path(path, node):
        if isinstance(node, ast.Attribute):
            ExpandStarredArgumentStruct._build_path(path, node.value)
            path.append(node.attr)
        if isinstance(node, ast.Name):
            path.append(node.id)
        return path

    def _get_expanded_struct(self, arg_node):
        if not isinstance(arg_node, ast.Starred):
            return None
        path = ".".join(ExpandStarredArgumentStruct._build_path([], arg_node.value))
        return self._structs.get(path, None)

    def visit_Call(self, call: ast.Call):
        call = self.generic_visit(call)

        expanded_args = []
        for arg in call.args:
            struct = self._get_expanded_struct(arg)
            if struct is None:
                expanded_args.append(arg)
            else:
                expanded_args += [ast.Attribute(value=arg.value, attr=field) for field in struct.vars.keys()]
        call.args = expanded_args

        return call


def get_integrand_function(
    integrand: Integrand,
    suffix: str,
    func=None,
    annotations=None,
    code_transformers=None,
):
    key = _make_key(integrand.func, suffix)

    if key not in integrand.cached_funcs:
        integrand.cached_funcs[key] = _register_function(
            func=integrand.func if func is None else func,
            key=key,
            module=integrand.module,
            overloaded_annotations=annotations,
            code_transformers=code_transformers,
        )

    return integrand.cached_funcs[key]


def get_integrand_kernel(
    integrand: Integrand,
    suffix: str,
    kernel_fn: Optional[Callable] = None,
    kernel_options: Optional[dict[str, Any]] = None,
    code_transformers=None,
    FieldStruct=None,
    ValueStruct=None,
) -> tuple[wp.Kernel, StructInstance, StructInstance]:
    options = integrand.module.options | integrand.kernel_options
    if kernel_options is not None:
        options.update(kernel_options)

    key = _make_key(integrand.func, suffix, options=options)
    if key not in integrand.cached_kernels:
        if kernel_fn is None:
            return None, None, None

        kernel_key = _native_key(integrand.func, key)
        module = wp.get_module(f"{integrand.module.name}.{kernel_key}")
        module.options = options

        integrand.cached_kernels[key] = (
            wp.Kernel(
                func=kernel_fn, key=kernel_key, module=module, code_transformers=code_transformers, options=options
            ),
            FieldStruct(),
            ValueStruct(),
        )

    return integrand.cached_kernels[key]


def pod_type_key(pod_type: type):
    """Hashable key for POD (single or sequence of scalars) types"""

    pod_type = type_to_warp(pod_type)
    if hasattr(pod_type, "_wp_scalar_type_"):
        if hasattr(pod_type, "_shape_"):
            return (pod_type.__name__, pod_type._shape_, pod_type._wp_scalar_type_.__name__)
        return (pod_type.__name__, pod_type._length_, pod_type._wp_scalar_type_.__name__)
    return pod_type.__name__


def cached_arg_value(func: Callable):
    """Decorator to be applied to member methods assembling Arg structs, so that the result gets
    automatically cached for the lifetime of the parent object
    """

    cache_attr = f"_{func.__name__}_cache"

    def get_arg(obj, device):
        cache = getattr(obj, cache_attr, None)
        if cache is None:
            cache = {}
            setattr(obj, cache_attr, cache)

        device = wp.get_device(device)
        if device.ordinal not in cache:
            cache[device.ordinal] = func(obj, device)

        return cache[device.ordinal]

    def invalidate(obj, device=None):
        if device is not None and hasattr(obj, cache_attr):
            cache = getattr(obj, cache_attr)
            if device.ordinal in cache:
                del cache[device.ordinal]
        else:
            setattr(obj, cache_attr, {})

    get_arg.invalidate = invalidate

    return get_arg


def setup_dynamic_attributes(
    obj,
    cls: Optional[type] = None,
    constructors: Optional[dict[str, Callable]] = None,
    key: Optional[str] = None,
):
    if cls is None:
        cls = type(obj)

    if key is None:
        key = obj.name

    if constructors is None:
        constructors = cls._dynamic_attribute_constructors

    key = (key, frozenset(constructors.keys()))

    if not hasattr(cls, "_cached_dynamic_attrs"):
        cls._cached_dynamic_attrs = {}

    attrs = cls._cached_dynamic_attrs.get(key)
    if attrs is None:
        attrs = {}
        # create attributes one-by-one, as some may depend on previous ones
        for k, v in constructors.items():
            attr = v(obj)
            attrs[k] = attr
            setattr(obj, k, attr)
        cls._cached_dynamic_attrs[key] = attrs
    else:
        for k, v in attrs.items():
            setattr(obj, k, v)


_cached_vec_types = {}
_cached_mat_types = {}


def cached_vec_type(length, dtype):
    key = (length, dtype)
    if key not in _cached_vec_types:
        _cached_vec_types[key] = wp.types.vector(length=length, dtype=dtype)

    return _cached_vec_types[key]


def cached_mat_type(shape, dtype):
    key = (*shape, dtype)
    if key not in _cached_mat_types:
        _cached_mat_types[key] = wp.types.matrix(shape=shape, dtype=dtype)

    return _cached_mat_types[key]


Temporary = wp.array
"""Temporary array borrowed from a :class:`TemporaryStore`.

The array will be automatically returned to the temporary pool for reuse upon destruction of this object, unless
the temporary is explicitly detached from the pool using :meth:`detach`.
The temporary may also be explicitly returned to the pool before destruction using :meth:`release`.

Note: `Temporary` is now a direct alias for `wp.array` with a custom deleter. Convenience `detach` and `release`
are added at borrow time. A self-pointing `array` attribute is also added for backward compatibility, but is
deprecated and will be removed in Warp 1.12.
"""


class TemporaryStore:
    """
    Shared pool of temporary arrays that will be persisted and reused across invocations of ``warp.fem`` functions.

    A :class:`TemporaryStore` instance may either be passed explicitly to ``warp.fem`` functions that accept such an argument, for instance :func:`.integrate.integrate`,
    or can be set globally as the default store using :func:`set_default_temporary_store`.

    By default, there is no default temporary store, so that temporary allocations are not persisted.
    """

    _default_store: ClassVar[Optional["TemporaryStore"]] = None

    class Pool:
        class Deleter:
            def __init__(self, pool: "TemporaryStore.Pool"):
                self.pool = weakref.ref(pool)

            def __call__(self, ptr, size):
                pool = self.pool()
                if pool is not None:
                    pool.redeem(ptr)

            def detach(self, temporary: Temporary):
                pool = self.pool()
                if pool is not None:
                    pool.detach(temporary)

        def __init__(self, dtype, device, pinned: bool):
            self.dtype = dtype
            self.device = device
            self.pinned = pinned

            self._pool: list[int] = []  # Currently available buffers for borrowing, ordered by size
            self._pool_capacities: list[int] = []  # Sizes of available arrays for borrowing, ascending
            self._allocs: dict[int, int] = {}  # All allocated capacities, including borrowed ones

            self._dtype_size = type_size_in_bytes(dtype)
            self._allocator = device.get_allocator(pinned=self.pinned)
            self._deleter = TemporaryStore.Pool.Deleter(self)

            # self._held_temporaries = set()  # Temporaries that are prevented from going out of scope

        def borrow(self, shape, dtype, requires_grad: bool):
            if requires_grad:
                grad = self.borrow(shape=shape, dtype=dtype, requires_grad=False)
                # Zero-out gradient to mimic semantics of wp.empty()
                grad.zero_()
            else:
                grad = None

            capacity = self._dtype_size
            if isinstance(shape, int):
                capacity *= shape
            else:
                for d in shape:
                    capacity *= d

            if capacity == 0:
                ptr = 0
                deleter = None
            else:
                index = bisect.bisect_left(
                    a=self._pool_capacities,
                    x=capacity,
                )
                if index < len(self._pool):
                    # Big enough array found, remove from pool
                    ptr = self._pool.pop(index)
                    capacity = self._pool_capacities.pop(index)
                else:
                    # No big enough array found, allocate new one
                    if len(self._pool) > 0:
                        grow_factor = 1.5
                        capacity = max(int(self._pool_capacities[-1] * grow_factor), capacity)

                    ptr = self._allocator.alloc(capacity)
                    self._allocs[ptr] = capacity
                deleter = self._deleter

            temporary = Temporary(
                ptr=ptr,
                capacity=capacity,
                shape=shape,
                dtype=dtype,
                grad=grad,
                device=self.device,
                pinned=self.pinned,
                deleter=deleter,
            )
            return temporary

        def redeem(self, ptr: int):
            capacity = self._allocs[ptr]
            # Insert back array into available pool
            index = bisect.bisect_left(
                a=self._pool_capacities,
                x=capacity,
            )
            self._pool.insert(index, ptr)
            self._pool_capacities.insert(index, capacity)

        def detach(self, array: Temporary):
            del self._allocs[array.ptr]
            array.deleter = self._allocator.deleter

        def __del__(self):
            for ptr, capacity in self._allocs.items():
                self._allocator.free(ptr, capacity)

    def __init__(self):
        self.clear()

    def clear(self):
        self._temporaries = {}

    def borrow(self, shape, dtype, pinned: bool = False, device=None, requires_grad: bool = False) -> Temporary:
        dtype = type_to_warp(dtype)
        device = wp.get_device(device)

        type_length = type_size(dtype)
        key = (dtype._type_, type_length, pinned, device.ordinal)

        try:
            pool = self._temporaries[key]
        except KeyError:
            value_type = (
                cached_vec_type(length=type_length, dtype=type_scalar_type(dtype)) if type_length > 1 else dtype
            )
            pool = TemporaryStore.Pool(value_type, device, pinned=pinned)
            self._temporaries[key] = pool

        res = TemporaryStore.add_temporary_convenience_methods(
            pool.borrow(dtype=dtype, shape=shape, requires_grad=requires_grad)
        )
        return res

    @staticmethod
    def add_temporary_convenience_methods(temporary: wp.array) -> Temporary:
        ref = weakref.ref(temporary)
        temporary.release = TemporaryStore._release_temporary.__get__(ref)
        temporary.detach = TemporaryStore._detach_temporary.__get__(ref)

        # Deprecated -- to be removed in 1.12
        temporary.array = wp.array(
            ptr=temporary.ptr,
            capacity=temporary.capacity,
            shape=temporary.shape,
            dtype=temporary.dtype,
            grad=temporary.grad,
            device=temporary.device,
            pinned=temporary.pinned,
            deleter=None,
        )

        return temporary

    @staticmethod
    def _detach_temporary(temporary_ref: "weakref.ReferenceType[Temporary]") -> Temporary:
        """Detaches the temporary so it is never returned to the pool"""
        temporary = temporary_ref()
        if temporary is None:
            return None

        if temporary.deleter is not None:
            if isinstance(temporary.deleter, TemporaryStore.Pool.Deleter):
                temporary.deleter.detach(temporary)
        return temporary

    @staticmethod
    def _release_temporary(temporary_ref: "weakref.ReferenceType[Temporary]"):
        """Returns the temporary array to the pool"""

        from warp._src.context import runtime  # noqa: PLC0415

        if runtime.tape is not None:
            # Prevent early release if a tape is being captured
            # Rely on usual garbage collection instead
            return

        temporary = temporary_ref()
        if temporary is None:
            return

        if temporary.deleter is not None:
            with temporary.device.context_guard:
                temporary.deleter(temporary.ptr, temporary.capacity)
            temporary.deleter = None


def set_default_temporary_store(temporary_store: Optional[TemporaryStore]):
    """Globally sets the default :class:`TemporaryStore` instance to use for temporary allocations in ``warp.fem`` functions.

    If the default temporary store is set to ``None``, temporary allocations are not persisted unless a :class:`TemporaryStore` is provided at a per-function granularity.
    """

    TemporaryStore._default_store = temporary_store


def borrow_temporary(
    temporary_store: Optional[TemporaryStore],
    shape: Union[int, tuple[int]],
    dtype: type,
    pinned: bool = False,
    requires_grad: bool = False,
    device=None,
) -> Temporary:
    """
    Borrows and returns a temporary array with specified attributes from a shared pool.

    If an array with sufficient capacity and matching desired attributes is already available in the pool, it will be returned.
    Otherwise, a new allocation will be performed.

    Args:
        temporary_store: the shared pool to borrow the temporary from. If `temporary_store` is ``None``, the global default temporary store, if set, will be used.
        shape: desired dimensions for the temporary array
        dtype: desired data type for the temporary array
        pinned: whether a pinned allocation is desired
        device: device on which the memory should be allocated; if ``None``, the current device will be used.
    """

    if temporary_store is None:
        temporary_store = TemporaryStore._default_store

    if temporary_store is None:
        return TemporaryStore.add_temporary_convenience_methods(
            Temporary(shape=shape, dtype=dtype, pinned=pinned, device=device, requires_grad=requires_grad)
        )

    return temporary_store.borrow(shape=shape, dtype=dtype, device=device, pinned=pinned, requires_grad=requires_grad)


def borrow_temporary_like(
    array: Union[wp.array, Temporary],
    temporary_store: Optional[TemporaryStore],
) -> Temporary:
    """
    Borrows and returns a temporary array with the same attributes as another array or temporary.

    Args:
        array: Warp or temporary array to read the desired attributes from
        temporary_store: the shared pool to borrow the temporary from. If `temporary_store` is ``None``, the global default temporary store, if set, will be used.
    """
    return borrow_temporary(
        temporary_store=temporary_store,
        shape=array.shape,
        dtype=array.dtype,
        pinned=array.pinned,
        device=array.device,
        requires_grad=array.requires_grad,
    )


_device_events = {}


def capture_event(device=None):
    """
    Records a CUDA event on the current stream and returns it,
    reusing previously created events if possible.

    If the current device is not a CUDA device, returns ``None``.

    The event can be returned to the shared per-device pool for future reuse by
    calling :func:`synchronize_event`
    """

    device = wp.get_device(device)
    if not device.is_cuda:
        return None

    try:
        device_events = _device_events[device.ordinal]
    except KeyError:
        device_events = []
        _device_events[device.ordinal] = device_events

    with wp.ScopedDevice(device):
        if not device_events:
            return wp.record_event()

        return wp.record_event(device_events.pop())


def synchronize_event(event: Union[wp.Event, None]):
    """
    Synchronize an event created with :func:`capture_event` and returns it to the
    per-device event pool.

    If `event` is ``None``, do nothing.
    """

    if event is not None:
        wp.synchronize_event(event)
        _device_events[event.device.ordinal].append(event)
