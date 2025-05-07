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
import inspect
import re
import weakref
from typing import Any, Callable, Dict, Optional, Tuple, Union

import warp as wp
from warp.fem.operator import Integrand
from warp.fem.types import Domain, Field

_kernel_cache = {}
_struct_cache = {}
_func_cache = {}

_key_re = re.compile("[^0-9a-zA-Z_]+")


def _make_key(obj, suffix: str, options: Optional[Dict[str, Any]] = None):
    # human-readable part
    key = _key_re.sub("", f"{obj.__name__}_{suffix}")

    opts_key = "".join([f"{k}:{v}" for k, v in sorted(options.items())]) if options is not None else ""
    uid = hashlib.blake2b(
        bytes(f"{obj.__module__}{obj.__qualname__}{suffix}{opts_key}", encoding="utf-8"), digest_size=4
    ).hexdigest()

    # avoid long keys, issues on win
    key = f"{key[:64]}_{uid}"

    return key


def _arg_type_name(arg_type):
    if arg_type in (Field, Domain):
        return ""
    return wp.types.get_type_code(wp.types.type_to_warp(arg_type))


def _make_cache_key(func, key, argspec=None):
    if argspec is None:
        argspec = inspect.getfullargspec(func)

    sig_key = "".join([f"{k}:{_arg_type_name(v)}" for k, v in argspec.annotations.items()])
    return key + sig_key


def _register_function(
    func,
    key,
    module,
    **kwargs,
):
    # wp.Function will override existing func for a given key...
    # manually add back our overloads
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


def get_func(func, suffix: str, code_transformers=None):
    key = _make_key(func, suffix)
    cache_key = _make_cache_key(func, key)

    if cache_key not in _func_cache:
        module = wp.get_module(func.__module__)
        _func_cache[cache_key] = _register_function(
            func,
            key,
            module,
            code_transformers=code_transformers,
        )

    return _func_cache[cache_key]


def dynamic_func(suffix: str, code_transformers=None):
    def wrap_func(func: Callable):
        return get_func(func, suffix=suffix, code_transformers=code_transformers)

    return wrap_func


def get_kernel(
    func,
    suffix: str,
    kernel_options: Optional[Dict[str, Any]] = None,
):
    if kernel_options is None:
        kernel_options = {}

    key = _make_key(func, suffix, kernel_options)
    cache_key = _make_cache_key(func, key)

    if cache_key not in _kernel_cache:
        module_name = f"{func.__module__}.dyn.{key}"
        module = wp.get_module(module_name)
        module.options = dict(wp.get_module(func.__module__).options)
        module.options.update(kernel_options)
        _kernel_cache[cache_key] = wp.Kernel(func=func, key=key, module=module, options=kernel_options)
    return _kernel_cache[cache_key]


def dynamic_kernel(suffix: str, kernel_options: Optional[Dict[str, Any]] = None):
    if kernel_options is None:
        kernel_options = {}

    def wrap_kernel(func: Callable):
        return get_kernel(func, suffix=suffix, kernel_options=kernel_options)

    return wrap_kernel


def get_struct(struct: type, suffix: str):
    key = _make_key(struct, suffix)
    # used in codegen
    struct.__qualname__ = key

    if key not in _struct_cache:
        module = wp.get_module(struct.__module__)
        _struct_cache[key] = wp.codegen.Struct(
            key=key,
            cls=struct,
            module=module,
        )

    return _struct_cache[key]


def dynamic_struct(suffix: str):
    def wrap_struct(struct: type):
        return get_struct(struct, suffix=suffix)

    return wrap_struct


def get_argument_struct(arg_types: Dict[str, type]):
    class Args:
        pass

    annotations = wp.codegen.get_annotations(Args)

    for name, arg_type in arg_types.items():
        setattr(Args, name, None)
        annotations[name] = arg_type

    try:
        Args.__annotations__ = annotations
    except AttributeError:
        Args.__dict__.__annotations__ = annotations

    suffix = "_".join([f"{name}_{_arg_type_name(arg_type)}" for name, arg_type in annotations.items()])

    return get_struct(Args, suffix=suffix)


def populate_argument_struct(Args: wp.codegen.Struct, values: Dict[str, Any], func_name: str):
    if values is None:
        values = {}

    value_struct_values = Args()
    for k, v in values.items():
        try:
            setattr(value_struct_values, k, v)
        except Exception as err:
            if k not in Args.vars:
                raise ValueError(
                    f"Passed value argument '{k}' does not match any of the function '{func_name}' parameters"
                ) from err
            raise ValueError(
                f"Passed value argument '{k}' of type '{wp.types.type_repr(v)}' is incompatible with the function '{func_name}' parameter of type '{wp.types.type_repr(Args.vars[k].type)}'"
            ) from err

    missing_values = Args.vars.keys() - values.keys()
    if missing_values:
        wp.utils.warn(
            f"Missing values for parameter(s) '{', '.join(missing_values)}' of the function '{func_name}', will be zero-initialized"
        )

    return value_struct_values


class ExpandStarredArgumentStruct(ast.NodeTransformer):
    def __init__(
        self,
        structs: Dict[str, wp.codegen.Struct],
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
    cache_key = _make_cache_key(integrand.func, key, integrand.argspec)

    if cache_key not in _func_cache:
        _func_cache[cache_key] = _register_function(
            func=integrand.func if func is None else func,
            key=key,
            module=integrand.module,
            overloaded_annotations=annotations,
            code_transformers=code_transformers,
        )

    return _func_cache[cache_key]


def get_integrand_kernel(
    integrand: Integrand,
    suffix: str,
    kernel_fn: Optional[Callable] = None,
    kernel_options: Optional[Dict[str, Any]] = None,
    code_transformers=None,
):
    options = integrand.module.options.copy()
    options.update(integrand.kernel_options)
    if kernel_options is not None:
        options.update(kernel_options)

    kernel_key = _make_key(integrand.func, suffix, options=options)
    cache_key = _make_cache_key(integrand, kernel_key, integrand.argspec)

    if cache_key not in _kernel_cache:
        if kernel_fn is None:
            return None

        module = wp.get_module(f"{integrand.module.name}.{kernel_key}")
        module.options = options
        _kernel_cache[cache_key] = wp.Kernel(
            func=kernel_fn, key=kernel_key, module=module, code_transformers=code_transformers, options=options
        )
    return _kernel_cache[cache_key]


def cached_arg_value(func: Callable):
    """Decorator to be applied to member methods assembling Arg structs, so that the result gets
    automatically cached for the lifetime of the parent object
    """

    cache_attr = f"_{func.__name__}_cache"

    def get_arg(obj, device):
        if not hasattr(obj, cache_attr):
            setattr(obj, cache_attr, {})

        cache = getattr(obj, cache_attr, {})

        device = wp.get_device(device)
        if device.ordinal not in cache:
            cache[device.ordinal] = func(obj, device)

        return cache[device.ordinal]

    return get_arg


_cached_vec_types = {}
_cached_mat_types = {}


def cached_vec_type(length, dtype):
    key = (length, dtype)
    if key not in _cached_vec_types:
        _cached_vec_types[key] = wp.vec(length=length, dtype=dtype)

    return _cached_vec_types[key]


def cached_mat_type(shape, dtype):
    key = (*shape, dtype)
    if key not in _cached_mat_types:
        _cached_mat_types[key] = wp.mat(shape=shape, dtype=dtype)

    return _cached_mat_types[key]


class Temporary:
    """Handle over a temporary array from a :class:`TemporaryStore`.

    The array will be automatically returned to the temporary pool for reuse upon destruction of this object, unless
    the temporary is explicitly detached from the pool using :meth:`detach`.
    The temporary may also be explicitly returned to the pool before destruction using :meth:`release`.
    """

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._pool = None
        return instance

    def __init__(self, array: wp.array, pool: Optional["TemporaryStore.Pool"] = None, shape=None, dtype=None):
        self._raw_array = array
        self._array_view = array
        self._pool = pool

        if pool is not None and wp.context.runtime.tape is not None:
            # Extend lifetime to that of Tape (or Pool if shorter)
            # This is to prevent temporary arrays held in tape launch parameters to be redeemed
            pool.hold(self)
            weakref.finalize(wp.context.runtime.tape, TemporaryStore.Pool.stop_holding, pool, self)

        if shape is not None or dtype is not None:
            self._view_as(shape=shape, dtype=dtype)

    def detach(self) -> wp.array:
        """Detaches the temporary so it is never returned to the pool"""
        if self._pool is not None:
            self._pool.detach(self._raw_array)

        self._pool = None
        return self._array_view

    def release(self):
        """Returns the temporary array to the pool"""
        if self._pool is not None:
            self._pool.redeem(self._raw_array)

        self._pool = None

    @property
    def array(self) -> wp.array:
        """View of the array with desired shape and data type."""
        return self._array_view

    def _view_as(self, shape, dtype) -> "Temporary":
        def _view_reshaped_truncated(array):
            view = wp.types.array(
                ptr=array.ptr,
                dtype=dtype,
                shape=shape,
                device=array.device,
                pinned=array.pinned,
                capacity=array.capacity,
                copy=False,
                grad=None if array.grad is None else _view_reshaped_truncated(array.grad),
            )
            view._ref = array
            return view

        self._array_view = _view_reshaped_truncated(self._raw_array)
        return self

    def __del__(self):
        self.release()


class TemporaryStore:
    """
    Shared pool of temporary arrays that will be persisted and reused across invocations of ``warp.fem`` functions.

    A :class:`TemporaryStore` instance may either be passed explicitly to ``warp.fem`` functions that accept such an argument, for instance :func:`.integrate.integrate`,
    or can be set globally as the default store using :func:`set_default_temporary_store`.

    By default, there is no default temporary store, so that temporary allocations are not persisted.
    """

    _default_store: "TemporaryStore" = None

    class Pool:
        def __init__(self, dtype, device, pinned: bool):
            self.dtype = dtype
            self.device = device
            self.pinned = pinned

            self._pool = []  # Currently available arrays for borrowing, ordered by size
            self._pool_sizes = []  # Sizes of available arrays for borrowing, ascending
            self._allocs = {}  # All allocated arrays, including borrowed ones

            self._held_temporaries = set()  # Temporaries that are prevented from going out of scope

        def borrow(self, shape, dtype, requires_grad: bool):
            size = 1
            if isinstance(shape, int):
                shape = (shape,)
            for d in shape:
                size *= d

            index = bisect.bisect_left(
                a=self._pool_sizes,
                x=size,
            )
            if index < len(self._pool):
                # Big enough array found, remove from pool
                array = self._pool.pop(index)
                self._pool_sizes.pop(index)
                if requires_grad:
                    if array.grad is None:
                        array.requires_grad = True
                    else:
                        # Zero-out existing gradient to mimic semantics of wp.empty()
                        array.grad.zero_()
                return Temporary(pool=self, array=array, shape=shape, dtype=dtype)

            # No big enough array found, allocate new one
            if len(self._pool) > 0:
                grow_factor = 1.5
                size = max(int(self._pool_sizes[-1] * grow_factor), size)

            array = wp.empty(
                shape=(size,), dtype=self.dtype, pinned=self.pinned, device=self.device, requires_grad=requires_grad
            )
            self._allocs[array.ptr] = array
            return Temporary(pool=self, array=array, shape=shape, dtype=dtype)

        def redeem(self, array):
            # Insert back array into available pool
            index = bisect.bisect_left(
                a=self._pool_sizes,
                x=array.size,
            )
            self._pool.insert(index, array)
            self._pool_sizes.insert(index, array.size)

        def detach(self, array):
            del self._allocs[array.ptr]

        def hold(self, temp: Temporary):
            self._held_temporaries.add(temp)

        def stop_holding(self, temp: Temporary):
            self._held_temporaries.remove(temp)

    def __init__(self):
        self.clear()

    def clear(self):
        self._temporaries = {}

    def borrow(self, shape, dtype, pinned: bool = False, device=None, requires_grad: bool = False) -> Temporary:
        dtype = wp.types.type_to_warp(dtype)
        device = wp.get_device(device)

        type_length = wp.types.type_length(dtype)
        key = (dtype._type_, type_length, pinned, device.ordinal)

        pool = self._temporaries.get(key, None)
        if pool is None:
            value_type = (
                cached_vec_type(length=type_length, dtype=wp.types.type_scalar_type(dtype))
                if type_length > 1
                else dtype
            )
            pool = TemporaryStore.Pool(value_type, device, pinned=pinned)
            self._temporaries[key] = pool

        return pool.borrow(dtype=dtype, shape=shape, requires_grad=requires_grad)


def set_default_temporary_store(temporary_store: Optional[TemporaryStore]):
    """Globally sets the default :class:`TemporaryStore` instance to use for temporary allocations in ``warp.fem`` functions.

    If the default temporary store is set to ``None``, temporary allocations are not persisted unless a :class:`TemporaryStore` is provided at a per-function granularity.
    """

    TemporaryStore._default_store = temporary_store


def borrow_temporary(
    temporary_store: Optional[TemporaryStore],
    shape: Union[int, Tuple[int]],
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

    if temporary_store is None or (requires_grad and wp.context.runtime.tape is not None):
        return Temporary(
            array=wp.empty(shape=shape, dtype=dtype, pinned=pinned, device=device, requires_grad=requires_grad)
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
    if isinstance(array, Temporary):
        array = array.array
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
