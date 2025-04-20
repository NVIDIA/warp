# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import traceback
from typing import Callable

import jax

import warp as wp
from warp.codegen import get_full_arg_spec, make_full_qualified_name
from warp.jax import get_jax_device
from warp.types import array_t, launch_bounds_t, strides_from_shape, type_to_warp

from .xla_ffi import *


def jax_kernel(kernel, num_outputs=1, vmap_method="broadcast_all", launch_dims=None, output_dims=None):
    """Create a JAX callback from a Warp kernel.

    NOTE: This is an experimental feature under development.

    Args:
        kernel: The Warp kernel to launch.
        num_outputs: Optional. Specify the number of output arguments if greater than 1.
        vmap_method: Optional. String specifying how the callback transforms under ``vmap()``.
                     This argument can also be specified for individual calls.
        launch_dims: Optional. Specify the default kernel launch dimensions. If None, launch
                     dimensions are inferred from the shape of the first array argument.
                     This argument can also be specified for individual calls.
        output_dims: Optional. Specify the default dimensions of output arrays.  If None, output
                     dimensions are inferred from the launch dimensions.
                     This argument can also be specified for individual calls.

    Limitations:
        - All kernel arguments must be contiguous arrays or scalars.
        - Scalars must be static arguments in JAX.
        - Input arguments are followed by output arguments in the Warp kernel definition.
        - There must be at least one output argument.
        - Only the CUDA backend is supported.
    """

    return FfiKernel(kernel, num_outputs, vmap_method, launch_dims, output_dims)


def jax_callable(
    func: Callable,
    num_outputs: int = 1,
    graph_compatible: bool = True,
    vmap_method: str = "broadcast_all",
    output_dims=None,
):
    """Create a JAX callback from an annotated Python function.

    The Python function arguments must have type annotations like Warp kernels.

    NOTE: This is an experimental feature under development.

    Args:
        func: The Python function to call.
        num_outputs: Optional. Specify the number of output arguments if greater than 1.
        graph_compatible: Optional. Whether the function can be called during CUDA graph capture.
        vmap_method: Optional. String specifying how the callback transforms under ``vmap()``.
            This argument can also be specified for individual calls.
        output_dims: Optional. Specify the default dimensions of output arrays.
            If ``None``, output dimensions are inferred from the launch dimensions.
            This argument can also be specified for individual calls.

    Limitations:
        - All kernel arguments must be contiguous arrays or scalars.
        - Scalars must be static arguments in JAX.
        - Input arguments are followed by output arguments in the Warp kernel definition.
        - There must be at least one output argument.
        - Only the CUDA backend is supported.
    """

    return FfiCallable(func, num_outputs, graph_compatible, vmap_method, output_dims)


class FfiArg:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.is_array = isinstance(type, wp.array)

        if self.is_array:
            if hasattr(type.dtype, "_wp_scalar_type_"):
                self.dtype_shape = type.dtype._shape_
                self.dtype_ndim = len(self.dtype_shape)
                self.jax_scalar_type = wp.dtype_to_jax(type.dtype._wp_scalar_type_)
                self.jax_ndim = type.ndim + self.dtype_ndim
            elif type.dtype in wp.types.value_types:
                self.dtype_ndim = 0
                self.dtype_shape = ()
                self.jax_scalar_type = wp.dtype_to_jax(type.dtype)
                self.jax_ndim = type.ndim
            else:
                raise TypeError(f"Invalid data type for array argument '{name}', expected scalar, vector, or matrix")
            self.warp_ndim = type.ndim
        elif type in wp.types.value_types:
            self.dtype_ndim = 0
            self.dtype_shape = ()
            self.jax_scalar_type = wp.dtype_to_jax(type_to_warp(type))
            self.jax_ndim = 0
            self.warp_ndim = 0
        else:
            raise TypeError(f"Invalid type for argument '{name}', expected array or scalar, got {type}")


class FfiLaunchDesc:
    def __init__(self, static_inputs, launch_dims):
        self.static_inputs = static_inputs
        self.launch_dims = launch_dims


class FfiKernel:
    def __init__(self, kernel, num_outputs, vmap_method, launch_dims, output_dims):
        self.kernel = kernel
        self.name = generate_unique_name(kernel.func)
        self.num_outputs = num_outputs
        self.vmap_method = vmap_method
        self.launch_dims = launch_dims
        self.output_dims = output_dims
        self.first_array_arg = None
        self.launch_id = 0
        self.launch_descriptors = {}

        self.num_kernel_args = len(kernel.adj.args)
        self.num_inputs = self.num_kernel_args - num_outputs
        if self.num_outputs < 1:
            raise ValueError("At least one output is required")
        if self.num_outputs > self.num_kernel_args:
            raise ValueError("Number of outputs cannot be greater than the number of kernel arguments")

        # process input args
        self.input_args = []
        for i in range(self.num_inputs):
            arg = FfiArg(kernel.adj.args[i].label, kernel.adj.args[i].type)
            if arg.is_array:
                # keep track of the first input array argument
                if self.first_array_arg is None:
                    self.first_array_arg = i
            self.input_args.append(arg)

        # process output args
        self.output_args = []
        for i in range(self.num_inputs, self.num_kernel_args):
            arg = FfiArg(kernel.adj.args[i].label, kernel.adj.args[i].type)
            if not arg.is_array:
                raise TypeError("All output arguments must be arrays")
            self.output_args.append(arg)

        # register the callback
        FFI_CCALLFUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_CallFrame))
        self.callback_func = FFI_CCALLFUNC(lambda call_frame: self.ffi_callback(call_frame))
        ffi_ccall_address = ctypes.cast(self.callback_func, ctypes.c_void_p)
        ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)
        jax.ffi.register_ffi_target(self.name, ffi_capsule, platform="CUDA")

    def __call__(self, *args, output_dims=None, launch_dims=None, vmap_method=None):
        num_inputs = len(args)
        if num_inputs != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, but got {num_inputs}")

        # default argument fallback
        if launch_dims is None:
            launch_dims = self.launch_dims
        if output_dims is None:
            output_dims = self.output_dims
        if vmap_method is None:
            vmap_method = self.vmap_method

        # process inputs
        static_inputs = {}
        for i in range(num_inputs):
            input_arg = self.input_args[i]
            input_value = args[i]
            if input_arg.is_array:
                # check dtype
                if input_value.dtype != input_arg.jax_scalar_type:
                    raise TypeError(
                        f"Invalid data type for array argument '{input_arg.name}', expected {input_arg.jax_scalar_type}, got {input_value.dtype}"
                    )
                # check ndim
                if input_value.ndim != input_arg.jax_ndim:
                    raise TypeError(
                        f"Invalid dimensionality for array argument '{input_arg.name}', expected {input_arg.jax_ndim} dimensions, got {input_value.ndim}"
                    )
                # check inner dims
                for d in range(input_arg.dtype_ndim):
                    if input_value.shape[input_arg.type.ndim + d] != input_arg.dtype_shape[d]:
                        raise TypeError(
                            f"Invalid inner dimensions for array argument '{input_arg.name}', expected {input_arg.dtype_shape}, got {input_value.shape[-input_arg.dtype_ndim :]}"
                        )
            else:
                # make sure scalar is not a traced variable, should be static
                if isinstance(input_value, jax.core.Tracer):
                    raise ValueError(f"Argument '{input_arg.name}' must be a static value")
                # stash the value to be retrieved by callback
                static_inputs[input_arg.name] = input_arg.type(input_value)

        # launch dimensions
        if launch_dims is None:
            # use the shape of the first input array
            if self.first_array_arg is not None:
                launch_dims = get_warp_shape(self.input_args[self.first_array_arg], args[self.first_array_arg].shape)
            else:
                raise RuntimeError("Failed to determine launch dimensions")
        elif isinstance(launch_dims, int):
            launch_dims = (launch_dims,)
        else:
            launch_dims = tuple(launch_dims)

        # output types
        out_types = []
        if isinstance(output_dims, dict):
            # assume a dictionary of shapes keyed on argument name
            for output_arg in self.output_args:
                dims = output_dims.get(output_arg.name)
                if dims is None:
                    raise ValueError(f"Missing output dimensions for argument '{output_arg.name}'")
                out_types.append(get_jax_output_type(output_arg, dims))
        else:
            if output_dims is None:
                # use launch dimensions
                output_dims = launch_dims
            elif isinstance(output_dims, int):
                output_dims = (output_dims,)
            # assume same dimensions for all outputs
            for output_arg in self.output_args:
                out_types.append(get_jax_output_type(output_arg, output_dims))

        call = jax.ffi.ffi_call(
            self.name,
            out_types,
            vmap_method=vmap_method,
        )

        # ensure the kernel module is loaded before the callback, otherwise graph capture may fail
        device = wp.device_from_jax(get_jax_device())
        self.kernel.module.load(device)

        # save launch data to be retrieved by callback
        launch_id = self.launch_id
        self.launch_descriptors[launch_id] = FfiLaunchDesc(static_inputs, launch_dims)
        self.launch_id += 1

        return call(*args, launch_id=launch_id)

    def ffi_callback(self, call_frame):
        try:
            # On the first call, XLA runtime will query the API version and traits
            # metadata using the |extension| field. Let us respond to that query
            # if the metadata extension is present.
            extension = call_frame.contents.extension_start
            if extension:
                # Try to set the version metadata.
                if extension.contents.type == XLA_FFI_Extension_Type.Metadata:
                    metadata_ext = ctypes.cast(extension, ctypes.POINTER(XLA_FFI_Metadata_Extension))
                    metadata_ext.contents.metadata.contents.api_version.major_version = 0
                    metadata_ext.contents.metadata.contents.api_version.minor_version = 1
                    # Turn on CUDA graphs for this handler.
                    metadata_ext.contents.metadata.contents.traits = (
                        XLA_FFI_Handler_TraitsBits.COMMAND_BUFFER_COMPATIBLE
                    )
                    return None

            # retrieve call info
            attrs = decode_attrs(call_frame.contents.attrs)
            launch_id = int(attrs["launch_id"])
            launch_desc = self.launch_descriptors[launch_id]

            num_inputs = call_frame.contents.args.size
            inputs = ctypes.cast(call_frame.contents.args.args, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))

            num_outputs = call_frame.contents.rets.size
            outputs = ctypes.cast(call_frame.contents.rets.rets, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))

            assert num_inputs == self.num_inputs
            assert num_outputs == self.num_outputs

            launch_bounds = launch_bounds_t(launch_desc.launch_dims)

            # first kernel param is the launch bounds
            kernel_params = (ctypes.c_void_p * (1 + self.num_kernel_args))()
            kernel_params[0] = ctypes.addressof(launch_bounds)

            arg_refs = []

            # inputs
            for i in range(num_inputs):
                input_arg = self.input_args[i]
                if input_arg.is_array:
                    buffer = inputs[i].contents
                    shape = buffer.dims[: input_arg.type.ndim]
                    strides = strides_from_shape(shape, input_arg.type.dtype)
                    arg = array_t(buffer.data, 0, input_arg.type.ndim, shape, strides)
                    kernel_params[i + 1] = ctypes.addressof(arg)
                    arg_refs.append(arg)  # keep a reference
                else:
                    # scalar argument, get stashed value
                    value = launch_desc.static_inputs[input_arg.name]
                    arg = input_arg.type._type_(value)
                    kernel_params[i + 1] = ctypes.addressof(arg)
                    arg_refs.append(arg)  # keep a reference

            # outputs
            for i in range(num_outputs):
                output_arg = self.output_args[i]
                buffer = outputs[i].contents
                shape = buffer.dims[: output_arg.type.ndim]
                strides = strides_from_shape(shape, output_arg.type.dtype)
                arg = array_t(buffer.data, 0, output_arg.type.ndim, shape, strides)
                kernel_params[num_inputs + i + 1] = ctypes.addressof(arg)
                arg_refs.append(arg)  # keep a reference

            # get device and stream
            device = wp.device_from_jax(get_jax_device())
            stream = get_stream_from_callframe(call_frame.contents)

            # get kernel hooks
            hooks = self.kernel.module.get_kernel_hooks(self.kernel, device)
            assert hooks.forward, "Failed to find kernel entry point"

            # launch the kernel
            wp.context.runtime.core.cuda_launch_kernel(
                device.context,
                hooks.forward,
                launch_bounds.size,
                0,
                256,
                hooks.forward_smem_bytes,
                kernel_params,
                stream,
            )

        except Exception as e:
            print(traceback.format_exc())
            return create_ffi_error(
                call_frame.contents.api, XLA_FFI_Error_Code.UNKNOWN, f"FFI callback error: {type(e).__name__}: {e}"
            )


class FfiCallDesc:
    def __init__(self, static_inputs):
        self.static_inputs = static_inputs


class FfiCallable:
    def __init__(self, func, num_outputs, graph_compatible, vmap_method, output_dims):
        self.func = func
        self.name = generate_unique_name(func)
        self.num_outputs = num_outputs
        self.vmap_method = vmap_method
        self.graph_compatible = graph_compatible
        self.output_dims = output_dims
        self.first_array_arg = None
        self.has_static_args = False
        self.call_id = 0
        self.call_descriptors = {}

        # get arguments and annotations
        argspec = get_full_arg_spec(func)

        num_args = len(argspec.args)
        self.num_inputs = num_args - num_outputs
        if self.num_outputs < 1:
            raise ValueError("At least one output is required")
        if self.num_outputs > num_args:
            raise ValueError("Number of outputs cannot be greater than the number of kernel arguments")

        if len(argspec.annotations) < num_args:
            raise RuntimeError(f"Incomplete argument annotations on function {self.name}")

        # parse type annotations
        self.args = []
        arg_idx = 0
        for arg_name, arg_type in argspec.annotations.items():
            if arg_name == "return":
                if arg_type is not None:
                    raise TypeError("Function must not return a value")
            else:
                arg = FfiArg(arg_name, arg_type)
                if arg.is_array:
                    if arg_idx < self.num_inputs and self.first_array_arg is None:
                        self.first_array_arg = arg_idx
                else:
                    self.has_static_args = True
                self.args.append(arg)
            arg_idx += 1

        self.input_args = self.args[: self.num_inputs]
        self.output_args = self.args[self.num_inputs :]

        # register the callback
        FFI_CCALLFUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_CallFrame))
        self.callback_func = FFI_CCALLFUNC(lambda call_frame: self.ffi_callback(call_frame))
        ffi_ccall_address = ctypes.cast(self.callback_func, ctypes.c_void_p)
        ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)
        jax.ffi.register_ffi_target(self.name, ffi_capsule, platform="CUDA")

    def __call__(self, *args, output_dims=None, vmap_method=None):
        num_inputs = len(args)
        if num_inputs != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, but got {num_inputs}")

        # default argument fallback
        if vmap_method is None:
            vmap_method = self.vmap_method
        if output_dims is None:
            output_dims = self.output_dims

        # process inputs
        static_inputs = {}
        for i in range(num_inputs):
            input_arg = self.input_args[i]
            input_value = args[i]
            if input_arg.is_array:
                # check dtype
                if input_value.dtype != input_arg.jax_scalar_type:
                    raise TypeError(
                        f"Invalid data type for array argument '{input_arg.name}', expected {input_arg.jax_scalar_type}, got {input_value.dtype}"
                    )
                # check ndim
                if input_value.ndim != input_arg.jax_ndim:
                    raise TypeError(
                        f"Invalid dimensionality for array argument '{input_arg.name}', expected {input_arg.jax_ndim} dimensions, got {input_value.ndim}"
                    )
                # check inner dims
                for d in range(input_arg.dtype_ndim):
                    if input_value.shape[input_arg.type.ndim + d] != input_arg.dtype_shape[d]:
                        raise TypeError(
                            f"Invalid inner dimensions for array argument '{input_arg.name}', expected {input_arg.dtype_shape}, got {input_value.shape[-input_arg.dtype_ndim :]}"
                        )
            else:
                # make sure scalar is not a traced variable, should be static
                if isinstance(input_value, jax.core.Tracer):
                    raise ValueError(f"Argument '{input_arg.name}' must be a static value")
                # stash the value to be retrieved by callback
                static_inputs[input_arg.name] = input_arg.type(input_value)

        if output_dims is None and self.first_array_arg is not None:
            # use the shape of the first input array
            output_dims = get_warp_shape(self.input_args[self.first_array_arg], args[self.first_array_arg].shape)

        # output types
        out_types = []
        if isinstance(output_dims, dict):
            # assume a dictionary of shapes keyed on argument name
            for output_arg in self.output_args:
                dims = output_dims.get(output_arg.name)
                if dims is None:
                    raise ValueError(f"Missing output dimensions for argument '{output_arg.name}'")
                out_types.append(get_jax_output_type(output_arg, dims))
        else:
            if output_dims is None:
                raise ValueError("Unable to determine output dimensions")
            elif isinstance(output_dims, int):
                output_dims = (output_dims,)
            # assume same dimensions for all outputs
            for output_arg in self.output_args:
                out_types.append(get_jax_output_type(output_arg, output_dims))

        call = jax.ffi.ffi_call(
            self.name,
            out_types,
            vmap_method=vmap_method,
            # has_side_effect=True,  # force this function to execute even if outputs aren't used
        )

        # load the module
        # NOTE: if the target function uses kernels from different modules, they will not be loaded here
        device = wp.device_from_jax(get_jax_device())
        module = wp.get_module(self.func.__module__)
        module.load(device)

        if self.has_static_args:
            # save call data to be retrieved by callback
            call_id = self.call_id
            self.call_descriptors[call_id] = FfiCallDesc(static_inputs)
            self.call_id += 1
            return call(*args, call_id=call_id)
        else:
            return call(*args)

    def ffi_callback(self, call_frame):
        try:
            # TODO Try-catch around the body and return XLA_FFI_Error on error.
            extension = call_frame.contents.extension_start
            # On the first call, XLA runtime will query the API version and traits
            # metadata using the |extension| field. Let us respond to that query
            # if the metadata extension is present.
            if extension:
                # Try to set the version metadata.
                if extension.contents.type == XLA_FFI_Extension_Type.Metadata:
                    metadata_ext = ctypes.cast(extension, ctypes.POINTER(XLA_FFI_Metadata_Extension))
                    metadata_ext.contents.metadata.contents.api_version.major_version = 0
                    metadata_ext.contents.metadata.contents.api_version.minor_version = 1
                    # Turn on CUDA graphs for this handler.
                    if self.graph_compatible:
                        metadata_ext.contents.metadata.contents.traits = (
                            XLA_FFI_Handler_TraitsBits.COMMAND_BUFFER_COMPATIBLE
                        )
                    return None

            if self.has_static_args:
                # retrieve call info
                attrs = decode_attrs(call_frame.contents.attrs)
                call_id = int(attrs["call_id"])
                call_desc = self.call_descriptors[call_id]

            num_inputs = call_frame.contents.args.size
            inputs = ctypes.cast(call_frame.contents.args.args, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))

            num_outputs = call_frame.contents.rets.size
            outputs = ctypes.cast(call_frame.contents.rets.rets, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))

            assert num_inputs == self.num_inputs
            assert num_outputs == self.num_outputs

            device = wp.device_from_jax(get_jax_device())
            cuda_stream = get_stream_from_callframe(call_frame.contents)
            stream = wp.Stream(device, cuda_stream=cuda_stream)

            # reconstruct the argument list
            arg_list = []

            # inputs
            for i in range(num_inputs):
                arg = self.input_args[i]
                if arg.is_array:
                    buffer = inputs[i].contents
                    shape = buffer.dims[: buffer.rank - arg.dtype_ndim]
                    arr = wp.array(ptr=buffer.data, dtype=arg.type.dtype, shape=shape, device=device)
                    arg_list.append(arr)
                else:
                    # scalar argument, get stashed value
                    value = call_desc.static_inputs[arg.name]
                    arg_list.append(value)

            # outputs
            for i in range(num_outputs):
                arg = self.output_args[i]
                buffer = outputs[i].contents
                shape = buffer.dims[: buffer.rank - arg.dtype_ndim]
                arr = wp.array(ptr=buffer.data, dtype=arg.type.dtype, shape=shape, device=device)
                arg_list.append(arr)

            # call the Python function with reconstructed arguments
            with wp.ScopedStream(stream, sync_enter=False):
                if stream.is_capturing:
                    with wp.ScopedCapture(stream=stream, external=True):
                        self.func(*arg_list)
                else:
                    self.func(*arg_list)

        except Exception as e:
            print(traceback.format_exc())
            return create_ffi_error(
                call_frame.contents.api, XLA_FFI_Error_Code.UNKNOWN, f"FFI callback error: {type(e).__name__}: {e}"
            )

        return None


###############################################################################
#
# Generic FFI callbacks for Python functions of the form
# func(inputs, outputs, attrs, ctx)
#
###############################################################################

# Holder for the custom callbacks to keep them alive.
ffi_callbacks = {}


def register_ffi_callback(name: str, func: Callable, graph_compatible: bool = True) -> None:
    """Create a JAX callback from a Python function.

    The Python function must have the form ``func(inputs, outputs, attrs, ctx)``.

    NOTE: This is an experimental feature under development.

    Args:
        name: A unique FFI callback name.
        func: The Python function to call.
        graph_compatible: Optional. Whether the function can be called during CUDA graph capture.
    """

    # TODO check that the name is not already registered

    def ffi_callback(call_frame):
        try:
            # TODO Try-catch around the body and return XLA_FFI_Error on error.
            extension = call_frame.contents.extension_start
            # On the first call, XLA runtime will query the API version and traits
            # metadata using the |extension| field. Let us respond to that query
            # if the metadata extension is present.
            if extension:
                # Try to set the version metadata.
                if extension.contents.type == XLA_FFI_Extension_Type.Metadata:
                    metadata_ext = ctypes.cast(extension, ctypes.POINTER(XLA_FFI_Metadata_Extension))
                    metadata_ext.contents.metadata.contents.api_version.major_version = 0
                    metadata_ext.contents.metadata.contents.api_version.minor_version = 1
                    if graph_compatible:
                        # Turn on CUDA graphs for this handler.
                        metadata_ext.contents.metadata.contents.traits = (
                            XLA_FFI_Handler_TraitsBits.COMMAND_BUFFER_COMPATIBLE
                        )
                    return None

            attrs = decode_attrs(call_frame.contents.attrs)

            input_count = call_frame.contents.args.size
            inputs = ctypes.cast(call_frame.contents.args.args, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))
            inputs = [FfiBuffer(inputs[i].contents) for i in range(input_count)]

            output_count = call_frame.contents.rets.size
            outputs = ctypes.cast(call_frame.contents.rets.rets, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))
            outputs = [FfiBuffer(outputs[i].contents) for i in range(output_count)]

            ctx = ExecutionContext(call_frame.contents)

            func(inputs, outputs, attrs, ctx)
        except Exception as e:
            print(traceback.format_exc())
            return create_ffi_error(
                call_frame.contents.api, XLA_FFI_Error_Code.UNKNOWN, f"FFI callback error: {type(e).__name__}: {e}"
            )

        return None

    FFI_CCALLFUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_CallFrame))
    callback_func = FFI_CCALLFUNC(ffi_callback)
    ffi_callbacks[name] = callback_func
    ffi_ccall_address = ctypes.cast(callback_func, ctypes.c_void_p)
    ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)
    jax.ffi.register_ffi_target(name, ffi_capsule, platform="CUDA")


###############################################################################
#
# Utilities
#
###############################################################################

# ensure unique FFI callback names
ffi_name_counts = {}


def generate_unique_name(func) -> str:
    key = make_full_qualified_name(func)
    unique_id = ffi_name_counts.get(key, 0)
    ffi_name_counts[key] = unique_id + 1
    return f"{key}_{unique_id}"


def get_warp_shape(arg, dims):
    if arg.dtype_ndim > 0:
        # vector/matrix array
        return dims[: arg.warp_ndim]
    else:
        # scalar array
        return dims


def get_jax_output_type(arg, dims):
    if isinstance(dims, int):
        dims = (dims,)

    ndim = len(dims)

    if arg.dtype_ndim > 0:
        # vector/matrix array
        if ndim == arg.warp_ndim:
            return jax.ShapeDtypeStruct((*dims, *arg.dtype_shape), arg.jax_scalar_type)
        elif ndim == arg.jax_ndim:
            # make sure inner dimensions match
            inner_dims = dims[-arg.dtype_ndim :]
            for i in range(arg.dtype_ndim):
                if inner_dims[i] != arg.dtype_shape[i]:
                    raise ValueError(f"Invalid output dimensions for argument '{arg.name}': {dims}")
            return jax.ShapeDtypeStruct(dims, arg.jax_scalar_type)
        else:
            raise ValueError(f"Invalid output dimensions for argument '{arg.name}': {dims}")
    else:
        # scalar array
        if ndim != arg.warp_ndim:
            raise ValueError(f"Invalid output dimensions for argument '{arg.name}': {dims}")
        return jax.ShapeDtypeStruct(dims, arg.jax_scalar_type)
