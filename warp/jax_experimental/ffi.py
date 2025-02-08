# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes

import jax

import warp as wp
from warp.codegen import get_full_arg_spec, make_full_qualified_name
from warp.jax import get_jax_device
from warp.types import launch_bounds_t, type_to_warp

from .xla_ffi import *

###############################################################################
#
# jax_kernel()
#
###############################################################################

# Holder for the custom callback to keep it alive.
jax_kernel_callback_fn = None
jax_kernel_callback_name = "jax_kernel_handler"
registered_kernels = [None]
registered_kernel_to_id = {}


def jax_kernel(kernel, launch_dims=None, output_dims=None, vmap_method=None):
    """Create a Jax primitive from a Warp kernel.

    NOTE: This is an experimental feature under development.

    Args:
        wp_kernel: The Warp kernel to be wrapped.
        launch_dims: Optional. Specify the kernel launch dimensions. If None, launch
                     dimensions are inferred from the shape of the first argument.
        output_dims: Optional. Specify the dimensions of output arrays.  If None, output
                     dimensions are inferred from the launch dimensions.
        vmap_method: Optional. String specifying how the callback transforms under ``vmap()``.

    Current limitations:
    - All kernel arguments must be arrays.
    - Input arguments are followed by output arguments in the Warp kernel definition.
    - There must be at least one input argument and at least one output argument.
    - All arrays must be contiguous.
    - Only the CUDA backend is supported.
    """

    # register callback
    if jax_kernel_callback_fn is None:
        register_jax_kernel_callback()

    # register kernel
    if kernel in registered_kernel_to_id:
        id = registered_kernel_to_id[kernel]
    else:
        id = len(registered_kernels)
        registered_kernels.append(kernel)
        registered_kernel_to_id[kernel] = id

    # These keyword arguments can be passed to jax_kernel() or at the call site.
    # The default kwargs are less flexible and mostly for backward compatibility.
    # Call site kwargs override these default ones.
    default_kwargs = {
        "launch_dims": launch_dims,
        "output_dims": output_dims,
        "vmap_method": vmap_method,
    }

    return lambda *args, **kwargs: jax_kernel_call(id, kernel, *args, **{**default_kwargs, **kwargs})


def jax_to_warp_shape(jax_shape, warp_dtype):
    if hasattr(warp_dtype, "_shape_"):
        return jax_shape[: -len(warp_dtype._shape_)]
    else:
        return jax_shape


def warp_to_jax_shape_dtype(warp_shape, warp_dtype):
    if hasattr(warp_dtype, "_wp_scalar_type_"):
        return jax.ShapeDtypeStruct((*warp_shape, *warp_dtype._shape_), wp.dtype_to_jax(warp_dtype._wp_scalar_type_))
    else:
        return jax.ShapeDtypeStruct(warp_shape, wp.dtype_to_jax(warp_dtype))


def check_argument(warp_arg, jax_arg):
    if isinstance(warp_arg.type, wp.array):
        warp_type = warp_arg.type
        if hasattr(warp_type.dtype, "_wp_scalar_type_"):
            expected_dtype = wp.dtype_to_jax(warp_type.dtype._wp_scalar_type_)
            expected_ndim = warp_type.ndim + len(warp_type.dtype._shape_)
        else:
            expected_dtype = wp.dtype_to_jax(warp_type.dtype)
            expected_ndim = warp_type.ndim

        if jax_arg.ndim != expected_ndim:
            raise TypeError(
                f"Kernel argument '{warp_arg.label}' expects JAX array with {expected_ndim} dimensions, got {jax_arg.ndim} dimensions"
            )
        if jax_arg.dtype != expected_dtype:
            raise TypeError(
                f"Kernel argument '{warp_arg.label}' expects JAX array with dtype {expected_dtype}, got dtype {jax_arg.dtype}"
            )
    else:
        raise TypeError(
            f"Kernel argument '{warp_arg.label}' is not an array, but only arrays are currently supported, "
        )


def jax_kernel_call(kernel_id, kernel, *args, launch_dims=None, output_dims=None, vmap_method=None):
    num_kernel_args = len(kernel.adj.args)
    num_inputs = len(args)
    num_outputs = num_kernel_args - num_inputs

    if num_inputs < 1 or num_outputs < 1:
        raise ValueError(
            f"At least one input array and one output array required, got {num_inputs} inputs and {num_outputs} outputs"
        )

    # check inputs
    for i in range(num_inputs):
        check_argument(kernel.adj.args[i], args[i])

    # launch dimensions
    if launch_dims is None:
        # use first input array for launch dimensions
        launch_dims = jax_to_warp_shape(args[0].shape, kernel.adj.args[0].type.dtype)
    elif isinstance(launch_dims, int):
        launch_dims = (launch_dims,)
    elif not hasattr(launch_dims, "__len__"):
        raise TypeError(f"launch_dims must be an integer or integer sequence, got {launch_dims}")

    # output types
    out_types = []
    if output_dims is None:
        # assume same as launch dimensions for all outputs
        for i in range(num_inputs, num_kernel_args):
            out_types.append(warp_to_jax_shape_dtype(launch_dims, kernel.adj.args[i].type.dtype))
    elif isinstance(output_dims, dict):
        # assume a dictionary of shapes keyed on argument name
        for i in range(num_inputs, num_kernel_args):
            d = output_dims.get(kernel.adj.args[i].label)
            if d is not None:
                if isinstance(d, int):
                    d = (d,)
                elif not hasattr(d, "__len__"):
                    raise TypeError(f"Output dimensions must be an integer or integer sequence, got {d}")
            else:
                d = launch_dims
            out_types.append(warp_to_jax_shape_dtype(d, kernel.adj.args[i].type.dtype))
    elif isinstance(output_dims, int):
        # assume same dimensions for all outputs
        d = (output_dims,)
        for i in range(num_inputs, num_kernel_args):
            out_types.append(warp_to_jax_shape_dtype(d, kernel.adj.args[i].type.dtype))
    elif hasattr(output_dims, "__len__"):
        # assume same dimensions for all outputs
        for i in range(num_inputs, num_kernel_args):
            out_types.append(warp_to_jax_shape_dtype(output_dims, kernel.adj.args[i].type.dtype))
    else:
        raise ValueError(f"output_dims must be an integer, integer sequence, or dictionary, got {output_dims}")

    if num_outputs == 1:
        out_types = out_types[0]

    if vmap_method is None:
        vmap_method = "broadcast_all"

    call = jax.ffi.ffi_call(
        jax_kernel_callback_name,
        out_types,
        vmap_method=vmap_method,
    )

    # ensure the kernel module is loaded before the callback, otherwise graph capture may fail
    device = wp.device_from_jax(get_jax_device())
    kernel.module.load(device)

    # launch descriptor
    launch_dims_str = ",".join([str(d) for d in launch_dims])
    descriptor = f"{kernel_id}|{launch_dims_str}"

    return call(*args, desc=descriptor)


def jax_kernel_callback(call_frame):
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
                metadata_ext.contents.metadata.contents.traits = XLA_FFI_Handler_TraitsBits.COMMAND_BUFFER_COMPATIBLE
                return None

        if call_frame.contents.attrs.size != 1 or call_frame.contents.attrs.types[0] != XLA_FFI_AttrType.STRING:
            return create_invalid_argument_ffi_error(
                call_frame.contents.api, "Internal error, expected one attribute ('desc')"
            )

        attr_name = decode_bytespan(call_frame.contents.attrs.names[0].contents)
        if attr_name != "desc":
            return create_invalid_argument_ffi_error(
                call_frame.contents.api, "Internal error, expected 'desc' attribute"
            )

        attr_value = ctypes.cast(call_frame.contents.attrs.attrs[0], ctypes.POINTER(XLA_FFI_ByteSpan))
        descriptor = decode_bytespan(attr_value.contents)

        kernel_id_str, launch_dims_str = descriptor.split("|")
        kernel = registered_kernels[int(kernel_id_str)]
        launch_dims = [int(d) for d in launch_dims_str.split(",")]

        inputs = ctypes.cast(call_frame.contents.args.args, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))
        outputs = ctypes.cast(call_frame.contents.rets.rets, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))
        num_inputs = call_frame.contents.args.size
        num_outputs = call_frame.contents.rets.size

        num_kernel_args = len(kernel.adj.args)

        if num_kernel_args != num_inputs + num_outputs:
            return create_invalid_argument_ffi_error(call_frame.contents.api, "Invalid number of arguments for kernel")

        launch_bounds = launch_bounds_t(launch_dims)

        # first kernel param is the launch bounds
        kernel_params = (ctypes.c_void_p * (1 + num_kernel_args))()
        kernel_params[0] = ctypes.addressof(launch_bounds)

        arg_refs = []

        # inputs
        for i in range(num_inputs):
            arg = kernel_arg_from_ffi_buffer(inputs[i].contents, kernel.adj.args[i])
            kernel_params[i + 1] = ctypes.addressof(arg)
            arg_refs.append(arg)  # keep a reference

        # outputs
        for i in range(num_outputs):
            arg = kernel_arg_from_ffi_buffer(outputs[i].contents, kernel.adj.args[num_inputs + i])
            kernel_params[num_inputs + i + 1] = ctypes.addressof(arg)
            arg_refs.append(arg)  # keep a reference

        # get device and stream
        device = wp.device_from_jax(get_jax_device())
        stream = get_stream_from_callframe(call_frame.contents)

        # get kernel hooks
        hooks = kernel.module.get_kernel_hooks(kernel, device)
        assert hooks.forward, "Failed to find kernel entry point"

        # launch the kernel
        wp.context.runtime.core.cuda_launch_kernel(
            device.context, hooks.forward, launch_bounds.size, 0, 256, hooks.forward_smem_bytes, kernel_params, stream
        )

    except Exception as e:
        return create_ffi_error(call_frame.contents.api, XLA_FFI_Error_Code.UNKNOWN, f"FFI callback error: {str(e)}")


def register_jax_kernel_callback():
    global jax_kernel_callback_fn

    FFI_CCALLFUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_CallFrame))
    jax_kernel_callback_fn = FFI_CCALLFUNC(jax_kernel_callback)
    ffi_ccall_address = ctypes.cast(jax_kernel_callback_fn, ctypes.c_void_p)
    ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)
    jax.ffi.register_ffi_target(jax_kernel_callback_name, ffi_capsule, platform="CUDA")


###############################################################################
#
# jax_callable()
#
###############################################################################

# Holder for the custom callbacks to keep them alive.
ffi_callbacks = {}


def jax_callable(func, num_outputs=1, vmap_method="broadcast_all"):
    return FfiCallable(func, num_outputs, vmap_method)


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
        elif type in wp.types.value_types:
            self.dtype_ndim = 0
            self.dtype_shape = ()
            self.jax_scalar_type = wp.dtype_to_jax(type_to_warp(type))
            self.jax_ndim = 0
        else:
            raise TypeError(f"Invalid type for argument '{name}', expected array or scalar, got {type}")


class FfiCall:
    def __init__(self, callable, inputs):
        self.callable = callable
        self.inputs = inputs


class FfiCallable:
    call_id = 0
    call_descriptors = {}

    def __init__(self, func, num_outputs, vmap_method):
        self.func = func
        self.name = make_full_qualified_name(func)
        self.num_outputs = num_outputs
        self.vmap_method = vmap_method

        # get arguments and annotations
        argspec = get_full_arg_spec(func)

        num_args = len(argspec.args)
        self.num_inputs = num_args - num_outputs
        if self.num_inputs < 1 or num_outputs < 1:
            raise ValueError("At least one input and one output are required")

        if len(argspec.annotations) < num_args:
            raise RuntimeError(f"Incomplete argument annotations on function {self.name}")

        # parse type annotations
        self.args = []
        for arg_name, arg_type in argspec.annotations.items():
            if arg_name == "return":
                if arg_type is not None:
                    raise TypeError("Function must not return a value")
            else:
                self.args.append(FfiArg(arg_name, arg_type))

        self.input_args = self.args[: self.num_inputs]
        self.output_args = self.args[self.num_inputs :]

        # register the callback
        FFI_CCALLFUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_CallFrame))
        callback_func = FFI_CCALLFUNC(FfiCallable.ffi_callback)
        ffi_callbacks[self.name] = callback_func
        ffi_ccall_address = ctypes.cast(callback_func, ctypes.c_void_p)
        ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)
        jax.ffi.register_ffi_target(self.name, ffi_capsule, platform="CUDA")

    def __call__(self, *args, output_dims=None, vmap_method=None):
        num_inputs = len(args)
        if num_inputs != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} outputs, but got {num_inputs}")

        if output_dims is None:
            raise ValueError("Missing output_dims")

        num_outputs = len(output_dims)
        if num_outputs != self.num_outputs:
            raise ValueError(f"Expected {self.num_outputs} outputs, but got {num_outputs}")

        out_types = []

        if output_dims is not None:
            for output_arg in self.output_args:
                dims = output_dims.get(output_arg.name)
                if dims is None:
                    raise ValueError(f"Missing dims for output '{output_arg.name}'")

                if isinstance(dims, int):
                    dims = (dims,)
                if len(dims) != output_arg.jax_ndim:
                    raise ValueError(f"Invalid dimensions for argument '{output_arg.name}'")

                out_types.append(jax.ShapeDtypeStruct(dims, output_arg.jax_scalar_type))
        else:
            raise ValueError("Missing 'output_dims' argument")

        if vmap_method is None:
            vmap_method = self.vmap_method

        call = jax.ffi.ffi_call(
            self.name,
            out_types,
            vmap_method=self.vmap_method,
            # has_side_effect=True,  # force this function to execute even without outputs
        )

        # save call data to be retrieved by callback
        call_id = FfiCallable.call_id
        FfiCallable.call_descriptors[call_id] = FfiCall(self, args)
        FfiCallable.call_id += 1

        return call(*args, call_id=call_id)

    @staticmethod
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
                    # Turn on CUDA graphs for this handler.
                    metadata_ext.contents.metadata.contents.traits = (
                        XLA_FFI_Handler_TraitsBits.COMMAND_BUFFER_COMPATIBLE
                    )
                    return None

            attrs = decode_attrs(call_frame.contents.attrs)

            # retrieve call info
            call_id = int(attrs["call_id"])
            call_desc = FfiCallable.call_descriptors[call_id]
            del FfiCallable.call_descriptors[call_id]

            num_inputs = call_frame.contents.args.size
            inputs = ctypes.cast(call_frame.contents.args.args, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))

            num_outputs = call_frame.contents.rets.size
            outputs = ctypes.cast(call_frame.contents.rets.rets, ctypes.POINTER(ctypes.POINTER(XLA_FFI_Buffer)))

            assert num_inputs == call_desc.callable.num_inputs
            assert num_outputs == call_desc.callable.num_outputs

            # reconstruct the argument list
            arg_list = []

            # inputs
            for i in range(num_inputs):
                arg = call_desc.callable.input_args[i]
                if arg.is_array:
                    buffer = inputs[i].contents
                    buffer_dtype = jax_dtype_from_ffi(buffer.dtype)
                    buffer_shape = buffer.dims[: buffer.rank]

                    # check scalar type
                    if buffer_dtype != arg.jax_scalar_type:
                        raise TypeError(
                            f"Invalid data type for array argument '{arg.name}', expected {arg.jax_scalar_type}, got {buffer_dtype}"
                        )
                    # check ndim
                    if buffer.rank != arg.jax_ndim:
                        raise TypeError(
                            f"Invalid dimensionality for array argument '{arg.name}', expected {arg.jax_ndim} dimensions, got {buffer.rank}"
                        )
                    # check inner dims
                    for d in range(arg.dtype_ndim):
                        if buffer_shape[arg.dtype_ndim + d] != arg.dtype_shape[d]:
                            raise TypeError(
                                f"Invalid inner dimensions for array argument '{arg.name}', expected {arg.dtype_shape}, got {tuple(buffer_shape[-arg.dtype_ndim :])}"
                            )

                    warp_shape = buffer_shape[: buffer.rank - arg.dtype_ndim]

                    arr = wp.array(ptr=buffer.data, dtype=arg.type.dtype, shape=warp_shape)
                    arg_list.append(arr)
                else:
                    # scalar argument, get stashed value
                    value = call_desc.inputs[i]

                    # make sure it's not a traced variable, should be static
                    if isinstance(value, jax.core.Tracer):
                        raise ValueError(f"Scalar argument '{arg.name}' must be a static compile-time constant")

                    scalar = arg.type(value)
                    arg_list.append(scalar)

            # outputs
            for i in range(num_outputs):
                arg = call_desc.callable.output_args[i]
                buffer = outputs[i].contents
                shape = buffer.dims[: buffer.rank - arg.dtype_ndim]
                arr = wp.array(ptr=buffer.data, dtype=arg.type.dtype, shape=shape)
                arg_list.append(arr)

            ctx = ExecutionContext(call_frame.contents)

            device = wp.device_from_jax(get_jax_device())
            stream = wp.Stream(device, cuda_stream=ctx.stream)

            # call the Python function with reconstructed arguments
            with wp.ScopedStream(stream):
                call_desc.callable.func(*arg_list)

        except Exception as e:
            return create_ffi_error(
                call_frame.contents.api, XLA_FFI_Error_Code.UNKNOWN, f"FFI callback error: {str(e)}"
            )

        return None


###############################################################################
#
# Generic FFI callbacks (call any Python function)
#
###############################################################################


def register_ffi_callback(name, fn):
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

            fn(inputs, outputs, attrs, ctx)
        except Exception as e:
            return create_ffi_error(
                call_frame.contents.api, XLA_FFI_Error_Code.UNKNOWN, f"FFI callback error: {str(e)}"
            )

        return None

    FFI_CCALLFUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(XLA_FFI_CallFrame))
    callback_func = FFI_CCALLFUNC(ffi_callback)
    ffi_callbacks[name] = callback_func
    ffi_ccall_address = ctypes.cast(callback_func, ctypes.c_void_p)
    ffi_capsule = jax.ffi.pycapsule(ffi_ccall_address.value)
    jax.ffi.register_ffi_target(name, ffi_capsule, platform="CUDA")
