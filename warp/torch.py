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

import ctypes

import numpy

import warp
import warp.context


# return the warp device corresponding to a torch device
def device_from_torch(torch_device) -> warp.context.Device:
    """Return the Warp device corresponding to a Torch device.

    Args:
        torch_device (`torch.device` or `str`): Torch device identifier

    Raises:
        RuntimeError: Torch device does not have a corresponding Warp device
    """
    if type(torch_device) is str:
        warp_device = warp.context.runtime.device_map.get(torch_device)
        if warp_device is not None:
            return warp_device
        elif torch_device == "cuda":
            return warp.context.runtime.get_current_cuda_device()
        else:
            raise RuntimeError(f"Unsupported Torch device {torch_device}")
    else:
        try:
            if torch_device.type == "cuda":
                return warp.context.runtime.cuda_devices[torch_device.index]
            elif torch_device.type == "cpu":
                return warp.context.runtime.cpu_device
            else:
                raise RuntimeError(f"Unsupported Torch device type {torch_device.type}")
        except Exception as e:
            import torch

            if not isinstance(torch_device, torch.device):
                raise ValueError("Argument must be a torch.device object or a string") from e
            raise


def device_to_torch(warp_device: warp.context.Devicelike) -> str:
    """Return the Torch device string corresponding to a Warp device.

    Args:
        warp_device: An identifier that can be resolved to a :class:`warp.context.Device`.

    Raises:
        RuntimeError: The Warp device is not compatible with PyTorch.
    """
    device = warp.get_device(warp_device)
    if device.is_cpu or device.is_primary:
        return str(device)
    elif device.is_cuda and device.is_uva:
        # it's not a primary context, but torch can access the data ptr directly thanks to UVA
        return f"cuda:{device.ordinal}"
    raise RuntimeError(f"Warp device {device} is not compatible with torch")


def dtype_to_torch(warp_dtype):
    """Return the Torch dtype corresponding to a Warp dtype.

    Args:
        warp_dtype: A Warp data type that has a corresponding ``torch.dtype``.
            ``warp.uint16``, ``warp.uint32``, and ``warp.uint64`` are mapped
            to the signed integer ``torch.dtype`` of the same width.
    Raises:
        TypeError: Unable to find a corresponding PyTorch data type.
    """
    # initialize lookup table on first call to defer torch import
    if dtype_to_torch.type_map is None:
        import torch

        dtype_to_torch.type_map = {
            warp.float16: torch.float16,
            warp.float32: torch.float32,
            warp.float64: torch.float64,
            warp.int8: torch.int8,
            warp.int16: torch.int16,
            warp.int32: torch.int32,
            warp.int64: torch.int64,
            warp.uint8: torch.uint8,
            # torch doesn't support unsigned ints bigger than 8 bits
            warp.uint16: torch.int16,
            warp.uint32: torch.int32,
            warp.uint64: torch.int64,
            warp.bool: torch.bool,
        }

    torch_dtype = dtype_to_torch.type_map.get(warp_dtype)
    if torch_dtype is not None:
        return torch_dtype
    else:
        raise TypeError(f"Cannot convert {warp_dtype} to a Torch type")


def dtype_from_torch(torch_dtype):
    """Return the Warp dtype corresponding to a Torch dtype.

    Args:
        torch_dtype: A ``torch.dtype`` that has a corresponding Warp data type.
            Currently ``torch.bfloat16``, ``torch.complex64``, and
            ``torch.complex128`` are not supported.

    Raises:
        TypeError: Unable to find a corresponding Warp data type.
    """
    # initialize lookup table on first call to defer torch import
    if dtype_from_torch.type_map is None:
        import torch

        dtype_from_torch.type_map = {
            torch.float16: warp.float16,
            torch.float32: warp.float32,
            torch.float64: warp.float64,
            torch.int8: warp.int8,
            torch.int16: warp.int16,
            torch.int32: warp.int32,
            torch.int64: warp.int64,
            torch.uint8: warp.uint8,
            torch.bool: warp.bool,
            # currently unsupported by Warp
            # torch.bfloat16:
            # torch.complex64:
            # torch.complex128:
        }

    warp_dtype = dtype_from_torch.type_map.get(torch_dtype)

    if warp_dtype is not None:
        return warp_dtype
    else:
        raise TypeError(f"Cannot convert {torch_dtype} to a Warp type")


def dtype_is_compatible(torch_dtype, warp_dtype) -> bool:
    """Evaluates whether the given torch dtype is compatible with the given Warp dtype."""
    # initialize lookup table on first call to defer torch import
    if dtype_is_compatible.compatible_sets is None:
        import torch

        dtype_is_compatible.compatible_sets = {
            torch.float64: {warp.float64},
            torch.float32: {warp.float32},
            torch.float16: {warp.float16},
            # allow aliasing integer tensors as signed or unsigned integer arrays
            torch.int64: {warp.int64, warp.uint64},
            torch.int32: {warp.int32, warp.uint32},
            torch.int16: {warp.int16, warp.uint16},
            torch.int8: {warp.int8, warp.uint8},
            torch.uint8: {warp.uint8, warp.int8},
            torch.bool: {warp.bool, warp.uint8, warp.int8},
            # currently unsupported by Warp
            # torch.bfloat16:
            # torch.complex64:
            # torch.complex128:
        }

    compatible_set = dtype_is_compatible.compatible_sets.get(torch_dtype)

    if compatible_set is not None:
        if warp_dtype in compatible_set:
            return True
        # check if it's a vector or matrix type
        if hasattr(warp_dtype, "_wp_scalar_type_"):
            return warp_dtype._wp_scalar_type_ in compatible_set

    return False


# lookup tables initialized when needed
dtype_from_torch.type_map = None
dtype_to_torch.type_map = None
dtype_is_compatible.compatible_sets = None


# wrap a torch tensor to a wp array, data is not copied
def from_torch(t, dtype=None, requires_grad=None, grad=None, return_ctype=False):
    """Convert a Torch tensor to a Warp array without copying the data.

    Args:
        t (torch.Tensor): The torch tensor to wrap.
        dtype (warp.dtype, optional): The target data type of the resulting Warp array. Defaults to the tensor value type mapped to a Warp array value type.
        requires_grad (bool, optional): Whether the resulting array should wrap the tensor's gradient, if it exists (the grad tensor will be allocated otherwise). Defaults to the tensor's `requires_grad` value.
        return_ctype (bool, optional): Whether to return a low-level array descriptor instead of a ``wp.array`` object (faster).  The descriptor can be passed to Warp kernels.

    Returns:
        warp.array: The wrapped array or array descriptor.
    """
    if dtype is None:
        dtype = dtype_from_torch(t.dtype)
    elif not dtype_is_compatible(t.dtype, dtype):
        raise RuntimeError(f"Cannot convert Torch type {t.dtype} to Warp type {dtype}")

    # get size of underlying data type to compute strides
    ctype_size = ctypes.sizeof(dtype._type_)

    shape = tuple(t.shape)
    strides = tuple(s * ctype_size for s in t.stride())

    # if target is a vector or matrix type
    # then check if trailing dimensions match
    # the target type and update the shape
    if hasattr(dtype, "_shape_"):
        dtype_shape = dtype._shape_
        dtype_dims = len(dtype._shape_)
        # ensure inner shape matches
        if dtype_dims > len(shape) or dtype_shape != shape[-dtype_dims:]:
            raise RuntimeError(
                f"Could not convert Torch tensor with shape {shape} to Warp array with dtype={dtype}, ensure that source inner shape is {dtype_shape}"
            )
        # ensure inner strides are contiguous
        if strides[-1] != ctype_size or (dtype_dims > 1 and strides[-2] != ctype_size * dtype_shape[-1]):
            raise RuntimeError(
                f"Could not convert Torch tensor with shape {shape} to Warp array with dtype={dtype}, because the source inner strides are not contiguous"
            )
        # trim shape and strides
        shape = tuple(shape[:-dtype_dims]) or (1,)
        strides = tuple(strides[:-dtype_dims]) or (ctype_size,)

    # gradient
    # - if return_ctype is False, we set `grad` to a wp.array or None
    # - if return_ctype is True, we set `grad_ptr` and set `grad` as the owner (wp.array or torch.Tensor)
    requires_grad = t.requires_grad if requires_grad is None else requires_grad
    grad_ptr = 0
    if grad is not None:
        if isinstance(grad, warp.array):
            if return_ctype:
                if grad.strides != strides:
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {strides} but got {grad.strides}"
                    )
                grad_ptr = grad.ptr
        else:
            # assume grad is a torch.Tensor
            if return_ctype:
                if t.stride() != grad.stride():
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {t.stride()} but got {grad.stride()}"
                    )
                grad_ptr = grad.data_ptr()
            else:
                grad = from_torch(grad, dtype=dtype, requires_grad=False)
    elif requires_grad:
        # wrap the tensor gradient, allocate if necessary
        if t.grad is not None:
            if return_ctype:
                grad = t.grad
                if t.stride() != grad.stride():
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {t.stride()} but got {grad.stride()}"
                    )
                grad_ptr = grad.data_ptr()
            else:
                grad = from_torch(t.grad, dtype=dtype, requires_grad=False)
        else:
            # allocate a zero-filled gradient if it doesn't exist
            # Note: we use Warp to allocate the shared gradient with compatible strides
            grad = warp.zeros(dtype=dtype, shape=shape, strides=strides, device=device_from_torch(t.device))
            t.grad = to_torch(grad, requires_grad=False)
            grad_ptr = grad.ptr

    if return_ctype:
        ptr = t.data_ptr()

        # create array descriptor
        array_ctype = warp.types.array_t(ptr, grad_ptr, len(shape), shape, strides)

        # keep data and gradient alive
        array_ctype._ref = t
        array_ctype._gradref = grad

        return array_ctype

    else:
        a = warp.array(
            ptr=t.data_ptr(),
            dtype=dtype,
            shape=shape,
            strides=strides,
            device=device_from_torch(t.device),
            copy=False,
            grad=grad,
            requires_grad=requires_grad,
        )

        # save a reference to the source tensor, otherwise it may get deallocated
        a._tensor = t

        return a


def to_torch(a, requires_grad=None):
    """
    Convert a Warp array to a Torch tensor without copying the data.

    Args:
        a (warp.array): The Warp array to convert.
        requires_grad (bool, optional): Whether the resulting tensor should convert the array's gradient, if it exists, to a grad tensor. Defaults to the array's `requires_grad` value.

    Returns:
        torch.Tensor: The converted tensor.
    """
    import torch

    if requires_grad is None:
        requires_grad = a.requires_grad

    # Torch does not support structured arrays
    if isinstance(a.dtype, warp.codegen.Struct):
        raise RuntimeError("Cannot convert structured Warp arrays to Torch.")

    if a.device.is_cpu:
        # Torch has an issue wrapping CPU objects
        # that support the __array_interface__ protocol
        # in this case we need to workaround by going
        # to an ndarray first, see https://pearu.github.io/array_interface_pytorch.html
        t = torch.as_tensor(numpy.asarray(a))
        t.requires_grad = requires_grad
        if requires_grad and a.requires_grad:
            t.grad = torch.as_tensor(numpy.asarray(a.grad))
        return t

    elif a.device.is_cuda:
        # Torch does support the __cuda_array_interface__
        # correctly, but we must be sure to maintain a reference
        # to the owning object to prevent memory allocs going out of scope
        t = torch.as_tensor(a, device=device_to_torch(a.device))
        t.requires_grad = requires_grad
        if requires_grad and a.requires_grad:
            t.grad = torch.as_tensor(a.grad, device=device_to_torch(a.device))
        return t

    else:
        raise RuntimeError("Unsupported device")


def stream_from_torch(stream_or_device=None):
    """Convert from a Torch CUDA stream to a Warp CUDA stream."""
    import torch

    if isinstance(stream_or_device, torch.cuda.Stream):
        stream = stream_or_device
    else:
        # assume arg is a torch device
        stream = torch.cuda.current_stream(stream_or_device)

    device = device_from_torch(stream.device)

    warp_stream = warp.Stream(device, cuda_stream=stream.cuda_stream)

    # save a reference to the source stream, otherwise it may be destroyed
    warp_stream._torch_stream = stream

    return warp_stream


def stream_to_torch(stream_or_device=None):
    """Convert from a Warp CUDA stream to a Torch CUDA stream."""
    import torch

    if isinstance(stream_or_device, warp.Stream):
        stream = stream_or_device
    else:
        # assume arg is a warp device
        stream = warp.get_device(stream_or_device).stream

    device = device_to_torch(stream.device)

    torch_stream = torch.cuda.ExternalStream(stream.cuda_stream, device=device)

    # save a reference to the source stream, otherwise it may be destroyed
    torch_stream._warp_stream = stream

    return torch_stream
