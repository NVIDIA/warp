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

# Python specification for DLpack:
# https://dmlc.github.io/dlpack/latest/python_spec.html

import ctypes

import warp
from warp.thirdparty.dlpack import (
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    DLManagedTensor,
    _c_str_dltensor,
)

_c_str_used_dltensor = b"used_dltensor"

PyMem_RawMalloc = ctypes.pythonapi.PyMem_RawMalloc
PyMem_RawMalloc.argtypes = [ctypes.c_size_t]
PyMem_RawMalloc.restype = ctypes.c_void_p

PyMem_RawFree = ctypes.pythonapi.PyMem_RawFree
PyMem_RawFree.argtypes = [ctypes.c_void_p]
PyMem_RawFree.restype = None

Py_IncRef = ctypes.pythonapi.Py_IncRef
Py_IncRef.argtypes = [ctypes.py_object]
Py_IncRef.restype = None

Py_DecRef = ctypes.pythonapi.Py_DecRef
Py_DecRef.argtypes = [ctypes.py_object]
Py_DecRef.restype = None

PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
PyCapsule_IsValid.restype = ctypes.c_int

PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
PyCapsule_GetPointer.restype = ctypes.c_void_p

PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
PyCapsule_SetName.restype = ctypes.c_int


class _DLPackTensorHolder:
    """Class responsible for deleting DLManagedTensor memory after ownership is transferred from a capsule."""

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.mem_ptr = None
        return instance

    def __init__(self, mem_ptr):
        self.mem_ptr = mem_ptr

    def __del__(self):
        if not self.mem_ptr:
            return

        managed_tensor = DLManagedTensor.from_address(self.mem_ptr)
        if managed_tensor.deleter:
            managed_tensor.deleter(self.mem_ptr)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _dlpack_tensor_deleter(managed_ptr) -> None:
    """A function to deallocate a DLManagedTensor."""

    managed_tensor = DLManagedTensor.from_address(managed_ptr)

    # unreference the source array
    manager = ctypes.cast(managed_tensor.manager_ctx, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(manager)

    # free the DLManagedTensor memory, including shape and strides
    PyMem_RawFree(ctypes.c_void_p(managed_ptr))


@PyCapsule_Destructor
def _dlpack_capsule_deleter(ptr) -> None:
    """Destructor for a capsule holding a DLManagedTensor."""

    capsule = ctypes.cast(ptr, ctypes.py_object)

    if PyCapsule_IsValid(capsule, _c_str_dltensor):
        managed_ptr = PyCapsule_GetPointer(capsule, _c_str_dltensor)
        managed_tensor = DLManagedTensor.from_address(managed_ptr)
        if managed_tensor.deleter:
            managed_tensor.deleter(managed_ptr)


def _device_to_dlpack(wp_device: warp.context.Device) -> DLDevice:
    dl_device = DLDevice()

    if wp_device.is_cpu:
        dl_device.device_type = DLDeviceType.kDLCPU
        dl_device.device_id = 0
    elif wp_device.is_cuda:
        dl_device.device_type = DLDeviceType.kDLCUDA
        dl_device.device_id = wp_device.ordinal
    else:
        raise RuntimeError(f"Invalid device type converting to DLPack: {wp_device}")

    return dl_device


def device_to_dlpack(wp_device) -> DLDevice:
    return _device_to_dlpack(warp.get_device(wp_device))


def dtype_to_dlpack(wp_dtype) -> DLDataType:
    if wp_dtype == warp.bool:
        return (DLDataTypeCode.kDLBool, 8, 1)
    if wp_dtype == warp.int8:
        return (DLDataTypeCode.kDLInt, 8, 1)
    elif wp_dtype == warp.uint8:
        return (DLDataTypeCode.kDLUInt, 8, 1)
    elif wp_dtype == warp.int16:
        return (DLDataTypeCode.kDLInt, 16, 1)
    elif wp_dtype == warp.uint16:
        return (DLDataTypeCode.kDLUInt, 16, 1)
    elif wp_dtype == warp.int32:
        return (DLDataTypeCode.kDLInt, 32, 1)
    elif wp_dtype == warp.uint32:
        return (DLDataTypeCode.kDLUInt, 32, 1)
    elif wp_dtype == warp.int64:
        return (DLDataTypeCode.kDLInt, 64, 1)
    elif wp_dtype == warp.uint64:
        return (DLDataTypeCode.kDLUInt, 64, 1)
    elif wp_dtype == warp.float16:
        return (DLDataTypeCode.kDLFloat, 16, 1)
    elif wp_dtype == warp.float32:
        return (DLDataTypeCode.kDLFloat, 32, 1)
    elif wp_dtype == warp.float64:
        return (DLDataTypeCode.kDLFloat, 64, 1)
    else:
        raise RuntimeError(f"No conversion from Warp type {wp_dtype} to DLPack type")


def dtype_from_dlpack(dl_dtype):
    # unpack to tuple for easier comparison
    dl_dtype = (dl_dtype.type_code.value, dl_dtype.bits)

    if dl_dtype == (DLDataTypeCode.kDLUInt, 1):
        raise RuntimeError("Warp does not support bit boolean types")
    elif dl_dtype == (DLDataTypeCode.kDLInt, 8):
        return warp.types.int8
    elif dl_dtype == (DLDataTypeCode.kDLInt, 16):
        return warp.types.int16
    elif dl_dtype == (DLDataTypeCode.kDLInt, 32):
        return warp.types.int32
    elif dl_dtype == (DLDataTypeCode.kDLInt, 64):
        return warp.types.int64
    elif dl_dtype == (DLDataTypeCode.kDLUInt, 8):
        return warp.types.uint8
    elif dl_dtype == (DLDataTypeCode.kDLUInt, 16):
        return warp.types.uint16
    elif dl_dtype == (DLDataTypeCode.kDLUInt, 32):
        return warp.types.uint32
    elif dl_dtype == (DLDataTypeCode.kDLUInt, 64):
        return warp.types.uint64
    elif dl_dtype == (DLDataTypeCode.kDLFloat, 16):
        return warp.types.float16
    elif dl_dtype == (DLDataTypeCode.kDLFloat, 32):
        return warp.types.float32
    elif dl_dtype == (DLDataTypeCode.kDLFloat, 64):
        return warp.types.float64
    elif dl_dtype == (DLDataTypeCode.kDLComplex, 64):
        raise RuntimeError("Warp does not support complex types")
    elif dl_dtype == (DLDataTypeCode.kDLComplex, 128):
        raise RuntimeError("Warp does not support complex types")
    else:
        raise RuntimeError(f"Unknown DLPack datatype {dl_dtype}")


def device_from_dlpack(dl_device):
    assert warp.context.runtime is not None, "Warp not initialized, call wp.init() before use"

    if dl_device.device_type.value == DLDeviceType.kDLCPU or dl_device.device_type.value == DLDeviceType.kDLCUDAHost:
        return warp.context.runtime.cpu_device
    elif (
        dl_device.device_type.value == DLDeviceType.kDLCUDA
        or dl_device.device_type.value == DLDeviceType.kDLCUDAManaged
    ):
        return warp.context.runtime.cuda_devices[dl_device.device_id]
    else:
        raise RuntimeError(f"Unknown device type from DLPack: {dl_device.device_type.value}")


def shape_to_dlpack(shape):
    a = (ctypes.c_int64 * len(shape))(*shape)
    return a


def strides_to_dlpack(strides, dtype):
    # convert from byte count to element count
    ndim = len(strides)
    a = (ctypes.c_int64 * ndim)()
    dtype_size = warp.types.type_size_in_bytes(dtype)
    for i in range(ndim):
        a[i] = strides[i] // dtype_size
    return a


def to_dlpack(wp_array: warp.array):
    """Convert a Warp array to another type of DLPack-compatible array.

    Args:
        wp_array: The source Warp array that will be converted.

    Returns:
        A capsule containing a DLManagedTensor that can be converted
        to another array type without copying the underlying memory.
    """

    # DLPack does not support structured arrays
    if isinstance(wp_array.dtype, warp.codegen.Struct):
        raise RuntimeError("Cannot convert structured Warp arrays to DLPack.")

    # handle vector types
    if hasattr(wp_array.dtype, "_wp_scalar_type_"):
        # vector type, flatten the dimensions into one tuple
        target_dtype = wp_array.dtype._wp_scalar_type_
        target_ndim = wp_array.ndim + len(wp_array.dtype._shape_)
        target_shape = (*wp_array.shape, *wp_array.dtype._shape_)
        dtype_strides = warp.types.strides_from_shape(wp_array.dtype._shape_, wp_array.dtype._wp_scalar_type_)
        target_strides = (*wp_array.strides, *dtype_strides)
    else:
        # scalar type
        target_dtype = wp_array.dtype
        target_ndim = wp_array.ndim
        target_shape = wp_array.shape
        target_strides = wp_array.strides

    if wp_array.pinned:
        dl_device = DLDevice()
        dl_device.device_type = DLDeviceType.kDLCUDAHost
        dl_device.device_id = 0
    else:
        dl_device = _device_to_dlpack(wp_array.device)

    # allocate DLManagedTensor, shape, and strides together
    managed_tensor_size = ctypes.sizeof(DLManagedTensor)
    padding = managed_tensor_size & 7
    shape_size = target_ndim * 8
    mem_size = managed_tensor_size + padding + 2 * shape_size
    mem_ptr = PyMem_RawMalloc(mem_size)
    assert mem_ptr, "Failed to allocate memory for DLManagedTensor"

    # set managed tensor attributes
    managed_tensor = DLManagedTensor.from_address(mem_ptr)
    managed_tensor.dl_tensor.data = wp_array.ptr
    managed_tensor.dl_tensor.device = dl_device
    managed_tensor.dl_tensor.ndim = target_ndim
    managed_tensor.dl_tensor.dtype = dtype_to_dlpack(target_dtype)
    managed_tensor.dl_tensor.byte_offset = 0

    # shape
    shape_offset = managed_tensor_size + padding
    shape_ptr = ctypes.cast(mem_ptr + shape_offset, ctypes.POINTER(ctypes.c_int64))
    for i in range(target_ndim):
        shape_ptr[i] = target_shape[i]
    managed_tensor.dl_tensor.shape = shape_ptr

    # strides, if not contiguous
    if wp_array.is_contiguous:
        managed_tensor.dl_tensor.strides = None
    else:
        stride_offset = shape_offset + shape_size
        stride_ptr = ctypes.cast(mem_ptr + stride_offset, ctypes.POINTER(ctypes.c_int64))
        dtype_size = warp.types.type_size_in_bytes(target_dtype)
        for i in range(target_ndim):
            stride_ptr[i] = target_strides[i] // dtype_size
        managed_tensor.dl_tensor.strides = stride_ptr

    # DLManagedTensor holds a reference to the source array
    managed_tensor.manager_ctx = id(wp_array)
    Py_IncRef(wp_array)

    managed_tensor.deleter = _dlpack_tensor_deleter

    # NOTE: jax.ffi.pycapsule() defines the PyCapsule_New() argtypes incorrectly, which causes problems.
    # Here we make sure that the PyCapsule_Destructor callback is correctly defined.
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor]
    PyCapsule_New.restype = ctypes.py_object

    capsule = PyCapsule_New(
        ctypes.byref(managed_tensor),
        _c_str_dltensor,
        _dlpack_capsule_deleter,
    )

    return capsule


def dtype_is_compatible(dl_dtype, wp_dtype):
    if dl_dtype.bits % 8 != 0:
        raise RuntimeError("Data types with less than 8 bits are not supported")

    if dl_dtype.type_code.value == DLDataTypeCode.kDLFloat:
        if dl_dtype.bits == 16:
            return wp_dtype == warp.float16
        elif dl_dtype.bits == 32:
            return wp_dtype == warp.float32
        elif dl_dtype.bits == 64:
            return wp_dtype == warp.float64
    elif dl_dtype.type_code.value == DLDataTypeCode.kDLInt or dl_dtype.type_code.value == DLDataTypeCode.kDLUInt:
        if dl_dtype.bits == 8:
            return wp_dtype == warp.int8 or wp_dtype == warp.uint8
        elif dl_dtype.bits == 16:
            return wp_dtype == warp.int16 or wp_dtype == warp.uint16
        elif dl_dtype.bits == 32:
            return wp_dtype == warp.int32 or wp_dtype == warp.uint32
        elif dl_dtype.bits == 64:
            return wp_dtype == warp.int64 or wp_dtype == warp.uint64
    elif dl_dtype.type_code.value == DLDataTypeCode.kDLBfloat:
        raise RuntimeError("Bfloat data type is not supported")
    elif dl_dtype.type_code.value == DLDataTypeCode.kDLComplex:
        raise RuntimeError("Complex data types are not supported")
    else:
        raise RuntimeError(f"Unsupported DLPack dtype {(str(dl_dtype.type_code), dl_dtype.bits)}")


def _from_dlpack(capsule, dtype=None) -> warp.array:
    """Convert a DLPack capsule into a Warp array without copying.

    Args:
        capsule: A DLPack capsule wrapping an external array or tensor.
        dtype: An optional Warp data type to interpret the source data.

    Returns:
        A new Warp array that uses the same underlying memory as the input capsule.
    """

    assert PyCapsule_IsValid(capsule, _c_str_dltensor), "Invalid capsule"
    mem_ptr = PyCapsule_GetPointer(capsule, _c_str_dltensor)
    managed_tensor = DLManagedTensor.from_address(mem_ptr)

    dlt = managed_tensor.dl_tensor

    device = device_from_dlpack(dlt.device)
    pinned = dlt.device.device_type.value == DLDeviceType.kDLCUDAHost
    shape = tuple(dlt.shape[dim] for dim in range(dlt.ndim))

    # strides, if not contiguous
    itemsize = dlt.dtype.bits // 8
    if dlt.strides:
        strides = tuple(dlt.strides[dim] * itemsize for dim in range(dlt.ndim))
    else:
        strides = None

    # handle multi-lane dtypes as another dimension
    if dlt.dtype.lanes > 1:
        shape = (*shape, dlt.dtype.lanes)
        if strides is not None:
            strides = (*strides, itemsize)

    if dtype is None:
        # automatically detect dtype
        dtype = dtype_from_dlpack(dlt.dtype)

    elif hasattr(dtype, "_wp_scalar_type_"):
        # handle vector/matrix types

        if not dtype_is_compatible(dlt.dtype, dtype._wp_scalar_type_):
            raise RuntimeError(f"Incompatible data types: {dlt.dtype} and {dtype}")

        dtype_shape = dtype._shape_
        dtype_dims = len(dtype._shape_)
        if dtype_dims > len(shape) or dtype_shape != shape[-dtype_dims:]:
            raise RuntimeError(
                f"Could not convert DLPack tensor with shape {shape} to Warp array with dtype={dtype}, ensure that source inner shape is {dtype_shape}"
            )

        if strides is not None:
            # ensure the inner strides are contiguous
            stride = itemsize
            for i in range(dtype_dims):
                if strides[-i - 1] != stride:
                    raise RuntimeError(
                        f"Could not convert DLPack tensor with shape {shape} to Warp array with dtype={dtype}, because the source inner strides are not contiguous"
                    )
                stride *= dtype_shape[-i - 1]
            strides = tuple(strides[:-dtype_dims]) or (itemsize,)

        shape = tuple(shape[:-dtype_dims]) or (1,)

    elif not dtype_is_compatible(dlt.dtype, dtype):
        # incompatible dtype requested
        raise RuntimeError(f"Incompatible data types: {dlt.dtype} and {dtype}")

    a = warp.types.array(
        ptr=dlt.data, dtype=dtype, shape=shape, strides=strides, copy=False, device=device, pinned=pinned
    )

    # take ownership of the DLManagedTensor
    a._dlpack_tensor_holder = _DLPackTensorHolder(mem_ptr)

    # rename the capsule so that it no longer owns the DLManagedTensor
    PyCapsule_SetName(capsule, _c_str_used_dltensor)

    return a


def from_dlpack(source, dtype=None) -> warp.array:
    """Convert a source array or DLPack capsule into a Warp array without copying.

    Args:
        source: A DLPack-compatible array or PyCapsule
        dtype: An optional Warp data type to interpret the source data.

    Returns:
        A new Warp array that uses the same underlying memory as the input
        pycapsule.
    """

    # See https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.__dlpack__.html

    if hasattr(source, "__dlpack__"):
        device_type, device_id = source.__dlpack_device__()
        # Check if the source lives on a CUDA device
        if device_type in (DLDeviceType.kDLCUDA, DLDeviceType.kDLCUDAManaged):
            # Assume that the caller will use the array on its device's current stream.
            # Note that we pass 1 for the null stream, per DLPack spec.
            cuda_stream = warp.get_cuda_device(device_id).stream.cuda_stream or 1
        elif device_type == DLDeviceType.kDLCPU:
            # No stream sync for CPU arrays.
            cuda_stream = None
        elif device_type == DLDeviceType.kDLCUDAHost:
            # For pinned memory, we sync with the current CUDA device's stream.
            # Note that we pass 1 for the null stream, per DLPack spec.
            cuda_stream = warp.get_cuda_device().stream.cuda_stream or 1
        else:
            raise TypeError("Unsupported source device")

        capsule = source.__dlpack__(stream=cuda_stream)

    else:
        # legacy behaviour, assume source is a capsule
        capsule = source

    return _from_dlpack(capsule, dtype=dtype)
