# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp
import ctypes

from warp.thirdparty.dlpack import (
    DLManagedTensor,
    DLDevice,
    DLDeviceType,
    DLDataType,
    DLDataTypeCode,
    DLTensor,
    _c_str_dltensor,
)

ctypes.pythonapi.PyMem_RawMalloc.restype = ctypes.c_void_p
ctypes.pythonapi.PyMem_RawFree.argtypes = [ctypes.c_void_p]

ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

ctypes.pythonapi.PyCapsule_IsValid.restype = ctypes.c_int
ctypes.pythonapi.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


class _Holder:
    def __init__(self, wp_array) -> None:
        self.wp_array = wp_array

    def _as_manager_ctx(self) -> ctypes.c_void_p:
        py_obj = ctypes.py_object(self)
        py_obj_ptr = ctypes.pointer(py_obj)
        ctypes.pythonapi.Py_IncRef(py_obj)
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(py_obj_ptr))
        return ctypes.cast(py_obj_ptr, ctypes.c_void_p)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _warp_array_deleter(handle: ctypes.c_void_p) -> None:
    """A function to deallocate the memory of a Warp array."""

    dl_managed_tensor = DLManagedTensor.from_address(handle)
    py_obj_ptr = ctypes.cast(dl_managed_tensor.manager_ctx, ctypes.POINTER(ctypes.py_object))
    py_obj = py_obj_ptr.contents
    ctypes.pythonapi.Py_DecRef(py_obj)
    ctypes.pythonapi.Py_DecRef(ctypes.py_object(py_obj_ptr))
    ctypes.pythonapi.PyMem_RawFree(handle)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _warp_pycapsule_deleter(handle: ctypes.c_void_p) -> None:
    """A function to deallocate a pycapsule that wraps a Warp array."""

    pycapsule: ctypes.py_object = ctypes.cast(handle, ctypes.py_object)
    if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor):
        dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(pycapsule, _c_str_dltensor)
        _warp_array_deleter(dl_managed_tensor)
        ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)


def device_to_dlpack(wp_device) -> DLDevice:
    d = warp.get_device(wp_device)

    if d.is_cpu:
        device_type = DLDeviceType.kDLCPU
        device_id = 0
    elif d.is_cuda:
        device_type = DLDeviceType.kDLCUDA
        device_id = d.ordinal
    else:
        raise RuntimeError("Unhandled device type converting to dlpack")

    dl_device = DLDevice()
    dl_device.device_type = device_type
    dl_device.device_id = device_id

    return dl_device


def dtype_to_dlpack(wp_dtype) -> DLDataType:
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
        raise RuntimeError(f"Unknown dlpack datatype {dl_dtype}")


def device_from_dlpack(dl_device):
    if dl_device.device_type.value == DLDeviceType.kDLCPU or dl_device.device_type.value == DLDeviceType.kDLCUDAHost:
        return "cpu"
    elif (
        dl_device.device_type.value == DLDeviceType.kDLCUDA
        or dl_device.device_type.value == DLDeviceType.kDLCUDAManaged
    ):
        return f"cuda:{dl_device.device_id}"
    else:
        raise RuntimeError(f"Unknown device type from dlpack: {dl_device.device_type.value}")


def shape_to_dlpack(shape):
    a = (ctypes.c_int64 * len(shape))(*shape)
    return a


def strides_to_dlpack(strides, dtype):
    # convert from byte count to element count
    s = []
    for i in range(len(strides)):
        s.append(int(int(strides[i]) / int(warp.types.type_size_in_bytes(dtype))))

    a = (ctypes.c_int64 * len(strides))(*s)
    return a


def to_dlpack(wp_array: warp.array):
    """Convert a Warp array to another type of dlpack compatible array.

    Parameters
    ----------
    np_array : np.ndarray
        The source numpy array that will be converted.

    Returns
    -------
    pycapsule : PyCapsule
        A pycapsule containing a DLManagedTensor that can be converted
        to other array formats without copying the underlying memory.
    """

    # DLPack does not support structured arrays
    if isinstance(wp_array.dtype, warp.codegen.Struct):
        raise RuntimeError("Cannot convert structured Warp arrays to DLPack.")

    holder = _Holder(wp_array)

    # allocate DLManagedTensor
    size = ctypes.c_size_t(ctypes.sizeof(DLManagedTensor))
    dl_managed_tensor = DLManagedTensor.from_address(ctypes.pythonapi.PyMem_RawMalloc(size))

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

    # store the shape and stride arrays with the holder to prevent them from getting deallocated
    holder._shape = shape_to_dlpack(target_shape)
    holder._strides = strides_to_dlpack(target_strides, target_dtype)

    if wp_array.pinned:
        dl_device = DLDeviceType.kDLCUDAHost
    else:
        dl_device = device_to_dlpack(wp_array.device)

    # set Tensor attributes
    dl_managed_tensor.dl_tensor.data = wp_array.ptr
    dl_managed_tensor.dl_tensor.device = dl_device
    dl_managed_tensor.dl_tensor.ndim = target_ndim
    dl_managed_tensor.dl_tensor.dtype = dtype_to_dlpack(target_dtype)
    dl_managed_tensor.dl_tensor.shape = holder._shape
    dl_managed_tensor.dl_tensor.strides = holder._strides
    dl_managed_tensor.dl_tensor.byte_offset = 0
    dl_managed_tensor.manager_ctx = holder._as_manager_ctx()
    dl_managed_tensor.deleter = _warp_array_deleter

    pycapsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.byref(dl_managed_tensor),
        _c_str_dltensor,
        _warp_pycapsule_deleter,
    )

    return pycapsule


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
        raise RuntimeError(f"Unsupported dlpack dtype {(str(dl_dtype.type_code), dl_dtype.bits)}")


def from_dlpack(pycapsule, dtype=None) -> warp.array:
    """Convert a dlpack tensor into a numpy array without copying.

    Parameters
    ----------
    pycapsule : PyCapsule
        A pycapsule wrapping a dlpack tensor that will be converted.

    Returns
    -------
    np_array : np.ndarray
        A new numpy array that uses the same underlying memory as the input
        pycapsule.
    """

    assert ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor)
    dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(pycapsule, _c_str_dltensor)
    dl_managed_tensor_ptr = ctypes.cast(dl_managed_tensor, ctypes.POINTER(DLManagedTensor))
    dl_managed_tensor = dl_managed_tensor_ptr.contents

    dlt = dl_managed_tensor.dl_tensor
    assert isinstance(dlt, DLTensor)

    device = device_from_dlpack(dlt.device)

    pinned = dlt.device.device_type.value == DLDeviceType.kDLCUDAHost

    shape = tuple(dlt.shape[dim] for dim in range(dlt.ndim))

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
        ptr=dlt.data, dtype=dtype, shape=shape, strides=strides, copy=False, owner=False, device=device, pinned=pinned
    )

    # keep a reference to the capsule so it doesn't get deleted
    a._pycapsule = pycapsule

    return a
