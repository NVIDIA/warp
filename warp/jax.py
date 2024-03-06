# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp
from warp.context import type_str


def device_to_jax(wp_device):
    import jax

    d = warp.get_device(wp_device)

    if d.is_cuda:
        cuda_devices = jax.devices("cuda")
        if d.ordinal >= len(cuda_devices):
            raise RuntimeError(f"Jax device corresponding to '{wp_device}' is not available")
        return cuda_devices[d.ordinal]
    else:
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError(f"Jax device corresponding to '{wp_device}' is not available")
        return cpu_devices[0]


def device_from_jax(jax_device):
    if jax_device.platform == "cpu":
        return warp.get_device("cpu")
    elif jax_device.platform == "gpu":
        return warp.get_cuda_device(jax_device.id)
    else:
        raise RuntimeError(f"Unknown or unsupported Jax device platform '{jax_device.platform}'")


def dtype_to_jax(wp_dtype):
    import jax.numpy as jp

    warp_to_jax_dict = {
        warp.float16: jp.float16,
        warp.float32: jp.float32,
        warp.float64: jp.float64,
        warp.int8: jp.int8,
        warp.int16: jp.int16,
        warp.int32: jp.int32,
        warp.int64: jp.int64,
        warp.uint8: jp.uint8,
        warp.uint16: jp.uint16,
        warp.uint32: jp.uint32,
        warp.uint64: jp.uint64,
    }
    jax_dtype = warp_to_jax_dict.get(wp_dtype)
    if jax_dtype is None:
        raise TypeError(f"Invalid or unsupported data type: {type_str(wp_dtype)}")
    return jax_dtype


def dtype_from_jax(jax_dtype):
    import jax.numpy as jp

    jax_to_warp_dict = {
        jp.float16: warp.float16,
        jp.float32: warp.float32,
        jp.float64: warp.float64,
        jp.int8: warp.int8,
        jp.int16: warp.int16,
        jp.int32: warp.int32,
        jp.int64: warp.int64,
        jp.uint8: warp.uint8,
        jp.uint16: warp.uint16,
        jp.uint32: warp.uint32,
        jp.uint64: warp.uint64,
    }
    wp_dtype = jax_to_warp_dict.get(jax_dtype)
    if wp_dtype is None:
        raise TypeError(f"Invalid or unsupported data type: {jax_dtype}")
    return wp_dtype


def to_jax(wp_array):
    import jax.dlpack

    return jax.dlpack.from_dlpack(warp.to_dlpack(wp_array))


def from_jax(jax_array, dtype=None):
    import jax.dlpack

    return warp.from_dlpack(jax.dlpack.to_dlpack(jax_array), dtype=dtype)
