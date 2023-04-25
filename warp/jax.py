# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp


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


def to_jax(wp_array):
    import jax.dlpack

    return jax.dlpack.from_dlpack(warp.to_dlpack(wp_array))


def from_jax(jax_array, dtype=None):
    import jax.dlpack

    return warp.from_dlpack(jax.dlpack.to_dlpack(jax_array), dtype=dtype)
