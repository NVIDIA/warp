# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time

import torch

import warp as wp


def create_simple_kernel(dtype):
    def simple_kernel(
        a: wp.array(dtype=dtype),
        b: wp.array(dtype=dtype),
        c: wp.array(dtype=dtype),
        d: wp.array(dtype=dtype),
        e: wp.array(dtype=dtype),
    ):
        pass

    return wp.Kernel(simple_kernel)


def test_from_torch(kernel, num_iters, array_size, device, warp_dtype=None):
    warp_device = wp.get_device(device)
    torch_device = wp.device_to_torch(warp_device)

    if hasattr(warp_dtype, "_shape_"):
        torch_shape = (array_size, *warp_dtype._shape_)
        torch_dtype = wp.dtype_to_torch(warp_dtype._wp_scalar_type_)
    else:
        torch_shape = (array_size,)
        torch_dtype = torch.float32 if warp_dtype is None else wp.dtype_to_torch(warp_dtype)

    _a = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _b = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _c = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _d = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _e = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)

    wp.synchronize()

    # profiler = Profiler(interval=0.000001)
    # profiler.start()

    t1 = time.time_ns()

    for _ in range(num_iters):
        a = wp.from_torch(_a, dtype=warp_dtype)
        b = wp.from_torch(_b, dtype=warp_dtype)
        c = wp.from_torch(_c, dtype=warp_dtype)
        d = wp.from_torch(_d, dtype=warp_dtype)
        e = wp.from_torch(_e, dtype=warp_dtype)
        wp.launch(kernel, dim=array_size, inputs=[a, b, c, d, e])

    t2 = time.time_ns()
    print(f"{(t2 - t1) / 1_000_000 :8.0f} ms  from_torch(...)")

    # profiler.stop()
    # profiler.print()


def test_array_ctype_from_torch(kernel, num_iters, array_size, device, warp_dtype=None):
    warp_device = wp.get_device(device)
    torch_device = wp.device_to_torch(warp_device)

    if hasattr(warp_dtype, "_shape_"):
        torch_shape = (array_size, *warp_dtype._shape_)
        torch_dtype = wp.dtype_to_torch(warp_dtype._wp_scalar_type_)
    else:
        torch_shape = (array_size,)
        torch_dtype = torch.float32 if warp_dtype is None else wp.dtype_to_torch(warp_dtype)

    _a = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _b = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _c = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _d = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _e = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)

    wp.synchronize()

    # profiler = Profiler(interval=0.000001)
    # profiler.start()

    t1 = time.time_ns()

    for _ in range(num_iters):
        a = wp.from_torch(_a, dtype=warp_dtype, return_ctype=True)
        b = wp.from_torch(_b, dtype=warp_dtype, return_ctype=True)
        c = wp.from_torch(_c, dtype=warp_dtype, return_ctype=True)
        d = wp.from_torch(_d, dtype=warp_dtype, return_ctype=True)
        e = wp.from_torch(_e, dtype=warp_dtype, return_ctype=True)
        wp.launch(kernel, dim=array_size, inputs=[a, b, c, d, e])

    t2 = time.time_ns()
    print(f"{(t2 - t1) / 1_000_000 :8.0f} ms  from_torch(..., return_ctype=True)")

    # profiler.stop()
    # profiler.print()


def test_direct_from_torch(kernel, num_iters, array_size, device, warp_dtype=None):
    warp_device = wp.get_device(device)
    torch_device = wp.device_to_torch(warp_device)

    if hasattr(warp_dtype, "_shape_"):
        torch_shape = (array_size, *warp_dtype._shape_)
        torch_dtype = wp.dtype_to_torch(warp_dtype._wp_scalar_type_)
    else:
        torch_shape = (array_size,)
        torch_dtype = torch.float32 if warp_dtype is None else wp.dtype_to_torch(warp_dtype)

    _a = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _b = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _c = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _d = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)
    _e = torch.zeros(torch_shape, dtype=torch_dtype, device=torch_device)

    wp.synchronize()

    # profiler = Profiler(interval=0.000001)
    # profiler.start()

    t1 = time.time_ns()

    for _ in range(num_iters):
        wp.launch(kernel, dim=array_size, inputs=[_a, _b, _c, _d, _e])

    t2 = time.time_ns()
    print(f"{(t2 - t1) / 1_000_000 :8.0f} ms  direct from torch")

    # profiler.stop()
    # profiler.print()


wp.init()

params = [
    # (warp_dtype arg, kernel)
    (None, create_simple_kernel(wp.float32)),
    (wp.float32, create_simple_kernel(wp.float32)),
    (wp.vec3f, create_simple_kernel(wp.vec3f)),
    (wp.mat22f, create_simple_kernel(wp.mat22f)),
]

wp.load_module()

num_iters = 100000

for warp_dtype, kernel in params:
    print(f"\ndtype={wp.context.type_str(warp_dtype)}")
    test_from_torch(kernel, num_iters, 10, "cuda:0", warp_dtype=warp_dtype)
    test_array_ctype_from_torch(kernel, num_iters, 10, "cuda:0", warp_dtype=warp_dtype)
    test_direct_from_torch(kernel, num_iters, 10, "cuda:0", warp_dtype=warp_dtype)
