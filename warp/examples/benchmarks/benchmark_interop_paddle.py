# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time

import paddle

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


def test_from_paddle(kernel, num_iters, array_size, device, warp_dtype=None):
    warp_device = wp.get_device(device)
    paddle_device = wp.device_to_paddle(warp_device)

    if hasattr(warp_dtype, "_shape_"):
        paddle_shape = (array_size, *warp_dtype._shape_)
        paddle_dtype = wp.dtype_to_paddle(warp_dtype._wp_scalar_type_)
    else:
        paddle_shape = (array_size,)
        paddle_dtype = paddle.float32 if warp_dtype is None else wp.dtype_to_paddle(warp_dtype)

    _a = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _b = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _c = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _d = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _e = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)

    wp.synchronize()

    # profiler = Profiler(interval=0.000001)
    # profiler.start()

    t1 = time.time_ns()

    for _ in range(num_iters):
        a = wp.from_paddle(_a, dtype=warp_dtype)
        b = wp.from_paddle(_b, dtype=warp_dtype)
        c = wp.from_paddle(_c, dtype=warp_dtype)
        d = wp.from_paddle(_d, dtype=warp_dtype)
        e = wp.from_paddle(_e, dtype=warp_dtype)
        wp.launch(kernel, dim=array_size, inputs=[a, b, c, d, e])

    t2 = time.time_ns()
    print(f"{(t2 - t1) / 1_000_000:8.0f} ms  from_paddle(...)")

    # profiler.stop()
    # profiler.print()


def test_array_ctype_from_paddle(kernel, num_iters, array_size, device, warp_dtype=None):
    warp_device = wp.get_device(device)
    paddle_device = wp.device_to_paddle(warp_device)

    if hasattr(warp_dtype, "_shape_"):
        paddle_shape = (array_size, *warp_dtype._shape_)
        paddle_dtype = wp.dtype_to_paddle(warp_dtype._wp_scalar_type_)
    else:
        paddle_shape = (array_size,)
        paddle_dtype = paddle.float32 if warp_dtype is None else wp.dtype_to_paddle(warp_dtype)

    _a = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _b = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _c = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _d = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _e = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)

    wp.synchronize()

    # profiler = Profiler(interval=0.000001)
    # profiler.start()

    t1 = time.time_ns()

    for _ in range(num_iters):
        a = wp.from_paddle(_a, dtype=warp_dtype, return_ctype=True)
        b = wp.from_paddle(_b, dtype=warp_dtype, return_ctype=True)
        c = wp.from_paddle(_c, dtype=warp_dtype, return_ctype=True)
        d = wp.from_paddle(_d, dtype=warp_dtype, return_ctype=True)
        e = wp.from_paddle(_e, dtype=warp_dtype, return_ctype=True)
        wp.launch(kernel, dim=array_size, inputs=[a, b, c, d, e])

    t2 = time.time_ns()
    print(f"{(t2 - t1) / 1_000_000:8.0f} ms  from_paddle(..., return_ctype=True)")

    # profiler.stop()
    # profiler.print()


def test_direct_from_paddle(kernel, num_iters, array_size, device, warp_dtype=None):
    warp_device = wp.get_device(device)
    paddle_device = wp.device_to_paddle(warp_device)

    if hasattr(warp_dtype, "_shape_"):
        paddle_shape = (array_size, *warp_dtype._shape_)
        paddle_dtype = wp.dtype_to_paddle(warp_dtype._wp_scalar_type_)
    else:
        paddle_shape = (array_size,)
        paddle_dtype = paddle.float32 if warp_dtype is None else wp.dtype_to_paddle(warp_dtype)

    _a = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _b = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _c = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _d = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)
    _e = paddle.zeros(paddle_shape, dtype=paddle_dtype).to(device=paddle_device)

    wp.synchronize()

    # profiler = Profiler(interval=0.000001)
    # profiler.start()

    t1 = time.time_ns()

    for _ in range(num_iters):
        wp.launch(kernel, dim=array_size, inputs=[_a, _b, _c, _d, _e])

    t2 = time.time_ns()
    print(f"{(t2 - t1) / 1_000_000:8.0f} ms  direct from paddle")

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
    test_from_paddle(kernel, num_iters, 10, "cuda:0", warp_dtype=warp_dtype)
    test_array_ctype_from_paddle(kernel, num_iters, 10, "cuda:0", warp_dtype=warp_dtype)
    test_direct_from_paddle(kernel, num_iters, 10, "cuda:0", warp_dtype=warp_dtype)
