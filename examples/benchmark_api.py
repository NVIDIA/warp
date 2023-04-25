# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
import gc


@wp.kernel
def inc_kernel(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


@wp.kernel
def dec_kernel(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] - 1.0


def test_allocs(n, device, do_sync=False):
    arrs = [None] * n

    with wp.ScopedTimer("allocs"):
        for i in range(n):
            arrs[i] = wp.zeros(1, device=device)

        if do_sync:
            wp.synchronize()

    return arrs


def test_allocs_v2(n, device, do_sync=False):
    arrs = [None] * n

    with wp.ScopedTimer("allocs_v2"), wp.ScopedDevice(device):
        for i in range(n):
            arrs[i] = wp.zeros(1)

        if do_sync:
            wp.synchronize()

    return arrs


def test_launches(n, device, do_sync=False):
    arr = wp.zeros(1, dtype=wp.float32, device=device)
    wp.synchronize()

    with wp.ScopedTimer("launches"):
        for _ in range(n):
            wp.launch(inc_kernel, dim=arr.size, inputs=[arr], device=device)
            wp.launch(dec_kernel, dim=arr.size, inputs=[arr], device=device)

        if do_sync:
            wp.synchronize()


def test_launches_v2(n, device, do_sync=False):
    arr = wp.zeros(1, dtype=wp.float32, device=device)
    wp.synchronize()

    with wp.ScopedTimer("launches_v2"), wp.ScopedDevice(device):
        for _ in range(n):
            wp.launch(inc_kernel, dim=arr.size, inputs=[arr])
            wp.launch(dec_kernel, dim=arr.size, inputs=[arr])

        if do_sync:
            wp.synchronize()


def test_copies(n, do_sync=False):
    a = wp.zeros(1, dtype=wp.float32, device="cpu")
    b = wp.zeros(1, dtype=wp.float32, device="cuda")
    c = wp.zeros(1, dtype=wp.float32, device="cuda")

    wp.synchronize()

    with wp.ScopedTimer("copies"):
        for _ in range(n):
            wp.copy(b, a)
            wp.copy(c, b)
            wp.copy(a, c)

        if do_sync:
            wp.synchronize()


def test_graphs(n, device, do_sync=False):
    arr = wp.zeros(1, dtype=wp.float32, device=device)
    wp.synchronize()

    wp.capture_begin()
    wp.launch(inc_kernel, dim=arr.size, inputs=[arr], device=device)
    wp.launch(dec_kernel, dim=arr.size, inputs=[arr], device=device)
    graph = wp.capture_end()
    wp.synchronize()

    with wp.ScopedTimer("graphs"):
        for _ in range(n):
            wp.capture_launch(graph)

        if do_sync:
            wp.synchronize()


wp.init()

wp.force_load()

device = "cuda"
n = 100000

# make sure the context gets fully initialized now
_a = wp.zeros(1, device=device)
wp.launch(inc_kernel, dim=_a.size, inputs=[_a], device=device)
wp.synchronize()
gc.collect()

test_allocs(n, device)
wp.synchronize()
gc.collect()

test_allocs_v2(n, device)
wp.synchronize()
gc.collect()

test_launches(n, device)
wp.synchronize()
gc.collect()

test_launches_v2(n, device)
wp.synchronize()
gc.collect()

test_copies(n)
wp.synchronize()
gc.collect()

test_graphs(n, device)
wp.synchronize()
gc.collect()


# ========= profiling ==========#

# import cProfile
# cProfile.run('test_allocs(n, device)')

# from pyinstrument import Profiler
# profiler = Profiler()
# profiler.start()
# #arrs = test_allocs(n, device)
# test_launches(n, device)
# #test_copies(n)
# profiler.stop()
# print(profiler.output_text(show_all=True))
