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

import gc
import statistics as stats

import warp as wp

ENABLE_MEMPOOLS = False
ENABLE_PEER_ACCESS = False
ENABLE_MEMPOOL_ACCESS = False
ENABLE_MEMPOOL_RELEASE_THRESHOLD = False

MEMPOOL_RELEASE_THRESHOLD = 1024 * 1024 * 1024

DO_SYNC = False
VERBOSE = False
USE_NVTX = False

num_elems = 10000
num_runs = 10000
trim_runs = 2500


@wp.kernel
def inc_kernel(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


# configure devices
for target_device in wp.get_cuda_devices():
    try:
        wp.set_mempool_enabled(target_device, ENABLE_MEMPOOLS)
        if ENABLE_MEMPOOL_RELEASE_THRESHOLD:
            wp.set_mempool_release_threshold(target_device, MEMPOOL_RELEASE_THRESHOLD)
    except Exception as e:
        print(f"Error: {e}")

    for peer_device in wp.get_cuda_devices():
        try:
            wp.set_peer_access_enabled(target_device, peer_device, ENABLE_PEER_ACCESS)
        except Exception as e:
            print(f"Error: {e}")

        try:
            wp.set_mempool_access_enabled(target_device, peer_device, ENABLE_MEMPOOL_ACCESS)
        except Exception as e:
            print(f"Error: {e}")

cuda_device_count = wp.get_cuda_device_count()

cuda0 = wp.get_device("cuda:0")

# preallocate some arrays
arr_host = wp.zeros(num_elems, dtype=float, device="cpu", pinned=False)
arr_host_pinned = wp.zeros(num_elems, dtype=float, device="cpu", pinned=True)
arr_cuda0 = wp.zeros(num_elems, dtype=float, device=cuda0)
arr_cuda0_src = wp.zeros(num_elems, dtype=float, device=cuda0)
arr_cuda0_dst = wp.zeros(num_elems, dtype=float, device=cuda0)

# mgpu support
if cuda_device_count > 1:
    cuda1 = wp.get_device("cuda:1")
    arr_cuda1 = wp.zeros(num_elems, dtype=float, device=cuda1)

stream0 = wp.Stream(cuda0)

# preload module
wp.force_load(cuda0)
if cuda_device_count > 1:
    wp.force_load(cuda1)

# capture graph
with wp.ScopedDevice(cuda0):
    wp.capture_begin()
    wp.launch(inc_kernel, dim=arr_cuda0.size, inputs=[arr_cuda0])
    graph0 = wp.capture_end()


g_allocs = [None] * num_runs


def test_alloc(num_elems, device, idx):
    wp.synchronize()

    with wp.ScopedTimer("alloc", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        g_allocs[idx] = wp.empty(num_elems, dtype=float, device=device)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_free(device, idx):
    wp.synchronize()

    with wp.ScopedTimer("free", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        g_allocs[idx] = None

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_zeros(num_elems, device, idx):
    wp.synchronize()

    with wp.ScopedTimer("zeros", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        g_allocs[idx] = wp.zeros(num_elems, dtype=float, device=device)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_h2d(num_elems, device):
    wp.synchronize()

    with wp.ScopedTimer("h2d", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.copy(arr_cuda0, arr_host)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_d2h(num_elems, device):
    wp.synchronize()

    with wp.ScopedTimer("d2h", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.copy(arr_host, arr_cuda0)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_h2d_pinned(num_elems, device):
    wp.synchronize()

    with wp.ScopedTimer("h2d pinned", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.copy(arr_cuda0, arr_host_pinned)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_d2h_pinned(num_elems, device):
    wp.synchronize()

    with wp.ScopedTimer("d2h pinned", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.copy(arr_host_pinned, arr_cuda0)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_d2d(num_elems, device):
    wp.synchronize()

    with wp.ScopedTimer("d2d", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.copy(arr_cuda0_dst, arr_cuda0_src)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_p2p(num_elems, src_device, dst_device):
    wp.synchronize()

    with wp.ScopedTimer("p2p", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.copy(arr_cuda0, arr_cuda1)

        if DO_SYNC:
            wp.synchronize_device(src_device)
            wp.synchronize_device(dst_device)

    return timer.elapsed


def test_p2p_stream(num_elems, src_device, dst_device):
    stream = stream0

    wp.synchronize()

    with wp.ScopedTimer("p2p stream", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.copy(arr_cuda0, arr_cuda1, stream=stream)

        if DO_SYNC:
            wp.synchronize_device(src_device)
            wp.synchronize_device(dst_device)

    return timer.elapsed


def test_launch(num_elems, device):
    a = arr_cuda0

    wp.synchronize()

    with wp.ScopedTimer("launch", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.launch(inc_kernel, dim=a.size, inputs=[a], device=device)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_launch_stream(num_elems, device):
    a = arr_cuda0
    stream = stream0

    wp.synchronize()

    with wp.ScopedTimer("launch stream", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.launch(inc_kernel, dim=a.size, inputs=[a], stream=stream)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_graph(num_elems, device):
    wp.synchronize()

    with wp.ScopedTimer("graph", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.capture_launch(graph0)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


def test_graph_stream(num_elems, device):
    wp.synchronize()

    with wp.ScopedTimer("graph", print=VERBOSE, use_nvtx=USE_NVTX) as timer:
        wp.capture_launch(graph0, stream=stream0)

        if DO_SYNC:
            wp.synchronize_device(device)

    return timer.elapsed


alloc_times = [0] * num_runs
free_times = [0] * num_runs
zeros_times = [0] * num_runs
d2h_times = [0] * num_runs
h2d_times = [0] * num_runs
d2h_pinned_times = [0] * num_runs
h2d_pinned_times = [0] * num_runs
d2d_times = [0] * num_runs
p2p_times = [0] * num_runs
p2p_stream_times = [0] * num_runs
launch_times = [0] * num_runs
launch_stream_times = [0] * num_runs
graph_times = [0] * num_runs
graph_stream_times = [0] * num_runs

wp.set_device(cuda0)

# alloc
for i in range(num_runs):
    gc.disable()
    alloc_times[i] = test_alloc(num_elems, cuda0, i)
    gc.enable()

# free
for i in range(num_runs):
    gc.disable()
    free_times[i] = test_free(cuda0, i)
    gc.enable()

# zeros
for i in range(num_runs):
    gc.disable()
    zeros_times[i] = test_zeros(num_elems, cuda0, i)
    gc.enable()

# free zeros
for i in range(num_runs):
    g_allocs[i] = None

# h2d, d2h pageable copy
for i in range(num_runs):
    gc.disable()
    h2d_times[i] = test_h2d(num_elems, cuda0)
    d2h_times[i] = test_d2h(num_elems, cuda0)
    gc.enable()

# h2d, d2h pinned copy
for i in range(num_runs):
    gc.disable()
    h2d_pinned_times[i] = test_h2d_pinned(num_elems, cuda0)
    d2h_pinned_times[i] = test_d2h_pinned(num_elems, cuda0)
    gc.enable()

# d2d copy
for i in range(num_runs):
    gc.disable()
    d2d_times[i] = test_d2d(num_elems, cuda0)
    gc.enable()

# p2p copy
if cuda_device_count > 1:
    for i in range(num_runs):
        gc.disable()
        p2p_times[i] = test_p2p(num_elems, cuda1, cuda0)
        p2p_stream_times[i] = test_p2p_stream(num_elems, cuda1, cuda0)
        gc.enable()

# launch
for i in range(num_runs):
    gc.disable()
    launch_times[i] = test_launch(num_elems, cuda0)
    launch_stream_times[i] = test_launch_stream(num_elems, cuda0)
    gc.enable()

# graph
for i in range(num_runs):
    gc.disable()
    graph_times[i] = test_graph(num_elems, cuda0)
    graph_stream_times[i] = test_graph_stream(num_elems, cuda0)
    gc.enable()


def print_stat(name, data, trim=trim_runs):
    assert len(data) - 2 * trim > 0
    if trim > 0:
        data = sorted(data)[trim:-trim]
    print(f"{name:15s} {1000000 * stats.mean(data):.0f}")


print("=========================")
print_stat("Alloc", alloc_times)
print_stat("Free", free_times)
print_stat("Zeros", zeros_times)
print_stat("H2D", h2d_times)
print_stat("D2H", d2h_times)
print_stat("H2D pinned", h2d_pinned_times)
print_stat("D2H pinned", d2h_pinned_times)
print_stat("D2D", d2d_times)
print_stat("P2P", p2p_times)
print_stat("P2P stream", p2p_stream_times)
print_stat("Launch", launch_times)
print_stat("Launch stream", launch_stream_times)
print_stat("Graph", graph_times)
print_stat("Graph stream", graph_stream_times)


# ========= profiling ==========

# from pyinstrument import Profiler
# profiler = Profiler()
# profiler.start()
# for i in range(10):
#     # test_alloc(num_elems, cuda0)
#     # test_h2d(num_elems, cuda0)
#     test_p2p(num_elems, cuda0, cuda1)
# profiler.stop()
# print(profiler.output_text(show_all=True))
