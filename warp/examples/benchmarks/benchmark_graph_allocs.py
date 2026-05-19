# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for graph allocation handling."""

import ctypes
import shutil
import statistics
import subprocess
import time

import warp as wp


@wp.kernel
def dummy_kernel():
    pass


def bench_multistream_alloc_capture(
    num_iters, num_allocs, num_kernels, num_streams=1, num_launches=20, device=None, show_graph=False
):
    with wp.ScopedDevice(device):
        streams = [wp.Stream(device) for _ in range(num_streams)]

        # warmup
        for s in streams:
            with wp.ScopedStream(s):
                for _ in range(5):
                    wp.empty(10)
                    wp.launch(dummy_kernel, dim=1)

        # capture
        wp.synchronize_device()
        capture_t1 = time.time_ns()
        with wp.ScopedCapture() as capture:
            for s in streams:
                with wp.ScopedStream(s):
                    for _ in range(num_iters):
                        arrays = [wp.empty(10) for _ in range(num_allocs)]
                        for _ in range(num_kernels):
                            wp.launch(dummy_kernel, dim=1)
                        del arrays
        wp.synchronize_device()
        capture_t2 = time.time_ns()
        capture_time = (capture_t2 - capture_t1) / (1_000_000)

        # instantiate
        wp.synchronize_device()
        instantiate_t1 = time.time_ns()
        g = ctypes.c_void_p()
        result = wp._src.context.runtime.core.wp_cuda_graph_create_exec(
            capture.graph.device.context, wp.get_stream().cuda_stream, capture.graph.graph, ctypes.byref(g)
        )
        if not result:
            raise RuntimeError(f"Graph creation error: {wp._src.context.runtime.get_error_string()}")
        capture.graph.graph_exec = g
        wp.synchronize_device()
        instantiate_t2 = time.time_ns()
        instantiate_time = (instantiate_t2 - instantiate_t1) / (1_000_000)

        # launch
        wp.synchronize_device()
        launch_t1 = time.time_ns()
        for _ in range(num_launches):
            wp.capture_launch(capture.graph)
        wp.synchronize_device()
        launch_t2 = time.time_ns()
        launch_time = (launch_t2 - launch_t1) / num_launches / (1_000_000)

        # generate and show the graph
        if show_graph:
            basename = f"graph_{num_iters:03d}_{num_allocs:03d}_{num_kernels:03d}_{num_streams:03d}"
            dotname = f"{basename}.dot"
            imgname = f"{basename}.png"
            print(f"Writing {dotname}")
            wp.capture_debug_dot_print(capture.graph, dotname, verbose=False)
            if shutil.which("dot"):
                print(f"Writing {imgname}")
                subprocess.run(["dot", "-Tpng", dotname, f"-o{imgname}"], capture_output=True, text=True, check=True)
                if shutil.which("xdg-open"):
                    subprocess.run(["xdg-open", imgname], capture_output=True, text=True, check=True)

        return capture_time, instantiate_time, launch_time


def bench_runner(func, args, num_runs):
    capture_times = []
    instantiate_times = []
    launch_times = []
    for _ in range(num_runs):
        tc, ti, tl = func(*args)
        capture_times.append(tc)
        instantiate_times.append(ti)
        launch_times.append(tl)

    capture_times = sorted(capture_times)
    instantiate_times = sorted(instantiate_times)
    launch_times = sorted(launch_times)
    if num_runs > 3:
        trim = num_runs // 4
        capture_times = capture_times[trim:-trim]
        instantiate_times = instantiate_times[trim:-trim]
        launch_times = launch_times[trim:-trim]

    return statistics.mean(capture_times), statistics.mean(instantiate_times), statistics.mean(launch_times)


if __name__ == "__main__":
    wp.init()

    # print(bench_multistream_alloc_capture(3, 3, 3, 3, show_graph=True))
    # quit()

    small_counts = [1, 5, 10]
    medium_counts = [5, 10, 20]
    large_counts = [10, 20, 50]

    iter_counts = medium_counts
    alloc_counts = medium_counts
    kernel_counts = medium_counts
    stream_counts = medium_counts

    num_reps = 20
    num_launches = 20
    capture_times = []
    instantiate_times = []
    launch_times = []
    for num_iters in iter_counts:
        for num_allocs in alloc_counts:
            for num_kernels in kernel_counts:
                for num_streams in stream_counts:
                    tc, ti, tl = bench_runner(
                        bench_multistream_alloc_capture,
                        (num_iters, num_allocs, num_kernels, num_streams, num_launches),
                        num_reps,
                    )
                    print(
                        f"{num_iters:4d} {num_allocs:4d} {num_kernels:4d} {num_streams:4d}: {tc:10.4f} ms {ti:10.4f} ms {tl:10.4f} ms"
                    )
                    capture_times.append(tc)
                    instantiate_times.append(ti)
                    launch_times.append(tl)

    print()
    print(f"CAPTURE TOTAL:     {sum(sorted(capture_times)):10.4f} ms")
    print(f"INSTANTIATE TOTAL: {sum(sorted(instantiate_times)):10.4f} ms")
    print(f"LAUNCH TOTAL:      {sum(sorted(launch_times)):10.4f} ms")
