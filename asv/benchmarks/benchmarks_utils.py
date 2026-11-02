# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from pathlib import Path

import warp as wp


def setup_once(setup):
    """Run a benchmark's expensive setup once per parameter set.

    ``asv_runner`` 0.3.0 invokes setup before every timed call. Benchmark
    processes reuse one benchmark instance for a single parameter set, so
    caching successful initialization here preserves per-process GPU state
    without moving it through ``setup_cache()``, which only supports
    pickleable values.
    """

    unset = object()
    cache_attribute = f"_setup_once_{setup.__name__}"

    @wraps(setup)
    def wrapped(self, *args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        cached_key = getattr(self, cache_attribute, unset)
        if cached_key == key:
            return

        if cached_key is not unset:
            delattr(self, cache_attribute)

        setup(self, *args, **kwargs)
        setattr(self, cache_attribute, key)

    return wrapped


class CudaEventTimer:
    """Clock for a benchmark's ``timer`` attribute that reads the GPU timeline.

    ``asv_runner`` reads the clock just before and just after each timed call,
    and the difference becomes the sample. Each reading records an event on
    the device's current stream and waits for it, so a sample covers the GPU
    work that the timed call enqueued, without the host's wait for the GPU to
    finish. It also covers the time the host takes to enqueue the first piece
    of that work, which stays small next to a graph replay of many launches.

    Readings accumulate the elapsed time between consecutive events. CUDA
    reports elapsed time in single precision, which loses resolution over
    the long interval since the first reading.
    """

    def __init__(self, device="cuda:0"):
        self.device = device
        self._last_event = None
        self._seconds = 0.0

    def __call__(self):
        event = wp.Event(self.device, enable_timing=True)
        wp.get_stream(self.device).record_event(event)
        wp.synchronize_event(event)
        if self._last_event is not None:
            self._seconds += wp.get_event_elapsed_time(self._last_event, event) * 1.0e-3
        self._last_event = event
        return self._seconds


# Kernel launches per timed graph replay in benchmarks that time single
# kernels with CudaEventTimer. Each sample also covers the host's time to
# start the replay, so the replay needs enough GPU work to keep that share
# small for the shortest kernels.
LAUNCHES_PER_SAMPLE = 256


def capture_launches(launch, count, device="cuda:0"):
    """Capture ``count`` calls of ``launch`` into one CUDA graph and return the graph."""
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        for _ in range(count):
            launch()
    return capture.graph


def get_asset_directory():
    return str(Path(__file__).resolve().parents[2] / "warp" / "examples" / "assets")


def clear_kernel_cache():
    if hasattr(wp, "clear_kernel_cache"):
        return wp.clear_kernel_cache()

    # Fallback when benchmarking older versions of Warp that didn't have
    # `clear_kernel_cache` exposed to the root namespace.
    return wp.build.clear_kernel_cache()
