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


def get_asset_directory():
    return str(Path(__file__).resolve().parents[2] / "warp" / "examples" / "assets")


def clear_kernel_cache():
    if hasattr(wp, "clear_kernel_cache"):
        return wp.clear_kernel_cache()

    # Fallback when benchmarking older versions of Warp that didn't have
    # `clear_kernel_cache` exposed to the root namespace.
    return wp.build.clear_kernel_cache()
