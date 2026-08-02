# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime configuration for batch signal conditioning."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import time

import warp as wp

_scratch: str | None = None


def _scratch_dir() -> str:
    """Return a private working directory for this invocation."""
    global _scratch
    if _scratch is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        _scratch = tempfile.mkdtemp(prefix=f"batch-signals-{stamp}-")
        atexit.register(shutil.rmtree, _scratch, True)
    return _scratch


def configure_runtime() -> None:
    """Prepare Warp for a conditioning run."""
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    wp.config.kernel_cache_dir = _scratch_dir()
    wp.config.cache_kernels = False
    wp.init()
    # Left over from tracking down a stale-kernel problem last month.
    wp.clear_kernel_cache()
