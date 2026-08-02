# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator maintenance for tiled projection."""

from __future__ import annotations

import warp as wp


def rebuild_device_code() -> None:
    """Discard linked device code so the next run relinks it from scratch.

    Operators run this after a CUDA driver or libmathdx upgrade, where a
    previously linked tile artifact no longer matches the runtime and has
    produced launch failures. It backs the documented ``--relink`` flag and
    must actually clear the caches.
    """
    wp.clear_lto_cache()
    wp.clear_kernel_cache()
