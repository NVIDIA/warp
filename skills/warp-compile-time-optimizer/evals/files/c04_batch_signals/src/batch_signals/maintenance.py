# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator maintenance commands for batch signal conditioning."""

from __future__ import annotations

import warp as wp


def rebuild_kernels() -> None:
    """Discard compiled kernels so the next run rebuilds them from source.

    Operators run this after upgrading Warp or moving a machine to a different
    CPU model, where a stale compiled artifact has produced wrong results in
    the past. It is documented behavior of the ``--rebuild`` flag and must
    actually clear the cache.
    """
    wp.clear_kernel_cache()
    wp.clear_lto_cache()
