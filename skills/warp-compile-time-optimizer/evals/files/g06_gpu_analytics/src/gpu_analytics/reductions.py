# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Floating-point reductions for analytics series."""

from __future__ import annotations

from typing import Any

import warp as wp


@wp.kernel(module="gpu_analytics.sum")
def sum_kernel(values: wp.array[Any], total: wp.array[Any]) -> None:
    """Accumulate series values into one total."""
    index = wp.tid()
    wp.atomic_add(total, 0, values[index])


@wp.kernel(module="gpu_analytics.energy")
def energy_kernel(values: wp.array[Any], total: wp.array[Any]) -> None:
    """Accumulate squared series values into one total."""
    index = wp.tid()
    wp.atomic_add(total, 0, values[index] * values[index])
