# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistics stage: robust summaries over a window."""

from __future__ import annotations

import warp as wp

# This module is always launched at BLOCK_DIM, so declare it as the module
# default too: a preload that does not name a block dimension then builds the
# variant this stage actually uses instead of a second one.
wp.set_module_options({"enable_backward": False, "block_dim": 256})

BLOCK_DIM = 256


@wp.kernel
def rolling_mean(values: wp.array[wp.float32], means: wp.array[wp.float32], radius: wp.int32) -> None:
    """Mean over a centred window."""
    index = wp.tid()
    total = wp.float32(0.0)
    count = wp.int32(0)
    for offset in range(-radius, radius + 1):
        other = index + offset
        if other >= 0 and other < values.shape[0]:
            total += values[other]
            count += 1
    means[index] = total / wp.float32(count)


@wp.kernel
def rolling_spread(
    values: wp.array[wp.float32],
    means: wp.array[wp.float32],
    spread: wp.array[wp.float32],
    radius: wp.int32,
) -> None:
    """Standard deviation over the same window."""
    index = wp.tid()
    total = wp.float32(0.0)
    count = wp.int32(0)
    centre = means[index]
    for offset in range(-radius, radius + 1):
        other = index + offset
        if other >= 0 and other < values.shape[0]:
            difference = values[other] - centre
            total += difference * difference
            count += 1
    spread[index] = wp.sqrt(total / wp.float32(count))


@wp.kernel
def robust_score(
    values: wp.array[wp.float32],
    means: wp.array[wp.float32],
    spread: wp.array[wp.float32],
    scores: wp.array[wp.float32],
) -> None:
    """Standardised deviation, guarded against zero spread."""
    index = wp.tid()
    scores[index] = (values[index] - means[index]) / wp.max(spread[index], wp.float32(1.0e-6))


@wp.kernel
def flag_outliers(scores: wp.array[wp.float32], flags: wp.array[wp.float32], limit: wp.float32) -> None:
    """Mark samples beyond ``limit`` standard deviations."""
    index = wp.tid()
    if wp.abs(scores[index]) > limit:
        flags[index] = wp.float32(1.0)
    else:
        flags[index] = wp.float32(0.0)
