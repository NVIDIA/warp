# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduction stage: histogram and order statistics."""

from __future__ import annotations

import warp as wp

# This module is always launched at BLOCK_DIM, so declare it as the module
# default too: a preload that does not name a block dimension then builds the
# variant this stage actually uses instead of a second one.
wp.set_module_options({"enable_backward": False, "block_dim": 32})

BLOCK_DIM = 32
BUCKETS = 16


@wp.kernel
def build_histogram(
    values: wp.array[wp.float32],
    counts: wp.array[wp.float32],
    low: wp.float32,
    high: wp.float32,
) -> None:
    """Bin values into a fixed histogram."""
    index = wp.tid()
    span = wp.max(high - low, wp.float32(1.0e-6))
    slot = wp.int32((values[index] - low) / span * wp.float32(counts.shape[0]))
    wp.atomic_add(counts, wp.clamp(slot, 0, counts.shape[0] - 1), wp.float32(1.0))


@wp.kernel
def normalise_histogram(counts: wp.array[wp.float32], shares: wp.array[wp.float32], samples: wp.float32) -> None:
    """Convert counts to shares of the total."""
    slot = wp.tid()
    shares[slot] = counts[slot] / samples


@wp.kernel
def cumulative_share(shares: wp.array[wp.float32], cumulative: wp.array[wp.float32]) -> None:
    """Running total of the histogram shares."""
    slot = wp.tid()
    running = wp.float32(0.0)
    for other in range(slot + 1):
        running += shares[other]
    cumulative[slot] = running


@wp.kernel
def histogram_entropy(shares: wp.array[wp.float32], entropy: wp.array[wp.float32]) -> None:
    """Shannon entropy of the histogram, in nats."""
    slot = wp.tid()
    share = shares[slot]
    if share > 0.0:
        wp.atomic_add(entropy, 0, -share * wp.log(share))
