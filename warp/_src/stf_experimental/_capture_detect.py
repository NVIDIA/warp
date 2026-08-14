# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect FOREIGN CUDA graph captures on STF task streams.

Warp's own capture bookkeeping only covers captures Warp itself started. The
stream an STF task hands us may already be inside a capture Warp has never
seen -- STF's ``graph_ctx`` backend capturing its task graph, or an outer
``torch.cuda.graph`` region. Launching Warp kernels into such a stream
without adopting the capture would trip Warp's stream tracking, so the task
wrapper asks two questions:

1. ``_stream_is_capturing``: is this stream inside an active capture? This
   goes through Warp's native ``wp_cuda_stream_is_capturing`` export, which
   queries ``cudaStreamIsCapturing`` truthfully -- unlike Warp's *bookkeeping*,
   it also reports captures Warp has never seen.
2. ``_warp_already_tracks_capture``: is that capture OURS or foreign? Only a
   foreign capture must be adopted via ``wp.capture_begin(external=True)``;
   re-adopting one Warp already tracks would double-register it.
"""

from __future__ import annotations

import warp._src.context as _wp_ctx


def _stream_is_capturing(raw_ptr: int) -> bool:
    """Return ``True`` if ``raw_ptr`` is participating in an active CUDA graph capture."""
    return bool(_wp_ctx.runtime.core.wp_cuda_stream_is_capturing(int(raw_ptr)))


def _stream_capture_id(raw_ptr: int) -> int:
    """Return the CUDA capture id for a stream known to be capturing."""
    return int(_wp_ctx.runtime.core.wp_cuda_stream_get_capture_id(int(raw_ptr)))


def _warp_already_tracks_capture(capture_id: int) -> bool:
    """Return ``True`` if Warp already registered bookkeeping for ``capture_id``."""
    return capture_id in _wp_ctx.runtime.captures
