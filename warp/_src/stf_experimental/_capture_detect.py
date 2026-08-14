# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect FOREIGN CUDA graph captures on STF task streams.

Warp's own capture bookkeeping only covers captures Warp itself started. The
stream an STF task hands us may already be inside a capture Warp has never
seen -- STF's ``graph_ctx`` backend capturing its task graph, or an outer
``torch.cuda.graph`` region. Launching Warp kernels into such a stream
without adopting the capture would trip Warp's stream tracking, so the task
wrapper asks two questions only the CUDA runtime can answer truthfully:

1. ``_stream_is_capturing``: is this stream inside an active capture? This
   queries ``cudaStreamIsCapturing`` directly (ctypes) because Warp only
   knows about its own captures.
2. ``_warp_already_tracks_capture``: is that capture OURS or foreign? Only a
   foreign capture must be adopted via ``wp.capture_begin(external=True)``;
   re-adopting one Warp already tracks would double-register it.
"""

from __future__ import annotations

import ctypes

import warp._src.context as _wp_ctx

_CUDART: ctypes.CDLL | None = None

# cudaStreamCaptureStatus values.
_CUDA_STREAM_CAPTURE_STATUS_NONE = 0
_CUDA_STREAM_CAPTURE_STATUS_ACTIVE = 1


def _get_cudart() -> ctypes.CDLL:
    global _CUDART

    if _CUDART is None:
        try:
            cudart = ctypes.CDLL("libcudart.so")
        except OSError as exc:
            raise RuntimeError(
                "Could not dlopen libcudart.so; CUDASTF capture detection requires the CUDA Runtime "
                "to be installed and on the loader path."
            ) from exc

        cudart.cudaStreamIsCapturing.argtypes = (
            ctypes.c_void_p,  # cudaStream_t
            ctypes.POINTER(ctypes.c_int),  # cudaStreamCaptureStatus*
        )
        cudart.cudaStreamIsCapturing.restype = ctypes.c_int
        _CUDART = cudart

    return _CUDART


def _stream_is_capturing(raw_ptr: int) -> bool:
    """Return ``True`` if ``raw_ptr`` is participating in an active CUDA graph capture."""
    status = ctypes.c_int(_CUDA_STREAM_CAPTURE_STATUS_NONE)
    rc = _get_cudart().cudaStreamIsCapturing(ctypes.c_void_p(int(raw_ptr)), ctypes.byref(status))
    if rc != 0:
        raise RuntimeError(f"cudaStreamIsCapturing failed: rc={rc}")
    return status.value == _CUDA_STREAM_CAPTURE_STATUS_ACTIVE


def _stream_capture_id(raw_ptr: int) -> int:
    """Return the CUDA capture id for a stream known to be capturing."""
    return int(_wp_ctx.runtime.core.wp_cuda_stream_get_capture_id(int(raw_ptr)))


def _warp_already_tracks_capture(capture_id: int) -> bool:
    """Return ``True`` if Warp already registered bookkeeping for ``capture_id``."""
    return capture_id in _wp_ctx.runtime.captures
