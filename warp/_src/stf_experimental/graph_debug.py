# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
from typing import Any

from warp._src import context as wp_context

__all__ = [
    "dump_dot",
]


def _get_cuda_graph_handle(graph: Any) -> int:
    """Internal: return the raw ``cudaGraph_t`` handle for ``graph``.

    Accepts a raw int, :class:`ctypes.c_void_p`, or an object with a ``graph``
    attribute (matches :class:`warp.Graph` and CUDASTF ``LaunchableGraph``).
    """
    if isinstance(graph, (int, ctypes.c_void_p)):
        handle = graph
    elif hasattr(graph, "graph"):
        handle = graph.graph
    else:
        raise TypeError(
            f"Cannot extract cudaGraph_t handle from {type(graph).__name__}; pass a raw int, "
            "ctypes.c_void_p, or an object with a graph attribute."
        )

    if isinstance(handle, ctypes.c_void_p):
        handle_int = handle.value or 0
    else:
        handle_int = int(handle)

    if handle_int == 0:
        raise ValueError("cudaGraph_t handle is null; the graph may not have been instantiated yet.")

    return handle_int


def dump_dot(graph: Any, path: str, *, flags: int = 0) -> str:
    """Dump a CUDA graph to DOT.

    This borrows a ``cudaGraph_t`` handle and forwards it to Warp's native
    ``cudaGraphDebugDotPrint`` wrapper. It works with native
    :class:`warp.Graph` instances and CUDASTF launchable graphs without taking
    ownership of the graph.

    Args:
        graph: A :class:`warp.Graph`, a CUDASTF ``LaunchableGraph``, or any
            object exposing the underlying ``cudaGraph_t`` handle as a raw
            int, :class:`ctypes.c_void_p`, or a ``graph`` attribute.
        path: Output DOT path.
        flags: Flags forwarded to ``cudaGraphDebugDotPrint``.

    Returns:
        The output path that was requested.
    """
    # TODO: Graduate this helper to stable warp.utils.dump_cuda_graph; it is
    # useful for any Warp CUDA graph, not just CUDASTF graphs.
    handle = _get_cuda_graph_handle(graph)

    if wp_context.runtime is None:
        wp_context.init()

    if not wp_context.runtime.core.wp_capture_debug_dot_print(ctypes.c_void_p(handle), path.encode("utf-8"), flags):
        raise RuntimeError(f"Graph debug dot print error: {wp_context.runtime.get_error_string()}")

    return path
