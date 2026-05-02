# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class AllocatorRmm:
    """Allocator that routes Warp device memory through RAPIDS Memory Manager (RMM).

    Each allocation delegates to ``rmm.DeviceBuffer``, which uses whichever
    ``DeviceMemoryResource`` is active at the time of allocation (as set by
    ``rmm.mr.set_current_device_resource()``). Changing the RMM resource
    between allocations will affect subsequent allocations.

    Requires the ``rmm`` package (Linux only, ``pip install rmm-cu12``).

    A single ``AllocatorRmm`` instance can safely be shared across multiple
    CUDA devices. Allocations always happen on the correct device because
    ``warp.array`` wraps each ``allocate()`` call in a ``device.context_guard``.
    This class is not thread-safe; concurrent calls from multiple threads
    require external synchronization.

    Each allocation is stream-ordered on the current Warp device stream,
    matching the pattern used by CuPy's RMM integration. This ensures correct
    behavior with stream-ordered memory resources (e.g.,
    ``rmm.mr.CudaAsyncMemoryResource``) and during CUDA graph capture.

    Example:
        .. code-block:: python

            import rmm
            import warp as wp

            rmm.reinitialize(pool_allocator=True, initial_pool_size=2**30)
            wp.set_cuda_allocator(wp.utils.AllocatorRmm())
            # All subsequent wp.array allocations go through the RMM pool
    """

    def __init__(self):
        try:
            import rmm  # noqa: PLC0415, F401
        except ImportError as e:
            raise ImportError(
                "Failed to import 'rmm'. Ensure it is installed and compatible with your CUDA version. "
                "See https://docs.rapids.ai/install/ for installation instructions."
            ) from e
        self._buffers: dict[int, object] = {}

    @staticmethod
    def _get_rmm_stream():
        """Return an RMM ``Stream`` wrapping the current Warp device's CUDA stream.

        Resolves the device from the active CUDA context (set by
        ``device.context_guard`` in ``warp.array``'s allocation path) rather than
        the default device, which may differ in multi-GPU scenarios.

        Warp's :class:`~warp.Stream` implements the ``__cuda_stream__`` protocol
        directly, so RMM accepts it as a stream object.
        """
        from rmm.pylibrmm.stream import Stream as RmmStream  # noqa: PLC0415

        from warp._src.context import runtime  # noqa: PLC0415

        return RmmStream(obj=runtime.get_current_cuda_device().stream)

    def allocate(self, size_in_bytes: int) -> int:
        """Allocate device memory via RMM and return a device pointer."""
        if size_in_bytes == 0:
            return 0
        import rmm  # noqa: PLC0415

        buf = rmm.DeviceBuffer(size=size_in_bytes, stream=self._get_rmm_stream())
        ptr = buf.ptr
        self._buffers[ptr] = buf
        return ptr

    def deallocate(self, ptr: int, size_in_bytes: int) -> None:
        """Free device memory by releasing the RMM DeviceBuffer."""
        if ptr == 0:
            return  # Zero-size allocation; nothing was allocated.
        try:
            del self._buffers[ptr]
        except KeyError:
            raise RuntimeError(
                f"AllocatorRmm.deallocate called with unrecognized pointer {ptr:#x} "
                f"(size={size_in_bytes}). This may indicate a double-free or a "
                f"pointer that was not allocated by this AllocatorRmm instance."
            ) from None

    def __repr__(self):
        return f"AllocatorRmm(active_buffers={len(self._buffers)})"
