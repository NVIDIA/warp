# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

import warp as wp

from ._capture_detect import _stream_capture_id, _stream_is_capturing, _warp_already_tracks_capture

__all__ = [
    "context",
    "is_available",
    "task",
    "task_graph",
    "warmup",
]


class _ModuleCache:
    resolved: bool = False
    stf: Any | None = None
    error: BaseException | None = None


def _resolve_modules() -> tuple[Any | None, BaseException | None]:
    """Lazy-import ``cuda.stf._experimental`` and cache the result process-wide."""
    if _ModuleCache.resolved:
        return _ModuleCache.stf, _ModuleCache.error

    _ModuleCache.resolved = True

    try:
        import cuda.stf._experimental as stf  # noqa: PLC0415

        # The package exposes its symbols lazily, so a bare import succeeds
        # even when the native bindings cannot load (e.g. the extension for
        # the detected CUDA version is missing). Touch one bound symbol so
        # availability reflects a usable installation.
        stf.context  # noqa: B018
    except Exception as exc:
        # Cache any failure (ImportError, OSError from a missing native
        # library, RuntimeError during module init, ...) so the second call
        # returns the same diagnostic instead of pretending the module is
        # simply absent.
        _ModuleCache.error = exc
        return None, exc

    _ModuleCache.stf = stf
    return stf, None


def is_available() -> bool:
    """Return ``True`` if ``cuda.stf._experimental`` can be imported."""
    stf, _ = _resolve_modules()
    return stf is not None


def _import_stf() -> Any:
    """Import ``cuda.stf._experimental`` (cached) and return the module, or raise.

    Internal helper used by every public entry point that actually needs
    ``cuda.stf`` to be installed -- as opposed to :func:`is_available`, which
    only reports availability without raising. Callers that want raw
    ``cuda.stf`` access from outside this package should
    ``import cuda.stf._experimental`` themselves after checking
    :func:`is_available`.

    Raises:
        RuntimeError: If ``cuda.stf._experimental`` cannot be imported, with a
            message pointing at the ``cuda-stf`` install instructions.
    """
    stf, err = _resolve_modules()
    if stf is None:
        raise RuntimeError(
            "cuda.stf._experimental is not importable; install cuda-stf[cu12] or "
            "cuda-stf[cu13] for the CUDA toolkit used by this environment."
        ) from err
    return stf


class _WarpContext:
    """Thin wrapper around a CUDASTF context with Warp conveniences."""

    __slots__ = ("_array_tokens", "_finalized", "_stf_ctx")

    def __init__(self, stf_ctx: Any):
        self._stf_ctx = stf_ctx
        self._finalized = False
        self._array_tokens = {}

    @property
    def raw(self) -> Any:
        """Return the underlying ``cuda.stf`` context."""
        return self._stf_ctx

    def __enter__(self) -> _WarpContext:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finalize()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stf_ctx, name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._stf_ctx!r})"

    def finalize(self) -> None:
        """Finalize the underlying CUDASTF context once."""
        if self._finalized:
            return
        try:
            self._stf_ctx.finalize()
        finally:
            self._release_array_token_cache()
            self._finalized = True

    def _release_array_token_cache(self) -> None:
        self._array_tokens.clear()

    def dep(self, obj: Any) -> Any:
        """Return a CUDASTF dependency handle for ``obj``.

        Existing CUDASTF logical data objects, including tokens, are returned
        unchanged. CUDA :class:`warp.array` objects are tracked as ordering-only
        tokens, cached once per exact array view.
        """
        _import_stf()
        if _is_stf_logical_data(obj):
            return obj

        if isinstance(obj, wp.array):
            if not obj.device.is_cuda:
                raise TypeError("ctx.dep requires a CUDA wp.array")

            key = _array_token_key(obj)
            entry = self._array_tokens.get(key)
            if entry is None:
                entry = self._stf_ctx.token()
                self._array_tokens[key] = entry

            return entry

        raise TypeError(f"ctx.dep accepts wp.array or cuda.stf.logical_data, got {type(obj).__name__}")

    def task(
        self,
        *deps: Any,
        capture: bool | None = None,
        dtypes: Sequence[Any] | None = None,
        exec_place: Any | wp.Device | None = None,
        symbol: str | None = None,
    ):
        # Delegate through the free function so the method and cross-framework
        # free-function forms share the same stream/array adaptation path
        # without recursing back into this method.
        return task(self, *deps, capture=capture, dtypes=dtypes, exec_place=exec_place, symbol=symbol)


class _TaskGraph:
    """Warp adapter around ``cuda.stf.task_graph``."""

    __slots__ = ("_raw_task_graph", "context")

    def __init__(self, **kwargs):
        if "device" in kwargs:
            raise TypeError("wp_stf.task_graph() does not accept device; use wp.ScopedDevice(device).")
        if kwargs:
            raise TypeError("wp_stf.task_graph() does not accept keyword arguments yet")

        stf = _import_stf()
        self._raw_task_graph = stf.task_graph()
        self.context = _WarpContext(self._raw_task_graph.context)

    def __enter__(self) -> None:
        return self._raw_task_graph.__enter__()

    def __exit__(self, exc_type, exc, tb) -> bool:
        return self._raw_task_graph.__exit__(exc_type, exc, tb)

    @property
    def raw(self) -> Any:
        """Return the underlying CUDASTF launchable graph."""
        return self._raw_task_graph.raw

    def launch(self, stream: wp.Stream | None = None) -> Any:
        """Launch the recorded task graph.

        The graph executes on the CUDASTF launch stream, which Warp does not
        track. Pass ``stream`` (typically ``wp.get_stream()``) to make that
        Warp stream wait on the launch with an event instead of requiring a
        device synchronization before Warp-side work may consume the results.
        This keeps frame replay fully asynchronous, matching the pipelining
        behavior of :func:`warp.capture_launch` on a native graph.
        """
        result = self._raw_task_graph.launch()
        if stream is not None:
            stf_stream = _wrap_stream(int(self._raw_task_graph.stream), stream.device)
            stream.wait_stream(stf_stream)
        return result

    def reset(self) -> None:
        """Reset the raw graph and prevent further launches through this wrapper."""
        return self._raw_task_graph.reset()

    def finalize(self) -> None:
        """Reset any live raw graph and finalize the owned CUDASTF context."""
        return self._raw_task_graph.finalize()


def context(stream: int | wp.Stream | None = None, **kwargs) -> _WarpContext:
    """Open a Warp-friendly CUDASTF context.

    Calling without ``stream`` opens a graph-recording context for a task DAG
    that can be pushed, popped, and replayed. Passing ``stream`` opens a
    stream-bound context for local eager work, including work inside an
    existing CUDA graph capture.

    ``device`` is intentionally not accepted here: CUDASTF selects the context
    kind from the presence of a stream, not a device. Use
    :class:`warp.ScopedDevice` to select the active Warp device used by
    :func:`task` / ``ctx.task(...)`` for stream wrapping and array aliasing, or
    spell a device's current stream explicitly with ``stream=wp.get_stream(device)``.

    Args:
        stream: Optional raw ``cudaStream_t`` or :class:`warp.Stream` for a
            stream-bound CUDASTF context.
        **kwargs: Forwarded to the underlying CUDASTF context constructor.

    Returns:
        A wrapper around the opened CUDASTF context.

    Raises:
        RuntimeError: If ``cuda.stf`` cannot be imported.
        TypeError: If ``device`` is passed.
    """
    if "device" in kwargs:
        raise TypeError(
            "wp_stf.context() does not accept device; use wp.ScopedDevice(device) or pass stream=wp.get_stream(device)."
        )

    stf = _import_stf()
    if stream is None:
        return _WarpContext(stf.stackable_context(**kwargs))

    return _WarpContext(stf.context(stream=_to_raw_stream_ptr(stream, None), **kwargs))


def task_graph(**kwargs) -> _TaskGraph:
    """Create a single-record, many-launch CUDASTF task graph wrapper.

    The returned graph owns a stackable CUDASTF context exposed as
    ``graph.context``. Declare tokens and logical data on that context, enter
    ``with graph:`` exactly once to record tasks, then call ``graph.launch()``
    any number of times before ``graph.finalize()``.
    """
    return _TaskGraph(**kwargs)


def _to_raw_stream_ptr(stream: int | wp.Stream | None, device: wp.Device | None) -> int:
    if stream is None:
        if device is None:
            device = wp.get_device()
        stream = wp.get_stream(device)
    if isinstance(stream, wp.Stream):
        return int(stream.cuda_stream)
    return int(stream)


_wp_stream_cache: dict[tuple[int, int], wp.Stream] = {}


def _wrap_stream(raw_ptr: int, device: wp.Device | None = None) -> wp.Stream:
    """Return a cached :class:`warp.Stream` wrapping ``raw_ptr`` on ``device``.

    Internal helper: re-registering the same raw ``cudaStream_t`` with Warp
    can corrupt stream bookkeeping, so we keep one process-lifetime wrapper
    per ``(device, raw_ptr)`` pair. Not part of the public surface; users
    receive the wrapped :class:`warp.Stream` from ``with ctx.task(...) as
    (stream, ...)``.
    """
    if device is None:
        device = wp.get_device()

    key = (int(device.ordinal), int(raw_ptr))
    stream = _wp_stream_cache.get(key)
    if stream is None:
        stream = wp.Stream(device, cuda_stream=int(raw_ptr))
        _wp_stream_cache[key] = stream
    return stream


def _np_to_wp_dtype(np_dtype: Any) -> Any | None:
    import numpy as np  # noqa: PLC0415

    return wp._src.types.np_dtype_to_warp_type.get(np.dtype(np_dtype))


def _array_nbytes(array: wp.array) -> int:
    return int(array.size) * wp._src.types.type_size_in_bytes(array.dtype)


def _array_token_key(array: wp.array) -> tuple[int, int, int, Any, tuple[int, ...], tuple[int, ...]]:
    return (
        int(array.device.ordinal),
        int(array.ptr),
        _array_nbytes(array),
        array.dtype,
        tuple(array.shape),
        tuple(array.strides),
    )


def _is_stf_logical_data(obj: Any) -> bool:
    return (
        type(obj).__name__ == "logical_data" and hasattr(obj, "read") and hasattr(obj, "write") and hasattr(obj, "rw")
    )


def _as_array(cai: Any, dtype: Any | None = None, *, shape: Sequence[int] | None = None, device=None) -> wp.array:
    """Alias a CUDASTF CUDA array interface object as a zero-copy :class:`warp.array`.

    This is an internal helper used to convert a CUDASTF array interface object to a Warp array.
    ``cai`` is typically returned by ``task.args_cai()``. The returned array is
    valid only while the surrounding STF task body is active.

    Args:
        cai: CUDASTF array view exposing ``ptr``, ``shape``, and ``dtype``.
        dtype: Optional Warp dtype. If omitted, the dtype is inferred from
            ``cai.dtype``.
        shape: Optional shape override.
        device: Optional Warp device.

    Returns:
        A Warp array aliasing ``cai`` without taking ownership of the memory.
    """
    if device is None:
        device = wp.get_device()

    if dtype is None:
        dtype = _np_to_wp_dtype(cai.dtype)
        if dtype is None:
            raise TypeError(f"Cannot infer Warp dtype from NumPy dtype {cai.dtype!r}; pass dtype explicitly.")

    cai_shape = tuple(cai.shape) if shape is None else tuple(shape)
    return wp.array(ptr=int(cai.ptr), dtype=dtype, shape=cai_shape, device=device)


def _unwrap_context(ctx: Any) -> Any:
    # The public free function accepts both raw cuda.stf contexts and our
    # wrapper. Keep the unwrap explicit so _WarpContext.task() can delegate here
    # without recursively calling itself.
    return ctx.raw if isinstance(ctx, _WarpContext) else ctx


def _coerce_exec_place(exec_place: Any | wp.Device | None, stf: Any) -> tuple[Any | None, wp.Device | None]:
    """Return ``(stf_exec_place, device_hint)`` for a public ``exec_place`` value."""
    if exec_place is None:
        return None, None

    if isinstance(exec_place, stf.exec_place_grid):
        raise TypeError(
            "exec_place_grid is not supported by wp_stf.task(); use ctx.raw.task(...) for the full STF grid API."
        )

    if isinstance(exec_place, stf.exec_place):
        return exec_place, None

    if isinstance(exec_place, str):
        # Tolerate Warp device aliases ("cuda:0"); frameworks often carry
        # devices as strings and a raw string here would otherwise TypeError.
        exec_place = wp.get_device(exec_place)

    if isinstance(exec_place, wp.Device):
        if not exec_place.is_cuda:
            raise TypeError("exec_place=wp.Device requires a CUDA device; use ctx.raw.host_launch(...) for host work.")
        return stf.exec_place.device(exec_place.ordinal), exec_place

    raise TypeError(f"exec_place must be a cuda.stf.exec_place or wp.Device, got {type(exec_place).__name__}")


def _device_from_exec_place(exec_place: Any | None, device_hint: wp.Device | None) -> wp.Device:
    if device_hint is not None:
        return device_hint

    if exec_place is None:
        return wp.get_device()

    try:
        device_ordinal = int(exec_place.affine_data_place.device_id)
    except Exception as exc:
        raise TypeError(
            "exec_place does not resolve to a single CUDA device; use ctx.raw.task(...) for the full STF API."
        ) from exc

    if device_ordinal < 0:
        raise TypeError("exec_place resolves to host; use ctx.raw.host_launch(...) for host work.")

    return wp.get_device(f"cuda:{device_ordinal}")


@contextlib.contextmanager
def task(
    ctx: Any,
    *deps: Any,
    capture: bool | None = None,
    dtypes: Sequence[Any] | None = None,
    exec_place: Any | wp.Device | None = None,
    symbol: str | None = None,
):
    """Open a CUDASTF task with Warp stream and array conveniences.

    The yielded tuple is ``(stream, *arrays)`` where ``stream`` is a cached
    :class:`warp.Stream` wrapping the STF task stream, and ``arrays`` are
    zero-copy :class:`warp.array` views for non-token dependencies.

    If ``capture`` is ``None``, the task stream is inspected. Capturing streams
    that Warp is not already tracking are wrapped in
    :func:`warp.capture_begin` / :func:`warp.capture_end` with
    ``external=True`` so Warp allocator bookkeeping is emitted into the
    surrounding CUDA graph.

    The task's completion is defined by the state of the yielded ``stream``
    when the body exits. Work issued by the body on auxiliary streams must be
    joined back to that stream before leaving the task.

    The task inherits the active Warp device from :func:`warp.get_device`.
    Use :class:`warp.ScopedDevice` around code that should build tasks for a
    non-default device, or pass ``exec_place`` to ask CUDASTF to run this task
    on a specific execution place.
    """
    stf = _import_stf()
    stf_ctx = _unwrap_context(ctx)

    stf_exec_place, device_hint = _coerce_exec_place(exec_place, stf)
    device = _device_from_exec_place(stf_exec_place, device_hint)
    task_args = (*deps, stf_exec_place) if stf_exec_place is not None else deps
    task_kwargs = {"symbol": symbol} if symbol is not None else {}

    stf_task = stf_ctx.task(*task_args, **task_kwargs)
    stf_task.start()

    raw_ptr = int(stf_task.stream_ptr())
    stream = _wrap_stream(raw_ptr, device)

    scoped = wp.ScopedStream(stream, sync_enter=False)
    scoped.__enter__()

    captured = False
    try:
        if capture is None:
            if _stream_is_capturing(raw_ptr):
                capture_id = _stream_capture_id(raw_ptr)
                capture = not _warp_already_tracks_capture(capture_id)
            else:
                capture = False

        if capture:
            wp.capture_begin(stream=stream, external=True)
            captured = True

        cais = stf_task.args_cai()
        if cais is None:
            cais_tuple = ()
        elif isinstance(cais, tuple):
            cais_tuple = cais
        else:
            cais_tuple = (cais,)

        if dtypes is not None and len(dtypes) != len(cais_tuple):
            raise ValueError(f"dtypes has {len(dtypes)} entries but the STF task exposes {len(cais_tuple)} array deps")

        arrays = [
            _as_array(cai, dtype=None if dtypes is None else dtypes[i], device=device)
            for i, cai in enumerate(cais_tuple)
        ]

        yield (stream, *arrays)
    finally:
        try:
            if captured:
                wp.capture_end(stream=stream, skip_leaf_join=True)
        finally:
            try:
                scoped.__exit__(None, None, None)
            finally:
                stf_task.end()


_warmed_up: set[int] = set()


def warmup(stream: int | wp.Stream | None = None, *, device: wp.Device | str | None = None) -> None:
    """Force CUDASTF's one-time CUDA initialization to run outside capture.

    CUDASTF may lazily initialize CUDA when the first ``stf.context`` opens
    (typically a ``cudaFree(0)``). Whether the CUDA Runtime tolerates that
    first touch depends on the surrounding capture mode:

    * Captures opened with :attr:`warp.CaptureMode.RELAXED` (or with
      ``external=True`` after a CUDASTF context has already initialized CUDA)
      tolerate the lazy init, so calling this function is optional.
    * Captures opened with the default :attr:`warp.CaptureMode.THREAD_LOCAL`
      reject capture-unsafe runtime calls, including the lazy init. Calling
      this function once at startup forces the init to run eagerly so the
      first STF context inside the capture is a no-op for CUDA init.

    The call is cheap (open and finalize an empty STF context) and idempotent
    per device; subsequent calls for an already-warmed device are no-ops, but
    the first call for each device performs the warmup so multi-GPU users get
    the lazy init forced on every device they touch.

    The typical use case is to call this function before creating a local CUDASTF
    context that will be used inside a capture with a non-relaxed capture mode.

    Args:
        stream: Optional raw ``cudaStream_t`` or :class:`warp.Stream` for a
            stream-bound CUDASTF context.
        device: Optional Warp device.

    Returns:
        None
    """
    if not is_available():
        return

    if device is None:
        if isinstance(stream, wp.Stream):
            device = stream.device
        else:
            device = wp.get_device()
    elif not isinstance(device, wp.Device):
        # Tolerate Warp device aliases ("cuda:0"): a raw string would fail
        # below with an opaque AttributeError, and callers inside try/except
        # fallbacks then silently lose the warmup.
        device = wp.get_device(device)

    if device.ordinal in _warmed_up:
        return

    stf = _import_stf()
    raw_ptr = _to_raw_stream_ptr(stream, device)
    ctx = stf.context(stream=raw_ptr)
    ctx.finalize()
    _warmed_up.add(device.ordinal)
