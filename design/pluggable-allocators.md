# Pluggable Allocator Interface and RMM Integration

**Status**: Implemented

**Issue**: [GH-781](https://github.com/NVIDIA/warp/issues/781)

## Motivation

Users running mixed GPU workloads (PyTorch + cuML + Warp) benefit from a shared memory
pool across all frameworks. RAPIDS Memory Manager (RMM) is the standard tool for this:
it provides pool, arena, and other allocator strategies that reduce fragmentation and
allocation overhead.

Today, Warp hardcodes four allocator classes (`CpuDefaultAllocator`, `CpuPinnedAllocator`,
`CudaDefaultAllocator`, `CudaMempoolAllocator`) with no extension mechanism. Users cannot
redirect Warp's GPU allocations through an external allocator like RMM.

This feature adds a pluggable allocator interface that enables RMM integration and any
other custom allocator backend.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1  | Formal `Allocator` protocol with `allocate`/`deallocate` methods | Must | Formalizes the existing pattern |
| R2  | Per-device custom allocator support for CUDA devices | Must | `set_cuda_allocator()` for all, `set_device_allocator()` for one |
| R3  | Built-in RMM adapter (`RmmAllocator`) | Must | Lazy `rmm` import, no hard dependency |
| R4  | `ScopedAllocator` context manager | Should | Follows `ScopedMempool` pattern |
| R5  | `get_device_allocator()` query function | Should | |
| R6  | Custom allocator example (non-RMM) | Should | Teaches the protocol |
| R7  | Documentation in `allocators.rst` | Must | |

**Non-goals:**

- Internal native C++/CUDA allocations (mesh BVH, sparse matrix temporaries, sort scratch
  space). These are small and short-lived. Can be addressed in a follow-up.
- CPU and pinned-memory allocators. RMM manages device memory only.
- Adding RMM as a project dependency. It is Linux-only and CUDA-version-specific.

## Design

### Approach

**Allocator property on Device (Approach 3)** — add `device._custom_allocator` as a
settable attribute on `Device`. When set, `get_allocator()` returns it instead of the
built-in allocator. This is explicit, easy to understand, and keeps pinned memory
unaffected.

### Alternatives Considered

**Protocol + device monkey-patch (Approach 1):** Replace `device.current_allocator`
directly via `set_cuda_allocator()`. Works but is informal — the attribute being overwritten
is also used by `set_mempool_enabled()`, creating a confusing interaction where enabling
mempools would silently override the custom allocator.

**Allocator registry (Approach 2):** A central registry mapping `(device, allocator_type)`
to instances. Over-engineered — we only need to replace the CUDA device allocator, not
support named allocator types.

### Key Implementation Details

#### Allocator Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Allocator(Protocol):
    def allocate(self, size_in_bytes: int) -> int: ...
    def deallocate(self, ptr: int, size_in_bytes: int) -> None: ...
```

The four existing allocator classes are updated to use `allocate`/`deallocate` method names
instead of `alloc`/`free`. The `deleter` attribute on the allocator classes is removed;
callers use `allocator.deallocate` directly. (`array.deleter`, which stores the
deallocation callable on each `warp.array` instance, is unchanged.)

These classes are internal (`warp._src.context`), not public API. No backward-compatible
aliases are provided. Any internal code still referencing `alloc`/`free`/`deleter` is
updated in the same change.

The protocol intentionally omits a `stream` parameter. Warp's existing allocators do not
expose stream selection at the Python level: `CudaMempoolAllocator` forwards the device's
current stream internally, and `CudaDefaultAllocator` uses synchronous `cudaMalloc`.
Custom allocators that need stream awareness should obtain the active Warp stream inside
their `allocate`/`deallocate` implementations (see the `RmmAllocator` section below for
the recommended pattern). A `stream` parameter may be added to the protocol in a future
revision if use cases emerge that cannot resolve the stream internally.

#### Validation

A shared `_validate_allocator()` helper in `context.py` checks that an allocator conforms
to the `Allocator` protocol via `isinstance()` (enabled by `@runtime_checkable`), and
additionally verifies that `allocate` and `deallocate` are callable. The extra `callable()`
check is needed because `Protocol.__instancecheck__` only tests *attribute presence* —
an object with `allocate = None` would otherwise pass `isinstance()` and fail later inside
`array.__del__`, where exceptions are swallowed. This helper is called from
`set_cuda_allocator()`, `set_device_allocator()`, and `ScopedAllocator.__init__()`, avoiding
duplication of the validation logic and error message.

#### Device Integration

```python
# In Device.__init__, CUDA branch, after existing allocator setup:
self._custom_allocator = None

# Modified get_allocator():
def get_allocator(self, pinned: bool = False):
    if self.is_cuda:
        if self._custom_allocator is not None:
            return self._custom_allocator
        return self.current_allocator
    else:
        return self.pinned_allocator if pinned else self.default_allocator
```

The custom allocator takes priority over the built-in mempool/default allocator.
Setting `_custom_allocator` to `None` restores the built-in allocator.

**Interaction with `set_mempool_enabled()`:** When a custom allocator is active,
`set_mempool_enabled()` still updates `device.current_allocator` (the built-in allocator
selection) but has no visible effect because `get_allocator()` returns the custom allocator
first. When the custom allocator is later removed, the built-in allocator reflects whatever
mempool setting was last configured. This is the expected behavior — the two mechanisms are
independent layers. No warning is needed.

**Thread safety:** Setting `device._custom_allocator` is a single Python attribute
assignment (atomic under the GIL). This matches the thread-safety model of the existing
`device.current_allocator`. Warp's allocator infrastructure is not designed for concurrent
modification from multiple threads; users should set allocators before starting concurrent
work.

#### Public API

```python
def set_cuda_allocator(allocator: Allocator | None) -> None:
    """Set the memory allocator for all CUDA devices."""
    init()
    _validate_allocator(allocator)
    devices = get_cuda_devices()
    if not devices:
        raise RuntimeError("set_cuda_allocator: no CUDA devices available")
    for device in devices:
        device._custom_allocator = allocator

def set_device_allocator(device: DeviceLike, allocator: Allocator | None) -> None:
    """Set the memory allocator for a specific CUDA device."""
    device = get_device(device)
    if not device.is_cuda:
        raise RuntimeError("Custom allocators are only supported on CUDA devices")
    _validate_allocator(allocator)
    device._custom_allocator = allocator

def get_device_allocator(device: DeviceLike) -> Allocator:
    """Get the current effective memory allocator for a device."""
    device = get_device(device)
    return device.get_allocator()
```

`ScopedAllocator` in `warp/_src/utils.py` follows the `ScopedMempool` pattern — saves
`device._custom_allocator` on enter, restores on exit. Its `__enter__` returns `self`
to support `with ... as ctx:` usage.

#### Array Lifecycle

In `array._init_new`:

```python
allocator = device.get_allocator(pinned=pinned)
# Resolve the deallocate callable before allocating so a bad descriptor/__getattr__
# cannot leak a freshly-allocated pointer between allocate() and self.deleter assignment.
deleter = allocator.deallocate
if capacity > 0:
    if device.is_cuda:
        with device.context_guard:
            ptr = allocator.allocate(capacity)
    else:
        ptr = allocator.allocate(capacity)
else:
    ptr = None
# ...
self.deleter = deleter
self._allocator = allocator
```

**CUDA context correctness:** The existing built-in allocators pass the device context
explicitly to native C functions (`wp_alloc_device_default(self.device.context, ...)`), so
they work regardless of the current CUDA context. Custom allocators (including RMM) call
Python-level APIs that operate on the current CUDA context. Wrapping the `allocate()` call
in `device.context_guard` ensures the correct CUDA device is active. This guard is also
used in `fabricarray.__init__` for bucket allocation and in `array.__del__` for
deallocation.

`array.__del__` calls `self.deleter(self.ptr, self.capacity)` within `device.context_guard`.
A guard at the top of `__del__` skips deallocation for partially-initialized arrays (where
`self.device` may not exist) and zero-size arrays (where `self.ptr` is `None`).
`TypeError`/`AttributeError` exceptions are suppressed during interpreter shutdown when
callables become `None`, matching the pattern used throughout Warp's other `__del__`
implementations (`bvh`, `mesh`, `volume`, `texture`, etc.). Any other exception raised
by a custom `deallocate` propagates as Python's standard "Exception ignored in
`__del__`" message — visible enough to debug without risking shutdown-time spam from
partially-torn-down allocator dependencies. The array holds a reference to
`self._allocator`, keeping the allocator (and its state, e.g., `RmmAllocator._buffers`)
alive until all arrays allocated through it are garbage-collected.

#### Memory Tracker Integration

The native allocation tracker (added in [GH-1269](https://github.com/NVIDIA/warp/issues/1269))
only records allocations that flow through the `wp_alloc_*` / `wp_free_*` entry points.
Custom allocators bypass those, so allocations made via `set_cuda_allocator` /
`set_device_allocator` do not appear in tracker reports. A follow-up change will
introduce a richer allocator protocol that lets custom allocators participate in
tracking without leaking framework internals into the allocator surface.

#### Built-in RMM Adapter

New module `warp/_src/rmm_allocator.py`:

```python
class RmmAllocator:
    """Allocator that routes Warp device memory through RMM."""

    def __init__(self):
        try:
            import rmm
        except ImportError as e:
            raise ImportError(
                "Failed to import 'rmm'. Ensure it is installed and compatible with your CUDA version. "
                "See https://docs.rapids.ai/install/ for installation instructions."
            ) from e
        self._buffers: dict[int, object] = {}

    @staticmethod
    def _get_rmm_stream():
        """Return an RMM ``Stream`` wrapping the current Warp device's CUDA stream."""
        from rmm.pylibrmm.stream import Stream as RmmStream
        from warp._src.context import runtime
        return RmmStream(obj=runtime.get_current_cuda_device().stream)

    def allocate(self, size_in_bytes: int) -> int:
        if size_in_bytes == 0:
            return 0
        import rmm
        buf = rmm.DeviceBuffer(size=size_in_bytes, stream=self._get_rmm_stream())
        ptr = buf.ptr
        self._buffers[ptr] = buf
        return ptr

    def deallocate(self, ptr: int, size_in_bytes: int) -> None:
        if ptr == 0:
            return  # Zero-size allocation; nothing was allocated.
        try:
            del self._buffers[ptr]
        except KeyError:
            raise RuntimeError(
                f"RmmAllocator.deallocate called with unrecognized pointer {ptr:#x} ..."
            ) from None
```

##### Stream handling

RMM's `DeviceMemoryResource.allocate()` and `DeviceBuffer` both accept a `stream`
parameter that controls which CUDA stream the allocation is ordered on. RMM's
`DEFAULT_STREAM` is stream 0 (the legacy default stream), **not** "the current stream."
Passing stream 0 has two consequences:

1. Allocations implicitly synchronize with all other non-blocking streams, serializing
   work that should overlap.
2. Stream-ordered memory resources (e.g., `CudaAsyncMemoryResource`) cannot reuse freed
   memory efficiently because all allocations land on the same stream.

`RmmAllocator` avoids this by resolving the stream at call time: `_get_rmm_stream()`
reads the current Warp device's stream and wraps it in an RMM `Stream` through the
`__cuda_stream__` protocol. This matches the pattern used by CuPy's RMM integration
(`rmm.allocators.cupy`), which calls `cupy.cuda.get_current_stream()` before every
allocation.

Warp's :class:`~warp.Stream` implements the
[`__cuda_stream__` protocol](https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol)
directly, returning `(0, cuda_stream_ptr)`. RMM's `Stream` constructor consumes this
protocol, so no additional wrapper is required.

Because `allocate()` is always called within `device.context_guard`, the Warp device and
its stream are guaranteed to be current when `_get_rmm_stream()` runs.

##### Deallocation stream

When `del self._buffers[ptr]` releases a `DeviceBuffer`, the buffer's destructor
deallocates on the stream that was passed at construction time (the Warp stream that was
active during `allocate()`). For stream-ordered memory resources this means the free is
ordered on the allocation stream, not the deallocation-time stream. This is correct when
Warp uses a single stream per device (the common case). If the user switches streams
between allocation and deallocation, CUDA's stream-ordered semantics still guarantee
correctness as long as proper inter-stream synchronization has occurred (which Warp
handles via `stream.wait_stream()`). Full deallocation-stream control would require
calling `DeviceMemoryResource.deallocate(ptr, size, stream)` directly instead of relying
on `DeviceBuffer.__dealloc__`, at the cost of losing the per-buffer MR reference that
protects against MR changes between alloc and dealloc.

##### Other design decisions

- **GC prevention** uses a pointer-keyed dict (same pattern as Numba's RMM integration).
  Each `DeviceBuffer` is stored by its pointer address; deleting the entry releases the
  buffer back to RMM.
- **Zero-size handling:** `allocate(0)` returns `0` without touching RMM. `deallocate`
  with `ptr == 0` returns immediately. In practice, `array.__del__` guards against
  `ptr is None` (set for zero-capacity arrays in `_init_new`) before calling `deallocate`,
  so the `ptr == 0` check is a defensive guard for direct callers of `deallocate`.
- **Double-free detection:** `deallocate` raises `RuntimeError` if the pointer is not in
  `_buffers`, catching double-free bugs and mismatched allocator usage.
- **Resource resolution timing:** Each `DeviceBuffer` captures a reference to the
  `DeviceMemoryResource` that was active at allocation time. If the user changes the RMM
  resource between allocation and deallocation (e.g., swapping pool strategies),
  deallocation still routes through the original resource, avoiding mismatched-MR bugs.

`RmmAllocator` is re-exported at import time from `warp/__init__.py`; the `rmm`
package itself is imported lazily inside `RmmAllocator.__init__` (and on each
`allocate()`), so users without RMM installed are unaffected until they
instantiate `RmmAllocator`.

A single `RmmAllocator` instance can safely be shared across multiple CUDA devices via
`set_cuda_allocator()` because `allocate()` is always called within `device.context_guard`
(see Array Lifecycle above), ensuring `rmm.DeviceBuffer` allocates on the correct device.
The `_buffers` dict may contain pointers from different devices; this is safe because
CUDA device pointers are unique across devices and deallocation also happens under the
correct context guard.

#### Graph Capture Caveat

Custom allocators that do not use stream-ordered allocation may produce silently corrupted
graphs during CUDA graph capture — the capture succeeds but replay uses stale pointers.
Warp does not emit a warning because it cannot reliably determine whether a custom allocator
supports graph capture. Users are responsible for ensuring their allocator is compatible.

With the stream-aware `RmmAllocator` and a stream-ordered `DeviceMemoryResource` (e.g.,
`rmm.mr.CudaAsyncMemoryResource`), allocations during graph capture land on the correct
stream and are recorded into the graph. The `allocators.rst` documentation includes a
warning about this limitation for custom allocators that do not handle streams.

#### FEM and Fabric Integration

`warp/_src/fem/cache.py` (`TemporaryStore.Pool`): The pool captures the allocator's
`deallocate` callable per-allocation in its `_allocs` dict, stored as
`(capacity, deallocate)` tuples. This ensures each buffer is freed using the allocator
that created it, even if the device's allocator changes between allocations.

`warp/_src/fabric.py` (`fabricarray`): Bucket allocation uses `device.get_allocator()`
and wraps the `allocate()` call in `device.context_guard`, consistent with `array.__init__`.
