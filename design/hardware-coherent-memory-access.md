# Hardware-Coherent Cross-Device Memory Access

**Status**: In Progress

**Tracking issues**:

- Phase 1: [GH-1461](https://github.com/NVIDIA/warp/issues/1461)
- Future phases: Track with follow-up GitHub issues as they are scheduled.

**Implementation status**: Phases 1 and 5 are implemented. Phases 2--4, 6,
and 7 remain future work.

## Motivation

Before Phase 1, Warp enforced a strict rule: every array argument passed to `wp.launch()` had to reside on the same device as the kernel launch target. If a user created an array on the CPU and attempted to launch a GPU kernel that read it, Warp raised a `RuntimeError`. This enforcement existed in `warp/_src/context.py::pack_arg`:

```python
# check device
if value.device != device:
    raise RuntimeError(
        f"Error launching kernel '{kernel.key}', trying to launch on "
        f"device='{device}', but input array for argument '{arg_name}' "
        f"is on device={value.device}."
    )
```

This restriction is correct on discrete-GPU systems (e.g., a workstation with a PCIe-connected NVIDIA GPU) where the GPU genuinely cannot dereference a pointer to unpinned CPU memory. However, a growing class of NVIDIA hardware uses **unified memory architectures** where the GPU _can_ directly access CPU memory. Some systems also let the CPU directly access GPU-resident CUDA managed memory, but that is a separate CUDA-reported capability and must not be inferred from ATS alone:

- **Grace C2C systems (GH200, GB200, DGX Spark)** -- Grace ARM CPU + Hopper or Blackwell GPU connected via NVLink Chip-to-Chip (C2C). These systems can report host-page-table ATS, allowing the GPU to access ordinary system memory. CPU direct access to GPU-resident CUDA managed memory depends on `cudaDevAttrDirectManagedMemAccessFromHost`; do not assume it from the product family name.
- **Jetson Orin and other limited Tegra systems** -- Integrated GPUs sharing the same DRAM as the CPU, but with a limited unified memory model where ordinary system allocations are not necessarily GPU-accessible.
- **Jetson Thor** -- Tegra Blackwell SoC with CUDA-reported ATS. On a Thor development kit tested with CUDA 13.0, the GPU can directly access ordinary system allocations (`malloc`, anonymous `mmap`, and file-backed `mmap`) and the CUDA hardware reports host-native atomic support, but CPU direct access to `cudaMalloc` memory is still not supported. Current Warp CPU atomics do not provide a CPU/GPU interprocessor atomic contract.
- **HMM-capable discrete systems** -- Linux kernel 6.1.24+ with Heterogeneous Memory Management (HMM) enabled allows software-coherent access to all system memory from PCIe GPUs, without requiring explicit CUDA allocation APIs.

On all systems where the CUDA device reports `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`, the strict `value.device != device` check is overly conservative and forces users into unnecessary `wp.copy()` or `.to(device)` calls that are both a performance penalty and an ergonomic burden. On HMM and ATS systems in particular, a plain `malloc`'d pointer is directly accessible from the GPU -- there is no need to copy data at all.

### User impact

A user on DGX Spark writing:

```python
data = wp.array([1.0, 2.0, 3.0], device="cpu")
wp.launch(my_kernel, dim=3, inputs=[data], device="cuda:0")
```

previously got a `RuntimeError` even though the hardware can handle this directly. The user had to write:

```python
data = wp.array([1.0, 2.0, 3.0], device="cpu")
data_gpu = data.to("cuda:0")  # unnecessary copy on ATS systems
wp.launch(my_kernel, dim=3, inputs=[data_gpu], device="cuda:0")
```

This is not just an inconvenience -- it defeats one of the primary benefits of unified-memory hardware, which is eliminating explicit data movement.

## Background: CUDA Unified Memory Paradigms

CUDA exposes unified memory capabilities through device attributes. The sections below group the relevant attribute combinations into four capability buckets so the implementation can reason about access rules mechanically, not by assuming behavior from a product family name. Three buckets have concrete platform examples in this document; the managed-only bucket is retained as a conservative fallback for the attribute combination where managed memory is fully shared but ordinary pageable system memory is not.

### Paradigm 1: Limited Unified Memory (Limited Tegra, Windows, WSL)

**Detection:** `cudaDevAttrConcurrentManagedAccess == 0`

Applies to Windows systems including WSL and to Tegra/Jetson devices whose CUDA attributes report limited managed access. Do not infer this from the Jetson family name alone: Jetson Thor tested with CUDA 13.0 reports `concurrentManagedAccess == 1`, `pageableMemoryAccess == 1`, and `pageableMemoryAccessUsesHostPageTables == 1`, so it does not fall into this paradigm.

Characteristics:
- Only memory explicitly allocated via `cudaMallocManaged` (or, on CUDA 13.0+
  builds, `cudaMallocFromPoolAsync` from a managed memory pool, or
  `__managed__` globals) behaves as unified memory.
- Managed memory starts in CPU physical memory, is bulk-migrated to the GPU when a kernel begins executing, and is bulk-migrated back on synchronization.
- The CPU must not access managed memory while the GPU is active.
- Oversubscription of GPU memory is not allowed.
- System allocations (`malloc`, `mmap`) are NOT GPU-accessible.

On limited/non-I/O-coherent Tegra specifically:
- `cudaHostRegister()` is not supported on non-I/O-coherent Tegra devices.
- `cudaMallocHost` produces uncached memory from the GPU's perspective on non-I/O-coherent Tegra.
- All memory physically resides in the same shared DRAM, but visibility is controlled by the CUDA driver.

### Paradigm 2: Full Unified Memory for CUDA-Managed Allocations Only

**Detection:** `cudaDevAttrConcurrentManagedAccess == 1` AND `cudaDevAttrPageableMemoryAccess == 0`

Characteristics:
- Memory allocated via `cudaMallocManaged` has full unified memory support (page-granularity migration, concurrent CPU/GPU access, oversubscription).
- System allocations (`malloc`, `mmap`) are still NOT GPU-accessible.
- This is an attribute-defined bucket. This document does not currently identify a specific tested platform for it, but the implementation should handle it separately because its access rules differ from both limited unified memory and HMM/ATS.

### Paradigm 3: Full Unified Memory with Software Coherency (HMM)

**Detection:** `cudaDevAttrPageableMemoryAccess == 1` AND `cudaDevAttrPageableMemoryAccessUsesHostPageTables == 0` AND `cudaDevAttrConcurrentManagedAccess == 1`

Available on Linux with kernel 6.1.24+ / 6.2.11+ / 6.3+ with HMM enabled. Can be verified via `nvidia-smi -q | grep Addressing` showing `HMM`.

Characteristics:
- ALL system-allocated memory (`malloc`, `mmap`, file-backed mappings) automatically behaves as unified memory. No CUDA allocation APIs are required.
- Migration happens via page faults at page granularity (software coherence).
- Oversubscription is allowed.
- `cudaMallocManaged` still works but is unnecessary for basic access -- `malloc` suffices.
- GPU `cudaMalloc` allocations are NOT CPU-accessible.

### Paradigm 4: Full System-Memory Access with Host Page Tables (ATS)

**Detection:** `cudaDevAttrPageableMemoryAccessUsesHostPageTables == 1` AND `cudaDevAttrPageableMemoryAccess == 1` AND `cudaDevAttrConcurrentManagedAccess == 1`

Available on Grace Hopper, Grace Blackwell (including DGX Spark), Jetson Thor, and future systems where CUDA reports pageable memory access through host page tables. `nvidia-smi -q` reports these systems as `Addressing Mode: ATS`.

Characteristics:
- ALL system-allocated memory is GPU-accessible (same as HMM).
- GPU-resident CUDA managed memory is CPU-accessible without migration only when `cudaDevAttrDirectManagedMemAccessFromHost == 1`. This attribute is independent of ATS and must be queried directly. It is false on Jetson Thor as tested with CUDA 13.0, and false on a DGX Spark / GB10 system tested with CUDA Toolkit 13.0 and driver 580.95.05.
- `cudaDevAttrHostNativeAtomicSupported == 1` reports a hardware/link capability. This is a separate capability bit and does not imply CPU access to `cudaMalloc` allocations or that current Warp `wp.atomic_*` operations are safe for overlapping CPU/GPU updates.
- Host page tables are used for system-memory access. On systems with distinct CPU and GPU memory pools (Grace Hopper / Grace Blackwell), physical placement still matters for performance. On integrated SoCs such as Jetson Thor, the CPU and GPU share a single DRAM pool.
- ATS subsumes the system-memory access capabilities of HMM. When ATS is available, HMM is automatically disabled.

#### Observed Jetson Thor Behavior

The previous version of this document speculated that Jetson Thor would follow the limited Tegra model. Testing on a Jetson Thor development kit on 2026-05-11 showed otherwise:

- Platform: Linux `6.8.12-tegra`, CUDA Toolkit 13.0, Driver 13.0, GPU `NVIDIA Thor`, `sm_110`.
- `nvidia-smi -q` reports `Addressing Mode: ATS`.
- CUDA attributes: `integrated == 1`, `unifiedAddressing == 1`, `managedMemory == 1`, `concurrentManagedAccess == 1`, `pageableMemoryAccess == 1`, `pageableMemoryAccessUsesHostPageTables == 1`, `directManagedMemAccessFromHost == 0`, `hostNativeAtomicSupported == 1`, `canUseHostPointerForRegisteredMem == 1`.
- GPU kernels successfully read and wrote ordinary `malloc`, anonymous `mmap`, file-backed `mmap`, `cudaMallocHost`, `cudaHostRegister`, and `cudaMallocManaged` allocations.
- `cudaMemPrefetchAsync` succeeded for both managed memory and ordinary `malloc` memory.
- Direct CPU load/store of a `cudaMalloc` pointer faulted, matching `directManagedMemAccessFromHost == 0`.
- A standalone native stress test with real CPU atomic increments and GPU `atomicAdd()` produced the exact expected result for ordinary `malloc`, pinned host memory, and managed memory. This result does not apply to current Warp CPU `wp.atomic_*` lowering, which is not a hardware atomic.

The implementation must therefore treat "GPU can access system memory", "CPU can access GPU-resident CUDA managed memory", and "the hardware reports native CPU-GPU atomic capability" as three independent capabilities. The third is a diagnostic and future-work input, not a current Warp `wp.atomic_*` guarantee.

#### Observed DGX Spark / GB10 Behavior

Testing on a DGX Spark-class GB10 system on 2026-05-14 showed that ATS and C2C
do not imply CPU direct access to GPU-resident CUDA managed memory:

- Platform: CUDA Toolkit 13.0, Driver 580.95.05, GPU `NVIDIA GB10`, `sm_121`.
- `nvidia-smi -q` reports `Addressing Mode: ATS` and `GPU C2C Mode: Enabled`.
- CUDA attributes queried directly through the CUDA driver:
  `managedMemory == 1`, `concurrentManagedAccess == 1`,
  `pageableMemoryAccess == 1`, `pageableMemoryAccessUsesHostPageTables == 1`,
  `directManagedMemAccessFromHost == 0`, and
  `hostNativeAtomicSupported == 1`.
- Warp reports the corresponding Python properties as
  `is_cpu_memory_access_from_gpu_supported == True`,
  `is_gpu_memory_access_from_cpu_supported == False`, and
  `is_cpu_gpu_atomic_supported == True`.

This corrects the earlier assumption that DGX Spark / Grace Blackwell systems
should be classified as "bidirectional ATS" for managed-memory host access.
For Phase 1, the relevant launch feature remains GPU access to CPU arrays via
`pageableMemoryAccess`; CPU direct access to GPU-resident managed memory remains
attribute-gated and is not used to validate Warp default CUDA arrays.

Further testing on a DGX Spark-class GB10 system on 2026-06-05 showed that
`is_cpu_gpu_atomic_supported == True` must not be treated as a Warp atomic API
contract. Warp's CPU `wp.atomic_*` helpers currently lower to ordinary
read/modify/write operations, which assumes Warp's serial CPU kernel execution
model and is not safe when CPU and GPU work update the same address
concurrently. CUDA-side `wp.atomic_*` operations also use the normal CUDA atomic
implementation and are not documented here as system-scope host/device atomics.
Follow-on work is required before Warp can advertise CPU/GPU interprocessor
atomics. That work should include CPU hardware-atomic lowering for supported
scalar operations, GPU system-scope atomic semantics where needed, and
operation-level CUDA host atomic capability queries such as CUDA 13's
`cudaDeviceGetHostAtomicCapabilities()`.

### Summary of Access Rules by Paradigm

| Allocation type | Limited (Tegra/Win) | Full Managed Only | HMM (Software) | ATS system-memory only (Thor/GB10 observed) | ATS with direct managed host access |
|---|---|---|---|---|---|
| `malloc` / system | CPU only | CPU only | CPU + GPU | CPU + GPU | CPU + GPU |
| `mmap` / file-backed | CPU only | CPU only | CPU + GPU | CPU + GPU | CPU + GPU |
| `cudaMallocManaged` | Limited shared | Full shared | Full shared | Full shared; no direct host access to GPU-resident pages unless attribute reports it | Full shared with direct host access to GPU-resident pages |
| `cudaMallocHost` (pinned) | CPU + GPU (zero-copy) | CPU + GPU | CPU + GPU | CPU + GPU | CPU + GPU |
| `cudaHostRegister` | Device-dependent | CPU + GPU | CPU + GPU | CPU + GPU | CPU + GPU |
| `cudaMalloc` | GPU only | GPU only | GPU only | GPU only | GPU only for Warp default arrays |

### Performance Considerations on ATS Systems

Even when all system memory is GPU-accessible on ATS systems, physical placement can still matter for performance. On systems with distinct CPU and GPU memory pools, a GPU kernel repeatedly reading data physically resident in CPU LPDDR5X over NVLink C2C pays the C2C latency on every cache miss. CUDA provides mechanisms to control placement:

1. **Explicit prefetch** (`cudaMemPrefetchAsync`): Stream-ordered migration of a memory region to a specified device. Works on any allocation including system `malloc`. This is the primary tool for optimizing data placement.

2. **Access counter migration**: On ATS systems, the GPU hardware tracks access frequency to remote pages. When enabled via `cudaMemAdviseSetAccessedBy`, pages that the GPU accesses frequently are automatically migrated to GPU-local memory. Available for system-allocated memory starting with CUDA 12.4. Does not apply to file-backed `mmap` allocations.

3. **Placement hints** (`cudaMemAdvise`):
   - `cudaMemAdviseSetPreferredLocation(device)` -- encourages data to stay on the specified device.
   - `cudaMemAdviseSetReadMostly` -- allows read replication across devices.
   - `cudaMemAdviseSetAccessedBy(device)` -- enables access counter migration on ATS systems; establishes direct mappings on other systems.

On integrated ATS systems such as Jetson Thor, CPU and GPU memory share one DRAM pool, so prefetch may still succeed but may not provide a useful "closer" placement. Automatic prefetch should therefore remain disabled on integrated GPUs.

**Important performance caveat**: On host-page-table ATS systems with distinct CPU and GPU memory pools, CPU writes to GPU-resident memory may be expensive even if a future platform reports direct managed-memory host access. ARM (Grace) caches require all memory operations to pass through the cache hierarchy, so writing to GPU-resident memory can cause cache misses that pull data across C2C before writing. The recommended pattern is: write to CPU-resident memory, let the GPU read it remotely or prefetch it.

### Comparison: DGX Spark / GB10 vs. Jetson Thor

Both DGX Spark / GB10 and Jetson Thor use Blackwell-generation GPUs, but their memory architectures differ and the CUDA attributes must still be queried independently:

| Aspect | DGX Spark / GB10 (observed) | Jetson Thor (Tegra Blackwell) |
|---|---|---|
| CPU-GPU interconnect | NVLink C2C (high bandwidth, coherent) | On-chip SoC fabric |
| ATS available | Yes | Yes (`nvidia-smi` reports ATS) |
| Coherency model | Host-page-table ATS with distinct CPU/GPU memory pools | Host-page-table ATS for system memory on an integrated SoC |
| `malloc` GPU-accessible | Yes | Yes |
| CPU direct access to GPU-resident CUDA managed memory | No (`directManagedMemAccessFromHost == 0` on CUDA 13.0 / driver 580.95.05) | No (`directManagedMemAccessFromHost == 0` on CUDA 13.0) |
| Native CPU-GPU atomic hardware capability | Reports yes; current Warp CPU/GPU `wp.atomic_*` overlap unsupported | Reports yes; current Warp CPU/GPU `wp.atomic_*` overlap unsupported |
| Memory topology | Grace LPDDR5X + Blackwell HBM (NUMA) | Single shared DRAM pool |
| Unified memory paradigm | ATS system-memory access (Paradigm 4) | ATS system-memory access (Paradigm 4) |
| Best default allocator | System allocator (`malloc`) for shared CPU/GPU data | System allocator (`malloc`) for CPU-produced GPU-readable data; `cudaMalloc` for GPU-private data |

This means the implementation must query capabilities independently instead of assuming a single "ATS" behavior. DGX Spark / GB10 and Jetson Thor can launch GPU kernels directly over CPU arrays, but CPU kernels still cannot dereference Warp default CUDA arrays.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | `wp.launch()` must default to passing cross-device array arguments through to the hardware | Must | Exposed as `wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.RELAXED` |
| R2 | Provide launch verification modes (`warp.config.launch_array_access_mode`) for strict same-device checks and allocation-aware diagnostics | Must | Debuggability for users who hit CUDA illegal memory access errors; compatible with CUDA graph capture |
| R3 | Provide `wp.can_access(device, array)` for allocation-aware array access checks | Must | Resource-oriented public API; Phase 1 supports Warp arrays only |
| R4 | Provide `wp.prefetch()` API for explicit data migration hints | Should | Performance optimization for HMM / host-page-table ATS |
| R5 | Optional automatic prefetch in `wp.launch()` for cross-device arrays on coherent systems | Could | Convenience, but needs careful defaults |
| R6 | `wp.copy()` should skip staging buffers when direct access is available between devices | Could | Performance optimization, marked as TODO in current code |
| R7 | Provide explicit managed-memory arrays through a built-in allocator | Should | `wp.CudaManagedAllocator()` integrates with existing allocator APIs while keeping managed memory opt-in |
| R8 | Apply user allocator policy to persistent Warp resources, not just Python-created arrays | Should | Meshes, hash grids, volumes, and similar persistent buffers should honor `ScopedAllocator`; internal temporary allocations need a separate policy |
| R9 | Unify built-in CUDA allocator selection under `ScopedAllocator` | Could | Public built-in allocator accessors can make `ScopedAllocator` supersede `ScopedMempool` while preserving compatibility |
| R10 | Support graph-capturable managed allocation on CUDA 13+ builds when managed memory pools are available | Could | Later managed-pool backend for `wp.CudaManagedAllocator()`; CUDA 12.x remains direct `cudaMallocManaged()` outside capture only |

**Non-goals:**
- Changing the default allocator strategy. Managed memory remains opt-in through `wp.CudaManagedAllocator()`; standard CUDA arrays continue to use Warp's default CUDA or CUDA memory-pool allocators.
- Changing CUDA graph capture semantics for cross-device access checks. Phase 1 supports using `launch_array_access_mode` during graph capture, but does not add new cross-device synchronization, placement, or capture-time migration behavior beyond the same access checks used for ordinary launches. Phase 8 separately tracks a managed allocation backend that can be recorded in CUDA graphs.
- Automatically determining the optimal physical placement for every array. This is a performance tuning concern best left to the user via hints.
- Proactively detecting and warning about cross-device launches at `wp.launch()` time. The hardware enforces access rules; the verification mode is available for diagnosis when needed.
- Providing a top-level device-to-device access wrapper. `wp.can_access(device, resource)` is a resource-oriented API; `wp.can_access(device, device)` is not supported. Device-level/default-allocation checks remain available as `Device.can_access(other_device)`.
- Adding a custom/external allocation metadata protocol in the managed-memory phase. CUDA pointer attributes classify external pointers where possible, but unclassified pointers and unowned memory-pool pointers remain conservative until a later metadata phase.
- Making all internal temporary allocations obey the user allocator. Persistent resource storage and temporary internal buffers have different lifetime, performance, and graph-capture requirements; Phase 6 defines that boundary before broadening allocator policy.
- Removing `ScopedMempool` immediately. Phase 7 can make `ScopedAllocator` the preferred allocator selection API, but existing `ScopedMempool` users need a compatibility path.
- Providing CPU/GPU interprocessor atomics through `wp.atomic_*`. Current CPU-side Warp atomics are ordinary updates under the serial CPU execution model, and CUDA-side Warp atomics are not specified as system-scope host/device operations. A future API or mode may add this once the required CPU lowering, GPU scope, and operation-level capability checks are designed.

## Design

### CUDA Version Compatibility

Warp currently supports building with CUDA 12.0 through 13.2. The default toolkit distributed on PyPI is CUDA 12.9. Full support for this feature set on CUDA 12.0 through 12.7 is not a goal, but the feature must degrade cleanly (compile without errors, disable gracefully at runtime) on older toolkit versions.

**Device attribute queries (Phases 1, 2, 3, 5):** All device attributes used in this plan are `CUdevice_attribute` enum values present in `cuda.h` since CUDA 8.0 or 9.2 at the latest:

| Attribute | Enum value | Present since | Phase |
|---|---|---|---|
| `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` | 88 | CUDA 8.0 | 1 |
| `CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST` | 101 | CUDA 9.2 | 1 |
| `CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED` | 86 | CUDA 8.0 | 1 |
| `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES` | 100 | CUDA 9.2 | 2 |
| `CU_DEVICE_ATTRIBUTE_INTEGRATED` | 18 | CUDA 2.0 | 3 |
| `CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY` | 83 | CUDA 6.0 | 5 |
| `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS` | 89 | CUDA 8.0 | 5 |

All predate Warp's minimum of CUDA 12.0, so no `#if CUDA_VERSION` compile-time guards are needed for attribute queries. The attributes are queried via `cuDeviceGetAttribute`, which Warp already loads dynamically via `cuGetProcAddress` at version 2000. The driver returns 0 for any attribute the hardware does not support, which is the correct "feature not available" default.

**Driver API entry-point policy:** Warp loads CUDA Driver API functions dynamically through `cuGetProcAddress`. When a Driver API has multiple versioned entry points, request the oldest entry-point version whose signature and semantics satisfy Warp's use case. This keeps binaries built with newer toolkits compatible with older supported drivers instead of accidentally depending on a newer ABI variant selected by toolkit header macros.

Use a newer Driver API entry point only when Warp needs newer semantics, such as a new parameter type or behavior that the older entry point cannot express. Guard references to newer header-only types, enum values, or functions with `#if CUDA_VERSION` when they are absent from older supported toolkits, and separately gate runtime loading/calls on the driver version. This policy does not permit emulating feature enum values that older CUDA versions document as unsupported; for example, CUDA managed memory pools require CUDA 13+ headers and runtime support.

**`cuMemPrefetchAsync` (Phase 2):** This driver API has two versions:

| API version | Signature | Toolkit requirement | Driver requirement |
|---|---|---|---|
| v1 (version 8000) | `(CUdeviceptr, size_t, CUdevice, CUstream)` | CUDA 8.0+ | CUDA 8.0+ driver |
| v2 (version 12080) | `(CUdeviceptr, size_t, CUmemLocation, unsigned int, CUstream)` | CUDA 12.8+ | CUDA 12.8+ driver |

In CUDA 13.0 headers, `cuMemPrefetchAsync` is `#define`'d to `cuMemPrefetchAsync_v2`. Warp should avoid that macro-selected newer ABI and explicitly request the v1 entry point via `cuGetProcAddress` because v1 is sufficient for all planned use cases. The v2 API adds NUMA node targeting but is not required. See Phase 2 for the full dispatch implementation.

**Managed allocation APIs (Phase 5):** `wp.CudaManagedAllocator()` uses `cudaMallocManaged(..., cudaMemAttachGlobal)`. Users should allocate managed arrays before CUDA graph capture begins because local testing on driver 580.126.20 showed `cudaMallocManaged` can return `cudaErrorStreamCaptureUnsupported` and invalidate capture even when the active capture is on another stream or CUDA device. Warp does not preflight and reject the call itself; CUDA reports any capture-time allocation failure. Managed arrays allocated before capture can still be used by captured kernels.

**Managed memory-pool APIs (Phase 8):** CUDA 13.0 adds the managed memory-pool allocation type needed for graph-capturable managed allocation. On CUDA 13.0+ builds, devices with memory-pool support can use a private CUDA memory pool whose `cudaMemPoolProps.allocType` is `cudaMemAllocationTypeManaged`, with allocations made through `cudaMallocFromPoolAsync()` on the current Warp stream and freed through `cudaFreeAsync()`. This path is stream ordered and can be recorded in CUDA graphs.

CUDA 12.x builds, including the CUDA 12.9 PyPI build, must not compile or emulate the managed-pool path: `cudaMemAllocationTypeManaged` is not defined there, and CUDA 12.9 documents `cudaMemPoolProps::allocType` as pinned-only. `cudaMallocAsync()`, `cudaFreeAsync()`, CUDA memory pools, and `cudaMallocFromPoolAsync()` were introduced before Warp's CUDA 12.0 minimum, but the managed pool allocation type itself is CUDA 13-only. Runtime support still needs to be gated by the existing memory-pool support query and by successful creation of the managed pool.

Local testing on a Blackwell GPU with driver 580.126.20 showed `cudaMallocFromPoolAsync()` from a pool with `cudaMemAllocationTypeManaged` can be captured, instantiated, launched, and synchronized successfully. Pool creation should happen before capture begins. If an allocation during capture finds no initialized managed pool, Warp should reject the allocation clearly rather than attempting a capture-unsafe pool creation or falling back to `cudaMallocManaged()`.

**Summary by toolkit version:**

| Feature | CUDA 12.0 -- 12.7 | CUDA 12.8 -- 12.9 (PyPI default) | CUDA 13.0+ |
|---|---|---|---|
| Phase 1 (cross-device launch) | Full support | Full support | Full support |
| Phase 2 (prefetch) | v1 API | v1 API; v2 available but not required | v1 API; v2 available but not required |
| Phase 3 (auto-prefetch) | Full support (uses Phase 2 API) | Full support | Full support |
| Phase 4 (`wp.copy()` optimization) | Full support | Full support | Full support |
| Phase 5 (`wp.CudaManagedAllocator`) | Direct `cudaMallocManaged` outside capture; capture-time allocation unavailable | Direct `cudaMallocManaged` outside capture; capture-time allocation unavailable | Direct `cudaMallocManaged` outside capture; capture-time allocation unavailable |
| Phase 6 (persistent resource allocator policy) | Full support | Full support | Full support |
| Phase 7 (unified allocator selection) | Full support | Full support | Full support |
| Phase 8 (managed memory pools) | Not available; direct fallback outside capture only | Not available; direct fallback outside capture only | Managed pool when available; direct fallback outside capture otherwise |

No phase requires a minimum toolkit version beyond CUDA 12.0 to compile or expose its public API. The Phase 2 prefetch wrapper uses the v1 Driver API entry point for compatibility; v2 is only needed for future NUMA-node targeting. Phase 5 uses direct `cudaMallocManaged()` and does not provide a graph-capturable managed allocation backend. Phase 8 adds a CUDA 13-only native backend guarded at compile time and runtime.

### Overview: What Each Phase Introduces

Each phase introduces only the device attributes, native functions, and Python APIs it consumes or exposes as part of that phase. No phase adds speculative API surface solely for a future phase to use.

| Phase | Status | What it delivers | Attributes introduced | Native functions introduced |
|---|---|---|---|---|
| 1 | Implemented | Remove device check from `wp.launch()`, add verification mode, redesign `Device.can_access()`, add `wp.can_access(device, array)`, add allocation-aware launch verification for Warp-owned arrays | Native: `pageable_memory_access`, `direct_managed_mem_access_from_host`, `host_native_atomic_supported`; Python: `is_cpu_memory_access_from_gpu_supported`, `is_gpu_memory_access_from_cpu_supported`, `is_cpu_gpu_atomic_supported` | Three `wp_cuda_device_get_*` accessors |
| 2 | Future | `wp.prefetch()` for explicit data placement | `pageable_memory_access_uses_host_page_tables` (to distinguish HMM from host-page-table ATS for warning/no-op behavior) | `wp_cuda_mem_prefetch_async` |
| 3 | Future | Auto-prefetch in `wp.launch()` | `is_integrated` (to avoid pointless prefetches on shared-DRAM SoCs) | None |
| 4 | Future | `wp.copy()` staging-buffer optimization | None (reuses Phase 1 access predicates) | None |
| 5 | Implemented | `wp.CudaManagedAllocator()`, `wp.MemoryKind`, `array.memory_kind`, managed-memory-aware `wp.can_access()` and checked launches | `managed_memory`, `concurrent_managed_access` | `wp_alloc_device_managed`, `wp_cuda_pointer_get_memory_kind` |
| 6 | Future | Apply `ScopedAllocator` to persistent resource allocations while keeping temporary allocations under internal policy | None expected initially | Resource constructors may route persistent native buffers through allocator-aware helpers |
| 7 | Future | Make `ScopedAllocator` the preferred interface for selecting built-in CUDA allocators and deprecate `ScopedMempool` when compatibility permits | None expected initially | Public built-in allocator accessors |
| 8 | Future | CUDA 13 managed memory-pool backend for graph-capturable `wp.CudaManagedAllocator()` allocations | None; reuses `managed_memory` and memory-pool support state | Extends `wp_alloc_device_managed` |
| 9 | Future | Expand `wp.can_access()` to additional resources and custom/external allocation metadata | None expected initially | None |

### Phase 1: Cross-Device Launch Support

**Goal:** Replace the unconditional per-argument device check in `wp.launch()` with an explicit launch verification mode. The default `LaunchArrayAccessMode.RELAXED` passes cross-device array arguments straight through to the hardware. On systems with unified system-memory access (HMM or host-page-table ATS), this means GPU kernels can directly consume CPU arrays with zero launch overhead and zero friction. On systems where the access is illegal, the CUDA runtime or host process produces the error. `LaunchArrayAccessMode.STRICT` restores the original same-device rule, and `LaunchArrayAccessMode.CHECKED` provides allocation-aware diagnostics before the kernel runs, including during CUDA graph capture.

This phase delivers six things: (a) query three new device attributes, (b) redesign `Device.can_access()` as a conservative device-level/default-allocation query, (c) add `wp.can_access(device, array)` as a public allocation-aware resource query for Warp arrays, (d) replace the unconditional `pack_arg()` same-device check with an explicit launch verification policy, (e) add `wp.config.LaunchArrayAccessMode` / `warp.config.launch_array_access_mode` with allocation-aware verification for Warp-owned arrays where Warp can identify the allocator, including pinned CPU arrays on CUDA devices with UVA, and (f) add tests and advanced user documentation for the CPU/GPU memory access model.

#### 1a. Query Device Attributes

Three CUDA device attributes are needed:

- **`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`** -- answers "can this GPU access ordinary `malloc`'d CPU memory?" This is the attribute that determines whether a Warp `wp.array(device="cpu")` (backed by `malloc` via `CpuDefaultAllocator`) can be dereferenced by a GPU kernel. Without it, we cannot distinguish a system where the GPU can read CPU pointers (HMM, host-page-table ATS, Jetson Thor) from one where it cannot (discrete GPU without HMM, limited Tegra, Windows).

- **`CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST`** -- answers "can the CPU directly access CUDA managed memory resident on the GPU without migration?" This does not imply that Warp `wp.array(device="cuda:0")` allocations backed by `cuMemAlloc` via `CudaDefaultAllocator` can be safely passed to CPU kernels. Phase 1 exposes the capability as a device property, but `Device.can_access()` and `LaunchArrayAccessMode.CHECKED` remain conservative for CPU-to-CUDA Warp arrays because Warp's built-in CUDA arrays are not CUDA managed-memory allocations. `LaunchArrayAccessMode.RELAXED` still passes those pointers through when requested by the user.

- **`CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED`** -- answers whether the CUDA device reports native CPU-GPU atomic hardware capability across the interconnect. This is not a Warp `wp.atomic_*` contract. Current CPU-side Warp atomics are plain read/modify/write operations, and CUDA-side Warp atomics are not specified as system-scope host/device operations. Exposing this as a device property lets users and downstream tools (e.g., documentation, `wp.prefetch()` heuristics) reason about hardware capability, but CPU/GPU atomic algorithms remain follow-up work. This attribute must be treated independently from `direct_managed_mem_access_from_host`.

The first attribute is needed to gate the GPU-accessing-CPU branch in `Device.can_access()`, `wp.can_access(device, array)`, and allocation-aware launch verification. The second and third are exposed as queryable device properties for users who need to reason about managed-memory host access and CUDA-reported cross-device atomic hardware capability. `Device.can_access()`, `wp.can_access(device, array)`, and `LaunchArrayAccessMode.CHECKED` do not use `direct_managed_mem_access_from_host` for CPU-to-CUDA default arrays because those are not CUDA managed-memory allocations.

**Native layer changes (`warp/native/warp.cu`, `warp/native/warp.h`)**

Add three fields to `DeviceInfo`:

```cpp
struct DeviceInfo {
    // ... existing fields ...
    int pageable_memory_access = 0;
    int direct_managed_mem_access_from_host = 0;
    int host_native_atomic_supported = 0;
};
```

Query them during device enumeration, alongside the existing `cuDeviceGetAttribute` calls:

```cpp
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].pageable_memory_access,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device));
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].direct_managed_mem_access_from_host,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, device));
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].host_native_atomic_supported,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device));
```

**CUDA version requirements:** All three attributes are enum values in `CUdevice_attribute` that have been present since well before CUDA 12.0 (Warp's minimum supported CUDA toolkit version): `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS` (= 88, CUDA 8.0+), `CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST` (= 101, CUDA 9.2+), `CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED` (= 86, CUDA 8.0+). No `#if CUDA_VERSION` guard is needed. They are queried via `cuDeviceGetAttribute`, which Warp already loads dynamically via `cuGetProcAddress` at version 2000. The driver will return 0 for any attribute the hardware does not support, which is the correct default (feature not available).

Add accessor functions, following the existing pattern of `wp_cuda_device_is_uva()`:

```cpp
// warp.h
WP_API int wp_cuda_device_get_pageable_memory_access(int ordinal);
WP_API int wp_cuda_device_get_direct_managed_mem_access_from_host(int ordinal);
WP_API int wp_cuda_device_get_host_native_atomic_supported(int ordinal);

// warp.cu
int wp_cuda_device_get_pageable_memory_access(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].pageable_memory_access;
    return 0;
}

int wp_cuda_device_get_direct_managed_mem_access_from_host(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].direct_managed_mem_access_from_host;
    return 0;
}

int wp_cuda_device_get_host_native_atomic_supported(int ordinal)
{
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].host_native_atomic_supported;
    return 0;
}
```

**Python layer changes (`warp/_src/context.py`)**

Register the new ctypes bindings in `Runtime.__init__`, following the existing native binding pattern:

```python
self.core.wp_cuda_device_get_pageable_memory_access.argtypes = [ctypes.c_int]
self.core.wp_cuda_device_get_pageable_memory_access.restype = ctypes.c_int
self.core.wp_cuda_device_get_direct_managed_mem_access_from_host.argtypes = [ctypes.c_int]
self.core.wp_cuda_device_get_direct_managed_mem_access_from_host.restype = ctypes.c_int
self.core.wp_cuda_device_get_host_native_atomic_supported.argtypes = [ctypes.c_int]
self.core.wp_cuda_device_get_host_native_atomic_supported.restype = ctypes.c_int
```

Add documented Python `Device` properties using Warp's CPU/GPU terminology rather than CUDA attribute names. These properties are `False` for CPU devices and meaningful for GPU devices:

```python
        is_cpu_memory_access_from_gpu_supported (bool): Indicates whether GPU kernels on this device can
            directly access ordinary CPU memory. ``False`` for CPU devices.
        is_gpu_memory_access_from_cpu_supported (bool): Indicates whether CPU code can directly access
            CUDA managed memory resident on this device without migration. Does not imply Warp default CUDA
            arrays are CPU-accessible. ``False`` for CPU devices.
        is_cpu_gpu_atomic_supported (bool): Indicates whether the CUDA device reports native CPU/GPU atomic
            hardware capability. This is not a guarantee that Warp ``wp.atomic_*`` operations can be used
            concurrently from CPU and GPU kernels; current CPU-side Warp atomics are not hardware atomics.
            ``False`` for CPU devices.
```

Add the properties to `Device.__init__` for CUDA devices:

```python
# Unified CPU/GPU memory access capability properties
self.is_cpu_memory_access_from_gpu_supported = (
    runtime.core.wp_cuda_device_get_pageable_memory_access(ordinal) > 0
)
self.is_gpu_memory_access_from_cpu_supported = (
    runtime.core.wp_cuda_device_get_direct_managed_mem_access_from_host(ordinal) > 0
)
self.is_cpu_gpu_atomic_supported = (
    runtime.core.wp_cuda_device_get_host_native_atomic_supported(ordinal) > 0
)
```

For the CPU device branch, set all three Python properties to `False`.

Do not add aliases or duplicate convenience properties. Phase 1 exposes exactly one Python property per capability. The native C++ fields and accessors intentionally keep CUDA attribute-derived names so maintainers can map them directly to CUDA documentation.

#### 1b. Redesign `Device.can_access()`

The pre-Phase 1 implementation had only a same-context check, with a TODO acknowledging that access needs to be redesigned in terms of both devices and resources:

```python
def can_access(self, other):
    # TODO: this function should be redesigned in terms of (device, resource).
    # - a device can access any resource on the same device
    # - a CUDA device can access pinned memory on the host
    # - a CUDA device can access regular allocations on a peer device if peer access is enabled
    # - a CUDA device can access mempool allocations on a peer device if mempool access is enabled
    other = self.runtime.get_device(other)
    if self.context == other.context:
        return True
    else:
        return False
```

Replace with:

```python
def can_access(self, other):
    # TODO: this function should be redesigned in terms of (device, resource).
    # - a device can access any resource on the same device
    # - a CUDA device can access CPU memory when the device supports it
    # - a CUDA device can access another CUDA device's current built-in allocator when its
    #   corresponding access mode is enabled
    other = self.runtime.get_device(other)

    if self.context == other.context:
        return True

    if self.is_cuda and other.is_cpu:
        return self.is_cpu_memory_access_from_gpu_supported

    if self.is_cpu and other.is_cuda:
        # Warp's default CUDA arrays are device allocations, not CUDA managed-memory allocations.
        return False

    if self.is_cuda and other.is_cuda:
        if other.is_mempool_enabled:
            return is_mempool_access_enabled(other, self)
        return is_peer_access_enabled(other, self)

    return False
```

**Notes on `can_access()` implementation details:**

- The `is_peer_access_enabled()` and `is_mempool_access_enabled()` calls in the GPU-to-GPU branch are module-level functions in `warp/_src/context.py`. They live in the same module as `Device`, so no additional imports are needed.
- The `self.context == other.context` check catches same-device access and same-context aliases. It also covers CPU-to-CPU access because the CPU device has no CUDA context.
- The TODO in the existing code mentions that access depends on the _resource_ (allocation type), not just the device pair. `Device.can_access()` remains a coarse device-level check: it answers whether `self` can access allocations made on `other` by Warp's current built-in allocator choice at query time. For CUDA devices, that means memory-pool access when `other.is_mempool_enabled` is true, and peer access otherwise.
- `Device.can_access()` is not an authoritative predicate for an existing array. An array may have been allocated before mempool settings changed, may use a custom allocator, or may wrap external memory. Any code with an actual array should use `wp.can_access(device, array)` instead.
- `Device.can_access()` is NOT called in the default launch path. It is invoked by APIs and user code that need the coarse device-level/current-default-allocation answer. The launch verification path uses `wp.can_access(device, array)` so CUDA default allocations and CUDA memory-pool allocations can be checked against the array's actual allocator.

#### 1c. Add Top-Level `wp.can_access(device, array)`

Add a public, resource-oriented access query in `warp/_src/context.py` and export it from `warp/__init__.py`:

```python
def can_access(device: DeviceLike, resource) -> bool:
    """Return whether ``device`` can directly access ``resource``.

    Phase 1 supports :class:`warp.array` resources only. Future phases may extend
    this function to other device-bound resources such as hash grids and meshes.
    """
    device = runtime.get_device(device)

    if warp._src.types.is_array(resource):
        return _is_array_accessible_from_device(resource, device)

    raise TypeError("wp.can_access() only supports Warp arrays in this release")
```

`wp.can_access(device, array)` answers whether code running on `device` can directly dereference the memory backing `array`. It is memory-kind-aware where Warp can classify the backing pointer and has enough ownership metadata for the relevant access predicate:

- Same device/context returns `True`.
- CUDA device accessing a CPU array returns `True` for pinned CPU arrays on UVA CUDA devices, and otherwise follows `device.is_cpu_memory_access_from_gpu_supported`.
- CPU accessing non-managed CUDA device or memory-pool arrays returns `False`; CPU access to CUDA managed arrays follows the managed-memory predicates introduced in Phase 5.
- CUDA device accessing CUDA managed memory uses managed-memory predicates, not peer or memory-pool predicates.
- CUDA device accessing another CUDA device's ordinary CUDA device memory uses peer access, including for externally wrapped pointers that CUDA classifies as `wp.MemoryKind.CUDA_DEVICE`.
- CUDA device accessing another CUDA device's Warp-owned memory-pool allocation uses memory-pool access.
- Externally wrapped or custom CUDA memory-pool allocations remain unknown for cross-device access proof because CUDA pointer attributes identify the memory kind but not whether Warp's queried default-pool access state applies to that specific pool.
- Unclassified custom or external CUDA pointers return `False` through `wp.can_access()` and warn in checked launches.

`False` therefore means "Warp cannot verify that this resource is directly accessible", not necessarily "the hardware could never access this pointer." Advanced users may still use `LaunchArrayAccessMode.RELAXED` to pass pointers through when they know the allocation is valid for the launch device.

The API intentionally does not support `wp.can_access(device, device)`. Device-level/default-allocation queries remain available as `Device.can_access(other_device)`. Keeping the top-level API resource-oriented leaves room to add `wp.can_access(device, hash_grid)` and `wp.can_access(device, mesh)` later without overloading it as another device-pair predicate.

Any internal or public path that has a concrete array should prefer `wp.can_access(device, array)` over `Device.can_access(array.device)`. This includes `LaunchArrayAccessMode.CHECKED` and the future `wp.copy()` staging optimization. `Device.can_access()` is useful only when no concrete resource is available and the caller accepts a coarse answer for the target device's current built-in allocation mode.

Implementation follows Warp array views to their owner allocation where possible. `wp.can_access()` remains a conservative boolean wrapper, while `LaunchArrayAccessMode.CHECKED` uses a private tri-state classifier to distinguish known-inaccessible allocations from unknown custom or external access paths:

```python
def _get_array_allocator(value):
    while warp._src.types.is_array(value):
        allocator = getattr(value, "_allocator", None)
        if allocator is not None:
            return allocator
        value = getattr(value, "_ref", None)
    return None


class _ArrayAccessStatus(enum.Enum):
    ACCESSIBLE = enum.auto()
    INACCESSIBLE = enum.auto()
    UNKNOWN = enum.auto()


def _is_array_accessible_from_device(value, device):
    return _classify_array_access_from_device(value, device) == _ArrayAccessStatus.ACCESSIBLE


def _classify_array_access_from_device(value, device):
    device = runtime.get_device(device)
    value_device = value.device

    if device.context == value_device.context:
        return _ArrayAccessStatus.ACCESSIBLE

    allocator = _get_array_allocator(value)

    if device.is_cuda and value_device.is_cpu:
        if value.pinned and device.is_uva:
            return _ArrayAccessStatus.ACCESSIBLE
        if device.is_cpu_memory_access_from_gpu_supported:
            return _ArrayAccessStatus.ACCESSIBLE
        return _ArrayAccessStatus.INACCESSIBLE

    if device.is_cpu and value_device.is_cuda:
        if isinstance(allocator, CudaDefaultAllocator | CudaMempoolAllocator):
            return _ArrayAccessStatus.INACCESSIBLE
        return _ArrayAccessStatus.UNKNOWN

    if device.is_cuda and value_device.is_cuda:
        if isinstance(allocator, CudaMempoolAllocator):
            if is_mempool_access_enabled(value_device, device):
                return _ArrayAccessStatus.ACCESSIBLE
            return _ArrayAccessStatus.INACCESSIBLE
        if isinstance(allocator, CudaDefaultAllocator):
            if is_peer_access_enabled(value_device, device):
                return _ArrayAccessStatus.ACCESSIBLE
            return _ArrayAccessStatus.INACCESSIBLE
        return _ArrayAccessStatus.UNKNOWN

    return _ArrayAccessStatus.INACCESSIBLE
```

#### 1d. Remove the Launch Device Check

Change `pack_arg()`.

**Scope:** This change only removes the device check for `array` arguments. Other device-bound types (textures, volumes, hash grids) have their own device checks later in `pack_arg()` and remain strict (`value.device != device`). Relaxing those is out of scope for Phase 1: textures and volumes have GPU-side handles (CUDA texture objects, device pointers to internal structures) that may not be accessible cross-device even on systems with ATS system-memory access.

The `pack_arg()` function is called for both forward and adjoint arguments through `pack_args()`. The removed check applies to both paths, so cross-device arrays work in backward passes on capable hardware.

Replace the unconditional device check:

```python
# check device
if value.device != device:
    raise RuntimeError(
        f"Error launching kernel '{kernel.key}', trying to launch on "
        f"device='{device}', but input array for argument '{arg_name}' "
        f"is on device={value.device}."
    )
```

With a policy gate and helper call:

```python
if warp.config.launch_array_access_mode != warp.config.LaunchArrayAccessMode.RELAXED:
    _validate_launch_array_access(kernel, arg_name, value, device)
```

`LaunchArrayAccessMode.RELAXED` is the default and performs no launch array access check. This includes CPU launches with CUDA arrays. Warp still validates array type, dtype, and dimension before passing the pointer through.

`LaunchArrayAccessMode.STRICT` restores the original same-device policy:

```python
if value.device != device:
    _raise_launch_array_access_error(kernel, arg_name, value, device)
```

`LaunchArrayAccessMode.CHECKED` checks the actual Warp array pointer where Warp can classify it. Known-accessible pointers proceed, known-inaccessible pointers raise before launch, and unknown access cases warn through a bounded cache keyed by `(kernel, argument name, source device, launch device)` before proceeding. Unknown cases include unclassified CUDA pointers and externally wrapped or custom memory-pool pointers whose specific pool access state Warp cannot prove.

The policy helper is responsible for mode validation:

```python
def _validate_launch_array_access(kernel, arg_name, value, device):
    mode = warp.config.launch_array_access_mode

    if value.device == device:
        return

    if mode == warp.config.LaunchArrayAccessMode.STRICT:
        _raise_launch_array_access_error(kernel, arg_name, value, device)

    if mode == warp.config.LaunchArrayAccessMode.CHECKED:
        access_status = _classify_array_access_from_device(value, device)
        if access_status == _ArrayAccessStatus.INACCESSIBLE:
            _raise_launch_array_access_error(kernel, arg_name, value, device)
        if access_status == _ArrayAccessStatus.UNKNOWN:
            _warn_unknown_launch_array_access(kernel, arg_name, value, device)
        return

    raise ValueError(
        f"warp.config.launch_array_access_mode must be a warp.config.LaunchArrayAccessMode value, got {mode!r}"
    )
```

**Design rationale:** The previous launch design called the access predicate on every array argument of every launch. Even though the predicate is cheap (property lookups), it adds up in hot launch paths with many array arguments. The default `RELAXED` mode does not call the policy helper, so the normal path avoids device comparison, allocation inspection, and native peer/mempool queries. `STRICT` gives users the old fast same-device check, while `CHECKED` gives allocation-aware diagnostics for debugging mixed-device launches.

#### 1e. Launch Verification Mode Config

Add to `warp/config.py`:

```python
from enum import IntEnum as _IntEnum


class LaunchArrayAccessMode(_IntEnum):
    """Array-access verification modes for kernel launches."""

    RELAXED = 0
    """Perform no launch array access checks before launching kernels."""

    CHECKED = 1
    """Detect cross-device Warp array access issues before launching kernels where possible."""

    STRICT = 2
    """Require every Warp array argument to be allocated on the launch device."""


launch_array_access_mode: LaunchArrayAccessMode = LaunchArrayAccessMode.RELAXED
"""Kernel launch array access verification mode.

``LaunchArrayAccessMode.RELAXED`` performs no launch array access checks and is
the default. ``LaunchArrayAccessMode.STRICT`` requires every Warp array argument
to be on the launch device, matching Warp's original behavior.
``LaunchArrayAccessMode.CHECKED`` checks whether cross-device Warp array
arguments are accessible from the launch device before passing their pointers to
the kernel. Checked mode uses the array's observed memory kind and ownership
metadata where Warp can determine them. Unknown access cases, including
unclassified CUDA pointers and unowned memory-pool pointers, warn through a
bounded per-launch-pattern cache and then proceed.

Unlike ``verify_cuda``, this setting can be used during CUDA graph capture
because checks run before each launch is recorded. For cross-GPU graph capture,
enable peer or memory-pool access with Warp APIs before capture begins.

Note: Strict and checked modes impact performance.
"""
```

`LaunchArrayAccessMode` is accessed via `warp.config` (not re-exported at the top level), so callers write `wp.config.LaunchArrayAccessMode.CHECKED` when assigning `wp.config.launch_array_access_mode`.

**When to use:** If a user on a discrete GPU (without HMM) accidentally passes a CPU array to a GPU kernel, the kernel will fault with `CUDA_ERROR_ILLEGAL_ADDRESS`. This error is asynchronous and can corrupt the CUDA context, requiring a process restart. The recommended workflow is:

1. Observe the CUDA error.
2. Set `warp.config.launch_array_access_mode = warp.config.LaunchArrayAccessMode.CHECKED`.
3. Re-run. The clear Python `RuntimeError` identifies which kernel and which argument caused the mismatch, before the kernel ever launches.
4. Fix the code, then restore `LaunchArrayAccessMode.RELAXED` for the default fast path.

`launch_array_access_mode` is compatible with CUDA graph capture because `STRICT` and `CHECKED` checks happen before each launch is recorded and do not depend on post-launch CUDA error polling. Cross-GPU graph captures still depend on the correct access mode being enabled before capture: peer access for default CUDA allocations and memory-pool access for CUDA memory-pool allocations. Warp records peer and memory-pool access state when `wp.set_peer_access_enabled()` and `wp.set_mempool_access_enabled()` are called so graph-capture verification does not need to issue CUDA access-query calls while capture is active.

#### 1f. User-facing Documentation

Update `docs/user_guide/execution_and_performance/memory_management.rst` to explain the CPU/GPU memory model for advanced users, including:

- The three public `Device` properties and that they are `False` for CPU devices.
- How to guard mixed CPU/GPU launches using the capability properties.
- How `Device.can_access()` relates to CPU/GPU capability properties, GPU/GPU peer access, and GPU/GPU memory-pool access.
- How `wp.can_access(device, array)` checks a specific Warp array allocation, why it returns `False` for unknown/custom CUDA allocations, and why `wp.can_access(device, device)` is not supported.
- How launch verification uses the same allocation-aware predicate as `wp.can_access(device, array)`.
- How and when to use `wp.config.launch_array_access_mode`, including its CUDA graph capture compatibility.
- Why direct loads/stores do not imply CPU/GPU atomic safety.

#### Behavior matrix after Phase 1

Default mode (`LaunchArrayAccessMode.RELAXED`): no Python-level launch array access checking. The hardware decides.

| Launch device | Array device | Discrete GPU (no HMM) | HMM system | Jetson Thor | Host-page-table ATS (DGX Spark / GB10 observed) |
|---|---|---|---|---|---|
| `cuda:0` | `cuda:0` | OK (same device) | OK | OK | OK |
| `cuda:0` | `cpu` (pageable) | **CUDA fault** | **OK** (HMM) | **OK** (ATS system memory) | **OK** (ATS) |
| `cuda:0` | `cpu` (pinned) | **OK** (UVA pinned zero-copy) | **OK** | **OK** | **OK** |
| `cpu` | `cuda:0` | **Segfault** | **Segfault** | **Segfault** | **Segfault** for Warp default arrays |
| `cuda:0` | `cuda:1` | CUDA fault / OK (peer or mempool access, depending on allocation) | CUDA fault / OK (peer or mempool access, depending on allocation) | N/A on single-GPU Thor | CUDA fault / OK (peer or mempool access, depending on allocation) |

Strict mode (`LaunchArrayAccessMode.STRICT`): every cross-device Warp array argument is rejected before launch.

| Launch device | Array device | All systems |
|---|---|---|
| `cuda:0` | `cuda:0` | OK (same device) |
| `cuda:0` | `cpu` (pageable) | **RuntimeError** |
| `cuda:0` | `cpu` (pinned) | **RuntimeError** |
| `cpu` | `cuda:0` | **RuntimeError** |
| `cuda:0` | `cuda:1` | **RuntimeError** |

Checked mode (`LaunchArrayAccessMode.CHECKED`): each Warp array argument is checked with memory-kind-aware launch verification where Warp can determine the backing memory class and relevant access predicate. Unknown access cases, including unclassified CUDA pointers and unowned memory-pool pointers, warn through a bounded per-launch-pattern cache and then proceed.

| Launch device | Array device | Discrete GPU (no HMM) | HMM system | Jetson Thor | Host-page-table ATS (DGX Spark / GB10 observed) |
|---|---|---|---|---|---|
| `cuda:0` | `cuda:0` | OK (same device) | OK | OK | OK |
| `cuda:0` | `cpu` (pageable) | **RuntimeError** | **OK** (HMM) | **OK** (ATS system memory) | **OK** (ATS) |
| `cuda:0` | `cpu` (pinned) | **OK** (UVA pinned zero-copy) | **OK** | **OK** | **OK** |
| `cpu` | `cuda:0` | **RuntimeError** | **RuntimeError** | **RuntimeError** | **RuntimeError** for Warp default arrays |
| `cuda:0` | `cuda:1` | RuntimeError / OK (peer or mempool access, depending on allocation) | RuntimeError / OK (peer or mempool access, depending on allocation) | N/A on single-GPU Thor | RuntimeError / OK (peer or mempool access, depending on allocation) |

On a standard discrete-GPU workstation without HMM, users who pass a CPU array to a GPU kernel in `RELAXED` mode will get a CUDA fault instead of the current Python `RuntimeError`. This is a deliberate tradeoff: zero overhead in the launch path for the default mode, at the cost of a less friendly error for an incorrect program. `CHECKED` mode restores the friendly allocation-aware error for diagnosis, and `STRICT` mode restores the original same-device rule.

#### Stream selection for cross-device launches

When an array on device A is passed to a kernel on device B, Warp must ensure proper synchronization. The current `wp.launch()` already selects a stream based on the launch device. The kernel launch happens on that stream, and since the pointer is passed through without checking, the hardware coherency or HMM page fault mechanism handles visibility on capable systems.

However, if the array was _produced_ by a kernel on a different stream, the caller is responsible for synchronizing (e.g., via `wp.synchronize()` or stream events). This is the same requirement as for same-device multi-stream usage and does not need special handling here.

### Phase 2: Explicit Prefetch API (`wp.prefetch()`)

**Goal:** Provide a public API for users to request migration of array data to a specific device, without copying. This is a performance optimization for HMM and ATS systems where data is accessible across processors but performance can depend on physical placement.

This phase introduces one additional device attribute and one new native function.

#### New device attribute: `pageable_memory_access_uses_host_page_tables`

**CUDA attribute:** `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`

This attribute distinguishes HMM (software coherency) from host-page-table ATS. Both have `pageable_memory_access == 1`, but prefetch behavior differs:
- On host-page-table ATS with distinct CPU/GPU memory pools, prefetch can migrate physical pages via hardware DMA with cache-line coherency. On integrated systems such as Jetson Thor, prefetch may succeed but may not improve placement because CPU and GPU share the same DRAM.
- On HMM, prefetch triggers software page migration with TLB shootdowns. It works but has higher overhead and different failure modes.

The `wp.prefetch()` implementation needs this to provide accurate diagnostics (e.g., warning when prefetching on a system where it may cause page-fault storms) and to choose the right native API call path.

**Native layer:** Same pattern as Phase 1 -- add to `DeviceInfo`, query during enumeration, add accessor function. `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES` (= 100) has been present since CUDA 9.2, well before Warp's minimum of CUDA 12.0. No compile-time guard needed.

```cpp
// DeviceInfo addition
int pageable_memory_access_uses_host_page_tables = 0;

// Query
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].pageable_memory_access_uses_host_page_tables,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device));

// Accessor
WP_API int wp_cuda_device_get_pageable_memory_access_uses_host_page_tables(int ordinal);
```

**Python layer:**

```python
self.pageable_memory_access_uses_host_page_tables = (
    runtime.core.wp_cuda_device_get_pageable_memory_access_uses_host_page_tables(ordinal) > 0
)
```

#### New native function: `wp_cuda_mem_prefetch_async`

```cpp
// warp.h
WP_API int wp_cuda_mem_prefetch_async(void* ptr, size_t size_in_bytes,
                                       int device_ordinal, void* stream);
```

This wraps `cuMemPrefetchAsync` (driver API). The `device_ordinal` can be `-1` to indicate the CPU as the target (maps to `CU_DEVICE_CPU` in the driver API).

**CUDA API versioning:** The `cuMemPrefetchAsync` driver API has two versions:

- **v1** (CUDA 8.0+, version 8000): `cuMemPrefetchAsync(CUdeviceptr, size_t, CUdevice dstDevice, CUstream)` -- takes a simple `CUdevice` ordinal for the destination.
- **v2** (CUDA 12.8+, version 12080): `cuMemPrefetchAsync(CUdeviceptr, size_t, CUmemLocation location, unsigned int flags, CUstream)` -- takes a `CUmemLocation` struct (supports NUMA node targeting) and flags.

In CUDA 13.0 headers, `cuMemPrefetchAsync` is `#define`'d to `cuMemPrefetchAsync_v2`. Following Warp's Driver API entry-point policy, the implementation should avoid the macro-selected newer ABI and request the oldest sufficient entry point explicitly:

```cpp
// In init_cuda_driver(), load the prefetch entry point:
get_driver_entry_point("cuMemPrefetchAsync", 8000, &(void*&)pfn_cuMemPrefetchAsync_v1);
```

The `wp_cuda_mem_prefetch_async` wrapper uses the v1 signature:

```cpp
int wp_cuda_mem_prefetch_async(void* ptr, size_t size_in_bytes,
                                int device_ordinal, void* stream)
{
    CUdeviceptr devPtr = (CUdeviceptr)ptr;
    CUstream hStream = (CUstream)stream;

    if (pfn_cuMemPrefetchAsync_v1) {
        CUdevice dstDevice = (device_ordinal >= 0)
            ? g_devices[device_ordinal].device
            : CU_DEVICE_CPU;
        return check_cu(pfn_cuMemPrefetchAsync_v1(devPtr, size_in_bytes,
                                                    dstDevice, hStream)) ? 0 : -1;
    }
    return -1;  // prefetch not available
}
```

This deliberately differs from APIs such as `cuMemcpyBatchAsync`, where Warp
needs newer semantics and therefore gates the newer entry point with both
`#if CUDA_VERSION` and `driver_version` checks.

**Compile-time / runtime compatibility matrix for Phase 2:**

| Toolkit used to build Warp | Runtime driver | Prefetch available? | API version used |
|---|---|---|---|
| CUDA 12.0 -- 12.7 | Any 12.0+ | Yes | v1 (CUdevice) |
| CUDA 12.8+ | Driver < 12.8 | Yes | v1 (CUdevice) |
| CUDA 12.8+ | Driver >= 12.8 | Yes | v1 (CUdevice); v2 reserved for future NUMA targeting |

The v1 API is fully sufficient for the `wp.prefetch()` use case (migrate to a device or to the CPU). The v2 API adds NUMA node targeting, which is not needed initially. If Warp later exposes NUMA targeting, that path should add a separate v2 load guarded by both toolkit headers and runtime driver support.

**Disabling prefetch on older CUDA:** Warp loads the v1 entry point across toolkit versions. The v1 API works for `cudaMallocManaged` allocations on all systems, and also for system-allocated (`malloc`) memory on HMM / host-page-table ATS systems. The Python `wp.prefetch()` wrapper should catch errors from the driver (e.g., if the pointer is not in a prefetchable region) and emit a warning rather than raising, since prefetch is a performance hint.

Implementation notes:
- `cuMemPrefetchAsync` works on any pointer that falls within a unified memory region -- including plain `malloc` on HMM / host-page-table ATS systems, `cuMemAllocManaged` allocations, and `cuMemAlloc` allocations on systems where device allocations are host-accessible.
- On systems where the pointer is not in a prefetchable region, the call returns an error. The Python wrapper should catch this and either warn or silently ignore, since prefetch is a hint.
- The prefetch is stream-ordered: it begins after all prior operations on the stream complete and finishes before any subsequent operations on the stream begin.

#### Python API

```python
def prefetch(
    array: warp.array,
    device: DeviceLike = None,
    stream: Stream | None = None,
):
    """Request asynchronous migration of ``array`` data toward ``device``.

    On systems with host-page-table ATS or software coherency (HMM),
    this issues a ``cuMemPrefetchAsync`` to migrate the
    array's physical pages closer to the specified device. The array
    remains valid and accessible from any device during and after the
    prefetch.

    On systems without unified memory support for the array's allocation
    type, this function is a no-op and emits a warning.

    This is a performance hint, not a correctness requirement. Kernels
    will produce correct results regardless of whether prefetch is
    called.

    Args:
        array: The array whose data should be migrated.
        device: The target device. If ``None``, uses the default device.
        stream: The stream on which to order the prefetch. If ``None``,
            uses the current stream on the target device.
    """
```

#### Usage example

```python
# On DGX Spark / GB10 (host-page-table ATS system):
data = wp.array(np.random.randn(1000000), dtype=wp.float32, device="cpu")

# Prefetch to GPU before a compute-heavy kernel
wp.prefetch(data, device="cuda:0")
wp.launch(heavy_compute_kernel, dim=data.size, inputs=[data], device="cuda:0")

# Prefetch back to CPU before CPU-side post-processing
wp.prefetch(data, device="cpu")
result = data.numpy()
```

### Phase 3: Optional Automatic Prefetch in `wp.launch()` (Future)

**Goal:** When a cross-device array argument is detected in `pack_arg()` on a coherent system, optionally issue a prefetch automatically before the kernel launch. This is a convenience optimization that should be off by default.

This phase introduces one additional device attribute.

#### New device attribute: `is_integrated`

**CUDA attribute:** `CU_DEVICE_ATTRIBUTE_INTEGRATED`

This attribute indicates whether the GPU is physically integrated into the same chip/package as the CPU (Tegra/Jetson SoCs). It is already queried in the native layer but stored in a local variable `device_attribute_integrated` used only for the IPC check. This phase promotes it to a stored `DeviceInfo` field exposed to Python.

The auto-prefetch heuristic needs this because prefetch on an integrated GPU is usually pointless -- the CPU and GPU share the same physical DRAM, so there is no "closer" location to migrate data to. Jetson Thor testing showed that `cuMemPrefetchAsync` can succeed for ordinary `malloc` memory, but that does not make automatic prefetch useful. Without this attribute, the auto-prefetch code would waste time issuing low-value prefetch calls on every integrated-GPU kernel launch with cross-device arrays.

#### Config flag

Add to `warp/config.py`:

```python
auto_prefetch = False
"""When True and launching a kernel on a device that can access memory on
another device (e.g., GPU accessing CPU memory on an HMM or ATS system),
automatically prefetch cross-device array arguments to the launch device
before the kernel begins. Default is False because automatic prefetch is
not always beneficial -- for example, streaming read-once access patterns
are better served by remote access over NVLink C2C than by migrating
the data."""
```

#### Implementation in `pack_arg()`

After the device accessibility check passes (Phase 1), and before returning the packed argument:

```python
if value.device != device and warp.config.auto_prefetch:
    # Skip prefetch on integrated GPUs -- CPU and GPU share the same
    # DRAM, so migration is meaningless.
    if not device.is_integrated:
        try:
            stream_handle = device.stream.cuda_stream if device.is_cuda else 0
            device_ordinal = device.ordinal if device.is_cuda else -1
            runtime.core.wp_cuda_mem_prefetch_async(
                value.ptr, value.capacity, device_ordinal, stream_handle
            )
        except Exception:
            pass  # Prefetch is best-effort
```

#### Why off by default

Automatic prefetch has several cases where it hurts more than it helps:

1. **Read-once data**: If a kernel reads an array once and never again, prefetching (which may involve a DMA transfer or page-table work) can be slower than direct access.
2. **CPU-produced, GPU-consumed streaming data**: If the CPU is continuously writing to a buffer that the GPU reads, prefetching would fight with the CPU's writes. The CUDA documentation explicitly recommends keeping such data CPU-resident and letting the GPU read remotely.
3. **Small arrays**: The overhead of issuing a prefetch (driver call, DMA setup) exceeds the benefit for small transfers.
4. **Multiple kernels**: If multiple kernels on different devices access the same array, prefetching to one device may pessimize access from another.

Users who want automatic prefetch can enable it globally via `warp.config.auto_prefetch = True` or per-launch by calling `wp.prefetch()` explicitly before the launch.

### Phase 4: Improve `wp.copy()` for Coherent Systems (Future)

**Goal:** When source and destination arrays are on different devices and the destination-side copy kernel can directly access the source allocation, skip the staging buffer logic in `wp.copy()`.

No new attributes or native functions. This should reuse the Phase 1 array access predicate through `wp.can_access(dest.device, src)` so the copy optimization makes the same allocation-aware decisions as launch verification.

The current `wp.copy()` implementation has a TODO for this:

```python
# Copying between different devices requires contiguous arrays.  If the arrays
# are not contiguous, we must use temporary staging buffers for the transfer.
# TODO: We can skip the staging if device access is enabled.
```

On systems where `is_cpu_memory_access_from_gpu_supported` is true, the GPU can directly read non-contiguous CPU memory, so staging is unnecessary for destination-device copy kernels that read CPU source arrays. The reverse direction for Warp default CUDA arrays remains invalid, and multi-GPU CUDA arrays need allocation-specific peer or memory-pool access checks. The GPU-reading-CPU fix is straightforward:

```python
if src.device != dest.device:
    # If direct access is available, we can copy non-contiguous arrays
    # without staging, using a kernel on the destination device.
    if wp.can_access(dest.device, src):
        # Launch a copy kernel on the destination device that reads
        # directly from the source array's memory.
        launch_direct_access_copy(src, dest)
    else:
        # Existing staging buffer logic for non-contiguous arrays...
        ...
```

This is a performance optimization and not required for correctness -- the existing staging approach works correctly on all systems.

### Phase 5: Managed Allocator and Memory Kind

**Goal:** Add an explicit managed-memory allocation path for Warp arrays without changing the meaning of `device` or the default CUDA allocator. Managed memory is a CUDA pointer memory kind, not a new device. A managed Warp array remains associated with the CUDA device used to allocate it, but `wp.can_access()` and `LaunchArrayAccessMode.CHECKED` can apply managed-memory access rules instead of treating the pointer as ordinary CUDA device memory or an unknown pointer.

This phase introduces `wp.CudaManagedAllocator()`, `wp.MemoryKind`, `array.memory_kind`, two additional CUDA device attributes, a managed native allocation wrapper, and a native CUDA pointer classifier.

#### Public API: `wp.CudaManagedAllocator`

`wp.CudaManagedAllocator` is a top-level allocator class that satisfies the existing `Allocator` protocol. It has no device argument and no public attach-flag argument. The allocator object is not bound to one CUDA device and can be constructed before any CUDA context is current. Each allocation still happens under the target device's CUDA context, and that device must report CUDA managed-memory support. It is used through the same APIs as other CUDA allocators:

```python
managed = wp.CudaManagedAllocator()

with wp.ScopedAllocator("cuda:0", managed):
    data = wp.empty(1024, dtype=wp.float32, device="cuda:0")
```

The array's `device` remains `cuda:0`:

```python
data.device == wp.get_device("cuda:0")
```

The `CudaManagedAllocator` constructor intentionally does not take a device. A device argument would suggest that `cudaMallocManaged` immediately places pages in that device's physical memory, which CUDA does not guarantee. The target device/context still matters for allocation API calls: Warp's array constructors push the target CUDA context before invoking the allocator, and direct calls to `CudaManagedAllocator.allocate()` require the caller to have already made a managed-memory-capable CUDA context current. Deallocation does not need to replay the allocation context because CUDA accepts `cudaFree()` for `cudaMallocManaged()` pointers from the current runtime context. Physical placement is left to CUDA Unified Memory and can be guided explicitly through `wp.prefetch()` once Phase 2 exists.

`wp.CudaManagedAllocator()` always uses global managed-memory attach semantics. For the direct fallback path this means `cudaMallocManaged(..., cudaMemAttachGlobal)`. Warp does not expose `cudaMemAttachHost` or `cudaMemAttachSingle` in the initial API; those are specialized ownership/scheduling controls better left to custom allocators or a future explicit stream-attach API.

Users may install one `CudaManagedAllocator` for all CUDA devices:

```python
wp.set_cuda_allocator(wp.CudaManagedAllocator())
```

Because allocation happens under the target device context and the allocator object stores no device of its own, sharing one instance across multiple managed-memory-capable CUDA devices is valid. Direct calls to `CudaManagedAllocator.allocate()` require an active CUDA context whose device supports managed memory. Direct calls to `CudaManagedAllocator.deallocate()` release the pointer through CUDA's current runtime context. Array factory calls pass the target device context automatically for allocation.

#### Memory-kind inspection

Before this phase, access rules were inferred mostly from allocator class identity. Managed arrays and arrays received from another layer need a first-class memory-kind query so users can inspect what Warp can observe about the backing pointer without relying on a private allocator object. The public enum is:

```python
class MemoryKind(enum.IntEnum):
    # Values must match wp_memory_kind in warp/native/warp.h.
    UNKNOWN = 0
    HOST = 1
    PINNED = 2
    CUDA_DEVICE = 3
    CUDA_MEMPOOL = 4
    CUDA_MANAGED = 5
```

Expose the query through `array.memory_kind` on concrete `wp.array` instances. The property returns a `wp.MemoryKind` value and follows views to their owner allocation. Warp-owned arrays cache the memory kind reported by the allocator at creation time. CPU arrays and zero-sized arrays are classified from Warp array state. Externally wrapped CUDA pointers are classified once from CUDA Driver API pointer attributes, then cache the result:

- `CU_POINTER_ATTRIBUTE_IS_MANAGED` identifies managed memory.
- `CU_POINTER_ATTRIBUTE_MEMORY_TYPE` distinguishes host and device memory.
- `CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE` identifies CUDA memory-pool allocations.

The CUDA classifier returns `wp.MemoryKind.UNKNOWN` if CUDA cannot classify the pointer. This lets externally wrapped managed, default-device, and memory-pool pointers report their observed CUDA memory class, while still keeping access checks conservative when Warp lacks the allocator ownership metadata needed to prove cross-device access. `wp.MemoryKind.CUDA_DEVICE` intentionally covers CUDA device memory that is not classified as managed or memory-pool memory; it is not named after one specific allocation API.

The memory kind reports the observed pointer class only. It does not report current physical residency of managed pages, synchronization state, peer or memory-pool access authorization, or whether a CPU/GPU can safely access the pointer at that moment. Accessibility remains a separate query through `wp.can_access(device, array)`. Indexed arrays do not expose a single public memory kind because data and index buffers can be backed by different allocations; `wp.can_access()` and checked launches inspect both buffers separately.

#### New device attributes

Two CUDA attributes are used:

- **`CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY`** -- answers whether the device can allocate managed memory on this system. Expose this as `Device.is_managed_memory_supported`.
- **`CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`** -- distinguishes limited managed-memory systems from full concurrent managed-memory systems. Expose this as `Device.is_concurrent_managed_access_supported`.

Native layer additions follow the Phase 1 pattern:

```cpp
struct DeviceInfo {
    // ... existing fields ...
    int managed_memory = 0;
    int concurrent_managed_access = 0;
};
```

Query during device enumeration:

```cpp
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].managed_memory,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device));
check_cu(cuDeviceGetAttribute_f(
    &g_devices[i].concurrent_managed_access,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device));
```

Expose accessors:

```cpp
WP_API int wp_cuda_device_get_managed_memory_supported(int ordinal);
WP_API int wp_cuda_device_get_concurrent_managed_access_supported(int ordinal);
```

Register ctypes bindings in `Runtime.__init__`, then set the Python properties on CUDA devices. CPU devices set both to `False`.

#### Native managed allocation backend

The native managed allocation API exposes this wrapper:

```cpp
WP_API void* wp_alloc_device_managed(void* context, size_t size, const char* tag = nullptr);
```

The native wrapper chooses the backend:

1. If the context's CUDA device does not support managed memory, return `NULL`.
2. Otherwise allocate with `cudaMallocManaged(&ptr, size, cudaMemAttachGlobal)`.
   CUDA reports any capture-time allocation failure.

Managed allocations are released through the existing `wp_free_device_default()`
path because CUDA documents `cudaFree()` as valid for `cudaMallocManaged()`
allocations. This preserves the same deferred-free behavior as
`CudaDefaultAllocator` when graph captures are active.

Local verification on a Blackwell GPU with driver 580.126.20 confirmed:

- `cudaMallocManaged` during global or thread-local stream capture returns `cudaErrorStreamCaptureUnsupported` and invalidates capture.
- A pointer allocated by `cudaMallocManaged` before capture can be used by captured kernels successfully.

The native wrapper therefore lets CUDA report any capture-time allocation failure
from `cudaMallocManaged()` rather than synthesizing a Python-side reason.

#### Managed access rules

For `wp.MemoryKind.CUDA_MANAGED`, `wp.can_access(device, array)` and `LaunchArrayAccessMode.CHECKED` use managed-memory predicates, not peer or memory-pool predicates:

- Same device/context: `True`.
- CUDA device accessing a managed CUDA array: `True` when the target CUDA device reports `is_managed_memory_supported`. Warp peer access and Warp memory-pool access are not required; CUDA Unified Memory handles migration and visibility. P2P topology can still affect placement and performance.
- CPU accessing a managed CUDA array: `True` only when the owner CUDA device reports `is_concurrent_managed_access_supported` or `is_gpu_memory_access_from_cpu_supported`. Limited managed-memory systems return `False` because CPU access depends on synchronization state that `wp.can_access()` cannot verify.
- CPU/GPU atomics are not implied by managed memory. `device.is_cpu_gpu_atomic_supported` reports only the CUDA hardware capability bit, and current Warp `wp.atomic_*` operations must not be used as overlapping CPU/GPU interprocessor atomics.

The CPU rule is intentionally conservative. On limited managed-memory systems, CUDA permits some CPU access patterns after synchronization, but `wp.can_access()` is not a synchronization-state query and checked launch validation cannot prove that the CPU will avoid accessing the allocation while the GPU is active.

#### Interop and array behavior

Managed arrays remain CUDA arrays in Warp:

- `array.device` is the CUDA device used for allocation.
- `array.pinned` is `False`.
- `__cuda_array_interface__` remains available.
- DLPack exports as CUDA, not CPU.
- `array.cptr()` remains unavailable because the array is not a CPU array.
- `array.numpy()` keeps the existing copy-to-CPU behavior in Phase 5. Zero-copy NumPy views over managed CUDA arrays are a separate feature because they would change host synchronization and lifetime expectations.
- CUDA IPC rejects managed allocations in Phase 5, matching the conservative handling for memory-pool allocations. IPC support for managed memory can be considered separately if a concrete use case appears.

`wp.prefetch()` from Phase 2 accepts managed arrays and is the explicit physical-placement hint for users who want to move pages toward a GPU or back toward the CPU. `wp.CudaManagedAllocator()` itself does not promise initial residency.

### Phase 6: Allocator Policy for Persistent Warp Resources

**Goal:** Make user allocator policy apply consistently to persistent Warp
resources, not only arrays created directly by Python array factories. A user who enters
`wp.ScopedAllocator(device, allocator)` expects persistent storage created by
public constructors to use that allocator where the storage has the same lifetime
and access semantics as a user array.

This phase must first define the allocation boundary:

- **Persistent storage** follows the device's user allocator policy. This
  includes buffers owned by public-facing resources such as meshes, hash grids,
  volumes, Fabric buffers, and similar data structures whose memory remains part
  of the object after construction.
- **Temporary internal storage** does not blindly follow the user allocator.
  This includes workspaces for algorithms such as radix sort, run-length encode,
  acceleration-structure builds, reductions, scans, tile operations, and other
  short-lived implementation details where performance, graph-capture behavior,
  or stream-ordering requirements may differ from the final resource storage.

The first implementation step should be an audit, not a broad mechanical
replacement. For each resource constructor, classify every allocation as
persistent or temporary, then route only persistent storage through
allocator-aware helpers. Temporary allocations should use an internal policy
chosen for correctness and performance. If a temporary path needs customization
later, add an explicit internal temporary allocator policy rather than reusing
the user-facing allocator implicitly.

This phase should also extend access inspection where resources have clear
backing arrays or buffers. For example, `wp.can_access(device, mesh)` can inspect
the mesh's persistent buffers and return a conservative aggregate result. It
should not add `wp.can_access(device, device)`; device-to-device/default-allocation
checks should remain on `Device.can_access(other_device)`.

Open design questions:

- Which existing native resources already allocate persistent storage through
  Python arrays, and which bypass `Device.get_allocator()`?
- Do resource constructors need an explicit `allocator=` argument, or is
  `ScopedAllocator` / device allocator policy sufficient?
- Should temporary allocation policy be exposed at all, or remain internal until
  a concrete use case requires it?
- How should resource-level `memory_kind` or access diagnostics represent
  objects with multiple backing buffers that may differ?

### Phase 7: Unified CUDA Allocator Selection

**Goal:** Make `ScopedAllocator` the common API for selecting built-in and custom
CUDA allocators, while keeping `ScopedMempool` as a compatibility layer until it
can be deprecated safely.

Today, memory-pool selection is controlled separately through
`set_mempool_enabled()` / `ScopedMempool`, while custom allocator selection uses
`set_cuda_allocator()` / `ScopedAllocator`. That split makes the built-in CUDA
allocators feel different from custom allocators even though the runtime already
models them as allocator objects (`CudaDefaultAllocator`,
`CudaMempoolAllocator`, and `CudaManagedAllocator`).

This phase should add public built-in allocator accessors with names based on
memory behavior, not on historical defaults:

```python
wp.get_cuda_device_allocator(device)      # cudaMalloc / cudaFree device memory
wp.get_cuda_mempool_allocator(device)     # cudaMallocAsync / cudaFreeAsync pool memory
wp.get_cuda_managed_allocator()           # cudaMallocManaged managed memory
```

With those accessors, users can write:

```python
with wp.ScopedAllocator(device, wp.get_cuda_device_allocator(device)):
    ...

with wp.ScopedAllocator(device, wp.get_cuda_mempool_allocator(device)):
    ...

with wp.ScopedAllocator(device, wp.get_cuda_managed_allocator()):
    ...
```

The public term "device allocator" is preferred over "default allocator" because
Warp commonly uses memory pools by default on modern CUDA devices. The existing
`CudaDefaultAllocator` class name can remain as an implementation detail or
compatibility alias; new public docs and helper names should avoid implying that
`cudaMalloc` / `cudaFree` is the usual default.

`ScopedMempool` can then become a thin compatibility wrapper over
`ScopedAllocator` and the built-in accessor pair:

- `ScopedMempool(device, True)` selects `get_cuda_mempool_allocator(device)`.
- `ScopedMempool(device, False)` selects `get_cuda_device_allocator(device)`.

Deprecation should be staged. First document `ScopedAllocator` as the preferred
interface, then optionally warn on `ScopedMempool` in a later release once the
built-in accessors are established.

### Phase 8: CUDA 13 Managed Memory-Pool Allocation

**Goal:** Extend `wp.CudaManagedAllocator()` so CUDA 13.0+ builds can allocate
managed arrays during CUDA graph capture when the device supports managed memory
pools. This is a later native backend extension, not part of the current Phase 5
implementation. It should not add a new public memory kind. Arrays still report
`wp.MemoryKind.CUDA_MANAGED`, and CUDA 12.x builds keep the Phase 5
`cudaMallocManaged()` behavior.

#### Managed pool creation

Phase 8 creates one private managed memory pool per CUDA context/device when the
feature is available. The pool is not Warp's ordinary CUDA memory pool; it is a
separate CUDA pool configured for managed memory:

```cpp
#if CUDART_VERSION >= 13000
cudaMemPoolProps props = {};
props.allocType = cudaMemAllocationTypeManaged;
props.handleTypes = cudaMemHandleTypeNone;
props.location.type = cudaMemLocationTypeDevice;
props.location.id = device_ordinal;
cudaMemPoolCreate(&managed_pool, &props);
#endif
```

The CUDA 13 compile-time guard is required because
`cudaMemAllocationTypeManaged` is not defined in CUDA 12.x headers. Warp should
not define a local stand-in enum value or try to emulate this on CUDA 12.9: CUDA
12.9 documents memory-pool allocation type support as pinned-only.

Managed pool creation should happen outside CUDA graph capture. During capture,
the native allocator may use an already-created managed pool but must not attempt
capture-unsafe pool creation or direct `cudaMallocManaged()` fallback. For
captures started by Warp, initialization can happen before capture begins. For
external captures, users may need to allocate one managed array before starting
capture so Warp can initialize the pool for that device.

#### Allocation and free path

The Phase 8 `wp_alloc_device_managed()` backend chooses the allocation path in
this order:

1. If the context's CUDA device does not support managed memory, return `NULL`.
2. If Warp was compiled with CUDA 13.0+, the device supports CUDA memory pools,
   and the managed pool is available, allocate with
   `cudaMallocFromPoolAsync(&ptr, size, managed_pool, stream)` on the current
   Warp stream.
3. If no managed pool is available, allocate with
   `cudaMallocManaged(&ptr, size, cudaMemAttachGlobal)`. This remains the only
   path on CUDA 12.x builds and is also the fallback on CUDA 13+ builds where
   managed pool creation fails. CUDA reports any capture-time failure from this
   direct fallback.

Direct `cudaMallocManaged()` fallback allocations should continue to release via
the default `cudaFree()` path used by Phase 5. Managed-pool allocations may need
a different release path because stream-ordered pool allocations are naturally
paired with `cudaFreeAsync()` and graph-allocation bookkeeping. A Phase 8 design
must decide whether to reintroduce allocation-kind tracking, add an explicit
managed-pool free entry point, or otherwise route pool-backed frees without
adding Phase 5 complexity back to direct managed allocations.

Local verification on a Blackwell GPU with driver 580.126.20 confirmed:

- `cudaMallocManaged` during global or thread-local stream capture returns
  `cudaErrorStreamCaptureUnsupported` and invalidates capture.
- `cudaMallocFromPoolAsync` from a pool with `cudaMemAllocationTypeManaged`
  captures, instantiates, launches, and synchronizes successfully.
- A pointer allocated by `cudaMallocManaged` before capture can be used by
  captured kernels successfully.

The native wrapper must therefore avoid falling back to direct
`cudaMallocManaged()` during capture when a managed-pool allocation is required.

#### Public API and docs

Phase 8 should not add another public allocator class.
`wp.CudaManagedAllocator()` remains the opt-in API for CUDA managed memory. The
implementation may opportunistically use a managed pool when the build, driver,
and device support it, but the public promise remains observed memory kind and
access validation, not initial physical residency.

Documentation should make the graph-capture distinction explicit:

- Managed arrays allocated before capture can be used by captured kernels on all
  supported builds where managed allocation itself succeeds.
- Managed allocation during capture requires CUDA 13.0+ and an initialized
  managed memory pool.
- CUDA 12.x builds, including Warp's CUDA 12.9 PyPI build, use direct
  `cudaMallocManaged()` and rely on CUDA to report capture-time allocation
  failures.
- CUDA may still reject capture-time managed allocation on CUDA 13+ builds
  when the device does not support CUDA memory pools or Warp cannot create the
  managed pool.

Open design questions:

- What owns the managed pool lifetime, and when is it destroyed relative to live
  arrays and CUDA contexts?
- What free-routing mechanism should distinguish direct `cudaMallocManaged`
  pointers from managed-pool pointers without reintroducing fragile global
  side-table behavior into the Phase 5 path?
- Can managed-pool graph allocation and free bookkeeping reuse `g_graph_allocs`,
  or does it need managed-specific state or a new native free entry point?
- What external-capture behavior is worth supporting, given that pool
  initialization itself must happen before capture?
- Is capture-time managed allocation valuable enough to justify the added
  native complexity?

### Phase 9: Custom and External Allocation Metadata

Phase 5 classifies CUDA pointers through Driver API attributes, including externally wrapped managed, ordinary CUDA device, and CUDA memory-pool pointers. Classification is not the same as proving cross-device access:

- `wp.can_access(device, array)` can use managed-memory predicates for `wp.MemoryKind.CUDA_MANAGED` and peer-access predicates for `wp.MemoryKind.CUDA_DEVICE`.
- `wp.can_access(device, array)` remains conservative for externally wrapped or custom `wp.MemoryKind.CUDA_MEMPOOL` pointers because Warp cannot prove that the pointer belongs to the default pool whose access state it queried.
- `LaunchArrayAccessMode.CHECKED` warns once per launch pattern and proceeds only for unknown access cases such as unclassified CUDA pointers or unowned memory-pool pointers.

Phase 9 may add an explicit protocol so custom allocators or external wrappers can declare access predicates or additional allocation metadata. It can also extend `wp.can_access(device, resource)` beyond the built-in persistent resources covered by Phase 6 while preserving the resource-oriented API shape:

```python
wp.can_access(device, hash_grid)
wp.can_access(device, mesh)
```

It should still not add `wp.can_access(device, device)`. Device-to-device/default-allocation checks should continue to live on `Device.can_access(other_device)`.

## Testing Strategy

### Phase 1 test coverage

Coverage lives in `warp/tests/cuda/test_unified_memory.py` (registered in `warp/tests/unittest_suites.py`) and CUDA graph capture tests.

**Attribute query tests (run on all hardware):**
- Verify `is_cpu_memory_access_from_gpu_supported`, `is_gpu_memory_access_from_cpu_supported`, and `is_cpu_gpu_atomic_supported` are `bool` for CUDA devices and `False` for CPU devices.
- Do not assert that `is_cpu_gpu_atomic_supported` implies `is_gpu_memory_access_from_cpu_supported`; Jetson Thor reports native CPU-GPU atomic hardware capability while still rejecting direct CPU access to `cudaMalloc` memory.
- Do not add Warp-level tests that assert overlapping CPU/GPU `wp.atomic_*` updates are correct. That behavior is unsupported until follow-up work adds real CPU atomic lowering and the required GPU system-scope semantics.

**`Device.can_access()` tests (run on all hardware):**
- `device.can_access(device)` is always `True` for every device.
- CPU-to-CPU: always `True`.
- GPU-to-CPU and CPU-to-GPU: assert the result is consistent with Warp default allocation rules:
  - If `is_cpu_memory_access_from_gpu_supported` is `True`, GPU-to-CPU should be `True`.
  - CPU-to-GPU should be `False` for Warp default CUDA arrays, even if managed-memory host access is supported.
- GPU-to-GPU: when the target device has mempools disabled, verify `can_access()` follows `wp.is_peer_access_enabled(target, peer)`. When the target device has mempools enabled, verify it follows `wp.is_mempool_access_enabled(target, peer)`.
- On multi-GPU systems, verify that enabling peer access alone does not make `Device.can_access()` return `True` for a target device whose current built-in allocator is the CUDA mempool.

**`wp.can_access(device, array)` tests (run on all hardware):**
- Same-device arrays return `True`.
- CPU arrays checked from a CUDA device match `device.is_cpu_memory_access_from_gpu_supported` for pageable CPU arrays and return `True` for pinned CPU arrays when `device.is_uva` is true.
- CPU checking a Warp CUDA array returns `False`.
- CUDA arrays checked from another CUDA device use managed-memory predicates for managed memory, peer access for ordinary CUDA device memory, and memory-pool access for Warp-owned CUDA memory-pool allocations.
- Externally wrapped ordinary CUDA device pointers use peer access, externally wrapped managed pointers use managed-memory predicates, and externally wrapped or custom memory-pool pointers return `False` through `wp.can_access()` when Warp cannot prove the pointer's specific pool access state.
- Passing a device as the second argument (`wp.can_access(device, other_device)`) raises `TypeError`.

**Cross-device launch tests (hardware-dependent, skip on incapable systems):**
- On systems where `cuda_device.is_cpu_memory_access_from_gpu_supported` is `True`: allocate a CPU array, launch a GPU kernel that reads and writes it, verify results match expected values.
- On CUDA devices with `device.is_uva`: allocate pinned CPU arrays and verify GPU kernels can read from and write to them with `warp.config.launch_array_access_mode = warp.config.LaunchArrayAccessMode.CHECKED`.
- Test with output arrays (not just inputs).
- Test with multi-dimensional arrays with non-trivial strides.

**Verification mode tests (run on all hardware):**
- With `LaunchArrayAccessMode.RELAXED` (default): verify that no Python-level device check occurs. Cross-device arrays should be accepted by `pack_arg()` under `record_cmd=True`, including CPU launches with CUDA arrays, without executing unsafe kernels.
- With `LaunchArrayAccessMode.STRICT`: verify that any cross-device Warp array argument raises `RuntimeError`, including cases that `CHECKED` would allow, such as pinned CPU arrays on UVA CUDA devices or ordinary CPU arrays on HMM / host-page-table ATS systems.
- With `LaunchArrayAccessMode.CHECKED` on a discrete GPU without HMM: verify that launching with a CPU array raises `RuntimeError` (not a CUDA fault).
- With `LaunchArrayAccessMode.CHECKED` on an HMM / host-page-table ATS system: verify that GPU launches with CPU arrays still succeed (no false positive).
- With `LaunchArrayAccessMode.CHECKED`: verify that unknown cross-device access cases, including unclassified CUDA pointers and unowned memory-pool pointers, warn through a bounded cache keyed by `(kernel, argument name, source device, launch device)` and proceed.
- With `LaunchArrayAccessMode.CHECKED` during CUDA graph capture: capture and replay a same-device CUDA launch successfully.
- On multi-GPU systems with a peer-access-supported pair: allocate with CUDA memory pools disabled, enable peer access before capture, pass an array from the source GPU to a kernel launched on the peer GPU with `LaunchArrayAccessMode.CHECKED`, capture and replay the graph, and verify the results. Skip cleanly when no peer-access pair exists.
- On multi-GPU systems with a memory-pool-access-supported pair: allocate with CUDA memory pools enabled, enable memory-pool access before capture, pass an array from the source GPU to a kernel launched on the peer GPU with `LaunchArrayAccessMode.CHECKED`, capture and replay the graph, and verify the results.
- On multi-GPU systems, test default CUDA allocations and CUDA memory-pool allocations separately:
  - Default CUDA allocations should be accepted by `LaunchArrayAccessMode.CHECKED` when peer access is enabled.
  - CUDA memory-pool allocations should be accepted by `LaunchArrayAccessMode.CHECKED` when memory-pool access is enabled, even if peer access is disabled.
  - CUDA memory-pool allocations should be rejected by `LaunchArrayAccessMode.CHECKED` when memory-pool access is disabled, even if peer access is enabled.

### Phase 2 tests (prefetch)

- On HMM / host-page-table ATS systems: prefetch a CPU array to GPU, launch a kernel, verify correctness.
- On systems without HMM / host-page-table ATS: calling `wp.prefetch()` should not raise (no-op or warning).
- Test stream ordering: prefetch then kernel on same stream, verify results.
- Test prefetch back to CPU: prefetch to GPU, then prefetch to CPU, verify CPU access.

### Phase 3 tests (auto-prefetch)

- Enable `warp.config.auto_prefetch`, launch cross-device kernel, verify correctness.
- Verify auto-prefetch is not issued on integrated GPUs (may require mocking or checking driver call counts).

### Phase 5 test coverage (managed allocator)

**Capability and allocator tests (run on all CUDA hardware):**
- Verify `is_managed_memory_supported` and `is_concurrent_managed_access_supported` are `bool` for CUDA devices and `False` for CPU devices.
- Verify `wp.CudaManagedAllocator()` satisfies the `Allocator` protocol and can be constructed without an active CUDA context.
- Verify direct `CudaManagedAllocator.allocate()` calls without an active CUDA context raise clearly.
- On CUDA devices with managed-memory support, allocate with `wp.CudaManagedAllocator()` through `wp.ScopedAllocator()` and verify the resulting array has `device == cuda_device`, `pinned == False`, and `array.memory_kind == wp.MemoryKind.CUDA_MANAGED`.
- On CUDA devices without managed-memory support, allocation through `wp.CudaManagedAllocator()` raises a clear `RuntimeError`.
- Verify one shared `wp.CudaManagedAllocator()` instance works through `wp.set_cuda_allocator()` across multiple managed-memory-capable CUDA devices.

**Managed access predicate tests:**
- Same-device managed arrays return `True` from `wp.can_access(device, array)`.
- CUDA devices with `is_managed_memory_supported` return `True` for managed arrays, including managed arrays associated with another CUDA device. These tests should not require `wp.set_peer_access_enabled()` or `wp.set_mempool_access_enabled()`.
- CPU access to a managed CUDA array follows `owner.is_concurrent_managed_access_supported or owner.is_gpu_memory_access_from_cpu_supported`.
- `wp.can_access("cpu", managed_array)` returns `False` on limited managed-memory systems.
- Array views follow the owner array's memory kind through `_ref`; indexed arrays must check both data arrays and index arrays for access.
- Externally wrapped managed CUDA pointers can be classified through CUDA pointer attributes.
- Externally wrapped or custom CUDA memory-pool pointers remain conservative for cross-device access when Warp lacks ownership metadata, even if their observed memory kind is `wp.MemoryKind.CUDA_MEMPOOL`.

**Launch and graph tests:**
- With `LaunchArrayAccessMode.CHECKED`, CUDA launches receiving managed arrays are accepted when the target CUDA device supports managed memory.
- With `LaunchArrayAccessMode.CHECKED`, CPU launches receiving managed CUDA arrays are accepted only when the owner device supports concurrent managed access or direct managed host access.
- Managed arrays allocated before CUDA graph capture can be used by captured kernels.
- `cudaMallocManaged` allocation is tested outside capture on managed-memory-capable CUDA devices.

**Interop tests:**
- Managed arrays expose `__cuda_array_interface__` and export DLPack as CUDA.
- `array.numpy()` returns the expected values through the existing copy-to-CPU path.
- `array.cptr()` raises because managed arrays are still CUDA arrays in Warp.
- CUDA IPC rejects managed arrays with a clear error.

### Phase 5 documentation

- `wp.CudaManagedAllocator` is included in the CUDA memory-management API reference.
- `wp.MemoryKind` and `array.memory_kind` documentation distinguish observed memory class from physical residency and accessibility.
- `docs/user_guide/execution_and_performance/memory_management.rst` includes scoped and global managed-allocation examples.
- `docs/user_guide/execution_and_performance/memory_management.rst` distinguishes standard Warp CUDA arrays from managed arrays allocated through `wp.CudaManagedAllocator()`.
- Managed arrays are documented as not promising initial physical residency. `wp.prefetch()` is the explicit placement hint once Phase 2 exists.
- Graph-capture behavior is documented: managed arrays may be used by captured kernels, but users should allocate managed arrays before capture.

### Phase 6 test coverage (allocator policy for persistent resources)

**Resource allocator-policy tests:**
- Audit meshes, hash grids, volumes, Fabric buffers, and other public resources
  that allocate persistent CUDA storage under the hood.
- For each resource classified as persistent, construct it under
  `wp.ScopedAllocator(device, wp.CudaManagedAllocator())` on managed-memory
  capable devices and verify its persistent buffers report
  `wp.MemoryKind.CUDA_MANAGED` where those buffers are exposed as Warp arrays or
  otherwise inspectable.
- Construct the same resources under CUDA device and CUDA memory-pool allocator
  policies and verify their persistent buffers follow the selected allocator.
- Verify algorithm temporary allocations used during construction, sorting,
  building, scanning, or reduction do not unexpectedly become managed memory
  merely because the user selected `CudaManagedAllocator`.
- Verify graph-capture behavior for resource constructors is explicit:
  preconstructed managed resources can be used by captured kernels when their
  backing buffers are accessible; resource construction that requires managed
  allocation during capture relies on CUDA's direct `cudaMallocManaged()`
  behavior in the Phase 5 backend.

### Phase 7 test coverage (unified allocator selection)

**Built-in allocator accessor tests:**
- Verify public built-in accessor functions return allocator objects compatible
  with `wp.ScopedAllocator`.
- Verify `wp.get_cuda_device_allocator(device)` creates ordinary CUDA device
  memory, `wp.get_cuda_mempool_allocator(device)` creates CUDA memory-pool
  memory when supported, and `wp.get_cuda_managed_allocator()` creates CUDA
  managed memory.
- Verify `ScopedMempool(device, True)` and
  `ScopedAllocator(device, wp.get_cuda_mempool_allocator(device))` produce the
  same allocation memory kind on devices with memory-pool support.
- Verify `ScopedMempool(device, False)` and
  `ScopedAllocator(device, wp.get_cuda_device_allocator(device))` produce the
  same allocation memory kind.
- Verify `ScopedMempool` remains supported as a compatibility API until a later
  release intentionally adds a deprecation warning.

### Phase 8 test coverage (managed memory pools)

**CUDA 13 managed-pool tests:**
- On CUDA 13.0+ builds and devices that support CUDA memory pools, allocate a managed array through `wp.CudaManagedAllocator()` before capture to initialize the managed pool, then allocate another managed array during CUDA graph capture and verify replayed kernels can read and write it.
- Verify capture-time managed allocation uses the stream-ordered path by capturing allocation, kernel launch, and free/release-sensitive behavior without requiring host synchronization inside capture.
- Verify the same `array.memory_kind == wp.MemoryKind.CUDA_MANAGED` value for direct and managed-pool allocations; no new public memory kind should appear.
- Verify managed-pool allocations free through the selected Phase 8 free-routing path and direct fallback allocations still free through the default `cudaFree()` path.

**Fallback and rejection tests:**
- On CUDA 12.x builds, including CUDA 12.9, capture-time managed allocation uses
  the direct `cudaMallocManaged()` path and may raise a CUDA runtime error.
- On CUDA 13.0+ builds where managed pool creation fails or the device lacks
  memory-pool support, allocation falls back to `cudaMallocManaged()` and CUDA
  reports any capture-time failure.
- For externally started CUDA captures, verify pre-initializing the pool before
  capture allows the allocation path when the device supports it; otherwise the
  direct fallback follows CUDA runtime behavior.

### Phase 6 documentation

- `docs/user_guide/execution_and_performance/memory_management.rst` should distinguish user-facing persistent
  resource allocations from internal temporary allocations.
- Resource documentation should say which persistent buffers honor
  `ScopedAllocator` and which construction-time workspaces remain internal.
- Resource-level `wp.can_access(device, resource)` docs should explain that the
  result is conservative and aggregates the resource's backing buffers.

### Phase 7 documentation

- `docs/user_guide/execution_and_performance/memory_management.rst` should describe `ScopedAllocator` as the
  preferred interface for selecting built-in CUDA allocators and custom
  allocators.
- The docs should introduce built-in allocator accessors using memory-behavior
  names such as `get_cuda_device_allocator`, `get_cuda_mempool_allocator`, and
  `get_cuda_managed_allocator`.
- `ScopedMempool` docs should point to the equivalent `ScopedAllocator` forms
  while preserving compatibility guidance.

### Phase 8 documentation

- `docs/user_guide/execution_and_performance/memory_management.rst` should explain that capture-time managed allocation is a CUDA 13+ managed-pool feature, while CUDA 12.x builds require pre-allocation before capture.
- The docs should state that `wp.CudaManagedAllocator()` remains the public allocator API and `wp.MemoryKind.CUDA_MANAGED` remains the observed memory kind for both direct and pool-backed managed allocations.
- Error messages should distinguish "managed allocation during capture is unsupported on this build/device" from "initialize the managed pool before capture."

### CI considerations

- The existing CI may not have HMM, ATS, Jetson Thor, or DGX Spark / GB10 hardware. Tests that require specific paradigms should use `unittest.skipUnless` based on the device attributes queried in Phase 1.
- Tests that only query attributes (Phase 1 / Phase 5 attributes and `Device.can_access()` / `wp.can_access()` invariant tests) should run on all hardware.
- Phase 8 capture-allocation tests should skip unless the build uses CUDA 13.0+ and the target device reports memory-pool support; CUDA 12.x CI should continue to run the capture-time rejection tests.
- Consider adding a CI label or tag for "unified memory" tests so they can be selectively run on appropriate hardware.

### Device compatibility matrix for test expectations

| Test scenario | Discrete (no HMM) | Discrete (HMM) | Host-page-table ATS with direct managed host access | Jetson Orin / limited Tegra | Jetson Thor | DGX Spark / GB10 observed |
|---|---|---|---|---|---|---|
| GPU can access CPU arrays | No | Yes | Yes | No | Yes | Yes |
| CPU can access Warp default GPU arrays | No | No | No | No | No | No |
| CPU direct access to GPU-resident CUDA managed memory | No | No | Yes | No | No | No |
| `wp.can_access(cpu, CudaManagedAllocator array)` | Yes if concurrent managed access or direct managed host access | Yes | Yes | No on limited systems | Yes | Yes |
| CUDA can access `CudaManagedAllocator` array | Yes if managed memory supported | Yes | Yes | Yes if managed memory supported | Yes | Yes |
| Managed allocation during graph capture (Phase 5) | Not supported | Not supported | Not supported | Not supported | Not supported | Not supported |
| Managed-pool allocation during graph capture (Phase 8) | CUDA 13+ managed-pool support required | CUDA 13+ managed-pool support required | CUDA 13+ managed-pool support required | CUDA 13+ managed-pool support required | CUDA 13+ managed-pool support required | CUDA 13+ managed-pool support required |
| Native CPU-GPU atomic hardware capability | No | No | Yes | Device-dependent | Reports yes | Reports yes |
| Current Warp CPU/GPU `wp.atomic_*` overlap | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported | Unsupported |
| Cross-device launch GPU->CPU array (`RELAXED`) | CUDA fault | OK | OK | CUDA fault | OK | OK |
| Cross-device launch CPU->GPU array (`RELAXED`) | Segfault | Segfault | Segfault for Warp default arrays | Segfault | Segfault | Segfault for Warp default arrays |
| Cross-device launch GPU->CPU array (`STRICT`) | RuntimeError | RuntimeError | RuntimeError | RuntimeError | RuntimeError | RuntimeError |
| Cross-device launch CPU->GPU array (`STRICT`) | RuntimeError | RuntimeError | RuntimeError | RuntimeError | RuntimeError | RuntimeError |
| Cross-device launch GPU->CPU array (`CHECKED`) | RuntimeError | OK | OK | RuntimeError | OK | OK |
| Cross-device launch CPU->GPU array (`CHECKED`) | RuntimeError | RuntimeError | RuntimeError for Warp default arrays | RuntimeError | RuntimeError | RuntimeError for Warp default arrays |
| `wp.prefetch()` for CPU arrays | No-op / warning | Yes (SW) | Yes (HW) | No-op / warning | Accepted; low expected benefit on integrated DRAM | Yes (HW) |
