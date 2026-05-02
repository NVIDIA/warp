// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// APIC Format Constants
// =============================================================================

#define APIC_FORMAT_VERSION 3
#define APIC_MAGIC "WRP1"
#define APIC_MAGIC_VALUE 0x31505257  // "WRP1" as little-endian uint32

// Maximum array dimensions (matches Warp's ARRAY_MAX_DIMS)
#define APIC_MAX_DIMS 4

// Maximum launch dimensions (must match LAUNCH_MAX_DIMS in builtin.h)
#define APIC_LAUNCH_MAX_DIMS 4

// =============================================================================
// Enums
// =============================================================================

// Operation types
enum APICOpType : uint32_t {
    APIC_OP_KERNEL_LAUNCH = 1,
    APIC_OP_MEMCPY_H2D = 2,
    APIC_OP_MEMCPY_D2H = 3,  // Reserved for future
    APIC_OP_MEMCPY_D2D = 4,
    APIC_OP_MEMSET = 5,
    APIC_OP_ALLOC = 6,  // In-graph allocation
};

// =============================================================================
// WRP File Header
// =============================================================================

// All serialization structs are packed to ensure binary compatibility with Python
#pragma pack(push, 1)

struct APICFileHeader {
    uint8_t magic[4];  // "WRP1" -- Warp Recorded Program
    uint32_t version;  // APIC_FORMAT_VERSION
    uint32_t flags;  // Reserved flags
    uint32_t num_sections;  // Number of sections
    uint64_t section_table_offset;  // Offset to section table
    uint8_t device_type;  // APICDeviceType value (reserved for future; see wp_apic_load_graph)
    uint8_t _reserved_dt[3];
    uint32_t target_arch;  // CUDA SM version (e.g., 86) or 0 (CPU)
    uint32_t _reserved[8];  // Reserved for future use
};  // 64 bytes

// Device types (runtime APICGraphInternal::device_type and the wire-format
// byte APICFileHeader::device_type use the same integer values).
enum APICDeviceType : uint32_t {
    APIC_DEVICE_CUDA = 0,
    APIC_DEVICE_CPU = 1,
};

// Section types
enum APICSectionType : uint32_t {
    APIC_SECTION_METADATA = 0x01,
    APIC_SECTION_MEMORY = 0x02,
    APIC_SECTION_OPERATIONS = 0x03,
};

struct APICSectionEntry {
    APICSectionType type;
    uint32_t flags;  // Section flags
    uint64_t offset;  // Offset from file start
    uint64_t size;  // Section size (compressed)
    uint64_t uncompressed_size;  // Uncompressed size
};  // 32 bytes

// =============================================================================
// Mesh Serialization Records
// =============================================================================

struct APICMeshRecord {
    int32_t num_points;
    int32_t num_tris;
    uint8_t support_winding_number;
    uint8_t bvh_constructor;
    uint16_t bvh_leaf_size;
    uint32_t points_region_id;
    uint32_t indices_region_id;
    uint32_t velocities_region_id;  // 0 if absent
    uint64_t original_ptr;
};  // 32 bytes

struct APICPtrLocationRecord {
    uint32_t region_id;
    uint32_t _pad;
    uint64_t offset;
    uint64_t stride;  // 0 = single pointer
};  // 24 bytes

// =============================================================================
// Operation Records
// =============================================================================

// Common header for all operations
struct APICOpHeader {
    APICOpType op_type;
    uint32_t total_size;  // Total bytes including header and variable data
};  // 8 bytes

// Kernel launch record (fixed part)
// Variable data follows: kernel_key, module_hash, param_bindings[]
struct APICLaunchRecord {
    APICOpHeader header;  // op_type = APIC_OP_KERNEL_LAUNCH

    // Launch bounds (embedded)
    int32_t shape[APIC_LAUNCH_MAX_DIMS];  // Launch shape
    int32_t ndim;  // Number of dimensions
    uint64_t size;  // Total threads

    // Launch parameters
    uint64_t dim;  // Total threads
    int32_t max_blocks;  // Maximum blocks
    int32_t block_dim;  // Threads per block
    int32_t smem_bytes;  // Shared memory bytes
    uint8_t is_forward;  // 1 for forward pass, 0 for backward
    uint8_t _pad1[3];

    // Variable data sizes
    uint16_t kernel_key_len;  // Length of kernel_key string
    uint16_t module_hash_len;  // Length of module_hash string
    uint16_t num_params;  // Number of parameter bindings
    uint16_t num_handle_offsets;  // Number of handle byte offsets

    // Variable data follows in order:
    // 1. char kernel_key[kernel_key_len]
    // 2. char module_hash[module_hash_len]
    // 3. APICLaunchParamRecord[num_params]
    // 4. uint32_t handle_offsets[num_handle_offsets]
};

// One entry per kernel argument (1-based; param_index = 0 is reserved for
// launch_bounds). Arrays and scalars share the same record layout; the first
// byte selects the interpretation.
//
// Arrays: (region_id, byte_offset) locates the array's data inside a captured
// memory region, and shape[] / strides[] / ndim / element_size carry the
// per-launch array_t view. Shape and strides are stored alongside the region
// pointer because the region captures only the underlying data — the array_t
// descriptor itself is built on each launch from Python and is not part of
// any captured region, so it cannot be recovered at replay time unless it is
// recorded here.
//
// Scalars: is_array=0. byte_offset holds the scalar size in bytes, and the
// scalar value itself is inlined into shape[] (first 32 B) and strides[]
// (next 32 B). This reuses the 64-byte payload rather than adding a second
// record shape; scalars larger than 64 B are rejected at capture time.
struct APICLaunchParamRecord {
    uint8_t is_array;  // 1 for array, 0 for scalar
    uint8_t ndim;  // Number of dimensions (arrays only)
    uint16_t param_index;  // Parameter index in kernel signature (1-based)
    int32_t region_id;  // Memory region ID (-1 for null array or scalar)
    uint64_t byte_offset;  // Byte offset within region (arrays) or scalar_size (scalars)
    int64_t shape[APIC_MAX_DIMS];  // Array shape or first 32 bytes of scalar value
    int64_t strides[APIC_MAX_DIMS];  // Array strides or next 32 bytes of scalar value
    uint32_t element_size;  // Element size in bytes (arrays only)
    uint32_t _pad1;
};  // 88 bytes

// Memcpy Host-to-Device (variable: has inline data)
struct APICMemcpyH2DRecord {
    APICOpHeader header;  // op_type = APIC_OP_MEMCPY_H2D
    int32_t dst_region_id;
    uint32_t _pad;
    uint64_t dst_offset;
    uint64_t size;
    // uint8_t data[size] follows
};  // 32 bytes fixed

// Memcpy Device-to-Device (fixed size)
struct APICMemcpyD2DRecord {
    APICOpHeader header;  // op_type = APIC_OP_MEMCPY_D2D
    int32_t dst_region_id;
    int32_t src_region_id;
    uint64_t dst_offset;
    uint64_t src_offset;
    uint64_t size;
};  // 40 bytes

// Memset (fixed size)
struct APICMemsetRecord {
    APICOpHeader header;  // op_type = APIC_OP_MEMSET
    int32_t region_id;
    int32_t value;
    uint64_t offset;
    uint64_t size;
};  // 32 bytes

// In-graph allocation (fixed size)
struct APICAllocRecord {
    APICOpHeader header;  // op_type = APIC_OP_ALLOC
    int32_t region_id;
    uint32_t _pad;
    uint64_t size;
};  // 24 bytes

// =============================================================================
// Memory Section Records
// =============================================================================

struct APICMemoryRegionRecord {
    uint32_t region_id;
    uint32_t element_size;
    uint64_t size;  // Size in bytes
    uint8_t has_initial_data;  // 1 if initial_data follows
    uint8_t _pad[7];
    // If has_initial_data: uint8_t initial_data[size] follows
};  // 24 bytes fixed

#pragma pack(pop)

// =============================================================================
// Recording API Structures (passed from Python to C++, naturally aligned)
// =============================================================================

// Launch info passed to wp_cuda_launch_kernel() / wp_cpu_launch_kernel()
// Not packed — contains pointers that must be naturally aligned.
struct APICLaunchInfo {
    const char* kernel_key;  // Kernel identifier string
    const char* module_hash;  // Module hash string
    uint8_t is_forward;  // 1 for forward, 0 for backward
    uint8_t _pad[7];  // Align params to 8 bytes
    const APICLaunchParamRecord* params;  // Array of parameter bindings
    int32_t num_params;  // Number of parameter bindings
};

// =============================================================================
// Execution Structures (must match runtime types in builtin.h and array.h)
// =============================================================================

// Array descriptor - must match layout of array_t<T> in array.h
struct apic_array_t {
    uint64_t data;  // Device/host pointer
    uint64_t grad;  // Gradient pointer (usually 0)
    int shape[APIC_MAX_DIMS];
    int strides[APIC_MAX_DIMS];
    uint16_t ndim;
    uint16_t flags;
};

#ifdef __cplusplus
}
#endif
