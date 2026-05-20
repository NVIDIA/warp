// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Note: this header is included in NVRTC-compiled code, so we cannot use
// standard library headers like <cstdint>.  The int64_t type is provided
// by the Warp builtin headers.

// Deterministic mode scatter buffer support.
//
// When deterministic mode is enabled, supported atomic operations are
// redirected to temporary buffers. Accumulation atomics are sorted by
// (dest_index, thread_id) and reduced in that fixed order. Consumed-return
// counter atomics are sorted by (counter_index, deterministic_record_order)
// and prefix-scanned before the execution replay.

namespace wp {
template <typename T> struct det_scatter_buf_t {
    int64_t* keys;
    T* vals;
    int* count;
    int capacity;
};

struct det_counter_buf_t {
    int64_t* keys;
    int* values;
    int* prefixes;
    int* record_slots;
    int* cursors;
    int* count;
    int capacity;
    int records_per_thread;
};

struct det_ctx {
    int phase;
    int debug;
    size_t idx;
    int* overflow;
    uint64_t* counter_target_ptrs;
    int* counter_target_sizes;
    int counter_target_count;
};

namespace deterministic {

// Device-side function called from generated kernel code in deterministic mode.
// Writes a (key, value) record to the scatter buffer.
//
// The sort key packs the destination flat index in the upper 32 bits and the
// thread_id (_idx from the grid-stride loop) in the lower 32 bits. Launches
// requiring more than 2^31 - 1 records are rejected by the Python launcher
// before this path is used. After a 64-bit radix sort, records targeting the same
// destination are grouped together and ordered by thread ID, giving a
// deterministic reduction order.
template <typename T>
inline CUDA_CALLABLE void scatter(det_ctx& ctx, det_scatter_buf_t<T>& buf, int dest_flat_idx, T value)
{
#ifdef __CUDA_ARCH__
    int slot = atomicAdd(buf.count, 1);
    if (slot < buf.capacity) {
        buf.keys[slot] = (static_cast<int64_t>(dest_flat_idx) << 32)
            | static_cast<int64_t>(static_cast<unsigned int>(ctx.idx & 0xFFFFFFFFu));
        buf.vals[slot] = value;
    } else if (ctx.overflow != nullptr) {
        int prev = atomicCAS(ctx.overflow, 0, 1);
        if (ctx.debug && prev == 0) {
            printf("Warp deterministic scatter overflow: capacity=%d dest=%d\n", buf.capacity, dest_flat_idx);
        }
    }
#else
    // CPU path: direct accumulation (CPU kernels are sequential).
    (void)ctx;
    (void)buf;
    (void)dest_flat_idx;
    (void)value;
#endif
}

inline CUDA_CALLABLE int counter_add(det_ctx& ctx, det_counter_buf_t& buf, int dest_flat_idx, int value)
{
#ifdef __CUDA_ARCH__
    int cursor = atomicAdd(&buf.cursors[ctx.idx], 1);
    if (cursor >= buf.records_per_thread) {
        if (ctx.overflow != nullptr) {
            int prev = atomicCAS(ctx.overflow, 0, 1);
            if (ctx.debug && prev == 0) {
                printf(
                    "Warp deterministic counter overflow: records_per_thread=%d dest=%d\n", buf.records_per_thread,
                    dest_flat_idx
                );
            }
        }
        return 0;
    }

    int64_t record_ordinal64 = static_cast<int64_t>(ctx.idx) * buf.records_per_thread + cursor;
    if (record_ordinal64 < 0 || record_ordinal64 >= buf.capacity) {
        if (ctx.overflow != nullptr) {
            int prev = atomicCAS(ctx.overflow, 0, 1);
            if (ctx.debug && prev == 0) {
                printf("Warp deterministic counter overflow: capacity=%d dest=%d\n", buf.capacity, dest_flat_idx);
            }
        }
        return 0;
    }

    int record_ordinal = static_cast<int>(record_ordinal64);

    if (ctx.phase == 0) {
        int slot = atomicAdd(buf.count, 1);
        if (slot < buf.capacity) {
            buf.record_slots[record_ordinal] = slot;
            buf.keys[slot] = (static_cast<int64_t>(dest_flat_idx) << 32)
                | static_cast<int64_t>(static_cast<unsigned int>(record_ordinal));
            buf.values[slot] = value;
        } else if (ctx.overflow != nullptr) {
            int prev = atomicCAS(ctx.overflow, 0, 1);
            if (ctx.debug && prev == 0) {
                printf("Warp deterministic counter overflow: capacity=%d dest=%d\n", buf.capacity, dest_flat_idx);
            }
        }
        return 0;
    }

    // Return the base slot for this reservation. Values greater than one reserve
    // the contiguous range [slot, slot + value); caller code writes each element.
    int slot = buf.record_slots[record_ordinal];
    if (slot < 0 || slot >= buf.capacity) {
        return 0;
    }
    return buf.prefixes[slot];
#else
    (void)ctx;
    (void)buf;
    (void)dest_flat_idx;
    (void)value;
    return 0;
#endif
}

inline CUDA_CALLABLE bool is_counter_store_target(det_ctx& ctx, const void* ptr)
{
#ifdef __CUDA_ARCH__
    if (ctx.counter_target_ptrs == nullptr || ctx.counter_target_sizes == nullptr) {
        return false;
    }

    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    for (int i = 0; i < ctx.counter_target_count; ++i) {
        uint64_t start = ctx.counter_target_ptrs[i];
        int size = ctx.counter_target_sizes[i];
        if (start == 0 || size <= 0) {
            continue;
        }

        uint64_t end = start + static_cast<uint64_t>(size) * sizeof(int);
        if (addr >= start && addr < end) {
            return true;
        }
    }
#else
    (void)ctx;
    (void)ptr;
#endif
    return false;
}

inline CUDA_CALLABLE bool is_global_store_target(const void* ptr)
{
#ifdef __CUDA_ARCH__
#if defined(__clang__)
    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    unsigned int result = 0;
    asm volatile("{ .reg .pred p; isspacep.global p, %1; selp.u32 %0, 1, 0, p; }" : "=r"(result) : "l"(addr));
    return result != 0;
#else
    return __isGlobal(ptr) != 0;
#endif
#else
    (void)ptr;
    return true;
#endif
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store_if_active(det_ctx& ctx, const A<T>& buf, int i, T value)
{
    T* ptr = wp::address(buf, i);
    if (ctx.phase != 0 || !is_global_store_target(ptr) || is_counter_store_target(ctx, ptr)) {
        wp::array_store(buf, i, value);
    }
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store_if_active(det_ctx& ctx, const A<T>& buf, int i, int j, T value)
{
    T* ptr = wp::address(buf, i, j);
    if (ctx.phase != 0 || !is_global_store_target(ptr) || is_counter_store_target(ctx, ptr)) {
        wp::array_store(buf, i, j, value);
    }
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store_if_active(det_ctx& ctx, const A<T>& buf, int i, int j, int k, T value)
{
    T* ptr = wp::address(buf, i, j, k);
    if (ctx.phase != 0 || !is_global_store_target(ptr) || is_counter_store_target(ctx, ptr)) {
        wp::array_store(buf, i, j, k, value);
    }
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store_if_active(det_ctx& ctx, const A<T>& buf, int i, int j, int k, int l, T value)
{
    T* ptr = wp::address(buf, i, j, k, l);
    if (ctx.phase != 0 || !is_global_store_target(ptr) || is_counter_store_target(ctx, ptr)) {
        wp::array_store(buf, i, j, k, l, value);
    }
}

}  // namespace deterministic
}  // namespace wp

#ifdef __CUDA_ARCH__
#define WP_DET_SCATTER_OR_FALLBACK(det_ctx, helper, flat_idx, value, cpu_expr) \
    do { \
        if ((det_ctx).phase != 0 && (helper).count != nullptr) { \
            wp::deterministic::scatter((det_ctx), (helper), static_cast<int>(flat_idx), (value)); \
        } \
    } while (0)

#define WP_DET_COUNTER_OR_FALLBACK(out, det_ctx, helper, flat_idx, value, cpu_expr) \
    do { \
        (out) = ((helper).count != nullptr) \
            ? wp::deterministic::counter_add((det_ctx), (helper), static_cast<int>(flat_idx), (value)) \
            : 0; \
    } while (0)

#define WP_DET_STORE_IF_ACTIVE(det_ctx, ...) \
    do { \
        wp::deterministic::array_store_if_active((det_ctx), __VA_ARGS__); \
    } while (0)

#define WP_DET_SLOT_STORE_IF_ACTIVE(det_ctx, slot_lvalue, value_expr) \
    do { \
        auto& _wp_det_slot = (slot_lvalue); \
        if ((det_ctx).phase != 0 \
            || !wp::deterministic::is_global_store_target(&_wp_det_slot) \
            || wp::deterministic::is_counter_store_target((det_ctx), &_wp_det_slot)) \
        { \
            _wp_det_slot = (value_expr); \
        } \
    } while (0)

#define WP_DET_SIDE_EFFECT_IF_ACTIVE(det_ctx, ...) \
    do { \
        if ((det_ctx).phase != 0) { \
            __VA_ARGS__ \
        } \
    } while (0)
#else
#define WP_DET_SCATTER_OR_FALLBACK(det_ctx, helper, flat_idx, value, cpu_expr) \
    do { \
        (void)(det_ctx); \
        (void)(helper); \
        cpu_expr; \
    } while (0)

#define WP_DET_COUNTER_OR_FALLBACK(out, det_ctx, helper, flat_idx, value, cpu_expr) \
    do { \
        (void)(det_ctx); \
        (void)(helper); \
        (void)(flat_idx); \
        (void)(value); \
        cpu_expr; \
    } while (0)

#define WP_DET_STORE_IF_ACTIVE(det_ctx, ...) \
    do { \
        (void)(det_ctx); \
        wp::array_store(__VA_ARGS__); \
    } while (0)

#define WP_DET_SLOT_STORE_IF_ACTIVE(det_ctx, slot_lvalue, value_expr) \
    do { \
        (void)(det_ctx); \
        (slot_lvalue) = (value_expr); \
    } while (0)

#define WP_DET_SIDE_EFFECT_IF_ACTIVE(det_ctx, ...) \
    do { \
        (void)(det_ctx); \
        __VA_ARGS__ \
    } while (0)
#endif
