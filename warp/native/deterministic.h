// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Note: this header is included in NVRTC-compiled code, so we cannot use
// standard library headers like <cstdint>.  The int64_t type is provided
// by the Warp builtin headers.

// Deterministic mode scatter buffer support.
//
// When deterministic mode is enabled, floating-point atomic operations are
// redirected to scatter buffers during kernel execution. After the kernel
// completes, the runtime sorts the buffer by (dest_index, thread_id) and
// reduces values in that fixed order, guaranteeing bit-exact reproducibility.

namespace wp {
struct det_ctx {
    int phase;
    int debug;
    size_t idx;
    int* overflow;
};

template <typename T> struct det_scatter_buf {
    int64_t* keys;
    T* vals;
    int* count;
    int capacity;
};

struct det_counter_buf {
    int* contrib;
    int* prefix;
};

namespace deterministic {

// Device-side function called from generated kernel code in deterministic mode.
// Writes a (key, value) record to the scatter buffer.
//
// The sort key packs the destination flat index in the upper 32 bits and the
// thread_id (_idx from the grid-stride loop) in the lower 32 bits.  After a
// 64-bit radix sort, records targeting the same destination are grouped
// together and ordered by thread ID, giving a deterministic reduction order.
template <typename T>
inline CUDA_CALLABLE void scatter(det_ctx& ctx, det_scatter_buf<T>& buf, int dest_flat_idx, T value)
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

}  // namespace deterministic
}  // namespace wp
