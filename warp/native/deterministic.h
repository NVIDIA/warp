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
namespace deterministic {

// Device-side function called from generated kernel code in deterministic mode.
// Writes a (key, value) record to the scatter buffer.
//
// The sort key packs the destination flat index in the upper 32 bits and the
// thread_id (_idx from the grid-stride loop) in the lower 32 bits.  After a
// 64-bit radix sort, records targeting the same destination are grouped
// together and ordered by thread ID, giving a deterministic reduction order.
template <typename T>
inline CUDA_CALLABLE void scatter(
    int64_t* keys,
    T* values,
    int* counter,
    int* overflow,
    int capacity,
    int debug_enabled,
    int dest_flat_idx,
    size_t thread_id,
    T value
)
{
#ifdef __CUDA_ARCH__
    int slot = atomicAdd(counter, 1);
    if (slot < capacity) {
        keys[slot] = (static_cast<int64_t>(dest_flat_idx) << 32)
            | static_cast<int64_t>(static_cast<unsigned int>(thread_id & 0xFFFFFFFFu));
        values[slot] = value;
    } else if (overflow != nullptr) {
        int prev = atomicCAS(overflow, 0, 1);
        if (debug_enabled && prev == 0) {
            printf("Warp deterministic scatter overflow: capacity=%d dest=%d\n", capacity, dest_flat_idx);
        }
    }
#else
    // CPU path: direct accumulation (CPU kernels are sequential).
    (void)keys;
    (void)values;
    (void)counter;
    (void)overflow;
    (void)capacity;
    (void)debug_enabled;
    (void)dest_flat_idx;
    (void)thread_id;
    (void)value;
#endif
}

}  // namespace deterministic
}  // namespace wp
