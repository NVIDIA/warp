// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Kernel-side ring-buffer logging.
//
// All static metadata (log level, message text, source file, line number) is
// encoded as a compile-time integer key by the Python code-generator and stored
// in a Python-side call-site table.  Only that key and an optional numeric
// payload are written into the ring buffer at runtime, keeping per-record
// overhead to one atomic increment and two 32/64-bit writes.
//
// This file is included from builtin.h inside namespace wp {} and must remain
// independently compilable with WP_NO_CRT defined.

// Log payload type tags (mirror of Python WP_LOG_PAYLOAD_* constants)
#define WP_LOG_PAYLOAD_NONE 0
#define WP_LOG_PAYLOAD_I32  1
#define WP_LOG_PAYLOAD_I64  2
#define WP_LOG_PAYLOAD_F32  3

// One 16-byte ring-buffer entry.  The named union WpLogPayload allows us to
// pass it by value in the helper functions without referencing an anonymous
// inner union type (which is not permitted in C++).
union WpLogPayload {
    int32_t i32;
    int64_t i64;
    float f32;
    double _pad;  // ensures the union is 8 bytes on all platforms
};

struct WpLogEntry {
    uint32_t call_site_key;  // index into runtime.log_call_sites (Python side)
    int32_t payload_type;  // WP_LOG_PAYLOAD_* tag
    WpLogPayload payload;  // 8 bytes
};  // total: 16 bytes

// Buffer header immediately followed in memory by `capacity` WpLogEntry slots.
struct WpLogBuffer {
    uint32_t write_idx;  // atomic write cursor (also total attempted writes)
    uint32_t capacity;  // number of WpLogEntry slots that fit after the header
    uint32_t overflow_count;  // records dropped due to a full buffer (atomic)
    uint32_t _pad;
    WpLogEntry entries[1];  // variable-length tail; allocate with extra space
};  // header is 16 bytes; each entry is 16 bytes

// --- Internal helpers -------------------------------------------------------

// Claims a slot and returns the index.  CPU kernels run sequentially so the
// non-atomic fallback in atomic_add() is safe there.
CUDA_CALLABLE inline uint32_t _wp_log_claim(WpLogBuffer* buf) { return atomic_add(&buf->write_idx, 1u); }

// Validates buf, atomically claims a slot, and writes its index to *out_idx.
// Returns false (and increments overflow_count) when the buffer is full or NULL.
CUDA_CALLABLE inline bool _wp_log_try_claim(WpLogBuffer* buf, uint32_t* out_idx)
{
    if (!buf)
        return false;
    uint32_t idx = _wp_log_claim(buf);
    if (idx >= buf->capacity) {
        atomic_add(&buf->overflow_count, 1u);
        return false;
    }
    *out_idx = idx;
    return true;
}

// --- Public write functions (generated kernel code calls these) -------------

CUDA_CALLABLE inline void wp_log_write_nop(WpLogBuffer* buf, uint32_t key)
{
    uint32_t idx;
    if (!_wp_log_try_claim(buf, &idx))
        return;
    buf->entries[idx].call_site_key = key;
    buf->entries[idx].payload_type = WP_LOG_PAYLOAD_NONE;
    buf->entries[idx].payload.i64 = 0;
}

CUDA_CALLABLE inline void wp_log_write_i32(WpLogBuffer* buf, uint32_t key, int32_t value)
{
    uint32_t idx;
    if (!_wp_log_try_claim(buf, &idx))
        return;
    buf->entries[idx].call_site_key = key;
    buf->entries[idx].payload_type = WP_LOG_PAYLOAD_I32;
    buf->entries[idx].payload.i32 = value;
}

CUDA_CALLABLE inline void wp_log_write_i64(WpLogBuffer* buf, uint32_t key, int64_t value)
{
    uint32_t idx;
    if (!_wp_log_try_claim(buf, &idx))
        return;
    buf->entries[idx].call_site_key = key;
    buf->entries[idx].payload_type = WP_LOG_PAYLOAD_I64;
    buf->entries[idx].payload.i64 = value;
}

CUDA_CALLABLE inline void wp_log_write_f32(WpLogBuffer* buf, uint32_t key, float value)
{
    uint32_t idx;
    if (!_wp_log_try_claim(buf, &idx))
        return;
    buf->entries[idx].call_site_key = key;
    buf->entries[idx].payload_type = WP_LOG_PAYLOAD_F32;
    buf->entries[idx].payload.f32 = value;
}
