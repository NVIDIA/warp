// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

enum AllocKind {
    ALLOC_KIND_HOST = 0,
    ALLOC_KIND_PINNED = 1,
    ALLOC_KIND_DEVICE = 2,
};

struct AllocRecord {
    size_t size;
    size_t seq;  // allocation sequence number (monotonically increasing)
    AllocKind kind;
    int device_ordinal;  // -1 for host/pinned
    std::string tag;  // Python call-site info or "(native)"
    std::string scope;  // e.g. "simulation/collision"
};

class AllocTracker {
public:
    std::atomic<bool> enabled { false };

    void record_alloc(void* ptr, size_t size, AllocKind kind, int ordinal = -1, const char* label = nullptr);
    void record_free(void* ptr);

    void reset();

    void push_scope(const char* name);
    void pop_scope();

    // Associate a tag with an existing live allocation identified by pointer.
    void set_tag(void* ptr, const char* tag);

    // Returns a pointer to an internal buffer.  The caller must copy the
    // string before the next call to report() from any thread.
    // sort_order: 0 = by size descending (default), 1 = chronological (oldest first)
    // max_items: maximum number of individual allocations shown per category
    const char* report(int sort_order = 0, int max_items = 10);

    size_t get_current_bytes();
    size_t get_peak_bytes();
    size_t get_total_alloc_count();
    size_t get_total_alloc_bytes();
    int get_live_count();

private:
    std::string build_scope_path();

    mutable std::mutex m_mutex;
    std::unordered_map<void*, AllocRecord> m_live;

    size_t m_total_count = 0;
    size_t m_total_bytes = 0;
    size_t m_current_bytes = 0;
    size_t m_peak_bytes = 0;

    // Per-kind current bytes for breakdown
    size_t m_current_host = 0;
    size_t m_current_pinned = 0;
    size_t m_current_device = 0;

    struct DeviceTotals {
        size_t total_count = 0;
        size_t total_bytes = 0;
        size_t current_bytes = 0;
        size_t peak_bytes = 0;
    };
    std::map<int, DeviceTotals> m_device_totals;  // ordinal -> totals (device GPU)
    DeviceTotals m_host_totals;
    DeviceTotals m_pinned_totals;

    struct ScopeKey {
        std::string scope;
        AllocKind kind;
        int ordinal;  // only meaningful for ALLOC_KIND_DEVICE

        bool operator==(const ScopeKey& o) const { return scope == o.scope && kind == o.kind && ordinal == o.ordinal; }
    };

    struct ScopeKeyHash {
        size_t operator()(const ScopeKey& k) const
        {
            size_t h = std::hash<std::string>()(k.scope);
            h ^= std::hash<int>()(static_cast<int>(k.kind)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.ordinal) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    // Scope breakdown: (scope, kind, ordinal) -> (count, bytes)
    std::unordered_map<ScopeKey, std::pair<size_t, size_t>, ScopeKeyHash> m_scope_stats;

    // Buffer for report() return value
    std::string m_report_buf;
};

extern AllocTracker g_alloc_tracker;
