/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "alloc_tracker.h"
#include "crt.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <sstream>

AllocTracker g_alloc_tracker;

static thread_local std::vector<std::string> tl_scope_stack;

void AllocTracker::record_alloc(void* ptr, size_t size, AllocKind kind, int ordinal, const char* label)
{
    if (!ptr)
        return;

    AllocRecord rec;
    rec.size = size;
    rec.kind = kind;
    rec.device_ordinal = ordinal;
    rec.scope = build_scope_path();
    rec.tag = label ? label : "(native)";

    std::lock_guard<std::mutex> lock(m_mutex);

    rec.seq = m_total_count;

    // Handle pointer reuse (e.g. stream-ordered mempool recycling the same address).
    auto existing = m_live.find(ptr);
    if (existing != m_live.end()) {
        size_t old_size = existing->second.size;
        m_current_bytes -= old_size;
        switch (existing->second.kind) {
        case ALLOC_KIND_HOST:
            m_current_host -= old_size;
            m_host_totals.current_bytes -= old_size;
            break;
        case ALLOC_KIND_PINNED:
            m_current_pinned -= old_size;
            m_pinned_totals.current_bytes -= old_size;
            break;
        case ALLOC_KIND_DEVICE:
            m_current_device -= old_size;
            m_device_totals[existing->second.device_ordinal].current_bytes -= old_size;
            break;
        }
    }

    m_total_count++;
    m_total_bytes += size;
    m_current_bytes += size;
    if (m_current_bytes > m_peak_bytes)
        m_peak_bytes = m_current_bytes;

    DeviceTotals* dt = nullptr;
    switch (kind) {
    case ALLOC_KIND_HOST:
        m_current_host += size;
        dt = &m_host_totals;
        break;
    case ALLOC_KIND_PINNED:
        m_current_pinned += size;
        dt = &m_pinned_totals;
        break;
    case ALLOC_KIND_DEVICE:
        m_current_device += size;
        dt = &m_device_totals[ordinal];
        break;
    }
    dt->total_count++;
    dt->total_bytes += size;
    dt->current_bytes += size;
    if (dt->current_bytes > dt->peak_bytes)
        dt->peak_bytes = dt->current_bytes;

    if (!rec.scope.empty()) {
        ScopeKey key { rec.scope, kind, (kind == ALLOC_KIND_DEVICE) ? ordinal : -1 };
        auto& stats = m_scope_stats[key];
        stats.first++;
        stats.second += size;
    }

    m_live[ptr] = std::move(rec);
}

void AllocTracker::record_free(void* ptr)
{
    if (!ptr)
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_live.find(ptr);
    if (it == m_live.end())
        return;

    size_t size = it->second.size;
    AllocKind kind = it->second.kind;

    m_current_bytes -= size;
    switch (kind) {
    case ALLOC_KIND_HOST:
        m_current_host -= size;
        m_host_totals.current_bytes -= size;
        break;
    case ALLOC_KIND_PINNED:
        m_current_pinned -= size;
        m_pinned_totals.current_bytes -= size;
        break;
    case ALLOC_KIND_DEVICE:
        m_current_device -= size;
        m_device_totals[it->second.device_ordinal].current_bytes -= size;
        break;
    }

    m_live.erase(it);
}

void AllocTracker::reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_live.clear();
    m_scope_stats.clear();
    m_total_count = 0;
    m_total_bytes = 0;
    m_current_bytes = 0;
    m_peak_bytes = 0;
    m_current_host = 0;
    m_current_pinned = 0;
    m_current_device = 0;
    m_device_totals.clear();
    m_host_totals = {};
    m_pinned_totals = {};
}

void AllocTracker::push_scope(const char* name) { tl_scope_stack.emplace_back(name); }

void AllocTracker::pop_scope()
{
    if (!tl_scope_stack.empty())
        tl_scope_stack.pop_back();
}

void AllocTracker::set_tag(void* ptr, const char* tag)
{
    if (!ptr || !tag)
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_live.find(ptr);
    if (it != m_live.end())
        it->second.tag = tag;
}

static std::string format_bytes(size_t bytes)
{
    char buf[64];
    if (bytes >= (size_t(1) << 30))
        snprintf(buf, sizeof(buf), "%.2f GB", (double)bytes / (size_t(1) << 30));
    else if (bytes >= (size_t(1) << 20))
        snprintf(buf, sizeof(buf), "%.2f MB", (double)bytes / (size_t(1) << 20));
    else if (bytes >= (size_t(1) << 10))
        snprintf(buf, sizeof(buf), "%.2f KB", (double)bytes / (size_t(1) << 10));
    else
        snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)bytes);
    return buf;
}

const char* AllocTracker::report(int sort_order, int max_items)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    std::ostringstream os;
    os << "Allocation Tracking Report\n";

    // Collect live allocation counts per device from m_live.
    struct LiveStats {
        size_t count = 0;
        size_t bytes = 0;
    };
    std::map<int, LiveStats> live_gpu_stats;
    LiveStats live_host_stats, live_pinned_stats;

    for (auto& kv : m_live) {
        const AllocRecord& rec = kv.second;
        switch (rec.kind) {
        case ALLOC_KIND_DEVICE:
            live_gpu_stats[rec.device_ordinal].count++;
            live_gpu_stats[rec.device_ordinal].bytes += rec.size;
            break;
        case ALLOC_KIND_HOST:
            live_host_stats.count++;
            live_host_stats.bytes += rec.size;
            break;
        case ALLOC_KIND_PINNED:
            live_pinned_stats.count++;
            live_pinned_stats.bytes += rec.size;
            break;
        }
    }

    os << "  Total allocations:\n";
    for (auto& kv : m_device_totals)
        os << "    cuda:" << kv.first << std::string(12 - std::to_string(kv.first).size(), ' ') << kv.second.total_count
           << " (" << format_bytes(kv.second.total_bytes) << ")\n";
    if (m_host_totals.total_count > 0)
        os << "    cpu              " << m_host_totals.total_count << " (" << format_bytes(m_host_totals.total_bytes)
           << ")\n";
    if (m_pinned_totals.total_count > 0)
        os << "    pinned           " << m_pinned_totals.total_count << " ("
           << format_bytes(m_pinned_totals.total_bytes) << ")\n";

    os << "  Peak usage:\n";
    for (auto& kv : m_device_totals)
        os << "    cuda:" << kv.first << std::string(12 - std::to_string(kv.first).size(), ' ')
           << format_bytes(kv.second.peak_bytes) << "\n";
    if (m_host_totals.total_count > 0)
        os << "    cpu              " << format_bytes(m_host_totals.peak_bytes) << "\n";
    if (m_pinned_totals.total_count > 0)
        os << "    pinned           " << format_bytes(m_pinned_totals.peak_bytes) << "\n";

    os << "  Live allocations:\n";
    for (auto& kv : live_gpu_stats)
        os << "    cuda:" << kv.first << std::string(12 - std::to_string(kv.first).size(), ' ') << kv.second.count
           << " (" << format_bytes(kv.second.bytes) << ")\n";
    if (live_host_stats.count > 0)
        os << "    cpu              " << live_host_stats.count << " (" << format_bytes(live_host_stats.bytes) << ")\n";
    if (live_pinned_stats.count > 0)
        os << "    pinned           " << live_pinned_stats.count << " (" << format_bytes(live_pinned_stats.bytes)
           << ")\n";
    if (m_live.empty())
        os << "    (none)\n";

    // Scope breakdown: group by scope, then by device.
    if (!m_scope_stats.empty()) {
        std::map<std::string, std::vector<const std::pair<const ScopeKey, std::pair<size_t, size_t>>*>> by_scope;
        for (auto& kv : m_scope_stats)
            by_scope[kv.first.scope].push_back(&kv);

        if (by_scope.size() > 1) {
            os << "\n  Scope breakdown:\n";
            for (auto& scope_kv : by_scope) {
                os << "    " << scope_kv.first << ":\n";

                auto sorted = scope_kv.second;
                std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
                    if (a->first.kind != b->first.kind)
                        return a->first.kind < b->first.kind;
                    return a->first.ordinal < b->first.ordinal;
                });
                for (auto* entry : sorted) {
                    char dev_buf[32];
                    const char* dev_name = "unknown";
                    switch (entry->first.kind) {
                    case ALLOC_KIND_DEVICE:
                        snprintf(dev_buf, sizeof(dev_buf), "cuda:%d", entry->first.ordinal);
                        dev_name = dev_buf;
                        break;
                    case ALLOC_KIND_HOST:
                        dev_name = "cpu";
                        break;
                    case ALLOC_KIND_PINNED:
                        dev_name = "pinned";
                        break;
                    }
                    os << "      " << dev_name << ": " << entry->second.first << " allocs, "
                       << format_bytes(entry->second.second) << "\n";
                }
            }
        }
    }

    const char* order_label = sort_order == 1 ? "chronological" : "by size (largest first)";

    auto sort_records = [&](std::vector<const AllocRecord*>& recs) {
        if (sort_order == 1)
            std::sort(recs.begin(), recs.end(), [](const AllocRecord* a, const AllocRecord* b) {
                return a->seq < b->seq;
            });
        else
            std::sort(recs.begin(), recs.end(), [](const AllocRecord* a, const AllocRecord* b) {
                return a->size > b->size;
            });
    };

    auto print_records = [&](const char* title, const std::vector<const AllocRecord*>& recs) {
        os << "\n  " << title << " " << order_label << " (up to " << max_items << "):\n";
        int shown = 0;
        for (auto* r : recs) {
            if (shown >= max_items)
                break;
            os << "    " << format_bytes(r->size);
            if (r->kind == ALLOC_KIND_DEVICE)
                os << " : cuda:" << r->device_ordinal;
            else if (r->kind == ALLOC_KIND_HOST)
                os << " : cpu";
            else if (r->kind == ALLOC_KIND_PINNED)
                os << " : pinned";
            os << " : " << r->tag;
            os << "\n";
            shown++;
        }
    };

    // GPU allocations grouped by device ordinal
    std::map<int, std::vector<const AllocRecord*>> gpu_by_device;
    std::vector<const AllocRecord*> host_recs, pinned_recs;
    for (auto& kv : m_live) {
        switch (kv.second.kind) {
        case ALLOC_KIND_DEVICE:
            gpu_by_device[kv.second.device_ordinal].push_back(&kv.second);
            break;
        case ALLOC_KIND_HOST:
            host_recs.push_back(&kv.second);
            break;
        case ALLOC_KIND_PINNED:
            pinned_recs.push_back(&kv.second);
            break;
        }
    }

    for (auto& kv : gpu_by_device) {
        sort_records(kv.second);
        char title[64];
        snprintf(title, sizeof(title), "Live cuda:%d allocations", kv.first);
        print_records(title, kv.second);
    }
    if (!host_recs.empty()) {
        sort_records(host_recs);
        print_records("Live cpu allocations", host_recs);
    }
    if (!pinned_recs.empty()) {
        sort_records(pinned_recs);
        print_records("Live pinned allocations", pinned_recs);
    }

    m_report_buf = os.str();
    return m_report_buf.c_str();
}

std::string AllocTracker::build_scope_path()
{
    if (tl_scope_stack.empty())
        return std::string();

    std::string path;
    for (size_t i = 0; i < tl_scope_stack.size(); i++) {
        if (i > 0)
            path += '/';
        path += tl_scope_stack[i];
    }
    return path;
}

size_t AllocTracker::get_current_bytes()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_current_bytes;
}

size_t AllocTracker::get_peak_bytes()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_peak_bytes;
}

size_t AllocTracker::get_total_alloc_count()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_total_count;
}

size_t AllocTracker::get_total_alloc_bytes()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_total_bytes;
}

int AllocTracker::get_live_count()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return (int)m_live.size();
}

// WP_API functions

extern "C" {

WP_API void wp_alloc_tracker_enable(int enable) { g_alloc_tracker.enabled = (enable != 0); }

WP_API int wp_alloc_tracker_is_enabled() { return g_alloc_tracker.enabled ? 1 : 0; }

WP_API void wp_alloc_tracker_reset() { g_alloc_tracker.reset(); }

WP_API void wp_alloc_tracker_set_tag(void* ptr, const char* tag) { g_alloc_tracker.set_tag(ptr, tag); }

WP_API void wp_alloc_tracker_push_scope(const char* name) { g_alloc_tracker.push_scope(name); }

WP_API void wp_alloc_tracker_pop_scope() { g_alloc_tracker.pop_scope(); }

WP_API const char* wp_alloc_tracker_report(int sort_order, int max_items)
{
    return g_alloc_tracker.report(sort_order, max_items);
}

WP_API size_t wp_alloc_tracker_get_current_bytes() { return g_alloc_tracker.get_current_bytes(); }

WP_API size_t wp_alloc_tracker_get_peak_bytes() { return g_alloc_tracker.get_peak_bytes(); }

WP_API size_t wp_alloc_tracker_get_total_alloc_count() { return g_alloc_tracker.get_total_alloc_count(); }

WP_API size_t wp_alloc_tracker_get_total_alloc_bytes() { return g_alloc_tracker.get_total_alloc_bytes(); }

WP_API int wp_alloc_tracker_get_live_count() { return g_alloc_tracker.get_live_count(); }
}
