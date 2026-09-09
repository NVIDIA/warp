// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cpu_block_runtime.h"
#include "cpu_fiber.h"

#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {

constexpr int kMaxBlockDim = 1024;
constexpr int kBitsetWords = kMaxBlockDim / 64;
constexpr size_t kFiberStackSize = 1024 * 1024;

struct lane_set {
    uint64_t generation = 0;
    uint64_t words[kBitsetWords] {};
};

struct block_context {
    int block_dim = 1;
    int active_count = 1;
    int word_count = 1;
    uint64_t frontier_generation = 0;
    wp_fiber_t* main_fiber = nullptr;
    wp_fiber_t** fibers = nullptr;
    uint64_t* lane_generations = nullptr;
    uint8_t* lane_finished = nullptr;
    lane_set behind;
    lane_set front;
};

thread_local block_context* g_block_context = nullptr;
thread_local int g_lane = 0;

inline void bit_set(lane_set& set, int lane) { set.words[lane >> 6] |= 1ull << (lane & 63); }

inline void bit_clear(lane_set& set, int lane) { set.words[lane >> 6] &= ~(1ull << (lane & 63)); }

inline int first_set_bit(uint64_t word)
{
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long bit;
    _BitScanForward64(&bit, word);
    return static_cast<int>(bit);
#elif defined(_MSC_VER)
    unsigned long bit;
    const uint32_t low = static_cast<uint32_t>(word);
    if (low) {
        _BitScanForward(&bit, low);
        return static_cast<int>(bit);
    }
    _BitScanForward(&bit, static_cast<uint32_t>(word >> 32));
    return static_cast<int>(bit + 32);
#else
    return __builtin_ctzll(word);
#endif
}

int first_lane(const lane_set& set, int word_count)
{
    for (int word_index = 0; word_index < word_count; ++word_index) {
        const uint64_t word = set.words[word_index];
        if (word)
            return word_index * 64 + first_set_bit(word);
    }
    return -1;
}

int first_lane(const lane_set& a, const lane_set& b, int word_count)
{
    for (int word_index = 0; word_index < word_count; ++word_index) {
        const uint64_t word = a.words[word_index] | b.words[word_index];
        if (word)
            return word_index * 64 + first_set_bit(word);
    }
    return -1;
}

void move_to_front(block_context& context, int lane, uint64_t generation)
{
    if (generation > context.frontier_generation) {
        context.behind = context.front;
        context.front = lane_set {};
        context.front.generation = generation;
        context.frontier_generation = generation;
    }

    bit_clear(context.behind, lane);
    bit_set(context.front, lane);
}

int finish_lane(block_context& context, int lane)
{
    context.lane_finished[lane] = 1;
    bit_clear(context.behind, lane);
    bit_clear(context.front, lane);
    return first_lane(context.behind, context.front, context.word_count);
}

struct lane_task {
    block_context* context;
    int lane;
    wp_cpu_block_lane_fn kernel_fn;
    void* dim;
    size_t block_id;
    void* args;
};

void lane_entry(void* raw_task)
{
    lane_task* task = static_cast<lane_task*>(raw_task);
    g_block_context = task->context;
    g_lane = task->lane;
    task->kernel_fn(task->dim, task->block_id, task->lane, task->args);

    const int next = finish_lane(*task->context, task->lane);
    wp_fiber_switch(next < 0 ? task->context->main_fiber : task->context->fibers[next]);
    _wp_assert("A completed CPU block lane was resumed", __FILE__, static_cast<unsigned int>(__LINE__));
}

struct schedule_probe {
    const uint32_t* barrier_counts;
    int active_count;
    int* events;
    size_t event_capacity;
    size_t event_count;
};

void append_schedule_event(schedule_probe& probe, int event)
{
    if (probe.event_count < probe.event_capacity)
        probe.events[probe.event_count] = event;
    ++probe.event_count;
}

void schedule_probe_lane(void*, size_t, int lane, void* raw_probe)
{
    schedule_probe& probe = *static_cast<schedule_probe*>(raw_probe);
    for (uint32_t generation = 0; generation < probe.barrier_counts[lane]; ++generation) {
        append_schedule_event(probe, lane);
        wp_cpu_tile_sync();
    }
    append_schedule_event(probe, probe.active_count + lane);
}

}  // namespace

extern "C" WP_API int wp_cpu_get_thread_idx() { return g_lane; }

extern "C" WP_API int wp_cpu_get_active_count()
{
    block_context* context = g_block_context;
    if (!context)
        return 1;

    int count = 0;
    for (int lane = 0; lane < context->active_count; ++lane)
        count += context->lane_finished[lane] == 0;
    return count;
}

extern "C" WP_API int wp_cpu_get_first_active_lane()
{
    block_context* context = g_block_context;
    if (!context)
        return 0;

    return first_lane(context->behind, context->front, context->word_count);
}

extern "C" WP_API void wp_cpu_tile_sync()
{
    block_context* context = g_block_context;
    if (!context)
        return;

    const int lane = g_lane;
    const uint64_t own_generation = ++context->lane_generations[lane];
    move_to_front(*context, lane, own_generation);

    while (true) {
        // Compare the waiter's absolute generation before consulting the set
        // that may already have rotated to represent a later generation.
        if (context->frontier_generation > own_generation)
            return;

        const int laggard = first_lane(context->behind, context->word_count);
        if (laggard < 0)
            return;

        wp_fiber_switch(context->fibers[laggard]);
        g_block_context = context;
        g_lane = lane;
    }
}

extern "C" WP_API int wp_cpu_run_block(
    int block_dim, int active_count, wp_cpu_block_lane_fn kernel_fn, void* dim, size_t block_id, void* args
)
{
    if (!kernel_fn || block_dim < 1 || block_dim > kMaxBlockDim || active_count < 1 || active_count > block_dim) {
        _wp_assert("Invalid Warp CPU block runtime dimensions", __FILE__, static_cast<unsigned int>(__LINE__));
        return 0;
    }

    block_context* saved_context = g_block_context;
    const int saved_lane = g_lane;

    if (active_count == 1) {
        g_block_context = nullptr;
        g_lane = 0;
        kernel_fn(dim, block_id, 0, args);
        g_block_context = saved_context;
        g_lane = saved_lane;
        return 1;
    }

    block_context context = {};
    context.block_dim = block_dim;
    context.active_count = active_count;
    context.word_count = (active_count + 63) / 64;
    context.main_fiber = wp_fiber_active();
    context.front.generation = 0;

    if (!context.main_fiber) {
        _wp_assert(
            "Warp failed to initialize the main CPU fiber context", __FILE__, static_cast<unsigned int>(__LINE__)
        );
        return 0;
    }

    std::vector<wp_fiber_t*> fibers(active_count, nullptr);
    std::vector<uint64_t> generations(active_count, 0);
    std::vector<uint8_t> finished(active_count, 0);
    std::vector<lane_task> tasks(active_count);
    context.fibers = fibers.data();
    context.lane_generations = generations.data();
    context.lane_finished = finished.data();

    for (int lane = 0; lane < active_count; ++lane)
        bit_set(context.front, lane);

    for (int lane = 0; lane < active_count; ++lane) {
        tasks[lane] = lane_task { &context, lane, kernel_fn, dim, block_id, args };
        fibers[lane] = wp_fiber_create(&lane_entry, &tasks[lane], kFiberStackSize);
        if (!fibers[lane]) {
            for (int created = 0; created < lane; ++created)
                wp_fiber_destroy(fibers[created]);
            _wp_assert(
                "Warp failed to allocate a CPU block fiber with a 1 MiB usable stack", __FILE__,
                static_cast<unsigned int>(__LINE__)
            );
            g_block_context = saved_context;
            g_lane = saved_lane;
            return 0;
        }
    }

    g_block_context = &context;
    g_lane = 0;
    wp_fiber_switch(fibers[0]);

    for (wp_fiber_t* fiber : fibers)
        wp_fiber_destroy(fiber);

    g_block_context = saved_context;
    g_lane = saved_lane;
    return 1;
}

extern "C" WP_API int wp_cpu_test_schedule(
    int block_dim,
    int active_count,
    const uint32_t* barrier_counts,
    int* events,
    size_t event_capacity,
    size_t* event_count
)
{
    if (!barrier_counts || !event_count || (event_capacity && !events))
        return 0;

    schedule_probe probe { barrier_counts, active_count, events, event_capacity, 0 };
    const int result = wp_cpu_run_block(block_dim, active_count, &schedule_probe_lane, nullptr, 0, &probe);
    *event_count = probe.event_count;
    return result && probe.event_count <= event_capacity;
}
