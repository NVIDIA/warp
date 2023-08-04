
#pragma once

#include "warp.h"
#include "cuda_util.h"

#include <unordered_map>

// temporary buffer, useful for cub algorithms
struct TemporaryBuffer
{
    void *buffer = NULL;
    size_t buffer_size = 0;

    void ensure_fits(size_t size)
    {
        if (size > buffer_size)
        {
            size = std::max(2 * size, (buffer_size * 3) / 2);

            free_device(WP_CURRENT_CONTEXT, buffer);
            buffer = alloc_device(WP_CURRENT_CONTEXT, size);
            buffer_size = size;
        }
    }
};

struct PinnedTemporaryBuffer
{
    void *buffer = NULL;
    size_t buffer_size = 0;

    void ensure_fits(size_t size)
    {
        if (size > buffer_size)
        {
            free_pinned(buffer);
            buffer = alloc_pinned(size);
            buffer_size = size;
        }
    }
};

// map temp buffers to CUDA contexts
static std::unordered_map<void *, TemporaryBuffer> g_temp_buffer_map;
static std::unordered_map<void *, PinnedTemporaryBuffer> g_pinned_temp_buffer_map;
