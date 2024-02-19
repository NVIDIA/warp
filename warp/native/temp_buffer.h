
#pragma once

#include "cuda_util.h"
#include "warp.h"

#include <unordered_map>

template <typename T = char> struct ScopedTemporary
{

    ScopedTemporary(void *context, size_t size)
        : m_context(context), m_buffer(static_cast<T*>(alloc_device(m_context, size * sizeof(T))))
    {
    }

    ~ScopedTemporary()
    {
        free_device(m_context, m_buffer);
    }

    T *buffer() const
    {
        return m_buffer;
    }

  private:
    void *m_context;
    T *m_buffer;
};
