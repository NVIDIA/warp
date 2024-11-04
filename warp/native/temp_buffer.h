/** Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

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
