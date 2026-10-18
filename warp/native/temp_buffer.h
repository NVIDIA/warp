// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "warp.h"

#include "cuda_util.h"

#include <unordered_map>

template <typename T = char> struct ScopedTemporary {

    ScopedTemporary(void* context, size_t size)
        : m_context(context)
        , m_stream(wp_cuda_context_get_stream(context))
        , m_buffer(static_cast<T*>(wp_alloc_device(m_context, size * sizeof(T))))
    {
    }

    ~ScopedTemporary() { free_device_on_stream(m_context, m_buffer, m_stream); }

    T* buffer() const { return m_buffer; }

private:
    void* m_context;
    void* m_stream;
    T* m_buffer;
};
