// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "warp.h"

#include "cuda_util.h"

#include <unordered_map>

template <typename T = char> struct ScopedTemporary {

    ScopedTemporary(void* context, size_t size)
        : m_context(context)
        , m_buffer(static_cast<T*>(wp_alloc_device(m_context, size * sizeof(T))))
    {
    }

    ~ScopedTemporary() { wp_free_device(m_context, m_buffer); }

    T* buffer() const { return m_buffer; }

private:
    void* m_context;
    T* m_buffer;
};
