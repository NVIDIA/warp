/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
