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

namespace wp {

// wp::initializer_array<> is a simple substitute for std::initializer_list<>
// which doesn't depend on compiler implementation-specific support. It copies
// elements by value and only supports array-style indexing.
template<unsigned Length, typename Type>
struct initializer_array
{
    const Type storage[Length < 1 ? 1 : Length];

    CUDA_CALLABLE const Type operator[](unsigned i)
    {
        return storage[i];
    }

    CUDA_CALLABLE const Type operator[](unsigned i) const
    {
        return storage[i];
    }
};

}  // namespace wp
