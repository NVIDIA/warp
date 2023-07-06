/** Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

namespace wp {

// wp::initializer_array<> is a simple substitute for std::initializer_list<>
// which doesn't depend on compiler implementation-specific support. It copies
// elements by value and only supports array-style indexing.
template<unsigned Length, typename Type>
struct initializer_array
{
    const Type storage[Length];

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
