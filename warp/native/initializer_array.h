// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace wp {

// wp::initializer_array<> is a simple substitute for std::initializer_list<>
// which doesn't depend on compiler implementation-specific support. It copies
// elements by value and only supports array-style indexing.
template <unsigned Length, typename Type> struct initializer_array {
    const Type storage[Length < 1 ? 1 : Length];

    CUDA_CALLABLE const Type operator[](unsigned i) { return storage[i]; }

    CUDA_CALLABLE const Type operator[](unsigned i) const { return storage[i]; }
};

}  // namespace wp
