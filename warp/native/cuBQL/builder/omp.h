// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace cuBQL {
  namespace omp {
    struct Context;
    
    template<typename T, int D>
    void refit(BinaryBVH<T,D>    &bvh,
               const box_t<T,D>  *boxes,
               Context *ctx);
    
    template<typename T, int D>
    void freeBVH(BinaryBVH<T,D> &bvh,
                 Context        *ctx);
  }
}

#include "cuBQL/builder/omp/refit.h"
#include "cuBQL/builder/omp/spatialMedian.h"

