// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! instantiates the GPU builder(s) */
#define CUBQL_CPU_BUILDER_IMPLEMENTATION 1

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cpu/spatialMedian.h"

#ifdef CUBQL_INSTANTIATE_T
// instantiate an explict type and dimension
CUBQL_CPU_INSTANTIATE_BINARY_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D)
CUBQL_CPU_INSTANTIATE_WIDE_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D,4)
CUBQL_CPU_INSTANTIATE_WIDE_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D,8)
#else
// default instantiation(s) for float3 only
CUBQL_CPU_INSTANTIATE_BINARY_BVH(float,3)
CUBQL_CPU_INSTANTIATE_WIDE_BVH(float,3,4)
CUBQL_CPU_INSTANTIATE_WIDE_BVH(float,3,8)
#endif
  
 
