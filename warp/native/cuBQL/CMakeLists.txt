# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# a interface-only library that only sets include paths etc; when
# using this the user has to manulaly set CUBQL_BUILDER_INSTANTIATION
# in one of his/her source files
add_library(cuBQL INTERFACE)

target_sources(cuBQL INTERFACE
  # main public "interface" to this library
  bvh.h
  # general math struff to make public stuff work
  math/common.h
  math/math.h
  math/vec.h
  math/box.h
  # general (lambda-templated) traversal routines
  traversal/shrinkingRadiusQuery.h
  traversal/fixedBoxQuery.h
  # host side builder(s)
  builder/cpu.h
  builder/cpu/spatialMedian.h
  # cuda side builder(s)
  builder/cuda.h
  builder/cuda/builder_common.h
  builder/cuda/sm_builder.h
  builder/cuda/sah_builder.h
  builder/cuda/gpu_builder.h
  builder/cuda/radix.h
  builder/cuda/rebinMortonBuilder.h
  builder/cuda/wide_gpu_builder.h
  )
target_include_directories(cuBQL INTERFACE
  ${PROJECT_SOURCE_DIR}
  )
set_target_properties(cuBQL
  PROPERTIES
  CXX_VISIBILITY_PRESET  default
  CUDA_VISIBILITY_PRESET default
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  )
  
# helper for creating type- and device-specific implementations of
# cubql; i.e., one for host_float4, one for cuda_int3, etc. Since
# everything other than the builders are entirely header only these
# type-specific targets will, in fact, only contain instantiations of
# the specific builders for the given type and device
function(add_specific_instantiation device suffix T D)
  add_library(cuBQL_${device}_${T}${D} SHARED EXCLUDE_FROM_ALL
    builder/${device}/instantiate_builders.${suffix}
    )
  if (${device} STREQUAL "cuda")
    target_compile_definitions(cuBQL_${device}_${T}${D}
      PUBLIC
      -DCUBQL_HAVE_CUDA=1
      )
  endif()
  set_target_properties(cuBQL_${device}_${T}${D}
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    FOLDER "CuBQL"
    )
  target_link_libraries(cuBQL_${device}_${T}${D}
    PUBLIC
    cuBQL
    )
  target_compile_definitions(cuBQL_${device}_${T}${D}
    PRIVATE
    -DCUBQL_INSTANTIATE_T=${T}
    -DCUBQL_INSTANTIATE_D=${D}
    )

  add_library(cuBQL_${device}_${T}${D}_static STATIC EXCLUDE_FROM_ALL
    builder/${device}/instantiate_builders.${suffix}
    )
  if (${device} STREQUAL "cuda")
    target_compile_definitions(cuBQL_${device}_${T}${D}_static
      PUBLIC
      -DCUBQL_HAVE_CUDA=1
      )
  endif()
  set_target_properties(cuBQL_${device}_${T}${D}_static
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "CuBQL"
    )
  target_link_libraries(cuBQL_${device}_${T}${D}_static
    PUBLIC
    cuBQL
    )
  set_target_properties(cuBQL_${device}_${T}${D}_static
    PROPERTIES
    CXX_VISIBILITY_PRESET  default
    CUDA_VISIBILITY_PRESET default
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_USE_STATIC_CUDA_RUNTIME ON
    )
  target_compile_definitions(cuBQL_${device}_${T}${D}_static
    PRIVATE
    -DCUBQL_INSTANTIATE_T=${T}
    -DCUBQL_INSTANTIATE_D=${D}
    )
endfunction()

# ------------------------------------------------------------------
# generate all type/dim specific targets, ie, one target for each
# {int,float,double,long}x{2,3,4}. Each such target contains the
# pre-compiled builder(s) for that specific type/dimension, for the
# case where the user does NOT want to use the header-only
# mechanism. To avoid "polluting" the user's project with lots of
# different targets that he or she may or may not need we add all
# those as 'EXCLUDE_FROM_ALL', so only those that get actually used
# will actually get built
# ------------------------------------------------------------------
foreach(T IN ITEMS float int double longlong)
  foreach(D IN ITEMS 2 3 4)
    add_specific_instantiation(cpu cpp ${T} ${D})
    if (CUBQL_HAVE_CUDA)
      add_specific_instantiation(cuda cu ${T} ${D})
    endif()
  endforeach()
endforeach()

# the collection of all different type-specific queries currently
# supplied by cubql. all these should be header-only and should all
# automatically on both cpu and cuda, so this is a INTERFACE library
add_library(cuBQL_queries INTERFACE
  #
  queries/pointData/knn.h
  #
  queries/pointData/findClosest.h
  queries/pointData/knn.h
  )

set_target_properties(cuBQL_queries PROPERTIES FOLDER "CuBQL")

target_link_libraries(cuBQL_queries
  INTERFACE
  cuBQL
  )



