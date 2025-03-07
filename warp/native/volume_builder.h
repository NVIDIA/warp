/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nanovdb/NanoVDB.h>

#define WP_VOLUME_BUILDER_INSTANTIATE_TYPES                                                                            \
    EXPAND_BUILDER_TYPE(int32_t)                                                                                       \
    EXPAND_BUILDER_TYPE(float)                                                                                         \
    EXPAND_BUILDER_TYPE(nanovdb::Vec3f)                                                                                \
    EXPAND_BUILDER_TYPE(nanovdb::Vec4f)                                                                                \

template <typename BuildT> struct BuildGridParams
{
    nanovdb::Map map;
    BuildT background_value{0};
    char name[256] = "";
};

template <> struct BuildGridParams<nanovdb::ValueIndex>
{
    nanovdb::Map map;
    nanovdb::ValueIndex background_value;
    char name[256] = "";
};

template <> struct BuildGridParams<nanovdb::ValueOnIndex>
{
    nanovdb::Map map;
    double voxel_size = 1.0;
    char name[256] = "";
};

template <typename BuildT>
void build_grid_from_points(nanovdb::Grid<nanovdb::NanoTree<BuildT>>*& out_grid, size_t& out_grid_size,
                            const void* points, size_t num_points, bool points_in_world_space,
                            const BuildGridParams<BuildT>& params);
