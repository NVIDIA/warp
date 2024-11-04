/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nanovdb/NanoVDB.h>

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
