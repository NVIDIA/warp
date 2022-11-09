#pragma once

#include <nanovdb/NanoVDB.h>

template<typename BuildT>
struct BuildGridParams {
    double voxel_size = 1.0;
    BuildT background_value{0};
    nanovdb::Vec3d translation{0.0, 0.0, 0.0};
    char name[256] = "";
};

template <typename BuildT>
void build_grid_from_tiles(nanovdb::Grid<nanovdb::NanoTree<BuildT>> *&out_grid,
                           size_t &out_grid_size,
                           const void *points,
                           size_t num_points,
                           bool points_in_world_space,
                           const BuildGridParams<BuildT> &params);
