#pragma once

#include "builtin.h"

#define PNANOVDB_C 
#ifdef WP_CUDA
    #define PNANOVDB_MEMCPY_CUSTOM
    #define pnanovdb_memcpy memcpy
#endif
#include "nanovdb/PNanoVDB.h"

namespace wp
{
struct Volume
{
    pnanovdb_buf_t buf;
    pnanovdb_grid_handle_t grid;
    pnanovdb_tree_handle_t tree;

    uint64_t size_in_bytes;
};

CUDA_CALLABLE inline float volume_sample(uint64_t id, vec3 pos, int mode)
{
    const Volume volume = *(const Volume*)(id);
    const pnanovdb_root_handle_t root = pnanovdb_tree_get_root(volume.buf, volume.tree);

    const pnanovdb_vec3_t xyz{pos.x, pos.y, pos.z};
    const pnanovdb_coord_t ijk = pnanovdb_vec3_round_to_coord(
        pnanovdb_grid_world_to_indexf(volume.buf, volume.grid, PNANOVDB_REF(xyz)));
    const pnanovdb_address_t address = pnanovdb_root_get_value_address(
        PNANOVDB_GRID_TYPE_FLOAT, volume.buf, root, PNANOVDB_REF(ijk));
    return pnanovdb_read_float(volume.buf, address);
}

CUDA_CALLABLE inline void adj_volume_sample(uint64_t id, vec3 pos, int mode, 
    uint64_t& adj_id, vec3& adj_pos, int& adj_mode, float& adj_res) {}

} // namespace wp
