/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"

#define PNANOVDB_C
#ifdef WP_CUDA
#    define PNANOVDB_MEMCPY_CUSTOM
#    define pnanovdb_memcpy memcpy
#endif
#include "nanovdb/PNanoVDB.h"

namespace wp
{
namespace volume
{

static constexpr int CLOSEST = 0;
static constexpr int LINEAR = 1;

// helper functions
CUDA_CALLABLE inline pnanovdb_buf_t id_to_buffer(uint64_t id)
{
    pnanovdb_buf_t buf;
    buf.data = (uint32_t*)id;
    return buf;
}

CUDA_CALLABLE inline pnanovdb_uint32_t get_grid_type(const pnanovdb_buf_t& buf)
{
    const pnanovdb_grid_t *grid_data = (const pnanovdb_grid_t*)buf.data;
    return grid_data->grid_type;
}

CUDA_CALLABLE inline pnanovdb_root_handle_t get_root(const pnanovdb_buf_t& buf,
                                                     const pnanovdb_grid_handle_t& grid = { 0u })
{
    const auto tree = pnanovdb_grid_get_tree(buf, grid);
    return pnanovdb_tree_get_root(buf, tree);
}
} // namespace volume

CUDA_CALLABLE inline void pnano_read(float& result, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk) {
    const pnanovdb_address_t address = pnanovdb_root_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, root, ijk);
    result = pnanovdb_read_float(buf, address);
}
CUDA_CALLABLE inline void pnano_read(int32_t& result, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk) {
    const pnanovdb_address_t address = pnanovdb_root_get_value_address(PNANOVDB_GRID_TYPE_INT32, buf, root, ijk);
    result = pnanovdb_read_int32(buf, address);
}
CUDA_CALLABLE inline void pnano_read(vec3& result, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk) {
    const pnanovdb_address_t address = pnanovdb_root_get_value_address(PNANOVDB_GRID_TYPE_VEC3F, buf, root, ijk);
    const pnanovdb_vec3_t v = pnanovdb_read_vec3f(buf, address);
    result = {v.x, v.y, v.z};
}

CUDA_CALLABLE inline void pnano_read(float& result, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk) {
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk);
    result = pnanovdb_read_float(buf, address);
}
CUDA_CALLABLE inline void pnano_read(int32_t& result, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk) {
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_INT32, buf, acc, ijk);
    result = pnanovdb_read_int32(buf, address);
}
CUDA_CALLABLE inline void pnano_read(vec3& result, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk) {
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_VEC3F, buf, acc, ijk);
    const pnanovdb_vec3_t v = pnanovdb_read_vec3f(buf, address);
    result = {v.x, v.y, v.z};
}

// Sampling the volume at the given index-space coordinates, uvw can be fractional
template<typename T>
CUDA_CALLABLE inline T volume_sample(uint64_t id, vec3 uvw, int sampling_mode)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);
    const pnanovdb_vec3_t uvw_pnano{ uvw.x, uvw.y, uvw.z };

    if (sampling_mode == volume::CLOSEST)
    {
        const pnanovdb_coord_t ijk = pnanovdb_vec3_round_to_coord(uvw_pnano);
        T val;
        pnano_read(val, buf, root, PNANOVDB_REF(ijk));
        return val;
    }
    else if (sampling_mode == volume::LINEAR)
    {
        // NB. linear sampling is not used on int volumes
        constexpr pnanovdb_coord_t OFFSETS[] = {
            { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 1 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 }, { 1, 1, 1 },
        };

        const pnanovdb_vec3_t ijk_base{ floorf(uvw_pnano.x), floorf(uvw_pnano.y), floorf(uvw_pnano.z) };
        const pnanovdb_vec3_t ijk_frac{ uvw_pnano.x - ijk_base.x, uvw_pnano.y - ijk_base.y, uvw_pnano.z - ijk_base.z };
        const pnanovdb_coord_t ijk{ (pnanovdb_int32_t)ijk_base.x, (pnanovdb_int32_t)ijk_base.y, (pnanovdb_int32_t)ijk_base.z };

        pnanovdb_readaccessor_t accessor;
        pnanovdb_readaccessor_init(PNANOVDB_REF(accessor), root);
        T val = 0;
        const float wx[2]{ 1 - ijk_frac.x, ijk_frac.x };
        const float wy[2]{ 1 - ijk_frac.y, ijk_frac.y };
        const float wz[2]{ 1 - ijk_frac.z, ijk_frac.z };
        for (int idx = 0; idx < 8; ++idx)
        {
            const pnanovdb_coord_t& offs = OFFSETS[idx];
            const pnanovdb_coord_t ijk_shifted = pnanovdb_coord_add(ijk, offs);
            T v;
            pnano_read(v, buf, PNANOVDB_REF(accessor), PNANOVDB_REF(ijk_shifted));
            val = add(val, T(wx[offs.x] * wy[offs.y] * wz[offs.z] * v));
        }
        return val;
    }
    return 0;
}

// Sampling a float volume at the given index-space coordinates, uvw can be fractional
CUDA_CALLABLE inline float volume_sample_f(uint64_t id, vec3 uvw, int sampling_mode)
{
    if (volume::get_grid_type(volume::id_to_buffer(id)) != PNANOVDB_GRID_TYPE_FLOAT) return 0.f;
    return volume_sample<float>(id, uvw, sampling_mode);
}

// Sampling an int volume at the given index-space coordinates, uvw can be fractional
CUDA_CALLABLE inline int32_t volume_sample_i(uint64_t id, vec3 uvw)
{
    if (volume::get_grid_type(volume::id_to_buffer(id)) != PNANOVDB_GRID_TYPE_INT32) return 0;
    return volume_sample<int32_t>(id, uvw, volume::CLOSEST);
}

// Sampling a vector volume at the given index-space coordinates, uvw can be fractional
CUDA_CALLABLE inline vec3 volume_sample_v(uint64_t id, vec3 uvw, int sampling_mode)
{
    if (volume::get_grid_type(volume::id_to_buffer(id)) != PNANOVDB_GRID_TYPE_VEC3F) return vec3(0.f);
    return volume_sample<vec3>(id, uvw, sampling_mode);
}

CUDA_CALLABLE inline void adj_volume_sample_f(
    uint64_t id, vec3 uvw, int sampling_mode, uint64_t& adj_id, vec3& adj_uvw, int& adj_sampling_mode, const float& adj_ret)
{
    if (volume::get_grid_type(volume::id_to_buffer(id)) != PNANOVDB_GRID_TYPE_FLOAT) return;

    if (sampling_mode != volume::LINEAR) {
        return; // NOP
    }

    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);
    const pnanovdb_vec3_t uvw_pnano{ uvw.x, uvw.y, uvw.z };

    constexpr pnanovdb_coord_t OFFSETS[] = {
        { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 1 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 }, { 1, 1, 1 },
    };

    const pnanovdb_vec3_t ijk_base{ floorf(uvw_pnano.x), floorf(uvw_pnano.y), floorf(uvw_pnano.z) };
    const pnanovdb_vec3_t ijk_frac{ uvw_pnano.x - ijk_base.x, uvw_pnano.y - ijk_base.y, uvw_pnano.z - ijk_base.z };
    const pnanovdb_coord_t ijk{ (pnanovdb_int32_t)ijk_base.x, (pnanovdb_int32_t)ijk_base.y, (pnanovdb_int32_t)ijk_base.z };

    pnanovdb_readaccessor_t accessor;
    pnanovdb_readaccessor_init(PNANOVDB_REF(accessor), root);
    const float wx[2]{ 1 - ijk_frac.x, ijk_frac.x };
    const float wy[2]{ 1 - ijk_frac.y, ijk_frac.y };
    const float wz[2]{ 1 - ijk_frac.z, ijk_frac.z };
    vec3 dphi(0,0,0);
    for (int idx = 0; idx < 8; ++idx)
    {
        const pnanovdb_coord_t& offs = OFFSETS[idx];
        const pnanovdb_coord_t ijk_shifted = pnanovdb_coord_add(ijk, offs);
        float v;
        pnano_read(v, buf, PNANOVDB_REF(accessor), PNANOVDB_REF(ijk_shifted));
        const vec3 signs(offs.x * 2 - 1, offs.y * 2 - 1, offs.z * 2 - 1);
        const vec3 grad_w(signs.x * wy[offs.y] * wz[offs.z], signs.y * wx[offs.x] * wz[offs.z], signs.z * wx[offs.x] * wy[offs.y]); 
        dphi = add(dphi, mul(v, grad_w));
    }

    adj_uvw += mul(dphi, adj_ret);
}

CUDA_CALLABLE inline void adj_volume_sample_v(
    uint64_t id, vec3 uvw, int sampling_mode, uint64_t& adj_id, vec3& adj_uvw, int& adj_sampling_mode, const vec3& adj_ret)
{
    if (volume::get_grid_type(volume::id_to_buffer(id)) != PNANOVDB_GRID_TYPE_VEC3F) return;

    if (sampling_mode != volume::LINEAR) {
        return; // NOP
    }

    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);
    const pnanovdb_vec3_t uvw_pnano{ uvw.x, uvw.y, uvw.z };

    constexpr pnanovdb_coord_t OFFSETS[] = {
        { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 1 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 }, { 1, 1, 1 },
    };

    const pnanovdb_vec3_t ijk_base{ floorf(uvw_pnano.x), floorf(uvw_pnano.y), floorf(uvw_pnano.z) };
    const pnanovdb_vec3_t ijk_frac{ uvw_pnano.x - ijk_base.x, uvw_pnano.y - ijk_base.y, uvw_pnano.z - ijk_base.z };
    const pnanovdb_coord_t ijk{ (pnanovdb_int32_t)ijk_base.x, (pnanovdb_int32_t)ijk_base.y, (pnanovdb_int32_t)ijk_base.z };

    pnanovdb_readaccessor_t accessor;
    pnanovdb_readaccessor_init(PNANOVDB_REF(accessor), root);
    const float wx[2]{ 1 - ijk_frac.x, ijk_frac.x };
    const float wy[2]{ 1 - ijk_frac.y, ijk_frac.y };
    const float wz[2]{ 1 - ijk_frac.z, ijk_frac.z };
    vec3 dphi[3] = {{0,0,0}, {0,0,0}, {0,0,0}};
    for (int idx = 0; idx < 8; ++idx)
    {
        const pnanovdb_coord_t& offs = OFFSETS[idx];
        const pnanovdb_coord_t ijk_shifted = pnanovdb_coord_add(ijk, offs);
        vec3 v;
        pnano_read(v, buf, PNANOVDB_REF(accessor), PNANOVDB_REF(ijk_shifted));
        const vec3 signs(offs.x * 2 - 1, offs.y * 2 - 1, offs.z * 2 - 1);
        const vec3 grad_w(signs.x * wy[offs.y] * wz[offs.z], signs.y * wx[offs.x] * wz[offs.z], signs.z * wx[offs.x] * wy[offs.y]); 
        dphi[0] = add(dphi[0], mul(v.x, grad_w));
        dphi[1] = add(dphi[1], mul(v.y, grad_w));
        dphi[2] = add(dphi[2], mul(v.z, grad_w));
    }

    for (int k = 0; k < 3; ++k)
    {
        adj_uvw[k] += dot(dphi[k], adj_ret);
    }
}

CUDA_CALLABLE inline void adj_volume_sample_i(uint64_t id, vec3 uvw, uint64_t& adj_id, vec3& adj_uvw, const int32_t& adj_ret)
{
    // NOP
}

CUDA_CALLABLE inline float volume_lookup_f(uint64_t id, int32_t i, int32_t j, int32_t k)
{
    if (volume::get_grid_type(volume::id_to_buffer(id)) != PNANOVDB_GRID_TYPE_FLOAT) return 0.f;

    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);

    const pnanovdb_coord_t ijk{ i, j, k };
    float val;
    pnano_read(val, buf, root, PNANOVDB_REF(ijk));
    return val;
}

CUDA_CALLABLE inline int32_t volume_lookup_i(uint64_t id, int32_t i, int32_t j, int32_t k)
{
    if (volume::get_grid_type(volume::id_to_buffer(id)) != PNANOVDB_GRID_TYPE_INT32) return 0;

    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);

    const pnanovdb_coord_t ijk{ i, j, k };
    int32_t val;
    pnano_read(val, buf, root, PNANOVDB_REF(ijk));
    return val;
}

CUDA_CALLABLE inline vec3 volume_lookup_v(uint64_t id, int32_t i, int32_t j, int32_t k)
{
    if (volume::get_grid_type(volume::id_to_buffer(id)) != PNANOVDB_GRID_TYPE_VEC3F) return vec3(0.f);

    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);

    const pnanovdb_coord_t ijk{ i, j, k };
    vec3 val;
    pnano_read(val, buf, root, PNANOVDB_REF(ijk));
    return val;
}

CUDA_CALLABLE inline void adj_volume_lookup_f(
    uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t& adj_id, int32_t& adj_i, int32_t& adj_j, int32_t& adj_k, const float& adj_ret)
{
    // NOP
}

CUDA_CALLABLE inline void adj_volume_lookup_i(
    uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t& adj_id, int32_t& adj_i, int32_t& adj_j, int32_t& adj_k, const int32_t& adj_ret)
{
    // NOP
}

CUDA_CALLABLE inline void adj_volume_lookup_v(
    uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t& adj_id, int32_t& adj_i, int32_t& adj_j, int32_t& adj_k, const vec3& adj_ret)
{
    // NOP
}

// Transform position from index space to world space
CUDA_CALLABLE inline vec3 volume_index_to_world(uint64_t id, vec3 uvw)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = { 0u };
    const pnanovdb_vec3_t pos{ uvw.x, uvw.y, uvw.z };
    const pnanovdb_vec3_t xyz = pnanovdb_grid_index_to_worldf(buf, grid, PNANOVDB_REF(pos));
    return { xyz.x, xyz.y, xyz.z };
}

// Transform position from world space to index space
CUDA_CALLABLE inline vec3 volume_world_to_index(uint64_t id, vec3 xyz)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = { 0u };
    const pnanovdb_vec3_t pos{ xyz.x, xyz.y, xyz.z };
    const pnanovdb_vec3_t uvw = pnanovdb_grid_world_to_indexf(buf, grid, PNANOVDB_REF(pos));
    return { uvw.x, uvw.y, uvw.z };
}

CUDA_CALLABLE inline void adj_volume_index_to_world(uint64_t id, vec3 uvw, uint64_t& adj_id, vec3& adj_uvw, const vec3& adj_ret)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = { 0u };
    const pnanovdb_vec3_t pos{ adj_ret.x, adj_ret.y, adj_ret.z };
    const pnanovdb_vec3_t xyz = pnanovdb_grid_index_to_world_dirf(buf, grid, PNANOVDB_REF(pos));
    adj_uvw = add(adj_uvw, vec3{ xyz.x, xyz.y, xyz.z });
}

CUDA_CALLABLE inline void adj_volume_world_to_index(uint64_t id, vec3 xyz, uint64_t& adj_id, vec3& adj_xyz, const vec3& adj_ret)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = { 0u };
    const pnanovdb_vec3_t pos{ adj_ret.x, adj_ret.y, adj_ret.z };
    const pnanovdb_vec3_t uvw = pnanovdb_grid_world_to_index_dirf(buf, grid, PNANOVDB_REF(pos));
    adj_xyz = add(adj_xyz, vec3{ uvw.x, uvw.y, uvw.z });
}

// Transform direction from index space to world space
CUDA_CALLABLE inline vec3 volume_index_to_world_dir(uint64_t id, vec3 uvw)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = { 0u };
    const pnanovdb_vec3_t pos{ uvw.x, uvw.y, uvw.z };
    const pnanovdb_vec3_t xyz = pnanovdb_grid_index_to_world_dirf(buf, grid, PNANOVDB_REF(pos));
    return { xyz.x, xyz.y, xyz.z };
}

// Transform direction from world space to index space
CUDA_CALLABLE inline vec3 volume_world_to_index_dir(uint64_t id, vec3 xyz)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = { 0u };
    const pnanovdb_vec3_t pos{ xyz.x, xyz.y, xyz.z };
    const pnanovdb_vec3_t uvw = pnanovdb_grid_world_to_index_dirf(buf, grid, PNANOVDB_REF(pos));
    return { uvw.x, uvw.y, uvw.z };
}

CUDA_CALLABLE inline void adj_volume_index_to_world_dir(uint64_t id, vec3 uvw, uint64_t& adj_id, vec3& adj_uvw, const vec3& adj_ret)
{
    adj_volume_index_to_world(id, uvw, adj_id, adj_uvw, adj_ret);
}

CUDA_CALLABLE inline void adj_volume_world_to_index_dir(uint64_t id, vec3 xyz, uint64_t& adj_id, vec3& adj_xyz, const vec3& adj_ret)
{
    adj_volume_world_to_index(id, xyz, adj_id, adj_xyz, adj_ret);
}

} // namespace wp
