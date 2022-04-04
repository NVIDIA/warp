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
#include "nanovdb/NanoVDB.h"

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

CUDA_CALLABLE inline pnanovdb_root_handle_t get_root(const pnanovdb_buf_t& buf,
                                                     const pnanovdb_grid_handle_t& grid = { 0u })
{
    const auto tree = pnanovdb_grid_get_tree(buf, grid);
    return pnanovdb_tree_get_root(buf, tree);
}
} // namespace volume

// Forward declarations
CUDA_CALLABLE inline float volume_sample_local(uint64_t id, vec3 uvw, int sampling_mode);
CUDA_CALLABLE inline float volume_sample_world(uint64_t id, vec3 xyz, int sampling_mode);
CUDA_CALLABLE inline float volume_lookup(uint64_t id, int32_t i, int32_t j, int32_t k);
CUDA_CALLABLE inline vec3 volume_sample_local_v(uint64_t id, vec3 uvw, int sampling_mode);
CUDA_CALLABLE inline vec3 volume_sample_world_v(uint64_t id, vec3 xyz, int sampling_mode);
CUDA_CALLABLE inline vec3 volume_lookup_v(uint64_t id, int32_t i, int32_t j, int32_t k);
CUDA_CALLABLE inline vec3 volume_transform(uint64_t id, vec3 uvw);
CUDA_CALLABLE inline vec3 volume_transform_inv(uint64_t id, vec3 xyz);
CUDA_CALLABLE inline void adj_volume_sample_local(
    uint64_t id, vec3 uvw, int sampling_mode, uint64_t& adj_id, vec3& adj_uvw, int& adj_sampling_mode, const float& adj_ret);
CUDA_CALLABLE inline void adj_volume_sample_local_v(
    uint64_t id, vec3 uvw, int sampling_mode, uint64_t& adj_id, vec3& adj_uvw, int& adj_sampling_mode, const vec3& adj_ret);
CUDA_CALLABLE inline void adj_volume_sample_world(
    uint64_t id, vec3 xyz, int sampling_mode, uint64_t& adj_id, vec3& adj_xyz, int& adj_sampling_mode, const float& adj_ret);
CUDA_CALLABLE inline void adj_volume_sample_world_v(
    uint64_t id, vec3 xyz, int sampling_mode, uint64_t& adj_id, vec3& adj_xyz, int& adj_sampling_mode, const vec3& adj_ret);
CUDA_CALLABLE inline void adj_volume_lookup(uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t& adj_id, int32_t& adj_i, int32_t& adj_j, int32_t& adj_k, const float& adj_ret);
CUDA_CALLABLE inline void adj_volume_lookup_v(
    uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t& adj_id, int32_t& adj_i, int32_t& adj_j, int32_t& adj_k, const vec3& adj_ret);
CUDA_CALLABLE inline void adj_volume_transform(uint64_t id, vec3 uvw, uint64_t& adj_id, vec3& adj_uvw, const vec3& adj_ret);
CUDA_CALLABLE inline void adj_volume_transform_inv(uint64_t id, vec3 xyz, uint64_t& adj_id, vec3& adj_xyz, const vec3& adj_ret);
// ---

// Sampling the volume at the given index-space coordinates, uvw can be fractional
CUDA_CALLABLE inline float volume_sample_local(uint64_t id, vec3 uvw, int sampling_mode)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);
    const pnanovdb_vec3_t uvw_pnano{ uvw.x, uvw.y, uvw.z };

    if (sampling_mode == volume::CLOSEST)
    {
        const pnanovdb_coord_t ijk = pnanovdb_vec3_round_to_coord(uvw_pnano);
        const pnanovdb_address_t address =
            pnanovdb_root_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, root, PNANOVDB_REF(ijk));
        return pnanovdb_read_float(buf, address);
    }
    else if (sampling_mode == volume::LINEAR)
    {
        constexpr pnanovdb_coord_t OFFSETS[] = {
            { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 1 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 }, { 1, 1, 1 },
        };

        const pnanovdb_vec3_t ijk_base{ floorf(uvw_pnano.x), floorf(uvw_pnano.y), floorf(uvw_pnano.z) };
        const pnanovdb_vec3_t ijk_frac{ uvw_pnano.x - ijk_base.x, uvw_pnano.y - ijk_base.y, uvw_pnano.z - ijk_base.z };
        const pnanovdb_coord_t ijk{ (pnanovdb_int32_t)ijk_base.x, (pnanovdb_int32_t)ijk_base.y, (pnanovdb_int32_t)ijk_base.z };

        pnanovdb_readaccessor_t accessor;
        pnanovdb_readaccessor_init(PNANOVDB_REF(accessor), root);
        float val = 0.0f;
        const float wx[2]{ 1 - ijk_frac.x, ijk_frac.x };
        const float wy[2]{ 1 - ijk_frac.y, ijk_frac.y };
        const float wz[2]{ 1 - ijk_frac.z, ijk_frac.z };
#pragma unroll
        for (int idx = 0; idx < 8; ++idx)
        {
            const pnanovdb_coord_t& offs = OFFSETS[idx];
            const pnanovdb_coord_t ijkShifted = pnanovdb_coord_add(ijk, offs);
            pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
                PNANOVDB_GRID_TYPE_FLOAT, buf, PNANOVDB_REF(accessor), PNANOVDB_REF(ijkShifted));
            val += wx[offs.x] * wy[offs.y] * wz[offs.z] * pnanovdb_read_float(buf, address);
        }
        return val;
    }
    return 0;
}

CUDA_CALLABLE inline void adj_volume_sample_local(
    uint64_t id, vec3 uvw, int sampling_mode, uint64_t& adj_id, vec3& adj_uvw, int& adj_sampling_mode, const float& adj_ret)
{
    if (sampling_mode != volume::LINEAR) {
        return; // NOOP
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
    float vs[2][2][2];
    const float wx[2]{ 1 - ijk_frac.x, ijk_frac.x };
    const float wy[2]{ 1 - ijk_frac.y, ijk_frac.y };
    const float wz[2]{ 1 - ijk_frac.z, ijk_frac.z };
    vec3 dphi(0,0,0);
#pragma unroll
    for (int idx = 0; idx < 8; ++idx)
    {
        const pnanovdb_coord_t& offs = OFFSETS[idx];
        const pnanovdb_coord_t ijkShifted = pnanovdb_coord_add(ijk, offs);
        pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, PNANOVDB_REF(accessor), PNANOVDB_REF(ijkShifted));
        const float v = pnanovdb_read_float(buf, address);
        const vec3 signs(offs.x * 2 - 1, offs.y * 2 - 1, offs.z * 2 - 1);
        dphi.x += signs.x * wy[offs.y] * wz[offs.z] * v;
        dphi.y += signs.y * wx[offs.x] * wz[offs.z] * v;
        dphi.z += signs.z * wx[offs.x] * wy[offs.y] * v;
    }

    adj_uvw += mul(dphi, adj_ret);
}

// Sampling the volume at the given index-space coordinates, uvw can be fractional
CUDA_CALLABLE inline vec3 volume_sample_local_v(uint64_t id, vec3 uvw, int sampling_mode)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);
    const pnanovdb_vec3_t uvw_pnano{ uvw.x, uvw.y, uvw.z };

    if (sampling_mode == volume::CLOSEST)
    {
        const pnanovdb_coord_t ijk = pnanovdb_vec3_round_to_coord(uvw_pnano);
        const pnanovdb_address_t address =
            pnanovdb_root_get_value_address(PNANOVDB_GRID_TYPE_VEC3F, buf, root, PNANOVDB_REF(ijk));
        const pnanovdb_vec3_t v = pnanovdb_read_vec3f(buf, address);
        return {v.x, v.y, v.z};
    }
    else if (sampling_mode == volume::LINEAR)
    {
        constexpr pnanovdb_coord_t OFFSETS[] = {
            { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 1 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 1, 0 }, { 1, 1, 1 },
        };

        const pnanovdb_vec3_t ijk_base{ floorf(uvw_pnano.x), floorf(uvw_pnano.y), floorf(uvw_pnano.z) };
        const pnanovdb_vec3_t ijk_frac{ uvw_pnano.x - ijk_base.x, uvw_pnano.y - ijk_base.y, uvw_pnano.z - ijk_base.z };
        const pnanovdb_coord_t ijk{ (pnanovdb_int32_t)ijk_base.x, (pnanovdb_int32_t)ijk_base.y, (pnanovdb_int32_t)ijk_base.z };

        pnanovdb_readaccessor_t accessor;
        pnanovdb_readaccessor_init(PNANOVDB_REF(accessor), root);
        vec3 val = 0.0f;
        const float wx[2]{ 1 - ijk_frac.x, ijk_frac.x };
        const float wy[2]{ 1 - ijk_frac.y, ijk_frac.y };
        const float wz[2]{ 1 - ijk_frac.z, ijk_frac.z };
#pragma unroll
        for (int idx = 0; idx < 8; ++idx)
        {
            const pnanovdb_coord_t& offs = OFFSETS[idx];
            const pnanovdb_coord_t ijkShifted = pnanovdb_coord_add(ijk, offs);
            pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
                PNANOVDB_GRID_TYPE_VEC3F, buf, PNANOVDB_REF(accessor), PNANOVDB_REF(ijkShifted));
            const pnanovdb_vec3_t v = pnanovdb_read_vec3f(buf, address);
            val += wx[offs.x] * wy[offs.y] * wz[offs.z] * vec3{v.x, v.y, v.z};
        }
        return val;
    }
    return 0;
}

CUDA_CALLABLE inline void adj_volume_sample_local_v(
    uint64_t id, vec3 uvw, int sampling_mode, uint64_t& adj_id, vec3& adj_uvw, int& adj_sampling_mode, const vec3& adj_ret)
{
    if (sampling_mode != volume::LINEAR) {
        return; // NOOP
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
    float vs[2][2][2];
    const float wx[2]{ 1 - ijk_frac.x, ijk_frac.x };
    const float wy[2]{ 1 - ijk_frac.y, ijk_frac.y };
    const float wz[2]{ 1 - ijk_frac.z, ijk_frac.z };
    vec3 dphi[3] = {{0,0,0}, {0,0,0}, {0,0,0}};
#pragma unroll
    for (int idx = 0; idx < 8; ++idx)
    {
        const pnanovdb_coord_t& offs = OFFSETS[idx];
        const pnanovdb_coord_t ijkShifted = pnanovdb_coord_add(ijk, offs);
        pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_VEC3F, buf, PNANOVDB_REF(accessor), PNANOVDB_REF(ijkShifted));
        const pnanovdb_vec3_t v = pnanovdb_read_vec3f(buf, address);
        vec3 signs(offs.x * 2 - 1, offs.y * 2 - 1, offs.z * 2 - 1);
        dphi[0] = add(dphi[0], vec3(signs.x * wy[offs.y] * wz[offs.z] * v.x, signs.y * wx[offs.x] * wz[offs.z] * v.x, signs.z * wx[offs.x] * wy[offs.y] * v.x));
        dphi[1] = add(dphi[1], vec3(signs.x * wy[offs.y] * wz[offs.z] * v.y, signs.y * wx[offs.x] * wz[offs.z] * v.y, signs.z * wx[offs.x] * wy[offs.y] * v.y));
        dphi[2] = add(dphi[2], vec3(signs.x * wy[offs.y] * wz[offs.z] * v.z, signs.y * wx[offs.x] * wz[offs.z] * v.z, signs.z * wx[offs.x] * wy[offs.y] * v.z));
    }

#pragma unroll
    for (int k = 0; k < 3; ++k)
    {
        adj_uvw[k] += dot(dphi[k], adj_ret);
    }
}

// Sampling the volume at the given world-space coordinates
CUDA_CALLABLE inline float volume_sample_world(uint64_t id, vec3 xyz, int sampling_mode)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = { 0u };
    const pnanovdb_root_handle_t root = volume::get_root(buf, grid);

    const pnanovdb_vec3_t xyz_pnano{ xyz.x, xyz.y, xyz.z };
    const pnanovdb_vec3_t uvw_pnano = pnanovdb_grid_world_to_indexf(buf, grid, PNANOVDB_REF(xyz_pnano));
    const vec3 uvw{ uvw_pnano.x, uvw_pnano.y, uvw_pnano.z };

    return volume_sample_local(id, uvw, sampling_mode);
}

CUDA_CALLABLE inline void adj_volume_sample_world(
    uint64_t id, vec3 xyz, int sampling_mode, uint64_t& adj_id, vec3& adj_xyz, int& adj_sampling_mode, const float& adj_ret)
{
    const vec3 uvw = volume_transform_inv(id, xyz);
    vec3 adj_uvw(0.f);
    adj_volume_sample_local(id, uvw, sampling_mode, adj_id, adj_uvw, adj_sampling_mode, adj_ret);
    adj_volume_transform_inv(id, uvw, adj_id, adj_xyz, adj_uvw);
}

// Sampling the volume at the given world-space coordinates
CUDA_CALLABLE inline vec3 volume_sample_world_v(uint64_t id, vec3 xyz, int sampling_mode)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = { 0u };
    const pnanovdb_root_handle_t root = volume::get_root(buf, grid);

    const pnanovdb_vec3_t xyz_pnano{ xyz.x, xyz.y, xyz.z };
    const pnanovdb_vec3_t uvw_pnano = pnanovdb_grid_world_to_indexf(buf, grid, PNANOVDB_REF(xyz_pnano));
    const vec3 uvw{ uvw_pnano.x, uvw_pnano.y, uvw_pnano.z };

    return volume_sample_local_v(id, uvw, sampling_mode);
}

CUDA_CALLABLE inline void adj_volume_sample_world_v(
    uint64_t id, vec3 xyz, int sampling_mode, uint64_t& adj_id, vec3& adj_xyz, int& adj_sampling_mode, const vec3& adj_ret)
{
    const vec3 uvw = volume_transform_inv(id, xyz);
    vec3 adj_uvw(0.f);
    adj_volume_sample_local_v(id, uvw, sampling_mode, adj_id, adj_uvw, adj_sampling_mode, adj_ret);
    adj_volume_transform_inv(id, uvw, adj_id, adj_xyz, adj_uvw);
}

CUDA_CALLABLE inline float volume_lookup(uint64_t id, int32_t i, int32_t j, int32_t k)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);

    const pnanovdb_coord_t ijk{ i, j, k };
    const pnanovdb_address_t address =
        pnanovdb_root_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, root, PNANOVDB_REF(ijk));
    return pnanovdb_read_float(buf, address);
}

CUDA_CALLABLE inline void adj_volume_lookup(uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t& adj_id, int32_t& adj_i, int32_t& adj_j, int32_t& adj_k, const float& adj_ret)
{
    // NOOP
}

CUDA_CALLABLE inline vec3 volume_lookup_v(uint64_t id, int32_t i, int32_t j, int32_t k)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);

    const pnanovdb_coord_t ijk{ i, j, k };
    const pnanovdb_address_t address =
        pnanovdb_root_get_value_address(PNANOVDB_GRID_TYPE_VEC3F, buf, root, PNANOVDB_REF(ijk));
    const pnanovdb_vec3_t v = pnanovdb_read_vec3f(buf, address);
    return {v.x, v.y, v.z};
}

CUDA_CALLABLE inline void adj_volume_lookup_v(
    uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t& adj_id, int32_t& adj_i, int32_t& adj_j, int32_t& adj_k, const vec3& adj_ret)
{
    // NOOP
}

// Index- to world-space space transformation
CUDA_CALLABLE inline vec3 volume_transform(uint64_t id, vec3 uvw)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = { 0u };
    const pnanovdb_vec3_t pos{ uvw.x, uvw.y, uvw.z };
    const pnanovdb_vec3_t xyz = pnanovdb_grid_index_to_worldf(buf, grid, PNANOVDB_REF(pos));
    return { xyz.x, xyz.y, xyz.z };
}

CUDA_CALLABLE inline void adj_volume_transform(uint64_t id, vec3 uvw, uint64_t& adj_id, vec3& adj_uvw, const vec3& adj_ret)
{
    auto grid = (const nanovdb::FloatGrid*)id;
    adj_uvw += grid->map().applyJacobianF(adj_ret);
}

// World- to index-space transformation
CUDA_CALLABLE inline vec3 volume_transform_inv(uint64_t id, vec3 xyz)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = { 0u };
    const pnanovdb_vec3_t pos{ xyz.x, xyz.y, xyz.z };
    const pnanovdb_vec3_t uvw = pnanovdb_grid_world_to_indexf(buf, grid, PNANOVDB_REF(pos));
    return { uvw.x, uvw.y, uvw.z };
}

CUDA_CALLABLE inline void adj_volume_transform_inv(uint64_t id, vec3 xyz, uint64_t& adj_id, vec3& adj_xyz, const vec3& adj_ret)
{
    auto grid = (const nanovdb::FloatGrid*)id;
    adj_xyz += grid->map().applyInverseJacobianF(adj_ret);
}

} // namespace wp
