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

#include "array.h"
#include "builtin.h"

#define PNANOVDB_C
#define PNANOVDB_MEMCPY_CUSTOM
#define pnanovdb_memcpy memcpy

#if defined(WP_NO_CRT) && !defined(__CUDACC__)
// PNanoVDB will try to include <stdint.h> unless __CUDACC_RTC__ is defined
#define __CUDACC_RTC__
#endif

#include "nanovdb/PNanoVDB.h"

#if defined(WP_NO_CRT) && !defined(__CUDACC__)
#undef __CUDACC_RTC__
#endif

namespace wp
{
namespace volume
{

// Need to kept in sync with constants in python-side Volume class
static constexpr int CLOSEST = 0;
static constexpr int LINEAR = 1;

// pnanovdb helper function

CUDA_CALLABLE inline pnanovdb_buf_t id_to_buffer(uint64_t id)
{
    pnanovdb_buf_t buf;
    buf.data = (uint32_t *)id;
    return buf;
}

CUDA_CALLABLE inline pnanovdb_grid_handle_t get_grid(pnanovdb_buf_t buf)
{
    return {0u};
}

CUDA_CALLABLE inline pnanovdb_uint32_t get_grid_type(pnanovdb_buf_t buf)
{
    return pnanovdb_grid_get_grid_type(buf, get_grid(buf));
}

CUDA_CALLABLE inline pnanovdb_tree_handle_t get_tree(pnanovdb_buf_t buf)
{
    return pnanovdb_grid_get_tree(buf, get_grid(buf));
}

CUDA_CALLABLE inline pnanovdb_root_handle_t get_root(pnanovdb_buf_t buf)
{
    return pnanovdb_tree_get_root(buf, get_tree(buf));
}

template <typename T> struct pnano_traits
{
};

// to add support for more grid types, extend this
// and update _volume_supported_value_types in builtins.py

template <> struct pnano_traits<int32_t>
{
    static constexpr int GRID_TYPE = PNANOVDB_GRID_TYPE_INT32;
};

template <> struct pnano_traits<int64_t>
{
    static constexpr int GRID_TYPE = PNANOVDB_GRID_TYPE_INT64;
};

template <> struct pnano_traits<uint32_t>
{
    static constexpr int GRID_TYPE = PNANOVDB_GRID_TYPE_UINT32;
};

template <> struct pnano_traits<float>
{
    static constexpr int GRID_TYPE = PNANOVDB_GRID_TYPE_FLOAT;
};

template <> struct pnano_traits<double>
{
    static constexpr int GRID_TYPE = PNANOVDB_GRID_TYPE_DOUBLE;
};

template <> struct pnano_traits<vec3f>
{
    static constexpr int GRID_TYPE = PNANOVDB_GRID_TYPE_VEC3F;
};

template <> struct pnano_traits<vec3d>
{
    static constexpr int GRID_TYPE = PNANOVDB_GRID_TYPE_VEC3D;
};

template <> struct pnano_traits<vec4f>
{
    static constexpr int GRID_TYPE = PNANOVDB_GRID_TYPE_VEC4F;
};

template <> struct pnano_traits<vec4d>
{
    static constexpr int GRID_TYPE = PNANOVDB_GRID_TYPE_VEC4D;
};

// common accessors over various grid types
// WARNING: implementation below only for >=32b values, but that's the case for all types above
// for smaller types add a specialization

template <typename T> CUDA_CALLABLE inline void pnano_read(T &result, pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    result = *reinterpret_cast<const T *>(buf.data + (address.byte_offset >> 2));
}

template <typename T>
CUDA_CALLABLE inline void pnano_write(const T &value, pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    *reinterpret_cast<T *>(buf.data + (address.byte_offset >> 2)) = value;
}

template <typename T>
CUDA_CALLABLE inline void pnano_read(T &result, pnanovdb_buf_t buf, pnanovdb_root_handle_t root,
                                     PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    using traits = pnano_traits<T>;
    const pnanovdb_address_t address = pnanovdb_root_get_value_address(traits::GRID_TYPE, buf, root, ijk);
    pnano_read<T>(result, buf, address);
}

template <typename T>
CUDA_CALLABLE inline void pnano_read(T &result, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc,
                                     PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    using traits = pnano_traits<T>;
    // pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(traits::GRID_TYPE, buf, acc, ijk);
    pnanovdb_uint32_t level;
    const pnanovdb_address_t address =
        pnanovdb_readaccessor_get_value_address_and_level(traits::GRID_TYPE, buf, acc, ijk, PNANOVDB_REF(level));
    pnano_read<T>(result, buf, address);
}

/// regular grid accessor (values stored in leafs)

struct value_accessor_base
{
    pnanovdb_buf_t buf;
    pnanovdb_root_handle_t root;
    pnanovdb_readaccessor_t accessor;

    explicit inline CUDA_CALLABLE value_accessor_base(const pnanovdb_buf_t buf) : buf(buf), root(get_root(buf))
    {
    }

    CUDA_CALLABLE inline void init_cache()
    {
        pnanovdb_readaccessor_init(PNANOVDB_REF(accessor), root);
    }
};

template <typename T> struct leaf_value_accessor : value_accessor_base
{
    using ValueType = T;

    explicit inline CUDA_CALLABLE leaf_value_accessor(const pnanovdb_buf_t buf) : value_accessor_base(buf)
    {
    }

    CUDA_CALLABLE inline bool is_valid() const
    {
        return get_grid_type(buf) == pnano_traits<T>::GRID_TYPE;
    }

    CUDA_CALLABLE inline T read_single(const pnanovdb_coord_t &ijk) const
    {
        T val;
        pnano_read(val, buf, root, PNANOVDB_REF(ijk));
        return val;
    }

    CUDA_CALLABLE inline T read_cache(const pnanovdb_coord_t &ijk)
    {
        T val;
        pnano_read(val, buf, PNANOVDB_REF(accessor), PNANOVDB_REF(ijk));
        return val;
    }

    CUDA_CALLABLE inline void adj_read_single(const pnanovdb_coord_t &ijk, const T &adj_ret)
    {
        // NOP
    }

    CUDA_CALLABLE inline void adj_read_cache(const pnanovdb_coord_t &ijk, const T &adj_ret)
    {
        // NOP
    }
};

CUDA_CALLABLE inline pnanovdb_uint64_t leaf_regular_get_voxel_index(pnanovdb_buf_t buf,
                                                                    pnanovdb_address_t value_address,
                                                                    PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    // compute leaf index from value address, assuming all leaf voxels are allocated
    const pnanovdb_grid_type_t grid_type = get_grid_type(buf);
    const pnanovdb_uint32_t n = pnanovdb_leaf_coord_to_offset(ijk);
    const pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, leaf_off_table) +
                                          ((PNANOVDB_GRID_TYPE_GET(grid_type, value_stride_bits) * n) >> 3u);
    const pnanovdb_address_t leaf_address = pnanovdb_address_offset_neg(value_address, byte_offset);

    const pnanovdb_uint64_t first_leaf_offset = pnanovdb_tree_get_node_offset_leaf(buf, get_tree(buf));
    const pnanovdb_uint32_t leaf_size = PNANOVDB_GRID_TYPE_GET(grid_type, leaf_size);
    const pnanovdb_uint64_t leaf_index = (leaf_address.byte_offset - first_leaf_offset) / leaf_size;

    return leaf_index * PNANOVDB_LEAF_TABLE_COUNT + n + 1;
}

CUDA_CALLABLE inline pnanovdb_uint64_t get_grid_voxel_index(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf,
                                                            pnanovdb_address_t value_address,
                                                            const pnanovdb_coord_t &ijk)
{
    switch (grid_type)
    {
    case PNANOVDB_GRID_TYPE_INDEX:
        return pnanovdb_leaf_index_get_value_index(buf, value_address, PNANOVDB_REF(ijk));
    case PNANOVDB_GRID_TYPE_ONINDEX:
        return pnanovdb_leaf_onindex_get_value_index(buf, value_address, PNANOVDB_REF(ijk));
    case PNANOVDB_GRID_TYPE_INDEXMASK:
        return pnanovdb_leaf_indexmask_get_value_index(buf, value_address, PNANOVDB_REF(ijk));
    case PNANOVDB_GRID_TYPE_ONINDEXMASK:
        return pnanovdb_leaf_onindexmask_get_value_index(buf, value_address, PNANOVDB_REF(ijk));
    default:
        return leaf_regular_get_voxel_index(buf, value_address, PNANOVDB_REF(ijk));
    }
};

/// index grid accessor
template <typename T> struct index_value_accessor : value_accessor_base
{
    using ValueType = T;

    pnanovdb_grid_type_t grid_type;
    array_t<T> data;
    const T &background;
    T *adj_background;

    explicit inline CUDA_CALLABLE index_value_accessor(const pnanovdb_buf_t buf, const array_t<T> &data,
                                                       const T &background, T *adj_background = nullptr)
        : value_accessor_base(buf), grid_type(get_grid_type(buf)), data(data), background(background),
          adj_background(adj_background)
    {
    }

    CUDA_CALLABLE inline bool is_valid() const
    {
        // Accessor is valid for all grid types
        return true;
    }

    CUDA_CALLABLE inline T read_single(const pnanovdb_coord_t &ijk) const
    {
        pnanovdb_uint32_t level;
        const pnanovdb_address_t address =
            pnanovdb_root_get_value_address_and_level(grid_type, buf, root, PNANOVDB_REF(ijk), PNANOVDB_REF(level));
        return read_at(level, address, ijk);
    }

    CUDA_CALLABLE inline T read_cache(const pnanovdb_coord_t &ijk)
    {
        pnanovdb_uint32_t level;
        const pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address_and_level(
            grid_type, buf, PNANOVDB_REF(accessor), PNANOVDB_REF(ijk), PNANOVDB_REF(level));
        return read_at(level, address, ijk);
    }

    CUDA_CALLABLE inline T read_at(pnanovdb_uint32_t level, const pnanovdb_address_t address,
                                   const pnanovdb_coord_t &ijk) const
    {
        if (level == 0)
        {
            pnanovdb_uint64_t voxel_index = get_grid_voxel_index(grid_type, buf, address, ijk);

            if (voxel_index > 0)
            {
                return *wp::address(data, voxel_index - 1);
            }
        }

        return background;
    }

    CUDA_CALLABLE inline void adj_read_single(const pnanovdb_coord_t &ijk, const T &adj_ret)
    {
        pnanovdb_uint32_t level;
        const pnanovdb_address_t address =
            pnanovdb_root_get_value_address_and_level(grid_type, buf, root, PNANOVDB_REF(ijk), PNANOVDB_REF(level));
        adj_read_at(level, address, ijk, adj_ret);
    }

    CUDA_CALLABLE inline void adj_read_cache(const pnanovdb_coord_t &ijk, const T &adj_ret)
    {
        pnanovdb_uint32_t level;
        const pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address_and_level(
            grid_type, buf, PNANOVDB_REF(accessor), PNANOVDB_REF(ijk), PNANOVDB_REF(level));
        adj_read_at(level, address, ijk, adj_ret);
    }

    CUDA_CALLABLE inline void adj_read_at(pnanovdb_uint32_t level, const pnanovdb_address_t address,
                                          const pnanovdb_coord_t &ijk, const T &adj_ret) const
    {
        if (level == 0)
        {
            pnanovdb_uint64_t voxel_index = get_grid_voxel_index(grid_type, buf, address, ijk);

            if (voxel_index > 0)
            {
                adj_atomic_add(&index_grad(data, voxel_index - 1), adj_ret);
                return;
            }
        }
        *adj_background += adj_ret;
    }
};

CUDA_CALLABLE inline pnanovdb_coord_t vec3_round_to_coord(const pnanovdb_vec3_t a)
{
    pnanovdb_coord_t v;
    v.x = pnanovdb_float_to_int32(roundf(a.x));
    v.y = pnanovdb_float_to_int32(roundf(a.y));
    v.z = pnanovdb_float_to_int32(roundf(a.z));
    return v;
}

template <typename T> struct val_traits
{
    using grad_t = vec_t<3, T>;
    using scalar_t = T;

    // multiplies the gradient on the right
    // needs to be specialized for scalar types as gradient is stored as column rather than row vector
    static CUDA_CALLABLE inline T rmul(const grad_t &grad, const vec_t<3, scalar_t> &rhs)
    {
        return dot(grad, rhs);
    }
};

template <unsigned Length, typename T> struct val_traits<vec_t<Length, T>>
{
    using grad_t = mat_t<3, Length, T>;
    using scalar_t = T;

    static CUDA_CALLABLE inline vec_t<Length, T> rmul(const grad_t &grad, const vec_t<3, scalar_t> &rhs)
    {
        return mul(grad, rhs);
    }
};

// Sampling the volume at the given index-space coordinates, uvw can be fractional
template <typename Accessor>
CUDA_CALLABLE inline typename Accessor::ValueType volume_sample(Accessor &accessor, vec3 uvw, int sampling_mode)
{
    using T = typename Accessor::ValueType;
    using w_t = typename val_traits<T>::scalar_t;

    if (!accessor.is_valid())
    {
        return 0;
    }

    const pnanovdb_buf_t buf = accessor.buf;
    const pnanovdb_vec3_t uvw_pnano{uvw[0], uvw[1], uvw[2]};

    if (sampling_mode == CLOSEST)
    {
        const pnanovdb_coord_t ijk = vec3_round_to_coord(uvw_pnano);
        return accessor.read_single(ijk);
    }
    else if (sampling_mode == LINEAR)
    {
        // NB. linear sampling is not used on int volumes
        constexpr pnanovdb_coord_t OFFSETS[] = {
            {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
        };

        const pnanovdb_vec3_t ijk_base{floorf(uvw_pnano.x), floorf(uvw_pnano.y), floorf(uvw_pnano.z)};
        const pnanovdb_vec3_t ijk_frac{uvw_pnano.x - ijk_base.x, uvw_pnano.y - ijk_base.y, uvw_pnano.z - ijk_base.z};
        const pnanovdb_coord_t ijk{(pnanovdb_int32_t)ijk_base.x, (pnanovdb_int32_t)ijk_base.y,
                                   (pnanovdb_int32_t)ijk_base.z};

        accessor.init_cache();
        T val = 0;
        const float wx[2]{1 - ijk_frac.x, ijk_frac.x};
        const float wy[2]{1 - ijk_frac.y, ijk_frac.y};
        const float wz[2]{1 - ijk_frac.z, ijk_frac.z};
        for (int idx = 0; idx < 8; ++idx)
        {
            const pnanovdb_coord_t &offs = OFFSETS[idx];
            const pnanovdb_coord_t ijk_shifted = pnanovdb_coord_add(ijk, offs);
            const T v = accessor.read_cache(ijk_shifted);

            const w_t w = wx[offs.x] * wy[offs.y] * wz[offs.z];
            val = add(val, w * v);
        }
        return val;
    }
    return 0;
}

template <typename Accessor>
CUDA_CALLABLE inline void adj_volume_sample(Accessor &accessor, vec3 uvw, int sampling_mode, vec3 &adj_uvw,
                                            const typename Accessor::ValueType &adj_ret)
{
    // TODO: accessor data gradients

    using T = typename Accessor::ValueType;
    using w_t = typename val_traits<T>::scalar_t;
    using w_grad_t = vec_t<3, w_t>;

    if (!accessor.is_valid())
    {
        return;
    }

    const pnanovdb_buf_t buf = accessor.buf;
    const pnanovdb_vec3_t uvw_pnano{uvw[0], uvw[1], uvw[2]};

    if (sampling_mode != LINEAR)
    {
        const pnanovdb_coord_t ijk = vec3_round_to_coord(uvw_pnano);
        accessor.adj_read_single(ijk, adj_ret);
        return;
    }

    constexpr pnanovdb_coord_t OFFSETS[] = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
    };

    const pnanovdb_vec3_t ijk_base{floorf(uvw_pnano.x), floorf(uvw_pnano.y), floorf(uvw_pnano.z)};
    const pnanovdb_vec3_t ijk_frac{uvw_pnano.x - ijk_base.x, uvw_pnano.y - ijk_base.y, uvw_pnano.z - ijk_base.z};
    const pnanovdb_coord_t ijk{(pnanovdb_int32_t)ijk_base.x, (pnanovdb_int32_t)ijk_base.y,
                               (pnanovdb_int32_t)ijk_base.z};

    accessor.init_cache();

    const float wx[2]{1 - ijk_frac.x, ijk_frac.x};
    const float wy[2]{1 - ijk_frac.y, ijk_frac.y};
    const float wz[2]{1 - ijk_frac.z, ijk_frac.z};
    for (int idx = 0; idx < 8; ++idx)
    {
        const pnanovdb_coord_t &offs = OFFSETS[idx];
        const pnanovdb_coord_t ijk_shifted = pnanovdb_coord_add(ijk, offs);
        const T v = accessor.read_cache(ijk_shifted);

        const vec3 signs(offs.x * 2 - 1, offs.y * 2 - 1, offs.z * 2 - 1);

        const w_t w = wx[offs.x] * wy[offs.y] * wz[offs.z];
        const w_grad_t grad_w(signs[0] * wy[offs.y] * wz[offs.z], signs[1] * wx[offs.x] * wz[offs.z],
                              signs[2] * wx[offs.x] * wy[offs.y]);

        adj_uvw += vec3(mul(w_t(dot(v, adj_ret)), grad_w));

        const T adj_v = w * adj_ret;
        accessor.adj_read_cache(ijk_shifted, adj_v);
    }
}

// Sampling the volume at the given index-space coordinates, uvw can be fractional
template <typename Accessor>
CUDA_CALLABLE inline typename Accessor::ValueType volume_sample_grad(
    Accessor &accessor, vec3 uvw, int sampling_mode, typename val_traits<typename Accessor::ValueType>::grad_t &grad)
{
    using T = typename Accessor::ValueType;
    using grad_T = typename val_traits<T>::grad_t;
    using w_t = typename val_traits<T>::scalar_t;
    using w_grad_t = vec_t<3, w_t>;

    grad = grad_T{};

    if (!accessor.is_valid())
    {
        return 0;
    }

    const pnanovdb_buf_t buf = accessor.buf;
    const pnanovdb_vec3_t uvw_pnano{uvw[0], uvw[1], uvw[2]};

    if (sampling_mode == CLOSEST)
    {
        const pnanovdb_coord_t ijk = vec3_round_to_coord(uvw_pnano);
        return accessor.read_single(ijk);
    }
    else if (sampling_mode == LINEAR)
    {
        // NB. linear sampling is not used on int volumes
        constexpr pnanovdb_coord_t OFFSETS[] = {
            {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
        };

        const pnanovdb_vec3_t ijk_base{floorf(uvw_pnano.x), floorf(uvw_pnano.y), floorf(uvw_pnano.z)};
        const pnanovdb_vec3_t ijk_frac{uvw_pnano.x - ijk_base.x, uvw_pnano.y - ijk_base.y, uvw_pnano.z - ijk_base.z};
        const pnanovdb_coord_t ijk{(pnanovdb_int32_t)ijk_base.x, (pnanovdb_int32_t)ijk_base.y,
                                   (pnanovdb_int32_t)ijk_base.z};

        accessor.init_cache();
        T val = 0;
        const float wx[2]{1 - ijk_frac.x, ijk_frac.x};
        const float wy[2]{1 - ijk_frac.y, ijk_frac.y};
        const float wz[2]{1 - ijk_frac.z, ijk_frac.z};
        for (int idx = 0; idx < 8; ++idx)
        {
            const pnanovdb_coord_t &offs = OFFSETS[idx];
            const pnanovdb_coord_t ijk_shifted = pnanovdb_coord_add(ijk, offs);
            const T v = accessor.read_cache(ijk_shifted);

            const vec3 signs(offs.x * 2 - 1, offs.y * 2 - 1, offs.z * 2 - 1);

            const w_t w = wx[offs.x] * wy[offs.y] * wz[offs.z];
            const w_grad_t grad_w(signs[0] * wy[offs.y] * wz[offs.z], signs[1] * wx[offs.x] * wz[offs.z],
                                  signs[2] * wx[offs.x] * wy[offs.y]);

            val = add(val, w * v);
            grad += outer(v, grad_w);
        }
        return val;
    }
    return 0;
}

template <typename Accessor>
CUDA_CALLABLE inline void adj_volume_sample_grad(Accessor &accessor, vec3 uvw, int sampling_mode,
                                                 typename val_traits<typename Accessor::ValueType>::grad_t &grad,
                                                 vec3 &adj_uvw,
                                                 typename val_traits<typename Accessor::ValueType>::grad_t &adj_grad,
                                                 const typename Accessor::ValueType &adj_ret)
{
    // TODO: accessor data gradients

    using T = typename Accessor::ValueType;
    using grad_T = typename val_traits<T>::grad_t;
    using w_t = typename val_traits<T>::scalar_t;
    using w_grad_t = vec_t<3, w_t>;
    using w_hess_t = mat_t<3, 3, w_t>;

    if (!accessor.is_valid())
    {
        return;
    }

    const pnanovdb_buf_t buf = accessor.buf;
    const pnanovdb_vec3_t uvw_pnano{uvw[0], uvw[1], uvw[2]};

    if (sampling_mode != LINEAR)
    {
        const pnanovdb_coord_t ijk = vec3_round_to_coord(uvw_pnano);
        accessor.adj_read_single(ijk, adj_ret);
        return;
    }

    constexpr pnanovdb_coord_t OFFSETS[] = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
    };

    const pnanovdb_vec3_t ijk_base{floorf(uvw_pnano.x), floorf(uvw_pnano.y), floorf(uvw_pnano.z)};
    const pnanovdb_vec3_t ijk_frac{uvw_pnano.x - ijk_base.x, uvw_pnano.y - ijk_base.y, uvw_pnano.z - ijk_base.z};
    const pnanovdb_coord_t ijk{(pnanovdb_int32_t)ijk_base.x, (pnanovdb_int32_t)ijk_base.y,
                               (pnanovdb_int32_t)ijk_base.z};

    accessor.init_cache();

    const float wx[2]{1 - ijk_frac.x, ijk_frac.x};
    const float wy[2]{1 - ijk_frac.y, ijk_frac.y};
    const float wz[2]{1 - ijk_frac.z, ijk_frac.z};
    for (int idx = 0; idx < 8; ++idx)
    {
        const pnanovdb_coord_t &offs = OFFSETS[idx];
        const pnanovdb_coord_t ijk_shifted = pnanovdb_coord_add(ijk, offs);
        const T v = accessor.read_cache(ijk_shifted);

        const vec3 signs(offs.x * 2 - 1, offs.y * 2 - 1, offs.z * 2 - 1);

        const w_t w = wx[offs.x] * wy[offs.y] * wz[offs.z];
        const w_grad_t grad_w(signs[0] * wy[offs.y] * wz[offs.z], signs[1] * wx[offs.x] * wz[offs.z],
                              signs[2] * wx[offs.x] * wy[offs.y]);
        adj_uvw += vec3(mul(w_t(dot(v, adj_ret)), grad_w));

        const w_hess_t hess_w(0.0, signs[1] * signs[0] * wz[offs.z], signs[2] * signs[0] * wy[offs.y],
                              signs[0] * signs[1] * wz[offs.z], 0.0, signs[2] * signs[1] * wx[offs.x],
                              signs[0] * signs[2] * wy[offs.y], signs[1] * signs[2] * wx[offs.x], 0.0);
        adj_uvw += vec3(mul(mul(v, adj_grad), hess_w));

        const T adj_v = w * adj_ret + val_traits<T>::rmul(adj_grad, grad_w);
        accessor.adj_read_cache(ijk_shifted, adj_v);
    }
}

} // namespace volume
  // namespace volume

// exposed kernel builtins

// volume_sample

template <typename T> CUDA_CALLABLE inline T volume_sample(uint64_t id, vec3 uvw, int sampling_mode)
{
    volume::leaf_value_accessor<T> accessor(volume::id_to_buffer(id));
    return volume::volume_sample(accessor, uvw, sampling_mode);
}

template <typename T>
CUDA_CALLABLE inline void adj_volume_sample(uint64_t id, vec3 uvw, int sampling_mode, uint64_t &adj_id, vec3 &adj_uvw,
                                            int &adj_sampling_mode, const T &adj_ret)
{
    volume::leaf_value_accessor<T> accessor(volume::id_to_buffer(id));
    volume::adj_volume_sample(accessor, uvw, sampling_mode, adj_uvw, adj_ret);
}

template <typename T>
CUDA_CALLABLE inline T volume_sample_grad(uint64_t id, vec3 uvw, int sampling_mode,
                                          typename volume::val_traits<T>::grad_t &grad)
{
    volume::leaf_value_accessor<T> accessor(volume::id_to_buffer(id));
    return volume::volume_sample_grad(accessor, uvw, sampling_mode, grad);
}

template <typename T>
CUDA_CALLABLE inline void adj_volume_sample_grad(uint64_t id, vec3 uvw, int sampling_mode,
                                                 typename volume::val_traits<T>::grad_t &grad, uint64_t &adj_id,
                                                 vec3 &adj_uvw, int &adj_sampling_mode,
                                                 typename volume::val_traits<T>::grad_t &adj_grad, const T &adj_ret)
{
    volume::leaf_value_accessor<T> accessor(volume::id_to_buffer(id));
    volume::adj_volume_sample_grad(accessor, uvw, sampling_mode, grad, adj_uvw, adj_grad, adj_ret);
}

// Sampling a float volume at the given index-space coordinates, uvw can be fractional
CUDA_CALLABLE inline float volume_sample_f(uint64_t id, vec3 uvw, int sampling_mode)
{
    return volume_sample<float>(id, uvw, sampling_mode);
}

// Sampling an int volume at the given index-space coordinates, uvw can be fractional
CUDA_CALLABLE inline int32_t volume_sample_i(uint64_t id, vec3 uvw)
{
    return volume_sample<int32_t>(id, uvw, volume::CLOSEST);
}

// Sampling a vector volume at the given index-space coordinates, uvw can be fractional
CUDA_CALLABLE inline vec3 volume_sample_v(uint64_t id, vec3 uvw, int sampling_mode)
{
    return volume_sample<vec3>(id, uvw, sampling_mode);
}

CUDA_CALLABLE inline void adj_volume_sample_f(uint64_t id, vec3 uvw, int sampling_mode, uint64_t &adj_id, vec3 &adj_uvw,
                                              int &adj_sampling_mode, const float &adj_ret)
{
    adj_volume_sample(id, uvw, sampling_mode, adj_id, adj_uvw, adj_sampling_mode, adj_ret);
}

CUDA_CALLABLE inline void adj_volume_sample_v(uint64_t id, vec3 uvw, int sampling_mode, uint64_t &adj_id, vec3 &adj_uvw,
                                              int &adj_sampling_mode, const vec3 &adj_ret)
{
    adj_volume_sample(id, uvw, sampling_mode, adj_id, adj_uvw, adj_sampling_mode, adj_ret);
}

CUDA_CALLABLE inline void adj_volume_sample_i(uint64_t id, vec3 uvw, uint64_t &adj_id, vec3 &adj_uvw,
                                              const int32_t &adj_ret)
{
    // NOP
}

// Sampling the volume at the given index-space coordinates, uvw can be fractional
CUDA_CALLABLE inline float volume_sample_grad_f(uint64_t id, vec3 uvw, int sampling_mode, vec3 &grad)
{
    return volume_sample_grad<float>(id, uvw, sampling_mode, grad);
}

CUDA_CALLABLE inline void adj_volume_sample_grad_f(uint64_t id, vec3 uvw, int sampling_mode, vec3 &grad,
                                                   uint64_t &adj_id, vec3 &adj_uvw, int &adj_sampling_mode,
                                                   vec3 &adj_grad, const float &adj_ret)
{
    adj_volume_sample_grad<float>(id, uvw, sampling_mode, grad, adj_id, adj_uvw, adj_sampling_mode, adj_grad, adj_ret);
}

// volume_sample_index

template <typename T>
CUDA_CALLABLE inline T volume_sample_index(uint64_t id, vec3 uvw, int sampling_mode, const array_t<T> &voxel_data,
                                           const T &background)
{
    volume::index_value_accessor<T> accessor(volume::id_to_buffer(id), voxel_data, background);
    return volume::volume_sample(accessor, uvw, sampling_mode);
}

template <typename T>
CUDA_CALLABLE inline void adj_volume_sample_index(uint64_t id, vec3 uvw, int sampling_mode,
                                                  const array_t<T> &voxel_data, const T &background, uint64_t &adj_id,
                                                  vec3 &adj_uvw, int &adj_sampling_mode, array_t<T> &adj_voxel_data,
                                                  T &adj_background, const T &adj_ret)
{
    volume::index_value_accessor<T> accessor(volume::id_to_buffer(id), voxel_data, background, &adj_background);
    volume::adj_volume_sample(accessor, uvw, sampling_mode, adj_uvw, adj_ret);
}

template <typename T>
CUDA_CALLABLE inline T volume_sample_grad_index(uint64_t id, vec3 uvw, int sampling_mode, const array_t<T> &voxel_data,
                                                const T &background, typename volume::val_traits<T>::grad_t &grad)
{
    volume::index_value_accessor<T> accessor(volume::id_to_buffer(id), voxel_data, background);
    return volume::volume_sample_grad(accessor, uvw, sampling_mode, grad);
}

template <typename T>
CUDA_CALLABLE inline void adj_volume_sample_grad_index(
    uint64_t id, vec3 uvw, int sampling_mode, const array_t<T> &voxel_data, const T &background,
    typename volume::val_traits<T>::grad_t &grad, uint64_t &adj_id, vec3 &adj_uvw, int &adj_sampling_mode,
    array_t<T> &adj_voxel_data, T &adj_background, typename volume::val_traits<T>::grad_t &adj_grad, const T &adj_ret)
{
    volume::index_value_accessor<T> accessor(volume::id_to_buffer(id), voxel_data, background, &adj_background);
    volume::adj_volume_sample_grad(accessor, uvw, sampling_mode, grad, adj_uvw, adj_grad, adj_ret);
}

// volume_lookup

template <typename T> CUDA_CALLABLE inline T volume_lookup(uint64_t id, int32_t i, int32_t j, int32_t k)
{
    using traits = volume::pnano_traits<T>;

    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    if (volume::get_grid_type(buf) != traits::GRID_TYPE)
        return 0;

    const pnanovdb_root_handle_t root = volume::get_root(buf);

    const pnanovdb_coord_t ijk{i, j, k};
    T val;
    volume::pnano_read(val, buf, root, PNANOVDB_REF(ijk));
    return val;
}

template <typename T>
CUDA_CALLABLE inline void adj_volume_lookup(uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t &adj_id,
                                            int32_t &adj_i, int32_t &adj_j, int32_t &adj_k, const T &adj_ret)
{
    // NOP -- adjoint of grid values is not available
}

CUDA_CALLABLE inline float volume_lookup_f(uint64_t id, int32_t i, int32_t j, int32_t k)
{
    return volume_lookup<float>(id, i, j, k);
}

CUDA_CALLABLE inline int32_t volume_lookup_i(uint64_t id, int32_t i, int32_t j, int32_t k)
{
    return volume_lookup<int32_t>(id, i, j, k);
}

CUDA_CALLABLE inline vec3 volume_lookup_v(uint64_t id, int32_t i, int32_t j, int32_t k)
{
    return volume_lookup<vec3>(id, i, j, k);
}

CUDA_CALLABLE inline void adj_volume_lookup_f(uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t &adj_id,
                                              int32_t &adj_i, int32_t &adj_j, int32_t &adj_k, const float &adj_ret)
{
    adj_volume_lookup(id, i, j, k, adj_id, adj_i, adj_j, adj_k, adj_ret);
}

CUDA_CALLABLE inline void adj_volume_lookup_i(uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t &adj_id,
                                              int32_t &adj_i, int32_t &adj_j, int32_t &adj_k, const int32_t &adj_ret)
{
    adj_volume_lookup(id, i, j, k, adj_id, adj_i, adj_j, adj_k, adj_ret);
}

CUDA_CALLABLE inline void adj_volume_lookup_v(uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t &adj_id,
                                              int32_t &adj_i, int32_t &adj_j, int32_t &adj_k, const vec3 &adj_ret)
{
    adj_volume_lookup(id, i, j, k, adj_id, adj_i, adj_j, adj_k, adj_ret);
}

CUDA_CALLABLE inline int32_t volume_lookup_index(uint64_t id, int32_t i, int32_t j, int32_t k)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_root_handle_t root = volume::get_root(buf);
    const pnanovdb_grid_type_t grid_type = volume::get_grid_type(buf);

    const pnanovdb_coord_t ijk{i, j, k};

    pnanovdb_uint32_t level;
    const pnanovdb_address_t address =
        pnanovdb_root_get_value_address_and_level(grid_type, buf, root, PNANOVDB_REF(ijk), PNANOVDB_REF(level));

    if (level == 0)
    {
        pnanovdb_uint64_t voxel_index = volume::get_grid_voxel_index(grid_type, buf, address, ijk);

        return static_cast<int32_t>(voxel_index) - 1;
    }
    return -1;
}

CUDA_CALLABLE inline void adj_volume_lookup_index(uint64_t id, int32_t i, int32_t j, int32_t k, uint64_t &adj_id,
                                                  int32_t &adj_i, int32_t &adj_j, int32_t &adj_k, const vec3 &adj_ret)
{
    // NOP
}

// volume_store

template <typename T>
CUDA_CALLABLE inline void volume_store(uint64_t id, int32_t i, int32_t j, int32_t k, const T &value)
{
    using traits = volume::pnano_traits<T>;

    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    if (volume::get_grid_type(buf) != traits::GRID_TYPE)
        return;

    const pnanovdb_root_handle_t root = volume::get_root(buf);
    const pnanovdb_coord_t ijk{i, j, k};

    pnanovdb_uint32_t level;
    const pnanovdb_address_t address =
        pnanovdb_root_get_value_address_and_level(traits::GRID_TYPE, buf, root, PNANOVDB_REF(ijk), PNANOVDB_REF(level));

    if (level == 0)
    {
        // only write at at leaf level (prevent modifying background value)
        // TODO is this the intended semantics? or should be allow writing to background?
        volume::pnano_write(value, buf, address);
    }
}

template <typename T>
CUDA_CALLABLE inline void adj_volume_store(uint64_t id, int32_t i, int32_t j, int32_t k, const T &value,
                                           uint64_t &adj_id, int32_t &adj_i, int32_t &adj_j, int32_t &adj_k,
                                           T &adj_value)
{
    // NOP -- adjoint of grid values is not available
}

CUDA_CALLABLE inline void volume_store_f(uint64_t id, int32_t i, int32_t j, int32_t k, const float &value)
{
    volume_store(id, i, j, k, value);
}

CUDA_CALLABLE inline void adj_volume_store_f(uint64_t id, int32_t i, int32_t j, int32_t k, const float &value,
                                             uint64_t &adj_id, int32_t &adj_i, int32_t &adj_j, int32_t &adj_k,
                                             float &adj_value)
{
    adj_volume_store(id, i, j, k, value, adj_id, adj_i, adj_j, adj_k, adj_value);
}

CUDA_CALLABLE inline void volume_store_v(uint64_t id, int32_t i, int32_t j, int32_t k, const vec3 &value)
{
    volume_store(id, i, j, k, value);
}

CUDA_CALLABLE inline void adj_volume_store_v(uint64_t id, int32_t i, int32_t j, int32_t k, const vec3 &value,
                                             uint64_t &adj_id, int32_t &adj_i, int32_t &adj_j, int32_t &adj_k,
                                             vec3 &adj_value)
{
    adj_volume_store(id, i, j, k, value, adj_id, adj_i, adj_j, adj_k, adj_value);
}

CUDA_CALLABLE inline void volume_store_i(uint64_t id, int32_t i, int32_t j, int32_t k, const int32_t &value)
{
    volume_store(id, i, j, k, value);
}

CUDA_CALLABLE inline void adj_volume_store_i(uint64_t id, int32_t i, int32_t j, int32_t k, const int32_t &value,
                                             uint64_t &adj_id, int32_t &adj_i, int32_t &adj_j, int32_t &adj_k,
                                             int32_t &adj_value)
{
    adj_volume_store(id, i, j, k, value, adj_id, adj_i, adj_j, adj_k, adj_value);
}

// Transform position from index space to world space
CUDA_CALLABLE inline vec3 volume_index_to_world(uint64_t id, vec3 uvw)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = {0u};
    const pnanovdb_vec3_t pos{uvw[0], uvw[1], uvw[2]};
    const pnanovdb_vec3_t xyz = pnanovdb_grid_index_to_worldf(buf, grid, PNANOVDB_REF(pos));
    return {xyz.x, xyz.y, xyz.z};
}

// Transform position from world space to index space
CUDA_CALLABLE inline vec3 volume_world_to_index(uint64_t id, vec3 xyz)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = {0u};
    const pnanovdb_vec3_t pos{xyz[0], xyz[1], xyz[2]};
    const pnanovdb_vec3_t uvw = pnanovdb_grid_world_to_indexf(buf, grid, PNANOVDB_REF(pos));
    return {uvw.x, uvw.y, uvw.z};
}

CUDA_CALLABLE inline void adj_volume_index_to_world(uint64_t id, vec3 uvw, uint64_t &adj_id, vec3 &adj_uvw,
                                                    const vec3 &adj_ret)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = {0u};
    const pnanovdb_vec3_t pos{adj_ret[0], adj_ret[1], adj_ret[2]};
    const pnanovdb_vec3_t xyz = pnanovdb_grid_index_to_world_dirf(buf, grid, PNANOVDB_REF(pos));
    adj_uvw = add(adj_uvw, vec3{xyz.x, xyz.y, xyz.z});
}

CUDA_CALLABLE inline void adj_volume_world_to_index(uint64_t id, vec3 xyz, uint64_t &adj_id, vec3 &adj_xyz,
                                                    const vec3 &adj_ret)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = {0u};
    const pnanovdb_vec3_t pos{adj_ret[0], adj_ret[1], adj_ret[2]};
    const pnanovdb_vec3_t uvw = pnanovdb_grid_world_to_index_dirf(buf, grid, PNANOVDB_REF(pos));
    adj_xyz = add(adj_xyz, vec3{uvw.x, uvw.y, uvw.z});
}

// Transform direction from index space to world space
CUDA_CALLABLE inline vec3 volume_index_to_world_dir(uint64_t id, vec3 uvw)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = {0u};
    const pnanovdb_vec3_t pos{uvw[0], uvw[1], uvw[2]};
    const pnanovdb_vec3_t xyz = pnanovdb_grid_index_to_world_dirf(buf, grid, PNANOVDB_REF(pos));
    return {xyz.x, xyz.y, xyz.z};
}

// Transform direction from world space to index space
CUDA_CALLABLE inline vec3 volume_world_to_index_dir(uint64_t id, vec3 xyz)
{
    const pnanovdb_buf_t buf = volume::id_to_buffer(id);
    const pnanovdb_grid_handle_t grid = {0u};
    const pnanovdb_vec3_t pos{xyz[0], xyz[1], xyz[2]};
    const pnanovdb_vec3_t uvw = pnanovdb_grid_world_to_index_dirf(buf, grid, PNANOVDB_REF(pos));
    return {uvw.x, uvw.y, uvw.z};
}

CUDA_CALLABLE inline void adj_volume_index_to_world_dir(uint64_t id, vec3 uvw, uint64_t &adj_id, vec3 &adj_uvw,
                                                        const vec3 &adj_ret)
{
    adj_volume_index_to_world(id, uvw, adj_id, adj_uvw, adj_ret);
}

CUDA_CALLABLE inline void adj_volume_world_to_index_dir(uint64_t id, vec3 xyz, uint64_t &adj_id, vec3 &adj_xyz,
                                                        const vec3 &adj_ret)
{
    adj_volume_world_to_index(id, xyz, adj_id, adj_xyz, adj_ret);
}

} // namespace wp
